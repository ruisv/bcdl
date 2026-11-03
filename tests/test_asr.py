"""Speech front end + CIF + token decoding: numpy path, no model required.

    PYTHONPATH=build:python pytest -s tests/test_asr.py

A speech model trained on Kaldi fbank features does not fail on features that
are merely close to them — it transcribes worse, with nothing to show why. So
the front end is pinned here against an independent numpy implementation of
`torchaudio.compliance.kaldi.fbank` (dither 0) and FunASR's `apply_lfr`; that
reference itself agrees with torchaudio to ~2e-4 in the log domain, the
float32-vs-float64 floor.

CIF gets the same treatment, plus the one behaviour that cost real accuracy: the
tail weight must land at the true end of speech, not at the end of the padded
window, or every utterance whose last token needed it loses that token.
"""
import json

import numpy as np
import pytest

import bcdl


# --- references ---------------------------------------------------------------

def ref_fbank(pcm, sr=16000, num_mel=80, frame_ms=25.0, shift_ms=10.0, preemph=0.97,
              low=20.0, high=0.0, scale=32768.0):
    x = np.asarray(pcm, np.float64) * scale
    flen, fsh = int(sr * frame_ms / 1000), int(sr * shift_ms / 1000)
    if len(x) < flen:
        return np.zeros((0, num_mel), np.float32)
    nfft = 1
    while nfft < flen:
        nfft *= 2
    n = 1 + (len(x) - flen) // fsh
    fr = np.stack([x[i * fsh:i * fsh + flen] for i in range(n)])
    fr = fr - fr.mean(1, keepdims=True)
    fr = fr - preemph * np.concatenate([fr[:, :1], fr[:, :-1]], 1)
    fr = fr * (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(flen) / (flen - 1)))
    spec = np.abs(np.fft.rfft(fr, nfft)) ** 2

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    hi = high if high > 0 else sr / 2 + high
    ml, mh = mel(low), mel(hi)
    d = (mh - ml) / (num_mel + 1)
    fm = mel(sr / nfft * np.arange(nfft // 2))
    W = np.zeros((num_mel, nfft // 2 + 1))
    for b in range(num_mel):
        lo, c, r = ml + b * d, ml + (b + 1) * d, ml + (b + 2) * d
        W[b, :nfft // 2] = np.maximum(0, np.minimum((fm - lo) / (c - lo), (r - fm) / (r - c)))
    return np.log(np.maximum(spec @ W.T, np.finfo(np.float32).eps)).astype(np.float32)


def ref_lfr(f, m=7, n=6):
    tl = -(-len(f) // n)
    p = np.concatenate([np.repeat(f[:1], (m - 1) // 2, 0), f], 0)
    idx = np.minimum(np.arange(tl)[:, None] * n + np.arange(m)[None, :], len(p) - 1)
    return p[idx].reshape(tl, m * f.shape[1]).astype(np.float32)


def ref_cif(alphas, hidden, valid, tail=0.45, threshold=1.0, max_tokens=100):
    """FunASR's frame loop, with the tail placed at frame `valid`."""
    a = np.maximum(np.asarray(alphas, np.float64), 0)[:valid].tolist()
    h = [np.asarray(x, np.float64) for x in hidden[:valid]]
    if tail > 0:
        a.append(tail)
        h.append(np.zeros(hidden.shape[1]))
    out, frame, integ = [], np.zeros(hidden.shape[1]), 0.0
    for at, ht in zip(a, h):
        comp = 1.0 - integ
        integ += at
        fire = integ >= threshold
        cur = comp if fire else at
        frame = frame + cur * ht
        if fire:
            out.append(frame)
            integ -= 1.0
            frame = (at - cur) * ht
    emb = np.zeros((max_tokens, hidden.shape[1]), np.float32)
    k = min(len(out), max_tokens)
    if k:
        emb[:k] = np.stack(out[:k])
    return emb, k


def _signal(seconds, seed=0):
    """A chirp plus noise: energy across the whole band, so every mel bin is exercised."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(16000 * seconds)) / 16000.0
    chirp = 0.3 * np.sin(2 * np.pi * (100 + 3500 * t / max(seconds, 1e-3)) * t)
    return (chirp + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


# --- fbank --------------------------------------------------------------------

@pytest.mark.parametrize("seconds", [0.03, 0.5, 2.3])
def test_fbank_matches_kaldi_reference(seconds):
    pcm = _signal(seconds)
    fb = bcdl.Fbank()
    got = fb.compute(pcm)
    want = ref_fbank(pcm)
    assert got.shape == want.shape == (fb.num_frames(pcm.size), 80)
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_fbank_framing_is_kaldi_snip_edges():
    fb = bcdl.Fbank()
    assert (fb.frame_length, fb.frame_shift, fb.fft_size) == (400, 160, 512)
    assert fb.num_frames(399) == 0
    assert fb.num_frames(400) == 1
    assert fb.num_frames(400 + 159) == 1
    assert fb.num_frames(400 + 160) == 2
    assert fb.compute(np.zeros(100, np.float32)).shape == (0, 80)


def test_fbank_silence_hits_the_log_floor_not_nan():
    out = bcdl.Fbank().compute(np.zeros(1600, np.float32))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, np.log(np.finfo(np.float32).eps), rtol=1e-6)


def test_fbank_input_scale_is_applied():
    # Kaldi works on int16-scale samples: float PCM in [-1, 1] must be scaled up
    # first, and the same samples given pre-scaled must produce the same features.
    pcm = _signal(0.5)
    a = bcdl.Fbank().compute(pcm)
    cfg = bcdl.FbankConfig()
    cfg.input_scale = 1.0
    b = bcdl.Fbank(cfg).compute(pcm * 32768.0)
    np.testing.assert_allclose(a, b, atol=1e-3)


# --- LFR + CMVN -----------------------------------------------------------------

@pytest.mark.parametrize("frames", list(range(1, 20)) + [37, 248])
def test_lfr_matches_funasr(frames):
    f = np.random.default_rng(frames).standard_normal((frames, 5)).astype(np.float32)
    got = bcdl.lfr_stack(f, 7, 6)
    np.testing.assert_array_equal(got, ref_lfr(f, 7, 6))


def test_cmvn_parse_and_apply():
    text = ("<Nnet>\n<Splice> 3 3\n[ 0 ]\n<AddShift> 3 3\n<LearnRateCoef> 0 [ 1 -2 0.5 ]\n"
            "<Rescale> 3 3\n<LearnRateCoef> 0 [ 2 0.5 -1 ]\n</Nnet>\n")
    c = bcdl.parse_kaldi_cmvn(text)
    assert c.dim == 3
    np.testing.assert_allclose(c.shift, [1, -2, 0.5])
    np.testing.assert_allclose(c.scale, [2, 0.5, -1])
    x = np.array([[0, 0, 0], [1, 2, 3]], np.float32)
    np.testing.assert_allclose(c.apply(x), (x + c.shift) * c.scale)


def test_cmvn_rejects_a_file_without_statistics():
    with pytest.raises(Exception):
        bcdl.parse_kaldi_cmvn("<Nnet>\n</Nnet>\n")


# --- CIF -------------------------------------------------------------------------

def test_cif_matches_reference_loop():
    rng = np.random.default_rng(7)
    frames, dim, valid = 41, 6, 33
    alphas = rng.uniform(0.0, 0.6, frames).astype(np.float32)
    hidden = rng.standard_normal((frames, dim)).astype(np.float32)
    emb, n = bcdl.cif_integrate(alphas, hidden, valid)
    want, k = ref_cif(alphas, hidden, valid)
    assert n == k > 0
    np.testing.assert_allclose(emb, want, atol=1e-5)


def test_cif_tail_fires_the_last_token():
    # 2.7 of weight over the valid frames: without the tail only two tokens fire
    # and the third (the last syllable) is lost. This is the shipped fixed-shape
    # build's failure — its graph puts the tail at the end of the padding, where
    # masking the padding then deletes it.
    frames, dim = 12, 4
    alphas = np.zeros(frames, np.float32)
    alphas[:9] = 0.3                                       # 2.7 in total
    alphas[9:] = 0.9                                       # padding: must be ignored
    hidden = np.ones((frames, dim), np.float32)
    cfg = bcdl.CifConfig()
    _, with_tail = bcdl.cif_integrate(alphas, hidden, 9, cfg)
    cfg.tail_threshold = 0.0
    _, without = bcdl.cif_integrate(alphas, hidden, 9, cfg)
    assert (with_tail, without) == (3, 2)


def test_cif_ignores_padding_and_caps_tokens():
    frames, dim = 30, 2
    alphas = np.full(frames, 0.99, np.float32)
    hidden = np.ones((frames, dim), np.float32)
    _, n_valid = bcdl.cif_integrate(alphas, hidden, 5)
    assert n_valid == 5          # 5 * 0.99 + 0.45 tail -> 5 fires; padding never counts
    cfg = bcdl.CifConfig()
    cfg.max_tokens = 3
    emb, n = bcdl.cif_integrate(alphas, hidden, 20, cfg)
    assert n == 3 and emb.shape == (3, dim)


def test_cif_needs_a_frame_for_the_tail():
    a = np.full(4, 0.5, np.float32)
    h = np.ones((4, 2), np.float32)
    with pytest.raises(Exception):
        bcdl.cif_integrate(a, h, 4)          # no frame left after the valid ones
    cfg = bcdl.CifConfig()
    cfg.tail_threshold = 0.0
    assert bcdl.cif_integrate(a, h, 4, cfg)[1] == 2


# --- tokens ------------------------------------------------------------------------

def test_tokens_to_text_rules():
    vocab = ["<blank>", "<s>", "</s>", "你", "好", "hel@@", "lo", "world", "<unk>", "ok"]
    ids = [1, 3, 4, 5, 6, 7, 9, 2, 0, 8]
    assert bcdl.tokens_to_text(ids, vocab) == "你好hello world ok"


def test_tokens_to_text_rejects_out_of_range_ids():
    with pytest.raises(Exception):
        bcdl.tokens_to_text([5], ["a"])


def test_argmax_tokens():
    logits = np.array([[0, 3, 1], [5, 0, 0]], np.float32)
    assert bcdl.argmax_tokens(logits) == [1, 0]


def test_load_token_list_json_and_text(tmp_path):
    toks = ["<blank>", "你", "a\"b", "é", "\U0001F600"]
    j = tmp_path / "tokens.json"
    j.write_text(json.dumps(toks, ensure_ascii=True), encoding="utf-8")   # \u escapes
    assert bcdl.load_token_list(str(j)) == toks
    t = tmp_path / "tokens.txt"
    t.write_text("<blk> 0\n你 1\nab 2\n", encoding="utf-8")                 # k2 "token id"
    assert bcdl.load_token_list(str(t)) == ["<blk>", "你", "ab"]


def test_load_token_list_bom_and_bad_surrogates(tmp_path):
    t = tmp_path / "bom.txt"
    t.write_bytes("﻿<blk> 0\n你 1\n".encode("utf-8"))
    assert bcdl.load_token_list(str(t)) == ["<blk>", "你"]
    for bad in (r'["\ud83d"]', r'["\ud83dA"]', r'["\ud83dA"]', r'["\ude00"]'):
        j = tmp_path / "bad.json"
        j.write_text(bad, encoding="utf-8")
        with pytest.raises(Exception, match="surrogate"):
            bcdl.load_token_list(str(j))
