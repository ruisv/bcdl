"""On-board speech synthesis test: Matcha-TTS (two int16 graphs) + Vocos.

The acoustic model is checked against the float model's mel for the same
sentence with the noise switched off (noise_scale 0 makes the decoder
deterministic), and the whole chain - text, front end, acoustic model, vocoder
- is checked by running its audio back through Paraformer, which must hear the
sentence.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_tts_board_py.py

Models from scripts/fetch_models.sh. Env-overridable: BCDL_MODELS / BCDL_TTS_DATA.
"""
import os
import time

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.environ.get("BCDL_MODELS", os.path.join(_REPO, "models"))
_DATA = os.environ.get("BCDL_TTS_DATA", os.path.join(_REPO, "data", "tts"))
_F = {
    "encoder": "matcha_zh_en_enc_l128_int16_nashm.hbm",
    "decoder": "matcha_zh_en_dec_t512_int16_nashm.hbm",
    "embedding": "matcha_zh_en_embedding.npy",
    "vocoder": "vocos_16k_t256_int16_nashm.hbm",
    "tokens": "matcha_zh_en_tokens.txt",
    "lexicon": "matcha_zh_en_lexicon.txt",
}
_P = {k: os.path.join(_MODELS, v) for k, v in _F.items()}
_PARA = {
    "encoder": "paraformer_encoder_int16.hbm",
    "predictor": "paraformer_predictor_int16_v2.hbm",
    "decoder": "paraformer_decoder_int16_v2.hbm",
    "tokens": "paraformer_tokens_8404.json",
    "cmvn": "paraformer_am.mvn",
}

pytestmark = pytest.mark.skipif(not all(os.path.exists(p) for p in _P.values()),
                                reason="Matcha/Vocos build missing (scripts/fetch_models.sh)")

# Per-token mean log-mel, BPU vs float; measured 0.20-0.22 on the two sentences
# (log-mel spans about -23..5.5 here).
SEG_MAE_MAX = 0.35


def _quiet():
    c = bcdl.MatchaConfig()
    c.noise_scale = 0.0
    return c


@pytest.fixture(scope="module")
def am():
    return bcdl.MatchaAcoustic(_P["encoder"], _P["decoder"], _P["embedding"], _quiet())


@pytest.fixture(scope="module")
def tts():
    return bcdl.TextToSpeech.from_files(_P["encoder"], _P["decoder"], _P["embedding"],
                                        _P["vocoder"], _P["tokens"], _P["lexicon"])


def test_geometry(am):
    assert (am.max_tokens, am.max_frames, am.num_mels, am.vocab_size) == (128, 512, 80, 2190)


def _segment_means(mel, durations):
    out, t = [], 0
    for d in durations:
        out.append(mel[:, t:t + d].mean(1) if d > 0 else np.full(mel.shape[0], np.nan))
        t += d
    return np.stack(out)


def test_matches_float_without_noise(am):
    # Durations are ceil(exp(logw)): where exp(logw) sits on an integer, the
    # int16 build can land one frame either side of float. So durations must
    # agree to within a frame each, and the mel is compared per token (the
    # mean over the token's frames) rather than frame by frame.
    for k in ("00", "36"):
        ref = np.load(os.path.join(_DATA, f"matcha_ref_{k}.npz"))
        j = 0
        while f"ids{j}" in ref:
            t0 = time.perf_counter()
            r = am.synthesize(ref[f"ids{j}"].tolist())
            dt = time.perf_counter() - t0
            want_d = ref[f"dur{j}"]
            got_d = np.array(r.durations)
            assert len(got_d) == len(want_d) and not r.truncated
            assert np.abs(got_d - want_d).max() <= 1 and abs(int(got_d.sum() - want_d.sum())) <= 2
            assert r.mel.shape == (80, r.frames) and r.frames == got_d.sum()
            a = _segment_means(ref[f"mel{j}"], want_d)
            b = _segment_means(r.mel, got_d)
            mae = float(np.nanmean(np.abs(a - b)))
            print(f"\n  sentence {k}/{j}: {len(got_d)} tokens, {r.frames} frames "
                  f"({r.frames * 256 / 16000:.2f} s) in {dt * 1000:.1f} ms, "
                  f"{int((got_d != want_d).sum())} durations off by one, per-token mel MAE {mae:.3f}")
            assert mae < SEG_MAE_MAX
            j += 1


def test_seed_reproduces_the_noise():
    c = bcdl.MatchaConfig()
    c.seed = 7
    a = bcdl.MatchaAcoustic(_P["encoder"], _P["decoder"], _P["embedding"], c)
    ids = np.load(os.path.join(_DATA, "matcha_ref_00.npz"))["ids0"].tolist()
    m1 = a.synthesize(ids).mel
    m2 = a.synthesize(ids).mel        # RNG has moved on: different noise
    a.seed(7)
    m3 = a.synthesize(ids).mel        # back to the first draw
    np.testing.assert_array_equal(m1, m3)
    assert np.abs(m1 - m2).mean() > 1e-3


def test_too_long_is_truncated_and_flagged(am):
    fe = bcdl.MatchaFrontend(_P["tokens"], _P["lexicon"])
    one = fe.encode("我")[0][0]
    r = am.synthesize([one] * 200)                      # > 128 tokens
    assert r.truncated and len(r.durations) == 128 and r.frames <= 512
    r = am.synthesize([one] * 120)                      # fits L, but ~ 120 x 5 frames > 512
    if sum(r.durations) > 512:
        assert r.truncated and r.frames == 512


def test_empty_and_bad_input(am, tts):
    r = am.synthesize([])
    assert r.frames == 0 and r.mel.shape == (0, 0)
    out = tts.synthesize("，。")
    assert out.sentences == 0 and out.pcm.shape == (0,)


def test_text_to_speech_round_trip(tts):
    paths = {k: os.path.join(_MODELS, v) for k, v in _PARA.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip("Paraformer build missing")
    asr = bcdl.ParaformerAsr.from_files(paths["encoder"], paths["predictor"], paths["decoder"],
                                        paths["tokens"], paths["cmvn"])
    for text in ("好的，马上出发。", "晚上好，现在是北京时间八点二十分。", "前方三米处有一个纸箱，我会绕开它。"):
        t0 = time.perf_counter()
        out = tts.synthesize(text)
        dt = time.perf_counter() - t0
        sec = len(out.pcm) / out.sample_rate
        heard = asr.transcribe(out.pcm).text
        want = "".join(c for c in text if c.isalnum())
        print(f"\n  '{text}': {out.sentences} sentence(s), {sec:.2f} s audio in {dt * 1000:.0f} ms "
              f"(RTF {dt / sec:.3f}) -> '{heard}'")
        assert out.sample_rate == 16000 and not out.truncated
        assert heard == want


def test_wrong_models_refused():
    with pytest.raises(Exception, match="MatchaAcoustic"):
        bcdl.MatchaAcoustic(_P["vocoder"], _P["decoder"], _P["embedding"])
    with pytest.raises(Exception, match="not a .npy"):
        bcdl.MatchaAcoustic(_P["encoder"], _P["decoder"], _P["tokens"])


def test_bad_token_id_refused(am):
    with pytest.raises(Exception, match="outside"):
        am.synthesize([5, am.vocab_size])


def test_decoder_inputs_are_named():
    # mu_y and z share a shape; MatchaAcoustic tells them apart by these names.
    dec = bcdl.Engine(_P["decoder"])
    assert sorted(dec.input_name(i) for i in range(dec.num_inputs)) == ["mu_y", "y_mask", "z"]
    assert dec.output_name(0) == "mel"
    with pytest.raises(Exception):
        dec.input_name(dec.num_inputs)
