"""On-board voice activity detection test: Silero VAD v6 (int16) + VadSegmenter.

The test signal is two AISHELL-1 clips from data/audio with silence around
and between them. The BPU probabilities are checked against the float model's
(data/vad/silero_ref.npz, same signal, same chunking); the segmenter against a
plain re-implementation of its rule over those probabilities; and each segment
it cuts must transcribe to its clip's text through Paraformer.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_vad_board_py.py

Models from scripts/fetch_models.sh. Env-overridable: BCDL_MODELS / BCDL_AUDIO / BCDL_VAD_DATA.
"""
import os
import time
import wave

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.environ.get("BCDL_MODELS", os.path.join(_REPO, "models"))
_AUDIO = os.environ.get("BCDL_AUDIO", os.path.join(_REPO, "data", "audio"))
_DATA = os.environ.get("BCDL_VAD_DATA", os.path.join(_REPO, "data", "vad"))
_VAD = os.path.join(_MODELS, "silero_vad_16k_int16_nashm.hbm")
_PARA = {
    "encoder": "paraformer_encoder_int16.hbm",
    "predictor": "paraformer_predictor_int16_v2.hbm",
    "decoder": "paraformer_decoder_int16_v2.hbm",
    "tokens": "paraformer_tokens_8404.json",
    "cmvn": "paraformer_am.mvn",
}

pytestmark = pytest.mark.skipif(not os.path.exists(_VAD),
                                reason="Silero VAD build missing (scripts/fetch_models.sh)")

SR = 16000
CHUNK = 512


def _read(name):
    with wave.open(os.path.join(_AUDIO, name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def _texts():
    with open(os.path.join(_AUDIO, "transcripts.tsv"), encoding="utf-8") as f:
        return dict(line.rstrip("\n").split("\t") for line in f)


def _signal():
    """silence 0.5 s + clip 1 + 1.2 s + clip 3 + 1.2 s; also the clips' spans."""
    a, b = _read("aishell1_test_1.wav"), _read("aishell1_test_3.wav")
    z = lambda s: np.zeros(int(s * SR), np.float32)
    parts = [z(0.5), a, z(1.2), b, z(1.2)]
    x = np.concatenate(parts)
    s1 = len(parts[0])
    s2 = s1 + len(a) + len(parts[2])
    return x, [(s1, s1 + len(a), "aishell1_test_1.wav"), (s2, s2 + len(b), "aishell1_test_3.wav")]


@pytest.fixture(scope="module")
def vad():
    return bcdl.SileroVad(_VAD)


def test_matches_float(vad):
    x, _ = _signal()
    t0 = time.perf_counter()
    p = vad.probabilities(x)
    dt = (time.perf_counter() - t0) / len(p)
    ref = np.load(os.path.join(_DATA, "silero_ref.npz"))["probs"]
    assert p.shape == ref.shape
    d = np.abs(p - ref)
    agree = np.mean((p >= 0.5) == (ref >= 0.5))
    print(f"\n  {len(p)} chunks, {dt * 1000:.3f} ms each; |BPU - float| mean {d.mean():.4f} "
          f"max {d.max():.4f}; decisions agree {agree * 100:.1f}%")
    assert d.mean() < 0.02 and agree >= 0.98


def test_reset_restarts_the_stream(vad):
    x, _ = _signal()
    np.testing.assert_array_equal(vad.probabilities(x[:CHUNK * 40]),
                                  vad.probabilities(x[:CHUNK * 40]))


def _rule(p, thr, min_speech, min_silence):
    """The segmenter's rule in plain Python (chunk counts) -> [(start, end)] in samples."""
    segs, trig, run_sp, run_sil, st = [], False, 0, 0, 0
    for i, v in enumerate(p):
        if not trig:
            run_sp = run_sp + 1 if v >= thr else 0
            if run_sp >= min_speech:
                trig, run_sil, st = True, 0, (i + 1 - run_sp) * CHUNK
        elif v >= thr:
            run_sil = 0
        else:
            run_sil += 1
            if run_sil >= min_silence:
                segs.append((st, (i + 1 - run_sil) * CHUNK))
                trig, run_sp, run_sil = False, 0, 0
    if trig:
        segs.append((st, len(p) * CHUNK))
    return segs


def test_segmenter_follows_its_rule(vad):
    x, _ = _signal()
    p = vad.probabilities(x)
    cfg = bcdl.VadConfig()
    cfg.speech_pad_ms = 0
    got = [(s.start, s.end) for s in bcdl.VadSegmenter(vad, cfg).segments(x)]
    assert got == _rule(p, 0.5, 8, 10)          # 250 ms -> 8 chunks, 300 ms -> 10
    assert len(got) == 2


def test_segments_cover_the_speech_and_transcribe(vad):
    x, spans = _signal()
    segs = bcdl.VadSegmenter(vad).segments(x)
    assert len(segs) == 2
    for s, (a, b, _) in zip(segs, spans):
        # Each segment lies within its clip, give or take the padding and the
        # silence it takes to close (the clips carry ~0.5 s of their own
        # leading silence); that it holds the whole sentence is what the
        # transcription below checks.
        assert a - SR // 10 <= s.start and s.end <= b + 4 * SR // 10
        assert s.end - s.start == len(s.pcm)
        np.testing.assert_array_equal(s.pcm, x[s.start:s.end])
    paths = {k: os.path.join(_MODELS, v) for k, v in _PARA.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip("Paraformer build missing")
    asr = bcdl.ParaformerAsr.from_files(**paths)
    texts = _texts()
    for s, (_, _, name) in zip(segs, spans):
        heard = asr.transcribe(s.pcm).text
        print(f"\n  [{s.start / SR:.2f}, {s.end / SR:.2f}] s -> {heard!r}")
        assert heard == texts[name]


def test_streaming_matches_whole_recording(vad):
    x, _ = _signal()
    whole = [(s.start, s.end) for s in bcdl.VadSegmenter(vad).segments(x)]
    seg = bcdl.VadSegmenter(vad)
    rng = np.random.default_rng(0)
    got, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 3000))
        got += [(s.start, s.end) for s in seg.feed(x[i:i + n])]
        i += n
    got += [(s.start, s.end) for s in seg.flush()]
    assert got == whole
    # int16 input is rescaled, not cast
    i16 = [(s.start, s.end) for s in seg.segments((x * 32767).astype(np.int16))]
    assert i16 == whole


def test_max_speech_cuts_long_segments(vad):
    x, _ = _signal()
    cfg = bcdl.VadConfig()
    cfg.max_speech_s = 1.0
    segs = bcdl.VadSegmenter(vad, cfg).segments(x)
    assert len(segs) > 2
    assert all(s.end - s.start <= SR + CHUNK + 2 * 480 for s in segs)


def test_silence_gives_nothing(vad):
    seg = bcdl.VadSegmenter(vad)
    assert seg.segments(np.zeros(SR * 3, np.float32)) == []
    assert not seg.in_speech


def test_bad_input_refused(vad):
    with pytest.raises(Exception, match="512"):
        vad.probability(np.zeros(100, np.float32))
    cfg = bcdl.VadConfig()
    cfg.threshold = 1.5
    with pytest.raises(Exception, match="threshold"):
        bcdl.VadSegmenter(vad, cfg)
    cfg = bcdl.VadConfig()
    cfg.neg_threshold = 0.9
    with pytest.raises(Exception, match="neg_threshold"):
        bcdl.VadSegmenter(vad, cfg)


def test_wrong_model_refused():
    voc = os.path.join(_MODELS, "vocos_16k_t256_int16_nashm.hbm")
    if not os.path.exists(voc):
        pytest.skip("no other build to try")
    with pytest.raises(Exception, match="SileroVad"):
        bcdl.SileroVad(voc)
