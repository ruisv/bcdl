"""On-board vocoder test: VocosVocoder on the int16 Vocos build.

The gate is the float model's own waveform for the same mel (data/tts), compared
where the ear can hear it: the log-spectral distance over time-frequency cells
within 40 dB of the utterance's peak. A waveform SNR would be meaningless — any
phase difference sinks it while sounding identical. Across 60 sentences the
int16 build is 0.54 dB from float on that measure, and the int8 build — which
sounds noisy — 5.4 dB, so the threshold sits well between them.

The synthesized speech is also run back through Paraformer, which must
recognise the sentence.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_vocoder_board_py.py

Models from scripts/fetch_models.sh. Env-overridable: BCDL_MODELS / BCDL_TTS_DATA.
"""
import os
import time
import wave

import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.environ.get("BCDL_MODELS", os.path.join(_REPO, "models"))
_DATA = os.environ.get("BCDL_TTS_DATA", os.path.join(_REPO, "data", "tts"))
_VOCOS = os.path.join(_MODELS, "vocos_16k_t256_int16_nashm.hbm")
_PARA = {
    "encoder": "paraformer_encoder_int16.hbm",
    "predictor": "paraformer_predictor_int16_v2.hbm",
    "decoder": "paraformer_decoder_int16_v2.hbm",
    "tokens": "paraformer_tokens_8404.json",
    "cmvn": "paraformer_am.mvn",
}

pytestmark = pytest.mark.skipif(not os.path.exists(_VOCOS),
                                reason="Vocos build missing (scripts/fetch_models.sh)")

LSD_MAX_DB = 1.5


def _texts():
    with open(os.path.join(_DATA, "texts.tsv"), encoding="utf-8") as f:
        return dict(l.rstrip("\n").split("\t") for l in f if "\t" in l)


def _wav(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


def _logspec(x, n=1024, hop=256):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    xp = np.pad(x, n // 2, mode="reflect")
    frames = 1 + (len(xp) - n) // hop
    S = np.stack([np.fft.rfft(xp[t * hop:t * hop + n] * w) for t in range(frames)])
    return 10 * np.log10(np.abs(S) ** 2 + 1e-12)


def audible_lsd(ref, got):
    m = min(len(ref), len(got))
    A, B = _logspec(ref[:m]), _logspec(got[:m])
    mask = A > A.max() - 40
    return float(np.sqrt(np.mean((A - B)[mask] ** 2)))


@pytest.fixture(scope="module")
def voc():
    return bcdl.VocosVocoder(_VOCOS)


@pytest.fixture(scope="module")
def cases():
    return {k: (np.load(os.path.join(_DATA, k)),
                _wav(os.path.join(_DATA, k.replace("matcha_mel", "vocos_float").replace(".npy", ".wav"))),
                t) for k, t in _texts().items()}


def test_geometry(voc):
    assert (voc.num_mels, voc.chunk_frames, voc.body_frames, voc.sample_rate) == (80, 256, 192, 16000)


def test_matches_float_vocos(voc, cases):
    for name, (mel, ref, _) in cases.items():
        t0 = time.perf_counter()
        pcm = voc.synthesize(mel)
        dt = time.perf_counter() - t0
        assert pcm.dtype == np.float32 and pcm.shape == ((mel.shape[1] - 1) * 256,)
        lsd = audible_lsd(ref, pcm)
        print(f"\n  {name}: {mel.shape[1]} frames ({-(-mel.shape[1] // voc.body_frames)} chunk), "
              f"{len(pcm) / 16000:.2f} s audio in {dt * 1000:.1f} ms, audible LSD {lsd:.2f} dB")
        assert lsd < LSD_MAX_DB, name


def test_batch_dim_accepted(voc, cases):
    mel = cases["matcha_mel_00.npy"][0]
    np.testing.assert_array_equal(voc.synthesize(mel[None]), voc.synthesize(mel))


def test_round_trip_through_paraformer(voc, cases):
    paths = {k: os.path.join(_MODELS, v) for k, v in _PARA.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip("Paraformer build missing")
    asr = bcdl.ParaformerAsr.from_files(paths["encoder"], paths["predictor"], paths["decoder"],
                                        paths["tokens"], paths["cmvn"])
    for name, (mel, _, text) in cases.items():
        heard = asr.transcribe(voc.synthesize(mel)).text
        want = "".join(c for c in text if c.isalnum())
        print(f"\n  {name}: '{heard}'")
        assert heard == want, name


def test_silence_stays_quiet(voc):
    pcm = voc.synthesize(np.full((80, 300), -22.0, np.float32))
    assert np.sqrt(np.mean(pcm ** 2)) < 1e-3


def test_empty_and_bad_input(voc):
    assert voc.synthesize(np.zeros((80, 0), np.float32)).shape == (0,)
    with pytest.raises(ValueError):
        voc.synthesize(np.zeros((64, 10), np.float32))


def test_context_must_leave_frames():
    cfg = bcdl.VocosConfig()
    cfg.context = 128
    with pytest.raises(Exception, match="leaves no frames"):
        bcdl.VocosVocoder(_VOCOS, cfg)


def test_wrong_model_refused():
    other = os.path.join(_MODELS, _PARA["encoder"])
    if not os.path.exists(other):
        pytest.skip("no other model to try")
    with pytest.raises(Exception, match="VocosVocoder"):
        bcdl.VocosVocoder(other)
