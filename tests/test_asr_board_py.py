"""On-board speech recognition test: ParaformerAsr on the real three-stage build.

The four clips in data/audio are AISHELL-1 test utterances, and the shipped
build transcribes every one of them exactly — so the gate is exact match, not a
CER threshold that a regression could hide under. Clips 2-4 were picked because
their last character only fires through the CIF tail weight: with the tail
disabled — which is what the fixed-shape build's own graph effectively does —
they come back without it, and `test_tail_weight_is_what_keeps_the_last_character`
pins that, so a CIF that loses the tail fails here rather than in the field.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_asr_board_py.py

Models from scripts/fetch_models.sh. Env-overridable: BCDL_MODELS / BCDL_AUDIO.
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
_FILES = {
    "encoder": "paraformer_encoder_int16.hbm",
    "predictor": "paraformer_predictor_int16_v2.hbm",
    "decoder": "paraformer_decoder_int16_v2.hbm",
    "tokens": "paraformer_tokens_8404.json",
    "cmvn": "paraformer_am.mvn",
}
_PATHS = {k: os.path.join(_MODELS, v) for k, v in _FILES.items()}

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in _PATHS.values()),
    reason="Paraformer build missing (scripts/fetch_models.sh)")


def _read(path):
    with wave.open(path) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def _clips():
    out = []
    with open(os.path.join(_AUDIO, "transcripts.tsv"), encoding="utf-8") as f:
        for line in f:
            name, text = line.rstrip("\n").split("\t")
            out.append((name, _read(os.path.join(_AUDIO, name)), text))
    return out


@pytest.fixture(scope="module")
def asr():
    return bcdl.ParaformerAsr.from_files(**_PATHS)


def test_geometry(asr):
    assert asr.feature_dim == 560
    assert asr.max_frames == 400
    assert asr.max_seconds == pytest.approx(24.0)
    assert asr.max_tokens == 100
    assert asr.vocab_size == 8404


def test_transcribes_the_samples_exactly(asr):
    lat = []
    for name, pcm, text in _clips():
        t0 = time.perf_counter()
        r = asr.transcribe(pcm)
        lat.append((time.perf_counter() - t0) * 1000)
        print(f"  {name}: {r.text!r}  ({lat[-1]:.1f} ms)")
        assert not r.truncated
        assert r.text == text, f"{name}: got {r.text!r}, want {text!r}"
    print(f"  median {np.median(lat):.1f} ms per utterance (front end + 3 BPU stages + CIF)")


def test_tail_weight_is_what_keeps_the_last_character():
    cfg = bcdl.ParaformerConfig()
    cif = bcdl.CifConfig()
    cif.tail_threshold = 0.0
    cfg.cif = cif
    no_tail = bcdl.ParaformerAsr.from_files(**_PATHS, config=cfg)
    for name, pcm, text in _clips()[1:]:
        got = no_tail.transcribe(pcm).text
        print(f"  {name} without the tail: {got!r} (want {text!r})")
        assert got != text and not got.endswith(text[-1])


def test_int16_pcm_is_accepted(asr):
    name, pcm, text = _clips()[0]
    assert asr.transcribe((pcm * 32768).astype(np.int16)).text == text


def test_features_path_is_the_same_pipeline(asr):
    _, pcm, text = _clips()[1]
    feats = asr.features(pcm)
    assert feats.shape[1] == 560
    assert asr.transcribe_features(feats).text == asr.transcribe(pcm).text == text


def test_long_audio_is_truncated_and_flagged(asr):
    pcm = np.concatenate([c[1] for c in _clips()] * 3)       # ~37 s, past the 24 s window
    r = asr.transcribe(pcm)
    assert r.truncated and r.feature_frames > asr.max_frames
    assert len(r.text) > 0
    # transcribe() only computes the front end for the window it can use;
    # the result must equal running the whole recording's features.
    full = asr.features(pcm)
    assert r.feature_frames == full.shape[0]
    assert asr.transcribe_features(full).text == r.text


def test_silence_and_too_short_input(asr):
    assert asr.transcribe(np.zeros(100, np.float32)).text == ""   # shorter than one frame
    r = asr.transcribe(np.zeros(16000, np.float32))
    assert not r.truncated and len(r.text) <= 1                     # nothing to say


def test_wrong_token_list_is_refused():
    e, p, d = (bcdl.Engine(_PATHS[k]) for k in ("encoder", "predictor", "decoder"))
    with pytest.raises(Exception):
        bcdl.ParaformerAsr(e, p, d, ["<blank>", "a"], bcdl.load_kaldi_cmvn(_PATHS["cmvn"]))


def test_int16_features_match_float(asr):
    _, pcm, _ = _clips()[0]
    a = asr.features(pcm)
    b = asr.features((pcm * 32768).astype(np.int16))
    assert np.abs(a - b).max() < 1e-2


def test_max_tokens_caps_the_output():
    cfg = bcdl.ParaformerConfig()
    cif = bcdl.CifConfig()
    cif.max_tokens = 3
    cfg.cif = cif
    capped = bcdl.ParaformerAsr.from_files(**_PATHS, config=cfg)
    _, pcm, text = _clips()[1]
    assert len(text) > 3
    got = capped.transcribe(pcm).text
    assert len(got) <= 3
