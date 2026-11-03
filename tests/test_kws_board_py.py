"""On-board keyword spotting test: zipformer2 KWS (zh-en 3M) encoder on the BPU,
decoder + joiner + keyword search on the CPU.

The encoder is checked against the float graph on an AISHELL-1 clip
(data/kws/kws_encoder_ref.npz). The wake words are synthesised on the board
with TextToSpeech, so the test needs no recorded wake-word audio, and the
AISHELL clips in data/audio must not fire.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_kws_board_py.py

Models from scripts/fetch_models.sh. Env-overridable: BCDL_MODELS / BCDL_AUDIO / BCDL_KWS_DATA.
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
_DATA = os.environ.get("BCDL_KWS_DATA", os.path.join(_REPO, "data", "kws"))
_ENC = os.path.join(_MODELS, "kws_zipformer_zh_en_3m_c16_int16_nashm.hbm")
_TABLE = os.path.join(_MODELS, "kws_zipformer_zh_en_3m_decoder_joiner.npy")
_TOKENS = os.path.join(_MODELS, "kws_zipformer_zh_en_3m_tokens.txt")
_TTS = {
    "encoder": "matcha_zh_en_enc_l128_int16_nashm.hbm",
    "decoder": "matcha_zh_en_dec_t512_int16_nashm.hbm",
    "embedding": "matcha_zh_en_embedding.npy",
    "vocoder": "vocos_16k_t256_int16_nashm.hbm",
    "tokens": "matcha_zh_en_tokens.txt",
    "lexicon": "matcha_zh_en_lexicon.txt",
}

pytestmark = pytest.mark.skipif(not all(os.path.exists(p) for p in (_ENC, _TABLE, _TOKENS)),
                                reason="KWS build missing (scripts/fetch_models.sh)")

KEYWORDS = "n ǐ h ǎo x iǎo ch ē @你好小车\nx iǎo ch ē x iǎo ch ē @小车小车\n"
SR = 16000


def _read(name):
    with wave.open(os.path.join(_AUDIO, name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


@pytest.fixture(scope="module")
def kws():
    return bcdl.KeywordSpotter(_ENC, _TABLE, _TOKENS, KEYWORDS)


@pytest.fixture(scope="module")
def tts():
    paths = {k: os.path.join(_MODELS, v) for k, v in _TTS.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip("Matcha/Vocos build missing")
    return bcdl.TextToSpeech.from_files(paths["encoder"], paths["decoder"], paths["embedding"],
                                        paths["vocoder"], paths["tokens"], paths["lexicon"])


def _cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_encoder_matches_float(kws):
    ref = np.load(os.path.join(_DATA, "kws_encoder_ref.npz"))["encoder_out"]
    t0 = time.perf_counter()
    got = kws.encode(_read("aishell1_test_1.wav"))
    dt = time.perf_counter() - t0
    n = min(len(ref), len(got))
    assert n >= len(ref) - 4
    cos = [_cos(ref[i], got[i]) for i in range(n)]
    print(f"\n  {n} frames in {dt * 1000:.0f} ms; per-frame cosine min {min(cos):.4f} "
          f"mean {np.mean(cos):.4f}; whole {_cos(ref[:n], got[:n]):.4f}")
    # Measured: per frame mean 0.995, min 0.973. The whole-array cosine is
    # lower (0.915) because some frames come out smaller overall, not turned:
    # the joiner's tanh and the log-softmax barely notice, and recall on the
    # full test sets matches float.
    assert np.mean(cos) > 0.99 and min(cos) > 0.95


def _wake(tts, text):
    pcm = tts.synthesize(text).pcm
    z = lambda s: np.zeros(int(s * SR), np.float32)
    return np.concatenate([z(0.5), pcm, z(1.0)]).astype(np.float32)


@pytest.mark.parametrize("text,want", [("你好小车。", "你好小车"), ("小车小车。", "小车小车")])
def test_detects_synthesised_wake_word(kws, tts, text, want):
    x = _wake(tts, text)
    kws.reset()
    t0 = time.perf_counter()
    got = kws.feed(x)
    dt = time.perf_counter() - t0
    print(f"\n  {text}: {got} ({dt * 1000:.0f} ms for {len(x) / SR:.1f} s, RTF {dt / (len(x) / SR):.3f})")
    assert [d.keyword for d in got] == [want]
    d = got[0]
    assert 0.4 < d.start < d.end < len(x) / SR and d.index == ("你好小车", "小车小车").index(want)
    assert len(d.tokens) == 8


def test_flush_catches_a_wake_word_at_the_very_end(kws, tts):
    x = _wake(tts, "你好小车。")
    x = x[: len(x) - int(1.0 * SR)]        # the clip stops right after the phrase
    kws.reset()
    early = kws.feed(x)
    late = kws.flush()
    print(f"\n  feed: {early}, flush: {late}")
    assert [d.keyword for d in early + late] == ["你好小车"]
    assert kws.flush() == []               # flushed streams are reset


def test_speech_does_not_fire(kws):
    for name in ("aishell1_test_1.wav", "aishell1_test_2.wav", "aishell1_test_3.wav",
                 "aishell1_test_4.wav"):
        kws.reset()
        assert kws.feed(_read(name)) == []


def test_streaming_matches_whole(kws, tts):
    x = np.concatenate([_wake(tts, "小车小车。"), _read("aishell1_test_2.wav"),
                        _wake(tts, "你好小车。")])
    kws.reset()
    whole = [(d.keyword, round(d.end, 2)) for d in kws.feed(x)]
    kws.reset()
    rng = np.random.default_rng(1)
    got, i = [], 0
    while i < len(x):
        n = int(rng.integers(1, 4000))
        got += [(d.keyword, round(d.end, 2)) for d in kws.feed(x[i:i + n])]
        i += n
    assert got == whole and [k for k, _ in whole] == ["小车小车", "你好小车"]


def test_keywords_parsing(kws):
    k = bcdl.KeywordSpotter(_ENC, _TABLE, _TOKENS)
    k.set_keywords("n ǐ h ǎo x iǎo ch ē :1.5 #0.3 @你好小车\n\nx iǎo ch ē x iǎo ch ē\n")
    assert k.keywords == ["你好小车", "x iǎo ch ē x iǎo ch ē"]
    with pytest.raises(Exception, match="not in the model's token table"):
        k.set_keywords("n ǐ h ǎo QQQ @x")
    with pytest.raises(Exception, match="no keywords"):
        k.set_keywords("\n\n")
    with pytest.raises(Exception, match="no keywords set"):
        bcdl.KeywordSpotter(_ENC, _TABLE, _TOKENS).feed(np.zeros(16000, np.float32))
    assert k.token_id("<blk>") == 0 and k.token_id("nope") == -1


def test_wrong_model_refused():
    vad = os.path.join(_MODELS, "silero_vad_16k_int16_nashm.hbm")
    if not os.path.exists(vad):
        pytest.skip("no other build to try")
    with pytest.raises(Exception, match="KeywordSpotter"):
        bcdl.KeywordSpotter(vad, _TABLE, _TOKENS)
    with pytest.raises(Exception, match="decoder_joiner"):
        bcdl.KeywordSpotter(_ENC, os.path.join(_MODELS, "matcha_zh_en_embedding.npy"), _TOKENS)
