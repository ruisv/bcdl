"""MatchaFrontend — text -> Matcha-TTS token ids. CPU only; needs the model's
tokens.txt and lexicon.txt (scripts/fetch_models.sh), no BPU.

The reference is sherpa-onnx's own front end for this model, logged on 100
sentences (data/tts/frontend_ref.json): the ids must match exactly, because the
acoustic model was trained on what that front end produces.
"""
import json
import os

import pytest

bcdl = pytest.importorskip("bcdl")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.environ.get("BCDL_MODELS", os.path.join(_REPO, "models"))
_DATA = os.environ.get("BCDL_TTS_DATA", os.path.join(_REPO, "data", "tts"))
_TOKENS = os.path.join(_MODELS, "matcha_zh_en_tokens.txt")
_LEXICON = os.path.join(_MODELS, "matcha_zh_en_lexicon.txt")

pytestmark = pytest.mark.skipif(not (os.path.exists(_TOKENS) and os.path.exists(_LEXICON)),
                                reason="Matcha tokens/lexicon missing (scripts/fetch_models.sh)")


@pytest.fixture(scope="module")
def fe():
    return bcdl.MatchaFrontend(_TOKENS, _LEXICON)


def test_sizes(fe):
    assert fe.vocab_size == 2191 and fe.lexicon_size > 60000


def test_matches_sherpa_on_100_sentences(fe):
    with open(os.path.join(_DATA, "frontend_ref.json"), encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    assert len(cases) == 100
    bad = [c["text"] for c in cases if fe.encode(c["text"]) != c["ids"]]
    assert not bad, bad[:5]


@pytest.mark.parametrize("text,spoken", [
    ("90度", "九十度"),
    ("还剩35%", "还剩百分之三十五"),
    ("还剩35％", "还剩百分之三十五"),
    ("12.6伏", "十二点六伏"),
    ("2026年", "二零二六年"),
    ("1987年", "一九八七年"),
    ("第10个", "第十个"),
    ("105", "一百零五"),
    ("10010", "一万零一十"),
    ("0", "零"),
    ("007", "零零七"),
    ("13800138000", "一三八零零一三八零零零"),  # > 8 digits: digit by digit
    ("v1.5版", "v一点五版"),
    ("没有数字", "没有数字"),
])
def test_read_numbers(text, spoken):
    assert bcdl.MatchaFrontend.read_numbers(text) == spoken


def test_punctuation_folds_and_closes_sentences(fe):
    # Chinese punctuation becomes ASCII; the closing mark is the last token.
    ids = fe.encode("好的，马上出发。")
    assert len(ids) == 1
    assert ids[0][2] == 4 and ids[0][-1] == 5          # ',' and '.'
    assert fe.encode("好的,马上出发.") == ids


def test_short_pieces_merge_and_long_ones_split(fe):
    # "好的," alone is 2 words < 5, so it merges with the next piece.
    assert fe.sentences("好的，马上出发。") == ["好的,马上出发."]
    # 30 characters with no punctuation exceed max_words (20): split.
    long = "我" * 30
    s = fe.sentences(long)
    assert [len(x) for x in s] == [20, 10]
    cfg = bcdl.MatchaFrontendConfig()
    cfg.max_words = 40
    assert len(bcdl.MatchaFrontend(_TOKENS, _LEXICON, cfg).sentences(long)) == 1


def test_longest_match_reads_polyphones_in_context(fe):
    # 行 alone is xing2; in 银行 it must come from the phrase entry (hang2).
    alone = fe.encode("行")[0]
    in_word = fe.encode("银行")[0]
    assert len(in_word) == 2 and in_word[1] != alone[0]


def test_latin_is_dropped(fe):
    assert fe.encode("打开WiFi") == fe.encode("打开")
    assert fe.encode("hello world") == []


def test_empty_and_errors(fe):
    assert fe.encode("") == [] and fe.encode("，。！") == []
    with pytest.raises(Exception, match="cannot open"):
        bcdl.MatchaFrontend("/nonexistent/tokens.txt", _LEXICON)
    cfg = bcdl.MatchaFrontendConfig()
    cfg.min_words = 0
    with pytest.raises(Exception, match="min_words"):
        bcdl.MatchaFrontend(_TOKENS, _LEXICON, cfg)
