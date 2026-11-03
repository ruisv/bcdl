"""CommandMatcher / to_pinyin — pure CPU, no model needed.

The examples are real Paraformer misrecognitions from a 40-command set.
"""
import random

import pytest

import bcdl

COMMANDS = [
    "停下", "前进", "后退", "左转", "右转", "掉头", "加速", "减速", "停车", "出发",
    "跟着我", "别跟了", "过来", "回去充电", "去厨房", "去客厅", "去卧室", "回到起点",
    "向左转九十度", "往前走两米", "往后退一米", "原地转一圈", "慢一点", "快一点",
    "打开灯", "关灯", "拍张照片", "开始巡逻", "停止巡逻", "看看前面有什么",
]


def ref_edit(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


# --- to_pinyin --------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("左转", ["zuo", "zhuan"]),
    ("绿灯", ["lv", "deng"]),                 # ü is written v
    ("打开灯！", ["da", "kai", "deng"]),        # punctuation dropped
    ("OK，好的", ["ok", "hao", "de"]),          # ASCII word lower-cased, one token
    ("ＯＫ", ["ok"]),                           # full-width folded
    ("〇", ["ling"]),
    ("  ", []),
    ("", []),
    ("abc123", ["abc", "yi", "bai", "er", "shi", "san"]),  # letter run, then digit run
])
def test_to_pinyin(text, expected):
    assert bcdl.to_pinyin(text) == expected


@pytest.mark.parametrize("digits,spoken", [
    ("0", "零"), ("2", "二"), ("10", "十"), ("15", "十五"), ("20", "二十"),
    ("90", "九十"), ("100", "一百"), ("105", "一百零五"), ("110", "一百一十"),
    ("1001", "一千零一"), ("1010", "一千零一十"), ("10000", "一万"),
    ("100010", "十万零一十"), ("12345678", "一千二百三十四万五千六百七十八"),
    ("007", "零零七"),                       # leading zero: digit by digit
    ("123456789", "一二三四五六七八九"),       # > 8 digits: digit by digit
])
def test_numbers_read_as_spoken(digits, spoken):
    assert bcdl.to_pinyin(digits) == bcdl.to_pinyin(spoken)


def test_digits_match_chinese_numerals():
    assert bcdl.to_pinyin("转90度") == bcdl.to_pinyin("转九十度")


def test_replacement_and_other_scripts_are_dropped():
    assert bcdl.to_pinyin("左\ufffd转 é Ω") == ["zuo", "zhuan"]


# --- CommandMatcher ---------------------------------------------------------

@pytest.fixture(scope="module")
def m():
    return bcdl.CommandMatcher(COMMANDS)


@pytest.mark.parametrize("heard,meant", [
    ("打开灯", "打开灯"),
    ("大开灯", "打开灯"),        # homophones: distance 0
    ("处罚", "出发"),
    ("灌灯", "关灯"),
    ("原地转移权", "原地转一圈"),
    ("前进了", "前进"),          # trailing particle: 1 of 2 syllables
    ("往后退一名", "往后退一米"),
    ("去出发", "去厨房"),        # nearer 去厨房 (1/3) than 出发 (1/2)
    ("往前走2米", "往前走两米"),  # digit read as er, one syllable off
])
def test_real_misrecognitions_map_back(m, heard, meant):
    r = m.match(heard)
    assert r.command == meant, r
    assert r.index == COMMANDS.index(meant)
    assert r


def test_homophone_distance_is_zero(m):
    assert m.match("大开灯").distance == 0.0


@pytest.mark.parametrize("text", [
    "今天天气怎么样",
    "这是一个关于经济发展的新闻报道",
    "田径",   # real miss for 前进: tian jing shares no syllable with qian jin
])
def test_non_commands_are_rejected(m, text):
    r = m.match(text)
    assert r.index == -1 and r.command == "" and not r
    assert r.nearest >= 0 and r.distance > m.max_distance


def test_empty_text(m):
    r = m.match("，。 ")
    assert r.index == -1 and r.nearest == -1 and not r
    assert all(d >= 1e29 for d in m.distances(""))


def test_threshold_is_inclusive():
    m = bcdl.CommandMatcher(["前进"], max_distance=0.5)
    assert m.match("前进了").distance == 0.5 and m.match("前进了")
    assert not bcdl.CommandMatcher(["前进"], max_distance=0.49).match("前进了")


def test_tie_breaks_on_length_then_order():
    # 转 is 1/2 from both and equally long: the earlier command wins.
    assert bcdl.CommandMatcher(["左转", "右转"], max_distance=1.0).match("转").index == 0
    # Equal distance (1/2), different length gap: the closer length wins.
    m = bcdl.CommandMatcher(["前进了吗吗吗", "前进"], max_distance=1.0)  # 3/6 vs 1/2
    assert m.distances("前进了") == pytest.approx([0.5, 0.5])
    assert m.match("前进了").index == 1
    # A lower distance beats order and length.
    assert bcdl.CommandMatcher(["开灯", "开灯吧"], max_distance=1.0).match("开灯呀").index == 1


def test_distances_match_reference(m):
    rng = random.Random(0)
    pool = "".join(COMMANDS) + "的了吗呢我你他好"
    for _ in range(200):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(1, 8)))
        t = bcdl.to_pinyin(text)
        ref = [ref_edit(t, bcdl.to_pinyin(c)) / len(bcdl.to_pinyin(c)) for c in COMMANDS]
        got = m.distances(text)
        assert got == pytest.approx(ref, abs=1e-6)
        r = m.match(text)
        best = min(range(len(COMMANDS)),
                   key=lambda i: (ref[i], abs(len(t) - len(bcdl.to_pinyin(COMMANDS[i])))))
        assert r.nearest == best
        assert (r.index == best) == (ref[best] <= 0.5)


def test_latin_words_never_equal_pinyin(m):
    # Real miss: 掉头 recognised as English "you too". "you" is not 右.
    assert bcdl.to_pinyin("you too") == ["you", "too"]
    assert not m.match("you too")
    assert m.distances("you too")[COMMANDS.index("右转")] == 1.0
    mixed = bcdl.CommandMatcher(["打开wifi"], max_distance=0.0)
    assert mixed.match("打开 WiFi").index == 0
    assert not mixed.match("打开歪fai")


def test_unknown_tokens_never_match():
    m = bcdl.CommandMatcher(["ok"], max_distance=0.0)
    assert m.match("OK").index == 0
    assert m.match("no").index == -1


def test_properties(m):
    assert len(m) == len(COMMANDS)
    assert m.commands == COMMANDS
    assert m.max_distance == pytest.approx(0.5)


def test_construction_errors():
    with pytest.raises(Exception, match="empty"):
        bcdl.CommandMatcher([])
    with pytest.raises(Exception, match="no pinyin"):
        bcdl.CommandMatcher(["前进", "！？"])
    with pytest.raises(Exception, match="max_distance"):
        bcdl.CommandMatcher(["前进"], max_distance=-0.1)
