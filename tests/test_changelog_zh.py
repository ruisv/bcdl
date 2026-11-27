"""The Chinese changelog must track the English one, version for version.

`CHANGELOG.zh.md` is a translation of `CHANGELOG.md`, not a second history. The
way a second copy of a changelog dies is quietly: an entry lands in one file and
not the other, and nothing fails. So the two files must carry the same version
sections in the same order, and each section the same number of entries — a
translation may reword an entry, but it may not drop or merge one.

Needs no board and no model: it reads the two files.

    pytest -s tests/test_changelog_zh.py
"""
import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EN = os.path.join(_REPO, "CHANGELOG.md")
ZH = os.path.join(_REPO, "CHANGELOG.zh.md")

_VERSION = re.compile(r"^## \[([^\]]+)\]")
_ENTRY = re.compile(r"^\s*[-*] ")


def _sections(path):
    """[(version, entry count)] in file order. Entries inside code fences are
    not entries."""
    out = []
    fenced = False
    for line in open(path, encoding="utf-8"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = _VERSION.match(line)
        if m:
            out.append([m.group(1), 0])
        elif out and _ENTRY.match(line):
            out[-1][1] += 1
    return [tuple(s) for s in out]


def test_same_versions_in_the_same_order():
    en = [v for v, _ in _sections(EN)]
    zh = [v for v, _ in _sections(ZH)]
    assert en, "no version sections found in CHANGELOG.md — the heading format changed"
    assert en == zh, ("CHANGELOG.zh.md versions differ from CHANGELOG.md:\n  en %s\n  zh %s"
                      % (en, zh))


def test_same_number_of_entries_per_version():
    en, zh = dict(_sections(EN)), dict(_sections(ZH))
    drift = {v: (en[v], zh.get(v)) for v in en if en[v] != zh.get(v)}
    assert not drift, ("entries per version (en, zh) differ — translate the new entry "
                       "in the same commit: %s" % drift)


def test_the_check_would_catch_a_dropped_entry(tmp_path):
    """The guard on the guard: drop one entry from a copy and confirm the count
    moves, so a passing run means something."""
    text = open(ZH, encoding="utf-8").read().splitlines(keepends=True)
    first = next(i for i, l in enumerate(text) if _ENTRY.match(l))
    broken = tmp_path / "CHANGELOG.zh.md"
    broken.write_text("".join(text[:first] + text[first + 1:]), encoding="utf-8")
    assert _sections(str(broken)) != _sections(ZH)
