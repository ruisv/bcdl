"""The API reference must only promise things that exist.

Shipped broken once, in 0.7.1: `output_packed_bytes()` was added to the nanobind
binding and listed in `docs/API.md`'s `Engine` member table, right beside the
`input_packed_bytes()` that *is* forwarded — but the Python wrapper class never
forwarded it, so a user following the documentation got an `AttributeError`. It
survived the whole test suite because every test reached through the private
`engine._e` handle rather than the public class: the binding was covered, and
the surface people actually call was not.

The Python half of that hole is closed in `test_engine_output_stride.py`, which
asserts the documented members exist on the wrapper. This file closes the C++
half, where the same mistake is quieter still — a documented-but-absent member
is only discovered by whoever tries to compile against it.

Needs no board and no model: it reads the docs and the headers.

    pytest -s tests/test_api_docs_match_code.py
"""
import glob
import os
import re

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = {"en": os.path.join(_REPO, "docs", "CPP_API.md"),
        "zh": os.path.join(_REPO, "docs", "CPP_API.zh.md")}


def _headers():
    """Every public header, concatenated. The public surface is what ships in
    include/; anything not declared there cannot be called by a user."""
    files = glob.glob(os.path.join(_REPO, "include", "bcdl", "**", "*.h"), recursive=True)
    assert files, "no public headers found — the path assumption is wrong"
    return "\n".join(open(f, encoding="utf-8").read() for f in files)


def _documented_functions(path):
    """Identifiers the doc presents as callable.

    Two sources, both of which are *promises* rather than prose: the first column
    of a member table, and the `// pure:` lines that enumerate the free
    functions. Deliberately NOT every `word(` in the file — prose ("bound (…)"),
    example locals (`det`, `enc`) and system calls (`printf`) would drown the
    signal, and a check nobody trusts gets deleted.
    """
    names = set()
    for line in open(path, encoding="utf-8").read().splitlines():
        cell = re.match(r"^\|\s*(`[^|]+`)\s*\|", line)
        if cell:
            for ident in re.findall(r"`([A-Za-z_][A-Za-z0-9_:]*)\s*\(", cell.group(1)):
                names.add(ident.split("::")[-1])
        marker = re.search(r"//\s*(?:pure|纯函数)\s*[:：]", line)
        if marker:
            # Split on the marker, not on ":" — the Chinese doc writes a
            # FULL-WIDTH colon, and splitting on the ASCII one raised IndexError
            # on every line of it.
            for ident in re.findall(r"\b([a-z][A-Za-z0-9_]+)\s*\(", line[marker.end():]):
                names.add(ident)
    return names


def _documented_types(path):
    return set(re.findall(r"\bbcdl::([A-Z][A-Za-z0-9_]+)",
                          open(path, encoding="utf-8").read()))


@pytest.mark.parametrize("lang", sorted(DOCS))
def test_every_documented_member_exists(lang):
    hdr = _headers()
    documented = _documented_functions(DOCS[lang])
    # If the extractor silently matches nothing, the test passes while checking
    # nothing — the failure mode this whole file exists to prevent.
    assert len(documented) > 25, "extracted only %d members; the doc format changed" % len(documented)
    missing = sorted(n for n in documented
                     if not re.search(r"\b%s\s*\(" % re.escape(n), hdr))
    assert not missing, ("%s promises members that no public header declares: %s"
                         % (os.path.basename(DOCS[lang]), missing))


@pytest.mark.parametrize("lang", sorted(DOCS))
def test_every_documented_type_exists(lang):
    hdr = _headers()
    missing = sorted(
        t for t in _documented_types(DOCS[lang])
        if not re.search(r"\b(class|struct|using|enum\s+class)\s+%s\b" % re.escape(t), hdr))
    assert not missing, ("%s references bcdl:: types that do not exist: %s"
                         % (os.path.basename(DOCS[lang]), missing))


def test_the_two_language_docs_promise_the_same_api():
    """A translation that drifts is worse than no translation: the reader of one
    is told about an API the reader of the other is not."""
    en, zh = _documented_functions(DOCS["en"]), _documented_functions(DOCS["zh"])
    assert not (en - zh), "documented in CPP_API.md but not in CPP_API.zh.md: %s" % sorted(en - zh)
    assert not (zh - en), "documented in CPP_API.zh.md but not in CPP_API.md: %s" % sorted(zh - en)


def test_the_check_would_catch_a_broken_promise(tmp_path):
    """The guard on the guard. A member-existence check that cannot fail proves
    nothing, so break a copy of the doc on purpose and confirm it is caught —
    the same reason the padded-output test asserts the flat read DIFFERS."""
    broken = tmp_path / "CPP_API.md"
    text = open(DOCS["en"], encoding="utf-8").read()
    assert "| `modelName()` |" in text, "the doc no longer has the row this test breaks"
    broken.write_text(text.replace("| `modelName()` |",
                                   "| `modelNaem()` |\n| `outputAlignedBytes(i)` | invented |",
                                   1), encoding="utf-8")
    hdr = _headers()
    missing = sorted(n for n in _documented_functions(str(broken))
                     if not re.search(r"\b%s\s*\(" % re.escape(n), hdr))
    assert missing == ["modelNaem", "outputAlignedBytes"], missing


def test_umbrella_header_includes_every_public_header():
    """`docs/CPP_API.md` tells users that including `bcdl/bcdl.h` is enough.

    It was not: the umbrella carried 27 of the 45 installed headers, so a user
    following the doc got `undeclared identifier` for TrackingPipeline,
    outputAsFloat, AnomalyDetector and 15 more. The examples hid it by including
    sub-headers directly. A new header under include/ ships automatically, so
    nothing but this test notices when it is not re-exported.
    """
    inc = os.path.join(_REPO, "include")
    umbrella = os.path.join(inc, "bcdl", "bcdl.h")
    installed = {
        os.path.relpath(f, inc).replace(os.sep, "/")
        for f in glob.glob(os.path.join(inc, "bcdl", "**", "*.h"), recursive=True)
    } - {"bcdl/bcdl.h"}
    assert installed, "no public headers found — the path assumption is wrong"

    included = set(re.findall(r'#include\s+"(bcdl/[^"]+)"',
                              open(umbrella, encoding="utf-8").read()))
    missing = sorted(installed - included)
    assert not missing, "bcdl/bcdl.h does not re-export: %s" % missing
    stale = sorted(included - installed)
    assert not stale, "bcdl/bcdl.h includes headers that no longer exist: %s" % stale
