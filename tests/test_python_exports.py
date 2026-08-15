"""Everything the API reference calls ``bcdl.X`` has to be a public export.

0.7.0 added four task heads — optical flow, anomaly detection, end-to-end
driving and lidar 3-D detection — wired each one into the Python wrapper,
documented all four in `docs/API.md`, and left every one of them out of
``__all__``. Nothing raised. The classes are reachable as plain attributes, so
the examples, the board tests and the documented snippets all kept working; what
disagreed with the documentation was ``from bcdl import *`` and the type
checkers that read ``__all__`` as the public surface of a ``py.typed`` package.

This is the same defect the 0.7.2 fix chased on the Engine accessors and the
0.7.3 check closed on the C++ reference, in its cheapest-to-catch form: the doc
names a symbol, and the package either exports it or does not.

Two independent gates, because they fail in different situations. The first
reads the reference and needs the package importable — it is what a user hits.
The second is static and runs anywhere, so a wrapper class added without a
matching ``__all__`` entry is caught on the machine it was written on, before it
reaches a board at all.

Needs no board and no model.

    pytest -s tests/test_python_exports.py
"""
import ast
import os
import re

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DOC = os.path.join(_REPO, "docs", "API.md")
WRAPPER = os.path.join(_REPO, "python", "bcdl", "__init__.py")


def _documented_names():
    """Identifiers the reference presents as ``bcdl.Something``.

    Capitalised only: the lower-case attributes are free functions written in
    prose as often as in code, while a capitalised name after ``bcdl.`` is
    always a class or an enum the reader is being told to construct.
    """
    doc = open(API_DOC, encoding="utf-8").read()
    return sorted(set(re.findall(r"\bbcdl\.([A-Z][A-Za-z0-9_]*)", doc)))


def _wrapper_module():
    return ast.parse(open(WRAPPER, encoding="utf-8").read())


def test_every_documented_name_is_a_public_export():
    bcdl = pytest.importorskip("bcdl")
    names = _documented_names()
    assert len(names) > 30, "the doc scrape found almost nothing — the pattern is wrong"

    absent = [n for n in names if not hasattr(bcdl, n)]
    assert not absent, "docs/API.md promises names the package does not have: %s" % absent

    unexported = [n for n in names if n not in bcdl.__all__]
    assert not unexported, (
        "documented but missing from __all__, so `from bcdl import *` and a type "
        "checker disagree with the reference: %s" % unexported)


def test_every_public_wrapper_class_is_exported():
    """Static half: no board, no import, no compiled extension needed."""
    tree = _wrapper_module()
    defined = [n.name for n in tree.body
               if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and not n.name.startswith("_")]
    exported = [e.value for n in tree.body
                if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "__all__" for t in n.targets)
                for e in n.value.elts]
    assert exported, "__all__ is not a plain list of string literals any more"

    missing = [d for d in defined if d not in exported]
    assert not missing, "defined in the wrapper but absent from __all__: %s" % missing


def test_the_check_would_catch_a_broken_promise(tmp_path):
    """A gate that cannot fail proves nothing, so break one on purpose."""
    src = open(WRAPPER, encoding="utf-8").read()
    tree = ast.parse(src)
    exported = [e.value for n in tree.body
                if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "__all__" for t in n.targets)
                for e in n.value.elts]
    victim = next(d for d in (n.name for n in tree.body if isinstance(n, ast.ClassDef))
                  if d in exported)

    broken = tmp_path / "wrapper.py"
    broken.write_text(src.replace('    "%s",\n' % victim, "", 1), encoding="utf-8")
    tree = ast.parse(broken.read_text(encoding="utf-8"))
    still_exported = [e.value for n in tree.body
                      if isinstance(n, ast.Assign)
                      and any(getattr(t, "id", None) == "__all__" for t in n.targets)
                      for e in n.value.elts]
    assert victim not in still_exported, "the edit did not remove the entry it meant to"
