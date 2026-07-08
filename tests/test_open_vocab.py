"""Open-vocabulary LabelMap binding tests (bcdl.LabelMap via nanobind).

Exercises the class_id -> prompt-name table used by YOLOE prompt-free exports.
Detection decode itself is unchanged (reuse Detector / YoloLtrbDetector); only
the label semantics are new. Needs only the compiled bcdl extension.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_open_vocab.py
"""

import pytest

bcdl = pytest.importorskip("bcdl")


def _have():
    if not hasattr(bcdl, "LabelMap"):
        pytest.skip("LabelMap not exposed in this build")


def test_labelmap_from_list():
    _have()
    lm = bcdl.LabelMap.from_list(["person", "forklift", "pallet"])
    assert len(lm) == 3
    assert lm.name(0) == "person"
    assert lm.name(2) == "pallet"


def test_labelmap_out_of_range():
    _have()
    lm = bcdl.LabelMap.from_list(["a", "b"])
    assert lm.name(-1) == "?"
    assert lm.name(5) == "?"


def test_labelmap_from_file_trims_and_skips_blank(tmp_path):
    _have()
    p = tmp_path / "labels.txt"
    p.write_text("person\r\n  forklift  \n\n\tpallet\n")
    lm = bcdl.LabelMap.from_file(str(p))
    assert list(lm.names) == ["person", "forklift", "pallet"]  # trimmed, blanks dropped


def test_labelmap_from_file_missing_raises():
    _have()
    with pytest.raises(Exception):
        bcdl.LabelMap.from_file("/nonexistent/labels.txt")
