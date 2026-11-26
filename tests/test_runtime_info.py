"""bcdl.runtime_info(): which hobot runtime the process actually loaded.

BCDL links against the board's own hobot SDK instead of bundling one. A bundled
copy shadows the board's: on a board whose OS ships a different SDK it runs that
copy against the board's firmware, and outputs come back shifted or NaN while
nothing raises. runtime_info() is how a user (or a bug report) tells which copy
was in play, so it has to name a real file and real versions.

Runs on the board.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_runtime_info.py
"""

import os

import pytest

bcdl = pytest.importorskip("bcdl")


def test_runtime_info_names_the_loaded_runtime():
    info = bcdl.runtime_info()
    print(info)
    assert set(info) == {"dnn_version", "ucp_version", "soc_name", "dnn_library"}
    assert info["dnn_version"] and info["ucp_version"]
    assert "RDK" in info["soc_name"]
    lib = info["dnn_library"]
    assert os.path.basename(lib).startswith("libdnn.so"), lib
    assert os.path.isfile(lib), lib


def test_libdnn_comes_from_the_board_not_a_bundled_copy():
    """No copy of libdnn inside a conda env may win over the board's. That is
    exactly what an SDK bundled into the package did, and it is invisible
    from the results alone."""
    lib = os.path.realpath(bcdl.runtime_info()["dnn_library"])
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        assert not lib.startswith(os.path.realpath(prefix) + os.sep), (
            f"libdnn loaded from the conda env ({lib}), shadowing the board's SDK")
