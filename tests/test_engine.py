"""Smoke test for the BCDL Engine. Run on the board inside the `bcdl` conda env.

    cd ~/projects/bcdl
    PYTHONPATH=build:python pytest -s tests/test_engine.py --hbm /path/to/model.hbm
"""

import sys

import numpy as np
import pytest

import bcdl


def _hbm_path():
    # crude arg passthrough: pytest tests/test_engine.py --hbm model.hbm
    for i, a in enumerate(sys.argv):
        if a == "--hbm" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


@pytest.mark.skipif(_hbm_path() is None, reason="pass --hbm <model.hbm>")
def test_load_and_infer():
    engine = bcdl.Engine(_hbm_path())
    print(f"\nmodel: {engine.model_name}")
    print(f"inputs={engine.num_inputs} outputs={engine.num_outputs}")

    inputs = []
    for i in range(engine.num_inputs):
        # use the exact device-buffer size (handles NV12 / aligned strides)
        nbytes = engine.input_bytes(i)
        print(f"  in[{i}] shape={engine.input_shape(i)} bytes={nbytes}")
        inputs.append(np.zeros(nbytes, dtype=np.uint8))

    outs = engine.infer(inputs)
    assert len(outs) == engine.num_outputs
    for i, o in enumerate(outs):
        print(f"  out[{i}] shape={o.shape} dtype={o.dtype}")
        assert list(o.shape) == list(engine.output_shape(i))
