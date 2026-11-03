"""scale_silence — pause shortening for TTS output (sherpa-onnx's silence_scale).
Pure CPU; no model."""
import numpy as np
import pytest

bcdl = pytest.importorskip("bcdl")

SR = 16000


def _sherpa(samples, sr, scale):
    """sherpa-onnx GeneratedAudio::ScaleSilence, transcribed."""
    if scale == 1 or not (0.01 <= scale <= 10):
        return samples.copy()
    threshold = int(sr * 0.2)
    intervals, last, n = [], -1, len(samples)
    for i in range(n):
        if abs(samples[i]) <= 0.01:
            if last == -1:
                last = i
            continue
        if last != -1 and i - last < threshold:
            last = -1
            continue
        if last != -1:
            intervals.append((last, i))
            last = -1
    if last != -1 and n - last > threshold:
        intervals.append((last, n))
    out, i = [], 0
    for s, e in intervals:
        out.append(samples[i:s])
        i = e
        k = int((e - s) * scale)
        out.append(samples[s:s + min(k, e - s)])
        if k > e - s:
            out.append(np.zeros(k - (e - s), samples.dtype))
    out.append(samples[i:])
    return np.concatenate(out) if out else samples[:0]


def _tone(sec, amp=0.3, f=220.0):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t) + 0.05).astype(np.float32)   # never quiet


def _signal():
    z = lambda s: np.zeros(int(s * SR), np.float32)
    # short gap (kept), long gap (shortened), exactly-threshold gap, trailing silence
    return np.concatenate([z(0.5), _tone(0.4), z(0.1), _tone(0.3), z(1.0), _tone(0.2),
                           z(0.2), _tone(0.2), z(0.6)])


@pytest.mark.parametrize("scale", [0.2, 0.5, 1.0, 2.0, 0.001, 20.0])
def test_matches_sherpa(scale):
    x = _signal()
    np.testing.assert_array_equal(bcdl.scale_silence(x, SR, scale), _sherpa(x, SR, scale))


def test_shortens_only_long_pauses():
    x = _signal()
    y = bcdl.scale_silence(x, SR, 0.2)
    # pauses of 0.5 + 1.0 + 0.2 + 0.6 s become 0.2 of that; the 0.1 s one stays
    assert len(x) - len(y) == pytest.approx(0.8 * (0.5 + 1.0 + 0.2 + 0.6) * SR, abs=4)
    assert bcdl.scale_silence(np.zeros(0, np.float32), SR, 0.2).shape == (0,)
