"""Istft — pure CPU, no model needed.

Checked two independent ways: against a numpy transcription of torch.istft,
and by perfect reconstruction of a signal from its own STFT.
"""
import numpy as np
import pytest

import bcdl

N, HOP = 1024, 256


def hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)  # periodic


def ref_istft(real, imag, n=N, hop=HOP, center=True):
    """torch.istft(onesided, hann, normalized=False), in numpy float64."""
    frames = real.shape[0]
    w = hann(n)
    full = n + (frames - 1) * hop
    acc = np.zeros(full)
    den = np.zeros(full)
    for t in range(frames):
        x = np.fft.irfft(real[t] + 1j * imag[t], n)  # ignores DC/Nyquist imag
        acc[t * hop:t * hop + n] += x * w
        den[t * hop:t * hop + n] += w * w
    out = np.where(den > 1e-11, acc / np.maximum(den, 1e-30), acc)
    return out[n // 2:full - n // 2] if center else out


def stft(x, n=N, hop=HOP):
    """Centred, reflect-padded STFT -> (frames, n/2+1) complex."""
    xp = np.pad(x, n // 2, mode="reflect")
    frames = 1 + (len(xp) - n) // hop
    w = hann(n)
    return np.stack([np.fft.rfft(xp[t * hop:t * hop + n] * w) for t in range(frames)])


def test_matches_reference_on_random_spectra():
    rng = np.random.default_rng(0)
    for frames in (1, 2, 7, 40):
        re = rng.standard_normal((frames, N // 2 + 1)).astype(np.float32)
        im = rng.standard_normal((frames, N // 2 + 1)).astype(np.float32)
        got = bcdl.Istft().compute(re, im)
        ref = ref_istft(re.astype(np.float64), im.astype(np.float64))
        assert got.shape == ref.shape == ((frames - 1) * HOP,)
        if frames > 1:
            np.testing.assert_allclose(got, ref, atol=1e-5)


def test_perfect_reconstruction():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(16000).astype(np.float64) * 0.3
    S = stft(x)
    y = bcdl.Istft().compute(S.real.astype(np.float32), S.imag.astype(np.float32))
    # center=True returns (frames-1)*hop samples: the input, truncated to whole hops.
    np.testing.assert_allclose(y, x[:len(y)], atol=1e-5)


def test_dc_and_nyquist_imaginary_parts_are_ignored():
    rng = np.random.default_rng(2)
    re = rng.standard_normal((5, N // 2 + 1)).astype(np.float32)
    im = rng.standard_normal((5, N // 2 + 1)).astype(np.float32)
    im2 = im.copy()
    im2[:, 0] += 3.0
    im2[:, -1] -= 5.0
    ist = bcdl.Istft()
    np.testing.assert_array_equal(ist.compute(re, im), ist.compute(re, im2))


def test_polar_is_bin_major_magnitude_times_unit_phase():
    rng = np.random.default_rng(3)
    frames = 9
    mag = np.abs(rng.standard_normal((N // 2 + 1, frames))).astype(np.float32)
    ph = rng.uniform(-40, 40, (N // 2 + 1, frames)).astype(np.float32)
    ist = bcdl.Istft()
    got = ist.compute_polar(mag, np.cos(ph), np.sin(ph))
    want = ist.compute(np.ascontiguousarray((mag * np.cos(ph)).T),
                       np.ascontiguousarray((mag * np.sin(ph)).T))
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_uncentred_and_other_sizes():
    cfg = bcdl.IstftConfig()
    cfg.n_fft, cfg.hop_length, cfg.center = 512, 128, False
    ist = bcdl.Istft(cfg)
    assert ist.bins == 257
    rng = np.random.default_rng(4)
    re = rng.standard_normal((6, 257)).astype(np.float32)
    im = rng.standard_normal((6, 257)).astype(np.float32)
    got = ist.compute(re, im)
    # Uncentred ends divide by a near-zero window sum, so values there reach the
    # thousands: compare relatively.
    np.testing.assert_allclose(got, ref_istft(re, im, 512, 128, center=False), rtol=1e-5, atol=1e-5)
    assert len(got) == ist.num_samples(6) == 512 + 5 * 128


def test_num_samples_and_empty():
    ist = bcdl.Istft()
    assert ist.num_samples(0) == 0 and ist.num_samples(1) == 0 and ist.num_samples(3) == 2 * HOP
    empty = np.zeros((0, N // 2 + 1), np.float32)
    assert ist.compute(empty, empty).shape == (0,)


def test_bad_shapes_and_config():
    ist = bcdl.Istft()
    with pytest.raises(ValueError):
        ist.compute(np.zeros((3, 100), np.float32), np.zeros((3, 100), np.float32))
    with pytest.raises(ValueError):
        ist.compute_polar(np.zeros((513, 3), np.float32), np.zeros((513, 4), np.float32),
                          np.zeros((513, 3), np.float32))
    cfg = bcdl.IstftConfig()
    cfg.n_fft = 1000
    with pytest.raises(Exception, match="power of two"):
        bcdl.Istft(cfg)
    cfg.n_fft, cfg.hop_length = 1024, 0
    with pytest.raises(Exception, match="hop_length"):
        bcdl.Istft(cfg)
