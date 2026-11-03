#pragma once

// Internal: an in-place iterative radix-2 complex FFT in double precision,
// shared by the fbank front end and the iSTFT. Not installed.

#include <cmath>
#include <vector>

namespace bcdl {
namespace detail {

class Fft {
 public:
  /// `n` must be a power of two.
  explicit Fft(int n) : n_(n), bitrev_(n), cos_(n / 2), sin_(n / 2) {
    int bits = 0;
    while ((1 << bits) < n_) ++bits;
    for (int i = 0; i < n_; ++i) {
      int r = 0;
      for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
      bitrev_[i] = r;
    }
    for (int k = 0; k < n_ / 2; ++k) {
      cos_[k] = std::cos(2.0 * M_PI * k / n_);
      sin_[k] = -std::sin(2.0 * M_PI * k / n_);
    }
  }

  int size() const noexcept { return n_; }

  /// Position of input sample i in the bit-reversed order transform() expects.
  int bitrev(int i) const noexcept { return bitrev_[i]; }

  /// Forward DFT, X[k] = sum x[n] e^{-2 pi i k n / N}, of data already placed
  /// in bit-reversed order. With `inverse`, the conjugate kernel (no 1/N).
  void transform(double* re, double* im, bool inverse = false) const {
    const double sgn = inverse ? -1.0 : 1.0;
    for (int len = 2; len <= n_; len <<= 1) {
      const int half = len >> 1, step = n_ / len;
      for (int s = 0; s < n_; s += len) {
        for (int k = 0; k < half; ++k) {
          const double wr = cos_[k * step], wi = sgn * sin_[k * step];
          const int a = s + k, b = a + half;
          const double tr = wr * re[b] - wi * im[b];
          const double ti = wr * im[b] + wi * re[b];
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }

 private:
  int n_;
  std::vector<int> bitrev_;
  std::vector<double> cos_, sin_;  // twiddles, n/2 each
};

}  // namespace detail
}  // namespace bcdl
