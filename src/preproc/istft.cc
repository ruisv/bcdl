#include "bcdl/preproc/istft.h"

#include <algorithm>
#include <cmath>

#include "bcdl/core/status.h"
#include "fft.h"

namespace bcdl {

Istft::Istft(const IstftConfig& cfg) : cfg_(cfg) {
  const int n = cfg.n_fft;
  if (n < 2 || (n & (n - 1)) != 0) throw Error(-1, "BCDL Istft: n_fft must be a power of two");
  if (cfg.hop_length <= 0 || cfg.hop_length > n)
    throw Error(-1, "BCDL Istft: hop_length must be in (0, n_fft]");
  fft_ = std::make_shared<detail::Fft>(n);
  window_.resize(n);
  for (int i = 0; i < n; ++i) window_[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / n);  // periodic
  re_.resize(n);
  im_.resize(n);
}

int Istft::numSamples(int frames) const noexcept {
  if (frames <= 0) return 0;
  const int full = cfg_.n_fft + (frames - 1) * cfg_.hop_length;
  return cfg_.center ? full - cfg_.n_fft : full;
}

int Istft::compute(const float* real, const float* imag, int frames, std::vector<float>& out) {
  out.clear();
  if (frames <= 0) return 0;
  if (real == nullptr || imag == nullptr) throw Error(-1, "BCDL Istft::compute: null spectrum");
  const int n = cfg_.n_fft, hop = cfg_.hop_length, nb = bins();
  const int full = n + (frames - 1) * hop;
  acc_.assign(full, 0.0);
  norm_.assign(full, 0.0);
  const double inv_n = 1.0 / n;
  for (int t = 0; t < frames; ++t) {
    const float* pr = real + static_cast<std::size_t>(t) * nb;
    const float* pi = imag + static_cast<std::size_t>(t) * nb;
    // Rebuild the Hermitian full spectrum in bit-reversed order. DC and
    // Nyquist are real by construction; their imaginary parts are dropped.
    for (int k = 0; k < n; ++k) {
      double r, i;
      if (k == 0 || k == n / 2) {
        r = pr[k];
        i = 0.0;
      } else if (k < n / 2) {
        r = pr[k];
        i = pi[k];
      } else {
        r = pr[n - k];
        i = -pi[n - k];
      }
      const int j = fft_->bitrev(k);
      re_[j] = r;
      im_[j] = i;
    }
    fft_->transform(re_.data(), im_.data(), /*inverse=*/true);
    double* a = acc_.data() + static_cast<std::size_t>(t) * hop;
    double* w2 = norm_.data() + static_cast<std::size_t>(t) * hop;
    for (int s = 0; s < n; ++s) {
      a[s] += re_[s] * inv_n * window_[s];
      w2[s] += window_[s] * window_[s];
    }
  }
  const int skip = cfg_.center ? n / 2 : 0;
  const int len = numSamples(frames);
  out.resize(len);
  for (int s = 0; s < len; ++s) {
    const double d = norm_[skip + s];
    out[s] = static_cast<float>(d > 1e-11 ? acc_[skip + s] / d : acc_[skip + s]);
  }
  return len;
}

int Istft::computePolar(const float* mag, const float* cos, const float* sin, int frames,
                        std::vector<float>& out) {
  out.clear();
  if (frames <= 0) return 0;
  if (mag == nullptr || cos == nullptr || sin == nullptr)
    throw Error(-1, "BCDL Istft::computePolar: null input");
  const int nb = bins();
  real_.resize(static_cast<std::size_t>(frames) * nb);
  imag_.resize(real_.size());
  for (int b = 0; b < nb; ++b) {
    const std::size_t row = static_cast<std::size_t>(b) * frames;
    for (int t = 0; t < frames; ++t) {
      const float m = mag[row + t];
      real_[static_cast<std::size_t>(t) * nb + b] = m * cos[row + t];
      imag_[static_cast<std::size_t>(t) * nb + b] = m * sin[row + t];
    }
  }
  return compute(real_.data(), imag_.data(), frames, out);
}

}  // namespace bcdl
