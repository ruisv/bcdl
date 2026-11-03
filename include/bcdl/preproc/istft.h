#pragma once

#include <memory>
#include <vector>

namespace bcdl {

namespace detail {
class Fft;
}

// ===========================================================================
// Inverse STFT (overlap-add), the back end of a spectral vocoder
// ===========================================================================
//
// Matches torch.istft(window=torch.hann_window(n_fft), center=True,
// normalized=False, onesided=True) and kaldi-native-fbank's IStft, which is
// what sherpa-onnx runs after Vocos: each frame is inverse-FFT'd, windowed with
// a periodic Hann window, overlap-added, and divided by the summed squared
// window; with `center`, n_fft/2 samples are then trimmed from each end. The
// imaginary parts of the DC and Nyquist bins are ignored, as a real inverse FFT
// must.

struct IstftConfig {
  int n_fft = 1024;      ///< power of two; the window is n_fft long
  int hop_length = 256;
  bool center = true;    ///< trim n_fft/2 from each end (frames were centred)
};

class Istft {
 public:
  explicit Istft(const IstftConfig& cfg = {});

  /// Samples produced from `frames` frames: (frames - 1) * hop with `center`,
  /// otherwise n_fft + (frames - 1) * hop. 0 for no frames.
  int numSamples(int frames) const noexcept;

  int bins() const noexcept { return cfg_.n_fft / 2 + 1; }
  const IstftConfig& config() const noexcept { return cfg_; }

  /// `real` and `imag` are [frames, bins()] row-major. Fills `out` and returns
  /// its length.
  int compute(const float* real, const float* imag, int frames, std::vector<float>& out);

  /// Same, from magnitude and phase as unit vectors — Vocos's output layout:
  /// `mag`, `cos`, `sin` are [bins(), frames] row-major (bin-major), and the
  /// spectrum is mag * (cos + i sin).
  int computePolar(const float* mag, const float* cos, const float* sin, int frames,
                   std::vector<float>& out);

 private:
  IstftConfig cfg_;
  std::shared_ptr<const detail::Fft> fft_;
  std::vector<double> window_;
  std::vector<double> re_, im_, acc_, norm_;
  std::vector<float> real_, imag_;
};

}  // namespace bcdl
