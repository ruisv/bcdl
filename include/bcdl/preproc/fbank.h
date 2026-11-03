#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace bcdl {

namespace detail {
class Fft;
}

// ===========================================================================
// Speech front end: Kaldi-compatible log-mel filterbank, LFR stacking, CMVN
// ===========================================================================
//
// The audio counterpart of letterbox: it turns raw PCM into the exact tensor a
// speech model was trained on, on the CPU. "Exact" is the whole point — every
// FunASR / WeNet / icefall model was trained on `torchaudio.compliance.kaldi.fbank`
// features, and a front end that is merely *similar* (librosa mel, a different
// window, natural-log vs log10, no pre-emphasis) produces features the model
// has never seen. That does not fail; it transcribes worse, quietly.
//
// This reproduces kaldi.fbank with dither = 0: per-frame DC removal,
// pre-emphasis, a symmetric Hamming window, a power spectrum zero-padded to the
// next power of two, Kaldi's mel banks (1127·ln(1 + f/700)), and the log floored
// at FLT_EPSILON. Dither is left out on purpose — it is a training-time noise
// source, and leaving it in makes inference non-deterministic.

/// Filterbank parameters. The defaults are the FunASR / WeNet 16 kHz front end.
struct FbankConfig {
  int sample_rate = 16000;
  int num_mel_bins = 80;
  float frame_length_ms = 25.0f;
  float frame_shift_ms = 10.0f;
  float preemphasis = 0.97f;
  bool remove_dc_offset = true;
  float low_freq = 20.0f;
  /// Upper band edge in Hz; <= 0 means "Nyquist + high_freq", Kaldi's convention.
  float high_freq = 0.0f;
  /// Kaldi works on int16-scale samples. PCM given as float in [-1, 1] is
  /// multiplied by this first; pass 1 if the samples are already int16-scaled.
  /// icefall / k2 models (sherpa-onnx's normalize_samples) train on 1.
  float input_scale = 32768.0f;
  /// "hamming" (torchaudio's kaldi.fbank, FunASR) or "povey" (Kaldi's and
  /// kaldi-native-fbank's default: a Hann window raised to 0.85).
  std::string window = "hamming";
};

/// Kaldi-compatible log-mel filterbank. Construction precomputes the window,
/// the FFT twiddles and the mel weights; compute() is then allocation-free
/// beyond growing `out`. compute() uses per-instance scratch buffers, so give
/// each thread its own instance.
class Fbank {
 public:
  explicit Fbank(const FbankConfig& cfg = {});

  /// Frames produced from `samples` PCM samples (snip-edges framing):
  /// 1 + (samples - frame_length) / frame_shift, or 0 if shorter than one frame.
  int numFrames(std::size_t samples) const;

  /// Log-mel features of `samples` mono PCM samples into `out` as a row-major
  /// [frames, num_mel_bins] array. Returns the frame count.
  int compute(const float* pcm, std::size_t samples, std::vector<float>& out);

  const FbankConfig& config() const noexcept { return cfg_; }
  int frameLength() const noexcept { return frame_len_; }  ///< samples per frame
  int frameShift() const noexcept { return frame_shift_; }  ///< samples per hop
  int fftSize() const noexcept { return nfft_; }

 private:
  FbankConfig cfg_;
  int frame_len_ = 0;
  int frame_shift_ = 0;
  int nfft_ = 0;
  std::vector<double> window_;
  std::shared_ptr<const detail::Fft> fft_;  // immutable, shared by copies
  // Mel weights, sparse: bin b covers FFT bins [first_[b], first_[b] + weights_[b].size()).
  std::vector<int> first_;
  std::vector<std::vector<double>> weights_;
  // FFT scratch in double: a weak band next to a strong one (e^4 vs e^20 of
  // energy) loses ~0.3% to float32 round-off, which shows up in the log.
  std::vector<double> re_, im_, power_;
};

/// Low-frame-rate stacking (FunASR `apply_lfr`): output frame i concatenates
/// input frames i*lfr_n .. i*lfr_n + lfr_m - 1, after (lfr_m-1)/2 copies of the
/// first frame are prepended and the last frame is repeated past the end.
/// `feats` is [frames, dim]; `out` becomes [ceil(frames / lfr_n), lfr_m * dim].
/// Returns the output frame count. Pure; matches FunASR bit for bit.
int lfrStack(const float* feats, int frames, int dim, int lfr_m, int lfr_n,
             std::vector<float>& out);

/// Global CMVN in the Kaldi-nnet text form FunASR ships as `am.mvn`:
/// y = (x + shift) * scale, per dimension.
struct Cmvn {
  std::vector<float> shift;
  std::vector<float> scale;

  int dim() const noexcept { return static_cast<int>(shift.size()); }
  /// Normalise `frames` rows of `dim()` values in place.
  void apply(float* feats, int frames) const;
};

/// Parse an `am.mvn` (the `<AddShift>` / `<Rescale>` blocks). Throws on a file
/// that has neither, or whose two vectors differ in length.
Cmvn parseKaldiCmvn(const std::string& text);
Cmvn loadKaldiCmvn(const std::string& path);

}  // namespace bcdl
