#include "bcdl/preproc/fbank.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <sstream>

#include "bcdl/core/status.h"
#include "fft.h"

namespace bcdl {

namespace {

double melScale(double f) { return 1127.0 * std::log(1.0 + f / 700.0); }

}  // namespace

Fbank::Fbank(const FbankConfig& cfg) : cfg_(cfg) {
  if (cfg.sample_rate <= 0 || cfg.num_mel_bins <= 0 || cfg.frame_length_ms <= 0 ||
      cfg.frame_shift_ms <= 0) {
    throw Error(-1, "BCDL Fbank: sample_rate, num_mel_bins and frame sizes must be > 0");
  }
  // Kaldi truncates, not rounds: 25 ms at 16 kHz is 400 samples, 10 ms is 160.
  frame_len_ = static_cast<int>(cfg.sample_rate * 0.001 * cfg.frame_length_ms);
  frame_shift_ = static_cast<int>(cfg.sample_rate * 0.001 * cfg.frame_shift_ms);
  if (frame_len_ < 2 || frame_shift_ < 1) {
    throw Error(-1, "BCDL Fbank: frame length/shift round to fewer than 2/1 samples");
  }
  nfft_ = 1;
  while (nfft_ < frame_len_) nfft_ *= 2;

  // Symmetric windows (periodic = false): Hamming as torch.hamming_window in
  // kaldi.fbank, Povey as Kaldi's feature-window.cc.
  window_.resize(frame_len_);
  if (cfg.window != "hamming" && cfg.window != "povey") {
    throw Error(-1, "BCDL Fbank: window must be \"hamming\" or \"povey\", got \"" + cfg.window + "\"");
  }
  for (int i = 0; i < frame_len_; ++i) {
    const double a = 2.0 * M_PI * i / static_cast<double>(frame_len_ - 1);
    window_[i] = cfg.window == "povey" ? std::pow(0.5 - 0.5 * std::cos(a), 0.85)
                                        : 0.54 - 0.46 * std::cos(a);
  }

  fft_ = std::make_shared<detail::Fft>(nfft_);

  // Kaldi mel banks: num_bins + 2 points evenly spaced on the mel axis between
  // low_freq and high_freq, triangles evaluated at each FFT bin's mel value.
  // The Nyquist bin gets weight 0 (kaldi pads that column with zeros).
  const double nyquist = 0.5 * cfg.sample_rate;
  const double high = cfg.high_freq > 0 ? cfg.high_freq : nyquist + cfg.high_freq;
  if (!(cfg.low_freq >= 0 && high > cfg.low_freq && high <= nyquist)) {
    throw Error(-1, "BCDL Fbank: need 0 <= low_freq < high_freq <= Nyquist");
  }
  const double mel_low = melScale(cfg.low_freq), mel_high = melScale(high);
  const double delta = (mel_high - mel_low) / (cfg.num_mel_bins + 1);
  const double bin_hz = static_cast<double>(cfg.sample_rate) / nfft_;
  first_.assign(cfg.num_mel_bins, 0);
  weights_.assign(cfg.num_mel_bins, {});
  for (int b = 0; b < cfg.num_mel_bins; ++b) {
    const double l = mel_low + b * delta, c = l + delta, r = l + 2 * delta;
    int first = -1;
    std::vector<double> w;
    for (int k = 0; k < nfft_ / 2; ++k) {
      const double m = melScale(bin_hz * k);
      const double v = std::max(0.0, std::min((m - l) / (c - l), (r - m) / (r - c)));
      if (v > 0) {
        if (first < 0) first = k;
        w.resize(k - first + 1, 0.0);
        w[k - first] = v;
      }
    }
    first_[b] = first < 0 ? 0 : first;
    weights_[b] = std::move(w);
  }
  re_.resize(nfft_);
  im_.resize(nfft_);
  power_.resize(nfft_ / 2 + 1);
}

int Fbank::numFrames(std::size_t samples) const {
  if (samples < static_cast<std::size_t>(frame_len_)) return 0;
  return 1 + static_cast<int>((samples - frame_len_) / frame_shift_);
}

int Fbank::compute(const float* pcm, std::size_t samples, std::vector<float>& out) {
  const int frames = numFrames(samples);
  const int nmel = cfg_.num_mel_bins;
  out.assign(static_cast<std::size_t>(frames) * nmel, 0.0f);
  if (frames == 0) return 0;
  if (pcm == nullptr) throw Error(-1, "BCDL Fbank::compute: null PCM");

  const double scale = cfg_.input_scale;
  std::vector<double> buf(frame_len_);
  for (int t = 0; t < frames; ++t) {
    const float* x = pcm + static_cast<std::size_t>(t) * frame_shift_;
    double mean = 0.0;
    for (int i = 0; i < frame_len_; ++i) {
      buf[i] = x[i] * scale;
      mean += buf[i];
    }
    if (cfg_.remove_dc_offset) {
      mean /= frame_len_;
      for (int i = 0; i < frame_len_; ++i) buf[i] -= mean;
    }
    // Pre-emphasis with the first sample replicated (kaldi's left pad), then the
    // window; walk backwards so each step still sees the un-emphasised x[i-1].
    const double p = cfg_.preemphasis;
    for (int i = frame_len_ - 1; i > 0; --i) buf[i] -= p * buf[i - 1];
    buf[0] -= p * buf[0];

    std::fill(re_.begin(), re_.end(), 0.0);
    std::fill(im_.begin(), im_.end(), 0.0);
    for (int i = 0; i < frame_len_; ++i) {
      re_[fft_->bitrev(i)] = buf[i] * window_[i];
    }
    // The zero tail lands on the bit-reversed slots of indices >= frame_len_.
    fft_->transform(re_.data(), im_.data());
    for (int k = 0; k <= nfft_ / 2; ++k) power_[k] = re_[k] * re_[k] + im_[k] * im_[k];

    float* o = out.data() + static_cast<std::size_t>(t) * nmel;
    for (int b = 0; b < nmel; ++b) {
      const std::vector<double>& w = weights_[b];
      const double* pw = power_.data() + first_[b];
      double e = 0.0;
      for (std::size_t k = 0; k < w.size(); ++k) e += w[k] * pw[k];
      o[b] = static_cast<float>(std::log(std::max(e, static_cast<double>(FLT_EPSILON))));
    }
  }
  return frames;
}

int lfrStack(const float* feats, int frames, int dim, int lfr_m, int lfr_n,
             std::vector<float>& out) {
  if (lfr_m <= 0 || lfr_n <= 0 || dim <= 0) {
    throw Error(-1, "BCDL lfrStack: lfr_m, lfr_n and dim must be > 0");
  }
  if (frames <= 0) {
    out.clear();
    return 0;
  }
  if (feats == nullptr) throw Error(-1, "BCDL lfrStack: null features");
  const int left = (lfr_m - 1) / 2;
  const int padded = frames + left;
  const int out_frames = (frames + lfr_n - 1) / lfr_n;
  out.resize(static_cast<std::size_t>(out_frames) * lfr_m * dim);
  for (int i = 0; i < out_frames; ++i) {
    for (int j = 0; j < lfr_m; ++j) {
      // Index into the left-padded sequence, clamped to its last frame.
      const int pi = std::min(i * lfr_n + j, padded - 1);
      const int src = std::max(pi - left, 0);
      std::copy_n(feats + static_cast<std::size_t>(src) * dim, dim,
                  out.data() + (static_cast<std::size_t>(i) * lfr_m + j) * dim);
    }
  }
  return out_frames;
}

void Cmvn::apply(float* feats, int frames) const {
  const int d = dim();
  if (d == 0 || static_cast<int>(scale.size()) != d) {
    throw Error(-1, "BCDL Cmvn::apply: empty or inconsistent CMVN");
  }
  if (frames <= 0) return;
  if (feats == nullptr) throw Error(-1, "BCDL Cmvn::apply: null features");
  for (int t = 0; t < frames; ++t) {
    float* x = feats + static_cast<std::size_t>(t) * d;
    for (int i = 0; i < d; ++i) x[i] = (x[i] + shift[i]) * scale[i];
  }
}

Cmvn parseKaldiCmvn(const std::string& text) {
  // Layout (FunASR am.mvn):
  //   <AddShift> 560 560
  //   <LearnRateCoef> 0 [ v0 v1 ... ]
  //   <Rescale> 560 560
  //   <LearnRateCoef> 0 [ v0 v1 ... ]
  auto vectorAfter = [&](const std::string& tag) {
    std::vector<float> v;
    const std::size_t at = text.find(tag);
    if (at == std::string::npos) return v;
    const std::size_t open = text.find('[', at);
    const std::size_t close = open == std::string::npos ? open : text.find(']', open);
    if (open == std::string::npos || close == std::string::npos) return v;
    std::istringstream in(text.substr(open + 1, close - open - 1));
    float x;
    while (in >> x) v.push_back(x);
    return v;
  };
  Cmvn c;
  c.shift = vectorAfter("<AddShift>");
  c.scale = vectorAfter("<Rescale>");
  if (c.shift.empty() || c.scale.empty()) {
    throw Error(-1, "BCDL parseKaldiCmvn: no <AddShift>/<Rescale> vectors found");
  }
  if (c.shift.size() != c.scale.size()) {
    throw Error(-1, "BCDL parseKaldiCmvn: <AddShift> has " + std::to_string(c.shift.size()) +
                        " values but <Rescale> has " + std::to_string(c.scale.size()));
  }
  return c;
}

Cmvn loadKaldiCmvn(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw Error(-1, "BCDL loadKaldiCmvn: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return parseKaldiCmvn(ss.str());
}

}  // namespace bcdl
