#include "bcdl/tasks/vocoder.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/input_writer.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"
#include "task_util.h"

namespace bcdl {

VocosVocoder::VocosVocoder(Engine& engine, VocosConfig cfg)
    : eng_(engine), cfg_(cfg), istft_(cfg.stft) {
  if (eng_.numInputs() != 1 || eng_.numOutputs() != 2)
    throw Error(-1, "BCDL VocosVocoder: expected 1 input and 2 outputs (log_mag, phase), got " +
                        std::to_string(eng_.numInputs()) + " / " +
                        std::to_string(eng_.numOutputs()));
  const std::vector<int> in = eng_.inputShape(0);
  if (in.size() != 3 || in[0] != 1)
    throw Error(-1, "BCDL VocosVocoder: input must be [1, mels, T], got " + detail::shapeStr(in));
  mels_ = in[1];
  chunk_ = in[2];
  bins_ = istft_.bins();
  for (int i = 0; i < 2; ++i) {
    const std::vector<int> o = eng_.outputShape(i);
    if (o != std::vector<int>{1, bins_, chunk_})
      throw Error(-1, "BCDL VocosVocoder: output " + std::to_string(i) + " is " + detail::shapeStr(o) +
                          ", expected [1, " + std::to_string(bins_) + ", " +
                          std::to_string(chunk_) + "] for n_fft " +
                          std::to_string(cfg_.stft.n_fft));
  }
  if (cfg_.context < 0 || bodyFrames() < 1)
    throw Error(-1, "BCDL VocosVocoder: context " + std::to_string(cfg_.context) +
                        " leaves no frames of a " + std::to_string(chunk_) + "-frame chunk");
}

int VocosVocoder::synthesize(const float* mel, int frames, std::vector<float>& pcm,
                             int timeout_ms) {
  pcm.clear();
  if (frames <= 0) return 0;
  if (mel == nullptr) throw Error(-1, "BCDL VocosVocoder::synthesize: null mel");
  const int C = cfg_.context, B = bodyFrames(), T = chunk_;
  mag_.assign(static_cast<std::size_t>(bins_) * frames, 0.0f);
  cos_.assign(mag_.size(), 0.0f);
  sin_.assign(mag_.size(), 0.0f);
  in_.resize(static_cast<std::size_t>(mels_) * T);
  std::vector<float> scratch;
  std::vector<int> shape;
  for (int b = 0; b < frames; b += B) {
    // Chunk window [b - C, b - C + T), padded past either end of the utterance.
    const int lo = b - C;
    for (int m = 0; m < mels_; ++m) {
      const float* src = mel + static_cast<std::size_t>(m) * frames;
      float* dst = in_.data() + static_cast<std::size_t>(m) * T;
      for (int t = 0; t < T; ++t) {
        const int s = lo + t;
        dst[t] = (s >= 0 && s < frames) ? src[s] : cfg_.pad_value;
      }
    }
    setInputFromFloat(eng_, 0, in_.data(), in_.size(), stage_);
    eng_.infer(timeout_ms);
    const int keep = std::min(B, frames - b);
    const float* lm = outputAsFloat(eng_, 0, scratch, shape);
    for (int k = 0; k < bins_; ++k) {
      const float* src = lm + static_cast<std::size_t>(k) * T + C;
      float* dst = mag_.data() + static_cast<std::size_t>(k) * frames + b;
      for (int t = 0; t < keep; ++t) dst[t] = std::min(std::exp(src[t]), cfg_.max_magnitude);
    }
    const float* ph = outputAsFloat(eng_, 1, scratch, shape);
    for (int k = 0; k < bins_; ++k) {
      const float* src = ph + static_cast<std::size_t>(k) * T + C;
      const std::size_t o = static_cast<std::size_t>(k) * frames + b;
      for (int t = 0; t < keep; ++t) {
        cos_[o + t] = std::cos(src[t]);
        sin_[o + t] = std::sin(src[t]);
      }
    }
  }
  return istft_.computePolar(mag_.data(), cos_.data(), sin_.data(), frames, pcm);
}

}  // namespace bcdl
