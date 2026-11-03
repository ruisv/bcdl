#include "bcdl/tasks/vad.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/input_writer.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"
#include "task_util.h"
#include "../preproc/fft.h"

namespace bcdl {

namespace {

constexpr int kWindow = SileroVad::kContext + SileroVad::kChunk + SileroVad::kContext;  // 640
constexpr int kState = 128;
constexpr int kFft = 256, kHop = 128, kBins = kFft / 2 + 1, kFrames = (kWindow - kFft) / kHop + 1;

// Index of the tensor called `name`, else of the one at `fallback` (a build
// without the export's names keeps the export's order).
int byName(const std::vector<std::string>& names, const char* name, int fallback) {
  for (std::size_t i = 0; i < names.size(); ++i)
    if (names[i] == name) return static_cast<int>(i);
  return fallback;
}

int chunksFor(int ms) {
  // Whole chunks covering `ms` (a run of k chunks lasts k * 32 ms).
  const double samples = static_cast<double>(ms) * SileroVad::kSampleRate / 1000.0;
  return std::max(0, static_cast<int>(std::ceil(samples / SileroVad::kChunk - 1e-9)));
}

}  // namespace

// ---------------------------------------------------------------------------
// SileroVad
// ---------------------------------------------------------------------------

SileroVad::SileroVad(Engine& engine) : eng_(engine) {
  if (eng_.numInputs() != 3 || eng_.numOutputs() != 3)
    throw Error(-1, "BCDL SileroVad: expected 3 inputs (mag, h, c) and 3 outputs (prob, h_out, "
                    "c_out), got " + std::to_string(eng_.numInputs()) + " / " +
                        std::to_string(eng_.numOutputs()));
  std::vector<std::string> in, out;
  for (int i = 0; i < 3; ++i) {
    in.push_back(eng_.inputName(i));
    out.push_back(eng_.outputName(i));
  }
  in_mag_ = byName(in, "mag", 0);
  in_h_ = byName(in, "h", 1);
  in_c_ = byName(in, "c", 2);
  out_p_ = byName(out, "prob", 0);
  out_h_ = byName(out, "h_out", 1);
  out_c_ = byName(out, "c_out", 2);
  const std::vector<int> st{1, kState};
  if (eng_.inputShape(in_mag_) != std::vector<int>{1, kBins, kFrames} ||
      eng_.inputShape(in_h_) != st || eng_.inputShape(in_c_) != st ||
      eng_.outputShape(out_h_) != st || eng_.outputShape(out_c_) != st ||
      eng_.outputShape(out_p_) != std::vector<int>{1, 1})
    throw Error(-1, "BCDL SileroVad: not a 16 kHz Silero build: mag " +
                        detail::shapeStr(eng_.inputShape(in_mag_)) + ", h " +
                        detail::shapeStr(eng_.inputShape(in_h_)) + ", prob " +
                        detail::shapeStr(eng_.outputShape(out_p_)) +
                        " (want mag [1,129,4], h/c [1,128], prob [1,1])");
  x_.assign(kWindow, 0.0f);
  mag_.assign(static_cast<std::size_t>(kBins) * kFrames, 0.0f);
  fft_ = std::make_shared<detail::Fft>(kFft);
  win_.resize(kFft);
  for (int n = 0; n < kFft; ++n) win_[n] = 0.5 - 0.5 * std::cos(2.0 * M_PI * n / kFft);
  re_.resize(kFft);
  im_.resize(kFft);
  reset();
}

void SileroVad::reset() {
  h_.assign(kState, 0.0f);
  c_.assign(kState, 0.0f);
  ctx_.assign(kContext, 0.0f);
}

float SileroVad::probability(const float* chunk, int timeout_ms) {
  if (chunk == nullptr) throw Error(-1, "BCDL SileroVad::probability: null chunk");
  // [context | chunk | reflection of the chunk's tail, edge excluded]
  std::copy(ctx_.begin(), ctx_.end(), x_.begin());
  std::copy_n(chunk, kChunk, x_.begin() + kContext);
  const int last = kContext + kChunk - 1;
  for (int j = 0; j < kContext; ++j) x_[last + 1 + j] = x_[last - 1 - j];
  std::copy_n(chunk + kChunk - kContext, kContext, ctx_.begin());

  // The model's STFT: |FFT| of 4 Hann-windowed frames -> mag [bins, frames].
  for (int f = 0; f < kFrames; ++f) {
    for (int n = 0; n < kFft; ++n) {
      const int r = fft_->bitrev(n);
      re_[r] = static_cast<double>(x_[f * kHop + n]) * win_[n];
      im_[r] = 0.0;
    }
    fft_->transform(re_.data(), im_.data());
    for (int k = 0; k < kBins; ++k)
      mag_[static_cast<std::size_t>(k) * kFrames + f] =
          static_cast<float>(std::sqrt(re_[k] * re_[k] + im_[k] * im_[k]));
  }

  setInputFromFloat(eng_, in_mag_, mag_.data(), mag_.size(), stage_);
  setInputFromFloat(eng_, in_h_, h_.data(), h_.size(), stage_);
  setInputFromFloat(eng_, in_c_, c_.data(), c_.size(), stage_);
  eng_.infer(timeout_ms);
  std::vector<float> scratch;
  std::vector<int> shape;
  const float p = outputAsFloat(eng_, out_p_, scratch, shape)[0];
  const float* h = outputAsFloat(eng_, out_h_, scratch, shape);
  std::copy_n(h, kState, h_.begin());
  const float* c = outputAsFloat(eng_, out_c_, scratch, shape);
  std::copy_n(c, kState, c_.begin());
  return p;
}

// ---------------------------------------------------------------------------
// VadSegmenter
// ---------------------------------------------------------------------------

VadSegmenter::VadSegmenter(SileroVad& vad, VadConfig cfg) : vad_(vad), cfg_(cfg) {
  if (!(cfg_.threshold > 0.0f && cfg_.threshold < 1.0f))
    throw Error(-1, "BCDL VadSegmenter: threshold must be in (0, 1)");
  neg_ = cfg_.neg_threshold < 0.0f ? cfg_.threshold : cfg_.neg_threshold;
  if (neg_ > cfg_.threshold)
    throw Error(-1, "BCDL VadSegmenter: neg_threshold must not exceed threshold");
  if (cfg_.min_speech_ms < 0 || cfg_.min_silence_ms < 0 || cfg_.speech_pad_ms < 0)
    throw Error(-1, "BCDL VadSegmenter: durations must be >= 0");
  min_speech_ = std::max(1, chunksFor(cfg_.min_speech_ms));
  min_silence_ = std::max(1, chunksFor(cfg_.min_silence_ms));
  pad_ = cfg_.speech_pad_ms * SileroVad::kSampleRate / 1000;
  max_speech_ = cfg_.max_speech_s > 0.0f
                    ? static_cast<std::int64_t>(cfg_.max_speech_s * SileroVad::kSampleRate)
                    : 0;
  if (max_speech_ > 0 && max_speech_ < SileroVad::kChunk)
    throw Error(-1, "BCDL VadSegmenter: max_speech_s is shorter than one chunk");
  reset();
}

void VadSegmenter::reset() {
  vad_.reset();
  pending_.clear();
  hist_.clear();
  hist_start_ = scored_ = 0;
  triggered_ = false;
  run_speech_ = run_silence_ = 0;
  seg_start_ = 0;
  last_p_ = 0.0f;
}

void VadSegmenter::close(std::int64_t end, std::vector<VadSegment>& out) {
  VadSegment s;
  s.start = std::max(hist_start_, seg_start_ - pad_);
  s.end = std::min(scored_, end + pad_);
  if (s.end > s.start)
    s.pcm.assign(hist_.begin() + (s.start - hist_start_), hist_.begin() + (s.end - hist_start_));
  out.push_back(std::move(s));
}

void VadSegmenter::step(float p, std::vector<VadSegment>& out) {
  // Called after the chunk [scored_ - kChunk, scored_) was scored.
  constexpr int C = SileroVad::kChunk;
  last_p_ = p;
  if (!triggered_) {
    if (p >= cfg_.threshold) {
      if (++run_speech_ >= min_speech_) {
        triggered_ = true;
        run_silence_ = 0;
        seg_start_ = scored_ - static_cast<std::int64_t>(run_speech_) * C;
      }
    } else {
      run_speech_ = 0;
    }
  } else {
    if (p >= cfg_.threshold) {
      run_silence_ = 0;
    } else if (p < neg_ && ++run_silence_ >= min_silence_) {
      close(scored_ - static_cast<std::int64_t>(run_silence_) * C, out);
      triggered_ = false;
      run_speech_ = run_silence_ = 0;
    }
    if (triggered_ && max_speech_ > 0 && scored_ - seg_start_ >= max_speech_) {
      // Too long for one utterance: cut here and carry on in a new one.
      const int pad = pad_;
      pad_ = 0;
      close(scored_, out);
      pad_ = pad;
      seg_start_ = scored_;
      run_silence_ = 0;
    }
  }
  // Keep only the audio a future segment can still start in.
  const std::int64_t keep_from =
      triggered_ ? seg_start_ - pad_
                 : scored_ - static_cast<std::int64_t>(run_speech_) * C - pad_;
  const std::int64_t drop = std::max<std::int64_t>(0, keep_from - hist_start_);
  if (drop > static_cast<std::int64_t>(16 * C)) {  // trim in batches, not per chunk
    hist_.erase(hist_.begin(), hist_.begin() + drop);
    hist_start_ += drop;
  }
}

std::vector<VadSegment> VadSegmenter::feed(const float* pcm, std::size_t n, int timeout_ms) {
  std::vector<VadSegment> out;
  if (n == 0) return out;
  if (pcm == nullptr) throw Error(-1, "BCDL VadSegmenter::feed: null pcm");
  constexpr std::size_t C = SileroVad::kChunk;
  std::size_t i = 0;
  // Finish a chunk left over from the previous call.
  if (!pending_.empty()) {
    const std::size_t take = std::min(n, C - pending_.size());
    pending_.insert(pending_.end(), pcm, pcm + take);
    i = take;
    if (pending_.size() < C) return out;
    hist_.insert(hist_.end(), pending_.begin(), pending_.end());
    const float p = vad_.probability(pending_.data(), timeout_ms);
    scored_ += C;
    pending_.clear();
    step(p, out);
  }
  for (; i + C <= n; i += C) {
    hist_.insert(hist_.end(), pcm + i, pcm + i + C);
    const float p = vad_.probability(pcm + i, timeout_ms);
    scored_ += C;
    step(p, out);
  }
  pending_.assign(pcm + i, pcm + n);
  return out;
}

std::vector<VadSegment> VadSegmenter::flush() {
  std::vector<VadSegment> out;
  if (triggered_) close(scored_, out);
  reset();
  return out;
}

std::vector<VadSegment> VadSegmenter::segments(const float* pcm, std::size_t n, int timeout_ms) {
  reset();
  std::vector<VadSegment> out = feed(pcm, n, timeout_ms);
  for (auto& s : flush()) out.push_back(std::move(s));
  return out;
}

}  // namespace bcdl
