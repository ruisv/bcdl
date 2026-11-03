#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.
namespace detail {
class Fft;
}

// ===========================================================================
// Voice activity detection: Silero VAD v6 (16 kHz)
// ===========================================================================
//
//   PCM -> 512-sample chunks (32 ms) -> [Silero] -> speech probability
//       -> VadSegmenter -> whole utterances, ready for ParaformerAsr
//
// The network is streaming: an LSTM carries its state from chunk to chunk,
// and each chunk is read together with the last 64 samples of the one before.
// Both live in SileroVad, so chunks must be fed in order, one stream per
// object; reset() starts a new stream.

/// Engine-bound Silero VAD.
///   in   mag [1, 129, 4]  (magnitude spectrum, see below)
///        h [1, 128], c [1, 128]  (LSTM state)
///   out  prob [1, 1], h_out [1, 128], c_out [1, 128]
/// The model's STFT runs on the CPU: 64 context samples + the 512 new ones +
/// 64 reflected, cut into 4 frames of 256 (hop 128, periodic Hann), |FFT|.
/// On the BPU, squaring the real and imaginary parts puts the power at a
/// range of ~5000 where int16 loses every quiet bin, and the probability
/// drifts by up to 0.8 mid-sentence. The build is identified by those shapes,
/// its inputs and outputs by name.
class SileroVad {
 public:
  static constexpr int kSampleRate = 16000;
  static constexpr int kChunk = 512;    ///< samples per call (32 ms)
  static constexpr int kContext = 64;

  explicit SileroVad(Engine& engine);

  /// Speech probability of the next kChunk samples (float PCM in [-1, 1]).
  float probability(const float* chunk, int timeout_ms = 0);

  /// Forget the stream: zero state and context.
  void reset();

 private:
  Engine& eng_;
  int in_mag_ = 0, in_h_ = 1, in_c_ = 2, out_p_ = 0, out_h_ = 1, out_c_ = 2;
  std::shared_ptr<const detail::Fft> fft_;
  std::vector<float> x_, mag_, h_, c_, ctx_;
  std::vector<double> win_, re_, im_;
  std::vector<std::uint8_t> stage_;
};

/// How probabilities become utterances. A segment opens once the probability
/// has stayed at or above `threshold` for `min_speech_ms`, and closes once it
/// has stayed below `neg_threshold` for `min_silence_ms`. The defaults were
/// chosen on noisy AISHELL sessions (SNR 10 / 0 dB) and MUSAN noise: no false
/// segments on pure noise, the end of speech declared a median 0.36 s after
/// the speaker stops.
struct VadConfig {
  float threshold = 0.5f;
  float neg_threshold = -1.0f;   ///< < 0: same as threshold
  int min_speech_ms = 250;
  int min_silence_ms = 300;
  int speech_pad_ms = 30;        ///< audio kept either side of a segment
  float max_speech_s = 20.0f;    ///< a longer segment is cut here (ASR window is 24 s)
};

/// One utterance. Sample offsets count from the first sample fed since the
/// last reset(); `pcm` is [start, end) including the padding.
struct VadSegment {
  std::int64_t start = 0;
  std::int64_t end = 0;
  std::vector<float> pcm;
};

/// Streaming segmenter over a SileroVad. Feed PCM of any length; each call
/// returns the utterances that closed during it.
class VadSegmenter {
 public:
  explicit VadSegmenter(SileroVad& vad, VadConfig cfg = {});

  std::vector<VadSegment> feed(const float* pcm, std::size_t n, int timeout_ms = 0);

  /// End of stream: close an open segment (the remainder shorter than a chunk
  /// is not scored) and reset.
  std::vector<VadSegment> flush();

  /// Whole recording -> its segments (feed + flush, from a reset).
  std::vector<VadSegment> segments(const float* pcm, std::size_t n, int timeout_ms = 0);

  void reset();
  bool inSpeech() const noexcept { return triggered_; }
  /// Probability of the last scored chunk.
  float lastProbability() const noexcept { return last_p_; }
  const VadConfig& config() const noexcept { return cfg_; }

 private:
  void step(float p, std::vector<VadSegment>& out);
  void close(std::int64_t end, std::vector<VadSegment>& out);
  float* historyAt(std::int64_t sample);

  SileroVad& vad_;
  VadConfig cfg_;
  float neg_ = 0.5f;
  int min_speech_ = 0, min_silence_ = 0, pad_ = 0;  // in chunks / samples (pad)
  std::int64_t max_speech_ = 0;                     // samples
  std::vector<float> pending_;                      // < kChunk samples not yet scored
  std::vector<float> hist_;                         // audio from hist_start_ on
  std::int64_t hist_start_ = 0, scored_ = 0;        // samples
  bool triggered_ = false;
  int run_speech_ = 0, run_silence_ = 0;
  std::int64_t seg_start_ = 0;
  float last_p_ = 0.0f;
};

}  // namespace bcdl
