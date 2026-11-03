#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "bcdl/preproc/fbank.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Keyword spotting (wake word): streaming Zipformer transducer, open vocabulary
// ===========================================================================
//
//   PCM -> fbank (CPU) -> [encoder, one 160 ms chunk per call] -> decoder +
//   joiner (CPU) -> keyword search over a phrase graph -> detections
//
// This reproduces sherpa-onnx's KeywordSpotter for icefall's zipformer2 KWS
// models (zh-en 3M): keywords are token sequences, so a new wake word is a new
// line of text, not a new model. The encoder carries 38 cache tensors from
// chunk to chunk; they live here, so one object reads one stream, in order.

/// Search settings; sherpa-onnx's defaults.
struct KwsConfig {
  float keywords_score = 1.0f;      ///< boost per matched token
  float keywords_threshold = 0.25f; ///< mean per-token probability to accept
  int num_trailing_blanks = 1;      ///< blank frames after the last token before firing
  int max_active_paths = 4;         ///< beam
};

/// One detection. Times are seconds since the last reset(), from the
/// encoder's 40 ms frames.
struct KwsDetection {
  std::string keyword;      ///< the phrase after '@' in its keywords line
  int index = -1;           ///< which keywords line
  double start = 0.0;       ///< first token
  double end = 0.0;         ///< last token
  std::vector<int> tokens;
};

/// Engine-bound keyword spotter.
///   encoder  in  x [1, T, 80] fbank frames, the cache tensors (cached_*,
///                embed_states), pad_mask [1, left + 8] (1 = cache slot not yet
///                filled — replaces the exported model's int64 processed_lens)
///            out encoder_out [1, frames, D] and new_<name> for every cache
///   decoder_joiner  the decoder and joiner weights as one float32 .npy table
///                (see the model's conversion notes for the layout)
///   tokens   "token id" per line (sherpa tokens.txt)
/// Inputs and outputs are matched by name.
class KeywordSpotter {
 public:
  KeywordSpotter(Engine& encoder, const std::string& decoder_joiner, const std::string& tokens,
                 KwsConfig cfg = {});
  ~KeywordSpotter();
  KeywordSpotter(const KeywordSpotter&) = delete;
  KeywordSpotter& operator=(const KeywordSpotter&) = delete;

  /// Keywords in sherpa-onnx's format, one per line:
  ///   n ǐ h ǎo x iǎo ch ē @你好小车
  ///   x iǎo ch ē x iǎo ch ē :2.0 #0.3 @小车小车
  /// tokens (from the model's token table), optional ":boost" and
  /// "#threshold" overriding the config for that line, then "@phrase".
  /// Replaces any previous set and resets the stream.
  void setKeywords(const std::string& text);
  void loadKeywords(const std::string& path);
  const std::vector<std::string>& keywords() const noexcept { return phrases_; }

  /// Feed 16 kHz float PCM in [-1, 1], any length; returns what fired.
  std::vector<KwsDetection> feed(const float* pcm, std::size_t n, int timeout_ms = 0);

  /// End of stream: pushes the audio still waiting for a full chunk (and the
  /// blank frames a keyword needs after its last token) through with
  /// silence, returns what fires, then resets. Without it a wake word in the
  /// last ~0.3 s of a recording is never seen.
  std::vector<KwsDetection> flush(int timeout_ms = 0);

  /// Start a new stream: features, encoder caches and search state.
  void reset();

  /// Encoder output for a whole recording, from a reset ([frames, D]); for tests.
  std::vector<float> encode(const float* pcm, std::size_t n, int& frames, int timeout_ms = 0);

  int vocabSize() const noexcept { return vocab_; }
  int tokenId(const std::string& token) const;  ///< -1 if absent
  const KwsConfig& config() const noexcept { return cfg_; }

 private:
  struct Graph;
  struct Hyp;
  void encodeChunk(int timeout_ms);
  void decodeFrame(const float* enc, std::vector<KwsDetection>& out);
  const float* decoderOut(int y0, int y1);

  Engine& enc_;
  KwsConfig cfg_;
  Fbank fbank_;
  std::unordered_map<std::string, int> token_;
  std::vector<std::string> inv_;
  int vocab_ = 0, dim_ = 0, unk_ = -1;
  std::vector<float> table_;  // [2V + 2D, D + 1]
  std::vector<float> zero_;   // [D], the decoder's embedding of "no token"
  // encoder geometry and state
  int in_x_ = -1, in_mask_ = -1, out_enc_ = -1, t_in_ = 0, shift_ = 0, out_frames_ = 0,
      mask_len_ = 0, left_ = 0;
  std::vector<std::pair<int, int>> caches_;  // (input index, output index)
  std::vector<std::vector<float>> state_;
  std::vector<float> mask_, enc_out_, scratch_;
  std::vector<std::uint8_t> stage_;
  // streaming front end
  std::vector<float> head_;      // first samples until the reflection pad can be built
  std::vector<float> wav_;       // samples from frame next_frame_ on
  std::vector<float> feats_;     // fbank frames not yet consumed by the encoder
  bool started_ = false;
  std::int64_t processed_ = 0, frame_offset_ = 0;
  // search
  std::unique_ptr<Graph> graph_;
  std::vector<std::string> phrases_;
  std::vector<Hyp> hyps_;
  std::unordered_map<std::int64_t, std::vector<float>> dec_cache_;
  std::vector<float> logits_;
};

}  // namespace bcdl
