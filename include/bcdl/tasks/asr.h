#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bcdl/preproc/fbank.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Speech recognition (Paraformer: non-autoregressive, CIF)
// ===========================================================================
//
// 16 kHz PCM in, text out, in one forward pass per utterance — no decoding
// loop, which is why it belongs in this library rather than next to the LLMs.
// The model is split into three compiled stages with a CPU step in the middle:
//
//   PCM -> fbank -> LFR -> CMVN -> [encoder] -> [predictor] -> CIF -> [decoder] -> argmax
//
// The predictor emits one weight ("alpha") per encoder frame; CIF
// (continuous integrate-and-fire) accumulates them and emits one acoustic
// embedding every time the running sum crosses 1. That count IS the number of
// output tokens, which is what lets the decoder run once, in parallel, instead
// of token by token.
//
// Input is ONE utterance, at most the encoder's fixed length (24 s for the
// shipped build). Longer audio is truncated and flagged; split it with a VAD
// first. Sample rate must be 16 kHz — there is no resampler here.

/// CIF parameters.
struct CifConfig {
  /// The running sum fires at this value. 1.0 for every Paraformer.
  float threshold = 1.0f;
  /// Weight added at the first frame past the end of speech so the last token
  /// can fire (predictor_conf.tail_threshold; 0.45 for paraformer-large). Set
  /// 0 to disable.
  ///
  /// It has to be applied HERE, at the true end of speech. A fixed-shape
  /// predictor graph only knows its padded length, so it adds the tail at the
  /// end of the padding; masking the padding then deletes it, and any utterance
  /// whose last token needed the tail to fire loses it. On the shipped build
  /// that alone took the short-utterance CER from 2.7% to 3.9%.
  float tail_threshold = 0.45f;
  /// Decoder token capacity; fires beyond it are dropped.
  int max_tokens = 100;
};

/// Continuous integrate-and-fire over the first `valid_frames` of a predictor
/// output.
///
/// `alphas` is [frames], `hidden` is [frames, dim] (row-major). Frames at and
/// beyond `valid_frames` are padding: their weights are ignored, and the
/// configured tail weight is placed at frame `valid_frames` (with a zero hidden
/// state), which therefore must be < frames when the tail is enabled.
/// Negative weights are clamped to 0.
///
/// Fills `embeds` with [cfg.max_tokens, dim] (rows past the count are zero) and
/// returns the number of tokens fired, capped at max_tokens. Pure; numpy-testable.
int cifIntegrate(const float* alphas, const float* hidden, int frames, int dim,
                 int valid_frames, const CifConfig& cfg, std::vector<float>& embeds);

/// Load a token list: either a JSON array of strings (FunASR `tokens.json`) or
/// plain text with one token per line. Order is the model's class order.
std::vector<std::string> loadTokenList(const std::string& path);

/// Row-wise argmax of `rows` rows of `vocab` logits.
std::vector<int> argmaxTokens(const float* logits, int rows, int vocab);

/// Join token pieces into text: special tokens (`<...>`) are dropped, a trailing
/// `@@` glues a word piece to the next one, and a space is put between two
/// consecutive Latin words. CJK characters are joined without spaces.
std::string tokensToText(const std::vector<int>& ids, const std::vector<std::string>& vocab);

/// One transcription.
struct AsrResult {
  std::string text;
  std::vector<int> token_ids;  ///< every argmax id, special tokens included
  int feature_frames = 0;      ///< LFR frames the utterance produced (before truncation)
  bool truncated = false;      ///< true if the audio was longer than the encoder window
};

/// Paraformer front end + CIF settings. The defaults match FunASR
/// paraformer-large (80 mel bins, LFR 7/6).
struct ParaformerConfig {
  FbankConfig fbank;
  int lfr_m = 7;
  int lfr_n = 6;
  CifConfig cif;
};

/// Engine-bound three-stage Paraformer.
///
/// The stages are told apart by shape, not by index, so the wiring does not
/// depend on the order the compiler happened to emit:
///   encoder   in [1,T,F]                 -> out [1,T,D]
///   predictor in [1,T,D]                 -> out [1,T+1] alphas, [1,T+1,D] hidden
///   decoder   in [1,T,D] context, [1] token count, [1,1,D] bias, [1,N,D] acoustic
///             -> out [1,N,V] logits
/// T (encoder frames), F (= num_mel_bins * lfr_m), D, N and V are all read from
/// the Engines and cross-checked.
///
/// The decoder's bias input is the contextual (hotword) slot. It is fed zeros:
/// the shipped build has a single slot, and on it zeros transcribe the same as
/// the model's own "no hotword" embedding.
class ParaformerAsr {
 public:
  ParaformerAsr(Engine& encoder, Engine& predictor, Engine& decoder,
                std::vector<std::string> vocab, Cmvn cmvn, ParaformerConfig cfg = {});

  /// One mono 16 kHz utterance (float samples in [-1, 1]) to text.
  AsrResult transcribe(const float* pcm, std::size_t samples, int timeout_ms = 0);

  /// Same, starting from front-end output: `feats` is [frames, featureDim()]
  /// LFR features with CMVN applied. For feeding features computed elsewhere.
  AsrResult transcribeFeatures(const float* feats, int frames, int timeout_ms = 0);

  /// Run the front end only: PCM -> [frames, featureDim()] normalised features.
  int features(const float* pcm, std::size_t samples, std::vector<float>& out);

  int maxFrames() const noexcept { return frames_; }    ///< encoder window, LFR frames
  float maxSeconds() const noexcept;                     ///< same, in seconds of audio
  int featureDim() const noexcept { return feat_dim_; }
  int maxTokens() const noexcept { return max_tokens_; }
  int vocabSize() const noexcept { return static_cast<int>(vocab_.size()); }
  const ParaformerConfig& config() const noexcept { return cfg_; }

 private:
  Engine& enc_;
  Engine& pred_;
  Engine& dec_;
  std::vector<std::string> vocab_;
  Cmvn cmvn_;
  ParaformerConfig cfg_;
  Fbank fbank_;

  int frames_ = 0, feat_dim_ = 0, hid_dim_ = 0, max_tokens_ = 0;
  int pred_alpha_ = 0, pred_hidden_ = 0;
  int dec_ctx_ = 0, dec_count_ = 0, dec_bias_ = 0, dec_acoustic_ = 0;

  std::vector<float> fb_, lfr_, x_, ctx_, embeds_, alphas_, hidden_, bias_;
  std::vector<std::uint8_t> stage_;
};

}  // namespace bcdl
