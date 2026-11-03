#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.
class VocosVocoder;

// ===========================================================================
// Speech synthesis: Matcha-TTS (zh-en) — text -> mel, non-autoregressive
// ===========================================================================
//
//   text -> MatchaFrontend -> token ids per sentence
//        -> [encoder] -> durations -> length regulation (CPU) -> [decoder] -> mel
//        -> VocosVocoder -> 16 kHz PCM
//
// The encoder predicts each token's mean mel frame and its duration; the CPU
// repeats each mean by its duration; the decoder — a 1-D U-Net integrated
// over three Euler steps of a flow-matching ODE, all inside one compiled graph
// — refines that into the mel. Every stage is a fixed-shape single pass: no
// decoding loop, so it belongs here rather than next to the LLMs.

/// Text -> Matcha token ids, reproducing sherpa-onnx's lexicon front end for
/// this model: Chinese punctuation folded to ASCII, the text split at
/// punctuation into sentences (short ones merged up to `min_words`, long ones
/// split at `max_words`), each sentence looked up in the lexicon by longest
/// match (so 银行 and 长城 read right where 行 and 长 alone would not), and the
/// sentence's closing punctuation kept as a token.
///
/// Digits are read out first (90 -> 九十, 1.5 -> 一点五, 35% -> 百分之三十五,
/// 2026年 -> 二零二六年); runs of more than 8 digits or with a leading zero are
/// read digit by digit. Latin words are dropped — this model's lexicon has no
/// English, and sherpa's lexicon path drops them too.
struct MatchaFrontendConfig {
  int min_words = 5;   ///< merge sentences shorter than this (CJK char = 1 word)
  int max_words = 20;  ///< split sentences longer than this
};

class MatchaFrontend {
 public:
  /// `tokens`: sherpa tokens.txt ("token id" per line; a line with only an id
  /// is the space token). `lexicon`: "word phone phone ..." per line.
  MatchaFrontend(const std::string& tokens, const std::string& lexicon,
                 MatchaFrontendConfig cfg = {});

  /// One id sequence per sentence; a sentence with no spoken token (only
  /// punctuation, or only words the lexicon lacks) is dropped.
  std::vector<std::vector<int>> encode(const std::string& text) const;

  /// The sentences encode() tokenises, after number reading and punctuation
  /// folding — for inspection.
  std::vector<std::string> sentences(const std::string& text) const;

  /// The text with digits read out as characters (the first step of encode()).
  static std::string readNumbers(const std::string& text);

  int vocabSize() const noexcept { return vocab_; }
  int lexiconSize() const noexcept { return static_cast<int>(lexicon_.size()); }

 private:
  std::vector<int> tokenize(const std::string& sentence) const;

  MatchaFrontendConfig cfg_;
  std::unordered_map<std::string, int> token_;
  std::unordered_map<std::string, std::vector<int>> lexicon_;
  std::unordered_set<int> silent_;  // space and punctuation token ids
  int max_phrase_ = 1, vocab_ = 0;
};

/// Acoustic-model settings.
struct MatchaConfig {
  float noise_scale = 1.0f;    ///< temperature of the decoder's starting noise
  float length_scale = 1.0f;   ///< > 1 speaks slower (sherpa: 1 / speed)
  std::uint32_t seed = 0;      ///< noise RNG seed; the same seed gives the same mel
};

/// One sentence's mel.
struct MatchaResult {
  std::vector<float> mel;          ///< [num_mels, frames] row-major
  int frames = 0;
  std::vector<int> durations;      ///< frames per token (before truncation)
  bool truncated = false;          ///< too many tokens or frames for the build
};

/// Engine-bound Matcha acoustic model: two compiled graphs and the token
/// embedding table.
///   encoder in  x_emb [1, D, L], x_mask [1, 1, L]
///           out mu_x [1, mels, L], logw [1, 1, L]
///   decoder in  mu_y [1, mels, T], y_mask [1, 1, T], z [1, mels, T]  (in that order)
///           out mel [1, mels, T]
/// The embedding lookup runs on the CPU from `embedding` ([vocab, D], already
/// scaled by sqrt(D), as a float32 .npy): fed to the encoder as a float
/// input, token ids are quantised like any activation, and the build gave the
/// id input a range of 127 against a vocabulary of 2190.
/// L, T, D and mels are read from the Engines. The decoder graph carries a
/// GroupNorm and an attention mask rewritten so that padding up to T does not
/// change the result: stock Matcha normalises over, and adds a +0 attention
/// bias to, every padded frame, so a fixed-length build of it is not the model.
class MatchaAcoustic {
 public:
  MatchaAcoustic(Engine& encoder, Engine& decoder, const std::string& embedding,
                 MatchaConfig cfg = {});
  /// Same, with the table in memory: `table` is [vocab, dim] row-major.
  MatchaAcoustic(Engine& encoder, Engine& decoder, std::vector<float> table, int vocab,
                 MatchaConfig cfg = {});

  /// Token ids of ONE sentence -> mel. More than maxTokens() ids, or a
  /// sentence longer than maxFrames(), is truncated and flagged.
  MatchaResult synthesize(const int* ids, int n, int timeout_ms = 0);

  void seed(std::uint32_t s) { rng_.seed(s); }
  int maxTokens() const noexcept { return L_; }
  int maxFrames() const noexcept { return T_; }
  int numMels() const noexcept { return mels_; }
  int vocabSize() const noexcept { return vocab_; }
  const MatchaConfig& config() const noexcept { return cfg_; }

 private:
  void init();

  Engine& enc_;
  Engine& dec_;
  MatchaConfig cfg_;
  std::mt19937 rng_;
  std::vector<float> table_;
  int vocab_ = 0, dim_ = 0;
  int L_ = 0, T_ = 0, mels_ = 0;
  int enc_tok_ = 0, enc_mask_ = 1, enc_mu_ = 0, enc_logw_ = 1;
  int dec_mu_ = 0, dec_mask_ = 1, dec_z_ = 2;
  std::vector<float> emb_, xmask_, mu_x_, logw_, mu_y_, ymask_, z_;
  std::vector<std::uint8_t> stage_;
};

/// Shorten pauses: every run of samples with |x| <= `level` lasting at least
/// `min_pause_s` keeps only `scale` of its length (its start). This is
/// sherpa-onnx's silence_scale, which its Matcha voices are heard with: the
/// model's own pauses at punctuation run long. scale 1 (or outside
/// [0.01, 10]) returns the input; scale > 1 lengthens a pause with zeros.
std::vector<float> scaleSilence(const std::vector<float>& pcm, int sample_rate, float scale,
                                float min_pause_s = 0.2f, float level = 0.01f);

struct TextToSpeechConfig {
  float silence_scale = 0.2f;   ///< see scaleSilence(); 1 keeps the model's pauses
};

/// Text in, 16 kHz PCM out: front end, acoustic model and vocoder, one
/// sentence at a time, concatenated, pauses then shortened.
struct TtsResult {
  std::vector<float> pcm;
  int sample_rate = 16000;
  int sentences = 0;
  bool truncated = false;   ///< some sentence was cut to fit the build
};

class TextToSpeech {
 public:
  TextToSpeech(MatchaFrontend frontend, MatchaAcoustic& acoustic, VocosVocoder& vocoder,
               TextToSpeechConfig cfg = {});
  TtsResult synthesize(const std::string& text, int timeout_ms = 0);
  const MatchaFrontend& frontend() const noexcept { return fe_; }
  const TextToSpeechConfig& config() const noexcept { return cfg_; }

 private:
  MatchaFrontend fe_;
  TextToSpeechConfig cfg_;
  MatchaAcoustic& am_;
  VocosVocoder& voc_;
};

}  // namespace bcdl
