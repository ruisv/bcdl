#pragma once

#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Post-processing parameters for an image-classification head.
struct ClsConfig {
  /// Number of top-scoring classes to return, sorted by score descending. A
  /// value <= 0 (or one exceeding the class count) returns ALL classes sorted.
  int top_k = 5;
  /// true  => scores are softmax probabilities over the `num_classes` logits.
  /// false => scores are the raw logits (argmax / ordering is identical, but
  ///          values are not normalised to [0,1]). Softmax is monotonic, so the
  ///          ranking and the chosen class ids are the same either way; the flag
  ///          only changes the reported `score` magnitude.
  bool apply_softmax = true;
};

/// One Top-K classification entry.
struct ClsResult {
  int class_id;   ///< class index (e.g. 0..999 for ImageNet)
  float score;    ///< softmax probability in [0,1], or the raw logit when
                  ///< ClsConfig::apply_softmax is false.
};

/// Decode `num_classes` contiguous logits into Top-K classification results.
///
/// `logits`      : pointer to `num_classes` floats in row-major order.
/// `num_classes` : number of class scores (e.g. 1000 for ImageNet).
/// `cfg`         : top_k + softmax toggle.
///
/// When `cfg.apply_softmax` is set the scores are a numerically-stable softmax
/// (subtract the max logit, exponentiate, normalise by the sum) so the returned
/// `score` is a probability in [0,1]. Otherwise the raw logit is used as the
/// score. Results are sorted by score descending and truncated to
/// `cfg.top_k` (all classes when top_k <= 0 or > num_classes). Returns empty if
/// `logits` is null or `num_classes <= 0`.
std::vector<ClsResult> decodeClassification(const float* logits, int num_classes,
                                            const ClsConfig& cfg);

/// Convenience wrapper binding an Engine output to a ClsConfig.
///
/// The caller is responsible for preprocessing + setInput + infer on the Engine
/// (the preproc layer / Python does that). `postprocess()` reads the selected
/// output (shape [1,1,1,C] or [1,C], etc.), flattens it to `C = product(shape)`
/// logits — converting F32 directly, F16/quantized via the dequant path in
/// outputAsFloat — and runs decodeClassification().
class Classifier {
 public:
  Classifier(Engine& engine, ClsConfig cfg = {}, int output_index = 0);

  std::vector<ClsResult> postprocess() const;

  const ClsConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  ClsConfig cfg_;
  int out_idx_;
};

}  // namespace bcdl
