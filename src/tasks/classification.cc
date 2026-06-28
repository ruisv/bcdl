#include "bcdl/tasks/classification.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

namespace bcdl {

std::vector<ClsResult> decodeClassification(const float* logits, int num_classes,
                                            const ClsConfig& cfg) {
  if (logits == nullptr || num_classes <= 0) return {};

  // Build (class_id, score) for every class. score = softmax prob when
  // requested, else the raw logit. Softmax uses the standard max-subtraction
  // trick so exp() never overflows for large logits.
  std::vector<ClsResult> all(static_cast<size_t>(num_classes));
  if (cfg.apply_softmax) {
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; ++i) {
      if (logits[i] > max_logit) max_logit = logits[i];
    }
    double sum = 0.0;  // accumulate in double; 1000 exp() terms add up.
    for (int i = 0; i < num_classes; ++i) {
      const float e = std::exp(logits[i] - max_logit);
      all[i].class_id = i;
      all[i].score = e;  // unnormalised exp for now; divide by sum below.
      sum += e;
    }
    const float inv = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
    for (int i = 0; i < num_classes; ++i) all[i].score *= inv;
  } else {
    for (int i = 0; i < num_classes; ++i) {
      all[i].class_id = i;
      all[i].score = logits[i];
    }
  }

  // How many to keep: all classes when top_k is unset (<=0) or oversized.
  int k = cfg.top_k;
  if (k <= 0 || k > num_classes) k = num_classes;

  const auto by_score_desc = [](const ClsResult& a, const ClsResult& b) {
    return a.score > b.score;
  };

  // Partial sort: only the top-k prefix is fully ordered (cheaper than a full
  // sort when k << num_classes, e.g. top-5 of 1000).
  if (k < num_classes) {
    std::partial_sort(all.begin(), all.begin() + k, all.end(), by_score_desc);
    all.resize(static_cast<size_t>(k));
  } else {
    std::sort(all.begin(), all.end(), by_score_desc);
  }
  return all;
}

Classifier::Classifier(Engine& engine, ClsConfig cfg, int output_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index) {}

std::vector<ClsResult> Classifier::postprocess() const {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "BCDL classification: output index out of range");
  }
  // Zero-copy for packed F32 (the common RDK case), dequant-into-scratch
  // otherwise. `scratch` must outlive `data`, so keep it in this scope.
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);

  // num_classes = product of all dims; robust to [1,1,1,C], [1,C], or [C].
  int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);
  if (total <= 0) return {};

  return decodeClassification(data, static_cast<int>(total), cfg_);
}

}  // namespace bcdl
