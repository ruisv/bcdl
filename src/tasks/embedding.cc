#include "bcdl/tasks/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "bcdl/backend/engine.h"
#include "bcdl/backend/output_reader.h"
#include "bcdl/core/status.h"

namespace bcdl {

namespace {

/// L2-normalize `v` in place. A zero (or denormal-magnitude) vector is left as
/// it is: dividing by ~0 would poison every later dot product with NaNs, and a
/// zero row simply scores 0 against everything, which is the sane answer.
void l2Normalize(float* v, int dim) {
  double sum = 0.0;
  for (int i = 0; i < dim; ++i) sum += static_cast<double>(v[i]) * v[i];
  const double norm = std::sqrt(sum);
  if (norm <= 1e-12) return;
  const float inv = static_cast<float>(1.0 / norm);
  for (int i = 0; i < dim; ++i) v[i] *= inv;
}

}  // namespace

std::vector<float> decodeEmbedding(const float* data, int dim,
                                   const EmbedConfig& cfg) {
  if (data == nullptr || dim <= 0) return {};
  std::vector<float> out(data, data + dim);
  if (cfg.l2_normalize) l2Normalize(out.data(), dim);
  return out;
}

// --- EmbeddingBank ---------------------------------------------------------

void EmbeddingBank::add(const std::vector<float>& vec, const std::string& label) {
  const int d = static_cast<int>(vec.size());
  if (d <= 0) throw Error(-1, "BCDL embedding: cannot add an empty vector to a bank");
  if (dim_ == 0) {
    dim_ = d;
  } else if (d != dim_) {
    throw Error(-1, "BCDL embedding: bank dimension mismatch (bank " +
                        std::to_string(dim_) + ", added " + std::to_string(d) + ")");
  }
  data_.insert(data_.end(), vec.begin(), vec.end());
  l2Normalize(data_.data() + data_.size() - dim_, dim_);
  labels_.push_back(label);
}

std::vector<EmbedMatch> EmbeddingBank::search(const std::vector<float>& query,
                                              int k) const {
  const int n = size();
  if (n == 0) return {};
  if (static_cast<int>(query.size()) != dim_) {
    throw Error(-1, "BCDL embedding: query dimension mismatch (bank " +
                        std::to_string(dim_) + ", query " +
                        std::to_string(query.size()) + ")");
  }

  // Normalize the query once, not once per entry: with unit rows in the bank,
  // each score is then a bare dot product.
  std::vector<float> q(query);
  l2Normalize(q.data(), dim_);

  std::vector<EmbedMatch> all(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const float* row = data_.data() + static_cast<size_t>(i) * dim_;
    float dot = 0.0f;
    for (int c = 0; c < dim_; ++c) dot += row[c] * q[c];
    all[static_cast<size_t>(i)] = EmbedMatch{i, dot, labels_[static_cast<size_t>(i)]};
  }

  const auto by_score_desc = [](const EmbedMatch& a, const EmbedMatch& b) {
    // Tie-break on index so equal scores come back in a stable, reproducible
    // order (matters for the fixed-vector unit tests).
    if (a.score != b.score) return a.score > b.score;
    return a.index < b.index;
  };

  int take = (k <= 0 || k > n) ? n : k;
  if (take < n) {
    std::partial_sort(all.begin(), all.begin() + take, all.end(), by_score_desc);
    all.resize(static_cast<size_t>(take));
  } else {
    std::sort(all.begin(), all.end(), by_score_desc);
  }
  return all;
}

// --- ImageEmbedder ---------------------------------------------------------

ImageEmbedder::ImageEmbedder(Engine& engine, EmbedConfig cfg, int output_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index) {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "BCDL embedding: output index out of range");
  }
}

int ImageEmbedder::dim() const {
  const std::vector<int> shape = engine_.outputShape(out_idx_);
  int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);
  return static_cast<int>(total);
}

std::vector<float> ImageEmbedder::postprocess() const {
  // Zero-copy for packed F32, dequant-into-scratch otherwise. `scratch` must
  // outlive `data`, so keep it in this scope.
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);

  // dim = product of all dims: robust to [1,D], [1,1,1,D] or a bare [D].
  int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);
  if (total <= 0) return {};

  return decodeEmbedding(data, static_cast<int>(total), cfg_);
}

}  // namespace bcdl
