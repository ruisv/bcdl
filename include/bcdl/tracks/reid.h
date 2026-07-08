#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace bcdl {

// ===========================================================================
// ReID appearance embeddings (BoT-SORT / StrongSORT association)
// ===========================================================================
//
// A lightweight ReID head (OSNet / fast-reid distillation) turns each detection
// crop into a D-dim appearance vector. For appearance-gated tracking the vector
// is L2-normalized so that cosine similarity is a plain dot product; matching a
// track's smoothed embedding to a detection is then `1 - dot`. These are the
// pure, board-independent primitives the tracker will call — the full BoT-SORT
// association (IoU + appearance cost matrix, EMA-smoothed gallery) is layered on
// top in tracks/byte_tracker once the ReID `.hbm` is available.

/// L2-normalize an embedding in place (no-op for a zero vector). After this,
/// cosineSimilarity() with another normalized vector is a dot product.
inline void l2Normalize(std::vector<float>& v) {
  double sum = 0.0;
  for (float x : v) sum += static_cast<double>(x) * x;
  const double norm = std::sqrt(sum);
  if (norm <= 0.0) return;
  const float inv = static_cast<float>(1.0 / norm);
  for (float& x : v) x *= inv;
}

/// Return an L2-normalized copy of a raw embedding (e.g. straight off the
/// engine output).
inline std::vector<float> normalizeEmbedding(const float* data, int dim) {
  std::vector<float> v;
  if (data == nullptr || dim <= 0) return v;
  v.assign(data, data + dim);
  l2Normalize(v);
  return v;
}

/// Cosine similarity of two equal-length embeddings in [-1, 1]. Computes the
/// normalized dot product, so the inputs need NOT be pre-normalized; returns 0
/// on length mismatch or a zero vector.
inline float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0f;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  if (na <= 0.0 || nb <= 0.0) return 0.0f;
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

}  // namespace bcdl
