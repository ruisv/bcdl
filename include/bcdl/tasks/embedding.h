#pragma once

#include <string>
#include <vector>

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Image / text embeddings (retrieval, zero-shot classification)
// ===========================================================================
//
// An embedding model maps an image to a fixed-length vector so that semantic
// similarity becomes geometric closeness. Once every vector is L2-normalized,
// cosine similarity is a plain dot product, which is all the matching below is.
//
// SPLIT ACROSS HOST AND BOARD. A dual-encoder model (SigLIP and friends) has an
// image tower and a text tower that were trained into a SHARED vector space.
// Only the image tower runs here: it is the one that has to see every frame.
// The text tower is run ONCE, offline, on a host — its output for a fixed set of
// prompts is a small table of vectors that ships next to the model. So:
//
//   zero-shot classification = embed the image, dot it against the class-name
//                              table, take the argmax
//   image retrieval          = embed the query, dot it against a table built
//                              from the gallery images (same tower, no text)
//
// Both are the same operation against a different table, which is why
// EmbeddingBank below does not care where its vectors came from.
//
// PACKAGED SUBMODELS. An embedding `.hbm` commonly holds MORE THAN ONE model:
// a pooled whole-image vector and the per-patch feature grid. Engine's
// constructor selects one by name and Engine::modelNames() lists them — pick the
// pooled one for everything here. Selecting the patch-feature submodel by
// accident yields a much larger output that is not a single embedding; the
// dimension check in ImageEmbedder is what catches that.
//
// PREPROCESSING IS NOT LETTERBOX. These towers are trained on a plain squashing
// resize to a square, not an aspect-preserving letterbox — feeding letterboxed
// input shifts the vector enough to cost retrieval accuracy. The Python wrapper
// does the right resize; there is no geometry to invert afterwards, which is why
// nothing in this header takes a LetterboxInfo.

/// Post-processing parameters for an embedding head.
struct EmbedConfig {
  /// L2-normalize the vector on read-out. Leave this on: every similarity in
  /// this header assumes unit vectors so that cosine reduces to a dot product.
  /// Turning it off is for callers that want the raw pooled activations.
  bool l2_normalize = true;
};

/// One entry of a similarity search, most-similar first.
struct EmbedMatch {
  int index;          ///< position of the entry in the bank
  float score;        ///< cosine similarity in [-1, 1] (dot of unit vectors)
  std::string label;  ///< the entry's label, empty if it was added without one
};

/// Read `dim` floats into an embedding vector, optionally L2-normalized.
///
/// `data` : pointer to `dim` contiguous floats (the pooled tower output).
/// `dim`  : embedding width (e.g. 768 / 1024 / 1152 depending on the tower).
/// `cfg`  : normalization toggle.
///
/// A zero vector is returned unchanged rather than producing NaNs. Returns empty
/// if `data` is null or `dim <= 0`.
std::vector<float> decodeEmbedding(const float* data, int dim,
                                   const EmbedConfig& cfg);

/// A searchable table of equal-length embeddings.
///
/// This is the CPU half of both retrieval and zero-shot classification: fill it
/// with gallery-image vectors or with offline-computed text vectors, then
/// search() a freshly embedded image against it. Vectors are normalized on
/// insert, so search() is a dot product per entry — linear scan, which is the
/// right choice at the table sizes that fit on a board (thousands of entries).
///
/// The first add() fixes `dim`; a later add() of a different length throws
/// Error(-1), since a silent dimension mismatch would otherwise surface as
/// meaningless similarity scores.
class EmbeddingBank {
 public:
  EmbeddingBank() = default;

  /// Append one entry. `vec` is L2-normalized into the bank (the caller's copy
  /// is untouched). `label` is free-form — a class name for zero-shot use, an
  /// image id for retrieval, or empty.
  void add(const std::vector<float>& vec, const std::string& label = "");

  /// Search `query` against every entry, returning the `k` best by cosine
  /// similarity, descending. `query` is normalized internally, so it may be a
  /// raw vector. A `k` <= 0 or beyond the entry count returns all entries,
  /// sorted. Returns empty for an empty bank; throws Error(-1) if `query`'s
  /// length disagrees with the bank's `dim`.
  std::vector<EmbedMatch> search(const std::vector<float>& query, int k = 5) const;

  int size() const noexcept { return static_cast<int>(labels_.size()); }
  int dim() const noexcept { return dim_; }
  const std::string& label(int i) const { return labels_.at(i); }

 private:
  int dim_ = 0;
  std::vector<float> data_;         ///< size() * dim_, row-major, unit rows
  std::vector<std::string> labels_;
};

/// Engine-bound image embedder.
///
/// The caller preprocesses + setInput + infer on the Engine. `postprocess()`
/// reads the selected output, flattens it to `dim = product(shape)` floats
/// (F32 directly, F16/quantized via the dequant path in outputAsFloat), and runs
/// decodeEmbedding().
///
/// The flatten is what makes a wrong submodel selection loud: a pooled head is
/// [1,D] (or [1,1,1,D]) and gives the expected width, whereas a patch-feature
/// head is [1,N,D] and flattens to N*D. Check dim() against the tower's
/// documented width once at startup rather than trusting the file.
class ImageEmbedder {
 public:
  ImageEmbedder(Engine& engine, EmbedConfig cfg = {}, int output_index = 0);

  std::vector<float> postprocess() const;

  /// Embedding width taken from the bound output's shape (product of dims).
  int dim() const;

  const EmbedConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  EmbedConfig cfg_;
  int out_idx_;
};

}  // namespace bcdl
