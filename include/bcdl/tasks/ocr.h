#pragma once

#include <string>
#include <vector>

#include "bcdl/preproc/geometry.h"

namespace bcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// PaddleOCR-style OCR post-processing (det + cls + rec)
// ===========================================================================
//
// Three independent stages, each a layout/precision-agnostic pure decode
// function plus a thin Engine-bound wrapper that adapts the device tensor via
// outputAsFloat(). The end-to-end pipeline (det boxes -> per-box crop ->
// optional cls 180° flip -> rec text) is intentionally NOT assembled here; the
// application / Python layer composes the three stages, since cropping and
// re-preprocessing each text line is a host-image operation outside this lib.
//
// NOTE: no OpenCV / third-party. Connected-component labelling and the unclip
// dilation are implemented by hand below.

// ---------------------------------------------------------------------------
// A. Text recognition (CRNN + CTC greedy decode)  — the important one
// ---------------------------------------------------------------------------

/// Load a character dictionary, one token per line (UTF-8 bytes kept verbatim).
/// Mirrors ccdl's `load_char_dict`: `dict[i]` is the token for class index `i`.
///
/// INDEX CONVENTION (matches both ccdl post_process_crnn and the official RDK
/// paddle_ocr ctc_greedy_decode): the CTC *blank* lives at index 0 and emitted
/// characters are looked up as `dict[idx]` directly (NOT `dict[idx-1]`). The
/// caller is therefore expected to pass a dictionary whose first line is the
/// blank placeholder (PaddleOCR's exported `*_dict.txt` convention is to
/// prepend "blank" and, for some models, append a trailing space token). This
/// function does NOT inject blank/space — the file is loaded as-is.
///
/// Throws Error(-1, ...) if the file cannot be opened (ccdl exit() -> throw).
std::vector<std::string> loadCharDict(const std::string& path);

/// One recognised text line.
struct RecResult {
  std::string text;  ///< decoded UTF-8 string (may be empty)
  float score;       ///< mean of the per-step max value over emitted steps,
                     ///< in [0,1] when the rec head emits softmax probs; 0 when
                     ///< nothing was emitted.
};

/// CTC best-path (greedy) decode of a recognition head's logits/probs.
///
/// `logits`      : row-major [num_steps, num_classes] (i.e. the [1,T,C] rec
///                 output with the batch dim dropped). logits[t*C + c].
/// `num_steps`   : T, the sequence length.
/// `num_classes` : C, vocabulary size INCLUDING the blank at index 0.
/// `dict`        : token table, `dict[idx]` for emitted idx (blank = index 0).
///
/// Algorithm (identical to ccdl post_process_crnn):
///   for each timestep t: idx = argmax_c logits[t]; if idx > 0 AND it differs
///   from the previous step's idx, append dict[idx]; always update last_index.
/// This collapses repeats and drops blanks. `score` is the mean of the max
/// value at exactly the steps that emitted a character (ccdl kept this commented
/// out; we implement it). Out-of-range idx is skipped defensively.
RecResult decodeCtc(const float* logits, int num_steps, int num_classes,
                    const std::vector<std::string>& dict);

/// Engine-bound text recogniser. Reads output[out_idx] as float (zero-copy F32
/// fast path, dequant otherwise), treats the last shape dim as num_classes and
/// the remaining product as num_steps (the [1,T,C] PaddleOCR rec layout), and
/// runs decodeCtc().
class TextRecognizer {
 public:
  /// Loads the dictionary from `dict_path` at construction.
  TextRecognizer(Engine& engine, std::string dict_path, int out_idx = 0);

  RecResult postprocess() const;

  const std::vector<std::string>& dict() const { return dict_; }

 private:
  Engine& engine_;
  std::vector<std::string> dict_;
  int out_idx_;
};

// ---------------------------------------------------------------------------
// B. Text-line orientation classifier (0° / 180°)
// ---------------------------------------------------------------------------

/// Result of the 2-class direction classifier.
struct ClsDirResult {
  int label;      ///< argmax: 0 == upright, 1 == rotated 180°
  float score;    ///< score of the chosen label (softmax prob when the head
                  ///< emits probabilities)
  bool flip180;   ///< true when label==1 AND score>thresh: the caller should
                  ///< rotate the cropped text line 180° before recognition.
};

/// Decode a direction-classifier logit/prob vector into a ClsDirResult
/// (Engine-free; the numpy entry point that mirrors TextAngleClassifier).
///
/// `logits` : row-major length-`n` scores ([1,2] / [1,N] with the batch dim
///            dropped). `n` : element count. `thresh` : flip gate.
/// Takes the argmax as the label and its value as the score; `flip180` is set
/// when label==1 and score > thresh. Returns {0,0,false} for empty input.
ClsDirResult decodeClsDir(const float* logits, int n, float thresh);

/// Engine-bound 0°/180° text-line orientation classifier. Reads output[out_idx]
/// as a length-2 vector ([1,2]), takes the argmax as the label and its value as
/// the score. `flip180` is set when label==1 and score exceeds `thresh`.
class TextAngleClassifier {
 public:
  TextAngleClassifier(Engine& engine, float thresh = 0.9f, int out_idx = 0);

  ClsDirResult postprocess() const;

  float threshold() const { return thresh_; }

 private:
  Engine& engine_;
  float thresh_;
  int out_idx_;
};

// ---------------------------------------------------------------------------
// C. Text detection (DBNet) — pure-C++ CCL + axis-aligned unclip
// ---------------------------------------------------------------------------

/// One detected text region, in ORIGINAL-image pixel coordinates (already
/// un-letterboxed).
///
/// PaddleOCR's DB head yields a rotated min-area-rect per region. With OpenCV
/// available, `pts` holds that rotated four-point box (clockwise); the
/// no-OpenCV fallback fills `pts` with the axis-aligned rectangle's corners.
/// `x1..y2` is always the axis-aligned bounding box of `pts` (handy for
/// filtering / cropping).
struct TextBox {
  float pts[8];   ///< 4 corner points (x0,y0,x1,y1,x2,y2,x3,y3), clockwise
  float x1, y1;   ///< axis-aligned bbox top-left     (min of pts)
  float x2, y2;   ///< axis-aligned bbox bottom-right  (max of pts)
  float score;    ///< mean DB-map probability inside the region
};

/// DBNet post-processing parameters (PaddleOCR DBPostProcess).
struct DbConfig {
  /// Binarisation threshold on the [0,1] probability map (PaddleOCR `thresh`).
  /// Foreground = prob > bin_thresh.
  float bin_thresh = 0.3f;
  /// Minimum mean probability INSIDE a region for it to be kept (PaddleOCR
  /// `box_thresh`), computed as box_score_fast (polygon-masked mean of the map).
  float box_thresh = 0.6f;
  /// Unclip dilation ratio (PaddleOCR `unclip_ratio`); larger => wider boxes.
  float unclip_ratio = 1.5f;
  /// Drop a region when its (pre-unclip) shorter side is below this many pixels.
  int min_size = 3;
  /// Connectivity for the no-OpenCV connected-component fallback: 4 or 8.
  int connectivity = 8;
};

/// Decode a DBNet probability map into axis-aligned text boxes.
///
/// `prob` : row-major [H, W] floats in [0,1] (the [1,1,H,W] / [1,H,W,1] map with
///          the unit dims dropped). prob[y*W + x].
/// `H,W`  : probability-map spatial dims.
/// `cfg`  : threshold / unclip / filter parameters.
/// `lb`   : letterbox geometry used at preprocess time. Probability-map pixels
///          are first scaled to model-input pixels by (lb.dstW/W, lb.dstH/H)
///          (handles a down-sampled DB head), then mapped back to original-image
///          pixels via lb.invX/invY and clamped to the source extent. When `lb`
///          is default-constructed (dstW/dstH == 0) the scale is identity and
///          coordinates are returned in probability-map pixels.
///
/// Steps (OpenCV path): binarise -> cv::findContours -> cv::minAreaRect per
/// region -> box_score_fast (polygon-masked mean) + box_thresh/min_size filter
/// -> unclip the rotated rect -> map its 4 points back to original-image pixels.
/// Without OpenCV, a hand-rolled BFS connected-component fallback yields the
/// axis-aligned box instead (pts = its 4 corners).
///
/// UNCLIP: expansion distance d = area * unclip_ratio / perimeter (area/perimeter
/// of the min-area rect); the rect's w,h are each grown by 2*d about its center,
/// approximating PaddleOCR's Pyclipper polygon offset.
std::vector<TextBox> decodeDbnet(const float* prob, int H, int W,
                                 const DbConfig& cfg, const LetterboxInfo& lb);

/// Engine-bound DBNet detector. Reads output[out_idx] as a float probability
/// map, infers H/W from its shape (the two non-unit dims, in order — works for
/// [1,1,H,W] and [1,H,W,1]), and runs decodeDbnet().
class DbTextDetector {
 public:
  DbTextDetector(Engine& engine, DbConfig cfg = {}, int out_idx = 0);

  std::vector<TextBox> postprocess(const LetterboxInfo& lb) const;

  const DbConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  DbConfig cfg_;
  int out_idx_;
};

}  // namespace bcdl
