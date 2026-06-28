#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace bcdl {

/// Result of an aspect-preserving "letterbox" fit of a source image of size
/// (srcW, srcH) into a model input canvas of size (dstW, dstH).
///
/// A single uniform `scale` is applied to both axes; the scaled image is then
/// centered in the canvas with `padX`/`padY` pixels of border on each leading
/// edge. This is the standard YOLO preprocessing geometry — the inverse map
/// below is what post-processing uses to project boxes from model-input
/// coordinates back to original-image pixels.
struct LetterboxInfo {
  float scale = 1.0f;  ///< srcPix * scale -> dstPix (same on x and y)
  float padX = 0.0f;   ///< left padding added in the canvas (model coords)
  float padY = 0.0f;   ///< top padding added in the canvas (model coords)
  int srcW = 0;
  int srcH = 0;
  int dstW = 0;
  int dstH = 0;

  /// Forward map: original-image pixel -> model-input pixel.
  float fwdX(float x) const { return x * scale + padX; }
  float fwdY(float y) const { return y * scale + padY; }

  /// Inverse map: model-input pixel -> original-image pixel (un-letterbox).
  float invX(float x) const { return (x - padX) / scale; }
  float invY(float y) const { return (y - padY) / scale; }

  /// Clamp an original-image x/y to the valid source range.
  float clampX(float x) const { return std::min(std::max(x, 0.0f), static_cast<float>(srcW)); }
  float clampY(float y) const { return std::min(std::max(y, 0.0f), static_cast<float>(srcH)); }
};

/// Compute the letterbox geometry to fit (srcW,srcH) into (dstW,dstH).
///
/// `centerPad` centers the scaled image (YOLOv5/v8 default); when false the
/// image is pinned to the top-left (padX=padY=0), matching some exporters.
inline LetterboxInfo computeLetterbox(int srcW, int srcH, int dstW, int dstH,
                                      bool centerPad = true) {
  LetterboxInfo lb;
  lb.srcW = srcW;
  lb.srcH = srcH;
  lb.dstW = dstW;
  lb.dstH = dstH;
  if (srcW <= 0 || srcH <= 0) return lb;
  lb.scale = std::min(static_cast<float>(dstW) / srcW, static_cast<float>(dstH) / srcH);
  const float newW = srcW * lb.scale;
  const float newH = srcH * lb.scale;
  if (centerPad) {
    lb.padX = (dstW - newW) * 0.5f;
    lb.padY = (dstH - newH) * 0.5f;
  }
  return lb;
}

}  // namespace bcdl
