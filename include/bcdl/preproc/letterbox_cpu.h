#pragma once

#include <cstdint>

#include "bcdl/preproc/geometry.h"
#include "bcdl/preproc/vp_image.h"

namespace bcdl {

/// Pure-CPU (vDSP-free) letterbox + color-conversion fallbacks for S100/S100P.
///
/// WHY THIS EXISTS: the hbVP geometric ops (letterbox()/cvtColor()/resizeExact()
/// in letterbox.h) are vDSP-backed, and the vDSP firmware is OFFLINE / root-only
/// on the deployed board. These functions reproduce the SAME preprocessing on the
/// A78AE CPUs (optionally OpenMP-parallelised) so the preproc layer works WITHOUT
/// the DSP. They mirror the proven cv::resize + cv::copyMakeBorder + BGR->NV12
/// deploy path. No OpenCV dependency — plain C++17.
///
/// Geometry is computed with computeLetterbox() exactly like the hbVP path, so
/// the returned LetterboxInfo (and therefore the un-letterbox inverse map used by
/// post-processing) is byte-for-byte the same contract as letterbox().

/// Aspect-preserving "letterbox" resize of `src` into the pre-allocated `dst`,
/// performed entirely on the CPU. Supports interleaved BGR / RGB (U8C3) and
/// grayscale Y (U8C1); `src` and `dst` must share the same format.
///
/// `dst` is first filled with `padValue` (the borders), then `src` is bilinearly
/// resized into the centered (padX, padY, newW, newH) region. Each image's row
/// stride (raw().stride) is honored, so width-misaligned strides are safe.
///
/// Sampling uses the OpenCV pixel-center convention
///     srcX = (dstX - padX + 0.5) / scale - 0.5
///     srcY = (dstY - padY + 0.5) / scale - 0.5
/// (clamped to the valid source range), so output matches cv::resize(INTER_LINEAR)
/// and the Python letterbox_numpy reference.
///
/// Cache discipline: this is a CPU-only producer. The data it writes is normally
/// consumed next by the BPU/Engine (a device reader), so dst.cleanCache() is
/// called at the end to flush the CPU cache to DRAM — a subsequent device
/// consumer then sees the pixels. (Harmless if the next reader is the CPU.)
/// Returns the geometry needed to un-letterbox detections.
LetterboxInfo letterboxCpu(VpImage& dst, const VpImage& src, uint8_t padValue = 114);

/// CPU BGR (U8C3) -> NV12 conversion. `dst` must be HB_VP_IMAGE_FORMAT_NV12 with
/// the SAME width/height as `src` (both must be even). The Y plane stride
/// (dst.raw().stride) and the interleaved UV plane stride (dst.raw().uvStride) are
/// honored. dst.cleanCache() is called at the end (same rationale as above).
///
/// COLOR CONVENTION: BT.601 FULL-RANGE, matching cv2's COLOR_BGR2YUV_I420 (the
/// proven RDK deploy preprocessing path):
///     Y =  0.299*R + 0.587*G + 0.114*B            (luma in [0,255])
///     U = -0.169*R - 0.331*G + 0.500*B + 128       (Cb)
///     V =  0.500*R - 0.419*G - 0.081*B + 128       (Cr)
/// Chroma is subsampled 4:2:0 by averaging each 2x2 RGB block BEFORE computing
/// U/V (closer to cv2 than even-pixel sampling). Results are rounded to nearest
/// and clamped to [0,255]. The interleaved UV layout is [U0 V0 U1 V1 ...].
///
/// CAVEAT (must be validated on-board): full- vs limited-range is model-specific.
/// If the .hbm was calibrated on limited/studio-range NV12, swap to the studio
/// coefficients (Y = 0.257R+0.504G+0.098B+16, ...). Verify against the model's
/// expected input range before trusting accuracy.
void bgrToNv12Cpu(VpImage& dstNv12, const VpImage& srcBgr);

/// CPU NV12 -> BGR (U8C3) conversion — the inverse of bgrToNv12Cpu(), using the
/// SAME BT.601 FULL-RANGE convention (so an NV12->BGR->NV12 round-trip is
/// self-consistent). `dst` must be HB_VP_IMAGE_FORMAT_BGR with the same
/// width/height as the NV12 `src`. The source Y stride (src.raw().stride) and
/// interleaved-UV stride (src.raw().uvStride) are honored; dst.cleanCache() is
/// called at the end. Hand-written (no OpenCV dependency), OpenMP over rows.
///     R = Y + 1.402*(V-128)
///     G = Y - 0.344*(U-128) - 0.714*(V-128)
///     B = Y + 1.772*(U-128)
void nv12ToBgrCpu(VpImage& dstBgr, const VpImage& srcNv12);

/// Convenience: letterbox a BGR (U8C3) `src` directly into an NV12 `dst`. Resizes
/// and pads in BGR via an internal temporary, then bgrToNv12Cpu()'s into `dst`.
/// `dst` must be HB_VP_IMAGE_FORMAT_NV12 (even dims). Returns the letterbox
/// geometry. Same color convention and cache discipline as the two helpers above.
LetterboxInfo letterboxToNv12Cpu(VpImage& dstNv12, const VpImage& srcBgr,
                                 uint8_t padValue = 114);

}  // namespace bcdl
