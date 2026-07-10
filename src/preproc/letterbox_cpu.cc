#include "bcdl/preproc/letterbox_cpu.h"

#ifdef BCDL_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // cv::resize
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bcdl/core/status.h"

namespace bcdl {
namespace {

/// Interleaved channel count for the CPU-supported VP formats.
int channelsOf(hbVPImageFormat fmt) {
  switch (fmt) {
    case HB_VP_IMAGE_FORMAT_BGR:
    case HB_VP_IMAGE_FORMAT_RGB:
      return 3;
    case HB_VP_IMAGE_FORMAT_Y:
      return 1;
    default:
      return 0;  // unsupported on the CPU path
  }
}

inline uint8_t clampU8(float v) {
  const int i = static_cast<int>(std::lround(v));
  return static_cast<uint8_t>(i < 0 ? 0 : (i > 255 ? 255 : i));
}

// Bilinear resize of an interleaved `chans`-channel U8 plane, OpenCV's
// pixel-center convention. Used when OpenCV is absent.
void bilinearPlane(uint8_t* dst, int dstStride, int dw, int dh, const uint8_t* src,
                   int srcStride, int sw, int sh, int chans) {
  const float sx = static_cast<float>(sw) / dw, sy = static_cast<float>(sh) / dh;
#pragma omp parallel for schedule(static)
  for (int y = 0; y < dh; ++y) {
    float fy = (y + 0.5f) * sy - 0.5f;
    if (fy < 0) fy = 0;
    int y0 = static_cast<int>(fy);
    if (y0 > sh - 1) y0 = sh - 1;
    const int y1 = std::min(y0 + 1, sh - 1);
    const float wy = fy - y0;
    uint8_t* d = dst + static_cast<std::size_t>(y) * dstStride;
    for (int x = 0; x < dw; ++x) {
      float fx = (x + 0.5f) * sx - 0.5f;
      if (fx < 0) fx = 0;
      int x0 = static_cast<int>(fx);
      if (x0 > sw - 1) x0 = sw - 1;
      const int x1 = std::min(x0 + 1, sw - 1);
      const float wx = fx - x0;
      for (int c = 0; c < chans; ++c) {
        const float a = src[static_cast<std::size_t>(y0) * srcStride + x0 * chans + c];
        const float b = src[static_cast<std::size_t>(y0) * srcStride + x1 * chans + c];
        const float e = src[static_cast<std::size_t>(y1) * srcStride + x0 * chans + c];
        const float f = src[static_cast<std::size_t>(y1) * srcStride + x1 * chans + c];
        const float top = a + (b - a) * wx, bot = e + (f - e) * wx;
        d[x * chans + c] = clampU8(top + (bot - top) * wy);
      }
    }
  }
}

}  // namespace

LetterboxInfo letterboxCpu(VpImage& dst, const VpImage& src, uint8_t padValue) {
  const int ch = channelsOf(dst.format());
  if (ch == 0 || dst.format() != src.format()) {
    throw Error(-1,
                "letterboxCpu: src/dst must share a CPU-supported format "
                "(BGR/RGB U8C3 or Y U8C1)");
  }

  const int srcW = src.width();
  const int srcH = src.height();
  const int dstW = dst.width();
  const int dstH = dst.height();
  const LetterboxInfo lb = computeLetterbox(srcW, srcH, dstW, dstH);

  // Pre-fill the whole destination buffer (including stride padding) with the
  // pad color so any pixel outside the resized region is already the border.
  std::memset(dst.data(), padValue, dst.mem().size());

  if (srcW <= 0 || srcH <= 0) {
    dst.cleanCache();
    return lb;
  }

  const int srcStride = src.raw().stride;
  const int dstStride = dst.raw().stride;
  const auto* srcBase = static_cast<const uint8_t*>(src.data());
  auto* dstBase = static_cast<uint8_t*>(dst.data());

  // Integer placement of the scaled image inside the canvas. The float scale /
  // padX / padY (which the returned LetterboxInfo carries) are reproduced here
  // as the nearest-integer destination region; per-pixel sampling still uses the
  // float scale below, matching the hbVP affine result.
  const float scale = lb.scale;
  const int padX = static_cast<int>(std::lround(lb.padX));
  const int padY = static_cast<int>(std::lround(lb.padY));
  int newW = static_cast<int>(std::lround(srcW * scale));
  int newH = static_cast<int>(std::lround(srcH * scale));
  newW = std::min(newW, dstW - padX);
  newH = std::min(newH, dstH - padY);

#ifdef BCDL_HAVE_OPENCV
  // cv::resize the source straight into the centered (padX,padY,newW,newH) ROI of
  // the already pad-filled canvas. The ROI Mat carries the canvas row stride, so
  // it writes in place; INTER_LINEAR uses the same pixel-center mapping as the
  // hand path. The border is already memset to padValue above.
  if (newW > 0 && newH > 0) {
    const int cvtype = (ch == 3) ? CV_8UC3 : CV_8UC1;
    const cv::Mat srcMat(srcH, srcW, cvtype, const_cast<uint8_t*>(srcBase),
                         static_cast<std::size_t>(srcStride));
    cv::Mat dstRoi(newH, newW, cvtype,
                   dstBase + static_cast<std::size_t>(padY) * dstStride +
                       static_cast<std::size_t>(padX) * ch,
                   static_cast<std::size_t>(dstStride));
    cv::resize(srcMat, dstRoi, cv::Size(newW, newH), 0.0, 0.0, cv::INTER_LINEAR);
  }
#else
  const float invScale = scale > 0.0f ? 1.0f / scale : 0.0f;

#pragma omp parallel for schedule(static)
  for (int dy = 0; dy < newH; ++dy) {
    // Pixel-center mapping dst -> src (OpenCV INTER_LINEAR convention).
    float fy = (dy + 0.5f) * invScale - 0.5f;
    if (fy < 0.0f) fy = 0.0f;
    if (fy > srcH - 1.0f) fy = static_cast<float>(srcH - 1);
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, srcH - 1);
    const float wy = fy - y0;

    const uint8_t* srcRow0 = srcBase + static_cast<std::size_t>(y0) * srcStride;
    const uint8_t* srcRow1 = srcBase + static_cast<std::size_t>(y1) * srcStride;
    uint8_t* dstRow =
        dstBase + static_cast<std::size_t>(dy + padY) * dstStride +
        static_cast<std::size_t>(padX) * ch;

    for (int dx = 0; dx < newW; ++dx) {
      float fx = (dx + 0.5f) * invScale - 0.5f;
      if (fx < 0.0f) fx = 0.0f;
      if (fx > srcW - 1.0f) fx = static_cast<float>(srcW - 1);
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, srcW - 1);
      const float wx = fx - x0;

      const int c00 = x0 * ch, c01 = x1 * ch;
      uint8_t* outPix = dstRow + static_cast<std::size_t>(dx) * ch;
      for (int c = 0; c < ch; ++c) {
        const float v00 = srcRow0[c00 + c];
        const float v01 = srcRow0[c01 + c];
        const float v10 = srcRow1[c00 + c];
        const float v11 = srcRow1[c01 + c];
        const float top = v00 + (v01 - v00) * wx;
        const float bot = v10 + (v11 - v10) * wx;
        outPix[c] = clampU8(top + (bot - top) * wy);
      }
    }
  }
#endif

  // CPU produced the pixels; flush so a subsequent device (BPU) consumer sees them.
  dst.cleanCache();
  return lb;
}

void bgrToNv12Cpu(VpImage& dstNv12, const VpImage& srcBgr) {
  if (dstNv12.format() != HB_VP_IMAGE_FORMAT_NV12 ||
      srcBgr.format() != HB_VP_IMAGE_FORMAT_BGR) {
    throw Error(-1, "bgrToNv12Cpu: dst must be NV12 and src must be BGR (U8C3)");
  }
  const int w = srcBgr.width();
  const int h = srcBgr.height();
  if (w != dstNv12.width() || h != dstNv12.height()) {
    throw Error(-1, "bgrToNv12Cpu: src and dst dimensions must match");
  }
  if ((w & 1) || (h & 1)) {
    throw Error(-1, "bgrToNv12Cpu: NV12 requires even width and height");
  }

  const int srcStride = srcBgr.raw().stride;
  const int yStride = dstNv12.raw().stride;
  const int uvStride = dstNv12.raw().uvStride;
  const auto* srcBase = static_cast<const uint8_t*>(srcBgr.data());
  auto* yBase = static_cast<uint8_t*>(dstNv12.data());
  // UV plane: use the descriptor's pointer if present, else it follows the Y plane.
  auto* uvBase = static_cast<uint8_t*>(dstNv12.raw().uvVirAddr);
  if (uvBase == nullptr) {
    uvBase = yBase + static_cast<std::size_t>(yStride) * static_cast<std::size_t>(h);
  }

  // BT.601 FULL-range BGR->YUV, matching cv2 COLOR_BGR2YUV_I420 (the proven RDK
  // deploy preprocessing). Full range: Y in [0,255], no +16 luma offset.
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const uint8_t* srcRow = srcBase + static_cast<std::size_t>(y) * srcStride;
    uint8_t* yRow = yBase + static_cast<std::size_t>(y) * yStride;
    for (int x = 0; x < w; ++x) {
      const uint8_t* p = srcRow + static_cast<std::size_t>(x) * 3;
      const float B = p[0], G = p[1], R = p[2];
      yRow[x] = clampU8(0.299f * R + 0.587f * G + 0.114f * B);
    }
  }

  // Chroma: average each 2x2 BGR block, then convert (4:2:0 subsampling).
#pragma omp parallel for schedule(static)
  for (int cy = 0; cy < h / 2; ++cy) {
    const int y0 = cy * 2;
    const uint8_t* srcRow0 = srcBase + static_cast<std::size_t>(y0) * srcStride;
    const uint8_t* srcRow1 = srcBase + static_cast<std::size_t>(y0 + 1) * srcStride;
    uint8_t* uvRow = uvBase + static_cast<std::size_t>(cy) * uvStride;
    for (int cx = 0; cx < w / 2; ++cx) {
      const int x0 = cx * 2;
      const uint8_t* a = srcRow0 + static_cast<std::size_t>(x0) * 3;
      const uint8_t* b = srcRow0 + static_cast<std::size_t>(x0 + 1) * 3;
      const uint8_t* c = srcRow1 + static_cast<std::size_t>(x0) * 3;
      const uint8_t* d = srcRow1 + static_cast<std::size_t>(x0 + 1) * 3;
      const float B = 0.25f * (a[0] + b[0] + c[0] + d[0]);
      const float G = 0.25f * (a[1] + b[1] + c[1] + d[1]);
      const float R = 0.25f * (a[2] + b[2] + c[2] + d[2]);
      uint8_t* uv = uvRow + static_cast<std::size_t>(cx) * 2;
      uv[0] = clampU8(-0.169f * R - 0.331f * G + 0.500f * B + 128.0f);  // U / Cb (full range)
      uv[1] = clampU8(0.500f * R - 0.419f * G - 0.081f * B + 128.0f);   // V / Cr (full range)
    }
  }

  // CPU produced the pixels; flush so a subsequent device (BPU) consumer sees them.
  dstNv12.cleanCache();
}

void nv12ToBgrCpu(VpImage& dstBgr, const VpImage& srcNv12, YuvRange range) {
  if (srcNv12.format() != HB_VP_IMAGE_FORMAT_NV12 ||
      dstBgr.format() != HB_VP_IMAGE_FORMAT_BGR) {
    throw Error(-1, "nv12ToBgrCpu: src must be NV12 and dst must be BGR (U8C3)");
  }
  const int w = srcNv12.width();
  const int h = srcNv12.height();
  if (w != dstBgr.width() || h != dstBgr.height()) {
    throw Error(-1, "nv12ToBgrCpu: src and dst dimensions must match");
  }

  const int yStride = srcNv12.raw().stride;
  const int uvStride = srcNv12.raw().uvStride;
  const int dstStride = dstBgr.raw().stride;
  const auto* yBase = static_cast<const uint8_t*>(srcNv12.data());
  const auto* uvBase = static_cast<const uint8_t*>(srcNv12.raw().uvVirAddr);
  if (uvBase == nullptr) {
    uvBase = yBase + static_cast<std::size_t>(yStride) * static_cast<std::size_t>(h);
  }
  auto* dstBase = static_cast<uint8_t*>(dstBgr.data());

#ifdef BCDL_HAVE_OPENCV
  if (range == YuvRange::kStudioToFull) {
    // Fast path: cv::cvtColor's NV12->BGR IS the studio-swing expansion — same
    // matrix as the hand path below, SIMD. Pack the strided planes into a tight
    // [h*3/2, w] Mat, convert, then copy rows into dst honoring its stride.
    cv::Mat nv12(h * 3 / 2, w, CV_8UC1);
    for (int r = 0; r < h; ++r)
      std::memcpy(nv12.ptr(r), yBase + static_cast<std::size_t>(r) * yStride, w);
    for (int r = 0; r < h / 2; ++r)
      std::memcpy(nv12.ptr(h + r), uvBase + static_cast<std::size_t>(r) * uvStride, w);
    cv::Mat bgr;
    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
    const std::size_t rowBytes = static_cast<std::size_t>(w) * 3;
    for (int r = 0; r < h; ++r)
      std::memcpy(dstBase + static_cast<std::size_t>(r) * dstStride, bgr.ptr(r), rowBytes);
    dstBgr.cleanCache();
    return;
  }
  // kAsIs has no OpenCV equivalent (cv::cvtColor always assumes studio swing),
  // so it falls through to the hand path.
#endif

  // BT.601, either studio-swing expanded (matching cv::cvtColor) or full-range
  // (the exact inverse of bgrToNv12Cpu's forward matrix). Both branches of the
  // build must produce the same pixels for the same `range` — that they did not
  // is the bug this signature fixes.
  const bool studio = (range == YuvRange::kStudioToFull);
  const float ky = studio ? 1.164f : 1.0f;
  const float kr = studio ? 1.596f : 1.402f;
  const float kgu = studio ? 0.391f : 0.344f;
  const float kgv = studio ? 0.813f : 0.714f;
  const float kb = studio ? 2.018f : 1.772f;
  const float yoff = studio ? 16.0f : 0.0f;
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const uint8_t* yRow = yBase + static_cast<std::size_t>(y) * yStride;
    const uint8_t* uvRow = uvBase + static_cast<std::size_t>(y / 2) * uvStride;
    uint8_t* dRow = dstBase + static_cast<std::size_t>(y) * dstStride;
    for (int x = 0; x < w; ++x) {
      const float Y = ky * (static_cast<float>(yRow[x]) - yoff);
      const uint8_t* uv = uvRow + static_cast<std::size_t>(x / 2) * 2;
      const float U = static_cast<float>(uv[0]) - 128.0f;
      const float V = static_cast<float>(uv[1]) - 128.0f;
      uint8_t* p = dRow + static_cast<std::size_t>(x) * 3;
      p[0] = clampU8(Y + kb * U);                 // B
      p[1] = clampU8(Y - kgu * U - kgv * V);      // G
      p[2] = clampU8(Y + kr * V);                 // R
    }
  }

  dstBgr.cleanCache();
}

LetterboxInfo letterboxToNv12Cpu(VpImage& dstNv12, const VpImage& srcBgr,
                                 uint8_t padValue) {
  if (dstNv12.format() != HB_VP_IMAGE_FORMAT_NV12) {
    throw Error(-1, "letterboxToNv12Cpu: dst must be NV12");
  }
  // Letterbox in BGR into a temporary sized to the NV12 canvas, then convert.
  VpImage tmpBgr(dstNv12.width(), dstNv12.height(), HB_VP_IMAGE_FORMAT_BGR);
  const LetterboxInfo lb = letterboxCpu(tmpBgr, srcBgr, padValue);
  bgrToNv12Cpu(dstNv12, tmpBgr);
  return lb;
}

LetterboxInfo letterboxNv12Cpu(VpImage& dstNv12, const VpImage& srcNv12, uint8_t padValue,
                               YuvRange range) {
  if (srcNv12.format() != HB_VP_IMAGE_FORMAT_NV12 ||
      dstNv12.format() != HB_VP_IMAGE_FORMAT_NV12) {
    throw Error(-1, "letterboxNv12Cpu: src and dst must both be NV12");
  }
  const int W = srcNv12.width(), H = srcNv12.height();
  const int OW = dstNv12.width(), OH = dstNv12.height();
  const LetterboxInfo lb = computeLetterbox(W, H, OW, OH);
  // Even-align the content window: NV12 chroma is 2x2 subsampled, so an odd
  // offset or extent would split a chroma sample.
  const int cw = static_cast<int>(std::lround(W * lb.scale)) & ~1;
  const int ch = static_cast<int>(std::lround(H * lb.scale)) & ~1;
  const int px = static_cast<int>(std::lround(lb.padX)) & ~1;
  const int py = static_cast<int>(std::lround(lb.padY)) & ~1;

  const auto& dr = dstNv12.raw();
  auto* dy = static_cast<uint8_t*>(dstNv12.data());
  auto* duv = static_cast<uint8_t*>(dr.uvVirAddr);
  for (int r = 0; r < OH; ++r) std::memset(dy + static_cast<std::size_t>(r) * dr.stride, padValue, OW);
  for (int r = 0; r < OH / 2; ++r) std::memset(duv + static_cast<std::size_t>(r) * dr.uvStride, 128, OW);

  if (cw > 0 && ch > 0) {
    const auto& sr = srcNv12.raw();
    auto* yroi = dy + static_cast<std::size_t>(py) * dr.stride + px;
    auto* uvroi = duv + static_cast<std::size_t>(py / 2) * dr.uvStride + px;
    const auto* sy_p = static_cast<const uint8_t*>(srcNv12.data());
    const auto* suv_p = static_cast<const uint8_t*>(sr.uvVirAddr);
#ifdef BCDL_HAVE_OPENCV
    // Resize straight into the centered ROI of the padded canvas; the ROI Mats
    // carry the canvas strides, so cv::resize writes in place. Interleaved UV is
    // one CV_8UC2 plane at half resolution, so U and V stay paired.
    cv::Mat sy(H, W, CV_8UC1, const_cast<uint8_t*>(sy_p), sr.stride);
    cv::Mat dyroi(ch, cw, CV_8UC1, yroi, dr.stride);
    cv::resize(sy, dyroi, cv::Size(cw, ch), 0, 0, cv::INTER_LINEAR);
    cv::Mat suv(H / 2, W / 2, CV_8UC2, const_cast<uint8_t*>(suv_p), sr.uvStride);
    cv::Mat duvroi(ch / 2, cw / 2, CV_8UC2, uvroi, dr.uvStride);
    cv::resize(suv, duvroi, cv::Size(cw / 2, ch / 2), 0, 0, cv::INTER_LINEAR);
#else
    bilinearPlane(yroi, dr.stride, cw, ch, sy_p, sr.stride, W, H, 1);
    bilinearPlane(uvroi, dr.uvStride, cw / 2, ch / 2, suv_p, sr.uvStride, W / 2, H / 2, 2);
#endif

    if (range == YuvRange::kStudioToFull) {
      uint8_t y_lut[256], uv_lut[256];
      for (int v = 0; v < 256; ++v) {
        y_lut[v] = clampU8((v - 16) * (255.0f / 219.0f));
        uv_lut[v] = clampU8(128.0f + (v - 128) * (255.0f / 224.0f));
      }
      for (int r = py; r < py + ch; ++r) {
        uint8_t* p = dy + static_cast<std::size_t>(r) * dr.stride + px;
        for (int c = 0; c < cw; ++c) p[c] = y_lut[p[c]];
      }
      for (int r = py / 2; r < (py + ch) / 2; ++r) {
        uint8_t* p = duv + static_cast<std::size_t>(r) * dr.uvStride + px;
        for (int c = 0; c < cw; ++c) p[c] = uv_lut[p[c]];
      }
    }
  }
  dstNv12.cleanCache();
  return lb;
}

}  // namespace bcdl
