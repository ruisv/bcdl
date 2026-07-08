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

void nv12ToBgrCpu(VpImage& dstBgr, const VpImage& srcNv12) {
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
  // Fast path: OpenCV's SIMD NV12->BGR (BT.601). Pack the strided planes into a
  // tight [h*3/2, w] Mat, convert, then copy rows into dst honoring its stride.
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
#else
  // BT.601 FULL-range YUV->BGR, exact inverse of bgrToNv12Cpu's forward matrix.
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const uint8_t* yRow = yBase + static_cast<std::size_t>(y) * yStride;
    const uint8_t* uvRow = uvBase + static_cast<std::size_t>(y / 2) * uvStride;
    uint8_t* dRow = dstBase + static_cast<std::size_t>(y) * dstStride;
    for (int x = 0; x < w; ++x) {
      const float Y = yRow[x];
      const uint8_t* uv = uvRow + static_cast<std::size_t>(x / 2) * 2;
      const float U = static_cast<float>(uv[0]) - 128.0f;
      const float V = static_cast<float>(uv[1]) - 128.0f;
      uint8_t* p = dRow + static_cast<std::size_t>(x) * 3;
      p[0] = clampU8(Y + 1.772f * U);                  // B
      p[1] = clampU8(Y - 0.344f * U - 0.714f * V);     // G
      p[2] = clampU8(Y + 1.402f * V);                  // R
    }
  }
#endif

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

}  // namespace bcdl
