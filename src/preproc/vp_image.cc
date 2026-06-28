#include "bcdl/preproc/vp_image.h"

#include <utility>

#include "bcdl/core/status.h"

namespace bcdl {

namespace {
// Row-stride alignment accepted by the VP units for U8 plane formats. 16 bytes
// is the conservative choice that satisfies BGR/RGB/Y/NV12 alike.
constexpr int32_t kRowAlign = 16;
inline int32_t alignUp(int32_t v, int32_t a) { return (v + (a - 1)) & ~(a - 1); }
}  // namespace

hbVPImageType VpImage::defaultType(hbVPImageFormat format) noexcept {
  switch (format) {
    case HB_VP_IMAGE_FORMAT_BGR:
    case HB_VP_IMAGE_FORMAT_RGB:
      return HB_VP_IMAGE_TYPE_U8C3;
    case HB_VP_IMAGE_FORMAT_Y:
    case HB_VP_IMAGE_FORMAT_NV12:
    default:
      return HB_VP_IMAGE_TYPE_U8C1;
  }
}

VpImage::VpImage(int width, int height, hbVPImageFormat format, hbVPImageType type)
    : width_(width), height_(height), format_(format) {
  if (width <= 0 || height <= 0) {
    throw Error(-1, "BCDL: VpImage dimensions must be positive");
  }

  int32_t stride = 0;    // primary (Y) plane row bytes
  int32_t uvStride = 0;  // NV12 interleaved-UV plane row bytes
  uint64_t ySize = 0;    // primary plane byte size (= NV12 UV plane offset)
  uint64_t total = 0;    // backing buffer byte size

  switch (format) {
    case HB_VP_IMAGE_FORMAT_BGR:
    case HB_VP_IMAGE_FORMAT_RGB:
      stride = alignUp(width * 3, kRowAlign);
      total = planeBytes(stride, height);
      break;
    case HB_VP_IMAGE_FORMAT_Y:
      stride = alignUp(width, kRowAlign);
      total = planeBytes(stride, height);
      break;
    case HB_VP_IMAGE_FORMAT_NV12:
      if ((width & 1) || (height & 1)) {
        throw Error(-1, "BCDL: NV12 requires even width/height");
      }
      stride = alignUp(width, kRowAlign);
      // Interleaved UV row covers width/2 (U,V) pairs == width bytes, same
      // stride as Y; the plane is half-height.
      uvStride = stride;
      ySize = planeBytes(stride, height);
      total = ySize + planeBytes(uvStride, height / 2);
      break;
    default:
      throw Error(-1, "BCDL: VpImage unsupported format");
  }

  mem_ = SysMem(total, /*cached=*/true);

  img_.imageFormat = static_cast<uint8_t>(format);
  img_.imageType = static_cast<uint8_t>(type);
  img_.width = width;
  img_.height = height;
  img_.stride = stride;
  img_.dataVirAddr = mem_.data();
  img_.dataPhyAddr = mem_.phyAddr();
  if (format == HB_VP_IMAGE_FORMAT_NV12) {
    img_.uvVirAddr = static_cast<uint8_t*>(mem_.data()) + ySize;
    img_.uvPhyAddr = mem_.phyAddr() + ySize;
    img_.uvStride = uvStride;
  } else {
    img_.uvVirAddr = nullptr;
    img_.uvPhyAddr = 0;
    img_.uvStride = 0;
  }
}

VpImage::VpImage(VpImage&& other) noexcept
    : img_(other.img_),
      mem_(std::move(other.mem_)),
      width_(other.width_),
      height_(other.height_),
      format_(other.format_) {
  // SysMem move preserves the virtual/physical addresses, so the copied img_
  // descriptor still points at the (now owned-by-us) buffer. Reset the source.
  other.img_ = hbVPImage{};
  other.width_ = 0;
  other.height_ = 0;
}

VpImage& VpImage::operator=(VpImage&& other) noexcept {
  if (this != &other) {
    mem_ = std::move(other.mem_);
    img_ = other.img_;
    width_ = other.width_;
    height_ = other.height_;
    format_ = other.format_;
    other.img_ = hbVPImage{};
    other.width_ = 0;
    other.height_ = 0;
  }
  return *this;
}

}  // namespace bcdl
