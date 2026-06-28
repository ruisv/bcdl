#pragma once

#include <cstdint>

#include "bcdl/core/sys_mem.h"
#include "hobot/vp/hb_vp.h"

namespace bcdl {

/// RAII, allocation-owning wrapper around an hbVPImage backed by a cached
/// bcdl::SysMem block. This is the unified-buffer image used for VP (vision
/// preprocessing) ops — resize / warp-affine / cvt-color — on S100.
///
/// The backing SysMem follows the same cache discipline as everything else in
/// BCDL: cleanCache() after the CPU writes pixels and before a
/// VP op reads them; invalidateCache() after a VP op (device) writes and before
/// the CPU reads. cleanCache()/invalidateCache() forward to the backing buffer.
///
/// Move-only, like SysMem.
///
/// Supported formats and their plane layout (row stride aligned up to 16 bytes,
/// the conservative alignment the VP units accept for all of these):
///   - HB_VP_IMAGE_FORMAT_BGR / RGB : interleaved C3, stride = align16(width*3).
///   - HB_VP_IMAGE_FORMAT_Y         : grayscale C1, stride = align16(width).
///   - HB_VP_IMAGE_FORMAT_NV12      : Y plane (stride*height) followed by an
///                                    interleaved UV plane (uvStride*height/2);
///                                    requires even width and height.
class VpImage {
 public:
  VpImage() = default;

  /// Allocate an image of (width, height) in `format`, picking the natural
  /// element type for that format via defaultType().
  VpImage(int width, int height, hbVPImageFormat format)
      : VpImage(width, height, format, defaultType(format)) {}

  /// Allocate an image with an explicit element type.
  VpImage(int width, int height, hbVPImageFormat format, hbVPImageType type);

  ~VpImage() = default;

  VpImage(const VpImage&) = delete;
  VpImage& operator=(const VpImage&) = delete;
  VpImage(VpImage&& other) noexcept;
  VpImage& operator=(VpImage&& other) noexcept;

  /// Raw hbVPImage descriptor to pass to hbVP* ops (dst takes the non-const
  /// overload, src the const one).
  hbVPImage& raw() noexcept { return img_; }
  const hbVPImage& raw() const noexcept { return img_; }

  int width() const noexcept { return width_; }
  int height() const noexcept { return height_; }
  hbVPImageFormat format() const noexcept { return format_; }
  bool valid() const noexcept { return mem_.valid(); }

  /// Backing shared-memory buffer.
  SysMem& mem() noexcept { return mem_; }
  const SysMem& mem() const noexcept { return mem_; }

  /// Virtual / physical address of the primary (Y) plane.
  void* data() const noexcept { return img_.dataVirAddr; }
  uint64_t phyAddr() const noexcept { return img_.dataPhyAddr; }

  /// Forward cache control to the backing buffer.
  void cleanCache() const { mem_.cleanCache(); }
  void invalidateCache() const { mem_.invalidateCache(); }

  /// Natural element type for a format (e.g. BGR/RGB -> U8C3, Y/NV12 -> U8C1).
  static hbVPImageType defaultType(hbVPImageFormat format) noexcept;

  /// Byte size of one plane given its row stride and height.
  static uint64_t planeBytes(int32_t stride, int height) noexcept {
    return static_cast<uint64_t>(stride) * static_cast<uint64_t>(height);
  }

 private:
  hbVPImage img_{};
  SysMem mem_;
  int width_ = 0;
  int height_ = 0;
  hbVPImageFormat format_ = HB_VP_IMAGE_FORMAT_BGR;
};

}  // namespace bcdl
