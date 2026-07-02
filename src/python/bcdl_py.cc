// nanobind bindings for BCDL.
//
// M0 keeps the data path explicit and dtype-agnostic: inputs come in as a
// C-contiguous numpy array (raw bytes copied into the device buffer), outputs
// go out as raw bytes + shape + dtype string. The pure-python `bcdl` wrapper
// (python/bcdl/__init__.py) turns those into numpy arrays.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bcdl/backend/engine.h"
#include "bcdl/version.h"
#include "bcdl/media/jpeg_codec.h"
#include "bcdl/media/video_codec.h"
#include "bcdl/preproc/geometry.h"
#include "bcdl/preproc/vp_image.h"
#ifdef BCDL_HAVE_GDC
#include "bcdl/preproc/gdc_letterbox.h"
#endif
#include "bcdl/tasks/classification.h"
#include "bcdl/tasks/depth.h"
#include "bcdl/tasks/detection.h"
#include "bcdl/tasks/instance_seg.h"
#include "bcdl/tasks/mono3d.h"
#include "bcdl/tasks/obb.h"
#include "bcdl/tasks/ocr.h"
#include "bcdl/tasks/pose.h"
#include "bcdl/tasks/segmentation.h"
#include "bcdl/pipeline/async_detection_pipeline.h"
#include "bcdl/pipeline/detection_pipeline.h"
#include "bcdl/pipeline/stereo_pipeline.h"
#include "bcdl/pipeline/tracking_pipeline.h"
#include "bcdl/tracks/byte_tracker.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

const char* dtypeName(int t) {
  switch (t) {
    case HB_DNN_TENSOR_TYPE_S8: return "int8";
    case HB_DNN_TENSOR_TYPE_U8: return "uint8";
    case HB_DNN_TENSOR_TYPE_F16: return "float16";
    case HB_DNN_TENSOR_TYPE_S16: return "int16";
    case HB_DNN_TENSOR_TYPE_U16: return "uint16";
    case HB_DNN_TENSOR_TYPE_F32: return "float32";
    case HB_DNN_TENSOR_TYPE_S32: return "int32";
    case HB_DNN_TENSOR_TYPE_U32: return "uint32";
    case HB_DNN_TENSOR_TYPE_F64: return "float64";
    case HB_DNN_TENSOR_TYPE_S64: return "int64";
    case HB_DNN_TENSOR_TYPE_U64: return "uint64";
    default: return "uint8";
  }
}

// Build an OWNING numpy array from a std::vector<T>. The vector is moved to the
// heap and freed via a capsule when the numpy array is garbage-collected, so the
// returned array owns its data — no dangling pointers, no extra copy of the
// payload. This is the ergonomic counterpart to the Engine's bytes path: tasks
// that produce typed buffers (depth float32, seg int32, colorized uint8) return
// real numpy arrays the caller can use directly.
template <typename T>
nb::ndarray<nb::numpy, T> toNumpy(std::vector<T> vec,
                                  std::initializer_list<size_t> shape) {
  auto* held = new std::vector<T>(std::move(vec));
  nb::capsule owner(held, [](void* p) noexcept {
    delete static_cast<std::vector<T>*>(p);
  });
  // nanobind copies the shape into its own storage at construction; the data
  // pointer is retained and freed via `owner`. Use the (data, ndim, shape*,
  // owner) constructor — the form documented across nanobind versions.
  std::vector<size_t> dims(shape);
  return nb::ndarray<nb::numpy, T>(held->data(), dims.size(), dims.data(),
                                   owner);
}

}  // namespace

NB_MODULE(bcdl_py, m) {
  m.doc() = "BCDL — BPU Computational Deep Learning (C++ core)";
  m.attr("__version__") = BCDL_VERSION_STRING;

  nb::class_<bcdl::Engine>(m, "Engine")
      .def(nb::init<const std::string&, const std::string&>(), "hbm_path"_a,
           "model_name"_a = "")
      .def_prop_ro("model_name", &bcdl::Engine::modelName)
      .def_prop_ro("num_inputs", &bcdl::Engine::numInputs)
      .def_prop_ro("num_outputs", &bcdl::Engine::numOutputs)
      .def("input_shape", &bcdl::Engine::inputShape, "index"_a)
      .def("output_shape", &bcdl::Engine::outputShape, "index"_a)
      .def("input_bytes",
           [](bcdl::Engine& e, int i) { return e.inputBytes(i); }, "index"_a)
      .def("input_dtype",
           [](bcdl::Engine& e, int i) { return dtypeName(e.inputType(i)); }, "index"_a)
      .def("output_dtype",
           [](bcdl::Engine& e, int i) { return dtypeName(e.outputType(i)); }, "index"_a)
      .def(
          "set_input",
          [](bcdl::Engine& e, int i, nb::ndarray<nb::c_contig> arr) {
            e.setInput(i, arr.data(), arr.nbytes());
          },
          "index"_a, "array"_a)
      .def("infer", &bcdl::Engine::infer, "timeout_ms"_a = 0)
      .def(
          "output_bytes",
          [](bcdl::Engine& e, int i) {
            return nb::bytes(static_cast<const char*>(e.outputData(i)),
                             e.outputBytes(i));
          },
          "index"_a);

  // --- preproc geometry: letterbox fit + un-letterbox map --------------------
  nb::class_<bcdl::LetterboxInfo>(m, "LetterboxInfo")
      .def(nb::init<>())
      .def_rw("scale", &bcdl::LetterboxInfo::scale)
      .def_rw("pad_x", &bcdl::LetterboxInfo::padX)
      .def_rw("pad_y", &bcdl::LetterboxInfo::padY)
      .def_rw("src_w", &bcdl::LetterboxInfo::srcW)
      .def_rw("src_h", &bcdl::LetterboxInfo::srcH)
      .def_rw("dst_w", &bcdl::LetterboxInfo::dstW)
      .def_rw("dst_h", &bcdl::LetterboxInfo::dstH)
      .def("fwd_x", &bcdl::LetterboxInfo::fwdX, "x"_a)
      .def("fwd_y", &bcdl::LetterboxInfo::fwdY, "y"_a)
      .def("inv_x", &bcdl::LetterboxInfo::invX, "x"_a)
      .def("inv_y", &bcdl::LetterboxInfo::invY, "y"_a);

  m.def("compute_letterbox", &bcdl::computeLetterbox, "src_w"_a, "src_h"_a,
        "dst_w"_a, "dst_h"_a, "center_pad"_a = true,
        "Aspect-preserving fit of (src_w,src_h) into (dst_w,dst_h).");

  // --- detection post-processing ---------------------------------------------
  nb::enum_<bcdl::DecodeLayout>(m, "DecodeLayout")
      .value("YoloV8", bcdl::DecodeLayout::kYoloV8)
      .value("YoloV5", bcdl::DecodeLayout::kYoloV5);

  nb::class_<bcdl::DetectConfig>(m, "DetectConfig")
      .def(nb::init<>())
      .def_rw("input_w", &bcdl::DetectConfig::input_w)
      .def_rw("input_h", &bcdl::DetectConfig::input_h)
      .def_rw("num_classes", &bcdl::DetectConfig::num_classes)
      .def_rw("conf_thresh", &bcdl::DetectConfig::conf_thresh)
      .def_rw("iou_thresh", &bcdl::DetectConfig::iou_thresh)
      .def_rw("max_dets", &bcdl::DetectConfig::max_dets)
      .def_rw("layout", &bcdl::DetectConfig::layout)
      .def_rw("channels_first", &bcdl::DetectConfig::channels_first)
      .def_rw("apply_sigmoid", &bcdl::DetectConfig::apply_sigmoid);

  nb::class_<bcdl::Detection>(m, "Detection")
      .def(nb::init<>())
      .def_rw("x1", &bcdl::Detection::x1)
      .def_rw("y1", &bcdl::Detection::y1)
      .def_rw("x2", &bcdl::Detection::x2)
      .def_rw("y2", &bcdl::Detection::y2)
      .def_rw("score", &bcdl::Detection::score)
      .def_rw("class_id", &bcdl::Detection::class_id)
      .def("__repr__", [](const bcdl::Detection& d) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "Detection(cls=%d score=%.3f box=[%.1f,%.1f,%.1f,%.1f])",
                      d.class_id, d.score, d.x1, d.y1, d.x2, d.y2);
        return std::string(buf);
      });

  // decode(): the numpy Python path — pass the (already-float, contiguous)
  // engine output as an ndarray; its shape drives the layout. Returns
  // un-letterboxed Detections in original-image pixels.
  m.def(
      "decode",
      [](nb::ndarray<float, nb::c_contig> arr, const bcdl::DetectConfig& cfg,
         const bcdl::LetterboxInfo& lb) {
        std::vector<int> shape;
        shape.reserve(arr.ndim());
        for (size_t d = 0; d < arr.ndim(); ++d)
          shape.push_back(static_cast<int>(arr.shape(d)));
        return bcdl::decode(arr.data(), shape, cfg, lb);
      },
      "output"_a, "config"_a, "letterbox"_a);

  m.def("nms", &bcdl::nms, "dets"_a, "iou_thresh"_a, "max_dets"_a = 300);
  m.def("iou", &bcdl::iou, "a"_a, "b"_a);

  // Detector: read the Engine's output directly (handles F16/quantized dequant
  // and device strides) and post-process. Run engine.infer() first.
  nb::class_<bcdl::Detector>(m, "Detector")
      .def(nb::init<bcdl::Engine&, bcdl::DetectConfig, int>(), "engine"_a,
           "config"_a, "output_index"_a = 0, nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::Detector::postprocess, "letterbox"_a)
      .def_prop_ro("config", &bcdl::Detector::config);

  // --- anchor-free LTRB multi-scale head (YOLO26 / RDK YOLO export) ----------
  nb::class_<bcdl::YoloLtrbConfig>(m, "YoloLtrbConfig")
      .def(nb::init<>())
      .def_rw("num_classes", &bcdl::YoloLtrbConfig::num_classes)
      .def_rw("conf_thresh", &bcdl::YoloLtrbConfig::conf_thresh)
      .def_rw("iou_thresh", &bcdl::YoloLtrbConfig::iou_thresh)
      .def_rw("max_dets", &bcdl::YoloLtrbConfig::max_dets)
      .def_rw("strides", &bcdl::YoloLtrbConfig::strides)
      .def_rw("reg_max", &bcdl::YoloLtrbConfig::reg_max);

  // YoloLtrbDetector: reads 2*len(strides) outputs as (cls,box) pairs from the
  // Engine, dequantizes + de-strides each, and runs the LTRB decode + NMS.
  nb::class_<bcdl::YoloLtrbDetector>(m, "YoloLtrbDetector")
      .def(nb::init<bcdl::Engine&, bcdl::YoloLtrbConfig, int>(), "engine"_a,
           "config"_a, "output_base"_a = 0, nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::YoloLtrbDetector::postprocess, "letterbox"_a)
      .def_prop_ro("config", &bcdl::YoloLtrbDetector::config);

  // ===========================================================================
  // Media: the unified VpImage + JPU/VPU codecs
  // ===========================================================================

  // hbVPImageFormat — only the layouts BCDL's VpImage / codecs handle today.
  nb::enum_<hbVPImageFormat>(m, "ImageFormat")
      .value("Y", HB_VP_IMAGE_FORMAT_Y)
      .value("NV12", HB_VP_IMAGE_FORMAT_NV12)
      .value("RGB", HB_VP_IMAGE_FORMAT_RGB)
      .value("BGR", HB_VP_IMAGE_FORMAT_BGR);

  // VpImage: the unified shared-memory image. Move-only — nanobind moves it into
  // a heap instance on return, so factories/decoders can hand one back by value.
  nb::class_<bcdl::VpImage>(m, "VpImage")
      .def(nb::init<int, int, hbVPImageFormat>(), "width"_a, "height"_a,
           "format"_a,
           "Allocate a (width, height) image in `format` (cached device buffer).")
      .def_prop_ro("width", &bcdl::VpImage::width)
      .def_prop_ro("height", &bcdl::VpImage::height)
      .def_prop_ro("format", &bcdl::VpImage::format)
      .def_prop_ro("valid", &bcdl::VpImage::valid)
      .def(
          "to_numpy",
          [](const bcdl::VpImage& im) {
            const hbVPImage& r = im.raw();
            const int w = im.width(), h = im.height();
            const auto* base = static_cast<const uint8_t*>(im.data());
            switch (im.format()) {
              case HB_VP_IMAGE_FORMAT_BGR:
              case HB_VP_IMAGE_FORMAT_RGB: {
                const size_t rowbytes = static_cast<size_t>(w) * 3;
                std::vector<uint8_t> out(static_cast<size_t>(h) * rowbytes);
                for (int y = 0; y < h; ++y)
                  std::memcpy(out.data() + static_cast<size_t>(y) * rowbytes,
                              base + static_cast<size_t>(y) * r.stride, rowbytes);
                return toNumpy(std::move(out), {static_cast<size_t>(h),
                                                static_cast<size_t>(w), 3});
              }
              case HB_VP_IMAGE_FORMAT_Y: {
                std::vector<uint8_t> out(static_cast<size_t>(h) * w);
                for (int y = 0; y < h; ++y)
                  std::memcpy(out.data() + static_cast<size_t>(y) * w,
                              base + static_cast<size_t>(y) * r.stride, w);
                return toNumpy(std::move(out),
                               {static_cast<size_t>(h), static_cast<size_t>(w)});
              }
              case HB_VP_IMAGE_FORMAT_NV12: {
                const size_t ysize = static_cast<size_t>(w) * h;
                std::vector<uint8_t> out(ysize + ysize / 2);
                for (int y = 0; y < h; ++y)
                  std::memcpy(out.data() + static_cast<size_t>(y) * w,
                              base + static_cast<size_t>(y) * r.stride, w);
                const auto* uvbase = static_cast<const uint8_t*>(r.uvVirAddr);
                for (int y = 0; y < h / 2; ++y)
                  std::memcpy(out.data() + ysize + static_cast<size_t>(y) * w,
                              uvbase + static_cast<size_t>(y) * r.uvStride, w);
                return toNumpy(std::move(out), {ysize + ysize / 2});
              }
              default:
                throw std::runtime_error("VpImage.to_numpy: unsupported format");
            }
          },
          "Copy the image out to a uint8 numpy array honoring the device row "
          "stride: BGR/RGB -> (H,W,3), Y -> (H,W), NV12 -> flat (W*H*3/2) "
          "[Y plane then interleaved UV].");

  // Factory: pack a contiguous HxWx3 uint8 BGR array into a BGR VpImage,
  // honoring the VpImage row stride; flushes the buffer so a codec can read it.
  m.def(
      "vp_image_from_bgr",
      [](nb::ndarray<uint8_t, nb::c_contig> arr) {
        if (arr.ndim() != 3 || arr.shape(2) != 3)
          throw std::runtime_error(
              "vp_image_from_bgr expects an (H, W, 3) uint8 array");
        const int h = static_cast<int>(arr.shape(0));
        const int w = static_cast<int>(arr.shape(1));
        bcdl::VpImage im(w, h, HB_VP_IMAGE_FORMAT_BGR);
        const auto* src = static_cast<const uint8_t*>(arr.data());
        auto* dst = static_cast<uint8_t*>(im.data());
        const int stride = im.raw().stride;
        const size_t rowbytes = static_cast<size_t>(w) * 3;
        for (int y = 0; y < h; ++y)
          std::memcpy(dst + static_cast<size_t>(y) * stride,
                      src + static_cast<size_t>(y) * rowbytes, rowbytes);
        im.cleanCache();
        return im;
      },
      "bgr"_a,
      "Build a BGR VpImage from an (H, W, 3) uint8 numpy array (copied, "
      "stride-honored, cache-flushed).");

  // Factory: pack a contiguous NV12 buffer (W*H Y bytes + W*H/2 interleaved UV
  // bytes, i.e. shape (H*3/2, W) or flat) into an NV12 VpImage.
  m.def(
      "vp_image_from_nv12",
      [](nb::ndarray<uint8_t, nb::c_contig> arr, int w, int h) {
        if (w <= 0 || h <= 0 || (w & 1) || (h & 1))
          throw std::runtime_error(
              "vp_image_from_nv12 requires positive even width/height");
        const size_t need = static_cast<size_t>(w) * h * 3 / 2;
        if (arr.nbytes() < need)
          throw std::runtime_error(
              "vp_image_from_nv12: array smaller than W*H*3/2");
        bcdl::VpImage im(w, h, HB_VP_IMAGE_FORMAT_NV12);
        const auto* src = static_cast<const uint8_t*>(arr.data());
        const hbVPImage& r = im.raw();
        auto* ybase = static_cast<uint8_t*>(r.dataVirAddr);
        auto* uvbase = static_cast<uint8_t*>(r.uvVirAddr);
        for (int y = 0; y < h; ++y)
          std::memcpy(ybase + static_cast<size_t>(y) * r.stride,
                      src + static_cast<size_t>(y) * w, w);
        const uint8_t* uvsrc = src + static_cast<size_t>(w) * h;
        for (int y = 0; y < h / 2; ++y)
          std::memcpy(uvbase + static_cast<size_t>(y) * r.uvStride,
                      uvsrc + static_cast<size_t>(y) * w, w);
        im.cleanCache();
        return im;
      },
      "nv12"_a, "width"_a, "height"_a,
      "Build an NV12 VpImage from a packed (W*H*3/2) uint8 buffer "
      "(stride-honored, cache-flushed).");

  // --- JPEG (JPU) ------------------------------------------------------------
  nb::class_<bcdl::JpegEncoder>(m, "JpegEncoder")
      .def(nb::init<int, int, int, hbVPImageFormat>(), "width"_a, "height"_a,
           "quality"_a = 50, "format"_a = HB_VP_IMAGE_FORMAT_NV12,
           "Fixed-size JPU encoder. width must align to 16, height to 8.")
      .def(
          "encode",
          [](bcdl::JpegEncoder& e, const bcdl::VpImage& src) {
            std::vector<uint8_t> b = e.encode(src);
            return nb::bytes(reinterpret_cast<const char*>(b.data()), b.size());
          },
          "image"_a, "Encode one VpImage to JPEG bytes.")
      .def_prop_ro("width", &bcdl::JpegEncoder::width)
      .def_prop_ro("height", &bcdl::JpegEncoder::height)
      .def_prop_ro("format", &bcdl::JpegEncoder::format);

  nb::class_<bcdl::JpegDecoder>(m, "JpegDecoder")
      .def(nb::init<hbVPImageFormat>(), "out_format"_a = HB_VP_IMAGE_FORMAT_NV12)
      .def(
          "decode",
          [](bcdl::JpegDecoder& d, nb::bytes data) {
            return d.decode(reinterpret_cast<const uint8_t*>(data.c_str()),
                            data.size());
          },
          "data"_a, "Decode JPEG bytes into an owned NV12 VpImage.")
      .def_prop_ro("out_format", &bcdl::JpegDecoder::outFormat);

#ifdef BCDL_HAVE_GDC
  // --- GDC hardware letterbox (VPS /dev/gdc, bypasses the offline vDSP) -------
  nb::class_<bcdl::GdcLetterbox>(m, "GdcLetterbox")
      .def(nb::init<const std::string&, int, int, int, int, uint8_t>(), "bin_path"_a,
           "in_w"_a, "in_h"_a, "out_w"_a, "out_h"_a, "pad"_a = 114,
           "Persistent GDC vnode + pre-generated warp-LUT bin for a fixed "
           "(in_w,in_h)->(out_w,out_h) letterbox. Generate the bin offline (SDK "
           "generate_bin) for the SAME geometry.")
      .def(
          "run",
          [](bcdl::GdcLetterbox& g, const bcdl::VpImage& src) {
            bcdl::VpImage dst(g.outputWidth(), g.outputHeight(), HB_VP_IMAGE_FORMAT_NV12);
            g.run(src, dst);
            return dst;
          },
          "src"_a, "GDC-letterbox an NV12 VpImage into a new out_w x out_h NV12 VpImage.")
      .def_prop_ro("info", &bcdl::GdcLetterbox::info)
      .def_prop_ro("output_width", &bcdl::GdcLetterbox::outputWidth)
      .def_prop_ro("output_height", &bcdl::GdcLetterbox::outputHeight);
#endif

  // --- H.264 / H.265 (VPU) ---------------------------------------------------
  nb::enum_<hbVPVideoType>(m, "VideoType")
      .value("H264", HB_VP_VIDEO_TYPE_H264)
      .value("H265", HB_VP_VIDEO_TYPE_H265);

  nb::class_<bcdl::VideoEncConfig>(m, "VideoEncConfig")
      .def(nb::init<>())
      .def_rw("type", &bcdl::VideoEncConfig::type)
      .def_rw("width", &bcdl::VideoEncConfig::width)
      .def_rw("height", &bcdl::VideoEncConfig::height)
      .def_rw("bitrate_kbps", &bcdl::VideoEncConfig::bitrate_kbps)
      .def_rw("framerate", &bcdl::VideoEncConfig::framerate)
      .def_rw("intra_period", &bcdl::VideoEncConfig::intra_period)
      .def_rw("format", &bcdl::VideoEncConfig::format);

  nb::class_<bcdl::VideoEncoder>(m, "VideoEncoder")
      .def(nb::init<const bcdl::VideoEncConfig&>(), "config"_a)
      .def(
          "encode",
          [](bcdl::VideoEncoder& e, const bcdl::VpImage& frame) {
            std::vector<uint8_t> b = e.encode(frame);
            return nb::bytes(reinterpret_cast<const char*>(b.data()), b.size());
          },
          "frame"_a,
          "Encode one NV12/YUV420 frame to compressed bytes (may be empty if "
          "the encoder buffered the frame).")
      .def_prop_ro("type", &bcdl::VideoEncoder::type)
      .def_prop_ro("width", &bcdl::VideoEncoder::width)
      .def_prop_ro("height", &bcdl::VideoEncoder::height)
      .def_prop_ro("format", &bcdl::VideoEncoder::format);

  nb::class_<bcdl::VideoDecConfig>(m, "VideoDecConfig")
      .def(nb::init<>())
      .def_rw("type", &bcdl::VideoDecConfig::type)
      .def_rw("format", &bcdl::VideoDecConfig::format)
      .def_rw("in_buf_size", &bcdl::VideoDecConfig::in_buf_size);

  nb::class_<bcdl::VideoDecoder>(m, "VideoDecoder")
      .def(nb::init<const bcdl::VideoDecConfig&>(), "config"_a)
      .def(
          "decode",
          [](bcdl::VideoDecoder& d,
             nb::bytes data) -> std::optional<bcdl::VpImage> {
            bcdl::VpImage out;
            if (d.decode(reinterpret_cast<const uint8_t*>(data.c_str()),
                         data.size(), out))
              return std::optional<bcdl::VpImage>(std::move(out));
            return std::nullopt;
          },
          "data"_a,
          "Feed one compressed chunk; returns an NV12 VpImage when a frame is "
          "ready, else None (the decoder is still buffering reference frames).")
      .def_prop_ro("type", &bcdl::VideoDecoder::type)
      .def_prop_ro("format", &bcdl::VideoDecoder::format);

  // ===========================================================================
  // Depth: single-channel dense head
  // ===========================================================================
  nb::class_<bcdl::DepthConfig>(m, "DepthConfig")
      .def(nb::init<>())
      .def_rw("width", &bcdl::DepthConfig::width)
      .def_rw("height", &bcdl::DepthConfig::height)
      .def_rw("normalize", &bcdl::DepthConfig::normalize)
      .def_rw("clip_lo", &bcdl::DepthConfig::clip_lo)
      .def_rw("clip_hi", &bcdl::DepthConfig::clip_hi);

  nb::class_<bcdl::DepthMap>(m, "DepthMap")
      .def(nb::init<>())
      .def_ro("width", &bcdl::DepthMap::width)
      .def_ro("height", &bcdl::DepthMap::height)
      .def_ro("vmin", &bcdl::DepthMap::vmin)
      .def_ro("vmax", &bcdl::DepthMap::vmax)
      .def_prop_ro(
          "data",
          [](const bcdl::DepthMap& mp) {
            return toNumpy(std::vector<float>(mp.data),
                           {static_cast<size_t>(mp.height),
                            static_cast<size_t>(mp.width)});
          },
          // The returned ndarray owns its data via a capsule, so override the
          // property default (reference_internal) which rejects an owned array.
          nb::rv_policy::reference,
          "Depth map as a float32 (H, W) numpy array (normalized to [0,1] when "
          "cfg.normalize, else raw).");

  // decode_depth(): numpy Python path — pass the (already-float, contiguous)
  // engine output; its shape drives H/W. See decodeDepth() in tasks/depth.h.
  m.def(
      "decode_depth",
      [](nb::ndarray<float, nb::c_contig> arr, const bcdl::DepthConfig& cfg) {
        std::vector<int> shape;
        shape.reserve(arr.ndim());
        for (size_t d = 0; d < arr.ndim(); ++d)
          shape.push_back(static_cast<int>(arr.shape(d)));
        return bcdl::decodeDepth(arr.data(), shape, cfg);
      },
      "output"_a, "config"_a,
      "Decode a single-channel float depth/disparity tensor into a DepthMap.");

  m.def(
      "depth_colorize",
      [](const bcdl::DepthMap& mp) {
        return toNumpy(bcdl::depthColorize(mp),
                       {static_cast<size_t>(mp.height),
                        static_cast<size_t>(mp.width), 3});
      },
      "depth_map"_a,
      "Turbo-colormap a DepthMap to a BGR (H, W, 3) uint8 numpy array.");

  m.def(
      "depth_to_gray8",
      [](const bcdl::DepthMap& mp) {
        return toNumpy(bcdl::depthToGray8(mp),
                       {static_cast<size_t>(mp.height),
                        static_cast<size_t>(mp.width)});
      },
      "depth_map"_a,
      "Min-max normalize a DepthMap to a grayscale (H, W) uint8 numpy array.");

  // DepthEstimator: read the Engine output directly (dequant + device strides)
  // and post-process. Run engine.infer() first. keep_alive ties the Engine ref.
  nb::class_<bcdl::DepthEstimator>(m, "DepthEstimator")
      .def(nb::init<bcdl::Engine&, bcdl::DepthConfig, int>(), "engine"_a,
           "config"_a = bcdl::DepthConfig{}, "output_index"_a = 0,
           nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::DepthEstimator::postprocess)
      .def_prop_ro("config", &bcdl::DepthEstimator::config);

  // ===========================================================================
  // Stereo: two-image disparity / depth pipeline (e.g. LAS2)
  // ===========================================================================
  nb::enum_<bcdl::StereoFit>(m, "StereoFit")
      .value("Resize", bcdl::StereoFit::kResize)
      .value("Crop", bcdl::StereoFit::kCrop);

  nb::class_<bcdl::StereoConfig>(m, "StereoConfig")
      .def(nb::init<>())
      .def_rw("input_w", &bcdl::StereoConfig::input_w)
      .def_rw("input_h", &bcdl::StereoConfig::input_h)
      .def_rw("fit", &bcdl::StereoConfig::fit)
      .def_rw("to_rgb", &bcdl::StereoConfig::to_rgb)
      .def_rw("left_index", &bcdl::StereoConfig::left_index)
      .def_rw("right_index", &bcdl::StereoConfig::right_index)
      .def_rw("output_index", &bcdl::StereoConfig::output_index)
      .def_rw("fx", &bcdl::StereoConfig::fx)
      .def_rw("baseline", &bcdl::StereoConfig::baseline)
      .def_rw("valid_mask", &bcdl::StereoConfig::valid_mask)
      .def_rw("disp_min", &bcdl::StereoConfig::disp_min)
      .def_rw("max_disp", &bcdl::StereoConfig::max_disp)
      .def_rw("left_margin", &bcdl::StereoConfig::left_margin)
      .def_rw("lr_check", &bcdl::StereoConfig::lr_check)
      .def_rw("lr_thresh", &bcdl::StereoConfig::lr_thresh);

  nb::class_<bcdl::StereoResult>(m, "StereoResult")
      .def_ro("disparity", &bcdl::StereoResult::disparity)
      .def_prop_ro(
          "depth",
          [](const bcdl::StereoResult& r) {
            const size_t h = r.disparity.height, w = r.disparity.width;
            if (r.depth.empty()) return toNumpy(std::vector<float>{}, {0});
            return toNumpy(std::vector<float>(r.depth), {h, w});
          },
          nb::rv_policy::reference,
          "Metric depth (m) as float32 (H, W); shape (0,) when fx/baseline unset.")
      .def_prop_ro(
          "valid",
          [](const bcdl::StereoResult& r) {
            const size_t h = r.disparity.height, w = r.disparity.width;
            if (r.valid.empty()) return toNumpy(std::vector<uint8_t>{}, {0});
            return toNumpy(std::vector<uint8_t>(r.valid), {h, w});
          },
          nb::rv_policy::reference,
          "Validity mask uint8 (H, W), 1=keep; shape (0,) when valid_mask off.");

  // pack_stereo_input(): numpy path — BGR HxWx3 uint8 -> planar (3, H, W) float32
  // (fit + channel swap), for testing the preproc without an Engine.
  m.def(
      "pack_stereo_input",
      [](nb::ndarray<const uint8_t, nb::c_contig> bgr, int out_h, int out_w,
         bcdl::StereoFit fit, bool to_rgb) {
        if (bgr.ndim() != 3 || bgr.shape(2) != 3)
          throw std::runtime_error("pack_stereo_input expects an HxWx3 uint8 array");
        std::vector<float> dst(static_cast<size_t>(3) * out_h * out_w);
        bcdl::packStereoInputCHW(bgr.data(), static_cast<int>(bgr.shape(1)),
                                 static_cast<int>(bgr.shape(0)), out_h, out_w, fit,
                                 to_rgb, dst.data());
        return toNumpy(std::move(dst),
                       {3, static_cast<size_t>(out_h), static_cast<size_t>(out_w)});
      },
      "bgr"_a, "out_h"_a, "out_w"_a, "fit"_a = bcdl::StereoFit::kResize,
      "to_rgb"_a = true,
      "Fit (resize/crop) + channel-swap a BGR frame to a planar (3,H,W) float32.");

  m.def(
      "disparity_to_depth",
      [](const bcdl::DepthMap& disp, float fx, float baseline) {
        return toNumpy(bcdl::disparityToDepth(disp, fx, baseline),
                       {static_cast<size_t>(disp.height),
                        static_cast<size_t>(disp.width)});
      },
      "disparity"_a, "fx"_a, "baseline"_a,
      "Convert a disparity DepthMap to metric depth (m): z = fx*baseline/disp.");

  m.def(
      "stereo_valid_mask",
      [](const bcdl::DepthMap& disp, float disp_min, float max_disp,
         int left_margin, std::optional<bcdl::DepthMap> disp_right,
         float lr_thresh) {
        const bcdl::DepthMap* dr = disp_right ? &*disp_right : nullptr;
        return toNumpy(bcdl::stereoValidMask(disp, disp_min, max_disp, left_margin,
                                             dr, lr_thresh),
                       {static_cast<size_t>(disp.height),
                        static_cast<size_t>(disp.width)});
      },
      "disparity"_a, "disp_min"_a = 0.0f, "max_disp"_a = 192.0f,
      "left_margin"_a = 0, "disp_right"_a = nb::none(), "lr_thresh"_a = 1.5f,
      "Geometry validity mask uint8 (H, W): disparity range + left-border + "
      "optional left-right consistency (pass disp_right).");

  // StereoPipeline: two HxWx3 uint8 BGR frames -> disparity (+ optional depth /
  // valid mask). keep_alive ties the Engine ref to the pipeline.
  nb::class_<bcdl::StereoPipeline>(m, "StereoPipeline")
      .def(nb::init<bcdl::Engine&, bcdl::StereoConfig>(), "engine"_a,
           "config"_a = bcdl::StereoConfig{}, nb::keep_alive<1, 2>())
      .def(
          "process",
          [](bcdl::StereoPipeline& p, nb::ndarray<const uint8_t, nb::c_contig> left,
             nb::ndarray<const uint8_t, nb::c_contig> right) {
            if (left.ndim() != 3 || left.shape(2) != 3 || right.ndim() != 3 ||
                right.shape(2) != 3)
              throw std::runtime_error(
                  "StereoPipeline.process expects two HxWx3 uint8 BGR arrays");
            if (left.shape(0) != right.shape(0) || left.shape(1) != right.shape(1))
              throw std::runtime_error("left/right must have identical dimensions");
            const int h = static_cast<int>(left.shape(0));
            const int w = static_cast<int>(left.shape(1));
            return p.process(left.data(), right.data(), w, h);
          },
          "left"_a, "right"_a,
          "Run one stereo pair (HxWx3 uint8 BGR); returns a StereoResult.")
      .def_prop_ro("config", &bcdl::StereoPipeline::config)
      .def_prop_ro("input_w", &bcdl::StereoPipeline::inputWidth)
      .def_prop_ro("input_h", &bcdl::StereoPipeline::inputHeight);

  // ===========================================================================
  // Segmentation: dense semantic head
  // ===========================================================================
  nb::class_<bcdl::SegConfig>(m, "SegConfig")
      .def(nb::init<>())
      .def_rw("num_classes", &bcdl::SegConfig::num_classes)
      .def_rw("channels_first", &bcdl::SegConfig::channels_first)
      .def_rw("argmaxed", &bcdl::SegConfig::argmaxed);

  nb::class_<bcdl::SegMask>(m, "SegMask")
      .def(nb::init<>())
      .def_ro("width", &bcdl::SegMask::width)
      .def_ro("height", &bcdl::SegMask::height)
      .def_ro("num_classes", &bcdl::SegMask::num_classes)
      .def_prop_ro(
          "labels",
          [](const bcdl::SegMask& mk) {
            return toNumpy(std::vector<int32_t>(mk.labels),
                           {static_cast<size_t>(mk.height),
                            static_cast<size_t>(mk.width)});
          },
          // Owned-via-capsule ndarray; override the property default policy.
          nb::rv_policy::reference,
          "Per-pixel class ids as an int32 (H, W) numpy array.");

  // decode_seg(): numpy Python path — argmax/pass-through a float tensor.
  m.def(
      "decode_seg",
      [](nb::ndarray<float, nb::c_contig> arr, const bcdl::SegConfig& cfg) {
        std::vector<int> shape;
        shape.reserve(arr.ndim());
        for (size_t d = 0; d < arr.ndim(); ++d)
          shape.push_back(static_cast<int>(arr.shape(d)));
        return bcdl::decodeSeg(arr.data(), shape, cfg);
      },
      "output"_a, "config"_a,
      "Argmax a logit tensor (or pass through an id tensor) into a SegMask.");

  m.def(
      "seg_colorize",
      [](const bcdl::SegMask& mk) {
        return toNumpy(bcdl::segColorize(mk),
                       {static_cast<size_t>(mk.height),
                        static_cast<size_t>(mk.width), 3});
      },
      "seg_mask"_a,
      "Color a SegMask with a fixed palette -> BGR (H, W, 3) uint8 numpy array.");

  nb::class_<bcdl::Segmenter>(m, "Segmenter")
      .def(nb::init<bcdl::Engine&, bcdl::SegConfig, int>(), "engine"_a,
           "config"_a = bcdl::SegConfig{}, "output_index"_a = 0,
           nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::Segmenter::postprocess)
      .def_prop_ro("config", &bcdl::Segmenter::config);

  // ===========================================================================
  // Classification (top-k)
  // ===========================================================================
  nb::class_<bcdl::ClsConfig>(m, "ClsConfig")
      .def(nb::init<>())
      .def_rw("top_k", &bcdl::ClsConfig::top_k)
      .def_rw("apply_softmax", &bcdl::ClsConfig::apply_softmax);

  nb::class_<bcdl::ClsResult>(m, "ClsResult")
      .def_ro("class_id", &bcdl::ClsResult::class_id)
      .def_ro("score", &bcdl::ClsResult::score)
      .def("__repr__", [](const bcdl::ClsResult& r) {
        char b[64];
        std::snprintf(b, sizeof(b), "ClsResult(cls=%d score=%.4f)", r.class_id, r.score);
        return std::string(b);
      });

  m.def("decode_classification",
        [](nb::ndarray<float, nb::c_contig> arr, const bcdl::ClsConfig& cfg) {
          return bcdl::decodeClassification(arr.data(),
                                            static_cast<int>(arr.size()), cfg);
        },
        "logits"_a, "config"_a, "Top-k decode of a flat logit array.");

  nb::class_<bcdl::Classifier>(m, "Classifier")
      .def(nb::init<bcdl::Engine&, bcdl::ClsConfig, int>(), "engine"_a,
           "config"_a = bcdl::ClsConfig{}, "output_index"_a = 0, nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::Classifier::postprocess)
      .def_prop_ro("config", &bcdl::Classifier::config);

  // ===========================================================================
  // Pose (keypoints)
  // ===========================================================================
  nb::class_<bcdl::Keypoint>(m, "Keypoint")
      .def_ro("x", &bcdl::Keypoint::x)
      .def_ro("y", &bcdl::Keypoint::y)
      .def_ro("score", &bcdl::Keypoint::score);

  nb::class_<bcdl::PoseDetection>(m, "PoseDetection")
      .def_ro("x1", &bcdl::PoseDetection::x1)
      .def_ro("y1", &bcdl::PoseDetection::y1)
      .def_ro("x2", &bcdl::PoseDetection::x2)
      .def_ro("y2", &bcdl::PoseDetection::y2)
      .def_ro("score", &bcdl::PoseDetection::score)
      .def_ro("class_id", &bcdl::PoseDetection::class_id)
      .def_ro("keypoints", &bcdl::PoseDetection::keypoints);

  nb::class_<bcdl::PoseConfig>(m, "PoseConfig")
      .def(nb::init<>())
      .def_rw("num_keypoints", &bcdl::PoseConfig::num_keypoints)
      .def_rw("conf_thresh", &bcdl::PoseConfig::conf_thresh)
      .def_rw("iou_thresh", &bcdl::PoseConfig::iou_thresh)
      .def_rw("max_dets", &bcdl::PoseConfig::max_dets)
      .def_rw("strides", &bcdl::PoseConfig::strides);

  nb::class_<bcdl::PoseEstimator>(m, "PoseEstimator")
      .def(nb::init<bcdl::Engine&, bcdl::PoseConfig, int>(), "engine"_a,
           "config"_a = bcdl::PoseConfig{}, "output_base"_a = 0, nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::PoseEstimator::postprocess, "letterbox"_a)
      .def_prop_ro("config", &bcdl::PoseEstimator::config);

  // decode_pose(): numpy Python path — pass per-scale float cls/box/kpt tensors
  // ([H,W,1] / [H,W,4] / [H,W,K*3]) as lists of c-contiguous ndarrays. Returns
  // un-letterboxed PoseDetections. Mirrors PoseEstimator without an Engine.
  m.def(
      "decode_pose",
      [](std::vector<nb::ndarray<const float, nb::c_contig>> cls,
         std::vector<nb::ndarray<const float, nb::c_contig>> box,
         std::vector<nb::ndarray<const float, nb::c_contig>> kpt,
         const bcdl::PoseConfig& cfg, const bcdl::LetterboxInfo& lb) {
        const size_t n = cls.size();
        if (box.size() != n || kpt.size() != n) {
          throw std::runtime_error(
              "decode_pose: cls/box/kpt must be lists of equal length (one per scale)");
        }
        std::vector<const float*> cp(n), bp(n), kp(n);
        std::vector<std::pair<int, int>> grid(n);
        for (size_t i = 0; i < n; ++i) {
          if (cls[i].ndim() != 3 || box[i].ndim() != 3 || kpt[i].ndim() != 3) {
            throw std::runtime_error("decode_pose: each scale tensor must be [H,W,C]");
          }
          // Bounds: the decoder strides box at 4 and kpt at num_keypoints*3 over
          // the cls grid — reject mismatched grids / too-few channels so it can't
          // read past the sibling buffers.
          if (box[i].shape(0) != cls[i].shape(0) || box[i].shape(1) != cls[i].shape(1) ||
              kpt[i].shape(0) != cls[i].shape(0) || kpt[i].shape(1) != cls[i].shape(1)) {
            throw std::runtime_error("decode_pose: box/kpt grid must match cls (H,W) per scale");
          }
          if (box[i].shape(2) < 4 ||
              static_cast<int>(kpt[i].shape(2)) < cfg.num_keypoints * 3) {
            throw std::runtime_error(
                "decode_pose: box needs >=4 channels and kpt needs num_keypoints*3");
          }
          cp[i] = cls[i].data();
          bp[i] = box[i].data();
          kp[i] = kpt[i].data();
          grid[i] = {static_cast<int>(cls[i].shape(0)),
                     static_cast<int>(cls[i].shape(1))};
        }
        return bcdl::decodePose(cp, bp, kp, grid, cfg, lb);
      },
      "cls"_a, "box"_a, "kpt"_a, "config"_a, "letterbox"_a,
      "Decode per-scale [H,W,1]/[H,W,4]/[H,W,K*3] float tensors -> PoseDetections.");

  // ===========================================================================
  // Instance segmentation
  // ===========================================================================
  nb::class_<bcdl::InstanceSegConfig>(m, "InstanceSegConfig")
      .def(nb::init<>())
      .def_rw("conf_thresh", &bcdl::InstanceSegConfig::conf_thresh)
      .def_rw("iou_thresh", &bcdl::InstanceSegConfig::iou_thresh)
      .def_rw("max_dets", &bcdl::InstanceSegConfig::max_dets)
      .def_rw("strides", &bcdl::InstanceSegConfig::strides)
      .def_rw("proto_index", &bcdl::InstanceSegConfig::proto_index)
      .def_rw("compute_masks", &bcdl::InstanceSegConfig::compute_masks);

  nb::class_<bcdl::InstanceMask>(m, "InstanceMask")
      .def_ro("x1", &bcdl::InstanceMask::x1)
      .def_ro("y1", &bcdl::InstanceMask::y1)
      .def_ro("x2", &bcdl::InstanceMask::x2)
      .def_ro("y2", &bcdl::InstanceMask::y2)
      .def_ro("score", &bcdl::InstanceMask::score)
      .def_ro("class_id", &bcdl::InstanceMask::class_id)
      .def_ro("mask_w", &bcdl::InstanceMask::mask_w)
      .def_ro("mask_h", &bcdl::InstanceMask::mask_h)
      .def_prop_ro(
          "mask",
          [](const bcdl::InstanceMask& im) {
            if (im.mask.empty())
              return toNumpy(std::vector<uint8_t>{}, {0, 0});
            return toNumpy(std::vector<uint8_t>(im.mask),
                           {static_cast<size_t>(im.mask_h),
                            static_cast<size_t>(im.mask_w)});
          },
          nb::rv_policy::reference,
          "Binary instance mask as a uint8 (H, W) numpy array (0/1), empty when "
          "compute_masks was false.");

  nb::class_<bcdl::InstanceSegmenter>(m, "InstanceSegmenter")
      .def(nb::init<bcdl::Engine&, bcdl::InstanceSegConfig, int>(), "engine"_a,
           "config"_a = bcdl::InstanceSegConfig{}, "output_base"_a = 0,
           nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::InstanceSegmenter::postprocess, "letterbox"_a,
           "orig_w"_a, "orig_h"_a)
      .def_prop_ro("config", &bcdl::InstanceSegmenter::config);

  // decode_instance_seg(): numpy Python path — per-scale float cls/box/mc tensors
  // ([H,W,nc]/[H,W,4]/[H,W,np]) plus the prototype ([mH,mW,np] or [1,mH,mW,np]).
  // Returns InstanceMasks (boxes un-letterboxed; full-frame binary masks when
  // cfg.compute_masks). Mirrors InstanceSegmenter without an Engine.
  m.def(
      "decode_instance_seg",
      [](std::vector<nb::ndarray<const float, nb::c_contig>> cls,
         std::vector<nb::ndarray<const float, nb::c_contig>> box,
         std::vector<nb::ndarray<const float, nb::c_contig>> mc,
         nb::ndarray<const float, nb::c_contig> proto,
         const bcdl::InstanceSegConfig& cfg, const bcdl::LetterboxInfo& lb,
         int orig_w, int orig_h) {
        const size_t n = cls.size();
        if (box.size() != n || mc.size() != n) {
          throw std::runtime_error(
              "decode_instance_seg: cls/box/mc must be lists of equal length");
        }
        if (n == 0) throw std::runtime_error("decode_instance_seg: empty scale list");
        std::vector<const float*> cp(n), bp(n), mp(n);
        std::vector<std::pair<int, int>> grid(n);
        for (size_t i = 0; i < n; ++i) {
          if (cls[i].ndim() != 3 || box[i].ndim() != 3 || mc[i].ndim() != 3) {
            throw std::runtime_error("decode_instance_seg: each scale tensor must be [H,W,C]");
          }
          // Bounds: nc/np below are taken from cls[0]/mc[0], box strides at 4 over
          // the cls grid — reject mismatched grids / channel counts so per-scale
          // indexing can't run off the sibling buffers.
          if (box[i].shape(0) != cls[i].shape(0) || box[i].shape(1) != cls[i].shape(1) ||
              mc[i].shape(0) != cls[i].shape(0) || mc[i].shape(1) != cls[i].shape(1)) {
            throw std::runtime_error("decode_instance_seg: box/mc grid must match cls (H,W) per scale");
          }
          if (box[i].shape(2) < 4 || cls[i].shape(2) != cls[0].shape(2) ||
              mc[i].shape(2) != mc[0].shape(2)) {
            throw std::runtime_error(
                "decode_instance_seg: box needs >=4 channels; cls/mc channels must match across scales");
          }
          cp[i] = cls[i].data();
          bp[i] = box[i].data();
          mp[i] = mc[i].data();
          grid[i] = {static_cast<int>(cls[i].shape(0)),
                     static_cast<int>(cls[i].shape(1))};
        }
        const int nc = static_cast<int>(cls[0].shape(2));
        const int np = static_cast<int>(mc[0].shape(2));
        int mH = 0, mW = 0, pC = 0;
        if (proto.ndim() == 3) {
          mH = static_cast<int>(proto.shape(0));
          mW = static_cast<int>(proto.shape(1));
          pC = static_cast<int>(proto.shape(2));
        } else if (proto.ndim() == 4) {
          mH = static_cast<int>(proto.shape(1));
          mW = static_cast<int>(proto.shape(2));
          pC = static_cast<int>(proto.shape(3));
        } else {
          throw std::runtime_error(
              "decode_instance_seg: proto must be [mH,mW,np] or [1,mH,mW,np]");
        }
        return bcdl::decodeInstanceSeg(cp, bp, mp, grid, nc, np, proto.data(), mH, mW,
                                       pC, cfg, lb, orig_w, orig_h);
      },
      "cls"_a, "box"_a, "mc"_a, "proto"_a, "config"_a, "letterbox"_a, "orig_w"_a,
      "orig_h"_a,
      "Decode per-scale [H,W,nc]/[H,W,4]/[H,W,np] + proto [mH,mW,np] -> InstanceMasks.");

  // ===========================================================================
  // Oriented bounding box (OBB)
  // ===========================================================================
  nb::class_<bcdl::RotatedBox>(m, "RotatedBox")
      .def_ro("cx", &bcdl::RotatedBox::cx)
      .def_ro("cy", &bcdl::RotatedBox::cy)
      .def_ro("w", &bcdl::RotatedBox::w)
      .def_ro("h", &bcdl::RotatedBox::h)
      .def_ro("angle", &bcdl::RotatedBox::angle);

  nb::class_<bcdl::ObbDetection>(m, "ObbDetection")
      .def_ro("rrect", &bcdl::ObbDetection::rrect)
      .def_ro("score", &bcdl::ObbDetection::score)
      .def_ro("class_id", &bcdl::ObbDetection::class_id);

  nb::class_<bcdl::ObbConfig>(m, "ObbConfig")
      .def(nb::init<>())
      .def_rw("num_classes", &bcdl::ObbConfig::num_classes)
      .def_rw("conf_thresh", &bcdl::ObbConfig::conf_thresh)
      .def_rw("iou_thresh", &bcdl::ObbConfig::iou_thresh)
      .def_rw("max_dets", &bcdl::ObbConfig::max_dets)
      .def_rw("strides", &bcdl::ObbConfig::strides)
      .def_rw("regularize", &bcdl::ObbConfig::regularize)
      .def_rw("angle_offset_rad", &bcdl::ObbConfig::angle_offset_rad)
      .def_rw("angle_sign", &bcdl::ObbConfig::angle_sign);

  nb::class_<bcdl::ObbDetector>(m, "ObbDetector")
      .def(nb::init<bcdl::Engine&, bcdl::ObbConfig, int>(), "engine"_a,
           "config"_a = bcdl::ObbConfig{}, "output_base"_a = 0, nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::ObbDetector::postprocess, "letterbox"_a)
      .def_prop_ro("config", &bcdl::ObbDetector::config);

  // decode_obb(): numpy Python path — per-scale float cls/box/angle tensors
  // ([H,W,nc] / [H,W,4] / [H,W,1]) as lists of c-contiguous ndarrays. Returns
  // un-letterboxed ObbDetections (rotated NMS already applied).
  m.def(
      "decode_obb",
      [](std::vector<nb::ndarray<const float, nb::c_contig>> cls,
         std::vector<nb::ndarray<const float, nb::c_contig>> box,
         std::vector<nb::ndarray<const float, nb::c_contig>> angle,
         const bcdl::ObbConfig& cfg, const bcdl::LetterboxInfo& lb) {
        const size_t n = cls.size();
        if (box.size() != n || angle.size() != n) {
          throw std::runtime_error(
              "decode_obb: cls/box/angle must be lists of equal length (one per scale)");
        }
        std::vector<const float*> cp(n), bp(n), ap(n);
        std::vector<std::pair<int, int>> grid(n);
        for (size_t i = 0; i < n; ++i) {
          if (cls[i].ndim() != 3 || box[i].ndim() != 3 || angle[i].ndim() != 3) {
            throw std::runtime_error("decode_obb: each scale tensor must be [H,W,C]");
          }
          // Bounds: decodeObb strides cls at cfg.num_classes, box at 4, angle at
          // 1 over the cls grid. Reject mismatched grids / too-few channels (cls
          // is indexed by the config num_classes, not the tensor — guard the OOB).
          if (box[i].shape(0) != cls[i].shape(0) || box[i].shape(1) != cls[i].shape(1) ||
              angle[i].shape(0) != cls[i].shape(0) || angle[i].shape(1) != cls[i].shape(1)) {
            throw std::runtime_error("decode_obb: box/angle grid must match cls (H,W) per scale");
          }
          if (static_cast<int>(cls[i].shape(2)) < cfg.num_classes ||
              box[i].shape(2) < 4 || angle[i].shape(2) < 1) {
            throw std::runtime_error(
                "decode_obb: cls needs >=num_classes channels, box >=4, angle >=1");
          }
          cp[i] = cls[i].data();
          bp[i] = box[i].data();
          ap[i] = angle[i].data();
          grid[i] = {static_cast<int>(cls[i].shape(0)),
                     static_cast<int>(cls[i].shape(1))};
        }
        return bcdl::decodeObb(cp, bp, ap, grid, cfg, lb);
      },
      "cls"_a, "box"_a, "angle"_a, "config"_a, "letterbox"_a,
      "Decode per-scale [H,W,nc]/[H,W,4]/[H,W,1] float tensors -> ObbDetections.");

  m.def(
      "rotated_iou",
      [](float acx, float acy, float aw, float ah, float aa, float bcx, float bcy,
         float bw, float bh, float ba) {
        return bcdl::rotatedIoU(bcdl::RotatedBox{acx, acy, aw, ah, aa},
                                bcdl::RotatedBox{bcx, bcy, bw, bh, ba});
      },
      "a_cx"_a, "a_cy"_a, "a_w"_a, "a_h"_a, "a_angle"_a, "b_cx"_a, "b_cy"_a,
      "b_w"_a, "b_h"_a, "b_angle"_a,
      "Rotated-rect IoU of two boxes (cx,cy,w,h,angle[rad]).");

  // ===========================================================================
  // Monocular 3D detection: SMOKE head
  // ===========================================================================
  nb::class_<bcdl::CameraIntrinsics>(m, "CameraIntrinsics")
      .def(nb::init<>())
      .def("__init__",
           [](bcdl::CameraIntrinsics* self, float fx, float fy, float cx, float cy) {
             new (self) bcdl::CameraIntrinsics{fx, fy, cx, cy};
           },
           "fx"_a, "fy"_a, "cx"_a, "cy"_a)
      .def_rw("fx", &bcdl::CameraIntrinsics::fx)
      .def_rw("fy", &bcdl::CameraIntrinsics::fy)
      .def_rw("cx", &bcdl::CameraIntrinsics::cx)
      .def_rw("cy", &bcdl::CameraIntrinsics::cy);

  nb::class_<bcdl::Mono3dBox>(m, "Mono3dBox")
      .def_ro("class_id", &bcdl::Mono3dBox::class_id)
      .def_ro("score", &bcdl::Mono3dBox::score)
      .def_ro("x", &bcdl::Mono3dBox::x)
      .def_ro("y", &bcdl::Mono3dBox::y)
      .def_ro("z", &bcdl::Mono3dBox::z)
      .def_ro("h", &bcdl::Mono3dBox::h)
      .def_ro("w", &bcdl::Mono3dBox::w)
      .def_ro("l", &bcdl::Mono3dBox::l)
      .def_ro("yaw", &bcdl::Mono3dBox::yaw)
      .def_ro("alpha", &bcdl::Mono3dBox::alpha)
      .def_ro("box2d", &bcdl::Mono3dBox::box2d)
      .def("__repr__", [](const bcdl::Mono3dBox& b) {
        char s[160];
        std::snprintf(s, sizeof(s),
                      "Mono3dBox(cls=%d score=%.2f xyz=[%.2f,%.2f,%.2f] hwl=[%.2f,%.2f,%.2f] "
                      "yaw=%.2f)",
                      b.class_id, b.score, b.x, b.y, b.z, b.h, b.w, b.l, b.yaw);
        return std::string(s);
      });

  nb::class_<bcdl::Mono3dConfig>(m, "Mono3dConfig")
      .def(nb::init<>())
      .def_rw("num_classes", &bcdl::Mono3dConfig::num_classes)
      .def_rw("conf_thresh", &bcdl::Mono3dConfig::conf_thresh)
      .def_rw("max_dets", &bcdl::Mono3dConfig::max_dets)
      .def_rw("nms_kernel", &bcdl::Mono3dConfig::nms_kernel)
      .def_rw("pred_2d", &bcdl::Mono3dConfig::pred_2d)
      .def_rw("depth_ref", &bcdl::Mono3dConfig::depth_ref)
      .def_rw("dim_ref", &bcdl::Mono3dConfig::dim_ref);

  m.def("compute_mono3d_feature_xform", &bcdl::computeMono3dFeatureXform, "orig_w"_a,
        "orig_h"_a, "feat_w"_a, "feat_h"_a,
        "Build the original<->feature affine (scale-to-width, center-height) for SMOKE decode.");

  // decode_mono3d(): numpy Python path — channel-first cls [nc,H,W] and reg
  // [8,H,W] c-contiguous float arrays (raw logits). Returns Mono3dBoxes.
  m.def(
      "decode_mono3d",
      [](nb::ndarray<const float, nb::c_contig> cls,
         nb::ndarray<const float, nb::c_contig> reg, bcdl::Mono3dConfig cfg,
         const bcdl::LetterboxInfo& feat_xform, const bcdl::CameraIntrinsics& K) {
        // Accept [nc,H,W] / [1,nc,H,W] (cls) and [8,H,W] / [1,8,H,W] (reg).
        auto squeeze = [](const nb::ndarray<const float, nb::c_contig>& a, int& C, int& H,
                          int& W) -> void {
          if (a.ndim() == 4 && a.shape(0) == 1) {
            C = static_cast<int>(a.shape(1));
            H = static_cast<int>(a.shape(2));
            W = static_cast<int>(a.shape(3));
          } else if (a.ndim() == 3) {
            C = static_cast<int>(a.shape(0));
            H = static_cast<int>(a.shape(1));
            W = static_cast<int>(a.shape(2));
          } else {
            throw std::runtime_error("decode_mono3d: expected [C,H,W] or [1,C,H,W]");
          }
        };
        int cC = 0, cH = 0, cW = 0, rC = 0, rH = 0, rW = 0;
        squeeze(cls, cC, cH, cW);
        squeeze(reg, rC, rH, rW);
        if (rH != cH || rW != cW) {
          throw std::runtime_error("decode_mono3d: cls and reg grids must match");
        }
        if (rC < 8) throw std::runtime_error("decode_mono3d: reg needs >=8 channels");
        cfg.num_classes = cC;  // authoritative from the tensor
        return bcdl::decodeMono3d(cls.data(), reg.data(), cH, cW, cfg, feat_xform, K);
      },
      "cls"_a, "reg"_a, "config"_a, "feat_xform"_a, "K"_a,
      "Decode channel-first cls[nc,H,W]/reg[8,H,W] SMOKE logits -> Mono3dBoxes.");

  nb::class_<bcdl::Mono3dDetector>(m, "Mono3dDetector")
      .def(nb::init<bcdl::Engine&, bcdl::Mono3dConfig, int>(), "engine"_a,
           "config"_a = bcdl::Mono3dConfig{}, "output_base"_a = 0, nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::Mono3dDetector::postprocess, "orig_w"_a, "orig_h"_a, "K"_a)
      .def_prop_ro("config", &bcdl::Mono3dDetector::config);

  // ===========================================================================
  // Multi-object tracking: ByteTrack
  // ===========================================================================
  nb::class_<bcdl::Track>(m, "Track")
      .def_ro("track_id", &bcdl::Track::track_id)
      .def_ro("x1", &bcdl::Track::x1)
      .def_ro("y1", &bcdl::Track::y1)
      .def_ro("x2", &bcdl::Track::x2)
      .def_ro("y2", &bcdl::Track::y2)
      .def_ro("score", &bcdl::Track::score)
      .def_ro("class_id", &bcdl::Track::class_id)
      .def("__repr__", [](const bcdl::Track& t) {
        char b[128];
        std::snprintf(b, sizeof(b), "Track(id=%d cls=%d score=%.2f box=[%.0f,%.0f,%.0f,%.0f])",
                      t.track_id, t.class_id, t.score, t.x1, t.y1, t.x2, t.y2);
        return std::string(b);
      });

  nb::class_<bcdl::ByteTrackConfig>(m, "ByteTrackConfig")
      .def(nb::init<>())
      .def_rw("track_thresh", &bcdl::ByteTrackConfig::track_thresh)
      .def_rw("high_thresh", &bcdl::ByteTrackConfig::high_thresh)
      .def_rw("match_thresh", &bcdl::ByteTrackConfig::match_thresh)
      .def_rw("track_buffer", &bcdl::ByteTrackConfig::track_buffer)
      .def_rw("frame_rate", &bcdl::ByteTrackConfig::frame_rate);

  nb::class_<bcdl::ByteTracker>(m, "ByteTracker")
      .def(nb::init<bcdl::ByteTrackConfig>(), "config"_a = bcdl::ByteTrackConfig{})
      .def("update", &bcdl::ByteTracker::update, "detections"_a,
           "Advance one frame with this frame's detections; returns active Tracks.")
      .def("reset", &bcdl::ByteTracker::reset)
      .def_prop_ro("config", &bcdl::ByteTracker::config);

  // --- detect + track in one call: TrackingPipeline ---------------------------
  // Wraps a DetectionPipeline (NV12 preproc + infer + decode/NMS, buffer-reuse)
  // feeding a ByteTracker. process() takes an HxWx3 uint8 BGR frame and returns
  // the active Tracks (stable id). Requires an NV12-input YOLO model.
  nb::enum_<bcdl::DetectHead>(m, "DetectHead")
      .value("Auto", bcdl::DetectHead::kAuto)
      .value("SingleTensor", bcdl::DetectHead::kSingleTensor)
      .value("YoloLtrb", bcdl::DetectHead::kYoloLtrb);

  nb::class_<bcdl::PipelineConfig>(m, "PipelineConfig")
      .def(nb::init<>())
      .def_rw("input_w", &bcdl::PipelineConfig::input_w)
      .def_rw("input_h", &bcdl::PipelineConfig::input_h)
      .def_rw("detect", &bcdl::PipelineConfig::detect)
      .def_rw("output_index", &bcdl::PipelineConfig::output_index)
      .def_rw("pad_value", &bcdl::PipelineConfig::pad_value)
      .def_rw("head", &bcdl::PipelineConfig::head)
      .def_rw("ltrb_strides", &bcdl::PipelineConfig::ltrb_strides);

  nb::class_<bcdl::TrackingPipeline>(m, "TrackingPipeline")
      .def(nb::init<bcdl::Engine&, bcdl::PipelineConfig, bcdl::ByteTrackConfig>(),
           "engine"_a, "det_config"_a = bcdl::PipelineConfig{},
           "track_config"_a = bcdl::ByteTrackConfig{}, nb::keep_alive<1, 2>())
      .def(
          "process",
          [](bcdl::TrackingPipeline& p, nb::ndarray<const uint8_t, nb::c_contig> bgr) {
            if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
              throw std::runtime_error(
                  "TrackingPipeline.process expects an HxWx3 uint8 BGR array");
            }
            const int h = static_cast<int>(bgr.shape(0));
            const int w = static_cast<int>(bgr.shape(1));
            return p.process(bgr.data(), w, h);
          },
          "bgr"_a, "Detect + track one HxWx3 uint8 BGR frame; returns list[Track].")
      .def_prop_ro("last_detections", &bcdl::TrackingPipeline::lastDetections)
      .def("reset", &bcdl::TrackingPipeline::reset);

  // --- streaming detect with preproc‖infer overlap: AsyncDetectionPipeline ----
  // Two C++ worker threads (preproc | infer+decode) overlap so steady-state
  // throughput approaches 1/max(preproc, infer+decode). submit()/next() block
  // (backpressure / wait-for-result) — both RELEASE the GIL so other Python
  // threads run while the C++ threads work. Results come back from next() in
  // submission order; next() returns None once finished AND fully drained.
  //
  //   p = bcdl.AsyncDetectionPipeline(engine, cfg, depth=3)
  //   for i, frame in enumerate(stream):
  //       p.submit(frame)                 # blocks if the pipeline is full
  //       if i >= depth:
  //           dets = p.next()             # keep ~depth in flight, drain in order
  //   p.finish()
  //   while (dets := p.next()) is not None:
  //       ...                             # handle the last in-flight results
  nb::class_<bcdl::AsyncDetectionPipeline>(m, "AsyncDetectionPipeline")
      .def(nb::init<bcdl::Engine&, bcdl::PipelineConfig, int>(), "engine"_a,
           "config"_a = bcdl::PipelineConfig{}, "depth"_a = 3,
           nb::keep_alive<1, 2>())
      .def(
          "submit",
          [](bcdl::AsyncDetectionPipeline& p,
             nb::ndarray<const uint8_t, nb::c_contig> bgr) {
            if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
              throw std::runtime_error(
                  "AsyncDetectionPipeline.submit expects an HxWx3 uint8 BGR array");
            }
            const int h = static_cast<int>(bgr.shape(0));
            const int w = static_cast<int>(bgr.shape(1));
            const uint8_t* data = bgr.data();
            // submit() copies the bytes, so the caller's array can be reused
            // right after; release the GIL while it blocks on backpressure.
            nb::gil_scoped_release rel;
            return p.submit(data, w, h);
          },
          "bgr"_a,
          "Enqueue an HxWx3 uint8 BGR frame (bytes copied). Blocks while the "
          "pipeline is full; returns False if finish()ed (frame not accepted).")
      .def(
          "next",
          [](bcdl::AsyncDetectionPipeline& p)
              -> std::optional<std::vector<bcdl::Detection>> {
            std::vector<bcdl::Detection> out;
            bool ok;
            {
              nb::gil_scoped_release rel;
              ok = p.next(out);
            }
            if (!ok) return std::nullopt;
            return out;
          },
          "Pop the next result in submission order (blocks until ready). Returns "
          "None once the pipeline is finished AND fully drained.")
      .def("finish", &bcdl::AsyncDetectionPipeline::finish,
           nb::call_guard<nb::gil_scoped_release>(),
           "Signal no more frames; next() drains the in-flight ones then "
           "returns None. Idempotent; also called on destruction.")
      .def_prop_ro("head", &bcdl::AsyncDetectionPipeline::head);

  // ===========================================================================
  // OCR: text detection (DBNet) + orientation cls + recognition (CTC)
  // ===========================================================================
  nb::class_<bcdl::RecResult>(m, "RecResult")
      .def_ro("text", &bcdl::RecResult::text)
      .def_ro("score", &bcdl::RecResult::score);

  nb::class_<bcdl::ClsDirResult>(m, "ClsDirResult")
      .def_ro("label", &bcdl::ClsDirResult::label)
      .def_ro("score", &bcdl::ClsDirResult::score)
      .def_ro("flip180", &bcdl::ClsDirResult::flip180);

  nb::class_<bcdl::TextBox>(m, "TextBox")
      .def_ro("x1", &bcdl::TextBox::x1)
      .def_ro("y1", &bcdl::TextBox::y1)
      .def_ro("x2", &bcdl::TextBox::x2)
      .def_ro("y2", &bcdl::TextBox::y2)
      .def_ro("score", &bcdl::TextBox::score)
      .def_prop_ro(
          "points",
          [](const bcdl::TextBox& b) {
            return toNumpy(std::vector<float>(b.pts, b.pts + 8), {4, 2});
          },
          nb::rv_policy::reference,
          "Rotated 4-point box (4,2) float, clockwise, original-image pixels.");

  nb::class_<bcdl::DbConfig>(m, "DbConfig")
      .def(nb::init<>())
      .def_rw("bin_thresh", &bcdl::DbConfig::bin_thresh)
      .def_rw("box_thresh", &bcdl::DbConfig::box_thresh)
      .def_rw("unclip_ratio", &bcdl::DbConfig::unclip_ratio)
      .def_rw("min_size", &bcdl::DbConfig::min_size)
      .def_rw("connectivity", &bcdl::DbConfig::connectivity);

  m.def("load_char_dict", &bcdl::loadCharDict, "path"_a,
        "Load a one-token-per-line OCR character dictionary.");

  m.def(
      "decode_ctc",
      [](nb::ndarray<float, nb::c_contig> arr, const std::vector<std::string>& dict) {
        const size_t C = arr.shape(arr.ndim() - 1);
        const size_t T = C > 0 ? arr.size() / C : 0;
        return bcdl::decodeCtc(arr.data(), static_cast<int>(T), static_cast<int>(C), dict);
      },
      "logits"_a, "dict"_a, "CTC greedy decode of a (T,C) logit array.");

  m.def(
      "decode_dbnet",
      [](nb::ndarray<float, nb::c_contig> arr, const bcdl::DbConfig& cfg,
         const bcdl::LetterboxInfo& lb) {
        if (arr.ndim() < 2)  // else ndim()-2 underflows size_t -> bogus shape index
          throw std::runtime_error("decode_dbnet: prob map must be at least 2-D (H,W)");
        const size_t W = arr.shape(arr.ndim() - 1);
        const size_t H = arr.shape(arr.ndim() - 2);
        return bcdl::decodeDbnet(arr.data(), static_cast<int>(H), static_cast<int>(W), cfg, lb);
      },
      "prob"_a, "config"_a, "letterbox"_a,
      "Connected-component + unclip decode of a DBNet (H,W) probability map.");

  m.def(
      "decode_cls_dir",
      [](nb::ndarray<float, nb::c_contig> arr, float thresh) {
        return bcdl::decodeClsDir(arr.data(), static_cast<int>(arr.size()), thresh);
      },
      "logits"_a, "thresh"_a = 0.9f,
      "Argmax a 0/180 text-direction logit vector -> ClsDirResult.");

  nb::class_<bcdl::TextRecognizer>(m, "TextRecognizer")
      .def(nb::init<bcdl::Engine&, std::string, int>(), "engine"_a, "dict_path"_a,
           "output_index"_a = 0, nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::TextRecognizer::postprocess);

  nb::class_<bcdl::TextAngleClassifier>(m, "TextAngleClassifier")
      .def(nb::init<bcdl::Engine&, float, int>(), "engine"_a, "thresh"_a = 0.9f,
           "output_index"_a = 0, nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::TextAngleClassifier::postprocess);

  nb::class_<bcdl::DbTextDetector>(m, "DbTextDetector")
      .def(nb::init<bcdl::Engine&, bcdl::DbConfig, int>(), "engine"_a,
           "config"_a = bcdl::DbConfig{}, "output_index"_a = 0, nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::DbTextDetector::postprocess, "letterbox"_a);
}
