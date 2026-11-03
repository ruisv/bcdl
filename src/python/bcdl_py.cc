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
#include <nanobind/stl/pair.h>
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
#include "bcdl/preproc/drive_inputs.h"
#include "bcdl/preproc/lidar_voxelize.h"
#include "bcdl/preproc/geometry.h"
#include "bcdl/preproc/vp_image.h"
#ifdef BCDL_HAVE_GDC
#include "bcdl/preproc/gdc_letterbox.h"
#include "bcdl/preproc/gdc_remap.h"
#endif
#include "bcdl/tasks/anomaly.h"
#include "bcdl/tasks/classification.h"
#include "bcdl/tasks/depth.h"
#include "bcdl/tasks/depth_refine.h"
#include "bcdl/tasks/e2e_drive.h"
#include "bcdl/tasks/embedding.h"
#include "bcdl/tasks/detection.h"
#include "bcdl/tasks/face.h"
#include "bcdl/tasks/wholebody.h"
#include "bcdl/tasks/features.h"
#include "bcdl/tasks/superres.h"
#include "bcdl/tasks/instance_seg.h"
#include "bcdl/tasks/lidar_det3d.h"
#include "bcdl/tasks/mono3d.h"
#include "bcdl/tasks/obb.h"
#include "bcdl/tasks/optical_flow.h"
#include "bcdl/tasks/asr.h"
#include "bcdl/tasks/command.h"
#include "bcdl/tasks/vocoder.h"
#include "bcdl/tasks/tts.h"
#include "bcdl/tasks/vad.h"
#include "bcdl/tasks/kws.h"
#include "bcdl/preproc/istft.h"
#include "bcdl/preproc/fbank.h"
#include "bcdl/tasks/panoptic_drive.h"
#include "bcdl/tasks/ocr.h"
#include "bcdl/tasks/open_vocab.h"
#include "bcdl/tasks/pose.h"
#include "bcdl/tasks/promptable_seg.h"
#include "bcdl/tasks/segmentation.h"
#include "bcdl/tracks/reid.h"
#include "bcdl/pipeline/async_detection_pipeline.h"
#include "bcdl/pipeline/async_video_detection_pipeline.h"
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
  m.doc() = "BCDL — BPU Common Deep Learning (C++ core)";
  m.attr("__version__") = BCDL_VERSION_STRING;

  nb::class_<bcdl::Engine>(m, "Engine")
      .def(nb::init<const std::string&, const std::string&>(), "hbm_path"_a,
           "model_name"_a = "")
      .def_prop_ro("model_name", &bcdl::Engine::modelName)
      .def_static("model_names", &bcdl::Engine::modelNames, "hbm_path"_a,
                  "List every model name packed into an .hbm (a package may "
                  "hold several — e.g. SigLIP's global-embedding and patch-"
                  "feature submodels). Pass one to Engine(path, model_name).")
      .def_prop_ro("num_inputs", &bcdl::Engine::numInputs)
      .def_prop_ro("num_outputs", &bcdl::Engine::numOutputs)
      .def("input_name", &bcdl::Engine::inputName, "index"_a)
      .def("output_name", &bcdl::Engine::outputName, "index"_a)
      .def("input_shape", &bcdl::Engine::inputShape, "index"_a)
      .def("output_shape", &bcdl::Engine::outputShape, "index"_a)
      .def("input_bytes",
           [](bcdl::Engine& e, int i) { return e.inputBytes(i); }, "index"_a)
      .def("input_packed_bytes",
           [](bcdl::Engine& e, int i) { return e.inputPackedBytes(i); }, "index"_a,
           "Byte size of input[i] as a contiguous row-major array. Smaller than "
           "input_bytes() when the model pads a dimension.")
      .def("input_stride", &bcdl::Engine::inputStride, "index"_a,
           "Resolved byte strides of input[i], innermost last.")
      .def("output_stride", &bcdl::Engine::outputStride, "index"_a,
           "Resolved byte strides of output[i], innermost last. Outputs are "
           "padded too — reshape output_bytes() flat and you shear the tensor.")
      .def(
          "input_buffer_bytes",
          [](bcdl::Engine& e, int i) {
            return nb::bytes(static_cast<const char*>(e.inputData(i)), e.inputBytes(i));
          },
          "index"_a,
          "Input[i]'s device buffer in the device layout — how set_input() "
          "actually laid the data out.")
      .def("input_dtype",
           [](bcdl::Engine& e, int i) { return dtypeName(e.inputType(i)); }, "index"_a)
      .def("output_dtype",
           [](bcdl::Engine& e, int i) { return dtypeName(e.outputType(i)); }, "index"_a)
      .def(
          "set_input",
          // `const void` matters: setInput only READS the array, and demanding a
          // writable one rejects the most ordinary multi-stage case there is —
          // feeding one model's output into the next. Engine.infer() builds its
          // outputs with np.frombuffer, which is read-only, so a two-model
          // pipeline failed here with a type error that named no cause.
          [](bcdl::Engine& e, int i, nb::ndarray<nb::ro, nb::c_contig> arr) {
            e.setInput(i, arr.data(), arr.nbytes());
          },
          "index"_a, "array"_a)
      // GIL released: infer() blocks in hbUCPWaitTaskDone for as long as the
      // model takes (116 ms for depth, 271 ms for SMOKE) and touches nothing
      // Python. Holding it there stalls every other thread in the interpreter.
      .def("infer", &bcdl::Engine::infer, "timeout_ms"_a = 0,
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "output_bytes",
          [](bcdl::Engine& e, int i) {
            return nb::bytes(static_cast<const char*>(e.outputData(i)),
                             e.outputBytes(i));
          },
          "index"_a,
          "Output[i]'s device buffer verbatim, padding and all. Reshaping it "
          "flat shears a padded tensor — use output_packed() unless you are "
          "walking output_stride() yourself.")
      .def("output_packed_bytes",
           [](bcdl::Engine& e, int i) { return e.outputPackedBytes(i); }, "index"_a,
           "Byte size of output[i] as a contiguous row-major array. Smaller than "
           "the device buffer whenever the model pads a dimension.")
      .def(
          "output_packed",
          [](bcdl::Engine& e, int i) {
            std::string buf(e.outputPackedBytes(i), '\0');
            e.copyOutput(i, buf.data(), buf.size());
            return nb::bytes(buf.data(), buf.size());
          },
          "index"_a,
          "Output[i] gathered into a contiguous row-major array of its declared "
          "shape and dtype (padding removed, quantisation left alone).");

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

  // --- open-vocabulary label table (YOLOE prompt-free) -----------------------
  nb::class_<bcdl::LabelMap>(m, "LabelMap")
      .def(nb::init<>())
      .def_static("from_file", &bcdl::LabelMap::fromFile, "path"_a)
      .def_static("from_list", &bcdl::LabelMap::fromList, "names"_a)
      .def_rw("names", &bcdl::LabelMap::names)
      .def("__len__", [](const bcdl::LabelMap& m) { return m.size(); })
      .def("name", &bcdl::LabelMap::name, "class_id"_a);

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
            std::vector<uint8_t> b;
            {
              nb::gil_scoped_release rel;  // JPU round trip
              b = e.encode(src);
            }
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
            // bytes are immutable, so the pointer stays valid without the GIL.
            const auto* p = reinterpret_cast<const uint8_t*>(data.c_str());
            const std::size_t n = data.size();
            nb::gil_scoped_release rel;  // JPU round trip
            return d.decode(p, n);
          },
          "data"_a, "Decode JPEG bytes into an owned NV12 VpImage.")
      .def_prop_ro("out_format", &bcdl::JpegDecoder::outFormat);

#ifdef BCDL_HAVE_GDC
  // --- GDC hardware letterbox (VPS /dev/gdc, bypasses the offline vDSP) -------
  nb::class_<bcdl::GdcLetterbox>(m, "GdcLetterbox")
      .def(nb::init<int, int, int, int, uint8_t>(), "in_w"_a, "in_h"_a, "out_w"_a,
           "out_h"_a, "pad"_a = 114,
           "Persistent GDC vnode for a fixed (in_w,in_h)->(out_w,out_h) hardware "
           "letterbox. The warp LUT is generated at construction.")
      .def(
          "run",
          [](bcdl::GdcLetterbox& g, const bcdl::VpImage& src) {
            nb::gil_scoped_release rel;  // copies + GDC hardware wait, no Python
            bcdl::VpImage dst(g.outputWidth(), g.outputHeight(), HB_VP_IMAGE_FORMAT_NV12);
            g.run(src, dst);
            return dst;
          },
          "src"_a, "GDC-letterbox an NV12 VpImage into a new out_w x out_h NV12 VpImage.")
      .def_prop_ro("info", &bcdl::GdcLetterbox::info)
      .def_prop_ro("output_width", &bcdl::GdcLetterbox::outputWidth)
      .def_prop_ro("output_height", &bcdl::GdcLetterbox::outputHeight);

  // --- GDC hardware dense remap (fixed warp, e.g. stereo rectification) -------
  nb::class_<bcdl::GdcRemap>(m, "GdcRemap")
      .def(
          "__init__",
          [](bcdl::GdcRemap* self, nb::ndarray<float, nb::c_contig> map_x,
             nb::ndarray<float, nb::c_contig> map_y, int in_w, int in_h,
             int grid_step) {
            if (map_x.ndim() != 2 || map_y.ndim() != 2 ||
                map_x.shape(0) != map_y.shape(0) || map_x.shape(1) != map_y.shape(1))
              throw std::runtime_error(
                  "GdcRemap expects two (out_h, out_w) float32 maps of equal shape");
            const int out_h = static_cast<int>(map_x.shape(0));
            const int out_w = static_cast<int>(map_x.shape(1));
            new (self) bcdl::GdcRemap(static_cast<const float*>(map_x.data()),
                                      static_cast<const float*>(map_y.data()),
                                      in_w, in_h, out_w, out_h, grid_step);
          },
          "map_x"_a, "map_y"_a, "in_w"_a, "in_h"_a, "grid_step"_a = 16,
          "Persistent GDC vnode for a FIXED dense warp with cv2.remap semantics "
          "(out(x,y) = in(map_x[y,x], map_y[y,x])). The (out_h, out_w) float32 "
          "maps are sampled every grid_step px into a CUSTOM warp grid whose "
          "LUT is generated at construction (hbn_gen_gdc_cfg) — no bin file.")
      .def(
          "run",
          [](bcdl::GdcRemap& g, const bcdl::VpImage& src) {
            // GIL released: the copies + hardware wait are pure C++, and two
            // GdcRemap instances (e.g. stereo top/bottom) can then overlap
            // their CPU work with each other's GDC op from Python threads.
            nb::gil_scoped_release rel;
            bcdl::VpImage dst(g.outputWidth(), g.outputHeight(), HB_VP_IMAGE_FORMAT_NV12);
            g.run(src, dst);
            return dst;
          },
          "src"_a, "Warp an NV12 VpImage into a new out_w x out_h NV12 VpImage.")
      .def_prop_ro("input_width", &bcdl::GdcRemap::inputWidth)
      .def_prop_ro("input_height", &bcdl::GdcRemap::inputHeight)
      .def_prop_ro("output_width", &bcdl::GdcRemap::outputWidth)
      .def_prop_ro("output_height", &bcdl::GdcRemap::outputHeight);
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
            std::vector<uint8_t> b;
            {
              nb::gil_scoped_release rel;  // VPU: queue + wait for a packet
              b = e.encode(frame);
            }
            return nb::bytes(reinterpret_cast<const char*>(b.data()), b.size());
          },
          "frame"_a,
          "Feed one NV12 frame and wait briefly for a packet; returns bytes "
          "(may be empty if the encoder buffered the frame).")
      .def(
          "feed",
          [](bcdl::VideoEncoder& e, const bcdl::VpImage& frame, uint64_t pts_us) {
            nb::gil_scoped_release rel;  // blocks up to 1 s on a free input buffer
            return e.feed(frame, pts_us);
          },
          "frame"_a, nb::arg("pts_us") = 0,
          "Queue one NV12 frame for encoding (does not wait for output). "
          "Returns False if the codec's input queue is full — drain with "
          "receive() and retry rather than treating it as fatal.")
      .def(
          "receive",
          [](bcdl::VideoEncoder& e, int timeout_ms) -> std::optional<nb::bytes> {
            std::vector<uint8_t> b;
            bool ok;
            {
              nb::gil_scoped_release rel;  // waits up to timeout_ms on the VPU
              ok = e.receive(b, timeout_ms);
            }
            if (!ok) return std::nullopt;
            return nb::bytes(reinterpret_cast<const char*>(b.data()), b.size());
          },
          nb::arg("timeout_ms") = 0,
          "Drain one compressed packet, or None if none ready.")
      .def(
          "flush",
          [](bcdl::VideoEncoder& e) -> std::optional<nb::bytes> {
            std::vector<uint8_t> b;
            bool ok;
            {
              nb::gil_scoped_release rel;  // drains the tail: up to 300 ms
              ok = e.flush(b);
            }
            if (!ok) return std::nullopt;
            return nb::bytes(reinterpret_cast<const char*>(b.data()), b.size());
          },
          "Signal end-of-stream, then drain the remaining packets; call until "
          "it returns None.")
      .def("feed_end_of_stream", &bcdl::VideoEncoder::feedEndOfStream,
           nb::call_guard<nb::gil_scoped_release>(),
           "Queue an end-of-stream marker without draining. Idempotent.")
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
            // bytes are immutable, so the pointer stays valid without the GIL.
            const auto* p = reinterpret_cast<const uint8_t*>(data.c_str());
            const std::size_t n = data.size();
            bcdl::VpImage out;
            bool ok;
            {
              nb::gil_scoped_release rel;  // VPU: queue + wait for a frame
              ok = d.decode(p, n, out);
            }
            if (ok) return std::optional<bcdl::VpImage>(std::move(out));
            return std::nullopt;
          },
          "data"_a,
          "Feed one access unit; returns an NV12 VpImage when a frame is "
          "ready, else None (the decoder is still buffering reference frames).")
      .def(
          "feed",
          [](bcdl::VideoDecoder& d, nb::bytes data) {
            const auto* p = reinterpret_cast<const uint8_t*>(data.c_str());
            const std::size_t n = data.size();
            nb::gil_scoped_release rel;  // blocks up to 1 s on a free input buffer
            return d.feed(p, n);
          },
          "data"_a, "Queue one access unit for decoding (does not wait for output).")
      .def(
          "receive",
          [](bcdl::VideoDecoder& d, int timeout_ms) -> std::optional<bcdl::VpImage> {
            bcdl::VpImage out;
            bool ok;
            {
              nb::gil_scoped_release rel;  // waits up to timeout_ms on the VPU
              ok = d.receive(out, timeout_ms);
            }
            if (ok) return std::optional<bcdl::VpImage>(std::move(out));
            return std::nullopt;
          },
          "timeout_ms"_a = 0,
          "Drain one decoded frame in display order (timeout_ms=0 is non-blocking); "
          "None on timeout / no frame.")
      .def(
          "flush",
          [](bcdl::VideoDecoder& d) -> std::optional<bcdl::VpImage> {
            bcdl::VpImage out;
            bool ok;
            {
              nb::gil_scoped_release rel;  // drains the reorder tail: up to 300 ms
              ok = d.flush(out);
            }
            if (ok) return std::optional<bcdl::VpImage>(std::move(out));
            return std::nullopt;
          },
          "After the last feed(): drain the reorder tail; call until it returns None.")
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
  // Depth refinement: RGB + raw sensor depth -> cleaned metric depth + trust mask
  // ===========================================================================
  nb::class_<bcdl::DepthRefineConfig>(m, "DepthRefineConfig")
      .def(nb::init<>())
      .def_rw("encoder_height", &bcdl::DepthRefineConfig::encoder_height)
      .def_rw("encoder_width", &bcdl::DepthRefineConfig::encoder_width)
      .def_rw("min_valid_depth", &bcdl::DepthRefineConfig::min_valid_depth)
      .def_rw("mask_threshold", &bcdl::DepthRefineConfig::mask_threshold)
      .def_rw("apply_mask", &bcdl::DepthRefineConfig::apply_mask);

  nb::class_<bcdl::Intrinsics>(m, "Intrinsics")
      .def(nb::init<>())
      .def("__init__",
           [](bcdl::Intrinsics* self, float fx, float fy, float cx, float cy) {
             new (self) bcdl::Intrinsics{fx, fy, cx, cy};
           },
           "fx"_a, "fy"_a, "cx"_a, "cy"_a)
      .def_rw("fx", &bcdl::Intrinsics::fx)
      .def_rw("fy", &bcdl::Intrinsics::fy)
      .def_rw("cx", &bcdl::Intrinsics::cx)
      .def_rw("cy", &bcdl::Intrinsics::cy);

  m.def("scale_intrinsics", &bcdl::scaleIntrinsics, "k"_a, "from_w"_a, "from_h"_a,
        "to_w"_a, "to_h"_a,
        "Rescale pinhole intrinsics between image sizes (plain resize, not crop).");

  nb::class_<bcdl::RefinedDepth>(m, "RefinedDepth")
      .def(nb::init<>())
      .def_ro("width", &bcdl::RefinedDepth::width)
      .def_ro("height", &bcdl::RefinedDepth::height)
      .def_ro("vmin", &bcdl::RefinedDepth::vmin)
      .def_ro("vmax", &bcdl::RefinedDepth::vmax)
      .def_prop_ro(
          "depth",
          [](const bcdl::RefinedDepth& r) {
            return toNumpy(std::vector<float>(r.depth),
                           {static_cast<size_t>(r.height), static_cast<size_t>(r.width)});
          },
          nb::rv_policy::reference,
          "Refined metric depth as a float32 (H, W) numpy array, 0 where rejected.")
      .def_prop_ro(
          "mask",
          [](const bcdl::RefinedDepth& r) {
            return toNumpy(std::vector<uint8_t>(r.mask),
                           {static_cast<size_t>(r.height), static_cast<size_t>(r.width)});
          },
          nb::rv_policy::reference,
          "Per-pixel trust mask as a uint8 (H, W) numpy array, 1 = trusted.");

  m.def(
      "preprocess_refine_image",
      [](nb::ndarray<const uint8_t, nb::c_contig> bgr, const bcdl::DepthRefineConfig& cfg) {
        if (bgr.ndim() != 3 || bgr.shape(2) != 3)
          throw std::invalid_argument("preprocess_refine_image expects an (H, W, 3) BGR array");
        std::vector<float> out;
        bcdl::preprocessRefineImage(bgr.data(), static_cast<int>(bgr.shape(1)),
                                    static_cast<int>(bgr.shape(0)),
                                    static_cast<int>(bgr.shape(1)) * 3, cfg, &out);
        return toNumpy(std::move(out), {1, 3, static_cast<size_t>(cfg.encoder_height),
                                        static_cast<size_t>(cfg.encoder_width)});
      },
      "bgr"_a, "config"_a = bcdl::DepthRefineConfig{},
      "BGR (H, W, 3) uint8 -> the model's `image` input, float32 (1, 3, eh, ew).");

  m.def(
      "preprocess_refine_depth",
      [](nb::ndarray<const float, nb::c_contig> depth, const bcdl::DepthRefineConfig& cfg) {
        if (depth.ndim() != 2)
          throw std::invalid_argument("preprocess_refine_depth expects an (H, W) float array");
        std::vector<float> out;
        bcdl::preprocessRefineDepth(depth.data(), static_cast<int>(depth.shape(1)),
                                    static_cast<int>(depth.shape(0)),
                                    static_cast<int>(depth.shape(1)), cfg, &out);
        return toNumpy(std::move(out), {1, 1, static_cast<size_t>(cfg.encoder_height),
                                        static_cast<size_t>(cfg.encoder_width)});
      },
      "depth_m"_a, "config"_a = bcdl::DepthRefineConfig{},
      "Metric depth (H, W) float32 -> the model's `depth_log` input, float32 (1, 1, eh, ew).");

  m.def(
      "decode_refined_depth",
      [](nb::ndarray<const float, nb::c_contig> depth, nb::object mask_logit,
         const bcdl::DepthRefineConfig& cfg) {
        std::vector<int> dims;
        for (size_t d = 0; d < depth.ndim(); ++d)
          if (depth.shape(d) > 1) dims.push_back(static_cast<int>(depth.shape(d)));
        if (dims.size() < 2)
          throw std::invalid_argument("decode_refined_depth: cannot resolve HxW");
        const int H = dims[dims.size() - 2], W = dims[dims.size() - 1];
        const float* mp = nullptr;
        nb::ndarray<const float, nb::c_contig> mask_arr;
        if (!mask_logit.is_none()) {
          mask_arr = nb::cast<nb::ndarray<const float, nb::c_contig>>(mask_logit);
          mp = mask_arr.data();
        }
        return bcdl::decodeRefinedDepth(depth.data(), mp, H, W, cfg);
      },
      "depth"_a, "mask_logit"_a = nb::none(), "config"_a = bcdl::DepthRefineConfig{},
      "Decode the refinement model's outputs into a RefinedDepth.");

  m.def(
      "depth_to_pointcloud",
      [](const bcdl::RefinedDepth& r, const bcdl::Intrinsics& k) {
        return toNumpy(bcdl::depthToPointCloud(r, k),
                       {static_cast<size_t>(r.height), static_cast<size_t>(r.width), 3});
      },
      "refined"_a, "intrinsics"_a,
      "Unproject to a camera-space point cloud, float32 (H, W, 3) in metres.");

  nb::class_<bcdl::DepthRefiner>(m, "DepthRefiner")
      .def(nb::init<bcdl::Engine&, bcdl::DepthRefineConfig, int, int, int, int>(), "engine"_a,
           "config"_a = bcdl::DepthRefineConfig{}, "image_input"_a = 0, "depth_input"_a = 1,
           "depth_output"_a = 0, "mask_output"_a = 1, nb::keep_alive<1, 2>())
      .def(
          "run",
          [](bcdl::DepthRefiner& self, nb::ndarray<const uint8_t, nb::c_contig> bgr,
             nb::ndarray<const float, nb::c_contig> depth_m) {
            if (bgr.ndim() != 3 || bgr.shape(2) != 3)
              throw std::invalid_argument("run() expects an (H, W, 3) BGR array");
            if (depth_m.ndim() != 2)
              throw std::invalid_argument("run() expects an (H, W) float depth array");
            nb::gil_scoped_release rel;  // preproc + BPU infer + decode
            return self.run(bgr.data(), static_cast<int>(bgr.shape(1)),
                            static_cast<int>(bgr.shape(0)), static_cast<int>(bgr.shape(1)) * 3,
                            depth_m.data(), static_cast<int>(depth_m.shape(1)));
          },
          "bgr"_a, "depth_m"_a,
          "Preprocess, infer and decode one RGB-D frame.")
      .def("postprocess", &bcdl::DepthRefiner::postprocess)
      .def_prop_ro("config", &bcdl::DepthRefiner::config);

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
            nb::gil_scoped_release rel;  // preproc + BPU infer + decode
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
  // Face detection (5 landmarks) + alignment
  // ===========================================================================
  nb::class_<bcdl::FaceDetection>(m, "FaceDetection")
      .def_ro("x1", &bcdl::FaceDetection::x1)
      .def_ro("y1", &bcdl::FaceDetection::y1)
      .def_ro("x2", &bcdl::FaceDetection::x2)
      .def_ro("y2", &bcdl::FaceDetection::y2)
      .def_ro("score", &bcdl::FaceDetection::score)
      .def_ro("landmarks", &bcdl::FaceDetection::landmarks)
      .def("__repr__", [](const bcdl::FaceDetection& f) {
        char b[128];
        std::snprintf(b, sizeof(b), "FaceDetection(%.0f,%.0f-%.0f,%.0f score=%.3f)",
                      f.x1, f.y1, f.x2, f.y2, f.score);
        return std::string(b);
      });

  nb::class_<bcdl::FaceDetectConfig>(m, "FaceDetectConfig")
      .def(nb::init<>())
      .def_rw("conf_thresh", &bcdl::FaceDetectConfig::conf_thresh)
      .def_rw("iou_thresh", &bcdl::FaceDetectConfig::iou_thresh)
      .def_rw("max_faces", &bcdl::FaceDetectConfig::max_faces)
      .def_rw("strides", &bcdl::FaceDetectConfig::strides)
      .def_rw("num_anchors", &bcdl::FaceDetectConfig::num_anchors);

  m.def("decode_scrfd",
        [](std::vector<nb::ndarray<const float, nb::c_contig>> score,
           std::vector<nb::ndarray<const float, nb::c_contig>> bbox,
           std::vector<nb::ndarray<const float, nb::c_contig>> kps,
           const bcdl::FaceDetectConfig& cfg, const bcdl::LetterboxInfo& lb) {
          std::vector<const float*> s, b, k;
          std::vector<int> counts;
          for (auto& a : score) { s.push_back(a.data()); counts.push_back(static_cast<int>(a.size())); }
          for (auto& a : bbox) b.push_back(a.data());
          for (auto& a : kps) k.push_back(a.data());
          return bcdl::decodeScrfd(s, b, k, counts, cfg, lb);
        },
        "score"_a, "bbox"_a, "kps"_a, "config"_a, "letterbox"_a,
        "Decode SCRFD score/bbox/kps tensors into faces with 5 landmarks.");

  m.def("arcface_template", &bcdl::arcFaceTemplate,
        "The canonical 112x112 five-point template used for face alignment.");

  m.def("similarity_transform", &bcdl::similarityTransform, "src"_a, "dst"_a,
        "Closed-form Umeyama similarity transform (rot+uniform scale+translation) "
        "as {a,b,tx,c,d,ty}.");

  m.def("align_face",
        [](nb::ndarray<const uint8_t, nb::c_contig> bgr,
           const std::vector<std::pair<float, float>>& landmarks, int size) {
          if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
            throw std::invalid_argument("align_face: expected an HxWx3 uint8 BGR image");
          }
          const int h = static_cast<int>(bgr.shape(0));
          const int w = static_cast<int>(bgr.shape(1));
          std::vector<uint8_t> out = bcdl::alignFace(bgr.data(), w, h, w * 3,
                                                     landmarks, size);
          auto* held = new std::vector<uint8_t>(std::move(out));
          nb::capsule owner(held, [](void* p) noexcept {
            delete static_cast<std::vector<uint8_t>*>(p);
          });
          const size_t shape[3] = {static_cast<size_t>(size),
                                   static_cast<size_t>(size), 3};
          return nb::ndarray<nb::numpy, uint8_t>(held->data(), 3, shape, owner);
        },
        "bgr"_a, "landmarks"_a, "size"_a = 112,
        "Warp a face into a size x size aligned BGR crop from its 5 landmarks.");

  nb::class_<bcdl::FaceDetector>(m, "FaceDetector")
      .def(nb::init<bcdl::Engine&, bcdl::FaceDetectConfig, int>(), "engine"_a,
           "config"_a = bcdl::FaceDetectConfig{}, "output_base"_a = 0,
           nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::FaceDetector::postprocess, "letterbox"_a)
      .def_prop_ro("config", &bcdl::FaceDetector::config);

  // ===========================================================================
  // ViTPose whole-body 133-keypoint head (top-down heatmaps)
  // ===========================================================================
  nb::class_<bcdl::WholeBodyCrop>(m, "WholeBodyCrop")
      .def(nb::init<>())
      .def_rw("x1", &bcdl::WholeBodyCrop::x1)
      .def_rw("y1", &bcdl::WholeBodyCrop::y1)
      .def_rw("pad_left", &bcdl::WholeBodyCrop::pad_left)
      .def_rw("pad_top", &bcdl::WholeBodyCrop::pad_top)
      .def_rw("padded_w", &bcdl::WholeBodyCrop::padded_w)
      .def_rw("padded_h", &bcdl::WholeBodyCrop::padded_h);

  nb::class_<bcdl::WholeBodyConfig>(m, "WholeBodyConfig")
      .def(nb::init<>())
      .def_rw("kpt_thresh", &bcdl::WholeBodyConfig::kpt_thresh)
      .def_rw("blur_kernel", &bcdl::WholeBodyConfig::blur_kernel)
      .def_rw("box_pad", &bcdl::WholeBodyConfig::box_pad)
      .def_prop_rw(
          "mean",
          [](const bcdl::WholeBodyConfig& c) {
            return std::vector<float>(c.mean, c.mean + 3);
          },
          [](bcdl::WholeBodyConfig& c, const std::vector<float>& v) {
            if (v.size() != 3) throw std::invalid_argument("mean must have 3 values (RGB)");
            std::copy(v.begin(), v.end(), c.mean);
          })
      .def_prop_rw(
          "std",
          [](const bcdl::WholeBodyConfig& c) {
            return std::vector<float>(c.std, c.std + 3);
          },
          [](bcdl::WholeBodyConfig& c, const std::vector<float>& v) {
            if (v.size() != 3) throw std::invalid_argument("std must have 3 values (RGB)");
            std::copy(v.begin(), v.end(), c.std);
          });

  // wholebody_preprocess(): returns (input, crop) — the NCHW float array to feed
  // the model and the geometry decode_wholebody() needs to invert.
  m.def(
      "wholebody_preprocess",
      [](nb::ndarray<const uint8_t, nb::c_contig> bgr, float x1, float y1, float x2, float y2,
         int in_w, int in_h, const bcdl::WholeBodyConfig& cfg) {
        if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
          throw std::invalid_argument(
              "wholebody_preprocess: expected an HxWx3 uint8 BGR image");
        }
        const int h = static_cast<int>(bgr.shape(0));
        const int w = static_cast<int>(bgr.shape(1));
        std::vector<float> out;
        bcdl::WholeBodyCrop crop = bcdl::wholeBodyPreprocess(
            bgr.data(), w, h, w * 3, x1, y1, x2, y2, in_w, in_h, cfg, out);
        auto* held = new std::vector<float>(std::move(out));
        nb::capsule owner(held, [](void* p) noexcept {
          delete static_cast<std::vector<float>*>(p);
        });
        const size_t shape[4] = {1, 3, static_cast<size_t>(in_h), static_cast<size_t>(in_w)};
        return nb::make_tuple(nb::ndarray<nb::numpy, float>(held->data(), 4, shape, owner),
                              crop);
      },
      "bgr"_a, "x1"_a, "y1"_a, "x2"_a, "y2"_a, "in_w"_a = 192, "in_h"_a = 256,
      "config"_a = bcdl::WholeBodyConfig{},
      "Crop a person box to the model's canvas: widen by box_pad, zero-pad to the "
      "3:4 aspect, resize, /255, ImageNet z-score, NCHW RGB. Returns (input, crop).");

  // decode_wholebody(): numpy path — pass the [K,H,W] (or [1,K,H,W]) heatmap.
  m.def(
      "decode_wholebody",
      [](nb::ndarray<const float, nb::c_contig> hm, const bcdl::WholeBodyCrop& crop,
         const bcdl::WholeBodyConfig& cfg) {
        int k, h, w;
        if (hm.ndim() == 4) {
          if (hm.shape(0) != 1) {
            throw std::invalid_argument("decode_wholebody: batch must be 1");
          }
          k = static_cast<int>(hm.shape(1));
          h = static_cast<int>(hm.shape(2));
          w = static_cast<int>(hm.shape(3));
        } else if (hm.ndim() == 3) {
          k = static_cast<int>(hm.shape(0));
          h = static_cast<int>(hm.shape(1));
          w = static_cast<int>(hm.shape(2));
        } else {
          throw std::invalid_argument("decode_wholebody: expected a [K,H,W] heatmap");
        }
        return bcdl::decodeWholeBody(hm.data(), k, h, w, crop, cfg);
      },
      "heatmaps"_a, "crop"_a, "config"_a = bcdl::WholeBodyConfig{},
      "Decode channel-first heatmaps into keypoints in original-image pixels "
      "(argmax + DARK-UDP sub-pixel refinement).");

  nb::class_<bcdl::WholeBodyEstimator>(m, "WholeBodyEstimator")
      .def(nb::init<bcdl::Engine&, bcdl::WholeBodyConfig, int>(), "engine"_a,
           "config"_a = bcdl::WholeBodyConfig{}, "output_index"_a = 0,
           nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::WholeBodyEstimator::postprocess, "crop"_a)
      .def_prop_ro("config", &bcdl::WholeBodyEstimator::config);

  // ===========================================================================
  // XFeat sparse local features + mutual-NN matching
  // ===========================================================================
  nb::class_<bcdl::Feature>(m, "Feature")
      .def_ro("x", &bcdl::Feature::x)
      .def_ro("y", &bcdl::Feature::y)
      .def_ro("score", &bcdl::Feature::score);

  nb::class_<bcdl::FeatureMatch>(m, "FeatureMatch")
      .def_ro("a", &bcdl::FeatureMatch::a)
      .def_ro("b", &bcdl::FeatureMatch::b)
      .def_ro("score", &bcdl::FeatureMatch::score);

  nb::class_<bcdl::XfeatConfig>(m, "XfeatConfig")
      .def(nb::init<>())
      .def_rw("detection_thresh", &bcdl::XfeatConfig::detection_thresh)
      .def_rw("nms_kernel", &bcdl::XfeatConfig::nms_kernel)
      .def_rw("top_k", &bcdl::XfeatConfig::top_k);

  // FeatureSet's descriptors come back as one (N, dim) float array rather than
  // per-feature lists: matching is a dense dot product, so callers want the
  // matrix, and copying N x 64 floats per frame into Python objects would cost
  // more than the decode.
  nb::class_<bcdl::FeatureSet>(m, "FeatureSet")
      .def_ro("keypoints", &bcdl::FeatureSet::keypoints)
      .def_ro("dim", &bcdl::FeatureSet::dim)
      // rv_policy::move, not the property default: these arrays already carry
      // their own capsule owner, and reference_internal cannot be applied on top.
      .def_prop_ro(
          "descriptors",
          [](const bcdl::FeatureSet& f) {
            auto* held = new std::vector<float>(f.descriptors);
            nb::capsule owner(held, [](void* p) noexcept {
              delete static_cast<std::vector<float>*>(p);
            });
            const size_t shape[2] = {f.size(), static_cast<size_t>(f.dim)};
            return nb::ndarray<nb::numpy, float>(held->data(), 2, shape, owner);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "xy",
          [](const bcdl::FeatureSet& f) {
            auto* held = new std::vector<float>(f.size() * 2);
            for (size_t i = 0; i < f.size(); ++i) {
              (*held)[2 * i] = f.keypoints[i].x;
              (*held)[2 * i + 1] = f.keypoints[i].y;
            }
            nb::capsule owner(held, [](void* p) noexcept {
              delete static_cast<std::vector<float>*>(p);
            });
            const size_t shape[2] = {f.size(), 2};
            return nb::ndarray<nb::numpy, float>(held->data(), 2, shape, owner);
          },
          nb::rv_policy::move)
      .def("__len__", [](const bcdl::FeatureSet& f) { return f.size(); });

  m.def(
      "xfeat_preprocess",
      [](nb::ndarray<const uint8_t, nb::c_contig> bgr, int in_w, int in_h) {
        if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
          throw std::invalid_argument("xfeat_preprocess: expected an HxWx3 uint8 BGR image");
        }
        const int h = static_cast<int>(bgr.shape(0));
        const int w = static_cast<int>(bgr.shape(1));
        std::vector<float> out;
        float sx = 1.0f, sy = 1.0f;
        bcdl::xfeatPreprocess(bgr.data(), w, h, w * 3, in_w, in_h, out, &sx, &sy);
        auto* held = new std::vector<float>(std::move(out));
        nb::capsule owner(held, [](void* p) noexcept {
          delete static_cast<std::vector<float>*>(p);
        });
        const size_t shape[4] = {1, 1, static_cast<size_t>(in_h), static_cast<size_t>(in_w)};
        return nb::make_tuple(nb::ndarray<nb::numpy, float>(held->data(), 4, shape, owner),
                              sx, sy);
      },
      "bgr"_a, "in_w"_a = 640, "in_h"_a = 480,
      "Grayscale (channel MEAN, not luma) + resize + InstanceNorm into the "
      "model's [1,1,H,W] input. Returns (input, scale_x, scale_y).");

  m.def(
      "decode_xfeat",
      [](nb::ndarray<const float, nb::c_contig> feats,
         nb::ndarray<const float, nb::c_contig> kpts,
         nb::ndarray<const float, nb::c_contig> heat, const bcdl::XfeatConfig& cfg,
         float scale_x, float scale_y) {
        auto dims = [](const auto& a) {
          return a.ndim() == 4 ? std::array<int, 3>{static_cast<int>(a.shape(1)),
                                                    static_cast<int>(a.shape(2)),
                                                    static_cast<int>(a.shape(3))}
                               : std::array<int, 3>{static_cast<int>(a.shape(0)),
                                                    static_cast<int>(a.shape(1)),
                                                    static_cast<int>(a.shape(2))};
        };
        if (feats.ndim() < 3 || kpts.ndim() < 3 || heat.ndim() < 3) {
          throw std::invalid_argument("decode_xfeat: expected [C,H,W] or [1,C,H,W] maps");
        }
        const auto f = dims(feats), k = dims(kpts), r = dims(heat);
        if (f[0] != 64 || k[0] != 65 || r[0] != 1) {
          throw std::invalid_argument(
              "decode_xfeat: channel counts must be 64 / 65 / 1 (feats/keypoints/heatmap)");
        }
        if (k[1] != f[1] || k[2] != f[2] || r[1] != f[1] || r[2] != f[2]) {
          throw std::invalid_argument("decode_xfeat: the three maps must share (H,W)");
        }
        return bcdl::decodeXfeat(feats.data(), kpts.data(), heat.data(), f[1], f[2],
                                 f[1] * 8, f[2] * 8, cfg, scale_x, scale_y);
      },
      "feats"_a, "keypoints"_a, "heatmap"_a, "config"_a = bcdl::XfeatConfig{},
      "scale_x"_a = 1.0f, "scale_y"_a = 1.0f,
      "Decode the three XFeat maps into sparse features (softmax -> NMS -> top-k "
      "-> bilinear descriptor sampling), in original-image pixels.");

  m.def("match_features", &bcdl::matchFeatures, "a"_a, "b"_a, "min_cossim"_a = 0.82f,
        "Mutual nearest-neighbour matching with a cosine floor. O(|a|*|b|*dim).");

  nb::class_<bcdl::FeatureExtractor>(m, "FeatureExtractor")
      .def(nb::init<bcdl::Engine&, bcdl::XfeatConfig, int>(), "engine"_a,
           "config"_a = bcdl::XfeatConfig{}, "output_base"_a = 0, nb::keep_alive<1, 2>())
      .def(
          "extract",
          [](bcdl::FeatureExtractor& e, nb::ndarray<const uint8_t, nb::c_contig> bgr,
             int timeout_ms) {
            if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
              throw std::invalid_argument("extract: expected an HxWx3 uint8 BGR image");
            }
            const int h = static_cast<int>(bgr.shape(0));
            const int w = static_cast<int>(bgr.shape(1));
            return e.extract(bgr.data(), w, h, w * 3, timeout_ms);
          },
          "bgr"_a, "timeout_ms"_a = 0)
      .def("postprocess", &bcdl::FeatureExtractor::postprocess, "scale_x"_a = 1.0f,
           "scale_y"_a = 1.0f)
      .def_prop_ro("config", &bcdl::FeatureExtractor::config);

  // ===========================================================================
  // Super-resolution (tiled)
  // ===========================================================================
  nb::class_<bcdl::SuperResConfig>(m, "SuperResConfig")
      .def(nb::init<>())
      .def_rw("overlap", &bcdl::SuperResConfig::overlap);

  nb::class_<bcdl::TilePlacement>(m, "TilePlacement")
      .def_ro("x", &bcdl::TilePlacement::x)
      .def_ro("y", &bcdl::TilePlacement::y);

  m.def("plan_tiles", &bcdl::planTiles, "width"_a, "height"_a, "tile_w"_a, "tile_h"_a,
        "overlap"_a,
        "Tile origins covering an image; the last tile on each axis is flush "
        "with the far edge, so the real overlap can exceed the requested one.");

  m.def("tile_weight", &bcdl::tileWeight, "i"_a, "len"_a, "ramp"_a,
        "Cross-fade weight along a tile axis. Always > 0, which is why the "
        "blend can normalize instead of special-casing image borders.");

  nb::class_<bcdl::SuperResolver>(m, "SuperResolver")
      .def(nb::init<bcdl::Engine&, bcdl::SuperResConfig, int, int>(), "engine"_a,
           "config"_a = bcdl::SuperResConfig{}, "input_index"_a = 0,
           "output_index"_a = 0, nb::keep_alive<1, 2>())
      .def(
          "upscale",
          [](bcdl::SuperResolver& s, nb::ndarray<const uint8_t, nb::c_contig> bgr,
             int timeout_ms) {
            if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
              throw std::invalid_argument("upscale: expected an HxWx3 uint8 BGR image");
            }
            const int h = static_cast<int>(bgr.shape(0));
            const int w = static_cast<int>(bgr.shape(1));
            bcdl::SrImage out = s.upscale(bgr.data(), w, h, w * 3, timeout_ms);
            auto* held = new std::vector<uint8_t>(std::move(out.data));
            nb::capsule owner(held, [](void* p) noexcept {
              delete static_cast<std::vector<uint8_t>*>(p);
            });
            const size_t shape[3] = {static_cast<size_t>(out.height),
                                     static_cast<size_t>(out.width), 3};
            return nb::ndarray<nb::numpy, uint8_t>(held->data(), 3, shape, owner);
          },
          "bgr"_a, "timeout_ms"_a = 0)
      .def_prop_ro("scale", &bcdl::SuperResolver::scale)
      .def_prop_ro("tile", &bcdl::SuperResolver::tile)
      .def_prop_ro("last_tile_count", &bcdl::SuperResolver::lastTileCount)
      .def_prop_ro("config", &bcdl::SuperResolver::config);

  // ===========================================================================
  // Anchor-based YOLOv5 detection head (panoptic driving models)
  // ===========================================================================
  nb::class_<bcdl::Anchor>(m, "Anchor")
      .def(nb::init<>())
      .def("__init__", [](bcdl::Anchor* a, float w, float h) { new (a) bcdl::Anchor{w, h}; },
           "w"_a, "h"_a)
      .def_rw("w", &bcdl::Anchor::w)
      .def_rw("h", &bcdl::Anchor::h);

  nb::class_<bcdl::AnchorDetectConfig>(m, "AnchorDetectConfig")
      .def(nb::init<>())
      .def_rw("num_classes", &bcdl::AnchorDetectConfig::num_classes)
      .def_rw("conf_thresh", &bcdl::AnchorDetectConfig::conf_thresh)
      .def_rw("iou_thresh", &bcdl::AnchorDetectConfig::iou_thresh)
      .def_rw("max_dets", &bcdl::AnchorDetectConfig::max_dets)
      .def_rw("strides", &bcdl::AnchorDetectConfig::strides)
      .def_rw("anchors", &bcdl::AnchorDetectConfig::anchors);

  m.def("decode_yolov5_anchor",
        [](std::vector<nb::ndarray<const float, nb::c_contig>> raw,
           const bcdl::AnchorDetectConfig& cfg, const bcdl::LetterboxInfo& lb) {
          std::vector<const float*> ptrs;
          std::vector<std::pair<int, int>> grid;
          ptrs.reserve(raw.size());
          grid.reserve(raw.size());
          for (auto& r : raw) {
            if (r.ndim() < 3) {
              throw std::invalid_argument(
                  "decode_yolov5_anchor: each raw head needs >= 3 dims "
                  "([C,H,W] or [1,C,H,W])");
            }
            ptrs.push_back(r.data());
            grid.emplace_back(static_cast<int>(r.shape(r.ndim() - 2)),
                              static_cast<int>(r.shape(r.ndim() - 1)));
          }
          return bcdl::decodeYoloV5Anchor(ptrs, grid, cfg, lb);
        },
        "raw"_a, "config"_a, "letterbox"_a,
        "Decode raw anchor-based head tensors ([1,na*(5+nc),H,W] per scale, "
        "unactivated) into Detections in original-image pixels.");

  nb::class_<bcdl::AnchorDetector>(m, "AnchorDetector")
      .def(nb::init<bcdl::Engine&, bcdl::AnchorDetectConfig, int>(), "engine"_a,
           "config"_a = bcdl::AnchorDetectConfig{}, "output_base"_a = 0,
           nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::AnchorDetector::postprocess, "letterbox"_a)
      .def_prop_ro("config", &bcdl::AnchorDetector::config);

  // ===========================================================================
  // Embeddings (retrieval, zero-shot classification)
  // ===========================================================================
  nb::class_<bcdl::EmbedConfig>(m, "EmbedConfig")
      .def(nb::init<>())
      .def_rw("l2_normalize", &bcdl::EmbedConfig::l2_normalize);

  nb::class_<bcdl::EmbedMatch>(m, "EmbedMatch")
      .def_ro("index", &bcdl::EmbedMatch::index)
      .def_ro("score", &bcdl::EmbedMatch::score)
      .def_ro("label", &bcdl::EmbedMatch::label)
      .def("__repr__", [](const bcdl::EmbedMatch& r) {
        char b[128];
        std::snprintf(b, sizeof(b), "EmbedMatch(idx=%d score=%.4f label='%s')",
                      r.index, r.score, r.label.c_str());
        return std::string(b);
      });

  m.def("decode_embedding",
        [](nb::ndarray<float, nb::c_contig> arr, const bcdl::EmbedConfig& cfg) {
          return bcdl::decodeEmbedding(arr.data(), static_cast<int>(arr.size()), cfg);
        },
        "data"_a, "config"_a = bcdl::EmbedConfig{},
        "Read a flat float array as an embedding (L2-normalized by default).");

  nb::class_<bcdl::EmbeddingBank>(m, "EmbeddingBank")
      .def(nb::init<>())
      .def("add", &bcdl::EmbeddingBank::add, "vec"_a, "label"_a = std::string(),
           "Append one entry; the vector is L2-normalized into the bank.")
      .def("search", &bcdl::EmbeddingBank::search, "query"_a, "k"_a = 5,
           "Top-k cosine matches against the bank, most similar first.")
      .def("label", &bcdl::EmbeddingBank::label, "index"_a)
      .def("__len__", &bcdl::EmbeddingBank::size)
      .def_prop_ro("dim", &bcdl::EmbeddingBank::dim);

  nb::class_<bcdl::ImageEmbedder>(m, "ImageEmbedder")
      .def(nb::init<bcdl::Engine&, bcdl::EmbedConfig, int>(), "engine"_a,
           "config"_a = bcdl::EmbedConfig{}, "output_index"_a = 0,
           nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::ImageEmbedder::postprocess)
      .def_prop_ro("dim", &bcdl::ImageEmbedder::dim)
      .def_prop_ro("config", &bcdl::ImageEmbedder::config);

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
  // ReID appearance embeddings (BoT-SORT association primitives)
  // ===========================================================================
  m.def(
      "normalize_embedding",
      [](nb::ndarray<const float, nb::c_contig> emb) {
        if (emb.ndim() != 1) {
          throw std::runtime_error("normalize_embedding: expected a 1-D [D] vector");
        }
        return bcdl::normalizeEmbedding(emb.data(), static_cast<int>(emb.shape(0)));
      },
      "embedding"_a, "L2-normalize a 1-D appearance embedding -> list[float].");
  m.def("cosine_similarity", &bcdl::cosineSimilarity, "a"_a, "b"_a,
        "Cosine similarity of two equal-length embeddings in [-1,1].");

  // ===========================================================================
  // Promptable segmentation (EdgeSAM / SAM mask decoder tail)
  // ===========================================================================
  nb::class_<bcdl::SamConfig>(m, "SamConfig")
      .def(nb::init<>())
      .def_rw("mask_threshold", &bcdl::SamConfig::mask_threshold)
      .def_rw("multimask", &bcdl::SamConfig::multimask);

  nb::class_<bcdl::SamMask>(m, "SamMask")
      .def_ro("index", &bcdl::SamMask::index)
      .def_ro("iou", &bcdl::SamMask::iou)
      .def_ro("mask_w", &bcdl::SamMask::mask_w)
      .def_ro("mask_h", &bcdl::SamMask::mask_h)
      .def_prop_ro(
          "mask",
          [](const bcdl::SamMask& sm) {
            if (sm.mask.empty()) return toNumpy(std::vector<uint8_t>{}, {0, 0});
            return toNumpy(std::vector<uint8_t>(sm.mask),
                           {static_cast<size_t>(sm.mask_h),
                            static_cast<size_t>(sm.mask_w)});
          },
          nb::rv_policy::reference,
          "Selected low-res binary mask as a uint8 (H, W) numpy array (0/1).");

  nb::class_<bcdl::SamMaskDecoder>(m, "SamMaskDecoder")
      .def(nb::init<bcdl::Engine&, bcdl::SamConfig, int, int>(), "engine"_a,
           "config"_a = bcdl::SamConfig{}, "masks_index"_a = 0, "iou_index"_a = 1,
           nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::SamMaskDecoder::postprocess)
      .def_prop_ro("config", &bcdl::SamMaskDecoder::config);

  // decode_sam_masks(): numpy path — low_res_logits [Nmask,Hm,Wm] + iou_pred
  // [Nmask]. Returns the best (argmax-IoU) low-res binary SamMask.
  m.def(
      "decode_sam_masks",
      [](nb::ndarray<const float, nb::c_contig> logits,
         nb::ndarray<const float, nb::c_contig> iou_pred, const bcdl::SamConfig& cfg) {
        if (logits.ndim() != 3) {
          throw std::runtime_error("decode_sam_masks: logits must be [Nmask,Hm,Wm]");
        }
        const int n = static_cast<int>(logits.shape(0));
        const int h = static_cast<int>(logits.shape(1));
        const int w = static_cast<int>(logits.shape(2));
        const float* ip = nullptr;
        if (iou_pred.ndim() == 1 && static_cast<int>(iou_pred.shape(0)) == n) {
          ip = iou_pred.data();
        }
        return bcdl::decodeSamMasks(logits.data(), n, h, w, ip, cfg);
      },
      "low_res_logits"_a, "iou_pred"_a, "config"_a,
      "Select+binarize the best SAM mask from [Nmask,Hm,Wm] logits + [Nmask] iou.");

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
      .def_rw("compute_masks", &bcdl::InstanceSegConfig::compute_masks)
      .def_rw("reg_max", &bcdl::InstanceSegConfig::reg_max);

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
        // Auto-detect box head format from channel count (4 => LTRB,
        // 4*reg_max => DFL) so the numpy path matches InstanceSegmenter.
        bcdl::InstanceSegConfig eff = cfg;
        const int box_ch = static_cast<int>(box[0].shape(2));
        if (box_ch == 4) {
          eff.reg_max = 0;
        } else if (box_ch % 4 == 0) {
          eff.reg_max = box_ch / 4;
        } else {
          throw std::runtime_error(
              "decode_instance_seg: box channels neither 4 nor 4*reg_max");
        }
        return bcdl::decodeInstanceSeg(cp, bp, mp, grid, nc, np, proto.data(), mH, mW,
                                       pC, eff, lb, orig_w, orig_h);
      },
      "cls"_a, "box"_a, "mc"_a, "proto"_a, "config"_a, "letterbox"_a, "orig_w"_a,
      "orig_h"_a,
      "Decode per-scale [H,W,nc]/[H,W,4|4*reg_max]/[H,W,np] + proto [mH,mW,np] -> InstanceMasks.");

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

  // --- ReID crop preprocessing ------------------------------------------------
  // Bound rather than reimplemented in numpy so there is exactly ONE definition
  // of what a ReID crop is; a Python-side copy would be free to drift from the
  // C++ TrackingPipeline's, and the two must feed the model identical pixels.
  nb::class_<bcdl::ReidConfig>(m, "ReidConfig")
      .def(nb::init<>())
      .def_prop_rw(
          "mean",
          [](const bcdl::ReidConfig& c) {
            return std::vector<float>(c.mean, c.mean + 3);
          },
          [](bcdl::ReidConfig& c, const std::vector<float>& v) {
            if (v.size() != 3) throw std::invalid_argument("mean must have 3 values (RGB)");
            std::copy(v.begin(), v.end(), c.mean);
          })
      .def_prop_rw(
          "std",
          [](const bcdl::ReidConfig& c) {
            return std::vector<float>(c.std, c.std + 3);
          },
          [](bcdl::ReidConfig& c, const std::vector<float>& v) {
            if (v.size() != 3) throw std::invalid_argument("std must have 3 values (RGB)");
            std::copy(v.begin(), v.end(), c.std);
          });

  m.def(
      "reid_crop_preprocess",
      [](nb::ndarray<const uint8_t, nb::c_contig> bgr, float x1, float y1, float x2,
         float y2, int in_w, int in_h, const bcdl::ReidConfig& cfg) {
        if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
          throw std::invalid_argument(
              "reid_crop_preprocess: expected an HxWx3 uint8 BGR image");
        }
        const int h = static_cast<int>(bgr.shape(0));
        const int w = static_cast<int>(bgr.shape(1));
        std::vector<float> out;
        bcdl::reidPreprocess(bgr.data(), w, h, w * 3, x1, y1, x2, y2, in_w, in_h, cfg,
                             out);
        auto* held = new std::vector<float>(std::move(out));
        nb::capsule owner(held, [](void* p) noexcept {
          delete static_cast<std::vector<float>*>(p);
        });
        const size_t shape[4] = {1, 3, static_cast<size_t>(in_h),
                                 static_cast<size_t>(in_w)};
        return nb::ndarray<nb::numpy, float>(held->data(), 4, shape, owner);
      },
      "bgr"_a, "x1"_a, "y1"_a, "x2"_a, "y2"_a, "in_w"_a = 128, "in_h"_a = 256,
      "config"_a = bcdl::ReidConfig{},
      "Cut a detection box out of a BGR frame into the ReID model's input: "
      "squashing resize to in_w x in_h (NOT a letterbox), BGR->RGB, /255, "
      "ImageNet z-score, NCHW float32.");

  nb::class_<bcdl::BoostConfig>(m, "BoostConfig")
      .def(nb::init<>())
      .def_rw("rich_similarity", &bcdl::BoostConfig::rich_similarity)
      .def_rw("soft_biou", &bcdl::BoostConfig::soft_biou)
      .def_rw("boost_detections", &bcdl::BoostConfig::boost_detections)
      .def_rw("lambda_iou", &bcdl::BoostConfig::lambda_iou)
      .def_rw("lambda_mhd", &bcdl::BoostConfig::lambda_mhd)
      .def_rw("lambda_shape", &bcdl::BoostConfig::lambda_shape)
      .def_rw("min_iou", &bcdl::BoostConfig::min_iou)
      .def_rw("dlo_alpha", &bcdl::BoostConfig::dlo_alpha)
      .def_rw("vt_start", &bcdl::BoostConfig::vt_start)
      .def_rw("vt_end", &bcdl::BoostConfig::vt_end)
      .def_rw("vt_steps", &bcdl::BoostConfig::vt_steps)
      .def_rw("duo", &bcdl::BoostConfig::duo)
      .def_rw("duo_iou", &bcdl::BoostConfig::duo_iou);

  nb::class_<bcdl::ByteTrackConfig>(m, "ByteTrackConfig")
      .def(nb::init<>())
      .def_rw("track_thresh", &bcdl::ByteTrackConfig::track_thresh)
      .def_rw("high_thresh", &bcdl::ByteTrackConfig::high_thresh)
      .def_rw("match_thresh", &bcdl::ByteTrackConfig::match_thresh)
      .def_rw("track_buffer", &bcdl::ByteTrackConfig::track_buffer)
      .def_rw("frame_rate", &bcdl::ByteTrackConfig::frame_rate)
      .def_rw("proximity_thresh", &bcdl::ByteTrackConfig::proximity_thresh)
      .def_rw("appearance_thresh", &bcdl::ByteTrackConfig::appearance_thresh)
      .def_rw("ema_alpha", &bcdl::ByteTrackConfig::ema_alpha)
      .def_rw("boost", &bcdl::ByteTrackConfig::boost);

  using TrackerUpdate =
      std::vector<bcdl::Track> (bcdl::ByteTracker::*)(const std::vector<bcdl::Detection>&);
  using TrackerUpdateReid = std::vector<bcdl::Track> (bcdl::ByteTracker::*)(
      const std::vector<bcdl::Detection>&, const std::vector<std::vector<float>>&);

  nb::class_<bcdl::ByteTracker>(m, "ByteTracker")
      .def(nb::init<bcdl::ByteTrackConfig>(), "config"_a = bcdl::ByteTrackConfig{})
      .def("update", static_cast<TrackerUpdate>(&bcdl::ByteTracker::update),
           "detections"_a,
           "Advance one frame with this frame's detections; returns active Tracks.")
      .def("update", static_cast<TrackerUpdateReid>(&bcdl::ByteTracker::update),
           "detections"_a, "embeddings"_a,
           "Advance one frame with detections AND their ReID embeddings (one "
           "entry per detection, parallel lists; an empty entry means 'no "
           "appearance for this detection'). Enables BoT-SORT appearance "
           "association: cost = min(IoU distance, gated cosine distance).")
      .def(
          "apply_camera_motion",
          [](bcdl::ByteTracker& t, nb::ndarray<const float, nb::c_contig> affine) {
            if (affine.size() != 6) {
              throw std::runtime_error(
                  "apply_camera_motion expects a 2x3 affine (6 floats), as "
                  "returned by cv2.estimateAffinePartial2D");
            }
            t.applyCameraMotion(affine.data());
          },
          "affine"_a,
          "Warp every tracklet by a camera-motion 2x3 affine mapping the "
          "PREVIOUS frame to this one, before the next update(). Only needed "
          "when the camera moves.")
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

  // Per-stage timing (preproc / infer / postproc), shared by the pipelines.
  // In AsyncDetectionPipeline these are per-thread SERVICE times, so the slowest
  // stage — not their sum — bounds throughput.
  nb::class_<bcdl::StageProfile>(m, "StageProfile")
      .def_ro("decode_ms", &bcdl::StageProfile::decode_ms)
      .def_ro("cvt_ms", &bcdl::StageProfile::cvt_ms)
      .def_ro("preproc_ms", &bcdl::StageProfile::preproc_ms)
      .def_ro("infer_ms", &bcdl::StageProfile::infer_ms)
      .def_ro("postproc_ms", &bcdl::StageProfile::postproc_ms)
      .def_ro("frames", &bcdl::StageProfile::frames)
      .def("total_ms", &bcdl::StageProfile::totalMs)
      .def("decode_per_frame", &bcdl::StageProfile::decodePerFrame)
      .def("cvt_per_frame", &bcdl::StageProfile::cvtPerFrame)
      .def("preproc_per_frame", &bcdl::StageProfile::preprocPerFrame)
      .def("infer_per_frame", &bcdl::StageProfile::inferPerFrame)
      .def("postproc_per_frame", &bcdl::StageProfile::postprocPerFrame);

  nb::class_<bcdl::TrackingReidConfig>(m, "TrackingReidConfig")
      .def(nb::init<>())
      .def_rw("min_score", &bcdl::TrackingReidConfig::min_score)
      .def_rw("max_crops", &bcdl::TrackingReidConfig::max_crops)
      .def_rw("crop", &bcdl::TrackingReidConfig::crop);

  nb::class_<bcdl::TrackingPipeline>(m, "TrackingPipeline")
      .def(nb::init<bcdl::Engine&, bcdl::PipelineConfig, bcdl::ByteTrackConfig>(),
           "engine"_a, "det_config"_a = bcdl::PipelineConfig{},
           "track_config"_a = bcdl::ByteTrackConfig{}, nb::keep_alive<1, 2>())
      // Appearance overload: keep_alive on BOTH engines — the pipeline holds
      // them by reference, so either being collected first is a use-after-free.
      .def(nb::init<bcdl::Engine&, bcdl::Engine&, bcdl::PipelineConfig,
                    bcdl::ByteTrackConfig, bcdl::TrackingReidConfig>(),
           "engine"_a, "reid_engine"_a, "det_config"_a = bcdl::PipelineConfig{},
           "track_config"_a = bcdl::ByteTrackConfig{},
           "reid_config"_a = bcdl::TrackingReidConfig{}, nb::keep_alive<1, 2>(),
           nb::keep_alive<1, 3>())
      .def_prop_ro("has_reid", &bcdl::TrackingPipeline::hasReid)
      .def_prop_ro("last_embed_count", &bcdl::TrackingPipeline::lastEmbedCount)
      .def(
          "process",
          [](bcdl::TrackingPipeline& p, nb::ndarray<const uint8_t, nb::c_contig> bgr) {
            if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
              throw std::runtime_error(
                  "TrackingPipeline.process expects an HxWx3 uint8 BGR array");
            }
            const int h = static_cast<int>(bgr.shape(0));
            const int w = static_cast<int>(bgr.shape(1));
            nb::gil_scoped_release rel;  // detect + (optional) re-ID + associate
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
      .def("profile", &bcdl::AsyncDetectionPipeline::profile,
           "Per-stage service timing (StageProfile); read after finish() + drain.")
      .def_prop_ro("head", &bcdl::AsyncDetectionPipeline::head);

  // --- full compressed-video -> detections pipeline (VPU decode ‖ CPU preproc ‖
  // BPU infer), all in C++ so a Python caller only pumps bytes. submit()/next()
  // RELEASE the GIL so the C++ worker threads run while Python does nothing.
  nb::class_<bcdl::AsyncVideoDetectionPipeline>(m, "AsyncVideoDetectionPipeline")
      .def(nb::init<bcdl::Engine&, bcdl::PipelineConfig, hbVPVideoType, int>(),
           "engine"_a, "config"_a = bcdl::PipelineConfig{},
           "codec"_a = HB_VP_VIDEO_TYPE_H264, "depth"_a = 4, nb::keep_alive<1, 2>())
      .def(
          "submit",
          [](bcdl::AsyncVideoDetectionPipeline& p, nb::bytes data) {
            const auto* d = reinterpret_cast<const uint8_t*>(data.c_str());
            const std::size_t n = data.size();
            nb::gil_scoped_release rel;
            return p.submit(d, n);
          },
          "data"_a,
          "Feed a chunk of Annex-B compressed bytes (segmented + VPU-decoded "
          "internally). Blocks on backpressure; False after finish().")
      .def(
          "next",
          [](bcdl::AsyncVideoDetectionPipeline& p)
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
          "Pop the next frame's detections in decode order (blocks). None once "
          "finished AND fully drained.")
      .def(
          "next_nowait",
          [](bcdl::AsyncVideoDetectionPipeline& p)
              -> std::optional<std::vector<bcdl::Detection>> {
            std::vector<bcdl::Detection> out;
            bool ok;
            {
              nb::gil_scoped_release rel;
              ok = p.tryNext(out);
            }
            if (!ok) return std::nullopt;
            return out;
          },
          "Non-blocking pop: detections if one is ready, else None. Never blocks.")
      .def("finish", &bcdl::AsyncVideoDetectionPipeline::finish,
           nb::call_guard<nb::gil_scoped_release>(),
           "Signal end of stream; drains in-flight frames then next() ends.")
      .def("profile", &bcdl::AsyncVideoDetectionPipeline::profile,
           "Per-stage timing incl. decode_ms (StageProfile); read after drain.");

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

  // ===========================================================================
  // Lidar: pillar encoding + 3-D detection
  // ===========================================================================
  nb::class_<bcdl::VoxelConfig>(m, "VoxelConfig")
      .def(nb::init<>())
      .def_rw("min_x", &bcdl::VoxelConfig::min_x)
      .def_rw("max_x", &bcdl::VoxelConfig::max_x)
      .def_rw("min_y", &bcdl::VoxelConfig::min_y)
      .def_rw("max_y", &bcdl::VoxelConfig::max_y)
      .def_rw("min_z", &bcdl::VoxelConfig::min_z)
      .def_rw("max_z", &bcdl::VoxelConfig::max_z)
      .def_rw("pillar_x", &bcdl::VoxelConfig::pillar_x)
      .def_rw("pillar_y", &bcdl::VoxelConfig::pillar_y)
      .def_rw("max_points_per_pillar", &bcdl::VoxelConfig::max_points_per_pillar)
      .def_rw("max_pillars", &bcdl::VoxelConfig::max_pillars)
      .def_rw("num_features", &bcdl::VoxelConfig::num_features)
      .def_prop_ro("grid_x", &bcdl::VoxelConfig::gridX)
      .def_prop_ro("grid_y", &bcdl::VoxelConfig::gridY);

  nb::class_<bcdl::Pillars>(m, "Pillars")
      .def_ro("num_pillars", &bcdl::Pillars::num_pillars)
      .def_ro("points_used", &bcdl::Pillars::points_used)
      .def_ro("points_out_of_range", &bcdl::Pillars::points_out_of_range)
      .def_ro("points_dropped_full", &bcdl::Pillars::points_dropped_full)
      .def_ro("pillars_dropped", &bcdl::Pillars::pillars_dropped)
      .def("__repr__", [](const bcdl::Pillars& p) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "Pillars(%d filled, %zu points used, %zu out of range, "
                      "%zu dropped full, %zu pillars dropped)",
                      p.num_pillars, p.points_used, p.points_out_of_range,
                      p.points_dropped_full, p.pillars_dropped);
        return std::string(buf);
      });

  m.def(
      "voxelize_pillars",
      [](nb::ndarray<const float, nb::c_contig> points, const bcdl::VoxelConfig& cfg) {
        if (points.ndim() != 2 || points.shape(1) < 3) {
          throw std::invalid_argument(
              "voxelize_pillars: expected an (N, >=3) float32 array of x, y, z[, ...]");
        }
        bcdl::Pillars p = bcdl::voxelizePillars(points.data(), points.shape(0),
                                                points.shape(1), cfg);
        const size_t mp = static_cast<size_t>(cfg.max_pillars);
        auto voxels = toNumpy(std::move(p.voxels),
                              {mp, static_cast<size_t>(cfg.max_points_per_pillar),
                               static_cast<size_t>(cfg.num_features)});
        auto npts = toNumpy(std::move(p.num_points), {mp});
        auto coords = toNumpy(std::move(p.coords), {mp, 4});
        p.voxels.clear();
        p.num_points.clear();
        p.coords.clear();
        return nb::make_tuple(voxels, npts, coords, p);
      },
      "points"_a, "config"_a = bcdl::VoxelConfig{},
      "Encode an ego-frame point cloud into (voxels, num_points, coords, stats). "
      "The caps are real: `stats` reports what was dropped.");

  m.def(
      "pillar_point_features",
      [](nb::ndarray<const float, nb::c_contig> voxels,
         nb::ndarray<const float, nb::c_contig> num_points,
         nb::ndarray<const float, nb::c_contig> coords, int num_pillars,
         const bcdl::VoxelConfig& cfg) {
        if (voxels.ndim() != 3 || num_points.ndim() != 1 || coords.ndim() != 2 ||
            coords.shape(1) != 4) {
          throw std::invalid_argument(
              "pillar_point_features: voxels (P, N, F), num_points (P,) and "
              "coords (P, 4) expected");
        }
        std::vector<float> feats = bcdl::pillarPointFeatures(
            voxels.data(), num_points.data(), coords.data(), num_pillars, cfg);
        return toNumpy(std::move(feats),
                       {1, static_cast<size_t>(cfg.num_features + 6),
                        static_cast<size_t>(cfg.max_pillars),
                        static_cast<size_t>(cfg.max_points_per_pillar)});
      },
      "voxels"_a, "num_points"_a, "coords"_a, "num_pillars"_a,
      "config"_a = bcdl::VoxelConfig{},
      "Expand pillar points into the (1, F+6, max_pillars, max_points) feature "
      "image a 1x1-convolution VFE takes: raw point, offset from the pillar's "
      "centroid, offset from its centre.");

  m.def(
      "scatter_pillars",
      [](nb::ndarray<const float, nb::c_contig> features,
         nb::ndarray<const float, nb::c_contig> coords, int num_pillars,
         const bcdl::VoxelConfig& cfg) {
        if (features.ndim() != 2 || coords.ndim() != 2 || coords.shape(1) != 4) {
          throw std::invalid_argument(
              "scatter_pillars: features (P, C) and coords (P, 4) expected");
        }
        const int channels = static_cast<int>(features.shape(1));
        std::vector<float> canvas = bcdl::scatterPillars(features.data(), coords.data(),
                                                         num_pillars, channels, cfg);
        return toNumpy(std::move(canvas), {1, static_cast<size_t>(channels),
                                           static_cast<size_t>(cfg.gridY()),
                                           static_cast<size_t>(cfg.gridX())});
      },
      "features"_a, "coords"_a, "num_pillars"_a, "config"_a = bcdl::VoxelConfig{},
      "Scatter (P, C) pillar features onto the dense (1, C, gridY, gridX) canvas.");

  nb::class_<bcdl::Box3D>(m, "Box3D")
      .def(nb::init<>())
      .def_rw("x", &bcdl::Box3D::x)
      .def_rw("y", &bcdl::Box3D::y)
      .def_rw("z", &bcdl::Box3D::z)
      .def_rw("dx", &bcdl::Box3D::dx)
      .def_rw("dy", &bcdl::Box3D::dy)
      .def_rw("dz", &bcdl::Box3D::dz)
      .def_rw("heading", &bcdl::Box3D::heading)
      .def_rw("label", &bcdl::Box3D::label)
      .def_rw("score", &bcdl::Box3D::score)
      .def("__repr__", [](const bcdl::Box3D& b) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "Box3D(cls=%d score=%.2f xyz=(%.2f,%.2f,%.2f) "
                      "size=(%.2f,%.2f,%.2f) yaw=%+.3f)",
                      b.label, b.score, b.x, b.y, b.z, b.dx, b.dy, b.dz, b.heading);
        return std::string(buf);
      });

  nb::class_<bcdl::Anchor3dConfig>(m, "Anchor3dConfig")
      .def(nb::init<>())
      .def_rw("min_x", &bcdl::Anchor3dConfig::min_x)
      .def_rw("max_x", &bcdl::Anchor3dConfig::max_x)
      .def_rw("min_y", &bcdl::Anchor3dConfig::min_y)
      .def_rw("max_y", &bcdl::Anchor3dConfig::max_y)
      .def_rw("grid_x", &bcdl::Anchor3dConfig::grid_x)
      .def_rw("grid_y", &bcdl::Anchor3dConfig::grid_y)
      .def_rw("sizes", &bcdl::Anchor3dConfig::sizes)
      .def_rw("centers", &bcdl::Anchor3dConfig::centers)
      .def_rw("rotations", &bcdl::Anchor3dConfig::rotations);

  nb::class_<bcdl::Det3dConfig>(m, "Det3dConfig")
      .def(nb::init<>())
      .def_rw("score_thresh", &bcdl::Det3dConfig::score_thresh)
      .def_rw("nms_iou", &bcdl::Det3dConfig::nms_iou)
      .def_rw("max_boxes", &bcdl::Det3dConfig::max_boxes)
      .def_rw("apply_sigmoid", &bcdl::Det3dConfig::apply_sigmoid)
      .def_rw("anchors", &bcdl::Det3dConfig::anchors);

  m.def(
      "generate_anchors_3d",
      [](const bcdl::Anchor3dConfig& cfg) {
        std::vector<float> a = bcdl::generateAnchors3d(cfg);
        const size_t rows = a.size() / 7;
        return toNumpy(std::move(a), {rows, 7});
      },
      "config"_a = bcdl::Anchor3dConfig{},
      "The anchor grid as (N, 7): x, y, z, dx, dy, dz, yaw.");

  m.def(
      "decode_det3d",
      [](nb::ndarray<const float, nb::c_contig> cls,
         nb::ndarray<const float, nb::c_contig> box,
         std::optional<nb::ndarray<const float, nb::c_contig>> dir,
         nb::ndarray<const float, nb::c_contig> anchors, const bcdl::Det3dConfig& cfg) {
        if (cls.ndim() != 4 || box.ndim() != 4) {
          throw std::invalid_argument("decode_det3d: cls and box must be {1,C,H,W}");
        }
        const int h = static_cast<int>(cls.shape(2)), w = static_cast<int>(cls.shape(3));
        const int num_anchors = static_cast<int>(cfg.anchors.sizes.size() *
                                                 cfg.anchors.rotations.size());
        if (num_anchors == 0 || cls.shape(1) % num_anchors != 0) {
          throw std::invalid_argument(
              "decode_det3d: class channels are not a multiple of the anchors per cell");
        }
        return bcdl::decodeDet3d(cls.data(), box.data(), dir ? dir->data() : nullptr,
                                 static_cast<int>(cls.shape(1)) / num_anchors, h, w,
                                 anchors.data(), cfg);
      },
      "cls"_a, "box"_a, "dir"_a = nb::none(), "anchors"_a,
      "config"_a = bcdl::Det3dConfig{},
      "Decode the three head tensors into Box3D list (anchor decode + BEV NMS).");

  m.def("bev_iou", &bcdl::bevIou, "a"_a, "b"_a,
        "Rotated IoU of two boxes' BEV footprints.");
  m.def("nms3d", &bcdl::nms3d, "boxes"_a, "iou_thresh"_a, "max_boxes"_a,
        "Greedy per-class NMS on BEV IoU.");

  nb::class_<bcdl::LidarDetector3d>(m, "LidarDetector3d")
      .def(nb::init<bcdl::Engine&, bcdl::Det3dConfig, int, int, int>(), "engine"_a,
           "config"_a = bcdl::Det3dConfig{}, "cls_index"_a = 0, "box_index"_a = 1,
           "dir_index"_a = 2, nb::keep_alive<1, 2>())
      .def("postprocess", &bcdl::LidarDetector3d::postprocess)
      .def_prop_ro("config", &bcdl::LidarDetector3d::config)
      .def_prop_ro(
          "anchors",
          [](const bcdl::LidarDetector3d& d) {
            return toNumpy(std::vector<float>(d.anchors()), {d.anchors().size() / 7, 7});
          },
          nb::rv_policy::reference);

  // ===========================================================================
  // Industrial anomaly detection (one-class)
  // ===========================================================================
  nb::class_<bcdl::AnomalyConfig>(m, "AnomalyConfig")
      .def(nb::init<>())
      .def_rw("out_width", &bcdl::AnomalyConfig::out_width)
      .def_rw("out_height", &bcdl::AnomalyConfig::out_height)
      .def_rw("blur_kernel", &bcdl::AnomalyConfig::blur_kernel)
      .def_rw("blur_sigma", &bcdl::AnomalyConfig::blur_sigma)
      .def_rw("threshold", &bcdl::AnomalyConfig::threshold);

  nb::class_<bcdl::AnomalyMap>(m, "AnomalyMap")
      .def(nb::init<>())
      .def_ro("width", &bcdl::AnomalyMap::width)
      .def_ro("height", &bcdl::AnomalyMap::height)
      .def_ro("score", &bcdl::AnomalyMap::score)
      .def_prop_ro(
          "data",
          [](const bcdl::AnomalyMap& a) {
            return toNumpy(std::vector<float>(a.data),
                           {static_cast<size_t>(a.height), static_cast<size_t>(a.width)});
          },
          // Owned-via-capsule ndarray; override the property default policy.
          nb::rv_policy::reference,
          "Per-pixel anomaly score as an (H, W) float32 array.")
      .def("__repr__", [](const bcdl::AnomalyMap& a) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "AnomalyMap(%dx%d, score=%.4f)", a.width, a.height,
                      a.score);
        return std::string(buf);
      });

  m.def(
      "decode_anomaly",
      [](nb::ndarray<const float, nb::c_contig> arr, const bcdl::AnomalyConfig& cfg) {
        std::vector<int> shape;
        shape.reserve(arr.ndim());
        for (size_t d = 0; d < arr.ndim(); ++d) shape.push_back(static_cast<int>(arr.shape(d)));
        return bcdl::decodeAnomaly(arr.data(), shape, cfg);
      },
      "output"_a, "config"_a = bcdl::AnomalyConfig{},
      "Upsample + smooth a patch score grid into an AnomalyMap (score = its max).");

  m.def(
      "anomaly_mask",
      [](const bcdl::AnomalyMap& map, float threshold) {
        return toNumpy(bcdl::anomalyMask(map, threshold),
                       {static_cast<size_t>(map.height), static_cast<size_t>(map.width)});
      },
      "map"_a, "threshold"_a,
      "Threshold a map into an (H, W) uint8 0/255 mask.");

  m.def(
      "anomaly_preprocess",
      [](nb::ndarray<const uint8_t, nb::c_contig> bgr, int in_w, int in_h) {
        if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
          throw std::invalid_argument("anomaly_preprocess: expected an (H, W, 3) uint8 BGR image");
        }
        std::vector<float> out;
        bcdl::anomalyPreprocess(bgr.data(), static_cast<int>(bgr.shape(1)),
                                static_cast<int>(bgr.shape(0)),
                                static_cast<int>(bgr.shape(1)) * 3, in_w, in_h, out);
        return toNumpy(std::move(out), {3, static_cast<size_t>(in_h), static_cast<size_t>(in_w)});
      },
      "bgr"_a, "in_w"_a, "in_h"_a,
      "Resize + BGR->RGB + /255 + ImageNet mean/std into the model's (3,H,W) input.");

  nb::class_<bcdl::AnomalyDetector>(m, "AnomalyDetector")
      .def(nb::init<bcdl::Engine&, bcdl::AnomalyConfig, int, int>(), "engine"_a,
           "config"_a = bcdl::AnomalyConfig{}, "output_index"_a = 0, "input_index"_a = 0,
           nb::keep_alive<1, 2>())
      .def(
          "detect",
          [](bcdl::AnomalyDetector& self, nb::ndarray<const uint8_t, nb::c_contig> bgr,
             int timeout_ms) {
            if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
              throw std::invalid_argument("detect: expected an (H, W, 3) uint8 BGR image");
            }
            return self.detect(bgr.data(), static_cast<int>(bgr.shape(1)),
                               static_cast<int>(bgr.shape(0)),
                               static_cast<int>(bgr.shape(1)) * 3, timeout_ms);
          },
          "bgr"_a, "timeout_ms"_a = 0, "One BGR image in, an AnomalyMap out.")
      .def("postprocess", &bcdl::AnomalyDetector::postprocess)
      .def_prop_ro("input_width", &bcdl::AnomalyDetector::inputWidth)
      .def_prop_ro("input_height", &bcdl::AnomalyDetector::inputHeight)
      .def_prop_ro("config", &bcdl::AnomalyDetector::config);

  // ===========================================================================
  // Dense optical flow (two frames -> per-pixel displacement)
  // ===========================================================================
  nb::class_<bcdl::FlowConfig>(m, "FlowConfig")
      .def(nb::init<>())
      .def_rw("scale_x", &bcdl::FlowConfig::scale_x)
      .def_rw("scale_y", &bcdl::FlowConfig::scale_y)
      .def_rw("channels_first", &bcdl::FlowConfig::channels_first);

  nb::class_<bcdl::FlowField>(m, "FlowField")
      .def(nb::init<>())
      .def_ro("width", &bcdl::FlowField::width)
      .def_ro("height", &bcdl::FlowField::height)
      .def_prop_ro(
          "data",
          [](const bcdl::FlowField& f) {
            return toNumpy(std::vector<float>(f.data),
                           {static_cast<size_t>(f.height), static_cast<size_t>(f.width), 2});
          },
          // Owned-via-capsule ndarray; override the property default policy.
          nb::rv_policy::reference,
          "Flow as an (H, W, 2) float32 array of (u, v) pixel displacements.");

  m.def(
      "decode_flow",
      [](nb::ndarray<const float, nb::c_contig> arr, const bcdl::FlowConfig& cfg) {
        std::vector<int> shape;
        shape.reserve(arr.ndim());
        for (size_t d = 0; d < arr.ndim(); ++d) shape.push_back(static_cast<int>(arr.shape(d)));
        return bcdl::decodeFlow(arr.data(), shape, cfg);
      },
      "output"_a, "config"_a = bcdl::FlowConfig{},
      "De-planarize a flow tensor into a FlowField, applying the pixel scale.");

  m.def(
      "flow_colorize",
      [](const bcdl::FlowField& f, float max_magnitude) {
        return toNumpy(bcdl::flowColorize(f, max_magnitude),
                       {static_cast<size_t>(f.height), static_cast<size_t>(f.width), 3});
      },
      "flow"_a, "max_magnitude"_a = 0.0f,
      "Middlebury colour wheel -> (H, W, 3) uint8 BGR. 0 normalises by the "
      "field's own 99th-percentile magnitude.");

  m.def("flow_endpoint_error", &bcdl::flowEndpointError, "a"_a, "b"_a,
        "Mean endpoint error (pixels) between two same-sized fields.");

  m.def(
      "flow_preprocess",
      [](nb::ndarray<const uint8_t, nb::c_contig> bgr, int in_w, int in_h) {
        if (bgr.ndim() != 3 || bgr.shape(2) != 3) {
          throw std::invalid_argument("flow_preprocess: expected an (H, W, 3) uint8 BGR image");
        }
        std::vector<float> out;
        bcdl::flowPreprocess(bgr.data(), static_cast<int>(bgr.shape(1)),
                             static_cast<int>(bgr.shape(0)),
                             static_cast<int>(bgr.shape(1)) * 3, in_w, in_h, out);
        return toNumpy(std::move(out), {3, static_cast<size_t>(in_h), static_cast<size_t>(in_w)});
      },
      "bgr"_a, "in_w"_a, "in_h"_a,
      "Resize one frame into the model's (3, H, W) float input. Stays BGR and "
      "0-255 — the graph normalises internally.");

  nb::class_<bcdl::OpticalFlowEstimator>(m, "OpticalFlowEstimator")
      .def(nb::init<bcdl::Engine&, bcdl::FlowConfig, int, int, int>(), "engine"_a,
           "config"_a = bcdl::FlowConfig{}, "output_index"_a = 0, "input0_index"_a = 0,
           "input1_index"_a = 1, nb::keep_alive<1, 2>())
      .def(
          "estimate",
          [](bcdl::OpticalFlowEstimator& self, nb::ndarray<const uint8_t, nb::c_contig> a,
             nb::ndarray<const uint8_t, nb::c_contig> b, int timeout_ms) {
            if (a.ndim() != 3 || a.shape(2) != 3 || b.ndim() != 3 || b.shape(2) != 3 ||
                a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1)) {
              throw std::invalid_argument(
                  "estimate: expected two same-sized (H, W, 3) uint8 BGR images");
            }
            return self.estimate(a.data(), b.data(), static_cast<int>(a.shape(1)),
                                 static_cast<int>(a.shape(0)),
                                 static_cast<int>(a.shape(1)) * 3, timeout_ms);
          },
          "frame0"_a, "frame1"_a, "timeout_ms"_a = 0,
          "Two BGR frames in, a FlowField in SOURCE pixels out.")
      .def("postprocess", &bcdl::OpticalFlowEstimator::postprocess)
      .def_prop_ro("input_width", &bcdl::OpticalFlowEstimator::inputWidth)
      .def_prop_ro("input_height", &bcdl::OpticalFlowEstimator::inputHeight)
      .def_prop_ro("config", &bcdl::OpticalFlowEstimator::config);

  // ===========================================================================
  // Speech: Kaldi fbank front end + Paraformer ASR
  // ===========================================================================
  nb::class_<bcdl::FbankConfig>(m, "FbankConfig")
      .def(nb::init<>())
      .def_rw("sample_rate", &bcdl::FbankConfig::sample_rate)
      .def_rw("num_mel_bins", &bcdl::FbankConfig::num_mel_bins)
      .def_rw("frame_length_ms", &bcdl::FbankConfig::frame_length_ms)
      .def_rw("frame_shift_ms", &bcdl::FbankConfig::frame_shift_ms)
      .def_rw("preemphasis", &bcdl::FbankConfig::preemphasis)
      .def_rw("remove_dc_offset", &bcdl::FbankConfig::remove_dc_offset)
      .def_rw("low_freq", &bcdl::FbankConfig::low_freq)
      .def_rw("high_freq", &bcdl::FbankConfig::high_freq)
      .def_rw("input_scale", &bcdl::FbankConfig::input_scale)
      .def_rw("window", &bcdl::FbankConfig::window);

  nb::class_<bcdl::Fbank>(m, "Fbank")
      .def(nb::init<const bcdl::FbankConfig&>(), "config"_a = bcdl::FbankConfig{})
      .def(
          "compute",
          [](bcdl::Fbank& self, nb::ndarray<const float, nb::ndim<1>, nb::c_contig> pcm) {
            std::vector<float> out;
            int frames;
            {
              nb::gil_scoped_release rel;
              frames = self.compute(pcm.data(), pcm.shape(0), out);
            }
            return toNumpy(std::move(out), {static_cast<size_t>(frames),
                                            static_cast<size_t>(self.config().num_mel_bins)});
          },
          "pcm"_a, "Mono float PCM -> (frames, num_mel_bins) log-mel features.")
      .def("num_frames", &bcdl::Fbank::numFrames, "samples"_a)
      .def_prop_ro("frame_length", &bcdl::Fbank::frameLength)
      .def_prop_ro("frame_shift", &bcdl::Fbank::frameShift)
      .def_prop_ro("fft_size", &bcdl::Fbank::fftSize)
      .def_prop_ro("config", &bcdl::Fbank::config);

  m.def(
      "lfr_stack",
      [](nb::ndarray<const float, nb::ndim<2>, nb::c_contig> feats, int lfr_m, int lfr_n) {
        std::vector<float> out;
        const int dim = static_cast<int>(feats.shape(1));
        const int n = bcdl::lfrStack(feats.data(), static_cast<int>(feats.shape(0)), dim,
                                     lfr_m, lfr_n, out);
        return toNumpy(std::move(out),
                       {static_cast<size_t>(n), static_cast<size_t>(lfr_m) * dim});
      },
      "feats"_a, "lfr_m"_a = 7, "lfr_n"_a = 6,
      "Low-frame-rate stacking: (T, D) -> (ceil(T / lfr_n), lfr_m * D), as FunASR.");

  nb::class_<bcdl::Cmvn>(m, "Cmvn")
      .def(nb::init<>())
      .def_prop_ro("dim", &bcdl::Cmvn::dim)
      .def_prop_ro(
          "shift", [](const bcdl::Cmvn& c) { return toNumpy(std::vector<float>(c.shift), {c.shift.size()}); },
          nb::rv_policy::reference)
      .def_prop_ro(
          "scale", [](const bcdl::Cmvn& c) { return toNumpy(std::vector<float>(c.scale), {c.scale.size()}); },
          nb::rv_policy::reference)
      .def(
          "apply",
          [](const bcdl::Cmvn& c, nb::ndarray<const float, nb::ndim<2>, nb::c_contig> feats) {
            if (static_cast<int>(feats.shape(1)) != c.dim())
              throw std::invalid_argument("Cmvn.apply: feature dim does not match the CMVN");
            std::vector<float> out(feats.data(), feats.data() + feats.size());
            c.apply(out.data(), static_cast<int>(feats.shape(0)));
            return toNumpy(std::move(out), {feats.shape(0), feats.shape(1)});
          },
          "feats"_a, "(x + shift) * scale, per dimension; returns a new array.");
  m.def("parse_kaldi_cmvn", &bcdl::parseKaldiCmvn, "text"_a,
        "Parse an am.mvn (<AddShift> / <Rescale>) into a Cmvn.");
  m.def("load_kaldi_cmvn", &bcdl::loadKaldiCmvn, "path"_a);

  nb::class_<bcdl::CifConfig>(m, "CifConfig")
      .def(nb::init<>())
      .def_rw("threshold", &bcdl::CifConfig::threshold)
      .def_rw("tail_threshold", &bcdl::CifConfig::tail_threshold)
      .def_rw("max_tokens", &bcdl::CifConfig::max_tokens);

  m.def(
      "cif_integrate",
      [](nb::ndarray<const float, nb::ndim<1>, nb::c_contig> alphas,
         nb::ndarray<const float, nb::ndim<2>, nb::c_contig> hidden, int valid_frames,
         const bcdl::CifConfig& cfg) {
        if (hidden.shape(0) != alphas.shape(0))
          throw std::invalid_argument("cif_integrate: alphas and hidden differ in frame count");
        std::vector<float> emb;
        const int dim = static_cast<int>(hidden.shape(1));
        const int n = bcdl::cifIntegrate(alphas.data(), hidden.data(),
                                         static_cast<int>(alphas.shape(0)), dim, valid_frames,
                                         cfg, emb);
        return nb::make_tuple(
            toNumpy(std::move(emb), {static_cast<size_t>(cfg.max_tokens), static_cast<size_t>(dim)}),
            n);
      },
      "alphas"_a, "hidden"_a, "valid_frames"_a, "config"_a = bcdl::CifConfig{},
      "Continuous integrate-and-fire -> ((max_tokens, D) embeddings, token count).");

  m.def("load_token_list", &bcdl::loadTokenList, "path"_a,
        "Token list from a JSON array (tokens.json) or one token per line.");
  m.def(
      "argmax_tokens",
      [](nb::ndarray<const float, nb::ndim<2>, nb::c_contig> logits) {
        return bcdl::argmaxTokens(logits.data(), static_cast<int>(logits.shape(0)),
                                  static_cast<int>(logits.shape(1)));
      },
      "logits"_a, "Row-wise argmax of (rows, vocab) logits.");
  m.def("tokens_to_text", &bcdl::tokensToText, "ids"_a, "vocab"_a,
        "Join token pieces: drop <specials>, glue '@@' pieces, space Latin words.");

  nb::class_<bcdl::AsrResult>(m, "AsrResult")
      .def(nb::init<>())
      .def_ro("text", &bcdl::AsrResult::text)
      .def_ro("token_ids", &bcdl::AsrResult::token_ids)
      .def_ro("feature_frames", &bcdl::AsrResult::feature_frames)
      .def_ro("truncated", &bcdl::AsrResult::truncated)
      .def("__repr__", [](const bcdl::AsrResult& r) {
        return "AsrResult(text='" + r.text + "', tokens=" + std::to_string(r.token_ids.size()) +
               (r.truncated ? ", truncated" : "") + ")";
      });

  nb::class_<bcdl::ParaformerConfig>(m, "ParaformerConfig")
      .def(nb::init<>())
      .def_rw("fbank", &bcdl::ParaformerConfig::fbank)
      .def_rw("lfr_m", &bcdl::ParaformerConfig::lfr_m)
      .def_rw("lfr_n", &bcdl::ParaformerConfig::lfr_n)
      .def_rw("cif", &bcdl::ParaformerConfig::cif);

  nb::class_<bcdl::ParaformerAsr>(m, "ParaformerAsr")
      .def(nb::init<bcdl::Engine&, bcdl::Engine&, bcdl::Engine&, std::vector<std::string>,
                    bcdl::Cmvn, bcdl::ParaformerConfig>(),
           "encoder"_a, "predictor"_a, "decoder"_a, "vocab"_a, "cmvn"_a,
           "config"_a = bcdl::ParaformerConfig{}, nb::keep_alive<1, 2>(),
           nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
      .def(
          "transcribe",
          [](bcdl::ParaformerAsr& self, nb::ndarray<const float, nb::ndim<1>, nb::c_contig> pcm,
             int timeout_ms) {
            nb::gil_scoped_release rel;  // front end + three BPU stages + CIF
            return self.transcribe(pcm.data(), pcm.shape(0), timeout_ms);
          },
          "pcm"_a, "timeout_ms"_a = 0, "One mono 16 kHz float utterance -> AsrResult.")
      .def(
          "transcribe_features",
          [](bcdl::ParaformerAsr& self, nb::ndarray<const float, nb::ndim<2>, nb::c_contig> f,
             int timeout_ms) {
            if (static_cast<int>(f.shape(1)) != self.featureDim())
              throw std::invalid_argument("transcribe_features: feature dim does not match the encoder");
            nb::gil_scoped_release rel;
            return self.transcribeFeatures(f.data(), static_cast<int>(f.shape(0)), timeout_ms);
          },
          "feats"_a, "timeout_ms"_a = 0,
          "(frames, feature_dim) LFR+CMVN features -> AsrResult.")
      .def(
          "features",
          [](bcdl::ParaformerAsr& self, nb::ndarray<const float, nb::ndim<1>, nb::c_contig> pcm) {
            std::vector<float> out;
            int n;
            {
              nb::gil_scoped_release rel;
              n = self.features(pcm.data(), pcm.shape(0), out);
            }
            return toNumpy(std::move(out),
                           {static_cast<size_t>(n), static_cast<size_t>(self.featureDim())});
          },
          "pcm"_a, "Front end only: PCM -> (frames, feature_dim) normalised features.")
      .def_prop_ro("max_frames", &bcdl::ParaformerAsr::maxFrames)
      .def_prop_ro("max_seconds", &bcdl::ParaformerAsr::maxSeconds)
      .def_prop_ro("feature_dim", &bcdl::ParaformerAsr::featureDim)
      .def_prop_ro("max_tokens", &bcdl::ParaformerAsr::maxTokens)
      .def_prop_ro("vocab_size", &bcdl::ParaformerAsr::vocabSize)
      .def_prop_ro("config", &bcdl::ParaformerAsr::config);

  m.def("to_pinyin", &bcdl::toPinyin, "text"_a,
        "Toneless pinyin of a UTF-8 string, one token per syllable.");
  nb::class_<bcdl::CommandMatch>(m, "CommandMatch")
      .def(nb::init<>())
      .def_ro("index", &bcdl::CommandMatch::index)
      .def_ro("nearest", &bcdl::CommandMatch::nearest)
      .def_ro("distance", &bcdl::CommandMatch::distance)
      .def_ro("command", &bcdl::CommandMatch::command)
      .def("__bool__", [](const bcdl::CommandMatch& r) { return r.index >= 0; })
      .def("__repr__", [](const bcdl::CommandMatch& r) {
        return "CommandMatch(index=" + std::to_string(r.index) + ", command='" + r.command +
               "', nearest=" + std::to_string(r.nearest) +
               ", distance=" + std::to_string(r.distance) + ")";
      });
  nb::class_<bcdl::CommandMatcher>(m, "CommandMatcher")
      .def(nb::init<std::vector<std::string>, float>(), "commands"_a, "max_distance"_a = 0.5f)
      .def("match", &bcdl::CommandMatcher::match, "text"_a,
           "Recogniser text -> CommandMatch (index -1 = not a command).")
      .def("distances", &bcdl::CommandMatcher::distances, "text"_a,
           "Normalised pinyin edit distance to every command, in list order.")
      .def_prop_ro("commands", &bcdl::CommandMatcher::commands)
      .def_prop_ro("max_distance", &bcdl::CommandMatcher::maxDistance)
      .def("__len__", &bcdl::CommandMatcher::size);

  // Speech synthesis back end: iSTFT + Vocos
  nb::class_<bcdl::IstftConfig>(m, "IstftConfig")
      .def(nb::init<>())
      .def_rw("n_fft", &bcdl::IstftConfig::n_fft)
      .def_rw("hop_length", &bcdl::IstftConfig::hop_length)
      .def_rw("center", &bcdl::IstftConfig::center);
  nb::class_<bcdl::Istft>(m, "Istft")
      .def(nb::init<const bcdl::IstftConfig&>(), "config"_a = bcdl::IstftConfig{})
      .def(
          "compute",
          [](bcdl::Istft& self, nb::ndarray<const float, nb::ndim<2>, nb::c_contig> real,
             nb::ndarray<const float, nb::ndim<2>, nb::c_contig> imag) {
            if (real.shape(0) != imag.shape(0) || real.shape(1) != imag.shape(1) ||
                static_cast<int>(real.shape(1)) != self.bins())
              throw std::invalid_argument("Istft.compute: real and imag must both be (frames, " +
                                          std::to_string(self.bins()) + ")");
            std::vector<float> out;
            {
              nb::gil_scoped_release rel;
              self.compute(real.data(), imag.data(), static_cast<int>(real.shape(0)), out);
            }
            const size_t n = out.size();
            return toNumpy(std::move(out), {n});
          },
          "real"_a, "imag"_a, "(frames, n_fft/2+1) real and imaginary parts -> waveform.")
      .def(
          "compute_polar",
          [](bcdl::Istft& self, nb::ndarray<const float, nb::ndim<2>, nb::c_contig> mag,
             nb::ndarray<const float, nb::ndim<2>, nb::c_contig> cos,
             nb::ndarray<const float, nb::ndim<2>, nb::c_contig> sin) {
            const auto ok = [&](const auto& a) {
              return a.shape(0) == mag.shape(0) && a.shape(1) == mag.shape(1);
            };
            if (static_cast<int>(mag.shape(0)) != self.bins() || !ok(cos) || !ok(sin))
              throw std::invalid_argument("Istft.compute_polar: mag, cos, sin must all be (" +
                                          std::to_string(self.bins()) + ", frames)");
            std::vector<float> out;
            {
              nb::gil_scoped_release rel;
              self.computePolar(mag.data(), cos.data(), sin.data(),
                                static_cast<int>(mag.shape(1)), out);
            }
            const size_t n = out.size();
            return toNumpy(std::move(out), {n});
          },
          "mag"_a, "cos"_a, "sin"_a,
          "(n_fft/2+1, frames) magnitude and phase cos/sin (Vocos layout) -> waveform.")
      .def("num_samples", &bcdl::Istft::numSamples, "frames"_a)
      .def_prop_ro("bins", &bcdl::Istft::bins)
      .def_prop_ro("config", &bcdl::Istft::config);
  nb::class_<bcdl::VocosConfig>(m, "VocosConfig")
      .def(nb::init<>())
      .def_rw("stft", &bcdl::VocosConfig::stft)
      .def_rw("context", &bcdl::VocosConfig::context)
      .def_rw("pad_value", &bcdl::VocosConfig::pad_value)
      .def_rw("max_magnitude", &bcdl::VocosConfig::max_magnitude)
      .def_rw("sample_rate", &bcdl::VocosConfig::sample_rate);
  nb::class_<bcdl::VocosVocoder>(m, "VocosVocoder")
      .def(nb::init<bcdl::Engine&, bcdl::VocosConfig>(), "engine"_a,
           "config"_a = bcdl::VocosConfig{}, nb::keep_alive<1, 2>())
      .def(
          "synthesize",
          [](bcdl::VocosVocoder& self, nb::ndarray<const float, nb::ndim<2>, nb::c_contig> mel,
             int timeout_ms) {
            if (static_cast<int>(mel.shape(0)) != self.numMels())
              throw std::invalid_argument("VocosVocoder.synthesize: mel must be (" +
                                          std::to_string(self.numMels()) + ", frames)");
            std::vector<float> out;
            {
              nb::gil_scoped_release rel;
              self.synthesize(mel.data(), static_cast<int>(mel.shape(1)), out, timeout_ms);
            }
            const size_t n = out.size();
            return toNumpy(std::move(out), {n});
          },
          "mel"_a, "timeout_ms"_a = 0, "(num_mels, frames) log-mel -> float waveform.")
      .def_prop_ro("num_mels", &bcdl::VocosVocoder::numMels)
      .def_prop_ro("chunk_frames", &bcdl::VocosVocoder::chunkFrames)
      .def_prop_ro("body_frames", &bcdl::VocosVocoder::bodyFrames)
      .def_prop_ro("sample_rate", &bcdl::VocosVocoder::sampleRate)
      .def_prop_ro("config", &bcdl::VocosVocoder::config);

  // Voice activity detection: Silero VAD + streaming segmenter
  nb::class_<bcdl::SileroVad>(m, "SileroVad")
      .def(nb::init<bcdl::Engine&>(), "engine"_a, nb::keep_alive<1, 2>())
      .def(
          "probability",
          [](bcdl::SileroVad& self, nb::ndarray<const float, nb::ndim<1>, nb::c_contig> chunk,
             int timeout_ms) {
            if (chunk.shape(0) != static_cast<size_t>(bcdl::SileroVad::kChunk))
              throw std::invalid_argument("SileroVad.probability: chunk must be " +
                                          std::to_string(bcdl::SileroVad::kChunk) +
                                          " samples");
            nb::gil_scoped_release rel;
            return self.probability(chunk.data(), timeout_ms);
          },
          "chunk"_a, "timeout_ms"_a = 0,
          "Speech probability of the next 512 samples (16 kHz float PCM); stateful.")
      .def("reset", &bcdl::SileroVad::reset)
      .def_prop_ro_static("chunk", [](nb::handle) { return bcdl::SileroVad::kChunk; })
      .def_prop_ro_static("sample_rate", [](nb::handle) { return bcdl::SileroVad::kSampleRate; });
  nb::class_<bcdl::VadConfig>(m, "VadConfig")
      .def(nb::init<>())
      .def_rw("threshold", &bcdl::VadConfig::threshold)
      .def_rw("neg_threshold", &bcdl::VadConfig::neg_threshold)
      .def_rw("min_speech_ms", &bcdl::VadConfig::min_speech_ms)
      .def_rw("min_silence_ms", &bcdl::VadConfig::min_silence_ms)
      .def_rw("speech_pad_ms", &bcdl::VadConfig::speech_pad_ms)
      .def_rw("max_speech_s", &bcdl::VadConfig::max_speech_s);
  nb::class_<bcdl::VadSegment>(m, "VadSegment")
      .def_ro("start", &bcdl::VadSegment::start)
      .def_ro("end", &bcdl::VadSegment::end)
      .def_prop_ro("pcm",
                   [](const bcdl::VadSegment& s) {
                     return toNumpy(std::vector<float>(s.pcm), {s.pcm.size()});
                   },
                   nb::rv_policy::reference);
  nb::class_<bcdl::VadSegmenter>(m, "VadSegmenter")
      .def(nb::init<bcdl::SileroVad&, bcdl::VadConfig>(), "vad"_a,
           "config"_a = bcdl::VadConfig{}, nb::keep_alive<1, 2>())
      .def(
          "feed",
          [](bcdl::VadSegmenter& self, nb::ndarray<const float, nb::ndim<1>, nb::c_contig> pcm,
             int timeout_ms) {
            nb::gil_scoped_release rel;
            return self.feed(pcm.data(), pcm.shape(0), timeout_ms);
          },
          "pcm"_a, "timeout_ms"_a = 0, "Float PCM of any length -> utterances that closed.")
      .def("flush", &bcdl::VadSegmenter::flush)
      .def(
          "segments",
          [](bcdl::VadSegmenter& self, nb::ndarray<const float, nb::ndim<1>, nb::c_contig> pcm,
             int timeout_ms) {
            nb::gil_scoped_release rel;
            return self.segments(pcm.data(), pcm.shape(0), timeout_ms);
          },
          "pcm"_a, "timeout_ms"_a = 0, "Whole recording -> its utterances (from a reset).")
      .def("reset", &bcdl::VadSegmenter::reset)
      .def_prop_ro("in_speech", &bcdl::VadSegmenter::inSpeech)
      .def_prop_ro("last_probability", &bcdl::VadSegmenter::lastProbability)
      .def_prop_ro("config", &bcdl::VadSegmenter::config);

  // Keyword spotting: streaming zipformer transducer
  nb::class_<bcdl::KwsConfig>(m, "KwsConfig")
      .def(nb::init<>())
      .def_rw("keywords_score", &bcdl::KwsConfig::keywords_score)
      .def_rw("keywords_threshold", &bcdl::KwsConfig::keywords_threshold)
      .def_rw("num_trailing_blanks", &bcdl::KwsConfig::num_trailing_blanks)
      .def_rw("max_active_paths", &bcdl::KwsConfig::max_active_paths);
  nb::class_<bcdl::KwsDetection>(m, "KwsDetection")
      .def_ro("keyword", &bcdl::KwsDetection::keyword)
      .def_ro("index", &bcdl::KwsDetection::index)
      .def_ro("start", &bcdl::KwsDetection::start)
      .def_ro("end", &bcdl::KwsDetection::end)
      .def_ro("tokens", &bcdl::KwsDetection::tokens)
      .def("__repr__", [](const bcdl::KwsDetection& d) {
        return "KwsDetection('" + d.keyword + "', " + std::to_string(d.start) + "-" +
               std::to_string(d.end) + " s)";
      });
  nb::class_<bcdl::KeywordSpotter>(m, "KeywordSpotter")
      .def(nb::init<bcdl::Engine&, const std::string&, const std::string&, bcdl::KwsConfig>(),
           "encoder"_a, "decoder_joiner"_a, "tokens"_a, "config"_a = bcdl::KwsConfig{},
           nb::keep_alive<1, 2>())
      .def("set_keywords", &bcdl::KeywordSpotter::setKeywords, "text"_a)
      .def("load_keywords", &bcdl::KeywordSpotter::loadKeywords, "path"_a)
      .def_prop_ro("keywords", &bcdl::KeywordSpotter::keywords)
      .def(
          "feed",
          [](bcdl::KeywordSpotter& self, nb::ndarray<const float, nb::ndim<1>, nb::c_contig> pcm,
             int timeout_ms) {
            nb::gil_scoped_release rel;
            return self.feed(pcm.data(), pcm.shape(0), timeout_ms);
          },
          "pcm"_a, "timeout_ms"_a = 0, "16 kHz float PCM of any length -> detections.")
      .def(
          "encode",
          [](bcdl::KeywordSpotter& self, nb::ndarray<const float, nb::ndim<1>, nb::c_contig> pcm,
             int timeout_ms) {
            int frames = 0;
            std::vector<float> e;
            {
              nb::gil_scoped_release rel;
              e = self.encode(pcm.data(), pcm.shape(0), frames, timeout_ms);
            }
            const size_t d = frames ? e.size() / frames : 0;
            return toNumpy(std::move(e), {static_cast<size_t>(frames), d});
          },
          "pcm"_a, "timeout_ms"_a = 0, "Encoder output (frames, D) of a whole recording, from a reset.")
      .def(
          "flush",
          [](bcdl::KeywordSpotter& self, int timeout_ms) {
            nb::gil_scoped_release rel;
            return self.flush(timeout_ms);
          },
          "timeout_ms"_a = 0, "End of stream: what still fires, then reset.")
      .def("reset", &bcdl::KeywordSpotter::reset)
      .def("token_id", &bcdl::KeywordSpotter::tokenId, "token"_a)
      .def_prop_ro("vocab_size", &bcdl::KeywordSpotter::vocabSize)
      .def_prop_ro("config", &bcdl::KeywordSpotter::config);

  // Speech synthesis: Matcha-TTS front end + acoustic model, and the whole chain
  nb::class_<bcdl::MatchaFrontendConfig>(m, "MatchaFrontendConfig")
      .def(nb::init<>())
      .def_rw("min_words", &bcdl::MatchaFrontendConfig::min_words)
      .def_rw("max_words", &bcdl::MatchaFrontendConfig::max_words);
  nb::class_<bcdl::MatchaFrontend>(m, "MatchaFrontend")
      .def(nb::init<const std::string&, const std::string&, bcdl::MatchaFrontendConfig>(),
           "tokens"_a, "lexicon"_a, "config"_a = bcdl::MatchaFrontendConfig{})
      .def("encode", &bcdl::MatchaFrontend::encode, "text"_a,
           "Text -> one token-id list per sentence.")
      .def("sentences", &bcdl::MatchaFrontend::sentences, "text"_a)
      .def_static("read_numbers", &bcdl::MatchaFrontend::readNumbers, "text"_a)
      .def_prop_ro("vocab_size", &bcdl::MatchaFrontend::vocabSize)
      .def_prop_ro("lexicon_size", &bcdl::MatchaFrontend::lexiconSize);
  nb::class_<bcdl::MatchaConfig>(m, "MatchaConfig")
      .def(nb::init<>())
      .def_rw("noise_scale", &bcdl::MatchaConfig::noise_scale)
      .def_rw("length_scale", &bcdl::MatchaConfig::length_scale)
      .def_rw("seed", &bcdl::MatchaConfig::seed);
  nb::class_<bcdl::MatchaResult>(m, "MatchaResult")
      .def_prop_ro("mel",
                   [](const bcdl::MatchaResult& r) {
                     const size_t mels = r.frames ? r.mel.size() / r.frames : 0;
                     return toNumpy(std::vector<float>(r.mel),
                                    {mels, static_cast<size_t>(r.frames)});
                   },
                   nb::rv_policy::reference)
      .def_ro("frames", &bcdl::MatchaResult::frames)
      .def_ro("durations", &bcdl::MatchaResult::durations)
      .def_ro("truncated", &bcdl::MatchaResult::truncated);
  nb::class_<bcdl::MatchaAcoustic>(m, "MatchaAcoustic")
      .def(nb::init<bcdl::Engine&, bcdl::Engine&, const std::string&, bcdl::MatchaConfig>(),
           "encoder"_a, "decoder"_a, "embedding"_a, "config"_a = bcdl::MatchaConfig{},
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
      .def(
          "synthesize",
          [](bcdl::MatchaAcoustic& self, const std::vector<int>& ids, int timeout_ms) {
            nb::gil_scoped_release rel;
            return self.synthesize(ids.data(), static_cast<int>(ids.size()), timeout_ms);
          },
          "ids"_a, "timeout_ms"_a = 0, "One sentence's token ids -> MatchaResult.")
      .def("seed", &bcdl::MatchaAcoustic::seed, "seed"_a)
      .def_prop_ro("max_tokens", &bcdl::MatchaAcoustic::maxTokens)
      .def_prop_ro("max_frames", &bcdl::MatchaAcoustic::maxFrames)
      .def_prop_ro("num_mels", &bcdl::MatchaAcoustic::numMels)
      .def_prop_ro("vocab_size", &bcdl::MatchaAcoustic::vocabSize)
      .def_prop_ro("config", &bcdl::MatchaAcoustic::config);
  nb::class_<bcdl::TtsResult>(m, "TtsResult")
      .def_prop_ro("pcm",
                   [](const bcdl::TtsResult& r) {
                     return toNumpy(std::vector<float>(r.pcm), {r.pcm.size()});
                   },
                   nb::rv_policy::reference)
      .def_ro("sample_rate", &bcdl::TtsResult::sample_rate)
      .def_ro("sentences", &bcdl::TtsResult::sentences)
      .def_ro("truncated", &bcdl::TtsResult::truncated);
  m.def(
      "scale_silence",
      [](nb::ndarray<const float, nb::ndim<1>, nb::c_contig> pcm, int sample_rate, float scale,
         float min_pause_s, float level) {
        std::vector<float> in(pcm.data(), pcm.data() + pcm.shape(0));
        std::vector<float> out = bcdl::scaleSilence(in, sample_rate, scale, min_pause_s, level);
        const size_t n = out.size();
        return toNumpy(std::move(out), {n});
      },
      "pcm"_a, "sample_rate"_a, "scale"_a, "min_pause_s"_a = 0.2f, "level"_a = 0.01f,
      "Shorten every pause (|x| <= level for >= min_pause_s) to scale of its length.");
  nb::class_<bcdl::TextToSpeechConfig>(m, "TextToSpeechConfig")
      .def(nb::init<>())
      .def_rw("silence_scale", &bcdl::TextToSpeechConfig::silence_scale);
  nb::class_<bcdl::TextToSpeech>(m, "TextToSpeech")
      .def(nb::init<bcdl::MatchaFrontend, bcdl::MatchaAcoustic&, bcdl::VocosVocoder&,
                    bcdl::TextToSpeechConfig>(),
           "frontend"_a, "acoustic"_a, "vocoder"_a, "config"_a = bcdl::TextToSpeechConfig{},
           nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>())
      .def(
          "synthesize",
          [](bcdl::TextToSpeech& self, const std::string& text, int timeout_ms) {
            nb::gil_scoped_release rel;
            return self.synthesize(text, timeout_ms);
          },
          "text"_a, "timeout_ms"_a = 0, "Text -> TtsResult (16 kHz float PCM).")
      .def_prop_ro("frontend", &bcdl::TextToSpeech::frontend)
      .def_prop_ro("config", &bcdl::TextToSpeech::config);

  // ===========================================================================
  // End-to-end driving: camera + lidar -> planned trajectory
  // ===========================================================================
  nb::class_<bcdl::LidarBevConfig>(m, "LidarBevConfig")
      .def(nb::init<>())
      .def_rw("min_x", &bcdl::LidarBevConfig::min_x)
      .def_rw("max_x", &bcdl::LidarBevConfig::max_x)
      .def_rw("min_y", &bcdl::LidarBevConfig::min_y)
      .def_rw("max_y", &bcdl::LidarBevConfig::max_y)
      .def_rw("pixels_per_meter", &bcdl::LidarBevConfig::pixels_per_meter)
      .def_rw("max_height", &bcdl::LidarBevConfig::max_height)
      .def_rw("split_height", &bcdl::LidarBevConfig::split_height)
      .def_rw("hist_max_per_pixel", &bcdl::LidarBevConfig::hist_max_per_pixel)
      .def_rw("use_ground_plane", &bcdl::LidarBevConfig::use_ground_plane);

  m.def(
      "lidar_bev_histogram",
      [](nb::ndarray<const float, nb::c_contig> points, const bcdl::LidarBevConfig& cfg) {
        if (points.ndim() != 2 || points.shape(1) < 3) {
          throw std::invalid_argument(
              "lidar_bev_histogram: expected an (N, >=3) float32 array of x, y, z");
        }
        bcdl::LidarBev bev = bcdl::lidarBevHistogram(
            points.data(), points.shape(0), points.shape(1), cfg);
        return toNumpy(std::move(bev.data), {static_cast<size_t>(bev.channels),
                                             static_cast<size_t>(bev.height),
                                             static_cast<size_t>(bev.width)});
      },
      "points"_a, "config"_a = bcdl::LidarBevConfig{},
      "Rasterize an (N,>=3) ego-frame lidar sweep into the model's (C,H,W) BEV "
      "occupancy histogram. Axis 0 of the grid is x (forward), axis 1 is y.");

  nb::class_<bcdl::DriveCameraConfig>(m, "DriveCameraConfig")
      .def(nb::init<>())
      .def_rw("crop_top", &bcdl::DriveCameraConfig::crop_top)
      .def_rw("crop_bottom", &bcdl::DriveCameraConfig::crop_bottom)
      .def_rw("crop_side", &bcdl::DriveCameraConfig::crop_side)
      .def_rw("out_width", &bcdl::DriveCameraConfig::out_width)
      .def_rw("out_height", &bcdl::DriveCameraConfig::out_height);

  m.def(
      "stitch_drive_cameras",
      [](nb::ndarray<const uint8_t, nb::c_contig> left,
         nb::ndarray<const uint8_t, nb::c_contig> front,
         nb::ndarray<const uint8_t, nb::c_contig> right,
         const bcdl::DriveCameraConfig& cfg) {
        auto check = [](const nb::ndarray<const uint8_t, nb::c_contig>& a, const char* who) {
          if (a.ndim() != 3 || a.shape(2) != 3) {
            throw std::invalid_argument(std::string("stitch_drive_cameras: ") + who +
                                        " must be an (H, W, 3) uint8 BGR image");
          }
        };
        check(left, "left");
        check(front, "front");
        check(right, "right");
        std::vector<float> out = bcdl::stitchDriveCameras(
            left.data(), static_cast<int>(left.shape(1)), static_cast<int>(left.shape(0)),
            static_cast<int>(left.shape(1)) * 3, front.data(),
            static_cast<int>(front.shape(1)), static_cast<int>(front.shape(0)),
            static_cast<int>(front.shape(1)) * 3, right.data(),
            static_cast<int>(right.shape(1)), static_cast<int>(right.shape(0)),
            static_cast<int>(right.shape(1)) * 3, cfg);
        return toNumpy(std::move(out), {3, static_cast<size_t>(cfg.out_height),
                                        static_cast<size_t>(cfg.out_width)});
      },
      "left"_a, "front"_a, "right"_a, "config"_a = bcdl::DriveCameraConfig{},
      "Crop + stitch three BGR cameras into the model's (3, H, W) RGB float "
      "canvas scaled to [0,1].");

  nb::class_<bcdl::DrivePose>(m, "DrivePose")
      .def_ro("x", &bcdl::DrivePose::x)
      .def_ro("y", &bcdl::DrivePose::y)
      .def_ro("heading", &bcdl::DrivePose::heading)
      .def("__repr__", [](const bcdl::DrivePose& p) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "DrivePose(x=%.2f, y=%.2f, heading=%.3f)", p.x, p.y,
                      p.heading);
        return std::string(buf);
      });

  nb::class_<bcdl::DriveAgent>(m, "DriveAgent")
      .def_ro("x", &bcdl::DriveAgent::x)
      .def_ro("y", &bcdl::DriveAgent::y)
      .def_ro("heading", &bcdl::DriveAgent::heading)
      .def_ro("length", &bcdl::DriveAgent::length)
      .def_ro("width", &bcdl::DriveAgent::width)
      .def_ro("score", &bcdl::DriveAgent::score)
      .def("__repr__", [](const bcdl::DriveAgent& a) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "DriveAgent(x=%.2f, y=%.2f, heading=%.3f, l=%.2f, w=%.2f, score=%.2f)",
                      a.x, a.y, a.heading, a.length, a.width, a.score);
        return std::string(buf);
      });

  nb::class_<bcdl::DriveResult>(m, "DriveResult")
      .def_ro("trajectory", &bcdl::DriveResult::trajectory)
      .def_ro("agents", &bcdl::DriveResult::agents)
      .def_ro("bev", &bcdl::DriveResult::bev);

  nb::class_<bcdl::DriveConfig>(m, "DriveConfig")
      .def(nb::init<>())
      .def_rw("agent_score_thresh", &bcdl::DriveConfig::agent_score_thresh)
      .def_rw("bev", &bcdl::DriveConfig::bev);

  nb::class_<bcdl::DriveStatus>(m, "DriveStatus")
      .def(nb::init<>())
      .def_rw("driving_command", &bcdl::DriveStatus::driving_command)
      .def_rw("velocity_x", &bcdl::DriveStatus::velocity_x)
      .def_rw("velocity_y", &bcdl::DriveStatus::velocity_y)
      .def_rw("accel_x", &bcdl::DriveStatus::accel_x)
      .def_rw("accel_y", &bcdl::DriveStatus::accel_y);

  nb::class_<bcdl::DriveAnchorConfig>(m, "DriveAnchorConfig")
      .def(nb::init<>())
      .def_rw("x_offset", &bcdl::DriveAnchorConfig::x_offset)
      .def_rw("x_span", &bcdl::DriveAnchorConfig::x_span)
      .def_rw("y_offset", &bcdl::DriveAnchorConfig::y_offset)
      .def_rw("y_span", &bcdl::DriveAnchorConfig::y_span)
      .def_rw("truncated_timestep", &bcdl::DriveAnchorConfig::truncated_timestep)
      .def_rw("num_train_timesteps", &bcdl::DriveAnchorConfig::num_train_timesteps)
      .def_rw("beta_start", &bcdl::DriveAnchorConfig::beta_start)
      .def_rw("beta_end", &bcdl::DriveAnchorConfig::beta_end)
      .def_rw("noise_scale", &bcdl::DriveAnchorConfig::noise_scale)
      .def_rw("seed", &bcdl::DriveAnchorConfig::seed);

  nb::class_<bcdl::DriveIoConfig>(m, "DriveIoConfig")
      .def(nb::init<>())
      .def_rw("camera_input", &bcdl::DriveIoConfig::camera_input)
      .def_rw("lidar_input", &bcdl::DriveIoConfig::lidar_input)
      .def_rw("status_input", &bcdl::DriveIoConfig::status_input)
      .def_rw("noise_input", &bcdl::DriveIoConfig::noise_input)
      .def_rw("trajectory_output", &bcdl::DriveIoConfig::trajectory_output)
      .def_rw("agent_states_output", &bcdl::DriveIoConfig::agent_states_output)
      .def_rw("agent_labels_output", &bcdl::DriveIoConfig::agent_labels_output)
      .def_rw("bev_output", &bcdl::DriveIoConfig::bev_output);

  m.def(
      "drive_status_vector",
      [](const bcdl::DriveStatus& status, int num_commands) {
        return toNumpy(bcdl::driveStatusVector(status, num_commands),
                       {static_cast<size_t>(num_commands) + 4});
      },
      "status"_a, "num_commands"_a = 4,
      "Pack a DriveStatus into the model's status vector (one-hot command + "
      "vx, vy, ax, ay).");

  m.def(
      "drive_anchor_noise",
      [](nb::ndarray<const float, nb::c_contig> anchors,
         const bcdl::DriveAnchorConfig& cfg) {
        if (anchors.ndim() != 3 || anchors.shape(2) != 2) {
          throw std::invalid_argument(
              "drive_anchor_noise: expected an (M, P, 2) float32 anchor array");
        }
        const size_t m_modes = anchors.shape(0), p_poses = anchors.shape(1);
        return toNumpy(bcdl::driveAnchorNoise(anchors.data(), static_cast<int>(m_modes),
                                              static_cast<int>(p_poses), cfg),
                       {m_modes, p_poses, 2});
      },
      "anchors"_a, "config"_a = bcdl::DriveAnchorConfig{},
      "Normalize (M,P,2) trajectory anchors into the diffusion input space and "
      "add the truncated DDIM noise. noise_scale=0 makes it deterministic.");

  // decode_drive(): numpy Python path. Pass None for a head the model lacks.
  m.def(
      "decode_drive",
      [](std::optional<nb::ndarray<const float, nb::c_contig>> trajectory,
         std::optional<nb::ndarray<const float, nb::c_contig>> agent_states,
         std::optional<nb::ndarray<const float, nb::c_contig>> agent_labels,
         std::optional<nb::ndarray<const float, nb::c_contig>> bev,
         const bcdl::DriveConfig& cfg) {
        auto shape_of = [](const std::optional<nb::ndarray<const float, nb::c_contig>>& a) {
          std::vector<int> s;
          if (a) {
            s.reserve(a->ndim());
            for (size_t d = 0; d < a->ndim(); ++d) s.push_back(static_cast<int>(a->shape(d)));
          }
          return s;
        };
        auto data_of = [](const std::optional<nb::ndarray<const float, nb::c_contig>>& a) {
          return a ? a->data() : nullptr;
        };
        return bcdl::decodeDrive(data_of(trajectory), shape_of(trajectory),
                                 data_of(agent_states), shape_of(agent_states),
                                 data_of(agent_labels), shape_of(agent_labels),
                                 data_of(bev), shape_of(bev), cfg);
      },
      "trajectory"_a = nb::none(), "agent_states"_a = nb::none(),
      "agent_labels"_a = nb::none(), "bev"_a = nb::none(),
      "config"_a = bcdl::DriveConfig{},
      "Decode the four driving heads into a DriveResult. Any head may be None.");

  nb::class_<bcdl::DrivePlanner>(m, "DrivePlanner")
      .def(
          "__init__",
          [](bcdl::DrivePlanner* self, bcdl::Engine& engine,
             nb::ndarray<const float, nb::c_contig> anchors, bcdl::DriveConfig cfg,
             bcdl::DriveAnchorConfig anchor_cfg, bcdl::DriveIoConfig io) {
            if (anchors.ndim() != 3 || anchors.shape(2) != 2) {
              throw std::invalid_argument(
                  "DrivePlanner: anchors must be an (M, P, 2) float32 array");
            }
            new (self) bcdl::DrivePlanner(engine, anchors.data(),
                                          static_cast<int>(anchors.shape(0)),
                                          static_cast<int>(anchors.shape(1)), cfg,
                                          anchor_cfg, io);
          },
          "engine"_a, "anchors"_a, "config"_a = bcdl::DriveConfig{},
          "anchor_config"_a = bcdl::DriveAnchorConfig{}, "io"_a = bcdl::DriveIoConfig{},
          nb::keep_alive<1, 2>())
      .def(
          "plan",
          [](bcdl::DrivePlanner& self, nb::ndarray<const float, nb::c_contig> camera,
             nb::ndarray<const float, nb::c_contig> lidar, const bcdl::DriveStatus& status,
             int timeout_ms) {
            std::vector<float> cam(camera.data(), camera.data() + camera.size());
            std::vector<float> lid(lidar.data(), lidar.data() + lidar.size());
            return self.plan(cam, lid, status, timeout_ms);
          },
          "camera"_a, "lidar"_a, "status"_a, "timeout_ms"_a = 0,
          "Write the four inputs (quantizing as the model declares), infer, decode.")
      .def("postprocess", &bcdl::DrivePlanner::postprocess)
      .def_prop_ro("config", &bcdl::DrivePlanner::config)
      .def_prop_ro(
          "diffusion_input",
          [](const bcdl::DrivePlanner& self) {
            return toNumpy(std::vector<float>(self.diffusionInput()),
                           {self.diffusionInput().size()});
          },
          // Owned-via-capsule ndarray; override the property default policy.
          nb::rv_policy::reference);
}
