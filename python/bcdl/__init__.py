"""BCDL — BPU Computational Deep Learning (Python wrapper).

Thin numpy-friendly layer over the compiled ``bcdl_py`` extension. The C++ core
exchanges raw bytes; this wrapper reshapes outputs into numpy arrays.
"""

from __future__ import annotations

import numpy as np

import bcdl_py

__version__ = bcdl_py.__version__


class Engine:
    """Load an ``.hbm`` model and run BPU inference with numpy in/out."""

    def __init__(self, hbm_path: str, model_name: str = ""):
        self._e = bcdl_py.Engine(hbm_path, model_name)

    @property
    def model_name(self) -> str:
        return self._e.model_name

    @property
    def num_inputs(self) -> int:
        return self._e.num_inputs

    @property
    def num_outputs(self) -> int:
        return self._e.num_outputs

    def input_shape(self, i: int) -> list[int]:
        return self._e.input_shape(i)

    def input_bytes(self, i: int) -> int:
        """Allocated device-buffer size for input i (post BPU stride alignment)."""
        return self._e.input_bytes(i)

    def output_shape(self, i: int) -> list[int]:
        return self._e.output_shape(i)

    def input_dtype(self, i: int) -> str:
        return self._e.input_dtype(i)

    def output_dtype(self, i: int) -> str:
        return self._e.output_dtype(i)

    def infer(self, inputs: list[np.ndarray], timeout_ms: int = 0) -> list[np.ndarray]:
        """Run one inference. ``inputs`` are copied into the device buffers in
        order; returns one numpy array per model output."""
        for i, arr in enumerate(inputs):
            self._e.set_input(i, np.ascontiguousarray(arr))
        self._e.infer(timeout_ms)
        outs = []
        for i in range(self._e.num_outputs):
            raw = self._e.output_bytes(i)
            dtype = np.dtype(self._e.output_dtype(i))
            shape = self._e.output_shape(i)
            n = int(np.prod(shape)) if shape else 0
            # output_bytes() returns the aligned device buffer; take the valid
            # prefix. (Stride-padded layouts are handled in the task layer.)
            flat = np.frombuffer(raw, dtype=dtype, count=n)
            outs.append(flat.reshape(shape))
        return outs


# --- detection: re-export the C++ decode/NMS + geometry --------------------
# These are the "numpy Python path": preprocess in numpy, run the Engine, then
# decode the float output with the same C++ kernel the native path uses.
#
# NOTE: decode() expects a float32 array. For an F16 output, cast first
# (np.asarray(out, np.float32)). For a QUANTIZED int output (int8/int16) decode()
# would see raw quantized integers with NO dequant and produce wrong boxes — use
# the native Detector.postprocess() path instead, which dequantizes via the
# tensor's scale/zero-point.
LetterboxInfo = bcdl_py.LetterboxInfo
DetectConfig = bcdl_py.DetectConfig
DecodeLayout = bcdl_py.DecodeLayout
Detection = bcdl_py.Detection
compute_letterbox = bcdl_py.compute_letterbox
decode = bcdl_py.decode
nms = bcdl_py.nms
iou = bcdl_py.iou

# Anchor-free LTRB multi-scale head (YOLO26 / standard RDK YOLO NV12 export):
# the model emits a (cls, box) output pair per stride. Use YoloLtrbConfig +
# the native YoloLtrbDetector (wrapped below) for that family.
YoloLtrbConfig = bcdl_py.YoloLtrbConfig


def letterbox_numpy(img: np.ndarray, dst_w: int, dst_h: int, pad: int = 114):
    """Aspect-preserving letterbox of an HxWxC (or HxW) uint8 image into a
    (dst_h, dst_w) canvas. Returns (canvas, LetterboxInfo). Requires OpenCV
    (imported lazily); the geometry matches the C++ VP letterbox()."""
    import cv2

    h0, w0 = img.shape[:2]
    lb = compute_letterbox(w0, h0, dst_w, dst_h, True)
    new_w = max(1, int(round(w0 * lb.scale)))
    new_h = max(1, int(round(h0 * lb.scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 3:
        canvas = np.full((dst_h, dst_w, img.shape[2]), pad, np.uint8)
    else:
        canvas = np.full((dst_h, dst_w), pad, np.uint8)
    x0 = int(round(lb.pad_x))
    y0 = int(round(lb.pad_y))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas, lb


def letterbox_chw_float(img: np.ndarray, dst_w: int, dst_h: int, pad: int = 114,
                        to_rgb: bool = True, scale: float = 1.0 / 255.0,
                        mean=None, std=None):
    """Preprocess a BGR (or gray) image for a FLOAT-input model — e.g. a
    **QAT-exported** YOLO whose input is ``[1,3,H,W]`` float32 rather than the
    NV12 two-plane layout the PTQ (hb_compile) export takes.

    Aspect-preserving letterbox to (dst_h, dst_w) with border value ``pad``,
    optional BGR->RGB, multiply by ``scale`` (default 1/255), optional per-channel
    ``(x - mean) / std`` (ImageNet-style), then HWC->NCHW contiguous float32.
    Returns ``(nchw, LetterboxInfo)``; feed ``nchw`` as the single model input to
    ``YoloLtrbDetector.detect([nchw], lb)`` (or use ``detect_float`` below).

    The letterbox geometry is identical to :func:`letterbox_numpy`, so the
    returned ``lb`` maps decoded boxes back to original-image pixels exactly as
    the NV12 path does.

    Fast path (``mean is None``): a single ``cv2.dnn.blobFromImage`` call fuses
    uint8->float32 + scale + BGR->RGB + HWC->NCHW into one optimized C pass (no
    intermediate numpy allocations). The ImageNet ``mean/std`` path preallocates
    the NCHW output and writes each channel in place."""
    import cv2

    canvas, lb = letterbox_numpy(img, dst_w, dst_h, pad)
    if canvas.ndim == 2:
        canvas = canvas[:, :, None]
    if mean is None:
        # scalefactor*(img) + swapRB + NCHW, all in one C call -> contiguous (1,C,H,W) f32.
        return cv2.dnn.blobFromImage(canvas, scalefactor=float(scale),
                                     swapRB=bool(to_rgb and canvas.shape[2] == 3),
                                     crop=False), lb
    # ImageNet-style (x*scale - mean)/std: one output alloc, per-channel in place.
    C = canvas.shape[2]
    out = np.empty((1, C, canvas.shape[0], canvas.shape[1]), np.float32)
    order = range(C - 1, -1, -1) if (to_rgb and C == 3) else range(C)
    m = np.asarray(mean, np.float32).reshape(-1)
    s = np.asarray(std, np.float32).reshape(-1)
    sc = np.float32(scale)
    for dc, srcc in enumerate(order):
        np.multiply(canvas[:, :, srcc], sc, out=out[0, dc])   # uint8->f32 + scale, no temp
        out[0, dc] -= m[dc]
        out[0, dc] /= s[dc]
    return out, lb


class Detector:
    """High-level detector: wraps an Engine + DetectConfig. Preprocessing is the
    caller's responsibility (model layout/normalization vary); pass the formatted
    model input and the LetterboxInfo from letterbox_numpy()."""

    def __init__(self, engine: "Engine", config: DetectConfig, output_index: int = 0):
        self.engine = engine
        self.config = config
        self._det = bcdl_py.Detector(engine._e, config, output_index)

    def detect(self, model_input: np.ndarray, lb: LetterboxInfo, timeout_ms: int = 0):
        """Run inference on one preprocessed input and post-process to a list of
        Detection (original-image pixel coords). `model_input` bytes must match
        the model's input buffer layout."""
        self.engine._e.set_input(0, np.ascontiguousarray(model_input))
        self.engine._e.infer(timeout_ms)
        return self._det.postprocess(lb)


class YoloLtrbDetector:
    """High-level anchor-free LTRB detector (YOLO26 / RDK YOLO). Wraps an Engine
    that emits one (cls, box) output pair per stride. Preprocessing is the
    caller's responsibility; for the two-input NV12 layout pass [Y, UV] planes."""

    def __init__(self, engine: "Engine", config: "YoloLtrbConfig | None" = None,
                 output_base: int = 0):
        self.engine = engine
        self.config = config if config is not None else YoloLtrbConfig()
        self._det = bcdl_py.YoloLtrbDetector(engine._e, self.config, output_base)

    def detect(self, inputs: "list[np.ndarray]", lb: LetterboxInfo, timeout_ms: int = 0):
        """Set the model inputs in order (e.g. [Y, UV] for the NV12 two-input
        layout), infer, and post-process to a list of Detection (original-image
        pixel coords)."""
        for i, arr in enumerate(inputs):
            self.engine._e.set_input(i, np.ascontiguousarray(arr))
        self.engine._e.infer(timeout_ms)
        return self._det.postprocess(lb)

    def postprocess(self, lb: LetterboxInfo):
        """Post-process the Engine's current output (run engine.infer() first)."""
        return self._det.postprocess(lb)

    def detect_float(self, img_bgr: np.ndarray, dst_w: int = 640, dst_h: int = 640,
                     pad: int = 114, to_rgb: bool = True, scale: float = 1.0 / 255.0,
                     mean=None, std=None, timeout_ms: int = 0):
        """One-call detect for a **FLOAT-input (QAT-exported) LTRB model**: letterbox
        + normalize + NCHW (:func:`letterbox_chw_float`), set the single input,
        infer, and decode to Detections in original-image pixels. ``img_bgr`` is
        the original frame. Use this instead of the NV12 ``detect([Y,UV], lb)``
        when the model input is ``[1,3,H,W]`` float32 (the horizon_plugin_pytorch
        QAT / hbdk4 export). Postprocess is identical — the LTRB head decode +
        sigmoid + NMS are the same for PTQ and QAT models."""
        nchw, lb = letterbox_chw_float(img_bgr, dst_w, dst_h, pad, to_rgb, scale, mean, std)
        return self.detect([nchw], lb, timeout_ms)


# --- media: VpImage + JPU/VPU codecs ---------------------------------------
# VpImage is the unified shared-memory image. Construct one from numpy via the
# vp_image_from_* factories (which honor the device row stride), and read a
# decoded image back with VpImage.to_numpy(). All of these allocate cached
# device buffers / drive the JPU/VPU, so they only work on the board.
ImageFormat = bcdl_py.ImageFormat
VideoType = bcdl_py.VideoType
VpImage = bcdl_py.VpImage
vp_image_from_bgr = bcdl_py.vp_image_from_bgr
vp_image_from_nv12 = bcdl_py.vp_image_from_nv12
JpegEncoder = bcdl_py.JpegEncoder
JpegDecoder = bcdl_py.JpegDecoder
# GDC hardware ops — only present when built with BCDL_HAVE_GDC (board).
GdcLetterbox = getattr(bcdl_py, "GdcLetterbox", None)
GdcRemap = getattr(bcdl_py, "GdcRemap", None)
VideoEncConfig = bcdl_py.VideoEncConfig
VideoDecConfig = bcdl_py.VideoDecConfig
VideoEncoder = bcdl_py.VideoEncoder
VideoDecoder = bcdl_py.VideoDecoder


def bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 uint8 BGR image to a packed NV12 buffer of shape
    (H*3//2, W). Requires even width/height and OpenCV (imported lazily).
    The result feeds vp_image_from_nv12() / jpeg_encode()."""
    import cv2

    h, w = bgr.shape[:2]
    if (w & 1) or (h & 1):
        raise ValueError("bgr_to_nv12 requires even width and height")
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)  # (H*3//2, W): Y, U, V planes
    nv12 = np.empty((h * 3 // 2, w), np.uint8)
    nv12[:h] = i420[:h]
    u = i420[h : h + h // 4].reshape(-1)
    v = i420[h + h // 4 :].reshape(-1)
    uv = nv12[h:].reshape(-1)
    uv[0::2] = u  # interleave U/V -> NV12 (UVUV...)
    uv[1::2] = v
    return nv12


def jpeg_encode(image, quality: int = 50) -> bytes:
    """One-shot hardware (JPU) JPEG encode. `image` is either an HxWx3 uint8 BGR
    ndarray, or an NV12 ``VpImage`` (e.g. straight from ``jpeg_decode`` /
    ``GdcLetterbox`` — no BGR round-trip). Returns the JPEG byte stream.

    The JPU requires width%16==0 and height%8==0. A misaligned BGR image is padded
    with a black bottom/right border to the aligned size (the padded dims are what
    get encoded); a misaligned NV12 VpImage is rejected (pad it before wrapping)."""
    # NV12 VpImage fast path — encode the device buffer directly, no CPU convert.
    if isinstance(image, VpImage):
        if image.format != ImageFormat.NV12:
            raise ValueError("jpeg_encode(VpImage) requires an NV12 image")
        w, h = image.width, image.height
        if (w & 15) or (h & 7):
            raise ValueError(
                f"jpeg_encode: NV12 VpImage {w}x{h} is not JPU-aligned (need w%16, h%8)")
        return JpegEncoder(w, h, quality, ImageFormat.NV12).encode(image)

    # BGR ndarray path.
    bgr = image
    h, w = bgr.shape[:2]
    aw, ah = (w + 15) & ~15, (h + 7) & ~7
    if aw != w or ah != h:
        import cv2

        bgr = cv2.copyMakeBorder(bgr, 0, ah - h, 0, aw - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w = ah, aw
    nv12 = bgr_to_nv12(bgr)
    enc = JpegEncoder(w, h, quality, ImageFormat.NV12)
    return enc.encode(vp_image_from_nv12(np.ascontiguousarray(nv12), w, h))


def jpeg_save(image, path: str, quality: int = 90) -> int:
    """Hardware (JPU) JPEG-encode `image` (BGR ndarray or NV12 ``VpImage``) and
    write it to `path`. Returns the number of bytes written. This is the JPU
    counterpart to ``cv2.imwrite`` — the encode runs on the JPU, not libjpeg."""
    data = jpeg_encode(image, quality)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def jpeg_decode(data) -> "VpImage":
    """Decode JPEG bytes (or a uint8 numpy buffer) into an NV12 VpImage via the
    JPU. Use the returned image's .to_numpy() for the raw planes."""
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    return JpegDecoder().decode(bytes(data))


# --- depth + segmentation post-processing ----------------------------------
# decode_depth/decode_seg are the "numpy Python path" (mirror of the native
# DepthEstimator/Segmenter): pass a float32 engine output, get a DepthMap/SegMask.
# camelCase aliases match the C++ names the tests probe for.
DepthConfig = bcdl_py.DepthConfig
DepthMap = bcdl_py.DepthMap
decode_depth = bcdl_py.decode_depth
decodeDepth = bcdl_py.decode_depth
depth_colorize = bcdl_py.depth_colorize
depth_to_gray8 = bcdl_py.depth_to_gray8

SegConfig = bcdl_py.SegConfig
SegMask = bcdl_py.SegMask
decode_seg = bcdl_py.decode_seg
decodeSeg = bcdl_py.decode_seg
seg_colorize = bcdl_py.seg_colorize


class DepthEstimator:
    """High-level depth head: wraps an Engine + DepthConfig. Preprocessing is the
    caller's responsibility (model layout/normalization vary); pass the formatted
    model input to estimate(), or call postprocess() yourself after engine.infer()."""

    def __init__(self, engine: "Engine", config: "DepthConfig | None" = None,
                 output_index: int = 0):
        self.engine = engine
        self.config = config if config is not None else DepthConfig()
        self._d = bcdl_py.DepthEstimator(engine._e, self.config, output_index)

    def estimate(self, model_input: np.ndarray, timeout_ms: int = 0) -> "DepthMap":
        """Run inference on one preprocessed input and post-process to a DepthMap."""
        self.engine._e.set_input(0, np.ascontiguousarray(model_input))
        self.engine._e.infer(timeout_ms)
        return self._d.postprocess()

    def postprocess(self) -> "DepthMap":
        """Post-process the Engine's current output (run engine.infer() first)."""
        return self._d.postprocess()


class Segmenter:
    """High-level segmentation head: wraps an Engine + SegConfig. Preprocessing is
    the caller's responsibility; pass the formatted model input to segment(), or
    call postprocess() yourself after engine.infer()."""

    def __init__(self, engine: "Engine", config: "SegConfig | None" = None,
                 output_index: int = 0):
        self.engine = engine
        self.config = config if config is not None else SegConfig()
        self._s = bcdl_py.Segmenter(engine._e, self.config, output_index)

    def segment(self, model_input: np.ndarray, timeout_ms: int = 0) -> "SegMask":
        """Run inference on one preprocessed input and post-process to a SegMask."""
        self.engine._e.set_input(0, np.ascontiguousarray(model_input))
        self.engine._e.infer(timeout_ms)
        return self._s.postprocess()

    def postprocess(self) -> "SegMask":
        """Post-process the Engine's current output (run engine.infer() first)."""
        return self._s.postprocess()


# --- stereo: two-image disparity / depth (e.g. LAS2) -----------------------
StereoFit = bcdl_py.StereoFit
StereoConfig = bcdl_py.StereoConfig
StereoResult = bcdl_py.StereoResult
pack_stereo_input = bcdl_py.pack_stereo_input
disparity_to_depth = bcdl_py.disparity_to_depth
stereo_valid_mask = bcdl_py.stereo_valid_mask


class StereoPipeline:
    """Streaming two-image stereo. Wraps an Engine + StereoConfig and turns a
    rectified BGR pair into a disparity map (+ optional metric depth / validity
    mask). Preprocessing (fit + BGR->RGB + F32 NCHW pack) happens in the C++
    core; pixel normalization is fused into the .hbm. The fit mode (resize/crop)
    MUST match the mode the model was calibrated with."""

    def __init__(self, engine: "Engine", config: "StereoConfig | None" = None):
        self.engine = engine
        self.config = config if config is not None else StereoConfig()
        self._p = bcdl_py.StereoPipeline(engine._e, self.config)

    def process(self, left: np.ndarray, right: np.ndarray) -> "StereoResult":
        """Run one stereo pair (two HxWx3 uint8 BGR frames). Returns a
        StereoResult (.disparity, and .depth/.valid when enabled in the config)."""
        l, r = np.asarray(left), np.asarray(right)
        if l.dtype != np.uint8 or r.dtype != np.uint8:
            raise TypeError("StereoPipeline.process expects uint8 BGR images")
        if l.ndim != 3 or l.shape[2] != 3 or r.shape != l.shape:
            raise ValueError(f"expected matching HxWx3 BGR images, got {l.shape}/{r.shape}")
        return self._p.process(np.ascontiguousarray(l), np.ascontiguousarray(r))

    @property
    def input_w(self) -> int:
        return self._p.input_w

    @property
    def input_h(self) -> int:
        return self._p.input_h


# --- extra task heads: classification / pose / instance-seg / obb ----------
# Each mirrors the standard RDK YOLO26 export. The struct/config types are
# re-exported raw; the high-level wrapper classes pass engine._e like Detector.
ClsConfig = bcdl_py.ClsConfig
ClsResult = bcdl_py.ClsResult
decode_classification = bcdl_py.decode_classification
Keypoint = bcdl_py.Keypoint
PoseDetection = bcdl_py.PoseDetection
PoseConfig = bcdl_py.PoseConfig
# Pose numpy path: per-scale [H,W,1]/[H,W,4]/[H,W,K*3] float tensors -> PoseDetections
# (no Engine; mirrors PoseEstimator). Lists are one ndarray per stride.
decode_pose = bcdl_py.decode_pose
InstanceSegConfig = bcdl_py.InstanceSegConfig
InstanceMask = bcdl_py.InstanceMask
# Instance-seg numpy path: per-scale [H,W,nc]/[H,W,4]/[H,W,np] + proto [mH,mW,np]
# float tensors -> InstanceMasks (no Engine; mirrors InstanceSegmenter).
decode_instance_seg = bcdl_py.decode_instance_seg
RotatedBox = bcdl_py.RotatedBox
ObbDetection = bcdl_py.ObbDetection
ObbConfig = bcdl_py.ObbConfig
rotated_iou = bcdl_py.rotated_iou
# OBB numpy path: per-scale [H,W,nc]/[H,W,4]/[H,W,1] float tensors -> ObbDetections.
decode_obb = bcdl_py.decode_obb
CameraIntrinsics = bcdl_py.CameraIntrinsics
Mono3dBox = bcdl_py.Mono3dBox
Mono3dConfig = bcdl_py.Mono3dConfig
compute_mono3d_feature_xform = bcdl_py.compute_mono3d_feature_xform
# Mono3d (SMOKE) numpy path: channel-first cls[nc,H,W]/reg[8,H,W] raw logits +
# feature affine + camera K -> Mono3dBoxes (no Engine; mirrors Mono3dDetector).
decode_mono3d = bcdl_py.decode_mono3d

# --- open-vocabulary label table (YOLOE prompt-free) -----------------------
# class_id -> prompt-name map; detection decode is unchanged (reuse
# YoloLtrbDetector / Detector with num_classes == len(LabelMap)).
LabelMap = bcdl_py.LabelMap

# --- ReID appearance embeddings (BoT-SORT association primitives) ----------
normalize_embedding = bcdl_py.normalize_embedding
cosine_similarity = bcdl_py.cosine_similarity

# --- promptable segmentation (EdgeSAM / SAM mask decoder tail) --------------
SamConfig = bcdl_py.SamConfig
SamMask = bcdl_py.SamMask
# SAM numpy path: low_res_logits [Nmask,Hm,Wm] + iou_pred [Nmask] -> best SamMask.
decode_sam_masks = bcdl_py.decode_sam_masks


def _read_packed_output(engine, idx, shape):
    """Read a PACKED (contiguous) float engine output, reshaped to `shape`, using
    the tensor's DECLARED dtype (F32/F16) rather than assuming float32. Used by
    SamSession for the encoder embedding + low-res mask logits (both packed F32
    from the EdgeSAM hbms). Rejects non-float (scaled-int) outputs — those need
    per-tensor dequant via the C++ `outputAsFloat` path, not a raw byte read —
    and rejects a short buffer. (Assumes no last-dim stride padding, which holds
    for these hbms; stride-padded outputs would need the C++ reader.)"""
    dt = np.dtype(engine.output_dtype(idx))
    if dt.kind != "f":
        raise RuntimeError(
            f"SamSession: output {idx} dtype {dt} is not float; a scaled-int output "
            "needs the C++ dequant path (outputAsFloat), not this packed reader")
    n = int(np.prod(shape))
    flat = np.frombuffer(engine._e.output_bytes(idx), dt)
    if flat.size < n:
        raise RuntimeError(
            f"SamSession: output {idx} has {flat.size} elems < {n} for shape {shape}")
    return flat[:n].reshape(shape)


class SamMaskDecoder:
    """SAM/EdgeSAM mask-decoder tail. 2 outputs: low-res mask logits + iou_pred."""

    def __init__(self, engine: "Engine", config: "SamConfig | None" = None,
                 masks_index: int = 0, iou_index: int = 1):
        self.engine = engine
        self.config = config if config is not None else SamConfig()
        self._d = bcdl_py.SamMaskDecoder(engine._e, self.config, masks_index, iou_index)

    def postprocess(self):
        return self._d.postprocess()


class SamSession:
    """EdgeSAM interactive segmentation over the two-hbm (encoder + decoder)
    pipeline.

    ``set_image(bgr)`` SAM-preprocesses one image (resize longest side to
    ``input_size`` + pad, BGR->RGB; the ImageNet norm is baked into the encoder
    ``.hbm`` so raw pixels are fed) and caches the encoder embedding. Then each
    ``predict(x, y)`` runs only the (lightweight) decoder against that cached
    embedding, so repeated clicks are cheap.

    Each deployed EdgeSAM decoder is a FIXED-point-count graph (statically shaped
    at conversion): the ``decoder`` handles a single click point; the optional
    ``box_decoder`` handles a 2-corner box. The iou-prediction head is int8-
    degraded, so mask selection among the multimask outputs defaults to ``"area"``
    (largest) rather than the unreliable predicted-iou argmax; pass ``select=<int>``
    to force an index. ``predict`` / ``predict_box`` return ``(mask, index)`` where
    ``mask`` is a full-resolution ``(H, W)`` uint8 (0/1) array in the ORIGINAL
    image size.
    """

    def __init__(self, encoder: "Engine", decoder: "Engine",
                 box_decoder: "Engine | None" = None, input_size: int = 1024):
        self.enc = encoder
        self.dec = decoder
        self.box_dec = box_decoder
        self.input_size = input_size
        self._emb = None
        self._scale = 1.0
        self._nh = self._nw = self._ow = self._oh = 0

    @staticmethod
    def _masks_out_index(decoder: "Engine") -> int:
        # The masks output is the 4-D one ([1, Nmask, mh, mw]); scores is [1, Nmask].
        return next((i for i in range(decoder.num_outputs)
                     if len(decoder.output_shape(i)) == 4), decoder.num_outputs - 1)

    def set_image(self, bgr) -> "SamSession":
        import cv2
        h, w = bgr.shape[:2]
        s = self.input_size / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        canvas = np.zeros((self.input_size, self.input_size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        x = np.ascontiguousarray(
            cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32))
        self.enc._e.set_input(0, x)
        self.enc._e.infer(0)
        eshape = self.enc.output_shape(0)
        self._emb = _read_packed_output(self.enc, 0, eshape).astype(np.float32, copy=True)
        self._scale, self._nh, self._nw, self._ow, self._oh = s, nh, nw, w, h
        return self

    def _decode(self, decoder, coords, labels, select):
        """Run one decoder pass. `coords` are ORIGINAL-image (x,y) pairs; `labels`
        the matching SAM prompt labels. Returns (full-res mask, index)."""
        import cv2
        if self._emb is None:
            raise RuntimeError("SamSession: call set_image() first")
        pt = np.array([[[cx * self._scale, cy * self._scale] for cx, cy in coords]], np.float32)
        lab = np.array([[float(v) for v in labels]], np.float32)
        decoder._e.set_input(0, self._emb)
        decoder._e.set_input(1, np.ascontiguousarray(pt))
        decoder._e.set_input(2, np.ascontiguousarray(lab))
        decoder._e.infer(0)
        mo = self._masks_out_index(decoder)
        _, nmask, mh, mw = decoder.output_shape(mo)
        masks = _read_packed_output(decoder, mo, (nmask, mh, mw))
        binm = (masks > 0).astype(np.uint8)
        areas = binm.reshape(nmask, -1).sum(1)
        idx = int(select) if isinstance(select, int) else int(areas.argmax())
        # low-res mask -> input canvas -> crop scaled region -> original size.
        up = cv2.resize(binm[idx], (self.input_size, self.input_size),
                        interpolation=cv2.INTER_NEAREST)[:self._nh, :self._nw]
        full = cv2.resize(up, (self._ow, self._oh), interpolation=cv2.INTER_NEAREST)
        return full.astype(np.uint8), idx

    def predict(self, x: float, y: float, label: int = 1, select="area"):
        return self._decode(self.dec, [(x, y)], [label], select)

    def predict_box(self, x1: float, y1: float, x2: float, y2: float, select="area"):
        if self.box_dec is None:
            raise RuntimeError("SamSession.predict_box: no box_decoder was provided")
        # SAM box convention: two corners with labels 2 (top-left) / 3 (bottom-right).
        return self._decode(self.box_dec, [(x1, y1), (x2, y2)], [2, 3], select)


class Classifier:
    """Top-k image classifier. Run engine.infer() (or pass via classify())."""

    def __init__(self, engine: "Engine", config: "ClsConfig | None" = None,
                 output_index: int = 0):
        self.engine = engine
        self.config = config if config is not None else ClsConfig()
        self._d = bcdl_py.Classifier(engine._e, self.config, output_index)

    def classify(self, inputs, timeout_ms: int = 0):
        """Set input(s) (a single array or a list, e.g. [Y, UV] for NV12), infer,
        return the top-k ClsResult list."""
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        for i, arr in enumerate(inputs):
            self.engine._e.set_input(i, np.ascontiguousarray(arr))
        self.engine._e.infer(timeout_ms)
        return self._d.postprocess()

    def postprocess(self):
        return self._d.postprocess()


class PoseEstimator:
    """LTRB multi-scale pose head (box + keypoints). 2-input NV12 [Y,UV]."""

    def __init__(self, engine: "Engine", config: "PoseConfig | None" = None,
                 output_base: int = 0):
        self.engine = engine
        self.config = config if config is not None else PoseConfig()
        self._d = bcdl_py.PoseEstimator(engine._e, self.config, output_base)

    def detect(self, inputs, lb, timeout_ms: int = 0):
        for i, arr in enumerate(inputs):
            self.engine._e.set_input(i, np.ascontiguousarray(arr))
        self.engine._e.infer(timeout_ms)
        return self._d.postprocess(lb)

    def postprocess(self, lb):
        return self._d.postprocess(lb)


class InstanceSegmenter:
    """LTRB instance segmentation (box + per-instance binary mask)."""

    def __init__(self, engine: "Engine", config: "InstanceSegConfig | None" = None,
                 output_base: int = 0):
        self.engine = engine
        self.config = config if config is not None else InstanceSegConfig()
        self._d = bcdl_py.InstanceSegmenter(engine._e, self.config, output_base)

    def detect(self, inputs, lb, orig_w, orig_h, timeout_ms: int = 0):
        for i, arr in enumerate(inputs):
            self.engine._e.set_input(i, np.ascontiguousarray(arr))
        self.engine._e.infer(timeout_ms)
        return self._d.postprocess(lb, orig_w, orig_h)

    def postprocess(self, lb, orig_w, orig_h):
        return self._d.postprocess(lb, orig_w, orig_h)


class ObbDetector:
    """LTRB+angle oriented-box detector (rotated NMS)."""

    def __init__(self, engine: "Engine", config: "ObbConfig | None" = None,
                 output_base: int = 0):
        self.engine = engine
        self.config = config if config is not None else ObbConfig()
        self._d = bcdl_py.ObbDetector(engine._e, self.config, output_base)

    def detect(self, inputs, lb, timeout_ms: int = 0):
        for i, arr in enumerate(inputs):
            self.engine._e.set_input(i, np.ascontiguousarray(arr))
        self.engine._e.infer(timeout_ms)
        return self._d.postprocess(lb)

    def postprocess(self, lb):
        return self._d.postprocess(lb)


class Mono3dDetector:
    """SMOKE monocular-3D detector. Single F32 NCHW RGB input [1,3,384,1280]
    (ImageNet-normalized); postprocess takes the ORIGINAL image (W,H) and its
    camera intrinsics K to lift heatmap peaks to 3D boxes."""

    def __init__(self, engine: "Engine", config: "Mono3dConfig | None" = None,
                 output_base: int = 0):
        self.engine = engine
        self.config = config if config is not None else Mono3dConfig()
        self._d = bcdl_py.Mono3dDetector(engine._e, self.config, output_base)

    def detect(self, image, orig_w, orig_h, K, timeout_ms: int = 0):
        self.engine._e.set_input(0, np.ascontiguousarray(image))
        self.engine._e.infer(timeout_ms)
        return self._d.postprocess(orig_w, orig_h, K)

    def postprocess(self, orig_w, orig_h, K):
        return self._d.postprocess(orig_w, orig_h, K)


# --- multi-object tracking (ByteTrack) -------------------------------------
# Directly usable, no model: ByteTracker(cfg).update(list[Detection]) per frame.
Track = bcdl_py.Track
ByteTrackConfig = bcdl_py.ByteTrackConfig
ByteTracker = bcdl_py.ByteTracker

# --- detect + track in one call: TrackingPipeline --------------------------
# Needs an NV12-input YOLO .hbm. process(bgr) runs the C++ DetectionPipeline
# (NV12 preproc + infer + decode/NMS, buffer-reuse) then ByteTracker, returning
# this frame's active Tracks. This is the native detect->track path; for the
# numpy-composition alternative, drive Detector/YoloLtrbDetector + ByteTracker.
DetectHead = bcdl_py.DetectHead
PipelineConfig = bcdl_py.PipelineConfig
StageProfile = bcdl_py.StageProfile  # per-stage timing (DetectionPipeline / async)


class TrackingPipeline:
    """Streaming detect-and-track. Wraps an Engine + (PipelineConfig, ByteTrackConfig)
    and tracks objects across frames with stable ids — all preprocessing happens
    in the C++ core (no per-frame numpy letterbox)."""

    def __init__(self, engine: "Engine",
                 det_config: "PipelineConfig | None" = None,
                 track_config: "ByteTrackConfig | None" = None):
        self.det_config = det_config if det_config is not None else PipelineConfig()
        self.track_config = track_config if track_config is not None else ByteTrackConfig()
        self._p = bcdl_py.TrackingPipeline(engine._e, self.det_config, self.track_config)

    def process(self, bgr: np.ndarray) -> list:
        """Detect + track one BGR frame (HxWx3 uint8). Returns this frame's Tracks
        (stable track_id, original-image pixel boxes)."""
        arr = np.asarray(bgr)
        # Reject non-uint8 loudly instead of silently truncating: a normalized
        # float image (e.g. [0,1]) cast to uint8 would become all zeros and
        # detect nothing, which is far harder to diagnose than a clear error.
        if arr.dtype != np.uint8:
            raise TypeError(
                f"TrackingPipeline.process expects a uint8 BGR image, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {arr.shape}")
        return self._p.process(np.ascontiguousarray(arr))

    @property
    def last_detections(self) -> list:
        """Detections from the most recent process(), pre-association."""
        return self._p.last_detections

    def reset(self) -> None:
        self._p.reset()


# --- streaming detect with preproc‖infer overlap: AsyncDetectionPipeline ----
# Two C++ worker threads (preproc | infer+decode) overlap so steady-state
# throughput approaches 1/max(preproc, infer+decode) instead of the sum. submit()
# and next() block (backpressure / wait-for-result) but RELEASE the GIL, so other
# Python threads keep running. Results come back from next() in submission order;
# next() returns None once finished AND fully drained.
class AsyncDetectionPipeline:
    """Threaded streaming detector that overlaps CPU preprocessing of later frames
    with BPU inference + decode of earlier ones. Needs an NV12-input YOLO .hbm.

    Typical streaming use (keep ~`depth` frames in flight, drain in order)::

        p = bcdl.AsyncDetectionPipeline(engine, cfg, depth=3)
        for i, frame in enumerate(stream):
            p.submit(frame)                 # blocks if the pipeline is full
            if i >= 3:
                for d in p.next():          # results in submission order
                    ...
        p.finish()
        while (dets := p.next()) is not None:
            ...                             # last in-flight results
    """

    def __init__(self, engine: "Engine",
                 config: "PipelineConfig | None" = None,
                 depth: int = 3):
        self.config = config if config is not None else PipelineConfig()
        self._p = bcdl_py.AsyncDetectionPipeline(engine._e, self.config, depth)

    def submit(self, bgr: np.ndarray) -> bool:
        """Enqueue an HxWx3 uint8 BGR frame (bytes copied, caller may reuse the
        array immediately). Blocks while the pipeline is full (backpressure).
        Returns False if the pipeline has been finish()ed (frame not accepted)."""
        arr = np.asarray(bgr)
        # Reject non-uint8 loudly: a normalized float image cast to uint8 would
        # become all zeros and detect nothing (see TrackingPipeline.process).
        if arr.dtype != np.uint8:
            raise TypeError(
                f"AsyncDetectionPipeline.submit expects a uint8 BGR image, got {arr.dtype}")
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {arr.shape}")
        return self._p.submit(np.ascontiguousarray(arr))

    def next(self) -> "list | None":
        """Pop the next result in submission order (blocks until ready). Returns
        the frame's Detections, or None once the pipeline is finished AND fully
        drained."""
        return self._p.next()

    def finish(self) -> None:
        """Signal that no more frames will be submitted; after the in-flight
        frames drain, next() returns None. Idempotent (also called on GC)."""
        self._p.finish()

    @property
    def head(self) -> "DetectHead":
        """Resolved decoder family (kAuto replaced by the concrete choice)."""
        return self._p.head

    def profile(self) -> "StageProfile":
        """Per-stage service timing (preproc / infer / postproc), each measured on
        its own worker thread — the slowest stage bounds throughput. Read after
        finish() + full drain for a settled value."""
        return self._p.profile()


class AsyncVideoDetectionPipeline:
    """Full compressed-video -> detections pipeline (VPU decode ‖ CPU preproc ‖
    BPU infer), all in C++. Feed raw Annex-B byte chunks; the pipeline segments
    and decodes them internally, so Python only pumps bytes (e.g. from an
    ``ffmpeg -c copy`` RTSP/mp4 demux). submit()/next() release the GIL.

        pipe = bcdl.AsyncVideoDetectionPipeline(engine, cfg, bcdl.VideoType.H264, depth=4)
        while chunk := ffmpeg.stdout.read(65536):
            pipe.submit(chunk)                       # blocks on backpressure
            while (dets := pipe.next_nowait()) is not None:
                draw(dets)
        pipe.finish()
        while (dets := pipe.next()) is not None:      # drain the last frames
            draw(dets)
    """

    def __init__(self, engine: "Engine",
                 config: "PipelineConfig | None" = None,
                 codec: "VideoType" = None,
                 depth: int = 4):
        self.config = config if config is not None else PipelineConfig()
        codec = codec if codec is not None else VideoType.H264
        self._p = bcdl_py.AsyncVideoDetectionPipeline(engine._e, self.config, codec, depth)

    def submit(self, data: bytes) -> bool:
        """Feed a chunk of Annex-B compressed bytes. Blocks on backpressure;
        returns False once finish()ed."""
        return self._p.submit(bytes(data))

    def next(self) -> "list | None":
        """Blocking pop of the next frame's detections in decode order; None once
        finished AND drained."""
        return self._p.next()

    def next_nowait(self) -> "list | None":
        """Non-blocking pop: detections if one is ready, else None."""
        return self._p.next_nowait()

    def finish(self) -> None:
        """Signal end of stream; drains in-flight frames then next() ends."""
        self._p.finish()

    def profile(self) -> "StageProfile":
        """Per-stage timing incl. decode_ms (VPU decode + NV12->BGR)."""
        return self._p.profile()

# --- OCR: DBNet detect / orientation cls / CTC recognize -------------------
# decode_ctc/decode_dbnet/load_char_dict are pure (no model). The Engine-bound
# TextRecognizer/TextAngleClassifier/DbTextDetector take engine._e and need an
# OCR .hbm on the board (model conversion lives in a sibling project).
RecResult = bcdl_py.RecResult
ClsDirResult = bcdl_py.ClsDirResult
TextBox = bcdl_py.TextBox
DbConfig = bcdl_py.DbConfig
load_char_dict = bcdl_py.load_char_dict
decode_ctc = bcdl_py.decode_ctc
decode_dbnet = bcdl_py.decode_dbnet
decode_cls_dir = bcdl_py.decode_cls_dir
TextRecognizer = bcdl_py.TextRecognizer
TextAngleClassifier = bcdl_py.TextAngleClassifier
DbTextDetector = bcdl_py.DbTextDetector


__all__ = [
    "Engine",
    "Detector",
    "DetectConfig",
    "DecodeLayout",
    "Detection",
    "LetterboxInfo",
    "compute_letterbox",
    "letterbox_numpy",
    "letterbox_chw_float",
    "decode",
    "nms",
    "iou",
    "YoloLtrbConfig",
    "YoloLtrbDetector",
    # media
    "ImageFormat",
    "VideoType",
    "VpImage",
    "vp_image_from_bgr",
    "vp_image_from_nv12",
    "JpegEncoder",
    "JpegDecoder",
    "VideoEncConfig",
    "VideoDecConfig",
    "VideoEncoder",
    "VideoDecoder",
    "bgr_to_nv12",
    "jpeg_encode",
    "jpeg_save",
    "jpeg_decode",
    "GdcLetterbox",
    "GdcRemap",
    # depth
    "DepthConfig",
    "DepthMap",
    "DepthEstimator",
    "decode_depth",
    "decodeDepth",
    "depth_colorize",
    "depth_to_gray8",
    # segmentation
    "SegConfig",
    "SegMask",
    "Segmenter",
    "decode_seg",
    "decodeSeg",
    "seg_colorize",
    # stereo
    "StereoFit",
    "StereoConfig",
    "StereoResult",
    "StereoPipeline",
    "pack_stereo_input",
    "disparity_to_depth",
    "stereo_valid_mask",
    # classification
    "ClsConfig",
    "ClsResult",
    "Classifier",
    "decode_classification",
    # pose
    "Keypoint",
    "PoseDetection",
    "PoseConfig",
    "PoseEstimator",
    "decode_pose",
    # instance segmentation
    "InstanceSegConfig",
    "InstanceMask",
    "InstanceSegmenter",
    "decode_instance_seg",
    # oriented bounding box
    "RotatedBox",
    "ObbDetection",
    "ObbConfig",
    "decode_obb",
    "ObbDetector",
    "rotated_iou",
    # monocular 3D detection (SMOKE)
    "CameraIntrinsics",
    "Mono3dBox",
    "Mono3dConfig",
    "compute_mono3d_feature_xform",
    "decode_mono3d",
    "Mono3dDetector",
    # open-vocabulary label table (YOLOE prompt-free)
    "LabelMap",
    # ReID appearance embeddings (BoT-SORT primitives)
    "normalize_embedding",
    "cosine_similarity",
    # promptable segmentation (EdgeSAM / SAM mask decoder tail)
    "SamConfig",
    "SamMask",
    "SamMaskDecoder",
    "SamSession",
    "decode_sam_masks",
    # multi-object tracking
    "Track",
    "ByteTrackConfig",
    "ByteTracker",
    "DetectHead",
    "PipelineConfig",
    "StageProfile",
    "TrackingPipeline",
    "AsyncDetectionPipeline",
    "AsyncVideoDetectionPipeline",
    # OCR
    "RecResult",
    "ClsDirResult",
    "TextBox",
    "DbConfig",
    "load_char_dict",
    "decode_ctc",
    "decode_dbnet",
    "decode_cls_dir",
    "TextRecognizer",
    "TextAngleClassifier",
    "DbTextDetector",
]
