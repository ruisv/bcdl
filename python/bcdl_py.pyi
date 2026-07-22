"""BCDL — BPU Computational Deep Learning (C++ core)"""

from collections.abc import Sequence
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


class Engine:
    def __init__(self, hbm_path: str, model_name: str = '') -> None: ...

    @property
    def model_name(self) -> str: ...

    @staticmethod
    def model_names(hbm_path: str) -> list[str]:
        """
        List every model name packed into an .hbm (a package may hold several — e.g. SigLIP's global-embedding and patch-feature submodels). Pass one to Engine(path, model_name).
        """

    @property
    def num_inputs(self) -> int: ...

    @property
    def num_outputs(self) -> int: ...

    def input_shape(self, index: int) -> list[int]: ...

    def output_shape(self, index: int) -> list[int]: ...

    def input_bytes(self, index: int) -> int: ...

    def input_packed_bytes(self, index: int) -> int:
        """
        Byte size of input[i] as a contiguous row-major array. Smaller than input_bytes() when the model pads a dimension.
        """

    def input_stride(self, index: int) -> list[int]:
        """Resolved byte strides of input[i], innermost last."""

    def output_stride(self, index: int) -> list[int]:
        """
        Resolved byte strides of output[i], innermost last. Outputs are padded too — reshape output_bytes() flat and you shear the tensor.
        """

    def input_buffer_bytes(self, index: int) -> bytes:
        """
        Input[i]'s device buffer in the device layout — how set_input() actually laid the data out.
        """

    def input_dtype(self, index: int) -> str: ...

    def output_dtype(self, index: int) -> str: ...

    def set_input(self, index: int, array: Annotated[NDArray, dict(order='C')]) -> None: ...

    def infer(self, timeout_ms: int = 0) -> None: ...

    def output_bytes(self, index: int) -> bytes: ...

class LetterboxInfo:
    def __init__(self) -> None: ...

    @property
    def scale(self) -> float: ...

    @scale.setter
    def scale(self, arg: float, /) -> None: ...

    @property
    def pad_x(self) -> float: ...

    @pad_x.setter
    def pad_x(self, arg: float, /) -> None: ...

    @property
    def pad_y(self) -> float: ...

    @pad_y.setter
    def pad_y(self, arg: float, /) -> None: ...

    @property
    def src_w(self) -> int: ...

    @src_w.setter
    def src_w(self, arg: int, /) -> None: ...

    @property
    def src_h(self) -> int: ...

    @src_h.setter
    def src_h(self, arg: int, /) -> None: ...

    @property
    def dst_w(self) -> int: ...

    @dst_w.setter
    def dst_w(self, arg: int, /) -> None: ...

    @property
    def dst_h(self) -> int: ...

    @dst_h.setter
    def dst_h(self, arg: int, /) -> None: ...

    def fwd_x(self, x: float) -> float: ...

    def fwd_y(self, y: float) -> float: ...

    def inv_x(self, x: float) -> float: ...

    def inv_y(self, y: float) -> float: ...

def compute_letterbox(src_w: int, src_h: int, dst_w: int, dst_h: int, center_pad: bool = True) -> LetterboxInfo:
    """Aspect-preserving fit of (src_w,src_h) into (dst_w,dst_h)."""

class DecodeLayout(enum.Enum):
    YoloV8 = 0

    YoloV5 = 1

class DetectConfig:
    def __init__(self) -> None: ...

    @property
    def input_w(self) -> int: ...

    @input_w.setter
    def input_w(self, arg: int, /) -> None: ...

    @property
    def input_h(self) -> int: ...

    @input_h.setter
    def input_h(self, arg: int, /) -> None: ...

    @property
    def num_classes(self) -> int: ...

    @num_classes.setter
    def num_classes(self, arg: int, /) -> None: ...

    @property
    def conf_thresh(self) -> float: ...

    @conf_thresh.setter
    def conf_thresh(self, arg: float, /) -> None: ...

    @property
    def iou_thresh(self) -> float: ...

    @iou_thresh.setter
    def iou_thresh(self, arg: float, /) -> None: ...

    @property
    def max_dets(self) -> int: ...

    @max_dets.setter
    def max_dets(self, arg: int, /) -> None: ...

    @property
    def layout(self) -> DecodeLayout: ...

    @layout.setter
    def layout(self, arg: DecodeLayout, /) -> None: ...

    @property
    def channels_first(self) -> bool: ...

    @channels_first.setter
    def channels_first(self, arg: bool, /) -> None: ...

    @property
    def apply_sigmoid(self) -> bool: ...

    @apply_sigmoid.setter
    def apply_sigmoid(self, arg: bool, /) -> None: ...

class Detection:
    def __init__(self) -> None: ...

    @property
    def x1(self) -> float: ...

    @x1.setter
    def x1(self, arg: float, /) -> None: ...

    @property
    def y1(self) -> float: ...

    @y1.setter
    def y1(self, arg: float, /) -> None: ...

    @property
    def x2(self) -> float: ...

    @x2.setter
    def x2(self, arg: float, /) -> None: ...

    @property
    def y2(self) -> float: ...

    @y2.setter
    def y2(self, arg: float, /) -> None: ...

    @property
    def score(self) -> float: ...

    @score.setter
    def score(self, arg: float, /) -> None: ...

    @property
    def class_id(self) -> int: ...

    @class_id.setter
    def class_id(self, arg: int, /) -> None: ...

    def __repr__(self) -> str: ...

def decode(output: Annotated[NDArray[numpy.float32], dict(order='C')], config: DetectConfig, letterbox: LetterboxInfo) -> list[Detection]: ...

def nms(dets: Sequence[Detection], iou_thresh: float, max_dets: int = 300) -> list[int]: ...

def iou(a: Detection, b: Detection) -> float: ...

class Detector:
    def __init__(self, engine: Engine, config: DetectConfig, output_index: int = 0) -> None: ...

    def postprocess(self, letterbox: LetterboxInfo) -> list[Detection]: ...

    @property
    def config(self) -> DetectConfig: ...

class YoloLtrbConfig:
    def __init__(self) -> None: ...

    @property
    def num_classes(self) -> int: ...

    @num_classes.setter
    def num_classes(self, arg: int, /) -> None: ...

    @property
    def conf_thresh(self) -> float: ...

    @conf_thresh.setter
    def conf_thresh(self, arg: float, /) -> None: ...

    @property
    def iou_thresh(self) -> float: ...

    @iou_thresh.setter
    def iou_thresh(self, arg: float, /) -> None: ...

    @property
    def max_dets(self) -> int: ...

    @max_dets.setter
    def max_dets(self, arg: int, /) -> None: ...

    @property
    def strides(self) -> list[int]: ...

    @strides.setter
    def strides(self, arg: Sequence[int], /) -> None: ...

    @property
    def reg_max(self) -> int: ...

    @reg_max.setter
    def reg_max(self, arg: int, /) -> None: ...

class YoloLtrbDetector:
    def __init__(self, engine: Engine, config: YoloLtrbConfig, output_base: int = 0) -> None: ...

    def postprocess(self, letterbox: LetterboxInfo) -> list[Detection]: ...

    @property
    def config(self) -> YoloLtrbConfig: ...

class LabelMap:
    def __init__(self) -> None: ...

    @staticmethod
    def from_file(path: str) -> LabelMap: ...

    @staticmethod
    def from_list(names: Sequence[str]) -> LabelMap: ...

    @property
    def names(self) -> list[str]: ...

    @names.setter
    def names(self, arg: Sequence[str], /) -> None: ...

    def __len__(self) -> int: ...

    def name(self, class_id: int) -> str: ...

class ImageFormat(enum.Enum):
    Y = 0

    NV12 = 1

    RGB = 3

    BGR = 5

class VpImage:
    def __init__(self, width: int, height: int, format: ImageFormat) -> None:
        """Allocate a (width, height) image in `format` (cached device buffer)."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def format(self) -> ImageFormat: ...

    @property
    def valid(self) -> bool: ...

    def to_numpy(self) -> NDArray[numpy.uint8]:
        """
        Copy the image out to a uint8 numpy array honoring the device row stride: BGR/RGB -> (H,W,3), Y -> (H,W), NV12 -> flat (W*H*3/2) [Y plane then interleaved UV].
        """

def vp_image_from_bgr(bgr: Annotated[NDArray[numpy.uint8], dict(order='C')]) -> VpImage:
    """
    Build a BGR VpImage from an (H, W, 3) uint8 numpy array (copied, stride-honored, cache-flushed).
    """

def vp_image_from_nv12(nv12: Annotated[NDArray[numpy.uint8], dict(order='C')], width: int, height: int) -> VpImage:
    """
    Build an NV12 VpImage from a packed (W*H*3/2) uint8 buffer (stride-honored, cache-flushed).
    """

class JpegEncoder:
    def __init__(self, width: int, height: int, quality: int = 50, format: ImageFormat = ImageFormat.NV12) -> None:
        """Fixed-size JPU encoder. width must align to 16, height to 8."""

    def encode(self, image: VpImage) -> bytes:
        """Encode one VpImage to JPEG bytes."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def format(self) -> ImageFormat: ...

class JpegDecoder:
    def __init__(self, out_format: ImageFormat = ImageFormat.NV12) -> None: ...

    def decode(self, data: bytes) -> VpImage:
        """Decode JPEG bytes into an owned NV12 VpImage."""

    @property
    def out_format(self) -> ImageFormat: ...

class GdcLetterbox:
    def __init__(self, in_w: int, in_h: int, out_w: int, out_h: int, pad: int = 114) -> None:
        """
        Persistent GDC vnode for a fixed (in_w,in_h)->(out_w,out_h) hardware letterbox. The warp LUT is generated at construction.
        """

    def run(self, src: VpImage) -> VpImage:
        """GDC-letterbox an NV12 VpImage into a new out_w x out_h NV12 VpImage."""

    @property
    def info(self) -> LetterboxInfo: ...

    @property
    def output_width(self) -> int: ...

    @property
    def output_height(self) -> int: ...

class GdcRemap:
    def __init__(self, map_x: Annotated[NDArray[numpy.float32], dict(order='C')], map_y: Annotated[NDArray[numpy.float32], dict(order='C')], in_w: int, in_h: int, grid_step: int = 16) -> None:
        """
        Persistent GDC vnode for a FIXED dense warp with cv2.remap semantics (out(x,y) = in(map_x[y,x], map_y[y,x])). The (out_h, out_w) float32 maps are sampled every grid_step px into a CUSTOM warp grid whose LUT is generated at construction (hbn_gen_gdc_cfg) — no bin file.
        """

    def run(self, src: VpImage) -> VpImage:
        """Warp an NV12 VpImage into a new out_w x out_h NV12 VpImage."""

    @property
    def input_width(self) -> int: ...

    @property
    def input_height(self) -> int: ...

    @property
    def output_width(self) -> int: ...

    @property
    def output_height(self) -> int: ...

class VideoType(enum.Enum):
    H264 = 0

    H265 = 1

class VideoEncConfig:
    def __init__(self) -> None: ...

    @property
    def type(self) -> VideoType: ...

    @type.setter
    def type(self, arg: VideoType, /) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def bitrate_kbps(self) -> int: ...

    @bitrate_kbps.setter
    def bitrate_kbps(self, arg: int, /) -> None: ...

    @property
    def framerate(self) -> int: ...

    @framerate.setter
    def framerate(self, arg: int, /) -> None: ...

    @property
    def intra_period(self) -> int: ...

    @intra_period.setter
    def intra_period(self, arg: int, /) -> None: ...

    @property
    def format(self) -> ImageFormat: ...

    @format.setter
    def format(self, arg: ImageFormat, /) -> None: ...

class VideoEncoder:
    def __init__(self, config: VideoEncConfig) -> None: ...

    def encode(self, frame: VpImage) -> bytes:
        """
        Feed one NV12 frame and wait briefly for a packet; returns bytes (may be empty if the encoder buffered the frame).
        """

    def feed(self, frame: VpImage, pts_us: int = 0) -> bool:
        """
        Queue one NV12 frame for encoding (does not wait for output). Returns False if the codec's input queue is full — drain with receive() and retry rather than treating it as fatal.
        """

    def receive(self, timeout_ms: int = 0) -> bytes | None:
        """Drain one compressed packet, or None if none ready."""

    def flush(self) -> bytes | None:
        """
        Signal end-of-stream, then drain the remaining packets; call until it returns None.
        """

    def feed_end_of_stream(self) -> None:
        """Queue an end-of-stream marker without draining. Idempotent."""

    @property
    def type(self) -> VideoType: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def format(self) -> ImageFormat: ...

class VideoDecConfig:
    def __init__(self) -> None: ...

    @property
    def type(self) -> VideoType: ...

    @type.setter
    def type(self, arg: VideoType, /) -> None: ...

    @property
    def format(self) -> ImageFormat: ...

    @format.setter
    def format(self, arg: ImageFormat, /) -> None: ...

    @property
    def in_buf_size(self) -> int: ...

    @in_buf_size.setter
    def in_buf_size(self, arg: int, /) -> None: ...

class VideoDecoder:
    def __init__(self, config: VideoDecConfig) -> None: ...

    def decode(self, data: bytes) -> VpImage | None:
        """
        Feed one access unit; returns an NV12 VpImage when a frame is ready, else None (the decoder is still buffering reference frames).
        """

    def feed(self, data: bytes) -> bool:
        """Queue one access unit for decoding (does not wait for output)."""

    def receive(self, timeout_ms: int = 0) -> VpImage | None:
        """
        Drain one decoded frame in display order (timeout_ms=0 is non-blocking); None on timeout / no frame.
        """

    def flush(self) -> VpImage | None:
        """
        After the last feed(): drain the reorder tail; call until it returns None.
        """

    @property
    def type(self) -> VideoType: ...

    @property
    def format(self) -> ImageFormat: ...

class DepthConfig:
    def __init__(self) -> None: ...

    @property
    def width(self) -> int: ...

    @width.setter
    def width(self, arg: int, /) -> None: ...

    @property
    def height(self) -> int: ...

    @height.setter
    def height(self, arg: int, /) -> None: ...

    @property
    def normalize(self) -> bool: ...

    @normalize.setter
    def normalize(self, arg: bool, /) -> None: ...

    @property
    def clip_lo(self) -> float: ...

    @clip_lo.setter
    def clip_lo(self, arg: float, /) -> None: ...

    @property
    def clip_hi(self) -> float: ...

    @clip_hi.setter
    def clip_hi(self, arg: float, /) -> None: ...

class DepthMap:
    def __init__(self) -> None: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def vmin(self) -> float: ...

    @property
    def vmax(self) -> float: ...

    @property
    def data(self) -> NDArray[numpy.float32]:
        """
        Depth map as a float32 (H, W) numpy array (normalized to [0,1] when cfg.normalize, else raw).
        """

def decode_depth(output: Annotated[NDArray[numpy.float32], dict(order='C')], config: DepthConfig) -> DepthMap:
    """Decode a single-channel float depth/disparity tensor into a DepthMap."""

def depth_colorize(depth_map: DepthMap) -> NDArray[numpy.uint8]:
    """Turbo-colormap a DepthMap to a BGR (H, W, 3) uint8 numpy array."""

def depth_to_gray8(depth_map: DepthMap) -> NDArray[numpy.uint8]:
    """Min-max normalize a DepthMap to a grayscale (H, W) uint8 numpy array."""

class DepthEstimator:
    def __init__(self, engine: Engine, config: DepthConfig = ..., output_index: int = 0) -> None: ...

    def postprocess(self) -> DepthMap: ...

    @property
    def config(self) -> DepthConfig: ...

class StereoFit(enum.Enum):
    Resize = 0

    Crop = 1

class StereoConfig:
    def __init__(self) -> None: ...

    @property
    def input_w(self) -> int: ...

    @input_w.setter
    def input_w(self, arg: int, /) -> None: ...

    @property
    def input_h(self) -> int: ...

    @input_h.setter
    def input_h(self, arg: int, /) -> None: ...

    @property
    def fit(self) -> StereoFit: ...

    @fit.setter
    def fit(self, arg: StereoFit, /) -> None: ...

    @property
    def to_rgb(self) -> bool: ...

    @to_rgb.setter
    def to_rgb(self, arg: bool, /) -> None: ...

    @property
    def left_index(self) -> int: ...

    @left_index.setter
    def left_index(self, arg: int, /) -> None: ...

    @property
    def right_index(self) -> int: ...

    @right_index.setter
    def right_index(self, arg: int, /) -> None: ...

    @property
    def output_index(self) -> int: ...

    @output_index.setter
    def output_index(self, arg: int, /) -> None: ...

    @property
    def fx(self) -> float: ...

    @fx.setter
    def fx(self, arg: float, /) -> None: ...

    @property
    def baseline(self) -> float: ...

    @baseline.setter
    def baseline(self, arg: float, /) -> None: ...

    @property
    def valid_mask(self) -> bool: ...

    @valid_mask.setter
    def valid_mask(self, arg: bool, /) -> None: ...

    @property
    def disp_min(self) -> float: ...

    @disp_min.setter
    def disp_min(self, arg: float, /) -> None: ...

    @property
    def max_disp(self) -> float: ...

    @max_disp.setter
    def max_disp(self, arg: float, /) -> None: ...

    @property
    def left_margin(self) -> int: ...

    @left_margin.setter
    def left_margin(self, arg: int, /) -> None: ...

    @property
    def lr_check(self) -> bool: ...

    @lr_check.setter
    def lr_check(self, arg: bool, /) -> None: ...

    @property
    def lr_thresh(self) -> float: ...

    @lr_thresh.setter
    def lr_thresh(self, arg: float, /) -> None: ...

class StereoResult:
    @property
    def disparity(self) -> DepthMap: ...

    @property
    def depth(self) -> NDArray[numpy.float32]:
        """Metric depth (m) as float32 (H, W); shape (0,) when fx/baseline unset."""

    @property
    def valid(self) -> NDArray[numpy.uint8]:
        """Validity mask uint8 (H, W), 1=keep; shape (0,) when valid_mask off."""

def pack_stereo_input(bgr: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)], out_h: int, out_w: int, fit: StereoFit = StereoFit.Resize, to_rgb: bool = True) -> NDArray[numpy.float32]:
    """
    Fit (resize/crop) + channel-swap a BGR frame to a planar (3,H,W) float32.
    """

def disparity_to_depth(disparity: DepthMap, fx: float, baseline: float) -> NDArray[numpy.float32]:
    """
    Convert a disparity DepthMap to metric depth (m): z = fx*baseline/disp.
    """

def stereo_valid_mask(disparity: DepthMap, disp_min: float = 0.0, max_disp: float = 192.0, left_margin: int = 0, disp_right: DepthMap | None = None, lr_thresh: float = 1.5) -> NDArray[numpy.uint8]:
    """
    Geometry validity mask uint8 (H, W): disparity range + left-border + optional left-right consistency (pass disp_right).
    """

class StereoPipeline:
    def __init__(self, engine: Engine, config: StereoConfig = ...) -> None: ...

    def process(self, left: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)], right: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)]) -> StereoResult:
        """Run one stereo pair (HxWx3 uint8 BGR); returns a StereoResult."""

    @property
    def config(self) -> StereoConfig: ...

    @property
    def input_w(self) -> int: ...

    @property
    def input_h(self) -> int: ...

class SegConfig:
    def __init__(self) -> None: ...

    @property
    def num_classes(self) -> int: ...

    @num_classes.setter
    def num_classes(self, arg: int, /) -> None: ...

    @property
    def channels_first(self) -> bool: ...

    @channels_first.setter
    def channels_first(self, arg: bool, /) -> None: ...

    @property
    def argmaxed(self) -> bool: ...

    @argmaxed.setter
    def argmaxed(self, arg: bool, /) -> None: ...

class SegMask:
    def __init__(self) -> None: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def num_classes(self) -> int: ...

    @property
    def labels(self) -> NDArray[numpy.int32]:
        """Per-pixel class ids as an int32 (H, W) numpy array."""

def decode_seg(output: Annotated[NDArray[numpy.float32], dict(order='C')], config: SegConfig) -> SegMask:
    """Argmax a logit tensor (or pass through an id tensor) into a SegMask."""

def seg_colorize(seg_mask: SegMask) -> NDArray[numpy.uint8]:
    """
    Color a SegMask with a fixed palette -> BGR (H, W, 3) uint8 numpy array.
    """

class Segmenter:
    def __init__(self, engine: Engine, config: SegConfig = ..., output_index: int = 0) -> None: ...

    def postprocess(self) -> SegMask: ...

    @property
    def config(self) -> SegConfig: ...

class ClsConfig:
    def __init__(self) -> None: ...

    @property
    def top_k(self) -> int: ...

    @top_k.setter
    def top_k(self, arg: int, /) -> None: ...

    @property
    def apply_softmax(self) -> bool: ...

    @apply_softmax.setter
    def apply_softmax(self, arg: bool, /) -> None: ...

class ClsResult:
    @property
    def class_id(self) -> int: ...

    @property
    def score(self) -> float: ...

    def __repr__(self) -> str: ...

def decode_classification(logits: Annotated[NDArray[numpy.float32], dict(order='C')], config: ClsConfig) -> list[ClsResult]:
    """Top-k decode of a flat logit array."""

class Classifier:
    def __init__(self, engine: Engine, config: ClsConfig = ..., output_index: int = 0) -> None: ...

    def postprocess(self) -> list[ClsResult]: ...

    @property
    def config(self) -> ClsConfig: ...

class FaceDetection:
    @property
    def x1(self) -> float: ...

    @property
    def y1(self) -> float: ...

    @property
    def x2(self) -> float: ...

    @property
    def y2(self) -> float: ...

    @property
    def score(self) -> float: ...

    @property
    def landmarks(self) -> list[tuple[float, float]]: ...

    def __repr__(self) -> str: ...

class FaceDetectConfig:
    def __init__(self) -> None: ...

    @property
    def conf_thresh(self) -> float: ...

    @conf_thresh.setter
    def conf_thresh(self, arg: float, /) -> None: ...

    @property
    def iou_thresh(self) -> float: ...

    @iou_thresh.setter
    def iou_thresh(self, arg: float, /) -> None: ...

    @property
    def max_faces(self) -> int: ...

    @max_faces.setter
    def max_faces(self, arg: int, /) -> None: ...

    @property
    def strides(self) -> list[int]: ...

    @strides.setter
    def strides(self, arg: Sequence[int], /) -> None: ...

    @property
    def num_anchors(self) -> int: ...

    @num_anchors.setter
    def num_anchors(self, arg: int, /) -> None: ...

def decode_scrfd(score: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], bbox: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], kps: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], config: FaceDetectConfig, letterbox: LetterboxInfo) -> list[FaceDetection]:
    """Decode SCRFD score/bbox/kps tensors into faces with 5 landmarks."""

def arcface_template() -> list[tuple[float, float]]:
    """The canonical 112x112 five-point template used for face alignment."""

def similarity_transform(src: Sequence[tuple[float, float]], dst: Sequence[tuple[float, float]]) -> list[float]:
    """
    Closed-form Umeyama similarity transform (rot+uniform scale+translation) as {a,b,tx,c,d,ty}.
    """

def align_face(bgr: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)], landmarks: Sequence[tuple[float, float]], size: int = 112) -> NDArray[numpy.uint8]:
    """Warp a face into a size x size aligned BGR crop from its 5 landmarks."""

class FaceDetector:
    def __init__(self, engine: Engine, config: FaceDetectConfig = ..., output_base: int = 0) -> None: ...

    def postprocess(self, letterbox: LetterboxInfo) -> list[FaceDetection]: ...

    @property
    def config(self) -> FaceDetectConfig: ...

class WholeBodyCrop:
    def __init__(self) -> None: ...

    @property
    def x1(self) -> int: ...

    @x1.setter
    def x1(self, arg: int, /) -> None: ...

    @property
    def y1(self) -> int: ...

    @y1.setter
    def y1(self, arg: int, /) -> None: ...

    @property
    def pad_left(self) -> int: ...

    @pad_left.setter
    def pad_left(self, arg: int, /) -> None: ...

    @property
    def pad_top(self) -> int: ...

    @pad_top.setter
    def pad_top(self, arg: int, /) -> None: ...

    @property
    def padded_w(self) -> int: ...

    @padded_w.setter
    def padded_w(self, arg: int, /) -> None: ...

    @property
    def padded_h(self) -> int: ...

    @padded_h.setter
    def padded_h(self, arg: int, /) -> None: ...

class WholeBodyConfig:
    def __init__(self) -> None: ...

    @property
    def kpt_thresh(self) -> float: ...

    @kpt_thresh.setter
    def kpt_thresh(self, arg: float, /) -> None: ...

    @property
    def blur_kernel(self) -> int: ...

    @blur_kernel.setter
    def blur_kernel(self, arg: int, /) -> None: ...

    @property
    def box_pad(self) -> int: ...

    @box_pad.setter
    def box_pad(self, arg: int, /) -> None: ...

    @property
    def mean(self) -> list[float]: ...

    @mean.setter
    def mean(self, arg: Sequence[float], /) -> None: ...

    @property
    def std(self) -> list[float]: ...

    @std.setter
    def std(self, arg: Sequence[float], /) -> None: ...

def wholebody_preprocess(bgr: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)], x1: float, y1: float, x2: float, y2: float, in_w: int = 192, in_h: int = 256, config: WholeBodyConfig = ...) -> tuple:
    """
    Crop a person box to the model's canvas: widen by box_pad, zero-pad to the 3:4 aspect, resize, /255, ImageNet z-score, NCHW RGB. Returns (input, crop).
    """

def decode_wholebody(heatmaps: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)], crop: WholeBodyCrop, config: WholeBodyConfig = ...) -> list[Keypoint]:
    """
    Decode channel-first heatmaps into keypoints in original-image pixels (argmax + DARK-UDP sub-pixel refinement).
    """

class WholeBodyEstimator:
    def __init__(self, engine: Engine, config: WholeBodyConfig = ..., output_index: int = 0) -> None: ...

    def postprocess(self, crop: WholeBodyCrop) -> list[Keypoint]: ...

    @property
    def config(self) -> WholeBodyConfig: ...

class Feature:
    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def score(self) -> float: ...

class FeatureMatch:
    @property
    def a(self) -> int: ...

    @property
    def b(self) -> int: ...

    @property
    def score(self) -> float: ...

class XfeatConfig:
    def __init__(self) -> None: ...

    @property
    def detection_thresh(self) -> float: ...

    @detection_thresh.setter
    def detection_thresh(self, arg: float, /) -> None: ...

    @property
    def nms_kernel(self) -> int: ...

    @nms_kernel.setter
    def nms_kernel(self, arg: int, /) -> None: ...

    @property
    def top_k(self) -> int: ...

    @top_k.setter
    def top_k(self, arg: int, /) -> None: ...

class FeatureSet:
    @property
    def keypoints(self) -> list[Feature]: ...

    @property
    def dim(self) -> int: ...

    @property
    def descriptors(self) -> NDArray[numpy.float32]: ...

    @property
    def xy(self) -> NDArray[numpy.float32]: ...

    def __len__(self) -> int: ...

def xfeat_preprocess(bgr: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)], in_w: int = 640, in_h: int = 480) -> tuple:
    """
    Grayscale (channel MEAN, not luma) + resize + InstanceNorm into the model's [1,1,H,W] input. Returns (input, scale_x, scale_y).
    """

def decode_xfeat(feats: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)], keypoints: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)], heatmap: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)], config: XfeatConfig = ..., scale_x: float = 1.0, scale_y: float = 1.0) -> FeatureSet:
    """
    Decode the three XFeat maps into sparse features (softmax -> NMS -> top-k -> bilinear descriptor sampling), in original-image pixels.
    """

def match_features(a: FeatureSet, b: FeatureSet, min_cossim: float = 0.8199999928474426) -> list[FeatureMatch]:
    """Mutual nearest-neighbour matching with a cosine floor. O(|a|*|b|*dim)."""

class FeatureExtractor:
    def __init__(self, engine: Engine, config: XfeatConfig = ..., output_base: int = 0) -> None: ...

    def extract(self, bgr: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)], timeout_ms: int = 0) -> FeatureSet: ...

    def postprocess(self, scale_x: float = 1.0, scale_y: float = 1.0) -> FeatureSet: ...

    @property
    def config(self) -> XfeatConfig: ...

class SuperResConfig:
    def __init__(self) -> None: ...

    @property
    def overlap(self) -> int: ...

    @overlap.setter
    def overlap(self, arg: int, /) -> None: ...

class TilePlacement:
    @property
    def x(self) -> int: ...

    @property
    def y(self) -> int: ...

def plan_tiles(width: int, height: int, tile_w: int, tile_h: int, overlap: int) -> list[TilePlacement]:
    """
    Tile origins covering an image; the last tile on each axis is flush with the far edge, so the real overlap can exceed the requested one.
    """

def tile_weight(i: int, len: int, ramp: int) -> float:
    """
    Cross-fade weight along a tile axis. Always > 0, which is why the blend can normalize instead of special-casing image borders.
    """

class SuperResolver:
    def __init__(self, engine: Engine, config: SuperResConfig = ..., input_index: int = 0, output_index: int = 0) -> None: ...

    def upscale(self, bgr: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)], timeout_ms: int = 0) -> NDArray[numpy.uint8]: ...

    @property
    def scale(self) -> int: ...

    @property
    def tile(self) -> int: ...

    @property
    def last_tile_count(self) -> int: ...

    @property
    def config(self) -> SuperResConfig: ...

class Anchor:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, w: float, h: float) -> None: ...

    @property
    def w(self) -> float: ...

    @w.setter
    def w(self, arg: float, /) -> None: ...

    @property
    def h(self) -> float: ...

    @h.setter
    def h(self, arg: float, /) -> None: ...

class AnchorDetectConfig:
    def __init__(self) -> None: ...

    @property
    def num_classes(self) -> int: ...

    @num_classes.setter
    def num_classes(self, arg: int, /) -> None: ...

    @property
    def conf_thresh(self) -> float: ...

    @conf_thresh.setter
    def conf_thresh(self, arg: float, /) -> None: ...

    @property
    def iou_thresh(self) -> float: ...

    @iou_thresh.setter
    def iou_thresh(self, arg: float, /) -> None: ...

    @property
    def max_dets(self) -> int: ...

    @max_dets.setter
    def max_dets(self, arg: int, /) -> None: ...

    @property
    def strides(self) -> list[int]: ...

    @strides.setter
    def strides(self, arg: Sequence[int], /) -> None: ...

    @property
    def anchors(self) -> list[list[Anchor]]: ...

    @anchors.setter
    def anchors(self, arg: Sequence[Sequence[Anchor]], /) -> None: ...

def decode_yolov5_anchor(raw: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], config: AnchorDetectConfig, letterbox: LetterboxInfo) -> list[Detection]:
    """
    Decode raw anchor-based head tensors ([1,na*(5+nc),H,W] per scale, unactivated) into Detections in original-image pixels.
    """

class AnchorDetector:
    def __init__(self, engine: Engine, config: AnchorDetectConfig = ..., output_base: int = 0) -> None: ...

    def postprocess(self, letterbox: LetterboxInfo) -> list[Detection]: ...

    @property
    def config(self) -> AnchorDetectConfig: ...

class EmbedConfig:
    def __init__(self) -> None: ...

    @property
    def l2_normalize(self) -> bool: ...

    @l2_normalize.setter
    def l2_normalize(self, arg: bool, /) -> None: ...

class EmbedMatch:
    @property
    def index(self) -> int: ...

    @property
    def score(self) -> float: ...

    @property
    def label(self) -> str: ...

    def __repr__(self) -> str: ...

def decode_embedding(data: Annotated[NDArray[numpy.float32], dict(order='C')], config: EmbedConfig = ...) -> list[float]:
    """Read a flat float array as an embedding (L2-normalized by default)."""

class EmbeddingBank:
    def __init__(self) -> None: ...

    def add(self, vec: Sequence[float], label: str = '') -> None:
        """Append one entry; the vector is L2-normalized into the bank."""

    def search(self, query: Sequence[float], k: int = 5) -> list[EmbedMatch]:
        """Top-k cosine matches against the bank, most similar first."""

    def label(self, index: int) -> str: ...

    def __len__(self) -> int: ...

    @property
    def dim(self) -> int: ...

class ImageEmbedder:
    def __init__(self, engine: Engine, config: EmbedConfig = ..., output_index: int = 0) -> None: ...

    def postprocess(self) -> list[float]: ...

    @property
    def dim(self) -> int: ...

    @property
    def config(self) -> EmbedConfig: ...

class Keypoint:
    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def score(self) -> float: ...

class PoseDetection:
    @property
    def x1(self) -> float: ...

    @property
    def y1(self) -> float: ...

    @property
    def x2(self) -> float: ...

    @property
    def y2(self) -> float: ...

    @property
    def score(self) -> float: ...

    @property
    def class_id(self) -> int: ...

    @property
    def keypoints(self) -> list[Keypoint]: ...

class PoseConfig:
    def __init__(self) -> None: ...

    @property
    def num_keypoints(self) -> int: ...

    @num_keypoints.setter
    def num_keypoints(self, arg: int, /) -> None: ...

    @property
    def conf_thresh(self) -> float: ...

    @conf_thresh.setter
    def conf_thresh(self, arg: float, /) -> None: ...

    @property
    def iou_thresh(self) -> float: ...

    @iou_thresh.setter
    def iou_thresh(self, arg: float, /) -> None: ...

    @property
    def max_dets(self) -> int: ...

    @max_dets.setter
    def max_dets(self, arg: int, /) -> None: ...

    @property
    def strides(self) -> list[int]: ...

    @strides.setter
    def strides(self, arg: Sequence[int], /) -> None: ...

class PoseEstimator:
    def __init__(self, engine: Engine, config: PoseConfig = ..., output_base: int = 0) -> None: ...

    def postprocess(self, letterbox: LetterboxInfo) -> list[PoseDetection]: ...

    @property
    def config(self) -> PoseConfig: ...

def decode_pose(cls: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], box: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], kpt: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], config: PoseConfig, letterbox: LetterboxInfo) -> list[PoseDetection]:
    """
    Decode per-scale [H,W,1]/[H,W,4]/[H,W,K*3] float tensors -> PoseDetections.
    """

def normalize_embedding(embedding: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]) -> list[float]:
    """L2-normalize a 1-D appearance embedding -> list[float]."""

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length embeddings in [-1,1]."""

class SamConfig:
    def __init__(self) -> None: ...

    @property
    def mask_threshold(self) -> float: ...

    @mask_threshold.setter
    def mask_threshold(self, arg: float, /) -> None: ...

    @property
    def multimask(self) -> bool: ...

    @multimask.setter
    def multimask(self, arg: bool, /) -> None: ...

class SamMask:
    @property
    def index(self) -> int: ...

    @property
    def iou(self) -> float: ...

    @property
    def mask_w(self) -> int: ...

    @property
    def mask_h(self) -> int: ...

    @property
    def mask(self) -> NDArray[numpy.uint8]:
        """Selected low-res binary mask as a uint8 (H, W) numpy array (0/1)."""

class SamMaskDecoder:
    def __init__(self, engine: Engine, config: SamConfig = ..., masks_index: int = 0, iou_index: int = 1) -> None: ...

    def postprocess(self) -> SamMask: ...

    @property
    def config(self) -> SamConfig: ...

def decode_sam_masks(low_res_logits: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)], iou_pred: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)], config: SamConfig) -> SamMask:
    """
    Select+binarize the best SAM mask from [Nmask,Hm,Wm] logits + [Nmask] iou.
    """

class InstanceSegConfig:
    def __init__(self) -> None: ...

    @property
    def conf_thresh(self) -> float: ...

    @conf_thresh.setter
    def conf_thresh(self, arg: float, /) -> None: ...

    @property
    def iou_thresh(self) -> float: ...

    @iou_thresh.setter
    def iou_thresh(self, arg: float, /) -> None: ...

    @property
    def max_dets(self) -> int: ...

    @max_dets.setter
    def max_dets(self, arg: int, /) -> None: ...

    @property
    def strides(self) -> list[int]: ...

    @strides.setter
    def strides(self, arg: Sequence[int], /) -> None: ...

    @property
    def proto_index(self) -> int: ...

    @proto_index.setter
    def proto_index(self, arg: int, /) -> None: ...

    @property
    def compute_masks(self) -> bool: ...

    @compute_masks.setter
    def compute_masks(self, arg: bool, /) -> None: ...

    @property
    def reg_max(self) -> int: ...

    @reg_max.setter
    def reg_max(self, arg: int, /) -> None: ...

class InstanceMask:
    @property
    def x1(self) -> float: ...

    @property
    def y1(self) -> float: ...

    @property
    def x2(self) -> float: ...

    @property
    def y2(self) -> float: ...

    @property
    def score(self) -> float: ...

    @property
    def class_id(self) -> int: ...

    @property
    def mask_w(self) -> int: ...

    @property
    def mask_h(self) -> int: ...

    @property
    def mask(self) -> NDArray[numpy.uint8]:
        """
        Binary instance mask as a uint8 (H, W) numpy array (0/1), empty when compute_masks was false.
        """

class InstanceSegmenter:
    def __init__(self, engine: Engine, config: InstanceSegConfig = ..., output_base: int = 0) -> None: ...

    def postprocess(self, letterbox: LetterboxInfo, orig_w: int, orig_h: int) -> list[InstanceMask]: ...

    @property
    def config(self) -> InstanceSegConfig: ...

def decode_instance_seg(cls: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], box: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], mc: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], proto: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)], config: InstanceSegConfig, letterbox: LetterboxInfo, orig_w: int, orig_h: int) -> list[InstanceMask]:
    """
    Decode per-scale [H,W,nc]/[H,W,4|4*reg_max]/[H,W,np] + proto [mH,mW,np] -> InstanceMasks.
    """

class RotatedBox:
    @property
    def cx(self) -> float: ...

    @property
    def cy(self) -> float: ...

    @property
    def w(self) -> float: ...

    @property
    def h(self) -> float: ...

    @property
    def angle(self) -> float: ...

class ObbDetection:
    @property
    def rrect(self) -> RotatedBox: ...

    @property
    def score(self) -> float: ...

    @property
    def class_id(self) -> int: ...

class ObbConfig:
    def __init__(self) -> None: ...

    @property
    def num_classes(self) -> int: ...

    @num_classes.setter
    def num_classes(self, arg: int, /) -> None: ...

    @property
    def conf_thresh(self) -> float: ...

    @conf_thresh.setter
    def conf_thresh(self, arg: float, /) -> None: ...

    @property
    def iou_thresh(self) -> float: ...

    @iou_thresh.setter
    def iou_thresh(self, arg: float, /) -> None: ...

    @property
    def max_dets(self) -> int: ...

    @max_dets.setter
    def max_dets(self, arg: int, /) -> None: ...

    @property
    def strides(self) -> list[int]: ...

    @strides.setter
    def strides(self, arg: Sequence[int], /) -> None: ...

    @property
    def regularize(self) -> bool: ...

    @regularize.setter
    def regularize(self, arg: bool, /) -> None: ...

    @property
    def angle_offset_rad(self) -> float: ...

    @angle_offset_rad.setter
    def angle_offset_rad(self, arg: float, /) -> None: ...

    @property
    def angle_sign(self) -> int: ...

    @angle_sign.setter
    def angle_sign(self, arg: int, /) -> None: ...

class ObbDetector:
    def __init__(self, engine: Engine, config: ObbConfig = ..., output_base: int = 0) -> None: ...

    def postprocess(self, letterbox: LetterboxInfo) -> list[ObbDetection]: ...

    @property
    def config(self) -> ObbConfig: ...

def decode_obb(cls: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], box: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], angle: Sequence[Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]], config: ObbConfig, letterbox: LetterboxInfo) -> list[ObbDetection]:
    """
    Decode per-scale [H,W,nc]/[H,W,4]/[H,W,1] float tensors -> ObbDetections.
    """

def rotated_iou(a_cx: float, a_cy: float, a_w: float, a_h: float, a_angle: float, b_cx: float, b_cy: float, b_w: float, b_h: float, b_angle: float) -> float:
    """Rotated-rect IoU of two boxes (cx,cy,w,h,angle[rad])."""

class CameraIntrinsics:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None: ...

    @property
    def fx(self) -> float: ...

    @fx.setter
    def fx(self, arg: float, /) -> None: ...

    @property
    def fy(self) -> float: ...

    @fy.setter
    def fy(self, arg: float, /) -> None: ...

    @property
    def cx(self) -> float: ...

    @cx.setter
    def cx(self, arg: float, /) -> None: ...

    @property
    def cy(self) -> float: ...

    @cy.setter
    def cy(self, arg: float, /) -> None: ...

class Mono3dBox:
    @property
    def class_id(self) -> int: ...

    @property
    def score(self) -> float: ...

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def z(self) -> float: ...

    @property
    def h(self) -> float: ...

    @property
    def w(self) -> float: ...

    @property
    def l(self) -> float: ...

    @property
    def yaw(self) -> float: ...

    @property
    def alpha(self) -> float: ...

    @property
    def box2d(self) -> list[float]: ...

    def __repr__(self) -> str: ...

class Mono3dConfig:
    def __init__(self) -> None: ...

    @property
    def num_classes(self) -> int: ...

    @num_classes.setter
    def num_classes(self, arg: int, /) -> None: ...

    @property
    def conf_thresh(self) -> float: ...

    @conf_thresh.setter
    def conf_thresh(self, arg: float, /) -> None: ...

    @property
    def max_dets(self) -> int: ...

    @max_dets.setter
    def max_dets(self, arg: int, /) -> None: ...

    @property
    def nms_kernel(self) -> int: ...

    @nms_kernel.setter
    def nms_kernel(self, arg: int, /) -> None: ...

    @property
    def pred_2d(self) -> bool: ...

    @pred_2d.setter
    def pred_2d(self, arg: bool, /) -> None: ...

    @property
    def depth_ref(self) -> list[float]: ...

    @depth_ref.setter
    def depth_ref(self, arg: Sequence[float], /) -> None: ...

    @property
    def dim_ref(self) -> list[list[float]]: ...

    @dim_ref.setter
    def dim_ref(self, arg: Sequence[Sequence[float]], /) -> None: ...

def compute_mono3d_feature_xform(orig_w: int, orig_h: int, feat_w: int, feat_h: int) -> LetterboxInfo:
    """
    Build the original<->feature affine (scale-to-width, center-height) for SMOKE decode.
    """

def decode_mono3d(cls: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)], reg: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)], config: Mono3dConfig, feat_xform: LetterboxInfo, K: CameraIntrinsics) -> list[Mono3dBox]:
    """
    Decode channel-first cls[nc,H,W]/reg[8,H,W] SMOKE logits -> Mono3dBoxes.
    """

class Mono3dDetector:
    def __init__(self, engine: Engine, config: Mono3dConfig = ..., output_base: int = 0) -> None: ...

    def postprocess(self, orig_w: int, orig_h: int, K: CameraIntrinsics) -> list[Mono3dBox]: ...

    @property
    def config(self) -> Mono3dConfig: ...

class Track:
    @property
    def track_id(self) -> int: ...

    @property
    def x1(self) -> float: ...

    @property
    def y1(self) -> float: ...

    @property
    def x2(self) -> float: ...

    @property
    def y2(self) -> float: ...

    @property
    def score(self) -> float: ...

    @property
    def class_id(self) -> int: ...

    def __repr__(self) -> str: ...

class ReidConfig:
    def __init__(self) -> None: ...

    @property
    def mean(self) -> list[float]: ...

    @mean.setter
    def mean(self, arg: Sequence[float], /) -> None: ...

    @property
    def std(self) -> list[float]: ...

    @std.setter
    def std(self, arg: Sequence[float], /) -> None: ...

def reid_crop_preprocess(bgr: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)], x1: float, y1: float, x2: float, y2: float, in_w: int = 128, in_h: int = 256, config: ReidConfig = ...) -> NDArray[numpy.float32]:
    """
    Cut a detection box out of a BGR frame into the ReID model's input: squashing resize to in_w x in_h (NOT a letterbox), BGR->RGB, /255, ImageNet z-score, NCHW float32.
    """

class BoostConfig:
    def __init__(self) -> None: ...

    @property
    def rich_similarity(self) -> bool: ...

    @rich_similarity.setter
    def rich_similarity(self, arg: bool, /) -> None: ...

    @property
    def soft_biou(self) -> bool: ...

    @soft_biou.setter
    def soft_biou(self, arg: bool, /) -> None: ...

    @property
    def boost_detections(self) -> bool: ...

    @boost_detections.setter
    def boost_detections(self, arg: bool, /) -> None: ...

    @property
    def lambda_iou(self) -> float: ...

    @lambda_iou.setter
    def lambda_iou(self, arg: float, /) -> None: ...

    @property
    def lambda_mhd(self) -> float: ...

    @lambda_mhd.setter
    def lambda_mhd(self, arg: float, /) -> None: ...

    @property
    def lambda_shape(self) -> float: ...

    @lambda_shape.setter
    def lambda_shape(self, arg: float, /) -> None: ...

    @property
    def min_iou(self) -> float: ...

    @min_iou.setter
    def min_iou(self, arg: float, /) -> None: ...

    @property
    def dlo_alpha(self) -> float: ...

    @dlo_alpha.setter
    def dlo_alpha(self, arg: float, /) -> None: ...

    @property
    def vt_start(self) -> float: ...

    @vt_start.setter
    def vt_start(self, arg: float, /) -> None: ...

    @property
    def vt_end(self) -> float: ...

    @vt_end.setter
    def vt_end(self, arg: float, /) -> None: ...

    @property
    def vt_steps(self) -> int: ...

    @vt_steps.setter
    def vt_steps(self, arg: int, /) -> None: ...

    @property
    def duo(self) -> bool: ...

    @duo.setter
    def duo(self, arg: bool, /) -> None: ...

    @property
    def duo_iou(self) -> float: ...

    @duo_iou.setter
    def duo_iou(self, arg: float, /) -> None: ...

class ByteTrackConfig:
    def __init__(self) -> None: ...

    @property
    def track_thresh(self) -> float: ...

    @track_thresh.setter
    def track_thresh(self, arg: float, /) -> None: ...

    @property
    def high_thresh(self) -> float: ...

    @high_thresh.setter
    def high_thresh(self, arg: float, /) -> None: ...

    @property
    def match_thresh(self) -> float: ...

    @match_thresh.setter
    def match_thresh(self, arg: float, /) -> None: ...

    @property
    def track_buffer(self) -> int: ...

    @track_buffer.setter
    def track_buffer(self, arg: int, /) -> None: ...

    @property
    def frame_rate(self) -> int: ...

    @frame_rate.setter
    def frame_rate(self, arg: int, /) -> None: ...

    @property
    def proximity_thresh(self) -> float: ...

    @proximity_thresh.setter
    def proximity_thresh(self, arg: float, /) -> None: ...

    @property
    def appearance_thresh(self) -> float: ...

    @appearance_thresh.setter
    def appearance_thresh(self, arg: float, /) -> None: ...

    @property
    def ema_alpha(self) -> float: ...

    @ema_alpha.setter
    def ema_alpha(self, arg: float, /) -> None: ...

    @property
    def boost(self) -> BoostConfig: ...

    @boost.setter
    def boost(self, arg: BoostConfig, /) -> None: ...

class ByteTracker:
    def __init__(self, config: ByteTrackConfig = ...) -> None: ...

    @overload
    def update(self, detections: Sequence[Detection]) -> list[Track]:
        """Advance one frame with this frame's detections; returns active Tracks."""

    @overload
    def update(self, detections: Sequence[Detection], embeddings: Sequence[Sequence[float]]) -> list[Track]:
        """
        Advance one frame with detections AND their ReID embeddings (one entry per detection, parallel lists; an empty entry means 'no appearance for this detection'). Enables BoT-SORT appearance association: cost = min(IoU distance, gated cosine distance).
        """

    def apply_camera_motion(self, affine: Annotated[NDArray[numpy.float32], dict(order='C', writable=False)]) -> None:
        """
        Warp every tracklet by a camera-motion 2x3 affine mapping the PREVIOUS frame to this one, before the next update(). Only needed when the camera moves.
        """

    def reset(self) -> None: ...

    @property
    def config(self) -> ByteTrackConfig: ...

class DetectHead(enum.Enum):
    Auto = 0

    SingleTensor = 1

    YoloLtrb = 2

class PipelineConfig:
    def __init__(self) -> None: ...

    @property
    def input_w(self) -> int: ...

    @input_w.setter
    def input_w(self, arg: int, /) -> None: ...

    @property
    def input_h(self) -> int: ...

    @input_h.setter
    def input_h(self, arg: int, /) -> None: ...

    @property
    def detect(self) -> DetectConfig: ...

    @detect.setter
    def detect(self, arg: DetectConfig, /) -> None: ...

    @property
    def output_index(self) -> int: ...

    @output_index.setter
    def output_index(self, arg: int, /) -> None: ...

    @property
    def pad_value(self) -> int: ...

    @pad_value.setter
    def pad_value(self, arg: int, /) -> None: ...

    @property
    def head(self) -> DetectHead: ...

    @head.setter
    def head(self, arg: DetectHead, /) -> None: ...

    @property
    def ltrb_strides(self) -> list[int]: ...

    @ltrb_strides.setter
    def ltrb_strides(self, arg: Sequence[int], /) -> None: ...

class StageProfile:
    @property
    def decode_ms(self) -> float: ...

    @property
    def cvt_ms(self) -> float: ...

    @property
    def preproc_ms(self) -> float: ...

    @property
    def infer_ms(self) -> float: ...

    @property
    def postproc_ms(self) -> float: ...

    @property
    def frames(self) -> int: ...

    def total_ms(self) -> float: ...

    def decode_per_frame(self) -> float: ...

    def cvt_per_frame(self) -> float: ...

    def preproc_per_frame(self) -> float: ...

    def infer_per_frame(self) -> float: ...

    def postproc_per_frame(self) -> float: ...

class TrackingReidConfig:
    def __init__(self) -> None: ...

    @property
    def min_score(self) -> float: ...

    @min_score.setter
    def min_score(self, arg: float, /) -> None: ...

    @property
    def max_crops(self) -> int: ...

    @max_crops.setter
    def max_crops(self, arg: int, /) -> None: ...

    @property
    def crop(self) -> ReidConfig: ...

    @crop.setter
    def crop(self, arg: ReidConfig, /) -> None: ...

class TrackingPipeline:
    @overload
    def __init__(self, engine: Engine, det_config: PipelineConfig = ..., track_config: ByteTrackConfig = ...) -> None: ...

    @overload
    def __init__(self, engine: Engine, reid_engine: Engine, det_config: PipelineConfig = ..., track_config: ByteTrackConfig = ..., reid_config: TrackingReidConfig = ...) -> None: ...

    @property
    def has_reid(self) -> bool: ...

    @property
    def last_embed_count(self) -> int: ...

    def process(self, bgr: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)]) -> list[Track]:
        """Detect + track one HxWx3 uint8 BGR frame; returns list[Track]."""

    @property
    def last_detections(self) -> list[Detection]: ...

    def reset(self) -> None: ...

class AsyncDetectionPipeline:
    def __init__(self, engine: Engine, config: PipelineConfig = ..., depth: int = 3) -> None: ...

    def submit(self, bgr: Annotated[NDArray[numpy.uint8], dict(order='C', writable=False)]) -> bool:
        """
        Enqueue an HxWx3 uint8 BGR frame (bytes copied). Blocks while the pipeline is full; returns False if finish()ed (frame not accepted).
        """

    def next(self) -> list[Detection] | None:
        """
        Pop the next result in submission order (blocks until ready). Returns None once the pipeline is finished AND fully drained.
        """

    def finish(self) -> None:
        """
        Signal no more frames; next() drains the in-flight ones then returns None. Idempotent; also called on destruction.
        """

    def profile(self) -> StageProfile:
        """Per-stage service timing (StageProfile); read after finish() + drain."""

    @property
    def head(self) -> DetectHead: ...

class AsyncVideoDetectionPipeline:
    def __init__(self, engine: Engine, config: PipelineConfig = ..., codec: VideoType = VideoType.H264, depth: int = 4) -> None: ...

    def submit(self, data: bytes) -> bool:
        """
        Feed a chunk of Annex-B compressed bytes (segmented + VPU-decoded internally). Blocks on backpressure; False after finish().
        """

    def next(self) -> list[Detection] | None:
        """
        Pop the next frame's detections in decode order (blocks). None once finished AND fully drained.
        """

    def next_nowait(self) -> list[Detection] | None:
        """Non-blocking pop: detections if one is ready, else None. Never blocks."""

    def finish(self) -> None:
        """Signal end of stream; drains in-flight frames then next() ends."""

    def profile(self) -> StageProfile:
        """Per-stage timing incl. decode_ms (StageProfile); read after drain."""

class RecResult:
    @property
    def text(self) -> str: ...

    @property
    def score(self) -> float: ...

class ClsDirResult:
    @property
    def label(self) -> int: ...

    @property
    def score(self) -> float: ...

    @property
    def flip180(self) -> bool: ...

class TextBox:
    @property
    def x1(self) -> float: ...

    @property
    def y1(self) -> float: ...

    @property
    def x2(self) -> float: ...

    @property
    def y2(self) -> float: ...

    @property
    def score(self) -> float: ...

    @property
    def points(self) -> NDArray[numpy.float32]:
        """Rotated 4-point box (4,2) float, clockwise, original-image pixels."""

class DbConfig:
    def __init__(self) -> None: ...

    @property
    def bin_thresh(self) -> float: ...

    @bin_thresh.setter
    def bin_thresh(self, arg: float, /) -> None: ...

    @property
    def box_thresh(self) -> float: ...

    @box_thresh.setter
    def box_thresh(self, arg: float, /) -> None: ...

    @property
    def unclip_ratio(self) -> float: ...

    @unclip_ratio.setter
    def unclip_ratio(self, arg: float, /) -> None: ...

    @property
    def min_size(self) -> int: ...

    @min_size.setter
    def min_size(self, arg: int, /) -> None: ...

    @property
    def connectivity(self) -> int: ...

    @connectivity.setter
    def connectivity(self, arg: int, /) -> None: ...

def load_char_dict(path: str) -> list[str]:
    """Load a one-token-per-line OCR character dictionary."""

def decode_ctc(logits: Annotated[NDArray[numpy.float32], dict(order='C')], dict: Sequence[str]) -> RecResult:
    """CTC greedy decode of a (T,C) logit array."""

def decode_dbnet(prob: Annotated[NDArray[numpy.float32], dict(order='C')], config: DbConfig, letterbox: LetterboxInfo) -> list[TextBox]:
    """Connected-component + unclip decode of a DBNet (H,W) probability map."""

def decode_cls_dir(logits: Annotated[NDArray[numpy.float32], dict(order='C')], thresh: float = 0.8999999761581421) -> ClsDirResult:
    """Argmax a 0/180 text-direction logit vector -> ClsDirResult."""

class TextRecognizer:
    def __init__(self, engine: Engine, dict_path: str, output_index: int = 0) -> None: ...

    def postprocess(self) -> RecResult: ...

class TextAngleClassifier:
    def __init__(self, engine: Engine, thresh: float = 0.8999999761581421, output_index: int = 0) -> None: ...

    def postprocess(self) -> ClsDirResult: ...

class DbTextDetector:
    def __init__(self, engine: Engine, config: DbConfig = ..., output_index: int = 0) -> None: ...

    def postprocess(self, letterbox: LetterboxInfo) -> list[TextBox]: ...
