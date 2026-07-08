"""BCDL — BPU Computational Deep Learning (C++ core)"""

from collections.abc import Sequence
import enum
from typing import Annotated

import numpy
from numpy.typing import NDArray


class Engine:
    def __init__(self, hbm_path: str, model_name: str = '') -> None: ...

    @property
    def model_name(self) -> str: ...

    @property
    def num_inputs(self) -> int: ...

    @property
    def num_outputs(self) -> int: ...

    def input_shape(self, index: int) -> list[int]: ...

    def output_shape(self, index: int) -> list[int]: ...

    def input_bytes(self, index: int) -> int: ...

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
        Encode one NV12/YUV420 frame to compressed bytes (may be empty if the encoder buffered the frame).
        """

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
        Feed one compressed chunk; returns an NV12 VpImage when a frame is ready, else None (the decoder is still buffering reference frames).
        """

    def feed(self, data: bytes) -> bool:
        """Queue one access unit for decoding (does not wait for output)."""

    def receive(self, timeout_ms: int = 0) -> VpImage | None:
        """Drain one decoded frame in display order (timeout_ms=0 is non-blocking); None on timeout / no frame."""

    def flush(self) -> VpImage | None:
        """After the last feed(): drain the reorder tail; call until it returns None."""

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
    Decode per-scale [H,W,nc]/[H,W,4]/[H,W,np] + proto [mH,mW,np] -> InstanceMasks.
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

class ByteTracker:
    def __init__(self, config: ByteTrackConfig = ...) -> None: ...

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        """Advance one frame with this frame's detections; returns active Tracks."""

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
    def preproc_ms(self) -> float: ...

    @property
    def infer_ms(self) -> float: ...

    @property
    def postproc_ms(self) -> float: ...

    @property
    def frames(self) -> int: ...

    def total_ms(self) -> float: ...

    def decode_per_frame(self) -> float: ...

    def preproc_per_frame(self) -> float: ...

    def infer_per_frame(self) -> float: ...

    def postproc_per_frame(self) -> float: ...

class TrackingPipeline:
    def __init__(self, engine: Engine, det_config: PipelineConfig = ..., track_config: ByteTrackConfig = ...) -> None: ...

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
    def __init__(self, engine: Engine, config: PipelineConfig = ..., codec: VideoType = ..., depth: int = 4) -> None: ...

    def submit(self, data: bytes) -> bool:
        """Feed Annex-B compressed bytes (segmented + VPU-decoded internally). Blocks on backpressure; False after finish()."""

    def next(self) -> list[Detection] | None:
        """Blocking pop of the next frame's detections in decode order; None once finished AND drained."""

    def next_nowait(self) -> list[Detection] | None:
        """Non-blocking pop: detections if one is ready, else None."""

    def finish(self) -> None:
        """Signal end of stream; drains in-flight frames then next() ends."""

    def profile(self) -> StageProfile:
        """Per-stage timing incl. decode_ms (VPU decode + NV12->BGR)."""

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
