# BCDL Roadmap

BPU-native counterpart to ccdl. Reuse ccdl's upper-layer multi-task API design;
rewrite backend / media / memory on S100's `hbDNN` + `hbUCP` + `hb_vp`.

## Scope

BCDL is an **application / on-board runtime** library: load a compiled `.hbm`,
run inference, pre/post-process, media codec, and pipeline — all on the board.
**Model conversion is OUT OF SCOPE here.** ONNX→`.hbm`, PTQ calibration, and
cosine verification are done offline on an x86 host with the D-Robotics
OpenExplorer toolchain by *separate* projects; BCDL only consumes the finished
`.hbm`.

## OpenCV (board, mamba)

OpenCV 5 (`libopencv`) is installed in the board's `bcdl` conda env, so the
image-heavy ops use mature implementations: `cv::resize` (letterbox + instance
mask), `cv::findContours`/`cv::minAreaRect` (OCR DBNet rotated 4-point boxes),
`cv::rotatedRectangleIntersection` (OBB rotated NMS). Each is guarded by
`#ifdef BCDL_HAVE_OPENCV` with a hand-written fallback. `bgrToNv12Cpu` stays
hand-written (OpenCV has no direct NV12 output, and the hand path matches the
model's full-range BT.601). Build note: OpenCV needs conda's newer libstdc++ via
`DT_RPATH` (`-Wl,--disable-new-dtags`), handled globally in CMakeLists.

## Core invariant

The whole S100 compute+media stack is unified on:
- `hbUCPSysMem {phyAddr, virAddr, memSize}` — one shared buffer (BPU tensor, JPU/VPU image, VP buffer).
- `hbUCPTaskHandle_t` — one task; BPU infer / codec / resize all submit+wait via `hbUCP`.

=> zero-copy across JPU → VP → BPU → VPU is the default. Cache coherency is explicit
(`hbUCPMallocCached` + `hbUCPMemFlush` CLEAN/INVALIDATE).

## Layers & API mapping (verified against board headers)

| Layer | hobot lib | key C API | status |
|------|-----------|-----------|--------|
| core/ SysMem · Task · Status | libhbucp | hbUCPMalloc/MallocCached/MemFlush/Free; hbUCPSubmitTask/WaitTaskDone/ReleaseTask | **M0 done** |
| backend/ Engine | libdnn | hbDNNInitializeFromFiles · GetModelHandle · Get{Input,Output}TensorProperties · hbDNNInferV2 | **M0 done** |
| preproc/ VpImage·letterbox·cvtColor | hb_vp (libhbvp) | hbVPWarpAffine · hbVPResize · hbVPCvtColor | **M1 code done; hbVP is vDSP-backed & OFFLINE on board (see findings)** |
| preproc/ CPU letterbox + BGR→NV12 | OpenCV/OpenMP | letterboxCpu(cv::resize) · bgrToNv12Cpu(自写 full-range) | **done + validated (sync 254 FPS @1280×720)** |
| media/ JpegCodec | hb_vp (libhbvp) | hbVPCreateJPEG{Enc,Dec}Context · hbVPJPEG{Encode,Decode} · hbVPGetJPEG*OutputBuffer | **M2 done + validated on JPU** |
| media/ VideoCodec H.264/H.265 | hb_vp (libhbvp) | hbVPCreateVideo{Enc,Dec}Context · hbVPVideo{Encode,Decode} | **M2b done + validated on VPU** |
| pipeline/ DetectionPipeline (sync, buffer-reuse) | — | letterbox→NV12→infer→decode, zero per-frame alloc | **M4 done + validated (216 FPS @1280×720 on yolo26)** |
| pipeline/ AsyncDetectionPipeline (threaded overlap) | std::thread | preproc‖infer pipeline, bounded channels, FIFO | **M4 done + validated (334 FPS, 1.55× over sync; async_check PASS)** |
| pipeline/ TrackingPipeline (detect + MOT) | — | DetectionPipeline → ByteTracker, header-only compose; per-frame tracks (stable id) | **done + 端到端验证 (yolo26s + bus.jpg 平移序列, examples/track_demo)** |
| pipeline/ StereoPipeline (双目视差/深度) | — | 双图 F32 NCHW RGB 输入(resize/crop)→ disparity(decodeDepth)→ 可选 depth(fx·B/d)+ 有效性掩码(LR一致性); packStereoInputCHW 纯函数 | **done + validated (LAS2-M: 与参考 cosine 1.0/EPE 0px, 70FPS; examples/stereo_demo + test_stereo_py 6 pass)** |
| tasks/ detection (decode + NMS) | CPU | YOLOv8/v5 fused decode + per-class NMS + dequant | **M1 done + validated** |
| tasks/ detection LTRB multi-scale + DFL | CPU | anchor-free (cls,box)×stride decode；box=4 LTRB 或 box=4×reg_max DFL(softmax 期望)自动识别 | **done + validated (yolo26 LTRB + yolov8 DFL 均 bus.jpg: 1 bus + 4 person)** |
| backend/ output_reader | — | 零拷贝 F32 / dequant 输出读取（sigmoid），全 task 复用 | **done** |
| tasks/ classification | CPU | softmax + top-k | **done + validated (真模型 zebra; numpy 路径 test_classification 3 pass)** |
| tasks/ pose (keypoints) | CPU | LTRB box + N×3 keypoint decode + box NMS；纯函数 decodePose(Engine-free) | **done + validated (4 persons, 17 kpts) + numpy 路径(合成张量解码测试)** |
| tasks/ instance segmentation | CPU+OpenCV | LTRB box + mask_coef·proto + cv::resize crop；纯函数 decodeInstanceSeg(Engine-free) | **done + validated (3 person + 1 bus 原图 mask) + numpy 路径(合成张量解码测试)** |
| tasks/ obb (oriented box) | CPU+OpenCV | LTRB+angle + cv::rotatedRectangleIntersection NMS；纯函数 decodeObb(Engine-free) | **done + validated (rotated_iou 数值精确) + numpy 路径(合成张量解码测试)** |
| tasks/ ocr (text) | OpenCV/CPU | DBNet cv::findContours+minAreaRect **旋转四点框** · 方向 cls(0°/180° 翻转, decodeClsDir) · CTC；三段均 Engine-free 纯函数 | **done + 端到端验证 (最新 PP-OCRv5 真实 hbm 全三段 det→cls→rec, examples/ocr_demo + OcrTask) + numpy 路径(test_ocr: ctc/dbnet/cls_dir)** |
| tracks/ bytetrack (MOT) | 自写 | 卡尔曼 + 匈牙利 + 两阶段关联 + 状态机 (纯算法) | **done + validated (合成序列:id 稳定/漏检重关联)** |
| tasks/ depth + segmentation | CPU | decodeDepth/colorize · decodeSeg/argmax | **M3 done + validated (depth on real model)** |
| python/ bindings | nanobind | …上述全部 + ByteTracker + OCR(decode_ctc/dbnet, TextRecognizer/DbTextDetector) + TrackingPipeline(numpy BGR→tracks) + AsyncDetectionPipeline(submit/next/finish, GIL-released) + pose/obb/instance-seg numpy 路径(decode_pose/decode_obb/decode_instance_seg) | **done + validated (pose/obb/instance_seg/detection/depth_seg 35 pass + tracking_py 板上 2 pass + async_detection_py 板上 3 pass)** |
| core/ MemPool | libhbucp | pre-alloc + reuse, RAII Lease, thread-safe | **M4 done + validated (mempool_demo)** |

## Milestones

- **M0 (this skeleton)** — core + Engine + C++ example + python stub + Mac→board workflow.
  Verify: load a real `.hbm`, print I/O signature, infer zero input, report latency.
- **M1** — det task: VP letterbox preproc + ported NMS/decode; numpy Python path.
  Verify: cosine > 0.99 vs host ONNX on a real image.
- **M2** — media: JpegCodec + VideoCodec; zero-copy JPU→BPU (no memcpy).
- **M3** — seg + depth tasks under one `DetectionResult`-style API.
- **M4** — **done.** sync DetectionPipeline (buffer reuse) + MemPool + LTRB multi-scale
  head + threaded AsyncDetectionPipeline (preproc‖infer overlap). Perf on yolo26s
  @1280×720: zero-copy F32 postproc (10.4→0.68ms) + OpenMP preproc (enabled the
  existing pragmas) took sync 52→216 FPS; async overlap → 334 FPS (1.55×), within
  ~10% of the max(preproc,infer+decode) bound. async_check validates the threaded
  path returns sync-identical results in order. (An hbUCPSetTaskDoneCb true-async
  graph is a possible future refinement but the thread pipeline already hits the
  overlap bound.)
- ~~**M5** — host_toolchain/~~ **OUT OF SCOPE.** Conversion (hb_compile + PTQ calib
  + cosine verify, normalization fused into the `.hbm`) lives in the offline host
  projects, not in this runtime repo. See Scope above.

## Candidate next work (application/runtime — pick per need)

- **Live camera → display path** (in progress, video-first):
  - **Step 1 — video-file source (硬解码)**: **done + validated.** VPU H.264/H.265
    hardware decode of a real elementary stream as a stand-in for live capture.
    `examples/video_decode.cc` reads an Annex-B file, splits it into access units
    (start-code scan + VCL-NAL boundary), feeds `VideoDecoder` (VPU), counts
    decoded frames + FPS, and dumps frames to JPEG (JPU) to verify pixels.
    `tests/test_video_decode_py.py` decodes a board `.h264` and asserts real frames
    out. Validated on board: `640x480_30fps.h264` 25/25 frames @480 FPS, and
    `1080P_test.h264` 186/186 frames @231 FPS (decoded NV12 1920×**1088** — VPU
    aligns height to 16; JPU dumps are valid 640×480 / 1920×1088 JPEGs). This is
    the decode half of the eventual camera path (decode → pipeline → encode).
  - **Step 1b — video → detect end-to-end**: **done + validated.**
    `examples/video_det_demo.cc` chains VPU decode → NV12→BGR (`cv::cvtColor`) →
    `DetectionPipeline` (BPU) → annotated JPEG. On `1080P_test.h264` + yolo26s
    det: 30 frames, decode 4.4 ms/f + detect 4.9 ms/f = **107 FPS end-to-end**,
    295 detections. Proves the decode→infer path the live camera will reuse.
  - **TODO — VIN/MIPI camera capture**: real camera ingest (the `hbVP`/`sp_dev`
    VIN/ISP units) + VOT/HDMI output, wired to the pipeline, for a fully live demo.
    Deferred — bring up after the video-file decode path is validated end-to-end.
- ~~**More task decoders**: pose / OBB / classification / OCR / semantic-seg /
  depth heads~~ **DONE** — all validated on real `.hbm`s (finding #7); detection
  now also handles the DFL head (yolov8/v10/v11), not just LTRB.
- ~~**Seg argmax perf**: deeplabv3plus full-res `[1024,2048,19]` argmax is ~662ms
  (single-thread CPU).~~ **DONE (CPU-side, no model change).** `Segmenter::
  postprocess()` now has an F32 fast path (`argmaxF32`) that argmaxes over the
  channel axis **directly on the device buffer** — honoring byte strides, OpenMP
  over rows, with no 40M-float materialization and no per-element index
  decomposition (the old `readElement`-per-element walk was the bottleneck, not
  the argmax). This is the official `np.argmax(axis=-1)` pattern. Result: full-
  res seg decode **~660ms → ~8.7ms (73× faster)**, byte-identical labels to the
  old path; semseg e2e 660→58ms (decode FPS 1.5→17). The materialize+`decodeSeg`
  fallback stays for F16 / quantized / pre-argmaxed outputs. (Fusing ArgMax into
  the `.hbm` offline remains a further option but needs a re-convert; not done.)
- ~~**Stereo / depth pipeline**: bind the LAS2 `.hbm` into a stereo pipeline
  mirroring DetectionPipeline (two-image input feed).~~ **DONE + validated.**
  `StereoPipeline` (`pipeline/stereo_pipeline.{h,cc}`) feeds two F32 NCHW RGB
  images (left->input0 / right->input1), decodes the disparity output via the
  existing `decodeDepth` (normalize off), and optionally derives metric depth
  (`z=fx·B/d`) + a geometry validity mask (disparity range + left-border +
  optional left-right consistency via a flipped second pass). Preproc is
  `packStereoInputCHW` (pure, Engine-free, testable): resize **or** center-crop
  (must match the model's calibration mode) + BGR→RGB + planar pack; norm is
  fused into the `.hbm`. Validated on LAS2-M (`las2_m_int16_nashm.hbm`,
  in[0/1]=`[1,3,480,640]` f32, out=`[1,1,480,640]` f32, 70 FPS): the pipeline's
  disparity is **byte-identical (cosine 1.0, EPE 0px)** to the LAS2 reference
  preproc (`las2-s100p/board/infer.py`) on `zed_test.png` (resize, mean 9.648);
  crop mode on the indoor pair gives disparity [44,129]px (mean 93.36 = fp32),
  79.7% valid (LR-check), depth 0.65–1.83 m. `examples/stereo_demo.cc`
  (side-by-side vis) + python `bcdl.StereoPipeline`/`pack_stereo_input`/
  `disparity_to_depth`/`stereo_valid_mask`. Tests: `tests/test_stereo_py.py`
  (6 numpy-path) + `tests/test_stereo_board_py.py` (3 on-board end-to-end on the
  repo-local `data/images/stereo_{left,right}.png` pair: sane disparity, depth +
  validity-mask geometry, and a cv2-reference preproc cross-check cosine>0.9999).
- **Public API & packaging**: stabilize headers, `install()` targets, a versioned
  C++/Python API surface, usage docs.
- ~~**Async Python binding**: expose AsyncDetectionPipeline submit/next to
  Python.~~ **DONE + validated.** `bcdl.AsyncDetectionPipeline(engine, cfg,
  depth)` with `submit(bgr)` / `next() -> list|None` / `finish()`; both blocking
  calls **release the GIL** so the C++ preproc‖infer threads overlap the Python
  loop. `tests/test_async_detection_py.py` (3 pass on board) mirrors
  `async_check.cc` — interleaved real/black frames assert FIFO order + no
  loss/dup, plus post-finish submit rejection and bad-shape guarding; loads the
  test image via bcdl's own JPU `JpegDecoder` (+ numpy NV12→BGR), so it needs no
  cv2. Python throughput on yolo26s @816×1088: **290 FPS** (vs C++ async 334;
  gap is per-call Python overhead).
- (Optional) `hbUCPSetTaskDoneCb` true-async graph — thread pipeline already hits
  the overlap bound, so low priority.

## On-board findings (validated 2026-06, S100P sw 4.0.5)

Build + run validated on the RDK S100P board inside the `bcdl` conda env. Examples:
`det_demo`, `jpeg_roundtrip`, `video_roundtrip`, `depth_demo`. Tests:
`tests/test_detection.py` (12 pass), `tests/test_depth_seg.py` (13 pass).

1. **hbVP geometric ops are vDSP-backed and the vDSP is OFFLINE / root-only.**
   `hbVPWarpAffine`/`hbVPResize`/`hbVPCvtColor` run on the vector DSP; `/dev/vdsp0`
   is `crw------- root root` and the core reports OFFLINE even under sudo for a
   short-lived process. WarpAffine also only accepts Y/NV12 source (not BGR).
   => The hbVP `letterbox()` code is kept (zero-copy goal for M4 camera pipeline,
   gated on bringing the vDSP online) but the **default preprocessing is CPU**
   (`letterboxCpu` + `bgrToNv12Cpu`), matching the proven RDK deploy pattern
   (OpenCV resize + copyMakeBorder + BGR→NV12). Python path uses cv2 + `bcdl.decode`.
2. **JPU / VPU codecs work without root.** `/dev/jpu` (`jpu` group), `/dev/vpu`
   (`vpu` group) — user is in both. JPEG roundtrip: Y-plane MAE 0.15 @ q90.
   H.264: 12-frame encode→Annex-B stream (`file` = "H.264 video @ L 30")→decode
   recovered all 12 frames.
3. **JPU JPEG encode requires width%16==0 and height%8==0** (enforced in
   `JpegEncoder` ctor with a clear error). Video encode: width%32, height%8.
4. **Chained pipeline validated**: depth model (Depth-Anything-V2, `[1,518,686]`
   f32) → `DepthEstimator` → `depthColorize` → `bgrToNv12Cpu` → JPU JPEG → file.
5. **Standard RDK YOLO NV12 export is 2-input / 6-output, NOT single-tensor.**
   `yolo26s_det_nashm_640x640_nv12.hbm`: inputs = Y `[1,640,640,1]` u8 (409600 B) +
   UV `[1,320,320,2]` u8 (204800 B); outputs = 3 scales × (cls `[1,H,W,80]`, box
   `[1,H,W,4]`) for strides 8/16/32. Head is **anchor-free LTRB** (not DFL, no
   objectness): `x1=(gx+0.5−l)·s`, `y2=(gy+0.5+b)·s`, score=sigmoid(max logit).
   => `DetectionPipeline` feeds Y→input0 / UV→input1 (both views of one NV12
   buffer, no extra copy) and auto-selects `YoloLtrbDetector`. Validated on
   `bus.jpg`: 1 bus + 4 person, matching the rdk_model_zoo reference runtime.
   The fused single-tensor `Detector` (kYoloV8/V5) remains for that head family.
6. **OCR end-to-end validated on pre-installed PP-OCRv3 `.hbm`** (no conversion
   needed). The board ships `/opt/hobot/model/s100/basic/cn_PP-OCRv3_det_…_640x640_nv12.hbm`
   (DBNet: 2-input NV12, output prob map `[1,1,640,640]` **S16+SCALE** → `outputAsFloat`
   dequantises to [0,1], no sigmoid) and `…_rec_…_48x320_rgb.hbm` (CRNN+CTC: **F32 RGB
   NCHW [1,3,48,320]**, output `[1,40,6625]` F32; rec preproc is just `/255` + BGR→RGB,
   no mean/std). Dict = `data/ppocr_keys_v1_6625.txt` = `blank` + PaddleOCR
   `ppocr_keys_v1.txt` (6623) + trailing space = 6625 classes. `examples/ocr_demo`
   runs det→`DbTextDetector`→crop→`TextRecognizer` and recognises Chinese text
   matching the `rdk_model_zoo/paddle_ocr` reference. **Gotcha:** the text-line crop
   must order its 4 corners itself (PaddleOCR `get_rotate_crop_image`); using
   `cv::minAreaRect`+angle transposes crops under OpenCV 5 (angle/size convention
   differs from OpenCV 4) → vertical crops → all-blank rec.
   **OCR now runs the LATEST PP-OCRv5 stack (all 3 stages converted offline).**
   The board ships only PP-OCRv3 det+rec, so the newer **PP-OCRv5** server det/rec
   + **PP-LCNet** textline direction cls were converted offline from the PaddleOCR
   ONNX (OE v3.7.0 `hb_compile`) and deployed to the board's `models/`. All three
   are **featuremap F32 RGB
   NCHW** (norm in preproc, not fused): det `[1,3,960,960]`→prob map `[1,1,960,960]`
   (ImageNet norm); rec `[1,3,48,320]`→`[1,40,18385]` softmax CTC (`(x/255−0.5)/0.5`,
   **18385-class v5 dict** `data/ppocr_keys_v5_18385.txt`); cls `[1,3,80,160]`→`[1,2]`
   (ImageNet norm). **Conversion gotcha:** the PaddleOCR ONNX choke hb_compile's
   opset 17→19 shape-inference (Constant/Reshape rank mismatch) — fix by simplifying
   with **onnxslim** (det/cls) or **onnxsim** (rec head) at a fixed input shape first.
   Both `examples/ocr_demo` and the python `OcrTask` run det→**cls (0°/180° flip)**→rec
   (cls OPTIONAL, env `BCDL_OCR_*`). v5 is far more accurate than v3 (e.g. `产品信息/参数`
   keeps the slash, `燕麦β-葡聚` keeps `β-`, `泛醌` vs v3's wrong `泛配`) at a speed cost
   (det 20 ms @960 server vs v3 1.2 ms; ~137 ms full 16-line decode). Validated:
   `test_ocr_direction_cls` (upright not flagged, 180° flagged) + `test_ocr_two_stage`.
   **Preprocessing aligned to ccdl/PaddleOCR source** (`get_rotate_crop_image` /
   `order_points_clockwise` / `resize_norm_img`): (a) the tall-crop rotate is
   **counter-clockwise** (`np.rot90`) — a clockwise rotate left vertical lines
   upside-down and the cls couldn't always recover them (e.g. "ODM OEM" → garbage);
   (b) box corners use **order_points_clockwise** (sort-by-x, split-by-y) not the
   x+y/y-x heuristic, so slanted boxes rectify/draw correctly; (c) rec/cls use
   **aspect-ratio-preserving resize + pad** (not anamorphic stretch), with rec/cls
   re-calibrated on matching crops. The check figure (`OcrTask.draw`) renders text
   back into each box direction-aware (vertical + slanted). Models the harness
   uses are copied into the repo-local `models/` dir (populated by
   `scripts/fetch_models.sh`); code references them by relative path (no absolute
   paths), env-overridable.
7. **Real-model board test + benchmark suite (9 tasks).** `tests/board_models.py`
   (shared preproc + per-task decode/draw) drives `tests/test_board_models.py`
   (pytest, 10 pass; full suite 74 pass + 1 skip) and `scripts/board_bench.py`
   (latency/FPS + isolated per-model RSS + check figures →
   `benchmarks/{RESULTS.md,results.json,figures/}`). Models: official **YOLO26n
   nash-m** (`ultralytics_yolo26/model/nash-m/yolo26n_{cls,detect,pose,seg,obb}`,
   the LTRB family — no conversion) for cls/det/pose/instance-seg/obb; **yolov8
   (det_dfl)** for the DFL head (box=64=4×16 → YoloLtrbDetector auto-detects
   reg_max, softmax-expectation per side); board `deeplabv3plus` (semantic seg,
   NV12 [1,H,W,19] NHWC → `SegConfig.channels_first=False`); downloaded
   `depth_any.hbm` (Depth-Anything-V2, **single F32 NCHW RGB [1,3,518,686]** +
   ImageNet norm — NOT NV12); pre-installed PP-OCRv3 (2-stage `OcrTask`). S100P
   infer: cls 0.43ms/2306FPS, det(LTRB) 1.21ms, det_dfl 1.52ms/658FPS, pose
   1.34ms, seg 1.59ms(+10.8ms mask), obb 1.18ms(96 DOTA boxes), semseg
   49.6ms/20FPS(+~8.7ms OpenMP channel-argmax, was +662ms), depth 112.9ms/9FPS, ocr det 1.24ms
   (3 中文 lines). Plus a JPEG-decode software(cv2) vs hardware(JPU) compare:
   reused `JpegDecoder` is **3.9–5.2× faster** than cv2 (one-shot `jpeg_decode`
   helper isn't — it rebuilds the JPU context each call). **JPU fix:** non-4:2:0
   JPEGs (4:2:2/4:4:4) used to segfault the JPU firmware at larger sizes — now
   parsed up front and rejected with a clean `bcdl::Error` (decode in software).
   **Check figures** now use repo-local sample images + a CJK font copied from
   ccdl (`data/images/{bus,zidane,ocr,obb,bird}.jpg`, `data/fonts/SourceHanSansSC-Regular.otf`)
   so they're reproducible without board-only `/app/res/assets` (cls/zebra +
   semseg/cityscapes have no ccdl match and stay on the board). Drawing was
   redone: per-class palette + filled-background labels rendered via
   `cv2.freetype`, pose skeleton bones, and a PaddleOCR-style side-by-side OCR
   figure with the recognised **Chinese** text warped back into each box.

## Decisions locked

1. Build natively on board (headers/libs already there); cross-compile later if needed.
2. Cache: outputs `MallocCached` + INVALIDATE before CPU read (in Engine). Never bypass.
3. Normalization fused into `.hbm` when possible; `hb_vp` only for geometric/colorspace ops.
4. M0–M3 synchronous (`WaitTaskDone`); async callback graph only at M4.
5. No on-board quantization/engine-building — conversion is offline on the x86 host.

## Risks

- VP resize/cvtColor mostly operate in NV12; letterbox padding + stride alignment must
  match the host ONNX preprocessing or accuracy drops — gate M1 on cosine verification.
- Codec output buffer ownership vs MemPool: avoid double-free.
- Large-resolution seg argmax/mask on A78AE may bottleneck — OpenMP + fixed-point if so.
