# Changelog

All notable changes to BCDL are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- This file ships verbatim with the published package: there is ONE copy, and
     it is written at a publishable register. Keep it free of local filesystem
     paths, machine names, customer references, and links to documents that are
     not published alongside it. Detail that cannot meet that bar belongs in the
     roadmap/design notes, not here. -->

## [0.4.0] — 2026-07

### Added
- **Appearance-aware multi-object tracking.** `ByteTracker::update()` gains an
  overload taking one ReID embedding per detection, and the first association's
  cost becomes `min(IoU distance, gated cosine distance)`. Taking the MINIMUM is
  what makes it safe: appearance can only lower a cost, so a bad embedding can
  never break a match that geometry already had — it can only rescue one that
  geometry was about to miss. Three gates keep it from inventing matches: a
  proximity gate on the RAW IoU distance (computed before score fusion, or a
  confident detection would widen its own spatial gate), a cosine gate, and a
  class gate (there is no shared appearance space across classes). The low-score
  second association stays pure IoU, since those crops are the occluded and
  blurred ones whose embeddings are worth least.

  Tracklet templates use a confidence-adaptive EMA: the smoothing factor rises
  toward 1 — no update at all — as a detection's score falls, so a half-occluded
  crop cannot poison what the target is remembered as looking like.

  An EMPTY embedding means "no appearance for this detection", which is the
  supported way to skip the ReID model on cheap crops. The single-argument
  `update()` is unchanged, bit for bit.
- **Person ReID.** New `tracks/reid.h` / `src/tracks/reid.cc`: `reidPreprocess()`
  cuts a detection box to the model's 256x128 input (a squashing resize, NOT a
  letterbox — these models are trained on squashed crops). Read-out reuses the
  embedding task. Python: `bcdl.ReIDExtractor`, `bcdl.reid_preprocess`,
  `bcdl.reid_crop_preprocess`. The Python preprocessing binds to the C++ one so
  the two paths cannot drift into feeding the model different pixels.
  **0.84 ms per crop** in a 6.6 MB model.
- **`TrackingPipeline` takes an optional ReID Engine**, so the native C++ path
  gets appearance too. Crop size is read from the model. `lastEmbedCount()`
  exposes the term that makes frame time scale with crowd size — the ReID model
  runs once per crop, so on a busy frame it, not the detector, sets the pace;
  `TrackingReidConfig` caps it.
- **BoostTrack++ additions**, each behind its own switch and all off by default
  (`BoostConfig`): soft buffered IoU, which grows both boxes in proportion to
  `1 - tracklet confidence` so a tracklet that has been coasting searches wider;
  Mahalanobis and shape-agreement terms in the association cost; and DLO/DUO
  detection-confidence boosting, which raises the scores of detections the
  tracklets vouch for BEFORE the high/low split.

  Measured per switch rather than shipped on faith. On a sequence where the
  detector performs normally, soft-BIoU and the similarity terms are small
  consistent wins (IDF1 0.510 to 0.554, with MOTA up); detection boosting is
  **situational** — there it costs 0.084 MOTA by turning detector noise into new
  tracks, but on a detection-starved sequence it is the only one of the three
  that helps at all. Which way it goes depends on whether your detector is
  limited by misses or by false positives, which is why it defaults to off.
- **Camera-motion compensation.** `ByteTracker::applyCameraMotion(affine)` warps
  every tracklet by a 2x3 transform before the next update. Position and size
  are warped, velocity is not: velocity is a property of the target in the
  world, and a one-frame camera jolt should not be read as the target
  accelerating. The transform comes from the caller because this class only ever
  sees boxes; Python takes whatever OpenCV's `estimateAffinePartial2D` returns.
  Under camera motion that pushes identity switches from 15 to 38, compensating
  recovers the static-camera result exactly, and it costs nothing when the
  camera is still.
- **A second x4 upscaler (SPAN), and it is not a replacement for the first.**
  Adding it needed no library code at all — `SuperResolver` reads scale and tile
  from the model, so a different upscaler is a path change. The two are
  complementary rather than ranked: SPAN is fidelity-oriented, wins on a clean
  downscale (32.95 dB against bicubic's 32.29 and the Compact model's 30.01),
  and is **5.8 MB at 1.13 ms/tile**; the perceptual, degradation-trained Compact
  model wins on blurred and JPEG-compressed input (28.54 vs 27.60 dB). Choose by
  what your input looks like.
- **x4 super-resolution, applied by tiling.** New `tasks/superres.{h,cc}`:
  `planTiles()`, `tileWeight()` and `SuperResolver`. Python:
  `bcdl.SuperResolver`, `bcdl.plan_tiles`, `bcdl.tile_weight`. The model
  upscales one fixed tile; an arbitrary image is cut into overlapping tiles and
  cross-faded back together. **2.04 ms per tile**; a 202x270 frame becomes
  808x1080 in 27.7 ms.

  The blend is a weighted average — each tile contributes `w * pixel` and `w` to
  two accumulators and the result is their ratio — which is what lets the image
  border need no special case: a border pixel is covered by one tile, divides by
  that tile's own weight, and comes back unchanged.

  **Pick the tile size with the `.hbm` in mind.** The compiled instruction
  stream scales with tile AREA, not with the weights: this net is 1.2 MB of int8
  weights, but compiling it for 256x256 yields a **147.9 MB** file where 128x128
  yields **37.1 MB** and 64x64 yields **11.4 MB**. Per-pixel throughput is
  identical across all three (0.126 vs 0.127 us per input pixel), so where you
  tile in software anyway a smaller model tile is nearly free — the only cost is
  that a fixed overlap is a larger share of a smaller tile. The shipped model
  uses 128.

  On what to expect from it: this is a perceptual, GAN-trained upscaler, so on a
  clean downscale it scores *below* bicubic on PSNR (20.5 vs 21.0 dB) while
  carrying visibly more detail — 0.56x of the ground truth's gradient energy
  against bicubic's 0.40x. On the degraded input it is actually meant for (blur
  plus JPEG q40) it wins on PSNR too, 28.5 vs 27.9 dB. The benchmark prints both
  numbers because either one alone is misleading.
- **Sparse local features + matching (XFeat).** New `tasks/features.{h,cc}`:
  `xfeatPreprocess()`, `decodeXfeat()`, `FeatureExtractor` and
  `matchFeatures()`. Python: `bcdl.FeatureExtractor`, `bcdl.decode_xfeat`,
  `bcdl.match_features`, `bcdl.xfeat_preprocess`. Repeatable keypoints with
  L2-normalized 64-d descriptors, for calibration, stitching, tracking and SLAM
  front-ends — the detector/descriptor half of the geometric-vision layer next
  to stereo and GDC.

  The compiled model is **3.1 MB and runs in ~1.0 ms**, because only the
  convolutional trunk is on the BPU. Everything data-dependent is deliberately
  outside it: the input's InstanceNorm (a per-image statistic, and the worst
  kind of thing to quantize) moves to preprocessing, and softmax, non-maximum
  suppression, top-k and sparse sampling live in the decode. That is what keeps
  the graph straight-line with no dynamic shapes.

  **Cost is dominated by the CPU, not the BPU**, and `XfeatConfig::top_k` is the
  knob: per 640x480 image, preprocessing 3.7 ms and inference 1.2 ms are fixed,
  the decode is 21-28 ms, and matching a pair is 130 ms at the default 4096
  features but 8.4 ms at 1024 and 2.2 ms at 512 — it grows with the product of
  the two feature counts. The decode's ~21 ms floor is the full-resolution
  heatmap scatter plus NMS over 307k pixels, which does not depend on top_k.

  Against the reference on identical inputs: keypoints 99.95% identical and
  descriptor cosine **1.000000**; on-board features reproduce 93% of the float
  ones within 2 px; and matching finds 326 pairs where the reference finds 325.
  Note one detail worth copying if you write your own decode — descriptors are
  sampled **bicubically** (the reference sampler's default) while the
  reliability map beside them is bilinear. Reading it as bilinear leaves every
  shape, count and keypoint correct and costs 0.4% of descriptor cosine, which
  looks exactly like float noise.
- **Whole-body pose — 133 keypoints (body, feet, face, hands).** New
  `tasks/wholebody.{h,cc}`: `wholeBodyPreprocess()` turns a person box into the
  model's canvas, `decodeWholeBody()` turns heatmaps back into original-image
  keypoints, and `WholeBodyEstimator` binds both to an Engine. Python:
  `bcdl.WholeBodyEstimator`, `bcdl.wholebody_preprocess`,
  `bcdl.decode_wholebody`. **1.68 ms / 589 FPS** per person in a 28 MB model.

  This is the library's first TOP-DOWN task: it runs once per person on a crop,
  so it needs a detector in front of it and its cost scales with the head count,
  unlike the bottom-up `PoseEstimator` that reads a frame once for every person.
  What you get for that is feet, a 68-point face and both hands, which a
  17-joint head does not have at all.

  Two details of the reference geometry are load-bearing and easy to get wrong:
  the model takes **RGB** (only the detector in front of it wants BGR — though
  `wholeBodyPreprocess` takes BGR like the rest of this library and flips
  internally), and the crop is **not** the usual mmpose affine — the box is
  widened, zero-padded to 3:4, and resized, making the inverse map a pure
  scale-and-translate. The sub-pixel refinement is DARK-UDP; bcdl blurs only a
  13x13 window around each peak rather than all 133 maps in full, which is the
  same arithmetic to within 2e-3 px for ~1/300 of the work.

  Against the float model on identical inputs: heatmap cosine 0.996/0.988, and
  for keypoints the float model is confident about (score >= 0.6) the error is
  2.07 px mean / 0.89 px median. The large-error tail is entirely
  low-confidence keypoints on occluded or out-of-frame limbs, where the heatmap
  is flat and float and int8 simply pick different noise peaks — the usual
  visibility threshold drops them.
- **Face detection with 5 landmarks + alignment.** New `tasks/face.{h,cc}`:
  `decodeScrfd()` (multi-scale distance regression plus landmarks),
  `similarityTransform()` (closed-form Umeyama) and `alignFace()` (warp onto the
  canonical 112x112 template), bound to Python as `bcdl.FaceDetector`,
  `bcdl.decode_scrfd`, `bcdl.align_face` and `bcdl.face_letterbox`. Against the
  float reference on the same frames: landmark error 0.35 px mean / 0.89 px max,
  box corners 0.52 px mean. 60.2 ms detection, 0.29 ms alignment.

  Recognition needs no new task — an aligned crop goes through `ImageEmbedder`
  and matching is `EmbeddingBank`. Two things to know: this detector pads
  BOTTOM-RIGHT rather than centring, so `face_letterbox()` returns a
  LetterboxInfo with zero padding (a centred one shifts every box and landmark);
  and alignment is a correctness requirement, not a refinement, because the
  recognition model only ever saw faces warped onto the template.

  Recognition is validated end-to-end on-board at **0.975 mean cosine** against
  the float reference (0.966 min), matching its separation between same-person
  and different-person pairs. It took a detour to get there: embeddings first
  came back near-orthogonal to the float model (0.015 cosine), which turned out
  to be the `setInput` stride bug fixed below rather than anything about the
  recognition backbone.
- **Panoptic driving perception — detection + drivable area + lane lines.** New
  `tasks/panoptic_drive.{h,cc}`: `decodeYoloV5Anchor()` and the Engine-bound
  `AnchorDetector`, the library's first ANCHOR-BASED decode (everything else here
  is anchor-free LTRB/DFL). One inference feeds all three heads at **2.80 ms /
  358 FPS** with 2.64 ms of CPU post-processing, in a 12 MB model. Bound to
  Python as `bcdl.AnchorDetector` / `bcdl.decode_yolov5_anchor`.

  The published export emits a ready-decoded `[1,25200,6]` tensor, which looks
  like it makes a CPU decode unnecessary. **It does not survive BPU
  compilation**: that decode is assembled from ScatterND, the compile succeeds
  with no error, and the resulting tensor never has its objectness or class
  columns written — only two of every eight floats carry data, all coordinates
  — so a detector reads zero objects at any threshold. The graph is therefore
  cut before the decode, exporting the three raw head convolutions, and the
  arithmetic runs on the CPU. That is also **10x faster**: the in-graph decode
  cost 28.33 ms on device versus 2.80 ms cut.
- Python `Detector.postprocess(lb)`, for multi-head models where one inference
  feeds several decoders — `detect()` could not express that.
- **Real-time semantic segmentation head (PIDNet-S).** Same 2048x1024 input as
  the existing deeplabv3plus head, at **4.52 ms / 221 FPS versus 49.6 ms / 20 FPS
  — 11x faster** in a 19 MB model instead of 38 MB, holding 0.9859 cosine and
  94.63% per-pixel agreement against the float reference. No new decoder: it
  runs through the existing `Segmenter` unchanged, registered as the `semseg_rt`
  task. The one difference to know about is that it emits logits at 1/8
  resolution (`[1,19,128,256]`), so the label map is 256x128 and the caller
  upsamples; set `SegConfig::channels_first = true` for it (NCHW), where
  deeplabv3plus is NHWC.
- **Image embeddings — retrieval and zero-shot classification.** New
  `tasks/embedding.{h,cc}`: `decodeEmbedding()` (read-out + L2 normalize),
  `EmbeddingBank` (cosine top-k over a table of vectors), and the Engine-bound
  `ImageEmbedder`. Python gets `bcdl.ImageEmbedder`, `bcdl.EmbeddingBank` and
  `bcdl.embed_preprocess()`, plus `examples/embed_demo.cc` (gallery retrieval).
  Measured against the model author's float implementation on identical input
  tensors: per-image cosine 0.9977–0.9995, and recall@1/@5 over a 200-image
  gallery identical to the float baseline (0.9250 / 1.0000).

  Two traps this API is shaped around. An embedding `.hbm` packs *several*
  submodels — a pooled whole-image vector and a per-patch feature grid — so bind
  the pooled one by name (`Engine::modelNames()` lists them) and check
  `ImageEmbedder::dim()`; the patch grid otherwise flattens into a very wide,
  meaningless "embedding". And `embed_preprocess()` deliberately does a
  *squashing* resize rather than a letterbox: these towers never saw padding
  bars, and there is no geometry to invert afterwards.
- **`VideoEncoder` decoupled API + H.265 encode.** `feed(frame, pts_us)` /
  `receive(timeout_ms)` / `flush()` / `feed_end_of_stream()`, bound to Python
  alongside the existing `encode()`. H.265 encoding now works and is covered by
  `tests/test_codec_py.py::test_video_encode_decoupled` (parametrized over both
  codecs); previously the encoder was only ever exercised as H.264.
- **`Engine::modelNames(path)` / `bcdl.Engine.model_names(path)`** — list every
  model packed into an `.hbm` without loading it. A `.hbm` is a *package* and may
  hold several models (the official SigLIP encoders ship a global-embedding and a
  patch-feature submodel in one file). The constructor could already select one
  by name; there was no way to discover the names.

### Changed
- **`VideoEncoder` rewritten on the `media_codec` (`hb_mm_mc_*`) streaming API**,
  replacing `hbVPVideoEncode` + an immediate `hbVPGetVideoEncOutputBuffer(task)`
  — the same "one frame in, that frame's output back from the same task" model
  that timed out on real HEVC decode, and equally wrong for an encoder whose rate
  controller may buffer a frame and emit nothing. Inputs are queued and packets
  drained independently. `encode()` keeps its signature and meaning (feed + short
  wait, may return empty).

  Validated H.264 and H.265 at 640×384 and 1920×1088, worst Y MAE 0.09–0.16 grey
  levels against the source pattern; the 1080p HEVC stream verified externally
  with `ffprobe`, keyframes matching the configured `intra_period`.

  **Callers must drain**: a packet does not necessarily come out per frame fed
  in, and on a `false` from `feed()` (input queue momentarily full — easy to hit
  at 1080p, where generating a frame is far cheaper than encoding one) the right
  response is to drain and retry, not to treat it as fatal. The tail needs
  `flush()`. `examples/video_roundtrip.cc` and `examples/video_det_demo.py` show
  the cadence; the latter previously assumed one packet per frame and so dropped
  frames under load and lost the tail.

### Fixed
- **`Engine::setInput` ignored the input's row stride.** Input buffers are sized
  from the model's stride-resolved properties, but the copy into them was a flat
  `memcpy` guarded only by `bytes > size` — so a packed array shorter than the
  device buffer went in silently. Models whose input rows are contiguous (nearly
  all of them) were unaffected; one whose rows are padded was not. ArcFace R50 is
  the case in point: 112 float columns are 448 B/row padded to a **512 B stride**,
  so every row after the first landed shifted and the model saw the same scrambled
  image no matter what was fed to it, producing near-collinear embeddings.

  `setInput` now takes either `inputPackedBytes(i)` — a contiguous row-major
  array, scattered row by row into the device layout with the pad bytes zeroed —
  or `inputBytes(i)`, data already in the device layout. **Any other size now
  throws** instead of being copied in as a short prefix. New accessors
  `inputPackedBytes()`, `inputStride()` and `inputData()` (Python:
  `input_packed_bytes`, `input_stride`, `input_buffer_bytes`) make the layout
  introspectable; anything writing an input buffer directly rather than through
  `setInput` must honor `inputStride`. `outputStride()` / `output_stride()` do
  the same for the output side, which pads as well: a `[1,133,64,48]` float
  heatmap comes back with a 256-byte row stride, so reshaping the raw output
  buffer flat shears every map. (`outputAsFloat()` and the task decoders already
  handled this — the accessor is for code reading the buffer directly.)
- **Encoder `bitstream_buf_size` clamped to the codec's 64 KiB floor.** It was
  sized as `w*h*3/2`, so a small frame (256×128 = 48 KiB) made
  `hb_mm_mc_configure` reject the entire context.
- **Two 1080p codec contexts do not fit in one process.** With an encoder alive
  (5 frame buffers) a decoder (8) never obtained buffers at 1920×1088: it
  produced no frame, its input queue filled, and `feed()` began returning false.
  This is a capacity limit, not a concurrency one — at 640×384 both fit, which
  hid it. `examples/video_roundtrip.cc` destroys the encoder before building the
  decoder; a real transcoder should size `frame_buf_count` deliberately rather
  than assume both defaults fit.
- **`scripts/fetch_models.sh` is self-healing.** Five models were missing from it
  and had never been staged onto the board either, so "re-run this to repopulate"
  was false. It now uses the board-local copy when present and otherwise pulls
  from the convert host and stages it for next time, via a `.part` temporary name
  so a truncated transfer cannot masquerade as a good model.

### Notes
- **The ReID model needs QAT; int8 PTQ is not adequate for this network.** PTQ
  compiles cleanly and returns perfectly well-formed unit vectors whose
  Market-1501 Rank-1 is 51% against the float model's 85% — three architecture
  variants, three calibration methods and a six-fold larger calibration set all
  failed the same way. QAT self-distillation against the float model's embedding
  (no identity labels and no ReID training set: the defect is function-matching,
  not identity learning) restores **84.6% Rank-1 and 0.980 board-vs-float
  cosine**. A quantized embedding tower that has genuinely collapsed still
  returns unit-length, deterministic, correctly-shaped vectors, so cosine
  against a float reference and a retrieval metric — not shape checks — are what
  catch it.

## [0.3.1] — 2026-07

Packaging fix. No library or binding changes.

### Fixed
- **`python/bcdl_py.pyi` regenerated.** The committed type stub had fallen behind
  the nanobind bindings and was missing 14 classes/enums, 8 module functions and
  `InstanceSegConfig.reg_max` — the stereo pipeline, the Mono3D head, the two GDC
  classes, and the SOTA heads (`LabelMap`, the SAM mask decoder, the ReID
  embedding helpers). `GdcLetterbox` also still advertised the `bin_path`
  argument removed in 0.3.0. Since the stub ships in the wheel and the conda
  package, the gap reached users as absent type information. Types and module
  functions now match the bindings 67/67 and 28/28.

  Regenerating needs `-DBCDL_BUILD_STUBS=ON` on a board, in a build with GDC
  enabled (the stub is produced by importing the module, so a GDC-less build
  silently drops `GdcLetterbox`/`GdcRemap`). Note that a plain rebuild will not
  refresh it: the stub target is keyed on `bcdl_py.so`, so once the `.pyi` is
  newer than the module, ninja skips it. Delete the `.pyi` to force it.

## [0.3.0] — 2026-07

Hardening of the video path, and NV12-native preprocessing. The compressed-video
pipeline is 1.8x faster and no longer detours through BGR.

### Changed (breaking)
- **`GdcLetterbox` builds its warp LUT at construction** — the `bin_path`
  argument is gone, along with the need for an offline `.bin`. The affine
  letterbox is expressed as a CUSTOM sampling grid and generated at runtime.
- **`VideoDecoder::feed()` / `decode()` take exactly one access unit.** The
  decoder now runs in `MC_FEEDING_MODE_FRAME_SIZE`; feeding arbitrary byte chunks
  is no longer supported. (`AsyncVideoDetectionPipeline::submit()` still takes
  arbitrary Annex-B bytes and does the reassembly for you.)
- **`StageProfile` gained `cvt_ms`**, which `totalMs()` now includes.

### Added
- **`AsyncDetectionPipeline::submitNv12()`** — letterboxes a decoded NV12 frame
  straight into a model-input slot, on the GDC hardware engine when present and
  on the CPU otherwise. No BGR round-trip and no copy of the frame. Split into
  `acquireSlot()` / `letterboxIntoSlot()` / `commitSlot()` for callers that must
  not hold a recycled buffer while waiting for pipeline capacity.
- **`letterboxNv12Cpu()`** — NV12→NV12 letterbox with no colour conversion; the
  fallback when GDC is unavailable. `BCDL_NO_GDC=1` forces it, for A/B.
- **`YuvRange`** (`preproc/geometry.h`) — makes the studio→full-range expansion
  explicit. Video carries studio-swing NV12 (Y in [16,235]) while models are
  calibrated on full-range pixels; the old BGR round-trip did this conversion by
  accident, as a side effect of `cv::cvtColor`.
- `StageProfile::cvt_ms` / `cvtPerFrame()` (Python: `cvt_ms`, `cvt_per_frame`).

### Fixed
- **VPU `INTERRUPT TIMEOUT` hang.** Feeding faster than dequeuing fills the
  decoder's frame buffers and the next hardware decode stalls. Every decode path
  now feeds one AU, drains every ready frame, then flushes the reorder tail.
- **Heap corruption / random `SIGSEGV`.** The decoder ran in
  `MC_FEEDING_MODE_STREAM_SIZE`, whose bitstream-ring update path corrupts the
  glibc heap from inside the codec. Single-threaded decode alone crashed 6 runs
  in 20, with no BPU in the process; `FRAME_SIZE` is clean and also faster (pure
  decode 452 vs 281 FPS) and no longer drops a frame.
- **H.265 stalled at 16 FPS and lost frames.** A blocking output dequeue holds
  the codec's internal lock, so a feed thread racing a receive thread stalls
  whenever the decoder has no frame ready. One thread now owns the codec context.
- **Deadlock under backpressure.** The caller-facing results queue was bounded,
  so a full queue blocked the infer stage, which stopped slots recycling, which
  stalled every upstream stage — while the caller, blocked inside `submit()`, was
  the only one who could drain it.
- **`~VideoDecoder` hung** when the decoder still held decoded frames:
  `hb_mm_mc_stop()` waits for the codec to empty its output port, and only the
  application can empty it. The destructor now returns every pending buffer first.
- **One `Component vdec_render isn't ready!` error line per decoded frame.** A
  dequeue that finds no frame prints it (to stdout), and a drain-to-empty cadence
  ends every access unit with such a call. `recv()` now asks
  `hb_mm_mc_get_status()` first, which is silent.
- **`AsyncVideoDetectionPipeline` dropped the last frame** at end of stream: the
  trailing buffer can hold two access units, and under `FRAME_SIZE` only the
  first of them decodes.
- **Per-frame device allocation** in the video pipeline (334 allocations per 319
  frames → 22, all warm-up).
- **`nv12ToBgrCpu()`'s two build configurations produced different pixels** — the
  OpenCV fast path expanded studio swing, the hand-written fallback did not. It
  now takes a `YuvRange` and both branches implement each range.

### Performance
- `AsyncVideoDetectionPipeline`, 1080p H.264, yolo26n: **249 → 441 FPS**
  (1385/1385 frames), decode-bound. H.265: **439–451 FPS** (300/300 frames).
- Pure VPU decode: **452 FPS** (H.264) / **481 FPS** (H.265).
- GDC hardware letterbox, 1920×1080 → 640×640: 0.97 ms/frame wall, ~0.3 ms of it
  CPU, against 4.71 ms of pure CPU for the BGR chain it replaces.

## [0.2.0] — 2026-07

### Added
- **Video object detection — `AsyncVideoDetectionPipeline`** — the whole
  compressed-video → detections path in C++: `submit(Annex-B bytes)` → internal
  AU segmentation → four overlapped stages (VPU decode ‖ NV12→BGR ‖ CPU preproc
  ‖ BPU infer+NMS) → `next()`/`next_nowait()`. A thin caller that only pumps an
  `ffmpeg -c copy` byte stream reaches the decode-bound ceiling (yolo26n 1080p
  H.264 ~240 FPS); the Python binding releases the GIL. New demos
  `examples/video_det_demo.{cc,py}`, `examples/video_det_demo_async.cc`, and the
  thin RTSP driver `examples/rtsp_det_demo.py`.
- **H.265 decode** — `VideoDecoder` rewritten on the `media_codec` (`hb_mm_mc_*`)
  streaming API with reorder support and a decoupled `feed()` / `receive()` /
  `flush()` interface (the old per-AU model timed out on HEVC). Caveat: a
  hierarchical-GOP HEVC stream (camera SVC / smart-codec) decodes its base
  temporal layer only.
- **Per-stage `StageProfile` timing** — `decode` / `preproc` / `infer` /
  `postproc` ms per frame from `DetectionPipeline` and `AsyncDetectionPipeline`
  (in the async path the slowest stage bounds throughput), plus
  `AsyncDetectionPipeline::tryNext()` (non-blocking pop) and `nv12ToBgrCpu()`
  (NV12→BGR, OpenCV SIMD + BT.601 fallback).
- **`bcdl::GdcRemap` — hardware dense remap on the VPS GDC engine** (+ Python
  binding `bcdl.GdcRemap(map_x, map_y, in_w, in_h, grid_step=16)`). An arbitrary
  FIXED warp with cv2.remap semantics (built for stereo rectification) compiled
  at construction into a GDC CUSTOM-grid warp LUT — generated **at runtime** by
  driving `libgdcbin` directly (`gdc_init`/`gdc_set_custom_points`/
  `gdc_calculate`); no offline bin file. The public `hbn_gen_gdc_cfg` cannot do
  this (it never forwards custom grid points — verified by disassembly). Key
  semantics recovered from `transform_custom` (the SDK header comments are
  wrong/misleading): `custom.w/h` are TILE counts (points = `(w+1)*(h+1)`
  nodes) and `centerx/centery` are in GRID-INDEX units (`w/2, h/2`), not input
  pixels. Measured (2448×2048 NV12, real stereo-rectify maps, S100P):
  **6.3 ms/frame wall, ~1 ms of it CPU** (copy-in 0.5 + copy-out 0.5; the
  5.3 ms GDC op leaves the CPU free) vs 14.7 ms on all 6 cores for `cv2.remap`;
  matches cv2.remap to mean 0.25 / p99 2 grey-levels over the valid map region.
  Correctness/bench suite: `tests/test_gdc_remap.py`. `GdcRemap.run` releases
  the GIL, so two instances (stereo top/bottom) overlap one camera's CPU
  conversions with the other's hardware op from Python threads (adam_fast
  measured 24.3 → 18.6 ms for both cams).
- **Open-vocabulary detection / segmentation (YOLOE)** — prompt-free YOLOE reuses
  the existing LTRB / DFL decode with a `bcdl::LabelMap` (COCO-80 by default,
  `from_file` / `from_list`); no new post-process math. Board bench + figures:
  `yoloe_det`, `yoloe_seg` (~520 / ~397 infer FPS on S100P).
- **Promptable segmentation (EdgeSAM / SAM mask-decoder tail)** — `SamConfig`,
  `SamMask`, `SamMaskDecoder`, `decode_sam_masks`, and the two-stage
  `SamSession` Python wrapper (RepViT encoder → cached embedding → point/box
  prompt decoder). `tests/test_promptable_seg.py`.
- **ReID appearance embeddings** — `normalize_embedding` (L2) + `cosine_similarity`
  BoT-SORT association primitives. `tests/test_reid.py`.
- **QAT float-input detection path** — `YoloLtrbDetector.detect_float()` and
  `letterbox_chw_float()` feed a `[1,3,H,W]` float32 (QAT-exported) model; the
  LTRB/DFL decode is identical to the NV12 PTQ path.
- **Instance-seg DFL box heads** — `InstanceSegmenter` auto-detects plain-LTRB
  (4 ch) vs DFL (`4*reg_max`, e.g. 64) box channels, so yolov8/v11/YOLOE seg
  heads decode without config; `InstanceSegConfig.reg_max` exposed for the raw
  `decode_instance_seg()` entry point.

### Changed
- **Semantic-seg argmax 73× faster** — `Segmenter::postprocess()` argmaxes over
  the channel axis directly on the F32 device buffer (byte-stride aware, OpenMP),
  instead of materializing the whole logit tensor element-by-element first.
  Full-res deeplabv3plus `[1,1024,2048,19]` decode ~660 ms → ~8.7 ms, labels
  byte-identical. No model change (the materialize fallback covers F16/quantized).
- **Docs overhaul** — reorganized README around a 12-task check-image Gallery,
  a Documentation hub, badges/TOC, a fuller private-conda install (env pin,
  channel config, smoke test), and a Quickstart. Documented platform support as
  RDK **S100 / S100P / S600** (same S-series runtime; a compiled `.hbm` is
  march-specific). `scripts/board_bench.py` now annotates the JPU check image
  with the measured cv2-vs-JPU bars and generates a ByteTrack check figure
  (`benchmarks/figures/track.jpg`).

### Added
- **Bilingual README** — Simplified Chinese as the default (`README.md`) with a
  full English translation (`README.en.md`) and a language switcher
  (`English | 简体中文`) at the top of both files.
- **Stereo** — `StereoPipeline`: two-image (left/right) F32 NCHW RGB input
  (resize **or** center-crop fit) → disparity, with optional metric depth
  (`z=fx·baseline/disp`) and a geometry validity mask (disparity range +
  left-border + optional left-right consistency). Pure `pack_stereo_input` /
  `disparity_to_depth` / `stereo_valid_mask` helpers + nanobind bindings +
  `examples/stereo_demo.cc`. Validated on LAS2-M (cosine 1.0 / EPE 0 px vs the
  reference preproc, 70 FPS @480×640).
- Packaging: `find_package(bcdl)` CMake export (`bcdlConfig.cmake` +
  `bcdlTargets.cmake`, bundled `FindHobot.cmake`), `install()` rules, and a
  generated `bcdl/version.h` (`BCDL_VERSION*` macros).
- Python: `pip install .` via scikit-build-core (`pyproject.toml`), `py.typed`
  marker, and `bcdl.__version__`.
- Docs: `CONTRIBUTING.md`, plus full Python (`docs/API.md`) and C++
  (`docs/CPP_API.md`) API references — every class, config, and decoder with a
  usage snippet per task.

### Fixed
- Docs: corrected stale benchmark numbers (semseg decode `660 → 58 ms` after the
  argmax speedup; JPU decode `2.6–3.5× → ≈3.6–5.3×`, now consistent with the
  measured table), the tracking module location (`tracks/`, not `tasks/`), and
  API-reference signature mismatches (the `Detector` wrapper has no
  `postprocess()`; the `decode_*` letterbox parameter name).

## [0.1.0] — 2026-06

First tagged release. On-board validated on RDK S100P (79 passed, 1 skipped).

### Added
- **Core** — `SysMem` / `Task` / `MemPool` (RAII over `hbUCP`), `Engine` over
  `hbDNN` with automatic cache discipline, zero-copy / dequantising output readers.
- **Tasks** — detection (LTRB multi-scale + DFL: YOLO26 / YOLOv8 / v5 / v11),
  classification, pose, instance segmentation, oriented boxes (OBB), semantic
  segmentation, monocular depth, and a 3-stage **PP-OCRv5** OCR pipeline
  (DBNet detect → PP-LCNet direction cls → CRNN/CTC recognize). Each decoder is
  also exposed as an Engine-free pure `decode_*` function.
- **Tracking** — ByteTrack multi-object tracker (Kalman + Hungarian + two-stage
  association).
- **Media** — hardware JPEG (JPU) and H.264 / H.265 (VPU) encode/decode.
- **Pipelines** — synchronous buffer-reuse `DetectionPipeline`, threaded
  `AsyncDetectionPipeline` (preproc ‖ infer overlap, GIL released in Python),
  `TrackingPipeline`, and a video-file → decode → detect path.
- **Python** — nanobind bindings (NumPy in/out) for every decoder + pipeline.
- **Tooling** — on-board benchmark + check-figure generator
  (`scripts/board_bench.py`), repo-local model dir (`scripts/fetch_models.sh`),
  two-tier test suite (Engine-free NumPy + on-board real-model).

[Unreleased]: https://example.com/compare/v0.1.0...HEAD
[0.1.0]: https://example.com/releases/tag/v0.1.0
