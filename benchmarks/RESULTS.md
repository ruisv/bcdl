# BCDL on-board benchmark (real models)

RDK S100P (BPU) · sw_version=4.0.5-Beta. Models: official YOLO26n **nash-m** from rdk_model_zoo (the LTRB family bcdl's decoders target). `infer` = BPU inference only; `decode` = infer + post-process (NMS / mask assembly / keypoints); `mem` = engine DRAM footprint (RSS delta around load+run, isolated per subprocess); `model` = `.hbm` size on disk.

| task | model | input | outs | infer ms | infer FPS | decode ms | decode FPS | model MB | mem MB | result |
|------|-------|-------|------|----------|-----------|-----------|------------|----------|--------|--------|
| cls | `yolo26n_cls_nashm_224x224_nv12.hbm` | 224x224 | 1 | 0.43 | 2347 | 0.48 | 2075 | 3.85 | 6.7 | 340:0.96, 292:0.03, 282:0.01 |
| det | `yolo26n_detect_nashm_640x640_nv12.hbm` | 640x640 | 6 | 1.20 | 830 | 2.05 | 489 | 7.75 | 12.1 | 5 box(es) |
| det_dfl | `yolov8_640x640_nv12.hbm` | 640x640 | 6 | 1.53 | 651 | 9.57 | 104 | 3.72 | 12.4 | 5 box(es) |
| pose | `yolo26n_pose_nashm_640x640_nv12.hbm` | 640x640 | 9 | 1.31 | 762 | 1.53 | 653 | 7.72 | 12.8 | 2 pose(s) |
| seg | `yolo26n_seg_nashm_640x640_nv12.hbm` | 640x640 | 10 | 1.62 | 616 | 11.36 | 88 | 9.88 | 12.2 | 4 instance(s) |
| obb | `yolo26n_obb_nashm_640x640_nv12.hbm` | 640x640 | 9 | 1.17 | 857 | 1.71 | 585 | 5.79 | 47.8 | 31 rotated box(es) |
| semseg | `deeplabv3plus_dilation1248_1024x2048_nv12.hbm` | 2048x1024 | 1 | 49.56 | 20 | 58.00 | 17 | 39.1 | 28.5 | 2048x1024 seg, 16 classes |
| depth | `depth_any.hbm` | 686x518 | 1 | 112.94 | 9 | 117.39 | 8 | 121.82 | 18.6 | 686x518 depth [0.58,17.96] |
| stereo | `las2_m_crop_nashm.hbm` | 640x480 | 1 | 14.17 | 71 | 21.76 | 46 | 40.7 | 17.8 | disp [44.4,129.0] mean 93.4px; 85% valid |
| ocr | `ppocrv5_server_det_960x960.hbm` | 960x960 | 1 | 20.17 | 50 | 137.98 | 7 | 35.33 | 25.9 | 16 lines; e.g. 发足够的滋养 |

Check figures: `benchmarks/figures/<task>.jpg` (boxes / keypoints / instance masks / rotated boxes / top-k).

## JPEG decode — software (cv2/libjpeg, CPU) vs hardware (JPU)

Decode straight to NV12 (what the BPU consumes). With a **reused** `JpegDecoder` (steady state — the way a real video / camera pipeline runs it) the JPU is **several times faster** (≈3.6–5.3× on these samples; see the table) than cv2/libjpeg, AND it **offloads the CPU** (frees the 6×A78AE cores for infer / post-process) and lands the result in a device NV12 buffer for **zero-copy NV12→BPU**. Caveat: the one-shot `bcdl.jpeg_decode` *helper* creates a fresh decoder per call (~5 ms JPU context setup), so it is NOT faster — reuse a `JpegDecoder` instance for the win. (Format note: the JPU decoder only supports **4:2:0** JPEGs; 4:2:2 / 4:4:4 streams are now rejected with a clean `bcdl::Error` — they used to segfault the JPU firmware — so decode those in software.) Check image: `benchmarks/figures/decode_jpu.jpg` (a JPU-decoded frame, annotated with the measured cv2-vs-JPU decode time).

| image | size | jpeg KB | cv2 ms | JPU ms | cv2/JPU |
|-------|------|---------|--------|--------|---------|
| bus.jpg | 816x1088 | 134.2 | 5.92 | 1.64 | 3.60 |
| gt_2322.jpg | 768x1088 | 268.8 | 6.80 | 1.29 | 5.27 |

Reproduce on the board (in the `bcdl` conda env): `PYTHONPATH=build:python python scripts/board_bench.py`.
