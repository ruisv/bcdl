# BCDL on-board benchmark (real models)

RDK S100P (BPU) · sw_version=4.0.5-Beta. Models: official YOLO26n **nash-m** from rdk_model_zoo (the LTRB family bcdl's decoders target). `infer` = BPU inference only; `decode` = infer + post-process (NMS / mask assembly / keypoints); `mem` = engine DRAM footprint (RSS delta around load+run, isolated per subprocess); `model` = `.hbm` size on disk.

| task | model | input | outs | infer ms | infer FPS | decode ms | decode FPS | model MB | mem MB | result |
|------|-------|-------|------|----------|-----------|-----------|------------|----------|--------|--------|
| cls | `yolo26n_cls_nashm_224x224_nv12.hbm` | 224x224 | 1 | 0.45 | 2240 | 0.54 | 1838 | 3.85 | 6.7 | 340:0.96, 292:0.03, 282:0.01 |
| det | `yolo26n_detect_nashm_640x640_nv12.hbm` | 640x640 | 6 | 1.22 | 820 | 2.05 | 487 | 7.75 | 11.9 | 5 box(es) |
| det_dfl | `yolov8_640x640_nv12.hbm` | 640x640 | 6 | 1.49 | 670 | 9.53 | 105 | 3.72 | 11.9 | 5 box(es) |
| pose | `yolo26n_pose_nashm_640x640_nv12.hbm` | 640x640 | 9 | 1.31 | 765 | 1.50 | 669 | 7.72 | 12.2 | 2 pose(s) |
| seg | `yolo26n_seg_nashm_640x640_nv12.hbm` | 640x640 | 10 | 1.62 | 616 | 11.38 | 88 | 9.88 | 11.8 | 4 instance(s) |
| obb | `yolo26n_obb_nashm_640x640_nv12.hbm` | 640x640 | 9 | 1.17 | 857 | 1.72 | 582 | 5.79 | 47.6 | 31 rotated box(es) |
| semseg | `deeplabv3plus_dilation1248_1024x2048_nv12.hbm` | 2048x1024 | 1 | 49.60 | 20 | 58.06 | 17 | 39.1 | 28.2 | 2048x1024 seg, 16 classes |
| depth | `depth_any.hbm` | 686x518 | 1 | 113.00 | 9 | 117.38 | 8 | 121.82 | 18.3 | 686x518 depth [0.58,17.96] |
| stereo | `las2_m_crop_nashm.hbm` | 640x480 | 1 | 14.15 | 71 | 21.72 | 46 | 40.7 | 17.4 | disp [44.4,129.0] mean 93.4px; 85% valid |
| ocr | `ppocrv5_server_det_960x960.hbm` | 960x960 | 1 | 20.14 | 50 | 138.26 | 7 | 35.33 | 25.6 | 16 lines; e.g. 发足够的滋养 |
| yoloe_det | `yoloe_11s_coco80_det_bpu_nashm_640x640_nv12.hbm` | 640x640 | 6 | 1.93 | 520 | 2.78 | 360 | 17.07 | 11.8 | 6 box(es) |
| yoloe_seg | `yoloe_11s_coco80_seg_bpu_nashm_640x640_nv12.hbm` | 640x640 | 10 | 2.52 | 397 | 12.35 | 81 | 19.15 | 11.9 | 4 instance(s) |
| edgesam | `edge_sam encoder + point/box decoder` | 1024x1024 | 2 | 16.72 | 60 | 21.93 | 46 | 47.58 | 4.5 | pt+box masks; enc 16.7ms/img + dec 5.2ms/prompt |

Check figures: `benchmarks/figures/<task>.jpg` (boxes / keypoints / instance masks / rotated boxes / top-k).

## JPEG decode — software (cv2/libjpeg, CPU) vs hardware (JPU)

Decode straight to NV12 (what the BPU consumes). With a **reused** `JpegDecoder` (steady state — the way a real video / camera pipeline runs it) the JPU is **several times faster** (≈3.6–5.3× on these samples; see the table) than cv2/libjpeg, AND it **offloads the CPU** (frees the 6×A78AE cores for infer / post-process) and lands the result in a device NV12 buffer for **zero-copy NV12→BPU**. Caveat: the one-shot `bcdl.jpeg_decode` *helper* creates a fresh decoder per call (~5 ms JPU context setup), so it is NOT faster — reuse a `JpegDecoder` instance for the win. (Format note: the JPU decoder only supports **4:2:0** JPEGs; 4:2:2 / 4:4:4 streams are now rejected with a clean `bcdl::Error` — they used to segfault the JPU firmware — so decode those in software.) Check image: `benchmarks/figures/decode_jpu.jpg` (a JPU-decoded frame, annotated with the measured cv2-vs-JPU decode time).

| image | size | jpeg KB | cv2 ms | JPU ms | cv2/JPU |
|-------|------|---------|--------|--------|---------|
| bus.jpg | 816x1088 | 134.2 | 5.92 | 1.62 | 3.65 |
| gt_2322.jpg | 768x1088 | 268.8 | 6.68 | 1.42 | 4.71 |

Reproduce on the board (in the `bcdl` conda env): `PYTHONPATH=build:python python scripts/board_bench.py`.
