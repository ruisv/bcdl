# BCDL on-board benchmark (real models)

RDK S100P (BPU) · sw_version=4.0.5-Beta. Models: official YOLO26n **nash-m** from rdk_model_zoo (the LTRB family bcdl's decoders target). `infer` = BPU inference only; `decode` = infer + post-process (NMS / mask assembly / keypoints); `mem` = engine DRAM footprint (RSS delta around load+run, isolated per subprocess); `model` = `.hbm` size on disk.

| task | model | input | outs | infer ms | infer FPS | decode ms | decode FPS | model MB | mem MB | result |
|------|-------|-------|------|----------|-----------|-----------|------------|----------|--------|--------|
| cls | `yolo26n_cls_nashm_224x224_nv12.hbm` | 224x224 | 1 | 0.43 | 2324 | 0.48 | 2084 | 3.85 | 6.5 | 340:0.96, 292:0.03, 282:0.01 |
| det | `yolo26n_detect_nashm_640x640_nv12.hbm` | 640x640 | 6 | 1.19 | 839 | 2.01 | 498 | 7.75 | 11.9 | 5 box(es) |
| det_dfl | `yolov8_640x640_nv12.hbm` | 640x640 | 6 | 1.47 | 679 | 9.56 | 105 | 3.72 | 12.1 | 5 box(es) |
| pose | `yolo26n_pose_nashm_640x640_nv12.hbm` | 640x640 | 9 | 1.28 | 780 | 1.47 | 680 | 7.72 | 12.3 | 2 pose(s) |
| seg | `yolo26n_seg_nashm_640x640_nv12.hbm` | 640x640 | 10 | 1.60 | 625 | 10.94 | 91 | 9.88 | 12.0 | 4 instance(s) |
| obb | `yolo26n_obb_nashm_640x640_nv12.hbm` | 640x640 | 9 | 1.14 | 874 | 1.60 | 623 | 5.79 | 47.6 | 31 rotated box(es) |
| semseg | `deeplabv3plus_dilation1248_1024x2048_nv12.hbm` | 2048x1024 | 1 | 48.94 | 20 | 57.16 | 18 | 39.1 | 28.4 | 2048x1024 seg, 16 classes |
| depth | `depth_any.hbm` | 686x518 | 1 | 110.86 | 9 | 115.17 | 9 | 121.82 | 18.4 | 686x518 depth [0.58,17.96] |
| stereo | `las2_m_crop_nashm.hbm` | 640x480 | 1 | 13.99 | 72 | 21.59 | 46 | 40.7 | 17.7 | disp [44.4,129.0] mean 93.4px; 85% valid |
| ocr | `ppocrv5_server_det_960x960.hbm` | 960x960 | 1 | 20.03 | 50 | 135.60 | 7 | 35.33 | 26.0 | 16 lines; e.g. 发足够的滋养 |
| yoloe_det | `yoloe_11s_coco80_det_bpu_nashm_640x640_nv12.hbm` | 640x640 | 6 | 1.90 | 528 | 2.71 | 369 | 17.07 | 12.1 | 6 box(es) |
| yoloe_seg | `yoloe_11s_coco80_seg_bpu_nashm_640x640_nv12.hbm` | 640x640 | 10 | 2.42 | 413 | 11.74 | 85 | 19.15 | 11.9 | 4 instance(s) |
| wholebody | `vitpose_s_wholebody_nashm_256x192.hbm` | 192x256 | 1 | 1.68 | 596 | 17.35 | 58 | 29.07 | 12.9 | 2 person(s), 133 keypoints, 98/104 visible |
| features | `xfeat_nashm_640x480.hbm` | 640x480 | 3 | 1.03 | 971 | 197.46 | 5 | 3.17 | 9.9 | 4096+4096 features, 2354 matches; 87% epipolar-consistent |
| superres | `realesr_general_x4v3_nashm_128.hbm` | 128x128 | 1 | 2.06 | 486 | 27.92 | 36 | 38.87 | 8.7 | 202x270 -> 808x1080, 6 tiles; PSNR 20.5 dB (bicubic 21.0); detail 0.56x of truth (bicubic 0.40x) |
| superres_span | `spanx4_ch48_nashm_128.hbm` | 128x128 | 1 | 1.16 | 861 | 22.29 | 45 | 6.08 | 8.6 | 202x270 -> 808x1080, 6 tiles; PSNR 21.0 dB (bicubic 21.0); detail 0.65x of truth (bicubic 0.40x) |
| reid | `osnet_ain_qat_nashm_256x128.hbm` | 128x256 | 1 | 0.84 | 1196 | 5.22 | 192 | 6.61 | 12.8 | 3 person(s), 512-d, cross-similarity max 0.491 mean 0.456 |
| edgesam | `edge_sam encoder + point/box decoder` | 1024x1024 | 2 | 16.23 | 62 | 21.39 | 47 | 47.58 | 4.6 | pt+box masks; enc 16.2ms/img + dec 5.2ms/prompt |

Check figures: `benchmarks/figures/<task>.jpg` (boxes / keypoints / instance masks / rotated boxes / top-k).

## JPEG decode — software (cv2/libjpeg, CPU) vs hardware (JPU)

Decode straight to NV12 (what the BPU consumes). With a **reused** `JpegDecoder` (steady state — the way a real video / camera pipeline runs it) the JPU is **several times faster** (≈3.6–5.3× on these samples; see the table) than cv2/libjpeg, AND it **offloads the CPU** (frees the 6×A78AE cores for infer / post-process) and lands the result in a device NV12 buffer for **zero-copy NV12→BPU**. Caveat: the one-shot `bcdl.jpeg_decode` *helper* creates a fresh decoder per call (~5 ms JPU context setup), so it is NOT faster — reuse a `JpegDecoder` instance for the win. (Format note: the JPU decoder only supports **4:2:0** JPEGs; 4:2:2 / 4:4:4 streams are now rejected with a clean `bcdl::Error` — they used to segfault the JPU firmware — so decode those in software.) Check image: `benchmarks/figures/decode_jpu.jpg` (a JPU-decoded frame, annotated with the measured cv2-vs-JPU decode time).

| image | size | jpeg KB | cv2 ms | JPU ms | cv2/JPU |
|-------|------|---------|--------|--------|---------|
| bus.jpg | 816x1088 | 134.2 | 5.85 | 1.66 | 3.52 |
| gt_2322.jpg | 768x1088 | 268.8 | 6.65 | 1.18 | 5.62 |

Reproduce on the board (in the `bcdl` conda env): `PYTHONPATH=build:python python scripts/board_bench.py`.
