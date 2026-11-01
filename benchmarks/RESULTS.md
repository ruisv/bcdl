# BCDL on-board benchmark (real models)

RDK S100P (BPU) · sw_version=4.0.5-Beta. Models: official YOLO26n **nash-m** from rdk_model_zoo (the LTRB family bcdl's decoders target). `infer` = BPU inference only; `decode` = infer + post-process (NMS / mask assembly / keypoints); `mem` = engine DRAM footprint (RSS delta around load+run, isolated per subprocess); `model` = `.hbm` size on disk.

| task | model | input | outs | infer ms | infer FPS | decode ms | decode FPS | model MB | mem MB | result |
|------|-------|-------|------|----------|-----------|-----------|------------|----------|--------|--------|
| cls | `yolo26n_cls_nashm_224x224_nv12.hbm` | 224x224 | 1 | 0.45 | 2242 | 0.52 | 1928 | 3.85 | 6.4 | 340:0.96, 292:0.03, 282:0.01 |
| det | `yolo26n_detect_nashm_640x640_nv12.hbm` | 640x640 | 6 | 1.21 | 826 | 2.07 | 484 | 7.75 | 12.1 | 5 box(es) |
| det_dfl | `yolov8_640x640_nv12.hbm` | 640x640 | 6 | 1.50 | 667 | 9.51 | 105 | 3.72 | 12.1 | 5 box(es) |
| pose | `yolo26n_pose_nashm_640x640_nv12.hbm` | 640x640 | 9 | 1.35 | 740 | 1.55 | 646 | 7.72 | 12.3 | 2 pose(s) |
| seg | `yolo26n_seg_nashm_640x640_nv12.hbm` | 640x640 | 10 | 1.63 | 612 | 12.39 | 81 | 9.88 | 12.5 | 4 instance(s) |
| obb | `yolo26n_obb_nashm_640x640_nv12.hbm` | 640x640 | 9 | 1.16 | 861 | 1.71 | 586 | 5.79 | 47.6 | 31 rotated box(es) |
| semseg | `deeplabv3plus_dilation1248_1024x2048_nv12.hbm` | 2048x1024 | 1 | 46.87 | 21 | 55.98 | 18 | 39.1 | 28.2 | 2048x1024 seg, 16 classes |
| depth | `depth_any.hbm` | 686x518 | 1 | 104.53 | 10 | 109.06 | 9 | 121.82 | 18.5 | 686x518 depth [0.58,17.96] |
| stereo | `las2_m_crop2_nashm.hbm` | 640x480 | 1 | 13.92 | 72 | 17.53 | 57 | 40.2 | 17.6 | disp [44.7,128.8] mean 93.4px; 85% valid |
| ocr | `ppocrv6_medium_det_960x960.hbm` | 960x960 | 1 | 8.69 | 115 | 92.53 | 11 | 23.48 | 25.5 | 16 lines; e.g. 发足够的滋养 |
| yoloe_det | `yoloe_11s_coco80_det_bpu_nashm_640x640_nv12.hbm` | 640x640 | 6 | 1.93 | 519 | 2.77 | 361 | 17.07 | 12.4 | 6 box(es) |
| yoloe_seg | `yoloe_11s_coco80_seg_bpu_nashm_640x640_nv12.hbm` | 640x640 | 10 | 2.45 | 408 | 13.21 | 76 | 19.15 | 12.3 | 4 instance(s) |
| wholebody | `vitpose_s_wholebody_nashm_256x192.hbm` | 192x256 | 1 | 1.65 | 605 | 6.56 | 152 | 29.07 | 12.8 | 2 person(s), 133 keypoints, 98/104 visible |
| features | `xfeat_nashm_640x480.hbm` | 640x480 | 3 | 1.05 | 953 | 228.39 | 4 | 3.17 | 9.8 | 4096+4096 features, 2354 matches; 87% epipolar-consistent |
| superres | `realesr_general_x4v3_nashm_128.hbm` | 128x128 | 1 | 2.08 | 481 | 28.12 | 36 | 38.87 | 8.6 | 202x270 -> 808x1080, 6 tiles; PSNR 20.5 dB (bicubic 21.0); detail 0.56x of truth (bicubic 0.40x) |
| superres_span | `spanx4_ch48_nashm_128.hbm` | 128x128 | 1 | 1.16 | 863 | 22.56 | 44 | 6.08 | 8.6 | 202x270 -> 808x1080, 6 tiles; PSNR 21.0 dB (bicubic 21.0); detail 0.65x of truth (bicubic 0.40x) |
| reid | `osnet_ain_qat_nashm_256x128.hbm` | 128x256 | 1 | 0.83 | 1200 | 5.31 | 188 | 6.61 | 12.8 | 3 person(s), 512-d, cross-similarity max 0.491 mean 0.456 |
| flow | `neuflow_v2_int16_nashm_512x384.hbm` | 512x384 | 1 | 19.45 | 51 | 26.41 | 38 | 46.37 | 13.9 | 512x384 field, |flow| mean 92.87 px, p95 125.03 px, max 137.94 px |
| anomaly | `glass_bottle_int8_nashm_288x288.hbm` | 288x288 | 1 | 72.62 | 14 | 163.22 | 6 | 96.6 | 6.2 | defect score 1.000, normal 0.876 (gap 0.124) |
| drive | `diffusiondrive_r34_256x1024_s100p.hbm` | 1024x256 | 4 | 14.37 | 70 | 22.24 | 45 | 135.58 | 12.4 | 8 poses, 24.8 m ahead, 7 vehicle(s), 1.74 m from the realized future |
| lidar3d | `pointpillars_vfe.hbm` | 432x496 | 3 | 460.05 | 2 | 465.83 | 2 | 44.94 | 154.6 | 7478 pillars from 120965 points; 8 boxes (8C/0P/0Y) |
| edgesam | `edge_sam encoder + point/box decoder` | 1024x1024 | 2 | 17.58 | 57 | 23.33 | 43 | 47.58 | 4.6 | pt+box masks; enc 17.6ms/img + dec 5.8ms/prompt |

The `lidar3d` row runs the **published three-stage PointPillars chain**, which is
what `scripts/fetch_models.sh` gives you — not the re-exported chain that is 46×
faster on the BPU but whose weights cannot be redistributed (see
[`docs/MODELS.md`](../docs/MODELS.md#pointpillars-comes-in-two-chains-and-one-is-46-faster)).
Its `image` column shows a staged KITTI frame; with none staged the task falls
back to the model zoo's own sample sweep and the counts differ.

Check figures: `benchmarks/figures/<task>.jpg` (boxes / keypoints / instance masks / rotated boxes / top-k).

## JPEG decode — software (cv2/libjpeg, CPU) vs hardware (JPU)

Decode straight to NV12 (what the BPU consumes). With a **reused** `JpegDecoder` (steady state — the way a real video / camera pipeline runs it) the JPU is **several times faster** (≈3.6–5.3× on these samples; see the table) than cv2/libjpeg, AND it **offloads the CPU** (frees the 6×A78AE cores for infer / post-process) and lands the result in a device NV12 buffer for **zero-copy NV12→BPU**. Caveat: the one-shot `bcdl.jpeg_decode` *helper* creates a fresh decoder per call (~5 ms JPU context setup), so it is NOT faster — reuse a `JpegDecoder` instance for the win. (Format note: the JPU decoder only supports **4:2:0** JPEGs; 4:2:2 / 4:4:4 streams are now rejected with a clean `bcdl::Error` — they used to segfault the JPU firmware — so decode those in software.) Check image: `benchmarks/figures/decode_jpu.jpg` (a JPU-decoded frame, annotated with the measured cv2-vs-JPU decode time).

| image | size | jpeg KB | cv2 ms | JPU ms | cv2/JPU |
|-------|------|---------|--------|--------|---------|
| bus.jpg | 816x1088 | 134.2 | 6.12 | 1.50 | 4.07 |
| gt_2322.jpg | 768x1088 | 268.8 | 6.95 | 1.25 | 5.57 |

Reproduce on the board (in the `bcdl` conda env): `PYTHONPATH=build:python python scripts/board_bench.py`.
