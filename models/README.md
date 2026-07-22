# models/

Compiled BPU model binaries (`.hbm`) the harness and examples consume. The code
references them ONLY by the repo-relative path `models/<name>.hbm` (no absolute
paths), env-overridable per task (`BCDL_MODELS`, `BCDL_OCR_DET`, …).

**The `.hbm` files are NOT committed** — 28 models totalling ~700 MB (incl. a
128 MB embedding model and a 116 MB depth model), too large for git
(`.gitignore` excludes `*.hbm`). Populate this dir on the board with:

```bash
scripts/fetch_models.sh        # copies from the board's source locations
```

`fetch_models.sh` is the single place the board's absolute source paths live
(the offline-converted models, `rdk_model_zoo`, `/opt/hobot`, and one public
download). The build workflow excludes `*.hbm` from sync so these survive an
`rsync --delete`.

**A model's file name matches the model name inside it, deliberately.** Several
of these have sibling builds that differ only in calibration or in where the
graph was cut — `_aligned`, `_cut`, `_v3` are which build you have, not noise —
and an earlier build usually compiles cleanly and decodes to garbage. To verify
a fetch: empty this directory, re-run the script, and run the test suite.

| file | task | origin |
|------|------|--------|
| `ppocrv5_server_det_960x960.hbm`  | OCR det | PP-OCRv5 server, converted offline |
| `ppocrv5_server_rec_48x320.hbm`   | OCR rec | PP-OCRv5 server, converted offline |
| `ppocrv5_lcnet_cls_80x160.hbm`    | OCR cls | PP-LCNet textline ori, converted offline |
| `yolo26s_det_nashm_640x640_nv12.hbm` | det / tracking / async | converted offline |
| `yolo26n_{cls,detect,pose,seg,obb}_nashm_*.hbm` | cls/det/pose/seg/obb | rdk_model_zoo |
| `depth_any.hbm`                   | depth | Depth-Anything-V2, rdk_model_zoo |
| `yolov8_640x640_nv12.hbm`         | det (DFL head) | board-shipped |
| `deeplabv3plus_dilation1248_1024x2048_nv12.hbm` | semantic seg | board-shipped |
| `las2_m_{crop,int16}_nashm.hbm`   | stereo disparity | LAS2, converted offline |
| `yoloe_11s_coco80_{det,seg}_bpu_nashm_*.hbm` | open-vocab det / seg | YOLOE, converted offline |
| `edge_sam_{encoder,decoder_sp1,decoder_bp2}_nashm.hbm` | promptable seg | EdgeSAM, converted offline |
| `scrfd_10g_nashm_640x640_nv12.hbm` | face detection | SCRFD, converted offline |
| `arcface_r50_aligned_nashm_112x112.hbm` | face recognition | ArcFace R50, calibrated on ALIGNED crops |
| `bpu-siglip-base-patch16-224.hbm` | image embeddings | SigLIP, official prebuilt (downloaded) |
| `yolop_cut_nashm_640x640_nv12.hbm` | panoptic driving | YOLOP, graph CUT before the in-graph decode |
| `pidnet_s_nashm_1024x2048_nv12_v3.hbm` | real-time semantic seg | PIDNet-S, third calibration attempt |
| `vitpose_s_wholebody_nashm_256x192.hbm` | whole-body pose (133 kpt) | ViTPose-S, converted offline |
| `xfeat_nashm_640x480.hbm`         | sparse features | XFeat, converted offline |
| `realesr_general_x4v3_nashm_128.hbm` | x4 super-res (perceptual) | Real-ESRGAN Compact, converted offline |
| `spanx4_ch48_nashm_128.hbm`       | x4 super-res (fidelity) | SPAN, converted offline |

Model conversion (ONNX → `.hbm`) is offline on the x86 host;
this repo only consumes the finished binaries.
