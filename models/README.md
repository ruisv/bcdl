# models/

Compiled BPU model binaries (`.hbm`) the harness and examples consume. The code
references them ONLY by the repo-relative path `models/<name>.hbm` (no absolute
paths), env-overridable per task (`BCDL_MODELS`, `BCDL_OCR_DET`, …).

**The `.hbm` files are NOT committed** — they total ~266 MB (incl. a 116 MB depth
model), too large for git (`.gitignore` excludes `*.hbm`). Populate this dir on
the board with:

```bash
scripts/fetch_models.sh        # copies from the board's source locations
```

`fetch_models.sh` is the single place the board's absolute source paths live
(the offline-converted models, `rdk_model_zoo`, and `/opt/hobot`). The build
workflow excludes `*.hbm` from sync so these survive an `rsync --delete`.

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

Model conversion (ONNX → `.hbm`) is offline on the x86 host;
this repo only consumes the finished binaries.
