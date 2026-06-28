# data/

Repo-local assets used by the on-board tests and check-figure benchmarks
(`tests/board_models.py`, `scripts/board_bench.py`), so figures are reproducible
without depending on board-only `/app/res/assets`. All are env-overridable
(`BCDL_IMAGES`, `BCDL_FONT`, `BCDL_OCR_DICT`).

## images/ — sample inputs (copied from ccdl `examples/images/`)

| file | task | origin |
|------|------|--------|
| `bus.jpg`    | det / det_dfl / seg / depth | Ultralytics COCO sample |
| `zidane.jpg` | pose                        | Ultralytics COCO sample |
| `ocr.jpg`    | ocr (Chinese text)          | PaddleOCR-style label image |
| `obb.jpg`    | obb                         | DOTA aerial scene |
| `bird.jpg`   | (unused; ImageNet bird)     | ImageNet sample |
| `stereo_left.png` / `stereo_right.png` | stereo (LAS2 disparity/depth) | rectified indoor pair (960×540), from the LiteAnyStereo demo set |

The stereo pair is a single rectified left/right scene used by
`tests/test_stereo_board_py.py` to drive `StereoPipeline` on the LAS2 `.hbm`.
It is committed losslessly (PNG) so the disparity is byte-reproducible — the
crop-calibrated model yields a stable mean ≈ 93 px on it.

cls (zebra) and semseg (cityscapes) have no ccdl counterpart and load from the
board's `/app/res/assets`.

## fonts/ — CJK font for figure text rendering

`SourceHanSansSC-Regular.otf` (Source Han Sans SC) — used by `cv2.freetype` to
render labels, including the recognised **Chinese** text in the OCR check figure.
Licensed under the SIL Open Font License 1.1 (see `OFL.txt`), so it ships freely
with the repo.

## OCR character dictionaries

- `ppocr_keys_v5_18385.txt` — **active (PP-OCRv5)**: `blank` + PaddleOCR
  `ppocr_keys_v5.txt` (18383) + trailing space = 18385 classes (matches the
  PP-OCRv5 server rec output channels).
- `ppocr_keys_v1_6625.txt` — legacy (PP-OCRv3): `blank` + `ppocr_keys_v1.txt`
  (6623) + space = 6625 classes.
