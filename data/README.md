# data/

Repo-local assets used by the on-board tests and check-figure benchmarks
(`tests/board_models.py`, `scripts/board_bench.py`), so figures are reproducible
without depending on board-only `/app/res/assets`. All are env-overridable
(`BCDL_IMAGES`, `BCDL_FONT`, `BCDL_OCR_DICT`, `BCDL_AUDIO`, `BCDL_VAD_DATA`, `BCDL_KWS_DATA`).

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

## audio/ — speech samples for ASR

Four utterances from the **AISHELL-1** test set (16 kHz mono PCM16, 2.7–3.9 s),
with their reference transcripts in `transcripts.tsv` (`file<TAB>text`). Used by
`tests/test_asr_board_py.py`, which expects the shipped Paraformer build to
transcribe all four exactly. Clips 2–4 end on a character that only the CIF tail
weight fires, so they also catch a CIF that loses it. AISHELL-1 is released under the Apache License 2.0
(OpenSLR SLR33), so the clips ship with the repo.

## kws/ — keyword-spotting reference

`kws_encoder_ref.npz` (`encoder_out`, 92 × 320): the float KWS encoder's output
for `audio/aishell1_test_1.wav`, streamed chunk by chunk as on the board.
`tests/test_kws_board_py.py` checks the int16 build against it; the wake
words themselves are synthesised on the board with `TextToSpeech`.

## vad/ — VAD reference

`silero_ref.npz` (`probs`): the float Silero VAD v6.2.3 ONNX's speech
probability for each 512-sample chunk of the signal `tests/test_vad_board_py.py`
builds from `audio/` (0.5 s silence, clip 1, 1.2 s, clip 3, 1.2 s). The board
build is checked against it.

## tts/ — vocoder inputs and references

Two Matcha-TTS (zh-en) mel spectrograms, `matcha_mel_00.npy` (80 × 136, one
chunk) and `matcha_mel_36.npy` (80 × 245, two chunks), with the float Vocos
output for each (`vocos_float_*.wav`, 16 kHz PCM16) and the sentences in
`texts.tsv`. Used by `tests/test_vocoder_board_py.py`, which checks the BPU
Vocos build against the float waveforms and runs the result back through
Paraformer. The mels are the acoustic model's output on sherpa-onnx's own front
end (punctuation tokens, sentence split, noise scale 1.0), so they are what the
vocoder sees in deployment. Both models are Apache-2.0.

`frontend_ref.json` holds 100 sentences with the token ids sherpa-onnx's own
front end produces for them (from its debug log); `tests/test_tts_frontend.py`
requires `MatchaFrontend` to reproduce every one. `matcha_ref_00.npz` and
`matcha_ref_36.npz` hold, per sentence, the token ids, the durations and the
float acoustic model's mel with the decoder noise off, which
`tests/test_tts_board_py.py` compares the BPU build against.

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
