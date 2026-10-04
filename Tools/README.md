# Tools

Python tooling. Use the env interpreter by full path:
`D:\Micromamba\micromambaenv\envs\pipelines\python.exe`.

Tokenizer goldens (WP0.2):

1. `fetch_tokenizers.py --out <models dir>` checks each model's license on its Hugging Face
   card (Apache-2.0, MIT or BSD only), then downloads the pinned tokenizer files and writes
   `SOURCE.json` per model. Safe to rerun; it skips files that already match.
2. `make_tokenizer_goldens.py --models <models dir> --strings <strings.json> --out <goldens dir>`
   writes one goldens JSON per tokenizer. The strings file is a JSON array of
   `{"id": "sNN", "text": "..."}`; `data/tokenizer_strings.json` holds the standard 24.
Test models and goldens (WP1.1), output under `<dev>/` (`MODELPIPELINES_DEV_DIR` or `../.pipelines-dev`):

- `make_test_models.py [--out DIR]` builds 4 tiny ONNX models (opset 17, IR 8) into
  `<dev>/models/test` and prints their SHA-256.
- `make_test_goldens.py [--models DIR] [--out FILE] [--seed 1234] [--batches 1,3,8,256]` runs them in
  onnxruntime (CPU) and writes `<dev>/goldens/test_models/test_models.json`.

3. `python -m pytest Tools/tests -q` runs the tests (offline; skips if files are absent).

Importer test data (WP2.1), output under `<dev>/`:

- `fetch_reference_models.py [--force] [--only slug,slug]` resolves pinned revisions to full SHAs, checks each
  model card says `apache-2.0` before downloading weights, exports ONNX (opset 17, single file) for the 4 reference
  models into `<dev>/models/reference/<slug>/` (the repo's own files unchanged + `modelpipelines.json` + `model.onnx`),
  runs a onnxruntime-vs-torch check (fails above 1e-3), and writes `<dev>/models/reference/manifest.json`.
  HF cache goes to `<dev>/cache/hf`. Skips models whose `model.onnx` sha256 matches the manifest.
- `make_import_test_data.py [--variant visible|hidden] [--out-models DIR] [--out-goldens FILE]` writes the 27 tiny
  importer folders to `<out-models>/import_test/` and the expectations JSON
  (`<dev>/goldens/import/import_expectations.json`), adding the reference cases when those folders exist. Reference
  cases are cross-checked against transformers; a disagreement fails the script.
  Resize-then-crop cases (WP4.0a, contract C25), tiny 3-label ImageClassification models: `deit_style` (DeiT, `size 40`,
  crop 32 -> StretchCenterCrop 40x40, image 32x32), `deit_hw_crop` (`{h 40, w 48}` + crop `{h 32, w 36}`),
  `deit_legacy_fe` (`feature_extractor_type DeiTFeatureExtractor`, as `deit_style`), `clip_style_crop` (CLIP, int size +
  crop -> ShortestEdgeCenterCrop 40, image 32x32) and `square_int_nocrop` (DeiT, `size 32`, no crop -> Stretch 32x32).
  Hidden variant uses other sizes (52/44, `{h 50, w 60}`/`{h 40, w 48}`). Each case is cross-checked against
  `AutoImageProcessor` (size, crop_size, do_center_crop, output shape for a 300x200 image). `preprocess` has
  `resize_width`/`resize_height` (0 unless StretchCenterCrop). A pytest re-scans transformers for the
  ShortestEdgeIntClasses list (union of a source scan and an empirical size=40 scan) (`SHORTEST_EDGE_INT_CLASSES`).
  WP4.4a (contract C32) adds `resample` (enumerator name), `nms_iou_threshold` 0.7 and `max_detections` 100 to every
  `preprocess` and the cases `levit_style` (Levit, `{shortest_edge 28}`, crop 28 -> ShortestEdgeCenterCrop, resize 32,
  Bicubic), `crop_no_size` (DeiT `{h,w 32}`, do_center_crop, no crop_size, resample 2 -> Stretch, Bilinear, warning
  UnsupportedResize), `resample_box` (ViT, resample 4 -> Box), `resample_bad` (ViT, resample "bicubic" -> Bilinear +
  UnsupportedResize) and `effnet_default` (EfficientNet, no resample -> Nearest). `RESAMPLE_DEFAULTS` (37 classes) is
  pinned by a pytest that imports every `image_processing_*.py` and prints modules that fail to import. Every image case
  is cross-checked against `AutoImageProcessor(use_fast=False).resample` (`resample_bad` skipped: HF keeps the string);
  LeViT is also compared pixel-wise with a manual PIL resize + centre crop. Hidden variant: sizes 48 (LeViT S 42, crop 44, resize 48).
  Worst-case cases (WP3.6b): `small_vocab` (Gather table 30000 rows, hidden: vocab-1000) and `short_positions`
  (position table 32 rows, hidden 16; MaxTokens 64, hidden 48) pass a Hello world run but fail the worst case
  (`SmokeRunFailed`); the script checks this on onnxruntime and also runs the worst case on the two reference text models.

`fetch_reference_models.py` also guarantees a `tokenizer.json` in text reference folders. A repo's own file is never
overwritten; when a repo ships none (distilbert-sst2 has only `vocab.txt`), it is converted with
`AutoTokenizer(use_fast=True).backend_tokenizer.save`, verified against the slow tokenizer on 6 strings (fails on any
difference), and recorded in `manifest.json` as `tokenizer_json`. This also runs on reruns without `--force`.

Text goldens (WP3.3), output `<dev>/goldens/text/text_goldens.json`:

- `make_text_goldens.py [--sentences FILE] [--out FILE] [--variant visible]` reads 50 sentences
  (`data/text_sentences.json`, schema `pipelines-text-sentences/1`; any file of that schema with 5+ entries works) and
  writes `pipelines-text-goldens/1`: for all-MiniLM-L6-v2 and distilbert-sst2 the PyTorch reference (batch 1, fp32,
  weights from the offline HF cache at the pinned revision) plus an onnxruntime CPU cross-check (one right-padded
  batch), and for the two tiny importer models the onnxruntime result as the reference. Each model block holds the
  `package` settings (checked against the importer expectations and derived independently from the folder files), the
  token ids from `tokenizers` and the per-sentence embedding or logits/scores/label; MiniLM also gets a top-5
  similarity table for the first 5 sentences. Fails if the ONNX/PyTorch gap is too large (cosine < 0.99999, logits
  > 1e-4) or an SST-2 top margin is < 0.01. No network; output is byte-identical on reruns. Runs in about 45-60 s on CPU.

Vision goldens, part A (WP4.1), output `<dev>/goldens/vision/vision_goldens.json` (schema `pipelines-vision-goldens/1`, C30):

- `make_vision_goldens.py [--variant visible|hidden] [--seed N] [--out-dir DIR] [--models-out DIR] [--source-models DIR]`
  writes 10 deterministic synthetic PNGs (`images/`), 60 Pillow resize results (6 filters x 10 size pairs, `resize/`),
  52 preprocess cases (7 HF slow-processor configs: tensor `.f32` plus the 8-bit cropped PNG, `preprocess/`) and 4 tiny
  models with onnxruntime references (`tiny_image_classification`, `tiny_object_detection` from `import_test/`, plus two
  new designed-output detection models written to `<models-out>/vision_test/tiny_det_designed_{sigmoid,softmax}`; the
  design table is a comment block in the script). Every preprocess case is cross-checked against an independent
  implementation of the C27 pixel rules (`cropped` byte-exact, tensor <= 1e-5, else the script exits non-zero with
  `NEEDS:`). `detections` come from a numpy implementation of the C29 decode rules 1-4 and C26 NMS (float64 IoU);
  tiny_object_detection cases are skipped when a score is within 0.02 of the threshold or a same-class IoU within 0.02
  of the NMS threshold (printed, not stored).
- Hidden variant: `--variant hidden --seed 777 --out-dir <dir> --models-out <dir>` (other images and resize pairs, designed
  models with 10 queries and labels w/x/y/z; the two importer tiny models are copied into `<models-out>/import_test`).
  `root` in the JSON is the absolute `--out-dir`, so the JSON bytes depend on that path (rerun into the same path: identical).
- No network. About 15 s. Tests: `tests/test_vision_goldens.py` (the hidden variant is run into a temp dir).

Reference photos and goldens, part B (WP4.2, C31):

- `fetch_test_images.py` reads `data/test_images.json` (schema `pipelines-test-images/1`, committed, metadata only: four pools
  `cls_visible` / `cls_hidden` / `det_visible` / `det_hidden`, each entry with id, Commons title, pinned upload.wikimedia.org
  URL, licence, author and sha256). Only CC0 and Public Domain files are allowed; any other licence string is refused.
  It downloads into `<dev>/cache/test_images/` (cached files with a matching sha256 are not downloaded again), verifies the
  sha256 (mismatch = error) and writes the canonical 8-bit RGB PNGs to `<dev>/goldens/vision/reference_images/<id>.png`
  (no EXIF transpose; thumbnail to at most 1024 px). `--pools` limits the pools. `--discover [--kinds cls,det]` rebuilds
  pools from the Commons API (search restricted to the CC-Zero / Public domain categories, licence from `extmetadata`,
  1024-px thumbnails, fixed shuffle, files split alternately into the visible and hidden pool, never in two pools); it
  overwrites the pinned JSON, so run it only to refresh the candidates. Network: Commons API and upload.wikimedia.org only, with a
  descriptive User-Agent and 0.6 s between requests. Images are never committed.
- `make_vision_goldens.py` now also adds `reference_images`, `reference_models` (resnet-50 with `scores` softmax float64,
  `index`, `top_margin`; rtdetr_r18vd with `detections` from the C29 decode and `hf_detections` from
  `post_process_object_detection(threshold=0.5)`) and `reference_generator`. Reference = PyTorch fp32 from the offline HF cache at the
  pinned revision (`HF_HUB_OFFLINE`), input = the HF slow processor on the PNG; onnxruntime CPU on the same tensor is the
  cross-check. Per pool it takes the first candidates that pass the selection rules (20 classification, 10 detection):
  top margin >= 0.05 and equal ORT argmax; for detection >= 1 detection, no torch score in [0.48, 0.52], no same-class pair
  with IoU in [0.65, 0.75]. A candidate whose ORT output differs from torch by more than 1e-3 (logits, classification and
  detection) or 1e-4 (detection boxes) is skipped with that reason (C31 says fail; the script ends with `NEEDS:` when a pool
  cannot be filled). Every skipped candidate is listed in `skipped` with its reason. `--reference-images` and `--test-images`
  override the PNG folder and the pools file. The hidden variant takes the `*_hidden` pools and copies the selected PNGs next to its JSON.

## For users

Everything here runs with plain Python 3.11 and needs no Unreal Engine and no dev folder.

1. Create a virtual environment and install the pinned packages (the CPU build of torch comes from the PyTorch index):
   `python -m venv .venv`, activate it, then
   `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`.
   (`requirements-dev.txt` adds `pytest` for running the Tools tests.)
2. Get the four reference models as ONNX folders:
   `python fetch_reference_models.py --out <folder>` (optionally `--only resnet-50`, `--cache <hf cache folder>`).
   It writes `<folder>/<slug>/` (`model.onnx`, the model's own config files, `modelpipelines.json` with the licence) for
   `all-MiniLM-L6-v2`, `distilbert-sst2`, `resnet-50` and `rtdetr_r18vd`, plus `<folder>/manifest.json`. Per model it
   prints the repo and revision, the licence read from the Hugging Face model card (only apache-2.0, mit or bsd-* are
   accepted, anything else stops the script) and the folder. A rerun skips models that are up to date.
   Then import each folder in the editor: in the Content Browser right-click the folder Content/Models > Model Pipelines > Import Hugging Face Model Folder... (assets go to /Game/Models).
3. Sample photos (CC0 or public domain only): `python fetch_sample_images.py [--out <folder>] [--check]` downloads the 6
   images listed in `data/sample_images.json`, verifies their sha256 and writes `<name>.png` (RGB, long edge <= 1024).
   `--check` runs resnet-50 and rtdetr_r18vd (PyTorch, the same preprocessing as the goldens) on the PNGs and asserts what
   each image should show; it needs the folders from step 2 via `--models <folder>` and the HF cache via
   `--hf-cache <folder>` (defaults point into the dev folder). `--verify-license` re-reads each licence from Wikimedia
   Commons.
