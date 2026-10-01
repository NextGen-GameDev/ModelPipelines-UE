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
- `make_import_test_data.py [--variant visible|hidden] [--out-models DIR] [--out-goldens FILE]` writes the 15 tiny
  importer folders to `<out-models>/import_test/` and the expectations JSON
  (`<dev>/goldens/import/import_expectations.json`), adding the reference cases when those folders exist. Reference
  cases are cross-checked against transformers; a disagreement fails the script.

`fetch_reference_models.py` also guarantees a `tokenizer.json` in text reference folders. A repo's own file is never
overwritten; when a repo ships none (distilbert-sst2 has only `vocab.txt`), it is converted with
`AutoTokenizer(use_fast=True).backend_tokenizer.save`, verified against the slow tokenizer on 6 strings (fails on any
difference), and recorded in `manifest.json` as `tokenizer_json`. This also runs on reruns without `--force`.
