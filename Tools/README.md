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
