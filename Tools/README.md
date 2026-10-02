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
- `make_import_test_data.py [--variant visible|hidden] [--out-models DIR] [--out-goldens FILE]` writes the 17 tiny
  importer folders to `<out-models>/import_test/` and the expectations JSON
  (`<dev>/goldens/import/import_expectations.json`), adding the reference cases when those folders exist. Reference
  cases are cross-checked against transformers; a disagreement fails the script.
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
