# Changelog

DRAFT for Noah. The version and date are proposals, see `.pipelines-dev/logs/m7/wp76-proposals.md`.

## [Unreleased]

First public release (proposed version 0.1.0, beta).

### Added

- Four task pipelines on Unreal's Neural Network Engine (NNE) with the ONNX Runtime backend: text embedding, text classification, image classification and object detection.
- Model packages: a package asset holds an ONNX model, its tokenizer or image settings, labels and license.
- Import: **Import Hugging Face Model Folder...** in the Content Browser, plus Blueprint and Python calls for batch import. A validation pass checks the tokenizer, CPU and GPU creation, task inputs and outputs, and a smoke run.
- Blueprint async nodes: Embed Texts, Classify Texts, Classify Image and Detect Objects. The same calls are available from C++.
- Helpers for results: cosine similarity and top-K similarity, top-K labels, box overlap and non-maximum suppression.
- Non-blocking image input: textures and render targets are read back without stalling the game thread.
- Four sample maps: `L_SemanticSearch`, `L_Sentiment`, `L_SceneClassifier` and `L_Detection`, with six sample photos.
- Scripts to download and convert the four sample models (`Tools/`).
- User documentation in `Docs/`: install, import a model, samples, nodes, troubleshooting, model licenses.
- Optional AI agent tools (MCP) in the editor, UE 5.8 only: list, import, validate, benchmark and try models. Importing asks for confirmation first. See `Docs/MCP.md`.
- Automation tests under `Pipelines.*`.

### Changed

- The plugin ships its own internal tokenizer module (a renamed build of the same tokenizer library) and no longer needs the separate Tokenizers plugin. You can still install the standalone Tokenizers plugin next to it. See `Docs/Install.md`.

### Known limitations

- Windows 64-bit only.
- The MCP tools need UE 5.8. They are not available on UE 5.7.
- Inference runs on the CPU or on the GPU through DirectML (NNE ORT). If DirectML is not available, the plugin uses the CPU.
- Tested on UE 5.8 and UE 5.7. UE 5.6 is planned and not tested.
- The plugin is in beta. The API may still change.
- You bring the models. Check each model's license (see `Docs/ModelLicenses.md`).
