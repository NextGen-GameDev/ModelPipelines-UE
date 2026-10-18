# Changelog

## [Unreleased]

### Added

- Speech to text (fifth task): `USpeechToTextPipeline`, created with `USpeechPipelineLibrary::CreateSpeechToTextPipeline` (Blueprint: **Create Speech To Text Pipeline**).
- Blueprint nodes **Transcribe Speech** and **Transcribe Sound Wave**, and the C++ calls `Transcribe` and `TranscribeSoundWave`. The result has the text and a start and end time per word (80 ms resolution).
- Supports Phonon-2 (Fermion Research, English, CC-BY-4.0, not bundled). One script downloads, verifies and converts it (`Tools/fetch_phonon2.py --out <dir>`); shipped variant `fp16w`. 2.16 % word error rate on LibriSpeech test-clean; real-time factor 0.076 on CPU and 0.053 on GPU (RTX 3080 laptop, UE 5.8).
- Any sample rate from 1000 to 384000 Hz and any channel count. Up to 600 s per call; audio over 35 s is split at quiet points into 25 to 35 s windows.
- Sound waves (PCM, ADPCM, Bink, Opus) are decoded off the game thread. Silence gate (`SilenceThresholdDb`, -60 dBFS by default).
- Docs: `Docs/SpeechToText.md`, and speech sections in the import, node, license and troubleshooting pages.
- Real-time speech: Blueprint nodes **Start Listening** (pins On Speech Started, On Partial, On Final, On Speech Ended, On Stopped, On Failure), **Stop Listening** and **Push Speech Audio**. C++: `USpeechToTextPipeline::StartStream`, `USpeechStream`, `USpeechMicrophoneSource`.
- An energy voice-activity detector finds utterances (12 dB over the noise floor, above -55 dBFS, 0.3 s hangover, 30 s cap). Finals work with the ONNX engine. Partials need the Phonon engine of ModelPipelines Generative.
- Docs: "Real-time speech" in `Docs/SpeechToText.md`, plus node and troubleshooting entries.

### Known limitations

- English only. About 4 GB of RAM per speech pipeline.
- Procedural sound waves are not supported.
- The decoder always runs on the CPU.

## [0.1.0] - 2026-10-06

First public release (beta).

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
