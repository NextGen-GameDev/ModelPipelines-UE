# Fab listing: Model Pipelines

Ready to paste (2026-10-10). Price: free. Field names follow the usual code-plugin form; check limits in the Fab publisher portal.

## Title (max 60 characters)

Model Pipelines: Hugging Face ONNX Models in Blueprints

(54 characters)

## Short description (max 200 characters)

Run Hugging Face ONNX models in Unreal: text embeddings, text classification, image classification and object detection, from Blueprints and C++, on NNE. (155 characters)

## Long description

Model Pipelines runs Hugging Face models inside Unreal Engine. You import a model once, then call it from Blueprints or C++. It runs on Unreal's Neural Network Engine (NNE) with the ONNX Runtime backend. You do not need to know machine learning. You do need to know Blueprints.

### Who it is for

Unreal developers who want to add a small language or vision model to a game or tool, such as search by meaning, sentiment, scene labels or object boxes, without writing inference code.

### What it does

Four tasks:

| Task | You give it | You get back |
|---|---|---|
| Text embedding | Sentences | A list of numbers per sentence. Similar meaning gives similar numbers. |
| Text classification | Sentences | A label and a score per sentence. |
| Image classification | A texture or render target | The best labels, with scores. |
| Object detection | A texture or render target | Boxes with a label and a score. |

### Features

- Blueprint async nodes: Embed Texts, Classify Texts, Classify Image and Detect Objects. The same calls are available from C++.
- Model packages: one asset holds the ONNX model, its tokenizer or image settings, labels and license.
- Import: **Import Hugging Face Model Folder...** in the Content Browser, plus Blueprint and Python calls for batch import. A validation pass checks the tokenizer, CPU and GPU creation, task inputs and outputs, and a smoke run.
- Result helpers: cosine similarity and top-K similarity, top-K labels, box overlap and non-maximum suppression.
- Non-blocking image input: textures and render targets are read back without stalling the game thread.
- Runs on the CPU, or on the GPU through DirectML (NNE ORT). If DirectML is not available, the plugin uses the CPU.
- Built-in tokenizer. No separate Tokenizers plugin is needed. The standalone Tokenizers plugin can still be installed next to it.
- Optional AI agent tools (MCP) in the editor, UE 5.8 only: list, import, validate, benchmark and try models. Importing asks for confirmation first.

### Four sample maps

| Map | Task |
|---|---|
| `L_SemanticSearch` | Text embedding |
| `L_Sentiment` | Text classification |
| `L_SceneClassifier` | Image classification (four sample photos) |
| `L_Detection` | Object detection (two sample photos) |

Each map needs one model package, which you import yourself (see below).

### Models are not included

The plugin ships no model. You bring the models, and each model has its own license. The plugin has scripts (`Tools/`) that download and convert four tested models from Hugging Face. All four are Apache-2.0:

| Task | Model | License |
|---|---|---|
| Text embedding | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 |
| Text classification | [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | Apache-2.0 |
| Image classification | [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) | Apache-2.0 |
| Object detection | [PekingU/rtdetr_r18vd](https://huggingface.co/PekingU/rtdetr_r18vd) | Apache-2.0 |

Read the license of any model before you ship it. See `Docs/ModelLicenses.md`.

### Performance

The model runs in the background, not on the game thread. Classifying a 1080p render target costs the game thread a median of 0.05 ms per call (90th percentile 0.06 ms; UE 5.8.3, D3D12, RTX 3080 Laptop GPU, measured with Unreal Insights), because reading the texture back does not block. Creating a pipeline loads the model once and takes longer (about 40 to 180 ms in our tests), so create pipelines during loading, not mid-game.

## Technical details

- Features: Blueprint async nodes; C++ API; model package asset; Content Browser import; four sample maps; optional MCP tools (UE 5.8).
- Code modules: ModelPipelinesCore, ModelPipelinesTokenizer, ModelPipelinesText, ModelPipelinesVision, ModelPipelinesSamples (Runtime); ModelPipelinesEditor, ModelPipelinesMCP (Editor). All Win64 only.
- Main types: the model package asset, the four pipeline classes and the result type (see `Docs/Nodes.md`).
- Platforms: Win64.
- Supported engine versions: 5.7 and 5.8. UE 5.6 is planned and not tested.
- Network use: none at runtime. The model download scripts use the network, on your machine, before you import.
- Dependencies: the engine plugin NNERuntimeORT. For the MCP tools on 5.8: Unreal MCP, Toolset Registry and Python Editor Script Plugin (all optional, editor only).
- Documentation: https://github.com/NextGen-GameDev/ModelPipelines-UE/blob/main/Docs/README.md (also `Docs/` in the plugin).
- Example project / sample maps: yes, the four sample maps above, in the plugin content under `/ModelPipelines/Samples/Maps/`.
- Version: 0.1.0 (beta).

## Tags (max 10)

AI, Machine Learning, NNE, ONNX, Hugging Face, Blueprint, Embeddings, Image Classification, Object Detection, Text Classification


## Category suggestion

Code Plugins (pick the closest subcategory in the portal). The `.uplugin` category is "AI".

## What's included

- The plugin (source code, Win64): Core, Tokenizer, Text, Vision, Samples, Editor and MCP modules.
- Four sample maps and six sample photos (CC0 or public domain).
- User documentation in `Docs/`: install, import a model, samples, nodes, troubleshooting, model licenses.
- Scripts to download and convert the four sample models (`Tools/`).
- `THIRD_PARTY_NOTICES.md` and `CHANGELOG.md`.
- Not included: any model file.

## Requirements

- Windows 64-bit.
- Unreal Engine 5.7 or 5.8.
- The engine plugin NNERuntimeORT (ships with the engine).
- To get the sample models: Python 3.11 or newer, the packages in `Tools/requirements.txt`, and network access, once (about 1 GB, about 6 minutes). See `Docs/ImportModel.md`.
- No extra build step: the prebuilt tokenizer library is included.
- For the MCP tools: UE 5.8 and an MCP client such as Claude Code or Cursor.

## Known limitations

- Windows 64-bit only.
- The MCP tools need UE 5.8. They are not available on UE 5.7.
- Inference runs on the CPU or on the GPU through DirectML (NNE ORT). If DirectML is not available, the plugin uses the CPU.
- Tested on UE 5.8 and UE 5.7. UE 5.6 is planned and not tested.
- The plugin is in beta. The API may still change.
- You bring the models. Check each model's license.

## Support

Email: support@holdoutlabs.ai
Issues: https://github.com/NextGen-GameDev/ModelPipelines-UE/issues

## License notes

- The plugin code is under the Apache License 2.0 (`LICENSE`). Contributions need the contributor agreement in `CLA.md`.
- Third-party components and their licenses (tokenizers-cpp, Hugging Face tokenizers, ONNX Runtime, the sample models and photos) are in `THIRD_PARTY_NOTICES.md`.
- Models carry their own licenses. See `Docs/ModelLicenses.md`.
- Free. The code stays under Apache-2.0; `LICENSE` ships in the package.

## Related

Tokenizers (free Fab listing) is a separate plugin by the same team. It is not needed for Model Pipelines.
