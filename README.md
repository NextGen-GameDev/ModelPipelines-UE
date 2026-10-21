# ModelPipelines for Unreal Engine

Run Hugging Face ONNX models as task pipelines in Unreal Engine: text embeddings, text
classification, image classification and object detection, from C++ and Blueprints, on
Unreal's Neural Network Engine (NNE).

**Status:** early development. Draft documentation below; the API may still change.

- Platform: Win64. Engines: UE 5.8 (primary) and 5.7.
- Depends only on the engine's NNERuntimeORT plugin. Tokenization is built in (an internal module with its own renamed copy
  of the tokenizer library), so it also works next to the [Tokenizers](https://github.com/NextGen-GameDev/Tokenizers-UE) plugin.

## Documentation

Start at [Docs/README.md](Docs/README.md).

- [Install](Docs/Install.md)
- [Get and import models](Docs/ImportModel.md)
- [Samples](Docs/Samples.md): four ready-to-play maps
- [Blueprint nodes and C++ calls](Docs/Nodes.md)
- [AI agent tools (MCP)](Docs/MCP.md): optional, UE 5.8 only
- [Troubleshooting](Docs/Troubleshooting.md)
- [Model licenses](Docs/ModelLicenses.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Changelog](CHANGELOG.md)

## License

From GitHub: Apache License 2.0, see [LICENSE](LICENSE). From Fab: the Fab license applies, and the package has no separate license file. Third-party licenses are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Contributions need the contributor agreement in
[CLA.md](https://github.com/NextGen-GameDev/ModelPipelines-UE/blob/main/CLA.md), see [CONTRIBUTING.md](https://github.com/NextGen-GameDev/ModelPipelines-UE/blob/main/CONTRIBUTING.md).
