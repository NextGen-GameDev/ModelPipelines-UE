# ModelPipelines for Unreal Engine

Run Hugging Face ONNX models as task pipelines in Unreal Engine: text embeddings, text
classification, image classification and object detection, from C++ and Blueprints, on
Unreal's Neural Network Engine (NNE).

**Status:** early development (milestone M1: plugin skeleton and NNE backend). Not ready for use.

- Platform: Win64. Engines: UE 5.8 (primary) and 5.7.
- Depends on the [Tokenizers](https://github.com/NextGen-GameDev/Tokenizers-UE) plugin and
  the engine's NNERuntimeORT plugin.

## License

Apache License 2.0, see [LICENSE](LICENSE). Contributions need the contributor agreement in
[CLA.md](CLA.md), see [CONTRIBUTING.md](CONTRIBUTING.md).
