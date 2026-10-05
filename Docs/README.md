# Model Pipelines documentation

Model Pipelines runs Hugging Face models inside Unreal Engine. You import a model once, then call it from Blueprints or C++. It runs on Unreal's Neural Network Engine (NNE) with the ONNX Runtime backend. Platform: Win64. Engines: UE 5.8 and UE 5.7.

You do not need to know machine learning. You do need to know Blueprints.

## The four tasks

| Task | What you give it | What you get back | Sample map |
|---|---|---|---|
| Text embedding | Sentences | One list of numbers per sentence. Similar meaning gives similar numbers. | `L_SemanticSearch` |
| Text classification | Sentences | A label and a score per sentence (for example POSITIVE 0.99). | `L_Sentiment` |
| Image classification | A texture or render target | The best labels for the picture, with scores. | `L_SceneClassifier` |
| Object detection | A texture or render target | Boxes with a label and a score. | `L_Detection` |

## Read in this order

1. [Install](Install.md): add the plugin to a project.
2. [Import a model](ImportModel.md): get the four sample models and import them.
3. [Samples](Samples.md): open and play the four sample maps. Each page also shows the same flow as Blueprint nodes.
4. [Nodes](Nodes.md): every Blueprint node, pin and result type, plus the C++ calls.
5. [AI agent tools (MCP)](MCP.md): optional. Let an AI agent import, validate, benchmark and try models in the editor (UE 5.8 only).
6. [Troubleshooting](Troubleshooting.md): messages you may see and what to do.
7. [Model licenses](ModelLicenses.md): licenses of the sample models and photos, and advice for your own models.

## License

The plugin is licensed under the Apache License 2.0. See the `LICENSE` file in the plugin folder.
