# Get and import models

Model Pipelines does not ship model files. You download the four sample models with a script, then import each folder into your project.

## Step 1: Get the models

You need Python 3.11. About 1 GB is downloaded, and the first run takes about 6 minutes.

1. Open a terminal in your project folder (the one with the `.uproject` file).
2. Create and activate a virtual environment:

   ```
   python -m venv .venv && .venv\Scripts\activate
   ```

   You should see `(.venv)` at the start of the prompt.
3. Install the packages. The first comment line in `requirements.txt` gives the CPU-only torch index URL, which keeps the download small:

   ```
   pip install -r Plugins/ModelPipelines/Tools/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

4. Download and convert the models. Replace `<some folder>` with any folder on your disk, for example `C:\PipelineModels`:

   ```
   python Plugins/ModelPipelines/Tools/fetch_reference_models.py --out <some folder>
   ```

   The script prints each model's license as it goes. When it ends, `<some folder>` holds four folders:

   | Folder (slug) | Task |
   |---|---|
   | `all-MiniLM-L6-v2` | Text embedding |
   | `distilbert-sst2` | Text classification |
   | `resnet-50` | Image classification |
   | `rtdetr_r18vd` | Object detection |

   Each folder contains `model.onnx` plus the Hugging Face files (`config.json`, and `tokenizer.json` or `preprocessor_config.json`).

Options:

- `--out <folder>`: where the model folders go.
- `--only <slug>`: download one model only, for example `--only resnet-50` (several: `--only a,b`).
- `--cache <folder>`: the Hugging Face download cache folder.
- `--force`: redo the download and export even if a folder is up to date.

A rerun skips models that are already up to date. Run the script with `--help` to see all options.

## Step 2: Import a model folder

Do this once per model folder.

1. Open your project in the editor.
2. In the Content Browser, right-click the **Content** folder and choose **New Folder**. Name it `Models`. (Do this once.)
3. Right-click the **Models** folder.
4. In the menu, find the section **Model Pipelines**. Choose **Import Hugging Face Model Folder...**.
5. In the folder dialog, pick the model folder, the one that contains `model.onnx`. For example `C:\PipelineModels\all-MiniLM-L6-v2`. Click **Select Folder**.
6. A progress dialog appears. Wait for it to end. The importer reads the files, detects the task, and runs the model once on the CPU to check it.
7. You should see a notification **Imported `<name>`**.
8. In the `Models` folder you now have two assets:
   - `<folder name>`: the **model package**. Samples and nodes use this one.
   - `<folder name>_Model`: the model data. You never use it directly.

If the import fails you see **Model folder import failed** with a report, and nothing is created. See [Troubleshooting](Troubleshooting.md).

Repeat steps 3 to 8 for all four folders. Keep the folder names the script made. The samples look for these exact paths:

- `/Game/Models/all-MiniLM-L6-v2`
- `/Game/Models/distilbert-sst2`
- `/Game/Models/resnet-50`
- `/Game/Models/rtdetr_r18vd`

## What the importer needs in a folder

- `model.onnx` (or `onnx/model.onnx`). It must be one single file. Models that use ONNX external data, and models over 2 GB, are not supported.
- `config.json`. Classification and detection models need its `id2label` list. It also tells the importer which task the model does.
- Text models need `tokenizer.json`. The command `optimum-cli export onnx` writes it. Some Hub models (distilbert-sst2 is one) ship without it; our script makes it for you.
- Image models use `preprocessor_config.json` for size, mean and standard deviation. Without it the importer warns and uses 224 x 224 and the usual ImageNet values.

The importer detects the task from these files:

- A `modules.json` file (sentence-transformers) means text embedding.
- Otherwise `config.json` `architectures[0]` is used. It ends with `ForSequenceClassification` (text classification), `ForImageClassification` (image classification) or `ForObjectDetection` (object detection). A name ending in `Model`, with a `tokenizer.json`, means text embedding.

If you want to force a task, put a file `modelpipelines.json` next to `model.onnx`:

```json
{ "task": "TextEmbedding", "preferred_device": "Auto",
  "license": { "spdx_id": "Apache-2.0", "source_url": "https://huggingface.co/<repo>", "attribution": "<text>" } }
```

All keys are optional.

## Edit a package

Double-click a package asset to open it. You can change these fields:

| Field | What it does |
|---|---|
| **Task** | Which pipeline accepts this package. |
| **Labels** | Class names. The position in the list is the class number. |
| **Preprocess** | Text: `MaxTokens` (longest input, including special tokens), `bAddSpecialTokens`, `Pooling` (Mean, CLS or None), `bNormalizeEmbeddings`. Image: `ImageWidth`, `ImageHeight`, `ResizeMode`, `Resample`, `RescaleFactor`, `ImageMean`, `ImageStd`. |
| **Postprocess** | `ScoreActivation` (Softmax or Sigmoid), `NmsIouThreshold` and `MaxDetections` (detection). |
| **PreferredDevice** | Auto, CPU or GPU. |
| **License** | SPDX id, source URL and attribution. |

Defaults come from the model folder. Change them only if you know why.

> **Warning: re-validate after editing `MaxTokens` or replacing the tokenizer.**
> A package whose tokens exceed the model's tables can pass in the editor and still crash a packaged game (an ONNX Runtime error outside the editor).
> Run **Validate Model Package** (a Blueprint or Python function in the category **Model Pipelines|Editor**) on the package. Read the report and fix every error.

How to run it from a Blueprint: in an Editor Utility Blueprint, call **Validate Model Package** with **Package** set to your asset and **Options** left at the defaults (smoke test on, GPU check on). It returns a report. Print it or check it for errors.

Example in Python (Output Log, **Python** console). The Python names follow the usual `unreal.*` naming of the C++ functions:

```python
import unreal
pkg = unreal.load_asset("/Game/Models/all-MiniLM-L6-v2")
report = unreal.PipelineEditorLibrary.validate_model_package(pkg, unreal.PipelineValidateOptions())
print(report)
```

Next: [Samples](Samples.md).
