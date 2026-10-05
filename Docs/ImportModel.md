# Get and import models

Model Pipelines does not ship model files. You download the four sample models with a script, then import each folder into your project.

## Step 1: Get the models

You need Python 3.11 or newer. Python 3.11 is tested, and 3.13 also worked in our check. About 1 GB is downloaded, and the first run takes about 6 minutes.

This virtual environment is only for the download script. The editor runs import scripts (see the batch import section below) with its own Python (UE 5.8 ships 3.11.8), so do not `pip install` anything into the engine.

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

## Import from Blueprint or Python (batch import)

The menu import handles one folder at a time. To import many folders, or to import without clicking, call the function **Import Model Folder** (category **Model Pipelines|Editor**). It does the same work as the menu command. It runs in the editor, on the game thread, and blocks until it is done.

In a Blueprint (an Editor Utility Blueprint or Editor Utility Widget), add the node **Import Model Folder**. It takes one **Options** struct (**Pipeline Import Options**) and returns a **Pipeline Import Result**.

### Options

| Field | Default | What it does |
|---|---|---|
| **Source Folder** | empty | The model folder on disk. It holds `model.onnx` (or `onnx/model.onnx`), `config.json` and so on. |
| **Destination Path** | `/Game/Models` | The Content path where the two assets are created. |
| **Asset Name** | empty | Name of the package asset. Empty means the folder name (cleaned up). The model data asset is `<Asset Name>_Model`. |
| **Task** | None | Forces a task. None means detect it from the folder. |
| **Preferred Device** | Auto | The device preference written to the package. Auto uses the folder's `modelpipelines.json` value, or Auto. |
| **Overwrite Existing** | off | Updates assets that already exist (of the right classes) in place. Other objects are never deleted or replaced. |
| **Save** | on | Saves both assets to disk after a successful import. |
| **Validate** | on | Runs the full validator: model creation, task inputs and outputs, and test runs. The package settings are always checked. |
| **Validation** | smoke test on, GPU check on | The validator options used when **Validate** is on: **Run Smoke Test** and **Check Gpu**. |

### Result

| Field | What it holds |
|---|---|
| **Success** | True when both assets exist, are valid and (with **Save**) were saved. |
| **Package** | The model package asset. Null unless the assets exist. |
| **Model** | The model data asset. Null unless the assets exist. |
| **Report** | The validation messages (each has a severity, a code and a text) and the device checks. |

A failed import leaves nothing in the project, except when saving fails: then the assets exist but are not saved, and **Success** is false.

### Example: import all four sample models with Python

This is an example, not part of the plugin. Change `MODELS_ROOT` to the folder you gave to `fetch_reference_models.py --out`. It imports the four folders into `/Game/Models`, saves them, and prints each result.

```python
import unreal

# Example script. Set this to the --out folder of fetch_reference_models.py.
MODELS_ROOT = "C:/PipelineModels"
SLUGS = ["all-MiniLM-L6-v2", "distilbert-sst2", "resnet-50", "rtdetr_r18vd"]

failed = 0
for slug in SLUGS:
    options = unreal.PipelineImportOptions()
    options.source_folder = f"{MODELS_ROOT}/{slug}"
    options.destination_path = "/Game/Models"
    options.asset_name = slug
    options.save = True
    # Set options.overwrite_existing = True to import again over existing assets.

    result = unreal.PipelineEditorLibrary.import_model_folder(options)

    print(f"{slug}: success={result.success}")
    if result.package:
        print(f"  package: {result.package.get_path_name()}")
    for message in result.report.messages:
        print(f"  {message.severity} {message.code}: {message.message}")
    if not result.success:
        failed += 1

print(f"Imported {len(SLUGS) - failed} of {len(SLUGS)} models.")
```

You should see one `success=True` line per model, then `Imported 4 of 4 models.` The four packages are now in `/Game/Models`, saved.

You can run the script two ways. First, turn on the Python plugin: **Edit > Plugins**, search for **Python Editor Script Plugin**, enable it, and restart the editor.

- **In the editor.** Open **Window > Developer Tools > Output Log**. Switch the console box at the bottom from **Cmd** to **Python**. Enter this, with your own script path. Use forward slashes:

  ```
  py "C:/Scripts/import_models.py"
  ```

  Or paste the script text into the Python console.
- **Headless, from a terminal.** Save the script as a file, for example `C:\Scripts\import_models.py`. Put it in a path without spaces, or quote the path. Run:

  ```
  "<UE>\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" "<Project>.uproject" -ExecutePythonScript="<script.py>" -unattended -nosplash -RenderOffscreen
  ```

  Replace `<UE>` with your engine folder, `<Project>` with the full path of your project, and `<script.py>` with your script path. With `-unattended` the prints do not appear in the terminal. They are written to the project log, `<Project folder>\Saved\Logs\<Project name>.log`, as lines that start with `LogPython:`. Open the log and search for `LogPython`. Do not run this while the same project is open in the editor.

  `-ExecutePythonScript` only works when the Python Editor Script Plugin is enabled in the project. If you enabled it in **Edit > Plugins**, the editor already added it to your `.uproject` file. If you never opened the project in the editor, add this entry by hand to the `"Plugins"` list in the `.uproject` file:

  ```json
  "Plugins": [
    { "Name": "PythonScriptPlugin", "Enabled": true }
  ]
  ```

  If the list already has other entries, add this one next to them.

In Python, names follow Unreal's rule: the class is `unreal.PipelineEditorLibrary`, the function is `import_model_folder`, and fields are written in snake_case. The `b` prefix of true-or-false fields is dropped, so **Save** is `save` and **Overwrite Existing** is `overwrite_existing`.

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
for message in report.messages:
    print(f"{message.severity} {message.code}: {message.message}")
if not report.messages:
    print("No problems found")
```

An empty messages list (no errors and no warnings) means the package passed. For a package that passed, the output is one line:

```
No problems found
```

Any other output is a list of messages. Fix every error and run the check again.

To read which task a package serves from Python, use its **Task** property:

```python
pkg = unreal.load_asset("/Game/Models/resnet-50")
print(pkg.get_editor_property("task"))
```

This prints the task, for example `IMAGE_CLASSIFICATION`. The other values are `TEXT_EMBEDDING`, `TEXT_CLASSIFICATION`, `OBJECT_DETECTION` and `NONE`.

Next: [Samples](Samples.md).
