# AI agent tools (MCP)

Model Pipelines adds five tools to the editor's MCP server. An AI agent such as Claude Code or Cursor can use them to import, validate, benchmark and try models in your project. You do not need this page to use the plugin. It is optional and for the editor only.

The tools:

- List the model packages in your project.
- Import a model folder (only when you say yes, see [The confirm rule](#the-confirm-rule)).
- Validate a model package.
- Benchmark a model package on the CPU and the GPU.
- Run a model once on your own text or texture and show the result.

## Requirements

- Unreal Engine 5.8. The tools are not available on 5.7 or 5.6. There they are simply absent, and the rest of the plugin works as before.
- Three engine plugins, all enabled in your project:
  - **Unreal MCP** (its settings page is called **Model Context Protocol**)
  - **Toolset Registry**
  - **Python Editor Script Plugin**
- An MCP client, for example Claude Code or Cursor.

If a plugin is missing, the editor writes one line to the Output Log (category `LogPython`) and the tools do not appear. Nothing else breaks. On UE 5.7, where the Toolset Registry plugin does not exist, the line is:

```
LogPython: Model Pipelines MCP tools not registered: the Toolset Registry plugin is not loaded
```

Other causes use the same start (`Model Pipelines MCP tools not registered: `) with a different reason after the colon.

## Step 1: Enable the plugins

1. Open your project in the editor.
2. Choose **Edit > Plugins**.
3. Search for **Unreal MCP**, **Toolset Registry** and **Python Editor Script Plugin**. Tick **Enabled** on each one.
4. Click **Restart Now**. The editor closes and opens again.

Unreal MCP and Toolset Registry are marked experimental in UE 5.8. Python Editor Script Plugin is marked beta.

## Step 2: Start the MCP server

The server is off by default. Start it in one of these ways.

**Start it by hand each time:**

1. Open **Window > Developer Tools > Output Log**.
2. In the console box at the bottom, make sure the mode is **Cmd**.
3. Enter `ModelContextProtocol.StartServer`.
4. You should see this line from `LogModelContextProtocol`:

```
LogModelContextProtocol: Starting MCP server on port 8000 (override with -ModelContextProtocolPort=N).
```

To use another port, enter `ModelContextProtocol.StartServer <port>`. To stop it, enter `ModelContextProtocol.StopServer`.

**Start it when the editor opens:**

- Start the editor with the command line option `-ModelContextProtocolStartServer`, or
- choose **Edit > Editor Preferences**, open **General > Model Context Protocol**, and tick **Auto Start Server**.

The same settings page holds the **Server Url Path** (default `/mcp`) and the **Server Port Number** (default `8000`).

The default address is:

```
http://127.0.0.1:8000/mcp
```

The server dies when you close the editor. Start it again after each restart.

> **Security.** By default the server listens on this computer only (loopback) and has no password. Do not change the engine's HTTP server bind address, and do not expose the port to a network, a tunnel or a port forward. Anyone who can reach it can run editor tools.

## Step 3: Connect a client

Claude Code reads a `.mcp.json` file in your project folder. Create it with this text:

```json
{
  "mcpServers": {
    "unreal": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Start Claude Code in that folder and approve the server when it asks. Other clients have their own setup. The engine can write the files for you: in the Output Log enter `ModelContextProtocol.GenerateClientConfig <ClaudeCode|Cursor|VSCode|Gemini|Codex|All>`. It uses the port and path from the settings page. The files go in your project folder (in a launcher install of the engine):

| Client | File |
|---|---|
| ClaudeCode | `.mcp.json` |
| Cursor | `.cursor/mcp.json` |
| VSCode | `.vscode/mcp.json` |
| Gemini | `.gemini/settings.json` |
| Codex | `.codex/config.toml` |

Keep the editor open while the client works.

## Find the tools

The server lists three general tools: `list_toolsets`, `describe_toolset` and `call_tool`. The agent finds ours like this:

1. It calls `list_toolsets`. You should see `model_pipelines_mcp.toolset.ModelPipelinesTools` in the list.
2. It calls `describe_toolset` with that name. You should see the five tools below with their parameters.
3. It calls `call_tool` with the toolset name, a tool name and the arguments.

The toolset name is:

```
model_pipelines_mcp.toolset.ModelPipelinesTools
```

**Enable Tool Search** is on by default (**Edit > Editor Preferences > General > Model Context Protocol > Tools**). If you turn it off, the server registers every tool as its own MCP tool when the editor starts, and the agent can call them directly without `call_tool`.

To find the sample textures, an agent can call `find_assets` in `editor_toolset.toolsets.asset.AssetTools` (see [Sample textures](#sample-textures)).

## The tools

Every tool returns text that holds JSON. A failure comes back as a tool error with a message, for example `Model package not found: /Game/Models/nothing`. The examples below are shortened: they keep only fields seen in a real run on UE 5.8.3.

Use the parameter names exactly as listed. An unknown name is ignored without an error, so a misspelt argument silently falls back to its default (for example `folder` instead of `path` lists all of `/Game`).

Packages are named by their Content path, for example `/Game/Models/Live/resnet-50`.

### Sample textures

The plugin ships six textures you can use with the image tools. They are in `/ModelPipelines/Samples/Textures/`:

- `T_classify_jellyfish`, `T_classify_elephant`, `T_classify_soccer_ball`, `T_classify_violin` (for image classification)
- `T_detect_dog_walk`, `T_detect_street_market` (for object detection)

Pass the full path, for example `/ModelPipelines/Samples/Textures/T_classify_jellyfish`, in `image`. An agent can list them with `find_assets` (toolset `editor_toolset.toolsets.asset.AssetTools`) and `folder_path` set to `/ModelPipelines/Samples/Textures`.

A texture that was just imported, or that is still loading after the editor started, may still be compiling. The image tools wait for it for up to 60 seconds. If it is still not ready they fail with `texture '<name>' is still compiling after 60 s`. Call again after a moment.

### list_model_packages

Lists the model packages under a Content path.

| Parameter | Default | What it does |
|---|---|---|
| `path` | `/Game` | Content path to search. It must start with `/`. |
| `recursive` | `true` | Also search sub folders. |

Example result:

```json
{
  "count": 2,
  "packages": [
    {
      "path": "/Game/Models/Live/resnet-50.resnet-50",
      "task": "ImageClassification",
      "labels": 1000
    },
    {
      "path": "/Game/Models/Live/rtdetr_r18vd.rtdetr_r18vd",
      "task": "ObjectDetection",
      "labels": 80
    }
  ]
}
```

Each entry also has `preferred_device`, `model`, `has_tokenizer` and `valid_settings`. They are left out of the example.

`valid_settings` only checks the package settings. It does not run the model. An empty project gives `{"count":0,"packages":[]}`.

### import_model_folder

Imports a Hugging Face model folder (the one with `model.onnx`) as a model package. This is the only tool that changes your project. See [Get and import models](ImportModel.md) for what a folder must contain.

| Parameter | Default | What it does |
|---|---|---|
| `source_folder` | none (required) | The model folder on disk. |
| `destination_path` | `/Game/Models` | Content path for the new assets. |
| `asset_name` | empty | Name of the package. Empty means the folder name. The model data asset is `<name>_Model`. |
| `overwrite_existing` | `false` | Update assets that already exist. |
| `confirm` | `false` | Must be `true` to change anything. |

#### The confirm rule

1. The first call, with `confirm=false`, is a dry run. It reads the folder and tells you what it would create. Nothing is created or changed.
2. Check the answer. If it holds errors, fix the folder first.
3. Call again with the same arguments and `confirm=true`. Now the import runs, the assets are saved and the model is validated.
4. If the assets already exist, the dry run says so. Pass `overwrite_existing=true` to update them in place, or choose another `asset_name`.

The tool never deletes anything. A failed import leaves nothing behind.

Dry run result:

```json
{
  "success": false,
  "needs_confirm": true,
  "would_create": {
    "package": "/Game/Models/Live/resnet-50.resnet-50",
    "model": "/Game/Models/Live/resnet-50_Model.resnet-50_Model"
  },
  "exists": false,
  "task": "ImageClassification"
}
```

The real answer also holds a `report` and a `message` that says nothing was changed.

Result after `confirm=true`:

```json
{
  "success": true,
  "package": "/Game/Models/all-MiniLM-L6-v2.all-MiniLM-L6-v2",
  "model": "/Game/Models/all-MiniLM-L6-v2_Model.all-MiniLM-L6-v2_Model",
  "task": "TextEmbedding",
  "report": {
    "messages": [],
    "device_checks": [
      { "device": "CPU", "available": true, "created": true, "ran": true, "inference_ms": 4.0 },
      { "device": "GPU", "available": true, "created": true, "ran": true, "inference_ms": 53.8 }
    ]
  }
}
```

Each device check also has an empty `message` when all is well. The GPU number is for the first run, which includes start-up cost.

The import blocks the editor while it runs, like the menu import. In a live run it took a few seconds for each model.

### validate_model_package

Checks a package and returns the report. It never changes the asset.

| Parameter | Default | What it does |
|---|---|---|
| `package` | none (required) | The package, for example `/Game/Models/Live/resnet-50`. |
| `run_smoke_test` | `true` | Run the model once to check it. |
| `check_gpu` | `true` | Also check the GPU. |

Example result (a real run, shortened):

```json
{
  "package": "/Game/Models/Live/resnet-50.resnet-50",
  "task": "ImageClassification",
  "valid": true,
  "errors": 0,
  "warnings": 0,
  "report": {
    "messages": [],
    "device_checks": [
      { "device": "CPU", "inference_ms": 43 },
      { "device": "GPU", "inference_ms": 84 }
    ]
  }
}
```

A message in `report.messages` has `severity` (`Info`, `Warning` or `Error`), `code` and `message`. `valid` is true when there are no errors. The editor is blocked for the short time the test runs take.

### benchmark_model_package

Runs a package many times and reports the timing for each device. It works in the background and the editor stays usable. The agent gets the answer when the run ends.

| Parameter | Default | What it does |
|---|---|---|
| `package` | none (required) | The package. |
| `runs` | `20` | Measured runs, 1 to 1000. One extra warm-up run comes first and is not counted. |
| `devices` | `both` | `cpu`, `gpu` or `both`. |
| `texts` | none (required, pass `[]` for image tasks) | Text for text models, up to 64 strings. One run processes all of them together. With `[]` on a text model, a built-in sentence is used. Image models ignore it. |
| `image` | empty | A texture asset for image models, for example `/ModelPipelines/Samples/Textures/T_classify_jellyfish` (see [Sample textures](#sample-textures)). With none, a flat gray picture of the model's input size is used. |

Always pass `texts`, even for an image model. The tool schema requires it. Use `[]` when you have no text.

The benchmark works on a temporary copy of the package. Your asset is not changed or saved.

Example result: resnet-50 on `T_classify_jellyfish`, 10 runs, `devices` set to `both` (a real run, shortened):

```json
{
  "package": "/Game/Models/Live/resnet-50.resnet-50",
  "task": "ImageClassification",
  "runs": 10,
  "devices": [
    {
      "device": "CPU",
      "inference_ms": { "median": 18.9 },
      "total_ms": { "median": 333.5 }
    },
    {
      "device": "GPU",
      "first_ms": 20.0,
      "inference_ms": { "median": 54.3 },
      "total_ms": { "median": 502.3 }
    }
  ]
}
```

Each device entry also has `available`, `actual_device`, `preprocess_ms`, `postprocess_ms` and `memory_mb`, and `inference_ms` and `total_ms` also have `p90`, `min` and `max`. They are left out here. A GPU that is not usable is `{"device":"GPU","available":false}`.

- All times are in milliseconds. `first_ms` is the warm-up run. The first call on a device loads the model, so it can be slower than the rest.
- `inference_ms` is the time the model itself takes. Use it to compare models and devices.
- `total_ms` is measured from the call to the moment the result arrives on the editor's game thread. It includes the wait for the next editor tick. See the note below.
- `p90` means 9 out of 10 runs were at least this fast.
- `actual_device` shows where the model really ran. If you asked for the GPU and got `CPU`, the plugin fell back.
- `memory_mb` is the memory use of the whole editor process. It is approximate. Do not read it as the size of the model.
- On a machine without a usable GPU, the GPU entry is `{"device":"GPU","available":false}`. That is not an error.
- The result does not name the GPU or the engine version. Write them down yourself when you keep the numbers.
- If one device fails (for example the texture is still compiling), the call still succeeds and that device's entry has an `error` text. Call again to get that device.

**Why `total_ms` looks large.** `total_ms` includes waiting for the editor to run its next tick. When the editor window is not in front, the editor slows down to a few frames a second, so `total_ms` can be hundreds of milliseconds even for a fast model. In the live run it was 333 to 500 ms while the model took 19 to 65 ms. For a realistic `total_ms`, bring the editor to the front, or untick **Edit > Editor Preferences > General > Performance > Use Less CPU when in Background**.

**GPU is not always faster.** In the same run, resnet-50 took a median of 18.9 ms on the CPU and 54.3 ms on the GPU. rtdetr_r18vd took 210.6 ms on the CPU and 64.9 ms on the GPU. Small models can run better on the CPU. Benchmark on the hardware you will ship on. These numbers are from one machine on UE 5.8.3 with 10 runs. They are not a promise for yours.

### run_pipeline_sample

Runs the package once with the package's own device setting and returns what the model found. It works in the background.

| Parameter | Default | What it does |
|---|---|---|
| `package` | none (required) | The package. |
| `texts` | none (required, pass `[]` for image tasks) | Text models: 1 to 64 strings. Image models ignore it. |
| `image` | empty | Image models: a texture asset, for example `/ModelPipelines/Samples/Textures/T_detect_dog_walk` (see [Sample textures](#sample-textures)). Text models ignore it. |
| `top_k` | `5` | How many top labels to return (classification). |
| `score_threshold` | `0.5` | Detection only: the lowest score to keep, from 0 to 1. |

Always pass `texts`, even for an image model. The tool schema requires it. Use `[]` when you have no text.

What you give depends on the task. Text embedding and text classification need `texts`. Image classification and object detection need `image`. A missing input gives an error that names it. If the texture is still compiling, the tool waits for it, as described in [Sample textures](#sample-textures).

The first call after the editor starts is slow. It loads the model and warms up the device. In a live run, the first call on rtdetr_r18vd had a preprocess time of 340 ms and an inference time of 1049 ms. Later calls are faster. The `timings.device` shows the device that ran: a package set to `Auto` reported `GPU`.

Example for image classification, resnet-50 on `T_classify_jellyfish` with `top_k` 3 (a real run, shortened):

```json
{
  "task": "ImageClassification",
  "top": [
    { "label": "jellyfish", "index": 107, "score": 1 },
    { "label": "tiger shark, Galeocerdo cuvieri", "index": 3, "score": 1.2e-10 },
    { "label": "spider web, spider's web", "index": 815, "score": 9.6e-11 }
  ]
}
```

The real answer also has `timings` (device, queue, preprocess, inference and postprocess times in ms).

Example for object detection, rtdetr_r18vd on `T_detect_dog_walk` with the default threshold 0.5. The real run found 9 objects: 4 dogs, 3 cars and 2 people. Shortened to the first three, best score first:

```json
{
  "task": "ObjectDetection",
  "detections": [
    { "label": "person", "score": 0.929 },
    { "label": "dog", "score": 0.914 },
    { "label": "dog", "score": 0.871 }
  ]
}
```

Each detection also has `index` and `box` as `[x0, y0, x1, y1]`, normalized from 0 to 1.

Example for text embedding with all-MiniLM-L6-v2 (a real run, shortened to the similarity of two sentences):

```json
{
  "task": "TextEmbedding",
  "embeddings": [
    { "text": "How do I craft a sword?" },
    { "text": "Where is the blacksmith?" }
  ],
  "similarity": [ [1.0, 0.407], [0.407, 1.0] ]
}
```

`similarity` is a cosine table. It appears when you give two or more texts. A value near 1 means the sentences are close in meaning. In the live run, an unrelated weather sentence scored about -0.05 and 0.06 against these two. Each embedding also has `dim`, `norm` and `head` (the first 8 numbers).

Text classification returns `results` with a `top` list of `label`, `index` and `score` for each text.

## Example: import and try a model

This is what you can ask an agent. The agent turns each request into tool calls. Say your download folder is `C:\PipelineModels` (see [Get and import models](ImportModel.md), step 1).

1. You: "Import the model in `C:\PipelineModels\all-MiniLM-L6-v2` into my project."
2. The agent calls `import_model_folder` with `source_folder` set to that folder and no `confirm`. It reports that it would create `/Game/Models/all-MiniLM-L6-v2` for the task `TextEmbedding`, and that nothing was changed.
3. You: "Yes, go ahead."
4. The agent calls `import_model_folder` again with `confirm=true`. You should see `"success": true` and two new assets in `/Game/Models`.
5. You: "Validate it." The agent calls `validate_model_package` with `package` set to `/Game/Models/all-MiniLM-L6-v2`. You should see `"valid": true`.
6. You: "How fast is it on my machine?" The agent calls `benchmark_model_package` with the same package and `texts` set to `[]`. You should see a CPU entry, and a GPU entry that has timings or `"available": false`.
7. You: "Compare 'A cat sits on the mat.' and 'A kitten rests on the rug.'" The agent calls `run_pipeline_sample` with those two strings in `texts`. You should see two embeddings and a `similarity` table with a number near 1 off the diagonal.

## Troubleshooting

**The client connects but the Model Pipelines tools are not listed.**

1. Open **Edit > Plugins** and check that **Unreal MCP**, **Toolset Registry** and **Python Editor Script Plugin** are enabled. Restart the editor after you change them.
2. Open the Output Log and search for `Model Pipelines MCP tools not registered`. The text after the colon says what is missing.
3. Check that the engine is 5.8. On 5.7 the tools do not exist.
4. Restart the client after the editor is back.

**The client cannot connect.**

1. Check that the editor is open and the server was started (step 2). It does not start by itself.
2. Check the port and path against **Edit > Editor Preferences > General > Model Context Protocol**.
3. Check that nothing else uses port 8000.

**`import_model_folder` returns `needs_confirm`.** This is normal for a first call. Read the answer, then call again with `confirm=true`. If `needs_confirm` is false, the `message` says why: errors in the folder, or assets that already exist.

**The benchmark memory numbers look odd.** They are for the whole editor process, not the model, so other work in the editor moves them. Use them as a rough guide.

**The GPU entry says `available: false`.** The machine has no GPU the plugin can use. The CPU entry still works.

**A benchmark or sample call returns an error.** The message names the cause, for example the package was not found, `runs` was outside 1 to 1000, `devices` was not `cpu`, `gpu` or `both`, or the input did not match the task. See also [Troubleshooting](Troubleshooting.md).

Back to [Documentation](README.md).
