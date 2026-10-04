# Blueprint nodes and C++ calls

## The flow

Every task works the same way:

1. Get a model package (an asset reference, see [Import a model](ImportModel.md)).
2. Create a pipeline from the package. Do this **once**, for example in Begin Play. Store the pipeline in a variable.
3. Call a task node as often as you like. The node runs the model in the background and fires an output pin later.

### Create nodes

Find these in the Blueprint action menu.

| Node | Category | Pipeline it returns |
|---|---|---|
| **Create Text Embedding Pipeline** | Model Pipelines > Text | Text Embedding Pipeline |
| **Create Text Classification Pipeline** | Model Pipelines > Text | Text Classification Pipeline |
| **Create Image Classification Pipeline** | Model Pipelines > Vision | Image Classification Pipeline |
| **Create Object Detection Pipeline** | Model Pipelines > Vision | Object Detection Pipeline |

Each one has an input **Package** and two outputs: the pipeline and **Result** (see below). These nodes have an exec pin (they are not pure), so wire the white exec line through them.

They return **None** when something is wrong. Check **Result**:

- `WrongTask`: the package does a different task than the node. Use the node that matches the package.
- `InvalidArgument`: the Package input is empty.

Always store the pipeline in a variable. If you only keep it on a wire it can be garbage collected.

### Task nodes

Each task node has one input exec pin and two output exec pins. **On Success** and **On Failure** carry the same data pins.

#### Embed Texts

Turns sentences into embeddings.

| | |
|---|---|
| Inputs | **Pipeline** (Text Embedding Pipeline), **Texts** (array of String) |
| Outputs | **Result**, **Embeddings** (array of Pipeline Embedding, one per text, same order) |

Each embedding has `Values`, an array of floats. The vector is scaled to length 1 (L2-normalized) when the package says so.

Example: Begin Play, **Create Text Embedding Pipeline**, set variable `Pipeline`. Then **Embed Texts** with `Pipeline` and a **Make Array** of your lines. On Success, store `Embeddings` in a variable `LineEmbeddings`.

#### Classify Texts

Gives each sentence a label and a score.

| | |
|---|---|
| Inputs | **Pipeline** (Text Classification Pipeline), **Texts**, **Top K** (default 1) |
| Outputs | **Result**, **Classifications** (array of Pipeline Classification, one per text) |

Example: On Success, **Get** element 0 of `Classifications`, then read **Label** (for example `POSITIVE`) and **Score**.

#### Classify Image

Gives the best labels for one picture.

| | |
|---|---|
| Inputs | **Pipeline** (Image Classification Pipeline), **Image** (Texture 2D or Texture Render Target 2D), **Top K** (default 5) |
| Outputs | **Result**, **Classification** (one Pipeline Classification) |

Example: pass the Render Target of a Scene Capture Component. On Success, loop over **Top K** of `Classification` and print each **Label** and **Score**.

#### Detect Objects

Finds objects and where they are.

| | |
|---|---|
| Inputs | **Pipeline** (Object Detection Pipeline), **Image** (Texture 2D or Texture Render Target 2D), **Score Threshold** (default 0.5) |
| Outputs | **Result**, **Detections** (array of Pipeline Detection) |

Only detections with a score above **Score Threshold** are returned. The value is clamped to 0 to 1.

A NaN threshold, an unsupported image or an empty pipeline ends in **On Failure**.

Example: On Success, **For Each Loop** over `Detections`. Use each **Box** to draw a rectangle on a HUD.

#### World context

Each task node has a hidden **World Context Object** input. In an Actor Blueprint it fills in on its own. In other Blueprints, pass an actor or any world object, or leave it empty.

#### Cancel a node

Each task node returns an async action object. Store it, and call **Cancel** on it to stop waiting. After **Cancel**, no output pin fires. The running call finishes in the background and its result is dropped.

#### Failure cases

For every task node, a null or not initialized pipeline ends in **On Failure** with the reason in **Result**. **Embed Texts** and **Classify Texts** also fail on an empty text list. **Classify Texts** and **Classify Image** keep the **Top K** best labels: below 1 gives 1, and above the number of labels gives all labels.

### When the output pins fire

- Exactly one of **On Success** or **On Failure** fires, and only once per call.
- It fires on the game thread, later. It never fires in the same execution path as the node. The top exec pin continues first.
- The work runs in the background. The game keeps running. In our tests a vision call costs about 0.2 ms of game-thread time, because reading the texture back does not block.
- The node keeps the pipeline and the image alive until it finishes.
- You can run many nodes at once. In a loop, each pass is its own call.
- Each node is a cancellable async action. Calling **Cancel** on the object it returns stops it. After a cancel, no output pin fires.

### Image rules

- **Texture 2D**: read back from the GPU without stalling the game. CPU access is not needed.
- **Render Target 2D**: format RGBA8 or RGBA16f.
- The first call on a new pipeline also loads the model, so it is slower. Later calls are fast.
- **Preferred Device** on the package: **Auto** tries the GPU (DirectML through NNERuntimeORT) and falls back to CPU. **CPU** always works.

## Result (Pipeline Result)

Every create node and every task node returns one.

| Field | Meaning |
|---|---|
| `bSuccess` | True when the call worked. |
| `Error` | A **Pipeline Error** value. `None` on success. |
| `Message` | A readable explanation. |
| `Device` | The device that ran the model: CPU or GPU. Auto means nothing ran. |
| `QueueMs` | Time spent waiting for the model. |
| `PreprocessMs` | Time to prepare the input. |
| `InferenceMs` | Time the model ran. |
| `PostprocessMs` | Time to read the output. |

### Error values

| Value | Meaning |
|---|---|
| `None` | No error. |
| `InvalidArgument` | A bad or empty input (no package, empty tensor, no callback). |
| `UnsupportedType` | The model uses an input or output type other than float32, int64 or int32. |
| `ShapeMismatch` | The input count, type or shape does not match the model. |
| `RuntimeUnavailable` | No NNE runtime is available for any allowed device. |
| `ModelLoadFailed` | A runtime refused the model, or it could not be created on any allowed device. |
| `InferenceFailed` | The model failed while running. |
| `Cancelled` | Cancelled while queued, or the engine shut down. |
| `WrongTask` | The package task does not match the pipeline. |
| `Internal` | An unexpected internal problem. |
| `PreprocessFailed` | The tokenizer or the image preparation failed. |

## Result types

| Type | Fields |
|---|---|
| **Pipeline Embedding** | `Values` (float array). |
| **Pipeline Classification** | `Label`, `Index`, `Score` (the best class), `TopK` (array, best first, each with `Label`, `Index`, `Score`), `Scores` (the score of every class, in class order). |
| **Pipeline Detection** | `Label`, `Index`, `Score`, `Box`. |
| **Pipeline Similarity Match** | `Index` (position in the corpus array), `Score` (cosine similarity). |

`Box` is a 2D box in image coordinates from 0 to 1. **Min** is (left, top). **Max** is (right, bottom). To draw on a 1920 x 1080 view, multiply x by 1920 and y by 1080.

## Similarity helpers

Category **Model Pipelines > Similarity**. These are pure nodes, they run at once.

| Node | Inputs | Output |
|---|---|---|
| **Cosine Similarity** | **A**, **B** (Pipeline Embedding) | A float from -1 to 1. Different lengths or an empty embedding gives 0 and a warning. |
| **Top K Similar** | **Query** (Pipeline Embedding), **Corpus** (array of Pipeline Embedding), **K** | Array of Pipeline Similarity Match, best first. Equal scores: the lower index first. |

## Pipeline queries

Category **Model Pipelines**. Call them on a pipeline variable.

| Node | Output |
|---|---|
| **Is Ready** | True when the pipeline was created from a package. |
| **Get Task** | The task this pipeline does. |
| **Get Package** | The package it uses. |
| **Get Device** | Auto until the model is loaded, then CPU or GPU. |
| **Get Num Pending Calls** | How many calls have not finished yet. |

## C++

The same calls exist in C++. The callback runs on the game thread, once, never inside the call. Keep the pipeline in a `UPROPERTY`.

```cpp
#include "TextPipelineLibrary.h"
#include "TextEmbeddingPipeline.h"
#include "PipelineModelPackage.h"
#include "PipelineLibrary.h"

// In a UObject or AActor with:
//   UPROPERTY() TObjectPtr<UTextEmbeddingPipeline> Pipeline;
//   UPROPERTY() TArray<FPipelineEmbedding> LineEmbeddings;

void AMyActor::Setup(UPipelineModelPackage* Package, const TArray<FString>& Lines)
{
	FPipelineResult Result;
	Pipeline = UTextPipelineLibrary::CreateTextEmbeddingPipeline(Package, Result);
	if (!Pipeline)
	{
		UE_LOG(LogTemp, Warning, TEXT("%s"), *Result.Message);
		return;
	}

	TWeakObjectPtr<AMyActor> Weak(this);
	Pipeline->EmbedAsync(Lines, [Weak](const FPipelineResult& Res, const TArray<FPipelineEmbedding>& Embeddings)
	{
		if (AMyActor* Self = Weak.Get(); Self && Res.bSuccess)
		{
			Self->LineEmbeddings = Embeddings;
		}
	});
}
```

The other pipelines:

| Pipeline class | Call |
|---|---|
| `UTextClassificationPipeline` | `ClassifyAsync(const TArray<FString>& Texts, int32 TopK, Callback)`; callback gets `(const FPipelineResult&, const TArray<FPipelineClassification>&)` |
| `UImageClassificationPipeline` | `ClassifyAsync(UTexture* Image, int32 TopK, Callback)`; callback gets `(const FPipelineResult&, const FPipelineClassification&)` |
| `UObjectDetectionPipeline` | `DetectAsync(UTexture* Image, float ScoreThreshold, Callback)`; callback gets `(const FPipelineResult&, const TArray<FPipelineDetection>&)` |

Create functions: `UTextPipelineLibrary::CreateTextEmbeddingPipeline`, `CreateTextClassificationPipeline`, `UVisionPipelineLibrary::CreateImageClassificationPipeline`, `CreateObjectDetectionPipeline` (all take `(UPipelineModelPackage*, FPipelineResult&)`).

For images that are not textures (for example from a server with no renderer), build an `FPipelineImage` and call `ClassifyImageAsync` or `DetectImageAsync`.
