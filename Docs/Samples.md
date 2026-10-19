# Samples

Five sample maps show the four tasks and speech to text. Each map needs one model package in `/Game/Models`.

| Map | Task | Package needed |
|---|---|---|
| `L_SemanticSearch` | Text embedding | `/Game/Models/all-MiniLM-L6-v2` |
| `L_Sentiment` | Text classification | `/Game/Models/distilbert-sst2` |
| `L_SceneClassifier` | Image classification | `/Game/Models/resnet-50` |
| `L_Detection` | Object detection | `/Game/Models/rtdetr_r18vd` |
| `L_TalkToText` | Speech to text | `/Game/Models/phonon-2-fp16w` |

The maps are in the plugin's content, in the folder `/ModelPipelines/Samples/Maps/`.

## Before you start

1. [Install the plugin](Install.md), including **Show Plugin Content**.
2. [Get and import the models](ImportModel.md) so the packages exist in `/Game/Models`, or run the batch import script from [Import a model](ImportModel.md#import-from-blueprint-or-python-batch-import).

## Open and play a map

1. Open the Content Browser.
2. Open the Content Browser settings and enable **Show Plugin Content** (if you have not yet).
3. In the folder tree, open the ModelPipelines plugin's content folder, then **Samples**, then **Maps**.
4. Double-click the map you want.
5. Click **Play** in the toolbar.

Each map has a PlayerStart, a sun, a sky light and one sample actor. The vision maps also have photo boards (below). Select the sample actor in the level to change its properties in the Details panel.

If the package is missing, the map shows this text on screen:

> Model package not found: `<path>`. Run Plugins/ModelPipelines/Tools/fetch_reference_models.py --only `<slug>`, import the folder into /Game/Models, then press Play again. See Docs/Samples.md.

Do what it says, then press Play again.

### Sample actor properties

All five actors share these properties (category **Model Pipelines|Sample**):

| Property | Default | What it does |
|---|---|---|
| **Package** | `/Game/Models/<slug>` for the sample's model | The model package to use. |
| **bShowOverlay** | on | Show the on-screen panel. |

The shared panel also shows a status line (loading, ready, busy, the last error, or the missing-package text).

## L_SemanticSearch

Search by meaning. The map has a list of NPC dialogue lines.

1. Press Play.
2. A panel shows a text box with the hint **Ask the villagers...**. Click it if it is not focused. The mouse cursor is on.
3. Type a question, for example `Where can I buy a sword?`.
4. Press **Enter**.
5. You should see your question and the 3 most similar NPC lines, each with its similarity score (for example `0.52   Blacksmith: Bring me your dented sword...`). A line about the blacksmith ranks above a line about fishing, even with no shared words.

Actor: `ModelPipelinesSemanticSearchSample`. Properties:

| Property | Default | What it does |
|---|---|---|
| **DialogueLines** | 24 NPC lines (blacksmith, innkeeper, guard, ...) | The searchable corpus. One line per entry. It is embedded once when the pipeline is ready. Edit the list to use your own lines. |
| **TopK** | 3 (minimum 1) | How many of the closest lines a search returns. |

Blueprint-callable on the actor: **Search** (Query), **Get Last Matches**, **Is Corpus Ready**.

Blueprint version of the core flow:

1. **Event Begin Play** > **Create Text Embedding Pipeline** (Package = the MiniLM package) > set variable `Pipeline`.
2. **Embed Texts** (Pipeline, Texts = `DialogueLines`).
3. On Success: set variable `LineEmbeddings` = `Embeddings`.
4. When the player commits text (for example the **On Text Committed** event of an Editable Text widget): **Embed Texts** (Pipeline, Texts = **Make Array** with `[Query]`).
5. On Success: **Top K Similar** (Query = `Embeddings` element 0, Corpus = `LineEmbeddings`, K = 3).
6. **For Each Loop** over the matches: **Print String** with `DialogueLines[Index]` and `Score`.

## L_Sentiment

Rate a chat message from negative to positive.

1. Press Play.
2. A panel shows a chat box with the hint **Say something in chat...**.
3. Type a chat message, for example `This game is great`.
4. Press **Enter**.
5. You should see the meter move from red (negative) to green (positive), the label **POSITIVE** or **NEGATIVE** with its score (for example `POSITIVE   (positive 0.99)`), and a list of your last 5 messages, each with its score and colored by sentiment.

Actor: `ModelPipelinesSentimentSample`. It has only the shared properties. Blueprint-callable on the actor: **Analyze** (Message), **Get Last Positive Score**, **Get Last Label**.

Blueprint version:

1. **Event Begin Play** > **Create Text Classification Pipeline** (Package = the distilbert package) > set variable `Pipeline`.
2. On text committed: **Classify Texts** (Pipeline, Texts = `[Message]`, Top K = 1).
3. On Success: take `Classifications` element 0. Read **Label** and **Score**.
4. Set the meter color from the label (green for POSITIVE, red for NEGATIVE). Add the message to a list of 5.

## L_SceneClassifier

The map classifies what you see. It has four photo boards in a row: a jellyfish, a soccer ball, a violin and an elephant. The PlayerStart faces them.

1. Press Play.
2. Fly with **W A S D** and the mouse.
3. A scene capture follows your camera. Once per second it is classified.
4. You should see a small preview of the captured image (256 px) and the top 3 labels with percentages (for example `jellyfish   97.3%`).
5. Look at each board. The top label should name what the photo shows (`jellyfish`, `soccer ball`, `violin`, an elephant label).
Actor: `ModelPipelinesSceneClassifierSample`. Properties:

| Property | Default | What it does |
|---|---|---|
| **CaptureResolution** | 512 (16 to 4096) | Width and height of the square capture render target. Set at Begin Play. |
| **IntervalSeconds** | 1.0 (minimum 0.05) | Seconds between two automatic classifications. |
| **TopK** | 3 (minimum 1) | How many labels a classification returns. |
| **bFollowPlayerCamera** | on | Copy the player camera's location, rotation and horizontal field of view every tick. Without a player controller the actor's own transform is used. |

The actor also has a **SceneCapture** component (read only). Blueprint-callable: **Classify Now**, **Get Last Classification**, **Get Render Target**.

Blueprint version:

1. **Event Begin Play** > **Create Image Classification Pipeline** (Package = the resnet-50 package) > set variable `Pipeline`.
2. **Set Timer by Event** (looping, 1 second).
3. In the timer event: **Classify Image** (Pipeline, Image = the **Texture Target** of a **Scene Capture Component 2D**, Top K = 3).
4. On Success: loop over `Classification` **Top K**, print **Label** and **Score** (as a percentage).

Make the capture target an RGBA8 render target, and keep the capture component attached to the camera.

## L_Detection

The same camera idea. Boxes are drawn over the view. The map has two photo boards: a woman walking four dogs, and a street market.

1. Press Play.
2. Fly around with **W A S D** and the mouse.
3. Twice per second the view is analyzed. You should see boxes with a label and a score drawn over the objects, for example `person 0.87` or `dog 0.91`.
4. Look at the boards. The dog-walk photo gives people and dogs. The market photo gives people and bicycles.

The boxes line up with the view because the capture copies your real camera projection.

Actor: `ModelPipelinesDetectionSample`. Properties:

| Property | Default | What it does |
|---|---|---|
| **CaptureWidth** | 960 (16 to 4096) | Capture width. The height follows the viewport aspect. Set at Begin Play. |
| **ScoreThreshold** | 0.5 (0 to 1) | Detections below this score are dropped. |
| **IntervalSeconds** | 0.5 (minimum 0.05) | Seconds between two automatic detections. |
| **bFollowPlayerCamera** | on | Copy the player camera's location, rotation and horizontal field of view every tick. |

The actor also has a read-only **SceneCapture** component. Blueprint-callable: **Detect Now**, **Get Last Detections**, **Get Render Target**.

Blueprint version:

1. **Event Begin Play** > **Create Object Detection Pipeline** (Package = the rtdetr_r18vd package) > set variable `Pipeline`.
2. **Set Timer by Event** (looping, 0.5 seconds).
3. In the timer event: **Detect Objects** (Pipeline, Image = the capture's render target, Score Threshold = 0.5).
4. On Success: store `Detections`.
5. In your HUD **Event Receive Draw HUD**: **For Each Loop** over `Detections`. For each, draw a rectangle from **Box Min** to **Box Max**, each multiplied by the canvas size, and the **Label** and **Score**.

If the boxes do not line up with the objects, see [Troubleshooting](Troubleshooting.md#detection-boxes-dont-line-up-with-the-objects).

## L_TalkToText

Press Play, talk, see the text. This map needs the speech model (about 1.3 GB). You get it in two steps.

Get the model first:

1. Run `Tools/fetch_phonon2.py --out <folder>` (see [Speech to text, step 1](SpeechToText.md#step-1-get-and-convert-the-model)). You should see a `phonon-2-fp16w` folder in `<folder>`.
2. In the Content Browser, right-click and choose **Import Hugging Face Model Folder...**. Pick the `phonon-2-fp16w` folder and import it into `/Game/Models` (see [Speech to text, step 2](SpeechToText.md#step-2-import-the-folder)). You should see the package `/Game/Models/phonon-2-fp16w`.

The model is Phonon-2 by Fermion Research, license CC-BY-4.0. If you ship it, you must credit it. See [Model licenses](ModelLicenses.md).

Then play the map:

1. Open `L_TalkToText` from the plugin content (see "Open and play a map" above).
2. Click **Play**.
3. Speak into your microphone.
4. You should see a panel titled **Talk to Text** with:
   - An engine line. With the free ONNX engine it reads `Engine: ONNX (final text only; partial text needs ModelPipelines Generative)`. With the Phonon engine it reads `Engine: Phonon (partial text while you speak)`.
   - A status line: first `Warming up the model...` for a few seconds (one short test transcription, so your first sentence is not slow), then `Listening (<microphone name>)`, then `Hearing speech...` while you talk, then `Transcribing...` until the text arrives.
   - The last 8 final lines. Each has its delay, for example `[0.31 s]`. The delay is the time from the end of your speech to the final text.
   - The partial text in grey while you speak. Only the Phonon engine gives it.
   - A **Stop listening** button. Click it to stop. It then reads **Start listening**. The mouse cursor is on.

Measured delay (end of speech to final text, Intel i7-11800H laptop CPU): about 0.5 to 0.7 s with the ONNX engine, about 0.3 s with the Phonon engine. The full table is in [Speech to text, Latency](SpeechToText.md#latency).

No microphone: the status says `No microphone (...)`. Audio you push with **Get Stream** then **Push Speech Audio** is still transcribed.

Phonon engine (the paid ModelPipelines Generative plugin): import the Phonon-2 weights folder as described in that plugin's docs. Then select the sample actor and set **Package** to that package.

If the package is missing, the map shows:

> Model package not found: `<path>`. Run Plugins/ModelPipelines/Tools/fetch_phonon2.py --out `<folder>`, import the phonon-2-fp16w folder into /Game/Models, then press Play again. With ModelPipelines Generative you can point Package at an imported Phonon-2 weights package instead. See Docs/Samples.md.

Actor: `ModelPipelinesTalkToTextSample`. Its **Package** defaults to `/Game/Models/phonon-2-fp16w`. Properties (category **Model Pipelines|Sample**):

| Property | Default | What it does |
|---|---|---|
| **bUseMicrophone** | on | Listen to the microphone. Turn it off to push your own audio. |
| **DeviceIndex** | -1 | The input device. -1 is the default device. |
| **bAutoStart** | on | Start listening when the pipeline is ready. |
| **StreamSettings** | defaults | The real-time settings. See [Speech to text, Settings](SpeechToText.md#settings). |
| **MaxLines** | 8 | How many final lines the panel keeps. |

Blueprint-callable on the actor: **Start Listening**, **Stop Listening**, **Is Listening**, **Is Microphone Active**, **Has Partial Text**, **Get Final Lines**, **Get Partial Text**, **Get Last Final Latency Seconds**, **Get Stream**. The event **On Final Text** fires for each final line.

Blueprint version of the core flow:

1. **Event Begin Play** > **Create Speech To Text Pipeline** (Package = the phonon-2-fp16w package) > set variable `Pipeline`.
2. The pipeline's **Start Listening** node (not the sample actor's function of the same name) (Pipeline = `Pipeline`, Use Microphone ticked, Device Index = -1). Store the returned action in a variable.
3. On **On Final**: append **Transcript** and a new line to the old text, then **Set Text** on a Text widget.
4. To stop, call the **Stop Listening** node with the stored action.

See [Nodes](Nodes.md#start-listening) for every pin.

## Performance

The model runs in the background, not on the game thread. In our tests a vision call costs about 0.2 ms of game-thread time, because reading the texture back does not block. The first call on a pipeline also loads the model, so it is slower.

## Photos

The photos on the boards are public domain or CC0. Their authors and licenses are listed in [Model licenses](ModelLicenses.md#sample-photos).

Next: [Nodes](Nodes.md) for every pin, or [Troubleshooting](Troubleshooting.md).
