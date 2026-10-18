# Troubleshooting

## A sample says "Model package not found"

The map cannot find its package. Fix:

1. Run `python Plugins/ModelPipelines/Tools/fetch_reference_models.py --only <slug> --out <folder>` (see [Get and import models](ImportModel.md)).
2. Import the folder into the `Content/Models` folder, keeping the folder name.
3. Press Play again.

The path must be exactly `/Game/Models/<slug>`.

## The import fails

You see **Model folder import failed** and a report. Nothing is created. Common reports:

| Report | Cause and fix |
|---|---|
| Missing model file | The folder has no `model.onnx` (or `onnx/model.onnx`), or it is empty. Pick the folder that contains it. |
| Missing tokenizer | A text model needs `tokenizer.json`. Export with `optimum-cli export onnx`, which writes it. |
| ONNX external data | The model is split into `model.onnx` plus `.onnx_data` files. Only one-file models work. Models over 2 GB are not supported. |
| Task not detected / unsupported task | `config.json` has no known architecture. Add a `modelpipelines.json` with `"task"` set to `TextEmbedding`, `TextClassification`, `ImageClassification` or `ObjectDetection`. |
| Missing labels | Classification and detection need `id2label` in `config.json`, numbered 0 to N-1 with no gaps. |
| Asset already exists | A package with that name is already in the folder. Delete it first, or import into another folder. |
| Tokenizer invalid / smoke run failed | The model or tokenizer is broken or not what `config.json` says. Re-export the model. |

## A warning about `hf_xet` while downloading models

The download script may print a warning about `hf_xet`. It is harmless. The download still works.

## Detection boxes don't line up with the objects

Check these causes:

- The model only sees what the capture sees. Anything outside the capture's view, or hidden by other actors, gets no box.
- A **ScoreThreshold** that is too low shows weak boxes on things that are not there. Raise it, for example to 0.5.
- Very small objects far away are missed. Fly closer.

## Wrong task

`Result.Error` is `WrongTask`. The package does a different task than the pipeline node. Use the create node that matches the package (see [Nodes](Nodes.md)). A text embedding package does not work in **Create Text Classification Pipeline**.

## The GPU is not used

If `Result.Device` is **CPU**, the model ran on the CPU. This is normal when DirectML is not available on the machine. The plugin falls back to the CPU on its own, and the CPU always works. To force the CPU on purpose, set **PreferredDevice** on the package to **CPU**.

## The first call is slow

The first call on a pipeline loads the model. Create the pipeline once (for example in Begin Play) and reuse it. Later calls are fast.

## PreprocessFailed

For text, the tokenizer did not load or could not read your text. Check that the package has a `tokenizer.json` that fits the model. Re-import the folder if you replaced it. For images, the image could not be read.

## Results differ from Python

- Make sure you exported the same model revision. The sample models are pinned to exact revisions (see [Model licenses](ModelLicenses.md)).
- Small differences are normal. We test against Python with these limits: embeddings cosine similarity above 0.999, class scores within 0.001, detection box overlap (IoU) above 0.9.

## The render target is black, or vision fails in a server

Textures and render targets need a rendering RHI. With `-nullrhi` or on a dedicated server there is nothing to read. In C++, build an `FPipelineImage` from your own pixels and call `ClassifyImageAsync` or `DetectImageAsync`.

## I edited MaxTokens or the tokenizer

Re-validate the package with **Validate Model Package** (see [Get and import models](ImportModel.md#edit-a-package)). An unvalidated package can crash a packaged game.

## The packaged game cannot find the model

A package that is only reached by a path string (for example `/Game/Models/resnet-50` typed in a node) is not cooked by default. Do one of these:

- Add `/Game/Models` to **Project Settings > Packaging > Additional Asset Directories to Cook**.
- Or use a hard reference to the package, for example a variable of type **Pipeline Model Package** set in the editor.

## Speech to text

**Out of memory when creating the speech pipeline.** One pipeline needs about 4 GB of RAM. Create one and share it. Do not create one per call.

**The transcript is empty.** The clip is quieter than `SilenceThresholdDb` (-60 dBFS by default), so the model did not run (**Inference Ms** is 0). Raise the volume, or set `SilenceThresholdDb` to -100 to turn the gate off.

**The call fails with "at most 600 s per call".** Cut the audio into pieces of 600 s or less.

**Transcribe Sound Wave fails.** Procedural sound waves are not supported. Use an imported sound wave.

**The first GPU call takes about 4 s.** DirectML warms up once. Later calls are faster.

**`LicenseNotPermissive` warning on import.** Expected for Phonon-2. It means attribution is required. See [Model licenses](ModelLicenses.md#cc-by-40-models-such-as-phonon-2).

**The import reports the wrong task or missing files.** A speech folder needs `encoder.onnx`, `decoder_joint.onnx`, `tokenizer.json` and `modelpipelines.json`. See [Get and import models](ImportModel.md#speech-to-text-folders).

## Real-time speech

**Start Listening ends in On Failure with "no audio capture device".** The engine found no input device. Check that a microphone is plugged in and enabled in Windows, and that Windows lets desktop apps use it. `USpeechMicrophoneSource::GetCaptureDevices()` shows what the engine finds.

**On Partial never fires.** Partials need the Phonon engine from the paid ModelPipelines Generative plugin. The free ONNX engine gives On Final only. See [Real-time speech](SpeechToText.md#real-time-speech).

**Quiet room: speech is missed, or small sounds start utterances.** Speech means 12 dB over a running noise floor and above -55 dBFS. A very quiet microphone stays under -55 dBFS: raise its level in Windows. If breath or clicks count as speech, raise `SpeechThresholdDb` (up to 40). If real speech is missed, lower it (down to 3).

**Stop Listening freezes the game for up to 1 s.** The microphone was unplugged during capture, and the engine's capture backend waits for the device. With a working device it takes about 10 ms.

## Still stuck

Open an issue: https://github.com/NextGen-GameDev/ModelPipelines-UE/issues. Include the Output Log and the report text.
