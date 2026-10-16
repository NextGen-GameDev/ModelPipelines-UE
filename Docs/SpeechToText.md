# Speech to text

Speech to text turns a recorded voice clip into text with a time for every word. The plugin runs Phonon-2, an English model, on your machine. No audio leaves the computer.

The model is **not** bundled. You download it, convert it to ONNX with the scripts in `Tools/`, and import the folder.

## What you need to know first

- Output is English, with capital letters and punctuation.
- One pipeline uses about 4 GB of RAM. Create one pipeline and share it.
- One call takes up to 600 seconds of audio.
- Microphone and streaming input are not supported yet.
- The model weights are licensed CC-BY-4.0. If you ship them, you must credit the authors. See [Attribution](#attribution-cc-by-40).

## Step 1: Get and convert the model

Phonon-2 is by Fermion Research: https://huggingface.co/FermionResearch/Phonon-2

One script downloads it, checks it and converts it to ONNX. It needs Python 3.11 with the packages in
`Tools/requirements-stt.txt` (transformers 5.20, so use a separate environment from the sample-model tools). From the
`Tools` folder:

```
python fetch_phonon2.py --out D:/Models
```

What it does:

1. Downloads Phonon-2 at a pinned revision (177 MB) and checks every file against the repository's `SHA256SUMS.txt`.
2. Downloads only the config and tokenizer files of the base model `nvidia/parakeet-tdt-0.6b-v3` (no weights).
3. Unpacks the model and exports it to ONNX, then builds the shipping variant `fp16w`.
4. Writes `D:/Models/phonon-2-fp16w/` with `encoder.onnx` (1.22 GB), `decoder_joint.onnx` (70 MB), `tokenizer.json`,
   the config files, `modelpipelines.json` (task and CC-BY-4.0 attribution), `NOTICE` and the licence text.

It needs about 4 GB of free disk while it runs (the temporary full-precision export is deleted at the end) and about
6.5 GB of RAM. Running it again with an up-to-date output folder does nothing. Other variants: `--variant fp16`
(16-bit compute, 1.22 GB) or `--variant int8` (0.84 GB, slightly less accurate: 2.15 % WER but 3 of 40 test clips differ
from the full-precision text).

If your Python environment has trouble loading numpy's maths libraries on Windows, run the script through your
environment manager (for example `micromamba run -n <env> python fetch_phonon2.py ...`) instead of calling `python.exe`
directly.

## Step 2: Import the folder

1. In the Content Browser, right-click and choose **Import Hugging Face Model Folder...**.
2. Pick the `fp16w` folder.
3. You should see a package asset. Open it. **Task** is `SpeechToText` and **License** shows `CC-BY-4.0`.

See [Get and import models](ImportModel.md#speech-to-text-folders) for what the importer looks for. The validator reports `LicenseNotPermissive` as a Warning. This is expected. It means "attribution required", not "do not use".

## Step 3: Transcribe from C++

```cpp
#include "SpeechPipelineLibrary.h"
#include "SpeechToTextPipeline.h"
#include "PipelineModelPackage.h"

// Once, for example in BeginPlay. Keep the pipeline in a UPROPERTY.
FPipelineResult CreateResult;
SpeechPipeline = USpeechPipelineLibrary::CreateSpeechToTextPipeline(Package, CreateResult);
if (!SpeechPipeline)
{
	UE_LOG(LogTemp, Warning, TEXT("Create failed: %s"), *CreateResult.Message);
	return;
}

// Later, on the game thread. Samples are interleaved floats.
SpeechPipeline->Transcribe(MoveTemp(Samples), SampleRate, NumChannels,
	[](const FPipelineResult& Result, const FSpeechTranscript& Transcript)
	{
		if (!Result.bSuccess) { return; }
		UE_LOG(LogTemp, Log, TEXT("%s"), *Transcript.Text);
		for (const FSpeechWord& W : Transcript.Words)
		{
			UE_LOG(LogTemp, Log, TEXT("%s  %.2f - %.2f"), *W.Word, W.StartSeconds, W.EndSeconds);
		}
	});
```

The callback (`FOnSpeechTranscribed`) fires once on the game thread, never inside `Transcribe`. `FSpeechTranscript` has `Text`, `Words` (each `FSpeechWord` has `Word`, `StartSeconds`, `EndSeconds`), `AudioSeconds` and `InferenceMs`. `InferenceMs` is 0 when the silence gate answered without running the model.

For a sound wave asset, call `TranscribeSoundWave(USoundWave* Sound, FOnSpeechTranscribed OnDone)` the same way.

A null callback logs a Warning and nothing runs.

## Step 4: Transcribe from Blueprints

1. Get the package asset. Call **Create Speech To Text Pipeline** (category **Model Pipelines > Speech**) once. Store the pipeline in a variable.
2. Call one of the two latent nodes:
   - **Transcribe Speech**: inputs **Pipeline**, **Samples** (array of floats), **Sample Rate** (default 16000) and **Num Channels** (default 1).
   - **Transcribe Sound Wave**: inputs **Pipeline** and **Sound** (a Sound Wave asset).
3. Use the two output exec pins. **On Success** and **On Failure** both carry **Result** and **Transcript**. Exactly one fires, once, on the game thread. On failure the transcript is empty and **Result** says why.
4. Break **Transcript** to read **Text**, **Words**, **Audio Seconds** and **Inference Ms**.

Store the node's return value and call **Cancel** to stop waiting. After **Cancel** no pin fires. More in [Nodes](Nodes.md#transcribe-speech).

## Long audio

A call takes up to 600 seconds. Audio longer than 35 seconds is split at quiet points into windows of 25 to 35 seconds. Each window is transcribed separately and the words are joined, with times counted from the start of the clip. Audio over 600 seconds ends in a failure. Cut it into pieces yourself.

## Settings

The importer fills these from the folder. They are the **Speech** section of the package asset (`FPipelineSpeechSettings`). Phonon-2 needs the defaults. Change them only for another model.

| Setting | Default | Meaning |
|---|---|---|
| `SampleRate` | 16000 | Sample rate (Hz) the model expects. Input is resampled to it. |
| `NumMelBins` | 128 | Mel filterbank bins (the encoder's feature size). |
| `FftSize` | 512 | FFT size in samples. |
| `WindowLength` | 400 | Analysis window in samples (at most `FftSize`). |
| `HopLength` | 160 | Hop between frames in samples. |
| `Preemphasis` | 0.97 | Pre-emphasis coefficient in [0, 1). 0 turns it off. |
| `SubsamplingFactor` | 8 | Feature frames per encoder frame. |
| `BlankTokenId` | 8192 | Token id of the blank symbol. |
| `Durations` | 0, 1, 2, 3, 4 | Frame advances the duration head can predict. |
| `MaxSymbolsPerStep` | 10 | Most symbols per encoder frame before the decoder must advance. |
| `SilenceThresholdDb` | -60 | Clips with RMS below this (dBFS) return an empty transcript without inference. -100 turns the gate off. |

The silence gate matters: on silence the model otherwise writes "Yeah.".

## Limits and measurements

| Item | Value |
|---|---|
| Language | English |
| Input | Any sample rate from 1000 to 384000 Hz, any number of channels. Mixed to mono, resampled to 16 kHz. |
| Length | Up to 600 s per call. Windows of 25 to 35 s above 35 s. |
| Memory | About 4 GB of RAM per pipeline. ONNX Runtime expands the 16-bit weights to 32-bit when it loads them. |
| Word times | Start and end per word, 80 ms resolution. |
| Sound waves | Any imported `USoundWave` (PCM, ADPCM, Bink, Opus; streaming or not), decoded off the game thread. Procedural waves (`USoundWaveProcedural`) are not supported. |
| Devices | The decoder always runs on the CPU. The encoder follows **Preferred Device**. |
| Platform | Windows 64-bit. Tested on UE 5.8; the UE 5.7 build is checked at the end of the milestone. |

Accuracy: 2.16 % word error rate on LibriSpeech test-clean (2620 utterances, our `fp16w` ONNX, Whisper English text normaliser). The PyTorch reference gives the same number. Fermion publishes 1.72 %, which comes from a different evaluation setup.

Speed, measured on a laptop with an RTX 3080 GPU (UE 5.8). Real-time factor is processing time divided by audio time.

| Device | Real-time factor | 10 s of audio |
|---|---|---|
| Laptop CPU | 0.076 | about 0.75 s |
| GPU (DirectML) | 0.053 | about 0.5 s |

The first GPU call takes about 4 s while DirectML warms up. A 497 s clip took 43 s on the CPU (real-time factor about 0.08).

## Attribution (CC-BY-4.0)

The Phonon-2 weights are CC-BY-4.0, and so is the NVIDIA model they come from. If your game includes the converted files, you must:

1. Credit Fermion Research (Phonon-2) and NVIDIA (parakeet-tdt-0.6b-v3).
2. Link the licence: https://creativecommons.org/licenses/by/4.0/
3. Say you changed it: "converted to ONNX".

Put this in your game's credits or notices. Each converted folder has a `NOTICE` file with the full text. See [Model licenses](ModelLicenses.md#cc-by-40-models-such-as-phonon-2) and the [third-party notices](../THIRD_PARTY_NOTICES.md#phonon-2-optional-user-downloaded-model). This page is not legal advice.

Problems? See [Troubleshooting](Troubleshooting.md#speech-to-text).
