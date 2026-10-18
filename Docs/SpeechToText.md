# Speech to text

Speech to text turns a recorded voice clip into text with a time for every word. The plugin runs Phonon-2, an English model, on your machine. No audio leaves the computer.

The model is **not** bundled. You download it, convert it to ONNX with the scripts in `Tools/`, and import the folder.

## What you need to know first

- Output is English, with capital letters and punctuation.
- One pipeline uses about 4 GB of RAM. Create one pipeline and share it.
- One call takes up to 600 seconds of audio.
- Real-time input (microphone or your own audio) is in [Real-time speech](#real-time-speech).
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

## Real-time speech

Real-time speech listens to a microphone, or to audio you push, and sends text events while people talk. It uses the same speech pipeline as above. It needs no extra model.

### How it works

An energy voice-activity detector cuts the audio into utterances. It works on 20 ms frames. A frame counts as speech when it is 12 dB over a running noise floor and above -55 dBFS. An utterance starts after 60 ms of speech and includes 0.3 s of audio from before that point (pre-roll). It ends after 0.2 s of silence by default. An utterance is cut at 30 s.

Events arrive on the game thread. For each utterance the order is: Speech Started, Partial (zero or more), Speech Ended, Final. Stopped comes once, at the end.

| Event | Meaning |
|---|---|
| Speech Started | An utterance began. |
| Partial | Text so far. **Only with the Phonon engine** (see "Stage B engine" below). |
| Speech Ended | The utterance ended. |
| Final | The full text of the utterance, with word times. |
| Stopped | The stream ended. Nothing follows. |

Both engines give finals. Only the Phonon engine gives partials.

### Start listening from a Blueprint

1. Create the speech pipeline and store it in a variable (see Step 4 above).
2. Add the node **Start Listening** (category **Model Pipelines > Speech**). Plug in **Pipeline**. Leave **Use Microphone** ticked and **Device Index** at -1 (the default device).
3. Wire the output pins you need: **On Speech Started**, **On Partial**, **On Final**, **On Speech Ended**, **On Stopped**, **On Failure**. Each carries **Transcript** and **Result**.
4. Store the node's return value in a variable.
5. Play and speak. You should see **On Speech Started**, then **On Final** after you stop. For the delay, see the latency table below.
6. To stop, call **Stop Listening** with the stored variable. **On Stopped** still fires once.

To feed your own audio (voice chat, game audio), untick **Use Microphone**. Call **Get Stream** on the stored node. Then call **Push Speech Audio** with the float samples, the sample rate and the channel count. Any rate from 1000 to 384000 Hz and any channel count work.

All pins: [Nodes](Nodes.md#start-listening).

### Settings

**Start Listening** takes a **Settings** struct (`FSpeechStreamSettings`). Out-of-range values are clamped.

| Setting | Default | Range | Meaning |
|---|---|---|---|
| `EndOfSpeechHangoverSeconds` | 0.2 | 0.1 to 2 | Silence that ends an utterance. |
| `PartialIntervalSeconds` | 0.5 | 0.1 to 5 | New speech audio between two partials. |
| `bPartials` | true | | Ask for partials. An engine without partials ignores it. |
| `SpeechThresholdDb` | 12 | 3 to 40 | Speech level over the noise floor. |
| `MaxUtteranceSeconds` | 30 | 2 to 120 | Longest utterance. |

### Speed with the ONNX engine (stage A)

The final text arrives after the 0.2 s hangover plus the decode time. The decode time grows with the length of the utterance.

### Latency

Measured on an Intel i7-11800H laptop CPU (8 cores), audio pushed at real-time speed, default settings. Latency is the time from the end of speech to **On Final**. Partial lag is the time from the end of a spoken word to the first partial text that contains it.

| Engine | Audio | Final text p50 / p90 | Partial lag p50 / p90 | WER |
|---|---|---|---|---|
| ONNX (stage A, free) | 12 short phrases, 2.4 to 3.9 s each | 0.60 / 0.65 s | no partials | 1.8 % |
| ONNX (stage A, free) | 45 s of read speech, 8 utterances | 0.53 / 0.82 s | no partials | 0.9 % |
| Phonon (stage B, ModelPipelines Generative) | 12 short phrases | 0.28 / 0.28 s | not measured | 1.8 % |
| Phonon (stage B, ModelPipelines Generative) | 45 s of read speech | 0.32 / 0.38 s | 0.16 / 0.52 s | 0.9 % |

A longer `EndOfSpeechHangoverSeconds` adds its difference to every final. A shorter one can split a sentence at a pause.

### Limits

- English only.
- Microphone capture uses the engine's AudioCapture backend on Windows. With no input device, **On Failure** fires.
- A quiet room can make speech detection less reliable. See [Troubleshooting](Troubleshooting.md#real-time-speech).
- **Stop Listening** can block the game thread for up to 1 s when the device was unplugged during capture. Normally it takes about 10 ms.

### Stage B engine (ModelPipelines Generative)

The paid ModelPipelines Generative plugin adds a second speech engine, Phonon. With it you get partial text, about 0.3 to 0.5 GB of RAM per engine (stage A: about 4 GB) and faster CPU decoding. Your Blueprints stay the same. See `Docs/SpeechStageB.md` in the ModelPipelines Generative plugin.

## Attribution (CC-BY-4.0)

The Phonon-2 weights are CC-BY-4.0, and so is the NVIDIA model they come from. If your game includes the converted files, you must:

1. Credit Fermion Research (Phonon-2) and NVIDIA (parakeet-tdt-0.6b-v3).
2. Link the licence: https://creativecommons.org/licenses/by/4.0/
3. Say you changed it: "converted to ONNX".

Put this in your game's credits or notices. Each converted folder has a `NOTICE` file with the full text. See [Model licenses](ModelLicenses.md#cc-by-40-models-such-as-phonon-2) and the [third-party notices](../THIRD_PARTY_NOTICES.md#phonon-2-optional-user-downloaded-model). This page is not legal advice.

Problems? See [Troubleshooting](Troubleshooting.md#speech-to-text).
