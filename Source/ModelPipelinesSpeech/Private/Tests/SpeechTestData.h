// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"
#include "PipelineTypes.h"

class UPipelineModelPackage;
class UNNEModelData;
class USoundWave;
class USpeechToTextPipeline;

namespace UE::ModelPipelines::SpeechTests
{
	/** env MODELPIPELINES_DEV_DIR, else <ProjectDir>/../.pipelines-dev (absolute). */
	FString GetDevDir();

	/** One onnx.word_frames entry: word text and start/end in encoder frames (seconds = frames * 0.08 at the default settings). */
	struct FSttGoldWord
	{
		FString Word;
		int32 StartFrame = 0;
		int32 EndFrame = 0;
	};

	/** One entry of stt_goldens.json "clips" (schema stt-goldens/1, contract C71). File fields are relative to <dev>/goldens/stt. */
	struct FSttClip
	{
		FString Id;
		bool bSynthetic = false;
		FString AudioFile;
		int32 NumSamples = 0;
		bool bHasRms = false;
		double RmsDbfs = 0.0;
		bool bHasFeatures = false;
		FString FeaturesFile;
		int32 FeatureFrames = 0;
		bool bHasLogmel = false;
		FString LogmelFile;
		int32 LogmelFrames = 0;

		// WP9S.5a: the "onnx" block (ONNX Runtime reference, greedy TDT loop) and hf.text. All clips of the file have it.
		bool bHasOnnx = false;
		FString OnnxText;               // onnx.text
		FString HfText;                 // hf.text
		TArray<int32> StepTokens;       // onnx.steps[i][0] (8192 = blank)
		TArray<int32> StepDurations;    // onnx.steps[i][1]
		TArray<int32> TokenIds;         // onnx.token_ids (non-blank k)
		TArray<FSttGoldWord> Words;     // onnx.word_frames
	};

	struct FSttGoldens
	{
		FString Dir;                 // <dev>/goldens/stt
		FString MelFile;             // relative
		int32 MelRows = 0;
		int32 MelCols = 0;
		TArray<FSttClip> Clips;

		const FSttClip* Find(const FString& Id) const;

		/** The 40 visible LibriSpeech clips (kind != synthetic), in file order. */
		TArray<const FSttClip*> VisibleClips() const;
	};

	/** Loads (and caches for the process) <dev>/goldens/stt/stt_goldens.json. False + OutError naming the path and Tools/make_stt_goldens.py. */
	bool LoadSttGoldens(const FSttGoldens*& OutGoldens, FString& OutError);

	/** Reads a raw little-endian float32 sidecar (path relative to the goldens dir). The byte size must equal ExpectedCount * 4. */
	bool LoadFloatFile(const FSttGoldens& G, const FString& RelFile, int64 ExpectedCount, TArray<float>& OutData, FString& OutError);

	// ---- WP9S.5a: model packages and clip audio (game thread only unless stated) ----

	/** The 16 kHz mono float32 samples of a golden clip. False + OutError (path + generator command) when missing. Any thread. */
	bool LoadClipAudio(const FSttGoldens& G, const FSttClip& Clip, TArray<float>& OutSamples, FString& OutError);

	/** env MODELPIPELINES_REQUIRE_GPU == "1". */
	bool IsGpuRequired();

	/** <dev>/models/stt/phonon-2-onnx/fp16w. */
	FString GetPhononModelDir();

	/**
	 * The Phonon-2 fp16w package (Task SpeechToText, Model = encoder.onnx, DecoderModel = decoder_joint.onnx, TokenizerJson,
	 * default Speech settings, License cc-by-4.0), built ONCE per process (about 1.3 GB read, takes seconds) and kept alive
	 * with a TStrongObjectPtr until the engine exits. PreferredDevice is set to Device on every call (a pipeline copies the
	 * package at InitializeFromPackage, so create the pipeline right after this call). Game thread. False + OutError naming
	 * the missing file and the fetch command.
	 */
	bool GetSharedPhononPackage(EPipelineDevice Device, UPipelineModelPackage*& OutPackage, FString& OutError);

	/**
	 * Tiny speech package for error-path tests (tiny_stt_encoder/decoder_joint, Phonon tokenizer.json, default Speech settings).
	 * Cached once per process like the Phonon one. Not meant for real transcription (toy sizes).
	 */
	bool GetSharedTinySttPackage(UPipelineModelPackage*& OutPackage, FString& OutError);

	/** A tiny .onnx from <dev>/models/test as a new UNNEModelData (outer = transient package). Game thread. */
	UNNEModelData* LoadTestModelData(const FString& FileName, FString& OutError);

	// ---- WP9S.5b ----

	/** ONE CPU Phonon pipeline for the whole process (WP9S.5b specs share it), created on first use. Game thread. Null + OutError on failure. */
	USpeechToTextPipeline* GetSharedCpuSttPipeline(FString& OutError);

	/** One pipeline on the tiny package (error paths only), created on first use. Game thread. */
	USpeechToTextPipeline* GetSharedTinySttPipeline(FString& OutError);

	/** round(s * 32768) clamped to int16. */
	void FloatToInt16(TConstArrayView<float> In, TArray<int16>& Out);

	/**
	 * A transient USoundWave filled like the engine's FSoundWavePCMWriter::ApplyBufferToSoundWave: SetSampleRate, NumChannels, Duration,
	 * TotalSamples, RawPCMDataSize, RawPCMData (int16 interleaved). No asset, no compressed data. Game thread. The caller owns it
	 * (wrap in a TStrongObjectPtr if a GC can run).
	 */
	USoundWave* MakeGeneratedSoundWave(const TArray<int16>& Interleaved, int32 SampleRate, int32 NumChannels);

	// ---- WP9S.5c2: long-audio goldens (<dev>/goldens/stt/long/long_goldens.json, schema stt-long-goldens/1) ----

	struct FSttLongWord
	{
		FString Word;
		double Start = 0.0;   // seconds (window-local in FSttLongWindow::Words, global in FSttLongRecording::Words)
		double End = 0.0;
	};

	struct FSttLongWindow
	{
		int32 Start = 0;      // [Start, End) in samples of the 16 kHz mono signal
		int32 End = 0;
		double RmsDbfs = 0.0;
		bool bGated = false;
		TArray<int32> TokenIds;
		FString Text;
		TArray<FSttLongWord> Words;
	};

	struct FSttLongRecording
	{
		FString Id;
		FString Kind;          // "librispeech", or the synthetic silence/gap ones
		FString AudioFile;     // relative to FSttLongGoldens::Dir
		int32 NumSamples = 0;
		FString Ref;           // reference transcript (may be empty)
		TArray<FSttLongWindow> Windows;
		FString Text;
		TArray<FSttLongWord> Words;
	};

	struct FSttLongGoldens
	{
		FString Dir;           // <dev>/goldens/stt/long
		TArray<FSttLongRecording> Recordings;
		const FSttLongRecording* Find(const FString& Id) const;
	};

	/**
	 * Drops the strong reference to the shared CPU pipeline and runs a GC, so it is destroyed (if no other strong ref exists).
	 * The next GetSharedCpuSttPipeline call builds a new one. Game thread. For the "destroyed mid-call" test, which must run last.
	 */
	void ReleaseSharedCpuSttPipeline();

	/** Loads (and caches) long_goldens.json. False + OutError naming the path and Tools/make_stt_long_goldens.py. Any thread. */
	bool LoadSttLongGoldens(const FSttLongGoldens*& OutGoldens, FString& OutError);

	/** The 16 kHz mono samples of a long recording (num_samples float32). False + OutError (path + generator command). Any thread. */
	bool LoadLongAudio(const FSttLongGoldens& G, const FSttLongRecording& Rec, TArray<float>& OutSamples, FString& OutError);
}

#endif // WITH_DEV_AUTOMATION_TESTS
