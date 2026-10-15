// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"

namespace UE::ModelPipelines::SpeechTests
{
	/** env MODELPIPELINES_DEV_DIR, else <ProjectDir>/../.pipelines-dev (absolute). */
	FString GetDevDir();

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
	};

	struct FSttGoldens
	{
		FString Dir;                 // <dev>/goldens/stt
		FString MelFile;             // relative
		int32 MelRows = 0;
		int32 MelCols = 0;
		TArray<FSttClip> Clips;

		const FSttClip* Find(const FString& Id) const;
	};

	/** Loads (and caches for the process) <dev>/goldens/stt/stt_goldens.json. False + OutError naming the path and Tools/make_stt_goldens.py. */
	bool LoadSttGoldens(const FSttGoldens*& OutGoldens, FString& OutError);

	/** Reads a raw little-endian float32 sidecar (path relative to the goldens dir). The byte size must equal ExpectedCount * 4. */
	bool LoadFloatFile(const FSttGoldens& G, const FString& RelFile, int64 ExpectedCount, TArray<float>& OutData, FString& OutError);
}

#endif // WITH_DEV_AUTOMATION_TESTS
