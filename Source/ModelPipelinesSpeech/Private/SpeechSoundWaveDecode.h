// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Runtime/Launch/Resources/Version.h"
#include "Sound/SoundWave.h"
#include "Templates/SharedPointer.h"
#include "PipelineTypes.h"

#define MODELPIPELINES_SPEECH_WAVE_DATA_REF (ENGINE_MAJOR_VERSION > 5 || (ENGINE_MAJOR_VERSION == 5 && ENGINE_MINOR_VERSION >= 8))

namespace UE::ModelPipelines::Speech
{
	/** What a USoundWave holds, captured on the game thread so the decode never touches the UObject (docs/contracts.md C76). */
	struct FSoundWaveSource
	{
		/** For messages. */
		FString Name;

		/** RawPCMData path: int16 interleaved bytes, copied from the wave. */
		bool bRawPcm = false;
		TArray<uint8> Pcm16;
		int32 PcmSampleRate = 0;
		int32 PcmNumChannels = 0;

#if MODELPIPELINES_SPEECH_WAVE_DATA_REF
		/** Compressed path (5.8): thread-safe snapshot of the wave's data. */
		TSharedPtr<const FSoundWaveData, ESPMode::ThreadSafe> WaveData;
#else
		/** Compressed path (5.7): the wave's proxy. */
		FSoundWaveProxyPtr WaveProxy;
#endif
	};

	/** Game thread. Copies RawPCMData when present, otherwise takes the wave's thread-safe snapshot. */
	void CaptureSoundWaveSource(USoundWave& Sound, FSoundWaveSource& Out);

	/** Any thread. Decodes to interleaved float samples. Failure -> PreprocessFailed "could not decode sound wave '<name>': ...". */
	FPipelineResult DecodeSoundWaveSource(FSoundWaveSource& Source, TArray<float>& OutSamples, int32& OutSampleRate, int32& OutNumChannels);
}
