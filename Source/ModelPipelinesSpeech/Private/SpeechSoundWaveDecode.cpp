// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechSoundWaveDecode.h"

#include "DSP/BufferVectorOperations.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "Math/UnrealMathUtility.h"
#include "Sound/SoundWaveProxyReader.h"

namespace UE::ModelPipelines::Speech
{
	namespace SoundWaveDecodePrivate
	{
		static FPipelineResult Fail(const FSoundWaveSource& Source, const FString& Reason)
		{
			return FPipelineResult::Fail(EPipelineError::PreprocessFailed,
				FString::Printf(TEXT("could not decode sound wave '%s': %s"), *Source.Name, *Reason));
		}

		static FPipelineResult DecodeRawPcm(FSoundWaveSource& Source, TArray<float>& Out, int32& OutRate, int32& OutChannels)
		{
			if (Source.PcmNumChannels <= 0)
			{
				return Fail(Source, FString::Printf(TEXT("invalid channel count %d"), Source.PcmNumChannels));
			}
			if (Source.PcmSampleRate <= 0)
			{
				return Fail(Source, FString::Printf(TEXT("invalid sample rate %d"), Source.PcmSampleRate));
			}
			const int32 NumFrames = (Source.Pcm16.Num() / (int32)sizeof(int16)) / Source.PcmNumChannels;
			if (NumFrames <= 0)
			{
				return Fail(Source, TEXT("the wave has no PCM frames"));
			}
			const int32 NumSamples = NumFrames * Source.PcmNumChannels;
			Out.SetNumUninitialized(NumSamples);
			const uint8* Bytes = Source.Pcm16.GetData();
			for (int32 i = 0; i < NumSamples; ++i)
			{
				int16 S;
				FMemory::Memcpy(&S, Bytes + (SIZE_T)i * sizeof(int16), sizeof(int16));
				Out[i] = (float)S / 32768.0f;
			}
			Source.Pcm16.Empty();
			OutRate = Source.PcmSampleRate;
			OutChannels = Source.PcmNumChannels;
			return FPipelineResult::Ok();
		}
	}

	void CaptureSoundWaveSource(USoundWave& Sound, FSoundWaveSource& Out)
	{
		Out.Name = Sound.GetName();
		if (Sound.RawPCMData != nullptr && Sound.RawPCMDataSize > 0)
		{
			Out.bRawPcm = true;
			Out.Pcm16 = TArray<uint8>(Sound.RawPCMData, Sound.RawPCMDataSize);
			Out.PcmSampleRate = FMath::RoundToInt32(Sound.GetSampleRateForCurrentPlatform());
			Out.PcmNumChannels = Sound.NumChannels;
			return;
		}
#if MODELPIPELINES_SPEECH_WAVE_DATA_REF
		// USoundWave::GetSoundWaveProxy (SoundWave.h:1207) -> FSoundWaveProxy::GetSoundWaveDataRef (SoundWave.h:1911).
		Out.WaveData = Sound.GetSoundWaveProxy()->GetSoundWaveDataRef().ToSharedPtr();
#else
		Out.WaveProxy = Sound.CreateSoundWaveProxy();
#endif
	}

	FPipelineResult DecodeSoundWaveSource(FSoundWaveSource& Source, TArray<float>& OutSamples, int32& OutSampleRate, int32& OutNumChannels)
	{
		using namespace SoundWaveDecodePrivate;

		OutSamples.Reset();
		OutSampleRate = 0;
		OutNumChannels = 0;
		if (Source.bRawPcm)
		{
			return DecodeRawPcm(Source, OutSamples, OutSampleRate, OutNumChannels);
		}

PRAGMA_DISABLE_DEPRECATION_WARNINGS
		FSoundWaveProxyReader::FSettings Settings;
		Settings.bIsLooping = false;
		TUniquePtr<FSoundWaveProxyReader> Reader;
#if MODELPIPELINES_SPEECH_WAVE_DATA_REF
		if (!Source.WaveData.IsValid())
		{
			return Fail(Source, TEXT("the wave has no sound wave data"));
		}
		Reader = FSoundWaveProxyReader::Create(Source.WaveData.ToSharedRef(), Settings);
		Source.WaveData.Reset();
#else
		if (!Source.WaveProxy.IsValid())
		{
			return Fail(Source, TEXT("the wave has no sound wave proxy"));
		}
		Reader = FSoundWaveProxyReader::Create(Source.WaveProxy.ToSharedRef(), Settings);
		Source.WaveProxy.Reset();
#endif
		if (!Reader.IsValid())
		{
			return Fail(Source, TEXT("no reader could be created (invalid sample rate, channel count or frame count)"));
		}
		if (Reader->HasFailed())
		{
			return Fail(Source, TEXT("the decoder failed"));
		}
		const int32 NumChannels = Reader->GetNumChannels();
		const int32 NumFrames = Reader->GetNumFramesInWave();
		const int32 SampleRate = FMath::RoundToInt32(Reader->GetSampleRate());
		if (NumChannels <= 0)
		{
			return Fail(Source, FString::Printf(TEXT("invalid channel count %d"), NumChannels));
		}
		if (NumFrames <= 0)
		{
			return Fail(Source, TEXT("the wave has 0 frames"));
		}
		if (SampleRate <= 0)
		{
			return Fail(Source, FString::Printf(TEXT("invalid sample rate %d"), SampleRate));
		}
		const int64 Total64 = (int64)NumFrames * (int64)NumChannels;
		if (Total64 > (int64)MAX_int32)
		{
			return Fail(Source, TEXT("the wave is too long"));
		}
		const int32 Total = (int32)Total64;

		OutSamples.Reserve(Total);
		Audio::FAlignedFloatBuffer Buffer;
		Buffer.SetNumZeroed((int32)Settings.MaxDecodeSizeInFrames * NumChannels);
		// Streamed waves (stream caching: LoadOnDemand, RetainOnLoad, ...) request their chunks asynchronously; PopAudio then
		// returns fewer samples than asked ("streaming data not being available yet", SoundWaveProxyReader.cpp:309-317).
		// Keep only what it wrote and retry after a short sleep until the chunk is in, up to StallTimeoutSeconds without progress.
		constexpr double StallTimeoutSeconds = 10.0;
		double LastProgress = FPlatformTime::Seconds();
		while (OutSamples.Num() < Total && Reader->CanProduceMoreAudio())
		{
			const int32 Written = Reader->PopAudio(Buffer);
			if (Reader->HasFailed())
			{
				break;
			}
			const int32 Keep = FMath::Min(FMath::Clamp(Written, 0, Buffer.Num()), Total - OutSamples.Num());
			if (Keep > 0)
			{
				OutSamples.Append(Buffer.GetData(), Keep);
				LastProgress = FPlatformTime::Seconds();
				continue;
			}
			if (FPlatformTime::Seconds() - LastProgress > StallTimeoutSeconds)
			{
				const int32 Got = OutSamples.Num();
				OutSamples.Reset();
				return Fail(Source, FString::Printf(TEXT("timed out waiting for streamed audio data after %d of %d samples"),
					Got, Total));
			}
			FPlatformProcess::Sleep(0.002f);
		}
		if (Reader->HasFailed())
		{
			OutSamples.Reset();
			return Fail(Source, TEXT("the decoder failed"));
		}
		// Exactly NumFrames * NumChannels samples; zero-pad when the decoder produced fewer.
		OutSamples.SetNumZeroed(Total);
		OutSampleRate = SampleRate;
		OutNumChannels = NumChannels;
		return FPipelineResult::Ok();
	}
PRAGMA_ENABLE_DEPRECATION_WARNINGS
}
