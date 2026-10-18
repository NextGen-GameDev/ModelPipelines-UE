// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechVad.h"

#include "Math/UnrealMathUtility.h"

namespace UE::ModelPipelines::Speech
{
	namespace VadPrivate
	{
		constexpr float InitialFloorDb = -70.f;
		constexpr float MinLevelDb = -90.f;
		constexpr float AbsoluteGateDb = -55.f;
	}

	FSpeechVad::FSpeechVad(const FSpeechVadSettings& InSettings)
	{
		HangoverFrames = FMath::Max(1, FMath::RoundToInt(InSettings.HangoverSeconds * 50.f));
		MaxUtteranceSamples = FMath::Max<int64>(FrameSamples, (int64)FMath::RoundToDouble((double)InSettings.MaxUtteranceSeconds * 16000.0));
		ThresholdDb = InSettings.ThresholdDb;
		Levels.Init(VadPrivate::InitialFloorDb, FloorFrames);
	}

	float FSpeechVad::GetNoiseFloorDb() const
	{
		float Floor = Levels[0];
		for (const float L : Levels)
		{
			Floor = FMath::Min(Floor, L);
		}
		return FMath::Max(Floor, VadPrivate::MinLevelDb);
	}

	void FSpeechVad::Process(TConstArrayView<float> Mono16k, TArray<FSpeechVadEvent>& OutEvents)
	{
		for (const float S : Mono16k)
		{
			FrameSumSquares += (double)S * (double)S;
			if (++FrameFill == FrameSamples)
			{
				ProcessFrame(OutEvents);
				FrameSumSquares = 0.0;
				FrameFill = 0;
			}
		}
	}

	void FSpeechVad::ProcessFrame(TArray<FSpeechVadEvent>& OutEvents)
	{
		const double MeanSquare = FrameSumSquares / (double)FrameSamples;
		const float Level = MeanSquare > 1e-9 ? FMath::Max((float)(10.0 * FMath::LogX(10.0, MeanSquare)), VadPrivate::MinLevelDb) : VadPrivate::MinLevelDb;
		const float Floor = GetNoiseFloorDb();
		const bool bSpeech = Level > Floor + ThresholdDb && Level > VadPrivate::AbsoluteGateDb;

		Levels[RingPos] = Level;
		RingPos = (RingPos + 1) % FloorFrames;

		const int64 Frame = FrameIndex++;
		const int64 FrameEnd = (Frame + 1) * FrameSamples;

		if (!bSpeaking)
		{
			if (!bSpeech)
			{
				Consecutive = 0;
				return;
			}
			if (Consecutive == 0)
			{
				RunStartFrame = Frame;
			}
			if (++Consecutive < StartFrames)
			{
				return;
			}
			bSpeaking = true;
			UtteranceStart = FMath::Max3<int64>(RunStartFrame * FrameSamples - PreRollSamples, 0, LastEnded);
			LastSpeechEnd = FrameEnd;
			SilentFrames = 0;
			OutEvents.Add({ ESpeechVadEventType::Started, UtteranceStart });
			return;
		}

		if (bSpeech)
		{
			LastSpeechEnd = FrameEnd;
			SilentFrames = 0;
		}
		else
		{
			++SilentFrames;
		}

		if (SilentFrames >= HangoverFrames || FrameEnd - UtteranceStart >= MaxUtteranceSamples)
		{
			bSpeaking = false;
			Consecutive = 0;
			LastEnded = FrameEnd;
			OutEvents.Add({ ESpeechVadEventType::Ended, FrameEnd });
		}
	}
}
