// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Containers/ArrayView.h"

namespace UE::ModelPipelines::Speech
{
	/** Energy VAD settings (WP9R.3; FSpeechStreamSettings after clamping). */
	struct FSpeechVadSettings
	{
		float HangoverSeconds = 0.3f;
		float ThresholdDb = 12.f;
		float MaxUtteranceSeconds = 30.f;
	};

	enum class ESpeechVadEventType : uint8
	{
		Started,
		Ended
	};

	/** SampleIndex: 16 kHz samples from the first sample given to the VAD. */
	struct FSpeechVadEvent
	{
		ESpeechVadEventType Type = ESpeechVadEventType::Started;
		int64 SampleIndex = 0;
	};

	/**
	 * Energy VAD on 20 ms frames of 16 kHz mono (WP9R.3 rule 3 + the lead's noise-floor clarification). Streaming state
	 * machine: the events do not depend on how the audio is chunked. One thread at a time.
	 *  - frame level = 10*log10(mean square), never below -90 dB;
	 *  - noise floor = minimum level over the previous 150 frames (3 s); slots not yet filled hold the initial -70 dB
	 *    (lead decision, WP9R.3 fix round 1, amends the clarification: the pre-filled ring lets a burst at sample 0 be detected;
	 *    the cost is at most one spurious detection in the first 3 s of a stream that starts in loud steady noise);
	 *  - speech frame: level > floor + ThresholdDb and level > -55 dBFS;
	 *  - Started after 3 consecutive speech frames, at the first of them - 0.3 s (clamped at 0 and at the previous end);
	 *  - Ended after HangoverSeconds without speech frames, at the end of the last speech frame + the hangover;
	 *  - an utterance reaching MaxUtteranceSeconds is ended at that frame end (a new one may start right away).
	 */
	class FSpeechVad
	{
	public:
		explicit FSpeechVad(const FSpeechVadSettings& InSettings);

		/** Appends the events completed by this audio (any chunk size). Samples must be finite. */
		void Process(TConstArrayView<float> Mono16k, TArray<FSpeechVadEvent>& OutEvents);

		bool IsSpeaking() const { return bSpeaking; }
		float GetNoiseFloorDb() const;

		static constexpr int32 FrameSamples = 320;
		static constexpr int32 PreRollSamples = 4800;
		static constexpr int32 FloorFrames = 150;
		static constexpr int32 StartFrames = 3;

	private:
		void ProcessFrame(TArray<FSpeechVadEvent>& OutEvents);

		int32 HangoverFrames = 15;
		int64 MaxUtteranceSamples = 480000;
		float ThresholdDb = 12.f;

		double FrameSumSquares = 0.0;
		int32 FrameFill = 0;
		int64 FrameIndex = 0;

		TArray<float> Levels;   // ring of FloorFrames levels
		int32 RingPos = 0;

		bool bSpeaking = false;
		int32 Consecutive = 0;
		int64 RunStartFrame = 0;
		int64 UtteranceStart = 0;
		int64 LastSpeechEnd = 0;
		int32 SilentFrames = 0;
		int64 LastEnded = 0;
	};
}
