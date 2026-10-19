// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"

#if !UE_BUILD_SHIPPING

struct FSpeechStreamSettings;

namespace UE::ModelPipelines::Speech
{
	/**
	 * C85: developer recording of one USpeechStream session (mp.Speech.RecordStreams 1). Game thread only; Finish() hands a copy of the
	 * data to a background task that writes Saved/ModelPipelines/SpeechRecordings/<utc>-<n>.wav + .json. Not compiled in Shipping.
	 */
	class FSpeechStreamRecorder
	{
	public:
		/** Reads mp.Speech.RecordStreams; null when recording is off. Game thread, called once when the stream starts. */
		static TUniquePtr<FSpeechStreamRecorder> CreateIfEnabled(bool bPhonon, const FString& PackagePath, const FSpeechStreamSettings& Settings, bool bPartialsEnabled);

		/** 16 kHz mono audio after the front end. Audio beyond RecordMaxSeconds is dropped (truncated). */
		void AppendAudio(const TArray<float>& Audio);

		void AddStarted() { AddEvent(TEXT("started")); }
		void AddEnded() { AddEvent(TEXT("ended")); }
		void AddPartial(const FString& Text);
		void AddFinal(const FString& Text, bool bReplaces, float InferenceMs);
		void AddStopped(bool bOk, const FString& Message);

		/** Moves the data into a background writer task. Call once; the recorder is empty afterwards. */
		void Finish();

	private:
		struct FEvent
		{
			FString Type;
			double T = 0.0;
			int64 Sample = 0;
			FString Text;
			bool bHasText = false;
			bool bHasFinal = false;
			bool bReplaces = false;
			float InferenceMs = 0.f;
			bool bHasStopped = false;
			bool bOk = false;
			FString Message;
		};

		FEvent& AddEvent(const TCHAR* Type);

		bool bPhonon = false;
		FString Package;
		float Hangover = 0.f, PartialInterval = 0.f, ThresholdDb = 0.f, MaxUtterance = 0.f, MergeWindow = 0.f, MinSpeech = 0.f;
		bool bPartials = false;
		int64 MaxSamples = 0;
		bool bTruncated = false;
		double StartSeconds = 0.0;
		FDateTime StartedUtc;
		TArray<float> Audio;
		TArray<FEvent> Events;
		bool bDone = false;
	};
}

#endif // !UE_BUILD_SHIPPING
