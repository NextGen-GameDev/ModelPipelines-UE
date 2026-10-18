// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

// WP9R.3: listener for USpeechStream's dynamic delegates (they bind only to UFUNCTIONs). Records every event with its
// order, time and thread. Not inside WITH_DEV_AUTOMATION_TESTS (UHT ignores preprocessor guards).

#include "CoreMinimal.h"
#include "HAL/PlatformTime.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "PipelineResults.h"
#include "SpeechTypes.h"
#include "SpeechStreamTestReceiver.generated.h"

enum class ESpeechStreamTestEvent : uint8 { Started, Partial, Ended, Final, Stopped };

struct FSpeechStreamTestRecord
{
	ESpeechStreamTestEvent Type = ESpeechStreamTestEvent::Started;
	int32 Seq = 0;              // global order of arrival
	double TimeSeconds = 0.0;   // FPlatformTime::Seconds() at the broadcast
	bool bGameThread = false;
	FSpeechTranscript Transcript;   // Partial / Final
	FPipelineResult Result;         // Stopped
};

UCLASS()
class USpeechStreamTestReceiver : public UObject
{
	GENERATED_BODY()

public:
	TArray<FSpeechStreamTestRecord> Records;

	int32 Count(ESpeechStreamTestEvent Type) const
	{
		int32 N = 0;
		for (const FSpeechStreamTestRecord& R : Records) { if (R.Type == Type) { ++N; } }
		return N;
	}

	/** Records of one type, in arrival order. */
	TArray<FSpeechStreamTestRecord> Of(ESpeechStreamTestEvent Type) const
	{
		TArray<FSpeechStreamTestRecord> Out;
		for (const FSpeechStreamTestRecord& R : Records) { if (R.Type == Type) { Out.Add(R); } }
		return Out;
	}

	bool AllOnGameThread() const
	{
		for (const FSpeechStreamTestRecord& R : Records) { if (!R.bGameThread) { return false; } }
		return true;
	}

	UFUNCTION() void HandleStarted() { Add(ESpeechStreamTestEvent::Started, FSpeechTranscript(), FPipelineResult()); }
	UFUNCTION() void HandlePartial(const FSpeechTranscript& Transcript) { Add(ESpeechStreamTestEvent::Partial, Transcript, FPipelineResult()); }
	UFUNCTION() void HandleFinal(const FSpeechTranscript& Transcript) { Add(ESpeechStreamTestEvent::Final, Transcript, FPipelineResult()); }
	UFUNCTION() void HandleEnded() { Add(ESpeechStreamTestEvent::Ended, FSpeechTranscript(), FPipelineResult()); }
	UFUNCTION() void HandleStopped(const FPipelineResult& Result) { Add(ESpeechStreamTestEvent::Stopped, FSpeechTranscript(), Result); }

private:
	void Add(ESpeechStreamTestEvent Type, const FSpeechTranscript& T, const FPipelineResult& R)
	{
		FSpeechStreamTestRecord& Rec = Records.AddDefaulted_GetRef();
		Rec.Type = Type;
		Rec.Seq = Records.Num() - 1;
		Rec.TimeSeconds = FPlatformTime::Seconds();
		Rec.bGameThread = IsInGameThread();
		Rec.Transcript = T;
		Rec.Result = R;
	}
};
