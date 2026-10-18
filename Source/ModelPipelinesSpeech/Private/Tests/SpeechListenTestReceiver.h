// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

// WP9R.4: listener for the six pins of USpeechListenAsyncAction (FSpeechListenPin: Transcript + Result). Dynamic delegates bind
// only to UFUNCTIONs. Not inside WITH_DEV_AUTOMATION_TESTS (UHT ignores preprocessor guards).

#include "CoreMinimal.h"
#include "HAL/PlatformTime.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "PipelineResults.h"
#include "SpeechTypes.h"
#include "SpeechListenTestReceiver.generated.h"

enum class ESpeechListenTestEvent : uint8 { Started, Partial, Final, Ended, Stopped, Failure };

struct FSpeechListenTestRecord
{
	ESpeechListenTestEvent Type = ESpeechListenTestEvent::Started;
	int32 Seq = 0;
	bool bGameThread = false;
	FSpeechTranscript Transcript;
	FPipelineResult Result;
};

UCLASS()
class USpeechListenTestReceiver : public UObject
{
	GENERATED_BODY()

public:
	TArray<FSpeechListenTestRecord> Records;

	int32 Count(ESpeechListenTestEvent Type) const
	{
		int32 N = 0;
		for (const FSpeechListenTestRecord& R : Records) { if (R.Type == Type) { ++N; } }
		return N;
	}

	TArray<FSpeechListenTestRecord> Of(ESpeechListenTestEvent Type) const
	{
		TArray<FSpeechListenTestRecord> Out;
		for (const FSpeechListenTestRecord& R : Records) { if (R.Type == Type) { Out.Add(R); } }
		return Out;
	}

	UFUNCTION() void HandleStarted(const FSpeechTranscript& T, const FPipelineResult& R) { Add(ESpeechListenTestEvent::Started, T, R); }
	UFUNCTION() void HandlePartial(const FSpeechTranscript& T, const FPipelineResult& R) { Add(ESpeechListenTestEvent::Partial, T, R); }
	UFUNCTION() void HandleFinal(const FSpeechTranscript& T, const FPipelineResult& R) { Add(ESpeechListenTestEvent::Final, T, R); }
	UFUNCTION() void HandleEnded(const FSpeechTranscript& T, const FPipelineResult& R) { Add(ESpeechListenTestEvent::Ended, T, R); }
	UFUNCTION() void HandleStopped(const FSpeechTranscript& T, const FPipelineResult& R) { Add(ESpeechListenTestEvent::Stopped, T, R); }
	UFUNCTION() void HandleFailure(const FSpeechTranscript& T, const FPipelineResult& R) { Add(ESpeechListenTestEvent::Failure, T, R); }

private:
	void Add(ESpeechListenTestEvent Type, const FSpeechTranscript& T, const FPipelineResult& R)
	{
		FSpeechListenTestRecord& Rec = Records.AddDefaulted_GetRef();
		Rec.Type = Type;
		Rec.Seq = Records.Num() - 1;
		Rec.bGameThread = IsInGameThread();
		Rec.Transcript = T;
		Rec.Result = R;
	}
};
