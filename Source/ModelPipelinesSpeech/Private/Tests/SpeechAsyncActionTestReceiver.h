// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "PipelineResults.h"
#include "SpeechTypes.h"
#include "SpeechAsyncActionTestReceiver.generated.h"

/** One delivery seen by the receiver. Plain data, filled on whatever thread the pin fired on. */
struct FSpeechAsyncActionTestRecord
{
	/** "Success" or "Failure". */
	FName Pin;
	FPipelineResult Result;
	FSpeechTranscript Transcript;
	bool bGameThread = false;
	/** True when the pin fired while USpeechAsyncActionTestReceiver::bInsideActivate was set (a bug). */
	bool bInsideActivate = false;
};

/** Pin listener for the speech async action tests (the delegates are dynamic, so the handlers are UFUNCTIONs). */
UCLASS()
class USpeechAsyncActionTestReceiver : public UObject
{
	GENERATED_BODY()

public:
	bool bInsideActivate = false;

	TArray<FSpeechAsyncActionTestRecord> Records;

	int32 Total() const { return Records.Num(); }

	int32 CountPin(FName Pin) const
	{
		int32 N = 0;
		for (const FSpeechAsyncActionTestRecord& R : Records) { if (R.Pin == Pin) { ++N; } }
		return N;
	}

	UFUNCTION()
	void OnSuccess(const FPipelineResult& Result, const FSpeechTranscript& Transcript)
	{
		Add(TEXT("Success"), Result, Transcript);
	}

	UFUNCTION()
	void OnFailure(const FPipelineResult& Result, const FSpeechTranscript& Transcript)
	{
		Add(TEXT("Failure"), Result, Transcript);
	}

private:
	void Add(const TCHAR* Pin, const FPipelineResult& Result, const FSpeechTranscript& Transcript)
	{
		FSpeechAsyncActionTestRecord& R = Records.AddDefaulted_GetRef();
		R.Pin = FName(Pin);
		R.Result = Result;
		R.Transcript = Transcript;
		R.bGameThread = IsInGameThread();
		R.bInsideActivate = bInsideActivate;
	}
};
