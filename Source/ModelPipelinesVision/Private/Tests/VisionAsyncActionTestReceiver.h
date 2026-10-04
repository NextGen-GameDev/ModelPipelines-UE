// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "VisionAsyncActionTestReceiver.generated.h"

/** One delivery seen by the receiver. Plain data, filled on whatever thread the pin fired on. */
struct FVisionAsyncActionTestRecord
{
	/** ClassifySuccess, ClassifyFailure, DetectSuccess or DetectFailure. */
	FName Pin;
	FPipelineResult Result;
	FPipelineClassification Classification;
	TArray<FPipelineDetection> Detections;
	bool bGameThread = false;
	/** The test sets UVisionAsyncActionTestReceiver::bInsideActivate around the Activate() call; a pin that fires while it is set is a bug. */
	bool bInsideActivate = false;
};

/**
 * Pin listener for the vision async action tests. The action delegates are dynamic, so the handlers must be UFUNCTIONs.
 * Every call is appended to Records; the counters are derived from it.
 */
UCLASS()
class UVisionAsyncActionTestReceiver : public UObject
{
	GENERATED_BODY()

public:
	/** Set by the test around Activate(). */
	bool bInsideActivate = false;

	TArray<FVisionAsyncActionTestRecord> Records;

	int32 Total() const { return Records.Num(); }

	int32 CountPin(FName Pin) const
	{
		int32 N = 0;
		for (const FVisionAsyncActionTestRecord& R : Records) { if (R.Pin == Pin) { ++N; } }
		return N;
	}

	UFUNCTION()
	void OnClassifySuccess(const FPipelineResult& Result, const FPipelineClassification& Classification)
	{
		Add(TEXT("ClassifySuccess"), Result, &Classification, nullptr);
	}

	UFUNCTION()
	void OnClassifyFailure(const FPipelineResult& Result, const FPipelineClassification& Classification)
	{
		Add(TEXT("ClassifyFailure"), Result, &Classification, nullptr);
	}

	UFUNCTION()
	void OnDetectSuccess(const FPipelineResult& Result, const TArray<FPipelineDetection>& Detections)
	{
		Add(TEXT("DetectSuccess"), Result, nullptr, &Detections);
	}

	UFUNCTION()
	void OnDetectFailure(const FPipelineResult& Result, const TArray<FPipelineDetection>& Detections)
	{
		Add(TEXT("DetectFailure"), Result, nullptr, &Detections);
	}

private:
	void Add(const TCHAR* Pin, const FPipelineResult& Result, const FPipelineClassification* C, const TArray<FPipelineDetection>* D)
	{
		FVisionAsyncActionTestRecord& R = Records.AddDefaulted_GetRef();
		R.Pin = FName(Pin);
		R.Result = Result;
		if (C) { R.Classification = *C; }
		if (D) { R.Detections = *D; }
		R.bGameThread = IsInGameThread();
		R.bInsideActivate = bInsideActivate;
	}
};
