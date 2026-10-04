// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "TextAsyncActionTestReceiver.generated.h"

/** One delivery seen by the receiver. Plain data, filled on whatever thread the pin fired on. */
struct FTextAsyncActionTestRecord
{
	/** EmbedSuccess, EmbedFailure, ClassifySuccess or ClassifyFailure. */
	FName Pin;
	FPipelineResult Result;
	TArray<FPipelineEmbedding> Embeddings;
	TArray<FPipelineClassification> Classifications;
	bool bGameThread = false;
	/** The test sets UTextAsyncActionTestReceiver::bInsideActivate around the Activate() call; a pin that fires while it is set is a bug. */
	bool bInsideActivate = false;
};

/**
 * Pin listener for the text async action tests. The action delegates are dynamic, so the handlers must be UFUNCTIONs.
 * Every call is appended to Records; the counters are derived from it.
 */
UCLASS()
class UTextAsyncActionTestReceiver : public UObject
{
	GENERATED_BODY()

public:
	/** Set by the test around Activate(). */
	bool bInsideActivate = false;

	TArray<FTextAsyncActionTestRecord> Records;

	int32 Total() const { return Records.Num(); }

	int32 CountPin(FName Pin) const
	{
		int32 N = 0;
		for (const FTextAsyncActionTestRecord& R : Records) { if (R.Pin == Pin) { ++N; } }
		return N;
	}

	UFUNCTION()
	void OnEmbedSuccess(const FPipelineResult& Result, const TArray<FPipelineEmbedding>& Embeddings)
	{
		Add(TEXT("EmbedSuccess"), Result, &Embeddings, nullptr);
	}

	UFUNCTION()
	void OnEmbedFailure(const FPipelineResult& Result, const TArray<FPipelineEmbedding>& Embeddings)
	{
		Add(TEXT("EmbedFailure"), Result, &Embeddings, nullptr);
	}

	UFUNCTION()
	void OnClassifySuccess(const FPipelineResult& Result, const TArray<FPipelineClassification>& Classifications)
	{
		Add(TEXT("ClassifySuccess"), Result, nullptr, &Classifications);
	}

	UFUNCTION()
	void OnClassifyFailure(const FPipelineResult& Result, const TArray<FPipelineClassification>& Classifications)
	{
		Add(TEXT("ClassifyFailure"), Result, nullptr, &Classifications);
	}

private:
	void Add(const TCHAR* Pin, const FPipelineResult& Result, const TArray<FPipelineEmbedding>* E, const TArray<FPipelineClassification>* C)
	{
		FTextAsyncActionTestRecord& R = Records.AddDefaulted_GetRef();
		R.Pin = FName(Pin);
		R.Result = Result;
		if (E) { R.Embeddings = *E; }
		if (C) { R.Classifications = *C; }
		R.bGameThread = IsInGameThread();
		R.bInsideActivate = bInsideActivate;
	}
};
