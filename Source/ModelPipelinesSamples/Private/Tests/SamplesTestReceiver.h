// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "PipelineTypes.h"
#include "SamplesTestReceiver.generated.h"

/**
 * Listener for AModelPipelinesSampleBase::OnSampleResult (a dynamic delegate, so the handler has to be a UFUNCTION).
 * Every delivery is appended to Results; nothing here is thread-safe on purpose: a delivery off the game thread is recorded
 * in bAllOnGameThread and fails the test.
 */
UCLASS()
class USamplesTestReceiver : public UObject
{
	GENERATED_BODY()

public:
	/** Every FPipelineResult delivered, in order. */
	TArray<FPipelineResult> Results;

	/** False as soon as one delivery arrived on another thread. */
	bool bAllOnGameThread = true;

	int32 Count() const { return Results.Num(); }

	UFUNCTION()
	void OnResult(const FPipelineResult& Result)
	{
		Results.Add(Result);
		if (!IsInGameThread())
		{
			bAllOnGameThread = false;
		}
	}
};
