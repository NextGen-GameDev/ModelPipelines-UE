// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "PipelineMcpTestReceiver.generated.h"

/**
 * Listener for UPipelineMcpJob::OnDone (a dynamic multicast delegate, so the handler has to be a UFUNCTION).
 * Every broadcast is appended; a broadcast off the game thread clears bAllOnGameThread and fails the test.
 */
UCLASS()
class UPipelineMcpTestReceiver : public UObject
{
	GENERATED_BODY()

public:
	/** Number of OnDone broadcasts received. */
	int32 Count = 0;
	TArray<bool> Successes;
	TArray<FString> Jsons;
	bool bAllOnGameThread = true;

	UFUNCTION()
	void OnDone(bool bSuccess, const FString& Json)
	{
		++Count;
		Successes.Add(bSuccess);
		Jsons.Add(Json);
		if (!IsInGameThread())
		{
			bAllOnGameThread = false;
		}
	}
};
