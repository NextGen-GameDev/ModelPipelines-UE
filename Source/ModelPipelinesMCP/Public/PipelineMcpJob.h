// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Containers/Ticker.h"
#include "Templates/SharedPointer.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineMcpJob.generated.h"

class FPipelineMcpRunner;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FPipelineMcpJobDone, bool, bSuccess, const FString&, Json);

/** One running MCP tool call (benchmark / sample run). Completes exactly once, on the game thread, and NEVER inside the call
    that created it (always at least one game-thread tick later), so callers can bind OnDone after creation. */
UCLASS(BlueprintType)
class MODELPIPELINESMCP_API UPipelineMcpJob : public UObject
{
	GENERATED_BODY()

public:
	/** Fires once when the job ends. bSuccess false -> Json is {"error":"<message>"}. */
	UPROPERTY(BlueprintAssignable, Category = "Model Pipelines|MCP")
	FPipelineMcpJobDone OnDone;

	UFUNCTION(BlueprintPure, Category = "Model Pipelines|MCP")
	bool IsDone() const;

	/** false until done. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|MCP")
	bool Succeeded() const;

	/** Result JSON once done ("" before). */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|MCP")
	FString GetJson() const;

private:
	/** Module-internal control of the job (PipelineMcpRun.h). */
	friend struct FPipelineMcpJobInternal;

	/** Objects the running job uses (pipelines, transient package duplicates, textures). Emptied when the job ends. */
	UPROPERTY(Transient)
	TArray<TObjectPtr<UObject>> KeepAlive;

	/** The work behind the job; released when the job ends. */
	TSharedPtr<FPipelineMcpRunner> Runner;

	/** Core ticker entry that starts the runner one tick after creation; reset once it fired or the job ended. */
	FTSTicker::FDelegateHandle StartTicker;

	bool bDone = false;
	bool bSucceeded = false;
	FString ResultJson;
};
