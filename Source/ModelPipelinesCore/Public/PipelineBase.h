// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Templates/Function.h"
#include "Templates/SharedPointer.h"
#include "HAL/ThreadSafeCounter.h"
#include "Tasks/Task.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineTypes.h"
#include "PipelineTensor.h"
#include "PipelineTaskIO.h"
#include "PipelineModelPackage.h"
#include "PipelineBase.generated.h"

class IPipelineBackend;
class IPipelineModel;

/** What the stage functions see. Filled once, on the first call, from the initialized model; immutable afterwards. */
struct MODELPIPELINESCORE_API FPipelineModelInfo
{
	/** ResolvePipelineTaskIO result (C13). */
	FPipelineTaskBindings Bindings;
	/** Model inputs, in model order. */
	TArray<FPipelineTensorDesc> InputDescs;
	/** Model outputs, in model order. */
	TArray<FPipelineTensorDesc> OutputDescs;
};

/** The three task-specific stages of one pipeline call (docs/contracts.md C19). */
struct MODELPIPELINESCORE_API FPipelineStages
{
	/** Background thread. Fill OutInputs in model input order (one tensor per InputDescs entry). Return Ok() or a
	    failed result (no inference then; Complete receives it). */
	TUniqueFunction<FPipelineResult(const FPipelineModelInfo& Info, TArray<FPipelineTensor>& OutInputs)> Preprocess;
	/** Background thread. Outputs are the model outputs in model order. Return Ok() or a failed result. */
	TUniqueFunction<FPipelineResult(const FPipelineModelInfo& Info, TArray<FPipelineTensor>&& Outputs)> Postprocess;
	/** Game thread, exactly once per RunStages call, never inside RunStages. Result carries Device and all timings. */
	TUniqueFunction<void(const FPipelineResult& Result)> Complete;
	/** Optional. Preprocess is launched only after this task completes (Vision: the GPU readback). Default: invalid
	    handle = no prerequisite. No worker blocks while waiting. */
	UE::Tasks::FTask Prerequisite;
};

/**
 * Base of every task pipeline. Binds to a UPipelineModelPackage and runs calls as
 * Preprocess (background) -> FPipelineExecutor (inference) -> Postprocess (background) -> Complete (game thread).
 * In-flight calls hold only a shared runtime struct, never this UObject; destroying the pipeline while calls are in
 * flight is allowed.
 */
UCLASS(Abstract, BlueprintType)
class MODELPIPELINESCORE_API UPipelineBase : public UObject
{
	GENERATED_BODY()

public:
	/** Game thread. Binds the pipeline to a package; uses FModelPipelinesCoreModule::Get().GetDefaultBackend(). */
	FPipelineResult InitializeFromPackage(UPipelineModelPackage* InPackage);

	/** Same with an explicit backend (tests pass one with a bogus GPU runtime name). */
	FPipelineResult InitializeFromPackage(UPipelineModelPackage* InPackage, IPipelineBackend& Backend);

	/** InitializeFromPackage succeeded. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines")
	bool IsReady() const;

	/** The task this class handles. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines")
	EPipelineTask GetTask() const;

	/** The package bound by InitializeFromPackage; null before. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines")
	UPipelineModelPackage* GetPackage() const;

	/** Auto until the model is initialized, then the device it runs on. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines")
	EPipelineDevice GetDevice() const;

	/** RunStages calls whose Complete has not run. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines")
	int32 GetNumPendingCalls() const;

protected:
	/** Set by each subclass constructor. */
	EPipelineTask PipelineTask = EPipelineTask::None;

	/** Game thread, called last by InitializeFromPackage. Subclass extras (e.g. the tokenizer). Default: Ok(). */
	virtual FPipelineResult OnInitialize(UPipelineModelPackage* InPackage);

	/** Game thread. Runs Preprocess (background) -> FPipelineExecutor (inference) -> Postprocess (background) ->
	    Complete (game thread). Called off the game thread or before IsReady(): Complete still runs once on the game
	    thread with Fail(InvalidArgument). */
	void RunStages(FPipelineStages&& Stages);

	/** Tests and tools. Any thread except the executor worker; the caller keeps the pipeline alive. Runs Preprocess,
	    FPipelineExecutor::RunBlocking and Postprocess on the calling thread and returns the result (Complete unused).
	    Not ready -> Fail(InvalidArgument). */
	FPipelineResult RunStagesBlocking(FPipelineStages&& Stages);

	/** Copies taken at InitializeFromPackage; later edits to the package asset do not affect this pipeline. Any thread. */
	const FPipelinePreprocessSettings& GetPreprocessSettings() const;
	const FPipelinePostprocessSettings& GetPostprocessSettings() const;
	const TArray<FString>& GetLabels() const;

	/** The model prepared by InitializeFromPackage (visible from OnInitialize on); null before and after a failed
	    initialization. Subclasses that run sessions (C61) create them on it. Any thread. (Noah 2026-10-13.) */
	TSharedPtr<IPipelineModel> GetModel() const;

private:
	UPROPERTY()
	TObjectPtr<UPipelineModelPackage> Package;

	/** Private struct in PipelineBase.cpp; shared with in-flight calls. */
	TSharedPtr<struct FPipelineRuntime, ESPMode::ThreadSafe> Runtime;

	/** Pending RunStages calls; shared with in-flight calls (also those started before initialization). */
	TSharedPtr<FThreadSafeCounter, ESPMode::ThreadSafe> PendingCalls = MakeShared<FThreadSafeCounter, ESPMode::ThreadSafe>();

	/** InitializeFromPackage succeeded (set after OnInitialize). */
	bool bReady = false;
};
