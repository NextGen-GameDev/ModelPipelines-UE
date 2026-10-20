// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"

class UNNEModelData;

/**
 * A model prepared by an IPipelineBackend. Holds no UObject reference.
 * Inference goes through FPipelineExecutor, which serializes calls per model and runs them on a worker thread.
 */
class MODELPIPELINESCORE_API IPipelineModel
{
public:
	virtual ~IPipelineModel() = default;

	/** Creates the runtime instance if not yet created. Blocking; call from a worker thread (the executor does).
	    Idempotent: later calls return the first call's result. Touches no UObject. */
	virtual FPipelineResult Initialize() = 0;

	/** Initialize succeeded. */
	virtual bool IsInitialized() const = 0;

	/** CPU or GPU once initialized; Auto before. */
	virtual EPipelineDevice GetDevice() const = 0;

	/** Empty before Initialize succeeded. */
	virtual TArray<FPipelineTensorDesc> GetInputDescs() const = 0;
	virtual TArray<FPipelineTensorDesc> GetOutputDescs() const = 0;

	/** Blocking inference; calls Initialize() first if needed. NOT thread-safe per model: the caller serializes
	    (FPipelineExecutor does). Validates everything before touching NNE. On success OutOutputs has one tensor
	    per model output, in model order, with the actual output shapes. Result.Device and Result.InferenceMs are set.
	    On failure OutOutputs is empty. */
	virtual FPipelineResult Run(TConstArrayView<FPipelineTensor> Inputs, TArray<FPipelineTensor>& OutOutputs) = 0;
};

/** Creates IPipelineModels from model data. */
class MODELPIPELINESCORE_API IPipelineBackend
{
public:
	virtual ~IPipelineBackend() = default;

	virtual FName GetName() const = 0;

	/** Game thread. true if a runtime for the device is registered (GPU: the GPU interface exists; hardware is
	    checked when the model is prepared). Auto -> true if CPU or GPU is available. */
	virtual bool IsDeviceAvailable(EPipelineDevice Device) const = 0;

	/** Game thread only (touches the UObject). Prepares the model for the allowed devices; creates no runtime instance.
	    Returns nullptr and sets OutResult (bSuccess false) on failure; on success OutResult = Ok().
	    The returned model holds no UObject reference; ModelData may be garbage-collected afterwards. */
	virtual TSharedPtr<IPipelineModel> PrepareModel(UNNEModelData* ModelData, EPipelineDevice Device, FPipelineResult& OutResult) = 0;
};
