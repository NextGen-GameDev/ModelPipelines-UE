// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "PipelineModelSource.h"
#include "PipelineSession.h"

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

	/** Bytes this model holds outside the engine's RHI (e.g. Vulkan buffers, host copies). 0 = unknown/none. Any thread. */
	virtual int64 GetExternalMemoryBytes() const { return 0; }

	/** Any thread; non-blocking. Asks a running Initialize to stop early (it then returns Cancelled). Default: no-op. */
	virtual void CancelInitialize() {}

	/** Creates a session for a long-running job (C61). Default: nullptr + Fail(InvalidArgument, "Model has no sessions").
	    Any thread. Creates no backend state yet (that happens in IPipelineSession::Begin on the worker). */
	virtual TSharedPtr<IPipelineSession> CreateSession(const FPipelineSessionRequest& Request, FPipelineResult& OutResult);
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

	/** Formats this backend can prepare. Default: NNE only. */
	virtual bool SupportsFormat(EPipelineModelFormat Format) const { return Format == EPipelineModelFormat::NNE; }

	/** Game thread. Default: NNE -> PrepareModel(Source.NNEModel, ...); other formats -> nullptr + Fail(RuntimeUnavailable). */
	virtual TSharedPtr<IPipelineModel> PrepareModelFromSource(const FPipelineModelSource& Source, EPipelineDevice Device,
		FPipelineResult& OutResult);
};
