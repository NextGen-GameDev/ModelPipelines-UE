// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "PipelineBackend.h"

/** Runtime names used by FPipelineNNEBackend. */
struct FPipelineNNEBackendConfig
{
	FString CpuRuntimeName = TEXT("NNERuntimeORTCpu");
	FString GpuRuntimeName = TEXT("NNERuntimeORTDml");   // tests set a bogus name to force "GPU unavailable"
};

/** IPipelineBackend over NNE's ONNX Runtime plugin (CPU and DirectML GPU interfaces, CPU fallback). */
class MODELPIPELINESCORE_API FPipelineNNEBackend : public IPipelineBackend
{
public:
	FPipelineNNEBackend();
	explicit FPipelineNNEBackend(const FPipelineNNEBackendConfig& Config);

	/** "NNE". */
	virtual FName GetName() const override;
	virtual bool IsDeviceAvailable(EPipelineDevice Device) const override;
	virtual TSharedPtr<IPipelineModel> PrepareModel(UNNEModelData* ModelData, EPipelineDevice Device, FPipelineResult& OutResult) override;

private:
	FPipelineNNEBackendConfig Config;
};
