// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "PipelineValidation.h"
#include "PipelineModelValidator.generated.h"

class UPipelineModelPackage;
class IPipelineBackend;

/** Options of ValidatePipelineModelPackage (docs/contracts.md C15). */
USTRUCT(BlueprintType)
struct MODELPIPELINESEDITOR_API FPipelineValidateOptions
{
	GENERATED_BODY()

	/** Run one small inference on each created device and check the output shapes and values. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	bool bRunSmokeTest = true;

	/** Also create (and smoke-run) the model on the GPU runtime. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	bool bCheckGpu = true;
};

/**
 * Proves a model package works: settings, tokenizer, model creation on CPU (and GPU), task I/O and a smoke run per
 * device. Everything found goes into the returned report; never crashes on a bad package and never modifies it.
 * Editor, game thread only (touches UObjects, calls PrepareModel). Blocking: meant for import and tools.
 * Uses FModelPipelinesCoreModule::Get().GetDefaultBackend().
 */
MODELPIPELINESEDITOR_API FPipelineValidationReport ValidatePipelineModelPackage(const UPipelineModelPackage* Package,
	const FPipelineValidateOptions& Options = FPipelineValidateOptions());

/**
 * Same as above with an explicit backend (tests pass an FPipelineNNEBackend with a bogus runtime name).
 * Smoke runs still go through FPipelineExecutor::Get(). Editor, game thread only. Blocking.
 */
MODELPIPELINESEDITOR_API FPipelineValidationReport ValidatePipelineModelPackage(const UPipelineModelPackage* Package,
	const FPipelineValidateOptions& Options, IPipelineBackend& Backend);
