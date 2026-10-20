// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "PipelineValidation.h"
#include "PipelineModelValidator.h"
#include "PipelineModelImporter.h"
#include "PipelineEditorLibrary.generated.h"

class UPipelineModelPackage;

/** Editor-only Blueprint/Python access to model import and validation. */
UCLASS()
class MODELPIPELINESEDITOR_API UPipelineEditorLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/** Same as ImportPipelineModelFolder. Editor, game thread only. Blocking. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Editor")
	static FPipelineImportResult ImportModelFolder(const FPipelineImportOptions& Options);

	/** Same as ValidatePipelineModelPackage. Editor, game thread only. Blocking. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Editor")
	static FPipelineValidationReport ValidateModelPackage(UPipelineModelPackage* Package, FPipelineValidateOptions Options);
};
