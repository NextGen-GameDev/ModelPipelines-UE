// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineTypes.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"
#include "PipelineModelValidator.h"
#include "PipelineModelImporter.generated.h"

class UNNEModelData;

/** Options of ImportPipelineModelFolder (docs/contracts.md C15). */
USTRUCT(BlueprintType)
struct MODELPIPELINESEDITOR_API FPipelineImportOptions
{
	GENERATED_BODY()

	/** Hugging Face-style model folder on disk (holds model.onnx or onnx/model.onnx, config.json, ...). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	FString SourceFolder;

	/** Long package path the two assets are created in, e.g. /Game/Models. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	FString DestinationPath = TEXT("/Game/Models");

	/** Name of the package asset; the model asset is <AssetName>_Model. "" -> sanitized folder name. Used as given otherwise. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	FString AssetName;

	/** Task override; None = auto-detect from the folder. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	EPipelineTask Task = EPipelineTask::None;

	/** Device preference written to the package; Auto = the folder's sidecar value (or Auto). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	EPipelineDevice PreferredDevice = EPipelineDevice::Auto;

	/** Update existing assets of the right classes in place (same objects). Never deletes or replaces other objects. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	bool bOverwriteExisting = false;

	/** Save both packages to disk after a successful import. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	bool bSave = true;

	/** Run the full validator (model creation, task I/O, smoke runs). Package settings are always checked. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	bool bValidate = true;

	/** Options passed to the validator when bValidate is true. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	FPipelineValidateOptions Validation;
};

/** Result of ImportPipelineModelFolder. */
USTRUCT(BlueprintType)
struct MODELPIPELINESEDITOR_API FPipelineImportResult
{
	GENERATED_BODY()

	/** True when both assets exist, are valid and (with bSave) were saved. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	bool bSuccess = false;

	/** The package asset; null unless the assets exist (set on success and on SaveFailed only). */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	TObjectPtr<UPipelineModelPackage> Package;

	/** The model data asset; null unless the assets exist (set on success and on SaveFailed only). */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	TObjectPtr<UNNEModelData> Model;

	/** Folder parse + settings + validation messages and device checks, then importer messages. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FPipelineValidationReport Report;
};

/**
 * Imports a model folder as two assets: <Dest>/<Name>_Model (UNNEModelData with the ONNX bytes) and <Dest>/<Name>
 * (UPipelineModelPackage referencing it). Both objects are built and validated in the transient package first; a failed
 * import (parse error, invalid destination, existing asset, validation error) leaves nothing in the project.
 * Editor, game thread only (off the game thread it returns one Error and touches nothing). Blocking.
 *
 * Logging: every report message is logged to LogModelPipelinesEditor with the folder name. Error-severity messages and
 * the failure summary are logged at Warning (not Error) on purpose, so that automation tests importing bad folders
 * deliberately do not fail on an unexpected Error log; the success summary is logged at Log.
 */
MODELPIPELINESEDITOR_API FPipelineImportResult ImportPipelineModelFolder(const FPipelineImportOptions& Options);
