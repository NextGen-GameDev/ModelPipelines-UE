// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "PipelineTypes.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"
#include "PipelineModelFolder.generated.h"

/** Everything a model package needs, read from a Hugging Face-style model folder (docs/contracts.md C14). */
USTRUCT(BlueprintType)
struct MODELPIPELINESEDITOR_API FPipelineModelFolderInfo
{
	GENERATED_BODY()

	/** Absolute folder path, '/' separators, no trailing '/'. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString FolderPath;

	/** Absolute path of the .onnx file ("" when it is missing). */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString ModelFile;

	/** Detected (or overridden) task. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	EPipelineTask Task = EPipelineTask::None;

	/** Where the task came from: "options" | "modelpipelines.json" | "modules.json" | "architectures". */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString TaskSource;

	/** tokenizer.json text (UTF-8 decoded); text tasks only, "" otherwise. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString TokenizerJson;

	/** Class labels from config.json id2label; index = class id. Empty for TextEmbedding. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	TArray<FString> Labels;

	/** Pre-processing settings derived from the folder's config files. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FPipelinePreprocessSettings Preprocess;

	/** Post-processing settings (score activation). */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FPipelinePostprocessSettings Postprocess;

	/** License record from the sidecar and/or the README front matter. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FPipelineModelLicense License;

	/** Device preference from the sidecar; Auto by default. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	EPipelineDevice PreferredDevice = EPipelineDevice::Auto;

	/** config.json top-level model_type, "" if absent. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString ModelType;

	/** config.json architectures[0], "" if absent. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString Architecture;

	/** Absolute path of decoder_joint.onnx; "" unless SpeechToText (docs/contracts.md C78). */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString DecoderModelFile;

	/** Speech settings from processor_config.json + config.json; defaults unless SpeechToText (C78). */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FPipelineSpeechSettings Speech;
};

/**
 * Reads a model folder's JSON/text files and fills OutInfo (reset first); appends findings (codes 40..51) to InOutReport.
 * Does not read the model bytes and creates no UObject; safe on any thread.
 * TaskOverride None = auto-detect. Returns true when this call added no Error.
 */
MODELPIPELINESEDITOR_API bool ParsePipelineModelFolder(const FString& Folder, EPipelineTask TaskOverride,
	FPipelineModelFolderInfo& OutInfo, FPipelineValidationReport& InOutReport);
