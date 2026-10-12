// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "PipelineTypes.generated.h"

/** Device a pipeline may run on, or the device that ran it. */
UENUM(BlueprintType)
enum class EPipelineDevice : uint8
{
	Auto = 0,
	CPU = 1,
	GPU = 2
};

/** Error codes carried by FPipelineResult. Values are fixed (docs/contracts.md C7). */
UENUM(BlueprintType)
enum class EPipelineError : uint8
{
	None = 0,
	/** Invalid tensor, empty tensor, null model/data, null callback. */
	InvalidArgument = 1,
	/** A model input/output type other than float32/int64/int32. */
	UnsupportedType = 2,
	/** Input count, type, rank or a fixed dim differs from the model. */
	ShapeMismatch = 3,
	/** No NNE runtime registered for any allowed device. */
	RuntimeUnavailable = 4,
	/** A runtime refused the model data, or instance creation failed on every allowed device. */
	ModelLoadFailed = 5,
	/** RunSync returned Fail. */
	InferenceFailed = 6,
	/** Cancelled while queued, or the executor shut down. */
	Cancelled = 7,
	/** (M2+) Package task does not match the pipeline. */
	WrongTask = 8,
	Internal = 9,
	/** (M3+) Tokenizer or image preprocessing failed (tokenizer did not load, EncodeBatch failed). */
	PreprocessFailed = 10,
	/** (M8+) The prompt (plus at least one new token) does not fit the context length. */
	ContextOverflow = 11,
};

/** Storage format of a package's model (docs/contracts.md C60c). */
UENUM(BlueprintType)
enum class EPipelineModelFormat : uint8
{
	None = 0,
	/** UNNEModelData (ONNX through NNE). */
	NNE = 1,
	/** A loose .gguf file described by a pointer asset implementing IPipelineFileModel. */
	GGUF = 2
};

/** Result returned by every pipeline API. A default-constructed result is a failure with Error None. */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelineResult
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	bool bSuccess = false;

	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	EPipelineError Error = EPipelineError::None;

	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString Message;

	/** Device that ran; Auto = nothing ran. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	EPipelineDevice Device = EPipelineDevice::Auto;

	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	float QueueMs = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	float PreprocessMs = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	float InferenceMs = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	float PostprocessMs = 0.f;

	/** bSuccess true, Error None, Message empty. */
	static FPipelineResult Ok();

	/** bSuccess false, Error and Message set. */
	static FPipelineResult Fail(EPipelineError Error, const FString& Message);
};
