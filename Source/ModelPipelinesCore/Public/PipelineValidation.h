// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "PipelineTypes.h"
#include "PipelineValidation.generated.h"

/** Severity of a validation message. Values are fixed (docs/contracts.md C12). */
UENUM(BlueprintType)
enum class EPipelineValidationSeverity : uint8
{
	Info = 0,
	Warning = 1,
	Error = 2
};

/** Code of a validation message. Values are fixed (docs/contracts.md C12). */
UENUM(BlueprintType)
enum class EPipelineValidationCode : uint8
{
	None = 0,
	// package settings (UPipelineModelPackage::ValidateSettings, WP2.2)
	/** Error: null package passed to a validator. */
	NoPackage = 1,
	/** Error: Task == None. */
	NoTask = 2,
	/** Error: Model == nullptr. */
	NoModel = 3,
	/** Error: text task and TokenizerJson empty. */
	NoTokenizer = 4,
	/** Error: text task and MaxTokens outside 1..65536. */
	InvalidMaxTokens = 5,
	/** Error: TextClassification/ImageClassification/ObjectDetection and Labels empty. */
	NoLabels = 6,
	/** Warning: a label string is empty after trimming (one message per package, names the first index). */
	EmptyLabel = 7,
	/** Error: image task and ImageWidth or ImageHeight outside 1..8192. */
	InvalidImageSize = 8,
	/** Error: image task and RescaleFactor or any ImageStd component is <= 0, NaN or Inf. */
	InvalidNormalization = 9,
	/** Error: image task, and either ShortestEdgeCenterCrop with ResizeShortestEdge < max(ImageWidth, ImageHeight), or StretchCenterCrop with ResizeWidth/ResizeHeight outside 1..8192 or smaller than ImageWidth/ImageHeight. */
	InvalidResize = 10,
	/** Warning: License.SpdxId empty (after trimming). */
	LicenseSpdxMissing = 11,
	/** Warning: License.SourceUrl empty (after trimming). */
	LicenseSourceMissing = 12,
	/** Warning: License.Attribution empty (after trimming). */
	LicenseAttributionMissing = 13,
	/** Error: ObjectDetection and NmsIouThreshold NaN or < 0, or MaxDetections outside 1..10000. */
	InvalidPostprocess = 14,
	/** Error: both Model and GgufModel are set. */
	BothModelsSet = 15,
	/** Error: TextGeneration with an NNE Model, or a non-generative task with a GgufModel. */
	WrongModelFormat = 16,
	/** Error: GgufModel does not implement IPipelineFileModel, its path is invalid, SizeBytes <= 0 or Sha256 is not 64 hex digits. */
	InvalidGgufModel = 17,
	/** Warning: License.SpdxId is not Apache-2.0, MIT, BSD-2-Clause or BSD-3-Clause (case-insensitive). */
	LicenseNotPermissive = 18,
	// task I/O (ResolvePipelineTaskIO, WP2.3)
	/** Error: an expected model input is absent. */
	MissingInput = 20,
	/** Error: the model has an input the task does not feed. */
	UnexpectedInput = 21,
	/** Error: a model input has the wrong element type. */
	InputTypeMismatch = 22,
	/** Error: a model input has the wrong rank. */
	InputRankMismatch = 23,
	/** Error: a fixed model input dimension does not fit the task. */
	InputDimMismatch = 24,
	/** Error: an expected model output is absent. */
	MissingOutput = 25,
	/** Error: a model output has the wrong element type. */
	OutputTypeMismatch = 26,
	/** Error: a model output has the wrong rank. */
	OutputRankMismatch = 27,
	/** Error: a fixed model output dimension does not fit the task. */
	OutputDimMismatch = 28,
	/** Error: the label count does not match the model's class dimension. */
	LabelCountMismatch = 29,
	/** Warning: expected output name absent, another output of the right type/rank was used. */
	OutputNameFallback = 30,
	/** Info: image model with batch dim fixed at 1. */
	FixedBatch = 31,
	// folder parsing (WP2.4)
	/** Error: the model folder does not exist. */
	FolderNotFound = 40,
	/** Error: neither <folder>/model.onnx nor <folder>/onnx/model.onnx, or the file is empty. */
	MissingModelFile = 41,
	/** Error: a JSON file that is present does not parse, or a known key has the wrong JSON type. */
	InvalidJson = 42,
	/** Error: config.json absent and it is needed (auto task detection, or labels). */
	MissingConfig = 43,
	/** Error: text task and tokenizer.json absent. */
	MissingTokenizer = 44,
	/** Error: no override, no sidecar task, no rule matched. */
	TaskUndetected = 45,
	/** Error: labelled task and config.json has no id2label. */
	MissingLabels = 46,
	/** Error: id2label keys are not exactly the integers 0..N-1. */
	InvalidLabels = 47,
	/** Warning: image task and preprocessor_config.json absent (defaults used). */
	MissingPreprocessorConfig = 48,
	/** Warning: sentence-transformers pooling other than mean/cls (Mean used). */
	UnsupportedPooling = 49,
	/** Warning: preprocessor size/crop combination not representable (Stretch used). */
	UnsupportedResize = 50,
	/** Warning: detection model_type not in the known table. */
	UnverifiedDetectionFamily = 51,
	// validator and importer (WP2.5, WP2.6)
	/** Error: TokenizerJson does not load, or encoding "Hello world" returns no ids. */
	TokenizerInvalid = 60,
	/** Error for CPU; Info for GPU. */
	RuntimeUnavailable = 61,
	/** Error on CPU (unsupported operator, corrupt model); Warning on GPU (runs on CPU instead). */
	ModelCreateFailed = 62,
	/** Error on CPU; on GPU Error if PreferredDevice != CPU, else Warning. */
	SmokeRunFailed = 63,
	/** Error: smoke-run output has the wrong shape or a non-finite value. */
	SmokeOutputInvalid = 64,
	/** Error: DestinationPath/AssetName do not form a valid long package name. */
	InvalidDestination = 65,
	/** Error: target asset exists and bOverwriteExisting is false. */
	AssetExists = 66,
	/** Error: saving the package failed. */
	SaveFailed = 67,
	// GGUF files (editor validator, M8)
	/** Error: the GGUF file does not exist. */
	GgufFileMissing = 68,
	/** Error: the GGUF file size differs from SizeBytes. */
	GgufSizeMismatch = 69,
	/** Error: the GGUF file SHA-256 differs from Sha256. */
	GgufShaMismatch = 70,
	/** Warning: the project does not stage Content/Models as NonUFS. */
	ModelsNotStagedNonUfs = 71,
};

/** Enumerator name of a validation code, e.g. "MissingInput"; "Unknown" for a value outside the enum. */
MODELPIPELINESCORE_API const TCHAR* LexToString(EPipelineValidationCode Code);

/** "Info" | "Warning" | "Error"; "Unknown" for a value outside the enum. */
MODELPIPELINESCORE_API const TCHAR* LexToString(EPipelineValidationSeverity Severity);

/** One finding of a validator. */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelineValidationMessage
{
	GENERATED_BODY()

	/** How serious the finding is; only Error makes a report invalid. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	EPipelineValidationSeverity Severity = EPipelineValidationSeverity::Info;

	/** Machine-readable code of the finding. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	EPipelineValidationCode Code = EPipelineValidationCode::None;

	/** Human-readable text; names the file, tensor or field. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString Message;
};

/** Result of checking one device (filled by the editor validator). */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelineDeviceCheck
{
	GENERATED_BODY()

	/** CPU or GPU. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	EPipelineDevice Device = EPipelineDevice::CPU;

	/** The runtime for this device is registered. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	bool bAvailable = false;

	/** An instance on this device was created. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	bool bCreated = false;

	/** The smoke run succeeded on this device. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	bool bRan = false;

	/** Inference time of the smoke run. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	float InferenceMs = 0.f;

	/** Failure reason; empty when fine. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString Message;
};

/** Ordered list of validation messages plus per-device checks. */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelineValidationReport
{
	GENERATED_BODY()

	/** Messages in the order they were added. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	TArray<FPipelineValidationMessage> Messages;

	/** Per-device results; filled by the editor validator. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	TArray<FPipelineDeviceCheck> DeviceChecks;

	/** Appends one message. */
	void Add(EPipelineValidationSeverity Severity, EPipelineValidationCode Code, const FString& Message);

	/** True when no message has Severity Error. */
	bool IsValid() const;

	/** Number of messages with Severity Error. */
	int32 NumErrors() const;

	/** Number of messages with Severity Warning. */
	int32 NumWarnings() const;

	/** True when any message (any severity) has this code. */
	bool HasCode(EPipelineValidationCode Code) const;

	/** True when a message with Severity Error has this code. */
	bool HasError(EPipelineValidationCode Code) const;

	/** Appends the messages and device checks of Other (safe when Other is this report). */
	void Append(const FPipelineValidationReport& Other);

	/** One line per message, "<Severity> <Code>: <Message>", joined with '\n', no trailing newline; "" when empty. */
	FString ToString() const;
};
