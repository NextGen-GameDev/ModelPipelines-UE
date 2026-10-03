// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "Engine/DataAsset.h"
#include "PipelineTypes.h"
#include "PipelineValidation.h"
#include "PipelineModelPackage.generated.h"

class UNNEModelData;

/** Task a model package serves. Values are fixed (docs/contracts.md C12). */
UENUM(BlueprintType)
enum class EPipelineTask : uint8
{
	None = 0,
	TextEmbedding = 1,
	TextClassification = 2,
	ImageClassification = 3,
	ObjectDetection = 4
};

/** How token embeddings are pooled into one sentence embedding. */
UENUM(BlueprintType)
enum class EPipelinePooling : uint8
{
	Mean = 0,
	CLS = 1,
	/** The model already outputs [batch, hidden]. */
	None = 2
};

/** How an input image is brought to the network input size. */
UENUM(BlueprintType)
enum class EPipelineImageResize : uint8
{
	Stretch = 0,
	ShortestEdgeCenterCrop = 1,
	/** Resize to exactly ResizeWidth x ResizeHeight (aspect not kept), then center-crop ImageWidth x ImageHeight (DeiT style). */
	StretchCenterCrop = 2
};

/** Activation applied to classification and detection logits. */
UENUM(BlueprintType)
enum class EPipelineScoreActivation : uint8
{
	Softmax = 0,
	Sigmoid = 1
};

/** Image resample filter. Values equal PIL's Image.Resampling (docs/contracts.md C26). */
UENUM(BlueprintType)
enum class EPipelineImageResample : uint8
{
	Nearest = 0,
	Lanczos = 1,
	Bilinear = 2,
	Bicubic = 3,
	Box = 4,
	Hamming = 5
};

/** Enumerator name of a resample filter ("Bicubic"); "Unknown" for a value outside the enum. */
MODELPIPELINESCORE_API const TCHAR* LexToString(EPipelineImageResample Value);

/** Enumerator name of a task: "None", "TextEmbedding", ...; "Unknown" for a value outside the enum. */
MODELPIPELINESCORE_API const TCHAR* LexToString(EPipelineTask Task);

/** Case-insensitive enumerator name (no trimming). Returns false and leaves Out unchanged on no match or null/empty. */
MODELPIPELINESCORE_API bool LexTryParseString(EPipelineTask& Out, const TCHAR* Str);

/** True for TextEmbedding and TextClassification. */
MODELPIPELINESCORE_API bool IsTextTask(EPipelineTask Task);

/** True for ImageClassification and ObjectDetection. */
MODELPIPELINESCORE_API bool IsImageTask(EPipelineTask Task);

/** True for every task except TextEmbedding and None. */
MODELPIPELINESCORE_API bool TaskUsesLabels(EPipelineTask Task);

/** Pre-processing settings of a model package. */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelinePreprocessSettings
{
	GENERATED_BODY()

	/** Text tasks: truncation length including special tokens. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Text", meta = (ClampMin = 1, ClampMax = 65536))
	int32 MaxTokens = 512;

	/** Text tasks: add the tokenizer's special tokens. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Text")
	bool bAddSpecialTokens = true;

	/** Text embedding: pooling of token embeddings. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Text")
	EPipelinePooling Pooling = EPipelinePooling::Mean;

	/** Text embedding: L2-normalize the result. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Text")
	bool bNormalizeEmbeddings = false;

	/** Image tasks: network input width. Pixel (0..255) * RescaleFactor, then (v - Mean) / Std per RGB channel, NCHW. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Image", meta = (ClampMin = 1, ClampMax = 8192))
	int32 ImageWidth = 224;

	/** Image tasks: network input height. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Image", meta = (ClampMin = 1, ClampMax = 8192))
	int32 ImageHeight = 224;

	/** Image tasks: resize mode. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Image")
	EPipelineImageResize ResizeMode = EPipelineImageResize::Stretch;

	/** ShortestEdgeCenterCrop: resize so the shortest edge is this, keep aspect, then center-crop ImageWidth x ImageHeight. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Image")
	int32 ResizeShortestEdge = 0;

	/** StretchCenterCrop: width to resize to before the center crop. 0 in the other modes. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Image")
	int32 ResizeWidth = 0;

	/** StretchCenterCrop: height to resize to before the center crop. 0 in the other modes. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Image")
	int32 ResizeHeight = 0;

	/** Image tasks: resample filter used when resizing. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Image")
	EPipelineImageResample Resample = EPipelineImageResample::Bilinear;

	/** Image tasks: factor applied to 0..255 pixel values. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Image")
	float RescaleFactor = 1.f / 255.f;

	/** Image tasks: per-channel mean (R,G,B). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Image")
	FVector ImageMean = FVector(0.485, 0.456, 0.406);

	/** Image tasks: per-channel standard deviation (R,G,B). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Image")
	FVector ImageStd = FVector(0.229, 0.224, 0.225);
};

/** Post-processing settings of a model package. */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelinePostprocessSettings
{
	GENERATED_BODY()

	/** Classification and detection scores. Detection with Softmax: the last logit column is the no-object class. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Scores")
	EPipelineScoreActivation ScoreActivation = EPipelineScoreActivation::Softmax;

	/** ObjectDetection: class-wise non-maximum suppression; a box is dropped when a kept box of the same class overlaps it
	    with IoU > NmsIouThreshold. >= 1 disables NMS. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Detection", meta = (ClampMin = 0, ClampMax = 1))
	float NmsIouThreshold = 0.7f;

	/** ObjectDetection: maximum number of detections returned (1..10000). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Detection", meta = (ClampMin = 1, ClampMax = 10000))
	int32 MaxDetections = 100;
};

/** License record of the packaged model. */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelineModelLicense
{
	GENERATED_BODY()

	/** SPDX identifier, e.g. "Apache-2.0". */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "License")
	FString SpdxId;

	/** Source of the model, e.g. https://huggingface.co/<repo>/tree/<rev>. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "License")
	FString SourceUrl;

	/** Attribution text. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "License")
	FString Attribution;
};

/** Data asset bundling an imported ONNX model with everything a task pipeline needs. */
UCLASS(BlueprintType)
class MODELPIPELINESCORE_API UPipelineModelPackage : public UPrimaryDataAsset
{
	GENERATED_BODY()

public:
	/** Task the package serves. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pipeline")
	EPipelineTask Task = EPipelineTask::None;

	/** The imported ONNX model. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pipeline")
	TObjectPtr<UNNEModelData> Model;

	/** Full tokenizer.json text; not shown in the details panel (megabytes). */
	UPROPERTY(BlueprintReadOnly, Category = "Pipeline")
	FString TokenizerJson;

	/** Class labels; index = class id. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pipeline")
	TArray<FString> Labels;

	/** Pre-processing settings. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pipeline")
	FPipelinePreprocessSettings Preprocess;

	/** Post-processing settings. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pipeline")
	FPipelinePostprocessSettings Postprocess;

	/** License of the model. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pipeline")
	FPipelineModelLicense License;

	/** Device the pipeline prefers. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Pipeline")
	EPipelineDevice PreferredDevice = EPipelineDevice::Auto;

	/**
	 * Settings only: loads neither model nor tokenizer. Appends messages (codes 2..14) to OutReport.
	 * Returns true when this call added no Error. Does not modify the package.
	 */
	UFUNCTION(BlueprintCallable, BlueprintPure = false, Category = "Pipelines")
	bool ValidateSettings(FPipelineValidationReport& OutReport) const;
};
