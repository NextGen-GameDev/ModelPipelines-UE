// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Templates/Function.h"
#include "UObject/ObjectMacros.h"
#include "PipelineBase.h"
#include "PipelineImage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "ObjectDetectionPipeline.generated.h"

class UTexture;

/** Game thread, exactly once per DetectAsync / DetectImageAsync call. Detections is empty on failure. */
using FPipelineDetectionCallback = TUniqueFunction<void(const FPipelineResult& Result, const TArray<FPipelineDetection>& Detections)>;

/**
 * Object detection pipeline (docs/contracts.md C29), DETR family (RT-DETR, DETR, YOLOS, ...): resize and normalize one
 * image per the package's settings, run the model, decode logits + pred_boxes with the package's score activation,
 * class-wise NMS and MaxDetections. Boxes are normalized to the input image.
 */
UCLASS(BlueprintType)
class MODELPIPELINESVISION_API UObjectDetectionPipeline : public UPipelineBase
{
	GENERATED_BODY()

public:
	/** PipelineTask = ObjectDetection. */
	UObjectDetectionPipeline();

	/** Game thread. Image: UTexture2D or UTextureRenderTarget2D (read back from the GPU without blocking the game thread).
	    ScoreThreshold: keep detections with score > it; NaN -> InvalidArgument; clamped to [0, 1].
	    Callback: game thread, exactly once, never inside the call. Null callback -> Warning, nothing runs. */
	void DetectAsync(UTexture* Image, float ScoreThreshold, FPipelineDetectionCallback OnComplete);

	/** Game thread. Same with a CPU image. */
	void DetectImageAsync(const FPipelineImage& Image, float ScoreThreshold, FPipelineDetectionCallback OnComplete);

	/** Tests and tools: blocking, any thread except the executor worker; the caller keeps the pipeline alive.
	    OutDetections is emptied on failure. */
	FPipelineResult DetectImageBlocking(const FPipelineImage& Image, float ScoreThreshold, TArray<FPipelineDetection>& OutDetections);
};
