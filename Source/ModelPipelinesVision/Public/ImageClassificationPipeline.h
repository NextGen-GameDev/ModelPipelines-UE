// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Templates/Function.h"
#include "UObject/ObjectMacros.h"
#include "PipelineBase.h"
#include "PipelineImage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "ImageClassificationPipeline.generated.h"

class UTexture;

/** Game thread, exactly once per ClassifyAsync / ClassifyImageAsync call. Classification is default-constructed on failure. */
using FPipelineImageClassificationCallback = TUniqueFunction<void(const FPipelineResult& Result, const FPipelineClassification& Classification)>;

/**
 * Image classification pipeline (docs/contracts.md C29): resize, crop and normalize one image per the package's settings,
 * run the model, apply the package's score activation and pick the top-k labels.
 */
UCLASS(BlueprintType)
class MODELPIPELINESVISION_API UImageClassificationPipeline : public UPipelineBase
{
	GENERATED_BODY()

public:
	/** PipelineTask = ImageClassification. */
	UImageClassificationPipeline();

	/** Game thread. Image: UTexture2D or UTextureRenderTarget2D (read back from the GPU without blocking the game thread).
	    TopK < 1 -> 1; > number of labels -> number of labels. Callback: game thread, exactly once, never inside the call.
	    Null callback -> Warning, nothing runs. */
	void ClassifyAsync(UTexture* Image, int32 TopK, FPipelineImageClassificationCallback OnComplete);

	/** Game thread. Same with a CPU image. */
	void ClassifyImageAsync(const FPipelineImage& Image, int32 TopK, FPipelineImageClassificationCallback OnComplete);

	/** Tests and tools: blocking, any thread except the executor worker; the caller keeps the pipeline alive.
	    OutResult is reset on failure. */
	FPipelineResult ClassifyImageBlocking(const FPipelineImage& Image, int32 TopK, FPipelineClassification& OutResult);

protected:
	/** Game thread. Builds SharedLabels from the labels copy taken at initialization. */
	virtual FPipelineResult OnInitialize(UPipelineModelPackage* InPackage) override;

private:
	/** Immutable labels shared by reference with in-flight calls (no per-call copy). Written only on the game thread by
	    OnInitialize; calls capture a reference and read it on background threads. */
	TSharedPtr<const TArray<FString>, ESPMode::ThreadSafe> SharedLabels;
};
