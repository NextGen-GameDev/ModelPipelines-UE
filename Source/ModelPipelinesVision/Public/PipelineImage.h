// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "PipelineModelPackage.h"
#include "PipelineTypes.h"
#include "PipelineImage.generated.h"

/** 8-bit RGB image handed to the vision pipelines (docs/contracts.md C27). */
USTRUCT(BlueprintType)
struct MODELPIPELINESVISION_API FPipelineImage
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	int32 Width = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	int32 Height = 0;

	/** Row-major, top row first, Width*Height entries. 8-bit values as stored in an image file (sRGB-encoded); A is ignored. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Pipelines")
	TArray<FColor> Pixels;

	static constexpr int32 MaxSide = 16384;

	/** Width and Height in 1..MaxSide and Pixels.Num() == Width*Height. OutReason names the failed rule. */
	bool IsValid(FString* OutReason = nullptr) const;
};

/** Where the network input sits in the source image (for mapping detection boxes back). */
struct FPipelineImageGeometry
{
	/** Input image. */
	int32 SourceWidth = 0, SourceHeight = 0;
	/** After the resize step. */
	int32 ResizedWidth = 0, ResizedHeight = 0;
	/** Crop origin in the resized image (0 for Stretch). */
	int32 CropLeft = 0, CropTop = 0;
	/** == ImageWidth / ImageHeight. */
	int32 CropWidth = 0, CropHeight = 0;
};

namespace UE::ModelPipelines::Vision
{
	/**
	 * Pillow 12 Image.resize((OutWidth, OutHeight), resample=Filter) on an 8-bit RGB image (box = whole image, reducing_gap
	 * None), bit-exact. Out size equal to In size -> copy. A of the output = 255.
	 * false (Out = {}) if In is invalid or OutWidth/OutHeight are outside 1..65536 or OutWidth*OutHeight > 2^26.
	 * Pure function, any thread.
	 */
	MODELPIPELINESVISION_API bool ResizeImage(const FPipelineImage& In, int32 OutWidth, int32 OutHeight, EPipelineImageResample Filter, FPipelineImage& Out);

	/**
	 * Resize per Settings.ResizeMode, centre crop, rescale, normalize. OutCHW = 3*ImageHeight*ImageWidth floats, planes R, G, B,
	 * row-major. OutCropped (optional) receives the resized+cropped 8-bit image; OutGeometry (optional) the geometry.
	 * Invalid image -> InvalidArgument. Settings not usable for an image task, or a resize size outside the ResizeImage
	 * limits -> PreprocessFailed naming the value. Pure function, any thread.
	 */
	MODELPIPELINESVISION_API FPipelineResult PreprocessImage(const FPipelineImage& In, const FPipelinePreprocessSettings& Settings,
		TArray<float>& OutCHW, FPipelineImage* OutCropped = nullptr, FPipelineImageGeometry* OutGeometry = nullptr);
}
