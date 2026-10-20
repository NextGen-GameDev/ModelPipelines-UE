// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Tasks/Task.h"
#include "Templates/SharedPointer.h"
#include "PipelineBase.h"
#include "PipelineImage.h"
#include "PipelineModelPackage.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"

class UTexture;

/**
 * Per-call code shared by the vision pipelines (docs/contracts.md C29). Module-private.
 *
 * Threading: a call's FVisionCallInput is filled on the game thread (settings, image) or by the texture read's OnRead
 * (background; it completes before the stages' Prerequisite does), then read and written only by Preprocess
 * (background), then read by Postprocess and Complete. The stage runner orders those steps, so no lock is needed.
 */
namespace UE::ModelPipelines::Vision::Private
{
	struct FVisionCallInput
	{
		/** Settings copy taken when the call was issued. */
		FPipelinePreprocessSettings Settings;
		/** The source image (FPipelineImage input, or the read texture). Released by Preprocess once the tensor is built. */
		FPipelineImage Image;
		/** Result of the texture read; Ok for FPipelineImage input. */
		FPipelineResult ReadResult = FPipelineResult::Ok();
		/** Texture input: call -> pixels on the CPU. 0 for FPipelineImage input. */
		double ReadbackWaitMs = 0.0;
		/** Filled by Preprocess: where the network input sits in the source image. */
		FPipelineImageGeometry Geometry;
	};
	using FVisionCallInputRef = TSharedRef<FVisionCallInput, ESPMode::ThreadSafe>;

	/** Immutable label list shared between a pipeline and its in-flight calls. Null reads as empty. */
	using FSharedLabels = TSharedPtr<const TArray<FString>, ESPMode::ThreadSafe>;

	/** Game thread (OnInitialize). One copy of Labels, shared from then on. */
	FSharedLabels MakeSharedLabels(const TArray<FString>& Labels);

	/** *Labels, or an empty array for null. Any thread. */
	const TArray<FString>& LabelsOrEmpty(const FSharedLabels& Labels);

	/** New call input with the settings copy. Any thread. */
	FVisionCallInputRef MakeVisionCallInput(const FPipelinePreprocessSettings& Settings);

	/**
	 * Game thread. Starts the texture read; OnRead stores the pixels (or the failure) and the wait time in Input.
	 * Returns the read task, to be used as FPipelineStages::Prerequisite.
	 */
	UE::Tasks::FTask StartTextureRead(UTexture* Texture, const FVisionCallInputRef& Input);

	/**
	 * Background (the Preprocess stage). Fails with the read result, InvalidArgument for an invalid image, or the
	 * PreprocessImage failure; otherwise fills OutInputs in Info.InputDescs order: pixel_values float32
	 * [1, 3, ImageHeight, ImageWidth] and pixel_mask (if bound) all 1 in its type. Each input must be IsCompatible with its
	 * desc, else ShapeMismatch. Fills Input.Geometry and releases Input.Image.
	 */
	FPipelineResult BuildVisionInputs(FVisionCallInput& Input, const FPipelineModelInfo& Info, TArray<FPipelineTensor>& OutInputs);

	/** Result with the readback wait added to PreprocessMs (C29: PreprocessMs = readback wait + Preprocess stage). */
	FPipelineResult WithReadbackWait(const FPipelineResult& Result, const FVisionCallInput& Input);
}
