// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Tasks/Task.h"
#include "PipelineTypes.h"
#include "PipelineImage.h"

class UTexture;

/** Receives the pixels of a texture read (docs/contracts.md C28). Runs on a background thread. */
using FPipelineImageReadCallback = TUniqueFunction<void(const FPipelineResult& Result, FPipelineImage&& Image)>;

namespace UE::ModelPipelines::Vision
{
	/**
	 * Game thread. Copies the top resident mip of a UTexture2D or UTextureRenderTarget2D into an FPipelineImage without
	 * blocking the game thread. OnRead runs exactly once, on a background thread, never inside this call (also on failure,
	 * then with an empty image). Returns a task that completes after OnRead returned (usable as FPipelineStages::Prerequisite).
	 */
	MODELPIPELINESVISION_API UE::Tasks::FTask ReadTexturePixelsAsync(UTexture* Texture, FPipelineImageReadCallback OnRead);

	struct FPipelineImageReadStats
	{
		int64 NumStarted = 0, NumSucceeded = 0, NumFailed = 0;
		/** Per read: the ReadTexturePixelsAsync call + every game-thread tick spent on it. */
		double MaxGameThreadMsPerRead = 0.0;
		double LastGameThreadMsPerRead = 0.0;
		/** Call -> pixels on the CPU. */
		double MaxReadbackWaitMs = 0.0;
	};

	/** Any thread. */
	MODELPIPELINESVISION_API FPipelineImageReadStats GetImageReadStats();
	MODELPIPELINESVISION_API void ResetImageReadStats();
}
