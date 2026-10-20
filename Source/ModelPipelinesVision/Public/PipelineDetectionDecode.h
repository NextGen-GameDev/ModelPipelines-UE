// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Containers/Array.h"
#include "Containers/ArrayView.h"
#include "PipelineImage.h"
#include "PipelineModelPackage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"

namespace UE::ModelPipelines::Vision
{
	/** Parameters of DecodeDetections (docs/contracts.md C29). */
	struct FPipelineDetectionDecodeParams
	{
		/** Sigmoid: one score per (query, class). Softmax: per-query softmax over C' = C + 1 columns, the last = no-object. */
		EPipelineScoreActivation Activation = EPipelineScoreActivation::Sigmoid;
		/** Keep candidates with score > it (strict). NaN -> InvalidArgument; otherwise clamped to [0, 1]. */
		float ScoreThreshold = 0.5f;
		/** Class-wise NMS: drop a box whose IoU with a kept box of the same class is > it. >= 1 disables NMS. */
		float NmsIouThreshold = 0.7f;
		/** At most this many detections (>= 1, else InvalidArgument). */
		int32 MaxDetections = 100;
		/** CropWidth/Height and ResizedWidth/Height all > 0 -> boxes are mapped from the network input to the source image;
		    all 0 -> boxes are taken as already normalized to the source image; any other combination -> InvalidArgument. */
		FPipelineImageGeometry Geometry;
	};

	/**
	 * Decodes a DETR-family detector's raw outputs. Logits [NumQueries, NumLogitClasses] and Boxes [NumQueries, 4]
	 * (cx, cy, w, h normalized to the network input), row-major. Labels.Num() must be NumLogitClasses (Sigmoid) or
	 * NumLogitClasses - 1 (Softmax), else InferenceFailed. Bad sizes or a NaN/Inf value -> InferenceFailed.
	 * Out: detections in NMS order (score descending), boxes clamped to [0, 1]; emptied on failure.
	 * Pure function, any thread.
	 */
	MODELPIPELINESVISION_API FPipelineResult DecodeDetections(TConstArrayView<float> Logits, int32 NumQueries, int32 NumLogitClasses,
		TConstArrayView<float> Boxes, const TArray<FString>& Labels, const FPipelineDetectionDecodeParams& Params, TArray<FPipelineDetection>& Out);
}
