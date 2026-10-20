// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Containers/ArrayView.h"
#include "PipelineTensor.h"
#include "PipelineValidation.h"
#include "PipelineModelPackage.h"

/** Indices into a model's input / output desc arrays that a task pipeline binds; -1 = absent. */
struct FPipelineTaskBindings
{
	/** Text tasks: input `input_ids`. */
	int32 InputIds = -1;

	/** Text tasks: input `attention_mask`. */
	int32 AttentionMask = -1;

	/** Text tasks: optional input `token_type_ids`. */
	int32 TokenTypeIds = -1;

	/** Image tasks: input `pixel_values`. */
	int32 PixelValues = -1;

	/** ObjectDetection: optional input `pixel_mask`. */
	int32 PixelMask = -1;

	/** Embeddings / logits output. */
	int32 MainOutput = -1;

	/** ObjectDetection: `pred_boxes` output. */
	int32 BoxesOutput = -1;
};

/**
 * Checks a model's input/output descriptions against a task and its settings (docs/contracts.md C13).
 * Pure function, any thread. Appends messages (codes 20..31, and NoTask) to InOutReport and fills OutBindings
 * (reset to all -1 at entry) with everything it could resolve. Returns true when this call added no Error.
 */
MODELPIPELINESCORE_API bool ResolvePipelineTaskIO(EPipelineTask Task, const FPipelinePreprocessSettings& Settings,
	const FPipelinePostprocessSettings& Postprocess, int32 NumLabels, TConstArrayView<FPipelineTensorDesc> Inputs,
	TConstArrayView<FPipelineTensorDesc> Outputs, FPipelineTaskBindings& OutBindings, FPipelineValidationReport& InOutReport);
