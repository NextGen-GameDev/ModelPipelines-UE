// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

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

	/** SpeechToText encoder: input `input_features`. */
	int32 InputFeatures = -1;

	/** Embeddings / logits output; SpeechToText: `encoder_projected`. */
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

/** SpeechToText (C72c): ResolvePipelineTaskIO with the package's speech settings (the encoder graph checks NumMelBins). */
MODELPIPELINESCORE_API bool ResolvePipelineTaskIO(EPipelineTask Task, const FPipelinePreprocessSettings& Settings,
	const FPipelinePostprocessSettings& Postprocess, const FPipelineSpeechSettings& Speech, int32 NumLabels,
	TConstArrayView<FPipelineTensorDesc> Inputs, TConstArrayView<FPipelineTensorDesc> Outputs,
	FPipelineTaskBindings& OutBindings, FPipelineValidationReport& InOutReport);

/** Indices into the decoder_joint graph's input / output descs; -1 = absent. */
struct FPipelineSpeechDecoderBindings
{
	int32 EncoderFrame = -1;
	int32 Token = -1;
	int32 StateH = -1;
	int32 StateC = -1;
	int32 TokenLogits = -1;
	int32 DurationLogits = -1;
	int32 StateHOut = -1;
	int32 StateCOut = -1;
};

/**
 * Pure, any thread. Exact names only (C70); codes 20-28. Checks: encoder_frame float32 [1|-1, EncoderDim|-1];
 * token int64 [1|-1]; state_h/state_c float32 rank 3 with equal dims; token_logits float32 [1|-1, BlankTokenId+1|-1];
 * duration_logits float32 [1|-1, Durations.Num()|-1]; state_*_out same rank/dims as state_*. EncoderDim <= 0 = don't check.
 * Returns true when this call added no Error. OutBindings reset to -1 at entry.
 */
MODELPIPELINESCORE_API bool ResolveSpeechDecoderIO(const FPipelineSpeechSettings& Speech, int32 EncoderDim,
	TConstArrayView<FPipelineTensorDesc> Inputs, TConstArrayView<FPipelineTensorDesc> Outputs,
	FPipelineSpeechDecoderBindings& OutBindings, FPipelineValidationReport& InOutReport);
