// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Templates/Function.h"
#include "PipelineBase.h"
#include "PipelineTypes.h"

namespace UE::ModelPipelines::Text
{
	/** Most texts one inference sees (docs/contracts.md C43). Larger calls run as sequential chunks of this size. */
	static constexpr int32 MaxTextsPerInference = 16;

	/** One chunk of a text call: its stages (Complete unused, the runner sets it) and a commit run on success. */
	struct FTextChunk
	{
		FPipelineStages Stages;
		/** Game thread (async) or the calling thread (blocking), after the chunk succeeded. Appends the chunk's outputs. */
		TUniqueFunction<void()> Commit;
	};

	/** Number of chunks for a call: ceil(N / MaxTextsPerInference); an invalid count (0 or > MaxTextsPerCall) is one chunk. */
	int32 NumTextChunks(int32 NumTexts);

	/**
	 * Issues chunk 0 now and chunk k+1 from inside chunk k's Complete (game thread, after the base counted chunk k done, before
	 * Complete returns), so the inferences run one after another and GetNumPendingCalls() reads exactly 1 for the whole call.
	 * MakeChunk builds the chunk for texts [First, First + Num); Issue hands its stages to RunStages. Both are kept until the
	 * final chunk and released on the game thread before OnDone runs (the caller pins the pipeline in them for multi-chunk calls).
	 * OnDone runs exactly once, on the game thread, after the last chunk: the first chunk error (no further chunk issued), or
	 * Ok() on the last chunk's device; timings are sums over the chunks that ran.
	 */
	void RunTextChunksAsync(int32 NumTexts, TUniqueFunction<FTextChunk(int32 First, int32 Num)> MakeChunk,
		TUniqueFunction<void(FPipelineStages&&)> Issue, TUniqueFunction<void(const FPipelineResult&)> OnDone);

	/** Calling thread. Runs the chunks one after another with RunBlocking; stops at the first error. Timings are sums. */
	FPipelineResult RunTextChunksBlocking(int32 NumTexts, TFunctionRef<FTextChunk(int32 First, int32 Num)> MakeChunk,
		TFunctionRef<FPipelineResult(FPipelineStages&&)> RunBlocking);
}
