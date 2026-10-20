// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "PipelineTextChunks.h"

#include "PipelineTextPreprocess.h"

namespace UE::ModelPipelines::Text::ChunksPrivate
{
	/** One async chunked call. Touched by the issuing thread for chunk 0, then only by chunk Completes (game thread). */
	struct FChunkedCall
	{
		int32 NumTexts = 0;
		int32 NumChunks = 1;
		int32 NextIndex = 0;
		TUniqueFunction<FTextChunk(int32, int32)> MakeChunk;
		TUniqueFunction<void(FPipelineStages&&)> Issue;
		TUniqueFunction<void(const FPipelineResult&)> OnDone;
		double QueueMs = 0.0;
		double PreprocessMs = 0.0;
		double InferenceMs = 0.0;
		double PostprocessMs = 0.0;
		EPipelineDevice Device = EPipelineDevice::Auto;
	};

	using FChunkedCallRef = TSharedRef<FChunkedCall, ESPMode::ThreadSafe>;

	static void AddTimings(double& QueueMs, double& PreprocessMs, double& InferenceMs, double& PostprocessMs,
		EPipelineDevice& Device, const FPipelineResult& R)
	{
		QueueMs += R.QueueMs;
		PreprocessMs += R.PreprocessMs;
		InferenceMs += R.InferenceMs;
		PostprocessMs += R.PostprocessMs;
		if (R.Device != EPipelineDevice::Auto)
		{
			Device = R.Device;
		}
	}

	/** Failure: the failing chunk's result (its code, message and device). Success: Ok() on the last chunk's device. Summed timings. */
	static FPipelineResult MakeFinal(const FPipelineResult& Last, double QueueMs, double PreprocessMs, double InferenceMs,
		double PostprocessMs, EPipelineDevice Device)
	{
		FPipelineResult Final = Last.bSuccess ? FPipelineResult::Ok() : Last;
		if (Last.bSuccess)
		{
			Final.Device = Device;
		}
		Final.QueueMs = (float)QueueMs;
		Final.PreprocessMs = (float)PreprocessMs;
		Final.InferenceMs = (float)InferenceMs;
		Final.PostprocessMs = (float)PostprocessMs;
		return Final;
	}

	static void ChunkRange(int32 NumTexts, int32 NumChunks, int32 Index, int32& OutFirst, int32& OutNum)
	{
		if (NumChunks == 1)
		{
			OutFirst = 0;
			OutNum = NumTexts;
			return;
		}
		OutFirst = Index * MaxTextsPerInference;
		OutNum = FMath::Min(MaxTextsPerInference, NumTexts - OutFirst);
	}

	static void IssueNext(const FChunkedCallRef& Call);

	/** Game thread (the chunk's Complete). Commit, then issue the next chunk or finish the call. */
	static void OnChunkComplete(const FChunkedCallRef& Call, const FPipelineResult& Result, TUniqueFunction<void()>& Commit)
	{
		FChunkedCall& C = *Call;
		AddTimings(C.QueueMs, C.PreprocessMs, C.InferenceMs, C.PostprocessMs, C.Device, Result);
		if (Result.bSuccess)
		{
			if (Commit)
			{
				Commit();
			}
			if (C.NextIndex < C.NumChunks)
			{
				// Issued before this Complete returns: the base already counted this chunk done, the next one counts now.
				IssueNext(Call);
				return;
			}
		}

		// Last chunk or first error: release the pipeline pin (here, on the game thread) and deliver once.
		const FPipelineResult Final = MakeFinal(Result, C.QueueMs, C.PreprocessMs, C.InferenceMs, C.PostprocessMs, C.Device);
		C.MakeChunk.Reset();
		C.Issue.Reset();
		TUniqueFunction<void(const FPipelineResult&)> Done = MoveTemp(C.OnDone);
		C.OnDone.Reset();
		if (Done)
		{
			Done(Final);
		}
	}

	static void IssueNext(const FChunkedCallRef& Call)
	{
		FChunkedCall& C = *Call;
		const int32 Index = C.NextIndex++;
		int32 First = 0;
		int32 Num = 0;
		ChunkRange(C.NumTexts, C.NumChunks, Index, First, Num);
		FTextChunk Chunk = C.MakeChunk(First, Num);
		FPipelineStages Stages = MoveTemp(Chunk.Stages);
		Stages.Complete = [Call, Commit = MoveTemp(Chunk.Commit)](const FPipelineResult& Result) mutable
		{
			OnChunkComplete(Call, Result, Commit);
		};
		C.Issue(MoveTemp(Stages));
	}
}

int32 UE::ModelPipelines::Text::NumTextChunks(int32 NumTexts)
{
	if (NumTexts < 1 || NumTexts > MaxTextsPerCall)
	{
		return 1; // one chunk with all texts: its Preprocess reports the count error as before
	}
	return (NumTexts + MaxTextsPerInference - 1) / MaxTextsPerInference;
}

void UE::ModelPipelines::Text::RunTextChunksAsync(int32 NumTexts, TUniqueFunction<FTextChunk(int32 First, int32 Num)> MakeChunk,
	TUniqueFunction<void(FPipelineStages&&)> Issue, TUniqueFunction<void(const FPipelineResult&)> OnDone)
{
	using namespace UE::ModelPipelines::Text::ChunksPrivate;

	const FChunkedCallRef Call = MakeShared<FChunkedCall, ESPMode::ThreadSafe>();
	Call->NumTexts = NumTexts;
	Call->NumChunks = NumTextChunks(NumTexts);
	Call->MakeChunk = MoveTemp(MakeChunk);
	Call->Issue = MoveTemp(Issue);
	Call->OnDone = MoveTemp(OnDone);
	IssueNext(Call);
}

FPipelineResult UE::ModelPipelines::Text::RunTextChunksBlocking(int32 NumTexts, TFunctionRef<FTextChunk(int32 First, int32 Num)> MakeChunk,
	TFunctionRef<FPipelineResult(FPipelineStages&&)> RunBlocking)
{
	using namespace UE::ModelPipelines::Text::ChunksPrivate;

	const int32 NumChunks = NumTextChunks(NumTexts);
	double QueueMs = 0.0;
	double PreprocessMs = 0.0;
	double InferenceMs = 0.0;
	double PostprocessMs = 0.0;
	EPipelineDevice Device = EPipelineDevice::Auto;
	FPipelineResult Last = FPipelineResult::Ok();
	for (int32 Index = 0; Index < NumChunks; ++Index)
	{
		int32 First = 0;
		int32 Num = 0;
		ChunkRange(NumTexts, NumChunks, Index, First, Num);
		FTextChunk Chunk = MakeChunk(First, Num);
		Last = RunBlocking(MoveTemp(Chunk.Stages));
		AddTimings(QueueMs, PreprocessMs, InferenceMs, PostprocessMs, Device, Last);
		if (!Last.bSuccess)
		{
			break;
		}
		if (Chunk.Commit)
		{
			Chunk.Commit();
		}
	}
	return MakeFinal(Last, QueueMs, PreprocessMs, InferenceMs, PostprocessMs, Device);
}
