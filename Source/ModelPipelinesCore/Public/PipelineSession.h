// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Templates/Function.h"
#include "PipelineTypes.h"

class IPipelineModel;

/** One unit of progress of a session (one generated token; a transcript segment in M9). Plain data. */
struct MODELPIPELINESCORE_API FPipelineSessionProgress
{
	/** 0, 1, 2, ... per session; set by the executor. */
	int32 Sequence = 0;

	/** e.g. the token id of this step. */
	TArray<int32> Ids;

	/** Text completed by this step (may be empty). */
	FString Text;
};

/** What IPipelineSession::Step reports. */
enum class EPipelineStep : uint8
{
	/** More to do. */
	Continue,
	/** Done (success). */
	Finished,
	/** OutResult carries the error. */
	Failed
};

/** Why IPipelineSession::End is called. */
enum class EPipelineSessionEnd : uint8
{
	Finished,
	Failed,
	Cancelled,
	Shutdown
};

/** What IPipelineModel::CreateSession is asked to create. Plain data. */
struct MODELPIPELINESCORE_API FPipelineSessionRequest
{
	/** e.g. "TextGeneration". */
	FName Kind;

	/** Defined by the module pair (C63 defines the TextGeneration one). */
	TSharedPtr<void, ESPMode::ThreadSafe> Payload;
};

/**
 * A long-running job created by a backend model and driven by FPipelineExecutor on its worker thread. Only the worker
 * calls these, one at a time, so the implementation needs no locks of its own.
 */
class MODELPIPELINESCORE_API IPipelineSession
{
public:
	virtual ~IPipelineSession() = default;

	/** Any thread. The model this session runs on (non-null). The executor holds it exclusively from Begin to End. */
	virtual TSharedPtr<IPipelineModel> GetModel() const = 0;

	/** Worker, once, after the executor initialized the model on a background task. Prepares the job (context,
	    prompt evaluation). Ok() -> Step follows. Failure -> End(Failed) follows, then OnComplete(failure). */
	virtual FPipelineResult Begin() = 0;

	/** Worker. Does one unit of work (one token) and may append at most one item to OutProgress (the executor
	    overwrites Sequence). Continue: more to do. Finished: done (success). Failed: OutResult carries the error. */
	virtual EPipelineStep Step(TArray<FPipelineSessionProgress>& OutProgress, FPipelineResult& OutResult) = 0;

	/** Worker, exactly once if Begin was called. Releases per-job state (context, KV cache). OutFinal: on Finished,
	    the success result with timings (InferenceMs = Begin..End); ignored otherwise. Must not fail. */
	virtual void End(EPipelineSessionEnd Reason, FPipelineResult& OutFinal) = 0;
};

/** Game-thread callbacks of one session (FPipelineExecutor::EnqueueSession). */
struct FPipelineSessionCallbacks
{
	/** Game thread, once per progress item, in Sequence order. Never inside EnqueueSession/Cancel, never after
	    OnComplete, never after Cancel(Id) returned true on the game thread. May be empty. */
	TUniqueFunction<void(const FPipelineSessionProgress& Progress)> OnProgress;

	/** Game thread, exactly once per accepted session (unless Owner died or Shutdown dropped it), never inside
	    EnqueueSession/Cancel, after every delivered OnProgress. Result: Ok with Device/QueueMs/InferenceMs, or the
	    failure, or Cancelled. */
	TUniqueFunction<void(const FPipelineResult& Result)> OnComplete;
};
