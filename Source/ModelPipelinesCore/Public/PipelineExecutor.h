// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Templates/Function.h"
#include "Templates/UniquePtr.h"
#include "PipelineBackend.h"
#include "PipelineSession.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"

class UObject;
class FRunnableThread;
struct FPipelineExecutorState;
class FPipelineExecutorWorker;

/** Identifies a job accepted by FPipelineExecutor. 0 = invalid / not accepted. Ids are unique per executor. */
using FPipelineJobId = uint64;

/**
 * Completion callback of FPipelineExecutor::Enqueue. Always called on the game thread, never inside Enqueue or Cancel.
 * Result carries the error code, Device, QueueMs and InferenceMs. Outputs holds one tensor per model output on success
 * and is empty on failure; the callback may move from it.
 */
using FPipelineInferenceCallback = TUniqueFunction<void(const FPipelineResult& Result, TArray<FPipelineTensor>&& Outputs)>;

/** What a scheduler sees about one queued job. Plain data; a copy taken under the queue lock. */
struct FPipelineJobInfo
{
	/** The id Enqueue returned (> 0). */
	FPipelineJobId Id = 0;
	/** FPlatformTime::Seconds() at Enqueue. */
	double EnqueueTimeSeconds = 0.0;
	/** The Priority passed to Enqueue. Ignored by the default FIFO order; meant for a Pro scheduler. */
	int32 Priority = 0;
};

/**
 * Pro extension point: decides which queued job the executor runs next.
 *
 * Threading: SelectNext is called on the executor's worker thread with the queue lock held, once per job the worker
 * picks while more than zero jobs wait. It must be fast, must not block, and must not call into the executor (that
 * would deadlock). Pending is in enqueue order and is only valid for the duration of the call.
 * The return value is an index into Pending; an out-of-range value (negative or >= Pending.Num()) means 0 (FIFO).
 * Without a scheduler the executor runs jobs in FIFO order.
 * Pending holds every queued job, including queued RunBlocking jobs: they appear with Priority = 0 and an internal
 * id (> 0) that Enqueue never returned to anyone.
 */
class IPipelineScheduler
{
public:
	virtual ~IPipelineScheduler() = default;

	/** Returns the index into Pending of the job to run next. Out of range -> 0. Worker thread, queue lock held. */
	virtual int32 SelectNext(TConstArrayView<FPipelineJobInfo> Pending) = 0; // index into Pending; out of range -> 0
};

/** Counters of an FPipelineExecutor since construction or the last ResetStats. */
struct FPipelineExecutorStats
{
	// NumCompleted: a job counts as soon as its game-thread delivery starts (callback about to run, or dropped because
	// its Owner died), so a GetStats() inside a callback includes that job. RunBlocking jobs count when the worker
	// releases the waiter.
	int64 NumCompleted = 0;               // jobs whose callback was dispatched (success or failure, incl. cancelled)
	int64 NumCancelled = 0;
	double MaxGameThreadDispatchMs = 0.0; // max time spent in one executor game-thread dispatch, EXCLUDING the user callback
	double MaxEnqueueMs = 0.0;            // max time spent inside one Enqueue call
};

/**
 * The single inference queue of ModelPipelines. Every inference in the plugin goes through an executor.
 *
 * - Jobs (a prepared IPipelineModel plus input tensors) are enqueued from any thread.
 * - One dedicated worker thread ("ModelPipelinesExecutor") runs them one at a time, so IPipelineModel::Run, which is
 *   not thread-safe per model, is always serialized. The order comes from the IPipelineScheduler (FIFO by default).
 * - Results are delivered on the game thread, in completion order. The game thread never waits for inference.
 *
 * All public functions are thread-safe unless noted. The module owns one instance (FPipelineExecutor::Get());
 * tests and tools may create private instances.
 */
class MODELPIPELINESCORE_API FPipelineExecutor
{
public:
	/** Starts one worker thread named "ModelPipelinesExecutor". */
	FPipelineExecutor();

	/**
	 * Calls Shutdown(). Must not be called from a callback of this executor or from the worker thread. May run on any
	 * other thread (e.g. a task worker dropping the last TryGetShared() reference after the module shut it down): once
	 * Shutdown() has run it only frees memory and never touches the game thread.
	 */
	~FPipelineExecutor();

	FPipelineExecutor(const FPipelineExecutor&) = delete;
	FPipelineExecutor& operator=(const FPipelineExecutor&) = delete;

	/** The module's instance (FModelPipelinesCoreModule::Get().GetExecutor()). Any thread once the module is loaded. */
	static FPipelineExecutor& Get();

	/** Any thread. The module's executor, or null before StartupModule / after ShutdownModule began.
	    The returned pointer keeps the object alive; after Shutdown(), Enqueue returns 0. */
	static TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> TryGetShared();

	/**
	 * Any thread. Never waits for inference (only a short lock). Returns an id > 0 if accepted, 0 after Shutdown
	 * (the callback is then never called). Null Model or empty Inputs are accepted and complete with InvalidArgument.
	 * The callback runs on the game thread exactly once per accepted job, never inside Enqueue, in completion order,
	 * unless Owner was non-null and is no longer valid at dispatch (then dropped, Verbose log).
	 * Jobs run one at a time on the worker, in the scheduler's order (FIFO by default).
	 * Owner is held weakly and only checked on the game thread. Priority is passed to the scheduler.
	 */
	FPipelineJobId Enqueue(TSharedPtr<IPipelineModel> Model, TArray<FPipelineTensor> Inputs,
		FPipelineInferenceCallback OnComplete, const UObject* Owner = nullptr, int32 Priority = 0);

	/**
	 * Tests and tools only. Same queue; waits for this job. Any thread incl. the game thread (the worker never waits
	 * on the game thread). After Shutdown: Fail(Cancelled). Null model: Fail(InvalidArgument).
	 * Inputs are copied. Released with Cancelled if Shutdown drops the job while it is queued; never hangs.
	 * Must not be called from the executor's own worker thread (for example from inside IPipelineModel::Run or a
	 * scheduler): the job would queue behind the running one and never start, so such a call returns
	 * Fail(Internal) at once without queueing anything.
	 */
	FPipelineResult RunBlocking(TSharedPtr<IPipelineModel> Model, TConstArrayView<FPipelineTensor> Inputs, TArray<FPipelineTensor>& OutOutputs);

	/**
	 * Any thread. Queues a session (C61); returns its id (> 0), or 0 after Shutdown (callbacks never called). A null
	 * session is accepted and completes with InvalidArgument. Owner/Priority as for Enqueue.
	 * The worker loads the model on a background task if needed, then runs Begin, one Step per turn (the session goes
	 * back to the end of the queue after each Step) and End. From Begin to End no other job runs on the session's model.
	 */
	FPipelineJobId EnqueueSession(TSharedPtr<IPipelineSession> Session, FPipelineSessionCallbacks Callbacks,
		const UObject* Owner = nullptr, int32 Priority = 0);

	/**
	 * Tests and tools; any thread except the worker (returns Fail(Internal) there at once). Runs the session to the end
	 * through the same queue; OutProgress gets every item in order. Same release rules as RunBlocking.
	 */
	FPipelineResult RunSessionBlocking(TSharedPtr<IPipelineSession> Session, TArray<FPipelineSessionProgress>& OutProgress);

	/**
	 * Any thread. Returns true if the job was still queued; it is removed and its callback then fires (game thread,
	 * never inside Cancel) with Cancelled. A running one-shot job, a finished job, id 0 or an unknown id returns false.
	 * A running (or loading) session returns true: the worker finishes at most the Step in flight, calls
	 * End(Cancelled), then OnComplete(Cancelled) follows; progress not yet delivered is dropped. A second Cancel of
	 * the same session returns false.
	 */
	bool Cancel(FPipelineJobId Id);

	/** Jobs waiting in the queue, not counting the running one. */
	int32 GetNumQueued() const;

	/**
	 * Queued + running + finished but callback not yet dispatched. Inside a callback, the job being delivered (and
	 * the rest of its dispatch batch) no longer counts.
	 */
	int32 GetNumPending() const;

	/**
	 * Any thread. Replaces the scheduler used for the next pick; nullptr restores FIFO. The executor keeps a shared
	 * reference until it is replaced or the executor is destroyed. See IPipelineScheduler for its threading rules.
	 */
	void SetScheduler(TSharedPtr<IPipelineScheduler> Scheduler);

	/** A snapshot of the counters. */
	FPipelineExecutorStats GetStats() const;

	/** Zeroes every counter and maximum. */
	void ResetStats();

	/**
	 * Idempotent. Stops accepting, drops queued jobs (callbacks NOT called), waits for the running job, joins the worker.
	 * Queued RunBlocking callers are released with Cancelled. Completed jobs whose dispatch is already scheduled may
	 * still be delivered while the executor lives. Any thread except the worker (never from a scheduler).
	 */
	void Shutdown();

private:
	TSharedPtr<FPipelineExecutorState, ESPMode::ThreadSafe> State;
	TUniquePtr<FPipelineExecutorWorker> Worker;
	FRunnableThread* Thread = nullptr;
};
