// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "PipelineExecutor.h"

#include "Async/Async.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTLS.h"
#include "HAL/PlatformTime.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "Misc/ScopeLock.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineLog.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "UObject/WeakObjectPtr.h"
#include "UObject/WeakObjectPtrTemplates.h"

// File-local helpers (internal linkage).
namespace
{
	/** Completion slot of a RunBlocking call. Shared by the caller and the worker; the last owner returns the event. */
	struct FBlockingWaiter
	{
		FBlockingWaiter()
			: Event(FPlatformProcess::GetSynchEventFromPool(true))
		{
		}

		~FBlockingWaiter()
		{
			FPlatformProcess::ReturnSynchEventToPool(Event);
		}

		FBlockingWaiter(const FBlockingWaiter&) = delete;
		FBlockingWaiter& operator=(const FBlockingWaiter&) = delete;

		FEvent* Event = nullptr;
		FPipelineResult Result;
		TArray<FPipelineTensor> Outputs;
	};

	/** One job, from Enqueue/RunBlocking to dispatch. Owned by exactly one list (or the worker) at a time. */
	struct FExecutorJob
	{
		FPipelineJobInfo Info;
		TSharedPtr<IPipelineModel> Model;
		TArray<FPipelineTensor> Inputs;
		FPipelineInferenceCallback Callback;
		/** Checked only on the game thread. */
		TWeakObjectPtr<const UObject> Owner;
		bool bHasOwner = false;
		/** Set for RunBlocking jobs: they complete through the waiter, never through a game-thread dispatch. */
		TSharedPtr<FBlockingWaiter, ESPMode::ThreadSafe> Waiter;
		FPipelineResult Result;
		TArray<FPipelineTensor> Outputs;
	};

	double CyclesToMs(uint64 Cycles)
	{
		return FPlatformTime::ToMilliseconds64(Cycles);
	}
}

/** Everything the worker, the game-thread dispatch and the executor share. Guarded by Lock unless noted. */
struct FPipelineExecutorState
{
	FPipelineExecutorState()
		: WakeEvent(FPlatformProcess::GetSynchEventFromPool(false))
	{
	}

	~FPipelineExecutorState()
	{
		FPlatformProcess::ReturnSynchEventToPool(WakeEvent);
	}

	mutable FCriticalSection Lock;
	/** Serializes Shutdown callers. */
	FCriticalSection ShutdownLock;

	TArray<TUniquePtr<FExecutorJob>> Queue;
	/** Finished async jobs waiting for the game-thread dispatch, in completion order. */
	TArray<TUniquePtr<FExecutorJob>> Completed;
	/** Reused buffer for the scheduler's view (worker only, under Lock). */
	TArray<FPipelineJobInfo> InfoScratch;
	TSharedPtr<IPipelineScheduler> Scheduler;

	/** Auto-reset; wakes the worker. Not guarded (FEvent is thread-safe). */
	FEvent* WakeEvent = nullptr;

	FPipelineJobId NextId = 1;
	/** Id of the worker thread (0 if it could not be created). Written once in the executor's constructor, before
	 *  the executor is shared; read-only afterwards, so not guarded. */
	uint32 WorkerThreadId = 0;
	bool bStopping = false;
	/** Under ShutdownLock. */
	bool bShutdownDone = false;
	/** A game-thread drain task is outstanding. */
	bool bDispatchScheduled = false;
	int32 NumRunning = 0;
	int32 NumAwaitingDispatch = 0;
	FPipelineExecutorStats Stats;

	FPipelineJobId AllocateIdLocked()
	{
		const FPipelineJobId Id = NextId++;
		if (NextId == 0)
		{
			NextId = 1;
		}
		return Id;
	}
};

// File-local helpers (internal linkage).
namespace
{
	using FStatePtr = TSharedPtr<FPipelineExecutorState, ESPMode::ThreadSafe>;
	using FStateWeakPtr = TWeakPtr<FPipelineExecutorState, ESPMode::ThreadSafe>;

	void ScheduleDispatch(const FStateWeakPtr& WeakState);

	/** Game thread. Delivers one snapshot of the completion list, then reschedules if more arrived meanwhile. */
	void DrainCompleted(const FStateWeakPtr& WeakState)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Executor.Dispatch");
		const FStatePtr State = WeakState.Pin();
		if (!State.IsValid())
		{
			return; // The executor is gone; its undelivered completions went with it.
		}

		const uint64 StartCycles = FPlatformTime::Cycles64();
		uint64 UserCycles = 0;

		TArray<TUniquePtr<FExecutorJob>> Batch;
		{
			FScopeLock Guard(&State->Lock);
			Swap(Batch, State->Completed);
			// A job counts as dispatched as soon as its delivery starts (callback invoked or dropped for a dead owner),
			// so stats and GetNumPending() read inside a callback already include the job in hand. One lock per drain.
			State->NumAwaitingDispatch -= Batch.Num();
			State->Stats.NumCompleted += Batch.Num();
		}

		for (TUniquePtr<FExecutorJob>& Job : Batch)
		{
			if (Job->bHasOwner && !Job->Owner.IsValid())
			{
				UE_LOG(LogModelPipelines, Verbose, TEXT("Executor: job %llu dropped, its owner is no longer valid."), (unsigned long long)Job->Info.Id);
			}
			else if (Job->Callback)
			{
				const uint64 CallStart = FPlatformTime::Cycles64();
				Job->Callback(Job->Result, MoveTemp(Job->Outputs));
				UserCycles += FPlatformTime::Cycles64() - CallStart;
			}
			Job.Reset();
		}
		Batch.Reset();

		bool bReschedule = false;
		{
			FScopeLock Guard(&State->Lock);
			if (State->Completed.Num() > 0)
			{
				bReschedule = true; // bDispatchScheduled stays true: the next task takes over
			}
			else
			{
				State->bDispatchScheduled = false;
			}
		}
		if (bReschedule)
		{
			ScheduleDispatch(WeakState);
		}

		const uint64 TotalCycles = FPlatformTime::Cycles64() - StartCycles;
		const double OwnMs = CyclesToMs(TotalCycles > UserCycles ? TotalCycles - UserCycles : 0);
		FScopeLock Guard(&State->Lock);
		State->Stats.MaxGameThreadDispatchMs = FMath::Max(State->Stats.MaxGameThreadDispatchMs, OwnMs);
	}

	void ScheduleDispatch(const FStateWeakPtr& WeakState)
	{
		AsyncTask(ENamedThreads::GameThread, [WeakState]()
		{
			DrainCompleted(WeakState);
		});
	}

	/**
	 * Called with State->Lock NOT held. Hands a finished or cancelled job to its consumer: the RunBlocking waiter
	 * directly, or the completion list plus (at most one outstanding) game-thread dispatch.
	 */
	void CompleteJob(const FStatePtr& State, TUniquePtr<FExecutorJob> Job, bool bWasRunning)
	{
		if (Job->Waiter.IsValid())
		{
			const TSharedPtr<FBlockingWaiter, ESPMode::ThreadSafe> Waiter = Job->Waiter;
			Waiter->Result = Job->Result;
			Waiter->Outputs = MoveTemp(Job->Outputs);
			Job.Reset();
			{
				FScopeLock Guard(&State->Lock);
				if (bWasRunning)
				{
					State->NumRunning = 0;
				}
				++State->Stats.NumCompleted;
			}
			Waiter->Event->Trigger();
			return;
		}

		bool bSchedule = false;
		{
			FScopeLock Guard(&State->Lock);
			if (bWasRunning)
			{
				State->NumRunning = 0;
			}
			++State->NumAwaitingDispatch;
			State->Completed.Add(MoveTemp(Job));
			if (!State->bDispatchScheduled)
			{
				State->bDispatchScheduled = true;
				bSchedule = true;
			}
		}
		if (bSchedule)
		{
			ScheduleDispatch(FStateWeakPtr(State));
		}
	}
}

/** The executor's worker thread body. Runs jobs one at a time until the state is stopping. */
class FPipelineExecutorWorker : public FRunnable
{
public:
	explicit FPipelineExecutorWorker(const FStatePtr& InState)
		: State(InState)
	{
	}

	virtual uint32 Run() override
	{
		for (;;)
		{
			TUniquePtr<FExecutorJob> Job;
			{
				FScopeLock Guard(&State->Lock);
				if (State->bStopping)
				{
					break;
				}
				if (State->Queue.Num() > 0)
				{
					int32 Index = 0;
					if (State->Scheduler.IsValid())
					{
						State->InfoScratch.Reset();
						for (const TUniquePtr<FExecutorJob>& Queued : State->Queue)
						{
							State->InfoScratch.Add(Queued->Info);
						}
						const int32 Selected = State->Scheduler->SelectNext(State->InfoScratch);
						if (Selected >= 0 && Selected < State->Queue.Num())
						{
							Index = Selected;
						}
					}
					Job = MoveTemp(State->Queue[Index]);
					State->Queue.RemoveAt(Index);
					State->NumRunning = 1;
				}
			}

			if (!Job.IsValid())
			{
				State->WakeEvent->Wait();
				continue;
			}

			Execute(*Job);
			CompleteJob(State, MoveTemp(Job), true);
		}
		return 0;
	}

	virtual void Stop() override
	{
		State->WakeEvent->Trigger();
	}

private:
	static void Execute(FExecutorJob& Job)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Executor.Job");
		const double StartSeconds = FPlatformTime::Seconds();
		const float QueueMs = (float)FMath::Max(0.0, (StartSeconds - Job.Info.EnqueueTimeSeconds) * 1000.0);

		if (!Job.Model.IsValid())
		{
			Job.Result = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("Executor: the model is null."));
		}
		else if (Job.Inputs.Num() == 0)
		{
			Job.Result = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("Executor: the job has no input tensors."));
		}
		else
		{
			Job.Result = Job.Model->Run(Job.Inputs, Job.Outputs);
			if (!Job.Result.bSuccess)
			{
				Job.Outputs.Reset();
			}
		}
		Job.Result.QueueMs = QueueMs;

		// Release the model and inputs here so the game-thread dispatch never pays for them.
		Job.Model.Reset();
		Job.Inputs.Empty();
	}

	FStatePtr State;
};

FPipelineExecutor::FPipelineExecutor()
	: State(MakeShared<FPipelineExecutorState, ESPMode::ThreadSafe>())
{
	Worker = MakeUnique<FPipelineExecutorWorker>(State);
	Thread = FRunnableThread::Create(Worker.Get(), TEXT("ModelPipelinesExecutor"), 0, TPri_Normal);
	if (Thread != nullptr)
	{
		State->WorkerThreadId = Thread->GetThreadID();
	}
	else
	{
		UE_LOG(LogModelPipelines, Error, TEXT("Executor: could not create the worker thread; the executor accepts no work."));
		FScopeLock Guard(&State->Lock);
		State->bStopping = true;
	}
}

FPipelineExecutor::~FPipelineExecutor()
{
	Shutdown();
	Worker.Reset();
	State.Reset();
}

FPipelineExecutor& FPipelineExecutor::Get()
{
	return FModelPipelinesCoreModule::Get().GetExecutor();
}

// FPipelineExecutor::TryGetShared() is defined in ModelPipelinesCoreModule.cpp, next to the slot the module owns.

FPipelineJobId FPipelineExecutor::Enqueue(TSharedPtr<IPipelineModel> Model, TArray<FPipelineTensor> Inputs,
	FPipelineInferenceCallback OnComplete, const UObject* Owner, int32 Priority)
{
	const uint64 StartCycles = FPlatformTime::Cycles64();

	TUniquePtr<FExecutorJob> Job = MakeUnique<FExecutorJob>();
	Job->Model = MoveTemp(Model);
	Job->Inputs = MoveTemp(Inputs);
	Job->Callback = MoveTemp(OnComplete);
	Job->bHasOwner = Owner != nullptr;
	Job->Owner = Owner;
	Job->Info.Priority = Priority;
	Job->Info.EnqueueTimeSeconds = FPlatformTime::Seconds();

	FPipelineJobId Id = 0;
	{
		FScopeLock Guard(&State->Lock);
		if (State->bStopping)
		{
			return 0; // Job (and its callback) is destroyed after the lock is released
		}
		Id = State->AllocateIdLocked();
		Job->Info.Id = Id;
		State->Queue.Add(MoveTemp(Job));
	}
	State->WakeEvent->Trigger();

	const double Ms = CyclesToMs(FPlatformTime::Cycles64() - StartCycles);
	FScopeLock Guard(&State->Lock);
	State->Stats.MaxEnqueueMs = FMath::Max(State->Stats.MaxEnqueueMs, Ms);
	return Id;
}

FPipelineResult FPipelineExecutor::RunBlocking(TSharedPtr<IPipelineModel> Model, TConstArrayView<FPipelineTensor> Inputs, TArray<FPipelineTensor>& OutOutputs)
{
	OutOutputs.Reset();
	if (State->WorkerThreadId != 0 && FPlatformTLS::GetCurrentThreadId() == State->WorkerThreadId)
	{
		// The job would queue behind the running one (this call) and never start.
		return FPipelineResult::Fail(EPipelineError::Internal, TEXT("RunBlocking called from the executor worker thread"));
	}
	if (!Model.IsValid())
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("Executor: the model is null."));
	}

	const TSharedPtr<FBlockingWaiter, ESPMode::ThreadSafe> Waiter = MakeShared<FBlockingWaiter, ESPMode::ThreadSafe>();
	TUniquePtr<FExecutorJob> Job = MakeUnique<FExecutorJob>();
	Job->Model = MoveTemp(Model);
	Job->Inputs.Append(Inputs.GetData(), Inputs.Num());
	Job->Waiter = Waiter;
	Job->Info.EnqueueTimeSeconds = FPlatformTime::Seconds();

	{
		FScopeLock Guard(&State->Lock);
		if (State->bStopping)
		{
			return FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("Executor: shut down."));
		}
		Job->Info.Id = State->AllocateIdLocked();
		State->Queue.Add(MoveTemp(Job));
	}
	State->WakeEvent->Trigger();

	// Released by the worker when the job finishes, or by Shutdown/Cancel with Cancelled while it is queued.
	Waiter->Event->Wait();
	OutOutputs = MoveTemp(Waiter->Outputs);
	return Waiter->Result;
}

bool FPipelineExecutor::Cancel(FPipelineJobId Id)
{
	if (Id == 0)
	{
		return false;
	}

	TUniquePtr<FExecutorJob> Job;
	{
		FScopeLock Guard(&State->Lock);
		const int32 Index = State->Queue.IndexOfByPredicate([Id](const TUniquePtr<FExecutorJob>& Queued)
		{
			return Queued->Info.Id == Id;
		});
		if (Index == INDEX_NONE)
		{
			return false;
		}
		Job = MoveTemp(State->Queue[Index]);
		State->Queue.RemoveAt(Index);
		++State->Stats.NumCancelled;
	}

	Job->Result = FPipelineResult::Fail(EPipelineError::Cancelled, FString::Printf(TEXT("Executor: job %llu was cancelled while queued."), (unsigned long long)Id));
	Job->Result.QueueMs = (float)FMath::Max(0.0, (FPlatformTime::Seconds() - Job->Info.EnqueueTimeSeconds) * 1000.0);
	Job->Model.Reset();
	Job->Inputs.Empty();
	CompleteJob(State, MoveTemp(Job), false);
	return true;
}

int32 FPipelineExecutor::GetNumQueued() const
{
	FScopeLock Guard(&State->Lock);
	return State->Queue.Num();
}

int32 FPipelineExecutor::GetNumPending() const
{
	FScopeLock Guard(&State->Lock);
	return State->Queue.Num() + State->NumRunning + State->NumAwaitingDispatch;
}

void FPipelineExecutor::SetScheduler(TSharedPtr<IPipelineScheduler> Scheduler)
{
	{
		FScopeLock Guard(&State->Lock);
		Swap(State->Scheduler, Scheduler);
	}
	// The previous scheduler (now in Scheduler) is released here, outside the lock.
}

FPipelineExecutorStats FPipelineExecutor::GetStats() const
{
	FScopeLock Guard(&State->Lock);
	return State->Stats;
}

void FPipelineExecutor::ResetStats()
{
	FScopeLock Guard(&State->Lock);
	State->Stats = FPipelineExecutorStats();
}

void FPipelineExecutor::Shutdown()
{
	if (!State.IsValid())
	{
		return;
	}

	FScopeLock ShutdownGuard(&State->ShutdownLock);
	if (State->bShutdownDone)
	{
		return;
	}

	// 1. Stop accepting. 2. Take the queued jobs.
	TArray<TUniquePtr<FExecutorJob>> Dropped;
	{
		FScopeLock Guard(&State->Lock);
		State->bStopping = true;
		Swap(Dropped, State->Queue);
	}

	// 2. Async callbacks are dropped without being called; RunBlocking waiters are released with Cancelled.
	for (TUniquePtr<FExecutorJob>& Job : Dropped)
	{
		if (Job->Waiter.IsValid())
		{
			Job->Waiter->Result = FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("Executor: shut down while the job was queued."));
			Job->Waiter->Event->Trigger();
		}
	}
	if (Dropped.Num() > 0)
	{
		UE_LOG(LogModelPipelines, Verbose, TEXT("Executor: shutdown dropped %d queued job(s)."), Dropped.Num());
	}
	Dropped.Reset();

	// 3./4. Wake the worker; it finishes the running job, sees bStopping and exits. Join it.
	State->WakeEvent->Trigger();
	if (Thread != nullptr)
	{
		Thread->Kill(true);
		delete Thread;
		Thread = nullptr;
	}

	State->bShutdownDone = true;
}
