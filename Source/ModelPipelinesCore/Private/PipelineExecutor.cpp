// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

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
#include "Tasks/Task.h"
#include "UObject/WeakObjectPtr.h"
#include "UObject/WeakObjectPtrTemplates.h"

#include <atomic>

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
		/** RunSessionBlocking: progress items, appended by the worker only (until the event fires). */
		TArray<FPipelineSessionProgress> Progress;
	};

	/**
	 * Shared by the worker, Cancel and the game-thread dispatch of one session. Callbacks are touched only on the game
	 * thread; bCancelled is atomic; the rest is written once before the session is queued.
	 */
	struct FSessionControl
	{
		FPipelineJobId Id = 0;
		FPipelineSessionCallbacks Callbacks;
		TWeakObjectPtr<const UObject> Owner;
		bool bHasOwner = false;
		std::atomic<bool> bCancelled{ false };
		/** Under the state Lock. The session is ending as Finished/Failed: a cancel request no longer applies (C61 rule 1). */
		bool bEnding = false;
		/** Set for RunSessionBlocking sessions: progress and result go to the waiter, never to the game thread. */
		TSharedPtr<FBlockingWaiter, ESPMode::ThreadSafe> Waiter;
	};

	using FSessionControlPtr = TSharedPtr<FSessionControl, ESPMode::ThreadSafe>;

	/** One job or session, from Enqueue/RunBlocking/EnqueueSession to dispatch. Owned by exactly one list (or the worker or a load task's slot) at a time. */
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

		// Sessions only.
		bool bIsSession = false;
		TSharedPtr<IPipelineSession> Session;
		/** Session->GetModel(), taken at enqueue. */
		TSharedPtr<IPipelineModel> SessionModel;
		FSessionControlPtr Ctl;
		/** The worker picked it once (QueueMs is set). */
		bool bStarted = false;
		/** The executor ran the model's load task for it (successfully). */
		bool bLoaded = false;
		/** Begin was called (End must follow exactly once). */
		bool bBegun = false;
		/** It holds its model in FPipelineExecutorState::HeldModels. */
		bool bHolding = false;
		int32 NextSequence = 0;
		float QueueMs = 0.f;
		double BeginSeconds = 0.0;
	};

	/** One entry of the game-thread dispatch list: a finished job/session, or one session progress item. */
	struct FDispatchItem
	{
		/** Finished job or session (null for progress items). */
		TUniquePtr<FExecutorJob> Job;
		/** Progress items only. */
		FSessionControlPtr Ctl;
		FPipelineSessionProgress Progress;
	};

	/** Who holds a model exclusively (a session from Begin to End, or while its load task runs). */
	struct FModelHold
	{
		FPipelineJobId Holder = 0;
		bool bLoading = false;
	};

	double CyclesToMs(uint64 Cycles)
	{
		return FPlatformTime::ToMilliseconds64(Cycles);
	}

	float MsSince(double StartSeconds)
	{
		return (float)FMath::Max(0.0, (FPlatformTime::Seconds() - StartSeconds) * 1000.0);
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
	/** Finished jobs/sessions and session progress waiting for the game-thread dispatch, in completion order. */
	TArray<FDispatchItem> Completed;
	/** Sessions whose model load task runs. */
	TArray<TUniquePtr<FExecutorJob>> Loading;
	/** Models held exclusively by a session (C61 rules 2 and 4). */
	TMap<IPipelineModel*, FModelHold> HeldModels;
	/** The session the worker is processing right now (null otherwise). */
	FSessionControlPtr RunningSessionCtl;
	/** Reused buffers for the scheduler's view (worker only, under Lock). */
	TArray<FPipelineJobInfo> InfoScratch;
	TArray<int32> EligibleScratch;
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
	int32 NumLoading = 0;
	int32 NumAwaitingDispatch = 0;
	/** Sessions accepted and not yet completed (anywhere). 0 = the M7 fast paths apply. */
	int32 NumSessionsLive = 0;
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

	/** Under Lock. May this queued entry run now (C61 rule 2)? */
	bool IsEligibleLocked(const FExecutorJob& Job) const
	{
		IPipelineModel* const ModelPtr = Job.bIsSession ? Job.SessionModel.Get() : Job.Model.Get();
		if (ModelPtr == nullptr)
		{
			return true;
		}
		const FModelHold* Hold = HeldModels.Find(ModelPtr);
		if (Hold == nullptr)
		{
			return true;
		}
		return Job.bIsSession && Hold->Holder == Job.Info.Id && !Hold->bLoading;
	}

	/** Under Lock. Releases the model hold of a session that ends here. */
	void ReleaseHoldLocked(FExecutorJob& Job)
	{
		if (Job.bHolding)
		{
			IPipelineModel* const ModelPtr = Job.SessionModel.Get();
			const FModelHold* Hold = HeldModels.Find(ModelPtr);
			if (Hold != nullptr && Hold->Holder == Job.Info.Id)
			{
				HeldModels.Remove(ModelPtr);
			}
			Job.bHolding = false;
		}
	}

	/**
	 * Under Lock. Marks a session cancelled (once). Returns false if it already was. OutLoadingModel is set if the
	 * session's load task runs (the caller calls CancelInitialize outside the lock).
	 */
	bool RequestCancelLocked(FSessionControl& Ctl, TSharedPtr<IPipelineModel>& OutLoadingModel)
	{
		if (Ctl.bEnding)
		{
			return false; // already finishing (Finished/Failed); Cancel returns false for it
		}
		bool bExpected = false;
		if (!Ctl.bCancelled.compare_exchange_strong(bExpected, true))
		{
			return false;
		}
		++Stats.NumCancelled;
		const FPipelineJobId Id = Ctl.Id;
		for (const TUniquePtr<FExecutorJob>& Loaded : Loading)
		{
			if (Loaded->Info.Id == Id)
			{
				OutLoadingModel = Loaded->SessionModel;
				break;
			}
		}
		return true;
	}
};

// File-local helpers (internal linkage).
namespace
{
	using FStatePtr = TSharedPtr<FPipelineExecutorState, ESPMode::ThreadSafe>;
	using FStateWeakPtr = TWeakPtr<FPipelineExecutorState, ESPMode::ThreadSafe>;

	void ScheduleDispatch(const FStateWeakPtr& WeakState);

	/** Lock NOT held. Asks a session's model load to stop and wakes the worker after a cancel request. */
	void AfterCancelRequest(const FStatePtr& State, const TSharedPtr<IPipelineModel>& LoadingModel)
	{
		if (LoadingModel.IsValid())
		{
			LoadingModel->CancelInitialize();
		}
		State->WakeEvent->Trigger();
	}

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

		TArray<FDispatchItem> Batch;
		{
			FScopeLock Guard(&State->Lock);
			Swap(Batch, State->Completed);
			// A job counts as dispatched as soon as its delivery starts (callback invoked or dropped for a dead owner),
			// so stats and GetNumPending() read inside a callback already include the job in hand. One lock per drain.
			int32 NumFinished = 0;
			for (const FDispatchItem& Item : Batch)
			{
				NumFinished += Item.Job.IsValid() ? 1 : 0;
			}
			State->NumAwaitingDispatch -= NumFinished;
			State->Stats.NumCompleted += NumFinished;
		}

		for (FDispatchItem& Item : Batch)
		{
			if (!Item.Job.IsValid())
			{
				// Session progress.
				FSessionControl& Ctl = *Item.Ctl;
				if (Ctl.bHasOwner && !Ctl.Owner.IsValid())
				{
					// C61 rule 6: cancelled as if by Cancel; nothing more is delivered.
					TSharedPtr<IPipelineModel> LoadingModel;
					bool bNewlyCancelled = false;
					{
						FScopeLock Guard(&State->Lock);
						bNewlyCancelled = State->RequestCancelLocked(Ctl, LoadingModel);
					}
					if (bNewlyCancelled)
					{
						UE_LOG(LogModelPipelines, Verbose, TEXT("Executor: session %llu cancelled, its owner is no longer valid."), (unsigned long long)Ctl.Id);
						AfterCancelRequest(State, LoadingModel);
					}
				}
				else if (!Ctl.bCancelled.load() && Ctl.Callbacks.OnProgress)
				{
					const uint64 CallStart = FPlatformTime::Cycles64();
					Ctl.Callbacks.OnProgress(Item.Progress);
					UserCycles += FPlatformTime::Cycles64() - CallStart;
				}
				Item.Ctl.Reset();
				continue;
			}

			FExecutorJob& Job = *Item.Job;
			if (Job.bIsSession)
			{
				FSessionControl& Ctl = *Job.Ctl;
				if (Ctl.bHasOwner && !Ctl.Owner.IsValid())
				{
					UE_LOG(LogModelPipelines, Verbose, TEXT("Executor: session %llu dropped, its owner is no longer valid."), (unsigned long long)Ctl.Id);
				}
				else if (Ctl.Callbacks.OnComplete)
				{
					const uint64 CallStart = FPlatformTime::Cycles64();
					Ctl.Callbacks.OnComplete(Job.Result);
					UserCycles += FPlatformTime::Cycles64() - CallStart;
				}
				// The user's captures are released here, on the game thread.
				const uint64 FreeStart = FPlatformTime::Cycles64();
				Ctl.Callbacks.OnProgress.Reset();
				Ctl.Callbacks.OnComplete.Reset();
				UserCycles += FPlatformTime::Cycles64() - FreeStart;
			}
			else if (Job.bHasOwner && !Job.Owner.IsValid())
			{
				UE_LOG(LogModelPipelines, Verbose, TEXT("Executor: job %llu dropped, its owner is no longer valid."), (unsigned long long)Job.Info.Id);
			}
			else if (Job.Callback)
			{
				const uint64 CallStart = FPlatformTime::Cycles64();
				Job.Callback(Job.Result, MoveTemp(Job.Outputs));
				UserCycles += FPlatformTime::Cycles64() - CallStart;
			}
			Item.Job.Reset();
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

	/** Under Lock. Appends a dispatch item; returns true if the caller must schedule the game-thread drain. */
	bool AddDispatchLocked(FPipelineExecutorState& State, FDispatchItem&& Item)
	{
		State.Completed.Add(MoveTemp(Item));
		if (!State.bDispatchScheduled)
		{
			State.bDispatchScheduled = true;
			return true;
		}
		return false;
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
			FDispatchItem Item;
			Item.Job = MoveTemp(Job);
			bSchedule = AddDispatchLocked(*State, MoveTemp(Item));
		}
		if (bSchedule)
		{
			ScheduleDispatch(FStateWeakPtr(State));
		}
	}

	/**
	 * Lock NOT held. A session ends with Job->Result (End already called if Begin ran). Releases its model hold and
	 * hands it to the waiter or the game-thread dispatch. bWasRunning: the worker was processing it.
	 */
	void CompleteSession(const FStatePtr& State, TUniquePtr<FExecutorJob> Job, bool bWasRunning)
	{
		Job->Result.QueueMs = Job->QueueMs;
		// Release the session (backend state) and the model here so the game-thread dispatch never pays for them.
		Job->Session.Reset();
		const TSharedPtr<FBlockingWaiter, ESPMode::ThreadSafe> Waiter = Job->Ctl->Waiter;

		TSharedPtr<IPipelineModel> ModelToRelease;
		bool bSchedule = false;
		{
			FScopeLock Guard(&State->Lock);
			if (bWasRunning)
			{
				State->NumRunning = 0;
				State->RunningSessionCtl.Reset();
			}
			State->ReleaseHoldLocked(*Job);
			--State->NumSessionsLive;
			ModelToRelease = MoveTemp(Job->SessionModel);
			if (Waiter.IsValid())
			{
				Waiter->Result = Job->Result;
				++State->Stats.NumCompleted;
			}
			else
			{
				++State->NumAwaitingDispatch;
				FDispatchItem Item;
				Item.Job = MoveTemp(Job);
				bSchedule = AddDispatchLocked(*State, MoveTemp(Item));
			}
		}
		Job.Reset();
		ModelToRelease.Reset();
		if (Waiter.IsValid())
		{
			Waiter->Event->Trigger();
		}
		if (bSchedule)
		{
			ScheduleDispatch(FStateWeakPtr(State));
		}
		State->WakeEvent->Trigger(); // a released model may unblock queued jobs
	}

	/**
	 * Lock NOT held. Shutdown: ends a session without callbacks (End(Shutdown) if Begin ran); a RunSessionBlocking
	 * waiter is released with Cancelled.
	 */
	void DropSessionForShutdown(const FStatePtr& State, TUniquePtr<FExecutorJob> Job, bool bWasRunning)
	{
		if (Job->bBegun && Job->Session.IsValid())
		{
			FPipelineResult Ignored;
			Job->Session->End(EPipelineSessionEnd::Shutdown, Ignored);
		}
		Job->Session.Reset();
		{
			FScopeLock Guard(&State->Lock);
			if (bWasRunning)
			{
				State->NumRunning = 0;
				State->RunningSessionCtl.Reset();
			}
			State->ReleaseHoldLocked(*Job);
			--State->NumSessionsLive;
		}
		TSharedPtr<FBlockingWaiter, ESPMode::ThreadSafe> Waiter;
		if (Job->Ctl.IsValid())
		{
			Waiter = Job->Ctl->Waiter;
		}
		Job.Reset();
		if (Waiter.IsValid())
		{
			Waiter->Result = FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("Executor: shut down while the session ran."));
			Waiter->Event->Trigger();
		}
	}

	/** Load task end (any thread). Puts the session back into the queue, or completes it (failure or cancel). */
	void FinishSessionLoad(const FStateWeakPtr& WeakState, FPipelineJobId Id, const FPipelineResult& LoadResult)
	{
		const FStatePtr State = WeakState.Pin();
		if (!State.IsValid())
		{
			return; // The executor is gone.
		}

		TUniquePtr<FExecutorJob> Failed;
		{
			FScopeLock Guard(&State->Lock);
			const int32 Index = State->Loading.IndexOfByPredicate([Id](const TUniquePtr<FExecutorJob>& Loaded)
			{
				return Loaded->Info.Id == Id;
			});
			if (Index == INDEX_NONE)
			{
				return; // Shutdown dropped it: only references are released, no callbacks.
			}
			TUniquePtr<FExecutorJob> Job = MoveTemp(State->Loading[Index]);
			State->Loading.RemoveAt(Index);
			--State->NumLoading;

			if (!LoadResult.bSuccess || Job->Ctl->bCancelled.load())
			{
				Failed = MoveTemp(Job);
			}
			else
			{
				if (FModelHold* Hold = State->HeldModels.Find(Job->SessionModel.Get()))
				{
					Hold->bLoading = false;
				}
				Job->bLoaded = true;
				State->Queue.Add(MoveTemp(Job));
			}
		}

		if (Failed.IsValid())
		{
			if (Failed->Ctl->bCancelled.load())
			{
				Failed->Result = FPipelineResult::Fail(EPipelineError::Cancelled, FString::Printf(TEXT("Executor: session %llu was cancelled while its model loaded."), (unsigned long long)Id));
			}
			else
			{
				Failed->Result = LoadResult;
			}
			CompleteSession(State, MoveTemp(Failed), false);
		}
		else
		{
			State->WakeEvent->Trigger();
		}
	}
}

/** The executor's worker thread body. Runs jobs and session steps one at a time until the state is stopping. */
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
			EAction Action = EAction::None;
			TArray<TUniquePtr<FExecutorJob>> Remaining;
			TSharedPtr<IPipelineModel> LoadModel;
			FPipelineJobId LoadId = 0;
			bool bExit = false;
			{
				FScopeLock Guard(&State->Lock);
				if (State->bStopping)
				{
					// Shutdown left only sessions that hold their model here; they end on this thread.
					Swap(Remaining, State->Queue);
					bExit = true;
				}
				else if (State->Queue.Num() > 0)
				{
					const int32 Index = PickLocked();
					if (Index != INDEX_NONE)
					{
						Job = MoveTemp(State->Queue[Index]);
						State->Queue.RemoveAt(Index);
						State->NumRunning = 1;
						Action = EAction::RunJob;
						if (Job->bIsSession)
						{
							Action = ClassifySessionLocked(*Job);
							if (Action == EAction::Load)
							{
								LoadModel = Job->SessionModel;
								LoadId = Job->Info.Id;
								State->NumRunning = 0;
								++State->NumLoading;
								State->Loading.Add(MoveTemp(Job));
							}
							else
							{
								State->RunningSessionCtl = Job->Ctl;
							}
						}
					}
				}
			}

			if (bExit)
			{
				for (TUniquePtr<FExecutorJob>& Left : Remaining)
				{
					if (Left->bIsSession)
					{
						DropSessionForShutdown(State, MoveTemp(Left), false);
					}
				}
				break;
			}

			switch (Action)
			{
			case EAction::None:
				State->WakeEvent->Wait();
				break;
			case EAction::RunJob:
				Execute(*Job);
				CompleteJob(State, MoveTemp(Job), true);
				break;
			case EAction::Load:
				LaunchLoad(MoveTemp(LoadModel), LoadId);
				break;
			case EAction::Invalid:
				Job->Result = FPipelineResult::Fail(EPipelineError::InvalidArgument, Job->Session.IsValid()
					? TEXT("Executor: the session has no model.") : TEXT("Executor: the session is null."));
				CompleteSession(State, MoveTemp(Job), true);
				break;
			case EAction::Cancel:
				EndSession(MoveTemp(Job), EPipelineSessionEnd::Cancelled, FPipelineResult());
				break;
			case EAction::Begin:
				BeginSession(MoveTemp(Job));
				break;
			case EAction::Step:
				StepSession(MoveTemp(Job));
				break;
			}
		}
		return 0;
	}

	virtual void Stop() override
	{
		State->WakeEvent->Trigger();
	}

private:
	enum class EAction : uint8
	{
		None,
		RunJob,
		Load,
		Invalid,
		Cancel,
		Begin,
		Step
	};

	/** Under Lock, Queue not empty. Index of the entry to run next, or INDEX_NONE if every entry is blocked. */
	int32 PickLocked()
	{
		FPipelineExecutorState& S = *State;
		if (S.NumSessionsLive > 0)
		{
			// A cancelled session that holds its model ends first, so Cancel completes quickly.
			for (int32 Index = 0; Index < S.Queue.Num(); ++Index)
			{
				const FExecutorJob& Queued = *S.Queue[Index];
				if (Queued.bIsSession && Queued.Ctl->bCancelled.load())
				{
					return Index;
				}
			}
		}

		if (S.HeldModels.Num() == 0)
		{
			// No model is held: every entry may run (the M7 path).
			int32 Index = 0;
			if (S.Scheduler.IsValid())
			{
				S.InfoScratch.Reset();
				for (const TUniquePtr<FExecutorJob>& Queued : S.Queue)
				{
					S.InfoScratch.Add(Queued->Info);
				}
				const int32 Selected = S.Scheduler->SelectNext(S.InfoScratch);
				if (Selected >= 0 && Selected < S.Queue.Num())
				{
					Index = Selected;
				}
			}
			return Index;
		}

		// Jobs blocked by a held model are not shown to the scheduler (C61 rule 2).
		S.EligibleScratch.Reset();
		for (int32 Index = 0; Index < S.Queue.Num(); ++Index)
		{
			if (S.IsEligibleLocked(*S.Queue[Index]))
			{
				S.EligibleScratch.Add(Index);
			}
		}
		if (S.EligibleScratch.Num() == 0)
		{
			return INDEX_NONE;
		}
		int32 Pick = 0;
		if (S.Scheduler.IsValid())
		{
			S.InfoScratch.Reset();
			for (const int32 Index : S.EligibleScratch)
			{
				S.InfoScratch.Add(S.Queue[Index]->Info);
			}
			const int32 Selected = S.Scheduler->SelectNext(S.InfoScratch);
			if (Selected >= 0 && Selected < S.EligibleScratch.Num())
			{
				Pick = Selected;
			}
		}
		return S.EligibleScratch[Pick];
	}

	/** Under Lock. What to do with a session the worker just took from the queue. */
	EAction ClassifySessionLocked(FExecutorJob& Job)
	{
		if (!Job.bStarted)
		{
			Job.bStarted = true;
			Job.QueueMs = MsSince(Job.Info.EnqueueTimeSeconds);
		}
		if (Job.Ctl->bCancelled.load())
		{
			return EAction::Cancel;
		}
		if (!Job.Session.IsValid() || !Job.SessionModel.IsValid())
		{
			return EAction::Invalid;
		}
		if (Job.bBegun)
		{
			return EAction::Step;
		}
		IPipelineModel* const ModelPtr = Job.SessionModel.Get();
		if (!Job.bLoaded && !Job.SessionModel->IsInitialized())
		{
			FModelHold& Hold = State->HeldModels.FindOrAdd(ModelPtr);
			Hold.Holder = Job.Info.Id;
			Hold.bLoading = true;
			Job.bHolding = true;
			return EAction::Load;
		}
		FModelHold& Hold = State->HeldModels.FindOrAdd(ModelPtr);
		Hold.Holder = Job.Info.Id;
		Hold.bLoading = false;
		Job.bHolding = true;
		return EAction::Begin;
	}

	/** C61 rule 4: the model loads on a background task, never on the worker. */
	void LaunchLoad(TSharedPtr<IPipelineModel> Model, FPipelineJobId Id)
	{
		const FStateWeakPtr WeakState(State);
		UE::Tasks::Launch(TEXT("ModelPipelines.SessionModelLoad"), [WeakState, Model, Id]()
		{
			TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Executor.SessionModelLoad");
			const FPipelineResult LoadResult = Model->Initialize();
			FinishSessionLoad(WeakState, Id, LoadResult);
		}, UE::Tasks::ETaskPriority::BackgroundNormal);
	}

	void BeginSession(TUniquePtr<FExecutorJob> Job)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Executor.SessionBegin");
		Job->bBegun = true;
		Job->BeginSeconds = FPlatformTime::Seconds();
		const FPipelineResult BeginResult = Job->Session->Begin();
		if (!BeginResult.bSuccess)
		{
			EndSession(MoveTemp(Job), EPipelineSessionEnd::Failed, BeginResult);
			return;
		}
		Requeue(MoveTemp(Job));
	}

	void StepSession(TUniquePtr<FExecutorJob> Job)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Executor.SessionStep");
		TArray<FPipelineSessionProgress> Progress;
		FPipelineResult StepResult = FPipelineResult::Ok();
		const EPipelineStep Outcome = Job->Session->Step(Progress, StepResult);
		PostProgress(*Job, Progress);

		switch (Outcome)
		{
		case EPipelineStep::Continue:
			Requeue(MoveTemp(Job));
			break;
		case EPipelineStep::Finished:
			EndSession(MoveTemp(Job), EPipelineSessionEnd::Finished, FPipelineResult::Ok());
			break;
		default:
			if (StepResult.bSuccess)
			{
				StepResult = FPipelineResult::Fail(EPipelineError::InferenceFailed, TEXT("Executor: the session step failed without an error."));
			}
			EndSession(MoveTemp(Job), EPipelineSessionEnd::Failed, StepResult);
			break;
		}
	}

	/** Worker. Numbers the step's progress items and hands them to the waiter or the game-thread dispatch. */
	void PostProgress(FExecutorJob& Job, TArray<FPipelineSessionProgress>& Progress)
	{
		if (Progress.Num() == 0)
		{
			return;
		}
		FSessionControl& Ctl = *Job.Ctl;
		if (Ctl.Waiter.IsValid())
		{
			for (FPipelineSessionProgress& Item : Progress)
			{
				Item.Sequence = Job.NextSequence++;
				Ctl.Waiter->Progress.Add(MoveTemp(Item));
			}
			return;
		}
		if (Ctl.bCancelled.load())
		{
			for (FPipelineSessionProgress& Item : Progress)
			{
				Item.Sequence = Job.NextSequence++; // dropped: undelivered progress of a cancelled session
			}
			return;
		}

		bool bSchedule = false;
		{
			FScopeLock Guard(&State->Lock);
			for (FPipelineSessionProgress& Item : Progress)
			{
				Item.Sequence = Job.NextSequence++;
				FDispatchItem Dispatch;
				Dispatch.Ctl = Job.Ctl;
				Dispatch.Progress = MoveTemp(Item);
				bSchedule |= AddDispatchLocked(*State, MoveTemp(Dispatch));
			}
		}
		if (bSchedule)
		{
			ScheduleDispatch(FStateWeakPtr(State));
		}
	}

	/** Worker. After Begin or a Step: back to the end of the queue (C61 rule 3), unless cancelled or stopping. */
	void Requeue(TUniquePtr<FExecutorJob> Job)
	{
		bool bStop = false;
		bool bCancelled = false;
		{
			FScopeLock Guard(&State->Lock);
			if (State->bStopping)
			{
				bStop = true;
			}
			else if (Job->Ctl->bCancelled.load())
			{
				bCancelled = true;
			}
			else
			{
				State->NumRunning = 0;
				State->RunningSessionCtl.Reset();
				State->Queue.Add(MoveTemp(Job));
			}
		}
		if (bStop)
		{
			DropSessionForShutdown(State, MoveTemp(Job), true);
		}
		else if (bCancelled)
		{
			EndSession(MoveTemp(Job), EPipelineSessionEnd::Cancelled, FPipelineResult());
		}
	}

	/** Worker. Calls End (if Begin ran) and completes the session. Result: the failure for Failed; ignored otherwise. */
	void EndSession(TUniquePtr<FExecutorJob> Job, EPipelineSessionEnd Reason, const FPipelineResult& Result)
	{
		bool bStopping = false;
		{
			FScopeLock Guard(&State->Lock);
			bStopping = State->bStopping;
			if (Reason == EPipelineSessionEnd::Finished || Reason == EPipelineSessionEnd::Failed)
			{
				FSessionControl& Ctl = *Job->Ctl;
				if (Ctl.bCancelled.load())
				{
					// A Cancel that returned true landed before this point (C61 rule 1): the session ends Cancelled.
					Reason = EPipelineSessionEnd::Cancelled;
				}
				else
				{
					// From here on Cancel returns false for this id (RequestCancelLocked checks bEnding), so the
					// delivered result is the Finished/Failed one.
					Ctl.bEnding = true;
					State->RunningSessionCtl.Reset();
				}
			}
		}
		if (bStopping && Reason != EPipelineSessionEnd::Failed && Reason != EPipelineSessionEnd::Finished)
		{
			DropSessionForShutdown(State, MoveTemp(Job), true);
			return;
		}

		FPipelineResult Final = FPipelineResult::Ok();
		if (Job->bBegun && Job->Session.IsValid())
		{
			Job->Session->End(Reason, Final);
		}

		switch (Reason)
		{
		case EPipelineSessionEnd::Finished:
			if (!Final.bSuccess)
			{
				Final = FPipelineResult::Ok();
			}
			if (Final.Device == EPipelineDevice::Auto && Job->SessionModel.IsValid())
			{
				Final.Device = Job->SessionModel->GetDevice();
			}
			if (Final.InferenceMs <= 0.f)
			{
				Final.InferenceMs = MsSince(Job->BeginSeconds);
			}
			Job->Result = Final;
			break;
		case EPipelineSessionEnd::Failed:
			Job->Result = Result;
			break;
		default:
			Job->Result = FPipelineResult::Fail(EPipelineError::Cancelled, FString::Printf(TEXT("Executor: session %llu was cancelled."), (unsigned long long)Job->Info.Id));
			break;
		}
		CompleteSession(State, MoveTemp(Job), true);
	}

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

FPipelineJobId FPipelineExecutor::EnqueueSession(TSharedPtr<IPipelineSession> Session, FPipelineSessionCallbacks Callbacks,
	const UObject* Owner, int32 Priority)
{
	const uint64 StartCycles = FPlatformTime::Cycles64();

	TUniquePtr<FExecutorJob> Job = MakeUnique<FExecutorJob>();
	Job->bIsSession = true;
	if (Session.IsValid())
	{
		Job->SessionModel = Session->GetModel();
	}
	Job->Session = MoveTemp(Session);
	Job->Ctl = MakeShared<FSessionControl, ESPMode::ThreadSafe>();
	Job->Ctl->Callbacks = MoveTemp(Callbacks);
	Job->Ctl->bHasOwner = Owner != nullptr;
	Job->Ctl->Owner = Owner;
	Job->Info.Priority = Priority;
	Job->Info.EnqueueTimeSeconds = FPlatformTime::Seconds();

	FPipelineJobId Id = 0;
	{
		FScopeLock Guard(&State->Lock);
		if (State->bStopping)
		{
			return 0; // Job (and its callbacks) is destroyed after the lock is released
		}
		Id = State->AllocateIdLocked();
		Job->Info.Id = Id;
		Job->Ctl->Id = Id;
		++State->NumSessionsLive;
		State->Queue.Add(MoveTemp(Job));
	}
	State->WakeEvent->Trigger();

	const double Ms = CyclesToMs(FPlatformTime::Cycles64() - StartCycles);
	FScopeLock Guard(&State->Lock);
	State->Stats.MaxEnqueueMs = FMath::Max(State->Stats.MaxEnqueueMs, Ms);
	return Id;
}

FPipelineResult FPipelineExecutor::RunSessionBlocking(TSharedPtr<IPipelineSession> Session, TArray<FPipelineSessionProgress>& OutProgress)
{
	OutProgress.Reset();
	if (State->WorkerThreadId != 0 && FPlatformTLS::GetCurrentThreadId() == State->WorkerThreadId)
	{
		// The session would queue behind the running one (this call) and never start.
		return FPipelineResult::Fail(EPipelineError::Internal, TEXT("RunSessionBlocking called from the executor worker thread"));
	}
	if (!Session.IsValid())
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("Executor: the session is null."));
	}

	const TSharedPtr<FBlockingWaiter, ESPMode::ThreadSafe> Waiter = MakeShared<FBlockingWaiter, ESPMode::ThreadSafe>();
	TUniquePtr<FExecutorJob> Job = MakeUnique<FExecutorJob>();
	Job->bIsSession = true;
	Job->SessionModel = Session->GetModel();
	Job->Session = MoveTemp(Session);
	Job->Ctl = MakeShared<FSessionControl, ESPMode::ThreadSafe>();
	Job->Ctl->Waiter = Waiter;
	Job->Info.EnqueueTimeSeconds = FPlatformTime::Seconds();

	{
		FScopeLock Guard(&State->Lock);
		if (State->bStopping)
		{
			return FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("Executor: shut down."));
		}
		Job->Info.Id = State->AllocateIdLocked();
		Job->Ctl->Id = Job->Info.Id;
		++State->NumSessionsLive;
		State->Queue.Add(MoveTemp(Job));
	}
	State->WakeEvent->Trigger();

	// Released by the worker when the session ends, or by Shutdown with Cancelled.
	Waiter->Event->Wait();
	OutProgress = MoveTemp(Waiter->Progress);
	return Waiter->Result;
}

bool FPipelineExecutor::Cancel(FPipelineJobId Id)
{
	if (Id == 0)
	{
		return false;
	}

	TUniquePtr<FExecutorJob> Job;
	TSharedPtr<IPipelineModel> LoadingModel;
	bool bRunningSessionCancelled = false;
	{
		FScopeLock Guard(&State->Lock);
		const int32 Index = State->Queue.IndexOfByPredicate([Id](const TUniquePtr<FExecutorJob>& Queued)
		{
			return Queued->Info.Id == Id;
		});
		if (Index != INDEX_NONE)
		{
			FExecutorJob& Queued = *State->Queue[Index];
			if (Queued.bIsSession && (Queued.bBegun || Queued.bHolding))
			{
				// A running session between two steps: the worker ends it (End(Cancelled)).
				bRunningSessionCancelled = State->RequestCancelLocked(*Queued.Ctl, LoadingModel);
				if (!bRunningSessionCancelled)
				{
					return false;
				}
			}
			else
			{
				Job = MoveTemp(State->Queue[Index]);
				State->Queue.RemoveAt(Index);
				++State->Stats.NumCancelled;
				if (Job->bIsSession)
				{
					Job->Ctl->bCancelled.store(true);
				}
			}
		}
		else
		{
			FSessionControlPtr Ctl;
			for (const TUniquePtr<FExecutorJob>& Loaded : State->Loading)
			{
				if (Loaded->Info.Id == Id)
				{
					Ctl = Loaded->Ctl;
					break;
				}
			}
			if (!Ctl.IsValid() && State->RunningSessionCtl.IsValid() && State->RunningSessionCtl->Id == Id)
			{
				Ctl = State->RunningSessionCtl;
			}
			if (!Ctl.IsValid() || !State->RequestCancelLocked(*Ctl, LoadingModel))
			{
				return false; // a running one-shot job, a finished or unknown id, or already cancelled
			}
			bRunningSessionCancelled = true;
		}
	}

	if (bRunningSessionCancelled)
	{
		AfterCancelRequest(State, LoadingModel);
		return true;
	}

	if (Job->bIsSession)
	{
		Job->Result = FPipelineResult::Fail(EPipelineError::Cancelled, FString::Printf(TEXT("Executor: session %llu was cancelled while queued."), (unsigned long long)Id));
		Job->QueueMs = MsSince(Job->Info.EnqueueTimeSeconds);
		CompleteSession(State, MoveTemp(Job), false);
		return true;
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
	int32 Num = State->Queue.Num();
	if (State->NumSessionsLive > 0)
	{
		for (const TUniquePtr<FExecutorJob>& Queued : State->Queue)
		{
			// A running session (begun, or holding its loaded model) counts as pending, not queued.
			if (Queued->bIsSession && (Queued->bBegun || Queued->bHolding))
			{
				--Num;
			}
		}
	}
	return Num;
}

int32 FPipelineExecutor::GetNumPending() const
{
	FScopeLock Guard(&State->Lock);
	return State->Queue.Num() + State->NumRunning + State->NumLoading + State->NumAwaitingDispatch;
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

	// 1. Stop accepting. 2. Take the queued jobs. Sessions that hold their model stay for the worker, which calls
	// End(Shutdown) on them. Sessions whose model loads are dropped now (the load task only releases references).
	TArray<TUniquePtr<FExecutorJob>> Dropped;
	TArray<TUniquePtr<FExecutorJob>> DroppedLoading;
	{
		FScopeLock Guard(&State->Lock);
		State->bStopping = true;
		if (State->NumSessionsLive == 0)
		{
			Swap(Dropped, State->Queue);
		}
		else
		{
			TArray<TUniquePtr<FExecutorJob>> Kept;
			for (TUniquePtr<FExecutorJob>& Queued : State->Queue)
			{
				if (Queued->bIsSession && (Queued->bBegun || Queued->bHolding) && Thread != nullptr)
				{
					Kept.Add(MoveTemp(Queued));
				}
				else
				{
					Dropped.Add(MoveTemp(Queued));
				}
			}
			State->Queue = MoveTemp(Kept);
			Swap(DroppedLoading, State->Loading);
			State->NumLoading = 0;
		}
	}

	// 2. Async callbacks are dropped without being called; RunBlocking waiters are released with Cancelled.
	for (TUniquePtr<FExecutorJob>& Job : Dropped)
	{
		const TSharedPtr<FBlockingWaiter, ESPMode::ThreadSafe> Waiter = Job->bIsSession ? Job->Ctl->Waiter : Job->Waiter;
		if (Waiter.IsValid())
		{
			Waiter->Result = FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("Executor: shut down while the job was queued."));
			Waiter->Event->Trigger();
		}
	}
	for (TUniquePtr<FExecutorJob>& Job : DroppedLoading)
	{
		if (Job->SessionModel.IsValid())
		{
			Job->SessionModel->CancelInitialize();
		}
		if (Job->Ctl->Waiter.IsValid())
		{
			Job->Ctl->Waiter->Result = FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("Executor: shut down while the session's model loaded."));
			Job->Ctl->Waiter->Event->Trigger();
		}
	}
	if (Dropped.Num() + DroppedLoading.Num() > 0)
	{
		UE_LOG(LogModelPipelines, Verbose, TEXT("Executor: shutdown dropped %d queued job(s)."), Dropped.Num() + DroppedLoading.Num());
	}
	Dropped.Reset();
	DroppedLoading.Reset();

	// 3./4. Wake the worker; it finishes the running job or session step, ends the sessions left in the queue, sees
	// bStopping and exits. Join it.
	State->WakeEvent->Trigger();
	if (Thread != nullptr)
	{
		Thread->Kill(true);
		delete Thread;
		Thread = nullptr;
	}

	State->bShutdownDone = true;
}
