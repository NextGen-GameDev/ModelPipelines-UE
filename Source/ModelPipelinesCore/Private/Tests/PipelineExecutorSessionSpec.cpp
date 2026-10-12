// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP8L.3b acceptance 1-13: executor sessions (C61). Written from the brief and contracts only, without the
// implementation. Every test uses a private FPipelineExecutor so stats, pending counts and Shutdown are isolated.
//
// Timing margins (all chosen so a busy laptop does not flake, but a wrong implementation still fails):
//  - step counts after Cancel are compared with the Step count AT the Cancel call (+1), never with wall-clock;
//  - cancel -> OnComplete(Cancelled): brief says 200 ms, tested at 400 ms (steps 50 ms; an implementation that ignored
//    the flag would run the 20 steps = 1000 ms);
//  - Shutdown with a 100 ms step: < 1 s (brief); Shutdown with a 1 s load: < 500 ms (brief says 200 ms; waiting for
//    the load would take >= 1000 ms);
//  - interleaving is checked on the worker's own order (trace), with 2 steps of slack for scheduling races;
//  - dispatch cost: MaxGameThreadDispatchMs < 2.0 ms exactly as the brief (M1 limit).

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Containers/Ticker.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeBool.h"
#include "HAL/ThreadSafeCounter.h"
#include "Misc/ScopeLock.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/GarbageCollection.h"
#include "NNEModelData.h"
#include "PipelineExecutor.h"
#include "PipelineSession.h"
#include "PipelineBackend.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "PipelineTestSession.h"

using namespace UE::ModelPipelines::Tests::Session;

namespace
{
	const FTimespan SessLatentLimit = FTimespan::FromSeconds(90);
	const double SessPollTimeout = 20.0;

	struct FSessProg
	{
		int32 Seq = -1;
		FString Text;
		TArray<int32> Ids;
		bool bGameThread = false;
	};

	/** Everything the game-thread callbacks of one session saw. Touched only on the game thread. */
	struct FSessLog
	{
		FPipelineJobId Id = 0;
		TArray<FSessProg> Progress;
		int32 Completes = 0;
		FPipelineResult Result;
		bool bCompleteOnGameThread = false;
		double CompleteTime = 0.0;
		int32 ProgressAfterComplete = 0;
		int32 ProgressAfterCancel = 0;
		bool bCancelReturned = false;
		bool bCancelResult = false;
		double CancelTime = 0.0;
		int32 StepsAtCancel = 0;
		bool bInCall = false;              // true while the test is inside EnqueueSession / Cancel on the game thread
		int32 CallbacksInsideCall = 0;     // callbacks that ran inside such a call (forbidden)
		bool bFlag = false;                // free-form observation taken at OnComplete
		TFunction<void(const FPipelineSessionProgress&)> OnProgressHook;
		TFunction<void()> OnCompleteHook;
	};

	struct FSessJobLog
	{
		int32 Completes = 0;
		FPipelineResult Result;
		bool bGameThread = false;
		bool bFlag = false;
		TFunction<void()> Hook;
	};

	FPipelineSessionCallbacks SessMakeCallbacks(const TSharedRef<FSessLog>& L)
	{
		FPipelineSessionCallbacks C;
		C.OnProgress = [L](const FPipelineSessionProgress& P)
		{
			if (L->bInCall) { ++L->CallbacksInsideCall; }
			if (L->Completes > 0) { ++L->ProgressAfterComplete; }
			if (L->bCancelReturned) { ++L->ProgressAfterCancel; }
			FSessProg R;
			R.Seq = P.Sequence;
			R.Text = P.Text;
			R.Ids = P.Ids;
			R.bGameThread = IsInGameThread();
			L->Progress.Add(MoveTemp(R));
			if (L->OnProgressHook) { L->OnProgressHook(P); }
		};
		C.OnComplete = [L](const FPipelineResult& R)
		{
			if (L->bInCall) { ++L->CallbacksInsideCall; }
			++L->Completes;
			L->Result = R;
			L->bCompleteOnGameThread = IsInGameThread();
			L->CompleteTime = FPlatformTime::Seconds();
			if (L->OnCompleteHook) { L->OnCompleteHook(); }
		};
		return C;
	}

	TArray<FPipelineTensor> SessOne(float V)
	{
		TArray<FPipelineTensor> R;
		R.Add(FPipelineTensor::FromFloat(TArray<int64>{1}, TArray<float>{V}));
		return R;
	}

	/** Picks the LAST pending entry and records every snapshot (time + copy of Pending). */
	class FSessLastPickScheduler : public IPipelineScheduler
	{
	public:
		struct FSnap
		{
			double Time = 0.0;
			TArray<FPipelineJobInfo> Items;
		};

		virtual int32 SelectNext(TConstArrayView<FPipelineJobInfo> Pending) override
		{
			FSnap S;
			S.Time = FPlatformTime::Seconds();
			S.Items.Append(Pending.GetData(), Pending.Num());
			if (IsInGameThread()) { OnGameThread.Increment(); }
			{
				FScopeLock L(&Lock);
				Snaps.Add(MoveTemp(S));
			}
			return Pending.Num() - 1;
		}

		TArray<FSnap> GetSnaps()
		{
			FScopeLock L(&Lock);
			return Snaps;
		}

		FThreadSafeCounter OnGameThread;

	private:
		FCriticalSection Lock;
		TArray<FSnap> Snaps;
	};

	/** Result slot for a RunSessionBlocking call made on another thread. */
	struct FSessBlockRes
	{
		FThreadSafeBool bDone;
		FPipelineResult R;
		TArray<FPipelineSessionProgress> P;
		double ElapsedSeconds = 0.0;
	};

	/** Shared state of the 8-thread test. */
	struct FSessThreadSet
	{
		FCriticalSection Lock;
		TArray<TSharedRef<FSessLog>> Logs;
		TArray<TSharedRef<FScriptedSession>> Sessions;
		TArray<FPipelineJobId> Ids;
		FThreadSafeCounter ThreadsDone;
		FThreadSafeCounter Completed;
	};
}

BEGIN_DEFINE_SPEC(FPipelineExecutorSessionSpec, "Pipelines.Core.ExecutorSession",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	TArray<TSharedPtr<FFakeSessionModel>> PrivModels;
	TUniquePtr<FPipelineExecutor> PrivExec;
	TSharedPtr<FTrace> TraceP;

	FTrace& Tr()
	{
		if (!TraceP.IsValid()) { TraceP = MakeShared<FTrace>(); }
		return *TraceP;
	}

	FPipelineExecutor& NewExec()
	{
		PrivExec = MakeUnique<FPipelineExecutor>();
		return *PrivExec;
	}

	TSharedRef<FFakeSessionModel> NewModel(const FString& Name, bool bPreInitialized = true)
	{
		Tr();
		TSharedRef<FFakeSessionModel> M = MakeShared<FFakeSessionModel>(Name, TraceP.ToSharedRef(), bPreInitialized);
		PrivModels.Add(M);
		return M;
	}

	TSharedRef<FScriptedSession> NewSess(const FString& Name, TSharedRef<FFakeSessionModel> M, int32 Steps, int32 StepMs)
	{
		Tr();
		return MakeShared<FScriptedSession>(Name, M, TraceP.ToSharedRef(), Steps, StepMs);
	}

	/** EnqueueSession with callback-inside-call detection. */
	FPipelineJobId EnqSess(FPipelineExecutor& E, TSharedPtr<IPipelineSession> S, const TSharedRef<FSessLog>& Log,
		const UObject* Owner = nullptr, int32 Priority = 0)
	{
		Log->bInCall = true;
		Log->Id = E.EnqueueSession(S, SessMakeCallbacks(Log), Owner, Priority);
		Log->bInCall = false;
		return Log->Id;
	}

	/** One-shot job whose input tag shows up in the trace as runtag:<model>:<Tag>. */
	FPipelineJobId EnqJob(FPipelineExecutor& E, TSharedRef<FFakeSessionModel> M, int32 Tag, const TSharedRef<FSessJobLog>& Log)
	{
		return E.Enqueue(M, SessOne((float)Tag), [Log](const FPipelineResult& R, TArray<FPipelineTensor>&&)
		{
			++Log->Completes;
			Log->Result = R;
			Log->bGameThread = IsInGameThread();
			if (Log->Hook) { Log->Hook(); }
		});
	}

	/** Calls Then(false) on the game thread once Cond() is true, or Then(true) after TimeoutSeconds. Never hangs. */
	void PollUntil(TFunction<bool()> Cond, double TimeoutSeconds, TFunction<void(bool)> Then)
	{
		const double Deadline = FPlatformTime::Seconds() + TimeoutSeconds;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("WP8L3bPoll"), 0.005f, [Cond, Then, Deadline](float)
		{
			if (Cond()) { Then(false); return false; }
			if (FPlatformTime::Seconds() > Deadline) { Then(true); return false; }
			return true;
		});
	}

	void Settle(double Seconds, TFunction<void()> Then)
	{
		PollUntil([]() { return false; }, Seconds, [Then](bool) { Then(); });
	}

	void VerifyProgress(const FSessLog& L, int32 N, const FString& What)
	{
		TestEqual(What + TEXT(": progress count"), L.Progress.Num(), N);
		int32 Reported = 0;
		for (int32 i = 0; i < L.Progress.Num(); ++i)
		{
			const FSessProg& P = L.Progress[i];
			const FString Want = FString::Printf(TEXT("s%d"), i);
			const bool bOk = P.Seq == i && P.Text == Want && P.Ids.Num() == 1 && P.Ids[0] == i && P.bGameThread;
			if (!bOk && Reported++ < 3)
			{
				AddError(FString::Printf(TEXT("%s: progress #%d: expected Sequence %d Text '%s' Ids [%d] on game thread; got Sequence %d Text '%s' Ids %d GT %d"),
					*What, i, i, *Want, i, P.Seq, *P.Text, P.Ids.Num(), P.bGameThread ? 1 : 0));
			}
		}
	}

	void VerifyOk(const FSessLog& L, int32 N, const FString& What)
	{
		VerifyProgress(L, N, What);
		TestEqual(What + TEXT(": OnComplete exactly once"), L.Completes, 1);
		TestTrue(*(What + FString::Printf(TEXT(": result success (error %d '%s')"), (int32)L.Result.Error, *L.Result.Message)), L.Result.bSuccess);
		TestTrue(What + TEXT(": OnComplete on the game thread"), L.bCompleteOnGameThread);
		TestEqual(What + TEXT(": no OnProgress after OnComplete"), L.ProgressAfterComplete, 0);
		TestEqual(What + TEXT(": no callback inside EnqueueSession/Cancel"), L.CallbacksInsideCall, 0);
		TestTrue(What + TEXT(": QueueMs and InferenceMs non-negative"), L.Result.QueueMs >= 0.f && L.Result.InferenceMs >= 0.f);
	}

	TSharedRef<FSessJobLog> NewJobLog() { return MakeShared<FSessJobLog>(); }

END_DEFINE_SPEC(FPipelineExecutorSessionSpec)

void FPipelineExecutorSessionSpec::Define()
{
	AfterEach([this]()
	{
		for (const TSharedPtr<FFakeSessionModel>& M : PrivModels) { M->OpenGate(); }
		PrivExec.Reset(); // Shutdown + join; every gate is open and every fake step is short
		PrivModels.Reset();
		TraceP.Reset();
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Happy path (acceptance 1)", [this]()
	{
		LatentIt("5-step session: progress 0..4 'sN' in order on the game thread, then one OnComplete Ok; pending 1 then 0; NumCompleted +1", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("A"), M, 5, 40);
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			E.ResetStats();
			FPipelineExecutor* EP = &E;
			const TSharedRef<int32> PendingAtFirstProgress = MakeShared<int32>(-1);
			Log->OnProgressHook = [EP, PendingAtFirstProgress](const FPipelineSessionProgress& P)
			{
				if (P.Sequence == 0) { *PendingAtFirstProgress = EP->GetNumPending(); }
			};

			const FPipelineJobId Id = EnqSess(E, S, Log);
			TestTrue(TEXT("EnqueueSession returns an id > 0"), Id > 0);
			// 5 x 40 ms is running or queued: a session counts as pending (not only queued) until its OnComplete dispatch.
			TestEqual(TEXT("GetNumPending right after EnqueueSession"), E.GetNumPending(), 1);

			PollUntil([Log]() { return Log->Completes >= 1; }, SessPollTimeout, [this, Done, Log, S, EP, PendingAtFirstProgress](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: OnComplete never arrived for a 5-step session")); Done.ExecuteIfBound(); return; }
				// Settle: a duplicated OnComplete or a late progress item would show up here.
				Settle(0.4, [this, Done, Log, S, EP, PendingAtFirstProgress]()
				{
					VerifyOk(*Log, 5, TEXT("happy path"));
					TestEqual(TEXT("GetNumPending while the session still ran (at progress 0)"), *PendingAtFirstProgress, 1);
					TestEqual(TEXT("GetNumPending afterwards"), EP->GetNumPending(), 0);
					TestEqual(TEXT("GetNumQueued afterwards"), EP->GetNumQueued(), 0);
					TestEqual(TEXT("NumCompleted counts the session once"), (int32)EP->GetStats().NumCompleted, 1);
					TestEqual(TEXT("NumCancelled"), (int32)EP->GetStats().NumCancelled, 0);
					TestEqual(TEXT("Begin called once"), S->BeginCalls.GetValue(), 1);
					TestEqual(TEXT("Step called 5 times"), S->StepCalls.GetValue(), 5);
					TestEqual(TEXT("End called once"), S->EndCalls.GetValue(), 1);
					TestEqual(TEXT("End reason Finished"), S->LastEndReason, (int32)EPipelineSessionEnd::Finished);
					Done.ExecuteIfBound();
				});
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Failures (acceptance 2)", [this]()
	{
		LatentIt("Begin fails: OnComplete carries the error, no progress, no Step, End(Failed) once", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("A"), M, 5, 10);
			S->bFailBegin = true;
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			EnqSess(E, S, Log);
			PollUntil([Log]() { return Log->Completes >= 1; }, SessPollTimeout, [this, Done, Log, S](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: OnComplete never arrived after a failing Begin")); Done.ExecuteIfBound(); return; }
				Settle(0.3, [this, Done, Log, S]()
				{
					TestEqual(TEXT("OnComplete exactly once"), Log->Completes, 1);
					TestFalse(TEXT("result is a failure"), Log->Result.bSuccess);
					TestEqual(TEXT("error code of Begin's failure"), (int32)Log->Result.Error, (int32)EPipelineError::InferenceFailed);
					TestTrue(TEXT("message of Begin's failure preserved"), Log->Result.Message.Contains(TEXT("fake begin failure")));
					TestEqual(TEXT("no OnProgress"), Log->Progress.Num(), 0);
					TestEqual(TEXT("Step never called"), S->StepCalls.GetValue(), 0);
					TestEqual(TEXT("End called once (Begin ran)"), S->EndCalls.GetValue(), 1);
					TestEqual(TEXT("End reason Failed"), S->LastEndReason, (int32)EPipelineSessionEnd::Failed);
					TestTrue(TEXT("callback on the game thread"), Log->bCompleteOnGameThread);
					Done.ExecuteIfBound();
				});
			});
		});

		LatentIt("Step fails on its 4th call: progress 0..2 delivered, OnComplete with the error, End(Failed) once", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("A"), M, 10, 10);
			S->FailAtStep = 3; // calls 0,1,2 produce progress; call 3 returns Failed
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			EnqSess(E, S, Log);
			PollUntil([Log]() { return Log->Completes >= 1; }, SessPollTimeout, [this, Done, Log, S](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: OnComplete never arrived after a failing Step")); Done.ExecuteIfBound(); return; }
				Settle(0.3, [this, Done, Log, S]()
				{
					VerifyProgress(*Log, 3, TEXT("failing step"));
					TestEqual(TEXT("OnComplete exactly once"), Log->Completes, 1);
					TestFalse(TEXT("result is a failure"), Log->Result.bSuccess);
					TestEqual(TEXT("error code"), (int32)Log->Result.Error, (int32)EPipelineError::InferenceFailed);
					TestTrue(TEXT("message preserved"), Log->Result.Message.Contains(TEXT("fake step failure")));
					TestEqual(TEXT("OnComplete after every progress item (none after)"), Log->ProgressAfterComplete, 0);
					TestEqual(TEXT("Step called 4 times, not more"), S->StepCalls.GetValue(), 4);
					TestEqual(TEXT("End called once"), S->EndCalls.GetValue(), 1);
					TestEqual(TEXT("End reason Failed"), S->LastEndReason, (int32)EPipelineSessionEnd::Failed);
					Done.ExecuteIfBound();
				});
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Cancel (acceptance 3)", [this]()
	{
		LatentIt("queued session: Cancel true, OnComplete(Cancelled) on the game thread, Begin never called, repeat is false", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> G = NewModel(TEXT("G"));
			const TSharedRef<FFakeSessionModel> N = NewModel(TEXT("N"));
			G->CloseGate();
			const TSharedRef<FSessJobLog> Blocker = NewJobLog();
			EnqJob(E, G, 1, Blocker);
			if (!G->WaitRunEntered())
			{
				AddError(TEXT("worker never started the blocking one-shot job"));
				G->OpenGate();
				Done.ExecuteIfBound();
				return;
			}
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("B"), N, 3, 10);
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			const FPipelineJobId Id = EnqSess(E, S, Log); // queued behind the blocked worker
			Log->bInCall = true;
			const bool bFirst = E.Cancel(Id);
			Log->bInCall = false;
			TestTrue(TEXT("Cancel of a queued session returns true"), bFirst);
			G->OpenGate();
			FPipelineExecutor* EP = &E;
			PollUntil([Log, Blocker]() { return Log->Completes >= 1 && Blocker->Completes >= 1; }, SessPollTimeout, [this, Done, Log, S, Id, EP](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: OnComplete(Cancelled) never arrived for a queued session")); Done.ExecuteIfBound(); return; }
				Settle(0.3, [this, Done, Log, S, Id, EP]()
				{
					TestEqual(TEXT("OnComplete exactly once"), Log->Completes, 1);
					TestEqual(TEXT("error code Cancelled"), (int32)Log->Result.Error, (int32)EPipelineError::Cancelled);
					TestTrue(TEXT("callback on the game thread"), Log->bCompleteOnGameThread);
					TestEqual(TEXT("callback not inside EnqueueSession/Cancel"), Log->CallbacksInsideCall, 0);
					TestEqual(TEXT("Begin never called"), S->BeginCalls.GetValue(), 0);
					TestEqual(TEXT("End never called (Begin did not run)"), S->EndCalls.GetValue(), 0);
					TestEqual(TEXT("no progress"), Log->Progress.Num(), 0);
					TestFalse(TEXT("Cancel of a finished id returns false"), EP->Cancel(Id));
					TestEqual(TEXT("NumCancelled counts the session once"), (int32)EP->GetStats().NumCancelled, 1);
					Done.ExecuteIfBound();
				});
			});
		});

		LatentIt("running session cancelled from inside OnProgress(2): true, at most one more Step, no OnProgress after, End(Cancelled) once, OnComplete(Cancelled) within 400 ms", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("A"), M, 20, 50);
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			FPipelineExecutor* EP = &E;
			FSessLog* LP = &Log.Get();
			Log->OnProgressHook = [LP, EP, S](const FPipelineSessionProgress& P)
			{
				if (P.Sequence == 2 && !LP->bCancelReturned)
				{
					LP->StepsAtCancel = S->StepCalls.GetValue();
					LP->bInCall = true;
					LP->bCancelResult = EP->Cancel(LP->Id);
					LP->bInCall = false;
					LP->CancelTime = FPlatformTime::Seconds();
					LP->bCancelReturned = true;
				}
			};
			EnqSess(E, S, Log);
			PollUntil([Log]() { return Log->Completes >= 1; }, SessPollTimeout, [this, Done, Log, S, EP](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: a cancelled running session never completed")); Done.ExecuteIfBound(); return; }
				Settle(0.5, [this, Done, Log, S, EP]()
				{
					TestTrue(TEXT("Cancel of a running session returns true"), Log->bCancelResult);
					TestEqual(TEXT("exactly progress 0..2 delivered (the one that cancelled was the 3rd)"), Log->Progress.Num(), 3);
					// A Step already in flight may finish; a new one may not start after the flag is set (rule 1).
					// Catches an implementation that ignores the flag: it would run all 20 steps.
					TestTrue(*FString::Printf(TEXT("Step calls after Cancel: expected <= %d (count at Cancel + 1), got %d"), Log->StepsAtCancel + 1, S->StepCalls.GetValue()),
						S->StepCalls.GetValue() <= Log->StepsAtCancel + 1);
					TestEqual(TEXT("no OnProgress after Cancel returned"), Log->ProgressAfterCancel, 0);
					TestEqual(TEXT("End called once"), S->EndCalls.GetValue(), 1);
					TestEqual(TEXT("End reason Cancelled"), S->LastEndReason, (int32)EPipelineSessionEnd::Cancelled);
					TestEqual(TEXT("OnComplete exactly once"), Log->Completes, 1);
					TestEqual(TEXT("error code Cancelled"), (int32)Log->Result.Error, (int32)EPipelineError::Cancelled);
					TestEqual(TEXT("callbacks not inside Cancel"), Log->CallbacksInsideCall, 0);
					const double Ms = (Log->CompleteTime - Log->CancelTime) * 1000.0;
					TestTrue(*FString::Printf(TEXT("OnComplete(Cancelled) expected within 400 ms of Cancel (brief: 200), got %.1f ms"), Ms), Ms < 400.0);
					TestEqual(TEXT("NumCancelled counts the session once"), (int32)EP->GetStats().NumCancelled, 1);
					TestEqual(TEXT("NumCompleted counts the session once"), (int32)EP->GetStats().NumCompleted, 1);
					TestFalse(TEXT("Cancel of the finished session returns false"), EP->Cancel(Log->Id));
					TestFalse(TEXT("Cancel(0) returns false"), EP->Cancel(0));
					TestFalse(TEXT("Cancel of an unknown id returns false"), EP->Cancel(987654321ull));
					Done.ExecuteIfBound();
				});
			});
		});

		LatentIt("a running one-shot job still cannot be cancelled", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> G = NewModel(TEXT("G"));
			G->CloseGate();
			const TSharedRef<FSessJobLog> JL = NewJobLog();
			const FPipelineJobId Id = EnqJob(E, G, 1, JL);
			if (!G->WaitRunEntered())
			{
				AddError(TEXT("worker never started the one-shot job"));
				G->OpenGate();
				Done.ExecuteIfBound();
				return;
			}
			TestFalse(TEXT("Cancel of a running one-shot job returns false"), E.Cancel(Id));
			G->OpenGate();
			PollUntil([JL]() { return JL->Completes >= 1; }, SessPollTimeout, [this, Done, JL](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out waiting for the one-shot job")); }
				else { TestTrue(TEXT("the job ran to success"), JL->Result.bSuccess); }
				Done.ExecuteIfBound();
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Interleaving (acceptance 4)", [this]()
	{
		LatentIt("a one-shot job on another model enqueued mid-session waits at most one step (+2 steps of slack)", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> MA = NewModel(TEXT("MA"));
			const TSharedRef<FFakeSessionModel> MJ = NewModel(TEXT("J"));
			const TSharedRef<FScriptedSession> A = NewSess(TEXT("A"), MA, 20, 25);
			const TSharedRef<FSessLog> LogA = MakeShared<FSessLog>();
			EnqSess(E, A, LogA);
			FPipelineExecutor* EP = &E;
			PollUntil([A]() { return A->StepCalls.GetValue() >= 3; }, SessPollTimeout, [this, Done, A, LogA, MJ, EP](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: the session never reached step 3")); Done.ExecuteIfBound(); return; }
				const int32 S0 = A->StepCalls.GetValue();
				const TSharedRef<FSessJobLog> JL = NewJobLog();
				EnqJob(*EP, MJ, 9, JL);
				PollUntil([JL, LogA]() { return JL->Completes >= 1 && LogA->Completes >= 1; }, SessPollTimeout, [this, Done, S0, A, JL, LogA](bool bTimedOut2)
				{
					if (bTimedOut2) { AddError(TEXT("timed out waiting for the one-shot job and the session")); Done.ExecuteIfBound(); return; }
					const int32 IJ = Tr().IndexOf(TEXT("runtag:J:9"));
					const int32 ILimit = Tr().IndexOf(FString::Printf(TEXT("step:A:%d"), S0 + 3));
					TestTrue(TEXT("one-shot job ran"), IJ != INDEX_NONE);
					// Round robin: J runs after the step in flight. Without interleaving it would run after step 19.
					TestTrue(*FString::Printf(TEXT("J must run before A's step %d (steps started at enqueue: %d). Worker order: %s"), S0 + 3, S0, *Tr().Dump()),
						IJ != INDEX_NONE && (ILimit == INDEX_NONE || IJ < ILimit));
					TestTrue(TEXT("one-shot job succeeded"), JL->Result.bSuccess);
					TestTrue(TEXT("the session still finished Ok"), LogA->Result.bSuccess);
					Done.ExecuteIfBound();
				});
			});
		});

		LatentIt("two sessions on different models alternate steps A,B,A,B... once both are queued", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> MA = NewModel(TEXT("MA"));
			const TSharedRef<FFakeSessionModel> MB = NewModel(TEXT("MB"));
			const TSharedRef<FScriptedSession> A = NewSess(TEXT("A"), MA, 10, 30);
			const TSharedRef<FScriptedSession> B = NewSess(TEXT("B"), MB, 10, 30);
			const TSharedRef<FSessLog> LogA = MakeShared<FSessLog>();
			const TSharedRef<FSessLog> LogB = MakeShared<FSessLog>();
			EnqSess(E, A, LogA);
			EnqSess(E, B, LogB);
			PollUntil([LogA, LogB]() { return LogA->Completes >= 1 && LogB->Completes >= 1; }, SessPollTimeout, [this, Done, LogA, LogB](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out waiting for both sessions")); Done.ExecuteIfBound(); return; }
				TArray<TCHAR> Seq;
				for (const FTrace::FEntry& En : Tr().Get())
				{
					if (En.Name.StartsWith(TEXT("step:A:"))) { Seq.Add(TEXT('A')); }
					else if (En.Name.StartsWith(TEXT("step:B:"))) { Seq.Add(TEXT('B')); }
				}
				FString Order;
				for (TCHAR C : Seq) { Order.AppendChar(C); }
				TestEqual(TEXT("20 steps ran in total"), Seq.Num(), 20);
				int32 FirstB = Seq.Find(TEXT('B'));
				TestTrue(*FString::Printf(TEXT("B must get a turn within the first 4 steps; order %s"), *Order), FirstB != INDEX_NONE && FirstB <= 3);
				int32 CountA = 0, CountB = 0;
				bool bAlternates = FirstB != INDEX_NONE;
				for (int32 i = 0; i < Seq.Num() && bAlternates; ++i)
				{
					(Seq[i] == TEXT('A') ? CountA : CountB)++;
					if (i >= FirstB && CountA < 10 && CountB < 10 && i + 1 < Seq.Num() && Seq[i + 1] == Seq[i])
					{
						bAlternates = false;
					}
				}
				// One long job running to completion first (A,A,A,...,B,B,B) fails here.
				TestTrue(*FString::Printf(TEXT("steps alternate while both sessions are live; order %s"), *Order), bAlternates);
				Done.ExecuteIfBound();
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Exclusive model (acceptance 5)", [this]()
	{
		LatentIt("jobs on the session's model wait for End (then FIFO); a job on another model runs meanwhile", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FFakeSessionModel> N = NewModel(TEXT("N"));
			const TSharedRef<FScriptedSession> A = NewSess(TEXT("A"), M, 8, 40);
			const TSharedRef<FScriptedSession> B = NewSess(TEXT("B"), M, 3, 10);
			const TSharedRef<FSessLog> LogA = MakeShared<FSessLog>();
			const TSharedRef<FSessLog> LogB = MakeShared<FSessLog>();
			EnqSess(E, A, LogA);
			PollUntil([A]() { return A->BeginCalls.GetValue() >= 1; }, SessPollTimeout, [this, Done, &E, M, N, A, B, LogA, LogB](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: session A never began")); Done.ExecuteIfBound(); return; }
				const TSharedRef<FSessJobLog> X = NewJobLog(), Z = NewJobLog(), Y = NewJobLog();
				EnqJob(E, M, 1, X);          // queued behind A on the same model
				EnqSess(E, B, LogB);         // second session on M
				EnqJob(E, M, 2, Z);          // queued behind B
				EnqJob(E, N, 3, Y);          // other model: must not be blocked
				PollUntil([LogA, LogB, X, Y, Z]() { return LogA->Completes >= 1 && LogB->Completes >= 1 && X->Completes >= 1 && Y->Completes >= 1 && Z->Completes >= 1; },
					SessPollTimeout, [this, Done, M, A, B, LogA, LogB, X, Y, Z](bool bTimedOut2)
				{
					if (bTimedOut2) { AddError(FString::Printf(TEXT("timed out; worker order: %s"), *Tr().Dump())); Done.ExecuteIfBound(); return; }
					const int32 EndA = Tr().IndexOf(TEXT("end:A")), RunX = Tr().IndexOf(TEXT("runtag:M:1"));
					const int32 BeginB = Tr().IndexOf(TEXT("begin:B")), EndB = Tr().IndexOf(TEXT("end:B"));
					const int32 RunZ = Tr().IndexOf(TEXT("runtag:M:2")), RunY = Tr().IndexOf(TEXT("runtag:N:3"));
					const FString Dump = Tr().Dump();
					TestTrue(*(TEXT("job on another model runs while A is running. Order: ") + Dump), RunY != INDEX_NONE && EndA != INDEX_NONE && RunY < EndA);
					TestTrue(*(TEXT("X (one-shot on M) starts only after A's End. Order: ") + Dump), RunX != INDEX_NONE && RunX > EndA);
					TestTrue(*(TEXT("FIFO after End: X before B. Order: ") + Dump), BeginB != INDEX_NONE && RunX < BeginB);
					TestTrue(*(TEXT("Z starts only after B's End (B holds M too). Order: ") + Dump), RunZ != INDEX_NONE && EndB != INDEX_NONE && RunZ > EndB);
					TestEqual(TEXT("no one-shot Run overlapped a session on M"), M->RunDuringSession.GetValue(), 0);
					TestEqual(TEXT("no two sessions held M at once"), M->SessionOverlap.GetValue(), 0);
					TestTrue(TEXT("all results Ok"), LogA->Result.bSuccess && LogB->Result.bSuccess && X->Result.bSuccess && Y->Result.bSuccess && Z->Result.bSuccess);
					VerifyProgress(*LogA, 8, TEXT("A"));
					VerifyProgress(*LogB, 3, TEXT("B"));
					Done.ExecuteIfBound();
				});
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Scheduler (acceptance 6)", [this]()
	{
		LatentIt("a last-entry scheduler sees the running session (its id, priority, enqueue time) and never a job blocked by the exclusive model", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FSessLastPickScheduler> Sched = MakeShared<FSessLastPickScheduler>();
			E.SetScheduler(Sched);
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FFakeSessionModel> N = NewModel(TEXT("N"));
			const TSharedRef<FScriptedSession> A = NewSess(TEXT("A"), M, 8, 40);
			const TSharedRef<FScriptedSession> B = NewSess(TEXT("B"), N, 6, 40);
			const TSharedRef<FSessLog> LogA = MakeShared<FSessLog>();
			const TSharedRef<FSessLog> LogB = MakeShared<FSessLog>();
			const FPipelineJobId IdA = EnqSess(E, A, LogA, nullptr, 7);
			PollUntil([A]() { return A->BeginCalls.GetValue() >= 1; }, SessPollTimeout, [this, Done, &E, Sched, M, N, A, B, LogA, LogB, IdA](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: session A never began")); Done.ExecuteIfBound(); return; }
				EnqSess(E, B, LogB);
				PollUntil([B]() { return B->BeginCalls.GetValue() >= 1; }, SessPollTimeout, [this, Done, &E, Sched, M, N, A, B, LogA, LogB, IdA](bool bTimedOut2)
				{
					if (bTimedOut2) { AddError(TEXT("timed out: session B never began (the last-pick scheduler starves A, B must still run)")); Done.ExecuteIfBound(); return; }
					const TSharedRef<FSessJobLog> X = NewJobLog(), Q = NewJobLog();
					const FPipelineJobId IdX = EnqJob(E, M, 1, X); // blocked while A holds M
					const FPipelineJobId IdQ = EnqJob(E, N, 2, Q); // blocked while B holds N
					PollUntil([LogA, LogB, X, Q]() { return LogA->Completes >= 1 && LogB->Completes >= 1 && X->Completes >= 1 && Q->Completes >= 1; },
						SessPollTimeout, [this, Done, Sched, A, B, LogA, LogB, IdA, IdX, IdQ](bool bTimedOut3)
					{
						if (bTimedOut3) { AddError(FString::Printf(TEXT("timed out; order: %s"), *Tr().Dump())); Done.ExecuteIfBound(); return; }
						const double BeginA = Tr().TimeOf(TEXT("begin:A")), EndA = Tr().TimeOf(TEXT("end:A"));
						const double EndB = Tr().TimeOf(TEXT("end:B"));
						const TArray<FSessLastPickScheduler::FSnap> Snaps = Sched->GetSnaps();
						int32 SeenARunning = 0, XTooEarly = 0, QTooEarly = 0, BadEntries = 0, WrongPrio = 0;
						double FirstEnqueueTime = -1.0;
						bool bEnqueueTimeStable = true;
						for (const FSessLastPickScheduler::FSnap& S : Snaps)
						{
							for (const FPipelineJobInfo& J : S.Items)
							{
								if (J.Id == 0 || J.EnqueueTimeSeconds <= 0.0) { ++BadEntries; }
								if (J.Id == IdA)
								{
									if (S.Time > BeginA && S.Time < EndA) { ++SeenARunning; }
									if (J.Priority != 7) { ++WrongPrio; }
									if (FirstEnqueueTime < 0.0) { FirstEnqueueTime = J.EnqueueTimeSeconds; }
									else if (FMath::Abs(J.EnqueueTimeSeconds - FirstEnqueueTime) > 1e-9) { bEnqueueTimeStable = false; }
								}
								if (J.Id == IdX && S.Time < EndA) { ++XTooEarly; }
								if (J.Id == IdQ && S.Time < EndB) { ++QTooEarly; }
							}
						}
						TestTrue(*FString::Printf(TEXT("scheduler saw the running session A between Begin and End (snapshots: %d)"), Snaps.Num()), SeenARunning > 0);
						TestEqual(TEXT("a running session keeps its original Priority in Pending"), WrongPrio, 0);
						TestTrue(TEXT("a running session keeps its original EnqueueTime in Pending"), bEnqueueTimeStable);
						TestTrue(TEXT("session's EnqueueTime is not after its Begin"), FirstEnqueueTime > 0.0 && FirstEnqueueTime <= BeginA);
						TestEqual(TEXT("job X on M never shown while A holds M"), XTooEarly, 0);
						TestEqual(TEXT("job Q on N never shown while B holds N"), QTooEarly, 0);
						TestEqual(TEXT("entries have a non-zero id and enqueue time"), BadEntries, 0);
						TestEqual(TEXT("scheduler never ran on the game thread"), Sched->OnGameThread.GetValue(), 0);
						TestTrue(TEXT("everything completed Ok"), LogA->Result.bSuccess && LogB->Result.bSuccess);
						VerifyProgress(*LogA, 8, TEXT("A"));
						VerifyProgress(*LogB, 6, TEXT("B"));
						Done.ExecuteIfBound();
					});
				});
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Owner (acceptance 7)", [this]()
	{
		LatentIt("owner garbage collected mid-session: nothing more delivered, the worker stops with End(Cancelled), the model is released", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("A"), M, 30, 30);
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			UObject* Owner = NewObject<UNNEModelData>(GetTransientPackage());
			EnqSess(E, S, Log, Owner);
			FPipelineExecutor* EP = &E;
			PollUntil([Log]() { return Log->Progress.Num() >= 2; }, SessPollTimeout, [this, Done, Log, S, M, EP, Owner](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: session never delivered 2 progress items")); Done.ExecuteIfBound(); return; }
				UObject* O = Owner;
				O->MarkAsGarbage();
				O = nullptr; // dangling after the collection: never touched again
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
				const int32 Delivered = Log->Progress.Num();
				PollUntil([S]() { return S->EndCalls.GetValue() >= 1; }, 5.0, [this, Done, Log, S, M, EP, Delivered](bool bTimedOut2)
				{
					if (bTimedOut2) { AddError(TEXT("worker never ended the session after its owner died")); Done.ExecuteIfBound(); return; }
					const int32 StepsAtEnd = S->StepCalls.GetValue();
					Settle(0.4, [this, Done, Log, S, M, EP, Delivered, StepsAtEnd]()
					{
						TestEqual(TEXT("no OnProgress delivered after the owner died"), Log->Progress.Num(), Delivered);
						TestEqual(TEXT("OnComplete never delivered"), Log->Completes, 0);
						TestEqual(TEXT("End called once"), S->EndCalls.GetValue(), 1);
						TestEqual(TEXT("End reason Cancelled"), S->LastEndReason, (int32)EPipelineSessionEnd::Cancelled);
						TestTrue(*FString::Printf(TEXT("worker stopped stepping early (steps %d of 30)"), S->StepCalls.GetValue()), S->StepCalls.GetValue() < 30);
						TestEqual(TEXT("no further Step after End"), S->StepCalls.GetValue(), StepsAtEnd);
						// The model must be free again: a one-shot job on it completes, proving the worker is alive too.
						const TSharedRef<FSessJobLog> JL = NewJobLog();
						EnqJob(*EP, M, 5, JL);
						PollUntil([JL]() { return JL->Completes >= 1; }, SessPollTimeout, [this, Done, JL, EP](bool bTimedOut3)
						{
							if (bTimedOut3) { AddError(TEXT("a follow-up job on the session's model never completed (model not released?)")); }
							else
							{
								TestTrue(TEXT("follow-up job succeeded"), JL->Result.bSuccess);
								TestEqual(TEXT("nothing left pending"), EP->GetNumPending(), 0);
							}
							Done.ExecuteIfBound();
						});
					});
				});
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Shutdown (acceptance 8)", [this]()
	{
		LatentIt("Shutdown during a running session returns within 1 s (100 ms steps), End(Shutdown) once, no OnComplete, a queued RunSessionBlocking waiter is released with Cancelled", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FScriptedSession> A = NewSess(TEXT("A"), M, 100, 100);
			const TSharedRef<FScriptedSession> B = NewSess(TEXT("B"), M, 5, 10); // blocked behind A on the same model
			const TSharedRef<FSessLog> LogA = MakeShared<FSessLog>();
			EnqSess(E, A, LogA);
			const TSharedRef<FSessBlockRes> Res = MakeShared<FSessBlockRes>();
			FPipelineExecutor* EP = &E;
			Async(EAsyncExecution::Thread, [EP, B, Res]()
			{
				const double T0 = FPlatformTime::Seconds();
				Res->R = EP->RunSessionBlocking(B, Res->P);
				Res->ElapsedSeconds = FPlatformTime::Seconds() - T0;
				Res->bDone = true;
			});
			PollUntil([EP, A]() { return A->StepCalls.GetValue() >= 2 && EP->GetNumPending() == 2; }, SessPollTimeout,
				[this, Done, EP, A, B, LogA, Res](bool bTimedOut)
			{
				if (bTimedOut)
				{
					AddError(TEXT("timed out: session A never ran 2 steps with the blocking waiter queued"));
					Done.ExecuteIfBound();
					return;
				}
				const double T0 = FPlatformTime::Seconds();
				EP->Shutdown();
				const double Ms = (FPlatformTime::Seconds() - T0) * 1000.0;
				TestTrue(*FString::Printf(TEXT("Shutdown must return after the current step: expected < 1000 ms, got %.1f ms"), Ms), Ms < 1000.0);
				TestEqual(TEXT("End called once"), A->EndCalls.GetValue(), 1);
				TestEqual(TEXT("End reason Shutdown"), A->LastEndReason, (int32)EPipelineSessionEnd::Shutdown);
				TestTrue(TEXT("session stopped early"), A->StepCalls.GetValue() < 100);
				TestEqual(TEXT("queued session B never began"), B->BeginCalls.GetValue(), 0);
				PollUntil([Res]() { return (bool)Res->bDone; }, 5.0, [this, Done, LogA, A, Res](bool bTimedOut2)
				{
					if (bTimedOut2) { AddError(TEXT("RunSessionBlocking waiter was never released by Shutdown")); Done.ExecuteIfBound(); return; }
					TestFalse(TEXT("waiter result is a failure"), Res->R.bSuccess);
					TestEqual(TEXT("waiter released with Cancelled"), (int32)Res->R.Error, (int32)EPipelineError::Cancelled);
					Settle(0.3, [this, Done, LogA, A]()
					{
						TestEqual(TEXT("OnComplete of the running session not called after Shutdown"), LogA->Completes, 0);
						TestEqual(TEXT("End still called exactly once"), A->EndCalls.GetValue(), 1);
						Done.ExecuteIfBound();
					});
				});
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("RunSessionBlocking (acceptance 9)", [this]()
	{
		LatentIt("from a background thread: Ok with 5 progress items in order", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("A"), M, 5, 10);
			const TSharedRef<FSessBlockRes> Res = MakeShared<FSessBlockRes>();
			FPipelineExecutor* EP = &E;
			Async(EAsyncExecution::Thread, [EP, S, Res]()
			{
				Res->R = EP->RunSessionBlocking(S, Res->P);
				Res->bDone = true;
			});
			PollUntil([Res]() { return (bool)Res->bDone; }, SessPollTimeout, [this, Done, Res, S](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("RunSessionBlocking from a background thread never returned")); Done.ExecuteIfBound(); return; }
				TestTrue(*FString::Printf(TEXT("result Ok (error %d '%s')"), (int32)Res->R.Error, *Res->R.Message), Res->R.bSuccess);
				TestEqual(TEXT("5 progress items"), Res->P.Num(), 5);
				for (int32 i = 0; i < Res->P.Num(); ++i)
				{
					TestEqual(*FString::Printf(TEXT("item %d Sequence"), i), Res->P[i].Sequence, i);
					TestEqual(*FString::Printf(TEXT("item %d Text"), i), Res->P[i].Text, FString::Printf(TEXT("s%d"), i));
				}
				TestEqual(TEXT("End called once"), S->EndCalls.GetValue(), 1);
				Done.ExecuteIfBound();
			});
		});

		LatentIt("from the game thread: returns Ok with all items (the worker never waits on the game thread)", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("A"), M, 3, 10);
			TArray<FPipelineSessionProgress> P;
			const FPipelineResult R = E.RunSessionBlocking(S, P);
			TestTrue(*FString::Printf(TEXT("result Ok (error %d '%s')"), (int32)R.Error, *R.Message), R.bSuccess);
			TestEqual(TEXT("3 progress items"), P.Num(), 3);
			for (int32 i = 0; i < P.Num(); ++i) { TestEqual(*FString::Printf(TEXT("item %d Sequence"), i), P[i].Sequence, i); }
			Done.ExecuteIfBound();
		});

		LatentIt("called from inside a Step on the worker: Fail(Internal) at once, the inner session never begins", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FFakeSessionModel> M2 = NewModel(TEXT("M2"));
			const TSharedRef<FScriptedSession> Inner = NewSess(TEXT("Inner"), M2, 3, 10);
			const TSharedRef<FScriptedSession> Outer = NewSess(TEXT("Outer"), M, 2, 10);
			const TSharedRef<FSessBlockRes> Res = MakeShared<FSessBlockRes>();
			FPipelineExecutor* EP = &E;
			Outer->OnStep = [EP, Inner, Res](int32 K)
			{
				if (K == 0)
				{
					const double T0 = FPlatformTime::Seconds();
					Res->R = EP->RunSessionBlocking(Inner, Res->P);
					Res->ElapsedSeconds = FPlatformTime::Seconds() - T0;
					Res->bDone = true;
				}
			};
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			EnqSess(E, Outer, Log);
			PollUntil([Log, Res]() { return Log->Completes >= 1 && (bool)Res->bDone; }, SessPollTimeout, [this, Done, Log, Res, Inner](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: RunSessionBlocking inside a worker Step probably deadlocked, or the outer session never completed")); Done.ExecuteIfBound(); return; }
				TestFalse(TEXT("nested call fails"), Res->R.bSuccess);
				TestEqual(TEXT("nested call error is Internal"), (int32)Res->R.Error, (int32)EPipelineError::Internal);
				TestTrue(*FString::Printf(TEXT("nested call returns at once (took %.1f ms)"), Res->ElapsedSeconds * 1000.0), Res->ElapsedSeconds < 0.5);
				TestEqual(TEXT("inner session never began"), Inner->BeginCalls.GetValue(), 0);
				TestEqual(TEXT("outer session unaffected: Ok"), Log->Result.bSuccess ? 1 : 0, 1);
				Done.ExecuteIfBound();
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Edge arguments (acceptance 10)", [this]()
	{
		LatentIt("null session is accepted (id > 0) and completes later with InvalidArgument", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			const FPipelineJobId Id = EnqSess(E, nullptr, Log);
			TestTrue(TEXT("null session accepted: id > 0"), Id > 0);
			TestEqual(TEXT("no callback inside EnqueueSession"), Log->Completes, 0);
			FPipelineExecutor* EP = &E;
			PollUntil([Log]() { return Log->Completes >= 1; }, SessPollTimeout, [this, Done, Log, EP](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: a null session never completed")); Done.ExecuteIfBound(); return; }
				Settle(0.2, [this, Done, Log, EP]()
				{
					TestEqual(TEXT("exactly one OnComplete"), Log->Completes, 1);
					TestEqual(TEXT("error is InvalidArgument"), (int32)Log->Result.Error, (int32)EPipelineError::InvalidArgument);
					TestTrue(TEXT("on the game thread"), Log->bCompleteOnGameThread);
					TestEqual(TEXT("no progress"), Log->Progress.Num(), 0);
					TestEqual(TEXT("callbacks not inside the call"), Log->CallbacksInsideCall, 0);
					TestEqual(TEXT("nothing pending"), EP->GetNumPending(), 0);
					Done.ExecuteIfBound();
				});
			});
		});

		LatentIt("EnqueueSession after Shutdown returns 0 and never calls back", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> M = NewModel(TEXT("M"));
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("A"), M, 3, 10);
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			E.Shutdown();
			const FPipelineJobId Id = EnqSess(E, S, Log);
			TestEqual(TEXT("id after Shutdown is 0"), (int32)Id, 0);
			const TSharedRef<FSessLog> LogNull = MakeShared<FSessLog>();
			TestEqual(TEXT("id of a null session after Shutdown is 0"), (int32)EnqSess(E, nullptr, LogNull), 0);
			Settle(0.3, [this, Done, Log, LogNull, S]()
			{
				TestEqual(TEXT("no OnComplete"), Log->Completes + LogNull->Completes, 0);
				TestEqual(TEXT("no OnProgress"), Log->Progress.Num(), 0);
				TestEqual(TEXT("Begin never called"), S->BeginCalls.GetValue(), 0);
				Done.ExecuteIfBound();
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Concurrency (acceptance 11)", [this]()
	{
		LatentIt("8 threads x 4 sessions: 32 OnComplete Ok, every session's progress in order", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			TArray<TSharedRef<FFakeSessionModel>> Models;
			for (int32 i = 0; i < 8; ++i) { Models.Add(NewModel(FString::Printf(TEXT("M%d"), i))); }
			Tr();
			const TSharedRef<FSessThreadSet> Set = MakeShared<FSessThreadSet>();
			FPipelineExecutor* EP = &E;
			const TSharedRef<FTrace> TraceRef = TraceP.ToSharedRef();
			for (int32 T = 0; T < 8; ++T)
			{
				const TSharedRef<FFakeSessionModel> Model = Models[T];
				Async(EAsyncExecution::Thread, [EP, Set, Model, TraceRef, T]()
				{
					for (int32 j = 0; j < 4; ++j)
					{
						const TSharedRef<FScriptedSession> S = MakeShared<FScriptedSession>(FString::Printf(TEXT("T%d_%d"), T, j), Model, TraceRef, 3, 5);
						const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
						Log->OnCompleteHook = [Set]() { Set->Completed.Increment(); };
						FPipelineSessionCallbacks C = SessMakeCallbacks(Log);
						const FPipelineJobId Id = EP->EnqueueSession(S, MoveTemp(C));
						FScopeLock L(&Set->Lock);
						Set->Logs.Add(Log);
						Set->Sessions.Add(S);
						Set->Ids.Add(Id);
					}
					Set->ThreadsDone.Increment();
				});
			}
			PollUntil([Set]() { return Set->ThreadsDone.GetValue() == 8 && Set->Completed.GetValue() == 32; }, 30.0, [this, Done, Set, Models, EP](bool bTimedOut)
			{
				if (bTimedOut)
				{
					AddError(FString::Printf(TEXT("timed out: %d of 32 sessions completed, %d of 8 threads finished enqueueing"), Set->Completed.GetValue(), Set->ThreadsDone.GetValue()));
					Done.ExecuteIfBound();
					return;
				}
				Settle(0.3, [this, Done, Set, Models, EP]()
				{
					FScopeLock L(&Set->Lock);
					TestEqual(TEXT("32 sessions were enqueued"), Set->Logs.Num(), 32);
					TSet<FPipelineJobId> Distinct;
					for (int32 i = 0; i < Set->Logs.Num(); ++i)
					{
						VerifyOk(*Set->Logs[i], 3, FString::Printf(TEXT("session #%d"), i));
						TestEqual(*FString::Printf(TEXT("session #%d: End once"), i), Set->Sessions[i]->EndCalls.GetValue(), 1);
						TestEqual(*FString::Printf(TEXT("session #%d: 3 steps"), i), Set->Sessions[i]->StepCalls.GetValue(), 3);
						Distinct.Add(Set->Ids[i]);
					}
					TestEqual(TEXT("ids are distinct"), Distinct.Num(), 32);
					TestFalse(TEXT("no id is 0"), Distinct.Contains(0));
					for (const TSharedRef<FFakeSessionModel>& M : Models)
					{
						TestEqual(TEXT("no two sessions held one model at once"), M->SessionOverlap.GetValue(), 0);
					}
					TestEqual(TEXT("nothing left pending"), EP->GetNumPending(), 0);
					Done.ExecuteIfBound();
				});
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Game-thread cost (acceptance 12)", [this]()
	{
		LatentIt("200 sessions x 5 steps: all delivered in order, MaxGameThreadDispatchMs < 2 ms", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			TArray<TSharedRef<FFakeSessionModel>> Models;
			for (int32 i = 0; i < 8; ++i) { Models.Add(NewModel(FString::Printf(TEXT("M%d"), i))); }
			const TSharedRef<TArray<TSharedRef<FSessLog>>> Logs = MakeShared<TArray<TSharedRef<FSessLog>>>();
			const TSharedRef<FThreadSafeCounter> Completed = MakeShared<FThreadSafeCounter>();
			E.ResetStats();
			for (int32 i = 0; i < 200; ++i)
			{
				const TSharedRef<FScriptedSession> S = NewSess(FString::Printf(TEXT("S%d"), i), Models[i % 8], 5, 0);
				const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
				Log->OnCompleteHook = [Completed]() { Completed->Increment(); };
				Logs->Add(Log);
				EnqSess(E, S, Log);
			}
			FPipelineExecutor* EP = &E;
			PollUntil([Completed]() { return Completed->GetValue() >= 200; }, 60.0, [this, Done, Logs, Completed, EP](bool bTimedOut)
			{
				if (bTimedOut)
				{
					AddError(FString::Printf(TEXT("timed out: %d of 200 sessions completed"), Completed->GetValue()));
					Done.ExecuteIfBound();
					return;
				}
				Settle(0.3, [this, Done, Logs, EP]()
				{
					int32 BadLogs = 0;
					for (const TSharedRef<FSessLog>& L : *Logs)
					{
						const bool bOk = L->Completes == 1 && L->Result.bSuccess && L->Progress.Num() == 5 && L->ProgressAfterComplete == 0;
						bool bOrder = L->Progress.Num() == 5;
						for (int32 k = 0; bOrder && k < 5; ++k) { bOrder = L->Progress[k].Seq == k && L->Progress[k].Text == FString::Printf(TEXT("s%d"), k); }
						if (!bOk || !bOrder) { ++BadLogs; }
					}
					TestEqual(TEXT("sessions with a wrong count/order/result"), BadLogs, 0);
					const FPipelineExecutorStats St = EP->GetStats();
					TestEqual(TEXT("NumCompleted"), (int32)St.NumCompleted, 200);
					TestTrue(*FString::Printf(TEXT("MaxGameThreadDispatchMs expected < 2.0, got %.3f"), St.MaxGameThreadDispatchMs), St.MaxGameThreadDispatchMs < 2.0);
					Done.ExecuteIfBound();
				});
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Load on its own thread (acceptance 13)", [this]()
	{
		LatentIt("1 s load does not block the worker; Initialize runs off game thread and worker; Begin after Initialize; a one-shot on the loading model waits", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> L = NewModel(TEXT("L"), false);
			L->InitSleepMs = 1000;
			const TSharedRef<FFakeSessionModel> N = NewModel(TEXT("N"));
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("S"), L, 3, 10);
			const TSharedRef<FSessLog> LogS = MakeShared<FSessLog>();
			const TSharedRef<FSessJobLog> J = NewJobLog(), K = NewJobLog();
			J->Hook = [J, L]() { J->bFlag = L->bInitFinished; }; // true would mean J had to wait for the load
			EnqSess(E, S, LogS);
			EnqJob(E, N, 1, J);
			EnqJob(E, L, 2, K);
			PollUntil([LogS, J, K]() { return LogS->Completes >= 1 && J->Completes >= 1 && K->Completes >= 1; }, SessPollTimeout,
				[this, Done, L, N, S, LogS, J, K](bool bTimedOut)
			{
				const FString Dump = Tr().Dump();
				if (bTimedOut) { AddError(TEXT("timed out; worker order: ") + Dump); Done.ExecuteIfBound(); return; }
				const int32 IJ = Tr().IndexOf(TEXT("runtag:N:1")), IInitEnd = Tr().IndexOf(TEXT("init-end:L"));
				const int32 IBegin = Tr().IndexOf(TEXT("begin:S")), IEndS = Tr().IndexOf(TEXT("end:S")), IK = Tr().IndexOf(TEXT("runtag:L:2"));
				TestFalse(TEXT("J (other model) completed while L was still loading (worker not blocked)"), J->bFlag);
				TestTrue(*(TEXT("J ran before the load finished. Order: ") + Dump), IJ != INDEX_NONE && IInitEnd != INDEX_NONE && IJ < IInitEnd);
				TestEqual(TEXT("Initialize body ran once"), L->InitBodyRuns.GetValue(), 1);
				TestFalse(TEXT("Initialize did not run on the game thread"), (bool)L->bInitOnGameThread);
				TestTrue(TEXT("Initialize did not run on the executor worker (the worker ran J)"), L->InitThreadId != 0 && L->InitThreadId != N->RunThreadId);
				TestTrue(*(TEXT("Begin only after Initialize returned. Order: ") + Dump), IBegin != INDEX_NONE && IInitEnd < IBegin);
				TestTrue(TEXT("model reported initialized when Begin ran"), (bool)S->bModelInitializedAtBegin);
				TestEqual(TEXT("Begin ran on the worker thread"), S->BeginThreadId, N->RunThreadId);
				TestTrue(*(TEXT("one-shot K on the loading model waited for the load and the session. Order: ") + Dump), IK != INDEX_NONE && IK > IInitEnd && IK > IEndS);
				TestEqual(TEXT("K never ran during the session"), L->RunDuringSession.GetValue(), 0);
				TestTrue(TEXT("J and K succeeded"), J->Result.bSuccess && K->Result.bSuccess);
				VerifyOk(*LogS, 3, TEXT("loaded session"));
				Done.ExecuteIfBound();
			});
		});

		LatentIt("Initialize fails: OnComplete carries the failure; Begin and End never called; worker still alive", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> L = NewModel(TEXT("L"), false);
			L->InitSleepMs = 100;
			L->bInitFails = true;
			const TSharedRef<FFakeSessionModel> N = NewModel(TEXT("N"));
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("S"), L, 3, 10);
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			EnqSess(E, S, Log);
			FPipelineExecutor* EP = &E;
			PollUntil([Log]() { return Log->Completes >= 1; }, SessPollTimeout, [this, Done, Log, S, N, EP](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: a session whose model failed to load never completed")); Done.ExecuteIfBound(); return; }
				Settle(0.3, [this, Done, Log, S, N, EP]()
				{
					TestEqual(TEXT("OnComplete exactly once"), Log->Completes, 1);
					TestFalse(TEXT("failure"), Log->Result.bSuccess);
					TestEqual(TEXT("error code of Initialize's failure"), (int32)Log->Result.Error, (int32)EPipelineError::ModelLoadFailed);
					TestTrue(TEXT("message of Initialize's failure"), Log->Result.Message.Contains(TEXT("fake init failure")));
					TestEqual(TEXT("Begin never called"), S->BeginCalls.GetValue(), 0);
					TestEqual(TEXT("End never called"), S->EndCalls.GetValue(), 0);
					TestEqual(TEXT("Step never called"), S->StepCalls.GetValue(), 0);
					TestEqual(TEXT("no progress"), Log->Progress.Num(), 0);
					TestEqual(TEXT("nothing pending"), EP->GetNumPending(), 0);
					const TSharedRef<FSessJobLog> JL = NewJobLog();
					EnqJob(*EP, N, 4, JL);
					PollUntil([JL]() { return JL->Completes >= 1; }, SessPollTimeout, [this, Done, JL](bool bTimedOut2)
					{
						if (bTimedOut2) { AddError(TEXT("worker did not run a later job after the failed load")); }
						else { TestTrue(TEXT("later job succeeds"), JL->Result.bSuccess); }
						Done.ExecuteIfBound();
					});
				});
			});
		});

		LatentIt("Cancel during the load: true, CancelInitialize once, OnComplete(Cancelled) only after the load task ended, Begin never called", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> L = NewModel(TEXT("L"), false);
			L->InitSleepMs = 800; // the fake does not abort on CancelInitialize: the task ends 800 ms after it started
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("S"), L, 3, 10);
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			Log->OnCompleteHook = [L, LP = &Log.Get()]() { LP->bFlag = L->bInitFinished; };
			const FPipelineJobId Id = EnqSess(E, S, Log);
			PollUntil([L]() { return (bool)L->bInitStarted; }, SessPollTimeout, [this, Done, &E, L, S, Log, Id](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: the load never started")); Done.ExecuteIfBound(); return; }
				Log->bInCall = true;
				const bool bCancelled = E.Cancel(Id);
				Log->bInCall = false;
				TestTrue(TEXT("Cancel during the load returns true"), bCancelled);
				PollUntil([Log]() { return Log->Completes >= 1; }, SessPollTimeout, [this, Done, L, S, Log](bool bTimedOut2)
				{
					if (bTimedOut2) { AddError(TEXT("timed out: OnComplete(Cancelled) never arrived after cancelling a load")); Done.ExecuteIfBound(); return; }
					Settle(0.3, [this, Done, L, S, Log]()
					{
						TestEqual(TEXT("OnComplete exactly once"), Log->Completes, 1);
						TestEqual(TEXT("error code Cancelled"), (int32)Log->Result.Error, (int32)EPipelineError::Cancelled);
						TestTrue(TEXT("OnComplete came only after the load task ended"), Log->bFlag);
						TestEqual(TEXT("CancelInitialize called once"), L->CancelInitCalls.GetValue(), 1);
						TestEqual(TEXT("Begin never called"), S->BeginCalls.GetValue(), 0);
						TestEqual(TEXT("End never called"), S->EndCalls.GetValue(), 0);
						TestEqual(TEXT("callbacks not inside Cancel"), Log->CallbacksInsideCall, 0);
						Done.ExecuteIfBound();
					});
				});
			});
		});

		LatentIt("Shutdown during a 1 s load returns in < 500 ms (brief: 200), no callbacks, no crash when the task ends", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> L = NewModel(TEXT("L"), false);
			L->InitSleepMs = 1000;
			const TSharedRef<FScriptedSession> S = NewSess(TEXT("S"), L, 3, 10);
			const TSharedRef<FSessLog> Log = MakeShared<FSessLog>();
			EnqSess(E, S, Log);
			PollUntil([L]() { return (bool)L->bInitStarted; }, SessPollTimeout, [this, Done, &E, L, S, Log](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: the load never started")); Done.ExecuteIfBound(); return; }
				const double T0 = FPlatformTime::Seconds();
				E.Shutdown();
				const double Ms = (FPlatformTime::Seconds() - T0) * 1000.0;
				TestTrue(*FString::Printf(TEXT("Shutdown must not wait for the load: expected < 500 ms, got %.1f ms"), Ms), Ms < 500.0);
				TestFalse(TEXT("the load was still running when Shutdown returned"), (bool)L->bInitFinished);
				PollUntil([L]() { return (bool)L->bInitFinished; }, 10.0, [this, Done, L, S, Log](bool bTimedOut2)
				{
					if (bTimedOut2) { AddError(TEXT("the fake load never finished")); Done.ExecuteIfBound(); return; }
					Settle(0.4, [this, Done, S, Log]()
					{
						TestEqual(TEXT("no OnComplete"), Log->Completes, 0);
						TestEqual(TEXT("no OnProgress"), Log->Progress.Num(), 0);
						TestEqual(TEXT("Begin never called"), S->BeginCalls.GetValue(), 0);
						TestEqual(TEXT("End never called"), S->EndCalls.GetValue(), 0);
						Done.ExecuteIfBound();
					});
				});
			});
		});

		LatentIt("two sessions on one unloaded model: Initialize runs once, both complete Ok", SessLatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FFakeSessionModel> L = NewModel(TEXT("L"), false);
			L->InitSleepMs = 300;
			const TSharedRef<FScriptedSession> S1 = NewSess(TEXT("S1"), L, 3, 5);
			const TSharedRef<FScriptedSession> S2 = NewSess(TEXT("S2"), L, 3, 5);
			const TSharedRef<FSessLog> Log1 = MakeShared<FSessLog>();
			const TSharedRef<FSessLog> Log2 = MakeShared<FSessLog>();
			EnqSess(E, S1, Log1);
			EnqSess(E, S2, Log2);
			PollUntil([Log1, Log2]() { return Log1->Completes >= 1 && Log2->Completes >= 1; }, SessPollTimeout, [this, Done, L, S1, S2, Log1, Log2](bool bTimedOut)
			{
				if (bTimedOut) { AddError(TEXT("timed out: two sessions on one unloaded model")); Done.ExecuteIfBound(); return; }
				Settle(0.2, [this, Done, L, S1, S2, Log1, Log2]()
				{
					TestEqual(TEXT("Initialize called once"), L->InitCalls.GetValue(), 1);
					TestEqual(TEXT("load body ran once"), L->InitBodyRuns.GetValue(), 1);
					VerifyOk(*Log1, 3, TEXT("first session"));
					VerifyOk(*Log2, 3, TEXT("second session"));
					TestTrue(TEXT("both Begins saw an initialized model"), S1->bModelInitializedAtBegin && S2->bModelInitializedAtBegin);
					TestEqual(TEXT("sessions did not overlap on the model"), L->SessionOverlap.GetValue(), 0);
					Done.ExecuteIfBound();
				});
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
