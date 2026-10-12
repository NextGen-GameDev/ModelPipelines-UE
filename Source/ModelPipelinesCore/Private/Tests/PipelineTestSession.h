// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// Test-only doubles for the executor session specs (WP8L.3b): a scripted IPipelineSession and a fake IPipelineModel.
// Header-only; include from exactly one spec file. Nothing here touches the executor implementation.

#pragma once

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/PlatformTLS.h"
#include "HAL/ThreadSafeBool.h"
#include "HAL/ThreadSafeCounter.h"
#include "Misc/ScopeLock.h"
#include "PipelineBackend.h"
#include "PipelineSession.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"

namespace UE::ModelPipelines::Tests::Session
{
	/** Thread-safe ordered event list with timestamps. Worker-side doubles append; tests read it afterwards. */
	class FTrace
	{
	public:
		struct FEntry
		{
			double Time = 0.0;
			FString Name;
		};

		void Add(const FString& Name)
		{
			FScopeLock L(&Lock);
			FEntry E;
			E.Time = FPlatformTime::Seconds();
			E.Name = Name;
			Entries.Add(MoveTemp(E));
		}

		TArray<FEntry> Get() const
		{
			FScopeLock L(&Lock);
			return Entries;
		}

		/** Index of the first entry with exactly this name, or INDEX_NONE. */
		int32 IndexOf(const FString& Name) const
		{
			FScopeLock L(&Lock);
			for (int32 i = 0; i < Entries.Num(); ++i)
			{
				if (Entries[i].Name == Name) { return i; }
			}
			return INDEX_NONE;
		}

		/** Time of the first entry with exactly this name, or -1. */
		double TimeOf(const FString& Name) const
		{
			FScopeLock L(&Lock);
			for (const FEntry& E : Entries)
			{
				if (E.Name == Name) { return E.Time; }
			}
			return -1.0;
		}

		/** Names only, in order. */
		FString Dump() const
		{
			FScopeLock L(&Lock);
			FString S;
			for (const FEntry& E : Entries) { S += E.Name + TEXT(" "); }
			return S;
		}

	private:
		mutable FCriticalSection Lock;
		TArray<FEntry> Entries;
	};

	/**
	 * Fake model with identity per instance. One-shot Run echoes 2x+1 and records "run:<Name>" with the thread id.
	 * Initialize records its thread, sleeps InitSleepMs (it does NOT abort on CancelInitialize, so tests can see that the
	 * executor waits for the task end), and may fail. Optional Gate blocks Run (manual-reset event).
	 * Also detects exclusivity violations: Run or a second session Begin while a session holds the model.
	 */
	class FFakeSessionModel : public IPipelineModel
	{
	public:
		FFakeSessionModel(const FString& InName, TSharedRef<FTrace> InTrace, bool bInPreInitialized)
			: Name(InName)
			, Trace(InTrace)
			, bInitialized(bInPreInitialized)
		{
			Gate = FPlatformProcess::GetSynchEventFromPool(true);
			Gate->Trigger(); // pool events start non-signalled (WindowsEvent.h:37); the gate is open unless a test closes it (lead fix)
			RunEntered = FPlatformProcess::GetSynchEventFromPool(true);
		}

		virtual ~FFakeSessionModel() override
		{
			Gate->Trigger();
			FPlatformProcess::ReturnSynchEventToPool(Gate);
			FPlatformProcess::ReturnSynchEventToPool(RunEntered);
		}

		void OpenGate() { Gate->Trigger(); }
		bool WaitRunEntered(double Seconds = 15.0) { return RunEntered->Wait(FTimespan::FromSeconds(Seconds)); }

		virtual FPipelineResult Initialize() override
		{
			InitCalls.Increment();
			if (bInitialized)
			{
				return FPipelineResult::Ok();
			}
			// The first caller does the work; the thread id is the one that ran the body.
			bInitStarted = true;
			InitBodyRuns.Increment();
			InitThreadId = FPlatformTLS::GetCurrentThreadId();
			bInitOnGameThread = IsInGameThread();
			Trace->Add(TEXT("init-start:") + Name);
			if (InitSleepMs > 0)
			{
				FPlatformProcess::Sleep((float)InitSleepMs / 1000.f);
			}
			FPipelineResult R;
			if (bInitFails)
			{
				R = FPipelineResult::Fail(EPipelineError::ModelLoadFailed, TEXT("fake init failure"));
			}
			else
			{
				bInitialized = true;
				R = FPipelineResult::Ok();
			}
			Trace->Add(TEXT("init-end:") + Name);
			bInitFinished = true;
			return R;
		}

		virtual void CancelInitialize() override
		{
			CancelInitCalls.Increment();
			Trace->Add(TEXT("cancel-init:") + Name);
		}

		virtual bool IsInitialized() const override { return bInitialized; }
		virtual EPipelineDevice GetDevice() const override { return EPipelineDevice::CPU; }
		virtual TArray<FPipelineTensorDesc> GetInputDescs() const override { return TArray<FPipelineTensorDesc>(); }
		virtual TArray<FPipelineTensorDesc> GetOutputDescs() const override { return TArray<FPipelineTensorDesc>(); }

		virtual FPipelineResult Run(TConstArrayView<FPipelineTensor> Inputs, TArray<FPipelineTensor>& OutOutputs) override
		{
			if (!bInitialized)
			{
				const FPipelineResult I = Initialize();
				if (!I.bSuccess) { OutOutputs.Reset(); return I; }
			}
			RunCalls.Increment();
			RunThreadId = FPlatformTLS::GetCurrentThreadId();
			if (SessionsActive.GetValue() > 0)
			{
				RunDuringSession.Increment(); // rule 2 broken
			}
			Trace->Add(TEXT("run:") + Name);
			if (Inputs.Num() > 0 && Inputs[0].GetType() == EPipelineTensorType::Float32 && Inputs[0].Num() > 0)
			{
				// "runtag:<model>:<first input as int>" tells apart several one-shot jobs on one model.
				Trace->Add(FString::Printf(TEXT("runtag:%s:%d"), *Name, (int32)Inputs[0].GetFloatData()[0]));
			}
			RunEntered->Trigger();
			Gate->Wait(FTimespan::FromSeconds(120)); // manual reset: open unless a test closed it; bounded
			if (RunSleepMs > 0) { FPlatformProcess::Sleep((float)RunSleepMs / 1000.f); }

			OutOutputs.Reset();
			for (const FPipelineTensor& In : Inputs)
			{
				TArray<float> V;
				V.Append(In.GetFloatData().GetData(), (int32)In.Num());
				for (float& F : V) { F = 2.f * F + 1.f; }
				TArray<int64> Shape;
				Shape.Append(In.GetShape().GetData(), In.GetShape().Num());
				OutOutputs.Add(FPipelineTensor::FromFloat(Shape, V));
			}
			FPipelineResult R = FPipelineResult::Ok();
			R.Device = EPipelineDevice::CPU;
			R.InferenceMs = 0.1f;
			Trace->Add(TEXT("run-done:") + Name);
			return R;
		}

		/** Closes the gate so the next Run blocks. Call before enqueueing the blocking job. */
		void CloseGate() { Gate->Reset(); }

		const FString Name;
		const TSharedRef<FTrace> Trace;

		// Configuration (set before enqueueing).
		int32 InitSleepMs = 0;
		bool bInitFails = false;
		int32 RunSleepMs = 0;

		// Observations.
		FThreadSafeCounter InitCalls;
		FThreadSafeCounter InitBodyRuns;       // times the load body ran (a double load shows as 2)
		FThreadSafeCounter CancelInitCalls;
		FThreadSafeCounter RunCalls;
		FThreadSafeCounter RunDuringSession;   // Run executed while a session held the model
		FThreadSafeCounter SessionsActive;     // Begin..End of sessions on this model (maintained by FScriptedSession)
		FThreadSafeCounter SessionOverlap;     // a second session began while one held the model
		FThreadSafeBool bInitStarted;
		FThreadSafeBool bInitFinished;
		FThreadSafeBool bInitOnGameThread;
		uint32 InitThreadId = 0;               // written once by the Initialize body, read after bInitFinished
		uint32 RunThreadId = 0;                // last Run's thread (the worker, for one-shot jobs)

	private:
		FThreadSafeBool bInitialized;
		FEvent* Gate = nullptr;
		FEvent* RunEntered = nullptr;
	};

	/**
	 * Scripted session. One Step = optional hook (on the worker), sleep StepMs, then one progress item {Ids={k}, Text="s<k>"}.
	 * Returns Finished on step NumSteps-1, Failed at step FailAtStep (before producing progress), Continue otherwise.
	 * Counters are atomics so the game thread can poll them. Trace entries: begin:<Name>, step:<Name>:<k>, end:<Name>.
	 */
	class FScriptedSession : public IPipelineSession
	{
	public:
		FScriptedSession(const FString& InName, TSharedPtr<FFakeSessionModel> InModel, TSharedRef<FTrace> InTrace, int32 InNumSteps, int32 InStepMs)
			: Name(InName)
			, Model(InModel)
			, Trace(InTrace)
			, NumSteps(InNumSteps)
			, StepMs(InStepMs)
		{
		}

		virtual TSharedPtr<IPipelineModel> GetModel() const override { return Model; }

		virtual FPipelineResult Begin() override
		{
			BeginCalls.Increment();
			BeginThreadId = FPlatformTLS::GetCurrentThreadId();
			bModelInitializedAtBegin = Model.IsValid() && Model->IsInitialized();
			bInitFinishedAtBegin = Model.IsValid() && (Model->bInitFinished || Model->InitCalls.GetValue() == 0);
			Trace->Add(TEXT("begin:") + Name);
			if (Model.IsValid())
			{
				if (Model->SessionsActive.Increment() > 1) { Model->SessionOverlap.Increment(); }
			}
			bHoldingModel = true;
			if (bFailBegin)
			{
				return FPipelineResult::Fail(EPipelineError::InferenceFailed, TEXT("fake begin failure"));
			}
			return FPipelineResult::Ok();
		}

		virtual EPipelineStep Step(TArray<FPipelineSessionProgress>& OutProgress, FPipelineResult& OutResult) override
		{
			const int32 K = StepCalls.Increment() - 1;
			StepThreadId = FPlatformTLS::GetCurrentThreadId();
			Trace->Add(FString::Printf(TEXT("step:%s:%d"), *Name, K));
			if (OnStep) { OnStep(K); }
			if (K == FailAtStep)
			{
				OutResult = FPipelineResult::Fail(EPipelineError::InferenceFailed, TEXT("fake step failure"));
				return EPipelineStep::Failed;
			}
			if (StepMs > 0) { FPlatformProcess::Sleep((float)StepMs / 1000.f); }
			FPipelineSessionProgress P;
			P.Ids.Add(K);
			P.Text = FString::Printf(TEXT("s%d"), K);
			OutProgress.Add(MoveTemp(P));
			return K >= NumSteps - 1 ? EPipelineStep::Finished : EPipelineStep::Continue;
		}

		virtual void End(EPipelineSessionEnd Reason, FPipelineResult& OutFinal) override
		{
			EndCalls.Increment();
			LastEndReason = (int32)Reason;
			Trace->Add(TEXT("end:") + Name);
			if (bHoldingModel && Model.IsValid())
			{
				Model->SessionsActive.Decrement();
			}
			bHoldingModel = false;
			if (Reason == EPipelineSessionEnd::Finished)
			{
				OutFinal = FPipelineResult::Ok();
			}
		}

		const FString Name;
		const TSharedPtr<FFakeSessionModel> Model;
		const TSharedRef<FTrace> Trace;
		const int32 NumSteps;
		const int32 StepMs;

		// Configuration.
		bool bFailBegin = false;
		int32 FailAtStep = -1;
		/** Runs on the worker at the start of each Step (before the sleep). May block. */
		TFunction<void(int32 StepIndex)> OnStep;

		// Observations.
		FThreadSafeCounter BeginCalls;
		FThreadSafeCounter StepCalls;
		FThreadSafeCounter EndCalls;
		int32 LastEndReason = -1;             // EPipelineSessionEnd, written on the worker, read after EndCalls > 0
		uint32 BeginThreadId = 0;
		uint32 StepThreadId = 0;
		FThreadSafeBool bModelInitializedAtBegin;
		FThreadSafeBool bInitFinishedAtBegin;

	private:
		bool bHoldingModel = false;
	};
}

#endif // WITH_DEV_AUTOMATION_TESTS
