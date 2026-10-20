// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "Misc/ScopeLock.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/GarbageCollection.h"
#include "NNEModelData.h"
#include "PipelineExecutor.h"
#include "PipelineBackend.h"
#include "PipelineNNEBackend.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "PipelineLog.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineTestData.h"

using namespace UE::ModelPipelines::Tests;

namespace
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(60);
	const TCHAR* const AffineFile = TEXT("affine_dynamic.onnx");
	const TCHAR* const FetchHint = TEXT("Fetch: run Tools/make_test_goldens.py (and the test-model step of docs/briefs/WP1.1.md) so that .pipelines-dev/models/test and .pipelines-dev/goldens/test_models exist.");

	/**
	 * Test double. Computes y = 2x + 1 per float input (same as affine_dynamic). Records the call order (the first
	 * element of the first input, as an int tag), the maximum number of concurrent Run calls, and whether Run ever
	 * executed on the game thread. With bBlocks, Run signals Entered, then waits on Gate (manual reset, so once opened
	 * every later Run passes straight through).
	 */
	class FTestModel : public IPipelineModel
	{
	public:
		explicit FTestModel(bool bInBlocks)
			: bBlocks(bInBlocks)
		{
			Gate = FPlatformProcess::GetSynchEventFromPool(true);
			Entered = FPlatformProcess::GetSynchEventFromPool(true);
		}

		virtual ~FTestModel() override
		{
			Open();
			FPlatformProcess::ReturnSynchEventToPool(Gate);
			FPlatformProcess::ReturnSynchEventToPool(Entered);
		}

		void Open() { Gate->Trigger(); }

		/** Game-thread helper: waits until the first Run has started (the worker is now blocked inside it). */
		bool WaitEntered(double Seconds = 15.0) { return Entered->Wait(FTimespan::FromSeconds(Seconds)); }

		TArray<int32> GetOrder()
		{
			FScopeLock L(&Lock);
			return Order;
		}

		int32 GetMaxConcurrent()
		{
			FScopeLock L(&Lock);
			return MaxConcurrent;
		}

		virtual FPipelineResult Initialize() override { return FPipelineResult::Ok(); }
		virtual bool IsInitialized() const override { return true; }
		virtual EPipelineDevice GetDevice() const override { return EPipelineDevice::CPU; }
		virtual TArray<FPipelineTensorDesc> GetInputDescs() const override { return TArray<FPipelineTensorDesc>(); }
		virtual TArray<FPipelineTensorDesc> GetOutputDescs() const override { return TArray<FPipelineTensorDesc>(); }

		virtual FPipelineResult Run(TConstArrayView<FPipelineTensor> Inputs, TArray<FPipelineTensor>& OutOutputs) override
		{
			Calls.Increment();
			if (IsInGameThread())
			{
				GameThreadRuns.Increment();
			}
			{
				FScopeLock L(&Lock);
				++Current;
				MaxConcurrent = FMath::Max(MaxConcurrent, Current);
				if (Inputs.Num() > 0 && Inputs[0].GetType() == EPipelineTensorType::Float32 && Inputs[0].Num() > 0)
				{
					Order.Add((int32)Inputs[0].GetFloatData()[0]);
				}
			}
			if (bBlocks)
			{
				Entered->Trigger();
				Gate->Wait(FTimespan::FromSeconds(120)); // bounded so a forgotten gate can never hang forever
			}
			if (SleepSeconds > 0.f)
			{
				FPlatformProcess::Sleep(SleepSeconds);
			}

			OutOutputs.Reset();
			FPipelineResult R;
			if (FailWith != EPipelineError::None)
			{
				R = FPipelineResult::Fail(FailWith, TEXT("test model failure"));
			}
			else
			{
				for (const FPipelineTensor& In : Inputs)
				{
					TArray<float> V;
					V.Append(In.GetFloatData().GetData(), (int32)In.Num());
					for (float& F : V) { F = 2.f * F + 1.f; }
					TArray<int64> Shape;
					Shape.Append(In.GetShape().GetData(), In.GetShape().Num());
					OutOutputs.Add(FPipelineTensor::FromFloat(Shape, V));
				}
				R = FPipelineResult::Ok();
			}
			R.Device = EPipelineDevice::CPU;
			R.InferenceMs = 1.5f;
			{
				FScopeLock L(&Lock);
				--Current;
			}
			Completed.Increment();
			return R;
		}

		const bool bBlocks;
		float SleepSeconds = 0.f;
		EPipelineError FailWith = EPipelineError::None;
		FThreadSafeCounter Calls;
		FThreadSafeCounter Completed;
		FThreadSafeCounter GameThreadRuns;

	private:
		FEvent* Gate = nullptr;
		FEvent* Entered = nullptr;
		FCriticalSection Lock;
		TArray<int32> Order;
		int32 Current = 0;
		int32 MaxConcurrent = 0;
	};

	/** Picks the pending job with the highest Priority (first one wins ties). Records what it was shown. */
	class FMaxPriorityScheduler : public IPipelineScheduler
	{
	public:
		virtual int32 SelectNext(TConstArrayView<FPipelineJobInfo> Pending) override
		{
			Calls.Increment();
			if (IsInGameThread()) { CalledOnGameThread.Increment(); }
			int32 Best = 0;
			for (int32 i = 0; i < Pending.Num(); ++i)
			{
				if (Pending[i].Id == 0 || Pending[i].EnqueueTimeSeconds <= 0.0) { BadInfo.Increment(); }
				if (Pending[i].Priority > Pending[Best].Priority) { Best = i; }
			}
			if (Pending.Num() == 3)
			{
				FScopeLock L(&Lock);
				if (SeenThree.Num() == 0)
				{
					for (const FPipelineJobInfo& J : Pending) { SeenThree.Add(J.Priority); }
					SeenThree.Sort();
				}
			}
			return Best;
		}

		TArray<int32> GetSeenThree()
		{
			FScopeLock L(&Lock);
			return SeenThree;
		}

		FThreadSafeCounter Calls;
		FThreadSafeCounter CalledOnGameThread;
		FThreadSafeCounter BadInfo;

	private:
		FCriticalSection Lock;
		TArray<int32> SeenThree;
	};

	/** Always returns the same index, even when it is out of range. */
	class FFixedScheduler : public IPipelineScheduler
	{
	public:
		explicit FFixedScheduler(int32 InIndex) : Index(InIndex) {}
		virtual int32 SelectNext(TConstArrayView<FPipelineJobInfo> Pending) override
		{
			Calls.Increment();
			return Index;
		}
		FThreadSafeCounter Calls;
	private:
		int32 Index;
	};

	/** One callback arrival, recorded on the game thread. */
	struct FCbRecord
	{
		int32 Tag = 0;
		bool bGameThread = false;
		bool bInCall = false;
		FPipelineResult Result;
		TArray<FPipelineTensor> Outputs;
	};

	/**
	 * Collects callbacks. When Expected records have arrived it waits DelaySeconds (so duplicate or late callbacks
	 * show up), runs Verify on the game thread and then signals Done. Verify must not capture the FLog.
	 */
	struct FLog
	{
		TArray<FCbRecord> Records;
		bool bInCall = false;
		int32 Expected = -1;
		float DelaySeconds = 0.3f;
		bool bScheduled = false;
		FDoneDelegate Done;
		TFunction<void(FLog&)> Verify;

		int32 CountTag(int32 Tag) const
		{
			int32 N = 0;
			for (const FCbRecord& R : Records) { if (R.Tag == Tag) { ++N; } }
			return N;
		}
		int32 IndexOfTag(int32 Tag) const
		{
			for (int32 i = 0; i < Records.Num(); ++i) { if (Records[i].Tag == Tag) { return i; } }
			return INDEX_NONE;
		}
		TArray<int32> Tags() const
		{
			TArray<int32> Out;
			for (const FCbRecord& R : Records) { Out.Add(R.Tag); }
			return Out;
		}
	};

	struct FJobSpec
	{
		int32 Tag;
		int32 Priority;
	};

	FString Join(const TArray<int32>& A)
	{
		FString S = TEXT("[");
		for (int32 i = 0; i < A.Num(); ++i)
		{
			S += FString::Printf(TEXT("%s%d"), i ? TEXT(",") : TEXT(""), A[i]);
		}
		return S + TEXT("]");
	}

	/** Schedules Verify + Done once, DelaySeconds from now, on the game thread. */
	void ScheduleFinish(const TSharedRef<FLog>& Log)
	{
		if (Log->bScheduled) { return; }
		Log->bScheduled = true;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("WP14Finish"), Log->DelaySeconds, [Log](float)
		{
			if (Log->Verify) { Log->Verify(*Log); }
			Log->Done.ExecuteIfBound();
			return false;
		});
	}

	/** Expected = number of records that trigger the finish; pass 0 and call ScheduleFinish yourself for "no callbacks" tests. */
	void Arm(const TSharedRef<FLog>& Log, const FDoneDelegate& Done, int32 Expected, TFunction<void(FLog&)> Verify, float DelaySeconds = 0.3f)
	{
		Log->Done = Done;
		Log->Expected = Expected > 0 ? Expected : -1;
		Log->Verify = MoveTemp(Verify);
		Log->DelaySeconds = DelaySeconds;
	}

	FPipelineInferenceCallback MakeCb(const TSharedRef<FLog>& Log, int32 Tag)
	{
		return [Log, Tag](const FPipelineResult& Result, TArray<FPipelineTensor>&& Outputs)
		{
			FCbRecord Rec;
			Rec.Tag = Tag;
			Rec.bGameThread = IsInGameThread();
			Rec.bInCall = Log->bInCall;
			Rec.Result = Result;
			Rec.Outputs = MoveTemp(Outputs);
			Log->Records.Add(MoveTemp(Rec));
			if (Log->Expected > 0 && Log->Records.Num() >= Log->Expected)
			{
				ScheduleFinish(Log);
			}
		};
	}

	TArray<FPipelineTensor> One(float V)
	{
		TArray<FPipelineTensor> R;
		R.Add(FPipelineTensor::FromFloat(TArray<int64>{1}, TArray<float>{V}));
		return R;
	}

	float FirstOut(const FCbRecord& R)
	{
		if (R.Outputs.Num() > 0 && R.Outputs[0].GetType() == EPipelineTensorType::Float32 && R.Outputs[0].Num() > 0)
		{
			return R.Outputs[0].GetFloatData()[0];
		}
		return -12345.f;
	}

	bool PollUntil(TFunctionRef<bool()> Pred, double Seconds)
	{
		const double End = FPlatformTime::Seconds() + Seconds;
		while (FPlatformTime::Seconds() < End)
		{
			if (Pred()) { return true; }
			FPlatformProcess::Sleep(0.001f);
		}
		return Pred();
	}

	// ---- golden helpers -------------------------------------------------------------------------------------

	TArray<TSharedPtr<FJsonObject>> GetObjectArray(const TSharedPtr<FJsonObject>& Parent, const TCHAR* Field)
	{
		TArray<TSharedPtr<FJsonObject>> Out;
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (Parent.IsValid() && Parent->TryGetArrayField(Field, Arr))
		{
			for (const TSharedPtr<FJsonValue>& V : *Arr)
			{
				const TSharedPtr<FJsonObject>* O = nullptr;
				if (V.IsValid() && V->TryGetObject(O)) { Out.Add(*O); }
			}
		}
		return Out;
	}

	TArray<FPipelineTensor> CaseInputs(const TSharedPtr<FJsonObject>& Case)
	{
		TArray<FPipelineTensor> Out;
		for (const TSharedPtr<FJsonObject>& O : GetObjectArray(Case, TEXT("inputs")))
		{
			Out.Add(TensorFromGoldenJson(O));
		}
		return Out;
	}

	/** Compares all outputs of a case. "" on match. Thread-safe (read-only on the JSON). */
	FString CompareCase(const TArray<FPipelineTensor>& Outputs, const TSharedPtr<FJsonObject>& Case, double& InOutWorst)
	{
		const TArray<TSharedPtr<FJsonObject>> Expected = GetObjectArray(Case, TEXT("outputs"));
		if (Outputs.Num() != Expected.Num())
		{
			return FString::Printf(TEXT("got %d outputs, expected %d"), Outputs.Num(), Expected.Num());
		}
		for (int32 i = 0; i < Expected.Num(); ++i)
		{
			double Worst = 0.0;
			const FString M = CompareTensorToGolden(Outputs[i], Expected[i], &Worst);
			InOutWorst = FMath::Max(InOutWorst, Worst);
			if (!M.IsEmpty()) { return FString::Printf(TEXT("output #%d: %s"), i, *M); }
		}
		return FString();
	}

	struct FRealModel
	{
		TSharedPtr<IPipelineModel> Model;
		TSharedPtr<FJsonObject> Goldens;
		TArray<TSharedPtr<FJsonObject>> Cases;
	};

	/** Loads goldens + affine_dynamic and prepares it with the module's default backend on the game thread. */
	bool PrepareAffine(FAutomationTestBase& T, EPipelineDevice Device, FRealModel& Out)
	{
		FString Err;
		Out.Goldens = LoadTestGoldens(Err);
		if (!Out.Goldens.IsValid())
		{
			T.AddError(FString::Printf(TEXT("Goldens missing or invalid: %s. %s"), *Err, FetchHint));
			return false;
		}
		const TSharedPtr<FJsonObject> GM = FindGoldenModel(Out.Goldens, AffineFile);
		Out.Cases = GetObjectArray(GM, TEXT("cases"));
		if (Out.Cases.Num() == 0)
		{
			T.AddError(FString::Printf(TEXT("No golden cases for %s in test_models.json. %s"), AffineFile, FetchHint));
			return false;
		}
		UNNEModelData* Data = LoadTestModel(AffineFile, Err);
		if (!Data)
		{
			T.AddError(FString::Printf(TEXT("Model missing: %s. %s"), *Err, FetchHint));
			return false;
		}
		TStrongObjectPtr<UNNEModelData> Keep(Data);
		FPipelineResult PR;
		Out.Model = FModelPipelinesCoreModule::Get().GetDefaultBackend().PrepareModel(Data, Device, PR);
		if (!Out.Model.IsValid() || !PR.bSuccess)
		{
			T.AddError(FString::Printf(TEXT("PrepareModel failed: error %d, '%s'"), (int32)PR.Error, *PR.Message));
			return false;
		}
		return true;
	}

	/** GPU skip rule. Returns true if the test must stop. Never a silent pass. */
	bool GpuSkip(FAutomationTestBase& T)
	{
		if (!FModelPipelinesCoreModule::Get().GetDefaultBackend().IsDeviceAvailable(EPipelineDevice::GPU))
		{
			if (IsGpuRequired())
			{
				T.AddError(TEXT("GPU not available but MODELPIPELINES_REQUIRE_GPU=1"));
			}
			else
			{
				T.AddInfo(TEXT("GPU not available: skipped"));
			}
			return true;
		}
		return false;
	}
}

BEGIN_DEFINE_SPEC(FPipelineExecutorSpec, "Pipelines.Core.Executor",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	/** Models and the private executor created by a test. AfterEach opens every gate so no worker stays blocked. */
	TArray<TSharedPtr<FTestModel>> PrivModels;
	TUniquePtr<FPipelineExecutor> PrivExec;

	TSharedPtr<FTestModel> NewModel(bool bBlocks)
	{
		TSharedPtr<FTestModel> M = MakeShared<FTestModel>(bBlocks);
		PrivModels.Add(M);
		return M;
	}

	FPipelineExecutor& NewExec()
	{
		PrivExec = MakeUnique<FPipelineExecutor>();
		return *PrivExec;
	}

	void ExpectTags(const TCHAR* What, const TArray<int32>& Actual, const TArray<int32>& Expected)
	{
		TestEqual(FString::Printf(TEXT("%s (actual %s)"), What, *Join(Actual)), Join(Actual), Join(Expected));
	}

	/** Acceptance 1 for one device. */
	void RunStallTest(const FDoneDelegate& Done, EPipelineDevice Device)
	{
		FRealModel Real;
		if (!PrepareAffine(*this, Device, Real))
		{
			Done.ExecuteIfBound();
			return;
		}
		constexpr int32 N = 100;
		FPipelineExecutor& E = FPipelineExecutor::Get();
		E.ResetStats();
		const int64 Before = E.GetStats().NumCompleted;

		const TSharedRef<FLog> Log = MakeShared<FLog>();
		const TSharedRef<FRealModel> RealRef = MakeShared<FRealModel>(Real);
		FPipelineExecutor* EPtr = &E;
		Arm(Log, Done, N, [this, RealRef, EPtr, Before, Device](FLog& L)
		{
			// Catches a dropped, duplicated or reordered callback.
			TestEqual(TEXT("callbacks arrived: expected 100"), L.Records.Num(), 100);
			double Worst = 0.0;
			int32 Bad = 0;
			for (int32 k = 0; k < L.Records.Num(); ++k)
			{
				const FCbRecord& R = L.Records[k];
				if (R.Tag != k) { if (Bad++ < 5) { AddError(FString::Printf(TEXT("callback #%d carries tag %d: not in enqueue order / not exactly once"), k, R.Tag)); } continue; }
				if (!R.bGameThread) { if (Bad++ < 5) { AddError(FString::Printf(TEXT("callback %d ran off the game thread"), k)); } }
				if (!R.Result.bSuccess) { if (Bad++ < 5) { AddError(FString::Printf(TEXT("job %d failed: error %d '%s'"), k, (int32)R.Result.Error, *R.Result.Message)); } continue; }
				const FString M = CompareCase(R.Outputs, RealRef->Cases[k % RealRef->Cases.Num()], Worst);
				if (!M.IsEmpty() && Bad++ < 5) { AddError(FString::Printf(TEXT("job %d (case %d) mismatch vs golden: %s"), k, k % RealRef->Cases.Num(), *M)); }
				if (R.Result.QueueMs < 0.f || R.Result.InferenceMs < 0.f) { if (Bad++ < 5) { AddError(FString::Printf(TEXT("job %d has negative timing"), k)); } }
			}
			if (Device == EPipelineDevice::CPU && L.Records.Num() > 0)
			{
				TestTrue(TEXT("CPU request: Device expected CPU"), L.Records[0].Result.Device == EPipelineDevice::CPU);
			}
			if (Device == EPipelineDevice::GPU && L.Records.Num() > 0 && L.Records[0].Result.Device != EPipelineDevice::GPU)
			{
				if (IsGpuRequired()) { AddError(TEXT("GPU requested, MODELPIPELINES_REQUIRE_GPU=1, but the job ran on another device")); }
				else { AddInfo(TEXT("GPU requested but the job fell back to CPU")); }
			}
			const FPipelineExecutorStats S = EPtr->GetStats();
			AddInfo(FString::Printf(TEXT("worst |actual-golden| = %.3g; MaxEnqueueMs = %.3f, MaxGameThreadDispatchMs = %.3f (limit 2.0 each)"), Worst, S.MaxEnqueueMs, S.MaxGameThreadDispatchMs));
			// Catches Enqueue that waits for inference, and a game-thread drain that does real work.
			TestTrue(*FString::Printf(TEXT("MaxEnqueueMs expected <= 2.0, got %.3f"), S.MaxEnqueueMs), S.MaxEnqueueMs <= 2.0);
			TestTrue(*FString::Printf(TEXT("MaxGameThreadDispatchMs expected <= 2.0, got %.3f"), S.MaxGameThreadDispatchMs), S.MaxGameThreadDispatchMs <= 2.0);
			TestEqual(TEXT("NumCompleted should rise by 100"), S.NumCompleted - Before, (int64)100);
			TestEqual(TEXT("GetNumPending should be back to 0"), EPtr->GetNumPending(), 0);
		});

		TSet<FPipelineJobId> Ids;
		for (int32 i = 0; i < N; ++i)
		{
			const FPipelineJobId Id = E.Enqueue(Real.Model, CaseInputs(Real.Cases[i % Real.Cases.Num()]), MakeCb(Log, i));
			if (Id == 0) { AddError(FString::Printf(TEXT("Enqueue %d returned id 0"), i)); }
			if (Ids.Contains(Id)) { AddError(FString::Printf(TEXT("Enqueue %d returned a duplicate id %llu"), i, (uint64)Id)); }
			Ids.Add(Id);
		}
	}

	/**
	 * Acceptance 5 helper. The worker is blocked in job 0, Jobs are queued behind it, optionally the scheduler is
	 * cleared, then the gate opens. Expected is the run order of the tags after 0.
	 */
	void RunOrderTest(const FDoneDelegate& Done, TSharedPtr<IPipelineScheduler> Sched, bool bClearBeforeOpen,
		const TArray<FJobSpec>& Jobs, const TArray<int32>& Expected, const FString& What)
	{
		FPipelineExecutor& E = NewExec();
		const TSharedPtr<FTestModel> G = NewModel(true);
		if (Sched.IsValid()) { E.SetScheduler(Sched); }
		const TSharedRef<FLog> Log = MakeShared<FLog>();
		TArray<int32> Want;
		Want.Add(0);
		Want.Append(Expected);
		Arm(Log, Done, Jobs.Num() + 1, [this, G, Want, What](FLog& L)
		{
			ExpectTags(*(What + TEXT(": model run order")), G->GetOrder(), Want);
			ExpectTags(*(What + TEXT(": callback (completion) order")), L.Tags(), Want);
		});
		E.Enqueue(G, One(0.f), MakeCb(Log, 0));
		if (!G->WaitEntered()) { AddError(TEXT("worker never started the blocker job")); G->Open(); ScheduleFinish(Log); return; }
		for (const FJobSpec& J : Jobs)
		{
			E.Enqueue(G, One((float)J.Tag), MakeCb(Log, J.Tag), nullptr, J.Priority);
		}
		TestEqual(*(What + TEXT(": GetNumQueued with the worker blocked")), E.GetNumQueued(), Jobs.Num());
		if (bClearBeforeOpen) { E.SetScheduler(nullptr); }
		G->Open();
	}

	/** Acceptance 4 core: cancels queued job 1 among {0 running, 1, 2 queued}. */
	void RunCancelTest(const FDoneDelegate& Done)
	{
		FPipelineExecutor& E = NewExec();
		E.ResetStats();
		const TSharedPtr<FTestModel> G = NewModel(true);
		const TSharedRef<FLog> Log = MakeShared<FLog>();
		FPipelineExecutor* EPtr = &E;
		const TSharedRef<FPipelineJobId> IdA = MakeShared<FPipelineJobId>(0);
		Arm(Log, Done, 3, [this, G, EPtr, IdA](FLog& L)
		{
			ExpectTags(TEXT("model run order: cancelled job must never run"), G->GetOrder(), TArray<int32>({0, 2}));
			const int32 IB = L.IndexOfTag(1);
			const int32 IA = L.IndexOfTag(0);
			const int32 IC = L.IndexOfTag(2);
			TestTrue(TEXT("all three callbacks arrived exactly once"), L.Records.Num() == 3 && IA != INDEX_NONE && IB != INDEX_NONE && IC != INDEX_NONE);
			if (IB != INDEX_NONE)
			{
				const FCbRecord& R = L.Records[IB];
				TestFalse(TEXT("cancelled job: bSuccess should be false"), R.Result.bSuccess);
				TestTrue(TEXT("cancelled job: error should be Cancelled"), R.Result.Error == EPipelineError::Cancelled);
				TestTrue(TEXT("cancelled job: callback on game thread"), R.bGameThread);
				TestFalse(TEXT("cancelled job: callback must not run inside Cancel"), R.bInCall);
				// Completion order: the cancelled job completed while job 0 was still running.
				TestTrue(TEXT("cancelled job completes before the running job and the later queued job"), IB < IA && IB < IC);
			}
			if (IC != INDEX_NONE)
			{
				TestTrue(TEXT("surviving queued job succeeds"), L.Records[IC].Result.bSuccess);
				TestEqual(TEXT("surviving queued job output expected 2*2+1"), FirstOut(L.Records[IC]), 5.f);
			}
			TestFalse(TEXT("Cancel of a finished job returns false"), EPtr->Cancel(*IdA));
			const FPipelineExecutorStats S = EPtr->GetStats();
			TestEqual(TEXT("NumCancelled expected 1"), S.NumCancelled, (int64)1);
			TestEqual(TEXT("NumCompleted counts cancelled jobs too: expected 3"), S.NumCompleted, (int64)3);
			TestEqual(TEXT("GetNumPending expected 0"), EPtr->GetNumPending(), 0);
		});

		*IdA = E.Enqueue(G, One(0.f), MakeCb(Log, 0));
		if (!G->WaitEntered()) { AddError(TEXT("worker never started the blocker job")); G->Open(); ScheduleFinish(Log); return; }
		const FPipelineJobId IdB = E.Enqueue(G, One(1.f), MakeCb(Log, 1));
		const FPipelineJobId IdC = E.Enqueue(G, One(2.f), MakeCb(Log, 2));
		TestTrue(TEXT("ids are distinct and non-zero"), *IdA != 0 && IdB != 0 && IdC != 0 && *IdA != IdB && IdB != IdC && *IdA != IdC);
		TestEqual(TEXT("GetNumQueued: 2 waiting, running job not counted"), E.GetNumQueued(), 2);
		TestEqual(TEXT("GetNumPending: 2 queued + 1 running"), E.GetNumPending(), 3);

		Log->bInCall = true;
		const bool bFirst = E.Cancel(IdB);
		Log->bInCall = false;
		TestTrue(TEXT("Cancel of a queued job returns true"), bFirst);
		TestEqual(TEXT("GetNumQueued after first cancel"), E.GetNumQueued(), 1);
		TestFalse(TEXT("Cancel of the same job again returns false"), E.Cancel(IdB));
		TestFalse(TEXT("Cancel of the running job returns false"), E.Cancel(*IdA));
		TestFalse(TEXT("Cancel(0) returns false"), E.Cancel(0));
		TestFalse(TEXT("Cancel of an unknown id returns false"), E.Cancel(IdC + 1000000));
		TestEqual(TEXT("GetNumQueued unchanged by failed cancels"), E.GetNumQueued(), 1);
		G->Open();
	}

END_DEFINE_SPEC(FPipelineExecutorSpec)

void FPipelineExecutorSpec::Define()
{
	AfterEach([this]()
	{
		for (const TSharedPtr<FTestModel>& M : PrivModels) { M->Open(); }
		PrivExec.Reset(); // Shutdown + join; safe because every gate is open
		PrivModels.Reset();
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Stall criterion (item 1)", [this]()
	{
		LatentIt("100 enqueues on CPU: game-thread callbacks in order, golden outputs, Enqueue/dispatch <= 2 ms", LatentLimit, [this](const FDoneDelegate& Done)
		{
			RunStallTest(Done, EPipelineDevice::CPU);
		});

		Describe("GPU", [this]()
		{
			LatentIt("100 enqueues on GPU: same criteria", LatentLimit, [this](const FDoneDelegate& Done)
			{
				if (GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
				RunStallTest(Done, EPipelineDevice::GPU);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Callback never inside Enqueue (item 2)", [this]()
	{
		LatentIt("null model completes later with InvalidArgument", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			Arm(Log, Done, 1, [this](FLog& L)
			{
				TestEqual(TEXT("exactly one callback"), L.Records.Num(), 1);
				if (L.Records.Num() == 1)
				{
					TestFalse(TEXT("callback must not run inside Enqueue"), L.Records[0].bInCall);
					TestTrue(TEXT("callback on game thread"), L.Records[0].bGameThread);
					TestFalse(TEXT("bSuccess should be false"), L.Records[0].Result.bSuccess);
					TestTrue(TEXT("error expected InvalidArgument"), L.Records[0].Result.Error == EPipelineError::InvalidArgument);
				}
			});
			Log->bInCall = true;
			const FPipelineJobId Id = E.Enqueue(nullptr, One(1.f), MakeCb(Log, 7));
			Log->bInCall = false;
			TestTrue(TEXT("null model is still accepted: id > 0"), Id > 0);
			TestEqual(TEXT("callback not yet delivered when Enqueue returns"), Log->Records.Num(), 0);
		});

		LatentIt("empty inputs complete later with InvalidArgument and never call Run", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedPtr<FTestModel> M = NewModel(false);
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			Arm(Log, Done, 1, [this, M](FLog& L)
			{
				TestEqual(TEXT("exactly one callback"), L.Records.Num(), 1);
				if (L.Records.Num() == 1)
				{
					TestFalse(TEXT("callback must not run inside Enqueue"), L.Records[0].bInCall);
					TestTrue(TEXT("error expected InvalidArgument"), L.Records[0].Result.Error == EPipelineError::InvalidArgument);
				}
				// Catches a model that is run with no inputs.
				TestEqual(TEXT("Run must not be called for empty inputs"), M->Calls.GetValue(), 0);
			});
			Log->bInCall = true;
			const FPipelineJobId Id = E.Enqueue(M, TArray<FPipelineTensor>(), MakeCb(Log, 1));
			Log->bInCall = false;
			TestTrue(TEXT("empty inputs accepted: id > 0"), Id > 0);
			TestEqual(TEXT("callback not yet delivered when Enqueue returns"), Log->Records.Num(), 0);
		});

		LatentIt("a valid job enqueued from a worker thread still calls back once on the game thread", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			FPipelineExecutor* EPtr = &E;
			const TSharedPtr<FTestModel> M = NewModel(false);
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			Arm(Log, Done, 1, [this](FLog& L)
			{
				TestEqual(TEXT("exactly one callback"), L.Records.Num(), 1);
				if (L.Records.Num() == 1)
				{
					TestTrue(TEXT("callback on game thread"), L.Records[0].bGameThread);
					TestTrue(TEXT("success"), L.Records[0].Result.bSuccess);
					TestEqual(TEXT("output expected 2*20+1"), FirstOut(L.Records[0]), 41.f);
				}
			});
			TFuture<FPipelineJobId> F = Async(EAsyncExecution::Thread, [EPtr, M, Log]()
			{
				return EPtr->Enqueue(M, One(20.f), MakeCb(Log, 3));
			});
			if (!F.WaitFor(FTimespan::FromSeconds(10)) || F.Get() == 0)
			{
				AddError(TEXT("Enqueue from a worker thread did not return an id > 0 in time"));
			}
		});

		LatentIt("a failing model's error and message reach the callback", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedPtr<FTestModel> M = NewModel(false);
			M->FailWith = EPipelineError::InferenceFailed;
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			Arm(Log, Done, 1, [this](FLog& L)
			{
				if (L.Records.Num() == 1)
				{
					TestTrue(TEXT("error expected InferenceFailed"), L.Records[0].Result.Error == EPipelineError::InferenceFailed);
					TestTrue(TEXT("message preserved"), L.Records[0].Result.Message.Contains(TEXT("test model failure")));
				}
				else { AddError(FString::Printf(TEXT("expected 1 callback, got %d"), L.Records.Num())); }
			});
			E.Enqueue(M, One(1.f), MakeCb(Log, 1));
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Owner lifetime (item 3)", [this]()
	{
		LatentIt("owner garbage collected while its job is queued: callback dropped, later ownerless job fires", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedPtr<FTestModel> G = NewModel(true);
			const TSharedPtr<FTestModel> Echo = NewModel(false);
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			// Expect the blocker (0) and the sentinel (2); the owner's job (1) must never call back.
			Arm(Log, Done, 2, [this](FLog& L)
			{
				ExpectTags(TEXT("callbacks: blocker then sentinel only"), L.Tags(), TArray<int32>({0, 2}));
				TestEqual(TEXT("owner's callback must never fire"), L.CountTag(1), 0);
			}, 0.5f);

			E.Enqueue(G, One(0.f), MakeCb(Log, 0));
			if (!G->WaitEntered()) { AddError(TEXT("worker never started the blocker job")); G->Open(); ScheduleFinish(Log); return; }

			UObject* Owner = NewObject<UNNEModelData>(GetTransientPackage());
			E.Enqueue(Echo, One(1.f), MakeCb(Log, 1), Owner);
			Owner->MarkAsGarbage();
			Owner = nullptr; // dangling after the collection: never touched again
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);

			G->Open();
			E.Enqueue(Echo, One(2.f), MakeCb(Log, 2)); // sentinel: completes after the owner's job
		});

		LatentIt("ungated variant on the module executor: owner destroyed right after Enqueue", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = FPipelineExecutor::Get();
			const TSharedPtr<FTestModel> Echo = NewModel(false);
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			FPipelineExecutor* EPtr = &E;
			Arm(Log, Done, 1, [this, EPtr](FLog& L)
			{
				ExpectTags(TEXT("only the sentinel calls back"), L.Tags(), TArray<int32>({2}));
				TestEqual(TEXT("nothing left pending afterwards (dropped job is still retired)"), EPtr->GetNumPending(), 0);
			}, 0.5f);
			UObject* Owner = NewObject<UNNEModelData>(GetTransientPackage());
			E.Enqueue(Echo, One(1.f), MakeCb(Log, 1), Owner);
			Owner->MarkAsGarbage();
			Owner = nullptr;
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			E.Enqueue(Echo, One(3.f), MakeCb(Log, 2));
		});

		LatentIt("control: a live owner's callback does fire", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedPtr<FTestModel> Echo = NewModel(false);
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			// Catches an implementation that drops every job that has an owner.
			const TSharedRef<TStrongObjectPtr<UObject>> Owner = MakeShared<TStrongObjectPtr<UObject>>(NewObject<UNNEModelData>(GetTransientPackage()));
			Arm(Log, Done, 1, [this, Owner](FLog& L)
			{
				TestEqual(TEXT("live owner: callback fires exactly once"), L.CountTag(1), 1);
				if (L.Records.Num() == 1) { TestTrue(TEXT("success"), L.Records[0].Result.bSuccess); }
			});
			E.Enqueue(Echo, One(1.f), MakeCb(Log, 1), Owner->Get());
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Cancel (item 4)", [this]()
	{
		LatentIt("queued job cancels (true, callback Cancelled); repeat, running, 0 and unknown ids return false", LatentLimit, [this](const FDoneDelegate& Done)
		{
			RunCancelTest(Done);
		});

		LatentIt("cancelling every queued job drains the queue; none of them runs", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedPtr<FTestModel> G = NewModel(true);
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			Arm(Log, Done, 4, [this, G](FLog& L)
			{
				ExpectTags(TEXT("only the running job ran"), G->GetOrder(), TArray<int32>({0}));
				for (int32 Tag = 1; Tag <= 3; ++Tag)
				{
					const int32 I = L.IndexOfTag(Tag);
					TestTrue(*FString::Printf(TEXT("job %d called back once"), Tag), L.CountTag(Tag) == 1);
					if (I != INDEX_NONE)
					{
						TestTrue(*FString::Printf(TEXT("job %d error expected Cancelled"), Tag), L.Records[I].Result.Error == EPipelineError::Cancelled);
					}
				}
			});
			E.Enqueue(G, One(0.f), MakeCb(Log, 0));
			if (!G->WaitEntered()) { AddError(TEXT("worker never started the blocker job")); G->Open(); ScheduleFinish(Log); return; }
			TArray<FPipelineJobId> Ids;
			for (int32 Tag = 1; Tag <= 3; ++Tag) { Ids.Add(E.Enqueue(G, One((float)Tag), MakeCb(Log, Tag))); }
			TestEqual(TEXT("3 queued"), E.GetNumQueued(), 3);
			// Cancel from the middle out to catch an index-shifting bug in the queue.
			TestTrue(TEXT("cancel middle"), E.Cancel(Ids[1]));
			TestEqual(TEXT("2 queued"), E.GetNumQueued(), 2);
			TestTrue(TEXT("cancel last"), E.Cancel(Ids[2]));
			TestEqual(TEXT("1 queued"), E.GetNumQueued(), 1);
			TestTrue(TEXT("cancel first"), E.Cancel(Ids[0]));
			TestEqual(TEXT("0 queued"), E.GetNumQueued(), 0);
			G->Open();
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Scheduler (item 5)", [this]()
	{
		LatentIt("max-priority scheduler runs priorities 1,5,3 as 5,3,1", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Tag = priority so the order is readable.
			RunOrderTest(Done, MakeShared<FMaxPriorityScheduler>(), false, {{1, 1}, {5, 5}, {3, 3}}, {5, 3, 1}, TEXT("max-priority"));
		});

		LatentIt("scheduler runs on the worker and sees every pending job with priority, id and time", LatentLimit, [this](const FDoneDelegate& Done)
		{
			const TSharedRef<FMaxPriorityScheduler> Sched = MakeShared<FMaxPriorityScheduler>();
			FPipelineExecutor& E = NewExec();
			const TSharedPtr<FTestModel> G = NewModel(true);
			E.SetScheduler(Sched);
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			Arm(Log, Done, 4, [this, Sched](FLog& L)
			{
				ExpectTags(TEXT("Pending priorities shown to the scheduler (sorted)"), Sched->GetSeenThree(), TArray<int32>({-2, 4, 9}));
				TestEqual(TEXT("no Pending entry had Id 0 or a zero enqueue time"), Sched->BadInfo.GetValue(), 0);
				TestEqual(TEXT("scheduler must never run on the game thread"), Sched->CalledOnGameThread.GetValue(), 0);
				TestTrue(TEXT("scheduler was consulted"), Sched->Calls.GetValue() > 0);
				ExpectTags(TEXT("completion order follows priority: 3 (9), 1 (4), 2 (-2)"), L.Tags(), TArray<int32>({0, 3, 1, 2}));
			});
			E.Enqueue(G, One(0.f), MakeCb(Log, 0));
			if (!G->WaitEntered()) { AddError(TEXT("worker never started the blocker job")); G->Open(); ScheduleFinish(Log); return; }
			E.Enqueue(G, One(1.f), MakeCb(Log, 1), nullptr, 4);
			E.Enqueue(G, One(2.f), MakeCb(Log, 2), nullptr, -2);
			E.Enqueue(G, One(3.f), MakeCb(Log, 3), nullptr, 9);
			G->Open();
		});

		LatentIt("a scheduler returning 99 falls back to index 0 (FIFO)", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Same priorities everywhere; FIFO expected even though the scheduler says 99.
			RunOrderTest(Done, MakeShared<FFixedScheduler>(99), false, {{1, 0}, {2, 0}, {3, 0}}, {1, 2, 3}, TEXT("index 99"));
		});

		LatentIt("a scheduler returning -5 also falls back to index 0", LatentLimit, [this](const FDoneDelegate& Done)
		{
			RunOrderTest(Done, MakeShared<FFixedScheduler>(-5), false, {{4, 0}, {5, 0}, {6, 0}}, {4, 5, 6}, TEXT("index -5"));
		});

		LatentIt("max-priority scheduler with 4 jobs reverses ascending priorities", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches an implementation that ignores the scheduler entirely.
			RunOrderTest(Done, MakeShared<FMaxPriorityScheduler>(), false, {{1, 10}, {2, 20}, {3, 30}, {4, 40}}, {4, 3, 2, 1}, TEXT("max-priority 4 jobs"));
		});

		LatentIt("SetScheduler(nullptr) restores FIFO, ignoring priorities", LatentLimit, [this](const FDoneDelegate& Done)
		{
			RunOrderTest(Done, MakeShared<FMaxPriorityScheduler>(), true, {{1, 1}, {5, 5}, {3, 3}}, {1, 5, 3}, TEXT("nullptr restores FIFO"));
		});

		LatentIt("default scheduler is FIFO and ignores Priority", LatentLimit, [this](const FDoneDelegate& Done)
		{
			RunOrderTest(Done, nullptr, false, {{8, 2}, {6, 9}, {7, 1}}, {8, 6, 7}, TEXT("default FIFO"));
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Serialization (item 6)", [this]()
	{
		It("8 threads x 20 RunBlocking (one is the game thread): correct results, max 1 concurrent Run, never on game thread", [this]()
		{
			FPipelineExecutor& E = NewExec();
			E.ResetStats();
			const TSharedPtr<FTestModel> M = NewModel(false);
			M->SleepSeconds = 0.0005f; // widens the window in which a second concurrent Run would be seen
			const int32 NumThreads = 8;
			const int32 Iters = 20;
			const TSharedRef<TArray<FString>> Errs = MakeShared<TArray<FString>>();
			Errs->SetNum(NumThreads);
			FPipelineExecutor* EPtr = &E;

			auto Work = [EPtr, M, Errs, Iters](int32 Idx)
			{
				for (int32 i = 0; i < Iters; ++i)
				{
					const float V = (float)(Idx * 1000 + i); // unique per call: a cross-wired result shows up
					TArray<FPipelineTensor> Out;
					const FPipelineResult R = EPtr->RunBlocking(M, One(V), Out);
					if (!R.bSuccess) { (*Errs)[Idx] = FString::Printf(TEXT("call %d failed: %s"), i, *R.Message); return; }
					if (Out.Num() != 1 || Out[0].Num() != 1 || Out[0].GetFloatData()[0] != 2.f * V + 1.f)
					{
						(*Errs)[Idx] = FString::Printf(TEXT("call %d returned a wrong/foreign result for input %.0f"), i, V);
						return;
					}
				}
			};

			TArray<TFuture<void>> Futures;
			for (int32 t = 0; t < NumThreads - 1; ++t)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [Work, t]() { Work(t); }));
			}
			Work(NumThreads - 1); // the game thread is the 8th caller
			for (TFuture<void>& F : Futures)
			{
				if (!F.WaitFor(FTimespan::FromSeconds(60))) { AddError(TEXT("a RunBlocking thread did not finish within 60 s")); }
			}
			for (int32 t = 0; t < NumThreads; ++t)
			{
				TestTrue(*FString::Printf(TEXT("thread %d: %s"), t, *(*Errs)[t]), (*Errs)[t].IsEmpty());
			}
			TestEqual(TEXT("total Run calls expected 160"), M->Calls.GetValue(), NumThreads * Iters);
			// Catches two jobs running at once (no serialization).
			TestEqual(TEXT("max concurrent Run expected 1"), M->GetMaxConcurrent(), 1);
			TestEqual(TEXT("Run must never execute on the game thread"), M->GameThreadRuns.GetValue(), 0);
			PollUntil([&]() { return E.GetStats().NumCompleted >= NumThreads * Iters && E.GetNumPending() == 0; }, 5.0);
			TestEqual(TEXT("RunBlocking jobs count in NumCompleted"), E.GetStats().NumCompleted, (int64)(NumThreads * Iters));
			TestEqual(TEXT("GetNumPending expected 0"), E.GetNumPending(), 0);
		});

		It("8 threads x 20 RunBlocking on the real CPU model: every output matches the goldens", [this]()
		{
			FRealModel Real;
			if (!PrepareAffine(*this, EPipelineDevice::CPU, Real)) { return; }
			const TSharedRef<FRealModel> RealRef = MakeShared<FRealModel>(Real);
			const TSharedRef<TArray<FString>> Errs = MakeShared<TArray<FString>>();
			const int32 NumThreads = 8;
			Errs->SetNum(NumThreads);
			FPipelineExecutor* EPtr = &FPipelineExecutor::Get();
			auto Work = [EPtr, RealRef, Errs](int32 Idx)
			{
				for (int32 i = 0; i < 20; ++i)
				{
					const TSharedPtr<FJsonObject>& Case = RealRef->Cases[(Idx + i) % RealRef->Cases.Num()];
					const TArray<FPipelineTensor> In = CaseInputs(Case);
					TArray<FPipelineTensor> Out;
					const FPipelineResult R = EPtr->RunBlocking(RealRef->Model, In, Out);
					if (!R.bSuccess) { (*Errs)[Idx] = FString::Printf(TEXT("call %d failed: error %d %s"), i, (int32)R.Error, *R.Message); return; }
					double Worst = 0.0;
					const FString M = CompareCase(Out, Case, Worst);
					if (!M.IsEmpty()) { (*Errs)[Idx] = FString::Printf(TEXT("call %d mismatch: %s"), i, *M); return; }
				}
			};
			TArray<TFuture<void>> Futures;
			for (int32 t = 0; t < NumThreads - 1; ++t)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [Work, t]() { Work(t); }));
			}
			Work(NumThreads - 1);
			for (TFuture<void>& F : Futures)
			{
				if (!F.WaitFor(FTimespan::FromSeconds(120))) { AddError(TEXT("a RunBlocking thread did not finish within 120 s")); }
			}
			for (int32 t = 0; t < NumThreads; ++t)
			{
				TestTrue(*FString::Printf(TEXT("thread %d: %s"), t, *(*Errs)[t]), (*Errs)[t].IsEmpty());
			}
		});

		It("RunBlocking with a null model returns InvalidArgument without hanging", [this]()
		{
			FPipelineExecutor& E = NewExec();
			TArray<FPipelineTensor> Out;
			const TArray<FPipelineTensor> In = One(1.f);
			const FPipelineResult R = E.RunBlocking(nullptr, In, Out);
			TestFalse(TEXT("bSuccess should be false"), R.bSuccess);
			TestTrue(TEXT("error expected InvalidArgument"), R.Error == EPipelineError::InvalidArgument);
		});

		It("RunBlocking and Enqueue share one queue: a blocking call waits behind a running async job", [this]()
		{
			FPipelineExecutor& E = NewExec();
			const TSharedPtr<FTestModel> G = NewModel(true);
			const TSharedPtr<FTestModel> Echo = NewModel(false);
			E.Enqueue(G, One(0.f), [](const FPipelineResult&, TArray<FPipelineTensor>&&) {});
			if (!G->WaitEntered()) { AddError(TEXT("worker never started the blocker job")); G->Open(); return; }
			FPipelineExecutor* EPtr = &E;
			TFuture<FPipelineResult> F = Async(EAsyncExecution::Thread, [EPtr, Echo]()
			{
				TArray<FPipelineTensor> Out;
				return EPtr->RunBlocking(Echo, One(4.f), Out);
			});
			// Catches a RunBlocking that bypasses the queue and runs on the caller's thread.
			TestFalse(TEXT("RunBlocking must not finish while the worker is busy with another job"), F.WaitFor(FTimespan::FromMilliseconds(250)));
			TestEqual(TEXT("blocking model not run yet"), Echo->Calls.GetValue(), 0);
			G->Open();
			TestTrue(TEXT("RunBlocking finishes once the worker is free"), F.WaitFor(FTimespan::FromSeconds(30)));
			if (F.IsReady()) { TestTrue(TEXT("RunBlocking succeeded"), F.Get().bSuccess); }
			TestEqual(TEXT("blocking model ran exactly once"), Echo->Calls.GetValue(), 1);
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Shutdown (item 7)", [this]()
	{
		LatentIt("waits for the running job, drops queued callbacks, then refuses work", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			FPipelineExecutor* EPtr = &E;
			const TSharedPtr<FTestModel> G = NewModel(true);
			const TSharedPtr<FTestModel> Echo = NewModel(false);
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			Arm(Log, Done, 0, [this](FLog& L)
			{
				// Queued jobs 1..3 must never call back (job 0 may or may not be delivered: not asserted).
				for (int32 Tag = 1; Tag <= 3; ++Tag)
				{
					TestEqual(*FString::Printf(TEXT("queued job %d callback must never fire"), Tag), L.CountTag(Tag), 0);
				}
				TestEqual(TEXT("post-shutdown Enqueue callback must never fire"), L.CountTag(9), 0);
			}, 0.6f);

			E.Enqueue(G, One(0.f), MakeCb(Log, 0));
			if (!G->WaitEntered()) { AddError(TEXT("worker never started the blocker job")); G->Open(); ScheduleFinish(Log); return; }
			for (int32 Tag = 1; Tag <= 3; ++Tag) { E.Enqueue(Echo, One((float)Tag), MakeCb(Log, Tag)); }
			// A RunBlocking waiter that is queued behind them must be released with Cancelled.
			TFuture<FPipelineResult> Waiter = Async(EAsyncExecution::Thread, [EPtr, Echo]()
			{
				TArray<FPipelineTensor> Out;
				return EPtr->RunBlocking(Echo, One(50.f), Out);
			});
			if (!PollUntil([&]() { return E.GetNumQueued() == 4; }, 10.0)) { AddError(FString::Printf(TEXT("expected 4 queued, have %d"), E.GetNumQueued())); }

			TFuture<void> Shut = Async(EAsyncExecution::Thread, [EPtr]() { EPtr->Shutdown(); });
			// Catches a Shutdown that returns before the running job is done.
			TestFalse(TEXT("Shutdown must wait for the running job"), Shut.WaitFor(FTimespan::FromMilliseconds(250)));
			TestEqual(TEXT("Enqueue after Shutdown began returns 0"), E.Enqueue(Echo, One(9.f), MakeCb(Log, 9)), (FPipelineJobId)0);

			G->Open();
			TestTrue(TEXT("Shutdown returns once the running job is released"), Shut.WaitFor(FTimespan::FromSeconds(30)));
			TestEqual(TEXT("the running job completed before Shutdown returned"), G->Completed.GetValue(), 1);
			TestEqual(TEXT("queued jobs never ran"), Echo->Calls.GetValue(), 0);
			if (Waiter.WaitFor(FTimespan::FromSeconds(10)))
			{
				TestTrue(TEXT("queued RunBlocking released with Cancelled"), Waiter.Get().Error == EPipelineError::Cancelled);
			}
			else { AddError(TEXT("queued RunBlocking waiter was never released by Shutdown")); }

			TestEqual(TEXT("Enqueue after Shutdown returns 0"), E.Enqueue(Echo, One(9.f), MakeCb(Log, 9)), (FPipelineJobId)0);
			TArray<FPipelineTensor> Out;
			const TArray<FPipelineTensor> In = One(1.f);
			const FPipelineResult R = E.RunBlocking(Echo, In, Out);
			TestFalse(TEXT("RunBlocking after Shutdown fails"), R.bSuccess);
			TestTrue(TEXT("RunBlocking after Shutdown: error expected Cancelled"), R.Error == EPipelineError::Cancelled);
			E.Shutdown(); // second call: no-op, must not hang or crash
			TestEqual(TEXT("GetNumQueued after Shutdown"), E.GetNumQueued(), 0);
			PrivExec.Reset(); // destruction after Shutdown is clean
			ScheduleFinish(Log); // give any wrongly delivered queued callback time to show up
		});

		LatentIt("destroying an executor with completed jobs awaiting dispatch is safe", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// The game-thread drain task may run after the executor is gone; it must not touch freed memory.
			const TSharedPtr<FTestModel> Echo = NewModel(false);
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			Arm(Log, Done, 0, [this](FLog& L)
			{
				TestTrue(TEXT("no more callbacks than jobs"), L.Records.Num() <= 5);
				for (const FCbRecord& R : L.Records) { TestTrue(TEXT("late callbacks still on the game thread"), R.bGameThread); }
			}, 0.6f);
			{
				FPipelineExecutor Local;
				for (int32 i = 0; i < 5; ++i) { Local.Enqueue(Echo, One((float)i), MakeCb(Log, i)); }
			} // destructor runs Shutdown here, in the same tick the jobs were enqueued
			ScheduleFinish(Log);
		});

		It("Shutdown on an idle executor is idempotent", [this]()
		{
			FPipelineExecutor& E = NewExec();
			E.Shutdown();
			E.Shutdown();
			TestEqual(TEXT("Enqueue after Shutdown returns 0"), E.Enqueue(NewModel(false), One(1.f), [](const FPipelineResult&, TArray<FPipelineTensor>&&) {}), (FPipelineJobId)0);
			TestEqual(TEXT("GetNumPending after Shutdown"), E.GetNumPending(), 0);
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Module (item 8)", [this]()
	{
		It("Get() is the module's executor and the default backend is NNE", [this]()
		{
			FModelPipelinesCoreModule& Module = FModelPipelinesCoreModule::Get();
			TestTrue(TEXT("FPipelineExecutor::Get() is the module's executor"), &FPipelineExecutor::Get() == &Module.GetExecutor());
			TestTrue(TEXT("stable across calls"), &Module.GetExecutor() == &FModelPipelinesCoreModule::Get().GetExecutor());
			TestEqual(TEXT("default backend name expected NNE"), Module.GetDefaultBackend().GetName().ToString(), FString(TEXT("NNE")));
			TestTrue(TEXT("default backend: same object every call"), &Module.GetDefaultBackend() == &Module.GetDefaultBackend());
			TestTrue(TEXT("default backend reports CPU available"), Module.GetDefaultBackend().IsDeviceAvailable(EPipelineDevice::CPU));
		});

		It("the module executor is alive and accepts work", [this]()
		{
			const TSharedPtr<FTestModel> Echo = NewModel(false);
			TArray<FPipelineTensor> Out;
			const TArray<FPipelineTensor> In = One(3.f);
			const FPipelineResult R = FPipelineExecutor::Get().RunBlocking(Echo, In, Out);
			TestTrue(TEXT("RunBlocking on the module executor succeeds"), R.bSuccess);
			if (Out.Num() == 1) { TestEqual(TEXT("output expected 2*3+1"), Out[0].GetFloatData()[0], 7.f); }
			else { AddError(TEXT("expected exactly one output")); }
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Timing and stats (item 9)", [this]()
	{
		LatentIt("real CPU model: QueueMs >= 0, InferenceMs >= 0, Device CPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FRealModel Real;
			if (!PrepareAffine(*this, EPipelineDevice::CPU, Real)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			Arm(Log, Done, 1, [this](FLog& L)
			{
				if (L.Records.Num() != 1) { AddError(FString::Printf(TEXT("expected 1 callback, got %d"), L.Records.Num())); return; }
				const FPipelineResult& R = L.Records[0].Result;
				TestTrue(*FString::Printf(TEXT("success (error %d: %s)"), (int32)R.Error, *R.Message), R.bSuccess);
				TestTrue(*FString::Printf(TEXT("QueueMs expected >= 0, got %f"), R.QueueMs), R.QueueMs >= 0.f);
				TestTrue(*FString::Printf(TEXT("InferenceMs expected >= 0, got %f"), R.InferenceMs), R.InferenceMs >= 0.f);
				TestTrue(TEXT("Device expected CPU"), R.Device == EPipelineDevice::CPU);
			});
			FPipelineExecutor::Get().Enqueue(Real.Model, CaseInputs(Real.Cases[0]), MakeCb(Log, 1));
		});

		LatentIt("QueueMs reflects the wait behind a running job and InferenceMs/Device are preserved", LatentLimit, [this](const FDoneDelegate& Done)
		{
			FPipelineExecutor& E = NewExec();
			const TSharedPtr<FTestModel> G = NewModel(true);
			const TSharedRef<FLog> Log = MakeShared<FLog>();
			Arm(Log, Done, 2, [this](FLog& L)
			{
				const int32 I = L.IndexOfTag(1);
				if (I == INDEX_NONE) { AddError(TEXT("queued job did not call back")); return; }
				const FPipelineResult& R = L.Records[I].Result;
				// Catches a QueueMs that is never filled in (always 0) or measured from the wrong timestamp.
				TestTrue(*FString::Printf(TEXT("QueueMs expected >= 50 after a 100 ms wait, got %f"), R.QueueMs), R.QueueMs >= 50.f);
				TestEqual(TEXT("InferenceMs from the model is preserved"), R.InferenceMs, 1.5f);
				TestTrue(TEXT("Device from the model is preserved"), R.Device == EPipelineDevice::CPU);
			});
			E.Enqueue(G, One(0.f), MakeCb(Log, 0));
			if (!G->WaitEntered()) { AddError(TEXT("worker never started the blocker job")); G->Open(); ScheduleFinish(Log); return; }
			E.Enqueue(G, One(1.f), MakeCb(Log, 1));
			FPlatformProcess::Sleep(0.1f);
			G->Open();
		});

		It("ResetStats zeroes every counter; RunBlocking jobs count in NumCompleted", [this]()
		{
			FPipelineExecutor& E = NewExec();
			const TSharedPtr<FTestModel> Echo = NewModel(false);
			for (int32 i = 0; i < 3; ++i)
			{
				TArray<FPipelineTensor> Out;
				const TArray<FPipelineTensor> In = One((float)i);
				E.RunBlocking(Echo, In, Out);
			}
			PollUntil([&]() { return E.GetStats().NumCompleted >= 3; }, 2.0);
			TestEqual(TEXT("NumCompleted expected 3"), E.GetStats().NumCompleted, (int64)3);
			E.ResetStats();
			const FPipelineExecutorStats S = E.GetStats();
			TestEqual(TEXT("NumCompleted after reset"), S.NumCompleted, (int64)0);
			TestEqual(TEXT("NumCancelled after reset"), S.NumCancelled, (int64)0);
			TestEqual(TEXT("MaxEnqueueMs after reset"), S.MaxEnqueueMs, 0.0);
			TestEqual(TEXT("MaxGameThreadDispatchMs after reset"), S.MaxGameThreadDispatchMs, 0.0);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
