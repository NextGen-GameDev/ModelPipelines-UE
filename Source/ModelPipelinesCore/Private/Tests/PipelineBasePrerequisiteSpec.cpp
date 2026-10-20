// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.
// WP4.3 (docs/contracts.md C26): FPipelineStages::Prerequisite gates the Preprocess stage.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Tasks/Task.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/GarbageCollection.h"
#include "NNEModelData.h"
#include "PipelineBase.h"
#include "PipelineModelPackage.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "PipelineBaseTestPipeline.h"
#include "PipelineTestData.h"

namespace PipelineBasePrerequisiteSpecPrivate
{
	using namespace UE::ModelPipelines::Tests;

	const FTimespan LatentLimit = FTimespan::FromSeconds(60);
	const TCHAR* const ClassifierFolder = TEXT("tiny_text_classification");
	const TCHAR* const FetchHint = TEXT("Fetch: run ModelPipelines-UE/Tools/make_import_test_data.py (see CLAUDE.md Commands) so <dev>/models/import_test/tiny_text_classification exists.");

	/** What one call records. Shared between the stage functions and the test. */
	struct FState
	{
		FThreadSafeCounter PreCalls, PostCalls, CompleteCalls, CompleteOnGT;
		FPipelineResult Result;
		FString InfoProblem;
	};

	FPipelineStages MakeStages(const TSharedRef<FState>& S, const UE::Tasks::FTask& Prerequisite)
	{
		FPipelineStages St;
		// One sentence of 4 tokens, all id 0 (valid in any vocabulary), mask 1. Only Ok vs. not Ok matters here.
		St.Preprocess = [S](const FPipelineModelInfo& Info, TArray<FPipelineTensor>& Out) -> FPipelineResult
		{
			S->PreCalls.Increment();
			if (!Info.InputDescs.IsValidIndex(Info.Bindings.InputIds) || !Info.InputDescs.IsValidIndex(Info.Bindings.AttentionMask))
			{
				return FPipelineResult::Fail(EPipelineError::Internal, TEXT("test: bindings out of range"));
			}
			const TArray<int64> Shape = {1, 4};
			const TArray<int64> Ids = {0, 0, 0, 0};
			const TArray<int64> Mask = {1, 1, 1, 1};
			Out.Reset();
			Out.SetNum(Info.InputDescs.Num());
			Out[Info.Bindings.InputIds] = FPipelineTensor::FromInt64(Shape, Ids);
			Out[Info.Bindings.AttentionMask] = FPipelineTensor::FromInt64(Shape, Mask);
			return FPipelineResult::Ok();
		};
		St.Postprocess = [S](const FPipelineModelInfo&, TArray<FPipelineTensor>&&) -> FPipelineResult
		{
			S->PostCalls.Increment();
			return FPipelineResult::Ok();
		};
		St.Complete = [S](const FPipelineResult& R)
		{
			S->CompleteCalls.Increment();
			if (IsInGameThread()) { S->CompleteOnGT.Increment(); }
			S->Result = R;
		};
		St.Prerequisite = Prerequisite;
		return St;
	}

	/** A task that completes when Event is triggered (the shape the Vision readback will hand over). */
	UE::Tasks::FTask GateOn(UE::Tasks::FTaskEvent& Event)
	{
		return UE::Tasks::Launch(UE_SOURCE_LOCATION, []() {}, UE::Tasks::Prerequisites(Event));
	}

	/** Polls Pred on the game thread every 20 ms; Then(true) when it held, Then(false) after TimeoutSeconds. */
	void PollUntil(TFunction<bool()> Pred, double TimeoutSeconds, TFunction<void(bool)> Then)
	{
		const double Deadline = FPlatformTime::Seconds() + TimeoutSeconds;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("WP43Poll"), 0.02f, [Pred = MoveTemp(Pred), Then = MoveTemp(Then), Deadline](float)
		{
			if (Pred()) { Then(true); return false; }
			if (FPlatformTime::Seconds() > Deadline) { Then(false); return false; }
			return true;
		});
	}

	/** Runs Fn on the game thread DelaySeconds from now. */
	void After(float DelaySeconds, TFunction<void()> Fn)
	{
		FTSTicker::GetCoreTicker().AddTicker(TEXT("WP43After"), DelaySeconds, [Fn = MoveTemp(Fn)](float) { Fn(); return false; });
	}

	struct FFixture
	{
		TStrongObjectPtr<UPipelineModelPackage> Pkg;
		TStrongObjectPtr<UNNEModelData> Model;
		TStrongObjectPtr<UPipelineBaseTestPipeline> Pipe;

		void Reset() { Pipe.Reset(); Pkg.Reset(); Model.Reset(); }

		bool MakeReady(FAutomationTestBase& T)
		{
			const FString Dir = GetDevDir() / TEXT("models/import_test") / ClassifierFolder;
			FString Err;
			UNNEModelData* M = LoadModelFile(Dir / TEXT("model.onnx"), Err);
			if (!M)
			{
				T.AddError(FString::Printf(TEXT("%s %s"), *Err, FetchHint));
				return false;
			}
			FString Tok;
			if (!FFileHelper::LoadFileToString(Tok, *(Dir / TEXT("tokenizer.json"))) || Tok.IsEmpty())
			{
				T.AddError(FString::Printf(TEXT("Missing tokenizer.json in '%s'. %s"), *Dir, FetchHint));
				return false;
			}
			Model.Reset(M);
			UPipelineModelPackage* P = NewObject<UPipelineModelPackage>(GetTransientPackage());
			Pkg.Reset(P);
			P->Task = EPipelineTask::TextClassification;
			P->Model = M;
			P->TokenizerJson = Tok;
			P->Labels = {TEXT("negative"), TEXT("neutral"), TEXT("positive")};
			P->Preprocess.MaxTokens = 64;
			P->Preprocess.bAddSpecialTokens = true;
			P->Postprocess.ScoreActivation = EPipelineScoreActivation::Sigmoid;
			P->License.SpdxId = TEXT("Apache-2.0");
			P->License.SourceUrl = TEXT("https://example.test/model");
			P->License.Attribution = TEXT("test");
			P->PreferredDevice = EPipelineDevice::CPU;
			Pipe.Reset(NewObject<UPipelineBaseTestPipeline>(GetTransientPackage()));
			const FPipelineResult R = Pipe->InitializeFromPackage(P);
			if (!R.bSuccess)
			{
				T.AddError(FString::Printf(TEXT("InitializeFromPackage failed (error %d): %s"), (int32)R.Error, *R.Message));
				return false;
			}
			return true;
		}
	};
}

BEGIN_DEFINE_SPEC(FPipelineBasePrerequisiteSpec, "Pipelines.Core.PipelineBase.Prerequisite",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	PipelineBasePrerequisiteSpecPrivate::FFixture Fx;
END_DEFINE_SPEC(FPipelineBasePrerequisiteSpec)

void FPipelineBasePrerequisiteSpec::Define()
{
	using namespace PipelineBasePrerequisiteSpecPrivate;

	AfterEach([this]()
	{
		Fx.Reset();
		CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
	});

	It("FPipelineStages::Prerequisite defaults to an invalid handle (no prerequisite)", [this]()
	{
		FPipelineStages St;
		TestFalse(TEXT("default Prerequisite.IsValid()"), St.Prerequisite.IsValid());
	});

	// Catches a Prerequisite that is ignored (Preprocess would already have run at the 0.5 s check) and one that
	// is never honoured after Trigger (Complete would never come).
	LatentIt("async: Preprocess does not start before the event is triggered, then runs; Complete exactly once on the game thread, Ok", LatentLimit, [this](const FDoneDelegate& Done)
	{
		if (!Fx.MakeReady(*this)) { Done.ExecuteIfBound(); return; }
		TSharedRef<FState> S = MakeShared<FState>();
		TSharedRef<UE::Tasks::FTaskEvent> Event = MakeShared<UE::Tasks::FTaskEvent>(TEXT("WP43Gate"));
		UE::Tasks::FTask Gate = GateOn(*Event);
		TWeakObjectPtr<UPipelineBaseTestPipeline> Pipe(Fx.Pipe.Get());
		Fx.Pipe->Run(MakeStages(S, Gate));
		TestEqual(TEXT("GetNumPendingCalls right after RunStages"), Fx.Pipe->GetNumPendingCalls(), 1);

		After(0.5f, [this, S, Event, Pipe, Done]()
		{
			TestEqual(TEXT("Preprocess must not run before the prerequisite is triggered (0.5 s later)"), S->PreCalls.GetValue(), 0);
			TestEqual(TEXT("Complete must not run before the prerequisite is triggered"), S->CompleteCalls.GetValue(), 0);
			if (Pipe.IsValid())
			{
				TestEqual(TEXT("the waiting call still counts as pending"), Pipe->GetNumPendingCalls(), 1);
			}
			Event->Trigger();
			PollUntil([S]() { return S->CompleteCalls.GetValue() > 0; }, 30.0, [this, S, Done](bool bGot)
			{
				if (!bGot)
				{
					AddError(TEXT("Complete did not run within 30 s after the event was triggered"));
					Done.ExecuteIfBound();
					return;
				}
				// Linger so a duplicate Complete would show up.
				After(0.3f, [this, S, Done]()
				{
					TestEqual(TEXT("Preprocess ran once after Trigger"), S->PreCalls.GetValue(), 1);
					TestEqual(TEXT("Postprocess ran once"), S->PostCalls.GetValue(), 1);
					TestEqual(TEXT("Complete called exactly once"), S->CompleteCalls.GetValue(), 1);
					TestEqual(TEXT("Complete on the game thread"), S->CompleteOnGT.GetValue(), 1);
					TestTrue(*FString::Printf(TEXT("call succeeded (error %d: %s)"), (int32)S->Result.Error, *S->Result.Message), S->Result.bSuccess);
					Done.ExecuteIfBound();
				});
			});
		});
	});

	// Without a prerequisite the call must not wait for anything.
	LatentIt("async: default (invalid) prerequisite behaves as before: Complete once, Ok, without any trigger", LatentLimit, [this](const FDoneDelegate& Done)
	{
		if (!Fx.MakeReady(*this)) { Done.ExecuteIfBound(); return; }
		TSharedRef<FState> S = MakeShared<FState>();
		Fx.Pipe->Run(MakeStages(S, UE::Tasks::FTask()));
		PollUntil([S]() { return S->CompleteCalls.GetValue() > 0; }, 30.0, [this, S, Done](bool bGot)
		{
			if (!bGot)
			{
				AddError(TEXT("Complete did not run within 30 s with no prerequisite"));
				Done.ExecuteIfBound();
				return;
			}
			After(0.3f, [this, S, Done]()
			{
				TestEqual(TEXT("Complete exactly once"), S->CompleteCalls.GetValue(), 1);
				TestEqual(TEXT("Preprocess once"), S->PreCalls.GetValue(), 1);
				TestTrue(*FString::Printf(TEXT("Ok (error %d: %s)"), (int32)S->Result.Error, *S->Result.Message), S->Result.bSuccess);
				Done.ExecuteIfBound();
			});
		});
	});

	// An already-completed prerequisite must not delay or block anything.
	LatentIt("async: an already-completed prerequisite lets the call run to Complete (Ok, once)", LatentLimit, [this](const FDoneDelegate& Done)
	{
		if (!Fx.MakeReady(*this)) { Done.ExecuteIfBound(); return; }
		TSharedRef<FState> S = MakeShared<FState>();
		UE::Tasks::FTask Quick = UE::Tasks::Launch(UE_SOURCE_LOCATION, []() {});
		Quick.Wait();
		Fx.Pipe->Run(MakeStages(S, Quick));
		PollUntil([S]() { return S->CompleteCalls.GetValue() > 0; }, 30.0, [this, S, Done](bool bGot)
		{
			if (!bGot)
			{
				AddError(TEXT("Complete did not run within 30 s with a completed prerequisite"));
				Done.ExecuteIfBound();
				return;
			}
			After(0.3f, [this, S, Done]()
			{
				TestEqual(TEXT("Complete exactly once"), S->CompleteCalls.GetValue(), 1);
				TestTrue(*FString::Printf(TEXT("Ok (error %d: %s)"), (int32)S->Result.Error, *S->Result.Message), S->Result.bSuccess);
				Done.ExecuteIfBound();
			});
		});
	});

	// The pipeline object may die while the gate is closed; the caller still gets its one Complete (no leak, no crash).
	LatentIt("async: pipeline destroyed (MarkAsGarbage + GC) while the event is untriggered; triggering afterwards -> Complete still runs exactly once", LatentLimit, [this](const FDoneDelegate& Done)
	{
		if (!Fx.MakeReady(*this)) { Done.ExecuteIfBound(); return; }
		TSharedRef<FState> S = MakeShared<FState>();
		TSharedRef<UE::Tasks::FTaskEvent> Event = MakeShared<UE::Tasks::FTaskEvent>(TEXT("WP43GateDestroy"));
		UE::Tasks::FTask Gate = GateOn(*Event);
		UPipelineBaseTestPipeline* P = Fx.Pipe.Get();
		P->Run(MakeStages(S, Gate));
		P->MarkAsGarbage();
		Fx.Pipe.Reset();
		CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);

		After(0.3f, [this, S, Event, Done]()
		{
			TestEqual(TEXT("Complete must not run before the trigger, even though the pipeline is gone"), S->CompleteCalls.GetValue(), 0);
			TestEqual(TEXT("Preprocess must not run before the trigger"), S->PreCalls.GetValue(), 0);
			Event->Trigger();
			PollUntil([S]() { return S->CompleteCalls.GetValue() > 0; }, 30.0, [this, S, Done](bool bGot)
			{
				if (!bGot)
				{
					AddError(TEXT("Complete never ran after the pipeline was destroyed and the event triggered (a lost callback)"));
					Done.ExecuteIfBound();
					return;
				}
				After(0.5f, [this, S, Done]()
				{
					TestEqual(TEXT("Complete exactly once"), S->CompleteCalls.GetValue(), 1);
					TestEqual(TEXT("Complete on the game thread"), S->CompleteOnGT.GetValue(), 1);
					Done.ExecuteIfBound();
				});
			});
		});
	});

	It("RunStagesBlocking from a background thread waits for the event (still running ~50 ms in), then returns Ok after Trigger", [this]()
	{
		if (!Fx.MakeReady(*this)) { return; }
		TSharedRef<FState> S = MakeShared<FState>();
		TSharedRef<UE::Tasks::FTaskEvent> Event = MakeShared<UE::Tasks::FTaskEvent>(TEXT("WP43GateBlocking"));
		UE::Tasks::FTask Gate = GateOn(*Event);
		UPipelineBaseTestPipeline* P = Fx.Pipe.Get();
		TFuture<FPipelineResult> F = Async(EAsyncExecution::Thread, [P, S, Gate]() { return P->RunBlocking(MakeStages(S, Gate)); });

		FPlatformProcess::Sleep(0.05f);
		TestFalse(TEXT("RunStagesBlocking returned before the prerequisite was triggered (prerequisite ignored)"), F.IsReady());
		TestEqual(TEXT("Preprocess must not have run yet"), S->PreCalls.GetValue(), 0);
		Event->Trigger();

		if (!F.WaitFor(FTimespan::FromSeconds(30)))
		{
			AddError(TEXT("RunStagesBlocking did not return within 30 s after the event was triggered"));
			return;
		}
		const FPipelineResult R = F.Get();
		TestTrue(*FString::Printf(TEXT("RunStagesBlocking Ok (error %d: %s)"), (int32)R.Error, *R.Message), R.bSuccess);
		TestEqual(TEXT("Preprocess ran once"), S->PreCalls.GetValue(), 1);
		TestEqual(TEXT("Postprocess ran once"), S->PostCalls.GetValue(), 1);
		TestEqual(TEXT("blocking calls do not use Complete"), S->CompleteCalls.GetValue(), 0);
		TestEqual(TEXT("no pending calls afterwards"), Fx.Pipe->GetNumPendingCalls(), 0);
	});

	It("RunStagesBlocking with the default (invalid) prerequisite returns Ok without any trigger", [this]()
	{
		if (!Fx.MakeReady(*this)) { return; }
		TSharedRef<FState> S = MakeShared<FState>();
		UPipelineBaseTestPipeline* P = Fx.Pipe.Get();
		TFuture<FPipelineResult> F = Async(EAsyncExecution::Thread, [P, S]() { return P->RunBlocking(MakeStages(S, UE::Tasks::FTask())); });
		if (!F.WaitFor(FTimespan::FromSeconds(30)))
		{
			AddError(TEXT("RunStagesBlocking without a prerequisite did not return within 30 s"));
			return;
		}
		const FPipelineResult R = F.Get();
		TestTrue(*FString::Printf(TEXT("Ok (error %d: %s)"), (int32)R.Error, *R.Message), R.bSuccess);
		TestEqual(TEXT("Preprocess ran once"), S->PreCalls.GetValue(), 1);
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
