// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/ScopeLock.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/GarbageCollection.h"
#include "NNEModelData.h"
#include "PipelineBase.h"
#include "PipelineExecutor.h"
#include "PipelineModelPackage.h"
#include "PipelineNNEBackend.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "PipelineValidation.h"
#include "PipelineBaseTestPipeline.h"
#include "PipelineTestData.h"

using namespace UE::ModelPipelines::Tests;

namespace
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(60);
	const TCHAR* const ClassifierFolder = TEXT("tiny_text_classification");
	const TCHAR* const EmbeddingFolder = TEXT("tiny_text_embedding");
	const TCHAR* const UnknownOpFolder = TEXT("unknown_op");
	const TCHAR* const FetchHint = TEXT("Fetch: run ModelPipelines-UE/Tools/make_import_test_data.py, fetch_reference_models.py and make_text_goldens.py (see CLAUDE.md Commands) so <dev>/models/import_test and <dev>/goldens/text exist.");

	// ---- golden data ------------------------------------------------------------------------------------------

	/** The tiny_text_classification cases of text_goldens.json. */
	struct FGolden
	{
		TArray<TArray<int64>> Ids;      // unpadded
		TArray<TArray<float>> Logits;   // [3] each
		int32 MaxLen = 0;               // longest sentence in the whole golden set (the goldens were made in one batch padded to it)
	};

	bool LoadGoldenData(FAutomationTestBase& T, TSharedPtr<FGolden>& Out)
	{
		FString Err;
		TSharedPtr<FJsonObject> Root = LoadTextGoldens(Err);
		if (!Root.IsValid())
		{
			T.AddError(FString::Printf(TEXT("%s %s"), *Err, FetchHint));
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>* Models = nullptr;
		if (!Root->TryGetArrayField(TEXT("models"), Models))
		{
			T.AddError(TEXT("text_goldens.json has no 'models' array"));
			return false;
		}
		for (const TSharedPtr<FJsonValue>& MV : *Models)
		{
			const TSharedPtr<FJsonObject>* M = nullptr;
			FString Slug;
			if (!MV.IsValid() || !MV->TryGetObject(M) || !(*M)->TryGetStringField(TEXT("slug"), Slug) || Slug != ClassifierFolder)
			{
				continue;
			}
			const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
			if (!(*M)->TryGetArrayField(TEXT("cases"), Cases))
			{
				break;
			}
			TSharedPtr<FGolden> G = MakeShared<FGolden>();
			for (const TSharedPtr<FJsonValue>& CV : *Cases)
			{
				const TSharedPtr<FJsonObject>* C = nullptr;
				const TArray<TSharedPtr<FJsonValue>>* IdsA = nullptr;
				const TArray<TSharedPtr<FJsonValue>>* LogA = nullptr;
				if (!CV.IsValid() || !CV->TryGetObject(C) || !(*C)->TryGetArrayField(TEXT("input_ids"), IdsA) || !(*C)->TryGetArrayField(TEXT("logits"), LogA))
				{
					T.AddError(TEXT("a golden case lacks input_ids or logits"));
					return false;
				}
				TArray<int64> Ids;
				for (const TSharedPtr<FJsonValue>& V : *IdsA) { Ids.Add((int64)V->AsNumber()); }
				TArray<float> Lg;
				for (const TSharedPtr<FJsonValue>& V : *LogA) { Lg.Add((float)V->AsNumber()); }
				G->MaxLen = FMath::Max(G->MaxLen, Ids.Num());
				G->Ids.Add(MoveTemp(Ids));
				G->Logits.Add(MoveTemp(Lg));
			}
			if (G->Ids.Num() < 3 || G->MaxLen <= 0)
			{
				T.AddError(FString::Printf(TEXT("tiny_text_classification golden has only %d cases"), G->Ids.Num()));
				return false;
			}
			Out = G;
			return true;
		}
		T.AddError(FString::Printf(TEXT("text_goldens.json has no model with slug 'tiny_text_classification'. %s"), FetchHint));
		return false;
	}

	/** "" when every logit is within |a-e| <= 1e-5 + 1e-5|e|; otherwise a description incl. the worst abs diff. */
	FString CompareLogits(const TArray<float>& Actual, const TArray<int32>& Rows, const FGolden& G, double* OutWorst = nullptr)
	{
		const int32 Expected = Rows.Num() * 3;
		if (OutWorst) { *OutWorst = 0.0; }
		if (Actual.Num() != Expected)
		{
			return FString::Printf(TEXT("expected %d logits (%d rows x 3), got %d"), Expected, Rows.Num(), Actual.Num());
		}
		double Worst = 0.0;
		FString First;
		for (int32 r = 0; r < Rows.Num(); ++r)
		{
			for (int32 c = 0; c < 3; ++c)
			{
				const double A = Actual[r * 3 + c];
				const double E = G.Logits[Rows[r]][c];
				const double D = FMath::Abs(A - E);
				Worst = FMath::Max(Worst, D);
				if (First.IsEmpty() && !(D <= 1e-5 + 1e-5 * FMath::Abs(E)))
				{
					First = FString::Printf(TEXT("row %d (case %d) col %d: actual %.9g, expected %.9g"), r, Rows[r], c, A, E);
				}
			}
		}
		if (OutWorst) { *OutWorst = Worst; }
		return First.IsEmpty() ? FString() : FString::Printf(TEXT("%s (worst abs diff %.9g)"), *First, Worst);
	}

	// ---- stage bookkeeping --------------------------------------------------------------------------------------

	/** Records on which thread an object captured in a stage function was destroyed. Move-only. */
	struct FDtorLog
	{
		FThreadSafeCounter OnGameThread;
		FThreadSafeCounter OffGameThread;
	};

	struct FDtorToken
	{
		explicit FDtorToken(const TSharedPtr<FDtorLog>& InLog) : Log(InLog) {}
		FDtorToken(FDtorToken&& Other) : Log(MoveTemp(Other.Log)) { Other.Log.Reset(); }
		FDtorToken(const FDtorToken&) = delete;
		FDtorToken& operator=(const FDtorToken&) = delete;
		~FDtorToken()
		{
			if (Log.IsValid())
			{
				(IsInGameThread() ? Log->OnGameThread : Log->OffGameThread).Increment();
			}
		}
		TSharedPtr<FDtorLog> Log;
	};

	/** Everything one RunStages/RunStagesBlocking call records. Shared between the stage functions and the test. */
	struct FCallState
	{
		TArray<int32> Rows;
		TSharedPtr<FGolden> Golden;
		TWeakObjectPtr<UPipelineBase> Pipe;

		// behaviour
		bool bFailPre = false;
		FPipelineResult PreFail;
		bool bFailPost = false;
		FPipelineResult PostFail;
		TSharedPtr<FDtorLog> Dtor;

		// observations
		FThreadSafeCounter PreCalls, PostCalls, CompleteCalls;
		FThreadSafeCounter PreOnGT, PostOnGT, CompleteOnGT, CompleteInsideRun;
		FString InfoProblem;
		TArray<float> Logits;
		FPipelineResult Result;
		int32 PendingInComplete = -99;
		bool bInRun = false;
		TFunction<void(FCallState&)> OnComplete;
	};

	TSharedRef<FCallState> MakeState(const TSharedPtr<FGolden>& G, UPipelineBase* Pipe, std::initializer_list<int32> Rows)
	{
		TSharedRef<FCallState> S = MakeShared<FCallState>();
		S->Golden = G;
		S->Pipe = Pipe;
		for (int32 R : Rows) { S->Rows.Add(R); }
		return S;
	}

	/** Preprocess: right-padded batch (pad id 0, mask 0), padded to the longest sentence of the whole golden set. */
	FPipelineResult DoPreprocess(FCallState& S, const FPipelineModelInfo& Info, TArray<FPipelineTensor>& Out)
	{
		S.PreCalls.Increment();
		if (IsInGameThread()) { S.PreOnGT.Increment(); }
		if (Info.InputDescs.Num() != 2 || Info.OutputDescs.Num() != 1 || Info.Bindings.InputIds < 0 || Info.Bindings.AttentionMask < 0 || Info.Bindings.MainOutput < 0)
		{
			S.InfoProblem = FString::Printf(TEXT("FPipelineModelInfo unexpected: %d inputs, %d outputs, ids=%d mask=%d main=%d"),
				Info.InputDescs.Num(), Info.OutputDescs.Num(), Info.Bindings.InputIds, Info.Bindings.AttentionMask, Info.Bindings.MainOutput);
		}
		if (S.bFailPre)
		{
			return S.PreFail;
		}
		if (!Info.InputDescs.IsValidIndex(Info.Bindings.InputIds) || !Info.InputDescs.IsValidIndex(Info.Bindings.AttentionMask))
		{
			return FPipelineResult::Fail(EPipelineError::Internal, TEXT("test: bindings out of range"));
		}
		const int32 L = S.Golden->MaxLen;
		const int32 B = S.Rows.Num();
		TArray<int64> Ids, Mask;
		Ids.SetNumZeroed(B * L);
		Mask.SetNumZeroed(B * L);
		for (int32 r = 0; r < B; ++r)
		{
			const TArray<int64>& Src = S.Golden->Ids[S.Rows[r]];
			for (int32 i = 0; i < Src.Num(); ++i)
			{
				Ids[r * L + i] = Src[i];
				Mask[r * L + i] = 1;
			}
		}
		const TArray<int64> Shape = {(int64)B, (int64)L};
		Out.Reset();
		Out.SetNum(Info.InputDescs.Num());
		Out[Info.Bindings.InputIds] = FPipelineTensor::FromInt64(Shape, Ids);
		Out[Info.Bindings.AttentionMask] = FPipelineTensor::FromInt64(Shape, Mask);
		return FPipelineResult::Ok();
	}

	FPipelineResult DoPostprocess(FCallState& S, const FPipelineModelInfo& Info, TArray<FPipelineTensor>&& Outputs)
	{
		S.PostCalls.Increment();
		if (IsInGameThread()) { S.PostOnGT.Increment(); }
		if (Outputs.IsValidIndex(Info.Bindings.MainOutput) && Outputs[Info.Bindings.MainOutput].GetType() == EPipelineTensorType::Float32)
		{
			TConstArrayView64<float> D = Outputs[Info.Bindings.MainOutput].GetFloatData();
			S.Logits.Reset();
			S.Logits.Append(D.GetData(), (int32)D.Num());
		}
		if (S.bFailPost)
		{
			return S.PostFail;
		}
		return FPipelineResult::Ok();
	}

	FPipelineStages MakeStages(const TSharedRef<FCallState>& S)
	{
		FPipelineStages St;
		const TSharedPtr<FDtorLog> D = S->Dtor;
		if (D.IsValid())
		{
			St.Preprocess = [S, Tok = FDtorToken(D)](const FPipelineModelInfo& Info, TArray<FPipelineTensor>& Out) -> FPipelineResult
			{
				return DoPreprocess(*S, Info, Out);
			};
			St.Postprocess = [S, Tok = FDtorToken(D)](const FPipelineModelInfo& Info, TArray<FPipelineTensor>&& Outputs) -> FPipelineResult
			{
				return DoPostprocess(*S, Info, MoveTemp(Outputs));
			};
		}
		else
		{
			St.Preprocess = [S](const FPipelineModelInfo& Info, TArray<FPipelineTensor>& Out) -> FPipelineResult
			{
				return DoPreprocess(*S, Info, Out);
			};
			St.Postprocess = [S](const FPipelineModelInfo& Info, TArray<FPipelineTensor>&& Outputs) -> FPipelineResult
			{
				return DoPostprocess(*S, Info, MoveTemp(Outputs));
			};
		}
		auto Complete = [S](const FPipelineResult& R)
		{
			S->CompleteCalls.Increment();
			if (IsInGameThread()) { S->CompleteOnGT.Increment(); }
			if (S->bInRun) { S->CompleteInsideRun.Increment(); }
			S->Result = R;
			UPipelineBase* P = S->Pipe.Get();
			S->PendingInComplete = P ? P->GetNumPendingCalls() : -99;
			TFunction<void(FCallState&)> Hook = MoveTemp(S->OnComplete);
			S->OnComplete = nullptr;
			if (Hook) { Hook(*S); }
		};
		if (D.IsValid())
		{
			St.Complete = [Complete, Tok = FDtorToken(D)](const FPipelineResult& R) { Complete(R); };
		}
		else
		{
			St.Complete = Complete;
		}
		return St;
	}

	/** Runs Verify and Done on the game thread DelaySeconds from now (so duplicate or late Completes show up first). */
	void FinishLater(TFunction<void()> Verify, const FDoneDelegate& Done, float DelaySeconds = 0.3f)
	{
		FTSTicker::GetCoreTicker().AddTicker(TEXT("WP32Finish"), DelaySeconds, [Verify = MoveTemp(Verify), Done](float)
		{
			if (Verify) { Verify(); }
			Done.ExecuteIfBound();
			return false;
		});
	}

	/** Common checks that every successful golden call must pass. */
	void VerifyGoldenCall(FAutomationTestBase& T, const FString& What, FCallState& S)
	{
		T.TestEqual(*(What + TEXT(": Complete called exactly once")), S.CompleteCalls.GetValue(), 1);
		T.TestEqual(*(What + TEXT(": Complete on the game thread")), S.CompleteOnGT.GetValue(), 1);
		T.TestTrue(*FString::Printf(TEXT("%s: call succeeded (error %d: %s)"), *What, (int32)S.Result.Error, *S.Result.Message), S.Result.bSuccess);
		T.TestTrue(*(What + TEXT(": FPipelineModelInfo problem: ") + S.InfoProblem), S.InfoProblem.IsEmpty());
		double Worst = 0.0;
		const FString Cmp = CompareLogits(S.Logits, S.Rows, *S.Golden, &Worst);
		T.TestTrue(*FString::Printf(TEXT("%s: logits vs golden (tol 1e-5+1e-5|e|; worst abs diff %.9g): %s"), *What, Worst, *Cmp), Cmp.IsEmpty());
	}

	// ---- fixture ------------------------------------------------------------------------------------------------

	struct FFixture
	{
		TStrongObjectPtr<UPipelineModelPackage> Pkg;
		TStrongObjectPtr<UNNEModelData> Model;
		TStrongObjectPtr<UPipelineBaseTestPipeline> Pipe;
		TSharedPtr<FGolden> Golden;
		TSharedPtr<FPipelineNNEBackend> Backend;

		void Reset()
		{
			Pipe.Reset();
			Pkg.Reset();
			Model.Reset();
			Golden.Reset();
			Backend.Reset();
		}

		bool LoadGolden(FAutomationTestBase& T) { return LoadGoldenData(T, Golden); }

		bool MakePackage(FAutomationTestBase& T, const TCHAR* Folder, EPipelineTask Task, EPipelineDevice Device)
		{
			const FString Dir = GetDevDir() / TEXT("models/import_test") / Folder;
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
			P->Task = Task;
			P->Model = M;
			P->TokenizerJson = Tok;
			P->Labels = {TEXT("negative"), TEXT("neutral"), TEXT("positive")};
			P->Preprocess.MaxTokens = 64;
			P->Preprocess.bAddSpecialTokens = true;
			P->Postprocess.ScoreActivation = EPipelineScoreActivation::Sigmoid;
			P->License.SpdxId = TEXT("Apache-2.0");
			P->License.SourceUrl = TEXT("https://example.test/model");
			P->License.Attribution = TEXT("test");
			P->PreferredDevice = Device;
			return true;
		}

		UPipelineBaseTestPipeline* NewPipeline()
		{
			Pipe.Reset(NewObject<UPipelineBaseTestPipeline>(GetTransientPackage()));
			return Pipe.Get();
		}

		/** Golden + classifier package + initialized pipeline (default backend). */
		bool MakeReady(FAutomationTestBase& T, EPipelineDevice Device)
		{
			if (!LoadGolden(T) || !MakePackage(T, ClassifierFolder, EPipelineTask::TextClassification, Device))
			{
				return false;
			}
			const FPipelineResult R = NewPipeline()->InitializeFromPackage(Pkg.Get());
			if (!R.bSuccess)
			{
				T.AddError(FString::Printf(TEXT("InitializeFromPackage failed (error %d): %s"), (int32)R.Error, *R.Message));
				return false;
			}
			return true;
		}
	};

	/** Runs S through Pipe->Run on the game thread; Verify runs (then Done) 0.3 s after the Complete. */
	void RunAndVerify(FAutomationTestBase& T, UPipelineBaseTestPipeline* Pipe, const TSharedRef<FCallState>& S, const FDoneDelegate& Done,
		TFunction<void(FCallState&)> Verify, bool bCheckPendingOne = true)
	{
		S->OnComplete = [&T, S, Done, Verify = MoveTemp(Verify)](FCallState&)
		{
			FinishLater([&T, S, Verify]() { Verify(*S); }, Done);
		};
		S->bInRun = true;
		Pipe->Run(MakeStages(S));
		S->bInRun = false;
		T.TestEqual(TEXT("Complete must not run inside RunStages"), S->CompleteCalls.GetValue(), 0);
		if (bCheckPendingOne)
		{
			T.TestEqual(TEXT("GetNumPendingCalls right after RunStages"), Pipe->GetNumPendingCalls(), 1);
		}
	}

	bool ContainsNoCase(const FString& Hay, const TCHAR* Needle)
	{
		return Hay.Contains(Needle, ESearchCase::IgnoreCase);
	}
}

// =====================================================================================================================
BEGIN_DEFINE_SPEC(FPipelineBaseSpec, "Pipelines.Core.PipelineBase",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	FFixture Fx;
END_DEFINE_SPEC(FPipelineBaseSpec)

void FPipelineBaseSpec::Define()
{
	AfterEach([this]()
	{
		Fx.Reset();
		CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
	});

	// ------------------------------------------------------------------------------------------------------------
	Describe("Initialization", [this]()
	{
		It("valid CPU package: Ok, IsReady, task TextClassification, device Auto, package returned, no pending calls", [this]()
		{
			if (!Fx.LoadGolden(*this) || !Fx.MakePackage(*this, ClassifierFolder, EPipelineTask::TextClassification, EPipelineDevice::CPU)) { return; }
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			TestFalse(TEXT("IsReady before initialize"), P->IsReady());
			TestTrue(TEXT("GetPackage null before initialize"), P->GetPackage() == nullptr);
			const FPipelineResult R = P->InitializeFromPackage(Fx.Pkg.Get());
			TestTrue(*FString::Printf(TEXT("Initialize ok (error %d: %s)"), (int32)R.Error, *R.Message), R.bSuccess);
			TestTrue(TEXT("IsReady after initialize"), P->IsReady());
			TestTrue(TEXT("GetTask is TextClassification"), P->GetTask() == EPipelineTask::TextClassification);
			TestTrue(TEXT("GetDevice is Auto before the first call (model not initialized yet)"), P->GetDevice() == EPipelineDevice::Auto);
			TestTrue(TEXT("GetPackage returns the package"), P->GetPackage() == Fx.Pkg.Get());
			TestEqual(TEXT("no pending calls"), P->GetNumPendingCalls(), 0);
		});

		It("package task ImageClassification: WrongTask, message names both tasks, pipeline not ready", [this]()
		{
			if (!Fx.MakePackage(*this, ClassifierFolder, EPipelineTask::ImageClassification, EPipelineDevice::CPU)) { return; }
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			const FPipelineResult R = P->InitializeFromPackage(Fx.Pkg.Get());
			TestFalse(TEXT("must fail"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("error should be WrongTask, got %d"), (int32)R.Error), R.Error == EPipelineError::WrongTask);
			TestTrue(*FString::Printf(TEXT("message must name 'ImageClassification' (package task): '%s'"), *R.Message), R.Message.Contains(TEXT("ImageClassification")));
			TestTrue(*FString::Printf(TEXT("message must name 'TextClassification' (pipeline task): '%s'"), *R.Message), R.Message.Contains(TEXT("TextClassification")));
			TestFalse(TEXT("not ready"), P->IsReady());
		});

		It("task check comes before the settings check (wrong task and empty tokenizer -> WrongTask)", [this]()
		{
			// Catches an implementation that validates settings first and reports InvalidArgument.
			if (!Fx.MakePackage(*this, ClassifierFolder, EPipelineTask::TextEmbedding, EPipelineDevice::CPU)) { return; }
			Fx.Pkg->TokenizerJson.Reset();
			const FPipelineResult R = Fx.NewPipeline()->InitializeFromPackage(Fx.Pkg.Get());
			TestTrue(*FString::Printf(TEXT("error should be WrongTask, got %d: %s"), (int32)R.Error, *R.Message), R.Error == EPipelineError::WrongTask);
		});

		It("null package: InvalidArgument, not ready", [this]()
		{
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			const FPipelineResult R = P->InitializeFromPackage(nullptr);
			TestFalse(TEXT("must fail"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("error should be InvalidArgument, got %d"), (int32)R.Error), R.Error == EPipelineError::InvalidArgument);
			TestFalse(TEXT("not ready"), P->IsReady());
		});

		It("package without a model: InvalidArgument with the report's Error line as message", [this]()
		{
			if (!Fx.MakePackage(*this, ClassifierFolder, EPipelineTask::TextClassification, EPipelineDevice::CPU)) { return; }
			Fx.Pkg->Model = nullptr;
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			const FPipelineResult R = P->InitializeFromPackage(Fx.Pkg.Get());
			TestFalse(TEXT("must fail"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("error should be InvalidArgument, got %d"), (int32)R.Error), R.Error == EPipelineError::InvalidArgument);
			TestFalse(TEXT("message must not be empty"), R.Message.IsEmpty());
			TestTrue(*FString::Printf(TEXT("message should mention the model: '%s'"), *R.Message), ContainsNoCase(R.Message, TEXT("model")));
			TestFalse(TEXT("not ready"), P->IsReady());
		});

		It("invalid settings (empty tokenizer): InvalidArgument; the failed call leaves nothing behind, a fixed package then initializes", [this]()
		{
			if (!Fx.MakePackage(*this, ClassifierFolder, EPipelineTask::TextClassification, EPipelineDevice::CPU)) { return; }
			const FString Tok = Fx.Pkg->TokenizerJson;
			Fx.Pkg->TokenizerJson.Reset();
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			const FPipelineResult Bad = P->InitializeFromPackage(Fx.Pkg.Get());
			TestFalse(TEXT("must fail"), Bad.bSuccess);
			TestTrue(*FString::Printf(TEXT("error should be InvalidArgument, got %d"), (int32)Bad.Error), Bad.Error == EPipelineError::InvalidArgument);
			TestTrue(*FString::Printf(TEXT("message should name the tokenizer: '%s'"), *Bad.Message), ContainsNoCase(Bad.Message, TEXT("tokenizer")));
			TestFalse(TEXT("not ready after failure"), P->IsReady());
			Fx.Pkg->TokenizerJson = Tok;
			const FPipelineResult Good = P->InitializeFromPackage(Fx.Pkg.Get());
			TestTrue(*FString::Printf(TEXT("second attempt with a fixed package must succeed, got error %d: %s"), (int32)Good.Error, *Good.Message), Good.bSuccess);
			TestTrue(TEXT("ready"), P->IsReady());
		});

		It("second Initialize after success: InvalidArgument 'already initialized', even with a null package; stays ready", [this]()
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { return; }
			const FPipelineResult R1 = Fx.Pipe->InitializeFromPackage(Fx.Pkg.Get());
			TestFalse(TEXT("second init must fail"), R1.bSuccess);
			TestTrue(TEXT("InvalidArgument"), R1.Error == EPipelineError::InvalidArgument);
			TestTrue(*FString::Printf(TEXT("message must say 'already initialized': '%s'"), *R1.Message), ContainsNoCase(R1.Message, TEXT("already initialized")));
			const FPipelineResult R2 = Fx.Pipe->InitializeFromPackage(nullptr);
			TestTrue(*FString::Printf(TEXT("the already-initialized check precedes the null check: '%s'"), *R2.Message), ContainsNoCase(R2.Message, TEXT("already initialized")));
			TestTrue(TEXT("still ready"), Fx.Pipe->IsReady());
		});

		It("Initialize from a background thread: InvalidArgument, nothing prepared (a later game-thread call succeeds)", [this]()
		{
			if (!Fx.MakePackage(*this, ClassifierFolder, EPipelineTask::TextClassification, EPipelineDevice::CPU)) { return; }
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			UPipelineModelPackage* Pkg = Fx.Pkg.Get();
			TFuture<FPipelineResult> F = Async(EAsyncExecution::Thread, [P, Pkg]() { return P->InitializeFromPackage(Pkg); });
			if (!F.WaitFor(FTimespan::FromSeconds(30))) { AddError(TEXT("background Initialize did not return within 30 s")); return; }
			const FPipelineResult R = F.Get();
			TestFalse(TEXT("must fail off the game thread"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("error should be InvalidArgument, got %d"), (int32)R.Error), R.Error == EPipelineError::InvalidArgument);
			TestFalse(TEXT("not ready"), P->IsReady());
			const FPipelineResult Again = P->InitializeFromPackage(Pkg);
			TestTrue(*FString::Printf(TEXT("a game-thread call afterwards must succeed (no 'already initialized' leftover): %s"), *Again.Message), Again.bSuccess);
		});

		It("settings and labels are copied at initialize: later package edits do not change the pipeline", [this]()
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { return; }
			Fx.Pkg->Labels.Add(TEXT("extra"));
			Fx.Pkg->Labels[0] = TEXT("changed");
			Fx.Pkg->Preprocess.MaxTokens = 7;
			Fx.Pkg->Preprocess.bAddSpecialTokens = false;
			Fx.Pkg->Postprocess.ScoreActivation = EPipelineScoreActivation::Softmax;
			TestEqual(TEXT("labels count unchanged"), Fx.Pipe->PeekLabels().Num(), 3);
			if (Fx.Pipe->PeekLabels().Num() == 3)
			{
				TestEqual(TEXT("label 0 unchanged"), Fx.Pipe->PeekLabels()[0], FString(TEXT("negative")));
			}
			TestEqual(TEXT("MaxTokens unchanged"), Fx.Pipe->PeekPreprocessSettings().MaxTokens, 64);
			TestTrue(TEXT("bAddSpecialTokens unchanged"), Fx.Pipe->PeekPreprocessSettings().bAddSpecialTokens);
			TestTrue(TEXT("ScoreActivation unchanged (Sigmoid)"), Fx.Pipe->PeekPostprocessSettings().ScoreActivation == EPipelineScoreActivation::Sigmoid);
		});

		It("RunStagesBlocking before InitializeFromPackage: InvalidArgument, stages never called", [this]()
		{
			if (!Fx.LoadGolden(*this)) { return; }
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			TSharedRef<FCallState> S = MakeState(Fx.Golden, P, {0});
			const FPipelineResult R = P->RunBlocking(MakeStages(S));
			TestFalse(TEXT("must fail"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("error should be InvalidArgument, got %d"), (int32)R.Error), R.Error == EPipelineError::InvalidArgument);
			TestEqual(TEXT("Preprocess not called"), S->PreCalls.GetValue(), 0);
			TestEqual(TEXT("Postprocess not called"), S->PostCalls.GetValue(), 0);
		});
	});

	// ------------------------------------------------------------------------------------------------------------
	Describe("Async calls", [this]()
	{
		LatentIt("first 3 golden sentences: Complete once on the game thread, later than RunStages, golden logits, CPU, timings, pending 1 then 0", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { Done.ExecuteIfBound(); return; }
			TSharedRef<FCallState> S = MakeState(Fx.Golden, Fx.Pipe.Get(), {0, 1, 2});
			RunAndVerify(*this, Fx.Pipe.Get(), S, Done, [this, Pipe = TWeakObjectPtr<UPipelineBaseTestPipeline>(Fx.Pipe.Get())](FCallState& St)
			{
				VerifyGoldenCall(*this, TEXT("3-row call"), St);
				TestEqual(TEXT("Complete never ran inside RunStages"), St.CompleteInsideRun.GetValue(), 0);
				TestTrue(TEXT("Result.Device is CPU"), St.Result.Device == EPipelineDevice::CPU);
				TestTrue(TEXT("PreprocessMs >= 0"), St.Result.PreprocessMs >= 0.f);
				TestTrue(*FString::Printf(TEXT("InferenceMs > 0 (got %g; includes the first call's model initialization)"), St.Result.InferenceMs), St.Result.InferenceMs > 0.f);
				TestTrue(TEXT("PostprocessMs >= 0"), St.Result.PostprocessMs >= 0.f);
				TestTrue(TEXT("QueueMs >= 0"), St.Result.QueueMs >= 0.f);
				TestEqual(TEXT("Preprocess ran once"), St.PreCalls.GetValue(), 1);
				TestEqual(TEXT("Postprocess ran once"), St.PostCalls.GetValue(), 1);
				TestEqual(TEXT("Preprocess not on the game thread"), St.PreOnGT.GetValue(), 0);
				TestEqual(TEXT("Postprocess not on the game thread"), St.PostOnGT.GetValue(), 0);
				TestEqual(TEXT("pending count inside Complete is 0"), St.PendingInComplete, 0);
				if (Pipe.IsValid())
				{
					TestEqual(TEXT("pending count after Complete is 0"), Pipe->GetNumPendingCalls(), 0);
					TestTrue(TEXT("GetDevice is CPU once the model is initialized"), Pipe->GetDevice() == EPipelineDevice::CPU);
				}
			});
		});

		LatentIt("a second call on the same pipeline (cached model info) gives the golden logits for different rows", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { Done.ExecuteIfBound(); return; }
			TSharedRef<FCallState> First = MakeState(Fx.Golden, Fx.Pipe.Get(), {4});
			TSharedRef<FCallState> Second = MakeState(Fx.Golden, Fx.Pipe.Get(), {9, 3});
			UPipelineBaseTestPipeline* P = Fx.Pipe.Get();
			FCallState& Ref1 = *First;
			Ref1.OnComplete = [this, P, First, Second, Done](FCallState&)
			{
				VerifyGoldenCall(*this, TEXT("first call"), *First);
				RunAndVerify(*this, P, Second, Done, [this](FCallState& St) { VerifyGoldenCall(*this, TEXT("second call"), St); }, true);
			};
			First->bInRun = true;
			P->Run(MakeStages(First));
			First->bInRun = false;
		});

		LatentIt("Preprocess failing with PreprocessFailed 'x': Complete gets exactly that, Postprocess not called, executor job count unchanged", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { Done.ExecuteIfBound(); return; }
			TSharedRef<FCallState> S = MakeState(Fx.Golden, Fx.Pipe.Get(), {0});
			S->bFailPre = true;
			S->PreFail = FPipelineResult::Fail(EPipelineError::PreprocessFailed, TEXT("x"));
			const int64 Before = FPipelineExecutor::Get().GetStats().NumCompleted;
			RunAndVerify(*this, Fx.Pipe.Get(), S, Done, [this, Before](FCallState& St)
			{
				TestEqual(TEXT("Complete exactly once"), St.CompleteCalls.GetValue(), 1);
				TestFalse(TEXT("call failed"), St.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("error should be PreprocessFailed, got %d"), (int32)St.Result.Error), St.Result.Error == EPipelineError::PreprocessFailed);
				TestEqual(TEXT("message kept unchanged"), St.Result.Message, FString(TEXT("x")));
				TestEqual(TEXT("Postprocess must not run"), St.PostCalls.GetValue(), 0);
				TestEqual(TEXT("no inference job was enqueued"), FPipelineExecutor::Get().GetStats().NumCompleted, Before);
				TestEqual(TEXT("pending back to 0 in Complete"), St.PendingInComplete, 0);
			});
		});

		LatentIt("Postprocess failing with InferenceFailed 'y': Complete gets it with InferenceMs > 0", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { Done.ExecuteIfBound(); return; }
			TSharedRef<FCallState> S = MakeState(Fx.Golden, Fx.Pipe.Get(), {1});
			S->bFailPost = true;
			S->PostFail = FPipelineResult::Fail(EPipelineError::InferenceFailed, TEXT("y"));
			RunAndVerify(*this, Fx.Pipe.Get(), S, Done, [this](FCallState& St)
			{
				TestEqual(TEXT("Complete exactly once"), St.CompleteCalls.GetValue(), 1);
				TestFalse(TEXT("call failed"), St.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("error should be InferenceFailed, got %d"), (int32)St.Result.Error), St.Result.Error == EPipelineError::InferenceFailed);
				TestEqual(TEXT("message kept unchanged"), St.Result.Message, FString(TEXT("y")));
				TestEqual(TEXT("Preprocess ran once"), St.PreCalls.GetValue(), 1);
				TestEqual(TEXT("Postprocess ran once"), St.PostCalls.GetValue(), 1);
				TestTrue(*FString::Printf(TEXT("a failed call keeps the timings of the stages that ran: InferenceMs %g > 0"), St.Result.InferenceMs), St.Result.InferenceMs > 0.f);
			});
		});

		LatentIt("RunStages before InitializeFromPackage: Complete once on the game thread, later, with InvalidArgument", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (!Fx.LoadGolden(*this)) { Done.ExecuteIfBound(); return; }
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			TSharedRef<FCallState> S = MakeState(Fx.Golden, P, {0});
			RunAndVerify(*this, P, S, Done, [this](FCallState& St)
			{
				TestEqual(TEXT("Complete exactly once"), St.CompleteCalls.GetValue(), 1);
				TestEqual(TEXT("on the game thread"), St.CompleteOnGT.GetValue(), 1);
				TestEqual(TEXT("never inside RunStages"), St.CompleteInsideRun.GetValue(), 0);
				TestFalse(TEXT("failed"), St.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("error should be InvalidArgument, got %d"), (int32)St.Result.Error), St.Result.Error == EPipelineError::InvalidArgument);
				TestEqual(TEXT("Preprocess not called"), St.PreCalls.GetValue(), 0);
			}, /*bCheckPendingOne*/ false); // whether a not-ready call counts as pending is unspecified
		});

		LatentIt("RunStages from a background thread: Complete still runs once on the game thread with InvalidArgument", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { Done.ExecuteIfBound(); return; }
			UPipelineBaseTestPipeline* P = Fx.Pipe.Get();
			TSharedRef<FCallState> S = MakeState(Fx.Golden, P, {0});
			S->OnComplete = [this, S, Done](FCallState&)
			{
				FinishLater([this, S]()
				{
					TestEqual(TEXT("Complete exactly once"), S->CompleteCalls.GetValue(), 1);
					TestEqual(TEXT("on the game thread"), S->CompleteOnGT.GetValue(), 1);
					TestFalse(TEXT("failed"), S->Result.bSuccess);
					TestTrue(*FString::Printf(TEXT("error should be InvalidArgument, got %d"), (int32)S->Result.Error), S->Result.Error == EPipelineError::InvalidArgument);
					TestEqual(TEXT("Preprocess not called"), S->PreCalls.GetValue(), 0);
				}, Done);
			};
			TFuture<void> F = Async(EAsyncExecution::Thread, [P, S]() { P->Run(MakeStages(S)); });
			if (!F.WaitFor(FTimespan::FromSeconds(30))) { AddError(TEXT("background RunStages did not return within 30 s")); }
		});

		LatentIt("TextClassification package over the embedding model: first call ShapeMismatch naming the output, Preprocess never called; second call fails the same way", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches a runner that skips ResolvePipelineTaskIO, or one that re-runs Preprocess after a cached failure.
			if (!Fx.LoadGolden(*this) || !Fx.MakePackage(*this, EmbeddingFolder, EPipelineTask::TextClassification, EPipelineDevice::CPU)) { Done.ExecuteIfBound(); return; }
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			const FPipelineResult Init = P->InitializeFromPackage(Fx.Pkg.Get());
			if (!Init.bSuccess) { AddError(FString::Printf(TEXT("Initialize must succeed (the mismatch shows on the first call): %s"), *Init.Message)); Done.ExecuteIfBound(); return; }
			TSharedRef<FCallState> First = MakeState(Fx.Golden, P, {0});
			TSharedRef<FCallState> Second = MakeState(Fx.Golden, P, {0});
			First->OnComplete = [this, P, First, Second, Done](FCallState&)
			{
				TestEqual(TEXT("first: Complete once"), First->CompleteCalls.GetValue(), 1);
				TestTrue(*FString::Printf(TEXT("first: error should be ShapeMismatch, got %d: %s"), (int32)First->Result.Error, *First->Result.Message), First->Result.Error == EPipelineError::ShapeMismatch);
				TestTrue(*FString::Printf(TEXT("first: message should name the output ('logits' or 'output'): '%s'"), *First->Result.Message),
					ContainsNoCase(First->Result.Message, TEXT("logits")) || ContainsNoCase(First->Result.Message, TEXT("output")));
				TestEqual(TEXT("first: Preprocess never called"), First->PreCalls.GetValue(), 0);
				RunAndVerify(*this, P, Second, Done, [this, First](FCallState& St)
				{
					TestEqual(TEXT("second: Complete once"), St.CompleteCalls.GetValue(), 1);
					TestTrue(TEXT("second: ShapeMismatch again"), St.Result.Error == EPipelineError::ShapeMismatch);
					TestEqual(TEXT("second: same message (cached outcome)"), St.Result.Message, First->Result.Message);
					TestEqual(TEXT("second: Preprocess never called"), St.PreCalls.GetValue(), 0);
					TestEqual(TEXT("second: Postprocess never called"), St.PostCalls.GetValue(), 0);
				});
			};
			First->bInRun = true;
			P->Run(MakeStages(First));
			First->bInRun = false;
		});

		LatentIt("unknown_op model: ModelLoadFailed via Complete, Preprocess never called", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// ORT rejects the model at session creation; NNERuntimeORT logs it at Error level (the expected failure under test).
			AddExpectedMessage(TEXT("ONNX Runtime error 10: This is an invalid model[.] In Node.*No Op registered for ModelPipelinesNotARealOp"), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 1);
			AddExpectedMessage(TEXT("Session creation failed."), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, 1, false);
			if (!Fx.LoadGolden(*this) || !Fx.MakePackage(*this, UnknownOpFolder, EPipelineTask::TextClassification, EPipelineDevice::CPU)) { Done.ExecuteIfBound(); return; }
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			const FPipelineResult Init = P->InitializeFromPackage(Fx.Pkg.Get());
			if (!Init.bSuccess) { AddError(FString::Printf(TEXT("Initialize must succeed (ORT rejects the model only at instance creation): %s"), *Init.Message)); Done.ExecuteIfBound(); return; }
			TSharedRef<FCallState> S = MakeState(Fx.Golden, P, {0});
			RunAndVerify(*this, P, S, Done, [this](FCallState& St)
			{
				TestEqual(TEXT("Complete exactly once"), St.CompleteCalls.GetValue(), 1);
				TestFalse(TEXT("failed"), St.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("error should be ModelLoadFailed, got %d: %s"), (int32)St.Result.Error, *St.Result.Message), St.Result.Error == EPipelineError::ModelLoadFailed);
				TestEqual(TEXT("Preprocess never called"), St.PreCalls.GetValue(), 0);
				TestEqual(TEXT("Postprocess never called"), St.PostCalls.GetValue(), 0);
			});
		});

		LatentIt("100 calls queued back to back: 100 Completes, each exactly once with its own golden logits, pending 100 then 0", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { Done.ExecuteIfBound(); return; }
			struct FGroup
			{
				TArray<TSharedRef<FCallState>> States;
				int32 NumDone = 0;
				TWeakObjectPtr<UPipelineBaseTestPipeline> Pipe;
			};
			const TSharedRef<FGroup> Group = MakeShared<FGroup>();
			Group->Pipe = Fx.Pipe.Get();
			UPipelineBaseTestPipeline* P = Fx.Pipe.Get();
			const int32 N = Fx.Golden->Ids.Num();
			for (int32 i = 0; i < 100; ++i)
			{
				TSharedRef<FCallState> S = MakeState(Fx.Golden, P, {i % N, (i + 5) % N});
				S->OnComplete = [this, Group, Done](FCallState&)
				{
					if (++Group->NumDone == 100)
					{
						FinishLater([this, Group]()
						{
							int32 Bad = 0;
							for (int32 i = 0; i < Group->States.Num(); ++i)
							{
								FCallState& St = *Group->States[i];
								double Worst = 0.0;
								const FString Cmp = CompareLogits(St.Logits, St.Rows, *St.Golden, &Worst);
								if (St.CompleteCalls.GetValue() != 1 || St.CompleteOnGT.GetValue() != 1 || !St.Result.bSuccess || !Cmp.IsEmpty())
								{
									if (++Bad <= 5)
									{
										AddError(FString::Printf(TEXT("call %d: Completes %d (GT %d), success %d (%s), logits: %s"), i, St.CompleteCalls.GetValue(),
											St.CompleteOnGT.GetValue(), (int32)St.Result.bSuccess, *St.Result.Message, *Cmp));
									}
								}
							}
							TestEqual(TEXT("bad calls out of 100"), Bad, 0);
							if (Group->Pipe.IsValid())
							{
								TestEqual(TEXT("pending returns to 0"), Group->Pipe->GetNumPendingCalls(), 0);
							}
						}, Done, 0.5f);
					}
				};
				Group->States.Add(S);
			}
			for (const TSharedRef<FCallState>& S : Group->States)
			{
				P->Run(MakeStages(S));
			}
			TestEqual(TEXT("100 calls pending right after queuing them (the game thread has not ticked)"), P->GetNumPendingCalls(), 100);
		});

		LatentIt("pipeline marked garbage and collected right after RunStages: the call still completes successfully, no crash", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { Done.ExecuteIfBound(); return; }
			TSharedRef<FCallState> S = MakeState(Fx.Golden, nullptr, {0, 1, 2});
			TWeakObjectPtr<UPipelineBaseTestPipeline> Weak(Fx.Pipe.Get());
			S->OnComplete = [this, S, Done](FCallState&)
			{
				FinishLater([this, S]() { VerifyGoldenCall(*this, TEXT("call whose pipeline was destroyed"), *S); }, Done);
			};
			Fx.Pipe->Run(MakeStages(S));
			Fx.Pipe->MarkAsGarbage();
			Fx.Pipe.Reset();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS, true);
			TestFalse(TEXT("the pipeline UObject must really be destroyed (in-flight work must not keep or touch it)"), Weak.IsValid());
		});

		LatentIt("captured objects of all three stage functions are destroyed on the game thread", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches stage functions released on the background task that ran the last stage.
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { Done.ExecuteIfBound(); return; }
			const TSharedPtr<FDtorLog> Log = MakeShared<FDtorLog>();
			TSharedRef<FCallState> S = MakeState(Fx.Golden, Fx.Pipe.Get(), {2});
			S->Dtor = Log;
			RunAndVerify(*this, Fx.Pipe.Get(), S, Done, [this, Log](FCallState& St)
			{
				TestTrue(TEXT("call succeeded"), St.Result.bSuccess);
				TestEqual(TEXT("objects captured by Preprocess, Postprocess and Complete destroyed on the game thread"), Log->OnGameThread.GetValue(), 3);
				TestEqual(TEXT("none destroyed off the game thread"), Log->OffGameThread.GetValue(), 0);
			});
		});

		LatentIt("PreferredDevice GPU with a bogus GPU runtime name: Initialize ok, first call runs on CPU with golden logits", LatentLimit, [this](const FDoneDelegate& Done)
		{
			AddExpectedMessage(TEXT("GPU unavailable for this model, falling back to CPU: GPU runtime '[^']*' is not registered"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1); // the fallback warning, once (one call)
			if (!Fx.LoadGolden(*this) || !Fx.MakePackage(*this, ClassifierFolder, EPipelineTask::TextClassification, EPipelineDevice::GPU)) { Done.ExecuteIfBound(); return; }
			FPipelineNNEBackendConfig Cfg;
			Cfg.GpuRuntimeName = TEXT("NoSuchRuntime");
			Fx.Backend = MakeShared<FPipelineNNEBackend>(Cfg);
			UPipelineBaseTestPipeline* P = Fx.NewPipeline();
			const FPipelineResult Init = P->InitializeFromPackage(Fx.Pkg.Get(), *Fx.Backend);
			if (!Init.bSuccess) { AddError(FString::Printf(TEXT("Initialize with a bogus GPU runtime must still succeed (CPU fallback): %s"), *Init.Message)); Done.ExecuteIfBound(); return; }
			TSharedRef<FCallState> S = MakeState(Fx.Golden, P, {0, 1, 2});
			RunAndVerify(*this, P, S, Done, [this](FCallState& St)
			{
				VerifyGoldenCall(*this, TEXT("GPU requested, runtime missing"), St);
				TestTrue(TEXT("Result.Device must say CPU (the fallback is reported, not hidden)"), St.Result.Device == EPipelineDevice::CPU);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------------
	Describe("Blocking", [this]()
	{
		It("RunStagesBlocking on the calling thread: golden logits, stages run on the caller, CPU, timings set", [this]()
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { return; }
			TSharedRef<FCallState> S = MakeState(Fx.Golden, Fx.Pipe.Get(), {0, 1, 2});
			const FPipelineResult R = Fx.Pipe->RunBlocking(MakeStages(S));
			TestTrue(*FString::Printf(TEXT("must succeed (error %d: %s)"), (int32)R.Error, *R.Message), R.bSuccess);
			double Worst = 0.0;
			const FString Cmp = CompareLogits(S->Logits, S->Rows, *S->Golden, &Worst);
			TestTrue(*FString::Printf(TEXT("logits vs golden (worst abs diff %.9g): %s"), Worst, *Cmp), Cmp.IsEmpty());
			TestTrue(TEXT("Device CPU"), R.Device == EPipelineDevice::CPU);
			TestEqual(TEXT("Preprocess ran on the calling (game) thread"), S->PreOnGT.GetValue(), 1);
			TestEqual(TEXT("Postprocess ran on the calling (game) thread"), S->PostOnGT.GetValue(), 1);
			TestEqual(TEXT("Complete unused"), S->CompleteCalls.GetValue(), 0);
			TestTrue(TEXT("InferenceMs > 0"), R.InferenceMs > 0.f);
			TestTrue(TEXT("PreprocessMs >= 0 and PostprocessMs >= 0"), R.PreprocessMs >= 0.f && R.PostprocessMs >= 0.f);
			TestTrue(TEXT("GetDevice CPU afterwards"), Fx.Pipe->GetDevice() == EPipelineDevice::CPU);
		});

		It("RunStagesBlocking: a failing Preprocess keeps its Error and message and skips inference", [this]()
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { return; }
			TSharedRef<FCallState> S = MakeState(Fx.Golden, Fx.Pipe.Get(), {0});
			S->bFailPre = true;
			S->PreFail = FPipelineResult::Fail(EPipelineError::PreprocessFailed, TEXT("blocking x"));
			const FPipelineResult R = Fx.Pipe->RunBlocking(MakeStages(S));
			TestFalse(TEXT("failed"), R.bSuccess);
			TestTrue(TEXT("PreprocessFailed"), R.Error == EPipelineError::PreprocessFailed);
			TestEqual(TEXT("message unchanged"), R.Message, FString(TEXT("blocking x")));
			TestEqual(TEXT("Postprocess not called"), S->PostCalls.GetValue(), 0);
		});

		It("8 threads x 20 RunStagesBlocking on one pipeline (first calls race): all correct, no crash", [this]()
		{
			if (!Fx.MakeReady(*this, EPipelineDevice::CPU)) { return; }
			struct FShared
			{
				FCriticalSection Lock;
				TArray<FString> Errors;
				int32 Ok = 0;
				double Worst = 0.0;
			};
			const TSharedRef<FShared> Sh = MakeShared<FShared>();
			UPipelineBaseTestPipeline* P = Fx.Pipe.Get();
			const TSharedPtr<FGolden> G = Fx.Golden;
			const int32 N = G->Ids.Num();
			TArray<TFuture<void>> Futures;
			for (int32 t = 0; t < 8; ++t)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [P, G, Sh, t, N]()
				{
					for (int32 k = 0; k < 20; ++k)
					{
						const int32 Row = (t * 20 + k) % N;
						TSharedRef<FCallState> S = MakeState(G, nullptr, {Row, (Row + t + 1) % N});
						const FPipelineResult R = P->RunBlocking(MakeStages(S));
						double W = 0.0;
						const FString Cmp = CompareLogits(S->Logits, S->Rows, *G, &W);
						FScopeLock L(&Sh->Lock);
						Sh->Worst = FMath::Max(Sh->Worst, W);
						if (!R.bSuccess || !Cmp.IsEmpty())
						{
							if (Sh->Errors.Num() < 5)
							{
								Sh->Errors.Add(FString::Printf(TEXT("thread %d call %d: success %d (%s) %s"), t, k, (int32)R.bSuccess, *R.Message, *Cmp));
							}
						}
						else
						{
							++Sh->Ok;
						}
					}
				}));
			}
			for (TFuture<void>& F : Futures)
			{
				if (!F.WaitFor(FTimespan::FromSeconds(120))) { AddError(TEXT("a worker thread did not finish within 120 s (deadlock?)")); return; }
			}
			for (const FString& E : Sh->Errors) { AddError(E); }
			TestEqual(*FString::Printf(TEXT("successful correct calls out of 160 (worst abs diff %.9g)"), Sh->Worst), Sh->Ok, 160);
		});
	});
}

// =====================================================================================================================
// GPU cases (headless DirectML works under -nullrhi, so they are not gated on FApp::CanEverRender()).
BEGIN_DEFINE_SPEC(FPipelineBaseGpuSpec, "Pipelines.Core.PipelineBase.GPU",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	FFixture Fx;

	/** True when the GPU test may run; otherwise reports (skip info, or an error when MODELPIPELINES_REQUIRE_GPU=1). */
	bool GpuAvailableOrReport()
	{
		FPipelineNNEBackend Probe;
		if (Probe.IsDeviceAvailable(EPipelineDevice::GPU))
		{
			return true;
		}
		if (IsGpuRequired())
		{
			AddError(TEXT("GPU not available, but MODELPIPELINES_REQUIRE_GPU=1"));
		}
		else
		{
			AddInfo(TEXT("GPU not available: skipped"));
		}
		return false;
	}
END_DEFINE_SPEC(FPipelineBaseGpuSpec)

void FPipelineBaseGpuSpec::Define()
{
	AfterEach([this]()
	{
		Fx.Reset();
		CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
	});

	LatentIt("PreferredDevice GPU: Result.Device GPU, golden logits within 1e-5 + 1e-5|e|", LatentLimit, [this](const FDoneDelegate& Done)
	{
		if (!GpuAvailableOrReport()) { Done.ExecuteIfBound(); return; }
		if (!Fx.MakeReady(*this, EPipelineDevice::GPU)) { Done.ExecuteIfBound(); return; }
		TSharedRef<FCallState> S = MakeState(Fx.Golden, Fx.Pipe.Get(), {0, 1, 2});
		RunAndVerify(*this, Fx.Pipe.Get(), S, Done, [this, Pipe = TWeakObjectPtr<UPipelineBaseTestPipeline>(Fx.Pipe.Get())](FCallState& St)
		{
			VerifyGoldenCall(*this, TEXT("GPU call"), St);
			TestTrue(*FString::Printf(TEXT("Result.Device must be GPU, got %d"), (int32)St.Result.Device), St.Result.Device == EPipelineDevice::GPU);
			TestTrue(TEXT("InferenceMs > 0"), St.Result.InferenceMs > 0.f);
			if (Pipe.IsValid())
			{
				TestTrue(TEXT("GetDevice is GPU after the first call"), Pipe->GetDevice() == EPipelineDevice::GPU);
			}
		});
	});

	LatentIt("GPU: 20 calls queued back to back all complete with golden logits", LatentLimit, [this](const FDoneDelegate& Done)
	{
		if (!GpuAvailableOrReport()) { Done.ExecuteIfBound(); return; }
		if (!Fx.MakeReady(*this, EPipelineDevice::GPU)) { Done.ExecuteIfBound(); return; }
		struct FGroup
		{
			TArray<TSharedRef<FCallState>> States;
			int32 NumDone = 0;
		};
		const TSharedRef<FGroup> Group = MakeShared<FGroup>();
		UPipelineBaseTestPipeline* P = Fx.Pipe.Get();
		const int32 N = Fx.Golden->Ids.Num();
		for (int32 i = 0; i < 20; ++i)
		{
			TSharedRef<FCallState> S = MakeState(Fx.Golden, P, {i % N, (i + 3) % N});
			S->OnComplete = [this, Group, Done](FCallState&)
			{
				if (++Group->NumDone == 20)
				{
					FinishLater([this, Group]()
					{
						for (int32 i = 0; i < Group->States.Num(); ++i)
						{
							FCallState& St = *Group->States[i];
							TestEqual(*FString::Printf(TEXT("call %d Complete count"), i), St.CompleteCalls.GetValue(), 1);
							TestTrue(*FString::Printf(TEXT("call %d on GPU"), i), St.Result.Device == EPipelineDevice::GPU);
							double Worst = 0.0;
							const FString Cmp = CompareLogits(St.Logits, St.Rows, *St.Golden, &Worst);
							TestTrue(*FString::Printf(TEXT("call %d logits (worst abs diff %.9g): %s"), i, Worst, *Cmp), Cmp.IsEmpty());
						}
					}, Done, 0.5f);
				}
			};
			Group->States.Add(S);
		}
		for (const TSharedRef<FCallState>& S : Group->States)
		{
			P->Run(MakeStages(S));
		}
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
