// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

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
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineBackend.h"
#include "PipelineModelPackage.h"
#include "PipelineNNEBackend.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "TextClassificationPipeline.h"
#include "TextPipelineLibrary.h"
#include "PipelineTextTestData.h"

using namespace UE::ModelPipelines::TextTests;

namespace TextClassSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(240);
	constexpr double CallTimeoutSeconds = 150.0;
	const TCHAR* const SstSlug = TEXT("distilbert-sst2");
	const TCHAR* const TinySlug = TEXT("tiny_text_classification");

	static TSharedPtr<FJsonObject> GCachedRoot;

	static TSharedPtr<FJsonObject> GetGoldens(FAutomationTestBase& T)
	{
		if (!GCachedRoot.IsValid())
		{
			FString Err;
			GCachedRoot = LoadTextGoldens(Err);
			if (!GCachedRoot.IsValid()) { T.AddError(Err); }
		}
		return GCachedRoot;
	}

	struct FGoldenCase
	{
		FString Label;
		int32 Index = -1;
		double TopMargin = 0.0;
		TArray<double> Scores;
	};

	struct FFixture
	{
		TSharedPtr<FJsonObject> Root;
		TSharedPtr<FJsonObject> Model;
		TStrongObjectPtr<UPipelineModelPackage> Package;
		TStrongObjectPtr<UTextClassificationPipeline> Pipeline;
		TArray<FString> Ids;
		TArray<FString> Texts;
		TMap<FString, FGoldenCase> Golden;
	};

	/** Loads goldens, sentences, golden cases and a package. Mutate may change the package before the pipeline is created. */
	static bool MakeFixture(FAutomationTestBase& T, const FString& Slug, EPipelineDevice Device, FFixture& F, bool bCreatePipeline = true,
		TFunction<void(UPipelineModelPackage&)> Mutate = nullptr)
	{
		F.Root = GetGoldens(T);
		if (!F.Root.IsValid()) { return false; }
		F.Model = FindModel(F.Root, Slug);
		if (!F.Model.IsValid())
		{
			T.AddError(FString::Printf(TEXT("Golden model '%s' not found in text_goldens.json. Regenerate with Tools/make_text_goldens.py."), *Slug));
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>* Sentences = nullptr;
		if (!F.Root->TryGetArrayField(TEXT("sentences"), Sentences) || Sentences->Num() != 50)
		{
			T.AddError(TEXT("Goldens: expected 50 sentences."));
			return false;
		}
		for (const TSharedPtr<FJsonValue>& V : *Sentences)
		{
			const TSharedPtr<FJsonObject> S = V->AsObject();
			F.Ids.Add(S->GetStringField(TEXT("id")));
			F.Texts.Add(S->GetStringField(TEXT("text")));
		}
		const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
		if (F.Model->TryGetArrayField(TEXT("cases"), Cases))
		{
			for (const TSharedPtr<FJsonValue>& V : *Cases)
			{
				const TSharedPtr<FJsonObject> C = V->AsObject();
				FGoldenCase& G = F.Golden.Add(C->GetStringField(TEXT("sentence_id")));
				G.Label = C->GetStringField(TEXT("label"));
				G.Index = (int32)C->GetNumberField(TEXT("index"));
				G.TopMargin = C->GetNumberField(TEXT("top_margin"));
				const TArray<TSharedPtr<FJsonValue>>* Sc = nullptr;
				if (C->TryGetArrayField(TEXT("scores"), Sc)) { for (const TSharedPtr<FJsonValue>& X : *Sc) { G.Scores.Add(X->AsNumber()); } }
			}
		}
		FString Err;
		UPipelineModelPackage* Pkg = MakePackage(F.Root, F.Model, Device, Err);
		if (!Pkg) { T.AddError(Err); return false; }
		F.Package.Reset(Pkg);
		if (Mutate) { Mutate(*Pkg); }
		if (bCreatePipeline)
		{
			FPipelineResult R;
			UTextClassificationPipeline* P = UTextPipelineLibrary::CreateTextClassificationPipeline(Pkg, R);
			if (!P)
			{
				T.AddError(FString::Printf(TEXT("CreateTextClassificationPipeline failed: %s"), *R.Message));
				return false;
			}
			F.Pipeline.Reset(P);
		}
		return true;
	}

	static TArray<FString> Pick(const FFixture& F, std::initializer_list<int32> Indices)
	{
		TArray<FString> Out;
		for (int32 I : Indices) { Out.Add(F.Texts[I]); }
		return Out;
	}

	struct FCall
	{
		FThreadSafeCounter Count;
		bool bInCall = false;
		bool bAnyInCall = false;
		bool bAllOnGameThread = true;
		FPipelineResult Result;
		TArray<FPipelineClassification> Out;
	};

	/** ClassifyAsync on the game thread; waits on a ticker (never blocks) for the callback plus 0.3 s to catch a second one. */
	static void RunCall(FAutomationTestBase* T, TSharedRef<FFixture> Fix, UTextClassificationPipeline* Pipe, const TArray<FString>& Texts, int32 TopK,
		const FDoneDelegate& Done, TFunction<void(FCall&)> Check, TFunction<void()> AfterIssue = nullptr)
	{
		const TSharedRef<FCall> C = MakeShared<FCall>();
		C->bInCall = true;
		Pipe->ClassifyAsync(Texts, TopK, [C](const FPipelineResult& R, const TArray<FPipelineClassification>& Res)
		{
			if (!IsInGameThread()) { C->bAllOnGameThread = false; }
			if (C->bInCall) { C->bAnyInCall = true; }
			if (C->Count.Increment() == 1)
			{
				C->Result = R;
				C->Out = Res;
			}
		});
		C->bInCall = false;
		if (AfterIssue) { AfterIssue(); }

		const double Start = FPlatformTime::Seconds();
		const TSharedRef<double> FirstSeen = MakeShared<double>(0.0);
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TextClassSpecWait"), 0.02f, [T, Fix, C, Start, FirstSeen, DoneCopy, Check](float)
		{
			const double Now = FPlatformTime::Seconds();
			if (C->Count.GetValue() > 0 && *FirstSeen == 0.0) { *FirstSeen = Now; }
			const bool bTimeout = (Now - Start) > CallTimeoutSeconds;
			if ((*FirstSeen > 0.0 && (Now - *FirstSeen) > 0.3) || bTimeout)
			{
				if (C->Count.GetValue() == 0)
				{
					T->AddError(FString::Printf(TEXT("callback never fired within %.0f s"), CallTimeoutSeconds));
				}
				else
				{
					T->TestEqual(TEXT("callback fired exactly once"), C->Count.GetValue(), 1);
					T->TestTrue(TEXT("callback ran on the game thread"), C->bAllOnGameThread);
					T->TestFalse(TEXT("callback must not run inside ClassifyAsync"), C->bAnyInCall);
					if (Check) { Check(*C); }
				}
				DoneCopy.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	static void RunWorker(FAutomationTestBase* T, TSharedRef<FFixture> Fix, const FDoneDelegate& Done, TFunction<void()> Work, TFunction<void()> Check)
	{
		const TSharedRef<TFuture<void>> F = MakeShared<TFuture<void>>(Async(EAsyncExecution::ThreadPool, MoveTemp(Work)));
		const double Start = FPlatformTime::Seconds();
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TextClassSpecWorker"), 0.02f, [T, Fix, F, Start, DoneCopy, Check](float)
		{
			if (F->IsReady())
			{
				if (Check) { Check(); }
				DoneCopy.ExecuteIfBound();
				return false;
			}
			if ((FPlatformTime::Seconds() - Start) > CallTimeoutSeconds)
			{
				T->AddError(TEXT("worker thread did not finish within the time limit (hang?)"));
				DoneCopy.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	static bool GpuSkip(FAutomationTestBase& T)
	{
		FPipelineNNEBackend Backend;
		if (Backend.IsDeviceAvailable(EPipelineDevice::GPU)) { return false; }
		if (IsGpuRequired()) { T.AddError(TEXT("GPU not available but MODELPIPELINES_REQUIRE_GPU=1")); }
		else { T.AddInfo(TEXT("GPU not available: skipped")); }
		return true;
	}

	static void ExpectOneWarning(FAutomationTestBase& T)
	{
		T.AddExpectedMessage(TEXT("."), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
	}

	/**
	 * Compares every result with the golden: label/index (a flip is tolerated only when the golden top margin is below 2*Tol,
	 * which a tolerance-sized score error could legitimately cause), every score within the tolerance, Label/Index/Score == TopK[0].
	 * Returns the worst score difference.
	 */
	static double CheckClassification(FAutomationTestBase& T, const FFixture& F, const TArray<FPipelineClassification>& Res, const TArray<int32>& Indices,
		const TCHAR* Label, int32 NumClasses, double AbsTol, double RelTol)
	{
		if (!T.TestEqual(*FString::Printf(TEXT("%s: result count"), Label), Res.Num(), Indices.Num())) { return -1.0; }
		double Worst = 0.0;
		FString WorstId;
		int32 LabelMismatches = 0, Matched = 0, ScoreBad = 0;
		FString FirstLabelBad;
		for (int32 k = 0; k < Indices.Num(); ++k)
		{
			const FString& Id = F.Ids[Indices[k]];
			const FGoldenCase* G = F.Golden.Find(Id);
			if (!G) { T.AddError(FString::Printf(TEXT("%s: no golden for %s"), Label, *Id)); continue; }
			const FPipelineClassification& R = Res[k];
			if (R.Scores.Num() != NumClasses || G->Scores.Num() != NumClasses)
			{
				T.AddError(FString::Printf(TEXT("%s: %s Scores has %d entries, golden %d, expected %d"), Label, *Id, R.Scores.Num(), G->Scores.Num(), NumClasses));
				continue;
			}
			for (int32 c = 0; c < NumClasses; ++c)
			{
				const double D = FMath::Abs((double)R.Scores[c] - G->Scores[c]);
				if (D > Worst) { Worst = D; WorstId = FString::Printf(TEXT("%s[%d]"), *Id, c); }
				if (!(D <= AbsTol + RelTol * FMath::Abs(G->Scores[c]))) { ++ScoreBad; }
			}
			const bool bLabelOk = R.Label == G->Label && R.Index == G->Index;
			if (bLabelOk) { ++Matched; }
			else if (G->TopMargin >= 2.0 * AbsTol)
			{
				if (LabelMismatches++ == 0) { FirstLabelBad = FString::Printf(TEXT("%s: got '%s'(%d) golden '%s'(%d) margin %.4g"), *Id, *R.Label, R.Index, *G->Label, G->Index, G->TopMargin); }
			}
			else
			{
				++Matched;
				T.AddInfo(FString::Printf(TEXT("%s: %s label flip tolerated, golden margin %.3g"), Label, *Id, G->TopMargin));
			}
			// Internal consistency: summary fields repeat TopK[0]; TopK holds label strings of the package.
			if (R.TopK.Num() < 1)
			{
				T.AddError(FString::Printf(TEXT("%s: %s TopK empty"), Label, *Id));
			}
			else
			{
				if (R.Label != R.TopK[0].Label || R.Index != R.TopK[0].Index || R.Score != R.TopK[0].Score)
				{
					T.AddError(FString::Printf(TEXT("%s: %s Label/Index/Score do not equal TopK[0]"), Label, *Id));
				}
				if (R.Scores.IsValidIndex(R.Index) && R.Scores[R.Index] != R.Score)
				{
					T.AddError(FString::Printf(TEXT("%s: %s Score %.6f != Scores[Index] %.6f"), Label, *Id, R.Score, R.Scores[R.Index]));
				}
			}
		}
		T.AddInfo(FString::Printf(TEXT("%s: max score diff vs golden = %.3g (at %s); labels equal %d/%d"), Label, Worst, *WorstId, Matched, Indices.Num()));
		T.TestEqual(*FString::Printf(TEXT("%s: label mismatches (first: %s)"), Label, *FirstLabelBad), LabelMismatches, 0);
		T.TestEqual(*FString::Printf(TEXT("%s: scores outside tolerance %.1e + %.1e*|e| (max diff %.3g at %s)"), Label, AbsTol, RelTol, Worst, *WorstId), ScoreBad, 0);
		return Worst;
	}

	static TArray<int32> AllIndices()
	{
		TArray<int32> I;
		for (int32 k = 0; k < 50; ++k) { I.Add(k); }
		return I;
	}

	static void CheckParityResult(FAutomationTestBase& T, const FFixture& F, FCall& C, EPipelineDevice ExpectedDevice, const TCHAR* Label)
	{
		T.TestTrue(*FString::Printf(TEXT("%s: bSuccess (%s)"), Label, *C.Result.Message), C.Result.bSuccess);
		T.TestTrue(*FString::Printf(TEXT("%s: Result.Device == %d, got %d"), Label, (int32)ExpectedDevice, (int32)C.Result.Device), C.Result.Device == ExpectedDevice);
		// Brief: labels equal PyTorch, every class score within 1e-3.
		CheckClassification(T, F, C.Out, AllIndices(), Label, 2, 1e-3, 0.0);
		for (int32 i = 0; i < C.Out.Num(); ++i)
		{
			if (C.Out[i].TopK.Num() != 2) { T.AddError(FString::Printf(TEXT("%s: sentence %d TopK has %d entries, expected 2 (TopK argument 2, 2 labels)"), Label, i, C.Out[i].TopK.Num())); }
		}
	}
}

BEGIN_DEFINE_SPEC(FTextClassificationPipelineSpec, "Pipelines.Text.Classification",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FTextClassificationPipelineSpec)

void FTextClassificationPipelineSpec::Define()
{
	using namespace TextClassSpecPrivate;

	// ------------------------------------------------------------------------------------------------------
	Describe("Parity", [this]()
	{
		LatentIt("CPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 50 golden sentences in one call, TopK 2: includes the 512-token truncated one, empty, whitespace-only, emoji, CJK, accents.
			// Softmax over the wrong axis, missing [CLS]/[SEP], wrong truncation or padding leaking in moves a score by more than 1e-3.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), F->Texts, 2, Done, [this, F](FCall& C)
			{
				CheckParityResult(*this, *F, C, EPipelineDevice::CPU, TEXT("sst2 CPU"));
			});
		});

		LatentIt("GPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::GPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), F->Texts, 2, Done, [this, F](FCall& C)
			{
				CheckParityResult(*this, *F, C, EPipelineDevice::GPU, TEXT("sst2 GPU"));
			});
		});

		LatentIt("TinySigmoid", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 3-label sigmoid model vs ORT golden, |a-e| <= 1e-5 + 1e-5|e|. A pipeline that always applies softmax would make the
			// scores sum to 1; the golden scores (e.g. 0.46 + 0.97 + 0.85) do not. Checked explicitly below.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, TinySlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), F->Texts, 3, Done, [this, F](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				CheckClassification(*this, *F, C.Out, AllIndices(), TEXT("tiny sigmoid"), 3, 1e-5, 1e-5);
				int32 NotSumOne = 0;
				for (const FPipelineClassification& R : C.Out)
				{
					float Sum = 0.f;
					for (float S : R.Scores) { Sum += S; }
					if (FMath::Abs(Sum - 1.f) > 1e-3f) { ++NotSumOne; }
					TestEqual(TEXT("TopK 3 of 3 labels gives 3 entries"), R.TopK.Num(), 3);
				}
				TestTrue(*FString::Printf(TEXT("sigmoid scores must not be normalised to sum 1 (%d of 50 rows do not sum to 1)"), NotSumOne), NotSumOne > 40);
			});
		});

		LatentIt("SoftmaxScoresSumToOne", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Independent of the goldens: softmax rows over 2 classes sum to 1 within 1e-5 and lie in [0,1].
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), Pick(*F, {17, 22, 39, 41}), 1, Done, [this](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				for (const FPipelineClassification& R : C.Out)
				{
					float Sum = 0.f;
					for (float S : R.Scores) { Sum += S; TestTrue(TEXT("score in [0,1]"), S >= 0.f && S <= 1.f); }
					TestTrue(*FString::Printf(TEXT("softmax sum %.7f == 1"), Sum), FMath::Abs(Sum - 1.f) < 1e-5f);
				}
			});
		});

		LatentIt("Order", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Results come back in input order: {17 (positive review), 22 (negative review)} vs the reversed call must swap.
			// A pipeline that sorted rows by length or reused row 0 would fail. Both are also checked against the golden.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			struct FOut { TArray<FPipelineClassification> AB, BA; FPipelineResult R1, R2; };
			const TSharedRef<FOut> Out = MakeShared<FOut>();
			UTextClassificationPipeline* P = F->Pipeline.Get();
			RunWorker(this, F, Done, [F, Out, P]()
			{
				Out->R1 = P->ClassifyBlocking(Pick(*F, {17, 22}), 2, Out->AB);
				Out->R2 = P->ClassifyBlocking(Pick(*F, {22, 17}), 2, Out->BA);
			}, [this, F, Out]()
			{
				TestTrue(*FString::Printf(TEXT("AB bSuccess (%s)"), *Out->R1.Message), Out->R1.bSuccess);
				TestTrue(*FString::Printf(TEXT("BA bSuccess (%s)"), *Out->R2.Message), Out->R2.bSuccess);
				if (Out->AB.Num() != 2 || Out->BA.Num() != 2) { AddError(TEXT("expected 2 results from each call")); return; }
				TestEqual(TEXT("AB[0] label == BA[1] label"), Out->AB[0].Label, Out->BA[1].Label);
				TestEqual(TEXT("AB[1] label == BA[0] label"), Out->AB[1].Label, Out->BA[0].Label);
				TestTrue(TEXT("the two sentences have different labels"), Out->AB[0].Label != Out->AB[1].Label);
				CheckClassification(*this, *F, Out->AB, TArray<int32>{17, 22}, TEXT("order AB"), 2, 1e-3, 0.0);
				CheckClassification(*this, *F, Out->BA, TArray<int32>{22, 17}, TEXT("order BA"), 2, 1e-3, 0.0);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("TopK", [this]()
	{
		LatentIt("ZeroGivesOne", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// TopK 0 clamps to 1: exactly one TopK entry, equal to the best score.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), Pick(*F, {17, 22, 0}), 0, Done, [this](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				if (!TestEqual(TEXT("result count"), C.Out.Num(), 3)) { return; }
				for (const FPipelineClassification& R : C.Out)
				{
					TestEqual(TEXT("TopK 0 -> 1 entry"), R.TopK.Num(), 1);
					TestEqual(TEXT("Scores still has every class"), R.Scores.Num(), 2);
				}
			});
		});

		LatentIt("NegativeGivesOne", LatentLimit, [this](const FDoneDelegate& Done)
		{
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), Pick(*F, {17}), -7, Done, [this](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				if (!TestEqual(TEXT("result count"), C.Out.Num(), 1)) { return; }
				TestEqual(TEXT("TopK -7 -> 1 entry"), C.Out[0].TopK.Num(), 1);
			});
		});

		LatentIt("FiveWithTwoLabelsGivesTwoSortedBestFirst", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// TopK 5 with 2 labels clamps to 2. Entries are best first, Label == TopK[0].Label, and each entry's label/score
			// matches the label list and Scores array (an off-by-one between index and label would show here).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), Pick(*F, {17, 22, 0, 39, 41}), 5, Done, [this, F](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				if (!TestEqual(TEXT("result count"), C.Out.Num(), 5)) { return; }
				const TArray<FString> Labels = F->Package->Labels;
				for (const FPipelineClassification& R : C.Out)
				{
					if (!TestEqual(TEXT("TopK 5 of 2 labels -> 2 entries"), R.TopK.Num(), 2)) { continue; }
					TestTrue(TEXT("best first"), R.TopK[0].Score >= R.TopK[1].Score);
					TestEqual(TEXT("Label == TopK[0].Label"), R.Label, R.TopK[0].Label);
					TestEqual(TEXT("Index == TopK[0].Index"), R.Index, R.TopK[0].Index);
					TestEqual(TEXT("distinct indices"), R.TopK[0].Index + R.TopK[1].Index, 1);
					for (const FPipelineLabelScore& E : R.TopK)
					{
						if (Labels.IsValidIndex(E.Index)) { TestEqual(TEXT("entry label matches package label of its index"), E.Label, Labels[E.Index]); }
						else { AddError(FString::Printf(TEXT("entry index %d out of range"), E.Index)); }
						if (R.Scores.IsValidIndex(E.Index)) { TestEqual(TEXT("entry score == Scores[index]"), E.Score, R.Scores[E.Index]); }
					}
				}
			});
		});

		LatentIt("TinyTopTwoOfThree", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 3 labels, TopK 2: the two best sigmoid scores, descending, index-consistent with Scores.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, TinySlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), Pick(*F, {0, 17, 22, 30}), 2, Done, [this](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				if (!TestEqual(TEXT("result count"), C.Out.Num(), 4)) { return; }
				for (const FPipelineClassification& R : C.Out)
				{
					if (!TestEqual(TEXT("2 entries"), R.TopK.Num(), 2) || !TestEqual(TEXT("3 scores"), R.Scores.Num(), 3)) { continue; }
					float Max = FMath::Max3(R.Scores[0], R.Scores[1], R.Scores[2]);
					TestEqual(TEXT("TopK[0].Score is the max score"), R.TopK[0].Score, Max);
					TestTrue(TEXT("descending"), R.TopK[0].Score >= R.TopK[1].Score);
					TestTrue(TEXT("TopK[1] >= the remaining score"), R.TopK[1].Score >= R.Scores[3 - R.TopK[0].Index - R.TopK[1].Index]);
				}
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Creator", [this]()
	{
		It("WrongTask", [this]()
		{
			// An embedding package must be refused with WrongTask, not crash and not produce a pipeline.
			FFixture F;
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, F, false, [](UPipelineModelPackage& P) { P.Task = EPipelineTask::TextEmbedding; })) { return; }
			FPipelineResult R;
			UTextClassificationPipeline* P = UTextPipelineLibrary::CreateTextClassificationPipeline(F.Package.Get(), R);
			TestNull(TEXT("pipeline"), P);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("Error WrongTask, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::WrongTask);
		});

		It("NullPackage", [this]()
		{
			FPipelineResult R;
			UTextClassificationPipeline* P = UTextPipelineLibrary::CreateTextClassificationPipeline(nullptr, R);
			TestNull(TEXT("pipeline"), P);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::InvalidArgument);
		});

		LatentIt("ThreeLabelsOnTwoClassModel", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// sst2 outputs [N, 2]; a package declaring 3 labels contradicts it. The first call must fail with ShapeMismatch
			// (base task I/O check) and return no results, not read out of bounds or fabricate a third class.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F, true, [](UPipelineModelPackage& P)
			{
				P.Labels = TArray<FString>{TEXT("A"), TEXT("B"), TEXT("C")};
			})) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), Pick(*F, {0, 1}), 2, Done, [this](FCall& C)
			{
				TestFalse(TEXT("bSuccess"), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error ShapeMismatch, got %d (%s)"), (int32)C.Result.Error, *C.Result.Message), C.Result.Error == EPipelineError::ShapeMismatch);
				TestEqual(TEXT("no results on failure"), C.Out.Num(), 0);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Calls", [this]()
	{
		LatentIt("EmptyTextList", LatentLimit, [this](const FDoneDelegate& Done)
		{
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), TArray<FString>(), 1, Done, [this](FCall& C)
			{
				TestFalse(TEXT("bSuccess"), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)C.Result.Error, *C.Result.Message), C.Result.Error == EPipelineError::InvalidArgument);
				TestEqual(TEXT("no results"), C.Out.Num(), 0);
			});
		});

		LatentIt("Over1024Texts", LatentLimit, [this](const FDoneDelegate& Done)
		{
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			TArray<FString> Texts;
			Texts.Init(TEXT("hello"), 1025);
			RunCall(this, F, F->Pipeline.Get(), Texts, 1, Done, [this](FCall& C)
			{
				TestFalse(TEXT("bSuccess"), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)C.Result.Error, *C.Result.Message), C.Result.Error == EPipelineError::InvalidArgument);
				TestEqual(TEXT("no results"), C.Out.Num(), 0);
			});
		});

		It("NullCallbackWarnsAndRunsNothing", [this]()
		{
			FFixture F;
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, F)) { return; }
			ExpectOneWarning(*this);
			F.Pipeline->ClassifyAsync(Pick(F, {0}), 1, FPipelineClassificationCallback());
			TestEqual(TEXT("GetNumPendingCalls"), F.Pipeline->GetNumPendingCalls(), 0);
		});

		LatentIt("DestroyedMidCall", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Pipeline marked garbage and collected right after ClassifyAsync: the callback still fires once with correct results.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UTextClassificationPipeline* P = F->Pipeline.Get();
			RunCall(this, F, P, Pick(*F, {17, 18, 22, 23, 0}), 2, Done, [this, F](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				CheckClassification(*this, *F, C.Out, TArray<int32>{17, 18, 22, 23, 0}, TEXT("destroyed mid-call"), 2, 1e-3, 0.0);
			}, [F, P]()
			{
				P->MarkAsGarbage();
				F->Pipeline.Reset();
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			});
		});

		LatentIt("EightThreadsBlocking", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 8 threads x 5 ClassifyBlocking calls on one pipeline, different 4-sentence subsets, every result matches the golden.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, SstSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			struct FSlot { TArray<int32> Indices; FPipelineResult R; TArray<FPipelineClassification> Out; };
			const TSharedRef<TArray<FSlot>> Slots = MakeShared<TArray<FSlot>>();
			Slots->SetNum(40);
			for (int32 t = 0; t < 8; ++t)
			{
				for (int32 c = 0; c < 5; ++c)
				{
					for (int32 k = 0; k < 4; ++k) { (*Slots)[t * 5 + c].Indices.Add((t * 7 + c * 11 + k * 13 + 3) % 50); }
				}
			}
			UTextClassificationPipeline* P = F->Pipeline.Get();
			RunWorker(this, F, Done, [F, Slots, P]()
			{
				TArray<TFuture<void>> Threads;
				for (int32 t = 0; t < 8; ++t)
				{
					Threads.Add(Async(EAsyncExecution::Thread, [F, Slots, P, t]()
					{
						for (int32 c = 0; c < 5; ++c)
						{
							FSlot& S = (*Slots)[t * 5 + c];
							TArray<FString> Texts;
							for (int32 I : S.Indices) { Texts.Add(F->Texts[I]); }
							S.R = P->ClassifyBlocking(Texts, 2, S.Out);
						}
					}));
				}
				for (TFuture<void>& Th : Threads) { Th.Wait(); }
			}, [this, F, Slots]()
			{
				double Worst = 0.0;
				for (int32 s = 0; s < Slots->Num(); ++s)
				{
					const FSlot& S = (*Slots)[s];
					if (!TestTrue(*FString::Printf(TEXT("slot %d bSuccess (%s)"), s, *S.R.Message), S.R.bSuccess)) { continue; }
					Worst = FMath::Max(Worst, CheckClassification(*this, *F, S.Out, S.Indices, *FString::Printf(TEXT("slot %d"), s), 2, 1e-3, 0.0));
				}
				AddInfo(FString::Printf(TEXT("8 threads x 5 calls: max score diff %.3g"), Worst));
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
