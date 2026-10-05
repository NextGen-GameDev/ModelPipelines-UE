// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
// WP4.7 visible tests (docs/contracts.md C29, C31, C33): UImageClassificationPipeline and UVisionPipelineLibrary's classification creator.
// Names: Pipelines.Vision.Classification.*  (parity: .Parity.CPU / .Parity.GPU; texture cases: .RHI.*).
// No test blocks the game thread on an async result: waits are FTSTicker polls inside LatentIt, with a timeout and a message.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureCube.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineBackend.h"
#include "PipelineImage.h"
#include "PipelineModelPackage.h"
#include "PipelineNNEBackend.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "ImageClassificationPipeline.h"
#include "VisionPipelineLibrary.h"
#include "PipelineVisionTestData.h"

using namespace UE::ModelPipelines::VisionTests;

namespace ImageClassSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(240);
	constexpr double CallTimeoutSeconds = 150.0;
	const TCHAR* const TinySlug = TEXT("tiny_image_classification");
	const TCHAR* const ResnetSlug = TEXT("resnet-50");

	// Tiny-model tolerance from the brief: |a - e| <= 1e-5 + 1e-5 |e|.
	constexpr double TinyAbs = 1e-5;
	constexpr double TinyRel = 1e-5;

	static TSharedPtr<FJsonObject> GCachedRoot;

	static TSharedPtr<FJsonObject> GetGoldens(FAutomationTestBase& T)
	{
		if (!GCachedRoot.IsValid())
		{
			FString Err;
			GCachedRoot = LoadVisionGoldens(Err);
			if (!GCachedRoot.IsValid()) { T.AddError(Err); }
		}
		return GCachedRoot;
	}

	static TSharedPtr<FJsonObject> FindModelIn(const TSharedPtr<FJsonObject>& Root, const TCHAR* Section, const FString& Slug)
	{
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (!Root->TryGetArrayField(Section, Arr)) { return nullptr; }
		for (const TSharedPtr<FJsonValue>& V : *Arr)
		{
			const TSharedPtr<FJsonObject> O = V->AsObject();
			if (O.IsValid() && O->GetStringField(TEXT("slug")) == Slug) { return O; }
		}
		return nullptr;
	}

	static TSharedPtr<FJsonObject> FindImageEntry(const TSharedPtr<FJsonObject>& Root, const FString& Id)
	{
		for (const TCHAR* Section : { TEXT("images"), TEXT("reference_images") })
		{
			const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
			if (!Root->TryGetArrayField(Section, Arr)) { continue; }
			for (const TSharedPtr<FJsonValue>& V : *Arr)
			{
				const TSharedPtr<FJsonObject> O = V->AsObject();
				if (O.IsValid() && O->GetStringField(TEXT("id")) == Id) { return O; }
			}
		}
		return nullptr;
	}

	struct FCase
	{
		FString Id;
		int32 Index = -1;
		FString Label;
		double TopMargin = 0.0;
		TArray<double> Scores;
		FPipelineImage Image;
	};

	struct FFixture
	{
		TSharedPtr<FJsonObject> Root;
		TSharedPtr<FJsonObject> Model;
		TStrongObjectPtr<UPipelineModelPackage> Package;
		TStrongObjectPtr<UImageClassificationPipeline> Pipeline;
		TArray<FCase> Cases;
		TArray<TStrongObjectPtr<UObject>> Keep; // textures / render targets
	};

	/** Loads goldens, the model's cases (with their PNG as FPipelineImage) and a package. Mutate may change the package before the pipeline exists. */
	static bool MakeFixture(FAutomationTestBase& T, const TCHAR* Section, const FString& Slug, EPipelineDevice Device, FFixture& F, bool bCreatePipeline = true,
		TFunction<void(UPipelineModelPackage&)> Mutate = nullptr)
	{
		F.Root = GetGoldens(T);
		if (!F.Root.IsValid()) { return false; }
		F.Model = FindModelIn(F.Root, Section, Slug);
		if (!F.Model.IsValid())
		{
			T.AddError(FString::Printf(TEXT("Golden model '%s' not found in '%s' of vision_goldens.json. Regenerate with Tools/make_vision_goldens.py (and Tools/fetch_test_images.py for reference models)."), *Slug, Section));
			return false;
		}
		FString RootDir;
		if (!F.Root->TryGetStringField(TEXT("root"), RootDir) || RootDir.IsEmpty()) { RootDir = GetDevDir() / TEXT("goldens/vision"); }
		const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
		if (!F.Model->TryGetArrayField(TEXT("cases"), Cases) || Cases->Num() == 0)
		{
			T.AddError(FString::Printf(TEXT("Golden model '%s' has no cases. Regenerate with Tools/make_vision_goldens.py."), *Slug));
			return false;
		}
		for (const TSharedPtr<FJsonValue>& V : *Cases)
		{
			const TSharedPtr<FJsonObject> C = V->AsObject();
			FCase& K = F.Cases.AddDefaulted_GetRef();
			K.Id = C->GetStringField(TEXT("image"));
			K.Index = (int32)C->GetNumberField(TEXT("index"));
			K.Label = C->GetStringField(TEXT("label"));
			C->TryGetNumberField(TEXT("top_margin"), K.TopMargin);
			const TArray<TSharedPtr<FJsonValue>>* Sc = nullptr;
			if (C->TryGetArrayField(TEXT("scores"), Sc)) { for (const TSharedPtr<FJsonValue>& X : *Sc) { K.Scores.Add(X->AsNumber()); } }
			const TSharedPtr<FJsonObject> Entry = FindImageEntry(F.Root, K.Id);
			if (!Entry.IsValid())
			{
				T.AddError(FString::Printf(TEXT("Golden image '%s' not listed in images/reference_images. Regenerate with Tools/make_vision_goldens.py."), *K.Id));
				return false;
			}
			FString Err;
			if (!LoadPng(RootDir / Entry->GetStringField(TEXT("file")), K.Image, Err)) { T.AddError(Err); return false; }
		}
		FString Err;
		UPipelineModelPackage* Pkg = MakePackage(F.Root, F.Model, Device, Err);
		if (!Pkg) { T.AddError(Err); return false; }
		F.Package.Reset(Pkg);
		if (Mutate) { Mutate(*Pkg); }
		if (bCreatePipeline)
		{
			FPipelineResult R;
			UImageClassificationPipeline* P = UVisionPipelineLibrary::CreateImageClassificationPipeline(Pkg, R);
			if (!P)
			{
				T.AddError(FString::Printf(TEXT("CreateImageClassificationPipeline failed: %s"), *R.Message));
				return false;
			}
			F.Pipeline.Reset(P);
		}
		return true;
	}

	static bool MakeTiny(FAutomationTestBase& T, EPipelineDevice Device, FFixture& F, bool bCreate = true, TFunction<void(UPipelineModelPackage&)> Mutate = nullptr)
	{
		return MakeFixture(T, TEXT("tiny_models"), TinySlug, Device, F, bCreate, Mutate);
	}

	// ------------------------------------------------------------------------------------------------------
	// Calls
	// ------------------------------------------------------------------------------------------------------
	struct FCall
	{
		FThreadSafeCounter Count;
		bool bInCall = false;
		bool bAnyInCall = false;
		bool bAllOnGameThread = true;
		FPipelineResult Result;
		FPipelineClassification Out;
	};

	using FIssuer = TFunction<void(FPipelineImageClassificationCallback)>;

	static FIssuer ImageIssuer(UImageClassificationPipeline* P, const FPipelineImage& Image, int32 TopK)
	{
		return [P, Image, TopK](FPipelineImageClassificationCallback CB) { P->ClassifyImageAsync(Image, TopK, MoveTemp(CB)); };
	}

	static FIssuer TextureIssuer(UImageClassificationPipeline* P, UTexture* Tex, int32 TopK)
	{
		return [P, Tex, TopK](FPipelineImageClassificationCallback CB) { P->ClassifyAsync(Tex, TopK, MoveTemp(CB)); };
	}

	/**
	 * Issues every call on the game thread, then polls on a ticker (never blocks) until each callback fired plus 0.3 s to catch a
	 * second one, or the timeout. Checks "exactly once", "game thread" and "not inside the call" for every call, then Check.
	 */
	static void RunCalls(FAutomationTestBase* T, TSharedRef<FFixture> Fix, const TArray<FIssuer>& Issuers, const FDoneDelegate& Done,
		TFunction<void(TArray<TSharedRef<FCall>>&)> Check, TFunction<void()> AfterIssue = nullptr)
	{
		const TSharedRef<TArray<TSharedRef<FCall>>> Calls = MakeShared<TArray<TSharedRef<FCall>>>();
		for (int32 i = 0; i < Issuers.Num(); ++i) { Calls->Add(MakeShared<FCall>()); }
		for (const TSharedRef<FCall>& C : *Calls) { C->bInCall = true; }
		for (int32 i = 0; i < Issuers.Num(); ++i)
		{
			const TSharedRef<FCall> C = (*Calls)[i];
			Issuers[i]([C](const FPipelineResult& R, const FPipelineClassification& Res)
			{
				if (!IsInGameThread()) { C->bAllOnGameThread = false; }
				if (C->bInCall) { C->bAnyInCall = true; }
				if (C->Count.Increment() == 1)
				{
					C->Result = R;
					C->Out = Res;
				}
			});
		}
		for (const TSharedRef<FCall>& C : *Calls) { C->bInCall = false; }
		if (AfterIssue) { AfterIssue(); }

		const double Start = FPlatformTime::Seconds();
		const TSharedRef<double> AllSeen = MakeShared<double>(0.0);
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("ImageClassSpecWait"), 0.02f, [T, Fix, Calls, Start, AllSeen, DoneCopy, Check](float)
		{
			const double Now = FPlatformTime::Seconds();
			bool bAll = true;
			for (const TSharedRef<FCall>& C : *Calls) { if (C->Count.GetValue() == 0) { bAll = false; } }
			if (bAll && *AllSeen == 0.0) { *AllSeen = Now; }
			const bool bTimeout = (Now - Start) > CallTimeoutSeconds;
			if ((*AllSeen > 0.0 && (Now - *AllSeen) > 0.3) || bTimeout)
			{
				if (!bAll)
				{
					int32 Missing = 0;
					for (const TSharedRef<FCall>& C : *Calls) { if (C->Count.GetValue() == 0) { ++Missing; } }
					T->AddError(FString::Printf(TEXT("%d of %d callbacks never fired within %.0f s"), Missing, Calls->Num(), CallTimeoutSeconds));
				}
				else
				{
					for (int32 i = 0; i < Calls->Num(); ++i)
					{
						const TSharedRef<FCall>& C = (*Calls)[i];
						T->TestEqual(*FString::Printf(TEXT("call %d: callback fired exactly once"), i), C->Count.GetValue(), 1);
						T->TestTrue(*FString::Printf(TEXT("call %d: callback ran on the game thread"), i), C->bAllOnGameThread);
						T->TestFalse(*FString::Printf(TEXT("call %d: callback must not run inside the Classify call"), i), C->bAnyInCall);
					}
					if (Check) { Check(*Calls); }
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
		FTSTicker::GetCoreTicker().AddTicker(TEXT("ImageClassSpecWorker"), 0.02f, [T, Fix, F, Start, DoneCopy, Check](float)
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

	static bool RhiSkip(FAutomationTestBase& T)
	{
		if (HasRhi()) { return false; }
		if (IsRhiRequired()) { T.AddError(TEXT("RHI not available but MODELPIPELINES_REQUIRE_RHI=1")); }
		else { T.AddInfo(TEXT("RHI not available: skipped")); }
		return true;
	}

	static void ExpectOneWarning(FAutomationTestBase& T)
	{
		T.AddExpectedMessage(TEXT("UImageClassificationPipeline::ClassifyImageAsync: null callback, nothing runs"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
	}

	static int32 ExpectedTopKCount(int32 TopK, int32 NumLabels)
	{
		return FMath::Clamp(TopK < 1 ? 1 : TopK, 1, NumLabels);
	}

	/**
	 * Checks one result against the golden case: index, label, every score within AbsTol + RelTol*|e| (only if AbsTol >= 0),
	 * TopK length per the TopK clamp rule, TopK order recomputed independently from the result's own Scores (descending, ties lower
	 * index), summary fields repeat TopK[0]. Returns the worst score difference (or -1 when the shape is wrong).
	 */
	static double CheckCase(FAutomationTestBase& T, const FFixture& F, const FCase& G, const FPipelineClassification& R, int32 TopK, const TCHAR* Label,
		double AbsTol, double RelTol)
	{
		const TArray<FString>& Labels = F.Package->Labels;
		const int32 NumLabels = Labels.Num();
		T.TestEqual(*FString::Printf(TEXT("%s %s: Index == golden (got %d '%s', golden %d '%s', margin %.4g)"), Label, *G.Id, R.Index, *R.Label, G.Index, *G.Label, G.TopMargin), R.Index, G.Index);
		T.TestEqual(*FString::Printf(TEXT("%s %s: Label == golden"), Label, *G.Id), R.Label, G.Label);
		if (R.Scores.Num() != NumLabels)
		{
			T.AddError(FString::Printf(TEXT("%s %s: Scores has %d entries, expected %d (one per label)"), Label, *G.Id, R.Scores.Num(), NumLabels));
			return -1.0;
		}
		double Worst = 0.0;
		if (AbsTol >= 0.0)
		{
			if (G.Scores.Num() != NumLabels)
			{
				T.AddError(FString::Printf(TEXT("%s %s: golden has %d scores, package has %d labels"), Label, *G.Id, G.Scores.Num(), NumLabels));
				return -1.0;
			}
			int32 Bad = 0;
			for (int32 c = 0; c < NumLabels; ++c)
			{
				const double D = FMath::Abs((double)R.Scores[c] - G.Scores[c]);
				Worst = FMath::Max(Worst, D);
				if (!(D <= AbsTol + RelTol * FMath::Abs(G.Scores[c]))) { ++Bad; }
			}
			T.TestEqual(*FString::Printf(TEXT("%s %s: scores outside %.0e + %.0e*|e| (worst diff %.3g)"), Label, *G.Id, AbsTol, RelTol, Worst), Bad, 0);
		}
		// TopK length and order.
		const int32 Expect = ExpectedTopKCount(TopK, NumLabels);
		if (!T.TestEqual(*FString::Printf(TEXT("%s %s: TopK entries for TopK=%d with %d labels"), Label, *G.Id, TopK, NumLabels), R.TopK.Num(), Expect)) { return Worst; }
		TArray<int32> Order;
		for (int32 c = 0; c < NumLabels; ++c) { Order.Add(c); }
		Order.StableSort([&R](int32 A, int32 B) { return R.Scores[A] > R.Scores[B]; });
		for (int32 k = 0; k < Expect; ++k)
		{
			const FPipelineLabelScore& E = R.TopK[k];
			T.TestEqual(*FString::Printf(TEXT("%s %s: TopK[%d].Index (best first, ties lower index)"), Label, *G.Id, k), E.Index, Order[k]);
			if (Labels.IsValidIndex(E.Index))
			{
				T.TestEqual(*FString::Printf(TEXT("%s %s: TopK[%d].Label is the package label of its index"), Label, *G.Id, k), E.Label, Labels[E.Index]);
			}
			if (R.Scores.IsValidIndex(E.Index))
			{
				T.TestEqual(*FString::Printf(TEXT("%s %s: TopK[%d].Score == Scores[Index]"), Label, *G.Id, k), E.Score, R.Scores[E.Index]);
			}
		}
		T.TestTrue(*FString::Printf(TEXT("%s %s: Label/Index/Score repeat TopK[0]"), Label, *G.Id), R.Label == R.TopK[0].Label && R.Index == R.TopK[0].Index && R.Score == R.TopK[0].Score);
		return Worst;
	}

	static void CheckSuccess(FAutomationTestBase& T, const FPipelineResult& R, EPipelineDevice ExpectedDevice, const TCHAR* Label)
	{
		T.TestTrue(*FString::Printf(TEXT("%s: bSuccess (%s)"), Label, *R.Message), R.bSuccess);
		T.TestTrue(*FString::Printf(TEXT("%s: Result.Device == %d, got %d"), Label, (int32)ExpectedDevice, (int32)R.Device), R.Device == ExpectedDevice);
	}

	static void CheckFailure(FAutomationTestBase& T, const FCall& C, EPipelineError Expected, const TCHAR* Label)
	{
		T.TestFalse(*FString::Printf(TEXT("%s: bSuccess"), Label), C.Result.bSuccess);
		T.TestTrue(*FString::Printf(TEXT("%s: Error %d expected, got %d (%s)"), Label, (int32)Expected, (int32)C.Result.Error, *C.Result.Message), C.Result.Error == Expected);
		T.TestEqual(*FString::Printf(TEXT("%s: no scores on failure"), Label), C.Out.Scores.Num(), 0);
		T.TestEqual(*FString::Printf(TEXT("%s: no TopK on failure"), Label), C.Out.TopK.Num(), 0);
	}

	static const int32 TopKValues[4] = { 1, 3, 0, 99 };

	/** Tiny model, every case x TopK {1,3,0,99}, ClassifyImageBlocking from a worker thread. */
	static void TinyBlocking(FAutomationTestBase* T, EPipelineDevice Device, const FDoneDelegate& Done)
	{
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeTiny(*T, Device, *F)) { Done.ExecuteIfBound(); return; }
		struct FSlot { int32 Case = 0; int32 TopK = 1; FPipelineResult R; FPipelineClassification Out; };
		const TSharedRef<TArray<FSlot>> Slots = MakeShared<TArray<FSlot>>();
		for (int32 c = 0; c < F->Cases.Num(); ++c)
		{
			for (int32 K : TopKValues) { FSlot& S = Slots->AddDefaulted_GetRef(); S.Case = c; S.TopK = K; }
		}
		UImageClassificationPipeline* P = F->Pipeline.Get();
		RunWorker(T, F, Done, [F, Slots, P]()
		{
			for (FSlot& S : *Slots) { S.R = P->ClassifyImageBlocking(F->Cases[S.Case].Image, S.TopK, S.Out); }
		}, [T, F, Slots, Device]()
		{
			double Worst = 0.0;
			for (const FSlot& S : *Slots)
			{
				const FString L = FString::Printf(TEXT("blocking TopK=%d"), S.TopK);
				CheckSuccess(*T, S.R, Device, *L);
				if (S.R.bSuccess) { Worst = FMath::Max(Worst, CheckCase(*T, *F, F->Cases[S.Case], S.Out, S.TopK, *L, TinyAbs, TinyRel)); }
			}
			T->AddInfo(FString::Printf(TEXT("tiny blocking: %d calls, max score diff vs golden %.3g"), Slots->Num(), Worst));
		});
	}

	/** Tiny model, every case x TopK {1,3,0,99}, ClassifyImageAsync, all in flight at once. */
	static void TinyAsync(FAutomationTestBase* T, EPipelineDevice Device, const FDoneDelegate& Done)
	{
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeTiny(*T, Device, *F)) { Done.ExecuteIfBound(); return; }
		TArray<FIssuer> Issuers;
		const TSharedRef<TArray<TPair<int32, int32>>> Map = MakeShared<TArray<TPair<int32, int32>>>();
		for (int32 c = 0; c < F->Cases.Num(); ++c)
		{
			for (int32 K : TopKValues)
			{
				Issuers.Add(ImageIssuer(F->Pipeline.Get(), F->Cases[c].Image, K));
				Map->Add(TPair<int32, int32>(c, K));
			}
		}
		RunCalls(T, F, Issuers, Done, [T, F, Map, Device](TArray<TSharedRef<FCall>>& Calls)
		{
			double Worst = 0.0;
			for (int32 i = 0; i < Calls.Num(); ++i)
			{
				const FString L = FString::Printf(TEXT("async TopK=%d"), (*Map)[i].Value);
				CheckSuccess(*T, Calls[i]->Result, Device, *L);
				if (Calls[i]->Result.bSuccess) { Worst = FMath::Max(Worst, CheckCase(*T, *F, F->Cases[(*Map)[i].Key], Calls[i]->Out, (*Map)[i].Value, *L, TinyAbs, TinyRel)); }
			}
			T->AddInfo(FString::Printf(TEXT("tiny async: %d calls, max score diff vs golden %.3g"), Calls.Num(), Worst));
		});
	}

	/** Reference model parity: top-1 index equal for every golden image (C31: 20/20). */
	static void ReferenceParity(FAutomationTestBase* T, EPipelineDevice Device, const FDoneDelegate& Done)
	{
		const TSharedPtr<FJsonObject> Root = GetGoldens(*T);
		if (!Root.IsValid()) { Done.ExecuteIfBound(); return; }
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (!Root->TryGetArrayField(TEXT("reference_models"), Arr) || !FindModelIn(Root, TEXT("reference_models"), ResnetSlug).IsValid())
		{
			T->AddError(TEXT("vision_goldens.json has no 'reference_models' entry 'resnet-50'. Generate it with Tools/fetch_reference_models.py, Tools/fetch_test_images.py then Tools/make_vision_goldens.py."));
			Done.ExecuteIfBound();
			return;
		}
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, TEXT("reference_models"), ResnetSlug, Device, *F)) { Done.ExecuteIfBound(); return; }
		T->TestEqual(TEXT("resnet-50 golden case count (C31 selects 20 images)"), F->Cases.Num(), 20);
		TArray<FIssuer> Issuers;
		for (const FCase& K : F->Cases) { Issuers.Add(ImageIssuer(F->Pipeline.Get(), K.Image, 5)); }
		RunCalls(T, F, Issuers, Done, [T, F, Device](TArray<TSharedRef<FCall>>& Calls)
		{
			int32 Equal = 0;
			double MaxDiff = 0.0, MinMargin = 1e9;
			FString FirstBad;
			for (int32 i = 0; i < Calls.Num(); ++i)
			{
				const FCase& G = F->Cases[i];
				const FCall& C = *Calls[i];
				CheckSuccess(*T, C.Result, Device, *FString::Printf(TEXT("resnet %s"), *G.Id));
				if (!C.Result.bSuccess) { continue; }
				if (C.Out.Index == G.Index) { ++Equal; }
				else if (FirstBad.IsEmpty()) { FirstBad = FString::Printf(TEXT("%s: got %d '%s', golden %d '%s' (margin %.3g)"), *G.Id, C.Out.Index, *C.Out.Label, G.Index, *G.Label, G.TopMargin); }
				T->TestEqual(*FString::Printf(TEXT("resnet %s: Label == golden label for the same index"), *G.Id), C.Out.Index == G.Index ? C.Out.Label : G.Label, G.Label);
				T->TestEqual(*FString::Printf(TEXT("resnet %s: Scores count == 1000 labels"), *G.Id), C.Out.Scores.Num(), F->Package->Labels.Num());
				T->TestEqual(*FString::Printf(TEXT("resnet %s: TopK 5 entries"), *G.Id), C.Out.TopK.Num(), 5);
				if (C.Out.Scores.Num() == G.Scores.Num())
				{
					for (int32 s = 0; s < G.Scores.Num(); ++s) { MaxDiff = FMath::Max(MaxDiff, FMath::Abs((double)C.Out.Scores[s] - G.Scores[s])); }
				}
				MinMargin = FMath::Min(MinMargin, G.TopMargin);
			}
			T->AddInfo(FString::Printf(TEXT("resnet-50 parity: top-1 equal %d/%d, max |score - golden| = %.4g, min golden margin = %.4g"), Equal, Calls.Num(), MaxDiff, MinMargin));
			T->TestEqual(*FString::Printf(TEXT("top-1 index equal for 20/20 images (first mismatch: %s)"), *FirstBad), Equal, 20);
		});
	}
}

BEGIN_DEFINE_SPEC(FImageClassificationPipelineSpec, "Pipelines.Vision.Classification",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FImageClassificationPipelineSpec)

void FImageClassificationPipelineSpec::Define()
{
	using namespace ImageClassSpecPrivate;

	// ------------------------------------------------------------------------------------------------------
	Describe("Tiny", [this]()
	{
		LatentIt("BlockingCPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 4 images x TopK {1,3,0,99} against the ORT golden, |a-e| <= 1e-5 + 1e-5|e|. Catches wrong channel order (BGR),
			// wrong normalisation, a transposed CHW layout, softmax over the wrong axis. TopK 0 -> 1 entry, 99 -> 4 labels (non-ASCII "cafe", "nihon" included).
			TinyBlocking(this, EPipelineDevice::CPU, Done);
		});

		LatentIt("AsyncCPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Same cases through ClassifyImageAsync, 16 calls in flight at once: results must not cross between calls.
			TinyAsync(this, EPipelineDevice::CPU, Done);
		});

		LatentIt("BlockingGPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
			TinyBlocking(this, EPipelineDevice::GPU, Done);
		});

		LatentIt("AsyncGPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
			TinyAsync(this, EPipelineDevice::GPU, Done);
		});

		LatentIt("TopKClamps", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// TopK -5 and 0 -> 1 entry, 4 -> 4, 99 -> 4 (number of labels). An implementation that indexes TopK entries beyond the
			// label count, or returns 0 entries for TopK 0, fails here.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			const int32 Ks[5] = { -5, 0, 1, 4, 99 };
			TArray<FIssuer> Issuers;
			for (int32 K : Ks) { Issuers.Add(ImageIssuer(F->Pipeline.Get(), F->Cases[0].Image, K)); }
			RunCalls(this, F, Issuers, Done, [this, F, Ks](TArray<TSharedRef<FCall>>& Calls)
			{
				for (int32 i = 0; i < Calls.Num(); ++i)
				{
					const FString L = FString::Printf(TEXT("TopK %d"), Ks[i]);
					TestTrue(*FString::Printf(TEXT("%s: bSuccess (%s)"), *L, *Calls[i]->Result.Message), Calls[i]->Result.bSuccess);
					CheckCase(*this, *F, F->Cases[0], Calls[i]->Out, Ks[i], *L, TinyAbs, TinyRel);
				}
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Parity", [this]()
	{
		LatentIt("CPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// resnet-50 (torch reference) on 20 real photos: the top-1 index must be the golden's for all 20 (margins >= 0.05 by selection).
			// Wrong resize/crop (stretch instead of shortest-edge crop), BGR input or missing normalisation flips several of them.
			ReferenceParity(this, EPipelineDevice::CPU, Done);
		});

		LatentIt("GPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
			ReferenceParity(this, EPipelineDevice::GPU, Done);
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("RHI", [this]()
	{
		LatentIt("Texture2DB8G8R8A8", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Texture input must give the golden result and match the FPipelineImage path; PreprocessMs includes the readback wait (> 0).
			// A BGRA/RGBA swap in the readback or a vertical flip changes the scores by far more than 1e-5.
			if (RhiSkip(*this)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			TArray<FIssuer> Issuers;
			const int32 N = F->Cases.Num();
			for (int32 c = 0; c < N; ++c) { Issuers.Add(ImageIssuer(F->Pipeline.Get(), F->Cases[c].Image, 3)); }
			for (int32 c = 0; c < N; ++c)
			{
				UTexture2D* Tex = MakeTexture(F->Cases[c].Image, PF_B8G8R8A8);
				if (!Tex) { AddError(FString::Printf(TEXT("MakeTexture(%s, B8G8R8A8) returned null although an RHI exists"), *F->Cases[c].Id)); Done.ExecuteIfBound(); return; }
				F->Keep.Add(TStrongObjectPtr<UObject>(Tex));
				Issuers.Add(TextureIssuer(F->Pipeline.Get(), Tex, 3));
			}
			RunCalls(this, F, Issuers, Done, [this, F, N](TArray<TSharedRef<FCall>>& Calls)
			{
				for (int32 c = 0; c < N; ++c)
				{
					const FCall& Img = *Calls[c];
					const FCall& Tx = *Calls[N + c];
					CheckSuccess(*this, Tx.Result, EPipelineDevice::CPU, TEXT("texture"));
					if (!Tx.Result.bSuccess || !Img.Result.bSuccess) { continue; }
					CheckCase(*this, *F, F->Cases[c], Tx.Out, 3, TEXT("texture B8G8R8A8"), TinyAbs, TinyRel);
					TestTrue(*FString::Printf(TEXT("%s: texture PreprocessMs > 0 (got %g)"), *F->Cases[c].Id, Tx.Result.PreprocessMs), Tx.Result.PreprocessMs > 0.f);
					if (Img.Out.Scores.Num() == Tx.Out.Scores.Num())
					{
						for (int32 s = 0; s < Img.Out.Scores.Num(); ++s)
						{
							TestTrue(*FString::Printf(TEXT("%s: texture score[%d] %.8g equals FPipelineImage path %.8g"), *F->Cases[c].Id, s, Tx.Out.Scores[s], Img.Out.Scores[s]),
								FMath::Abs((double)Tx.Out.Scores[s] - Img.Out.Scores[s]) <= TinyAbs + TinyRel * FMath::Abs((double)Img.Out.Scores[s]));
						}
					}
				}
			});
		});

		LatentIt("RenderTargetRGBA8", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Same as the texture case through a UTextureRenderTarget2D (RTF_RGBA8).
			if (RhiSkip(*this)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			TArray<FIssuer> Issuers;
			const int32 N = F->Cases.Num();
			for (int32 c = 0; c < N; ++c) { Issuers.Add(ImageIssuer(F->Pipeline.Get(), F->Cases[c].Image, 3)); }
			for (int32 c = 0; c < N; ++c)
			{
				UTextureRenderTarget2D* Rt = MakeRenderTarget(F->Cases[c].Image, RTF_RGBA8);
				if (!Rt) { AddError(FString::Printf(TEXT("MakeRenderTarget(%s, RTF_RGBA8) returned null although an RHI exists"), *F->Cases[c].Id)); Done.ExecuteIfBound(); return; }
				F->Keep.Add(TStrongObjectPtr<UObject>(Rt));
				Issuers.Add(TextureIssuer(F->Pipeline.Get(), Rt, 3));
			}
			RunCalls(this, F, Issuers, Done, [this, F, N](TArray<TSharedRef<FCall>>& Calls)
			{
				for (int32 c = 0; c < N; ++c)
				{
					const FCall& Img = *Calls[c];
					const FCall& Tx = *Calls[N + c];
					CheckSuccess(*this, Tx.Result, EPipelineDevice::CPU, TEXT("render target"));
					if (!Tx.Result.bSuccess || !Img.Result.bSuccess) { continue; }
					CheckCase(*this, *F, F->Cases[c], Tx.Out, 3, TEXT("render target RGBA8"), TinyAbs, TinyRel);
					TestTrue(*FString::Printf(TEXT("%s: render target PreprocessMs > 0 (got %g)"), *F->Cases[c].Id, Tx.Result.PreprocessMs), Tx.Result.PreprocessMs > 0.f);
					if (Img.Out.Scores.Num() == Tx.Out.Scores.Num())
					{
						for (int32 s = 0; s < Img.Out.Scores.Num(); ++s)
						{
							TestTrue(*FString::Printf(TEXT("%s: render target score[%d] %.8g equals FPipelineImage path %.8g"), *F->Cases[c].Id, s, Tx.Out.Scores[s], Img.Out.Scores[s]),
								FMath::Abs((double)Tx.Out.Scores[s] - Img.Out.Scores[s]) <= TinyAbs + TinyRel * FMath::Abs((double)Img.Out.Scores[s]));
						}
					}
				}
			});
		});

		LatentIt("TextureOnGpuDevice", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// GPU device + texture input: result equals the golden within the tiny tolerance and Device == GPU.
			if (RhiSkip(*this) || GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::GPU, *F)) { Done.ExecuteIfBound(); return; }
			UTexture2D* Tex = MakeTexture(F->Cases[1].Image, PF_B8G8R8A8);
			if (!Tex) { AddError(TEXT("MakeTexture returned null although an RHI exists")); Done.ExecuteIfBound(); return; }
			F->Keep.Add(TStrongObjectPtr<UObject>(Tex));
			RunCalls(this, F, { TextureIssuer(F->Pipeline.Get(), Tex, 2) }, Done, [this, F](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckSuccess(*this, Calls[0]->Result, EPipelineDevice::GPU, TEXT("texture on GPU"));
				if (Calls[0]->Result.bSuccess) { CheckCase(*this, *F, F->Cases[1], Calls[0]->Out, 2, TEXT("texture on GPU"), TinyAbs, TinyRel); }
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Creator", [this]()
	{
		It("WrongTask", [this]()
		{
			// A detection/embedding package must be refused with WrongTask: no pipeline, no crash.
			FFixture F;
			if (!MakeTiny(*this, EPipelineDevice::CPU, F, false, [](UPipelineModelPackage& P) { P.Task = EPipelineTask::TextEmbedding; })) { return; }
			FPipelineResult R;
			UImageClassificationPipeline* P = UVisionPipelineLibrary::CreateImageClassificationPipeline(F.Package.Get(), R);
			TestNull(TEXT("pipeline"), P);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("Error WrongTask, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::WrongTask);
		});

		It("NullPackage", [this]()
		{
			FPipelineResult R;
			UImageClassificationPipeline* P = UVisionPipelineLibrary::CreateImageClassificationPipeline(nullptr, R);
			TestNull(TEXT("pipeline"), P);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::InvalidArgument);
		});

		It("ValidPackageGivesPipeline", [this]()
		{
			FFixture F;
			if (!MakeTiny(*this, EPipelineDevice::CPU, F, false)) { return; }
			FPipelineResult R;
			UImageClassificationPipeline* P = UVisionPipelineLibrary::CreateImageClassificationPipeline(F.Package.Get(), R);
			TestNotNull(*FString::Printf(TEXT("pipeline (%s)"), *R.Message), P);
			TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *R.Message), R.bSuccess);
			if (P) { TestTrue(TEXT("PipelineTask is ImageClassification"), P->GetTask() == EPipelineTask::ImageClassification); }
		});

		LatentIt("FixedInputSizeContradictsSettings", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// The model input is fixed 32x32; settings say 64x64. The mismatch must surface as ShapeMismatch (from the stage
			// runner's resolve, or from the creator) with no crash and no scores, never as an out-of-bounds read.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F, false, [](UPipelineModelPackage& P) { P.Preprocess.ImageWidth = 64; P.Preprocess.ImageHeight = 64; P.Preprocess.ResizeShortestEdge = 72; })) { Done.ExecuteIfBound(); return; }
			FPipelineResult R;
			UImageClassificationPipeline* P = UVisionPipelineLibrary::CreateImageClassificationPipeline(F->Package.Get(), R);
			if (!P)
			{
				TestTrue(*FString::Printf(TEXT("creator refused with ShapeMismatch, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::ShapeMismatch);
				Done.ExecuteIfBound();
				return;
			}
			F->Pipeline.Reset(P);
			RunCalls(this, F, { ImageIssuer(P, F->Cases[0].Image, 1) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::ShapeMismatch, TEXT("64x64 settings on a 32x32 model"));
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Calls", [this]()
	{
		LatentIt("InvalidImagesGiveInvalidArgument", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Default image, zero size, and Pixels shorter than Width*Height: InvalidArgument through the callback, on the game
			// thread, never inside the call (RunCalls checks that), nothing run.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			FPipelineImage Short;
			Short.Width = 4; Short.Height = 4; Short.Pixels.Init(FColor::Red, 3);
			FPipelineImage Zero;
			Zero.Width = 0; Zero.Height = 5;
			TArray<FIssuer> Issuers = { ImageIssuer(F->Pipeline.Get(), FPipelineImage(), 1), ImageIssuer(F->Pipeline.Get(), Zero, 1), ImageIssuer(F->Pipeline.Get(), Short, 1) };
			RunCalls(this, F, Issuers, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::InvalidArgument, TEXT("default image"));
				CheckFailure(*this, *Calls[1], EPipelineError::InvalidArgument, TEXT("zero width"));
				CheckFailure(*this, *Calls[2], EPipelineError::InvalidArgument, TEXT("short pixel array"));
			});
		});

		LatentIt("InvalidImageBlocking", LatentLimit, [this](const FDoneDelegate& Done)
		{
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			struct FOut { FPipelineResult R; FPipelineClassification Out; };
			const TSharedRef<FOut> O = MakeShared<FOut>();
			UImageClassificationPipeline* P = F->Pipeline.Get();
			RunWorker(this, F, Done, [P, O]() { O->R = P->ClassifyImageBlocking(FPipelineImage(), 1, O->Out); }, [this, O]()
			{
				TestFalse(TEXT("bSuccess"), O->R.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)O->R.Error, *O->R.Message), O->R.Error == EPipelineError::InvalidArgument);
				TestEqual(TEXT("no scores"), O->Out.Scores.Num(), 0);
			});
		});

		LatentIt("NullAndCubeTextureGiveInvalidArgument", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// A null UTexture and a UTextureCube are not UTexture2D / render targets: InvalidArgument via the callback (works without an RHI).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UTextureCube* Cube = NewObject<UTextureCube>(GetTransientPackage());
			F->Keep.Add(TStrongObjectPtr<UObject>(Cube));
			RunCalls(this, F, { TextureIssuer(F->Pipeline.Get(), nullptr, 1), TextureIssuer(F->Pipeline.Get(), Cube, 1) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::InvalidArgument, TEXT("null texture"));
				CheckFailure(*this, *Calls[1], EPipelineError::InvalidArgument, TEXT("UTextureCube"));
			});
		});

		LatentIt("TextureUnderNullRhiFailsPreprocessFailed", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Under -nullrhi a 2D texture cannot be read back: PreprocessFailed via the callback, no crash, no hang. With a real RHI this case does not apply.
			if (HasRhi()) { AddInfo(TEXT("A real RHI is present: the nullrhi texture case does not apply, skipped (run with -nullrhi)")); Done.ExecuteIfBound(); return; }
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UTexture2D* Tex = UTexture2D::CreateTransient(8, 8, PF_B8G8R8A8);
			UTextureRenderTarget2D* Rt = NewObject<UTextureRenderTarget2D>(GetTransientPackage());
			Rt->RenderTargetFormat = RTF_RGBA8;
			Rt->InitAutoFormat(8, 8);
			if (!Tex) { AddError(TEXT("UTexture2D::CreateTransient returned null")); Done.ExecuteIfBound(); return; }
			F->Keep.Add(TStrongObjectPtr<UObject>(Tex));
			F->Keep.Add(TStrongObjectPtr<UObject>(Rt));
			RunCalls(this, F, { TextureIssuer(F->Pipeline.Get(), Tex, 1), TextureIssuer(F->Pipeline.Get(), Rt, 1) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::PreprocessFailed, TEXT("UTexture2D under nullrhi"));
				CheckFailure(*this, *Calls[1], EPipelineError::PreprocessFailed, TEXT("render target under nullrhi"));
			});
		});

		LatentIt("CalledOffGameThreadCompletesInvalidArgumentOnGameThread", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// C29: ClassifyImageAsync / ClassifyAsync off the game thread complete once, on the game thread, with InvalidArgument
			// (and a texture call off the game thread must not touch the texture). Even a valid image fails here.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UTexture2D* Tex = UTexture2D::CreateTransient(4, 4, PF_B8G8R8A8);
			if (Tex) { F->Keep.Add(TStrongObjectPtr<UObject>(Tex)); }
			UImageClassificationPipeline* P = F->Pipeline.Get();
			const FPipelineImage Img = F->Cases[0].Image;
			FIssuer ImgOff = [P, Img](FPipelineImageClassificationCallback CB)
			{
				const TSharedRef<FPipelineImageClassificationCallback> Box = MakeShared<FPipelineImageClassificationCallback>(MoveTemp(CB));
				Async(EAsyncExecution::ThreadPool, [P, Img, Box]() { P->ClassifyImageAsync(Img, 1, MoveTemp(*Box)); });
			};
			FIssuer TexOff = [P, Tex](FPipelineImageClassificationCallback CB)
			{
				const TSharedRef<FPipelineImageClassificationCallback> Box = MakeShared<FPipelineImageClassificationCallback>(MoveTemp(CB));
				Async(EAsyncExecution::ThreadPool, [P, Tex, Box]() { P->ClassifyAsync(Tex, 1, MoveTemp(*Box)); });
			};
			RunCalls(this, F, { ImgOff, TexOff }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::InvalidArgument, TEXT("ClassifyImageAsync off the game thread"));
				CheckFailure(*this, *Calls[1], EPipelineError::InvalidArgument, TEXT("ClassifyAsync off the game thread"));
			});
		});

		It("NullCallbackWarnsAndRunsNothing", [this]()
		{
			FFixture F;
			if (!MakeTiny(*this, EPipelineDevice::CPU, F)) { return; }
			ExpectOneWarning(*this);
			F.Pipeline->ClassifyImageAsync(F.Cases[0].Image, 1, FPipelineImageClassificationCallback());
			TestEqual(TEXT("GetNumPendingCalls"), F.Pipeline->GetNumPendingCalls(), 0);
		});

		LatentIt("DestroyedMidCall", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Pipeline marked garbage and collected right after 5 ClassifyImageAsync calls: every callback still fires once with correct results.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UImageClassificationPipeline* P = F->Pipeline.Get();
			TArray<FIssuer> Issuers;
			const TSharedRef<TArray<int32>> Which = MakeShared<TArray<int32>>();
			for (int32 i = 0; i < 5; ++i)
			{
				const int32 C = i % F->Cases.Num();
				Which->Add(C);
				Issuers.Add(ImageIssuer(P, F->Cases[C].Image, 3));
			}
			RunCalls(this, F, Issuers, Done, [this, F, Which](TArray<TSharedRef<FCall>>& Calls)
			{
				for (int32 i = 0; i < Calls.Num(); ++i)
				{
					const FString L = FString::Printf(TEXT("destroyed mid-call %d"), i);
					TestTrue(*FString::Printf(TEXT("%s: bSuccess (%s)"), *L, *Calls[i]->Result.Message), Calls[i]->Result.bSuccess);
					if (Calls[i]->Result.bSuccess) { CheckCase(*this, *F, F->Cases[(*Which)[i]], Calls[i]->Out, 3, *L, TinyAbs, TinyRel); }
				}
			}, [F, P]()
			{
				P->MarkAsGarbage();
				F->Pipeline.Reset();
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			});
		});

		LatentIt("EightThreadsBlocking", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 8 threads x 3 ClassifyImageBlocking calls on one pipeline, different images and TopK, every result matches the golden.
			// Shared mutable preprocessing state or a shared output buffer would mix results between threads.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeTiny(*this, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			struct FSlot { int32 Case = 0; int32 TopK = 1; FPipelineResult R; FPipelineClassification Out; };
			const TSharedRef<TArray<FSlot>> Slots = MakeShared<TArray<FSlot>>();
			Slots->SetNum(24);
			for (int32 t = 0; t < 8; ++t)
			{
				for (int32 c = 0; c < 3; ++c)
				{
					FSlot& S = (*Slots)[t * 3 + c];
					S.Case = (t * 3 + c * 5 + 1) % F->Cases.Num();
					S.TopK = 1 + (t + c) % 4;
				}
			}
			UImageClassificationPipeline* P = F->Pipeline.Get();
			RunWorker(this, F, Done, [F, Slots, P]()
			{
				TArray<TFuture<void>> Threads;
				for (int32 t = 0; t < 8; ++t)
				{
					Threads.Add(Async(EAsyncExecution::Thread, [F, Slots, P, t]()
					{
						for (int32 c = 0; c < 3; ++c)
						{
							FSlot& S = (*Slots)[t * 3 + c];
							S.R = P->ClassifyImageBlocking(F->Cases[S.Case].Image, S.TopK, S.Out);
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
					const FString L = FString::Printf(TEXT("slot %d"), s);
					if (!TestTrue(*FString::Printf(TEXT("%s bSuccess (%s)"), *L, *S.R.Message), S.R.bSuccess)) { continue; }
					Worst = FMath::Max(Worst, CheckCase(*this, *F, F->Cases[S.Case], S.Out, S.TopK, *L, TinyAbs, TinyRel));
				}
				AddInfo(FString::Printf(TEXT("8 threads x 3 calls: max score diff %.3g"), Worst));
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
