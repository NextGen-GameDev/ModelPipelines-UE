// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
// WP4.8b visible tests (docs/contracts.md C29, C31, C33): UObjectDetectionPipeline and UVisionPipelineLibrary's detection creator.
// Names: Pipelines.Vision.Detection.*  (not .Decode; parity: .Parity.CPU / .Parity.GPU; texture cases: .RHI.*).
// No test blocks the game thread on an async result: waits are FTSTicker polls inside LatentIt, with a timeout and a message.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include <limits>
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
#include "ObjectDetectionPipeline.h"
#include "VisionPipelineLibrary.h"
#include "PipelineVisionTestData.h"

using namespace UE::ModelPipelines::VisionTests;

namespace ObjDetSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(240);
	constexpr double CallTimeoutSeconds = 150.0;
	const TCHAR* const SigmoidSlug = TEXT("tiny_det_designed_sigmoid");
	const TCHAR* const SoftmaxSlug = TEXT("tiny_det_designed_softmax");
	const TCHAR* const RandomSlug = TEXT("tiny_object_detection");
	const TCHAR* const RtDetrSlug = TEXT("rtdetr_r18vd");

	// Tiny/designed detection tolerance (C33): same detections in the same order, score and box |a - e| <= 1e-5.
	constexpr double TinyTol = 1e-5;

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

	struct FGoldDet
	{
		FString Label;
		int32 Index = -1;
		double Score = 0.0;
		double Box[4] = { 0, 0, 0, 0 }; // x0, y0, x1, y1
	};

	struct FCase
	{
		FString Id;
		double Threshold = 0.5;
		double Nms = 0.7;
		int32 Max = 100;
		TArray<FGoldDet> Expected;
		FPipelineImage Image;
		int32 Pipe = 0; // index into FFixture::Pipelines
	};

	struct FFixture
	{
		TSharedPtr<FJsonObject> Root;
		TSharedPtr<FJsonObject> Model;
		TArray<TStrongObjectPtr<UPipelineModelPackage>> Packages;
		TArray<TStrongObjectPtr<UObjectDetectionPipeline>> Pipelines; // one per distinct (Nms, Max) among the cases
		TArray<FCase> Cases;
		TArray<TStrongObjectPtr<UObject>> Keep; // textures / render targets
		UObjectDetectionPipeline* PipeOf(int32 CaseIndex) const { return Pipelines[Cases[CaseIndex].Pipe].Get(); }
		const TArray<FString>& LabelsOf(int32 CaseIndex) const { return Packages[Cases[CaseIndex].Pipe]->Labels; }
	};

	static bool ParseDets(const TSharedPtr<FJsonObject>& C, TArray<FGoldDet>& Out)
	{
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (!C->TryGetArrayField(TEXT("detections"), Arr)) { return false; }
		for (const TSharedPtr<FJsonValue>& V : *Arr)
		{
			const TSharedPtr<FJsonObject> O = V->AsObject();
			FGoldDet& D = Out.AddDefaulted_GetRef();
			D.Label = O->GetStringField(TEXT("label"));
			D.Index = (int32)O->GetNumberField(TEXT("index"));
			D.Score = O->GetNumberField(TEXT("score"));
			const TArray<TSharedPtr<FJsonValue>>* B = nullptr;
			if (!O->TryGetArrayField(TEXT("box"), B) || B->Num() != 4) { return false; }
			for (int32 i = 0; i < 4; ++i) { D.Box[i] = (*B)[i]->AsNumber(); }
		}
		return true;
	}

	/**
	 * Loads goldens, the model's cases (PNG as FPipelineImage, per-case threshold / NMS / max) and one package + pipeline per distinct
	 * (NmsIouThreshold, MaxDetections). The package's Postprocess is set from the case before the pipeline exists (settings are copied at
	 * initialize). Mutate may change every package after that and before pipelines are created.
	 */
	static bool MakeFixture(FAutomationTestBase& T, const TCHAR* Section, const FString& Slug, EPipelineDevice Device, FFixture& F, bool bCreatePipeline = true,
		TFunction<void(UPipelineModelPackage&)> Mutate = nullptr)
	{
		F.Root = GetGoldens(T);
		if (!F.Root.IsValid()) { return false; }
		F.Model = FindModelIn(F.Root, Section, Slug);
		if (!F.Model.IsValid())
		{
			T.AddError(FString::Printf(TEXT("Golden model '%s' not found in '%s' of vision_goldens.json. Regenerate with Tools/make_vision_goldens.py (and Tools/fetch_test_images.py, Tools/fetch_reference_models.py for reference models)."), *Slug, Section));
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
		double ModelThreshold = 0.5;
		F.Model->TryGetNumberField(TEXT("score_threshold"), ModelThreshold);
		double ModelNms = 0.7, ModelMax = 100;
		const TSharedPtr<FJsonObject>* PkgJson = nullptr;
		if (F.Model->TryGetObjectField(TEXT("package"), PkgJson))
		{
			(*PkgJson)->TryGetNumberField(TEXT("nms_iou_threshold"), ModelNms);
			(*PkgJson)->TryGetNumberField(TEXT("max_detections"), ModelMax);
		}
		TMap<FString, FPipelineImage> ImageCache;
		for (const TSharedPtr<FJsonValue>& V : *Cases)
		{
			const TSharedPtr<FJsonObject> C = V->AsObject();
			FCase& K = F.Cases.AddDefaulted_GetRef();
			K.Id = C->GetStringField(TEXT("image"));
			K.Threshold = ModelThreshold; C->TryGetNumberField(TEXT("score_threshold"), K.Threshold);
			K.Nms = ModelNms; C->TryGetNumberField(TEXT("nms_iou_threshold"), K.Nms);
			double MaxD = ModelMax; C->TryGetNumberField(TEXT("max_detections"), MaxD); K.Max = (int32)MaxD;
			if (!ParseDets(C, K.Expected))
			{
				T.AddError(FString::Printf(TEXT("Golden case '%s' of '%s' has no parsable 'detections'. Regenerate with Tools/make_vision_goldens.py."), *K.Id, *Slug));
				return false;
			}
			if (const FPipelineImage* Cached = ImageCache.Find(K.Id)) { K.Image = *Cached; continue; }
			const TSharedPtr<FJsonObject> Entry = FindImageEntry(F.Root, K.Id);
			if (!Entry.IsValid())
			{
				T.AddError(FString::Printf(TEXT("Golden image '%s' not listed in images/reference_images. Regenerate with Tools/make_vision_goldens.py."), *K.Id));
				return false;
			}
			FString Err;
			if (!LoadPng(RootDir / Entry->GetStringField(TEXT("file")), K.Image, Err)) { T.AddError(Err); return false; }
			ImageCache.Add(K.Id, K.Image);
		}
		// One package per distinct (Nms, Max).
		TArray<TPair<double, int32>> Keys;
		for (FCase& K : F.Cases)
		{
			int32 Idx = Keys.IndexOfByPredicate([&K](const TPair<double, int32>& P) { return P.Key == K.Nms && P.Value == K.Max; });
			if (Idx == INDEX_NONE)
			{
				Idx = Keys.Add(TPair<double, int32>(K.Nms, K.Max));
				FString Err;
				UPipelineModelPackage* Pkg = MakePackage(F.Root, F.Model, Device, Err);
				if (!Pkg) { T.AddError(Err); return false; }
				Pkg->Postprocess.NmsIouThreshold = (float)K.Nms;
				Pkg->Postprocess.MaxDetections = K.Max;
				if (Mutate) { Mutate(*Pkg); }
				F.Packages.Add(TStrongObjectPtr<UPipelineModelPackage>(Pkg));
			}
			K.Pipe = Idx;
		}
		if (bCreatePipeline)
		{
			for (int32 i = 0; i < F.Packages.Num(); ++i)
			{
				FPipelineResult R;
				UObjectDetectionPipeline* P = UVisionPipelineLibrary::CreateObjectDetectionPipeline(F.Packages[i].Get(), R);
				if (!P)
				{
					T.AddError(FString::Printf(TEXT("CreateObjectDetectionPipeline failed: %s"), *R.Message));
					return false;
				}
				F.Pipelines.Add(TStrongObjectPtr<UObjectDetectionPipeline>(P));
			}
		}
		return true;
	}

	static bool MakeDesigned(FAutomationTestBase& T, const TCHAR* Slug, EPipelineDevice Device, FFixture& F, bool bCreate = true, TFunction<void(UPipelineModelPackage&)> Mutate = nullptr)
	{
		return MakeFixture(T, TEXT("tiny_models"), Slug, Device, F, bCreate, Mutate);
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
		TArray<FPipelineDetection> Out;
	};

	using FIssuer = TFunction<void(FPipelineDetectionCallback)>;

	static FIssuer ImageIssuer(UObjectDetectionPipeline* P, const FPipelineImage& Image, float Threshold)
	{
		return [P, Image, Threshold](FPipelineDetectionCallback CB) { P->DetectImageAsync(Image, Threshold, MoveTemp(CB)); };
	}

	static FIssuer TextureIssuer(UObjectDetectionPipeline* P, UTexture* Tex, float Threshold)
	{
		return [P, Tex, Threshold](FPipelineDetectionCallback CB) { P->DetectAsync(Tex, Threshold, MoveTemp(CB)); };
	}

	/** Issues every call on the game thread, then polls on a ticker (never blocks): exactly once, game thread, not inside the call, then Check. */
	static void RunCalls(FAutomationTestBase* T, TSharedRef<FFixture> Fix, const TArray<FIssuer>& Issuers, const FDoneDelegate& Done,
		TFunction<void(TArray<TSharedRef<FCall>>&)> Check, TFunction<void()> AfterIssue = nullptr)
	{
		const TSharedRef<TArray<TSharedRef<FCall>>> Calls = MakeShared<TArray<TSharedRef<FCall>>>();
		for (int32 i = 0; i < Issuers.Num(); ++i) { Calls->Add(MakeShared<FCall>()); }
		for (const TSharedRef<FCall>& C : *Calls) { C->bInCall = true; }
		for (int32 i = 0; i < Issuers.Num(); ++i)
		{
			const TSharedRef<FCall> C = (*Calls)[i];
			Issuers[i]([C](const FPipelineResult& R, const TArray<FPipelineDetection>& Res)
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
		FTSTicker::GetCoreTicker().AddTicker(TEXT("ObjDetSpecWait"), 0.02f, [T, Fix, Calls, Start, AllSeen, DoneCopy, Check](float)
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
						T->TestFalse(*FString::Printf(TEXT("call %d: callback must not run inside the Detect call"), i), C->bAnyInCall);
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
		FTSTicker::GetCoreTicker().AddTicker(TEXT("ObjDetSpecWorker"), 0.02f, [T, Fix, F, Start, DoneCopy, Check](float)
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
		T.AddExpectedMessage(TEXT("UObjectDetectionPipeline::DetectImageAsync: null callback, nothing runs"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
	}

	/**
	 * Compares a result with the golden detections: same count, same order, Index, Label (== package label of Index), score and all four
	 * box coordinates within Tol, boxes inside [0, 1]. Returns the worst |diff| (or -1 when the count is wrong).
	 */
	static double CheckDetections(FAutomationTestBase& T, const TArray<FString>& Labels, const TArray<FGoldDet>& Gold, const TArray<FPipelineDetection>& Got, const TCHAR* Tag, double Tol)
	{
		if (Got.Num() != Gold.Num())
		{
			FString Summary;
			for (const FPipelineDetection& D : Got) { Summary += FString::Printf(TEXT("[%d %.3f] "), D.Index, D.Score); }
			T.AddError(FString::Printf(TEXT("%s: %d detections, golden has %d (got %s)"), Tag, Got.Num(), Gold.Num(), *Summary));
			return -1.0;
		}
		double Worst = 0.0;
		for (int32 i = 0; i < Gold.Num(); ++i)
		{
			const FGoldDet& G = Gold[i];
			const FPipelineDetection& D = Got[i];
			T.TestEqual(*FString::Printf(TEXT("%s det %d: Index == golden (order matters: NMS order)"), Tag, i), D.Index, G.Index);
			T.TestEqual(*FString::Printf(TEXT("%s det %d: Label == golden"), Tag, i), D.Label, G.Label);
			if (Labels.IsValidIndex(D.Index)) { T.TestEqual(*FString::Printf(TEXT("%s det %d: Label is the package label of its Index"), Tag, i), D.Label, Labels[D.Index]); }
			const double Diffs[5] = { FMath::Abs((double)D.Score - G.Score), FMath::Abs(D.Box.Min.X - G.Box[0]), FMath::Abs(D.Box.Min.Y - G.Box[1]),
				FMath::Abs(D.Box.Max.X - G.Box[2]), FMath::Abs(D.Box.Max.Y - G.Box[3]) };
			double W = 0.0;
			for (double X : Diffs) { W = FMath::Max(W, X); }
			Worst = FMath::Max(Worst, W);
			T.TestTrue(*FString::Printf(TEXT("%s det %d: score/box within %.0e of golden (score %.7g vs %.7g, box [%.6g %.6g %.6g %.6g] vs [%.6g %.6g %.6g %.6g], worst %.3g)"),
				Tag, i, Tol, D.Score, G.Score, D.Box.Min.X, D.Box.Min.Y, D.Box.Max.X, D.Box.Max.Y, G.Box[0], G.Box[1], G.Box[2], G.Box[3], W), W <= Tol);
			T.TestTrue(*FString::Printf(TEXT("%s det %d: box clamped to [0,1] and valid"), Tag, i),
				D.Box.bIsValid && D.Box.Min.X >= 0.0 && D.Box.Min.Y >= 0.0 && D.Box.Max.X <= 1.0 && D.Box.Max.Y <= 1.0 && D.Box.Min.X <= D.Box.Max.X && D.Box.Min.Y <= D.Box.Max.Y);
		}
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
		T.TestEqual(*FString::Printf(TEXT("%s: no detections on failure"), Label), C.Out.Num(), 0);
	}

	/** Every case of a designed/tiny model through DetectImageBlocking from a worker thread. */
	static void ModelBlocking(FAutomationTestBase* T, const TCHAR* Slug, EPipelineDevice Device, const FDoneDelegate& Done)
	{
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeDesigned(*T, Slug, Device, *F)) { Done.ExecuteIfBound(); return; }
		const FString SlugStr = Slug;
		struct FSlot { FPipelineResult R; TArray<FPipelineDetection> Out; };
		const TSharedRef<TArray<FSlot>> Slots = MakeShared<TArray<FSlot>>();
		Slots->SetNum(F->Cases.Num());
		RunWorker(T, F, Done, [F, Slots]()
		{
			for (int32 c = 0; c < F->Cases.Num(); ++c)
			{
				(*Slots)[c].R = F->PipeOf(c)->DetectImageBlocking(F->Cases[c].Image, (float)F->Cases[c].Threshold, (*Slots)[c].Out);
			}
		}, [T, F, Slots, Device, SlugStr]()
		{
			double Worst = 0.0;
			for (int32 c = 0; c < F->Cases.Num(); ++c)
			{
				const FCase& K = F->Cases[c];
				const FString L = FString::Printf(TEXT("%s blocking %s thr=%g nms=%g max=%d"), *SlugStr, *K.Id, K.Threshold, K.Nms, K.Max);
				CheckSuccess(*T, (*Slots)[c].R, Device, *L);
				if ((*Slots)[c].R.bSuccess) { Worst = FMath::Max(Worst, CheckDetections(*T, F->LabelsOf(c), K.Expected, (*Slots)[c].Out, *L, TinyTol)); }
			}
			T->AddInfo(FString::Printf(TEXT("%s blocking: %d cases, worst |diff| vs golden %.3g"), *SlugStr, F->Cases.Num(), Worst));
		});
	}

	/** Every case through DetectImageAsync, all in flight at once. */
	static void ModelAsync(FAutomationTestBase* T, const TCHAR* Slug, EPipelineDevice Device, const FDoneDelegate& Done)
	{
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeDesigned(*T, Slug, Device, *F)) { Done.ExecuteIfBound(); return; }
		TArray<FIssuer> Issuers;
		for (int32 c = 0; c < F->Cases.Num(); ++c) { Issuers.Add(ImageIssuer(F->PipeOf(c), F->Cases[c].Image, (float)F->Cases[c].Threshold)); }
		const FString SlugStr = Slug;
		RunCalls(T, F, Issuers, Done, [T, F, Device, SlugStr](TArray<TSharedRef<FCall>>& Calls)
		{
			double Worst = 0.0;
			for (int32 c = 0; c < Calls.Num(); ++c)
			{
				const FCase& K = F->Cases[c];
				const FString L = FString::Printf(TEXT("%s async %s thr=%g nms=%g max=%d"), *SlugStr, *K.Id, K.Threshold, K.Nms, K.Max);
				CheckSuccess(*T, Calls[c]->Result, Device, *L);
				if (Calls[c]->Result.bSuccess) { Worst = FMath::Max(Worst, CheckDetections(*T, F->LabelsOf(c), K.Expected, Calls[c]->Out, *L, TinyTol)); }
			}
			T->AddInfo(FString::Printf(TEXT("%s async: %d calls, worst |diff| vs golden %.3g"), *SlugStr, Calls.Num(), Worst));
		});
	}

	// ------------------------------------------------------------------------------------------------------
	// Reference parity, C31 matching rule
	// ------------------------------------------------------------------------------------------------------
	static double BoxIoU(const double A[4], const FBox2D& B)
	{
		const double IW = FMath::Min(A[2], B.Max.X) - FMath::Max(A[0], B.Min.X);
		const double IH = FMath::Min(A[3], B.Max.Y) - FMath::Max(A[1], B.Min.Y);
		if (IW <= 0.0 || IH <= 0.0) { return 0.0; }
		const double Inter = IW * IH;
		const double AreaA = (A[2] - A[0]) * (A[3] - A[1]);
		const double AreaB = (B.Max.X - B.Min.X) * (B.Max.Y - B.Min.Y);
		const double U = AreaA + AreaB - Inter;
		return U > 0.0 ? Inter / U : 0.0;
	}

	struct FMatchStats
	{
		int32 GoldTotal = 0, GoldMatched = 0;
		double MinIou = 1.0, MaxScoreDiff = 0.0;
		int32 GoldHighUnmatched = 0, UeHighUnmatched = 0, LowIouPairs = 0;
	};

	/** C31: greedy per golden detection in order: among unmatched UE detections of the same index take the highest IoU; matched if IoU > 0.9. */
	static void MatchDetections(const TArray<FGoldDet>& Gold, const TArray<FPipelineDetection>& Got, FMatchStats& S, FString& FirstProblem, const FString& Id)
	{
		constexpr double ScoreGate = 0.52;
		TArray<bool> Used;
		Used.Init(false, Got.Num());
		for (int32 g = 0; g < Gold.Num(); ++g)
		{
			++S.GoldTotal;
			int32 Best = INDEX_NONE;
			double BestIou = -1.0;
			for (int32 u = 0; u < Got.Num(); ++u)
			{
				if (Used[u] || Got[u].Index != Gold[g].Index) { continue; }
				const double Iou = BoxIoU(Gold[g].Box, Got[u].Box);
				if (Iou > BestIou) { BestIou = Iou; Best = u; }
			}
			if (Best != INDEX_NONE && BestIou > 0.9)
			{
				Used[Best] = true;
				++S.GoldMatched;
				S.MinIou = FMath::Min(S.MinIou, BestIou);
				S.MaxScoreDiff = FMath::Max(S.MaxScoreDiff, FMath::Abs((double)Got[Best].Score - Gold[g].Score));
			}
			else
			{
				if (Best != INDEX_NONE) { S.MinIou = FMath::Min(S.MinIou, BestIou); }
				if (Gold[g].Score >= ScoreGate)
				{
					++S.GoldHighUnmatched;
					if (Best != INDEX_NONE) { ++S.LowIouPairs; }
					if (FirstProblem.IsEmpty())
					{
						FirstProblem = FString::Printf(TEXT("%s: golden det %d (class %d '%s', score %.3f) unmatched, best same-class IoU %.3f"), *Id, g, Gold[g].Index, *Gold[g].Label, Gold[g].Score, BestIou);
					}
				}
			}
		}
		for (int32 u = 0; u < Got.Num(); ++u)
		{
			if (!Used[u] && Got[u].Score >= ScoreGate)
			{
				++S.UeHighUnmatched;
				if (FirstProblem.IsEmpty())
				{
					FirstProblem = FString::Printf(TEXT("%s: UE det %d (class %d '%s', score %.3f) has no golden match"), *Id, u, Got[u].Index, *Got[u].Label, Got[u].Score);
				}
			}
		}
	}

	static void ReferenceParity(FAutomationTestBase* T, EPipelineDevice Device, const FDoneDelegate& Done)
	{
		const TSharedPtr<FJsonObject> Root = GetGoldens(*T);
		if (!Root.IsValid()) { Done.ExecuteIfBound(); return; }
		if (!FindModelIn(Root, TEXT("reference_models"), RtDetrSlug).IsValid())
		{
			T->AddError(TEXT("vision_goldens.json has no 'reference_models' entry 'rtdetr_r18vd'. Generate it with Tools/fetch_reference_models.py, Tools/fetch_test_images.py then Tools/make_vision_goldens.py."));
			Done.ExecuteIfBound();
			return;
		}
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, TEXT("reference_models"), RtDetrSlug, Device, *F)) { Done.ExecuteIfBound(); return; }
		T->TestEqual(TEXT("rtdetr_r18vd golden case count (C31 selects 10 images)"), F->Cases.Num(), 10);
		TArray<FIssuer> Issuers;
		for (int32 c = 0; c < F->Cases.Num(); ++c) { Issuers.Add(ImageIssuer(F->PipeOf(c), F->Cases[c].Image, (float)F->Cases[c].Threshold)); }
		RunCalls(T, F, Issuers, Done, [T, F, Device](TArray<TSharedRef<FCall>>& Calls)
		{
			FMatchStats S;
			FString FirstProblem;
			int32 FailedImages = 0;
			for (int32 c = 0; c < Calls.Num(); ++c)
			{
				const FCase& K = F->Cases[c];
				CheckSuccess(*T, Calls[c]->Result, Device, *FString::Printf(TEXT("rtdetr %s"), *K.Id));
				if (!Calls[c]->Result.bSuccess) { ++FailedImages; continue; }
				const int32 BeforeGoldHigh = S.GoldHighUnmatched, BeforeUeHigh = S.UeHighUnmatched;
				MatchDetections(K.Expected, Calls[c]->Out, S, FirstProblem, K.Id);
				T->TestEqual(*FString::Printf(TEXT("rtdetr %s: every golden detection with score >= 0.52 is matched (IoU > 0.9, same class)"), *K.Id), S.GoldHighUnmatched - BeforeGoldHigh, 0);
				T->TestEqual(*FString::Printf(TEXT("rtdetr %s: every UE detection with score >= 0.52 is matched"), *K.Id), S.UeHighUnmatched - BeforeUeHigh, 0);
				for (const FPipelineDetection& D : Calls[c]->Out)
				{
					T->TestTrue(*FString::Printf(TEXT("rtdetr %s: box inside [0,1], label == package label"), *K.Id),
						D.Box.bIsValid && D.Box.Min.X >= 0.0 && D.Box.Min.Y >= 0.0 && D.Box.Max.X <= 1.0 && D.Box.Max.Y <= 1.0 && F->LabelsOf(c).IsValidIndex(D.Index) && F->LabelsOf(c)[D.Index] == D.Label);
				}
			}
			T->AddInfo(FString::Printf(TEXT("rtdetr_r18vd parity: matched %d/%d golden detections, min IoU %.5f, max |score diff| %.4g, golden>=0.52 unmatched %d, UE>=0.52 unmatched %d"),
				S.GoldMatched, S.GoldTotal, S.MinIou, S.MaxScoreDiff, S.GoldHighUnmatched, S.UeHighUnmatched));
			T->TestEqual(*FString::Printf(TEXT("no golden detection with score >= 0.52 left unmatched (first problem: %s)"), *FirstProblem), S.GoldHighUnmatched, 0);
			T->TestEqual(*FString::Printf(TEXT("no UE detection with score >= 0.52 left unmatched (first problem: %s)"), *FirstProblem), S.UeHighUnmatched, 0);
			T->TestEqual(TEXT("no matched pair with IoU <= 0.9"), S.LowIouPairs, 0);
			T->TestEqual(TEXT("no failed images"), FailedImages, 0);
		});
	}

	static bool SameDetections(const TArray<FPipelineDetection>& A, const TArray<FPipelineDetection>& B, double Tol, double& OutWorst)
	{
		OutWorst = 0.0;
		if (A.Num() != B.Num()) { OutWorst = 1e9; return false; }
		for (int32 i = 0; i < A.Num(); ++i)
		{
			if (A[i].Index != B[i].Index || A[i].Label != B[i].Label) { OutWorst = 1e9; return false; }
			OutWorst = FMath::Max(OutWorst, (double)FMath::Abs(A[i].Score - B[i].Score));
			OutWorst = FMath::Max(OutWorst, FMath::Abs(A[i].Box.Min.X - B[i].Box.Min.X));
			OutWorst = FMath::Max(OutWorst, FMath::Abs(A[i].Box.Min.Y - B[i].Box.Min.Y));
			OutWorst = FMath::Max(OutWorst, FMath::Abs(A[i].Box.Max.X - B[i].Box.Max.X));
			OutWorst = FMath::Max(OutWorst, FMath::Abs(A[i].Box.Max.Y - B[i].Box.Max.Y));
		}
		return OutWorst <= Tol;
	}

	/** Texture or render target input vs the FPipelineImage path (and the golden when Tol applies): shared by the RHI cases. */
	static void TextureVsImage(FAutomationTestBase* T, const TCHAR* Section, const TCHAR* Slug, const TArray<int32>& CaseSel, bool bRenderTarget, bool bCheckGolden,
		EPipelineDevice Device, const FDoneDelegate& Done)
	{
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, Section, Slug, Device, *F)) { Done.ExecuteIfBound(); return; }
		TArray<FIssuer> Issuers;
		const int32 N = CaseSel.Num();
		for (int32 k = 0; k < N; ++k)
		{
			const int32 c = CaseSel[k];
			if (!F->Cases.IsValidIndex(c)) { T->AddError(FString::Printf(TEXT("case index %d out of range for '%s'"), c, Slug)); Done.ExecuteIfBound(); return; }
			Issuers.Add(ImageIssuer(F->PipeOf(c), F->Cases[c].Image, (float)F->Cases[c].Threshold));
		}
		for (int32 k = 0; k < N; ++k)
		{
			const int32 c = CaseSel[k];
			UTexture* Tex = nullptr;
			if (bRenderTarget) { Tex = MakeRenderTarget(F->Cases[c].Image, RTF_RGBA8); }
			else { Tex = MakeTexture(F->Cases[c].Image, PF_B8G8R8A8); }
			if (!Tex) { T->AddError(FString::Printf(TEXT("MakeTexture/MakeRenderTarget(%s) returned null although an RHI exists"), *F->Cases[c].Id)); Done.ExecuteIfBound(); return; }
			F->Keep.Add(TStrongObjectPtr<UObject>(Tex));
			Issuers.Add(TextureIssuer(F->PipeOf(c), Tex, (float)F->Cases[c].Threshold));
		}
		RunCalls(T, F, Issuers, Done, [T, F, N, CaseSel, bRenderTarget, bCheckGolden, Device](TArray<TSharedRef<FCall>>& Calls)
		{
			const TCHAR* Kind = bRenderTarget ? TEXT("render target") : TEXT("texture");
			for (int32 k = 0; k < N; ++k)
			{
				const int32 c = CaseSel[k];
				const FCall& Img = *Calls[k];
				const FCall& Tx = *Calls[N + k];
				const FString L = FString::Printf(TEXT("%s %s"), Kind, *F->Cases[c].Id);
				CheckSuccess(*T, Tx.Result, Device, *L);
				CheckSuccess(*T, Img.Result, Device, *FString::Printf(TEXT("%s (image path)"), *L));
				if (!Tx.Result.bSuccess || !Img.Result.bSuccess) { continue; }
				// Readback is bit-exact (C33), so the same pixels give the same detections: a BGRA/RGBA swap or a vertical flip in the readback
				// moves boxes and scores by far more than the tolerance.
				double Worst = 0.0;
				const bool bSame = SameDetections(Tx.Out, Img.Out, TinyTol, Worst);
				T->TestTrue(*FString::Printf(TEXT("%s: same detections as the FPipelineImage path (%d vs %d, worst %.3g)"), *L, Tx.Out.Num(), Img.Out.Num(), Worst), bSame);
				T->AddInfo(FString::Printf(TEXT("%s: worst |diff| vs image path %.3g, %d detections"), *L, Worst, Tx.Out.Num()));
				if (bCheckGolden) { CheckDetections(*T, F->LabelsOf(c), F->Cases[c].Expected, Tx.Out, *L, TinyTol); }
				T->TestTrue(*FString::Printf(TEXT("%s: PreprocessMs includes the readback wait (> 0, got %g)"), *L, Tx.Result.PreprocessMs), Tx.Result.PreprocessMs > 0.f);
			}
		});
	}
}

BEGIN_DEFINE_SPEC(FObjectDetectionPipelineSpec, "Pipelines.Vision.Detection",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FObjectDetectionPipelineSpec)

void FObjectDetectionPipelineSpec::Define()
{
	using namespace ObjDetSpecPrivate;

	// ------------------------------------------------------------------------------------------------------
	// tiny_models detection entries: random-weight model (Sigmoid, 2 labels) and the two designed models (Sigmoid / Softmax, 3 labels).
	// ------------------------------------------------------------------------------------------------------
	const TCHAR* const Slugs[3] = { SigmoidSlug, SoftmaxSlug, RandomSlug };
	for (const TCHAR* SlugC : Slugs)
	{
		const FString Slug = SlugC;
		Describe(Slug, [this, Slug]()
		{
			LatentIt("BlockingCPU", LatentLimit, [this, Slug](const FDoneDelegate& Done)
			{
				// Every golden case (own threshold, own NmsIouThreshold/MaxDetections on a per-parameter-set pipeline) through DetectImageBlocking:
				// same detections in the same order, score and box within 1e-5. Catches a wrong box layout (xywh vs xyxy), a missing sigmoid/softmax
				// no-object drop, wrong sort order (ties lower index), NMS applied per call instead of with the package parameters, or unclamped boxes.
				ModelBlocking(this, *Slug, EPipelineDevice::CPU, Done);
			});

			LatentIt("AsyncCPU", LatentLimit, [this, Slug](const FDoneDelegate& Done)
			{
				// Same cases through DetectImageAsync, all in flight at once: results must not cross between calls.
				ModelAsync(this, *Slug, EPipelineDevice::CPU, Done);
			});

			LatentIt("BlockingGPU", LatentLimit, [this, Slug](const FDoneDelegate& Done)
			{
				if (GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
				ModelBlocking(this, *Slug, EPipelineDevice::GPU, Done);
			});

			LatentIt("AsyncGPU", LatentLimit, [this, Slug](const FDoneDelegate& Done)
			{
				if (GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
				ModelAsync(this, *Slug, EPipelineDevice::GPU, Done);
			});
		});
	}

	// ------------------------------------------------------------------------------------------------------
	Describe("Parity", [this]()
	{
		LatentIt("CPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// RT-DETR R18 (torch reference) on 10 real photos, threshold 0.5, C31 greedy one-to-one matching: every detection with score >= 0.52
			// on either side matched with IoU > 0.9. A box in the wrong space (network input instead of source image), an off-by-one resize
			// mapping, cx,cy,w,h read as x0,y0,x1,y1 or softmax instead of sigmoid breaks several images.
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
		LatentIt("DesignedTexture2D", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Designed sigmoid model, every case through a B8G8R8A8 UTexture2D: golden detections and equal to the FPipelineImage path.
			if (RhiSkip(*this)) { Done.ExecuteIfBound(); return; }
			TextureVsImage(this, TEXT("tiny_models"), SigmoidSlug, { 0, 1, 2, 3, 4, 5, 6, 7 }, false, true, EPipelineDevice::CPU, Done);
		});

		LatentIt("DesignedRenderTarget", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Same through a UTextureRenderTarget2D (RTF_RGBA8).
			if (RhiSkip(*this)) { Done.ExecuteIfBound(); return; }
			TextureVsImage(this, TEXT("tiny_models"), SigmoidSlug, { 0, 1, 2, 3, 4, 5, 6, 7 }, true, true, EPipelineDevice::CPU, Done);
		});

		LatentIt("ReferenceTexture2D", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Two real reference images (d00, d06 are cases 0 and 3) through a texture: same detections as the FPipelineImage path.
			if (RhiSkip(*this)) { Done.ExecuteIfBound(); return; }
			TextureVsImage(this, TEXT("reference_models"), RtDetrSlug, { 0, 3 }, false, false, EPipelineDevice::CPU, Done);
		});

		LatentIt("ReferenceRenderTarget", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (RhiSkip(*this)) { Done.ExecuteIfBound(); return; }
			TextureVsImage(this, TEXT("reference_models"), RtDetrSlug, { 0, 3 }, true, false, EPipelineDevice::CPU, Done);
		});

		LatentIt("DesignedTextureOnGpuDevice", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (RhiSkip(*this) || GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
			TextureVsImage(this, TEXT("tiny_models"), SoftmaxSlug, { 0, 2 }, false, true, EPipelineDevice::GPU, Done);
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Creator", [this]()
	{
		It("WrongTask", [this]()
		{
			// A classification package must be refused with WrongTask: no pipeline, no crash.
			FFixture F;
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, F, false, [](UPipelineModelPackage& P) { P.Task = EPipelineTask::ImageClassification; })) { return; }
			FPipelineResult R;
			UObjectDetectionPipeline* P = UVisionPipelineLibrary::CreateObjectDetectionPipeline(F.Packages[0].Get(), R);
			TestNull(TEXT("pipeline"), P);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("Error WrongTask, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::WrongTask);
		});

		It("NullPackage", [this]()
		{
			FPipelineResult R;
			UObjectDetectionPipeline* P = UVisionPipelineLibrary::CreateObjectDetectionPipeline(nullptr, R);
			TestNull(TEXT("pipeline"), P);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::InvalidArgument);
		});

		It("ValidPackageGivesPipeline", [this]()
		{
			FFixture F;
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, F, false)) { return; }
			FPipelineResult R;
			UObjectDetectionPipeline* P = UVisionPipelineLibrary::CreateObjectDetectionPipeline(F.Packages[0].Get(), R);
			TestNotNull(*FString::Printf(TEXT("pipeline (%s)"), *R.Message), P);
			TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *R.Message), R.bSuccess);
			if (P) { TestTrue(TEXT("PipelineTask is ObjectDetection"), P->GetTask() == EPipelineTask::ObjectDetection); }
		});

		LatentIt("SigmoidLabelCountMismatch", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Designed sigmoid model has C' = 3 logit columns; a package with 2 labels cannot fit. ShapeMismatch from the creator or from the
			// stage resolve, no crash, no detections (never an out-of-bounds Labels[class]).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F, false, [](UPipelineModelPackage& P) { P.Labels = { TEXT("a"), TEXT("b") }; })) { Done.ExecuteIfBound(); return; }
			FPipelineResult R;
			UObjectDetectionPipeline* P = UVisionPipelineLibrary::CreateObjectDetectionPipeline(F->Packages[0].Get(), R);
			if (!P)
			{
				TestTrue(*FString::Printf(TEXT("creator refused with ShapeMismatch, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::ShapeMismatch);
				Done.ExecuteIfBound();
				return;
			}
			F->Pipelines.Add(TStrongObjectPtr<UObjectDetectionPipeline>(P));
			F->Cases[0].Pipe = 0;
			RunCalls(this, F, { ImageIssuer(P, F->Cases[0].Image, 0.5f) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::ShapeMismatch, TEXT("2 labels on a 3-column sigmoid model"));
			});
		});

		LatentIt("SoftmaxLabelCountMismatch", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Designed softmax model has C' = 4 columns (3 classes + no-object): it needs 3 labels. With 4 labels (the sigmoid count) it must
			// fail with ShapeMismatch, not read the no-object column as a class.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SoftmaxSlug, EPipelineDevice::CPU, *F, false, [](UPipelineModelPackage& P) { P.Labels = { TEXT("a"), TEXT("b"), TEXT("c"), TEXT("d") }; })) { Done.ExecuteIfBound(); return; }
			FPipelineResult R;
			UObjectDetectionPipeline* P = UVisionPipelineLibrary::CreateObjectDetectionPipeline(F->Packages[0].Get(), R);
			if (!P)
			{
				TestTrue(*FString::Printf(TEXT("creator refused with ShapeMismatch, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::ShapeMismatch);
				Done.ExecuteIfBound();
				return;
			}
			F->Pipelines.Add(TStrongObjectPtr<UObjectDetectionPipeline>(P));
			F->Cases[0].Pipe = 0;
			RunCalls(this, F, { ImageIssuer(P, F->Cases[0].Image, 0.5f) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::ShapeMismatch, TEXT("4 labels on a 4-column softmax model"));
			});
		});

		LatentIt("FixedInputSizeContradictsSettings", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Model input is fixed 32x32; settings say 64x64: ShapeMismatch (creator or call), never an out-of-bounds read.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F, false, [](UPipelineModelPackage& P) { P.Preprocess.ImageWidth = 64; P.Preprocess.ImageHeight = 64; })) { Done.ExecuteIfBound(); return; }
			FPipelineResult R;
			UObjectDetectionPipeline* P = UVisionPipelineLibrary::CreateObjectDetectionPipeline(F->Packages[0].Get(), R);
			if (!P)
			{
				TestTrue(*FString::Printf(TEXT("creator refused with ShapeMismatch, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::ShapeMismatch);
				Done.ExecuteIfBound();
				return;
			}
			F->Pipelines.Add(TStrongObjectPtr<UObjectDetectionPipeline>(P));
			F->Cases[0].Pipe = 0;
			RunCalls(this, F, { ImageIssuer(P, F->Cases[0].Image, 0.5f) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::ShapeMismatch, TEXT("64x64 settings on a 32x32 model"));
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Postprocess", [this]()
	{
		LatentIt("NmsIouThresholdOneKeepsOverlappingBoxes", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Designed sigmoid model: NmsIouThreshold 0.7 gives 6 detections, 1.0 gives 7 (the 0.85 'a' box overlapping the 0.92 'a' box
			// survives). Run both from the same image. NMS that ignores the package threshold or treats >= 1 as "suppress equal IoU" fails.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			int32 Keep = INDEX_NONE, Suppress = INDEX_NONE;
			for (int32 c = 0; c < F->Cases.Num(); ++c)
			{
				if (F->Cases[c].Id != F->Cases[0].Id || F->Cases[c].Threshold != 0.5 || F->Cases[c].Max != 100) { continue; }
				if (F->Cases[c].Nms >= 1.0) { Keep = c; } else if (F->Cases[c].Nms == 0.7) { Suppress = c; }
			}
			if (Keep == INDEX_NONE || Suppress == INDEX_NONE) { AddError(TEXT("golden has no NMS 0.7 / NMS 1.0 pair at threshold 0.5 for the first image; regenerate with Tools/make_vision_goldens.py")); Done.ExecuteIfBound(); return; }
			TestTrue(TEXT("golden: NMS 1.0 keeps more boxes than NMS 0.7"), F->Cases[Keep].Expected.Num() > F->Cases[Suppress].Expected.Num());
			RunCalls(this, F, { ImageIssuer(F->PipeOf(Suppress), F->Cases[Suppress].Image, 0.5f), ImageIssuer(F->PipeOf(Keep), F->Cases[Keep].Image, 0.5f) }, Done,
				[this, F, Keep, Suppress](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckSuccess(*this, Calls[0]->Result, EPipelineDevice::CPU, TEXT("nms 0.7"));
				CheckSuccess(*this, Calls[1]->Result, EPipelineDevice::CPU, TEXT("nms 1.0"));
				TestEqual(TEXT("NMS 0.7 detection count"), Calls[0]->Out.Num(), F->Cases[Suppress].Expected.Num());
				TestEqual(TEXT("NMS 1.0 detection count (overlapping boxes kept)"), Calls[1]->Out.Num(), F->Cases[Keep].Expected.Num());
				CheckDetections(*this, F->LabelsOf(Keep), F->Cases[Keep].Expected, Calls[1]->Out, TEXT("nms 1.0"), TinyTol);
			});
		});

		LatentIt("MaxDetectionsOneKeepsTheBest", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Package MaxDetections = 1 (set before the pipeline exists): exactly the first golden detection (best score) at threshold 0.5.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SoftmaxSlug, EPipelineDevice::CPU, *F, true, [](UPipelineModelPackage& P) { P.Postprocess.MaxDetections = 1; })) { Done.ExecuteIfBound(); return; }
			RunCalls(this, F, { ImageIssuer(F->PipeOf(0), F->Cases[0].Image, 0.5f) }, Done, [this, F](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckSuccess(*this, Calls[0]->Result, EPipelineDevice::CPU, TEXT("max 1"));
				TArray<FGoldDet> One;
				if (F->Cases[0].Expected.Num() > 0) { One.Add(F->Cases[0].Expected[0]); }
				CheckDetections(*this, F->LabelsOf(0), One, Calls[0]->Out, TEXT("MaxDetections 1"), TinyTol);
			});
		});

		LatentIt("ThresholdIsPerCall", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// One pipeline, thresholds 0.95 / 0.65 / 0.05 on the same image: counts must be non-increasing with the threshold and the 0.95 call
			// must be empty (best designed score is 0.92). A threshold baked in at creation fails.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UObjectDetectionPipeline* P = F->PipeOf(0);
			const FPipelineImage Img = F->Cases[0].Image;
			RunCalls(this, F, { ImageIssuer(P, Img, 0.95f), ImageIssuer(P, Img, 0.65f), ImageIssuer(P, Img, 0.05f) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				for (int32 i = 0; i < 3; ++i) { TestTrue(*FString::Printf(TEXT("call %d bSuccess (%s)"), i, *Calls[i]->Result.Message), Calls[i]->Result.bSuccess); }
				TestEqual(TEXT("threshold 0.95: no detection scores above 0.92"), Calls[0]->Out.Num(), 0);
				TestTrue(*FString::Printf(TEXT("counts %d (0.65) <= %d (0.05)"), Calls[1]->Out.Num(), Calls[2]->Out.Num()), Calls[1]->Out.Num() <= Calls[2]->Out.Num());
				TestTrue(TEXT("threshold 0.65 gives at least the 0.92 and 0.88 detections"), Calls[1]->Out.Num() >= 2);
				for (const FPipelineDetection& D : Calls[1]->Out) { TestTrue(*FString::Printf(TEXT("score %.4f > 0.65"), D.Score), D.Score > 0.65f); }
			});
		});

		LatentIt("ThresholdAboveOneClampsToEmptyOk", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Threshold 2 is clamped to 1: no score exceeds 1, so the result is Ok with zero detections (not an error). Also 1.0 itself.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UObjectDetectionPipeline* P = F->PipeOf(0);
			const FPipelineImage Img = F->Cases[0].Image;
			RunCalls(this, F, { ImageIssuer(P, Img, 2.f), ImageIssuer(P, Img, 1.f) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				for (int32 i = 0; i < 2; ++i)
				{
					TestTrue(*FString::Printf(TEXT("call %d bSuccess (%s)"), i, *Calls[i]->Result.Message), Calls[i]->Result.bSuccess);
					TestEqual(*FString::Printf(TEXT("call %d: empty detections"), i), Calls[i]->Out.Num(), 0);
				}
			});
		});

		LatentIt("NegativeThresholdClampsToZero", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Threshold -5 is clamped to 0: identical result to threshold 0 (and not an error, not empty: sigmoid scores are all > 0).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UObjectDetectionPipeline* P = F->PipeOf(0);
			const FPipelineImage Img = F->Cases[0].Image;
			RunCalls(this, F, { ImageIssuer(P, Img, -5.f), ImageIssuer(P, Img, 0.f) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *Calls[0]->Result.Message), Calls[0]->Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *Calls[1]->Result.Message), Calls[1]->Result.bSuccess);
				double Worst = 0.0;
				const bool bSame = SameDetections(Calls[0]->Out, Calls[1]->Out, 0.0, Worst);
				TestTrue(*FString::Printf(TEXT("-5 behaves like 0 (%d vs %d detections)"), Calls[0]->Out.Num(), Calls[1]->Out.Num()), bSame);
				TestTrue(TEXT("threshold 0 returns detections"), Calls[1]->Out.Num() > 0);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Calls", [this]()
	{
		LatentIt("NaNThresholdGivesInvalidArgument", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// NaN threshold: InvalidArgument through the callback (game thread, once, not inside the call), nothing returned. A clamp that lets
			// NaN through would compare false against every score and return Ok with no detections.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			const float NaN = std::numeric_limits<float>::quiet_NaN();
			RunCalls(this, F, { ImageIssuer(F->PipeOf(0), F->Cases[0].Image, NaN) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::InvalidArgument, TEXT("NaN threshold (async)"));
			});
		});

		LatentIt("NaNThresholdBlocking", LatentLimit, [this](const FDoneDelegate& Done)
		{
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			struct FOut { FPipelineResult R; TArray<FPipelineDetection> Out; };
			const TSharedRef<FOut> O = MakeShared<FOut>();
			UObjectDetectionPipeline* P = F->PipeOf(0);
			const FPipelineImage Img = F->Cases[0].Image;
			RunWorker(this, F, Done, [P, O, Img]() { O->R = P->DetectImageBlocking(Img, std::numeric_limits<float>::quiet_NaN(), O->Out); }, [this, O]()
			{
				TestFalse(TEXT("bSuccess"), O->R.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)O->R.Error, *O->R.Message), O->R.Error == EPipelineError::InvalidArgument);
				TestEqual(TEXT("no detections"), O->Out.Num(), 0);
			});
		});

		LatentIt("InvalidImagesGiveInvalidArgument", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Default image, zero size, and Pixels shorter than Width*Height: InvalidArgument through the callback, nothing run.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			FPipelineImage Short;
			Short.Width = 4; Short.Height = 4; Short.Pixels.Init(FColor::Red, 3);
			FPipelineImage Zero;
			Zero.Width = 0; Zero.Height = 5;
			UObjectDetectionPipeline* P = F->PipeOf(0);
			TArray<FIssuer> Issuers = { ImageIssuer(P, FPipelineImage(), 0.5f), ImageIssuer(P, Zero, 0.5f), ImageIssuer(P, Short, 0.5f) };
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
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			struct FOut { FPipelineResult R; TArray<FPipelineDetection> Out; };
			const TSharedRef<FOut> O = MakeShared<FOut>();
			UObjectDetectionPipeline* P = F->PipeOf(0);
			RunWorker(this, F, Done, [P, O]() { O->R = P->DetectImageBlocking(FPipelineImage(), 0.5f, O->Out); }, [this, O]()
			{
				TestFalse(TEXT("bSuccess"), O->R.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)O->R.Error, *O->R.Message), O->R.Error == EPipelineError::InvalidArgument);
				TestEqual(TEXT("no detections"), O->Out.Num(), 0);
			});
		});

		LatentIt("NullAndCubeTextureGiveInvalidArgument", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// A null UTexture and a UTextureCube are not UTexture2D / render targets: InvalidArgument via the callback (works without an RHI).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UTextureCube* Cube = NewObject<UTextureCube>(GetTransientPackage());
			F->Keep.Add(TStrongObjectPtr<UObject>(Cube));
			UObjectDetectionPipeline* P = F->PipeOf(0);
			RunCalls(this, F, { TextureIssuer(P, nullptr, 0.5f), TextureIssuer(P, Cube, 0.5f) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::InvalidArgument, TEXT("null texture"));
				CheckFailure(*this, *Calls[1], EPipelineError::InvalidArgument, TEXT("UTextureCube"));
			});
		});

		LatentIt("TextureUnderNullRhiFailsPreprocessFailed", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Under -nullrhi a 2D texture cannot be read back: PreprocessFailed via the callback, no crash, no hang.
			if (HasRhi()) { AddInfo(TEXT("A real RHI is present: the nullrhi texture case does not apply, skipped (run with -nullrhi)")); Done.ExecuteIfBound(); return; }
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UTexture2D* Tex = UTexture2D::CreateTransient(8, 8, PF_B8G8R8A8);
			UTextureRenderTarget2D* Rt = NewObject<UTextureRenderTarget2D>(GetTransientPackage());
			Rt->RenderTargetFormat = RTF_RGBA8;
			Rt->InitAutoFormat(8, 8);
			if (!Tex) { AddError(TEXT("UTexture2D::CreateTransient returned null")); Done.ExecuteIfBound(); return; }
			F->Keep.Add(TStrongObjectPtr<UObject>(Tex));
			F->Keep.Add(TStrongObjectPtr<UObject>(Rt));
			UObjectDetectionPipeline* P = F->PipeOf(0);
			RunCalls(this, F, { TextureIssuer(P, Tex, 0.5f), TextureIssuer(P, Rt, 0.5f) }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::PreprocessFailed, TEXT("UTexture2D under nullrhi"));
				CheckFailure(*this, *Calls[1], EPipelineError::PreprocessFailed, TEXT("render target under nullrhi"));
			});
		});

		LatentIt("CalledOffGameThreadCompletesInvalidArgumentOnGameThread", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// C29: DetectImageAsync / DetectAsync off the game thread complete once, on the game thread, with InvalidArgument.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UTexture2D* Tex = UTexture2D::CreateTransient(4, 4, PF_B8G8R8A8);
			if (Tex) { F->Keep.Add(TStrongObjectPtr<UObject>(Tex)); }
			UObjectDetectionPipeline* P = F->PipeOf(0);
			const FPipelineImage Img = F->Cases[0].Image;
			FIssuer ImgOff = [P, Img](FPipelineDetectionCallback CB)
			{
				const TSharedRef<FPipelineDetectionCallback> Box = MakeShared<FPipelineDetectionCallback>(MoveTemp(CB));
				Async(EAsyncExecution::ThreadPool, [P, Img, Box]() { P->DetectImageAsync(Img, 0.5f, MoveTemp(*Box)); });
			};
			FIssuer TexOff = [P, Tex](FPipelineDetectionCallback CB)
			{
				const TSharedRef<FPipelineDetectionCallback> Box = MakeShared<FPipelineDetectionCallback>(MoveTemp(CB));
				Async(EAsyncExecution::ThreadPool, [P, Tex, Box]() { P->DetectAsync(Tex, 0.5f, MoveTemp(*Box)); });
			};
			RunCalls(this, F, { ImgOff, TexOff }, Done, [this](TArray<TSharedRef<FCall>>& Calls)
			{
				CheckFailure(*this, *Calls[0], EPipelineError::InvalidArgument, TEXT("DetectImageAsync off the game thread"));
				CheckFailure(*this, *Calls[1], EPipelineError::InvalidArgument, TEXT("DetectAsync off the game thread"));
			});
		});

		It("NullCallbackWarnsAndRunsNothing", [this]()
		{
			FFixture F;
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, F)) { return; }
			ExpectOneWarning(*this);
			F.PipeOf(0)->DetectImageAsync(F.Cases[0].Image, 0.5f, FPipelineDetectionCallback());
			TestEqual(TEXT("GetNumPendingCalls"), F.PipeOf(0)->GetNumPendingCalls(), 0);
		});

		LatentIt("DestroyedMidCall", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Pipeline marked garbage and collected right after 5 DetectImageAsync calls: every callback still fires once with the golden result.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SigmoidSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			const int32 PipeIdx = F->Cases[0].Pipe;
			UObjectDetectionPipeline* P = F->Pipelines[PipeIdx].Get();
			TArray<int32> Mine;
			for (int32 c = 0; c < F->Cases.Num(); ++c) { if (F->Cases[c].Pipe == PipeIdx) { Mine.Add(c); } }
			const TSharedRef<TArray<int32>> Which = MakeShared<TArray<int32>>();
			TArray<FIssuer> Issuers;
			for (int32 i = 0; i < 5; ++i)
			{
				const int32 C = Mine[i % Mine.Num()];
				Which->Add(C);
				Issuers.Add(ImageIssuer(P, F->Cases[C].Image, (float)F->Cases[C].Threshold));
			}
			RunCalls(this, F, Issuers, Done, [this, F, Which](TArray<TSharedRef<FCall>>& Calls)
			{
				for (int32 i = 0; i < Calls.Num(); ++i)
				{
					const FString L = FString::Printf(TEXT("destroyed mid-call %d"), i);
					TestTrue(*FString::Printf(TEXT("%s: bSuccess (%s)"), *L, *Calls[i]->Result.Message), Calls[i]->Result.bSuccess);
					if (Calls[i]->Result.bSuccess) { CheckDetections(*this, F->LabelsOf((*Which)[i]), F->Cases[(*Which)[i]].Expected, Calls[i]->Out, *L, TinyTol); }
				}
			}, [F, P, PipeIdx]()
			{
				P->MarkAsGarbage();
				F->Pipelines[PipeIdx].Reset();
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			});
		});

		LatentIt("EightThreadsBlocking", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 8 threads x 3 DetectImageBlocking calls on one pipeline, different images/thresholds from its cases: every result matches the golden.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeDesigned(*this, SoftmaxSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			const int32 PipeIdx = F->Cases[0].Pipe;
			TArray<int32> Mine;
			for (int32 c = 0; c < F->Cases.Num(); ++c) { if (F->Cases[c].Pipe == PipeIdx) { Mine.Add(c); } }
			struct FSlot { int32 Case = 0; FPipelineResult R; TArray<FPipelineDetection> Out; };
			const TSharedRef<TArray<FSlot>> Slots = MakeShared<TArray<FSlot>>();
			Slots->SetNum(24);
			for (int32 s = 0; s < 24; ++s) { (*Slots)[s].Case = Mine[(s * 7 + 1) % Mine.Num()]; }
			UObjectDetectionPipeline* P = F->Pipelines[PipeIdx].Get();
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
							S.R = P->DetectImageBlocking(F->Cases[S.Case].Image, (float)F->Cases[S.Case].Threshold, S.Out);
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
					Worst = FMath::Max(Worst, CheckDetections(*this, F->LabelsOf(S.Case), F->Cases[S.Case].Expected, S.Out, *L, TinyTol));
				}
				AddInfo(FString::Printf(TEXT("8 threads x 3 calls: worst |diff| %.3g"), Worst));
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
