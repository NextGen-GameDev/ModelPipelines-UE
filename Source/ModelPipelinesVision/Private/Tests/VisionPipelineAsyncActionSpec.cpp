// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
// WP5.2 visible tests (docs/contracts.md C34, vision part): UClassifyImageAsyncAction and UDetectObjectsAsyncAction.
// Names: Pipelines.Vision.AsyncAction.*
//  - headless (run under -nullrhi and with an RHI): PassThrough, Failure, Cancel, Lifetime, Reflection. They compare the node with the
//    direct C++ call (ClassifyAsync / DetectAsync) on the same pipeline and texture, whatever that call returns (PreprocessFailed under
//    -nullrhi, success with an RHI).
//  - RHI only (skipped under -nullrhi, an error when MODELPIPELINES_REQUIRE_RHI=1): Success, Concurrency.
// No test blocks the game thread on an async result: waits are FTSTicker polls inside LatentIt, with a timeout and a message.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include <limits>
#include "Async/Async.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Engine/CancellableAsyncAction.h"
#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureCube.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/UnrealType.h"
#include "PipelineBase.h"
#include "PipelineImage.h"
#include "PipelineModelPackage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "ImageClassificationPipeline.h"
#include "ObjectDetectionPipeline.h"
#include "VisionPipelineAsyncActions.h"
#include "VisionPipelineLibrary.h"
#include "PipelineVisionTestData.h"
#include "VisionAsyncActionTestReceiver.h"

using namespace UE::ModelPipelines::VisionTests;

namespace VisionAsyncSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(240);
	constexpr double WaitSeconds = 150.0;
	constexpr double GcWaitSeconds = 20.0;
	/** After the awaited pin: wait this long to catch a second (wrong) delivery. */
	constexpr double SettleSeconds = 0.4;

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

	/** Which action/pipeline pair and golden model a scenario runs on. */
	struct FKind
	{
		const TCHAR* Name = TEXT("");
		bool bDetect = false;
		const TCHAR* Section = TEXT("tiny_models");
		const TCHAR* Slug = TEXT("");
		int32 TopK = 3;
		float Threshold = 0.5f;
		/** Classification success only: expected TopK entries; -1 = do not check, -2 = number of package labels. */
		int32 ExpectEntries = -1;
		/** Number of distinct golden images to load. */
		int32 MaxImages = 2;
		/** Classification success only: top-1 index must equal the golden case's index (reference models only). */
		bool bGolden = false;
	};

	static FName SuccessPin(const FKind& K) { return K.bDetect ? FName(TEXT("DetectSuccess")) : FName(TEXT("ClassifySuccess")); }
	static FName FailurePin(const FKind& K) { return K.bDetect ? FName(TEXT("DetectFailure")) : FName(TEXT("ClassifyFailure")); }
	static FName PinFor(const FKind& K, bool bSuccess) { return bSuccess ? SuccessPin(K) : FailurePin(K); }

	struct FFixture
	{
		TSharedPtr<FJsonObject> Root;
		TStrongObjectPtr<UPipelineModelPackage> Package;
		TStrongObjectPtr<UImageClassificationPipeline> Class;
		TStrongObjectPtr<UObjectDetectionPipeline> Det;
		TArray<FString> Ids;
		TArray<FPipelineImage> Images;
		TArray<int32> GoldIndex;                 // classification goldens only, else -1
		TArray<TStrongObjectPtr<UObject>> Keep;  // textures / render targets made for the test
		UObject* Pipe() const { return Class.IsValid() ? static_cast<UObject*>(Class.Get()) : static_cast<UObject*>(Det.Get()); }
		void DropPipeline() { Class.Reset(); Det.Reset(); }
	};

	static bool MakeFixture(FAutomationTestBase& T, const FKind& K, FFixture& F)
	{
		F.Root = GetGoldens(T);
		if (!F.Root.IsValid()) { return false; }
		const TSharedPtr<FJsonObject> Model = FindModelIn(F.Root, K.Section, K.Slug);
		if (!Model.IsValid())
		{
			T.AddError(FString::Printf(TEXT("Golden model '%s' not found in '%s' of vision_goldens.json. Regenerate with Tools/make_vision_goldens.py (and Tools/fetch_reference_models.py, Tools/fetch_test_images.py for reference models)."), K.Slug, K.Section));
			return false;
		}
		FString RootDir;
		if (!F.Root->TryGetStringField(TEXT("root"), RootDir) || RootDir.IsEmpty()) { RootDir = GetDevDir() / TEXT("goldens/vision"); }
		const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
		if (!Model->TryGetArrayField(TEXT("cases"), Cases) || Cases->Num() == 0)
		{
			T.AddError(FString::Printf(TEXT("Golden model '%s' has no cases. Regenerate with Tools/make_vision_goldens.py."), K.Slug));
			return false;
		}
		for (const TSharedPtr<FJsonValue>& V : *Cases)
		{
			if (F.Images.Num() >= K.MaxImages) { break; }
			const TSharedPtr<FJsonObject> C = V->AsObject();
			const FString Id = C->GetStringField(TEXT("image"));
			if (F.Ids.Contains(Id)) { continue; }
			const TSharedPtr<FJsonObject> Entry = FindImageEntry(F.Root, Id);
			if (!Entry.IsValid())
			{
				T.AddError(FString::Printf(TEXT("Golden image '%s' not listed in images/reference_images. Regenerate with Tools/make_vision_goldens.py."), *Id));
				return false;
			}
			FString Err;
			FPipelineImage Img;
			if (!LoadPng(RootDir / Entry->GetStringField(TEXT("file")), Img, Err)) { T.AddError(Err); return false; }
			F.Ids.Add(Id);
			F.Images.Add(MoveTemp(Img));
			double Idx = -1.0;
			if (!K.bDetect) { C->TryGetNumberField(TEXT("index"), Idx); }
			F.GoldIndex.Add((int32)Idx);
		}
		if (F.Images.Num() < K.MaxImages)
		{
			T.AddError(FString::Printf(TEXT("Golden model '%s' has only %d distinct images, the test needs %d."), K.Slug, F.Images.Num(), K.MaxImages));
			return false;
		}
		FString Err;
		UPipelineModelPackage* Pkg = MakePackage(F.Root, Model, EPipelineDevice::CPU, Err);
		if (!Pkg) { T.AddError(Err); return false; }
		F.Package.Reset(Pkg);
		FPipelineResult R;
		if (K.bDetect)
		{
			UObjectDetectionPipeline* P = UVisionPipelineLibrary::CreateObjectDetectionPipeline(Pkg, R);
			if (!P) { T.AddError(FString::Printf(TEXT("CreateObjectDetectionPipeline failed: %s"), *R.Message)); return false; }
			F.Det.Reset(P);
		}
		else
		{
			UImageClassificationPipeline* P = UVisionPipelineLibrary::CreateImageClassificationPipeline(Pkg, R);
			if (!P) { T.AddError(FString::Printf(TEXT("CreateImageClassificationPipeline failed: %s"), *R.Message)); return false; }
			F.Class.Reset(P);
		}
		return true;
	}

	static bool RhiSkip(FAutomationTestBase& T)
	{
		if (HasRhi()) { return false; }
		if (IsRhiRequired()) { T.AddError(TEXT("RHI not available but MODELPIPELINES_REQUIRE_RHI=1")); }
		else { T.AddInfo(TEXT("RHI not available: skipped")); }
		return true;
	}

	/** Texture for image i: real pixels with an RHI, an 8x8 transient shell (which the pipeline must fail with PreprocessFailed) without. Kept alive by F.Keep. */
	static UTexture* MakeTex(FAutomationTestBase& T, FFixture& F, int32 I)
	{
		UTexture2D* Tex = HasRhi() ? MakeTexture(F.Images[I], PF_B8G8R8A8) : UTexture2D::CreateTransient(8, 8, PF_B8G8R8A8);
		if (!Tex) { T.AddError(FString::Printf(TEXT("texture for image %s is null (MakeTexture / CreateTransient failed)"), *F.Ids[I])); return nullptr; }
		F.Keep.Add(TStrongObjectPtr<UObject>(Tex));
		return Tex;
	}

	static UTexture* MakeRt(FAutomationTestBase& T, FFixture& F, int32 I)
	{
		UTextureRenderTarget2D* Rt = nullptr;
		if (HasRhi()) { Rt = MakeRenderTarget(F.Images[I], RTF_RGBA8); }
		else
		{
			Rt = NewObject<UTextureRenderTarget2D>(GetTransientPackage());
			Rt->RenderTargetFormat = RTF_RGBA8;
			Rt->InitAutoFormat(8, 8);
		}
		if (!Rt) { T.AddError(FString::Printf(TEXT("render target for image %s is null"), *F.Ids[I])); return nullptr; }
		F.Keep.Add(TStrongObjectPtr<UObject>(Rt));
		return Rt;
	}

	// ---------------------------------------------------------------- waiting (never blocks the game thread)

	/** Polls Cond on the core ticker. On timeout: AddError(What) and Done. On success: Next(), which must call Done itself. */
	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Next, double Timeout = WaitSeconds)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("VisionAsyncSpecWait"), 0.02f, [T, Cond, What, Done, Next, Start, Timeout](float)
		{
			if (Cond())
			{
				Next();
				return false;
			}
			if ((FPlatformTime::Seconds() - Start) > Timeout)
			{
				T->AddError(FString::Printf(TEXT("timed out after %.0f s waiting for: %s"), Timeout, *What));
				Done.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	/** Lets the game thread tick for Secs, then Next() (which must call Done itself). */
	static void Settle(double Secs, TFunction<void()> Next)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("VisionAsyncSpecSettle"), 0.02f, [Start, Secs, Next](float)
		{
			if ((FPlatformTime::Seconds() - Start) >= Secs)
			{
				Next();
				return false;
			}
			return true;
		});
	}

	/** Waits for Cond, settles (to catch a second delivery), runs Check, then Done. */
	static void WaitSettled(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Check)
	{
		WaitFor(T, Cond, What, Done, [Done, Check]()
		{
			Settle(SettleSeconds, [Done, Check]()
			{
				Check();
				Done.ExecuteIfBound();
			});
		});
	}

	// ---------------------------------------------------------------- reference results (the direct C++ call)

	struct FExpected
	{
		FThreadSafeCounter Count;
		FPipelineResult R;
		FPipelineClassification C;
		TArray<FPipelineDetection> D;
	};

	/** Issues the direct ClassifyAsync / DetectAsync call (game thread). Done when Count > 0. */
	static void StartExpected(const FKind& K, FFixture& F, UTexture* Tex, const TSharedRef<FExpected>& Out, float Threshold)
	{
		if (K.bDetect)
		{
			F.Det->DetectAsync(Tex, Threshold, [Out](const FPipelineResult& R, const TArray<FPipelineDetection>& D)
			{
				if (Out->Count.GetValue() == 0) { Out->R = R; Out->D = D; }
				Out->Count.Increment();
			});
		}
		else
		{
			F.Class->ClassifyAsync(Tex, K.TopK, [Out](const FPipelineResult& R, const FPipelineClassification& C)
			{
				if (Out->Count.GetValue() == 0) { Out->R = R; Out->C = C; }
				Out->Count.Increment();
			});
		}
	}

	/** Direct call on the same pipeline and texture, then Next(Exp) (which must call Done itself). */
	static void WithExpected(FAutomationTestBase* T, const FKind& K, const TSharedRef<FFixture>& F, UTexture* Tex, FDoneDelegate Done, TFunction<void(const TSharedRef<FExpected>&)> Next)
	{
		const TSharedRef<FExpected> Exp = MakeShared<FExpected>();
		StartExpected(K, *F, Tex, Exp, K.Threshold);
		WaitFor(T, [Exp]() { return Exp->Count.GetValue() > 0; }, TEXT("the direct ClassifyAsync/DetectAsync reference call"), Done, [Next, Exp, F]() { Next(Exp); });
	}

	// ---------------------------------------------------------------- actions

	static TStrongObjectPtr<UVisionAsyncActionTestReceiver> NewReceiver()
	{
		return TStrongObjectPtr<UVisionAsyncActionTestReceiver>(NewObject<UVisionAsyncActionTestReceiver>(GetTransientPackage()));
	}

	/** Factory call with a null world context (no game instance in the automation run), pins bound to R. */
	static UCancellableAsyncAction* NewAction(const FKind& K, UObject* Pipe, UTexture* Tex, UVisionAsyncActionTestReceiver* R)
	{
		if (K.bDetect)
		{
			UDetectObjectsAsyncAction* A = UDetectObjectsAsyncAction::DetectObjects(nullptr, Cast<UObjectDetectionPipeline>(Pipe), Tex, K.Threshold);
			if (A)
			{
				A->OnSuccess.AddDynamic(R, &UVisionAsyncActionTestReceiver::OnDetectSuccess);
				A->OnFailure.AddDynamic(R, &UVisionAsyncActionTestReceiver::OnDetectFailure);
			}
			return A;
		}
		UClassifyImageAsyncAction* A = UClassifyImageAsyncAction::ClassifyImage(nullptr, Cast<UImageClassificationPipeline>(Pipe), Tex, K.TopK);
		if (A)
		{
			A->OnSuccess.AddDynamic(R, &UVisionAsyncActionTestReceiver::OnClassifySuccess);
			A->OnFailure.AddDynamic(R, &UVisionAsyncActionTestReceiver::OnClassifyFailure);
		}
		return A;
	}

	static void Activate(UVisionAsyncActionTestReceiver* R, UCancellableAsyncAction* A)
	{
		R->bInsideActivate = true;
		A->Activate();
		R->bInsideActivate = false;
	}

	static void ExpectNullPipelineWarning(FAutomationTestBase& T, const FKind& K)
	{
		T.AddExpectedMessage(FString::Printf(TEXT("%s: Pipeline is null"), K.bDetect ? TEXT("Detect Objects") : TEXT("Classify Image")), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
	}

	static void ExpectSecondActivateWarning(FAutomationTestBase& T, const FKind& K)
	{
		T.AddExpectedMessage(FString::Printf(TEXT("%s: Activate called again on an action that was already activated or cancelled; ignored"), K.bDetect ? TEXT("Detect Objects") : TEXT("Classify Image")), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
	}

	// ---------------------------------------------------------------- checks

	static void CheckClassificationEqual(FAutomationTestBase& T, const FPipelineClassification& Got, const FPipelineClassification& Want, const FString& Label)
	{
		T.TestEqual(*(Label + TEXT(": Index")), Got.Index, Want.Index);
		T.TestEqual(*(Label + TEXT(": Label")), Got.Label, Want.Label);
		T.TestTrue(*FString::Printf(TEXT("%s: Score %.9g equals %.9g exactly"), *Label, Got.Score, Want.Score), Got.Score == Want.Score);
		if (!T.TestEqual(*(Label + TEXT(": Scores count")), Got.Scores.Num(), Want.Scores.Num())) { return; }
		if (!T.TestEqual(*(Label + TEXT(": TopK count")), Got.TopK.Num(), Want.TopK.Num())) { return; }
		int32 Bad = 0;
		double Worst = 0.0;
		for (int32 c = 0; c < Got.Scores.Num(); ++c)
		{
			if (Got.Scores[c] != Want.Scores[c]) { ++Bad; Worst = FMath::Max(Worst, (double)FMath::Abs(Got.Scores[c] - Want.Scores[c])); }
		}
		for (int32 c = 0; c < Got.TopK.Num(); ++c)
		{
			if (Got.TopK[c].Label != Want.TopK[c].Label || Got.TopK[c].Index != Want.TopK[c].Index || Got.TopK[c].Score != Want.TopK[c].Score) { ++Bad; }
		}
		T.TestEqual(*FString::Printf(TEXT("%s: scores/TopK fields differing from ClassifyAsync (exact equality required; worst score diff %.3g)"), *Label, Worst), Bad, 0);
	}

	static void CheckDetectionsEqual(FAutomationTestBase& T, const TArray<FPipelineDetection>& Got, const TArray<FPipelineDetection>& Want, const FString& Label)
	{
		if (!T.TestEqual(*(Label + TEXT(": detection count")), Got.Num(), Want.Num())) { return; }
		int32 Bad = 0;
		double Worst = 0.0;
		for (int32 i = 0; i < Got.Num(); ++i)
		{
			const FPipelineDetection& G = Got[i];
			const FPipelineDetection& W = Want[i];
			if (G.Index != W.Index || G.Label != W.Label || G.Box.bIsValid != W.Box.bIsValid) { ++Bad; }
			const double Diffs[5] = { FMath::Abs((double)G.Score - W.Score), FMath::Abs(G.Box.Min.X - W.Box.Min.X), FMath::Abs(G.Box.Min.Y - W.Box.Min.Y),
				FMath::Abs(G.Box.Max.X - W.Box.Max.X), FMath::Abs(G.Box.Max.Y - W.Box.Max.Y) };
			for (double D : Diffs) { if (D != 0.0) { ++Bad; Worst = FMath::Max(Worst, D); } }
		}
		T.TestEqual(*FString::Printf(TEXT("%s: detection fields differing from DetectAsync (exact equality required; worst diff %.3g)"), *Label, Worst), Bad, 0);
	}

	/** Exactly one record in total, on the expected pin, on the game thread, not inside Activate. Returns false if the count is wrong. */
	static bool CheckSingleDelivery(FAutomationTestBase& T, const UVisionAsyncActionTestReceiver& R, FName Pin, FName OtherPin, const FString& Label)
	{
		T.TestEqual(*FString::Printf(TEXT("%s: %s count"), *Label, *Pin.ToString()), R.CountPin(Pin), 1);
		T.TestEqual(*FString::Printf(TEXT("%s: %s count (the other pin)"), *Label, *OtherPin.ToString()), R.CountPin(OtherPin), 0);
		if (!T.TestEqual(*FString::Printf(TEXT("%s: total deliveries (exactly once)"), *Label), R.Total(), 1)) { return false; }
		T.TestTrue(*FString::Printf(TEXT("%s: delivered on the game thread"), *Label), R.Records[0].bGameThread);
		T.TestFalse(*FString::Printf(TEXT("%s: delivered inside Activate()"), *Label), R.Records[0].bInsideActivate);
		return true;
	}

	/**
	 * The record must be what the direct call returned: same pin (success or failure), same Result.Error / Device, equal outputs on success,
	 * empty outputs on failure. GoldIndex >= 0 also pins the classification top-1 to the golden (an independent check of the success path).
	 */
	static void CheckRecordMatches(FAutomationTestBase& T, const FKind& K, const FFixture& F, const FVisionAsyncActionTestRecord& Rec, const FExpected& Exp, const FString& Label, int32 GoldIndex = -1)
	{
		T.TestEqual(*(Label + TEXT(": pin")), Rec.Pin.ToString(), PinFor(K, Exp.R.bSuccess).ToString());
		T.TestTrue(*FString::Printf(TEXT("%s: Result.bSuccess %d == the direct call's %d (%s)"), *Label, (int32)Rec.Result.bSuccess, (int32)Exp.R.bSuccess, *Rec.Result.Message), Rec.Result.bSuccess == Exp.R.bSuccess);
		T.TestTrue(*FString::Printf(TEXT("%s: Result.Error %d == the direct call's %d"), *Label, (int32)Rec.Result.Error, (int32)Exp.R.Error), Rec.Result.Error == Exp.R.Error);
		if (Exp.R.bSuccess)
		{
			T.TestTrue(*FString::Printf(TEXT("%s: Result.Device %d == %d"), *Label, (int32)Rec.Result.Device, (int32)Exp.R.Device), Rec.Result.Device == Exp.R.Device);
			if (K.bDetect)
			{
				CheckDetectionsEqual(T, Rec.Detections, Exp.D, Label);
			}
			else
			{
				CheckClassificationEqual(T, Rec.Classification, Exp.C, Label);
				const int32 Want = K.ExpectEntries == -2 ? (F.Package.IsValid() ? F.Package->Labels.Num() : -1) : K.ExpectEntries;
				if (Want >= 0) { T.TestEqual(*FString::Printf(TEXT("%s: TopK entries for TopK=%d"), *Label, K.TopK), Rec.Classification.TopK.Num(), Want); }
				if (GoldIndex >= 0) { T.TestEqual(*FString::Printf(TEXT("%s: top-1 index equals the golden's (got '%s')"), *Label, *Rec.Classification.Label), Rec.Classification.Index, GoldIndex); }
			}
		}
		else
		{
			if (K.bDetect) { T.TestEqual(*(Label + TEXT(": no detections on failure")), Rec.Detections.Num(), 0); }
			else
			{
				T.TestEqual(*(Label + TEXT(": no scores on failure")), Rec.Classification.Scores.Num(), 0);
				T.TestEqual(*(Label + TEXT(": no TopK on failure")), Rec.Classification.TopK.Num(), 0);
				T.TestTrue(*(Label + TEXT(": empty label on failure")), Rec.Classification.Label.IsEmpty());
			}
		}
	}

	static FKind ClassifyTiny(int32 TopK = 3, int32 ExpectEntries = -1)
	{
		FKind K; K.Name = TEXT("ClassifyImage"); K.bDetect = false; K.Section = TEXT("tiny_models"); K.Slug = TEXT("tiny_image_classification"); K.TopK = TopK; K.ExpectEntries = ExpectEntries;
		return K;
	}

	static FKind DetectTiny()
	{
		FKind K; K.Name = TEXT("DetectObjects"); K.bDetect = true; K.Section = TEXT("tiny_models"); K.Slug = TEXT("tiny_object_detection"); K.Threshold = 0.1f;
		return K;
	}

	static TArray<FKind> GenericKinds() { return TArray<FKind>{ ClassifyTiny(2), DetectTiny() }; }

	// ---------------------------------------------------------------- RHI success scenarios

	enum class ETexKind { Texture2D, RenderTarget };

	static void ScenarioSuccess(FAutomationTestBase* T, FKind K, ETexKind TexKind, FDoneDelegate Done)
	{
		// One action on a real GPU texture / render target must give exactly what ClassifyAsync / DetectAsync give on the same pipeline and
		// texture: the node adds no logic, so a wrong TopK, a dropped clamp, a swapped threshold or a stale buffer shows up as a differing float.
		if (RhiSkip(*T)) { Done.ExecuteIfBound(); return; }
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		UTexture* Tex = TexKind == ETexKind::Texture2D ? MakeTex(*T, *F, 0) : MakeRt(*T, *F, 0);
		if (!Tex) { Done.ExecuteIfBound(); return; }
		WithExpected(T, K, F, Tex, Done, [T, K, F, Tex, Done](const TSharedRef<FExpected>& Exp)
		{
			if (!Exp->R.bSuccess) { T->AddError(FString::Printf(TEXT("reference direct call failed: %s"), *Exp->R.Message)); Done.ExecuteIfBound(); return; }
			const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R = NewReceiver();
			UCancellableAsyncAction* A = NewAction(K, F->Pipe(), Tex, R.Get());
			if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
			const TStrongObjectPtr<UCancellableAsyncAction> Hold(A);   // only so IsActive() can be read after delivery
			T->TestEqual(TEXT("nothing fired before Activate"), R->Total(), 0);
			Activate(R.Get(), A);
			T->TestEqual(TEXT("nothing fired inside Activate (count right after it returned)"), R->Total(), 0);
			T->TestTrue(TEXT("IsActive() true after Activate"), A->IsActive());
			WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("OnSuccess/OnFailure never fired"), Done, [T, K, F, R, Hold, Exp]()
			{
				if (CheckSingleDelivery(*T, *R, SuccessPin(K), FailurePin(K), K.Name)) { CheckRecordMatches(*T, K, *F, R->Records[0], *Exp, K.Name, K.bGolden ? F->GoldIndex[0] : -1); }
				T->TestFalse(TEXT("IsActive() false after delivery"), Hold->IsActive());
			});
		});
	}

	static void ScenarioEightConcurrent(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// 8 actions in flight on one pipeline, each with its own golden image and its own receiver. Receiver i must get exactly the result of
		// image i (a node that shared a result buffer, or delivered in completion order to the wrong listener, would mix them up).
		if (RhiSkip(*T)) { Done.ExecuteIfBound(); return; }
		K.MaxImages = 8;
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		const TSharedRef<TArray<UTexture*>> Texs = MakeShared<TArray<UTexture*>>();
		for (int32 i = 0; i < 8; ++i)
		{
			UTexture* Tex = MakeTex(*T, *F, i);
			if (!Tex) { Done.ExecuteIfBound(); return; }
			Texs->Add(Tex);
		}
		const TSharedRef<TArray<TSharedRef<FExpected>>> Exps = MakeShared<TArray<TSharedRef<FExpected>>>();
		for (int32 i = 0; i < 8; ++i)
		{
			Exps->Add(MakeShared<FExpected>());
			StartExpected(K, *F, (*Texs)[i], (*Exps)[i], K.Threshold);
		}
		WaitFor(T, [Exps]()
		{
			for (const TSharedRef<FExpected>& E : *Exps) { if (E->Count.GetValue() == 0) { return false; } }
			return true;
		}, TEXT("the 8 direct reference calls"), Done, [T, K, F, Texs, Exps, Done]()
		{
			TSet<FString> Signatures;
			for (int32 i = 0; i < 8; ++i)
			{
				const FExpected& E = *(*Exps)[i];
				if (!E.R.bSuccess) { T->AddError(FString::Printf(TEXT("reference call %d failed: %s"), i, *E.R.Message)); Done.ExecuteIfBound(); return; }
				FString Sig;
				if (K.bDetect) { for (const FPipelineDetection& D : E.D) { Sig += FString::Printf(TEXT("%d:%.6f:%.4f,%.4f,%.4f,%.4f;"), D.Index, D.Score, D.Box.Min.X, D.Box.Min.Y, D.Box.Max.X, D.Box.Max.Y); } }
				else { for (float V : E.C.Scores) { Sig += FString::Printf(TEXT("%.6f,"), V); } }
				Signatures.Add(Sig);
			}
			if (Signatures.Num() < 3) { T->AddError(FString::Printf(TEXT("the 8 test images give only %d distinct reference results; cross-delivery could not be detected"), Signatures.Num())); }
			const TSharedRef<TArray<TStrongObjectPtr<UVisionAsyncActionTestReceiver>>> Rs = MakeShared<TArray<TStrongObjectPtr<UVisionAsyncActionTestReceiver>>>();
			TArray<UCancellableAsyncAction*> Actions;
			for (int32 i = 0; i < 8; ++i)
			{
				Rs->Add(NewReceiver());
				UCancellableAsyncAction* A = NewAction(K, F->Pipe(), (*Texs)[i], (*Rs)[i].Get());
				if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
				Actions.Add(A);
			}
			for (int32 i = 0; i < 8; ++i) { Activate((*Rs)[i].Get(), Actions[i]); }
			WaitSettled(T, [Rs]()
			{
				for (const TStrongObjectPtr<UVisionAsyncActionTestReceiver>& R : *Rs) { if (R->Total() == 0) { return false; } }
				return true;
			}, TEXT("not all 8 actions delivered"), Done, [T, K, F, Rs, Exps]()
			{
				for (int32 i = 0; i < 8; ++i)
				{
					const FString Label = FString::Printf(TEXT("%s action %d (image %s)"), K.Name, i, *F->Ids[i]);
					if (CheckSingleDelivery(*T, *(*Rs)[i], SuccessPin(K), FailurePin(K), Label)) { CheckRecordMatches(*T, K, *F, (*Rs)[i]->Records[0], *(*Exps)[i], Label, K.bGolden ? F->GoldIndex[i] : -1); }
				}
			});
		});
	}

	// ---------------------------------------------------------------- pass-through scenarios (headless-capable)

	enum class EBadInput { NullTexture, CubeTexture, TransientTexture2D, TransientRenderTarget, NanThreshold };

	static void ScenarioPassThrough(FAutomationTestBase* T, FKind K, EBadInput Bad, FDoneDelegate Done)
	{
		// Every check except "Pipeline is null" belongs to the pipeline: the node must deliver on OnFailure exactly the Error the direct call returns
		// (it must not map it, pre-check it, or swallow it), once, with empty outputs.
		EPipelineError WantError = EPipelineError::InvalidArgument;
		if (Bad == EBadInput::TransientTexture2D || Bad == EBadInput::TransientRenderTarget)
		{
			if (HasRhi()) { T->AddInfo(TEXT("A real RHI is present: the nullrhi PreprocessFailed case does not apply, skipped (run with -nullrhi)")); Done.ExecuteIfBound(); return; }
			WantError = EPipelineError::PreprocessFailed;
		}
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		UTexture* Tex = nullptr;
		switch (Bad)
		{
		case EBadInput::NullTexture: break;
		case EBadInput::CubeTexture:
		{
			UTextureCube* Cube = NewObject<UTextureCube>(GetTransientPackage());
			F->Keep.Add(TStrongObjectPtr<UObject>(Cube));
			Tex = Cube;
			break;
		}
		case EBadInput::TransientTexture2D:
		case EBadInput::NanThreshold:
			Tex = MakeTex(*T, *F, 0);   // no RHI -> transient shell; NaN case: a texture that would otherwise be fine
			if (!Tex) { Done.ExecuteIfBound(); return; }
			break;
		case EBadInput::TransientRenderTarget:
			Tex = MakeRt(*T, *F, 0);
			if (!Tex) { Done.ExecuteIfBound(); return; }
			break;
		}
		if (Bad == EBadInput::NanThreshold) { K.Threshold = std::numeric_limits<float>::quiet_NaN(); }
		const FKind KK = K;
		const FString Label = FString::Printf(TEXT("%s bad input %d"), K.Name, (int32)Bad);
		WithExpected(T, K, F, Tex, Done, [T, KK, F, Tex, WantError, Label, Done](const TSharedRef<FExpected>& Exp)
		{
			T->TestFalse(*(Label + TEXT(": the direct call fails")), Exp->R.bSuccess);
			T->TestTrue(*FString::Printf(TEXT("%s: the direct call's Error is %d, got %d (%s)"), *Label, (int32)WantError, (int32)Exp->R.Error, *Exp->R.Message), Exp->R.Error == WantError);
			const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R = NewReceiver();
			UCancellableAsyncAction* A = NewAction(KK, F->Pipe(), Tex, R.Get());
			if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
			Activate(R.Get(), A);
			T->TestEqual(TEXT("nothing fired inside Activate"), R->Total(), 0);
			WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("failure never fired"), Done, [T, KK, F, R, Exp, Label]()
			{
				if (!CheckSingleDelivery(*T, *R, FailurePin(KK), SuccessPin(KK), Label)) { return; }
				CheckRecordMatches(*T, KK, *F, R->Records[0], *Exp, Label);
			});
		});
	}

	static void ScenarioNullPipeline(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// Null pipeline (with a perfectly fine texture): failure, InvalidArgument, "Pipeline is null", empty output, delivered later (never inside Activate).
		// A node that delivered synchronously would fire inside Activate and break latent-node semantics in Blueprints.
		const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R = NewReceiver();
		const TStrongObjectPtr<UObject> Tex(UTexture2D::CreateTransient(4, 4, PF_B8G8R8A8));
		ExpectNullPipelineWarning(*T, K);
		UCancellableAsyncAction* A = NewAction(K, nullptr, Cast<UTexture>(Tex.Get()), R.Get());
		if (!T->TestNotNull(TEXT("factory never fails, even with a null pipeline"), A)) { Done.ExecuteIfBound(); return; }
		const TStrongObjectPtr<UCancellableAsyncAction> Hold(A);
		Activate(R.Get(), A);
		T->TestEqual(TEXT("null pipeline: nothing fired inside Activate"), R->Total(), 0);
		T->TestTrue(TEXT("null pipeline: IsActive() true until the failure is delivered"), A->IsActive());
		WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("null-pipeline failure never fired"), Done, [T, K, R, Hold, Tex]()
		{
			if (!CheckSingleDelivery(*T, *R, FailurePin(K), SuccessPin(K), TEXT("null pipeline"))) { return; }
			const FVisionAsyncActionTestRecord& Rec = R->Records[0];
			T->TestFalse(TEXT("null pipeline: Result.bSuccess"), Rec.Result.bSuccess);
			T->TestTrue(*FString::Printf(TEXT("null pipeline: Error InvalidArgument, got %d"), (int32)Rec.Result.Error), Rec.Result.Error == EPipelineError::InvalidArgument);
			T->TestEqual(TEXT("null pipeline: message"), Rec.Result.Message, FString(TEXT("Pipeline is null")));
			T->TestEqual(TEXT("null pipeline: default Classification (no scores)"), Rec.Classification.Scores.Num(), 0);
			T->TestEqual(TEXT("null pipeline: default Classification (no TopK)"), Rec.Classification.TopK.Num(), 0);
			T->TestEqual(TEXT("null pipeline: empty detections"), Rec.Detections.Num(), 0);
			T->TestFalse(TEXT("null pipeline: IsActive() false after delivery"), Hold->IsActive());
		});
	}

	static void ScenarioNullPipelineCancelled(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// The null-pipeline failure is also deferred, so Cancel() right after Activate must suppress it.
		const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R = NewReceiver();
		ExpectNullPipelineWarning(*T, K);
		UCancellableAsyncAction* A = NewAction(K, nullptr, nullptr, R.Get());
		if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
		Activate(R.Get(), A);
		A->Cancel();
		T->TestFalse(TEXT("IsActive() false right after Cancel"), A->IsActive());
		Settle(1.0, [T, R, Done]()
		{
			T->TestEqual(TEXT("cancelled null-pipeline action: no pin fired"), R->Total(), 0);
			Done.ExecuteIfBound();
		});
	}

	static void ScenarioUninitializedPipeline(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// A pipeline that was never initialised fails by itself; the node passes that Error through unchanged (no node-side mapping to InvalidArgument).
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (K.bDetect) { F->Det.Reset(NewObject<UObjectDetectionPipeline>(GetTransientPackage())); }
		else { F->Class.Reset(NewObject<UImageClassificationPipeline>(GetTransientPackage())); }
		const TStrongObjectPtr<UObject> Tex(UTexture2D::CreateTransient(4, 4, PF_B8G8R8A8));
		UTexture* T2 = Cast<UTexture>(Tex.Get());
		WithExpected(T, K, F, T2, Done, [T, K, F, T2, Tex, Done](const TSharedRef<FExpected>& Exp)
		{
			if (!T->TestFalse(TEXT("direct call on an uninitialised pipeline fails"), Exp->R.bSuccess)) { Done.ExecuteIfBound(); return; }
			const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R = NewReceiver();
			UCancellableAsyncAction* A = NewAction(K, F->Pipe(), T2, R.Get());
			if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
			Activate(R.Get(), A);
			WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("failure never fired"), Done, [T, K, F, R, Exp, Tex]()
			{
				if (CheckSingleDelivery(*T, *R, FailurePin(K), SuccessPin(K), TEXT("uninitialised pipeline"))) { CheckRecordMatches(*T, K, *F, R->Records[0], *Exp, TEXT("uninitialised pipeline")); }
			});
		});
	}

	// ---------------------------------------------------------------- generic delivery scenarios (success with an RHI, PreprocessFailed without)

	static void ScenarioSecondActivate(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// A second Activate warns and does nothing: still exactly one delivery (a second call would deliver twice).
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		UTexture* Tex = MakeTex(*T, *F, 0);
		if (!Tex) { Done.ExecuteIfBound(); return; }
		WithExpected(T, K, F, Tex, Done, [T, K, F, Tex, Done](const TSharedRef<FExpected>& Exp)
		{
			const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R = NewReceiver();
			UCancellableAsyncAction* A = NewAction(K, F->Pipe(), Tex, R.Get());
			if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
			ExpectSecondActivateWarning(*T, K);
			Activate(R.Get(), A);
			Activate(R.Get(), A);
			WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("no pin fired"), Done, [T, K, F, R, Exp]()
			{
				if (CheckSingleDelivery(*T, *R, PinFor(K, Exp->R.bSuccess), PinFor(K, !Exp->R.bSuccess), TEXT("double Activate"))) { CheckRecordMatches(*T, K, *F, R->Records[0], *Exp, TEXT("double Activate")); }
				UPipelineBase* Base = Cast<UPipelineBase>(F->Pipe());
				if (Base) { T->TestEqual(TEXT("double Activate: no extra call pending"), Base->GetNumPendingCalls(), 0); }
			});
		});
	}

	static void ScenarioCancelRightAfterActivate(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// Cancel before delivery: no pin ever fires; the running call is not interrupted, its result is dropped, then the action is released.
		// A control action on the same pipeline proves the executor is not stuck behind the cancelled call.
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		UTexture* Tex = MakeTex(*T, *F, 0);
		if (!Tex) { Done.ExecuteIfBound(); return; }
		WithExpected(T, K, F, Tex, Done, [T, K, F, Tex, Done](const TSharedRef<FExpected>& Exp)
		{
			const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R1 = NewReceiver();
			const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R2 = NewReceiver();
			UCancellableAsyncAction* A = NewAction(K, F->Pipe(), Tex, R1.Get());
			if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
			const TWeakObjectPtr<UCancellableAsyncAction> Weak(A);
			Activate(R1.Get(), A);
			T->TestTrue(TEXT("IsActive() true after Activate"), A->IsActive());
			A->Cancel();
			T->TestFalse(TEXT("IsActive() false after Cancel"), A->IsActive());
			A->Cancel();   // second Cancel: no-op, no crash
			T->TestFalse(TEXT("IsActive() still false after a second Cancel"), A->IsActive());
			UCancellableAsyncAction* Control = NewAction(K, F->Pipe(), Tex, R2.Get());
			if (!T->TestNotNull(TEXT("control action"), Control)) { Done.ExecuteIfBound(); return; }
			Activate(R2.Get(), Control);
			WaitSettled(T, [R2]() { return R2->Total() > 0; }, TEXT("control action never delivered (executor stuck behind the cancelled call?)"), Done, [T, K, F, R1, R2, Weak, Exp, Done]()
			{
				T->TestEqual(TEXT("cancelled action: no pin fired"), R1->Total(), 0);
				if (CheckSingleDelivery(*T, *R2, PinFor(K, Exp->R.bSuccess), PinFor(K, !Exp->R.bSuccess), TEXT("control"))) { CheckRecordMatches(*T, K, *F, R2->Records[0], *Exp, TEXT("control")); }
				// The cancelled call's result may arrive after the control's: poll (collecting garbage) until the action is released and nothing is pending.
				WaitFor(T, [F, Weak]()
				{
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					UPipelineBase* Base = Cast<UPipelineBase>(F->Pipe());
					return !Weak.IsValid() && (!Base || Base->GetNumPendingCalls() == 0);
				}, TEXT("the cancelled action to be released (unrooted and collected) and no call left pending"), Done, [T, R1, Done]()
				{
					T->TestEqual(TEXT("cancelled action: still no pin after its result arrived"), R1->Total(), 0);
					Done.ExecuteIfBound();
				}, GcWaitSeconds);
			});
		});
	}

	static void ScenarioCancelAfterDelivery(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// Cancel after delivery (and twice) is a no-op: no second pin, no crash, IsActive stays false.
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		UTexture* Tex = MakeTex(*T, *F, 0);
		if (!Tex) { Done.ExecuteIfBound(); return; }
		const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R = NewReceiver();
		UCancellableAsyncAction* A = NewAction(K, F->Pipe(), Tex, R.Get());
		if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
		const TStrongObjectPtr<UCancellableAsyncAction> Hold(A);
		Activate(R.Get(), A);
		WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("no pin fired"), Done, [T, K, R, Hold, Done]()
		{
			T->TestEqual(TEXT("before Cancel: exactly one delivery"), R->Total(), 1);
			T->TestFalse(TEXT("IsActive() false after delivery"), Hold->IsActive());
			Hold->Cancel();
			Hold->Cancel();
			T->TestFalse(TEXT("IsActive() false after Cancel following delivery"), Hold->IsActive());
			Settle(SettleSeconds, [T, R, Done]()
			{
				T->TestEqual(TEXT("Cancel after delivery: still exactly one delivery in total"), R->Total(), 1);
				Done.ExecuteIfBound();
			});
		});
	}

	static void ScenarioKeepAlive(FAutomationTestBase* T, FKind K, bool bDropPipeline, bool bDropTexture, FDoneDelegate Done)
	{
		// The test keeps no reference to the action (weak pointer only) and collects garbage right after Activate. The action roots itself until
		// delivery, so the pin still fires with the right result. bDropTexture: the test also lets go of the texture, which then has no reference
		// but the action's Image property (a texture collected under an in-flight readback would crash or fail the call). bDropPipeline: same for the pipeline.
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		UTexture* Tex0 = MakeTex(*T, *F, 0);
		if (!Tex0) { Done.ExecuteIfBound(); return; }
		WithExpected(T, K, F, Tex0, Done, [T, K, F, Tex0, bDropPipeline, bDropTexture, Done](const TSharedRef<FExpected>& Exp)
		{
			const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R = NewReceiver();
			const TWeakObjectPtr<UTexture> WeakTex(Tex0);
			const TWeakObjectPtr<UObject> WeakPipe(F->Pipe());
			TWeakObjectPtr<UCancellableAsyncAction> Weak;
			{
				UCancellableAsyncAction* A = NewAction(K, F->Pipe(), Tex0, R.Get());
				if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
				Weak = A;
				Activate(R.Get(), A);
			}
			if (bDropTexture) { F->Keep.Empty(); }
			if (bDropPipeline) { F->DropPipeline(); }
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			T->TestTrue(TEXT("action survived GC while in flight (rooted)"), Weak.IsValid());
			if (bDropTexture) { T->TestTrue(TEXT("texture survived GC while the action holds it"), WeakTex.IsValid()); }
			if (bDropPipeline) { T->TestTrue(TEXT("pipeline survived GC while the action holds it"), WeakPipe.IsValid()); }
			WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("pin never fired after GC"), Done, [T, K, F, R, Exp]()
			{
				if (CheckSingleDelivery(*T, *R, PinFor(K, Exp->R.bSuccess), PinFor(K, !Exp->R.bSuccess), TEXT("keep-alive"))) { CheckRecordMatches(*T, K, *F, R->Records[0], *Exp, TEXT("keep-alive")); }
			});
		});
	}

	static void ScenarioInputsReleasedAfterDelivery(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// After delivery the action drops Pipeline and Image (review note on WP5.1: do not keep inputs alive longer than the call). With the test holding
		// nothing, a few GC passes after the pin must collect the action, the texture and the pipeline.
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		UTexture* Tex0 = MakeTex(*T, *F, 0);
		if (!Tex0) { Done.ExecuteIfBound(); return; }
		WithExpected(T, K, F, Tex0, Done, [T, K, F, Tex0, Done](const TSharedRef<FExpected>& Exp)
		{
			const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R = NewReceiver();
			const TWeakObjectPtr<UTexture> WeakTex(Tex0);
			const TWeakObjectPtr<UObject> WeakPipe(F->Pipe());
			TWeakObjectPtr<UCancellableAsyncAction> Weak;
			{
				UCancellableAsyncAction* A = NewAction(K, F->Pipe(), Tex0, R.Get());
				if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
				Weak = A;
				Activate(R.Get(), A);
			}
			F->Keep.Empty();
			F->DropPipeline();
			WaitFor(T, [R]() { return R->Total() > 0; }, TEXT("pin never fired"), Done, [T, K, F, R, Exp, Weak, WeakTex, WeakPipe, Done]()
			{
				T->TestEqual(TEXT("exactly one delivery"), R->Total(), 1);
				if (R->Total() >= 1) { CheckRecordMatches(*T, K, *F, R->Records[0], *Exp, TEXT("inputs released")); }
				WaitFor(T, [Weak, WeakTex, WeakPipe]()
				{
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					return !Weak.IsValid() && !WeakTex.IsValid() && !WeakPipe.IsValid();
				}, TEXT("action, texture and pipeline to be collected after delivery (action must null Pipeline and Image)"), Done, [Done]() { Done.ExecuteIfBound(); }, GcWaitSeconds);
			});
		});
	}

	static void ScenarioListenerDestroyed(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// The object bound to the pins is destroyed and collected while the call is in flight: the broadcast must skip it without a crash.
		// A control action (own receiver) proves the call finished; afterwards the first action must have been released.
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		UTexture* Tex = MakeTex(*T, *F, 0);
		if (!Tex) { Done.ExecuteIfBound(); return; }
		TStrongObjectPtr<UVisionAsyncActionTestReceiver> R1 = NewReceiver();
		const TStrongObjectPtr<UVisionAsyncActionTestReceiver> R2 = NewReceiver();
		UCancellableAsyncAction* A = NewAction(K, F->Pipe(), Tex, R1.Get());
		if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
		const TWeakObjectPtr<UCancellableAsyncAction> Weak(A);
		const TWeakObjectPtr<UVisionAsyncActionTestReceiver> WeakR1(R1.Get());
		Activate(R1.Get(), A);
		UCancellableAsyncAction* Control = NewAction(K, F->Pipe(), Tex, R2.Get());
		if (!T->TestNotNull(TEXT("control action"), Control)) { Done.ExecuteIfBound(); return; }
		Activate(R2.Get(), Control);
		R1.Reset();
		CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		T->TestFalse(TEXT("listener was collected"), WeakR1.IsValid());
		WaitSettled(T, [R2]() { return R2->Total() > 0; }, TEXT("control action never delivered"), Done, [T, R2, Weak, F, Done]()
		{
			T->TestEqual(TEXT("control: exactly one delivery"), R2->Total(), 1);
			WaitFor(T, [Weak]() { CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS); return !Weak.IsValid(); },
				TEXT("the action to be released after delivering to a dead listener"), Done, [Done]() { Done.ExecuteIfBound(); }, GcWaitSeconds);
		});
	}

	// ---------------------------------------------------------------- reflection

	static void CheckFactory(FAutomationTestBase& T, UClass* Cls, const TCHAR* FuncName, const TCHAR* DisplayName, const TCHAR* ThirdParam, double DefaultValue)
	{
		UFunction* Fn = Cls->FindFunctionByName(FuncName);
		if (!T.TestNotNull(*FString::Printf(TEXT("%s::%s exists as a UFUNCTION"), *Cls->GetName(), FuncName), Fn)) { return; }
		T.TestTrue(TEXT("factory is static"), Fn->HasAnyFunctionFlags(FUNC_Static));
		T.TestTrue(TEXT("factory is BlueprintCallable"), Fn->HasAnyFunctionFlags(FUNC_BlueprintCallable));
#if WITH_METADATA
		T.TestTrue(TEXT("meta BlueprintInternalUseOnly"), Fn->HasMetaData(TEXT("BlueprintInternalUseOnly")));
		T.TestEqual(TEXT("BlueprintInternalUseOnly value"), Fn->GetMetaData(TEXT("BlueprintInternalUseOnly")), FString(TEXT("true")));
		T.TestEqual(TEXT("meta WorldContext"), Fn->GetMetaData(TEXT("WorldContext")), FString(TEXT("WorldContextObject")));
		T.TestEqual(TEXT("meta DisplayName"), Fn->GetMetaData(TEXT("DisplayName")), FString(DisplayName));
		T.TestEqual(TEXT("meta Category"), Fn->GetMetaData(TEXT("Category")), FString(TEXT("Model Pipelines|Vision")));
		const FName DefaultKey(*FString::Printf(TEXT("CPP_Default_%s"), ThirdParam));
		if (T.TestTrue(*FString::Printf(TEXT("%s has a default value in metadata"), ThirdParam), Fn->HasMetaData(DefaultKey)))
		{
			const FString Def = Fn->GetMetaData(DefaultKey);
			T.TestTrue(*FString::Printf(TEXT("%s default is %g, metadata says '%s'"), ThirdParam, DefaultValue, *Def), FMath::IsNearlyEqual(FCString::Atod(*Def), DefaultValue, 1e-6));
		}
#else
		(void)DisplayName; (void)DefaultValue;
#endif
		T.TestNotNull(TEXT("parameter WorldContextObject"), Fn->FindPropertyByName(TEXT("WorldContextObject")));
		T.TestNotNull(TEXT("parameter Pipeline"), Fn->FindPropertyByName(TEXT("Pipeline")));
		T.TestNotNull(TEXT("parameter Image"), Fn->FindPropertyByName(TEXT("Image")));
		T.TestNotNull(*FString::Printf(TEXT("parameter %s"), ThirdParam), Fn->FindPropertyByName(ThirdParam));
		const FObjectProperty* Ret = CastField<FObjectProperty>(Fn->GetReturnProperty());
		if (T.TestNotNull(TEXT("factory has an object return value"), Ret)) { T.TestTrue(TEXT("factory returns the action class"), Ret->PropertyClass == Cls); }
	}

	static void CheckPins(FAutomationTestBase& T, UClass* Cls, const TCHAR* OutputParamName)
	{
		TArray<const FMulticastDelegateProperty*> Assignable;
		for (TFieldIterator<FMulticastDelegateProperty> It(Cls); It; ++It)
		{
			if (It->HasAnyPropertyFlags(CPF_BlueprintAssignable)) { Assignable.Add(*It); }
		}
		if (!T.TestEqual(*FString::Printf(TEXT("%s: number of BlueprintAssignable delegates"), *Cls->GetName()), Assignable.Num(), 2)) { return; }
		const FMulticastDelegateProperty* Ok = nullptr;
		const FMulticastDelegateProperty* Fail = nullptr;
		for (const FMulticastDelegateProperty* P : Assignable)
		{
			if (P->GetFName() == TEXT("OnSuccess")) { Ok = P; }
			if (P->GetFName() == TEXT("OnFailure")) { Fail = P; }
		}
		if (!T.TestNotNull(TEXT("property OnSuccess"), Ok) || !T.TestNotNull(TEXT("property OnFailure"), Fail)) { return; }
		T.TestTrue(TEXT("OnSuccess and OnFailure share one delegate signature (required by K2Node_BaseAsyncTask)"), Ok->SignatureFunction == Fail->SignatureFunction);
		if (Ok->SignatureFunction)
		{
			T.TestNotNull(TEXT("signature parameter Result"), Ok->SignatureFunction->FindPropertyByName(TEXT("Result")));
			T.TestNotNull(*FString::Printf(TEXT("signature parameter %s"), OutputParamName), Ok->SignatureFunction->FindPropertyByName(OutputParamName));
			int32 NumParams = 0;
			for (TFieldIterator<FProperty> It(Ok->SignatureFunction); It; ++It) { ++NumParams; }
			T.TestEqual(TEXT("signature has exactly two parameters"), NumParams, 2);
		}
	}
}

BEGIN_DEFINE_SPEC(FVisionPipelineAsyncActionSpec, "Pipelines.Vision.AsyncAction",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FVisionPipelineAsyncActionSpec)

void FVisionPipelineAsyncActionSpec::Define()
{
	using namespace VisionAsyncSpecPrivate;

	const TArray<FKind> Kinds = GenericKinds();

	// ------------------------------------------------------------------------------------------------------
	// RHI only: success through real textures.
	Describe("Success", [this]()
	{
		LatentIt("ClassifyTinyTexture2DEqualsClassifyAsync", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] tiny classifier, TopK 3 on a B8G8R8A8 texture: equal to ClassifyAsync on the same pipeline and texture, 3 TopK entries.
			ScenarioSuccess(this, ClassifyTiny(3, 3), ETexKind::Texture2D, Done);
		});

		LatentIt("ClassifyTinyRenderTargetEqualsClassifyAsync", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] same with a RGBA8 render target.
			ScenarioSuccess(this, ClassifyTiny(3, 3), ETexKind::RenderTarget, Done);
		});

		LatentIt("ClassifyTopK0ClampsToOneEntry", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] TopK 0 is clamped by the pipeline (to 1), not by the node: one TopK entry, identical to ClassifyAsync(TopK 0).
			ScenarioSuccess(this, ClassifyTiny(0, 1), ETexKind::Texture2D, Done);
		});

		LatentIt("ClassifyTopKAboveLabelCountClampsToLabelCount", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] TopK 99 on the 4-label tiny model gives as many entries as the package has labels.
			ScenarioSuccess(this, ClassifyTiny(99, -2), ETexKind::Texture2D, Done);
		});

		LatentIt("ClassifyNegativeTopKClampsToOneEntry", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] TopK -4 behaves like 1.
			ScenarioSuccess(this, ClassifyTiny(-4, 1), ETexKind::Texture2D, Done);
		});

		LatentIt("ClassifyResnet50Texture2DTopK3EqualsClassifyAsyncAndGolden", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] resnet-50 reference package on a golden photo: equal (label, index, every score, exactly) to ClassifyAsync, and the top-1 is the golden's.
			FKind K; K.Name = TEXT("ClassifyImage resnet-50"); K.bDetect = false; K.Section = TEXT("reference_models"); K.Slug = TEXT("resnet-50"); K.TopK = 3; K.ExpectEntries = 3; K.bGolden = true; K.MaxImages = 2;
			ScenarioSuccess(this, K, ETexKind::Texture2D, Done);
		});

		LatentIt("ClassifyResnet50RenderTargetTopK3EqualsClassifyAsyncAndGolden", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] same with an RGBA8 render target.
			FKind K; K.Name = TEXT("ClassifyImage resnet-50 RT"); K.bDetect = false; K.Section = TEXT("reference_models"); K.Slug = TEXT("resnet-50"); K.TopK = 3; K.ExpectEntries = 3; K.bGolden = true; K.MaxImages = 2;
			ScenarioSuccess(this, K, ETexKind::RenderTarget, Done);
		});

		LatentIt("DetectRtDetrTexture2DThreshold05EqualsDetectAsync", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] RT-DETR reference package, threshold 0.5: detections (count, order, index, label, score, box) equal to DetectAsync.
			FKind K; K.Name = TEXT("DetectObjects rtdetr"); K.bDetect = true; K.Section = TEXT("reference_models"); K.Slug = TEXT("rtdetr_r18vd"); K.Threshold = 0.5f; K.MaxImages = 2;
			ScenarioSuccess(this, K, ETexKind::Texture2D, Done);
		});

		LatentIt("DetectRtDetrRenderTargetThreshold05EqualsDetectAsync", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] same with a render target.
			FKind K; K.Name = TEXT("DetectObjects rtdetr RT"); K.bDetect = true; K.Section = TEXT("reference_models"); K.Slug = TEXT("rtdetr_r18vd"); K.Threshold = 0.5f; K.MaxImages = 2;
			ScenarioSuccess(this, K, ETexKind::RenderTarget, Done);
		});

		LatentIt("DetectRtDetrThreshold02PassesThresholdThrough", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] a different threshold gives (usually) more detections: a node that ignored its ScoreThreshold input and used a fixed 0.5 would not equal DetectAsync(0.2).
			FKind K; K.Name = TEXT("DetectObjects rtdetr 0.2"); K.bDetect = true; K.Section = TEXT("reference_models"); K.Slug = TEXT("rtdetr_r18vd"); K.Threshold = 0.2f; K.MaxImages = 2;
			ScenarioSuccess(this, K, ETexKind::Texture2D, Done);
		});

		LatentIt("DetectTinyDesignedSigmoidTexture2DEqualsDetectAsync", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] designed tiny detector (known detections), threshold 0.3.
			FKind K; K.Name = TEXT("DetectObjects designed sigmoid"); K.bDetect = true; K.Section = TEXT("tiny_models"); K.Slug = TEXT("tiny_det_designed_sigmoid"); K.Threshold = 0.3f; K.MaxImages = 1;
			ScenarioSuccess(this, K, ETexKind::Texture2D, Done);
		});
	});

	// ------------------------------------------------------------------------------------------------------
	// Headless-capable: the pipeline's own failures pass through the node.
	Describe("PassThrough", [this, Kinds]()
	{
		for (const FKind& K : Kinds)
		{
			Describe(K.Name, [this, K]()
			{
				LatentIt("NullTexturePassesPipelineErrorThrough", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioPassThrough(this, K, EBadInput::NullTexture, Done); });
				LatentIt("CubeTexturePassesPipelineErrorThrough", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioPassThrough(this, K, EBadInput::CubeTexture, Done); });
				LatentIt("Texture2DUnderNullRhiFailsPreprocessFailed", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioPassThrough(this, K, EBadInput::TransientTexture2D, Done); });
				LatentIt("RenderTargetUnderNullRhiFailsPreprocessFailed", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioPassThrough(this, K, EBadInput::TransientRenderTarget, Done); });
				LatentIt("UninitializedPipelinePassesPipelineErrorThrough", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioUninitializedPipeline(this, K, Done); });
			});
		}

		LatentIt("DetectNaNThresholdFailsInvalidArgumentViaPipeline", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// NaN threshold: the pipeline (not the node) rejects it with InvalidArgument, and the node passes that through on OnFailure.
			// A node that clamped NaN to 0.5 before the call would deliver OnSuccess (RHI) or PreprocessFailed (nullrhi) instead.
			ScenarioPassThrough(this, DetectTiny(), EBadInput::NanThreshold, Done);
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Failure", [this, Kinds]()
	{
		for (const FKind& K : Kinds)
		{
			Describe(K.Name, [this, K]()
			{
				LatentIt("NullPipelineFailsOnceWithInvalidArgumentLaterThanActivate", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioNullPipeline(this, K, Done); });
				LatentIt("CancelledNullPipelineActionFiresNothing", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioNullPipelineCancelled(this, K, Done); });
				LatentIt("SecondActivateWarnsAndDeliversOnce", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioSecondActivate(this, K, Done); });
			});
		}
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Cancel", [this, Kinds]()
	{
		for (const FKind& K : Kinds)
		{
			Describe(K.Name, [this, K]()
			{
				LatentIt("CancelRightAfterActivateFiresNoPinAndReleasesAction", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioCancelRightAfterActivate(this, K, Done); });
				LatentIt("CancelAfterDeliveryAndDoubleCancelAreNoOps", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioCancelAfterDelivery(this, K, Done); });
			});
		}
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Lifetime", [this, Kinds]()
	{
		for (const FKind& K : Kinds)
		{
			Describe(K.Name, [this, K]()
			{
				LatentIt("ActionSurvivesGCWithNoReferencesHeldByTest", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioKeepAlive(this, K, false, false, Done); });
				LatentIt("ActionKeepsTextureAliveWhenTestDropsIt", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioKeepAlive(this, K, false, true, Done); });
				LatentIt("ActionKeepsPipelineAndTextureAliveWhenTestDropsBoth", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioKeepAlive(this, K, true, true, Done); });
				LatentIt("ActionReleasesPipelineAndImageAfterDelivery", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioInputsReleasedAfterDelivery(this, K, Done); });
				LatentIt("ListenerDestroyedAndCollectedWhileInFlightIsNoCrash", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioListenerDestroyed(this, K, Done); });
			});
		}
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Concurrency", [this]()
	{
		LatentIt("ClassifyEightActionsOnResnet50EachGetsItsOwnResult", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] 8 actions at once on one resnet-50 pipeline with 8 different golden images; the 8 reference results must differ so crossed deliveries are detectable.
			FKind K; K.Name = TEXT("ClassifyImage resnet-50"); K.bDetect = false; K.Section = TEXT("reference_models"); K.Slug = TEXT("resnet-50"); K.TopK = 3; K.ExpectEntries = 3; K.bGolden = true;
			ScenarioEightConcurrent(this, K, Done);
		});

		LatentIt("DetectEightActionsOnRtDetrEachGetsItsOwnResult", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// [RHI] same for RT-DETR, threshold 0.5.
			FKind K; K.Name = TEXT("DetectObjects rtdetr"); K.bDetect = true; K.Section = TEXT("reference_models"); K.Slug = TEXT("rtdetr_r18vd"); K.Threshold = 0.5f;
			ScenarioEightConcurrent(this, K, Done);
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Reflection", [this]()
	{
		It("ClassifyImageFactoryMetadata", [this]()
		{
			CheckFactory(*this, UClassifyImageAsyncAction::StaticClass(), TEXT("ClassifyImage"), TEXT("Classify Image"), TEXT("TopK"), 5.0);
		});

		It("DetectObjectsFactoryMetadata", [this]()
		{
			CheckFactory(*this, UDetectObjectsAsyncAction::StaticClass(), TEXT("DetectObjects"), TEXT("Detect Objects"), TEXT("ScoreThreshold"), 0.5);
		});

		It("ClassifyImageHasExactlyTwoAssignablePinsWithOneSignature", [this]()
		{
			CheckPins(*this, UClassifyImageAsyncAction::StaticClass(), TEXT("Classification"));
		});

		It("DetectObjectsHasExactlyTwoAssignablePinsWithOneSignature", [this]()
		{
			CheckPins(*this, UDetectObjectsAsyncAction::StaticClass(), TEXT("Detections"));
		});

		It("ActionsDeriveFromCancellableAsyncAction", [this]()
		{
			TestTrue(TEXT("UClassifyImageAsyncAction is a UCancellableAsyncAction"), UClassifyImageAsyncAction::StaticClass()->IsChildOf(UCancellableAsyncAction::StaticClass()));
			TestTrue(TEXT("UDetectObjectsAsyncAction is a UCancellableAsyncAction"), UDetectObjectsAsyncAction::StaticClass()->IsChildOf(UCancellableAsyncAction::StaticClass()));
		});

		It("FactoryDoesNoWorkAndNeverFailsWithNullInputs", [this]()
		{
			// Factory with a null world context, null pipeline and null texture: returns an action, nothing fires, no warning from the factory itself.
			UClassifyImageAsyncAction* C = UClassifyImageAsyncAction::ClassifyImage(nullptr, nullptr, nullptr, 5);
			UDetectObjectsAsyncAction* D = UDetectObjectsAsyncAction::DetectObjects(nullptr, nullptr, nullptr, 0.5f);
			TestNotNull(TEXT("ClassifyImage(null, null, null, 5)"), C);
			TestNotNull(TEXT("DetectObjects(null, null, null, 0.5)"), D);
			// Never activated: nothing roots them; collect so they do not leak into later tests.
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
