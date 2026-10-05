// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
//
// WP5.5 visible tests (docs/briefs/WP5.5.md, contract C35 vision part): AModelPipelinesSceneClassifierSample and
// AModelPipelinesDetectionSample. Names: Pipelines.Samples.Vision.*
//
// Two families:
//  * "headless" tests (work with and without an RHI): defaults, capture setup, package/wrong-task messages, lifetime, pending slot.
//    In the -nullrhi run the world has no render scene; with an RHI the world has one (a scene capture there renders black).
//    The tests that need the call to FAIL (PreprocessFailed) skip when an RHI exists.
//  * "RHI" tests (Pipelines.Samples.Vision.RHI.*): skipped under -nullrhi with a message, an error when MODELPIPELINES_REQUIRE_RHI=1.
//    A plane with the golden photo (/Engine/BasicShapes/Plane + MID of /Engine/EngineMaterials/EmissiveTexturedMaterial, unlit,
//    texture parameter "Texture") fills the capture's view. No test reads pixels back; only labels / boxes are compared with goldens.
//
// The world is never ticked; every wait is an FTSTicker poll with a timeout and a message. Nothing blocks the game thread on a result.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Algo/Sort.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/StaticMeshComponent.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "HAL/PlatformTime.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
#include "UObject/Class.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#if WITH_EDITOR
#include "ShaderCompiler.h"
#endif
#include "PipelineBase.h"
#include "PipelineModelPackage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesSceneClassifierSample.h"
#include "ModelPipelinesDetectionSample.h"
#include "SampleOverlayWidgets.h"
#include "Engine/EngineTypes.h"
#include "SamplesTestData.h"
#include "SamplesTestReceiver.h"

using namespace UE::ModelPipelines::SamplesTests;

namespace VisionSamplesSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(300);
	constexpr double WaitSeconds = 150.0;
	/** After the awaited result: wait this long to catch a second (wrong) delivery. */
	constexpr double SettleSeconds = 0.6;
	/** Camera distance from the photo plane (cm). */
	constexpr double ViewDistance = 200.0;

	const TCHAR* const ClassSlug = TEXT("resnet-50");
	const TCHAR* const DetSlug = TEXT("rtdetr_r18vd");

	/** The one place where the brief fixes the exact wording (rule 1 of C35). */
	static FString MissingText(const FString& Path, const FString& Slug)
	{
		return FString::Printf(TEXT("Model package not found: %s. Run Plugins/ModelPipelines/Tools/fetch_reference_models.py --only %s, import the folder into /Game/Models, then press Play again. See Docs/Samples.md."), *Path, *Slug);
	}

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

	static bool StateIs(const AModelPipelinesSampleBase* A, EPipelineSampleState S) { return A && A->GetSampleState() == S; }

	static FString StateName(const AModelPipelinesSampleBase* A)
	{
		return A ? UEnum::GetValueAsString(A->GetSampleState()) : FString(TEXT("(null actor)"));
	}

	/** Ignore warning/error lines the engine or the creators log on failure paths (the sample's own text is checked directly). */
	static void IgnoreLogNoise(FAutomationTestBase& T)
	{
		T.AddExpectedMessage(TEXT("."), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
		T.AddExpectedMessage(TEXT("."), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, -1);
	}

	/** Rule 1 requires a Warning from the sample: expect at least one. */
	static void ExpectWarning(FAutomationTestBase& T)
	{
		T.AddExpectedMessage(TEXT("."), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 0);
	}

	/** true = skip this RHI test (message added). With MODELPIPELINES_REQUIRE_RHI=1 a missing RHI is an error instead. */
	static bool SkipWithoutRhi(FAutomationTestBase& T)
	{
		if (HasRhi()) { return false; }
		if (IsRhiRequired()) { T.AddError(TEXT("RHI not available but MODELPIPELINES_REQUIRE_RHI=1 (run without -nullrhi, D3D12)")); }
		else { T.AddInfo(TEXT("RHI not available (-nullrhi): this test was SKIPPED, it did not run")); }
		return true;
	}

	// ---------------------------------------------------------------- generic sample access

	static UClass* ClassOf(bool bDet)
	{
		return bDet ? AModelPipelinesDetectionSample::StaticClass() : AModelPipelinesSceneClassifierSample::StaticClass();
	}

	static void CallNow(AModelPipelinesSampleBase* A)
	{
		if (AModelPipelinesSceneClassifierSample* C = Cast<AModelPipelinesSceneClassifierSample>(A)) { C->ClassifyNow(); }
		else if (AModelPipelinesDetectionSample* D = Cast<AModelPipelinesDetectionSample>(A)) { D->DetectNow(); }
	}

	static UTextureRenderTarget2D* RenderTargetOf(AModelPipelinesSampleBase* A)
	{
		if (AModelPipelinesSceneClassifierSample* C = Cast<AModelPipelinesSceneClassifierSample>(A)) { return C->GetRenderTarget(); }
		if (AModelPipelinesDetectionSample* D = Cast<AModelPipelinesDetectionSample>(A)) { return D->GetRenderTarget(); }
		return nullptr;
	}

	// ---------------------------------------------------------------- fixture

	/** Everything one test owns: world (with a render scene when an RHI exists), packages, receiver, photo plane, pending-call monitor. */
	struct FFix : public TSharedFromThis<FFix>
	{
		FSamplesSceneWorld World;
		TStrongObjectPtr<UPipelineModelPackage> ClassPkg;
		TStrongObjectPtr<UPipelineModelPackage> DetPkg;
		TStrongObjectPtr<USamplesTestReceiver> Receiver;
		TWeakObjectPtr<AModelPipelinesSampleBase> Actor;
		TStrongObjectPtr<UTexture2D> PhotoTexture;
		TStrongObjectPtr<UMaterialInstanceDynamic> PhotoMaterial;
		TWeakObjectPtr<AStaticMeshActor> Plane;

		/** Highest GetNumPendingCalls() of the sample's pipeline seen on any tick (never more than 1 call in flight). */
		int32 MaxPending = 0;
		bool bMonitorSawPipeline = false;
		FTSTicker::FDelegateHandle Monitor;

		bool Setup(FAutomationTestBase& T, bool bClass, bool bDet, EPipelineDevice Device = EPipelineDevice::CPU)
		{
			const TSharedPtr<FJsonObject> Root = GetGoldens(T);
			if (!Root.IsValid()) { return false; }
			for (int32 Pass = 0; Pass < 2; ++Pass)
			{
				const bool bThisDet = (Pass == 1);
				if (!(bThisDet ? bDet : bClass)) { continue; }
				const FString Slug = bThisDet ? DetSlug : ClassSlug;
				const TSharedPtr<FJsonObject> Model = FindVisionModel(Root, Slug);
				if (!Model.IsValid())
				{
					T.AddError(FString::Printf(TEXT("Golden model '%s' not found in reference_models of vision_goldens.json. Regenerate with Tools/make_vision_goldens.py (after Tools/fetch_reference_models.py)."), *Slug));
					return false;
				}
				FString Err;
				UPipelineModelPackage* Pkg = MakeVisionPackage(Root, Model, Device, Err);
				if (!Pkg) { T.AddError(Err); return false; }
				(bThisDet ? DetPkg : ClassPkg).Reset(Pkg);
			}
			FString Err;
			const bool bOk = HasRhi() ? World.CreateWithScene(Err) : World.Create(Err);
			if (!bOk) { T.AddError(Err); return false; }
			Receiver.Reset(NewObject<USamplesTestReceiver>(GetTransientPackage()));
			return true;
		}

		/** Spawns the sample (BeginPlay runs), binds OnSampleResult before BeginPlay, starts the pending-call monitor. bFollowPlayerCamera is forced off. */
		AModelPipelinesSampleBase* Spawn(FAutomationTestBase& T, UClass* SampleClass, const TSoftObjectPtr<UPipelineModelPackage>& Package, bool bOverlay = false,
			TFunction<void(AModelPipelinesSampleBase*)> Pre = nullptr)
		{
			FString Err;
			USamplesTestReceiver* Rcv = Receiver.Get();
			AModelPipelinesSampleBase* A = World.SpawnSample(SampleClass, Package, bOverlay,
				[Rcv, Pre](AModelPipelinesSampleBase* NewActor)
				{
					NewActor->OnSampleResult.AddDynamic(Rcv, &USamplesTestReceiver::OnResult);
					if (AModelPipelinesSceneClassifierSample* C = Cast<AModelPipelinesSceneClassifierSample>(NewActor)) { C->bFollowPlayerCamera = false; }
					if (AModelPipelinesDetectionSample* D = Cast<AModelPipelinesDetectionSample>(NewActor)) { D->bFollowPlayerCamera = false; }
					if (Pre) { Pre(NewActor); }
				}, Err);
			if (!A) { T.AddError(Err); return nullptr; }
			Actor = A;
			TWeakPtr<FFix> Weak = AsShared();
			Monitor = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Weak](float)
			{
				const TSharedPtr<FFix> Self = Weak.Pin();
				if (!Self.IsValid()) { return false; }
				if (UPipelineBase* Pipe = FindSamplePipeline(Self->Actor.Get()))
				{
					Self->bMonitorSawPipeline = true;
					Self->MaxPending = FMath::Max(Self->MaxPending, Pipe->GetNumPendingCalls());
				}
				return true;
			}), 0.0f);
			return A;
		}

		/** Shortcut: the sample for the task with the fixture's own package. */
		AModelPipelinesSampleBase* SpawnFor(FAutomationTestBase& T, bool bDet, bool bOverlay = false, TFunction<void(AModelPipelinesSampleBase*)> Pre = nullptr)
		{
			UPipelineModelPackage* Pkg = bDet ? DetPkg.Get() : ClassPkg.Get();
			return Spawn(T, ClassOf(bDet), TSoftObjectPtr<UPipelineModelPackage>(Pkg), bOverlay, Pre);
		}

		/**
		 * A unit plane with the photo, unlit and emissive, facing -X at the origin (RHI only). false + error when a piece is missing.
		 * Parameter name "Texture" is the one of the texture-sample parameter in EmissiveTexturedMaterial (read from the asset's name table).
		 */
		bool SpawnPhotoPlane(FAutomationTestBase& T, const FVisionTestImage& Image)
		{
			UStaticMesh* Mesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
			UMaterialInterface* Parent = LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/EngineMaterials/EmissiveTexturedMaterial.EmissiveTexturedMaterial"));
			if (!Mesh || !Parent)
			{
				T.AddError(FString::Printf(TEXT("Engine content missing: plane mesh %s, material %s"), Mesh ? TEXT("ok") : TEXT("/Engine/BasicShapes/Plane NOT FOUND"), Parent ? TEXT("ok") : TEXT("/Engine/EngineMaterials/EmissiveTexturedMaterial NOT FOUND")));
				return false;
			}
			UTexture2D* Tex = MakeVisionTexture(Image);
			if (!Tex) { T.AddError(TEXT("MakeVisionTexture returned null although an RHI exists")); return false; }
			PhotoTexture.Reset(Tex);
			UMaterialInstanceDynamic* Mid = UMaterialInstanceDynamic::Create(Parent, GetTransientPackage());
			if (!Mid) { T.AddError(TEXT("UMaterialInstanceDynamic::Create returned null")); return false; }
			PhotoMaterial.Reset(Mid);
			const FName ParamName(TEXT("Texture"));
			if (!Mid->K2_GetTextureParameterValue(ParamName))
			{
				T.AddError(TEXT("EmissiveTexturedMaterial has no texture parameter named 'Texture' (K2_GetTextureParameterValue returned null); the engine material changed, pick another unlit texture material"));
				return false;
			}
			Mid->SetTextureParameterValue(ParamName, Tex);
			if (Mid->K2_GetTextureParameterValue(ParamName) != Tex)
			{
				T.AddError(TEXT("SetTextureParameterValue('Texture') did not stick"));
				return false;
			}
			AStaticMeshActor* P = World.World->SpawnActor<AStaticMeshActor>(FVector::ZeroVector, FRotator::ZeroRotator);
			if (!P || !P->GetStaticMeshComponent()) { T.AddError(TEXT("Spawning the photo plane failed")); return false; }
			UStaticMeshComponent* Comp = P->GetStaticMeshComponent();
			Comp->SetMobility(EComponentMobility::Movable);
			Comp->SetStaticMesh(Mesh);
			Comp->SetMaterial(0, Mid);
			Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
			Comp->SetCastShadow(false);
			Plane = P;
			// The first capture must not see the engine's "shaders compiling" fallback material.
			Mid->EnsureIsComplete();
#if WITH_EDITOR
			if (GShaderCompilingManager) { GShaderCompilingManager->FinishAllCompilation(); }
#endif
			return true;
		}

		void Teardown()
		{
			if (Monitor.IsValid())
			{
				FTSTicker::GetCoreTicker().RemoveTicker(Monitor);
				Monitor.Reset();
			}
			if (AModelPipelinesSampleBase* A = Actor.Get())
			{
				A->OnSampleResult.RemoveAll(Receiver.Get());
			}
			World.Destroy();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		}
	};

	static TSharedPtr<FFix> NewFix(FAutomationTestBase& T, bool bClass, bool bDet, EPipelineDevice Device = EPipelineDevice::CPU)
	{
		TSharedRef<FFix> F = MakeShared<FFix>();
		if (!F->Setup(T, bClass, bDet, Device))
		{
			F->Teardown();
			return nullptr;
		}
		return F;
	}

	// ---------------------------------------------------------------- waiting (never blocks the game thread)

	/** Polls Cond on the core ticker. On timeout: AddError(What) and Done. On success: Next(), which must call Done itself. */
	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Next, double Timeout = WaitSeconds)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("VisionSamplesSpecWait"), 0.02f, [T, Cond, What, Done, Next, Start, Timeout](float)
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
		FTSTicker::GetCoreTicker().AddTicker(TEXT("VisionSamplesSpecSettle"), 0.02f, [Start, Secs, Next](float)
		{
			if ((FPlatformTime::Seconds() - Start) >= Secs)
			{
				Next();
				return false;
			}
			return true;
		});
	}

	static void CheckPending(FAutomationTestBase& T, const FFix& F, const FString& Label)
	{
		T.TestTrue(*FString::Printf(TEXT("%s: the pipeline monitor found the sample's pipeline (otherwise the pending-call check is vacuous)"), *Label), F.bMonitorSawPipeline);
		T.TestTrue(*FString::Printf(TEXT("%s: GetNumPendingCalls() peaked at %d; the sample must never have more than 1 call in flight"), *Label, F.MaxPending), F.MaxPending <= 1);
	}

	// ---------------------------------------------------------------- photo placement (RHI tests)

	struct FFit
	{
		double VisW = 1.0, VisH = 1.0; // size of the capture's view at ViewDistance (cm)
		double FitW = 1.0, FitH = 1.0; // size of the photo (cm), inside the view
	};

	/** The capture's view at ViewDistance (from its real FOV and render target) and the photo size: stretched to fill, or contained with its aspect. */
	static FFit ComputeFit(AModelPipelinesSampleBase* S, const FVisionTestImage& Img, bool bStretch)
	{
		FFit R;
		const USceneCaptureComponent2D* Cap = S ? S->FindComponentByClass<USceneCaptureComponent2D>() : nullptr;
		const UTextureRenderTarget2D* RT = Cap ? Cast<UTextureRenderTarget2D>(Cap->TextureTarget) : nullptr;
		const double Fov = Cap ? (double)Cap->FOVAngle : 90.0;
		const double Aspect = (RT && RT->SizeY > 0) ? (double)RT->SizeX / (double)RT->SizeY : 1.0;
		R.VisW = 2.0 * ViewDistance * FMath::Tan(FMath::DegreesToRadians(Fov * 0.5));
		R.VisH = R.VisW / Aspect;
		if (bStretch)
		{
			R.FitW = R.VisW;
			R.FitH = R.VisH;
		}
		else
		{
			const double ImgAspect = (double)Img.Width / (double)FMath::Max(1, Img.Height);
			if (ImgAspect * R.VisH <= R.VisW) { R.FitH = R.VisH; R.FitW = ImgAspect * R.VisH; }
			else { R.FitW = R.VisW; R.FitH = R.VisW / ImgAspect; }
		}
		return R;
	}

	/**
	 * Camera = the sample actor at (-ViewDistance, 0, 0) looking +X; the plane at the origin faces -X. RollDeg spins the plane about the
	 * view axis; for odd quarter turns its X/Y scale is swapped. The plane's U axis lies along its local X (evidence: the first D3D12 run read
	 * the photo at 90 degrees and not at 0), so odd turns are the upright ones (X = horizontal = FitW), tried first (RollAt); the even turns
	 * (U vertical) stay as a fallback. At the matched roll the board is exactly photo-sized (unstretched); only mirroring is unknown.
	 */
	static void Place(FFix& F, AModelPipelinesSampleBase* S, const FVisionTestImage& Img, bool bStretch, int32 RollDeg)
	{
		const FFit Fit = ComputeFit(S, Img, bStretch);
		S->SetActorLocationAndRotation(FVector(-ViewDistance, 0.0, 0.0), FRotator::ZeroRotator);
		if (AStaticMeshActor* P = F.Plane.Get())
		{
			const bool bOdd = ((RollDeg / 90) % 2) != 0;
			const double ScaleX = (bOdd ? Fit.FitW : Fit.FitH) / 100.0; // the engine plane is 100 x 100 cm
			const double ScaleY = (bOdd ? Fit.FitH : Fit.FitW) / 100.0;
			P->SetActorLocation(FVector::ZeroVector);
			P->SetActorRotation(FQuat(FVector::XAxisVector, FMath::DegreesToRadians((double)RollDeg)) * FRotator(90.0, 0.0, 0.0).Quaternion());
			P->SetActorScale3D(FVector(ScaleX, ScaleY, 1.0));
		}
		// No eye adaptation / bloom: an unlit photo should come out as the photo, not as an exposure experiment.
		if (USceneCaptureComponent2D* Cap = S->FindComponentByClass<USceneCaptureComponent2D>())
		{
			TArray<FEngineShowFlagsSetting> Flags;
			for (const TCHAR* Name : { TEXT("EyeAdaptation"), TEXT("Bloom"), TEXT("MotionBlur"), TEXT("Vignette"), TEXT("LensFlares") })
			{
				FEngineShowFlagsSetting& Flag = Flags.AddDefaulted_GetRef();
				Flag.ShowFlagName = Name;
				Flag.Enabled = false;
			}
			Cap->SetShowFlagSettings(Flags);
		}
		F.World.World->SendAllEndOfFrameUpdates();
	}

	/** Tries the four quarter turns one after the other: Place, call, wait, Check; the first turn where Check passes goes to OnMatch (which calls Done). */
	struct FRollSearch
	{
		FAutomationTestBase* T = nullptr;
		TSharedPtr<FFix> F;
		TWeakObjectPtr<AModelPipelinesSampleBase> A;
		FVisionTestImage Img;
		bool bStretch = false;
		FString Description;
		TFunction<bool(FString& Seen)> Check;
		TFunction<void(int32 RollDeg)> OnMatch;
		FDoneDelegate Done;
		TArray<FString> SeenPerRoll;
	};

	/**
	 * Roll order of the search. The engine plane's U axis lies along its local X (log evidence of the first D3D12 run: the photo matched at 90 degrees
	 * and not at 0), so the photo is upright (and the board fitted, see Place) at the ODD quarter turns; those are tried first, the even ones
	 * stay as fallback in case the engine content changes.
	 */
	static int32 RollAt(int32 Idx)
	{
		static const int32 Order[4] = { 90, 270, 0, 180 };
		return Order[FMath::Clamp(Idx, 0, 3)];
	}

	static void RunRoll(TSharedRef<FRollSearch> S, int32 Idx)
	{
		AModelPipelinesSampleBase* A = S->A.Get();
		if (!A)
		{
			S->T->AddError(TEXT("the sample actor vanished during the roll search"));
			S->Done.ExecuteIfBound();
			return;
		}
		if (Idx >= 4)
		{
			S->T->AddError(FString::Printf(TEXT("%s did not match the golden in any of the 4 plane orientations. Saw: %s"), *S->Description, *FString::Join(S->SeenPerRoll, TEXT(" | "))));
			S->Done.ExecuteIfBound();
			return;
		}
		Place(*S->F, A, S->Img, S->bStretch, RollAt(Idx));
		const int32 Prev = S->F->Receiver->Count();
		CallNow(A);
		WaitFor(S->T, [S, Prev]() { return S->F->Receiver->Count() > Prev && StateIs(S->A.Get(), EPipelineSampleState::Ready); },
			FString::Printf(TEXT("%s: OnSampleResult and state Ready (roll %d)"), *S->Description, RollAt(Idx)), S->Done,
			[S, Idx]()
			{
				const FPipelineResult& R = S->F->Receiver->Results.Last();
				if (!R.bSuccess)
				{
					S->T->AddError(FString::Printf(TEXT("%s: the call failed (error %d): %s"), *S->Description, (int32)R.Error, *R.Message));
					S->Done.ExecuteIfBound();
					return;
				}
				FString Seen;
				if (S->Check(Seen))
				{
					S->T->AddInfo(FString::Printf(TEXT("%s: matched the golden with the plane turned %d degrees (%s)"), *S->Description, RollAt(Idx), *Seen));
					S->OnMatch(RollAt(Idx));
				}
				else
				{
					S->SeenPerRoll.Add(FString::Printf(TEXT("roll %d: %s"), RollAt(Idx), *Seen));
					RunRoll(S, Idx + 1);
				}
			});
	}

	// ---------------------------------------------------------------- golden data

	struct FGoldClass
	{
		FString Label;
		int32 Index = -1;
		FVisionTestImage Image;
	};

	static bool LoadGoldClass(FAutomationTestBase& T, const FString& ImageId, FGoldClass& Out)
	{
		const TSharedPtr<FJsonObject> Root = GetGoldens(T);
		if (!Root.IsValid()) { return false; }
		const TSharedPtr<FJsonObject> Model = FindVisionModel(Root, ClassSlug);
		const TSharedPtr<FJsonObject> Case = FindVisionCase(Model, ImageId);
		if (!Case.IsValid())
		{
			T.AddError(FString::Printf(TEXT("Golden case '%s' of '%s' not found. Regenerate with Tools/make_vision_goldens.py."), *ImageId, ClassSlug));
			return false;
		}
		Out.Label = Case->GetStringField(TEXT("label"));
		Out.Index = (int32)Case->GetNumberField(TEXT("index"));
		FString Err;
		if (!LoadVisionImage(Root, ImageId, Out.Image, Err)) { T.AddError(Err); return false; }
		return true;
	}

	struct FGoldDet
	{
		FString Label;
		double Score = 0.0;
		double Box[4] = { 0, 0, 0, 0 }; // x0, y0, x1, y1 (normalized)
	};

	struct FGoldDetCase
	{
		TArray<FGoldDet> Dets;
		TSet<FString> Labels;
		FVisionTestImage Image;
	};

	static bool LoadGoldDet(FAutomationTestBase& T, const FString& ImageId, FGoldDetCase& Out)
	{
		const TSharedPtr<FJsonObject> Root = GetGoldens(T);
		if (!Root.IsValid()) { return false; }
		const TSharedPtr<FJsonObject> Model = FindVisionModel(Root, DetSlug);
		const TSharedPtr<FJsonObject> Case = FindVisionCase(Model, ImageId);
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (!Case.IsValid() || !Case->TryGetArrayField(TEXT("detections"), Arr) || Arr->Num() == 0)
		{
			T.AddError(FString::Printf(TEXT("Golden case '%s' of '%s' missing or without detections. Regenerate with Tools/make_vision_goldens.py."), *ImageId, DetSlug));
			return false;
		}
		for (const TSharedPtr<FJsonValue>& V : *Arr)
		{
			const TSharedPtr<FJsonObject> O = V->AsObject();
			FGoldDet& D = Out.Dets.AddDefaulted_GetRef();
			D.Label = O->GetStringField(TEXT("label"));
			D.Score = O->GetNumberField(TEXT("score"));
			const TArray<TSharedPtr<FJsonValue>>* B = nullptr;
			if (!O->TryGetArrayField(TEXT("box"), B) || B->Num() != 4) { T.AddError(TEXT("golden detection without a 4-number box")); return false; }
			for (int32 i = 0; i < 4; ++i) { D.Box[i] = (*B)[i]->AsNumber(); }
			Out.Labels.Add(D.Label);
		}
		FString Err;
		if (!LoadVisionImage(Root, ImageId, Out.Image, Err)) { T.AddError(Err); return false; }
		return true;
	}

	static FString DescribeDetections(const TArray<FPipelineDetection>& Dets)
	{
		TArray<FString> Parts;
		for (int32 i = 0; i < Dets.Num() && i < 6; ++i)
		{
			Parts.Add(FString::Printf(TEXT("%s %.2f [%.2f %.2f %.2f %.2f]"), *Dets[i].Label, Dets[i].Score, Dets[i].Box.Min.X, Dets[i].Box.Min.Y, Dets[i].Box.Max.X, Dets[i].Box.Max.Y));
		}
		return Dets.Num() == 0 ? FString(TEXT("(no detections)")) : FString::Join(Parts, TEXT(", "));
	}

	/** Result checks shared by the classifier tests once the golden label matched. */
	static void CheckClassification(FAutomationTestBase& T, const FFix& F, const AModelPipelinesSceneClassifierSample* S, const FGoldClass& Gold)
	{
		const FPipelineClassification C = S->GetLastClassification();
		T.TestTrue(*FString::Printf(TEXT("top-1 label must equal the golden '%s', was '%s' (%.3f)"), *Gold.Label, *C.Label, C.Score), C.Label.Equals(Gold.Label));
		T.TestEqual(TEXT("top-1 class index equals the golden index"), C.Index, Gold.Index);
		T.TestEqual(TEXT("TopK entries == TopK property"), C.TopK.Num(), S->TopK);
		for (int32 i = 0; i < C.TopK.Num(); ++i)
		{
			T.TestTrue(*FString::Printf(TEXT("TopK[%d] score %.4f within 0..1"), i, C.TopK[i].Score), C.TopK[i].Score >= 0.f && C.TopK[i].Score <= 1.f);
			if (i > 0)
			{
				T.TestTrue(*FString::Printf(TEXT("TopK sorted best first: [%d]=%.4f must not exceed [%d]=%.4f"), i, C.TopK[i].Score, i - 1, C.TopK[i - 1].Score), C.TopK[i].Score <= C.TopK[i - 1].Score);
			}
		}
		T.TestTrue(TEXT("OnSampleResult arrived on the game thread"), F.Receiver->bAllOnGameThread);
		T.TestTrue(TEXT("GetLastResult is a success"), S->GetLastResult().bSuccess);
	}

	/**
	 * Position + aspect check of the detected box against the golden box, for the roll the label search is currently trying. Never skipped.
	 * The photo is upright and unstretched on the board (Place fits it that way); what is not known is whether the engine plane's UVs run
	 * mirrored, so the four flips (none, left-right, top-bottom, both; NO u/v swap, a swap would be a stretched photo) are the only
	 * hypotheses. The golden box (normalised in the photo) is mapped through the board into the view; the detected box centre must be within
	 * 0.05 of it for the best flip, and the box's pixel aspect (w/h) must match the golden box's aspect on the board within 25 %, so a capture
	 * that was squashed, or a sample that swapped x/y, fails. Reference box: the first golden detection whose label was detected.
	 */
	static bool CheckDetectionPosition(const AModelPipelinesDetectionSample* Sample, const FGoldDetCase& Gold, FString& Seen)
	{
		const TArray<FPipelineDetection> Dets = Sample->GetLastDetections();
		const FGoldDet* Ref = nullptr;
		const FPipelineDetection* Best = nullptr;
		for (const FGoldDet& G : Gold.Dets)
		{
			for (const FPipelineDetection& D : Dets)
			{
				if (D.Label == G.Label && D.Score >= Sample->ScoreThreshold && (!Best || D.Score > Best->Score)) { Best = &D; }
			}
			if (Best) { Ref = &G; break; }
		}
		if (!Ref || !Best)
		{
			Seen += TEXT(" | no golden label detected: position check FAILED");
			return false;
		}
		const FFit Fit = ComputeFit(const_cast<AModelPipelinesDetectionSample*>(Sample), Gold.Image, false);
		const double RW = Fit.FitW / Fit.VisW;
		const double RH = Fit.FitH / Fit.VisH;
		const FVector2D Got = Best->Box.GetCenter();
		double BestDist = 1e9;
		int32 BestFlip = -1;
		FVector2D BestWant(0, 0);
		for (int32 Flip = 0; Flip < 4; ++Flip)
		{
			double U0 = Ref->Box[0], U1 = Ref->Box[2], V0 = Ref->Box[1], V1 = Ref->Box[3];
			if (Flip & 1) { const double A = 1.0 - U1, B = 1.0 - U0; U0 = A; U1 = B; }
			if (Flip & 2) { const double A = 1.0 - V1, B = 1.0 - V0; V0 = A; V1 = B; }
			const FVector2D Want(0.5 + (0.5 * (U0 + U1) - 0.5) * RW, 0.5 + (0.5 * (V0 + V1) - 0.5) * RH);
			const double Dist = FVector2D::Distance(Got, Want);
			if (Dist < BestDist) { BestDist = Dist; BestFlip = Flip; BestWant = Want; }
		}
		const double GW = FMath::Max(1e-6, Ref->Box[2] - Ref->Box[0]);
		const double GH = FMath::Max(1e-6, Ref->Box[3] - Ref->Box[1]);
		const double WantAspect = (GW * Fit.FitW) / (GH * Fit.FitH);
		const double GotW = FMath::Max(1e-6, (double)Best->Box.GetSize().X);
		const double GotH = FMath::Max(1e-6, (double)Best->Box.GetSize().Y);
		const double GotAspect = (GotW * Fit.VisW) / (GotH * Fit.VisH);
		const double Ratio = GotAspect / WantAspect;
		const bool bPos = BestDist <= 0.05;
		const bool bAspect = Ratio >= 0.75 && Ratio <= 1.25;
		Seen += FString::Printf(TEXT(" | '%s' centre (%.3f, %.3f) vs golden-in-view (%.3f, %.3f) flip %d: distance %.3f (max 0.05)%s; box aspect %.3f vs golden-on-board %.3f, ratio %.2f (0.75..1.25)%s"),
			*Ref->Label, Got.X, Got.Y, BestWant.X, BestWant.Y, BestFlip, BestDist, bPos ? TEXT("") : TEXT(" FAILED"), GotAspect, WantAspect, Ratio, bAspect ? TEXT("") : TEXT(" FAILED"));
		return bPos && bAspect;
	}

	static float Median(TArray<double> V)
	{
		if (V.Num() == 0) { return 0.f; }
		Algo::Sort(V);
		return (float)V[V.Num() / 2];
	}
}

BEGIN_DEFINE_SPEC(FVisionSamplesSpec, "Pipelines.Samples.Vision",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	TSharedPtr<VisionSamplesSpecPrivate::FFix> Fix;
END_DEFINE_SPEC(FVisionSamplesSpec)

void FVisionSamplesSpec::Define()
{
	using namespace VisionSamplesSpecPrivate;

	AfterEach([this]()
	{
		if (Fix.IsValid())
		{
			Fix->Teardown();
			Fix.Reset();
		}
	});

	// ======================================================================================================
	Describe("Fov", [this]()
	{
		using UE::ModelPipelines::Samples::CaptureHorizontalFov;
		auto Near = [this](const TCHAR* What, float Got, float Want)
		{
			TestTrue(*FString::Printf(TEXT("%s: got %.4f deg, expected %.4f deg (tolerance 0.01, off by %.4f)"), What, Got, Want, FMath::Abs(Got - Want)), FMath::IsNearlyEqual(Got, Want, 0.01f));
		};

		It("MaintainYFov16x9", [this, Near]()
		{
			// CaptureHorizontalFov is the LEGACY formula (r.UseLegacyMaintainYFOVViewMatrix): it treats the camera FOV as vertical -> 2*atan(tan 45 * 16/9) = 121.28.
			// The real capture FOV comes from the projection matrix (ULocalPlayer::GetProjectionData); this formula is only the legacy-cvar fallback.
			// Returning 90 from this function is wrong for the legacy case: the capture would show only the middle 56 % and boxes come out 1.78x too big.
			Near(TEXT("90 MaintainYFOV 1920x1080"), CaptureHorizontalFov(90.f, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080)), 121.28f);
		});

		It("MaintainXFov", [this, Near]()
		{
			// X is the constrained axis: the camera FOV is already the horizontal one. Applying the aspect again would give 121.
			Near(TEXT("90 MaintainXFOV 1920x1080"), CaptureHorizontalFov(90.f, AspectRatio_MaintainXFOV, FIntPoint(1920, 1080)), 90.f);
			Near(TEXT("90 MaintainXFOV 1080x1920"), CaptureHorizontalFov(90.f, AspectRatio_MaintainXFOV, FIntPoint(1080, 1920)), 90.f);
		});

		It("MajorAxisFov", [this, Near]()
		{
			// Landscape: the major axis is X -> 90. Portrait: the major axis is Y (FOV vertical) -> HFOV = 2*atan(tan 45 * 1080/1920) = 58.72.
			// A version that treated MajorAxis like MaintainY would give 121.28 on the landscape case.
			Near(TEXT("90 MajorAxisFOV 1920x1080"), CaptureHorizontalFov(90.f, AspectRatio_MajorAxisFOV, FIntPoint(1920, 1080)), 90.f);
			Near(TEXT("90 MajorAxisFOV 1080x1920"), CaptureHorizontalFov(90.f, AspectRatio_MajorAxisFOV, FIntPoint(1080, 1920)), 58.72f);
		});

		It("SquareViewKeepsTheFov", [this, Near]()
		{
			// X == Y: every constraint is the identity (60 stays 60). An aspect factor applied twice, or in the wrong direction, still gives 60 only if X/Y is 1.
			Near(TEXT("60 MaintainYFOV 1000x1000"), CaptureHorizontalFov(60.f, AspectRatio_MaintainYFOV, FIntPoint(1000, 1000)), 60.f);
			Near(TEXT("60 MajorAxisFOV 1000x1000"), CaptureHorizontalFov(60.f, AspectRatio_MajorAxisFOV, FIntPoint(1000, 1000)), 60.f);
			Near(TEXT("60 MaintainXFOV 1000x1000"), CaptureHorizontalFov(60.f, AspectRatio_MaintainXFOV, FIntPoint(1000, 1000)), 60.f);
		});

		It("PortraitMaintainYFov", [this, Near]()
		{
			// Portrait with MaintainY: X/Y = 0.5625 -> 2*atan(0.5625) = 58.72 (narrower than the camera FOV, the factor is X/Y and not Y/X).
			Near(TEXT("90 MaintainYFOV 1080x1920"), CaptureHorizontalFov(90.f, AspectRatio_MaintainYFOV, FIntPoint(1080, 1920)), 58.72f);
		});

		It("ZeroSizeViewReturnsCameraFov", [this, Near]()
		{
			// No viewport yet (size 0): no division by zero, no NaN/inf, the camera FOV comes back. Any zero component counts.
			Near(TEXT("90 MaintainYFOV 0x0"), CaptureHorizontalFov(90.f, AspectRatio_MaintainYFOV, FIntPoint(0, 0)), 90.f);
			Near(TEXT("90 MaintainYFOV 1920x0"), CaptureHorizontalFov(90.f, AspectRatio_MaintainYFOV, FIntPoint(1920, 0)), 90.f);
			Near(TEXT("75 MaintainYFOV 0x1080"), CaptureHorizontalFov(75.f, AspectRatio_MaintainYFOV, FIntPoint(0, 1080)), 75.f);
			Near(TEXT("90 MajorAxisFOV 0x0"), CaptureHorizontalFov(90.f, AspectRatio_MajorAxisFOV, FIntPoint(0, 0)), 90.f);
		});

		It("ClampedTo1And170", [this, Near]()
		{
			// FOV 170 MaintainY on 3840x1080 would be ~177 deg (an invalid projection) -> clamped to 170. A missing clamp returns 177+.
			const float Wide = CaptureHorizontalFov(170.f, AspectRatio_MaintainYFOV, FIntPoint(3840, 1080));
			Near(TEXT("170 MaintainYFOV 3840x1080"), Wide, 170.f);
			TestTrue(*FString::Printf(TEXT("result %.3f is finite"), Wide), FMath::IsFinite(Wide));
			// Tiny FOV: clamped up to 1.
			Near(TEXT("0.2 MaintainXFOV 1920x1080"), CaptureHorizontalFov(0.2f, AspectRatio_MaintainXFOV, FIntPoint(1920, 1080)), 1.f);
		});

		It("MonotonicInAspectForMaintainY", [this]()
		{
			// The wider the view (same height) the wider the horizontal FOV, strictly while below the clamp. A swapped ratio (Y/X) would decrease.
			float Prev = 0.f;
			bool bOk = true;
			FString Seen;
			for (const int32 W : { 540, 810, 1080, 1440, 1920, 2560, 3440 })
			{
				const float H = CaptureHorizontalFov(60.f, AspectRatio_MaintainYFOV, FIntPoint(W, 1080));
				Seen += FString::Printf(TEXT("%d:%.2f "), W, H);
				if (!(H > Prev)) { bOk = false; }
				Prev = H;
			}
			TestTrue(*FString::Printf(TEXT("HFOV strictly increases with width for FOV 60, height 1080 (width:HFOV) %s"), *Seen), bOk);
			TestTrue(TEXT("and stays within [1, 170]"), Prev <= 170.f && Prev >= 1.f);
		});
	});

	// ======================================================================================================
	Describe("FovFallback", [this]()
	{
		using UE::ModelPipelines::Samples::FallbackCaptureFov;
		auto Near = [this](const TCHAR* What, float Got, float Want)
		{
			TestTrue(*FString::Printf(TEXT("%s: got %.4f deg, expected %.4f deg (tolerance 0.01, off by %.4f)"), What, Got, Want, FMath::Abs(Got - Want)), FMath::IsNearlyEqual(Got, Want, 0.01f));
		};
		const float NaN = FMath::Sqrt(-1.f);
		volatile float Big = TNumericLimits<float>::Max(); const float Inf = Big * Big;   // +inf at run time (no Infinity() in TNumericLimits)

		It("ModernCvarOffReturnsCameraFov", [this, Near]()
		{
			// Since 5.7 the view-info FOV is already horizontal: without the legacy cvar the camera FOV must come back unchanged.
			// The old fallback widened 90 on 16:9 to 121.28 (the F-5.5b bug).
			Near(TEXT("90 MaintainYFOV 1920x1080 legacy=false"), FallbackCaptureFov(90.f, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), false), 90.f);
			// The view size and the constraint are irrelevant when the cvar is off: portrait / MajorAxis / MaintainX / 0x0 all give 75.
			Near(TEXT("75 MaintainYFOV 1080x1920 legacy=false"), FallbackCaptureFov(75.f, AspectRatio_MaintainYFOV, FIntPoint(1080, 1920), false), 75.f);
			Near(TEXT("75 MajorAxisFOV 1080x1920 legacy=false"), FallbackCaptureFov(75.f, AspectRatio_MajorAxisFOV, FIntPoint(1080, 1920), false), 75.f);
			Near(TEXT("75 MaintainXFOV 1920x1080 legacy=false"), FallbackCaptureFov(75.f, AspectRatio_MaintainXFOV, FIntPoint(1920, 1080), false), 75.f);
			Near(TEXT("75 MaintainYFOV 0x0 legacy=false"), FallbackCaptureFov(75.f, AspectRatio_MaintainYFOV, FIntPoint(0, 0), false), 75.f);
		});

		It("LegacyCvarOnUsesTheLegacyFormula", [this, Near]()
		{
			// legacy=true delegates to CaptureHorizontalFov: 90 MaintainY 16:9 -> 121.28, portrait -> 58.72, MaintainX -> unchanged.
			// A fallback that ignored the flag would return 90 for the first one.
			Near(TEXT("90 MaintainYFOV 1920x1080 legacy=true"), FallbackCaptureFov(90.f, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), true), 121.28f);
			Near(TEXT("90 MaintainYFOV 1080x1920 legacy=true"), FallbackCaptureFov(90.f, AspectRatio_MaintainYFOV, FIntPoint(1080, 1920), true), 58.72f);
			Near(TEXT("90 MaintainXFOV 1920x1080 legacy=true"), FallbackCaptureFov(90.f, AspectRatio_MaintainXFOV, FIntPoint(1920, 1080), true), 90.f);
			Near(TEXT("90 MajorAxisFOV 1080x1920 legacy=true"), FallbackCaptureFov(90.f, AspectRatio_MajorAxisFOV, FIntPoint(1080, 1920), true), 58.72f);
			// Zero view size with the cvar on: no widening possible, the camera FOV comes back.
			Near(TEXT("90 MaintainYFOV 0x0 legacy=true"), FallbackCaptureFov(90.f, AspectRatio_MaintainYFOV, FIntPoint(0, 0), true), 90.f);
		});

		It("NonFiniteFovFallsBackTo90", [this, Near, NaN, Inf]()
		{
			// NaN or infinity must never reach the capture component: 90 instead. MaintainX keeps the legacy path an identity, so 90 is unambiguous.
			Near(TEXT("NaN MaintainYFOV 1920x1080 legacy=false"), FallbackCaptureFov(NaN, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), false), 90.f);
			Near(TEXT("+inf MaintainYFOV 1920x1080 legacy=false"), FallbackCaptureFov(Inf, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), false), 90.f);
			Near(TEXT("-inf MaintainYFOV 1920x1080 legacy=false"), FallbackCaptureFov(-Inf, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), false), 90.f);
			Near(TEXT("NaN MaintainXFOV 1920x1080 legacy=true"), FallbackCaptureFov(NaN, AspectRatio_MaintainXFOV, FIntPoint(1920, 1080), true), 90.f);
		});

		It("ClampedTo1And170", [this, Near]()
		{
			// 200 -> 170, 0.5 -> 1 (a missing clamp hands the capture an invalid projection); negative and zero go to 1; the bounds themselves stay.
			Near(TEXT("200 MaintainYFOV legacy=false"), FallbackCaptureFov(200.f, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), false), 170.f);
			Near(TEXT("0.5 MaintainYFOV legacy=false"), FallbackCaptureFov(0.5f, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), false), 1.f);
			Near(TEXT("-5 MaintainYFOV legacy=false"), FallbackCaptureFov(-5.f, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), false), 1.f);
			Near(TEXT("0 MaintainYFOV legacy=false"), FallbackCaptureFov(0.f, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), false), 1.f);
			Near(TEXT("170 MaintainYFOV legacy=false"), FallbackCaptureFov(170.f, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), false), 170.f);
			Near(TEXT("1 MaintainYFOV legacy=false"), FallbackCaptureFov(1.f, AspectRatio_MaintainYFOV, FIntPoint(1920, 1080), false), 1.f);
			// The clamp also applies after the legacy widening (170 MaintainY on 3840x1080 is ~177 unclamped).
			Near(TEXT("170 MaintainYFOV 3840x1080 legacy=true"), FallbackCaptureFov(170.f, AspectRatio_MaintainYFOV, FIntPoint(3840, 1080), true), 170.f);
		});
	});

	// ======================================================================================================
	Describe("ViewRect", [this]()
	{
		using UE::ModelPipelines::Samples::CaptureSizeForView;
		using UE::ModelPipelines::Samples::NeedsNewCaptureTarget;
		using UE::ModelPipelines::Samples::MapBoxToOverlay;

		auto SizeIs = [this](const TCHAR* What, FIntPoint Got, FIntPoint Want)
		{
			TestTrue(*FString::Printf(TEXT("%s: got %dx%d, expected %dx%d"), What, Got.X, Got.Y, Want.X, Want.Y), Got == Want);
		};
		auto Needs = [this](const TCHAR* What, FIntPoint Current, const FIntRect& View, bool bWant)
		{
			const bool bGot = NeedsNewCaptureTarget(Current, View);
			TestTrue(*FString::Printf(TEXT("%s: current %dx%d, view rect (%d,%d)-(%d,%d): got %s, expected %s"), What, Current.X, Current.Y,
				View.Min.X, View.Min.Y, View.Max.X, View.Max.Y, bGot ? TEXT("true") : TEXT("false"), bWant ? TEXT("true") : TEXT("false")), bGot == bWant);
		};
		auto BoxIs = [this](const TCHAR* What, const FBox2D& Got, FVector2D WantMin, FVector2D WantMax)
		{
			const bool bOk = FMath::IsNearlyEqual(Got.Min.X, (double)WantMin.X, 0.01) && FMath::IsNearlyEqual(Got.Min.Y, (double)WantMin.Y, 0.01)
				&& FMath::IsNearlyEqual(Got.Max.X, (double)WantMax.X, 0.01) && FMath::IsNearlyEqual(Got.Max.Y, (double)WantMax.Y, 0.01);
			TestTrue(*FString::Printf(TEXT("%s: got (%.3f,%.3f)-(%.3f,%.3f), expected (%.3f,%.3f)-(%.3f,%.3f) (tolerance 0.01)"), What,
				Got.Min.X, Got.Min.Y, Got.Max.X, Got.Max.Y, WantMin.X, WantMin.Y, WantMax.X, WantMax.Y), bOk);
		};

		It("CaptureSizeFollowsTheViewAspect", [this, SizeIs]()
		{
			// Full 16:9 viewport -> 960x540; the split-screen half (960x1080) must give 960x1080: sizing from a fixed 16:9 gives 540 here.
			SizeIs(TEXT("960, rect 0,0-1920,1080"), CaptureSizeForView(960, FIntRect(0, 0, 1920, 1080)), FIntPoint(960, 540));
			SizeIs(TEXT("960, split half 0,0-960,1080"), CaptureSizeForView(960, FIntRect(0, 0, 960, 1080)), FIntPoint(960, 1080));
			// Portrait 1080x1920: 960 / 0.5625 = 1706.67 -> rounds to 1707 (truncation would give 1706).
			SizeIs(TEXT("960, portrait 0,0-1080,1920"), CaptureSizeForView(960, FIntRect(0, 0, 1080, 1920)), FIntPoint(960, 1707));
			// Only the size of the rect counts, not where it sits: an offset rect and a negative-origin rect of 960x540 both give 960x540.
			// A version using Max.X / Max.Y as the size would give 960x(960/(1060/590)) = 960x534.
			SizeIs(TEXT("960, offset rect 100,50-1060,590"), CaptureSizeForView(960, FIntRect(100, 50, 1060, 590)), FIntPoint(960, 540));
			SizeIs(TEXT("960, negative-origin rect -960,-540-0,0"), CaptureSizeForView(960, FIntRect(-960, -540, 0, 0)), FIntPoint(960, 540));
			// The right half of a split screen (Min.X = 960): same size as the left half.
			SizeIs(TEXT("960, right half 960,0-1920,1080"), CaptureSizeForView(960, FIntRect(960, 0, 1920, 1080)), FIntPoint(960, 1080));
		});

		It("CaptureSizeEmptyRectUses16x9", [this, SizeIs]()
		{
			// An empty rect (no viewport yet) must not divide by zero: 16:9. Width or height <= 0 both count, and an inverted rect too.
			SizeIs(TEXT("960, rect 0,0-0,0"), CaptureSizeForView(960, FIntRect(0, 0, 0, 0)), FIntPoint(960, 540));
			SizeIs(TEXT("960, zero width 0,0-0,1080"), CaptureSizeForView(960, FIntRect(0, 0, 0, 1080)), FIntPoint(960, 540));
			SizeIs(TEXT("960, zero height 0,0-1920,0"), CaptureSizeForView(960, FIntRect(0, 0, 1920, 0)), FIntPoint(960, 540));
			SizeIs(TEXT("960, inverted 100,100-50,50"), CaptureSizeForView(960, FIntRect(100, 100, 50, 50)), FIntPoint(960, 540));
		});

		It("CaptureSizeClamps", [this, SizeIs]()
		{
			// Width clamp to [16, 4096]: 10 -> 16 (so 16x9), 0 and negative -> 16, 99999 on a 1x4000 sliver -> 4096 wide and 4096 high (both clamps).
			SizeIs(TEXT("10, rect 0,0-1920,1080"), CaptureSizeForView(10, FIntRect(0, 0, 1920, 1080)), FIntPoint(16, 9));
			SizeIs(TEXT("0, rect 0,0-1920,1080"), CaptureSizeForView(0, FIntRect(0, 0, 1920, 1080)), FIntPoint(16, 9));
			SizeIs(TEXT("-100, rect 0,0-1920,1080"), CaptureSizeForView(-100, FIntRect(0, 0, 1920, 1080)), FIntPoint(16, 9));
			SizeIs(TEXT("99999, rect 0,0-1,4000"), CaptureSizeForView(99999, FIntRect(0, 0, 1, 4000)), FIntPoint(4096, 4096));
			// The upper width bound keeps 16:9 (4096 -> 2304); a clamp that cut Y only would break the aspect.
			SizeIs(TEXT("4096, rect 0,0-1920,1080"), CaptureSizeForView(4096, FIntRect(0, 0, 1920, 1080)), FIntPoint(4096, 2304));
			// Height clamp to >= 1: 16 / (4000/1) = 0.004 -> 0 rounded, must be 1.
			SizeIs(TEXT("16, rect 0,0-4000,1"), CaptureSizeForView(16, FIntRect(0, 0, 4000, 1)), FIntPoint(16, 1));
		});

		It("NeedsNewTargetWorkedExamples", [this, Needs]()
		{
			// Same aspect: keep. Split screen half (aspect 0.89 vs 1.78): recreate. Empty rect: keep what we have. Zero current size: create.
			Needs(TEXT("same aspect"), FIntPoint(960, 540), FIntRect(0, 0, 1920, 1080), false);
			Needs(TEXT("split half"), FIntPoint(960, 540), FIntRect(0, 0, 960, 1080), true);
			Needs(TEXT("empty rect"), FIntPoint(960, 540), FIntRect(0, 0, 0, 0), false);
			Needs(TEXT("zero current"), FIntPoint(0, 0), FIntRect(0, 0, 1920, 1080), true);
		});

		It("NeedsNewTargetOnePercentBoundary", [this, Needs]()
		{
			// Square current target; the view's aspect differs by 0.5 % (keep) or 2 % (recreate), in both directions. A threshold of 0 would recreate
			// on every 1-pixel jitter during a window drag; a threshold of 5 % would keep the 2 % case.
			Needs(TEXT("0.5 % wider"), FIntPoint(1000, 1000), FIntRect(0, 0, 1005, 1000), false);
			Needs(TEXT("0.5 % narrower"), FIntPoint(1000, 1000), FIntRect(0, 0, 995, 1000), false);
			Needs(TEXT("2 % wider"), FIntPoint(1000, 1000), FIntRect(0, 0, 1020, 1000), true);
			Needs(TEXT("2 % narrower"), FIntPoint(1000, 1000), FIntRect(0, 0, 980, 1000), true);
			// Just inside (0.9 %) and just outside (1.1 %) the 1 % limit: 1000x1009 -> view aspect 0.9911 (0.9 % off), 1000x1011 -> 0.9891 (1.1 % off).
			Needs(TEXT("0.9 % taller view"), FIntPoint(1000, 1000), FIntRect(0, 0, 1000, 1009), false);
			Needs(TEXT("1.1 % taller view"), FIntPoint(1000, 1000), FIntRect(0, 0, 1000, 1011), true);
		});

		It("NeedsNewTargetIgnoresRectPositionAndHandlesEdges", [this, Needs]()
		{
			// Only width/height matter: a 960x540 rect at an offset or negative origin matches a 960x540 target (Max/Min mix-ups would flag it).
			Needs(TEXT("offset rect same aspect"), FIntPoint(960, 540), FIntRect(500, 300, 1460, 840), false);
			Needs(TEXT("negative-origin rect same aspect"), FIntPoint(960, 540), FIntRect(-960, -540, 0, 0), false);
			// Portrait target and portrait view match; landscape target vs portrait view does not.
			Needs(TEXT("portrait/portrait"), FIntPoint(540, 960), FIntRect(0, 0, 1080, 1920), false);
			Needs(TEXT("landscape target, portrait view"), FIntPoint(960, 540), FIntRect(0, 0, 1080, 1920), true);
			// A zero in either component of the current size forces a new target for a non-empty rect.
			Needs(TEXT("current 960x0"), FIntPoint(960, 0), FIntRect(0, 0, 1920, 1080), true);
			Needs(TEXT("current 0x540"), FIntPoint(0, 540), FIntRect(0, 0, 1920, 1080), true);
			// Empty rects keep the target, even with a zero current size, zero width/height, or an inverted rect.
			Needs(TEXT("empty rect, zero current"), FIntPoint(0, 0), FIntRect(0, 0, 0, 0), false);
			Needs(TEXT("zero-width rect"), FIntPoint(960, 540), FIntRect(0, 0, 0, 1080), false);
			Needs(TEXT("inverted rect"), FIntPoint(960, 540), FIntRect(100, 100, 50, 50), false);
		});

		It("MapBoxWorkedExamples", [this, BoxIs]()
		{
			// Right half of the viewport: the full capture box covers the right half of the widget (640..1280 of 1280), full height.
			BoxIs(TEXT("full box, right half"), MapBoxToOverlay(FBox2D(FVector2D(0, 0), FVector2D(1, 1)), FIntRect(960, 0, 1920, 1080), FIntPoint(1920, 1080), FVector2D(1280, 720)),
				FVector2D(640, 0), FVector2D(1280, 720));
			// Empty rect: the box maps onto the whole widget ((0.25,0.5)-(0.5,1) of 100x50).
			BoxIs(TEXT("empty rect"), MapBoxToOverlay(FBox2D(FVector2D(0.25, 0.5), FVector2D(0.5, 1)), FIntRect(0, 0, 0, 0), FIntPoint(1920, 1080), FVector2D(100, 50)),
				FVector2D(25, 25), FVector2D(50, 50));
			// Box outside 0..1 is clamped to the view rect: {-1,-1}-{2,2} on the right half gives exactly the right half, not 960 px beyond it.
			BoxIs(TEXT("oversize box clamped"), MapBoxToOverlay(FBox2D(FVector2D(-1, -1), FVector2D(2, 2)), FIntRect(960, 0, 1920, 1080), FIntPoint(1920, 1080), FVector2D(1280, 720)),
				FVector2D(640, 0), FVector2D(1280, 720));
			// Viewport with a zero component: whole widget.
			BoxIs(TEXT("viewport 0x0"), MapBoxToOverlay(FBox2D(FVector2D(0.25, 0.5), FVector2D(0.5, 1)), FIntRect(960, 0, 1920, 1080), FIntPoint(0, 0), FVector2D(100, 50)),
				FVector2D(25, 25), FVector2D(50, 50));
		});

		It("MapBoxOffsetRectAndEdges", [this, BoxIs]()
		{
			// A 200x100 view rect at (100,50) in a 400x200 viewport shown by an 800x400 widget (widget scale 2, differs from the viewport):
			// full box -> (100/400*800, 50/200*400)-(300/400*800, 150/200*400) = (200,100)-(600,300). Ignoring Min.X/Y would give (0,0)-(400,200).
			BoxIs(TEXT("offset rect, full box"), MapBoxToOverlay(FBox2D(FVector2D(0, 0), FVector2D(1, 1)), FIntRect(100, 50, 300, 150), FIntPoint(400, 200), FVector2D(800, 400)),
				FVector2D(200, 100), FVector2D(600, 300));
			// Inner box (0.25,0.5)-(0.75,1): x = (100+50)/400*800 = 300 .. (100+150)/400*800 = 500; y = (50+50)/200*400 = 200 .. 300.
			BoxIs(TEXT("offset rect, inner box"), MapBoxToOverlay(FBox2D(FVector2D(0.25, 0.5), FVector2D(0.75, 1)), FIntRect(100, 50, 300, 150), FIntPoint(400, 200), FVector2D(800, 400)),
				FVector2D(300, 200), FVector2D(500, 300));
			// Partly outside: (-0.5,0.2)-(0.5,1.5) clamps to (0,0.2)-(0.5,1) on the whole 1920x1080 view in a 192x108 widget -> (0,21.6)-(96,108).
			BoxIs(TEXT("partly outside, whole viewport"), MapBoxToOverlay(FBox2D(FVector2D(-0.5, 0.2), FVector2D(0.5, 1.5)), FIntRect(0, 0, 1920, 1080), FIntPoint(1920, 1080), FVector2D(192, 108)),
				FVector2D(0, 21.6), FVector2D(96, 108));
			// Left half of a split screen (Min = 0): full box -> left half of the widget, full height.
			BoxIs(TEXT("left half"), MapBoxToOverlay(FBox2D(FVector2D(0, 0), FVector2D(1, 1)), FIntRect(0, 0, 960, 1080), FIntPoint(1920, 1080), FVector2D(1280, 720)),
				FVector2D(0, 0), FVector2D(640, 720));
			// Zero height viewport, or zero-height rect: whole widget as well.
			BoxIs(TEXT("viewport 1920x0"), MapBoxToOverlay(FBox2D(FVector2D(0.25, 0.5), FVector2D(0.5, 1)), FIntRect(960, 0, 1920, 1080), FIntPoint(1920, 0), FVector2D(100, 50)),
				FVector2D(25, 25), FVector2D(50, 50));
			BoxIs(TEXT("zero-height rect"), MapBoxToOverlay(FBox2D(FVector2D(0.25, 0.5), FVector2D(0.5, 1)), FIntRect(0, 0, 1920, 0), FIntPoint(1920, 1080), FVector2D(100, 50)),
				FVector2D(25, 25), FVector2D(50, 50));
		});
	});

	// ======================================================================================================
	Describe("Defaults", [this]()
	{
		It("Classifier", [this]()
		{
			// Brief C35: default package /Game/Models/resnet-50..., 512 square capture, 1 s interval, TopK 3, follows the player camera.
			// A wrong slug would send every user to import the wrong model.
			const AModelPipelinesSceneClassifierSample* CDO = GetDefault<AModelPipelinesSceneClassifierSample>();
			TestEqual(TEXT("default Package"), CDO->Package.ToSoftObjectPath().ToString(), FString(TEXT("/Game/Models/resnet-50.resnet-50")));
			TestEqual(TEXT("default CaptureResolution"), CDO->CaptureResolution, 512);
			TestEqual(TEXT("default IntervalSeconds"), CDO->IntervalSeconds, 1.0f);
			TestEqual(TEXT("default TopK"), CDO->TopK, 3);
			TestTrue(TEXT("default bFollowPlayerCamera"), CDO->bFollowPlayerCamera);
			TestTrue(TEXT("default bShowOverlay"), CDO->bShowOverlay);
		});

		It("Detection", [this]()
		{
			const AModelPipelinesDetectionSample* CDO = GetDefault<AModelPipelinesDetectionSample>();
			TestEqual(TEXT("default Package"), CDO->Package.ToSoftObjectPath().ToString(), FString(TEXT("/Game/Models/rtdetr_r18vd.rtdetr_r18vd")));
			TestEqual(TEXT("default CaptureWidth"), CDO->CaptureWidth, 960);
			TestEqual(TEXT("default ScoreThreshold"), CDO->ScoreThreshold, 0.5f);
			TestEqual(TEXT("default IntervalSeconds"), CDO->IntervalSeconds, 0.5f);
			TestTrue(TEXT("default bFollowPlayerCamera"), CDO->bFollowPlayerCamera);
			TestTrue(TEXT("default bShowOverlay"), CDO->bShowOverlay);
		});
	});

	// ======================================================================================================
	Describe("Capture", [this]()
	{
		auto Body = [this](const FDoneDelegate& Done, bool bDet, FIntPoint WantSize, TFunction<void(AModelPipelinesSampleBase*)> Pre, const TCHAR* Label)
		{
			// Capture setup contract: a USceneCaptureComponent2D whose TextureTarget is the render target returned by GetRenderTarget (RGBA8, wanted size),
			// bCaptureEveryFrame/bCaptureOnMovement off (the sample captures on demand), final-color LDR.
			// A sample that left bCaptureEveryFrame on would render the scene twice per frame for nothing.
			Fix = NewFix(*this, !bDet, bDet);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSampleBase* A = F->SpawnFor(*this, bDet, false, Pre);
			if (!A) { Done.ExecuteIfBound(); return; }
			const FString L(Label);
			WaitFor(this, [A]() { return StateIs(A, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, A, WantSize, L, Done]()
			{
				const USceneCaptureComponent2D* Cap = A->FindComponentByClass<USceneCaptureComponent2D>();
				if (!TestNotNull(*(L + TEXT(": the sample owns a USceneCaptureComponent2D")), Cap)) { Done.ExecuteIfBound(); return; }
				UTextureRenderTarget2D* RT = RenderTargetOf(A);
				if (!TestNotNull(*(L + TEXT(": GetRenderTarget() is not null")), RT)) { Done.ExecuteIfBound(); return; }
				TestTrue(*(L + TEXT(": the component's TextureTarget is GetRenderTarget()")), Cap->TextureTarget == RT);
				TestEqual(*(L + TEXT(": render target width")), (int32)RT->SizeX, WantSize.X);
				TestEqual(*(L + TEXT(": render target height")), (int32)RT->SizeY, WantSize.Y);
				TestTrue(*(L + TEXT(": render target format is RGBA8")), RT->RenderTargetFormat == RTF_RGBA8);
				TestFalse(*(L + TEXT(": bCaptureEveryFrame is false")), Cap->bCaptureEveryFrame);
				TestFalse(*(L + TEXT(": bCaptureOnMovement is false")), Cap->bCaptureOnMovement);
				TestTrue(*(L + TEXT(": CaptureSource is SCS_FinalColorLDR")), Cap->CaptureSource == SCS_FinalColorLDR);
				Done.ExecuteIfBound();
			});
		};

		LatentIt("ClassifierRenderTarget512", LatentLimit, [this, Body](const FDoneDelegate& Done)
		{
			Body(Done, false, FIntPoint(512, 512), nullptr, TEXT("classifier default"));
		});

		LatentIt("ClassifierRenderTargetFollowsCaptureResolution", LatentLimit, [this, Body](const FDoneDelegate& Done)
		{
			// CaptureResolution = 256 must give a 256 x 256 target (square), proving the property is used and not a constant 512.
			Body(Done, false, FIntPoint(256, 256), [](AModelPipelinesSampleBase* A) { Cast<AModelPipelinesSceneClassifierSample>(A)->CaptureResolution = 256; }, TEXT("classifier 256"));
		});

		LatentIt("DetectionRenderTarget960x540WithoutViewport", LatentLimit, [this, Body](const FDoneDelegate& Done)
		{
			// No game viewport in a test world -> the brief's default aspect: 960 x 540.
			Body(Done, true, FIntPoint(960, 540), nullptr, TEXT("detection default"));
		});

		LatentIt("DetectionRenderTargetFollowsCaptureWidth", LatentLimit, [this, Body](const FDoneDelegate& Done)
		{
			// CaptureWidth = 640 -> height = 640 / (960/540) = 360. A sample that hard-codes 540 for the height ignores the aspect.
			Body(Done, true, FIntPoint(640, 360), [](AModelPipelinesSampleBase* A) { Cast<AModelPipelinesDetectionSample>(A)->CaptureWidth = 640; }, TEXT("detection 640"));
		});

		LatentIt("DetectionCaptureFovIsHorizontalFovWithoutViewport", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// No viewport and no local player in the test world: the sample's own default is the camera FOV 90 on a view of unknown size, so the capture's
			// FOVAngle must be 90 (CaptureHorizontalFov / the fallback with a 0x0 view returns the camera FOV; the capture component's FOV is horizontal).
			// Guards F1: a sample that applied the legacy 960x540 aspect widening (121.28) to a camera it does not have fails here.
			Fix = NewFix(*this, false, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSampleBase* A = F->SpawnFor(*this, true);
			if (!A) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [A]() { return StateIs(A, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, A, Done]()
			{
				const USceneCaptureComponent2D* Cap = A->FindComponentByClass<USceneCaptureComponent2D>();
				if (TestNotNull(TEXT("the sample owns a USceneCaptureComponent2D"), Cap))
				{
					const float Want = UE::ModelPipelines::Samples::CaptureHorizontalFov(90.f, AspectRatio_MaintainYFOV, FIntPoint(0, 0));
					TestTrue(*FString::Printf(TEXT("capture FOVAngle %.3f must equal CaptureHorizontalFov(90, MaintainYFOV, no viewport) = %.3f"), Cap->FOVAngle, Want), FMath::IsNearlyEqual(Cap->FOVAngle, Want, 0.01f));
				}
				Done.ExecuteIfBound();
			});
		});

		LatentIt("GettersBeforeAnyCall", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Fresh samples report empty results, not garbage and not a crash.
			Fix = NewFix(*this, true, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSceneClassifierSample* C = Cast<AModelPipelinesSceneClassifierSample>(F->SpawnFor(*this, false));
			if (!C) { Done.ExecuteIfBound(); return; }
			AModelPipelinesDetectionSample* D = Cast<AModelPipelinesDetectionSample>(F->Spawn(*this, AModelPipelinesDetectionSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->DetPkg.Get())));
			if (!D) { Done.ExecuteIfBound(); return; }
			TestTrue(TEXT("classifier: GetLastClassification() is empty before any call"), C->GetLastClassification().Label.IsEmpty() && C->GetLastClassification().TopK.Num() == 0);
			TestEqual(TEXT("detection: GetLastDetections() is empty before any call"), D->GetLastDetections().Num(), 0);
			TestEqual(TEXT("no OnSampleResult yet"), F->Receiver->Count(), 0);
			Done.ExecuteIfBound();
		});
	});

	// ======================================================================================================
	Describe("NoRhi", [this]()
	{
		auto Body = [this](const FDoneDelegate& Done, bool bDet)
		{
			// Without an RHI a texture call fails with PreprocessFailed: OnSampleResult fires once with that error, state goes back to Ready,
			// the last result/detections stay empty and nothing crashes (no scene exists, so CaptureScene is a no-op).
			// A sample that swallowed the failure would stay Busy forever; one that fired twice would report two results.
			if (HasRhi())
			{
				AddInfo(TEXT("an RHI is present: the 'texture call fails with PreprocessFailed' case does not apply, SKIPPED (run with -nullrhi)"));
				Done.ExecuteIfBound();
				return;
			}
			IgnoreLogNoise(*this);
			Fix = NewFix(*this, !bDet, bDet);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSampleBase* A = F->SpawnFor(*this, bDet);
			if (!A) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [A]() { return StateIs(A, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, A, bDet, Done]()
			{
				CallNow(A);
				TestTrue(*FString::Printf(TEXT("state right after the call is Busy (callbacks never run inside the call), was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::Busy));
				WaitFor(this, [F]() { return F->Receiver->Count() >= 1; }, TEXT("OnSampleResult"), Done, [this, F, A, bDet, Done]()
				{
					Settle(SettleSeconds, [this, F, A, bDet, Done]()
					{
						TestEqual(TEXT("OnSampleResult fired exactly once"), F->Receiver->Count(), 1);
						const FPipelineResult R = F->Receiver->Results[0];
						TestFalse(TEXT("the call failed (no RHI)"), R.bSuccess);
						TestTrue(*FString::Printf(TEXT("error is PreprocessFailed, was %d: %s"), (int32)R.Error, *R.Message), R.Error == EPipelineError::PreprocessFailed);
						TestTrue(TEXT("GetLastResult carries the same failure"), !A->GetLastResult().bSuccess && A->GetLastResult().Error == EPipelineError::PreprocessFailed);
						TestTrue(*FString::Printf(TEXT("state is back to Ready, was %s (status: %s)"), *StateName(A), *A->GetStatusText()), StateIs(A, EPipelineSampleState::Ready));
						TestTrue(TEXT("OnSampleResult arrived on the game thread"), F->Receiver->bAllOnGameThread);
						if (bDet) { TestEqual(TEXT("no detections after a failed call"), Cast<AModelPipelinesDetectionSample>(A)->GetLastDetections().Num(), 0); }
						else { TestEqual(TEXT("no classification after a failed call"), Cast<AModelPipelinesSceneClassifierSample>(A)->GetLastClassification().TopK.Num(), 0); }
						CheckPending(*this, *F, bDet ? TEXT("detection") : TEXT("classifier"));
						Done.ExecuteIfBound();
					});
				});
			});
		};

		LatentIt("ClassifyNowPreprocessFailed", LatentLimit, [this, Body](const FDoneDelegate& Done) { Body(Done, false); });
		LatentIt("DetectNowPreprocessFailed", LatentLimit, [this, Body](const FDoneDelegate& Done) { Body(Done, true); });

		auto EarlyBody = [this](const FDoneDelegate& Done, bool bDet)
		{
			// A call right after BeginPlay (state Loading, pipeline not created yet) must be harmless: no crash, the sample ends Ready (not
			// stuck Loading/Busy) and never reports more than the one result.
			IgnoreLogNoise(*this);
			Fix = NewFix(*this, !bDet, bDet);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSampleBase* A = F->SpawnFor(*this, bDet);
			if (!A) { Done.ExecuteIfBound(); return; }
			CallNow(A);
			WaitFor(this, [A]() { return StateIs(A, EPipelineSampleState::Ready); }, TEXT("state Ready after an early call"), Done, [this, F, A, Done]()
			{
				Settle(1.5, [this, F, A, Done]()
				{
					TestTrue(*FString::Printf(TEXT("at most one OnSampleResult for the one early call, was %d"), F->Receiver->Count()), F->Receiver->Count() <= 1);
					TestTrue(*FString::Printf(TEXT("state settled at Ready, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::Ready));
					CheckPending(*this, *F, TEXT("early call"));
					Done.ExecuteIfBound();
				});
			});
		};
		LatentIt("ClassifyNowBeforeReady", LatentLimit, [this, EarlyBody](const FDoneDelegate& Done) { EarlyBody(Done, false); });
		LatentIt("DetectNowBeforeReady", LatentLimit, [this, EarlyBody](const FDoneDelegate& Done) { EarlyBody(Done, true); });
	});

	// ======================================================================================================
	Describe("Package", [this]()
	{
		// Spawn with the given soft package, wait for the state to leave Loading, run Check, then call Now once more and confirm nothing ran.
		auto RunBad = [this](const FDoneDelegate& Done, bool bDet, const FSoftObjectPath* Path, bool bWrongTask, TFunction<void(AModelPipelinesSampleBase*)> Check)
		{
			// Wrong task: the OTHER task's package goes onto this sample.
			Fix = NewFix(*this, bWrongTask && bDet, bWrongTask && !bDet);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			TSoftObjectPtr<UPipelineModelPackage> Soft;
			if (Path) { Soft = TSoftObjectPtr<UPipelineModelPackage>(*Path); }
			else if (bWrongTask) { Soft = TSoftObjectPtr<UPipelineModelPackage>(bDet ? F->ClassPkg.Get() : F->DetPkg.Get()); }
			AModelPipelinesSampleBase* A = F->Spawn(*this, ClassOf(bDet), Soft);
			if (!A) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [A]() { return !StateIs(A, EPipelineSampleState::Loading); }, TEXT("state to leave Loading"), Done, [this, F, A, Check, Done]()
			{
				Settle(SettleSeconds, [this, F, A, Check, Done]()
				{
					Check(A);
					TestEqual(TEXT("no OnSampleResult: no call was made"), F->Receiver->Count(), 0);
					TestTrue(TEXT("no pipeline was created"), FindSamplePipeline(A) == nullptr);
					// A request on a sample without a pipeline must be a harmless no-op.
					CallNow(A);
					Settle(SettleSeconds, [this, F, A, Done]()
					{
						TestEqual(TEXT("still no OnSampleResult after a request on the unusable sample"), F->Receiver->Count(), 0);
						TestFalse(TEXT("state is not Busy after a request on the unusable sample"), StateIs(A, EPipelineSampleState::Busy));
						Done.ExecuteIfBound();
					});
				});
			});
		};

		LatentIt("MissingPackageClassifier", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			// Path that does not resolve -> MissingPackage + the exact text with slug resnet-50, a Warning, no call.
			const FSoftObjectPath Path(TEXT("/Game/Models/does-not-exist.does-not-exist"));
			ExpectWarning(*this);
			RunBad(Done, false, &Path, false, [this](AModelPipelinesSampleBase* A)
			{
				TestTrue(*FString::Printf(TEXT("state is MissingPackage, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::MissingPackage));
				TestEqual(TEXT("status text"), A->GetStatusText(), MissingText(TEXT("/Game/Models/does-not-exist.does-not-exist"), ClassSlug));
			});
		});

		LatentIt("MissingPackageDetection", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			const FSoftObjectPath Path(TEXT("/Game/Models/does-not-exist.does-not-exist"));
			ExpectWarning(*this);
			RunBad(Done, true, &Path, false, [this](AModelPipelinesSampleBase* A)
			{
				TestTrue(*FString::Printf(TEXT("state is MissingPackage, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::MissingPackage));
				TestEqual(TEXT("status text"), A->GetStatusText(), MissingText(TEXT("/Game/Models/does-not-exist.does-not-exist"), DetSlug));
			});
		});

		LatentIt("UnsetPackageClassifier", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			// Package = null -> same message with the path "(none)".
			ExpectWarning(*this);
			RunBad(Done, false, nullptr, false, [this](AModelPipelinesSampleBase* A)
			{
				TestTrue(*FString::Printf(TEXT("state is MissingPackage, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::MissingPackage));
				TestEqual(TEXT("status text"), A->GetStatusText(), MissingText(TEXT("(none)"), ClassSlug));
			});
		});

		LatentIt("UnsetPackageDetection", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			ExpectWarning(*this);
			RunBad(Done, true, nullptr, false, [this](AModelPipelinesSampleBase* A)
			{
				TestTrue(*FString::Printf(TEXT("state is MissingPackage, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::MissingPackage));
				TestEqual(TEXT("status text"), A->GetStatusText(), MissingText(TEXT("(none)"), DetSlug));
			});
		});

		LatentIt("WrongTaskClassifier", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			// An object-detection package on the classifier sample: the creator rejects it -> Failed, "Could not create the pipeline: <message>".
			IgnoreLogNoise(*this);
			RunBad(Done, false, nullptr, true, [this](AModelPipelinesSampleBase* A)
			{
				const FString Prefix(TEXT("Could not create the pipeline: "));
				TestTrue(*FString::Printf(TEXT("state is Failed, was %s (status: %s)"), *StateName(A), *A->GetStatusText()), StateIs(A, EPipelineSampleState::Failed));
				TestTrue(*FString::Printf(TEXT("status starts with '%s', was '%s'"), *Prefix, *A->GetStatusText()), A->GetStatusText().StartsWith(Prefix));
				TestTrue(TEXT("the status carries a non-empty creator message"), A->GetStatusText().Len() > Prefix.Len());
			});
		});

		LatentIt("WrongTaskDetection", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			// An image-classification package on the detection sample.
			IgnoreLogNoise(*this);
			RunBad(Done, true, nullptr, true, [this](AModelPipelinesSampleBase* A)
			{
				const FString Prefix(TEXT("Could not create the pipeline: "));
				TestTrue(*FString::Printf(TEXT("state is Failed, was %s (status: %s)"), *StateName(A), *A->GetStatusText()), StateIs(A, EPipelineSampleState::Failed));
				TestTrue(*FString::Printf(TEXT("status starts with '%s', was '%s'"), *Prefix, *A->GetStatusText()), A->GetStatusText().StartsWith(Prefix));
			});
		});
	});

	// ======================================================================================================
	Describe("Lifetime", [this]()
	{
		auto DestroyBody = [this](const FDoneDelegate& Done, bool bDet)
		{
			// Call, then destroy the actor in the same frame (the call is in flight), force a GC and let time pass: no crash, no OnSampleResult
			// afterwards (rule 4: weak this). A callback that kept a raw this pointer would crash or fire here.
			IgnoreLogNoise(*this);
			Fix = NewFix(*this, !bDet, bDet);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSampleBase* A = F->SpawnFor(*this, bDet);
			if (!A) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [A]() { return StateIs(A, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, A, Done]()
			{
				CallNow(A);
				TestTrue(*FString::Printf(TEXT("a call is in flight (state Busy) before the actor is destroyed, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::Busy));
				A->Destroy();
				Settle(0.05, [this, F, Done]()
				{
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					Settle(4.0, [this, F, Done]()
					{
						TestEqual(TEXT("no OnSampleResult after the actor was destroyed"), F->Receiver->Count(), 0);
						TestFalse(TEXT("the actor is gone"), F->Actor.IsValid());
						Done.ExecuteIfBound();
					});
				});
			});
		};
		LatentIt("ClassifierDestroyedWhileInFlight", LatentLimit, [this, DestroyBody](const FDoneDelegate& Done) { DestroyBody(Done, false); });
		LatentIt("DetectionDestroyedWhileInFlight", LatentLimit, [this, DestroyBody](const FDoneDelegate& Done) { DestroyBody(Done, true); });

		auto QuickBody = [this](const FDoneDelegate& Done, bool bDet)
		{
			// Two calls in the same frame: the first runs, the second waits in the single pending slot (latest wins, never dropped, never stacked).
			// Exactly two results arrive, the pipeline never has 2 calls in flight, the sample ends Ready.
			IgnoreLogNoise(*this);
			Fix = NewFix(*this, !bDet, bDet);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSampleBase* A = F->SpawnFor(*this, bDet);
			if (!A) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [A]() { return StateIs(A, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, A, Done]()
			{
				CallNow(A);
				CallNow(A);
				TestTrue(*FString::Printf(TEXT("state is Busy with a request pending, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::Busy));
				WaitFor(this, [F, A]() { return F->Receiver->Count() >= 2 && StateIs(A, EPipelineSampleState::Ready); }, TEXT("two results and state Ready"), Done, [this, F, A, Done]()
				{
					Settle(1.5, [this, F, A, Done]()
					{
						TestEqual(TEXT("exactly two OnSampleResult: the running call and the one served from the pending slot"), F->Receiver->Count(), 2);
						TestTrue(*FString::Printf(TEXT("state settled at Ready, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::Ready));
						CheckPending(*this, *F, TEXT("two quick calls"));
						Done.ExecuteIfBound();
					});
				});
			});
		};
		LatentIt("TwoQuickDetectNow", LatentLimit, [this, QuickBody](const FDoneDelegate& Done) { QuickBody(Done, true); });
		LatentIt("TwoQuickClassifyNow", LatentLimit, [this, QuickBody](const FDoneDelegate& Done) { QuickBody(Done, false); });

		LatentIt("DestroyedWhileLoading", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Destroy in the same frame as BeginPlay (async package load pending): the load completion must not touch the dead actor.
			Fix = NewFix(*this, false, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSampleBase* A = F->SpawnFor(*this, true);
			if (!A) { Done.ExecuteIfBound(); return; }
			A->Destroy();
			Settle(0.05, [this, F, Done]()
			{
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
				Settle(2.0, [this, F, Done]()
				{
					TestEqual(TEXT("no OnSampleResult"), F->Receiver->Count(), 0);
					Done.ExecuteIfBound();
				});
			});
		});
	});

	// ======================================================================================================
	Describe("Overlay", [this]()
	{
		auto Body = [this](const FDoneDelegate& Done, bool bDet)
		{
			// bShowOverlay = true in a world without a game viewport: no crash, no overlay, and the sample still works (reaches Ready, one call
			// delivers one result).
			IgnoreLogNoise(*this);
			Fix = NewFix(*this, !bDet, bDet);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSampleBase* A = F->SpawnFor(*this, bDet, /*bOverlay*/ true);
			if (!A) { Done.ExecuteIfBound(); return; }
			TestTrue(TEXT("bShowOverlay stayed true"), A->bShowOverlay);
			WaitFor(this, [A]() { return StateIs(A, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, A, Done]()
			{
				CallNow(A);
				WaitFor(this, [F]() { return F->Receiver->Count() >= 1; }, TEXT("OnSampleResult"), Done, [this, F, A, Done]()
				{
					TestTrue(*FString::Printf(TEXT("a result arrived without an overlay; state %s"), *StateName(A)), true);
					Done.ExecuteIfBound();
				});
			});
		};
		LatentIt("ClassifierShowOverlayWithoutViewport", LatentLimit, [this, Body](const FDoneDelegate& Done) { Body(Done, false); });
		LatentIt("DetectionShowOverlayWithoutViewport", LatentLimit, [this, Body](const FDoneDelegate& Done) { Body(Done, true); });
	});

	// ======================================================================================================
	Describe("RHI", [this]()
	{
		/** Classifier looking at a golden photo: top-1 equals the golden label. */
		auto ClassifierCase = [this](const FDoneDelegate& Done, const FString& ImageId)
		{
			// The photo fills the 512 x 512 capture (stretched), unlit and emissive; ClassifyNow() captures it and runs resnet-50 on the
			// capture. The golden label comes from the Python reference on the original photo (top margin > 0.9 for the chosen photos).
			// A capture that never rendered (black frame, wrong target, capture after the readback) gives a random label here;
			// a sample that ignored TopK would not return 3 sorted entries.
			if (SkipWithoutRhi(*this)) { Done.ExecuteIfBound(); return; }
			FGoldClass Gold;
			if (!LoadGoldClass(*this, ImageId, Gold)) { Done.ExecuteIfBound(); return; }
			Fix = NewFix(*this, true, false, IsGpuRequired() ? EPipelineDevice::GPU : EPipelineDevice::Auto);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			if (!F->SpawnPhotoPlane(*this, Gold.Image)) { Done.ExecuteIfBound(); return; }
			AModelPipelinesSceneClassifierSample* S = Cast<AModelPipelinesSceneClassifierSample>(F->SpawnFor(*this, false));
			if (!S) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, S, Gold, ImageId, Done]()
			{
				const TSharedRef<FRollSearch> Search = MakeShared<FRollSearch>();
				Search->T = this;
				Search->F = F;
				Search->A = S;
				Search->Img = Gold.Image;
				Search->bStretch = true;
				Search->Description = FString::Printf(TEXT("classifier on '%s' (golden '%s')"), *ImageId, *Gold.Label);
				Search->Done = Done;
				const TWeakObjectPtr<AModelPipelinesSceneClassifierSample> WeakS = S;
				Search->Check = [WeakS, Gold](FString& Seen)
				{
					const AModelPipelinesSceneClassifierSample* Sample = WeakS.Get();
					if (!Sample) { Seen = TEXT("(sample gone)"); return false; }
					const FPipelineClassification C = Sample->GetLastClassification();
					Seen = FString::Printf(TEXT("top-1 '%s' %.3f"), *C.Label, C.Score);
					return C.Label.Equals(Gold.Label);
				};
				Search->OnMatch = [this, F, WeakS, Gold, Done](int32)
				{
					if (const AModelPipelinesSceneClassifierSample* Sample = WeakS.Get()) { CheckClassification(*this, *F, Sample, Gold); }
					CheckPending(*this, *F, TEXT("classifier golden"));
					Done.ExecuteIfBound();
				};
				RunRoll(Search, 0);
			});
		};

		LatentIt("ClassifierSoccerBallLabel", LatentLimit, [this, ClassifierCase](const FDoneDelegate& Done) { ClassifierCase(Done, TEXT("c03")); });
		LatentIt("ClassifierJellyfishLabel", LatentLimit, [this, ClassifierCase](const FDoneDelegate& Done) { ClassifierCase(Done, TEXT("c02")); });

		/** Detection looking at a golden photo: >= 1 detection with a golden label, boxes in [0,1], box over the right place. */
		auto DetectionCase = [this](const FDoneDelegate& Done, const FString& ImageId)
		{
			// The photo keeps its aspect inside the 960 x 540 view (black bars), DetectNow() captures and runs RT-DETR. At least one
			// detection must carry a label the Python reference found in that photo with score >= ScoreThreshold, every box lies inside
			// [0,1], and the box of the first detected golden label sits where the golden box maps into the view (centre within 0.05) with the
			// golden aspect (25 %). The position check is part of the roll match and is never skipped: a sample that squashed the capture,
			// swapped x/y or mapped boxes to the wrong frame fails it at every roll.
			if (SkipWithoutRhi(*this)) { Done.ExecuteIfBound(); return; }
			FGoldDetCase Gold;
			if (!LoadGoldDet(*this, ImageId, Gold)) { Done.ExecuteIfBound(); return; }
			Fix = NewFix(*this, false, true, IsGpuRequired() ? EPipelineDevice::GPU : EPipelineDevice::Auto);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			if (!F->SpawnPhotoPlane(*this, Gold.Image)) { Done.ExecuteIfBound(); return; }
			AModelPipelinesDetectionSample* S = Cast<AModelPipelinesDetectionSample>(F->SpawnFor(*this, true));
			if (!S) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, S, Gold, ImageId, Done]()
			{
				const TSharedRef<FRollSearch> Search = MakeShared<FRollSearch>();
				Search->T = this;
				Search->F = F;
				Search->A = S;
				Search->Img = Gold.Image;
				Search->bStretch = false;
				Search->Description = FString::Printf(TEXT("detection on '%s' (golden labels: %s)"), *ImageId, *FString::Join(Gold.Labels.Array(), TEXT(",")));
				Search->Done = Done;
				const TWeakObjectPtr<AModelPipelinesDetectionSample> WeakS = S;
				Search->Check = [WeakS, Gold](FString& Seen)
				{
					const AModelPipelinesDetectionSample* Sample = WeakS.Get();
					if (!Sample) { Seen = TEXT("(sample gone)"); return false; }
					const TArray<FPipelineDetection> Dets = Sample->GetLastDetections();
					Seen = DescribeDetections(Dets);
					bool bLabel = false;
					for (const FPipelineDetection& D : Dets)
					{
						if (Gold.Labels.Contains(D.Label) && D.Score >= Sample->ScoreThreshold) { bLabel = true; break; }
					}
					// A roll counts only when the label matches AND the box lands where the golden box maps in the view with the golden aspect.
					const bool bPos = CheckDetectionPosition(Sample, Gold, Seen);
					return bLabel && bPos;
				};
				Search->OnMatch = [this, F, WeakS, Gold, ImageId, Done](int32)
				{
					const AModelPipelinesDetectionSample* Sample = WeakS.Get();
					if (!Sample) { Done.ExecuteIfBound(); return; }
					const TArray<FPipelineDetection> Dets = Sample->GetLastDetections();
					for (int32 i = 0; i < Dets.Num(); ++i)
					{
						const FBox2D& B = Dets[i].Box;
						const bool bInside = B.bIsValid && B.Min.X >= -1e-6 && B.Min.Y >= -1e-6 && B.Max.X <= 1.0 + 1e-6 && B.Max.Y <= 1.0 + 1e-6 && B.Min.X <= B.Max.X && B.Min.Y <= B.Max.Y;
						TestTrue(*FString::Printf(TEXT("detection %d '%s' box [%.3f %.3f %.3f %.3f] must be inside [0,1] and ordered"), i, *Dets[i].Label, B.Min.X, B.Min.Y, B.Max.X, B.Max.Y), bInside);
						TestTrue(*FString::Printf(TEXT("detection %d score %.3f must be >= ScoreThreshold %.2f"), i, Dets[i].Score, Sample->ScoreThreshold), Dets[i].Score >= Sample->ScoreThreshold - 1e-6f);
					}
					TestTrue(TEXT("OnSampleResult arrived on the game thread"), F->Receiver->bAllOnGameThread);
					TestTrue(TEXT("GetLastResult is a success"), Sample->GetLastResult().bSuccess);

					// The position/aspect check already passed inside the roll search (Check); a roll without it never matches.
					// FOV: the sample's capture is a horizontal-FOV view. Without a viewport or local player the sample uses the camera FOV 90.
					if (const USceneCaptureComponent2D* Cap = Sample->FindComponentByClass<USceneCaptureComponent2D>())
					{
						const float Want = UE::ModelPipelines::Samples::CaptureHorizontalFov(90.f, AspectRatio_MaintainYFOV, FIntPoint(0, 0));
						TestTrue(*FString::Printf(TEXT("detection capture FOV %.3f must equal CaptureHorizontalFov(90, MaintainYFOV, no viewport) = %.3f"), Cap->FOVAngle, Want), FMath::IsNearlyEqual(Cap->FOVAngle, Want, 0.01f));
					}
					CheckPending(*this, *F, TEXT("detection golden"));
					Done.ExecuteIfBound();
				};
				RunRoll(Search, 0);
			});
		};

		LatentIt("DetectionDogAndPerson", LatentLimit, [this, DetectionCase](const FDoneDelegate& Done) { DetectionCase(Done, TEXT("d06")); });
		LatentIt("DetectionBicycles", LatentLimit, [this, DetectionCase](const FDoneDelegate& Done) { DetectionCase(Done, TEXT("d14")); });

		LatentIt("DetectionRenderTargetSizeNoViewport", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// With a real RHI and still no game viewport: the target is 960 x 540 RGBA8 (brief acceptance), and a call on it succeeds.
			if (SkipWithoutRhi(*this)) { Done.ExecuteIfBound(); return; }
			Fix = NewFix(*this, false, true, IsGpuRequired() ? EPipelineDevice::GPU : EPipelineDevice::Auto);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesDetectionSample* S = Cast<AModelPipelinesDetectionSample>(F->SpawnFor(*this, true));
			if (!S) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, S, Done]()
			{
				const UTextureRenderTarget2D* RT = S->GetRenderTarget();
				if (TestNotNull(TEXT("GetRenderTarget()"), RT))
				{
					TestEqual(TEXT("render target width"), (int32)RT->SizeX, 960);
					TestEqual(TEXT("render target height"), (int32)RT->SizeY, 540);
				}
				S->DetectNow();
				WaitFor(this, [F]() { return F->Receiver->Count() >= 1; }, TEXT("OnSampleResult"), Done, [this, F, Done]()
				{
					TestTrue(*FString::Printf(TEXT("the call on an RHI succeeds: %s"), *F->Receiver->Results[0].Message), F->Receiver->Results[0].bSuccess);
					Done.ExecuteIfBound();
				});
			});
		});

		/** Median game-thread time of 20 Now() calls (capture + pipeline call), each waited out before the next. */
		auto TimingBody = [this](const FDoneDelegate& Done, bool bDet)
		{
			// The sample's own game-thread work per call must stay small (target < 1 ms; the M4 readback itself is ~0.05 ms): no flush, no ReadPixels, no
			// waiting. A sample that flushed the render thread or read pixels synchronously costs several ms per call.
			if (SkipWithoutRhi(*this)) { Done.ExecuteIfBound(); return; }
			Fix = NewFix(*this, !bDet, bDet, IsGpuRequired() ? EPipelineDevice::GPU : EPipelineDevice::Auto);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSampleBase* A = F->SpawnFor(*this, bDet);
			if (!A) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [A]() { return StateIs(A, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, A, bDet, Done]()
			{
				const TSharedRef<TArray<double>> Ms = MakeShared<TArray<double>>();
				const TSharedRef<TFunction<void()>> Step = MakeShared<TFunction<void()>>();
				const TWeakObjectPtr<AModelPipelinesSampleBase> WeakA = A;
				*Step = [this, F, WeakA, bDet, Ms, Step, Done]()
				{
					AModelPipelinesSampleBase* Sample = WeakA.Get();
					if (!Sample) { AddError(TEXT("sample vanished during the timing loop")); Done.ExecuteIfBound(); return; }
					if (Ms->Num() >= 20)
					{
						const float Median = VisionSamplesSpecPrivate::Median(*Ms);
						AddInfo(FString::Printf(TEXT("%s game-thread time per call over %d calls: median %.4f ms (min %.4f, max %.4f)"),
							bDet ? TEXT("DetectNow()") : TEXT("ClassifyNow()"), Ms->Num(), Median, FMath::Min(*Ms), FMath::Max(*Ms)));
						TestTrue(*FString::Printf(TEXT("median game-thread time per call %.4f ms must be < 1 ms"), Median), Median < 1.0f);
						Done.ExecuteIfBound();
						return;
					}
					const int32 Prev = F->Receiver->Count();
					const double T0 = FPlatformTime::Seconds();
					CallNow(Sample);
					Ms->Add((FPlatformTime::Seconds() - T0) * 1000.0);
					WaitFor(this, [F, Prev, WeakA]() { return F->Receiver->Count() > Prev && StateIs(WeakA.Get(), EPipelineSampleState::Ready); },
						TEXT("OnSampleResult and state Ready in the timing loop"), Done, [Step]() { (*Step)(); });
				};
				(*Step)();
			});
		};
		LatentIt("DetectNowGameThreadMs", LatentLimit, [this, TimingBody](const FDoneDelegate& Done) { TimingBody(Done, true); });
		LatentIt("ClassifyNowGameThreadMs", LatentLimit, [this, TimingBody](const FDoneDelegate& Done) { TimingBody(Done, false); });
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
