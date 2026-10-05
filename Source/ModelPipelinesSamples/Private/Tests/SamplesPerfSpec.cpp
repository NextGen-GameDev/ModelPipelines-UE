// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
//
// M5 performance capture (PLAN section 7: "Unreal Insights capture of the sample maps; report game-thread and inference ms
// per task"). Names: Pipelines.Samples.Perf.*
//
// One RHI-only test per sample. Each makes 20 sequential calls (every call waits for its OnSampleResult before the next starts),
// measures the game-thread cost of the call itself, reads QueueMs / PreprocessMs / InferenceMs / PostprocessMs / Device from
// GetLastResult(), and prints ONE parseable line per sample through AddInfo:
//
//   PERF <Sample> device=<GPU|CPU> first_ms=<e2e> call_gt_ms=<median>/<p90>/<max> inference_ms=<median>/<p90>
//        preprocess_ms=<median> postprocess_ms=<median> queue_ms=<median> e2e_ms=<median>/<p90> calls=<n>
//
// (one line; the key=value pairs are separated by single spaces; values are plain floats; <Sample> is one of
// SemanticSearch, Sentiment, SceneClassifier, Detection.) The first call (model load / warm-up) is excluded from every statistic
// and only reported as first_ms (its end-to-end time). Statistics are over the remaining 19 calls; p90 is nearest-rank.
//
// Every call is wrapped in TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.SamplePerf.<Sample>.Call") so an Unreal Insights
// export (TimingInsights.ExportTimingEvents) finds the game-thread cost of each call by timer name.
//
// end-to-end ms = from just before the call to the first game-thread poll (core ticker, every frame) that sees OnSampleResult
// delivered, so it can include up to one tick of polling latency on top of the true value.
//
// Assertions: every call succeeds; the median game-thread cost of the call is below 1 ms (p90 and max are only reported).
// Device: GPU when MODELPIPELINES_REQUIRE_GPU=1, else Auto. Skipped with an info under -nullrhi; an error with MODELPIPELINES_REQUIRE_RHI=1.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Algo/Sort.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/StaticMeshComponent.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "HAL/PlatformTime.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "UObject/Class.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#if WITH_EDITOR
#include "ShaderCompiler.h"
#endif
#include "PipelineModelPackage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesSemanticSearchSample.h"
#include "ModelPipelinesSentimentSample.h"
#include "ModelPipelinesSceneClassifierSample.h"
#include "ModelPipelinesDetectionSample.h"
#include "SamplesTestData.h"
#include "SamplesTestReceiver.h"

using namespace UE::ModelPipelines::SamplesTests;

namespace SamplesPerfSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(900);
	/** Waiting for the model load / corpus / first Ready state. */
	constexpr double SetupWaitSeconds = 300.0;
	/** One call in flight. */
	constexpr double CallWaitSeconds = 150.0;
	constexpr int32 NumCalls = 20;
	constexpr double ViewDistance = 200.0;
	constexpr double MedianLimitMs = 1.0;

	enum class EKind : uint8 { Search, Sentiment, Classify, Detect };

	static const TCHAR* SampleName(EKind K)
	{
		switch (K)
		{
		case EKind::Search: return TEXT("SemanticSearch");
		case EKind::Sentiment: return TEXT("Sentiment");
		case EKind::Classify: return TEXT("SceneClassifier");
		default: return TEXT("Detection");
		}
	}

	static const TCHAR* SlugOf(EKind K)
	{
		switch (K)
		{
		case EKind::Search: return TEXT("all-MiniLM-L6-v2");
		case EKind::Sentiment: return TEXT("distilbert-sst2");
		case EKind::Classify: return TEXT("resnet-50");
		default: return TEXT("rtdetr_r18vd");
		}
	}

	static const TCHAR* Questions[NumCalls] =
	{
		TEXT("Where can I get my sword repaired?"),
		TEXT("Is there a room where I can sleep tonight?"),
		TEXT("Who sells healing potions in this town?"),
		TEXT("How do I get to the castle?"),
		TEXT("What is the price of a night at the inn?"),
		TEXT("Do you know anything about the missing merchant?"),
		TEXT("Where is the nearest blacksmith?"),
		TEXT("Can you teach me how to fight?"),
		TEXT("Is it safe to travel through the forest at night?"),
		TEXT("Which road leads to the harbor?"),
		TEXT("I am hungry, where can I eat?"),
		TEXT("Who is the leader of the guard?"),
		TEXT("Are there any quests available for beginners?"),
		TEXT("Where can I sell the treasure I found?"),
		TEXT("Tell me about the old ruins to the north."),
		TEXT("Do you have a map of the region?"),
		TEXT("What happened to the village bridge?"),
		TEXT("Can I buy a horse somewhere around here?"),
		TEXT("Is there a healer who can cure poison?"),
		TEXT("What time does the market open?"),
	};

	static const TCHAR* ChatMessages[NumCalls] =
	{
		TEXT("I love this game, best quest ever!"),
		TEXT("This is the worst, I want a refund."),
		TEXT("The boss fight was absolutely amazing."),
		TEXT("Servers keep crashing, completely unplayable."),
		TEXT("Great teamwork everyone, we won!"),
		TEXT("Why is the matchmaking so terrible?"),
		TEXT("Such a beautiful world, I could explore for hours."),
		TEXT("The new patch broke everything again."),
		TEXT("Thanks for the help, you are a lifesaver."),
		TEXT("I hate how slow the loading screens are."),
		TEXT("What a fantastic soundtrack!"),
		TEXT("Boring and repetitive, I quit."),
		TEXT("That last level was brilliant."),
		TEXT("Awful lag, I lost all my progress."),
		TEXT("Best multiplayer experience I have had in years."),
		TEXT("The controls feel clunky and frustrating."),
		TEXT("Wow, the graphics are gorgeous."),
		TEXT("Terrible balance, the mage class is useless."),
		TEXT("I am having so much fun with my friends."),
		TEXT("Not worth the money at all."),
	};

	static bool SkipWithoutRhi(FAutomationTestBase& T)
	{
		if (HasRhi()) { return false; }
		if (IsRhiRequired()) { T.AddError(TEXT("RHI not available but MODELPIPELINES_REQUIRE_RHI=1 (run without -nullrhi, D3D12)")); }
		else { T.AddInfo(TEXT("RHI not available (-nullrhi): this perf test was SKIPPED, it did not run")); }
		return true;
	}

	// ---------------------------------------------------------------- statistics

	static double Median(TArray<double> V)
	{
		if (V.Num() == 0) { return 0.0; }
		V.Sort();
		const int32 N = V.Num();
		return (N % 2 == 1) ? V[N / 2] : 0.5 * (V[N / 2 - 1] + V[N / 2]);
	}

	/** Nearest-rank percentile (P in 0..1). */
	static double Percentile(TArray<double> V, double P)
	{
		if (V.Num() == 0) { return 0.0; }
		V.Sort();
		const int32 Rank = FMath::Clamp(FMath::CeilToInt(P * V.Num()), 1, V.Num());
		return V[Rank - 1];
	}

	static double MaxOf(const TArray<double>& V)
	{
		double M = 0.0;
		for (double D : V) { M = FMath::Max(M, D); }
		return M;
	}

	// ---------------------------------------------------------------- fixture

	struct FPerfFix
	{
		FSamplesSceneWorld World;
		TStrongObjectPtr<UPipelineModelPackage> Pkg;
		TStrongObjectPtr<USamplesTestReceiver> Receiver;
		TWeakObjectPtr<AModelPipelinesSampleBase> Actor;
		TStrongObjectPtr<UTexture2D> PhotoTexture;
		TStrongObjectPtr<UMaterialInstanceDynamic> PhotoMaterial;
		TWeakObjectPtr<AStaticMeshActor> Plane;
		FVisionTestImage Image;
		EPipelineDevice Device = EPipelineDevice::Auto;

		void Teardown()
		{
			if (AModelPipelinesSampleBase* A = Actor.Get())
			{
				A->OnSampleResult.RemoveAll(Receiver.Get());
			}
			World.Destroy();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		}
	};

	/** A unit plane with the photo, unlit/emissive, at the origin facing -X (same recipe as VisionSamplesSpec). */
	static bool SpawnPhotoPlane(FAutomationTestBase& T, FPerfFix& F)
	{
		UStaticMesh* Mesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
		UMaterialInterface* Parent = LoadObject<UMaterialInterface>(nullptr, TEXT("/Engine/EngineMaterials/EmissiveTexturedMaterial.EmissiveTexturedMaterial"));
		if (!Mesh || !Parent) { T.AddError(TEXT("Engine content missing: /Engine/BasicShapes/Plane or /Engine/EngineMaterials/EmissiveTexturedMaterial")); return false; }
		UTexture2D* Tex = MakeVisionTexture(F.Image);
		if (!Tex) { T.AddError(TEXT("MakeVisionTexture returned null although an RHI exists")); return false; }
		F.PhotoTexture.Reset(Tex);
		UMaterialInstanceDynamic* Mid = UMaterialInstanceDynamic::Create(Parent, GetTransientPackage());
		if (!Mid) { T.AddError(TEXT("UMaterialInstanceDynamic::Create returned null")); return false; }
		F.PhotoMaterial.Reset(Mid);
		Mid->SetTextureParameterValue(FName(TEXT("Texture")), Tex);
		AStaticMeshActor* P = F.World.World->SpawnActor<AStaticMeshActor>(FVector::ZeroVector, FRotator::ZeroRotator);
		if (!P || !P->GetStaticMeshComponent()) { T.AddError(TEXT("Spawning the photo plane failed")); return false; }
		UStaticMeshComponent* Comp = P->GetStaticMeshComponent();
		Comp->SetMobility(EComponentMobility::Movable);
		Comp->SetStaticMesh(Mesh);
		Comp->SetMaterial(0, Mid);
		Comp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
		Comp->SetCastShadow(false);
		F.Plane = P;
		Mid->EnsureIsComplete();
#if WITH_EDITOR
		if (GShaderCompilingManager) { GShaderCompilingManager->FinishAllCompilation(); }
#endif
		return true;
	}

	/**
	 * Sample at (-ViewDistance, 0, 0) looking +X, plane turned 90 degrees about the view axis (the upright orientation found by
	 * VisionSamplesSpec for the engine plane), photo stretched to fill the view (bStretch) or contained with its aspect.
	 * For a perf run the exact picture does not matter, only that the capture sees a real photo of the usual size.
	 */
	static void PlacePhoto(FPerfFix& F, AModelPipelinesSampleBase* S, bool bStretch)
	{
		const USceneCaptureComponent2D* Cap = S->FindComponentByClass<USceneCaptureComponent2D>();
		const UTextureRenderTarget2D* RT = Cap ? Cast<UTextureRenderTarget2D>(Cap->TextureTarget) : nullptr;
		const double Fov = Cap ? (double)Cap->FOVAngle : 90.0;
		const double Aspect = (RT && RT->SizeY > 0) ? (double)RT->SizeX / (double)RT->SizeY : 1.0;
		const double VisW = 2.0 * ViewDistance * FMath::Tan(FMath::DegreesToRadians(Fov * 0.5));
		const double VisH = VisW / Aspect;
		double FitW = VisW, FitH = VisH;
		if (!bStretch)
		{
			const double ImgAspect = (double)F.Image.Width / (double)FMath::Max(1, F.Image.Height);
			if (ImgAspect * VisH <= VisW) { FitH = VisH; FitW = ImgAspect * VisH; }
			else { FitW = VisW; FitH = VisW / ImgAspect; }
		}
		S->SetActorLocationAndRotation(FVector(-ViewDistance, 0.0, 0.0), FRotator::ZeroRotator);
		if (AStaticMeshActor* P = F.Plane.Get())
		{
			P->SetActorLocation(FVector::ZeroVector);
			P->SetActorRotation(FQuat(FVector::XAxisVector, FMath::DegreesToRadians(90.0)) * FRotator(90.0, 0.0, 0.0).Quaternion());
			P->SetActorScale3D(FVector(FitW / 100.0, FitH / 100.0, 1.0)); // odd quarter turn: X is horizontal; the engine plane is 100 x 100 cm
		}
		F.World.World->SendAllEndOfFrameUpdates();
	}

	// ---------------------------------------------------------------- the run

	struct FCallRecord
	{
		double CallGtMs = 0.0;
		double E2eMs = 0.0;
		FPipelineResult Result;
	};

	/** Drives the 20 sequential calls from the core ticker; never blocks the game thread. */
	struct FPerfRun : public TSharedFromThis<FPerfRun>
	{
		FAutomationTestBase* T = nullptr;
		TSharedPtr<FPerfFix> F;
		EKind Kind = EKind::Search;
		FDoneDelegate Done;
		/** True when the sample may take the next call (pipeline created / corpus ready, state Ready). */
		TFunction<bool()> CanIssue;
		/** Starts call number Index (inside the trace scope). */
		TFunction<void(int32 Index)> Issue;

		TArray<FCallRecord> Records;
		int32 BaseCount = 0;
		int32 NumIssued = 0;
		bool bInFlight = false;
		bool bFinished = false;
		double CallStart = 0.0;
		double CallGtMs = 0.0;
		double WaitStart = 0.0;

		void Start()
		{
			WaitStart = FPlatformTime::Seconds();
			// The ticker owns the run (strong ref) until Tick returns false (finished or failed); nothing else keeps it alive.
			TSharedRef<FPerfRun> Self = AsShared();
			FTSTicker::GetCoreTicker().AddTicker(TEXT("SamplesPerfSpecRun"), 0.0f, [Self](float)
			{
				return Self->Tick();
			});
		}

		bool Fail(const FString& Msg)
		{
			T->AddError(FString::Printf(TEXT("%s perf: %s"), SampleName(Kind), *Msg));
			return Finish();
		}

		bool Finish()
		{
			if (!bFinished)
			{
				bFinished = true;
				Done.ExecuteIfBound();
			}
			return false;
		}

		bool Tick()
		{
			if (bFinished) { return false; }
			AModelPipelinesSampleBase* A = F->Actor.Get();
			if (!A) { return Fail(TEXT("the sample actor is gone")); }
			const double Now = FPlatformTime::Seconds();

			if (bInFlight)
			{
				if (F->Receiver->Count() >= BaseCount + NumIssued)
				{
					const double E2e = (Now - CallStart) * 1000.0;
					bInFlight = false;
					FCallRecord Rec;
					Rec.CallGtMs = CallGtMs;
					Rec.E2eMs = E2e;
					Rec.Result = F->Receiver->Results[BaseCount + NumIssued - 1];
					if (!Rec.Result.bSuccess)
					{
						return Fail(FString::Printf(TEXT("call %d failed: %s (error %d)"), NumIssued, *Rec.Result.Message, (int32)Rec.Result.Error));
					}
					Records.Add(Rec);
					if (!F->Receiver->bAllOnGameThread) { return Fail(TEXT("OnSampleResult was delivered off the game thread")); }
					return true;
				}
				if (Now - CallStart > CallWaitSeconds)
				{
					return Fail(FString::Printf(TEXT("timed out after %.0f s waiting for the result of call %d (state %s, status: %s)"),
						CallWaitSeconds, NumIssued, *UEnum::GetValueAsString(A->GetSampleState()), *A->GetStatusText()));
				}
				return true;
			}

			if (NumIssued >= NumCalls)
			{
				Report();
				return Finish();
			}
			if (!CanIssue())
			{
				if (Now - WaitStart > SetupWaitSeconds)
				{
					return Fail(FString::Printf(TEXT("timed out after %.0f s waiting for the sample to become ready for call %d (state %s, status: %s)"),
						SetupWaitSeconds, NumIssued, *UEnum::GetValueAsString(A->GetSampleState()), *A->GetStatusText()));
				}
				return true;
			}
			if (NumIssued == 0) { BaseCount = F->Receiver->Count(); } // Search: the corpus call already delivered one result
			CallStart = FPlatformTime::Seconds();
			Issue(NumIssued); // inside the trace scope (see the Issue lambdas)
			CallGtMs = (FPlatformTime::Seconds() - CallStart) * 1000.0;
			++NumIssued;
			bInFlight = true;
			WaitStart = FPlatformTime::Seconds();
			return true;
		}

		void Report()
		{
			if (Records.Num() < 2) { T->AddError(FString::Printf(TEXT("%s perf: only %d call(s) recorded"), SampleName(Kind), Records.Num())); return; }
			TArray<double> Gt, Inf, Pre, Post, Que, E2e;
			for (int32 i = 1; i < Records.Num(); ++i) // the first call is the warm-up
			{
				const FCallRecord& R = Records[i];
				Gt.Add(R.CallGtMs);
				Inf.Add(R.Result.InferenceMs);
				Pre.Add(R.Result.PreprocessMs);
				Post.Add(R.Result.PostprocessMs);
				Que.Add(R.Result.QueueMs);
				E2e.Add(R.E2eMs);
			}
			const EPipelineDevice Dev = Records.Last().Result.Device;
			const TCHAR* DevName = Dev == EPipelineDevice::GPU ? TEXT("GPU") : (Dev == EPipelineDevice::CPU ? TEXT("CPU") : TEXT("UNKNOWN"));
			T->AddInfo(FString::Printf(
				TEXT("PERF %s device=%s first_ms=%.2f call_gt_ms=%.3f/%.3f/%.3f inference_ms=%.2f/%.2f preprocess_ms=%.3f postprocess_ms=%.3f queue_ms=%.3f e2e_ms=%.2f/%.2f calls=%d"),
				SampleName(Kind), DevName, Records[0].E2eMs,
				Median(Gt), Percentile(Gt, 0.9), MaxOf(Gt),
				Median(Inf), Percentile(Inf, 0.9),
				Median(Pre), Median(Post), Median(Que),
				Median(E2e), Percentile(E2e, 0.9), Records.Num()));
			if (Dev != EPipelineDevice::GPU && Dev != EPipelineDevice::CPU)
			{
				T->AddError(FString::Printf(TEXT("%s perf: the results report device Auto (nothing ran?)"), SampleName(Kind)));
			}
			const double GtMedian = Median(Gt);
			T->TestTrue(*FString::Printf(TEXT("%s: median game-thread cost of the call is %.3f ms, must be < %.1f ms (p90 %.3f ms, max %.3f ms)"),
				SampleName(Kind), GtMedian, MedianLimitMs, Percentile(Gt, 0.9), MaxOf(Gt)), GtMedian < MedianLimitMs);
		}
	};

	/** Builds the fixture for Kind and starts the run. Adds errors and calls Done on any setup failure. */
	static void BeginPerf(FAutomationTestBase& T, TSharedPtr<FPerfFix>& OutFix, EKind Kind, const FDoneDelegate& Done)
	{
		if (SkipWithoutRhi(T)) { Done.ExecuteIfBound(); return; }

		TSharedRef<FPerfFix> F = MakeShared<FPerfFix>();
		OutFix = F;
		F->Device = IsGpuRequired() ? EPipelineDevice::GPU : EPipelineDevice::Auto;
		const bool bVision = (Kind == EKind::Classify || Kind == EKind::Detect);

		FString Err;
		if (bVision)
		{
			const TSharedPtr<FJsonObject> Root = LoadVisionGoldens(Err);
			if (!Root.IsValid()) { T.AddError(Err); Done.ExecuteIfBound(); return; }
			const TSharedPtr<FJsonObject> Model = FindVisionModel(Root, SlugOf(Kind));
			if (!Model.IsValid())
			{
				T.AddError(FString::Printf(TEXT("Golden model '%s' not found in vision_goldens.json. Regenerate with Tools/make_vision_goldens.py (after Tools/fetch_reference_models.py)."), SlugOf(Kind)));
				Done.ExecuteIfBound();
				return;
			}
			UPipelineModelPackage* Pkg = MakeVisionPackage(Root, Model, F->Device, Err);
			if (!Pkg) { T.AddError(Err); Done.ExecuteIfBound(); return; }
			Pkg->PreferredDevice = F->Device;
			F->Pkg.Reset(Pkg);
			// A golden photo of the sample's own task (c03 = soccer ball for resnet-50, d06 = dog and person for RT-DETR).
			if (!LoadVisionImage(Root, Kind == EKind::Classify ? TEXT("c03") : TEXT("d06"), F->Image, Err)) { T.AddError(Err); Done.ExecuteIfBound(); return; }
			if (!F->World.CreateWithScene(Err)) { T.AddError(Err); Done.ExecuteIfBound(); return; }
			if (!SpawnPhotoPlane(T, *F)) { Done.ExecuteIfBound(); return; }
		}
		else
		{
			const TSharedPtr<FJsonObject> Root = LoadTextGoldens(Err);
			if (!Root.IsValid()) { T.AddError(Err); Done.ExecuteIfBound(); return; }
			const TSharedPtr<FJsonObject> Model = FindModel(Root, SlugOf(Kind));
			if (!Model.IsValid())
			{
				T.AddError(FString::Printf(TEXT("Golden model '%s' not found in text_goldens.json. Regenerate with Tools/make_text_goldens.py."), SlugOf(Kind)));
				Done.ExecuteIfBound();
				return;
			}
			UPipelineModelPackage* Pkg = MakePackage(Root, Model, F->Device, Err);
			if (!Pkg) { T.AddError(Err); Done.ExecuteIfBound(); return; }
			Pkg->PreferredDevice = F->Device;
			F->Pkg.Reset(Pkg);
			if (!F->World.CreateWithScene(Err)) { T.AddError(Err); Done.ExecuteIfBound(); return; }
		}
		F->Receiver.Reset(NewObject<USamplesTestReceiver>(GetTransientPackage()));

		UClass* Class = nullptr;
		switch (Kind)
		{
		case EKind::Search: Class = AModelPipelinesSemanticSearchSample::StaticClass(); break;
		case EKind::Sentiment: Class = AModelPipelinesSentimentSample::StaticClass(); break;
		case EKind::Classify: Class = AModelPipelinesSceneClassifierSample::StaticClass(); break;
		default: Class = AModelPipelinesDetectionSample::StaticClass(); break;
		}
		USamplesTestReceiver* Rcv = F->Receiver.Get();
		AModelPipelinesSampleBase* A = F->World.SpawnSample(Class, TSoftObjectPtr<UPipelineModelPackage>(F->Pkg.Get()), /*bShowOverlay*/ false,
			[Rcv](AModelPipelinesSampleBase* NewActor)
			{
				NewActor->OnSampleResult.AddDynamic(Rcv, &USamplesTestReceiver::OnResult);
				if (AModelPipelinesSceneClassifierSample* C = Cast<AModelPipelinesSceneClassifierSample>(NewActor)) { C->bFollowPlayerCamera = false; }
				if (AModelPipelinesDetectionSample* D = Cast<AModelPipelinesDetectionSample>(NewActor)) { D->bFollowPlayerCamera = false; }
			}, Err);
		if (!A) { T.AddError(Err); Done.ExecuteIfBound(); return; }
		F->Actor = A;
		if (Kind == EKind::Classify) { PlacePhoto(*F, A, /*bStretch*/ true); }
		if (Kind == EKind::Detect) { PlacePhoto(*F, A, /*bStretch*/ false); }

		const TSharedRef<FPerfRun> Run = MakeShared<FPerfRun>();
		Run->T = &T;
		Run->F = F;
		Run->Kind = Kind;
		Run->Done = Done;
		const TWeakObjectPtr<AModelPipelinesSampleBase> Weak = A;
		Run->CanIssue = [Weak, Kind]() -> bool
		{
			const AModelPipelinesSampleBase* S = Weak.Get();
			if (!S || S->GetSampleState() != EPipelineSampleState::Ready) { return false; }
			if (Kind == EKind::Search)
			{
				const AModelPipelinesSemanticSearchSample* Search = Cast<AModelPipelinesSemanticSearchSample>(S);
				return Search && Search->IsCorpusReady();
			}
			return true;
		};
		switch (Kind)
		{
		case EKind::Search:
			Run->Issue = [Weak](int32 I)
			{
				if (AModelPipelinesSemanticSearchSample* S = Cast<AModelPipelinesSemanticSearchSample>(Weak.Get()))
				{
					TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.SamplePerf.SemanticSearch.Call");
					S->Search(Questions[I % NumCalls]);
				}
			};
			break;
		case EKind::Sentiment:
			Run->Issue = [Weak](int32 I)
			{
				if (AModelPipelinesSentimentSample* S = Cast<AModelPipelinesSentimentSample>(Weak.Get()))
				{
					TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.SamplePerf.Sentiment.Call");
					S->Analyze(ChatMessages[I % NumCalls]);
				}
			};
			break;
		case EKind::Classify:
			Run->Issue = [Weak](int32)
			{
				if (AModelPipelinesSceneClassifierSample* S = Cast<AModelPipelinesSceneClassifierSample>(Weak.Get()))
				{
					TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.SamplePerf.SceneClassifier.Call");
					S->ClassifyNow();
				}
			};
			break;
		default:
			Run->Issue = [Weak](int32)
			{
				if (AModelPipelinesDetectionSample* S = Cast<AModelPipelinesDetectionSample>(Weak.Get()))
				{
					TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.SamplePerf.Detection.Call");
					S->DetectNow();
				}
			};
			break;
		}
		Run->Start();
	}
}

BEGIN_DEFINE_SPEC(FSamplesPerfSpec, "Pipelines.Samples.Perf",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	TSharedPtr<SamplesPerfSpecPrivate::FPerfFix> Fix;
END_DEFINE_SPEC(FSamplesPerfSpec)

void FSamplesPerfSpec::Define()
{
	using namespace SamplesPerfSpecPrivate;

	AfterEach([this]()
	{
		if (Fix.IsValid())
		{
			Fix->Teardown();
			Fix.Reset();
		}
	});

	// Each test: 20 sequential calls, one PERF line, success of every call, median game-thread cost of the call < 1 ms.
	// A sample that did its tokenizing / capture readback / pre-processing on the game thread inside the call would show here.
	LatentIt("SemanticSearch", LatentLimit, [this](const FDoneDelegate& Done) { BeginPerf(*this, Fix, EKind::Search, Done); });
	LatentIt("Sentiment", LatentLimit, [this](const FDoneDelegate& Done) { BeginPerf(*this, Fix, EKind::Sentiment, Done); });
	LatentIt("SceneClassifier", LatentLimit, [this](const FDoneDelegate& Done) { BeginPerf(*this, Fix, EKind::Classify, Done); });
	LatentIt("Detection", LatentLimit, [this](const FDoneDelegate& Done) { BeginPerf(*this, Fix, EKind::Detect, Done); });
}

#endif // WITH_DEV_AUTOMATION_TESTS
