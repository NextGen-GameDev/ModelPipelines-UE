// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesDetectionSample.h"

#include "Components/SceneCaptureComponent2D.h"
#include "Engine/GameViewportClient.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "ObjectDetectionPipeline.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "VisionPipelineLibrary.h"
#include "Widgets/SNullWidget.h"
#include "Widgets/SOverlay.h"
#include "SampleDetectionOverlay.h"
#include "SampleOverlayWidgets.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(ModelPipelinesDetectionSample)

AModelPipelinesDetectionSample::AModelPipelinesDetectionSample()
{
	// Where the importer puts the model by default (Tools/fetch_reference_models.py --only rtdetr_r18vd).
	Package = TSoftObjectPtr<UPipelineModelPackage>(FSoftObjectPath(TEXT("/Game/Models/rtdetr_r18vd.rtdetr_r18vd")));

	// Follows the camera and detects on an interval.
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bStartWithTickEnabled = true;

	SceneCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("SceneCapture"));
	RootComponent = SceneCapture;
	UE::ModelPipelines::Samples::ConfigureSampleCapture(SceneCapture);
}

// ---------------------------------------------------------------------------------------------------- API

void AModelPipelinesDetectionSample::DetectNow()
{
	// Runs now when idle, otherwise waits in the single pending slot (latest wins). The capture happens when the call
	// starts, so a queued request looks at the view of that moment.
	RunOrQueue([this]() { StartDetectCall(); });
}

TArray<FPipelineDetection> AModelPipelinesDetectionSample::GetLastDetections() const
{
	return LastDetections;
}

UTextureRenderTarget2D* AModelPipelinesDetectionSample::GetRenderTarget() const
{
	return RenderTarget;
}

void AModelPipelinesDetectionSample::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (bFollowPlayerCamera)
	{
		FollowPlayerCamera();
	}
	SecondsSinceLastCall += DeltaSeconds;
	// Only when idle: the timer never fills the pending slot (that is for explicit DetectNow calls).
	if (SecondsSinceLastCall >= IntervalSeconds && GetSampleState() == EPipelineSampleState::Ready)
	{
		SecondsSinceLastCall = 0.f;
		DetectNow();
	}
}

// ---------------------------------------------------------------------------------------------------- pipeline

void AModelPipelinesDetectionSample::BeginPlay()
{
	// Same aspect as the view, so normalized boxes map straight onto the viewport. 16:9 when there is no viewport (960 x 540).
	// This is the aspect CopyPlayerCamera uses for the horizontal FOV (the game viewport size); a later window resize is not followed.
	double Aspect = 16.0 / 9.0;
	UWorld* World = GetWorld();
	if (UGameViewportClient* Viewport = World ? World->GetGameViewport() : nullptr)
	{
		FVector2D ViewportSize = FVector2D::ZeroVector;
		Viewport->GetViewportSize(ViewportSize);
		if (ViewportSize.X > 0.0 && ViewportSize.Y > 0.0)
		{
			Aspect = ViewportSize.X / ViewportSize.Y;
		}
	}
	const int32 Width = FMath::Clamp(CaptureWidth, 16, 4096);
	const int32 Height = FMath::Clamp(FMath::RoundToInt32(Width / Aspect), 1, 4096);
	RenderTarget = UE::ModelPipelines::Samples::CreateSampleCaptureTarget(this, Width, Height);
	if (SceneCapture)
	{
		SceneCapture->TextureTarget = RenderTarget;
		UE::ModelPipelines::Samples::ConfigureSampleCapture(SceneCapture);
	}
	Super::BeginPlay();
}

FPipelineResult AModelPipelinesDetectionSample::CreatePipeline(UPipelineModelPackage* Loaded)
{
	FPipelineResult Result;
	Pipeline = UVisionPipelineLibrary::CreateObjectDetectionPipeline(Loaded, Result);
	return Result;
}

void AModelPipelinesDetectionSample::FollowPlayerCamera()
{
	// Same location, rotation and view as the player camera: CopyPlayerCamera sets the capture's FOVAngle to the view's HORIZONTAL
	// FOV (the camera FOV is vertical under the default MaintainYFOV constraint), and the render target has the viewport's aspect
	// (BeginPlay), so the capture frames exactly the view. Without a player controller (tests) the actor keeps its own transform
	// and the capture its own FOV (90).
	UE::ModelPipelines::Samples::CopyPlayerCamera(this, SceneCapture);
}

void AModelPipelinesDetectionSample::StartDetectCall()
{
	if (!Pipeline || !RenderTarget || !SceneCapture)
	{
		FinishCallNextTick(FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("The detection sample has no pipeline or render target")));
		return;
	}
	if (bFollowPlayerCamera)
	{
		FollowPlayerCamera();
	}

	// CaptureScene enqueues the scene render into RenderTarget on the render thread now (no flush, no wait). The pipeline's
	// readback (DetectAsync -> ReadTexturePixelsAsync) enqueues its GPU copy afterwards on the same render thread, so the
	// copy reads this capture. Without a render scene (-nullrhi) CaptureScene does nothing and the call fails cleanly.
	// Game-thread cost: CaptureScene() first runs World->SendAllEndOfFrameUpdates() on the game thread
	// (Engine/Private/Components/SceneCaptureComponent.cpp:832 in 5.8), which scales with the number of dirty components. The ~0.2 ms
	// per call measured by the tests is a near-empty test-world figure; a busy level costs more.
	SceneCapture->CaptureScene();

	// The callback runs on the game thread; the weak pointer skips it if the actor is gone (rule 4).
	Pipeline->DetectAsync(RenderTarget, ScoreThreshold, [WeakThis = TWeakObjectPtr<AModelPipelinesDetectionSample>(this)](const FPipelineResult& Result, const TArray<FPipelineDetection>& Detections)
	{
		AModelPipelinesDetectionSample* Self = WeakThis.Get();
		if (!Self || !Self->IsPlaying())
		{
			return;
		}
		// The boxes belong to the frame captured when this call started, not to the current one; they stay on screen until
		// the next result. At 2 Hz they trail fast camera moves a little, which is fine for a sample.
		Self->LastDetections = Result.bSuccess ? Detections : TArray<FPipelineDetection>();
		Self->FinishCall(Result);
	});
}

// ---------------------------------------------------------------------------------------------------- overlay

FString AModelPipelinesDetectionSample::GetModelSlug() const
{
	return TEXT("rtdetr_r18vd");
}

FString AModelPipelinesDetectionSample::GetReadyHint() const
{
	return TEXT("Ready: fly around (WASD + mouse), objects are detected twice per second");
}

FText AModelPipelinesDetectionSample::GetOverlayTitle() const
{
	return FText::FromString(TEXT("Object detection (rtdetr_r18vd)"));
}

TSharedRef<SWidget> AModelPipelinesDetectionSample::BuildOverlay()
{
	// The box layer fills the viewport (the capture has the view's aspect and FOV); the status panel sits on top, top-left.
	return SNew(SOverlay)
		+ SOverlay::Slot().HAlign(HAlign_Fill).VAlign(VAlign_Fill)
		[
			SAssignNew(BoxOverlay, SSampleDetectionOverlay)
		]
		+ SOverlay::Slot().HAlign(HAlign_Fill).VAlign(VAlign_Fill)
		[
			MakeOverlayPanel(SNullWidget::NullWidget)
		];
}

void AModelPipelinesDetectionSample::RefreshOverlay()
{
	if (BoxOverlay.IsValid())
	{
		BoxOverlay->SetDetections(LastDetections);
	}
}
