// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesDetectionSample.h"

#include "Components/SceneCaptureComponent2D.h"
#include "Engine/GameViewportClient.h"
#include "Engine/LocalPlayer.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "SceneView.h"
#include "Slate/SceneViewport.h"
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
	// Same aspect as the player's view rect, so normalized boxes map straight onto that rect. Fallbacks: the whole viewport, then
	// 16:9 without a viewport (960 x 540). Later aspect changes (window resize, split screen) are followed in StartDetectCall.
	FIntRect ViewRect(0, 0, 0, 0);
	FIntPoint ViewportSize = FIntPoint::ZeroValue;
	GetPlayerView(ViewRect, ViewportSize);
	const FIntPoint Size = UE::ModelPipelines::Samples::CaptureSizeForView(CaptureWidth, ViewRect);
	RenderTarget = UE::ModelPipelines::Samples::CreateSampleCaptureTarget(this, Size.X, Size.Y);
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
	// Same location, rotation and view as the player camera: CopyPlayerCamera sets the capture's FOVAngle to the view's horizontal FOV,
	// read from the player's projection matrix (fallback: the camera FOV, widened only under the legacy MaintainYFOV cvar), and the
	// render target has the aspect of the player's view rect (BeginPlay / UpdateCaptureTargetForView), so the capture frames exactly
	// the view. Without a player controller (tests) the actor keeps its own transform and the capture its own FOV (90).
	UE::ModelPipelines::Samples::CopyPlayerCamera(this, SceneCapture);
}

bool AModelPipelinesDetectionSample::GetPlayerView(FIntRect& OutViewRect, FIntPoint& OutViewportSize) const
{
	UWorld* World = GetWorld();
	UGameViewportClient* Client = World ? World->GetGameViewport() : nullptr;
	FSceneViewport* Viewport = Client ? Client->GetGameViewport() : nullptr;
	const FIntPoint Size = Viewport ? Viewport->GetSizeXY() : FIntPoint::ZeroValue;
	if (Size.X <= 0 || Size.Y <= 0)
	{
		return false;
	}
	OutViewportSize = Size;
	OutViewRect = FIntRect(FIntPoint::ZeroValue, Size);
	// The player's constrained view rect (split screen, bConstrainAspectRatio); LocalPlayer.h:623 (5.8) / :613 (5.7), SceneView.h:98.
	// It includes the viewport's initial position, so make it relative to the viewport (the overlay covers the viewport).
	APlayerController* PC = World->GetFirstPlayerController();
	ULocalPlayer* LocalPlayer = PC ? PC->GetLocalPlayer() : nullptr;
	FSceneViewProjectionData Data;
	if (LocalPlayer && LocalPlayer->GetProjectionData(Viewport, Data))
	{
		FIntRect Rect = Data.GetConstrainedViewRect();
		const FIntPoint Offset = Viewport->GetInitialPositionXY();
		Rect.Min -= Offset;
		Rect.Max -= Offset;
		if (Rect.Width() > 0 && Rect.Height() > 0)
		{
			OutViewRect = Rect;
		}
	}
	return true;
}

void AModelPipelinesDetectionSample::UpdateCaptureTargetForView(const FIntRect& ViewRect)
{
	// Called only from StartDetectCall: the previous call has finished, so no in-flight call still reads the old target.
	if (LastTargetCheckFrame == GFrameCounter)
	{
		return;
	}
	LastTargetCheckFrame = GFrameCounter;
	const FIntPoint Current = RenderTarget ? FIntPoint(RenderTarget->SizeX, RenderTarget->SizeY) : FIntPoint::ZeroValue;
	const FIntPoint Wanted = UE::ModelPipelines::Samples::CaptureSizeForView(CaptureWidth, ViewRect);
	// The size check avoids recreating the same size again when rounding a tiny CaptureWidth alone exceeds the 1 % aspect limit.
	if (!UE::ModelPipelines::Samples::NeedsNewCaptureTarget(Current, ViewRect) || Wanted == Current)
	{
		return;
	}
	RenderTarget = UE::ModelPipelines::Samples::CreateSampleCaptureTarget(this, Wanted.X, Wanted.Y);
	if (SceneCapture)
	{
		SceneCapture->TextureTarget = RenderTarget;
	}
}

void AModelPipelinesDetectionSample::StartDetectCall()
{
	if (!Pipeline || !RenderTarget || !SceneCapture)
	{
		FinishCallNextTick(FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("The detection sample has no pipeline or render target")));
		return;
	}
	// The view this call captures: the render target follows its aspect (recreated here only, between calls), and the overlay
	// draws this call's boxes inside it. Without a viewport (tests) the rect stays empty: 960 x 540, boxes over the whole overlay.
	FIntRect ViewRect(0, 0, 0, 0);
	FIntPoint ViewportSize = FIntPoint::ZeroValue;
	if (GetPlayerView(ViewRect, ViewportSize))
	{
		UpdateCaptureTargetForView(ViewRect);
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
	Pipeline->DetectAsync(RenderTarget, ScoreThreshold, [WeakThis = TWeakObjectPtr<AModelPipelinesDetectionSample>(this), ViewRect, ViewportSize](const FPipelineResult& Result, const TArray<FPipelineDetection>& Detections)
	{
		AModelPipelinesDetectionSample* Self = WeakThis.Get();
		if (!Self || !Self->IsPlaying())
		{
			return;
		}
		// The boxes belong to the frame captured when this call started, not to the current one; they stay on screen until
		// the next result. At 2 Hz they trail fast camera moves a little, which is fine for a sample.
		Self->LastDetections = Result.bSuccess ? Detections : TArray<FPipelineDetection>();
		Self->LastViewRect = ViewRect;
		Self->LastViewportSize = ViewportSize;
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
	// The box layer fills the viewport and draws inside the player's view rect (the capture has that rect's aspect and the view's FOV);
	// the status panel sits on top, top-left.
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
		BoxOverlay->SetDetections(LastDetections, LastViewRect, LastViewportSize);
	}
}
