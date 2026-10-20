// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesSceneClassifierSample.h"

#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "ImageClassificationPipeline.h"
#include "Styling/SlateBrush.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "VisionPipelineLibrary.h"
#include "Widgets/Images/SImage.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/SBoxPanel.h"
#include "SampleOverlayWidgets.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(ModelPipelinesSceneClassifierSample)

AModelPipelinesSceneClassifierSample::AModelPipelinesSceneClassifierSample()
{
	// Where the importer puts the model by default (Tools/fetch_reference_models.py --only resnet-50).
	Package = TSoftObjectPtr<UPipelineModelPackage>(FSoftObjectPath(TEXT("/Game/Models/resnet-50.resnet-50")));

	// Follows the camera and classifies on an interval.
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bStartWithTickEnabled = true;

	SceneCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("SceneCapture"));
	RootComponent = SceneCapture;
	UE::ModelPipelines::Samples::ConfigureSampleCapture(SceneCapture);
}

// ---------------------------------------------------------------------------------------------------- API

void AModelPipelinesSceneClassifierSample::ClassifyNow()
{
	// Runs now when idle, otherwise waits in the single pending slot (latest wins). The capture happens when the call
	// starts, so a queued request classifies the view of that moment.
	RunOrQueue([this]() { StartClassifyCall(); });
}

FPipelineClassification AModelPipelinesSceneClassifierSample::GetLastClassification() const
{
	return LastClassification;
}

UTextureRenderTarget2D* AModelPipelinesSceneClassifierSample::GetRenderTarget() const
{
	return RenderTarget;
}

void AModelPipelinesSceneClassifierSample::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	if (bFollowPlayerCamera)
	{
		FollowPlayerCamera();
	}
	SecondsSinceLastCall += DeltaSeconds;
	// Only when idle: the timer never fills the pending slot (that is for explicit ClassifyNow calls).
	if (SecondsSinceLastCall >= IntervalSeconds && GetSampleState() == EPipelineSampleState::Ready)
	{
		SecondsSinceLastCall = 0.f;
		ClassifyNow();
	}
}

// ---------------------------------------------------------------------------------------------------- pipeline

void AModelPipelinesSceneClassifierSample::BeginPlay()
{
	// The render target exists before the base class builds the overlay (its preview brush points at it).
	const int32 Size = FMath::Clamp(CaptureResolution, 16, 4096);
	RenderTarget = UE::ModelPipelines::Samples::CreateSampleCaptureTarget(this, Size, Size);
	if (SceneCapture)
	{
		SceneCapture->TextureTarget = RenderTarget;
		UE::ModelPipelines::Samples::ConfigureSampleCapture(SceneCapture);
	}
	Super::BeginPlay();
}

FPipelineResult AModelPipelinesSceneClassifierSample::CreatePipeline(UPipelineModelPackage* Loaded)
{
	FPipelineResult Result;
	Pipeline = UVisionPipelineLibrary::CreateImageClassificationPipeline(Loaded, Result);
	return Result;
}

void AModelPipelinesSceneClassifierSample::FollowPlayerCamera()
{
	// Without a player controller (tests, no local player) the actor keeps its own transform.
	UE::ModelPipelines::Samples::CopyPlayerCamera(this, SceneCapture);
}

void AModelPipelinesSceneClassifierSample::StartClassifyCall()
{
	if (!Pipeline || !RenderTarget || !SceneCapture)
	{
		FinishCallNextTick(FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("The scene classifier has no pipeline or render target")));
		return;
	}
	if (bFollowPlayerCamera)
	{
		FollowPlayerCamera();
	}

	// CaptureScene enqueues the scene render into RenderTarget on the render thread now (no flush, no wait). The pipeline's
	// readback (ClassifyAsync -> ReadTexturePixelsAsync) enqueues its GPU copy afterwards on the same render thread, so the
	// copy reads this capture. Without a render scene (-nullrhi) CaptureScene does nothing and the call fails cleanly.
	// Game-thread cost: CaptureScene() first runs World->SendAllEndOfFrameUpdates() on the game thread
	// (Engine/Private/Components/SceneCaptureComponent.cpp:832 in 5.8), which scales with the number of dirty components. The ~0.2 ms
	// per call measured by the tests is a near-empty test-world figure; a busy level costs more.
	SceneCapture->CaptureScene();

	// The callback runs on the game thread; the weak pointer skips it if the actor is gone (rule 4).
	Pipeline->ClassifyAsync(RenderTarget, TopK, [WeakThis = TWeakObjectPtr<AModelPipelinesSceneClassifierSample>(this)](const FPipelineResult& Result, const FPipelineClassification& Classification)
	{
		AModelPipelinesSceneClassifierSample* Self = WeakThis.Get();
		if (!Self || !Self->IsPlaying())
		{
			return;
		}
		Self->LastClassification = Result.bSuccess ? Classification : FPipelineClassification();
		Self->FinishCall(Result);
	});
}

// ---------------------------------------------------------------------------------------------------- overlay

FString AModelPipelinesSceneClassifierSample::GetModelSlug() const
{
	return TEXT("resnet-50");
}

FString AModelPipelinesSceneClassifierSample::GetReadyHint() const
{
	return TEXT("Ready: fly around (WASD + mouse), the view is classified every second");
}

FText AModelPipelinesSceneClassifierSample::GetOverlayTitle() const
{
	return FText::FromString(TEXT("Scene classifier (resnet-50)"));
}

TSharedRef<SWidget> AModelPipelinesSceneClassifierSample::BuildOverlay()
{
	// Preview of the capture: a Slate brush on the render target (updated whenever the capture renders).
	PreviewBrush = MakeShared<FSlateBrush>();
	PreviewBrush->SetResourceObject(RenderTarget);
	PreviewBrush->ImageSize = FVector2D(256.0, 256.0);
	PreviewBrush->DrawAs = ESlateBrushDrawType::Image;

	// The widget holds its own reference to the brush, so it stays valid as long as the widget draws it.
	const TSharedPtr<FSlateBrush> Brush = PreviewBrush;
	// Narrow left column (VisionPanelWidth): the panel must not cover the photo board in the middle of the view.
	using namespace UE::ModelPipelines::Samples;
	return MakePanel(GetOverlayTitle(),
		SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().HAlign(HAlign_Left)
		[
			SNew(SBox)
			.WidthOverride(256.f)
			.HeightOverride(256.f)
			[
				SNew(SImage)
				.Image(TAttribute<const FSlateBrush*>::CreateLambda([Brush]() -> const FSlateBrush* { return Brush.Get(); }))
			]
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 12.f, 0.f, 0.f)
		[
			SAssignNew(LabelList, SVerticalBox)
		],
		MakeStatusAttribute(this), VisionPanelWidth);
}

void AModelPipelinesSceneClassifierSample::RefreshOverlay()
{
	using namespace UE::ModelPipelines::Samples;
	if (!LabelList.IsValid())
	{
		return;
	}
	LabelList->ClearChildren();
	for (const FPipelineLabelScore& Entry : LastClassification.TopK)
	{
		LabelList->AddSlot().AutoHeight().Padding(0.f, 2.f)
		[
			MakeTextLine(FText::FromString(FString::Printf(TEXT("%s   %.1f%%"), *Entry.Label, Entry.Score * 100.f)))
		];
	}
}
