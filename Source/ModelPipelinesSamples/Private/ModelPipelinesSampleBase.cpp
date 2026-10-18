// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesSampleBase.h"

#include "Async/Async.h"
#include "Engine/AssetManager.h"
#include "Engine/GameViewportClient.h"
#include "Engine/World.h"
#include "Framework/Application/SlateApplication.h"
#include "GameFramework/PlayerController.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "Widgets/SNullWidget.h"
#include "SampleOverlayWidgets.h"
#include "SamplesLog.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(ModelPipelinesSampleBase)

AModelPipelinesSampleBase::AModelPipelinesSampleBase()
{
	// Samples are event driven (load completion, pipeline callbacks, text box commits); they never tick.
	PrimaryActorTick.bCanEverTick = false;
}

EPipelineSampleState AModelPipelinesSampleBase::GetSampleState() const
{
	return State;
}

FString AModelPipelinesSampleBase::GetStatusText() const
{
	return StatusText;
}

FPipelineResult AModelPipelinesSampleBase::GetLastResult() const
{
	return LastResult;
}

// ---------------------------------------------------------------------------------------------------- play

void AModelPipelinesSampleBase::BeginPlay()
{
	Super::BeginPlay();
	bPlaying = true;
	PendingCall.Reset();

	const FString Path = Package.IsNull() ? FString(TEXT("(none)")) : Package.ToSoftObjectPath().ToString();
	SetState(EPipelineSampleState::Loading, FString::Printf(TEXT("Loading %s ..."), *Path));
	ShowOverlay();

	if (Package.IsNull())
	{
		ReportMissingPackage();
		return;
	}

	// Never a blocking load: the streamable manager loads in the background (or finds the package in memory) and calls
	// back on a later game-thread tick. The weak lambda is skipped when this actor is gone.
	LoadHandle = GetStreamableManager().RequestAsyncLoad(Package.ToSoftObjectPath(),
		FStreamableDelegate::CreateWeakLambda(this, [this]() { OnPackageLoaded(); }));
	if (!LoadHandle.IsValid() && State == EPipelineSampleState::Loading)
	{
		ReportMissingPackage();
	}
}

void AModelPipelinesSampleBase::EndPlay(const EEndPlayReason::Type Reason)
{
	bPlaying = false;
	PendingCall.Reset();
	if (LoadHandle.IsValid())
	{
		LoadHandle->CancelHandle();
		LoadHandle.Reset();
	}
	HideOverlay();
	Super::EndPlay(Reason);
}

void AModelPipelinesSampleBase::OnPackageLoaded()
{
	LoadHandle.Reset();
	if (!bPlaying || State != EPipelineSampleState::Loading)
	{
		return;
	}

	// Get() resolves the soft pointer to the object now in memory; null if the path was missing or of another class.
	UPipelineModelPackage* Loaded = Package.Get();
	if (!Loaded)
	{
		ReportMissingPackage();
		return;
	}

	const FPipelineResult Created = CreatePipeline(Loaded);
	if (!Created.bSuccess)
	{
		PendingCall.Reset();
		UE_LOG(LogModelPipelinesSamples, Warning, TEXT("%s: could not create the pipeline from %s: %s"), *GetName(), *Loaded->GetPathName(), *Created.Message);
		SetState(EPipelineSampleState::Failed, FString::Printf(TEXT("Could not create the pipeline: %s"), *Created.Message));
		return;
	}

	SetState(EPipelineSampleState::Ready, GetReadyHint());
	OnPipelineReady();

	// A request made while loading runs now (unless OnPipelineReady started a call; then it waits for that one).
	if (State == EPipelineSampleState::Ready && PendingCall)
	{
		TFunction<void()> Call = MoveTemp(PendingCall);
		PendingCall.Reset();
		RunOrQueue(MoveTemp(Call));
	}
}

void AModelPipelinesSampleBase::ReportMissingPackage()
{
	PendingCall.Reset();
	const FString Path = Package.IsNull() ? FString(TEXT("(none)")) : Package.ToSoftObjectPath().ToString();
	const FString Text = GetMissingPackageText(Path);
	UE_LOG(LogModelPipelinesSamples, Warning, TEXT("%s: %s"), *GetName(), *Text);
	SetState(EPipelineSampleState::MissingPackage, Text);
}

FString AModelPipelinesSampleBase::GetMissingPackageText(const FString& PackagePath) const
{
	return FString::Printf(TEXT("Model package not found: %s. Run Plugins/ModelPipelines/Tools/fetch_reference_models.py --only %s, import the folder into /Game/Models, then press Play again. See Docs/Samples.md."),
		*PackagePath, *GetModelSlug());
}

// ---------------------------------------------------------------------------------------------------- calls

void AModelPipelinesSampleBase::RunOrQueue(TFunction<void()> StartCall)
{
	if (!bPlaying || !StartCall)
	{
		return;
	}
	switch (State)
	{
	case EPipelineSampleState::Ready:
		SetState(EPipelineSampleState::Busy, TEXT("Working..."));
		StartCall();
		break;
	case EPipelineSampleState::Loading:
	case EPipelineSampleState::Busy:
		// One pending slot: a newer request replaces the waiting one (latest wins), so calls never stack up.
		PendingCall = MoveTemp(StartCall);
		break;
	default:
		UE_LOG(LogModelPipelinesSamples, Log, TEXT("%s: request ignored, the sample has no pipeline (%s)"), *GetName(), *StatusText);
		break;
	}
}

void AModelPipelinesSampleBase::FinishCall(const FPipelineResult& Result)
{
	if (!bPlaying)
	{
		return;
	}
	LastResult = Result;
	if (Result.bSuccess)
	{
		const float TotalMs = Result.QueueMs + Result.PreprocessMs + Result.InferenceMs + Result.PostprocessMs;
		SetState(EPipelineSampleState::Ready, FString::Printf(TEXT("%s (last call %.1f ms on %s)"), *GetReadyHint(), TotalMs,
			*UEnum::GetDisplayValueAsText(Result.Device).ToString()));
	}
	else
	{
		UE_LOG(LogModelPipelinesSamples, Warning, TEXT("%s: call failed: %s"), *GetName(), *Result.Message);
		SetState(EPipelineSampleState::Ready, FString::Printf(TEXT("Error: %s"), *Result.Message));
	}

	// Start the waiting request before broadcasting, so a listener that makes a new request queues it instead of running
	// a second call next to the pending one.
	if (PendingCall)
	{
		TFunction<void()> Call = MoveTemp(PendingCall);
		PendingCall.Reset();
		RunOrQueue(MoveTemp(Call));
	}

	OnSampleResult.Broadcast(Result);
	if (bPlaying && OverlayWidget.IsValid())
	{
		RefreshOverlay();
	}
}

void AModelPipelinesSampleBase::FinishCallNextTick(const FPipelineResult& Result)
{
	AsyncTask(ENamedThreads::GameThread, [WeakThis = TWeakObjectPtr<AModelPipelinesSampleBase>(this), Result]()
	{
		if (AModelPipelinesSampleBase* Self = WeakThis.Get())
		{
			Self->FinishCall(Result);
		}
	});
}

void AModelPipelinesSampleBase::SetStatusText(const FString& InText)
{
	StatusText = InText;
}

void AModelPipelinesSampleBase::SetInputHint(const FString& InHint)
{
	if (State == EPipelineSampleState::Ready || State == EPipelineSampleState::Busy)
	{
		StatusText = InHint;
	}
}

void AModelPipelinesSampleBase::SetState(EPipelineSampleState InState, const FString& InStatus)
{
	State = InState;
	StatusText = InStatus;
}

FString AModelPipelinesSampleBase::GetReadyHint() const
{
	return TEXT("Ready");
}

FText AModelPipelinesSampleBase::GetOverlayTitle() const
{
	return FText::FromString(GetModelSlug());
}

FStreamableManager& AModelPipelinesSampleBase::GetStreamableManager()
{
	if (UAssetManager::IsInitialized())
	{
		return UAssetManager::GetStreamableManager();
	}
	if (!FallbackStreamable.IsValid())
	{
		FallbackStreamable = MakeUnique<FStreamableManager>();
	}
	return *FallbackStreamable;
}

// ---------------------------------------------------------------------------------------------------- overlay

TSharedRef<SWidget> AModelPipelinesSampleBase::BuildOverlay()
{
	return MakeOverlayPanel(SNullWidget::NullWidget);
}

TSharedRef<SWidget> AModelPipelinesSampleBase::MakeOverlayPanel(TSharedRef<SWidget> Body) const
{
	// The status line reads the actor every frame; the weak lambda returns empty text once the actor is gone.
	const TAttribute<FText> Status = TAttribute<FText>::CreateLambda([WeakThis = TWeakObjectPtr<const AModelPipelinesSampleBase>(this)]()
	{
		const AModelPipelinesSampleBase* Self = WeakThis.Get();
		return Self ? FText::FromString(Self->StatusText) : FText::GetEmpty();
	});
	return UE::ModelPipelines::Samples::MakePanel(GetOverlayTitle(), Body, Status);
}

void AModelPipelinesSampleBase::ShowOverlay()
{
	UWorld* World = GetWorld();
	UGameViewportClient* Viewport = World ? World->GetGameViewport() : nullptr;
	if (!bShowOverlay || !Viewport || !FSlateApplication::IsInitialized())
	{
		return;   // headless, tests, dedicated server: no overlay
	}

	OverlayWidget = BuildOverlay();
	Viewport->AddViewportWidgetContent(OverlayWidget.ToSharedRef(), /*ZOrder*/ 10);

	// Text samples type into a box: give it focus and show the mouse cursor. Vision samples keep game input.
	const TSharedPtr<SWidget> Focus = GetWidgetToFocus();
	APlayerController* PC = World->GetFirstPlayerController();
	if (Focus.IsValid() && PC)
	{
		FInputModeGameAndUI Mode;
		Mode.SetWidgetToFocus(Focus);
		Mode.SetHideCursorDuringCapture(false);
		PC->SetInputMode(Mode);
		bChangedInputMode = true;
		bPrevShowMouseCursor = PC->ShouldShowMouseCursor();
		if (!bPrevShowMouseCursor)
		{
			PC->SetShowMouseCursor(true);
			bChangedMouseCursor = true;
		}
	}
	if (Focus.IsValid())
	{
		FSlateApplication::Get().SetKeyboardFocus(Focus, EFocusCause::SetDirectly);
	}
}

void AModelPipelinesSampleBase::HideOverlay()
{
	if (!OverlayWidget.IsValid())
	{
		return;
	}
	UWorld* World = GetWorld();
	if (UGameViewportClient* Viewport = World ? World->GetGameViewport() : nullptr)
	{
		Viewport->RemoveViewportWidgetContent(OverlayWidget.ToSharedRef());
	}
	// Restore only what ShowOverlay changed. The engine has no getter for the previous input mode, so a changed mode goes
	// back to game-only (the mode a level starts in); the cursor gets its previous value back.
	if (bChangedInputMode || bChangedMouseCursor)
	{
		if (APlayerController* PC = World ? World->GetFirstPlayerController() : nullptr)
		{
			if (bChangedInputMode)
			{
				PC->SetInputMode(FInputModeGameOnly());
			}
			if (bChangedMouseCursor)
			{
				PC->SetShowMouseCursor(bPrevShowMouseCursor);
			}
		}
		bChangedInputMode = false;
		bChangedMouseCursor = false;
	}
	OverlayWidget.Reset();
}
