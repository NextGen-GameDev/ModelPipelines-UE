// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Engine/StreamableManager.h"
#include "GameFramework/Actor.h"
#include "Templates/Function.h"
#include "Templates/UniquePtr.h"
#include "UObject/SoftObjectPtr.h"
#include "PipelineModelPackage.h"
#include "PipelineTypes.h"
#include "ModelPipelinesSampleBase.generated.h"

class SWidget;

/** Where a sample is in its life: loading the package, unusable (missing package / creator failed), idle, or running a call. */
UENUM(BlueprintType)
enum class EPipelineSampleState : uint8
{
	/** The model package is being loaded (async). */
	Loading,
	/** The Package path is unset or does not resolve to a model package; the status text says how to import it. */
	MissingPackage,
	/** The package loaded but the pipeline could not be created (e.g. a package for another task). */
	Failed,
	/** Idle: the next request runs immediately. */
	Ready,
	/** A call is in flight; a new request waits in the single pending slot (latest wins). */
	Busy
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FPipelineSampleResultEvent, const FPipelineResult&, Result);

/**
 * Shared base of the Model Pipelines sample actors. It loads the model package asynchronously, lets the subclass create its
 * pipeline, runs at most one call at a time (a request made while busy waits in one "latest wins" slot), reports every
 * finished call through OnSampleResult and shows an optional Slate overlay. Drop a subclass into a level and press Play.
 */
UCLASS(Abstract)
class MODELPIPELINESSAMPLES_API AModelPipelinesSampleBase : public AActor
{
	GENERATED_BODY()

public:
	AModelPipelinesSampleBase();

	/** Package to use. Default set by each sample to /Game/Models/<slug>.<slug> (importer default destination). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample")
	TSoftObjectPtr<UPipelineModelPackage> Package;

	/** Show the Slate overlay (off in tests; also skipped when there is no game viewport). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample")
	bool bShowOverlay = true;

	/** Fires on the game thread after every finished sample call (success or failure). */
	UPROPERTY(BlueprintAssignable, Category = "Model Pipelines|Sample")
	FPipelineSampleResultEvent OnSampleResult;

	/** Where the sample is: Loading, MissingPackage, Failed, Ready (idle) or Busy (a call is in flight). */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	EPipelineSampleState GetSampleState() const;

	/** One line shown in the overlay: loading / ready / busy / last error / missing-package hint. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	FString GetStatusText() const;

	/** The result of the last finished call (default-constructed until the first call finishes). */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	FPipelineResult GetLastResult() const;

protected:
	/** Async-loads Package (FStreamableManager), never a blocking load; adds the overlay. */
	virtual void BeginPlay() override;

	/** Removes the overlay, cancels the load; in-flight results are ignored from now on. */
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;

	/** Subclass: create its pipeline from the loaded package (game thread). Return the creator's result. */
	virtual FPipelineResult CreatePipeline(UPipelineModelPackage* Loaded) PURE_VIRTUAL(AModelPipelinesSampleBase::CreatePipeline, return FPipelineResult(););

	/** Subclass: called once when the pipeline is ready (e.g. embed the corpus with RunOrQueue). */
	virtual void OnPipelineReady() {}

	/** Subclass: build its Slate content (called only when bShowOverlay and a game viewport exist). Default: title + status. */
	virtual TSharedRef<SWidget> BuildOverlay();

	/** Slug used in the missing-package hint, e.g. "resnet-50". */
	virtual FString GetModelSlug() const PURE_VIRTUAL(AModelPipelinesSampleBase::GetModelSlug, return FString(););

	/** Status text shown when Package does not resolve (PackagePath = the soft path or "(none)"). Default: the
	    fetch_reference_models.py hint with GetModelSlug(). */
	virtual FString GetMissingPackageText(const FString& PackagePath) const;

	/** Sets LastResult and the state (Ready), starts the pending request if any, broadcasts OnSampleResult, refreshes the overlay.
	    Ignored after EndPlay. Call it from the pipeline callback of every call started by RunOrQueue. */
	void FinishCall(const FPipelineResult& Result);

	// ------------------------------------------------------------------ helpers for subclasses

	/**
	 * Runs StartCall now when Ready (state becomes Busy), or keeps it as the single pending request while Loading/Busy
	 * (a newer request replaces an older one: latest wins). StartCall must start exactly one async call whose callback ends
	 * in FinishCall. On an unusable sample (MissingPackage/Failed) the request is dropped with a log line.
	 */
	void RunOrQueue(TFunction<void()> StartCall);

	/** FinishCall on the next game-thread tick (for failures detected before any pipeline call starts). */
	void FinishCallNextTick(const FPipelineResult& Result);

	/** Replaces the status line (e.g. "Type a question"). */
	void SetStatusText(const FString& InText);

	/** Shows an input hint (e.g. "Type a question") only while Ready or Busy, so it never replaces the missing-package or
	    creator-failure text a user needs to fix the sample. */
	void SetInputHint(const FString& InHint);

	/** Status shown while Ready. Default: "Ready". */
	virtual FString GetReadyHint() const;

	/** Overlay title. Default: the class display name and the model slug. */
	virtual FText GetOverlayTitle() const;

	/** Text samples return their text box: the base then switches to FInputModeGameAndUI with a mouse cursor and focuses
	    it. Default: null (vision samples keep game input). */
	virtual TSharedPtr<SWidget> GetWidgetToFocus() const { return nullptr; }

	/** Called after every FinishCall while the overlay is shown: rebuild dynamic lists here. */
	virtual void RefreshOverlay() {}

	/** The dark translucent panel with title, Body and the status line; for BuildOverlay overrides. */
	TSharedRef<SWidget> MakeOverlayPanel(TSharedRef<SWidget> Body) const;

	/** True between BeginPlay and EndPlay. Pipeline callbacks check it (with a weak pointer) before touching the actor. */
	bool IsPlaying() const { return bPlaying; }

	/** The overlay widget, valid while shown. */
	TSharedPtr<SWidget> OverlayWidget;

private:
	void OnPackageLoaded();
	void ReportMissingPackage();
	void SetState(EPipelineSampleState InState, const FString& InStatus);
	void ShowOverlay();
	void HideOverlay();
	FStreamableManager& GetStreamableManager();

	EPipelineSampleState State = EPipelineSampleState::Loading;
	FString StatusText;
	FPipelineResult LastResult;
	bool bPlaying = false;
	/** What ShowOverlay changed on the player controller, so HideOverlay restores only that (and the cursor's old value). */
	bool bChangedInputMode = false;
	bool bChangedMouseCursor = false;
	bool bPrevShowMouseCursor = false;

	/** The one "latest wins" pending request. */
	TFunction<void()> PendingCall;

	/** Keeps the async package load alive; cancelled in EndPlay. */
	TSharedPtr<FStreamableHandle> LoadHandle;

	/** Used only when the asset manager is not initialized (odd setups). */
	TUniquePtr<FStreamableManager> FallbackStreamable;
};
