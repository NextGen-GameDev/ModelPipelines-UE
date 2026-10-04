// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Engine/CancellableAsyncAction.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "VisionPipelineAsyncActions.generated.h"

class UImageClassificationPipeline;
class UObjectDetectionPipeline;
class UTexture;

/** Output pin of Classify Image. Result: what the pipeline reported. Classification: default (no scores, no labels) on failure. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FPipelineClassifyImagePin,
	const FPipelineResult&, Result, const FPipelineClassification&, Classification);

/** Output pin of Detect Objects. Result: what the pipeline reported. Detections: boxes normalized to the image; empty on failure. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FPipelineDetectObjectsPin,
	const FPipelineResult&, Result, const TArray<FPipelineDetection>&, Detections);

namespace UE::ModelPipelines::Vision
{
	/** Life cycle of one vision async action (docs/contracts.md C34). */
	enum class EAsyncActionState : uint8
	{
		/** Created, not activated. */
		Idle,
		/** Activated, result not delivered yet. */
		Running,
		/** A pin fired. */
		Delivered,
		/** Cancelled before delivery; the result is dropped when it arrives. */
		Cancelled
	};
}

/**
 * Blueprint latent node "Classify Image" (docs/contracts.md C34): wraps UImageClassificationPipeline::ClassifyAsync.
 * Exactly one of OnSuccess / OnFailure fires, once, on the game thread, never inside Activate. The action keeps itself,
 * the pipeline and the image alive until the call's result arrives, then releases them.
 */
UCLASS()
class MODELPIPELINESVISION_API UClassifyImageAsyncAction : public UCancellableAsyncAction
{
	GENERATED_BODY()

public:
	/** Fires once on the game thread when the call succeeded. */
	UPROPERTY(BlueprintAssignable)
	FPipelineClassifyImagePin OnSuccess;

	/** Fires once on the game thread when the call failed. Result says why; Classification is default. */
	UPROPERTY(BlueprintAssignable)
	FPipelineClassifyImagePin OnFailure;

	/** Classifies the image (Texture 2D or Render Target 2D, read back from the GPU in the background) and keeps the TopK
	    best labels (TopK below 1 gives 1, above the label count gives all labels). A null or not initialized pipeline,
	    or an unsupported image, ends in On Failure. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Vision",
		meta = (BlueprintInternalUseOnly = "true", WorldContext = "WorldContextObject", DisplayName = "Classify Image"))
	static UClassifyImageAsyncAction* ClassifyImage(UObject* WorldContextObject, UImageClassificationPipeline* Pipeline,
		UTexture* Image, int32 TopK = 5);

	/** Game thread. Starts the one call; a second Activate logs a Warning and does nothing. */
	virtual void Activate() override;

	/** Game thread. Before delivery: no pin fires afterwards (the running call finishes and its result is dropped).
	    After delivery or a second time: no-op. */
	virtual void Cancel() override;

	/** True from Activate until a pin fired or Cancel was called. */
	virtual bool IsActive() const override;

	/** Same as IsActive(). */
	virtual bool ShouldBroadcastDelegates() const override;

private:
	/** Game thread: the pipeline's (or the deferred null-pipeline) result. */
	void HandleResult(const FPipelineResult& Result, const FPipelineClassification& Classification);

	/** Kept alive with the action for the in-flight call; released after delivery. */
	UPROPERTY()
	TObjectPtr<UImageClassificationPipeline> Pipeline;

	/** Kept alive with the action for the in-flight call (GPU readback); released after delivery. */
	UPROPERTY()
	TObjectPtr<UTexture> Image;

	int32 TopK = 5;

	UE::ModelPipelines::Vision::EAsyncActionState State = UE::ModelPipelines::Vision::EAsyncActionState::Idle;

	/** Activate started a call whose result has not arrived yet (the action is rooted while this is set). */
	bool bInFlight = false;
};

/**
 * Blueprint latent node "Detect Objects" (docs/contracts.md C34): wraps UObjectDetectionPipeline::DetectAsync.
 * Exactly one of OnSuccess / OnFailure fires, once, on the game thread, never inside Activate. The action keeps itself,
 * the pipeline and the image alive until the call's result arrives, then releases them.
 */
UCLASS()
class MODELPIPELINESVISION_API UDetectObjectsAsyncAction : public UCancellableAsyncAction
{
	GENERATED_BODY()

public:
	/** Fires once on the game thread when the call succeeded. Detections: boxes normalized to the image. */
	UPROPERTY(BlueprintAssignable)
	FPipelineDetectObjectsPin OnSuccess;

	/** Fires once on the game thread when the call failed. Result says why; Detections is empty. */
	UPROPERTY(BlueprintAssignable)
	FPipelineDetectObjectsPin OnFailure;

	/** Detects objects in the image (Texture 2D or Render Target 2D, read back from the GPU in the background) and keeps
	    detections with a score above ScoreThreshold (clamped to 0..1). A null or not initialized pipeline, an unsupported
	    image or a NaN threshold ends in On Failure. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Vision",
		meta = (BlueprintInternalUseOnly = "true", WorldContext = "WorldContextObject", DisplayName = "Detect Objects"))
	static UDetectObjectsAsyncAction* DetectObjects(UObject* WorldContextObject, UObjectDetectionPipeline* Pipeline,
		UTexture* Image, float ScoreThreshold = 0.5f);

	/** Game thread. Starts the one call; a second Activate logs a Warning and does nothing. */
	virtual void Activate() override;

	/** Game thread. Before delivery: no pin fires afterwards (the running call finishes and its result is dropped).
	    After delivery or a second time: no-op. */
	virtual void Cancel() override;

	/** True from Activate until a pin fired or Cancel was called. */
	virtual bool IsActive() const override;

	/** Same as IsActive(). */
	virtual bool ShouldBroadcastDelegates() const override;

private:
	/** Game thread: the pipeline's (or the deferred null-pipeline) result. */
	void HandleResult(const FPipelineResult& Result, const TArray<FPipelineDetection>& Detections);

	/** Kept alive with the action for the in-flight call; released after delivery. */
	UPROPERTY()
	TObjectPtr<UObjectDetectionPipeline> Pipeline;

	/** Kept alive with the action for the in-flight call (GPU readback); released after delivery. */
	UPROPERTY()
	TObjectPtr<UTexture> Image;

	float ScoreThreshold = 0.5f;

	UE::ModelPipelines::Vision::EAsyncActionState State = UE::ModelPipelines::Vision::EAsyncActionState::Idle;

	/** Activate started a call whose result has not arrived yet (the action is rooted while this is set). */
	bool bInFlight = false;
};
