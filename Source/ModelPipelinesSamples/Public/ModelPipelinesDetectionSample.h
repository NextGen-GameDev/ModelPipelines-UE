// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectPtr.h"
#include "PipelineResults.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesDetectionSample.generated.h"

class SSampleDetectionOverlay;
class UObjectDetectionPipeline;
class USceneCaptureComponent2D;
class UTextureRenderTarget2D;

/**
 * Object detection sample: a scene capture with the viewport's aspect and the player view's horizontal FOV follows the camera; every
 * IntervalSeconds it runs the detector and the boxes are drawn over the view with "label 0.87". Model: rtdetr_r18vd.
 * Game input stays on.
 */
UCLASS(Blueprintable)
class MODELPIPELINESSAMPLES_API AModelPipelinesDetectionSample : public AModelPipelinesSampleBase
{
	GENERATED_BODY()

public:
	AModelPipelinesDetectionSample();

	/** Renders the view that is searched for objects (on demand only, right before each call). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Model Pipelines|Sample")
	TObjectPtr<USceneCaptureComponent2D> SceneCapture;

	/** Capture width; the height follows the viewport aspect (540 for 960 when there is no viewport). Set at BeginPlay. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample", meta = (ClampMin = "16", ClampMax = "4096"))
	int32 CaptureWidth = 960;

	/** Detections below this score are dropped. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample", meta = (ClampMin = "0", ClampMax = "1"))
	float ScoreThreshold = 0.5f;

	/** Seconds between two automatic detections (Tick). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample", meta = (ClampMin = "0.05"))
	float IntervalSeconds = 0.5f;

	/** Copy the player camera's location, rotation and horizontal view FOV every tick (without a player controller the actor's own transform is used). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample")
	bool bFollowPlayerCamera = true;

	/** Captures the view once and runs the detector. While a call is in flight the request waits in the pending slot (latest wins). */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Sample")
	void DetectNow();

	/** The detections of the last finished call (empty after a failed one); boxes normalized 0-1 to the capture, which matches the view. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	TArray<FPipelineDetection> GetLastDetections() const;

	/** The capture's render target (null before BeginPlay). */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	UTextureRenderTarget2D* GetRenderTarget() const;

	virtual void Tick(float DeltaSeconds) override;

protected:
	virtual void BeginPlay() override;
	virtual FPipelineResult CreatePipeline(UPipelineModelPackage* Loaded) override;
	virtual TSharedRef<SWidget> BuildOverlay() override;
	virtual FString GetModelSlug() const override;
	virtual FString GetReadyHint() const override;
	virtual FText GetOverlayTitle() const override;
	virtual void RefreshOverlay() override;

private:
	/** Capture + DetectAsync; the callback ends in FinishCall. */
	void StartDetectCall();
	void FollowPlayerCamera();

	UPROPERTY(Transient)
	TObjectPtr<UObjectDetectionPipeline> Pipeline;

	UPROPERTY(Transient)
	TObjectPtr<UTextureRenderTarget2D> RenderTarget;

	TArray<FPipelineDetection> LastDetections;
	float SecondsSinceLastCall = 0.f;

	TSharedPtr<SSampleDetectionOverlay> BoxOverlay;
};
