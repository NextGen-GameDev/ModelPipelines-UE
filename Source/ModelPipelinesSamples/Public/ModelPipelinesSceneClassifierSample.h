// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectPtr.h"
#include "PipelineResults.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesSceneClassifierSample.generated.h"

class SVerticalBox;
class UImageClassificationPipeline;
class USceneCaptureComponent2D;
class UTextureRenderTarget2D;
struct FSlateBrush;

/**
 * Scene classifier sample: a scene capture follows the player camera and, every IntervalSeconds, classifies what the player
 * sees (image classification on the capture's render target). The overlay shows a small live preview of the capture and the
 * top-K labels. Model: resnet-50. Game input stays on.
 */
UCLASS(Blueprintable)
class MODELPIPELINESSAMPLES_API AModelPipelinesSceneClassifierSample : public AModelPipelinesSampleBase
{
	GENERATED_BODY()

public:
	AModelPipelinesSceneClassifierSample();

	/** Renders the view that is classified (on demand only, right before each call). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Model Pipelines|Sample")
	TObjectPtr<USceneCaptureComponent2D> SceneCapture;

	/** Width and height of the square capture render target, created at BeginPlay. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample", meta = (ClampMin = "16", ClampMax = "4096"))
	int32 CaptureResolution = 512;

	/** Seconds between two automatic classifications (Tick). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample", meta = (ClampMin = "0.05"))
	float IntervalSeconds = 1.0f;

	/** How many labels a classification returns. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample", meta = (ClampMin = "1"))
	int32 TopK = 3;

	/** Copy the player camera's location, rotation and horizontal view FOV (from the player's projection; else the camera FOV) every tick
	    (without a player controller the actor's own transform is used). The capture stays square: it classifies the centre of the view. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample")
	bool bFollowPlayerCamera = true;

	/** Captures the view once and classifies it. While a call is in flight the request waits in the pending slot (latest wins). */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Sample")
	void ClassifyNow();

	/** The classification of the last finished call (empty before the first call and after a failed call). */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	FPipelineClassification GetLastClassification() const;

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
	/** Capture + ClassifyAsync; the callback ends in FinishCall. */
	void StartClassifyCall();
	void FollowPlayerCamera();

	UPROPERTY(Transient)
	TObjectPtr<UImageClassificationPipeline> Pipeline;

	UPROPERTY(Transient)
	TObjectPtr<UTextureRenderTarget2D> RenderTarget;

	FPipelineClassification LastClassification;
	float SecondsSinceLastCall = 0.f;

	/** Brush on the render target for the overlay preview (shared with the widget, which may outlive a frame of the actor). */
	TSharedPtr<FSlateBrush> PreviewBrush;
	TSharedPtr<SVerticalBox> LabelList;
};
