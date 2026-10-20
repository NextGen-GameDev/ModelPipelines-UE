// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectPtr.h"
#include "PipelineResults.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesSentimentSample.generated.h"

class SEditableTextBox;
class SVerticalBox;
class UTextClassificationPipeline;

/**
 * Sentiment meter sample: classifies each chat message as POSITIVE or NEGATIVE (text classification) and moves a meter
 * from red to green. Model: distilbert-sst2.
 */
UCLASS(Blueprintable)
class MODELPIPELINESSAMPLES_API AModelPipelinesSentimentSample : public AModelPipelinesSampleBase
{
	GENERATED_BODY()

public:
	AModelPipelinesSentimentSample();

	/** Classifies Message. Empty/whitespace message: no call, status "Type a message". While busy the message waits in the
	    pending slot; a newer message replaces it. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Sample")
	void Analyze(const FString& Message);

	/** Score (0..1) of the label named "POSITIVE" in the last result; for 2-label models without that name, label 1; else the top score. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	float GetLastPositiveScore() const;

	/** Top label of the last result (e.g. "POSITIVE"); empty before the first result. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	FString GetLastLabel() const;

protected:
	virtual FPipelineResult CreatePipeline(UPipelineModelPackage* Loaded) override;
	virtual TSharedRef<SWidget> BuildOverlay() override;
	virtual FString GetModelSlug() const override;
	virtual FString GetReadyHint() const override;
	virtual FText GetOverlayTitle() const override;
	virtual TSharedPtr<SWidget> GetWidgetToFocus() const override;
	virtual void RefreshOverlay() override;

private:
	void StartAnalyzeCall(const FString& Message);
	/** The positive-class score of one classification (see GetLastPositiveScore). */
	float ComputePositiveScore(const FPipelineClassification& Classification) const;

	UPROPERTY(Transient)
	TObjectPtr<UTextClassificationPipeline> Pipeline;

	float LastPositiveScore = 0.f;
	FString LastLabel;

	/** One analyzed chat message, for the overlay history. */
	struct FHistoryEntry
	{
		FString Message;
		FString Label;
		float PositiveScore = 0.f;
	};
	/** The last messages, newest first (at most MaxHistory). */
	TArray<FHistoryEntry> History;
	static constexpr int32 MaxHistory = 5;

	TSharedPtr<SEditableTextBox> ChatBox;
	TSharedPtr<SVerticalBox> HistoryList;
};
