// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectPtr.h"
#include "PipelineResults.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesSemanticSearchSample.generated.h"

class SEditableTextBox;
class SVerticalBox;
class UTextEmbeddingPipeline;

/**
 * Semantic search sample: embeds the NPC dialogue lines once, then finds the lines closest in meaning to a typed question
 * (text embedding + cosine similarity). Model: all-MiniLM-L6-v2.
 */
UCLASS(Blueprintable)
class MODELPIPELINESSAMPLES_API AModelPipelinesSemanticSearchSample : public AModelPipelinesSampleBase
{
	GENERATED_BODY()

public:
	AModelPipelinesSemanticSearchSample();

	/** The searchable corpus: one NPC line per entry, embedded once when the pipeline is ready. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample")
	TArray<FString> DialogueLines;

	/** How many of the closest lines a search returns. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample", meta = (ClampMin = "1"))
	int32 TopK = 3;

	/** Embeds Query and finds the TopK closest DialogueLines. Empty/whitespace query: no call, status "Type a question".
	    While busy (or before the corpus is ready) the query waits in the pending slot; a newer query replaces it. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Sample")
	void Search(const FString& Query);

	/** The matches of the last finished query, best first (Index into DialogueLines, Score = cosine similarity). */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	TArray<FPipelineSimilarityMatch> GetLastMatches() const;

	/** True once every DialogueLines entry is embedded. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	bool IsCorpusReady() const;

protected:
	virtual FPipelineResult CreatePipeline(UPipelineModelPackage* Loaded) override;
	virtual void OnPipelineReady() override;
	virtual TSharedRef<SWidget> BuildOverlay() override;
	virtual FString GetModelSlug() const override;
	virtual FString GetReadyHint() const override;
	virtual FText GetOverlayTitle() const override;
	virtual TSharedPtr<SWidget> GetWidgetToFocus() const override;
	virtual void RefreshOverlay() override;

private:
	/** Embeds the whole corpus in one call; on success, FollowUpQuery (if any) runs next through RunOrQueue. */
	void StartCorpusCall(const FString& FollowUpQuery = FString());
	/** Embeds one query and ranks the corpus. */
	void StartQueryCall(const FString& Query);

	UPROPERTY(Transient)
	TObjectPtr<UTextEmbeddingPipeline> Pipeline;

	/** One embedding per DialogueLines entry (the lines as they were when embedded). */
	TArray<FPipelineEmbedding> CorpusEmbeddings;
	TArray<FString> CorpusLines;
	bool bCorpusReady = false;

	FString LastQuery;
	TArray<FPipelineSimilarityMatch> LastMatches;

	TSharedPtr<SEditableTextBox> QueryBox;
	TSharedPtr<SVerticalBox> MatchList;
};
