// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Engine/CancellableAsyncAction.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "TextPipelineAsyncActions.generated.h"

class UTextClassificationPipeline;
class UTextEmbeddingPipeline;

/** Output pin of Embed Texts. Result: what the pipeline reported. Embeddings: one per text, same order; empty on failure. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FPipelineEmbedTextsPin,
	const FPipelineResult&, Result, const TArray<FPipelineEmbedding>&, Embeddings);

/** Output pin of Classify Texts. Result: what the pipeline reported. Classifications: one per text, same order; empty on failure. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FPipelineClassifyTextsPin,
	const FPipelineResult&, Result, const TArray<FPipelineClassification>&, Classifications);

namespace UE::ModelPipelines::Text
{
	/** Life cycle of one text async action (docs/contracts.md C34). */
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
 * Blueprint latent node "Embed Texts" (docs/contracts.md C34): wraps UTextEmbeddingPipeline::EmbedAsync. Exactly one of
 * OnSuccess / OnFailure fires, once, on the game thread, never inside Activate. The action keeps itself (and the
 * pipeline) alive until the call's result arrives.
 */
UCLASS()
class MODELPIPELINESTEXT_API UEmbedTextsAsyncAction : public UCancellableAsyncAction
{
	GENERATED_BODY()

public:
	/** Fires once on the game thread when the call succeeded. Embeddings: one per text, same order. */
	UPROPERTY(BlueprintAssignable)
	FPipelineEmbedTextsPin OnSuccess;

	/** Fires once on the game thread when the call failed. Result says why; Embeddings is empty. */
	UPROPERTY(BlueprintAssignable)
	FPipelineEmbedTextsPin OnFailure;

	/** Embeds each text with the pipeline, one vector per text in input order. Runs in the background; a null or not
	    initialized pipeline, or an empty list, ends in On Failure. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Text",
		meta = (BlueprintInternalUseOnly = "true", WorldContext = "WorldContextObject", DisplayName = "Embed Texts"))
	static UEmbedTextsAsyncAction* EmbedTexts(UObject* WorldContextObject, UTextEmbeddingPipeline* Pipeline,
		const TArray<FString>& Texts);

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
	void HandleResult(const FPipelineResult& Result, const TArray<FPipelineEmbedding>& Embeddings);

	/** Kept alive with the action for the in-flight call. */
	UPROPERTY()
	TObjectPtr<UTextEmbeddingPipeline> Pipeline;

	TArray<FString> Texts;

	UE::ModelPipelines::Text::EAsyncActionState State = UE::ModelPipelines::Text::EAsyncActionState::Idle;

	/** Activate started a call whose result has not arrived yet (the action is rooted while this is set). */
	bool bInFlight = false;
};

/**
 * Blueprint latent node "Classify Texts" (docs/contracts.md C34): wraps UTextClassificationPipeline::ClassifyAsync.
 * Exactly one of OnSuccess / OnFailure fires, once, on the game thread, never inside Activate. The action keeps itself
 * (and the pipeline) alive until the call's result arrives.
 */
UCLASS()
class MODELPIPELINESTEXT_API UClassifyTextsAsyncAction : public UCancellableAsyncAction
{
	GENERATED_BODY()

public:
	/** Fires once on the game thread when the call succeeded. Classifications: one per text, same order. */
	UPROPERTY(BlueprintAssignable)
	FPipelineClassifyTextsPin OnSuccess;

	/** Fires once on the game thread when the call failed. Result says why; Classifications is empty. */
	UPROPERTY(BlueprintAssignable)
	FPipelineClassifyTextsPin OnFailure;

	/** Classifies each text with the pipeline and keeps the TopK best labels per text (TopK below 1 gives 1, above the
	    label count gives all labels). Runs in the background; a null or not initialized pipeline, or an empty list, ends
	    in On Failure. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Text",
		meta = (BlueprintInternalUseOnly = "true", WorldContext = "WorldContextObject", DisplayName = "Classify Texts"))
	static UClassifyTextsAsyncAction* ClassifyTexts(UObject* WorldContextObject, UTextClassificationPipeline* Pipeline,
		const TArray<FString>& Texts, int32 TopK = 1);

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
	void HandleResult(const FPipelineResult& Result, const TArray<FPipelineClassification>& Classifications);

	/** Kept alive with the action for the in-flight call. */
	UPROPERTY()
	TObjectPtr<UTextClassificationPipeline> Pipeline;

	TArray<FString> Texts;

	int32 TopK = 1;

	UE::ModelPipelines::Text::EAsyncActionState State = UE::ModelPipelines::Text::EAsyncActionState::Idle;

	/** Activate started a call whose result has not arrived yet (the action is rooted while this is set). */
	bool bInFlight = false;
};
