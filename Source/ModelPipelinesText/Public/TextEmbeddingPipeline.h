// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Templates/Function.h"
#include "Templates/SharedPointer.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineBase.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "TextEmbeddingPipeline.generated.h"

class UPipelineTokenizer;
class UPipelineModelPackage;

namespace UE::ModelPipelines::Text
{
	struct FPipelineTextState;
}

/** Game thread, exactly once per EmbedAsync call. Embeddings: one per text, input order; empty on failure. */
using FPipelineEmbeddingCallback = TUniqueFunction<void(const FPipelineResult& Result, const TArray<FPipelineEmbedding>& Embeddings)>;

/**
 * Text embedding pipeline (docs/contracts.md C20): tokenize the batch, run the model, pool (mean over the attention mask,
 * CLS, or none per the package) and optionally L2-normalize.
 */
UCLASS(BlueprintType)
class MODELPIPELINESTEXT_API UTextEmbeddingPipeline : public UPipelineBase
{
	GENERATED_BODY()

public:
	/** PipelineTask = TextEmbedding. */
	UTextEmbeddingPipeline();

	/** Game thread. One embedding per text, same order. Callback: game thread, exactly once, never inside the call.
	    Null callback -> Warning, nothing runs. */
	void EmbedAsync(const TArray<FString>& Texts, FPipelineEmbeddingCallback OnComplete);

	/** Tests and tools: blocking, any thread except the executor worker. The caller keeps the pipeline alive for the
	    duration of the call. OutEmbeddings is emptied on failure. */
	FPipelineResult EmbedBlocking(const TArray<FString>& Texts, TArray<FPipelineEmbedding>& OutEmbeddings);

protected:
	/** Loads the tokenizer from the package's TokenizerJson. */
	virtual FPipelineResult OnInitialize(UPipelineModelPackage* InPackage) override;

private:
	/** Builds the three stages from State (null: the stages fail with InvalidArgument); Embeddings receives the result of Postprocess. */
	static FPipelineStages MakeStages(const TSharedPtr<UE::ModelPipelines::Text::FPipelineTextState, ESPMode::ThreadSafe>& State,
		const TArray<FString>& Texts, const TSharedRef<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe>& Embeddings);

	/** Keeps the tokenizer reachable for GC too (the shared state holds a TStrongObjectPtr to it). */
	UPROPERTY()
	TObjectPtr<UPipelineTokenizer> Tokenizer;

	/** Shared with in-flight calls. Read only on the game thread (Async) or by a blocking caller that keeps the pipeline
	    alive; released by the destructor. */
	TSharedPtr<UE::ModelPipelines::Text::FPipelineTextState, ESPMode::ThreadSafe> TextState;
};
