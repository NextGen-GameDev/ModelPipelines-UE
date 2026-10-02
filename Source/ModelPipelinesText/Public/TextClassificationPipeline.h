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
#include "TextClassificationPipeline.generated.h"

class UTokenizerWrapper;
class UPipelineModelPackage;

namespace UE::ModelPipelines::Text
{
	struct FPipelineTextState;
}

/** Game thread, exactly once per ClassifyAsync call. Results: one per text, input order; empty on failure. */
using FPipelineClassificationCallback = TUniqueFunction<void(const FPipelineResult& Result, const TArray<FPipelineClassification>& Results)>;

/**
 * Text classification pipeline (docs/contracts.md C20): tokenize the batch, run the model, apply the package's score
 * activation (softmax or sigmoid) per row and pick the top-k labels.
 */
UCLASS(BlueprintType)
class MODELPIPELINESTEXT_API UTextClassificationPipeline : public UPipelineBase
{
	GENERATED_BODY()

public:
	/** PipelineTask = TextClassification. */
	UTextClassificationPipeline();

	/** Game thread. TopK < 1 -> 1; TopK > number of labels -> number of labels. Callback: game thread, exactly once,
	    never inside the call. Null callback -> Warning, nothing runs. */
	void ClassifyAsync(const TArray<FString>& Texts, int32 TopK, FPipelineClassificationCallback OnComplete);

	/** Tests and tools: blocking, any thread except the executor worker. OutResults is emptied on failure. */
	FPipelineResult ClassifyBlocking(const TArray<FString>& Texts, int32 TopK, TArray<FPipelineClassification>& OutResults);

	/** Releases the pipeline's reference to the shared text state, then Super::BeginDestroy(). */
	virtual void BeginDestroy() override;

protected:
	/** Loads the tokenizer from the package's TokenizerJson. */
	virtual FPipelineResult OnInitialize(UPipelineModelPackage* InPackage) override;

private:
	/** Builds the three stages; Results receives the result of Postprocess. */
	FPipelineStages MakeStages(const TArray<FString>& Texts, int32 TopK,
		const TSharedRef<TArray<FPipelineClassification>, ESPMode::ThreadSafe>& Results);

	/** Keeps the tokenizer reachable for GC too (the shared state holds a TStrongObjectPtr to it). */
	UPROPERTY()
	TObjectPtr<UTokenizerWrapper> Tokenizer;

	/** Shared with in-flight calls. */
	TSharedPtr<UE::ModelPipelines::Text::FPipelineTextState, ESPMode::ThreadSafe> TextState;
};
