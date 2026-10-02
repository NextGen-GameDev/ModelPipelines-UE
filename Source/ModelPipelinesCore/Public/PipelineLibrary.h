// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "PipelineResults.h"
#include "PipelineLibrary.generated.h"

/** Blueprint-visible pipeline utilities (PLAN section 5). */
UCLASS()
class MODELPIPELINESCORE_API UPipelineLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/** Cosine similarity in [-1, 1]. Different lengths or an empty embedding -> 0 (one Warning). */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Similarity")
	static float CosineSimilarity(const FPipelineEmbedding& A, const FPipelineEmbedding& B);

	/** The K corpus entries most similar to Query, best first (ties: lower index first). K <= 0, empty Query or empty
	    Corpus -> empty. Corpus entries whose length differs from Query's are skipped (one Warning with the count).
	    K larger than the usable entries -> all usable entries. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Similarity")
	static TArray<FPipelineSimilarityMatch> TopKSimilar(const FPipelineEmbedding& Query, const TArray<FPipelineEmbedding>& Corpus, int32 K);
};
