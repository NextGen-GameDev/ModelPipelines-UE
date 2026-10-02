// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineLibrary.h"

#include "Math/UnrealMathUtility.h"
#include "PipelineLog.h"
#include "PipelineMath.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(PipelineLibrary)

float UPipelineLibrary::CosineSimilarity(const FPipelineEmbedding& A, const FPipelineEmbedding& B)
{
	if (A.Values.Num() == 0 || B.Values.Num() == 0 || A.Values.Num() != B.Values.Num())
	{
		UE_LOG(LogModelPipelines, Warning, TEXT("CosineSimilarity: embeddings of length %d and %d cannot be compared; returning 0."),
			A.Values.Num(), B.Values.Num());
		return 0.f;
	}
	return (float)UE::ModelPipelines::CosineSimilarity(A.Values, B.Values);
}

TArray<FPipelineSimilarityMatch> UPipelineLibrary::TopKSimilar(const FPipelineEmbedding& Query, const TArray<FPipelineEmbedding>& Corpus, int32 K)
{
	TArray<FPipelineSimilarityMatch> Result;
	if (K <= 0 || Query.Values.Num() == 0 || Corpus.Num() == 0)
	{
		return Result;
	}

	struct FScored
	{
		int32 Index;
		double Score;
	};

	TArray<FScored> Scored;
	Scored.Reserve(Corpus.Num());
	int32 Skipped = 0;
	for (int32 i = 0; i < Corpus.Num(); ++i)
	{
		if (Corpus[i].Values.Num() != Query.Values.Num())
		{
			++Skipped;
			continue;
		}
		Scored.Add({i, UE::ModelPipelines::CosineSimilarity(Query.Values, Corpus[i].Values)});
	}
	if (Skipped > 0)
	{
		UE_LOG(LogModelPipelines, Warning, TEXT("TopKSimilar: skipped %d corpus entries whose length differs from the query's (%d)."),
			Skipped, Query.Values.Num());
	}

	// Strict total order on the double score: non-NaN before NaN, higher first, then lower index.
	Scored.Sort([](const FScored& L, const FScored& R)
	{
		const bool bNaNL = FMath::IsNaN(L.Score);
		const bool bNaNR = FMath::IsNaN(R.Score);
		if (bNaNL != bNaNR)
		{
			return !bNaNL;
		}
		if (!bNaNL && L.Score != R.Score)
		{
			return L.Score > R.Score;
		}
		return L.Index < R.Index;
	});

	const int32 Count = FMath::Min(K, Scored.Num());
	Result.Reserve(Count);
	for (int32 i = 0; i < Count; ++i)
	{
		FPipelineSimilarityMatch& Match = Result.AddDefaulted_GetRef();
		Match.Index = Scored[i].Index;
		Match.Score = (float)Scored[i].Score;
	}
	return Result;
}
