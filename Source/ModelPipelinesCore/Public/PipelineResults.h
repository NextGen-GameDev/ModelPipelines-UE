// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "PipelineResults.generated.h"

/** One embedding vector (docs/contracts.md C18). */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelineEmbedding
{
	GENERATED_BODY()

	/** L2-normalized when the package says so. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	TArray<float> Values;
};

/** One top-k entry of a classification. */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelineLabelScore
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString Label;

	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	int32 Index = -1;

	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	float Score = 0.f;
};

/** Classification result. Label/Index/Score repeat TopK[0]. */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelineClassification
{
	GENERATED_BODY()

	/** == TopK[0].Label */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString Label;

	/** == TopK[0].Index */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	int32 Index = -1;

	/** == TopK[0].Score */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	float Score = 0.f;

	/** Best first. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	TArray<FPipelineLabelScore> TopK;

	/** Every class in index order, after activation. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	TArray<float> Scores;
};

/** One TopKSimilar result. */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelineSimilarityMatch
{
	GENERATED_BODY()

	/** Index into the corpus array. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	int32 Index = -1;

	/** Cosine similarity. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	float Score = 0.f;
};

/** One object detection (docs/contracts.md C26). */
USTRUCT(BlueprintType)
struct MODELPIPELINESCORE_API FPipelineDetection
{
	GENERATED_BODY()

	/** Labels[Index] of the package. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FString Label;

	/** Class id (index into the package's Labels). */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	int32 Index = -1;

	/** Class score after activation. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	float Score = 0.f;

	/** Normalized to the input image: Min = (left, top), Max = (right, bottom), each in [0, 1]; bIsValid true when set. */
	UPROPERTY(BlueprintReadOnly, Category = "Pipelines")
	FBox2D Box = FBox2D(ForceInit);
};
