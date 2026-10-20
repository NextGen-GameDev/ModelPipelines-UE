// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "UObject/ObjectMacros.h"
#include "PipelineTypes.h"
#include "TextPipelineLibrary.generated.h"

class UPipelineModelPackage;
class UTextClassificationPipeline;
class UTextEmbeddingPipeline;

/** Creators of the text task pipelines (docs/contracts.md C20). */
UCLASS()
class MODELPIPELINESTEXT_API UTextPipelineLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/** Game thread. New pipeline (outer: transient package; hold it in a UPROPERTY) or nullptr with OutResult saying why. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Text")
	static UTextEmbeddingPipeline* CreateTextEmbeddingPipeline(UPipelineModelPackage* Package, FPipelineResult& OutResult);

	/** Game thread. New pipeline (outer: transient package; hold it in a UPROPERTY) or nullptr with OutResult saying why. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Text")
	static UTextClassificationPipeline* CreateTextClassificationPipeline(UPipelineModelPackage* Package, FPipelineResult& OutResult);
};
