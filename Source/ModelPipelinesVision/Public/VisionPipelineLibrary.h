// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "UObject/ObjectMacros.h"
#include "PipelineTypes.h"
#include "VisionPipelineLibrary.generated.h"

class UPipelineModelPackage;
class UImageClassificationPipeline;

/** Creators of the vision task pipelines (docs/contracts.md C29). */
UCLASS()
class MODELPIPELINESVISION_API UVisionPipelineLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/** Game thread. New pipeline (outer: transient package; hold it in a UPROPERTY) or nullptr with OutResult saying why. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Vision")
	static UImageClassificationPipeline* CreateImageClassificationPipeline(UPipelineModelPackage* Package, FPipelineResult& OutResult);
};
