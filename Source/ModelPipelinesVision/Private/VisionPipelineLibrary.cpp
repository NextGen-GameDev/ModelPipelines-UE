// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "VisionPipelineLibrary.h"

#include "CoreGlobals.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "ImageClassificationPipeline.h"
#include "ObjectDetectionPipeline.h"
#include "PipelineModelPackage.h"
#include "PipelineVisionLog.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(VisionPipelineLibrary)

UImageClassificationPipeline* UVisionPipelineLibrary::CreateImageClassificationPipeline(UPipelineModelPackage* Package, FPipelineResult& OutResult)
{
	if (!IsInGameThread())
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("CreateImageClassificationPipeline must be called on the game thread"));
		UE_LOG(LogModelPipelinesVision, Warning, TEXT("%s"), *OutResult.Message);
		return nullptr;
	}
	if (!Package)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("CreateImageClassificationPipeline: package is null"));
		return nullptr;
	}

	UImageClassificationPipeline* Pipeline = NewObject<UImageClassificationPipeline>(GetTransientPackage());
	const FPipelineResult Init = Pipeline->InitializeFromPackage(Package);
	if (!Init.bSuccess)
	{
		Pipeline->MarkAsGarbage();
		OutResult = Init;
		return nullptr;
	}
	OutResult = FPipelineResult::Ok();
	return Pipeline;
}

UObjectDetectionPipeline* UVisionPipelineLibrary::CreateObjectDetectionPipeline(UPipelineModelPackage* Package, FPipelineResult& OutResult)
{
	if (!IsInGameThread())
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("CreateObjectDetectionPipeline must be called on the game thread"));
		UE_LOG(LogModelPipelinesVision, Warning, TEXT("%s"), *OutResult.Message);
		return nullptr;
	}
	if (!Package)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("CreateObjectDetectionPipeline: package is null"));
		return nullptr;
	}

	UObjectDetectionPipeline* Pipeline = NewObject<UObjectDetectionPipeline>(GetTransientPackage());
	const FPipelineResult Init = Pipeline->InitializeFromPackage(Package);
	if (!Init.bSuccess)
	{
		Pipeline->MarkAsGarbage();
		OutResult = Init;
		return nullptr;
	}
	OutResult = FPipelineResult::Ok();
	return Pipeline;
}
