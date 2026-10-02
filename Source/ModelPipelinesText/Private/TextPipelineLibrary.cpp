// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "TextPipelineLibrary.h"

#include "CoreGlobals.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineModelPackage.h"
#include "PipelineTextPreprocess.h"
#include "TextClassificationPipeline.h"
#include "TextEmbeddingPipeline.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(TextPipelineLibrary)

UTextEmbeddingPipeline* UTextPipelineLibrary::CreateTextEmbeddingPipeline(UPipelineModelPackage* Package, FPipelineResult& OutResult)
{
	if (!IsInGameThread())
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("CreateTextEmbeddingPipeline must be called on the game thread"));
		UE_LOG(LogModelPipelinesText, Warning, TEXT("%s"), *OutResult.Message);
		return nullptr;
	}
	if (!Package)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("CreateTextEmbeddingPipeline: package is null"));
		return nullptr;
	}

	UTextEmbeddingPipeline* Pipeline = NewObject<UTextEmbeddingPipeline>(GetTransientPackage());
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

UTextClassificationPipeline* UTextPipelineLibrary::CreateTextClassificationPipeline(UPipelineModelPackage* Package, FPipelineResult& OutResult)
{
	if (!IsInGameThread())
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("CreateTextClassificationPipeline must be called on the game thread"));
		UE_LOG(LogModelPipelinesText, Warning, TEXT("%s"), *OutResult.Message);
		return nullptr;
	}
	if (!Package)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("CreateTextClassificationPipeline: package is null"));
		return nullptr;
	}

	UTextClassificationPipeline* Pipeline = NewObject<UTextClassificationPipeline>(GetTransientPackage());
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
