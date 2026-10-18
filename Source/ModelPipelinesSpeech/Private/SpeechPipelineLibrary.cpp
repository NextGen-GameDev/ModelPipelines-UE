// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechPipelineLibrary.h"

#include "CoreGlobals.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineModelPackage.h"
#include "PipelineSpeechLog.h"
#include "SpeechToTextPipeline.h"
#include "SpeechListenAsyncAction.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(SpeechPipelineLibrary)

USpeechToTextPipeline* USpeechPipelineLibrary::CreateSpeechToTextPipeline(UPipelineModelPackage* Package, FPipelineResult& OutResult)
{
	if (!IsInGameThread())
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("CreateSpeechToTextPipeline must be called on the game thread"));
		UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("%s"), *OutResult.Message);
		return nullptr;
	}
	if (!Package)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("CreateSpeechToTextPipeline: package is null"));
		return nullptr;
	}

	USpeechToTextPipeline* Pipeline = NewObject<USpeechToTextPipeline>(GetTransientPackage());
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

void USpeechPipelineLibrary::StopListening(USpeechListenAsyncAction* Listener)
{
	if (!Listener)
	{
		UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("Stop Listening: Listener is null; nothing to stop"));
		return;
	}
	Listener->Cancel();
}
