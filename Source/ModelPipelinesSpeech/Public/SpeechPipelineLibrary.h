// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "UObject/ObjectMacros.h"
#include "PipelineTypes.h"
#include "SpeechPipelineLibrary.generated.h"

class UPipelineModelPackage;
class USpeechToTextPipeline;
class USpeechListenAsyncAction;

/** Creators of the speech task pipelines (docs/contracts.md C75). */
UCLASS()
class MODELPIPELINESSPEECH_API USpeechPipelineLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/** Game thread. New pipeline (outer: transient package; hold it in a UPROPERTY) or nullptr with OutResult saying why. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Speech")
	static USpeechToTextPipeline* CreateSpeechToTextPipeline(UPipelineModelPackage* Package, FPipelineResult& OutResult);

	/** Game thread (C81c). Ends a "Start Listening" action: stops its microphone and stream; OnStopped still fires once.
	    Null -> Warning, nothing happens. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Speech", meta = (DisplayName = "Stop Listening"))
	static void StopListening(USpeechListenAsyncAction* Listener);
};
