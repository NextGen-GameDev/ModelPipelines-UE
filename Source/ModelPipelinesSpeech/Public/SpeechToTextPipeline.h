// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Containers/ArrayView.h"
#include "Templates/SharedPointer.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineBase.h"
#include "PipelineTypes.h"
#include "SpeechTypes.h"
#include "SpeechToTextPipeline.generated.h"

class UPipelineTokenizer;
class UPipelineModelPackage;

namespace UE::ModelPipelines::Speech
{
	struct FSpeechPipelineState;
	struct FSpeechCall;
}

/**
 * Speech-to-text pipeline (docs/contracts.md C75): clips up to 35 s. The front end (mono mix, resampling, silence gate,
 * log-mel) runs on a background task, the encoder as an executor job, the greedy TDT loop as an executor session on the
 * decoder (CPU), and token decoding plus word times on the game thread.
 */
UCLASS(BlueprintType)
class MODELPIPELINESSPEECH_API USpeechToTextPipeline : public UPipelineBase
{
	GENERATED_BODY()

public:
	/** PipelineTask = SpeechToText. */
	USpeechToTextPipeline();

	/** Game thread. Samples interleaved, any rate 1000..384000, any channel count >= 1. Null callback -> Warning, nothing runs. */
	void Transcribe(TArray<float> Samples, int32 SampleRate, int32 NumChannels, FOnSpeechTranscribed OnDone);

	/** Tests/tools: blocking, any thread except the executor worker and the game thread; caller keeps the pipeline alive. */
	FPipelineResult TranscribeBlocking(TConstArrayView<float> Samples, int32 SampleRate, int32 NumChannels, FSpeechTranscript& Out);

protected:
	/** Loads the tokenizer from the package's TokenizerJson. */
	virtual FPipelineResult OnInitialize(UPipelineModelPackage* InPackage) override;

private:
	/** Game thread: runs the encoder for a call whose front end produced features. */
	void StartEncoder(const TSharedRef<UE::ModelPipelines::Speech::FSpeechCall, ESPMode::ThreadSafe>& Call);

	/** Keeps the tokenizer reachable for GC (the shared state also holds a TStrongObjectPtr to it). */
	UPROPERTY()
	TObjectPtr<UPipelineTokenizer> Tokenizer;

	/** Shared with in-flight calls; immutable after OnInitialize. */
	TSharedPtr<UE::ModelPipelines::Speech::FSpeechPipelineState, ESPMode::ThreadSafe> SpeechState;
};
