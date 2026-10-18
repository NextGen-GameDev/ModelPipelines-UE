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
#include "SpeechStream.h"
#include "SpeechToTextPipeline.generated.h"

class UPipelineTokenizer;
class UPipelineModelPackage;
class USoundWave;

namespace UE::ModelPipelines::Speech
{
	struct FSpeechPipelineState;
	struct FSpeechCall;
}

/**
 * Speech-to-text pipeline: calls up to 600 s (longer than 35 s are split at quiet points into 25-35 s windows). The front end (mono mix, resampling, silence gate,
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

	/** Game thread (docs/contracts.md C76). Decodes the wave's PCM off the game thread (the wave is kept alive until the decode
	    ends), then runs exactly the Transcribe path on the decoded interleaved float samples at the wave's own rate and channel
	    count. Every failure is delivered on a later game-thread tick. Null callback -> Warning, nothing runs. */
	void TranscribeSoundWave(USoundWave* Sound, FOnSpeechTranscribed OnDone);

	/** Tests/tools: blocking, any thread except the executor worker and the game thread; caller keeps the pipeline alive. */
	FPipelineResult TranscribeBlocking(TConstArrayView<float> Samples, int32 SampleRate, int32 NumChannels, FSpeechTranscript& Out);

	/** Game thread (docs/contracts.md C81c). Starts a real-time stream on this pipeline (must be ready). Null + OutResult
	    failure otherwise. Settings are clamped to their ranges. Keep a reference to the returned stream (a UPROPERTY or a
	    variable): dropping the last reference ends it at the next garbage collection, with no further events. The
	    "Start Listening" node keeps its own stream alive until Stop Listening. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Speech")
	USpeechStream* StartStream(const FSpeechStreamSettings& Settings, FPipelineResult& OutResult);

protected:
	/** Loads the tokenizer from the package's TokenizerJson. */
	virtual FPipelineResult OnInitialize(UPipelineModelPackage* InPackage) override;

private:
	friend class USpeechStream;

	/** WP9R.3: the model behind the Phonon engine path (C81a), or null on the NNE path. */
	TSharedPtr<IPipelineModel> GetSpeechEngineModel() const;

	/** Game thread: runs the encoder for a call whose front end produced features. */
	void StartEncoder(const TSharedRef<UE::ModelPipelines::Speech::FSpeechCall, ESPMode::ThreadSafe>& Call);

	/** Game thread, after validation (C81a rule 2): mono mix + silence gate on a worker, then one engine session. */
	void TranscribeWithEngine(const TSharedRef<UE::ModelPipelines::Speech::FSpeechCall, ESPMode::ThreadSafe>& Call);

	/** Blocking variant of the engine path (C81a rule 2), for TranscribeBlocking. */
	FPipelineResult TranscribeWithEngineBlocking(const TSharedRef<UE::ModelPipelines::Speech::FSpeechCall, ESPMode::ThreadSafe>& Call,
		FSpeechTranscript& Out);

	/** Set by OnInitialize: the package is in Phonon format and calls go to the model's ISpeechEngine (C81a). */
	bool bUseSpeechEngine = false;

	/** Keeps the tokenizer reachable for GC (the shared state also holds a TStrongObjectPtr to it). */
	UPROPERTY()
	TObjectPtr<UPipelineTokenizer> Tokenizer;

	/** Shared with in-flight calls; immutable after OnInitialize. */
	TSharedPtr<UE::ModelPipelines::Speech::FSpeechPipelineState, ESPMode::ThreadSafe> SpeechState;
};
