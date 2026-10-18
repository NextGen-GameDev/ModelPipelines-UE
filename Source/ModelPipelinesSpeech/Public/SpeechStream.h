// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Containers/ArrayView.h"
#include "Templates/SharedPointer.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineTypes.h"
#include "SpeechTypes.h"

#include "SpeechStream.generated.h"

class USpeechToTextPipeline;
class IPipelineModel;

namespace UE::ModelPipelines::Speech
{
	class FSpeechStreamImpl;
}

/** Settings of a real-time speech stream (docs/contracts.md C81c). Out-of-range values are clamped by StartStream. */
USTRUCT(BlueprintType)
struct MODELPIPELINESSPEECH_API FSpeechStreamSettings
{
	GENERATED_BODY()

	/** Silence after the last speech that ends an utterance, seconds (0.1..2). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Speech")
	float EndOfSpeechHangoverSeconds = 0.3f;

	/** New speech audio between two partial transcripts, seconds (0.1..5). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Speech")
	float PartialIntervalSeconds = 0.5f;

	/** Partial transcripts while speaking (only with an engine that supports them). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Speech")
	bool bPartials = true;

	/** Speech threshold over the noise floor, dB (3..40). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Speech")
	float SpeechThresholdDb = 12.f;

	/** Longest utterance; a longer one is ended there, seconds (2..120). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Speech")
	float MaxUtteranceSeconds = 30.f;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FSpeechStreamEvent);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FSpeechStreamTranscriptEvent, const FSpeechTranscript&, Transcript);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FSpeechStreamStoppedEvent, const FPipelineResult&, Result);

/**
 * Real-time speech-to-text on pushed audio (C81c): an energy VAD finds utterances, each gets one final transcript, and with
 * an engine whose SupportsPartials() partial transcripts arrive while speaking. Created by USpeechToTextPipeline::StartStream.
 * All events fire on the game thread: OnSpeechStarted -> OnPartial* -> OnSpeechEnded -> OnFinal per utterance, then
 * OnStopped exactly once. Transcript times are seconds from the stream's first pushed sample.
 */
UCLASS(BlueprintType)
class MODELPIPELINESSPEECH_API USpeechStream : public UObject
{
	GENERATED_BODY()

public:
	UPROPERTY(BlueprintAssignable, Category = "Model Pipelines|Speech")
	FSpeechStreamEvent OnSpeechStarted;

	/** Only with an engine whose SupportsPartials(). */
	UPROPERTY(BlueprintAssignable, Category = "Model Pipelines|Speech")
	FSpeechStreamTranscriptEvent OnPartial;

	UPROPERTY(BlueprintAssignable, Category = "Model Pipelines|Speech")
	FSpeechStreamTranscriptEvent OnFinal;

	UPROPERTY(BlueprintAssignable, Category = "Model Pipelines|Speech")
	FSpeechStreamEvent OnSpeechEnded;

	UPROPERTY(BlueprintAssignable, Category = "Model Pipelines|Speech")
	FSpeechStreamStoppedEvent OnStopped;

	/** Any thread. Interleaved float audio at any rate 1000..384000, any channel count; copied into a lock-free queue. Bad
	    arguments or non-finite samples: the chunk is dropped and one Warning is logged (never a crash). After Stop: ignored. */
	void PushAudio(TConstArrayView<float> Interleaved, int32 SampleRate, int32 NumChannels);

	/** Game thread. Finalises the current utterance (no hangover wait), then OnStopped. A second call does nothing. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Speech")
	void Stop();

	/** False once Stop() was called or the stream stopped with a failure. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Speech")
	bool IsRunning() const;

	/** True inside an utterance (game-thread view of the VAD). */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Speech")
	bool IsSpeaking() const;

	virtual void BeginDestroy() override;

private:
	friend class USpeechToTextPipeline;
	friend class UE::ModelPipelines::Speech::FSpeechStreamImpl;

	/** Called once by USpeechToTextPipeline::StartStream (settings already clamped). */
	void Start(USpeechToTextPipeline* InPipeline, const FSpeechStreamSettings& InSettings, TSharedPtr<IPipelineModel> InEngineModel);

	/** Keeps the pipeline alive while the stream exists. */
	UPROPERTY()
	TObjectPtr<USpeechToTextPipeline> Pipeline;

	TSharedPtr<UE::ModelPipelines::Speech::FSpeechStreamImpl, ESPMode::ThreadSafe> Impl;
};
