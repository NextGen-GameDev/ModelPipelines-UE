// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Engine/CancellableAsyncAction.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineResults.h"
#include "SpeechTypes.h"
#include "SpeechPipelineAsyncActions.generated.h"

class USoundWave;
class USpeechToTextPipeline;

/** Output pin of Transcribe Speech / Transcribe Sound Wave. Result: what the pipeline reported. Transcript: empty on failure. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FPipelineTranscribeSpeechPin,
	const FPipelineResult&, Result, const FSpeechTranscript&, Transcript);

namespace UE::ModelPipelines::Speech
{
	/** Life cycle of one speech async action (docs/contracts.md C34). */
	enum class EAsyncActionState : uint8
	{
		/** Created, not activated. */
		Idle,
		/** Activated, result not delivered yet. */
		Running,
		/** A pin fired. */
		Delivered,
		/** Cancelled before delivery; the result is dropped when it arrives. */
		Cancelled
	};
}

/**
 * Blueprint latent node "Transcribe Speech" (docs/contracts.md C76, C34 rules): wraps USpeechToTextPipeline::Transcribe.
 * Exactly one of OnSuccess / OnFailure fires, once, on the game thread, never inside Activate. The action keeps itself
 * (and the pipeline) alive until the call's result arrives.
 */
UCLASS()
class MODELPIPELINESSPEECH_API UTranscribeSpeechAsyncAction : public UCancellableAsyncAction
{
	GENERATED_BODY()

public:
	/** Fires once on the game thread when the call succeeded. */
	UPROPERTY(BlueprintAssignable)
	FPipelineTranscribeSpeechPin OnSuccess;

	/** Fires once on the game thread when the call failed. Result says why; Transcript is empty. */
	UPROPERTY(BlueprintAssignable)
	FPipelineTranscribeSpeechPin OnFailure;

	/** Transcribes interleaved float samples (any rate 1000..384000, any channel count, at most 35 s). Runs in the
	    background; a null or not initialized pipeline, or invalid audio, ends in On Failure. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Speech",
		meta = (BlueprintInternalUseOnly = "true", WorldContext = "WorldContextObject", DisplayName = "Transcribe Speech"))
	static UTranscribeSpeechAsyncAction* TranscribeSpeech(UObject* WorldContextObject, USpeechToTextPipeline* Pipeline,
		const TArray<float>& Samples, int32 SampleRate = 16000, int32 NumChannels = 1);

	/** Game thread. Starts the one call; a second Activate logs a Warning and does nothing. */
	virtual void Activate() override;

	/** Game thread. Before delivery: no pin fires afterwards (the running call finishes and its result is dropped).
	    After delivery or a second time: no-op. */
	virtual void Cancel() override;

	/** True from Activate until a pin fired or Cancel was called. */
	virtual bool IsActive() const override;

	/** Same as IsActive(). */
	virtual bool ShouldBroadcastDelegates() const override;

private:
	/** Game thread: the pipeline's (or the deferred null-pipeline) result. */
	void HandleResult(const FPipelineResult& Result, const FSpeechTranscript& Transcript);

	/** Kept alive with the action for the in-flight call. */
	UPROPERTY()
	TObjectPtr<USpeechToTextPipeline> Pipeline;

	TArray<float> Samples;
	int32 SampleRate = 16000;
	int32 NumChannels = 1;

	UE::ModelPipelines::Speech::EAsyncActionState State = UE::ModelPipelines::Speech::EAsyncActionState::Idle;

	/** Activate started a call whose result has not arrived yet (the action is rooted while this is set). */
	bool bInFlight = false;
};

/**
 * Blueprint latent node "Transcribe Sound Wave" (docs/contracts.md C76, C34 rules): wraps
 * USpeechToTextPipeline::TranscribeSoundWave. Exactly one of OnSuccess / OnFailure fires, once, on the game thread, never
 * inside Activate. The action keeps itself, the pipeline and the sound wave alive until the call's result arrives.
 */
UCLASS()
class MODELPIPELINESSPEECH_API UTranscribeSoundWaveAsyncAction : public UCancellableAsyncAction
{
	GENERATED_BODY()

public:
	/** Fires once on the game thread when the call succeeded. */
	UPROPERTY(BlueprintAssignable)
	FPipelineTranscribeSpeechPin OnSuccess;

	/** Fires once on the game thread when the call failed. Result says why; Transcript is empty. */
	UPROPERTY(BlueprintAssignable)
	FPipelineTranscribeSpeechPin OnFailure;

	/** Transcribes a sound wave asset (at most 35 s; procedural waves are not supported). The audio is decoded in the
	    background; a null pipeline or sound, or audio that cannot be decoded, ends in On Failure. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Speech",
		meta = (BlueprintInternalUseOnly = "true", WorldContext = "WorldContextObject", DisplayName = "Transcribe Sound Wave"))
	static UTranscribeSoundWaveAsyncAction* TranscribeSoundWave(UObject* WorldContextObject, USpeechToTextPipeline* Pipeline,
		USoundWave* Sound);

	/** Game thread. Starts the one call; a second Activate logs a Warning and does nothing. */
	virtual void Activate() override;

	/** Game thread. Before delivery: no pin fires afterwards (the running call finishes and its result is dropped).
	    After delivery or a second time: no-op. */
	virtual void Cancel() override;

	/** True from Activate until a pin fired or Cancel was called. */
	virtual bool IsActive() const override;

	/** Same as IsActive(). */
	virtual bool ShouldBroadcastDelegates() const override;

private:
	/** Game thread: the pipeline's (or the deferred null-pipeline) result. */
	void HandleResult(const FPipelineResult& Result, const FSpeechTranscript& Transcript);

	/** Kept alive with the action for the in-flight call. */
	UPROPERTY()
	TObjectPtr<USpeechToTextPipeline> Pipeline;

	/** Kept alive with the action until delivery. */
	UPROPERTY()
	TObjectPtr<USoundWave> Sound;

	UE::ModelPipelines::Speech::EAsyncActionState State = UE::ModelPipelines::Speech::EAsyncActionState::Idle;

	/** Activate started a call whose result has not arrived yet (the action is rooted while this is set). */
	bool bInFlight = false;
};
