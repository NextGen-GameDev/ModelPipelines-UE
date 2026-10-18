// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Engine/CancellableAsyncAction.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineTypes.h"
#include "SpeechTypes.h"
#include "SpeechStream.h"

#include "SpeechListenAsyncAction.generated.h"

class USpeechToTextPipeline;
class USpeechMicrophoneSource;

/** All pins of "Start Listening" share this signature (K2Node_BaseAsyncTask requirement). */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FSpeechListenPin, const FSpeechTranscript&, Transcript, const FPipelineResult&, Result);

/**
 * "Start Listening" (docs/contracts.md C81c): starts a speech stream on a pipeline, optionally fed by a microphone, and forwards
 * its events on the game thread in the stream's order. The action keeps itself, the stream, the pipeline and the microphone
 * alive from Activate until OnStopped or OnFailure. "Stop Listening" (or Cancel) ends it; OnStopped still fires, once.
 */
UCLASS()
class MODELPIPELINESSPEECH_API USpeechListenAsyncAction : public UCancellableAsyncAction
{
	GENERATED_BODY()

public:
	/** An utterance began. Transcript empty, Result Ok. */
	UPROPERTY(BlueprintAssignable)
	FSpeechListenPin OnSpeechStarted;

	/** Partial transcript while speaking (only with an engine that supports partials). */
	UPROPERTY(BlueprintAssignable)
	FSpeechListenPin OnPartial;

	/** Final transcript of an utterance. */
	UPROPERTY(BlueprintAssignable)
	FSpeechListenPin OnFinal;

	/** An utterance ended. Transcript empty, Result Ok. */
	UPROPERTY(BlueprintAssignable)
	FSpeechListenPin OnSpeechEnded;

	/** The stream stopped (Result = the stream's stop result). Fires once; nothing follows. */
	UPROPERTY(BlueprintAssignable)
	FSpeechListenPin OnStopped;

	/** Starting failed (null pipeline, not ready, no device). Fires once on a later tick; no OnStopped follows. */
	UPROPERTY(BlueprintAssignable)
	FSpeechListenPin OnFailure;

	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Speech",
		meta = (BlueprintInternalUseOnly = "true", WorldContext = "WorldContextObject", DisplayName = "Start Listening"))
	static USpeechListenAsyncAction* StartListening(UObject* WorldContextObject, USpeechToTextPipeline* Pipeline,
		FSpeechStreamSettings Settings, bool bUseMicrophone = true, int32 DeviceIndex = -1);

	/** The running stream (null before Activate / after failure), for "Push Speech Audio" when bUseMicrophone is false. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Speech")
	USpeechStream* GetStream() const;

	/** Starts the stream (+ microphone); a failure fires OnFailure on a later tick. A second call logs a Warning. */
	virtual void Activate() override;

	/** = Stop Listening: stops the microphone and the stream; OnStopped still fires; then the action is released. */
	virtual void Cancel() override;

	virtual bool IsActive() const override;

private:
	enum class EListenState : uint8 { Idle, Running, Stopping, Failing, Done };

	UFUNCTION()
	void HandleStreamSpeechStarted();

	UFUNCTION()
	void HandleStreamPartial(const FSpeechTranscript& Transcript);

	UFUNCTION()
	void HandleStreamFinal(const FSpeechTranscript& Transcript);

	UFUNCTION()
	void HandleStreamSpeechEnded();

	UFUNCTION()
	void HandleStreamStopped(const FPipelineResult& Result);

	/** Fires OnFailure on a later game-thread tick, then releases the action. */
	void FailLater(const FPipelineResult& Result);

	void StopMicrophone();
	void UnbindStream();
	void Release();

	UPROPERTY()
	TObjectPtr<USpeechToTextPipeline> Pipeline;

	UPROPERTY()
	TObjectPtr<USpeechStream> Stream;

	UPROPERTY()
	TObjectPtr<USpeechMicrophoneSource> Microphone;

	FSpeechStreamSettings Settings;
	bool bUseMicrophone = true;
	int32 DeviceIndex = -1;
	EListenState State = EListenState::Idle;
};
