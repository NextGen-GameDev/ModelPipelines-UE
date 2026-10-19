// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectPtr.h"
#include "SpeechMicrophoneSource.h"
#include "SpeechStream.h"
#include "SpeechTypes.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesTalkToTextSample.generated.h"

class SButton;
class SVerticalBox;
template <typename OptionType> class SComboBox;
class USpeechMicrophoneSource;
class USpeechToTextPipeline;

/**
 * Talk-to-text sample: listens to the microphone (or to audio pushed with Get Stream > Push Speech Audio) and shows each
 * final line with its latency. With a Phonon package (ModelPipelines Generative) the partial text appears while speaking.
 * Model: phonon-2-fp16w.
 */
UCLASS(Blueprintable)
class MODELPIPELINESSAMPLES_API AModelPipelinesTalkToTextSample : public AModelPipelinesSampleBase
{
	GENERATED_BODY()

public:
	AModelPipelinesTalkToTextSample();

	/** Capture the microphone while listening. Off: only audio pushed into GetStream() is transcribed. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample")
	bool bUseMicrophone = true;

	/** Capture device index (USpeechMicrophoneSource::GetCaptureDevices); -1 = the default device. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample")
	int32 DeviceIndex = -1;

	/** Start listening as soon as the pipeline is ready. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample")
	bool bAutoStart = true;

	/** Settings of the speech stream started by StartListening. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample")
	FSpeechStreamSettings StreamSettings;

	/** Final lines kept (the oldest is dropped above this). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Model Pipelines|Sample", meta = (ClampMin = "1"))
	int32 MaxLines = 8;

	/** Starts a stream (and the microphone when bUseMicrophone). No-op unless Ready and not already listening. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Sample")
	void StartListening();

	/** Stops the microphone and the stream; IsListening is false at once. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Sample")
	void StopListening();

	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	bool IsListening() const;

	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	bool IsMicrophoneActive() const;

	/** True when the package can deliver partial text while speaking (package format Phonon). */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	bool HasPartialText() const;

	/** Final lines, oldest first, at most MaxLines. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	TArray<FString> GetFinalLines() const;

	/** The partial text of the current utterance; "" when none. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	FString GetPartialText() const;

	/** Seconds from the end of speech to the final text of the last line; -1 before the first. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	float GetLastFinalLatencySeconds() const;

	/** The stream audio can be pushed into. Null once the stream has stopped (OnStopped); still set while a stopped stream delivers its last finals. */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	USpeechStream* GetStream() const;

	/** The input devices (USpeechMicrophoneSource::GetCaptureDevices, queried now; empty when none). */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Sample")
	TArray<FSpeechCaptureDevice> GetMicrophones() const;

	/** Use input device InDeviceIndex (-1 = default device) and turn bUseMicrophone on. While listening, only the capture restarts on the
	    same stream (no transcript is lost); a failure shows "No microphone (<message>)..." and keeps the stream. Not listening: just stored. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Sample")
	void SetMicrophone(int32 InDeviceIndex);

	/** Name of the device being captured ("" when the microphone is not active). */
	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Sample")
	FString GetMicrophoneName() const;

	/** Fires on the game thread for every non-empty final transcript. */
	UPROPERTY(BlueprintAssignable, Category = "Model Pipelines|Sample")
	FSpeechStreamTranscriptEvent OnFinalText;

protected:
	virtual void EndPlay(const EEndPlayReason::Type Reason) override;
	virtual FPipelineResult CreatePipeline(UPipelineModelPackage* Loaded) override;
	virtual void OnPipelineReady() override;
	virtual TSharedRef<SWidget> BuildOverlay() override;
	virtual FString GetModelSlug() const override;
	virtual FString GetReadyHint() const override;
	virtual FText GetOverlayTitle() const override;
	virtual TSharedPtr<SWidget> GetWidgetToFocus() const override;
	virtual void RefreshOverlay() override;
	virtual FString GetMissingPackageText(const FString& PackagePath) const override;

private:
	UFUNCTION()
	void HandleSpeechStarted();
	UFUNCTION()
	void HandlePartial(const FSpeechTranscript& Transcript);
	UFUNCTION()
	void HandleFinal(const FSpeechTranscript& Transcript);
	UFUNCTION()
	void HandleSpeechEnded();
	UFUNCTION()
	void HandleStopped(const FPipelineResult& Result);

	/** "Listening (<device>)", "Listening (pushed audio)" or the sticky microphone failure text. */
	FString GetListeningStatus() const;
	/** Unbinds this actor from Stream and drops the reference. */
	void ReleaseStream();
	void StopMicrophone();
	/** Starts Microphone (created when null) on Stream with DeviceIndex; sets MicrophoneName or MicrophoneFailure. */
	void StartMicrophone();
	void ToggleListening();
	/** Re-queries the devices into MicOptions (overlay combo). */
	void RefreshMicOptions();
	/** The combo text for DeviceIndex. */
	FString GetMicComboText() const;

	UPROPERTY(Transient)
	TObjectPtr<USpeechToTextPipeline> Pipeline;

	UPROPERTY(Transient)
	TObjectPtr<USpeechStream> Stream;

	UPROPERTY(Transient)
	TObjectPtr<USpeechMicrophoneSource> Microphone;

	/** Runs one throw-away transcription so the first real final does not pay the cold model start. */
	void StartWarmUp();
	void HandleWarmUpDone(const FPipelineResult& Result);

	bool bListening = false;
	bool bPartialCapable = false;
	/** The warm-up transcription is in flight. */
	bool bWarmingUp = false;
	/** StartListening was requested (or bAutoStart) while warming up: start when the warm-up completes. */
	bool bStartAfterWarmUp = false;
	FString MicrophoneName;
	FString MicrophoneFailure;
	TArray<FString> Lines;
	TArray<float> LineLatencies;
	FString PartialText;
	float LastLatency = -1.f;
	/** FPlatformTime::Seconds() of each SpeechEnded whose final has not arrived yet, oldest first (a final can trail the next
	    utterance's SpeechEnded when audio arrives faster than real time). */
	TArray<double> SpeechEndedTimes;

	TSharedPtr<SButton> ToggleButton;
	TSharedPtr<SVerticalBox> LinesList;
	/** Combo entries: DeviceIndex -1 = the default device, -2 = the disabled "No microphone found" entry; Name holds the shown text. */
	TArray<TSharedPtr<FSpeechCaptureDevice>> MicOptions;
	TSharedPtr<SComboBox<TSharedPtr<FSpeechCaptureDevice>>> MicCombo;
};
