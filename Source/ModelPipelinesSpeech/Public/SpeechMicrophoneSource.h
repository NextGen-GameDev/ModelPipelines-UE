// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Templates/UniquePtr.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectPtr.h"
#include "PipelineTypes.h"

#include "SpeechMicrophoneSource.generated.h"

class USpeechStream;

namespace Audio
{
	class FAudioCapture;
}

/** An audio input device as reported by the engine's audio capture backend (docs/contracts.md C81c). */
USTRUCT(BlueprintType)
struct MODELPIPELINESSPEECH_API FSpeechCaptureDevice
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	int32 DeviceIndex = -1;

	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	FString Name;

	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	int32 NumChannels = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	int32 SampleRate = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	bool bIsDefault = false;
};

/**
 * Microphone source for a USpeechStream (C81c): opens an input device through the engine's audio capture (AudioCapture plugin)
 * and pushes its float audio into the stream from the capture thread. Stop closes the device; it never stops the stream.
 */
UCLASS(BlueprintType)
class MODELPIPELINESSPEECH_API USpeechMicrophoneSource : public UObject
{
	GENERATED_BODY()

public:
	USpeechMicrophoneSource();
	virtual ~USpeechMicrophoneSource() override;

	/** Game thread. Available input devices (empty when none / no backend). Never blocks for long, never crashes without hardware. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Speech")
	static TArray<FSpeechCaptureDevice> GetCaptureDevices();

	/** Game thread. Opens the device (-1 = default) and pushes its audio into Stream (Stream->PushAudio from the capture thread).
	    False + OutResult (RuntimeUnavailable "no audio capture device" / InvalidArgument for a bad index or null stream).
	    Starting again while capturing -> InvalidArgument. */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Speech")
	bool Start(USpeechStream* Stream, int32 DeviceIndex, FPipelineResult& OutResult);

	/** Game thread. Closes the device; does not stop the stream. Waits for the capture thread to finish its last callback:
	    normally ~10 ms, up to 1 s if the device was unplugged mid-capture (the engine's WASAPI backend waits on its buffer event). */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|Speech")
	void Stop();

	UFUNCTION(BlueprintPure, Category = "Model Pipelines|Speech")
	bool IsCapturing() const;

	virtual void BeginDestroy() override;

private:
	/** The stream receiving audio; kept alive while capturing. */
	UPROPERTY()
	TObjectPtr<USpeechStream> TargetStream;

	TUniquePtr<Audio::FAudioCapture> Capture;

	bool bCapturing = false;
};
