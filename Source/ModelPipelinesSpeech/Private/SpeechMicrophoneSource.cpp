// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechMicrophoneSource.h"

#include "AudioCaptureCore.h"
#include "AudioCaptureDeviceInterface.h"
#include "Containers/ArrayView.h"
#include "PipelineSpeechLog.h"
#include "SpeechStream.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(SpeechMicrophoneSource)

// Engine API (UE 5.8; UE 5.7 has the same declarations):
//   Audio::FAudioCapture                     Runtime/AudioCaptureCore/Public/AudioCaptureCore.h:28
//     GetCaptureDevicesAvailable             :36 (returns a bool as int32: the count is OutDevices.Num())
//     GetCaptureDeviceInfo(Info, INDEX_NONE) :45 (INDEX_NONE = default device)
//     OpenAudioCaptureStream                 :54, CloseStream :57, StartStream :60, StopStream :63, IsStreamOpen :75
//   FAudioCaptureDeviceParams / FOnAudioCaptureFunction / DefaultDeviceEncoding FLOATING_POINT_32
//                                            Runtime/AudioCaptureCore/Public/AudioCaptureDeviceInterface.h:45 / :62 / :43
// The backend comes from a modular feature (AudioCaptureCore/Private/AudioCaptureInternal.h:34): without one (no AudioCapture
// plugin, -nosound, unsupported platform) every call answers false / no devices. The WASAPI backend's StopStream joins the
// capture thread (AudioCaptureWasapi/Private/WasapiCaptureThread.cpp:73), so no callback runs after Stop returns.

namespace UE::ModelPipelines::Speech::MicrophonePrivate
{
	static FPipelineResult NoDevice(const TCHAR* Detail)
	{
		return FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, FString::Printf(TEXT("no audio capture device (%s)"), Detail));
	}
}

USpeechMicrophoneSource::USpeechMicrophoneSource() = default;

USpeechMicrophoneSource::~USpeechMicrophoneSource() = default;

TArray<FSpeechCaptureDevice> USpeechMicrophoneSource::GetCaptureDevices()
{
	TArray<FSpeechCaptureDevice> Out;
	Audio::FAudioCapture Probe;
	TArray<Audio::FCaptureDeviceInfo> Infos;
	Probe.GetCaptureDevicesAvailable(Infos);
	if (Infos.Num() == 0)
	{
		return Out;
	}
	Audio::FCaptureDeviceInfo DefaultInfo;
	const bool bHasDefault = Probe.GetCaptureDeviceInfo(DefaultInfo, Audio::DefaultDeviceIndex);
	bool bDefaultFound = false;
	for (int32 Index = 0; Index < Infos.Num(); ++Index)
	{
		const Audio::FCaptureDeviceInfo& Info = Infos[Index];
		FSpeechCaptureDevice& Device = Out.AddDefaulted_GetRef();
		Device.DeviceIndex = Index;
		Device.Name = Info.DeviceName;
		Device.NumChannels = Info.InputChannels;
		Device.SampleRate = Info.PreferredSampleRate;
		Device.bIsDefault = bHasDefault && !bDefaultFound && !DefaultInfo.DeviceId.IsEmpty() && Info.DeviceId == DefaultInfo.DeviceId;
		bDefaultFound |= Device.bIsDefault;
	}
	return Out;
}

bool USpeechMicrophoneSource::Start(USpeechStream* Stream, int32 DeviceIndex, FPipelineResult& OutResult)
{
	using namespace UE::ModelPipelines::Speech::MicrophonePrivate;

	if (bCapturing)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("USpeechMicrophoneSource::Start: already capturing; call Stop first"));
		return false;
	}
	if (!Stream)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("USpeechMicrophoneSource::Start: Stream is null"));
		return false;
	}
	if (DeviceIndex < Audio::DefaultDeviceIndex)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument,
			FString::Printf(TEXT("USpeechMicrophoneSource::Start: device index %d is invalid (-1 = default device)"), DeviceIndex));
		return false;
	}

	TUniquePtr<Audio::FAudioCapture> NewCapture = MakeUnique<Audio::FAudioCapture>();
	TArray<Audio::FCaptureDeviceInfo> Infos;
	NewCapture->GetCaptureDevicesAvailable(Infos);
	if (Infos.Num() == 0)
	{
		OutResult = NoDevice(TEXT("no input device or no audio capture backend; is the AudioCapture plugin enabled?"));
		return false;
	}
	if (DeviceIndex >= Infos.Num())
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument,
			FString::Printf(TEXT("USpeechMicrophoneSource::Start: device index %d is out of range (%d device(s))"), DeviceIndex, Infos.Num()));
		return false;
	}
	// An explicit index takes its info from the capture-device list: UE 5.7's FAudioCaptureWasapiStream::GetCaptureDeviceInfo resolves an
	// explicit index over ALL endpoints (eAll; 5.8 uses eCapture, AudioCaptureWasapi.cpp:22), while OpenAudioCaptureStream and
	// GetCaptureDevicesAvailable use the capture subset in both versions. Only the default device is queried by GetCaptureDeviceInfo.
	Audio::FCaptureDeviceInfo Info;
	if (DeviceIndex != Audio::DefaultDeviceIndex)
	{
		Info = Infos[DeviceIndex];
	}
	else if (!NewCapture->GetCaptureDeviceInfo(Info, DeviceIndex))
	{
		OutResult = NoDevice(DeviceIndex == Audio::DefaultDeviceIndex ? TEXT("no default input device") : TEXT("the device could not be queried"));
		return false;
	}

	// Capture thread: one copy into the stream's lock-free queue. The stream is kept alive by TargetStream until Stop has joined
	// the capture thread; during GC every unreachable object gets BeginDestroy (where this source stops) before any is freed.
	USpeechStream* RawStream = Stream;
	Audio::FOnAudioCaptureFunction OnCapture = [RawStream](const void* InAudio, int32 NumFrames, int32 NumChannels, int32 SampleRate, double, bool)
	{
		if (InAudio == nullptr || NumFrames <= 0 || NumChannels <= 0)
		{
			return;
		}
		RawStream->PushAudio(TConstArrayView<float>(static_cast<const float*>(InAudio), NumFrames * NumChannels), SampleRate, NumChannels);
	};

	Audio::FAudioCaptureDeviceParams Params;
	Params.DeviceIndex = DeviceIndex;
	Params.PCMAudioEncoding = Audio::EPCMAudioEncoding::FLOATING_POINT_32;
	if (!NewCapture->OpenAudioCaptureStream(Params, MoveTemp(OnCapture), 1024))
	{
		OutResult = NoDevice(*FString::Printf(TEXT("'%s' could not be opened"), *Info.DeviceName));
		return false;
	}
	TargetStream = Stream;
	if (!NewCapture->StartStream())
	{
		NewCapture->CloseStream();
		TargetStream = nullptr;
		OutResult = NoDevice(*FString::Printf(TEXT("'%s' could not be started"), *Info.DeviceName));
		return false;
	}
	Capture = MoveTemp(NewCapture);
	bCapturing = true;
	UE_LOG(LogModelPipelinesSpeech, Log, TEXT("USpeechMicrophoneSource: capturing from '%s' (%d ch, %d Hz)"), *Info.DeviceName, Info.InputChannels, Info.PreferredSampleRate);
	OutResult = FPipelineResult::Ok();
	return true;
}

void USpeechMicrophoneSource::Stop()
{
	if (Capture.IsValid())
	{
		if (Capture->IsStreamOpen())
		{
			Capture->StopStream();
			Capture->CloseStream();
		}
		Capture.Reset();
	}
	bCapturing = false;
	TargetStream = nullptr;
}

bool USpeechMicrophoneSource::IsCapturing() const
{
	return bCapturing;
}

void USpeechMicrophoneSource::BeginDestroy()
{
	Stop();
	Super::BeginDestroy();
}
