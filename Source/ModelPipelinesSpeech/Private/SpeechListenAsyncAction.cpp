// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechListenAsyncAction.h"

#include "Async/Async.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "PipelineSpeechLog.h"
#include "SpeechMicrophoneSource.h"
#include "SpeechToTextPipeline.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(SpeechListenAsyncAction)

USpeechListenAsyncAction* USpeechListenAsyncAction::StartListening(UObject* WorldContextObject, USpeechToTextPipeline* Pipeline,
	FSpeechStreamSettings Settings, bool bUseMicrophone, int32 DeviceIndex)
{
	USpeechListenAsyncAction* Action = NewObject<USpeechListenAsyncAction>(GetTransientPackage());
	Action->Pipeline = Pipeline;
	Action->Settings = Settings;
	Action->bUseMicrophone = bUseMicrophone;
	Action->DeviceIndex = DeviceIndex;
	// Engine convention, except that a null context is a silent no-op (C34; the action roots itself in Activate anyway).
	if (WorldContextObject)
	{
		Action->RegisterWithGameInstance(WorldContextObject);
	}
	return Action;
}

USpeechStream* USpeechListenAsyncAction::GetStream() const
{
	return Stream;
}

void USpeechListenAsyncAction::Activate()
{
	if (State != EListenState::Idle)
	{
		UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("Start Listening: Activate called again on an action that was already activated or cancelled; ignored"));
		return;
	}
	State = EListenState::Running;
	AddToRoot();

	if (!Pipeline)
	{
		UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("Start Listening: Pipeline is null"));
		FailLater(FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("Pipeline is null")));
		return;
	}

	FPipelineResult StartResult;
	USpeechStream* NewStream = Pipeline->StartStream(Settings, StartResult);
	if (!NewStream)
	{
		if (StartResult.bSuccess)
		{
			StartResult = FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("Start Listening: the pipeline did not start a stream"));
		}
		FailLater(StartResult);
		return;
	}
	Stream = NewStream;
	Stream->OnSpeechStarted.AddDynamic(this, &USpeechListenAsyncAction::HandleStreamSpeechStarted);
	Stream->OnPartial.AddDynamic(this, &USpeechListenAsyncAction::HandleStreamPartial);
	Stream->OnFinal.AddDynamic(this, &USpeechListenAsyncAction::HandleStreamFinal);
	Stream->OnSpeechEnded.AddDynamic(this, &USpeechListenAsyncAction::HandleStreamSpeechEnded);
	Stream->OnStopped.AddDynamic(this, &USpeechListenAsyncAction::HandleStreamStopped);

	if (bUseMicrophone)
	{
		Microphone = NewObject<USpeechMicrophoneSource>(this);
		FPipelineResult MicResult;
		if (!Microphone->Start(Stream, DeviceIndex, MicResult))
		{
			// The stream ends silently (unbound first): a failed start gives OnFailure only, never OnStopped.
			Microphone = nullptr;
			UnbindStream();
			Stream->Stop();
			Stream = nullptr;
			FailLater(MicResult);
			return;
		}
	}
}

void USpeechListenAsyncAction::Cancel()
{
	switch (State)
	{
	case EListenState::Idle:
		// Never activated: nothing runs; release now.
		State = EListenState::Done;
		Super::Cancel();
		break;
	case EListenState::Running:
		State = EListenState::Stopping;
		StopMicrophone();
		if (Stream)
		{
			Stream->Stop();   // OnStopped arrives from the stream on a later tick (HandleStreamStopped).
		}
		break;
	default:
		// Stopping (OnStopped pending), Failing (OnFailure pending) or Done: nothing more to do.
		break;
	}
}

bool USpeechListenAsyncAction::IsActive() const
{
	return State == EListenState::Running || State == EListenState::Stopping;
}

void USpeechListenAsyncAction::HandleStreamSpeechStarted()
{
	if (IsActive())
	{
		OnSpeechStarted.Broadcast(FSpeechTranscript(), FPipelineResult::Ok());
	}
}

void USpeechListenAsyncAction::HandleStreamPartial(const FSpeechTranscript& Transcript)
{
	if (IsActive())
	{
		OnPartial.Broadcast(Transcript, FPipelineResult::Ok());
	}
}

void USpeechListenAsyncAction::HandleStreamFinal(const FSpeechTranscript& Transcript)
{
	if (IsActive())
	{
		OnFinal.Broadcast(Transcript, FPipelineResult::Ok());
	}
}

void USpeechListenAsyncAction::HandleStreamSpeechEnded()
{
	if (IsActive())
	{
		OnSpeechEnded.Broadcast(FSpeechTranscript(), FPipelineResult::Ok());
	}
}

void USpeechListenAsyncAction::HandleStreamStopped(const FPipelineResult& Result)
{
	if (!IsActive())
	{
		return;
	}
	State = EListenState::Done;
	StopMicrophone();
	UnbindStream();
	OnStopped.Broadcast(FSpeechTranscript(), Result);
	Release();
}

void USpeechListenAsyncAction::FailLater(const FPipelineResult& Result)
{
	State = EListenState::Failing;
	// The action is rooted until Release, so the weak pointer only guards engine shutdown.
	AsyncTask(ENamedThreads::GameThread, [WeakThis = TWeakObjectPtr<USpeechListenAsyncAction>(this), Result]()
	{
		USpeechListenAsyncAction* Action = WeakThis.Get();
		if (!Action || Action->State != EListenState::Failing)
		{
			return;
		}
		Action->State = EListenState::Done;
		Action->OnFailure.Broadcast(FSpeechTranscript(), Result);
		Action->Release();
	});
}

void USpeechListenAsyncAction::StopMicrophone()
{
	if (Microphone)
	{
		Microphone->Stop();
	}
}

void USpeechListenAsyncAction::UnbindStream()
{
	if (Stream)
	{
		Stream->OnSpeechStarted.RemoveAll(this);
		Stream->OnPartial.RemoveAll(this);
		Stream->OnFinal.RemoveAll(this);
		Stream->OnSpeechEnded.RemoveAll(this);
		Stream->OnStopped.RemoveAll(this);
	}
}

void USpeechListenAsyncAction::Release()
{
	Microphone = nullptr;
	Stream = nullptr;
	Pipeline = nullptr;
	RemoveFromRoot();
	SetReadyToDestroy();
}
