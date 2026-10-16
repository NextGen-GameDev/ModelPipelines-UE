// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechPipelineAsyncActions.h"

#include "Async/Async.h"
#include "Sound/SoundWave.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "PipelineSpeechLog.h"
#include "SpeechToTextPipeline.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(SpeechPipelineAsyncActions)

using UE::ModelPipelines::Speech::EAsyncActionState;

namespace UE::ModelPipelines::Speech::AsyncActionPrivate
{
	static FPipelineResult NullPipelineResult()
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("Pipeline is null"));
	}

	/** Engine convention, except that a null context is a silent no-op (the engine would log a script warning). */
	static void Register(UBlueprintAsyncActionBase* Action, const UObject* WorldContextObject)
	{
		if (WorldContextObject)
		{
			Action->RegisterWithGameInstance(WorldContextObject);
		}
	}

	/** Runs Fn on a later game-thread tick if the action still exists. */
	template <typename ActionType, typename FuncType>
	static void DeferOnGameThread(ActionType* Action, FuncType&& Fn)
	{
		AsyncTask(ENamedThreads::GameThread, [WeakAction = TWeakObjectPtr<ActionType>(Action), Fn = MoveTemp(Fn)]()
		{
			if (ActionType* Alive = WeakAction.Get())
			{
				Fn(Alive);
			}
		});
	}

	/** C34 rule 2: state, root and warnings. False: a second Activate (ignored). */
	template <typename ActionType>
	static bool BeginActivate(ActionType* Action, EAsyncActionState& State, bool& bInFlight, bool bHasPipeline, const TCHAR* NodeName)
	{
		if (State != EAsyncActionState::Idle)
		{
			UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("%s: Activate called again on an action that was already activated or cancelled; ignored"), NodeName);
			return false;
		}
		State = EAsyncActionState::Running;
		bInFlight = true;
		Action->AddToRoot();
		if (!bHasPipeline)
		{
			UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("%s: Pipeline is null"), NodeName);
		}
		return true;
	}

}

// ---------------------------------------------------------------------------------------------------- Transcribe Speech

UTranscribeSpeechAsyncAction* UTranscribeSpeechAsyncAction::TranscribeSpeech(UObject* WorldContextObject, USpeechToTextPipeline* Pipeline,
	const TArray<float>& Samples, int32 SampleRate, int32 NumChannels)
{
	UTranscribeSpeechAsyncAction* Action = NewObject<UTranscribeSpeechAsyncAction>(GetTransientPackage());
	Action->Pipeline = Pipeline;
	Action->Samples = Samples;
	Action->SampleRate = SampleRate;
	Action->NumChannels = NumChannels;
	UE::ModelPipelines::Speech::AsyncActionPrivate::Register(Action, WorldContextObject);
	return Action;
}

void UTranscribeSpeechAsyncAction::Activate()
{
	using namespace UE::ModelPipelines::Speech::AsyncActionPrivate;

	if (!BeginActivate(this, State, bInFlight, Pipeline != nullptr, TEXT("Transcribe Speech")))
	{
		return;
	}
	if (!Pipeline)
	{
		DeferOnGameThread(this, [](UTranscribeSpeechAsyncAction* Action)
		{
			Action->HandleResult(NullPipelineResult(), FSpeechTranscript());
		});
		return;
	}
	Pipeline->Transcribe(TArray<float>(Samples), SampleRate, NumChannels,
		[WeakThis = TWeakObjectPtr<UTranscribeSpeechAsyncAction>(this)](const FPipelineResult& Result, const FSpeechTranscript& Transcript)
	{
		if (UTranscribeSpeechAsyncAction* Action = WeakThis.Get())
		{
			Action->HandleResult(Result, Transcript);
		}
	});
}

void UTranscribeSpeechAsyncAction::HandleResult(const FPipelineResult& Result, const FSpeechTranscript& Transcript)
{
	if (!bInFlight)
	{
		return;
	}
	bInFlight = false;
	if (State == EAsyncActionState::Running)
	{
		State = EAsyncActionState::Delivered;
		if (Result.bSuccess)
		{
			OnSuccess.Broadcast(Result, Transcript);
		}
		else
		{
			OnFailure.Broadcast(Result, FSpeechTranscript());
		}
	}
	Pipeline = nullptr;
	Samples.Empty();
	RemoveFromRoot();
	SetReadyToDestroy();
}

void UTranscribeSpeechAsyncAction::Cancel()
{
	if (State == EAsyncActionState::Idle || State == EAsyncActionState::Running)
	{
		State = EAsyncActionState::Cancelled;
		// The base unregisters from the game instance; the root reference stays until the dropped result arrives (HandleResult).
		Super::Cancel();
	}
}

bool UTranscribeSpeechAsyncAction::IsActive() const
{
	return State == EAsyncActionState::Running;
}

bool UTranscribeSpeechAsyncAction::ShouldBroadcastDelegates() const
{
	return IsActive();
}

// ---------------------------------------------------------------------------------------------------- Transcribe Sound Wave

UTranscribeSoundWaveAsyncAction* UTranscribeSoundWaveAsyncAction::TranscribeSoundWave(UObject* WorldContextObject,
	USpeechToTextPipeline* Pipeline, USoundWave* Sound)
{
	UTranscribeSoundWaveAsyncAction* Action = NewObject<UTranscribeSoundWaveAsyncAction>(GetTransientPackage());
	Action->Pipeline = Pipeline;
	Action->Sound = Sound;
	UE::ModelPipelines::Speech::AsyncActionPrivate::Register(Action, WorldContextObject);
	return Action;
}

void UTranscribeSoundWaveAsyncAction::Activate()
{
	using namespace UE::ModelPipelines::Speech::AsyncActionPrivate;

	if (!BeginActivate(this, State, bInFlight, Pipeline != nullptr, TEXT("Transcribe Sound Wave")))
	{
		return;
	}
	if (!Pipeline)
	{
		DeferOnGameThread(this, [](UTranscribeSoundWaveAsyncAction* Action)
		{
			Action->HandleResult(NullPipelineResult(), FSpeechTranscript());
		});
		return;
	}
	Pipeline->TranscribeSoundWave(Sound,
		[WeakThis = TWeakObjectPtr<UTranscribeSoundWaveAsyncAction>(this)](const FPipelineResult& Result, const FSpeechTranscript& Transcript)
	{
		if (UTranscribeSoundWaveAsyncAction* Action = WeakThis.Get())
		{
			Action->HandleResult(Result, Transcript);
		}
	});
}

void UTranscribeSoundWaveAsyncAction::HandleResult(const FPipelineResult& Result, const FSpeechTranscript& Transcript)
{
	if (!bInFlight)
	{
		return;
	}
	bInFlight = false;
	if (State == EAsyncActionState::Running)
	{
		State = EAsyncActionState::Delivered;
		if (Result.bSuccess)
		{
			OnSuccess.Broadcast(Result, Transcript);
		}
		else
		{
			OnFailure.Broadcast(Result, FSpeechTranscript());
		}
	}
	Pipeline = nullptr;
	Sound = nullptr;
	RemoveFromRoot();
	SetReadyToDestroy();
}

void UTranscribeSoundWaveAsyncAction::Cancel()
{
	if (State == EAsyncActionState::Idle || State == EAsyncActionState::Running)
	{
		State = EAsyncActionState::Cancelled;
		Super::Cancel();
	}
}

bool UTranscribeSoundWaveAsyncAction::IsActive() const
{
	return State == EAsyncActionState::Running;
}

bool UTranscribeSoundWaveAsyncAction::ShouldBroadcastDelegates() const
{
	return IsActive();
}
