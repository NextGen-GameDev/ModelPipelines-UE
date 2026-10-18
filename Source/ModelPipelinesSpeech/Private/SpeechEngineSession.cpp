// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechEngineSession.h"

#include "SpeechEngine.h"

namespace UE::ModelPipelines::Speech
{
	FSpeechEngineSession::FSpeechEngineSession(TSharedPtr<IPipelineModel> InModel, TArray<float>&& InMono, int32 InSampleRate)
		: Model(MoveTemp(InModel))
		, Mono(MoveTemp(InMono))
		, SampleRate(InSampleRate)
	{
	}

	FPipelineResult FSpeechEngineSession::Begin()
	{
		if (!Model.IsValid())
		{
			return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("the pipeline has no model"));
		}
		// The executor normally loads the model before Begin; initialize here if it did not.
		if (!Model->IsInitialized())
		{
			const FPipelineResult Init = Model->Initialize();
			if (!Init.bSuccess)
			{
				return Init;
			}
		}
		if (Model->AsSpeechEngine() == nullptr)
		{
			return FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("the model has no speech engine"));
		}
		Device = Model->GetDevice();
		return FPipelineResult::Ok();
	}

	EPipelineStep FSpeechEngineSession::Step(TArray<FPipelineSessionProgress>& OutProgress, FPipelineResult& OutResult)
	{
		ISpeechEngine* Engine = Model.IsValid() ? Model->AsSpeechEngine() : nullptr;
		if (!Engine)
		{
			OutResult = FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("the model has no speech engine"));
			return EPipelineStep::Failed;
		}
		FSpeechTranscript Out;
		FPipelineResult R = Engine->Transcribe(Mono, SampleRate, Out);
		Mono.Empty();
		if (!R.bSuccess)
		{
			if (R.Error == EPipelineError::None)
			{
				R.Error = EPipelineError::InferenceFailed;
			}
			R.Message = FString::Printf(TEXT("engine: %s"), *R.Message);
			OutResult = MoveTemp(R);
			return EPipelineStep::Failed;
		}
		Transcript = MoveTemp(Out);
		return EPipelineStep::Finished;
	}

	void FSpeechEngineSession::End(EPipelineSessionEnd Reason, FPipelineResult& OutFinal)
	{
		Mono.Empty();
		if (Reason == EPipelineSessionEnd::Finished)
		{
			OutFinal = FPipelineResult::Ok();
			OutFinal.Device = Device;
			OutFinal.InferenceMs = Transcript.InferenceMs;
		}
	}
}
