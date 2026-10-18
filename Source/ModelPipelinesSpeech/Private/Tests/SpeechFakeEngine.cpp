// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "SpeechFakeEngine.h"

#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTLS.h"
#include "Misc/ScopeLock.h"

namespace UE::ModelPipelines::SpeechTests
{
	FPipelineResult FFakeSpeechEngine::Transcribe(TConstArrayView<float> Mono, int32 SampleRate, FSpeechTranscript& Out)
	{
		double Delay = 0.0;
		TFunction<FPipelineResult(TConstArrayView<float>, int32, FSpeechTranscript&)> Script;
		{
			FScopeLock S(&State->Lock);
			FFakeEngineCall Call;
			Call.ThreadId = FPlatformTLS::GetCurrentThreadId();
			Call.SampleRate = SampleRate;
			Call.Samples = TArray<float>(Mono.GetData(), Mono.Num());
			Call.bModelInitializedAtCall = State->bModelInitialized;
			State->Calls.Add(MoveTemp(Call));
			Delay = State->DelaySeconds;
			Script = State->Script;
		}
		if (Delay > 0.0) { FPlatformProcess::Sleep((float)Delay); }
		if (Script) { return Script(Mono, SampleRate, Out); }
		Out.Text = TEXT("fake");
		Out.InferenceMs = 1.f;
		return FPipelineResult::Ok();
	}

	TSharedPtr<ISpeechEngineStream> FFakeSpeechEngine::OpenStream(int32 SampleRate, FPipelineResult& OutResult)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("fake engine has no streaming"));
		return nullptr;
	}

	FFakePhononModel::FFakePhononModel(TSharedRef<FFakeEngineState, ESPMode::ThreadSafe> InState)
		: State(InState), Engine(InState)
	{
	}

	FPipelineResult FFakePhononModel::Initialize()
	{
		FScopeLock S(&State->Lock);
		++State->InitializeCount;
		State->bModelInitialized = true;
		return FPipelineResult::Ok();
	}

	bool FFakePhononModel::IsInitialized() const
	{
		FScopeLock S(&State->Lock);
		return State->bModelInitialized;
	}

	EPipelineDevice FFakePhononModel::GetDevice() const
	{
		return IsInitialized() ? EPipelineDevice::CPU : EPipelineDevice::Auto;
	}

	FPipelineResult FFakePhononModel::Run(TConstArrayView<FPipelineTensor> Inputs, TArray<FPipelineTensor>& OutOutputs)
	{
		OutOutputs.Reset();
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("fake Phonon model has no tensor graph"));
	}

	ISpeechEngine* FFakePhononModel::AsSpeechEngine()
	{
		FScopeLock S(&State->Lock);
		return State->bNoEngine ? nullptr : &Engine;
	}

	TSharedPtr<IPipelineModel> FFakePhononBackend::PrepareModel(UNNEModelData* ModelData, EPipelineDevice Device, FPipelineResult& OutResult)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("fake Phonon backend prepares no NNE models"));
		return nullptr;
	}

	TSharedPtr<IPipelineModel> FFakePhononBackend::PrepareModelFromSource(const FPipelineModelSource& Source, EPipelineDevice Device,
		FPipelineResult& OutResult)
	{
		if (Source.Format != EPipelineModelFormat::Phonon)
		{
			OutResult = FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("fake Phonon backend: not a Phonon source"));
			return nullptr;
		}
		{
			FScopeLock S(&State->Lock);
			++State->PrepareCount;
			State->LastPreparedFormat = Source.Format;
			State->LastPreparedPath = Source.FilePath;
		}
		OutResult = FPipelineResult::Ok();
		return MakeShared<FFakePhononModel>(State);
	}
}

#endif // WITH_DEV_AUTOMATION_TESTS
