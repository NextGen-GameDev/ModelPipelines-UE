// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "PipelineBackend.h"
#include "PipelineSession.h"
#include "SpeechTypes.h"

namespace UE::ModelPipelines::Speech
{
	/**
	 * Internal (C81a rule 2): one executor session (C61) whose single step gets the model's ISpeechEngine and runs
	 * Transcribe on the whole mono signal at the input rate. Only the executor worker calls Begin/Step/End; the getters are
	 * read on another thread only after the session completed.
	 */
	class FSpeechEngineSession final : public IPipelineSession
	{
	public:
		FSpeechEngineSession(TSharedPtr<IPipelineModel> InModel, TArray<float>&& InMono, int32 InSampleRate);

		virtual TSharedPtr<IPipelineModel> GetModel() const override { return Model; }
		virtual FPipelineResult Begin() override;
		virtual EPipelineStep Step(TArray<FPipelineSessionProgress>& OutProgress, FPipelineResult& OutResult) override;
		virtual void End(EPipelineSessionEnd Reason, FPipelineResult& OutFinal) override;

		/** After a successful completion: the engine's transcript (AudioSeconds not set). */
		const FSpeechTranscript& GetTranscript() const { return Transcript; }

	private:
		TSharedPtr<IPipelineModel> Model;
		TArray<float> Mono;
		int32 SampleRate = 0;
		FSpeechTranscript Transcript;
		EPipelineDevice Device = EPipelineDevice::CPU;
	};
}
