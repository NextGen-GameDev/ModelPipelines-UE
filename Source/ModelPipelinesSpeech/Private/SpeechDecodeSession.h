// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "PipelineBackend.h"
#include "PipelineModelPackage.h"
#include "PipelineSession.h"
#include "PipelineTaskIO.h"
#include "PipelineTensor.h"

namespace UE::ModelPipelines::Speech
{
	/** One greedy TDT step (C70): emitted token (BlankTokenId = blank), the encoder frame it was taken at, the frame advance. */
	struct FSpeechDecodeStep
	{
		int32 Token = 0;
		int32 Frame = 0;
		int32 Duration = 0;
	};

	/**
	 * Internal (C75 step 4): the greedy TDT loop on the decoder_joint graph as an executor session (C61). Each Step runs up to
	 * StepsPerTurn decoder steps, then yields. Only the executor worker calls Begin/Step/End; the getters are read on another
	 * thread only after the session completed.
	 */
	class FSpeechDecodeSession final : public IPipelineSession
	{
	public:
		static constexpr int32 StepsPerTurn = 32;

		/** Encoder: row-major [NumFrames][EncoderDim]. */
		FSpeechDecodeSession(TSharedPtr<IPipelineModel> InModel, const FPipelineSpeechSettings& InSpeech, TArray<float>&& InEncoder,
			int32 InNumFrames, int32 InEncoderDim);

		virtual TSharedPtr<IPipelineModel> GetModel() const override { return Model; }
		virtual FPipelineResult Begin() override;
		virtual EPipelineStep Step(TArray<FPipelineSessionProgress>& OutProgress, FPipelineResult& OutResult) override;
		virtual void End(EPipelineSessionEnd Reason, FPipelineResult& OutFinal) override;

		/** After completion: every greedy step in order. */
		const TArray<FSpeechDecodeStep>& GetSteps() const { return Steps; }
		/** After completion: summed wall time of the decoder Run calls. */
		double GetDecoderMs() const { return DecoderSeconds * 1000.0; }

	private:
		bool IsDone() const { return Frame >= NumFrames || Steps.Num() >= StepCap; }

		TSharedPtr<IPipelineModel> Model;
		FPipelineSpeechSettings Speech;
		TArray<float> Encoder;
		int32 NumFrames = 0;
		int32 EncoderDim = 0;

		FPipelineSpeechDecoderBindings Bindings;
		int32 NumOutputs = 0;
		TArray<FPipelineTensor> Inputs;
		int32 Frame = 0;
		int64 CurrentToken = 0;
		int32 StepCap = 0;
		TArray<FSpeechDecodeStep> Steps;
		double DecoderSeconds = 0.0;
	};
}
