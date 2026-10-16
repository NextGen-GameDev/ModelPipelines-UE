// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechDecodeSession.h"

#include "HAL/PlatformTime.h"
#include "PipelineValidation.h"

namespace UE::ModelPipelines::Speech
{
	namespace DecodePrivate
	{
		/** Index of the first maximum (ties -> lowest index). */
		static int32 ArgMax(TConstArrayView64<float> Values)
		{
			int32 Best = 0;
			for (int64 i = 1; i < Values.Num(); ++i)
			{
				if (Values[i] > Values[Best])
				{
					Best = (int32)i;
				}
			}
			return Best;
		}

		/** Desc shape with a dynamic batch dim (index 1 of [layers, batch, hidden]) set to 1; false if another dim is dynamic. */
		static bool ConcreteStateShape(const FPipelineTensorDesc& Desc, TArray<int64>& OutShape)
		{
			OutShape = Desc.Shape;
			for (int32 i = 0; i < OutShape.Num(); ++i)
			{
				if (OutShape[i] < 0)
				{
					if (i != 1)
					{
						return false;
					}
					OutShape[i] = 1;
				}
			}
			return true;
		}
	}

	FSpeechDecodeSession::FSpeechDecodeSession(TSharedPtr<IPipelineModel> InModel, const FPipelineSpeechSettings& InSpeech,
		TArray<float>&& InEncoder, int32 InNumFrames, int32 InEncoderDim)
		: Model(MoveTemp(InModel))
		, Speech(InSpeech)
		, Encoder(MoveTemp(InEncoder))
		, NumFrames(InNumFrames)
		, EncoderDim(InEncoderDim)
	{
	}

	FPipelineResult FSpeechDecodeSession::Begin()
	{
		using namespace DecodePrivate;

		if (!Model.IsValid())
		{
			return FPipelineResult::Fail(EPipelineError::Internal, TEXT("speech decoder session has no model"));
		}
		if (NumFrames < 1 || EncoderDim < 1 || (int64)Encoder.Num() != (int64)NumFrames * EncoderDim)
		{
			return FPipelineResult::Fail(EPipelineError::Internal,
				FString::Printf(TEXT("encoder output of %d values does not match %d frames x %d"), Encoder.Num(), NumFrames, EncoderDim));
		}
		if (Speech.Durations.Num() < 1 || Speech.BlankTokenId < 0)
		{
			return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("speech settings need at least one duration and BlankTokenId >= 0"));
		}

		const TArray<FPipelineTensorDesc> InDescs = Model->GetInputDescs();
		const TArray<FPipelineTensorDesc> OutDescs = Model->GetOutputDescs();
		FPipelineValidationReport Report;
		if (!ResolveSpeechDecoderIO(Speech, EncoderDim, InDescs, OutDescs, Bindings, Report))
		{
			return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
				FString::Printf(TEXT("decoder graph does not match the speech decoder contract: %s"), *Report.ToString()));
		}
		if (InDescs.Num() != 4)
		{
			return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
				FString::Printf(TEXT("decoder graph has %d inputs; expected encoder_frame, token, state_h, state_c"), InDescs.Num()));
		}
		NumOutputs = OutDescs.Num();

		TArray<int64> ShapeH, ShapeC;
		if (!ConcreteStateShape(InDescs[Bindings.StateH], ShapeH) || !ConcreteStateShape(InDescs[Bindings.StateC], ShapeC))
		{
			return FPipelineResult::Fail(EPipelineError::ShapeMismatch, TEXT("decoder state_h/state_c have a dynamic dim other than the batch"));
		}

		Inputs.Reset();
		Inputs.SetNum(InDescs.Num());
		Inputs[Bindings.EncoderFrame] = FPipelineTensor::Zeros(EPipelineTensorType::Float32, { 1, (int64)EncoderDim });
		Inputs[Bindings.Token] = FPipelineTensor::Zeros(EPipelineTensorType::Int64, { 1 });
		Inputs[Bindings.StateH] = FPipelineTensor::Zeros(EPipelineTensorType::Float32, ShapeH);
		Inputs[Bindings.StateC] = FPipelineTensor::Zeros(EPipelineTensorType::Float32, ShapeC);
		for (const FPipelineTensor& T : Inputs)
		{
			if (!T.IsValid())
			{
				return FPipelineResult::Fail(EPipelineError::Internal, TEXT("could not allocate the decoder input tensors"));
			}
		}

		Frame = 0;
		CurrentToken = Speech.BlankTokenId;
		const int64 Cap = (int64)FMath::Max(1, Speech.MaxSymbolsPerStep) * ((int64)NumFrames + 1) - 1;
		StepCap = (int32)FMath::Min<int64>(Cap, MAX_int32);
		Steps.Reset();
		Steps.Reserve(FMath::Min(StepCap, NumFrames * 2 + 16));
		DecoderSeconds = 0.0;
		return FPipelineResult::Ok();
	}

	EPipelineStep FSpeechDecodeSession::Step(TArray<FPipelineSessionProgress>& OutProgress, FPipelineResult& OutResult)
	{
		using namespace DecodePrivate;

		const int32 Blank = Speech.BlankTokenId;
		for (int32 Turn = 0; Turn < StepsPerTurn; ++Turn)
		{
			if (IsDone())
			{
				return EPipelineStep::Finished;
			}

			FMemory::Memcpy(Inputs[Bindings.EncoderFrame].GetFloatData().GetData(), Encoder.GetData() + (int64)Frame * EncoderDim,
				sizeof(float) * EncoderDim);
			Inputs[Bindings.Token].GetInt64Data()[0] = CurrentToken;

			TArray<FPipelineTensor> Outputs;
			const double Start = FPlatformTime::Seconds();
			const FPipelineResult R = Model->Run(Inputs, Outputs);
			DecoderSeconds += FPlatformTime::Seconds() - Start;
			if (!R.bSuccess)
			{
				OutResult = R;
				return EPipelineStep::Failed;
			}
			if (Outputs.Num() != NumOutputs)
			{
				OutResult = FPipelineResult::Fail(EPipelineError::InferenceFailed,
					FString::Printf(TEXT("decoder returned %d outputs, expected %d"), Outputs.Num(), NumOutputs));
				return EPipelineStep::Failed;
			}

			const FPipelineTensor& TokLogits = Outputs[Bindings.TokenLogits];
			const FPipelineTensor& DurLogits = Outputs[Bindings.DurationLogits];
			const bool bShapesOk = TokLogits.IsValid() && TokLogits.GetType() == EPipelineTensorType::Float32 && TokLogits.Num() == (int64)Blank + 1
				&& DurLogits.IsValid() && DurLogits.GetType() == EPipelineTensorType::Float32 && DurLogits.Num() == Speech.Durations.Num()
				&& Outputs[Bindings.StateHOut].IsValid() && Outputs[Bindings.StateHOut].Num() == Inputs[Bindings.StateH].Num()
				&& Outputs[Bindings.StateCOut].IsValid() && Outputs[Bindings.StateCOut].Num() == Inputs[Bindings.StateC].Num();
			if (!bShapesOk)
			{
				OutResult = FPipelineResult::Fail(EPipelineError::InferenceFailed,
					FString::Printf(TEXT("decoder outputs have unexpected sizes: token_logits %s (expected %d values), duration_logits %s (expected %d)"),
						*TokLogits.DescribeShape(), Blank + 1, *DurLogits.DescribeShape(), Speech.Durations.Num()));
				return EPipelineStep::Failed;
			}

			const int32 K = ArgMax(TokLogits.GetFloatData());
			int32 D = FMath::Max(0, Speech.Durations[ArgMax(DurLogits.GetFloatData())]);
			if (K == Blank && D == 0)
			{
				D = 1;
			}
			Steps.Add({ K, Frame, D });
			if (K != Blank)
			{
				CurrentToken = K;
				Inputs[Bindings.StateH] = MoveTemp(Outputs[Bindings.StateHOut]);
				Inputs[Bindings.StateC] = MoveTemp(Outputs[Bindings.StateCOut]);
			}
			Frame += D;
		}
		return IsDone() ? EPipelineStep::Finished : EPipelineStep::Continue;
	}

	void FSpeechDecodeSession::End(EPipelineSessionEnd Reason, FPipelineResult& OutFinal)
	{
		Inputs.Empty();
		Encoder.Empty();
		if (Reason == EPipelineSessionEnd::Finished)
		{
			OutFinal = FPipelineResult::Ok();
			OutFinal.Device = Model.IsValid() ? Model->GetDevice() : EPipelineDevice::CPU;
			OutFinal.InferenceMs = (float)GetDecoderMs();
		}
	}
}
