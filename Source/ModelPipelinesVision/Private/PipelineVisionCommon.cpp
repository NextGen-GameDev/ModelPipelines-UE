// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "PipelineVisionCommon.h"

#include "HAL/PlatformTime.h"
#include "PipelineTextureReader.h"

namespace UE::ModelPipelines::Vision::Private
{
	FSharedLabels MakeSharedLabels(const TArray<FString>& Labels)
	{
		return MakeShared<TArray<FString>, ESPMode::ThreadSafe>(Labels);
	}

	const TArray<FString>& LabelsOrEmpty(const FSharedLabels& Labels)
	{
		static const TArray<FString> Empty;
		return Labels.IsValid() ? *Labels : Empty;
	}

	FVisionCallInputRef MakeVisionCallInput(const FPipelinePreprocessSettings& Settings)
	{
		FVisionCallInputRef Input = MakeShared<FVisionCallInput, ESPMode::ThreadSafe>();
		Input->Settings = Settings;
		return Input;
	}

	UE::Tasks::FTask StartTextureRead(UTexture* Texture, const FVisionCallInputRef& Input)
	{
		const double StartSeconds = FPlatformTime::Seconds();
		return ReadTexturePixelsAsync(Texture, [Input, StartSeconds](const FPipelineResult& Result, FPipelineImage&& Image)
		{
			Input->ReadbackWaitMs = (FPlatformTime::Seconds() - StartSeconds) * 1000.0;
			Input->ReadResult = Result;
			if (Result.bSuccess)
			{
				Input->Image = MoveTemp(Image);
			}
		});
	}

	/** Builds the pixel_mask tensor (all 1) in the desc's element type. */
	static FPipelineTensor MakeAllOnes(EPipelineTensorType Type, TConstArrayView<int64> Shape, int64 Count)
	{
		FPipelineTensor T = FPipelineTensor::Zeros(Type, Shape);
		if (!T.IsValid() || T.Num() != Count)
		{
			return FPipelineTensor();
		}
		switch (Type)
		{
		case EPipelineTensorType::Int64:
			for (int64& V : T.GetInt64Data()) { V = 1; }
			break;
		case EPipelineTensorType::Int32:
			for (int32& V : T.GetInt32Data()) { V = 1; }
			break;
		case EPipelineTensorType::Float32:
			for (float& V : T.GetFloatData()) { V = 1.f; }
			break;
		default:
			return FPipelineTensor();
		}
		return T;
	}

	static FString DescribeDesc(const FPipelineTensorDesc& Desc)
	{
		FString Dims;
		for (int32 i = 0; i < Desc.Shape.Num(); ++i)
		{
			Dims += (i > 0 ? TEXT(",") : TEXT(""));
			Dims += FString::Printf(TEXT("%lld"), (long long)Desc.Shape[i]);
		}
		return FString::Printf(TEXT("%s [%s]"), Desc.bTypeSupported ? LexToString(Desc.Type) : TEXT("unsupported"), *Dims);
	}

	FPipelineResult BuildVisionInputs(FVisionCallInput& Input, const FPipelineModelInfo& Info, TArray<FPipelineTensor>& OutInputs)
	{
		OutInputs.Reset();
		if (!Input.ReadResult.bSuccess)
		{
			return Input.ReadResult;
		}
		FString Reason;
		if (!Input.Image.IsValid(&Reason))
		{
			return FPipelineResult::Fail(EPipelineError::InvalidArgument, FString::Printf(TEXT("invalid image: %s"), *Reason));
		}
		const FPipelineTaskBindings& B = Info.Bindings;
		if (!Info.InputDescs.IsValidIndex(B.PixelValues))
		{
			return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
				FString::Printf(TEXT("pixel_values binding %d not among %d model inputs"), B.PixelValues, Info.InputDescs.Num()));
		}

		TArray<float> CHW;
		const FPipelineResult Pre = PreprocessImage(Input.Image, Input.Settings, CHW, nullptr, &Input.Geometry);
		Input.Image = FPipelineImage();   // the source pixels are no longer needed
		if (!Pre.bSuccess)
		{
			return Pre;
		}
		const int64 H = Input.Settings.ImageHeight;
		const int64 W = Input.Settings.ImageWidth;
		if (CHW.Num() != 3 * H * W)
		{
			return FPipelineResult::Fail(EPipelineError::Internal,
				FString::Printf(TEXT("preprocessing produced %d values, expected 3x%lldx%lld"), CHW.Num(), (long long)H, (long long)W));
		}

		TArray<FPipelineTensor> Inputs;
		Inputs.SetNum(Info.InputDescs.Num());
		for (int32 i = 0; i < Info.InputDescs.Num(); ++i)
		{
			const FPipelineTensorDesc& Desc = Info.InputDescs[i];
			FPipelineTensor T;
			if (i == B.PixelValues)
			{
				const int64 Shape[] = { 1, 3, H, W };
				T = FPipelineTensor::FromFloat(Shape, CHW);
			}
			else if (i == B.PixelMask)
			{
				const int64 Shape[] = { 1, H, W };
				T = MakeAllOnes(Desc.bTypeSupported ? Desc.Type : EPipelineTensorType::Int64, Shape, H * W);
			}
			else
			{
				return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
					FString::Printf(TEXT("model input %d '%s' is neither pixel_values nor pixel_mask"), i, *Desc.Name));
			}
			if (!Desc.IsCompatible(T))
			{
				return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
					FString::Printf(TEXT("model input '%s' expects %s; preprocessing produced %s %s (settings ImageWidth %lld, ImageHeight %lld)"),
						*Desc.Name, *DescribeDesc(Desc), LexToString(T.GetType()), *T.DescribeShape(), (long long)W, (long long)H));
			}
			Inputs[i] = MoveTemp(T);
		}
		OutInputs = MoveTemp(Inputs);
		return FPipelineResult::Ok();
	}

	FPipelineResult WithReadbackWait(const FPipelineResult& Result, const FVisionCallInput& Input)
	{
		FPipelineResult Out = Result;
		Out.PreprocessMs += (float)Input.ReadbackWaitMs;
		return Out;
	}
}
