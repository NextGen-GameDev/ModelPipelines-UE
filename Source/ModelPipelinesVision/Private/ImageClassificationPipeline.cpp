// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "ImageClassificationPipeline.h"

#include "Containers/ArrayView.h"
#include "PipelineMath.h"
#include "PipelineModelPackage.h"
#include "PipelineTensor.h"
#include "PipelineVisionCommon.h"
#include "PipelineVisionLog.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(ImageClassificationPipeline)

namespace UE::ModelPipelines::Vision::ClassificationPrivate
{
	using namespace UE::ModelPipelines::Vision::Private;

	/** Per-call data: the shared labels and the activation copy taken when the call was issued, and the result of Postprocess. */
	struct FImageClassificationCall
	{
		FSharedLabels Labels;   // immutable, shared with the pipeline (no per-call copy)
		EPipelineScoreActivation Activation = EPipelineScoreActivation::Softmax;
		int32 TopK = 1;
		FPipelineClassification Result;
	};
	using FImageClassificationCallRef = TSharedRef<FImageClassificationCall, ESPMode::ThreadSafe>;

	/** Main output float32 [1, C] (C == Labels.Num()) -> activated scores and top-k (C20 rules). */
	static FPipelineResult ClassifyOutput(const FImageClassificationCall& Call, const FPipelineTensor& Out, FPipelineClassification& OutResult)
	{
		const TArray<FString>& Labels = LabelsOrEmpty(Call.Labels);
		const int32 NumClasses = Labels.Num();
		const TConstArrayView<int64> Shape = Out.GetShape();
		const bool bShapeOk = NumClasses >= 1 && Out.IsValid() && Out.GetType() == EPipelineTensorType::Float32 && Shape.Num() == 2
			&& Shape[0] == 1 && Shape[1] == NumClasses && Out.Num() == (int64)NumClasses;
		if (!bShapeOk)
		{
			return FPipelineResult::Fail(EPipelineError::InferenceFailed,
				FString::Printf(TEXT("model output is %s %s; expected float32 [1,%d] (%d labels)"),
					LexToString(Out.GetType()), *Out.DescribeShape(), NumClasses, NumClasses));
		}

		const TConstArrayView64<float> Data64 = Out.GetFloatData();
		const TConstArrayView<float> Logits(Data64.GetData(), NumClasses);
		FPipelineClassification R;
		const bool bOk = Call.Activation == EPipelineScoreActivation::Sigmoid ? Sigmoid(Logits, R.Scores) : Softmax(Logits, R.Scores);
		if (!bOk || R.Scores.Num() != NumClasses)
		{
			return FPipelineResult::Fail(EPipelineError::InferenceFailed, TEXT("model output contains NaN or Inf"));
		}
		const int32 K = FMath::Clamp(Call.TopK, 1, NumClasses);
		const TArray<int32> Best = TopKIndices(R.Scores, K);
		if (Best.Num() != K)
		{
			return FPipelineResult::Fail(EPipelineError::Internal, FString::Printf(TEXT("top-k returned %d of %d entries"), Best.Num(), K));
		}
		R.TopK.Reserve(K);
		for (const int32 Index : Best)
		{
			if (!R.Scores.IsValidIndex(Index))
			{
				return FPipelineResult::Fail(EPipelineError::Internal, FString::Printf(TEXT("top-k index %d out of range"), Index));
			}
			FPipelineLabelScore& E = R.TopK.AddDefaulted_GetRef();
			E.Label = Labels[Index];
			E.Index = Index;
			E.Score = R.Scores[Index];
		}
		R.Label = R.TopK[0].Label;
		R.Index = R.TopK[0].Index;
		R.Score = R.TopK[0].Score;
		OutResult = MoveTemp(R);
		return FPipelineResult::Ok();
	}

	/** Verbose diagnostics: the timings of a finished call. */
	static void LogTimings(const TCHAR* Kind, const FPipelineResult& R)
	{
		UE_LOG(LogModelPipelinesVision, Verbose, TEXT("ImageClassification %s: ok %d, device %d, queue %.2f ms, preprocess %.2f ms, inference %.2f ms, postprocess %.2f ms"),
			Kind, R.bSuccess ? 1 : 0, (int32)R.Device, R.QueueMs, R.PreprocessMs, R.InferenceMs, R.PostprocessMs);
	}

	/** Preprocess and Postprocess of one call. Captures only shared per-call data, never the pipeline. */
	static FPipelineStages MakeStages(const FVisionCallInputRef& Input, const FImageClassificationCallRef& Call)
	{
		FPipelineStages Stages;
		Stages.Preprocess = [Input](const FPipelineModelInfo& Info, TArray<FPipelineTensor>& OutInputs) -> FPipelineResult
		{
			return BuildVisionInputs(*Input, Info, OutInputs);
		};
		Stages.Postprocess = [Call](const FPipelineModelInfo& Info, TArray<FPipelineTensor>&& Outputs) -> FPipelineResult
		{
			if (!Outputs.IsValidIndex(Info.Bindings.MainOutput))
			{
				return FPipelineResult::Fail(EPipelineError::InferenceFailed,
					FString::Printf(TEXT("main output index %d not among %d model outputs"), Info.Bindings.MainOutput, Outputs.Num()));
			}
			return ClassifyOutput(*Call, Outputs[Info.Bindings.MainOutput], Call->Result);
		};
		return Stages;
	}
}

UImageClassificationPipeline::UImageClassificationPipeline()
{
	PipelineTask = EPipelineTask::ImageClassification;
}

FPipelineResult UImageClassificationPipeline::OnInitialize(UPipelineModelPackage* InPackage)
{
	FPipelineResult R = Super::OnInitialize(InPackage);
	if (R.bSuccess)
	{
		// One copy per initialization; calls share it read-only instead of copying the label list per call.
		SharedLabels = UE::ModelPipelines::Vision::Private::MakeSharedLabels(GetLabels());
	}
	return R;
}

void UImageClassificationPipeline::ClassifyAsync(UTexture* Image, int32 TopK, FPipelineImageClassificationCallback OnComplete)
{
	using namespace UE::ModelPipelines::Vision::Private;
	using namespace UE::ModelPipelines::Vision::ClassificationPrivate;

	if (!OnComplete)
	{
		UE_LOG(LogModelPipelinesVision, Warning, TEXT("UImageClassificationPipeline::ClassifyAsync: null callback, nothing runs."));
		return;
	}
	// Off the game thread: the texture is not touched and nothing is copied; RunStages fails the call (InvalidArgument).
	const bool bGameThread = IsInGameThread();
	const FVisionCallInputRef Input = MakeVisionCallInput(bGameThread ? GetPreprocessSettings() : FPipelinePreprocessSettings());
	const FImageClassificationCallRef Call = MakeShared<FImageClassificationCall, ESPMode::ThreadSafe>();
	Call->TopK = TopK;
	if (bGameThread)
	{
		Call->Labels = SharedLabels;
		Call->Activation = GetPostprocessSettings().ScoreActivation;
	}

	FPipelineStages Stages = MakeStages(Input, Call);
	if (bGameThread && IsReady())
	{
		// The read always completes (success, failure, timeout or Cancelled); a failed read fails Preprocess.
		Stages.Prerequisite = StartTextureRead(Image, Input);
	}
	Stages.Complete = [Input, Call, Callback = MoveTemp(OnComplete)](const FPipelineResult& Result) mutable
	{
		const FPipelineResult Final = WithReadbackWait(Result, *Input);
		LogTimings(TEXT("texture async"), Final);
		Callback(Final, Final.bSuccess ? Call->Result : FPipelineClassification());
	};
	RunStages(MoveTemp(Stages));
}

void UImageClassificationPipeline::ClassifyImageAsync(const FPipelineImage& Image, int32 TopK, FPipelineImageClassificationCallback OnComplete)
{
	using namespace UE::ModelPipelines::Vision::Private;
	using namespace UE::ModelPipelines::Vision::ClassificationPrivate;

	if (!OnComplete)
	{
		UE_LOG(LogModelPipelinesVision, Warning, TEXT("UImageClassificationPipeline::ClassifyImageAsync: null callback, nothing runs."));
		return;
	}
	const bool bGameThread = IsInGameThread();
	const FVisionCallInputRef Input = MakeVisionCallInput(bGameThread ? GetPreprocessSettings() : FPipelinePreprocessSettings());
	const FImageClassificationCallRef Call = MakeShared<FImageClassificationCall, ESPMode::ThreadSafe>();
	Call->TopK = TopK;
	if (bGameThread)
	{
		Input->Image = Image;
		Call->Labels = SharedLabels;
		Call->Activation = GetPostprocessSettings().ScoreActivation;
	}

	FPipelineStages Stages = MakeStages(Input, Call);
	Stages.Complete = [Input, Call, Callback = MoveTemp(OnComplete)](const FPipelineResult& Result) mutable
	{
		LogTimings(TEXT("image async"), Result);
		Callback(Result, Result.bSuccess ? Call->Result : FPipelineClassification());
	};
	RunStages(MoveTemp(Stages));
}

FPipelineResult UImageClassificationPipeline::ClassifyImageBlocking(const FPipelineImage& Image, int32 TopK, FPipelineClassification& OutResult)
{
	using namespace UE::ModelPipelines::Vision::Private;
	using namespace UE::ModelPipelines::Vision::ClassificationPrivate;

	OutResult = FPipelineClassification();
	FString Reason;
	if (!Image.IsValid(&Reason))
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, FString::Printf(TEXT("invalid image: %s"), *Reason));
	}
	const FVisionCallInputRef Input = MakeVisionCallInput(GetPreprocessSettings());
	Input->Image = Image;
	const FImageClassificationCallRef Call = MakeShared<FImageClassificationCall, ESPMode::ThreadSafe>();
	Call->TopK = TopK;
	Call->Labels = SharedLabels;
	Call->Activation = GetPostprocessSettings().ScoreActivation;

	// No prerequisite here (C26 note): blocking calls take only CPU images.
	const FPipelineResult R = RunStagesBlocking(MakeStages(Input, Call));
	LogTimings(TEXT("blocking"), R);
	if (R.bSuccess)
	{
		OutResult = MoveTemp(Call->Result);
	}
	return R;
}
