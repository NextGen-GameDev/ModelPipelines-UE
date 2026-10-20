// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "ObjectDetectionPipeline.h"

#include "Containers/ArrayView.h"
#include "PipelineDetectionDecode.h"
#include "PipelineModelPackage.h"
#include "PipelineTensor.h"
#include "PipelineVisionCommon.h"
#include "PipelineVisionLog.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(ObjectDetectionPipeline)

namespace UE::ModelPipelines::Vision::DetectionPrivate
{
	using namespace UE::ModelPipelines::Vision::Private;

	/** Per-call data: the shared labels and the postprocess settings copy taken when the call was issued, the threshold, the result. */
	struct FObjectDetectionCall
	{
		FSharedLabels Labels;   // immutable, shared with the pipeline (no per-call copy)
		FPipelinePostprocessSettings Postprocess;
		float ScoreThreshold = 0.5f;
		TArray<FPipelineDetection> Result;
	};
	using FObjectDetectionCallRef = TSharedRef<FObjectDetectionCall, ESPMode::ThreadSafe>;

	static FPipelineResult NaNThreshold()
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("ScoreThreshold is NaN"));
	}

	/** float32 [1, Q, Last] with Q >= 1 (Last > 0 fixed, or -1 = any >= 1). OutQ / OutLast receive the sizes. */
	static bool CheckDetectionOutput(const FPipelineTensor& T, int64 Last, int64& OutQ, int64& OutLast)
	{
		const TConstArrayView<int64> Shape = T.GetShape();
		if (!T.IsValid() || T.GetType() != EPipelineTensorType::Float32 || Shape.Num() != 3 || Shape[0] != 1 || Shape[1] < 1 || Shape[2] < 1)
		{
			return false;
		}
		if (Last > 0 && Shape[2] != Last)
		{
			return false;
		}
		if (Shape[1] * Shape[2] > (int64)MAX_int32 || T.Num() != Shape[1] * Shape[2])
		{
			return false;
		}
		OutQ = Shape[1];
		OutLast = Shape[2];
		return true;
	}

	/** Logits [1, Q, C'] and boxes [1, Q, 4] -> DecodeDetections with the call's settings and geometry. */
	static FPipelineResult DecodeOutputs(const FObjectDetectionCall& Call, const FPipelineImageGeometry& Geometry, const FPipelineModelInfo& Info,
		const TArray<FPipelineTensor>& Outputs, TArray<FPipelineDetection>& OutResult)
	{
		const int32 LogitsIndex = Info.Bindings.MainOutput;
		const int32 BoxesIndex = Info.Bindings.BoxesOutput;
		if (!Outputs.IsValidIndex(LogitsIndex) || !Outputs.IsValidIndex(BoxesIndex))
		{
			return FPipelineResult::Fail(EPipelineError::InferenceFailed,
				FString::Printf(TEXT("logits output %d / boxes output %d not among %d model outputs"), LogitsIndex, BoxesIndex, Outputs.Num()));
		}
		const FPipelineTensor& Logits = Outputs[LogitsIndex];
		const FPipelineTensor& Boxes = Outputs[BoxesIndex];
		int64 Q = 0, C = 0, BoxQ = 0, Four = 0;
		if (!CheckDetectionOutput(Logits, -1, Q, C))
		{
			return FPipelineResult::Fail(EPipelineError::InferenceFailed,
				FString::Printf(TEXT("logits output is %s %s; expected float32 [1,Q,C] with Q >= 1"), LexToString(Logits.GetType()), *Logits.DescribeShape()));
		}
		if (!CheckDetectionOutput(Boxes, 4, BoxQ, Four) || BoxQ != Q)
		{
			return FPipelineResult::Fail(EPipelineError::InferenceFailed,
				FString::Printf(TEXT("boxes output is %s %s; expected float32 [1,%lld,4]"), LexToString(Boxes.GetType()), *Boxes.DescribeShape(), (long long)Q));
		}

		FPipelineDetectionDecodeParams Params;
		Params.Activation = Call.Postprocess.ScoreActivation;
		Params.ScoreThreshold = Call.ScoreThreshold;
		Params.NmsIouThreshold = Call.Postprocess.NmsIouThreshold;
		Params.MaxDetections = Call.Postprocess.MaxDetections;
		Params.Geometry = Geometry;
		const TConstArrayView64<float> L64 = Logits.GetFloatData();
		const TConstArrayView64<float> B64 = Boxes.GetFloatData();
		return DecodeDetections(TConstArrayView<float>(L64.GetData(), (int32)(Q * C)), (int32)Q, (int32)C,
			TConstArrayView<float>(B64.GetData(), (int32)(Q * 4)), LabelsOrEmpty(Call.Labels), Params, OutResult);
	}

	/** Verbose diagnostics: the timings of a finished call. */
	static void LogTimings(const TCHAR* Kind, const FPipelineResult& R, int32 NumDetections)
	{
		UE_LOG(LogModelPipelinesVision, Verbose, TEXT("ObjectDetection %s: ok %d, device %d, %d detections, queue %.2f ms, preprocess %.2f ms, inference %.2f ms, postprocess %.2f ms"),
			Kind, R.bSuccess ? 1 : 0, (int32)R.Device, NumDetections, R.QueueMs, R.PreprocessMs, R.InferenceMs, R.PostprocessMs);
	}

	/** Preprocess and Postprocess of one call. Captures only shared per-call data, never the pipeline. */
	static FPipelineStages MakeStages(const FVisionCallInputRef& Input, const FObjectDetectionCallRef& Call)
	{
		FPipelineStages Stages;
		Stages.Preprocess = [Input, Call](const FPipelineModelInfo& Info, TArray<FPipelineTensor>& OutInputs) -> FPipelineResult
		{
			if (FMath::IsNaN(Call->ScoreThreshold))
			{
				return NaNThreshold();   // before inference: nothing runs
			}
			return BuildVisionInputs(*Input, Info, OutInputs);
		};
		Stages.Postprocess = [Input, Call](const FPipelineModelInfo& Info, TArray<FPipelineTensor>&& Outputs) -> FPipelineResult
		{
			// Input->Geometry was written by Preprocess; the stage runner orders Preprocess before Postprocess.
			return DecodeOutputs(*Call, Input->Geometry, Info, Outputs, Call->Result);
		};
		return Stages;
	}

	static FObjectDetectionCallRef MakeCall(float ScoreThreshold, bool bCopySettings,
		const FSharedLabels& Labels, const FPipelinePostprocessSettings& Postprocess)
	{
		const FObjectDetectionCallRef Call = MakeShared<FObjectDetectionCall, ESPMode::ThreadSafe>();
		Call->ScoreThreshold = ScoreThreshold;
		if (bCopySettings)
		{
			Call->Labels = Labels;
			Call->Postprocess = Postprocess;
		}
		return Call;
	}
}

UObjectDetectionPipeline::UObjectDetectionPipeline()
{
	PipelineTask = EPipelineTask::ObjectDetection;
}

FPipelineResult UObjectDetectionPipeline::OnInitialize(UPipelineModelPackage* InPackage)
{
	FPipelineResult R = Super::OnInitialize(InPackage);
	if (R.bSuccess)
	{
		// One copy per initialization; calls share it read-only instead of copying the label list per call.
		SharedLabels = UE::ModelPipelines::Vision::Private::MakeSharedLabels(GetLabels());
	}
	return R;
}

void UObjectDetectionPipeline::DetectAsync(UTexture* Image, float ScoreThreshold, FPipelineDetectionCallback OnComplete)
{
	using namespace UE::ModelPipelines::Vision::Private;
	using namespace UE::ModelPipelines::Vision::DetectionPrivate;

	if (!OnComplete)
	{
		UE_LOG(LogModelPipelinesVision, Warning, TEXT("UObjectDetectionPipeline::DetectAsync: null callback, nothing runs."));
		return;
	}
	// Off the game thread: the texture is not touched and nothing is copied; RunStages fails the call (InvalidArgument).
	const bool bGameThread = IsInGameThread();
	const FVisionCallInputRef Input = MakeVisionCallInput(bGameThread ? GetPreprocessSettings() : FPipelinePreprocessSettings());
	const FObjectDetectionCallRef Call = MakeCall(ScoreThreshold, bGameThread, SharedLabels, GetPostprocessSettings());

	FPipelineStages Stages = MakeStages(Input, Call);
	if (bGameThread && IsReady() && !FMath::IsNaN(ScoreThreshold))
	{
		// NaN threshold: no read; Preprocess fails the call with InvalidArgument before inference.
		// The read always completes (success, failure, timeout or Cancelled); a failed read fails Preprocess.
		Stages.Prerequisite = StartTextureRead(Image, Input);
	}
	Stages.Complete = [Input, Call, Callback = MoveTemp(OnComplete)](const FPipelineResult& Result) mutable
	{
		const FPipelineResult Final = WithReadbackWait(Result, *Input);
		LogTimings(TEXT("texture async"), Final, Call->Result.Num());
		Callback(Final, Final.bSuccess ? Call->Result : TArray<FPipelineDetection>());
	};
	RunStages(MoveTemp(Stages));
}

void UObjectDetectionPipeline::DetectImageAsync(const FPipelineImage& Image, float ScoreThreshold, FPipelineDetectionCallback OnComplete)
{
	using namespace UE::ModelPipelines::Vision::Private;
	using namespace UE::ModelPipelines::Vision::DetectionPrivate;

	if (!OnComplete)
	{
		UE_LOG(LogModelPipelinesVision, Warning, TEXT("UObjectDetectionPipeline::DetectImageAsync: null callback, nothing runs."));
		return;
	}
	const bool bGameThread = IsInGameThread();
	const FVisionCallInputRef Input = MakeVisionCallInput(bGameThread ? GetPreprocessSettings() : FPipelinePreprocessSettings());
	const FObjectDetectionCallRef Call = MakeCall(ScoreThreshold, bGameThread, SharedLabels, GetPostprocessSettings());
	if (bGameThread)
	{
		Input->Image = Image;
	}

	FPipelineStages Stages = MakeStages(Input, Call);
	Stages.Complete = [Call, Callback = MoveTemp(OnComplete)](const FPipelineResult& Result) mutable
	{
		LogTimings(TEXT("image async"), Result, Call->Result.Num());
		Callback(Result, Result.bSuccess ? Call->Result : TArray<FPipelineDetection>());
	};
	RunStages(MoveTemp(Stages));
}

FPipelineResult UObjectDetectionPipeline::DetectImageBlocking(const FPipelineImage& Image, float ScoreThreshold, TArray<FPipelineDetection>& OutDetections)
{
	using namespace UE::ModelPipelines::Vision::Private;
	using namespace UE::ModelPipelines::Vision::DetectionPrivate;

	OutDetections.Reset();
	if (FMath::IsNaN(ScoreThreshold))
	{
		return NaNThreshold();
	}
	FString Reason;
	if (!Image.IsValid(&Reason))
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, FString::Printf(TEXT("invalid image: %s"), *Reason));
	}
	const FVisionCallInputRef Input = MakeVisionCallInput(GetPreprocessSettings());
	Input->Image = Image;
	const FObjectDetectionCallRef Call = MakeCall(ScoreThreshold, true, SharedLabels, GetPostprocessSettings());

	// No prerequisite here (C26 note): blocking calls take only CPU images.
	const FPipelineResult R = RunStagesBlocking(MakeStages(Input, Call));
	LogTimings(TEXT("blocking"), R, Call->Result.Num());
	if (R.bSuccess)
	{
		OutDetections = MoveTemp(Call->Result);
	}
	return R;
}
