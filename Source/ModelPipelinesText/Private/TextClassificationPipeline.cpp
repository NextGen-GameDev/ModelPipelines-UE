// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "TextClassificationPipeline.h"

#include "Containers/ArrayView.h"
#include "PipelineMath.h"
#include "PipelineModelPackage.h"
#include "PipelineTensor.h"
#include "PipelineTextPreprocess.h"
#include "TokenizerWrapper.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(TextClassificationPipeline)

namespace UE::ModelPipelines::Text::ClassificationPrivate
{
	using FResultsRef = TSharedRef<TArray<FPipelineClassification>, ESPMode::ThreadSafe>;

	/** Per-call data: inputs, the settings copied when the call was issued, and the batch shape from Preprocess. */
	struct FClassificationCall
	{
		TArray<FString> Texts;
		TArray<FString> Labels;
		EPipelineScoreActivation Activation = EPipelineScoreActivation::Softmax;
		int32 TopK = 1;
		FPipelineTextBatch Batch;
	};

	/** Main output float32 [NumRows, C] (C == Labels.Num()) -> activated scores and top-k per row (C20). */
	static FPipelineResult ClassifyOutput(const FClassificationCall& Call, const FPipelineTensor& Out, TArray<FPipelineClassification>& OutResults)
	{
		OutResults.Reset();
		const int32 NumRows = Call.Batch.NumRows;
		const int32 NumClasses = Call.Labels.Num();
		const TConstArrayView<int64> Shape = Out.GetShape();

		const bool bShapeOk = NumClasses >= 1 && Out.IsValid() && Out.GetType() == EPipelineTensorType::Float32 && Shape.Num() == 2
			&& Shape[0] == NumRows && Shape[1] == NumClasses && Out.Num() == (int64)NumRows * NumClasses && Out.Num() <= MAX_int32;
		if (!bShapeOk)
		{
			return FPipelineResult::Fail(EPipelineError::InferenceFailed,
				FString::Printf(TEXT("model output is %s %s; expected float32 [%d,%d] (%d labels)"),
					LexToString(Out.GetType()), *Out.DescribeShape(), NumRows, NumClasses, NumClasses));
		}

		const TConstArrayView64<float> Data64 = Out.GetFloatData();
		const float* Data = Data64.GetData();
		const int32 K = FMath::Clamp(Call.TopK, 1, NumClasses);
		const bool bSigmoid = Call.Activation == EPipelineScoreActivation::Sigmoid;

		TArray<FPipelineClassification> Results;
		Results.SetNum(NumRows);
		for (int32 Row = 0; Row < NumRows; ++Row)
		{
			const TConstArrayView<float> Logits(Data + (int64)Row * NumClasses, NumClasses);
			FPipelineClassification& R = Results[Row];
			const bool bOk = bSigmoid ? Sigmoid(Logits, R.Scores) : Softmax(Logits, R.Scores);
			if (!bOk || R.Scores.Num() != NumClasses)
			{
				return FPipelineResult::Fail(EPipelineError::InferenceFailed,
					FString::Printf(TEXT("model output row %d contains NaN or Inf"), Row));
			}
			const TArray<int32> Best = TopKIndices(R.Scores, K);
			if (Best.Num() != K)
			{
				return FPipelineResult::Fail(EPipelineError::Internal,
					FString::Printf(TEXT("top-k returned %d of %d entries"), Best.Num(), K));
			}
			R.TopK.Reserve(K);
			for (const int32 Index : Best)
			{
				if (!R.Scores.IsValidIndex(Index))
				{
					return FPipelineResult::Fail(EPipelineError::Internal, FString::Printf(TEXT("top-k index %d out of range"), Index));
				}
				FPipelineLabelScore& E = R.TopK.AddDefaulted_GetRef();
				E.Label = Call.Labels[Index];
				E.Index = Index;
				E.Score = R.Scores[Index];
			}
			R.Label = R.TopK[0].Label;
			R.Index = R.TopK[0].Index;
			R.Score = R.TopK[0].Score;
		}
		OutResults = MoveTemp(Results);
		return FPipelineResult::Ok();
	}
}

UTextClassificationPipeline::UTextClassificationPipeline()
{
	PipelineTask = EPipelineTask::TextClassification;
}

FPipelineResult UTextClassificationPipeline::OnInitialize(UPipelineModelPackage* InPackage)
{
	using namespace UE::ModelPipelines::Text;

	if (!InPackage)
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("package is null"));
	}
	FPipelineTextStatePtr NewState;
	UTokenizerWrapper* NewTokenizer = nullptr;
	const FPipelineResult R = CreateTextState(this, *InPackage, NewState, NewTokenizer);
	if (!R.bSuccess)
	{
		UE_LOG(LogModelPipelinesText, Warning, TEXT("UTextClassificationPipeline: %s"), *R.Message);
		return R;
	}
	Tokenizer = NewTokenizer;
	TextState = MoveTemp(NewState);
	return FPipelineResult::Ok();
}

void UTextClassificationPipeline::BeginDestroy()
{
	// Game thread. In-flight calls keep their own references; the last one drops when the base destroys the stages there.
	TextState.Reset();
	Super::BeginDestroy();
}

FPipelineStages UTextClassificationPipeline::MakeStages(const TArray<FString>& Texts, int32 TopK,
	const TSharedRef<TArray<FPipelineClassification>, ESPMode::ThreadSafe>& Results)
{
	using namespace UE::ModelPipelines::Text;
	using namespace UE::ModelPipelines::Text::ClassificationPrivate;

	const TSharedRef<FClassificationCall, ESPMode::ThreadSafe> Call = MakeShared<FClassificationCall, ESPMode::ThreadSafe>();
	Call->Texts = Texts;
	Call->Labels = GetLabels();
	Call->Activation = GetPostprocessSettings().ScoreActivation;
	Call->TopK = TopK;
	const FPipelineTextStatePtr State = TextState;

	FPipelineStages Stages;
	Stages.Preprocess = [State, Call](const FPipelineModelInfo& Info, TArray<FPipelineTensor>& OutInputs) -> FPipelineResult
	{
		const FPipelineResult Count = ValidateTextCount(Call->Texts);
		if (!Count.bSuccess)
		{
			return Count;
		}
		if (!State.IsValid())
		{
			return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("pipeline is not initialized (no tokenizer)"));
		}
		return BuildTextInputs(*State, Call->Texts, Info, OutInputs, Call->Batch);
	};
	Stages.Postprocess = [State, Call, Results](const FPipelineModelInfo& Info, TArray<FPipelineTensor>&& Outputs) -> FPipelineResult
	{
		if (!State.IsValid())
		{
			return FPipelineResult::Fail(EPipelineError::Internal, TEXT("text state missing"));
		}
		if (!Outputs.IsValidIndex(Info.Bindings.MainOutput))
		{
			return FPipelineResult::Fail(EPipelineError::InferenceFailed,
				FString::Printf(TEXT("main output index %d not among %d model outputs"), Info.Bindings.MainOutput, Outputs.Num()));
		}
		TArray<FPipelineClassification> Out;
		const FPipelineResult R = ClassifyOutput(*Call, Outputs[Info.Bindings.MainOutput], Out);
		if (!R.bSuccess)
		{
			return R;
		}
		if (Out.Num() != Call->Texts.Num())
		{
			return FPipelineResult::Fail(EPipelineError::Internal,
				FString::Printf(TEXT("%d results for %d texts"), Out.Num(), Call->Texts.Num()));
		}
		*Results = MoveTemp(Out);
		return FPipelineResult::Ok();
	};
	return Stages;
}

void UTextClassificationPipeline::ClassifyAsync(const TArray<FString>& Texts, int32 TopK, FPipelineClassificationCallback OnComplete)
{
	if (!OnComplete)
	{
		UE_LOG(LogModelPipelinesText, Warning, TEXT("UTextClassificationPipeline::ClassifyAsync: null callback, nothing runs."));
		return;
	}

	const TSharedRef<TArray<FPipelineClassification>, ESPMode::ThreadSafe> Results = MakeShared<TArray<FPipelineClassification>, ESPMode::ThreadSafe>();
	FPipelineStages Stages = MakeStages(Texts, TopK, Results);
	Stages.Complete = [Results, Callback = MoveTemp(OnComplete)](const FPipelineResult& Result) mutable
	{
		if (Result.bSuccess)
		{
			Callback(Result, *Results);
		}
		else
		{
			Callback(Result, TArray<FPipelineClassification>());
		}
	};
	RunStages(MoveTemp(Stages));
}

FPipelineResult UTextClassificationPipeline::ClassifyBlocking(const TArray<FString>& Texts, int32 TopK, TArray<FPipelineClassification>& OutResults)
{
	OutResults.Reset();
	const FPipelineResult Count = UE::ModelPipelines::Text::ValidateTextCount(Texts);
	if (!Count.bSuccess)
	{
		return Count;
	}
	const TSharedRef<TArray<FPipelineClassification>, ESPMode::ThreadSafe> Results = MakeShared<TArray<FPipelineClassification>, ESPMode::ThreadSafe>();
	const FPipelineResult R = RunStagesBlocking(MakeStages(Texts, TopK, Results));
	if (R.bSuccess)
	{
		OutResults = MoveTemp(*Results);
	}
	return R;
}
