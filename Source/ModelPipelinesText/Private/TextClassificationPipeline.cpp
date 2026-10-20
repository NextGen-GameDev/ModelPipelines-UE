// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "TextClassificationPipeline.h"

#include "Containers/ArrayView.h"
#include "PipelineMath.h"
#include "UObject/StrongObjectPtr.h"
#include "PipelineModelPackage.h"
#include "PipelineTensor.h"
#include "PipelineTextChunks.h"
#include "PipelineTextPreprocess.h"
#include "PipelineTokenizer.h"

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
	UPipelineTokenizer* NewTokenizer = nullptr;
	const FPipelineResult R = CreateTextState(*InPackage, NewState, NewTokenizer);
	if (!R.bSuccess)
	{
		UE_LOG(LogModelPipelinesText, Warning, TEXT("UTextClassificationPipeline: %s"), *R.Message);
		return R;
	}
	Tokenizer = NewTokenizer;
	TextState = MoveTemp(NewState);
	return FPipelineResult::Ok();
}

FPipelineStages UTextClassificationPipeline::MakeStages(const UE::ModelPipelines::Text::FPipelineTextStatePtr& State,
	const TArray<FString>& Texts, int32 TopK,
	const TSharedRef<TArray<FPipelineClassification>, ESPMode::ThreadSafe>& Results)
{
	using namespace UE::ModelPipelines::Text;
	using namespace UE::ModelPipelines::Text::ClassificationPrivate;

	const TSharedRef<FClassificationCall, ESPMode::ThreadSafe> Call = MakeShared<FClassificationCall, ESPMode::ThreadSafe>();
	Call->Texts = Texts;
	Call->TopK = TopK;
	if (State.IsValid())
	{
		// Only with a state: a null state means the call is failed anyway (possibly issued off the game thread).
		Call->Labels = GetLabels();
		Call->Activation = GetPostprocessSettings().ScoreActivation;
	}
	
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
			return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("pipeline is not initialized (no tokenizer)"));
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

	using namespace UE::ModelPipelines::Text;
	using FResultsRef = TSharedRef<TArray<FPipelineClassification>, ESPMode::ThreadSafe>;

	// All chunks' results, in input order (C43). Appended by each chunk's commit on the game thread.
	const FResultsRef Total = MakeShared<TArray<FPipelineClassification>, ESPMode::ThreadSafe>();
	// Off the game thread pass no state: RunStages fails the call there, and TextState is only read on the game thread.
	const FPipelineTextStatePtr State = IsInGameThread() ? TextState : nullptr;
	// Multi-chunk calls issue chunk k+1 from chunk k's Complete via this UObject: pin it until the final callback, so a pipeline
	// released mid-call still finishes the call like an unchunked one (C19/C43). Released on the game thread by the runner.
	TStrongObjectPtr<UTextClassificationPipeline> Pin;
	if (IsInGameThread() && NumTextChunks(Texts.Num()) > 1)
	{
		Pin.Reset(this);
	}
	const bool bSingle = Texts.Num() <= MaxTextsPerInference;
	RunTextChunksAsync(Texts.Num(),
		[this, Pin, Texts, State, Total, TopK, bSingle](int32 First, int32 Num)
		{
			const TSharedRef<TArray<FPipelineClassification>, ESPMode::ThreadSafe> Chunk = MakeShared<TArray<FPipelineClassification>, ESPMode::ThreadSafe>();
			FTextChunk Out;
			Out.Stages = MakeStages(State, bSingle ? Texts : TArray<FString>(Texts.GetData() + First, Num), TopK, Chunk);
			Out.Commit = [Total, Chunk]() { Total->Append(MoveTemp(*Chunk)); };
			return Out;
		},
		[this, Pin](FPipelineStages&& Stages) { RunStages(MoveTemp(Stages)); },
		[Total, Callback = MoveTemp(OnComplete)](const FPipelineResult& Result) mutable
		{
			if (Result.bSuccess)
			{
				Callback(Result, *Total);
			}
			else
			{
				Callback(Result, TArray<FPipelineClassification>());
			}
		});
}

FPipelineResult UTextClassificationPipeline::ClassifyBlocking(const TArray<FString>& Texts, int32 TopK, TArray<FPipelineClassification>& OutResults)
{
	OutResults.Reset();
	const FPipelineResult Count = UE::ModelPipelines::Text::ValidateTextCount(Texts);
	if (!Count.bSuccess)
	{
		return Count;
	}
	using namespace UE::ModelPipelines::Text;
	using FResultsRef = TSharedRef<TArray<FPipelineClassification>, ESPMode::ThreadSafe>;

	TArray<FPipelineClassification> Total;
	const bool bSingle = Texts.Num() <= MaxTextsPerInference;
	const FPipelineResult R = RunTextChunksBlocking(Texts.Num(),
		[this, &Texts, &Total, TopK, bSingle](int32 First, int32 Num)
		{
			const FResultsRef Chunk = MakeShared<TArray<FPipelineClassification>, ESPMode::ThreadSafe>();
			FTextChunk Out;
			Out.Stages = MakeStages(TextState, bSingle ? Texts : TArray<FString>(Texts.GetData() + First, Num), TopK, Chunk);
			Out.Commit = [&Total, Chunk]() { Total.Append(MoveTemp(*Chunk)); };
			return Out;
		},
		[this](FPipelineStages&& Stages) { return RunStagesBlocking(MoveTemp(Stages)); });
	if (R.bSuccess)
	{
		OutResults = MoveTemp(Total);
	}
	return R;
}
