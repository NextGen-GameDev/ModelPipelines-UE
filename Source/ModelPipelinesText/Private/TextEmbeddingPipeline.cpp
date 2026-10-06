// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "TextEmbeddingPipeline.h"

#include "Containers/ArrayView.h"
#include "PipelineMath.h"
#include "PipelineModelPackage.h"
#include "PipelineTensor.h"
#include "PipelineTextPreprocess.h"
#include "PipelineTokenizer.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(TextEmbeddingPipeline)

namespace UE::ModelPipelines::Text::EmbeddingPrivate
{
	using FEmbeddingsRef = TSharedRef<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe>;

	/** Per-call data carried from Preprocess to Postprocess (the batch's mask and shape). */
	struct FEmbeddingCall
	{
		TArray<FString> Texts;
		FPipelineTextBatch Batch;
	};

	/** Pools the main output into one embedding per row and L2-normalizes when asked (C20). */
	static FPipelineResult PoolOutput(EPipelinePooling Pooling, bool bNormalize, const FPipelineTextBatch& Batch,
		const FPipelineTensor& Out, TArray<FPipelineEmbedding>& OutEmbeddings)
	{
		OutEmbeddings.Reset();
		const TConstArrayView<int64> Shape = Out.GetShape();
		const bool bPooled = (Pooling == EPipelinePooling::Mean || Pooling == EPipelinePooling::CLS);
		const int32 ExpectedRank = bPooled ? 3 : 2;

		bool bShapeOk = Out.IsValid() && Out.GetType() == EPipelineTensorType::Float32 && Shape.Num() == ExpectedRank
			&& Shape[0] == Batch.NumRows;
		if (bShapeOk && bPooled)
		{
			bShapeOk = Shape[1] == Batch.SeqLen;
		}
		const int64 Dim64 = bShapeOk ? Shape.Last() : 0;
		bShapeOk = bShapeOk && Dim64 >= 1 && Dim64 <= MAX_int32 && Out.Num() <= MAX_int32;
		if (!bShapeOk)
		{
			return FPipelineResult::Fail(EPipelineError::InferenceFailed,
				FString::Printf(TEXT("model output is %s %s; expected float32 %s for pooling %s"),
					LexToString(Out.GetType()), *Out.DescribeShape(),
					bPooled ? *FString::Printf(TEXT("[%d,%d,D]"), Batch.NumRows, Batch.SeqLen) : *FString::Printf(TEXT("[%d,D]"), Batch.NumRows),
					Pooling == EPipelinePooling::Mean ? TEXT("Mean") : (Pooling == EPipelinePooling::CLS ? TEXT("CLS") : TEXT("None"))));
		}
		const int32 Dim = (int32)Dim64;

		const TConstArrayView64<float> Data64 = Out.GetFloatData();
		const TConstArrayView<float> Hidden(Data64.GetData(), (int32)Data64.Num());
		if (!AllFinite(Hidden))
		{
			return FPipelineResult::Fail(EPipelineError::InferenceFailed, TEXT("model output contains NaN or Inf"));
		}

		TArray<float> Pooled;
		if (Pooling == EPipelinePooling::Mean)
		{
			if (!MeanPool(Hidden, Batch.AttentionMask, Batch.NumRows, Batch.SeqLen, Dim, Pooled))
			{
				return FPipelineResult::Fail(EPipelineError::InferenceFailed,
					FString::Printf(TEXT("mean pooling failed for output %s"), *Out.DescribeShape()));
			}
		}
		else if (Pooling == EPipelinePooling::CLS)
		{
			if (!ClsPool(Hidden, Batch.NumRows, Batch.SeqLen, Dim, Pooled))
			{
				return FPipelineResult::Fail(EPipelineError::InferenceFailed,
					FString::Printf(TEXT("CLS pooling failed for output %s"), *Out.DescribeShape()));
			}
		}
		else
		{
			Pooled = TArray<float>(Hidden.GetData(), Hidden.Num());
		}

		if (Pooled.Num() != Batch.NumRows * Dim)
		{
			return FPipelineResult::Fail(EPipelineError::Internal, TEXT("pooled size does not match the batch"));
		}

		OutEmbeddings.SetNum(Batch.NumRows);
		for (int32 Row = 0; Row < Batch.NumRows; ++Row)
		{
			TArray<float>& Values = OutEmbeddings[Row].Values;
			Values.SetNumUninitialized(Dim);
			FMemory::Memcpy(Values.GetData(), Pooled.GetData() + (int64)Row * Dim, sizeof(float) * Dim);
			if (bNormalize)
			{
				L2NormalizeInPlace(Values);
			}
		}
		return FPipelineResult::Ok();
	}
}

UTextEmbeddingPipeline::UTextEmbeddingPipeline()
{
	PipelineTask = EPipelineTask::TextEmbedding;
}

FPipelineResult UTextEmbeddingPipeline::OnInitialize(UPipelineModelPackage* InPackage)
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
		UE_LOG(LogModelPipelinesText, Warning, TEXT("UTextEmbeddingPipeline: %s"), *R.Message);
		return R;
	}
	Tokenizer = NewTokenizer;
	TextState = MoveTemp(NewState);
	return FPipelineResult::Ok();
}

FPipelineStages UTextEmbeddingPipeline::MakeStages(const UE::ModelPipelines::Text::FPipelineTextStatePtr& State,
	const TArray<FString>& Texts,
	const TSharedRef<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe>& Embeddings)
{
	using namespace UE::ModelPipelines::Text;
	using namespace UE::ModelPipelines::Text::EmbeddingPrivate;

	const TSharedRef<FEmbeddingCall, ESPMode::ThreadSafe> Call = MakeShared<FEmbeddingCall, ESPMode::ThreadSafe>();
	Call->Texts = Texts;
	
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
	Stages.Postprocess = [State, Call, Embeddings](const FPipelineModelInfo& Info, TArray<FPipelineTensor>&& Outputs) -> FPipelineResult
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
		TArray<FPipelineEmbedding> Result;
		const FPipelineResult R = PoolOutput(State->Settings.Pooling, State->Settings.bNormalizeEmbeddings, Call->Batch,
			Outputs[Info.Bindings.MainOutput], Result);
		if (!R.bSuccess)
		{
			return R;
		}
		if (Result.Num() != Call->Texts.Num())
		{
			return FPipelineResult::Fail(EPipelineError::Internal,
				FString::Printf(TEXT("%d embeddings for %d texts"), Result.Num(), Call->Texts.Num()));
		}
		*Embeddings = MoveTemp(Result);
		return FPipelineResult::Ok();
	};
	return Stages;
}

void UTextEmbeddingPipeline::EmbedAsync(const TArray<FString>& Texts, FPipelineEmbeddingCallback OnComplete)
{
	if (!OnComplete)
	{
		UE_LOG(LogModelPipelinesText, Warning, TEXT("UTextEmbeddingPipeline::EmbedAsync: null callback, nothing runs."));
		return;
	}

	const TSharedRef<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe> Embeddings = MakeShared<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe>();
	// Off the game thread pass no state: RunStages fails the call there, and TextState is only read on the game thread.
	const UE::ModelPipelines::Text::FPipelineTextStatePtr State = IsInGameThread() ? TextState : nullptr;
	FPipelineStages Stages = MakeStages(State, Texts, Embeddings);
	Stages.Complete = [Embeddings, Callback = MoveTemp(OnComplete)](const FPipelineResult& Result) mutable
	{
		if (Result.bSuccess)
		{
			Callback(Result, *Embeddings);
		}
		else
		{
			Callback(Result, TArray<FPipelineEmbedding>());
		}
	};
	RunStages(MoveTemp(Stages));
}

FPipelineResult UTextEmbeddingPipeline::EmbedBlocking(const TArray<FString>& Texts, TArray<FPipelineEmbedding>& OutEmbeddings)
{
	OutEmbeddings.Reset();
	const FPipelineResult Count = UE::ModelPipelines::Text::ValidateTextCount(Texts);
	if (!Count.bSuccess)
	{
		return Count;
	}
	const TSharedRef<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe> Embeddings = MakeShared<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe>();
	const FPipelineResult R = RunStagesBlocking(MakeStages(TextState, Texts, Embeddings));
	if (R.bSuccess)
	{
		OutEmbeddings = MoveTemp(*Embeddings);
	}
	return R;
}
