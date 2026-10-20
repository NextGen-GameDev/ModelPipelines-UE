// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "TextEmbeddingPipeline.h"

#include "Containers/ArrayView.h"
#include "PipelineMath.h"
#include "UObject/StrongObjectPtr.h"
#include "PipelineModelPackage.h"
#include "PipelineTensor.h"
#include "PipelineTextChunks.h"
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

	using namespace UE::ModelPipelines::Text;
	using FEmbeddingsRef = TSharedRef<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe>;

	// All chunks' embeddings, in input order (C43). Appended by each chunk's commit on the game thread.
	const FEmbeddingsRef Total = MakeShared<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe>();
	// Off the game thread pass no state: RunStages fails the call there, and TextState is only read on the game thread.
	const FPipelineTextStatePtr State = IsInGameThread() ? TextState : nullptr;
	// Multi-chunk calls issue chunk k+1 from chunk k's Complete via this UObject: pin it until the final callback, so a pipeline
	// released mid-call still finishes the call like an unchunked one (C19/C43). Released on the game thread by the runner.
	TStrongObjectPtr<UTextEmbeddingPipeline> Pin;
	if (IsInGameThread() && NumTextChunks(Texts.Num()) > 1)
	{
		Pin.Reset(this);
	}
	const bool bSingle = Texts.Num() <= MaxTextsPerInference;
	RunTextChunksAsync(Texts.Num(),
		[this, Pin, Texts, State, Total, bSingle](int32 First, int32 Num)
		{
			const TSharedRef<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe> Chunk = MakeShared<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe>();
			FTextChunk Out;
			Out.Stages = MakeStages(State, bSingle ? Texts : TArray<FString>(Texts.GetData() + First, Num), Chunk);
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
				Callback(Result, TArray<FPipelineEmbedding>());
			}
		});
}

FPipelineResult UTextEmbeddingPipeline::EmbedBlocking(const TArray<FString>& Texts, TArray<FPipelineEmbedding>& OutEmbeddings)
{
	OutEmbeddings.Reset();
	const FPipelineResult Count = UE::ModelPipelines::Text::ValidateTextCount(Texts);
	if (!Count.bSuccess)
	{
		return Count;
	}
	using namespace UE::ModelPipelines::Text;
	using FEmbeddingsRef = TSharedRef<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe>;

	TArray<FPipelineEmbedding> Total;
	const bool bSingle = Texts.Num() <= MaxTextsPerInference;
	const FPipelineResult R = RunTextChunksBlocking(Texts.Num(),
		[this, &Texts, &Total, bSingle](int32 First, int32 Num)
		{
			const FEmbeddingsRef Chunk = MakeShared<TArray<FPipelineEmbedding>, ESPMode::ThreadSafe>();
			FTextChunk Out;
			Out.Stages = MakeStages(TextState, bSingle ? Texts : TArray<FString>(Texts.GetData() + First, Num), Chunk);
			Out.Commit = [&Total, Chunk]() { Total.Append(MoveTemp(*Chunk)); };
			return Out;
		},
		[this](FPipelineStages&& Stages) { return RunStagesBlocking(MoveTemp(Stages)); });
	if (R.bSuccess)
	{
		OutEmbeddings = MoveTemp(Total);
	}
	return R;
}
