// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Logging/LogMacros.h"
#include "Templates/SharedPointer.h"
#include "UObject/StrongObjectPtr.h"
#include "PipelineBase.h"
#include "PipelineModelPackage.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "TokenizerWrapper.h"

DECLARE_LOG_CATEGORY_EXTERN(LogModelPipelinesText, Log, All);

namespace UE::ModelPipelines::Text
{
	/** Most texts one call accepts (docs/contracts.md C20). */
	static constexpr int32 MaxTextsPerCall = 1024;

	/**
	 * State shared by a text pipeline and its in-flight calls (C20). Not a UObject. The pipeline holds it as a thread-safe
	 * TSharedPtr and the stage lambdas capture it; the last reference drops on the game thread (BeginDestroy or the base
	 * destroying the stage functions there), so the TStrongObjectPtr is released there.
	 */
	struct FPipelineTextState
	{
		/** Initialized tokenizer; its public calls are serialized by its own lock. */
		TStrongObjectPtr<UTokenizerWrapper> Tokenizer;

		/** Copy of the package settings taken at initialization. */
		FPipelinePreprocessSettings Settings;
	};

	using FPipelineTextStatePtr = TSharedPtr<FPipelineTextState, ESPMode::ThreadSafe>;

	/** Tokenized batch of one call, carried from Preprocess to Postprocess. */
	struct FPipelineTextBatch
	{
		/** Row-major [NumRows * SeqLen]; 1 = token, 0 = padding. */
		TArray<int32> AttentionMask;
		int32 NumRows = 0;
		int32 SeqLen = 0;
	};

	/** Any thread. Empty -> InvalidArgument "no texts"; more than MaxTextsPerCall -> InvalidArgument; else Ok(). */
	FPipelineResult ValidateTextCount(const TArray<FString>& Texts);

	/** Game thread. New tokenizer (outer: Outer) loaded from Package->TokenizerJson and the package's settings.
	    Failure -> PreprocessFailed with the tokenizer's error; OutState untouched. */
	FPipelineResult CreateTextState(UObject* Outer, const UPipelineModelPackage& Package, FPipelineTextStatePtr& OutState,
		UTokenizerWrapper*& OutTokenizer);

	/**
	 * Background. Tokenizes Texts (special tokens and truncation per State.Settings, pad id 0) and builds one tensor per
	 * Info.InputDescs entry, in that order: input_ids, attention_mask, token_type_ids (zeros), each in its desc's int type.
	 * Validates ids >= 0, mask in {0,1}, equal shapes, SeqLen <= MaxTokens, NumRows == Texts.Num().
	 * OutBatch receives the mask and the batch shape.
	 */
	FPipelineResult BuildTextInputs(const FPipelineTextState& State, const TArray<FString>& Texts, const FPipelineModelInfo& Info,
		TArray<FPipelineTensor>& OutInputs, FPipelineTextBatch& OutBatch);
}
