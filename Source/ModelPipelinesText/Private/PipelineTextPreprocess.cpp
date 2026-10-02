// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineTextPreprocess.h"

#include "UObject/UObjectGlobals.h"

DEFINE_LOG_CATEGORY(LogModelPipelinesText);

namespace UE::ModelPipelines::Text
{
	namespace Private
	{
		static bool IsIntType(EPipelineTensorType Type)
		{
			return Type == EPipelineTensorType::Int64 || Type == EPipelineTensorType::Int32;
		}

		/** Tensor of Type (Int64 or Int32) and Shape holding Values. */
		static FPipelineTensor MakeIntTensor(EPipelineTensorType Type, TConstArrayView<int64> Shape, const TArray<int32>& Values)
		{
			if (Type == EPipelineTensorType::Int32)
			{
				return FPipelineTensor::FromInt32(Shape, Values);
			}
			TArray<int64> Wide;
			Wide.SetNumUninitialized(Values.Num());
			for (int32 i = 0; i < Values.Num(); ++i)
			{
				Wide[i] = Values[i];
			}
			return FPipelineTensor::FromInt64(Shape, Wide);
		}

		static FString TokenizerError(const UTokenizerWrapper& Tokenizer, const TCHAR* Fallback)
		{
			const FString Err = Tokenizer.GetLastError();
			return Err.IsEmpty() ? FString(Fallback) : Err;
		}

		static bool IsJsonSpace(TCHAR C)
		{
			return C == TCHAR(' ') || C == TCHAR('\t') || C == TCHAR('\n') || C == TCHAR('\r') || C == TCHAR(0xFEFF);
		}

		static void SkipJsonSpace(const FString& Json, int32& i)
		{
			while (i < Json.Len() && IsJsonSpace(Json[i]))
			{
				++i;
			}
		}

		/** i at the opening quote; on success i is one past the closing quote and OutRaw holds the content, still escaped. */
		static bool SkipJsonString(const FString& Json, int32& i, FString* OutRaw)
		{
			if (i >= Json.Len() || Json[i] != TCHAR('"'))
			{
				return false;
			}
			const int32 Start = ++i;
			while (i < Json.Len())
			{
				const TCHAR C = Json[i];
				if (C == TCHAR('\\'))
				{
					i += 2;
					continue;
				}
				if (C == TCHAR('"'))
				{
					if (OutRaw)
					{
						*OutRaw = Json.Mid(Start, i - Start);
					}
					++i;
					return true;
				}
				++i;
			}
			return false;
		}

		/** i at the first character of a value; on success i is one past its end. */
		static bool SkipJsonValue(const FString& Json, int32& i)
		{
			if (i >= Json.Len())
			{
				return false;
			}
			const TCHAR First = Json[i];
			if (First == TCHAR('"'))
			{
				return SkipJsonString(Json, i, nullptr);
			}
			if (First == TCHAR('{') || First == TCHAR('['))
			{
				int32 Depth = 0;
				while (i < Json.Len())
				{
					const TCHAR C = Json[i];
					if (C == TCHAR('"'))
					{
						if (!SkipJsonString(Json, i, nullptr))
						{
							return false;
						}
						continue;
					}
					if (C == TCHAR('{') || C == TCHAR('['))
					{
						++Depth;
					}
					else if (C == TCHAR('}') || C == TCHAR(']'))
					{
						if (--Depth == 0)
						{
							++i;
							return true;
						}
					}
					++i;
				}
				return false;
			}
			// Literal: number, true, false, null.
			const int32 Start = i;
			while (i < Json.Len() && !IsJsonSpace(Json[i]) && Json[i] != TCHAR(',') && Json[i] != TCHAR('}') && Json[i] != TCHAR(']'))
			{
				++i;
			}
			return i > Start;
		}

		/**
		 * Replaces the value of the top-level "padding" key of a tokenizer.json with null. The Python reference encodes with
		 * no_padding (padding is per batch, to the longest row, done by EncodeBatch); a file-level "Fixed" padding would add
		 * pad tokens the mask does not exclude. Only that span changes, so vocab and scores are kept byte for byte.
		 * Malformed JSON is returned unchanged (the tokenizer reports the error).
		 */
		static FString DisableFilePadding(const FString& Json)
		{
			int32 i = 0;
			SkipJsonSpace(Json, i);
			if (i >= Json.Len() || Json[i] != TCHAR('{'))
			{
				return Json;
			}
			++i;
			while (true)
			{
				SkipJsonSpace(Json, i);
				if (i >= Json.Len() || Json[i] == TCHAR('}'))
				{
					return Json;
				}
				FString Key;
				if (!SkipJsonString(Json, i, &Key))
				{
					return Json;
				}
				SkipJsonSpace(Json, i);
				if (i >= Json.Len() || Json[i] != TCHAR(':'))
				{
					return Json;
				}
				++i;
				SkipJsonSpace(Json, i);
				const int32 ValueStart = i;
				if (!SkipJsonValue(Json, i))
				{
					return Json;
				}
				if (Key.Equals(TEXT("padding"), ESearchCase::CaseSensitive))
				{
					if (Json.Mid(ValueStart, i - ValueStart).Equals(TEXT("null"), ESearchCase::CaseSensitive))
					{
						return Json;
					}
					return Json.Left(ValueStart) + TEXT("null") + Json.Mid(i);
				}
				SkipJsonSpace(Json, i);
				if (i < Json.Len() && Json[i] == TCHAR(','))
				{
					++i;
					continue;
				}
				return Json;
			}
		}

		/** Reads element i of an Int64/Int32 tensor as int64. */
		static int64 ReadInt(const FPipelineTensor& T, int64 i)
		{
			return T.GetType() == EPipelineTensorType::Int64 ? T.GetInt64Data()[i] : (int64)T.GetInt32Data()[i];
		}
	}

	FPipelineResult ValidateTextCount(const TArray<FString>& Texts)
	{
		if (Texts.Num() == 0)
		{
			return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("no texts"));
		}
		if (Texts.Num() > MaxTextsPerCall)
		{
			return FPipelineResult::Fail(EPipelineError::InvalidArgument,
				FString::Printf(TEXT("too many texts: %d (at most %d per call)"), Texts.Num(), MaxTextsPerCall));
		}
		return FPipelineResult::Ok();
	}

	FPipelineResult CreateTextState(UObject* Outer, const UPipelineModelPackage& Package, FPipelineTextStatePtr& OutState,
		UTokenizerWrapper*& OutTokenizer)
	{
		UTokenizerWrapper* Tokenizer = NewObject<UTokenizerWrapper>(Outer);
		if (!Tokenizer)
		{
			return FPipelineResult::Fail(EPipelineError::Internal, TEXT("could not create the tokenizer object"));
		}
		if (Package.TokenizerJson.IsEmpty())
		{
			Tokenizer->MarkAsGarbage();
			return FPipelineResult::Fail(EPipelineError::PreprocessFailed,
				FString::Printf(TEXT("package '%s' has no TokenizerJson"), *Package.GetName()));
		}
		if (!Tokenizer->InitializeTokenizerFromJson(Private::DisableFilePadding(Package.TokenizerJson)))
		{
			const FString Err = Private::TokenizerError(*Tokenizer, TEXT("unknown tokenizer error"));
			Tokenizer->MarkAsGarbage();
			return FPipelineResult::Fail(EPipelineError::PreprocessFailed,
				FString::Printf(TEXT("tokenizer of package '%s' did not load: %s"), *Package.GetName(), *Err));
		}

		FPipelineTextStatePtr State = MakeShared<FPipelineTextState, ESPMode::ThreadSafe>();
		State->Tokenizer.Reset(Tokenizer);
		State->Settings = Package.Preprocess;
		OutState = MoveTemp(State);
		OutTokenizer = Tokenizer;
		return FPipelineResult::Ok();
	}

	FPipelineResult BuildTextInputs(const FPipelineTextState& State, const TArray<FString>& Texts, const FPipelineModelInfo& Info,
		TArray<FPipelineTensor>& OutInputs, FPipelineTextBatch& OutBatch)
	{
		OutInputs.Reset();
		OutBatch = FPipelineTextBatch();

		const FPipelineResult CountCheck = ValidateTextCount(Texts);
		if (!CountCheck.bSuccess)
		{
			return CountCheck;
		}

		UTokenizerWrapper* Tokenizer = State.Tokenizer.Get();
		if (!Tokenizer)
		{
			return FPipelineResult::Fail(EPipelineError::PreprocessFailed, TEXT("tokenizer is not loaded"));
		}

		const FPipelineTaskBindings& B = Info.Bindings;
		const int32 NumInputs = Info.InputDescs.Num();
		if (!Info.InputDescs.IsValidIndex(B.InputIds) || !Info.InputDescs.IsValidIndex(B.AttentionMask)
			|| (B.TokenTypeIds >= 0 && !Info.InputDescs.IsValidIndex(B.TokenTypeIds)))
		{
			return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
				TEXT("model inputs are not bound: input_ids and attention_mask are required"));
		}
		for (int32 i = 0; i < NumInputs; ++i)
		{
			const FPipelineTensorDesc& Desc = Info.InputDescs[i];
			if (i != B.InputIds && i != B.AttentionMask && i != B.TokenTypeIds)
			{
				return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
					FString::Printf(TEXT("model input '%s' is not a text input (input_ids, attention_mask, token_type_ids)"), *Desc.Name));
			}
			if (!Desc.bTypeSupported || !Private::IsIntType(Desc.Type))
			{
				return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
					FString::Printf(TEXT("model input '%s' must be int64 or int32, is %s"), *Desc.Name,
						Desc.bTypeSupported ? LexToString(Desc.Type) : TEXT("unsupported")));
			}
		}

		const int32 MaxTokens = State.Settings.MaxTokens;
		FTokenizedBatch Batch;
		if (!Tokenizer->EncodeBatch(Texts, Batch, State.Settings.bAddSpecialTokens, MaxTokens, /*PadTokenId*/ 0))
		{
			return FPipelineResult::Fail(EPipelineError::PreprocessFailed,
				FString::Printf(TEXT("tokenization failed: %s"), *Private::TokenizerError(*Tokenizer, TEXT("EncodeBatch failed"))));
		}
		if (Batch.SeqLen == 0)
		{
			return FPipelineResult::Fail(EPipelineError::PreprocessFailed, TEXT("texts produced no tokens"));
		}

		// Validate before anything reaches ORT (runtime errors there can crash non-editor builds).
		const int64 Expected = (int64)Batch.NumRows * (int64)Batch.SeqLen;
		if (Batch.NumRows != Texts.Num() || Batch.SeqLen < 0 || Batch.InputIds.Num() != Expected || Batch.AttentionMask.Num() != Expected)
		{
			return FPipelineResult::Fail(EPipelineError::InvalidArgument,
				FString::Printf(TEXT("tokenizer returned %d rows x %d tokens (%d ids, %d mask values) for %d texts"),
					Batch.NumRows, Batch.SeqLen, Batch.InputIds.Num(), Batch.AttentionMask.Num(), Texts.Num()));
		}
		if (MaxTokens > 0 && Batch.SeqLen > MaxTokens)
		{
			return FPipelineResult::Fail(EPipelineError::InvalidArgument,
				FString::Printf(TEXT("tokenized length %d exceeds MaxTokens %d"), Batch.SeqLen, MaxTokens));
		}
		for (int32 i = 0; i < Batch.InputIds.Num(); ++i)
		{
			if (Batch.InputIds[i] < 0)
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument, FString::Printf(TEXT("negative token id %d"), Batch.InputIds[i]));
			}
			const int32 M = Batch.AttentionMask[i];
			if (M != 0 && M != 1)
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument, FString::Printf(TEXT("attention mask value %d is not 0 or 1"), M));
			}
		}

		const int64 ShapeArr[2] = { (int64)Batch.NumRows, (int64)Batch.SeqLen };
		const TConstArrayView<int64> Shape(ShapeArr, 2);
		OutInputs.SetNum(NumInputs);
		for (int32 i = 0; i < NumInputs; ++i)
		{
			const FPipelineTensorDesc& Desc = Info.InputDescs[i];
			FPipelineTensor& T = OutInputs[i];
			if (i == B.InputIds)
			{
				T = Private::MakeIntTensor(Desc.Type, Shape, Batch.InputIds);
			}
			else if (i == B.AttentionMask)
			{
				T = Private::MakeIntTensor(Desc.Type, Shape, Batch.AttentionMask);
			}
			else
			{
				T = FPipelineTensor::Zeros(Desc.Type, Shape);
			}
			if (!T.IsValid())
			{
				OutInputs.Reset();
				return FPipelineResult::Fail(EPipelineError::InvalidArgument,
					FString::Printf(TEXT("could not build input '%s' of shape [%d,%d]"), *Desc.Name, Batch.NumRows, Batch.SeqLen));
			}
		}

		// Final check of what will be sent: equal shapes, ids >= 0, mask in {0,1}.
		for (int32 i = 0; i < NumInputs; ++i)
		{
			const FPipelineTensor& T = OutInputs[i];
			const TConstArrayView<int64> S = T.GetShape();
			if (S.Num() != 2 || S[0] != ShapeArr[0] || S[1] != ShapeArr[1])
			{
				OutInputs.Reset();
				return FPipelineResult::Fail(EPipelineError::InvalidArgument,
					FString::Printf(TEXT("input '%s' has shape %s, expected [%d,%d]"), *Info.InputDescs[i].Name, *T.DescribeShape(), Batch.NumRows, Batch.SeqLen));
			}
			const bool bMask = (i == B.AttentionMask);
			for (int64 k = 0; k < T.Num(); ++k)
			{
				const int64 V = Private::ReadInt(T, k);
				if (V < 0 || (bMask && V > 1))
				{
					OutInputs.Reset();
					return FPipelineResult::Fail(EPipelineError::InvalidArgument,
						FString::Printf(TEXT("input '%s' has invalid value %lld"), *Info.InputDescs[i].Name, V));
				}
			}
		}

		OutBatch.AttentionMask = MoveTemp(Batch.AttentionMask);
		OutBatch.NumRows = Batch.NumRows;
		OutBatch.SeqLen = Batch.SeqLen;
		return FPipelineResult::Ok();
	}
}
