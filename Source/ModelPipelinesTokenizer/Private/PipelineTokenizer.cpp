// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineTokenizer.h"

#include "PipelineTokenizerLog.h"
#include "HAL/PlatformString.h"
#include "Math/NumericLimits.h"
#include "Misc/ScopeLock.h"

THIRD_PARTY_INCLUDES_START
#include <mpt_tokenizers_c.h>
THIRD_PARTY_INCLUDES_END

#include UE_INLINE_GENERATED_CPP_BY_NAME(PipelineTokenizer)

static_assert(sizeof(SIZE_T) == sizeof(size_t), "SIZE_T and size_t must have the same size");

namespace UE::ModelPipelines::Tokenizer::Private
{
	/** Converts UTF-8 bytes with an explicit length (never null-terminated) to an FString. */
	static FString Utf8ToFString(const char* Data, SIZE_T Len)
	{
		if (Data == nullptr || Len == 0 || Len > static_cast<SIZE_T>(MAX_int32))
		{
			return FString();
		}
		return FString::ConstructFromPtrSize(reinterpret_cast<const UTF8CHAR*>(Data), static_cast<int32>(Len));
	}

	/** Converts exactly SrcLen TCHARs to UTF-8 (a trailing 0 is kept). Out holds the bytes, no terminator. */
	static bool TCharToUtf8(const TCHAR* Src, int32 SrcLen, TArray<UTF8CHAR>& Out)
	{
		Out.Reset();
		if (Src == nullptr || SrcLen <= 0)
		{
			return true;
		}
		const int32 DestLen = FPlatformString::ConvertedLength<UTF8CHAR>(Src, SrcLen);
		if (DestLen <= 0)
		{
			return DestLen == 0;
		}
		Out.SetNumUninitialized(DestLen);
		const UTF8CHAR* End = FPlatformString::Convert(Out.GetData(), DestLen, Src, SrcLen);
		if (End == nullptr)
		{
			Out.Reset();
			return false;
		}
		Out.SetNum(static_cast<int32>(End - Out.GetData()), EAllowShrinking::No);
		return true;
	}

	/** Pointer for the C API: (NULL, 0) for an empty buffer. */
	static const char* Utf8Data(const TArray<UTF8CHAR>& Bytes)
	{
		return Bytes.Num() > 0 ? reinterpret_cast<const char*>(Bytes.GetData()) : nullptr;
	}

	/** The library's thread-local message for the most recent failure on this thread (owned by the library). */
	static FString LibraryError()
	{
		const char* Data = nullptr;
		size_t Len = 0;
		mpt_tokenizers_get_last_error(&Data, &Len);
		FString Message = Utf8ToFString(Data, Len);
		return Message.IsEmpty() ? FString(TEXT("(no message from tokenizers library)")) : Message;
	}
}

namespace TokPrivate = UE::ModelPipelines::Tokenizer::Private;

void UPipelineTokenizer::BeginDestroy()
{
	{
		FScopeLock Lock(&Mutex);
		if (Tokenizer != nullptr)
		{
			mpt_tokenizers_free(Tokenizer);
			Tokenizer = nullptr;
		}
	}
	Super::BeginDestroy();
}

void UPipelineTokenizer::SetError(const TCHAR* FunctionName, const FString& Message)
{
	LastError = FString::Printf(TEXT("UPipelineTokenizer::%s: %s"), FunctionName, *Message);
	UE_LOG(LogModelPipelinesTokenizer, Warning, TEXT("%s"), *LastError);
}

bool UPipelineTokenizer::InitializeTokenizerFromJson(const FString& JsonBlob)
{
	const TCHAR* Fn = TEXT("InitializeTokenizerFromJson");
	FScopeLock Lock(&Mutex);
	LastError.Reset();

	// A leading U+FEFF (byte order mark) is skipped; the parser rejects it.
	const int32 Skip = (JsonBlob.Len() > 0 && JsonBlob[0] == TCHAR(0xFEFF)) ? 1 : 0;
	TArray<UTF8CHAR> Utf8;
	if (!TokPrivate::TCharToUtf8(*JsonBlob + Skip, JsonBlob.Len() - Skip, Utf8))
	{
		SetError(Fn, TEXT("could not convert the JSON to UTF-8"));
		return false;
	}

	MptTokenizerHandle NewHandle = mpt_tokenizers_new_from_str(TokPrivate::Utf8Data(Utf8), static_cast<size_t>(Utf8.Num()));
	if (NewHandle == nullptr)
	{
		SetError(Fn, FString::Printf(TEXT("failed to parse tokenizer JSON: %s"), *TokPrivate::LibraryError()));
		return false;
	}

	// Replace only after the new tokenizer parsed, so a failure keeps the previous one.
	if (Tokenizer != nullptr)
	{
		mpt_tokenizers_free(Tokenizer);
	}
	Tokenizer = NewHandle;
	return true;
}

bool UPipelineTokenizer::IsInitialized() const
{
	FScopeLock Lock(&Mutex);
	return Tokenizer != nullptr;
}

TArray<int32> UPipelineTokenizer::Encode(const FString& Text, bool bAddSpecialTokens)
{
	const TCHAR* Fn = TEXT("Encode");
	FScopeLock Lock(&Mutex);
	LastError.Reset();

	TArray<int32> Ids;
	if (Tokenizer == nullptr)
	{
		SetError(Fn, TEXT("tokenizer is not initialized"));
		return Ids;
	}

	TArray<UTF8CHAR> Utf8;
	if (!TokPrivate::TCharToUtf8(*Text, Text.Len(), Utf8))
	{
		SetError(Fn, TEXT("could not convert the text to UTF-8"));
		return Ids;
	}

	// token_ids is allocated by the library; freed only with mpt_tokenizers_free_encode_results.
	MptTokenizerEncodeResult Result{};
	const int32 Status = mpt_tokenizers_encode(Tokenizer, TokPrivate::Utf8Data(Utf8),
		static_cast<size_t>(Utf8.Num()), bAddSpecialTokens ? 1 : 0, &Result);
	if (Status != MPT_TOKENIZERS_OK)
	{
		SetError(Fn, FString::Printf(TEXT("mpt_tokenizers_encode failed (%d): %s"), Status, *TokPrivate::LibraryError()));
	}
	else if (Result.len > static_cast<size_t>(MAX_int32))
	{
		// A result longer than TArray<int32> can index is dropped, never truncated.
		SetError(Fn, FString::Printf(TEXT("result has %llu ids, more than an array can hold"),
			static_cast<unsigned long long>(Result.len)));
	}
	else if (Result.token_ids != nullptr && Result.len > 0)
	{
		Ids.Append(reinterpret_cast<const int32*>(Result.token_ids), static_cast<int32>(Result.len));
	}
	mpt_tokenizers_free_encode_results(&Result, 1);
	return Ids;
}

bool UPipelineTokenizer::EncodeBatch(const TArray<FString>& Texts, FPipelineTokenizedBatch& OutBatch,
	bool bAddSpecialTokens, int32 MaxLength, int32 PadTokenId)
{
	const TCHAR* Fn = TEXT("EncodeBatch");
	FScopeLock Lock(&Mutex);
	LastError.Reset();
	OutBatch = FPipelineTokenizedBatch();

	if (Tokenizer == nullptr)
	{
		SetError(Fn, TEXT("tokenizer is not initialized"));
		return false;
	}
	if (MaxLength < 0)
	{
		SetError(Fn, FString::Printf(TEXT("MaxLength %d is negative (0 = no truncation)"), MaxLength));
		return false;
	}
	if (PadTokenId < 0)
	{
		SetError(Fn, FString::Printf(TEXT("PadTokenId %d is negative"), PadTokenId));
		return false;
	}

	const int32 NumTexts = Texts.Num();
	if (NumTexts == 0)
	{
		return true;
	}

	// One UTF-8 buffer per text, alive until the C call returns. Exact lengths, a trailing 0 is kept.
	TArray<TArray<UTF8CHAR>> Buffers;
	Buffers.SetNum(NumTexts);
	TArray<const char*> DataPtrs;
	DataPtrs.SetNumUninitialized(NumTexts);
	TArray<SIZE_T> Lens;
	Lens.SetNumUninitialized(NumTexts);
	for (int32 Row = 0; Row < NumTexts; ++Row)
	{
		if (!TokPrivate::TCharToUtf8(*Texts[Row], Texts[Row].Len(), Buffers[Row]))
		{
			SetError(Fn, FString::Printf(TEXT("could not convert text %d to UTF-8"), Row));
			return false;
		}
		DataPtrs[Row] = TokPrivate::Utf8Data(Buffers[Row]);
		Lens[Row] = static_cast<SIZE_T>(Buffers[Row].Num());
	}

	// Each results[i].token_ids is allocated by the library; freed below with mpt_tokenizers_free_encode_results.
	TArray<MptTokenizerEncodeResult> Results;
	Results.SetNumZeroed(NumTexts);
	const int32 Status = mpt_tokenizers_encode_batch_truncated(Tokenizer, DataPtrs.GetData(),
		reinterpret_cast<size_t*>(Lens.GetData()), static_cast<size_t>(NumTexts), bAddSpecialTokens ? 1 : 0,
		static_cast<size_t>(MaxLength), Results.GetData());
	if (Status != MPT_TOKENIZERS_OK)
	{
		SetError(Fn, FString::Printf(TEXT("mpt_tokenizers_encode_batch_truncated failed (%d): %s"),
			Status, *TokPrivate::LibraryError()));
		mpt_tokenizers_free_encode_results(Results.GetData(), static_cast<size_t>(NumTexts));
		return false;
	}

	size_t Longest = 0;
	for (const MptTokenizerEncodeResult& Result : Results)
	{
		const size_t RowLen = Result.token_ids != nullptr ? Result.len : 0;
		Longest = RowLen > Longest ? RowLen : Longest;
	}

	const uint64 Total = static_cast<uint64>(NumTexts) * static_cast<uint64>(Longest);
	if (Longest > static_cast<size_t>(MAX_int32) || Total > static_cast<uint64>(MAX_int32))
	{
		SetError(Fn, FString::Printf(TEXT("%d rows x %llu ids is more than an array can hold"),
			NumTexts, static_cast<unsigned long long>(Longest)));
		mpt_tokenizers_free_encode_results(Results.GetData(), static_cast<size_t>(NumTexts));
		return false;
	}

	const int32 SeqLen = static_cast<int32>(Longest);
	FPipelineTokenizedBatch Batch;
	Batch.NumRows = NumTexts;
	Batch.SeqLen = SeqLen;
	Batch.InputIds.Init(PadTokenId, static_cast<int32>(Total));
	Batch.AttentionMask.Init(0, static_cast<int32>(Total));
	for (int32 Row = 0; Row < NumTexts; ++Row)
	{
		const MptTokenizerEncodeResult& Result = Results[Row];
		if (Result.token_ids == nullptr || Result.len == 0)
		{
			continue;
		}
		const int32 RowLen = static_cast<int32>(Result.len);
		const int32 Base = Row * SeqLen;
		FMemory::Memcpy(Batch.InputIds.GetData() + Base, Result.token_ids, sizeof(int32) * RowLen);
		for (int32 Col = 0; Col < RowLen; ++Col)
		{
			Batch.AttentionMask[Base + Col] = 1;
		}
	}
	mpt_tokenizers_free_encode_results(Results.GetData(), static_cast<size_t>(NumTexts));

	OutBatch = MoveTemp(Batch);
	return true;
}

int32 UPipelineTokenizer::GetVocabSize()
{
	const TCHAR* Fn = TEXT("GetVocabSize");
	FScopeLock Lock(&Mutex);
	LastError.Reset();

	if (Tokenizer == nullptr)
	{
		SetError(Fn, TEXT("tokenizer is not initialized"));
		return -1;
	}

	size_t Size = 0;
	const int32 Status = mpt_tokenizers_get_vocab_size(Tokenizer, &Size);
	if (Status != MPT_TOKENIZERS_OK)
	{
		SetError(Fn, FString::Printf(TEXT("mpt_tokenizers_get_vocab_size failed (%d): %s"), Status, *TokPrivate::LibraryError()));
		return -1;
	}
	return Size > static_cast<size_t>(MAX_int32) ? MAX_int32 : static_cast<int32>(Size);
}

FString UPipelineTokenizer::GetLastError() const
{
	FScopeLock Lock(&Mutex);
	return LastError;
}
