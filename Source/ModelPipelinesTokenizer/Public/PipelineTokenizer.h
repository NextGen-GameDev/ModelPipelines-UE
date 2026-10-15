// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "HAL/CriticalSection.h"
#include "UObject/Object.h"
#include "PipelineTokenizer.generated.h"

/** Result of UPipelineTokenizer::EncodeBatch: NumRows rows of SeqLen ids each, row-major, right-padded. */
USTRUCT()
struct MODELPIPELINESTOKENIZER_API FPipelineTokenizedBatch
{
	GENERATED_BODY()

	/** Row-major [NumRows * SeqLen], right-padded; row r is InputIds[r*SeqLen .. r*SeqLen+SeqLen-1]. */
	TArray<int32> InputIds;

	/** Same shape; 1 = real token, 0 = padding. */
	TArray<int32> AttentionMask;

	int32 NumRows = 0;

	/** Length of the longest row after truncation. */
	int32 SeqLen = 0;
};

/**
 * ModelPipelines' internal tokenizer: one Hugging Face tokenizer.json over the renamed mpt_tokenizers_c lib
 * (independent of the Tokenizers plugin; both can be loaded in one process).
 *
 * Errors never crash: calls return false / an empty result / -1, GetLastError() returns
 * "UPipelineTokenizer::<FunctionName>: <reason>", and that same string is logged once as a Warning
 * (LogModelPipelinesTokenizer). Successful calls clear the last error and log nothing at Warning or above.
 *
 * Thread safety: every public call on one object is serialized by a per-object lock, so an object may be
 * called from any thread; different objects run in parallel. Keeping the object alive is the caller's job.
 */
UCLASS(NotBlueprintable, NotBlueprintType, Transient)
class MODELPIPELINESTOKENIZER_API UPipelineTokenizer : public UObject
{
	GENERATED_BODY()

public:
	/** Parse a tokenizer.json string. A leading U+FEFF (BOM) is skipped.
	    On success replaces (and frees) any previous tokenizer.
	    On failure the previous tokenizer stays loaded and false is returned. */
	bool InitializeTokenizerFromJson(const FString& JsonBlob);

	bool IsInitialized() const;

	/** Empty array on error (not initialized, library error). */
	TArray<int32> Encode(const FString& Text, bool bAddSpecialTokens = false);

	/** Encodes all texts in one C call. MaxLength 0 = no truncation; > 0 = HF truncation (special tokens included).
	    Pads on the right with PadTokenId to the longest row; mask 1/0.
	    Empty Texts -> true, NumRows = SeqLen = 0, arrays empty.
	    Errors (not initialized, MaxLength < 0, PadTokenId < 0, library error, NumRows*SeqLen > MAX_int32)
	    -> false and OutBatch reset to empty. */
	bool EncodeBatch(const TArray<FString>& Texts, FPipelineTokenizedBatch& OutBatch,
		bool bAddSpecialTokens = false, int32 MaxLength = 0, int32 PadTokenId = 0);

	/** Vocabulary size including added tokens. -1 when not initialized or on a library error. Clamped to MAX_int32. */
	int32 GetVocabSize();

	/** Decodes ids to text, like HF tokenizer.decode(ids, skip_special_tokens).
	    Empty Ids -> true, OutText "". A negative id -> false (no library call).
	    Not initialized / library error -> false, OutText "". */
	bool Decode(TConstArrayView<int32> Ids, bool bSkipSpecialTokens, FString& OutText);

	/** The vocabulary piece of one id (e.g. "▁hello"). An id not in the vocabulary -> true, OutPiece "".
	    Negative id / not initialized / library error -> false, OutPiece "". */
	bool IdToToken(int32 Id, FString& OutPiece);

	/** Message from the most recent call on this object: "" if that call succeeded. */
	FString GetLastError() const;

	/** Frees the handle (under the lock), then calls Super::BeginDestroy(). */
	virtual void BeginDestroy() override;

private:
	/** Sets LastError to "UPipelineTokenizer::<FunctionName>: <Message>" and logs it once as a Warning. Caller holds Mutex. */
	void SetError(const TCHAR* FunctionName, const FString& Message);

	/** Serializes every public call on this object (the C handle is not thread-safe). */
	mutable FCriticalSection Mutex;

	/** MptTokenizerHandle (mpt_tokenizers_c.h is included only in the .cpp). Owned; freed with mpt_tokenizers_free. */
	void* Tokenizer = nullptr;

	FString LastError;
};
