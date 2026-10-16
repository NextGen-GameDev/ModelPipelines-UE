// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Templates/Function.h"
#include "UObject/ObjectMacros.h"
#include "PipelineTypes.h"
#include "SpeechTypes.generated.h"

/** One recognised word with its time span in the input audio (docs/contracts.md C75). */
USTRUCT(BlueprintType)
struct MODELPIPELINESSPEECH_API FSpeechWord
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	FString Word;

	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	float StartSeconds = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	float EndSeconds = 0.f;
};

/** Result of one transcription (C75). Empty on failure. */
USTRUCT(BlueprintType)
struct MODELPIPELINESSPEECH_API FSpeechTranscript
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	FString Text;

	/** Words in order. Words whose text is empty after decoding are omitted. */
	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	TArray<FSpeechWord> Words;

	/** Input duration in seconds. */
	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	float AudioSeconds = 0.f;

	/** Encoder + decoder time; 0 when the silence gate answered without inference. */
	UPROPERTY(BlueprintReadOnly, Category = "Speech")
	float InferenceMs = 0.f;
};

/**
 * Game thread, exactly once per Transcribe call, never inside it. Transcript empty on failure.
 * If the executor shuts down while the call is in flight (including the decode leg), it fires with Cancelled.
 * Exception, as in Core: at engine exit (IsEngineExitRequested) a dropped call fires no callback.
 */
using FOnSpeechTranscribed = TUniqueFunction<void(const FPipelineResult& Result, const FSpeechTranscript& Transcript)>;
