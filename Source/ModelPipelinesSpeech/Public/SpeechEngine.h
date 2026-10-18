// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Containers/ArrayView.h"
#include "Templates/SharedPointer.h"
#include "PipelineTypes.h"
#include "SpeechTypes.h"

class ISpeechEngineStream;

/**
 * A complete speech-to-text engine behind one IPipelineModel (docs/contracts.md C81a). Implemented by a backend's model and
 * returned by IPipelineModel::AsSpeechEngine() (Core forward-declares this class in the global namespace). USpeechToTextPipeline
 * routes packages of format Phonon to it instead of the NNE encoder/decoder graphs.
 */
class MODELPIPELINESSPEECH_API ISpeechEngine
{
public:
	virtual ~ISpeechEngine() = default;

	/** Executor worker only. Mono float at any rate 1000..384000; the engine resamples and windows (C77 semantics). Out gets
	    Text, Words (seconds from the input start), InferenceMs; AudioSeconds is filled by the pipeline. */
	virtual FPipelineResult Transcribe(TConstArrayView<float> Mono, int32 SampleRate, FSpeechTranscript& Out) = 0;

	/** Executor worker only. Null + failure if the engine has no streaming. */
	virtual TSharedPtr<ISpeechEngineStream> OpenStream(int32 SampleRate, FPipelineResult& OutResult) = 0;

	/** Any thread. True if streams produce partials (Phonon: true). */
	virtual bool SupportsPartials() const = 0;
};

/** One streaming transcription. One thread at a time; all calls on the executor worker. */
class MODELPIPELINESSPEECH_API ISpeechEngineStream
{
public:
	virtual ~ISpeechEngineStream() = default;

	/** Appends mono audio at the stream's rate. */
	virtual FPipelineResult Feed(TConstArrayView<float> Mono) = 0;

	/** Provisional text of the pending audio. */
	virtual FPipelineResult Partial(FSpeechTranscript& Out) = 0;

	/** End of utterance: finals of everything pending. */
	virtual FPipelineResult Flush(TArray<FSpeechTranscript>& OutFinals) = 0;

	/** Windows the engine cut by itself since the last call. */
	virtual FPipelineResult TakeFinals(TArray<FSpeechTranscript>& OutFinals) = 0;
};
