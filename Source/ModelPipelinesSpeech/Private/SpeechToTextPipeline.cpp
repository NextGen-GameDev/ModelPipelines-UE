// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechToTextPipeline.h"

#include "Async/Async.h"
#include "CoreGlobals.h"
#include "Math/UnrealMathUtility.h"
#include "Tasks/Task.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "PipelineBackend.h"
#include "PipelineExecutor.h"
#include "PipelineModelPackage.h"
#include "PipelineSpeechDsp.h"
#include "PipelineSpeechLog.h"
#include "PipelineTensor.h"
#include "PipelineTokenizer.h"
#include "SpeechDecodeSession.h"
#include "SpeechSoundWaveDecode.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(SpeechToTextPipeline)

namespace UE::ModelPipelines::Speech
{
	/** Immutable after OnInitialize; shared with in-flight calls. */
	struct FSpeechPipelineState
	{
		TStrongObjectPtr<UPipelineTokenizer> Tokenizer;
		TSharedPtr<const FLogMelExtractor, ESPMode::ThreadSafe> Extractor;
	};

	using FSpeechStatePtr = TSharedPtr<FSpeechPipelineState, ESPMode::ThreadSafe>;

	/** Everything one call needs; the worker threads never touch the pipeline UObject. */
	struct FSpeechCall
	{
		TArray<float> Samples;
		int32 SampleRate = 0;
		int32 NumChannels = 0;
		float AudioSeconds = 0.f;
		FPipelineSpeechSettings Speech;
		TSharedPtr<IPipelineModel> Decoder;
		FSpeechStatePtr State;
		FOnSpeechTranscribed OnDone;

		/** Front end output. bGated: answered without inference (silence or too short). */
		bool bGated = false;
		FSpeechFeatures Features;

		/** Encoder output, row-major [NumFrames][EncoderDim]. */
		TArray<float> Encoder;
		int32 NumFrames = 0;
		int32 EncoderDim = 0;
		FPipelineResult EncoderResult;

		FSpeechCall() = default;
		FSpeechCall(const FSpeechCall&) = delete;
		FSpeechCall& operator=(const FSpeechCall&) = delete;

		/** Delivers Cancelled if the call is dropped with OnDone still bound (executor shutdown); none at engine exit. */
		~FSpeechCall();
	};

	using FSpeechCallRef = TSharedRef<FSpeechCall, ESPMode::ThreadSafe>;

	namespace PipelinePrivate
	{
		constexpr double MaxSecondsPerCall = 35.0;
		constexpr TCHAR WordStartMarker = TCHAR(0x2581);

		/** C75 step 1 checks, in order, except the non-finite scan (O(N); runs first in RunFrontEnd). Ok() when the call may run. */
		static FPipelineResult ValidateCall(bool bReady, TConstArrayView<float> Samples, int32 SampleRate, int32 NumChannels)
		{
			if (!bReady)
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument,
					TEXT("pipeline is not ready (InitializeFromPackage has not succeeded)"));
			}
			if (NumChannels < 1)
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument,
					FString::Printf(TEXT("NumChannels is %d; at least 1 channel is required"), NumChannels));
			}
			if (SampleRate < 1000 || SampleRate > 384000)
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument,
					FString::Printf(TEXT("sample rate %d Hz is outside 1000..384000"), SampleRate));
			}
			if (Samples.Num() % NumChannels != 0)
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument,
					FString::Printf(TEXT("sample count %d is not a multiple of the channel count %d"), Samples.Num(), NumChannels));
			}
			const double Seconds = (double)(Samples.Num() / NumChannels) / (double)SampleRate;
			if (Seconds > MaxSecondsPerCall)
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument,
					FString::Printf(TEXT("audio is %.2f s; at most 35 s per call until long-audio support"), Seconds));
			}
			return FPipelineResult::Ok();
		}

		/** Fills the call from the pipeline (game thread or a blocking caller that keeps the pipeline alive). */
		static FPipelineResult PrepareCall(FSpeechCall& Call, const FSpeechStatePtr& State, const FPipelineSpeechSettings& Speech,
			TSharedPtr<IPipelineModel> Decoder)
		{
			Call.Speech = Speech;
			Call.Decoder = MoveTemp(Decoder);
			Call.State = State;
			Call.AudioSeconds = Call.NumChannels > 0 && Call.SampleRate > 0
				? (float)((double)(Call.Samples.Num() / Call.NumChannels) / (double)Call.SampleRate) : 0.f;
			if (!State.IsValid() || !State->Tokenizer.IsValid() || !State->Extractor.IsValid())
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("pipeline is not initialized (no tokenizer)"));
			}
			if (!Call.Decoder.IsValid())
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("the package has no DecoderModel"));
			}
			return FPipelineResult::Ok();
		}

		/** C75 step 2 (any thread): mono, resample, silence gate, log-mel. Sets bGated or Features. Frees Samples. */
		static FPipelineResult RunFrontEnd(FSpeechCall& Call)
		{
			// C75 step 1, last check: done here so the game thread never scans the samples. Before any inference.
			int32 NonFinite = 0;
			for (const float S : Call.Samples)
			{
				if (!FMath::IsFinite(S))
				{
					++NonFinite;
				}
			}
			if (NonFinite > 0)
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument,
					FString::Printf(TEXT("audio contains %d non-finite samples"), NonFinite));
			}

			FString Err;
			TArray<float> Mono;
			if (!MixToMono(Call.Samples, Call.NumChannels, Mono, Err))
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument, Err);
			}
			Call.Samples.Empty();
			TArray<float> Resampled;
			if (!Resample(Mono, Call.SampleRate, Call.Speech.SampleRate, Resampled, Err))
			{
				return FPipelineResult::Fail(EPipelineError::PreprocessFailed, FString::Printf(TEXT("resampling failed: %s"), *Err));
			}
			Mono.Empty();

			const bool bGateOn = Call.Speech.SilenceThresholdDb > -100.f;
			if (bGateOn && ComputeRmsDbfs(Resampled) < Call.Speech.SilenceThresholdDb)
			{
				Call.bGated = true;
				return FPipelineResult::Ok();
			}
			if (NumValidFrames(Resampled.Num(), Call.Speech) < 2)
			{
				Call.bGated = true;
				return FPipelineResult::Ok();
			}
			if (!Call.State->Extractor->Extract(Resampled, Call.Features, Err))
			{
				return FPipelineResult::Fail(EPipelineError::PreprocessFailed, FString::Printf(TEXT("feature extraction failed: %s"), *Err));
			}
			return FPipelineResult::Ok();
		}

		/** C75 step 3: the encoder stages (Complete is set by the caller for async calls). */
		static FPipelineStages MakeEncoderStages(const FSpeechCallRef& Call)
		{
			FPipelineStages Stages;
			Stages.Preprocess = [Call](const FPipelineModelInfo& Info, TArray<FPipelineTensor>& OutInputs) -> FPipelineResult
			{
				const int32 Index = Info.Bindings.InputFeatures;
				if (!Info.InputDescs.IsValidIndex(Index))
				{
					return FPipelineResult::Fail(EPipelineError::ShapeMismatch, TEXT("encoder has no input_features input"));
				}
				if (Info.InputDescs.Num() != 1)
				{
					return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
						FString::Printf(TEXT("encoder has %d inputs; only input_features is supported"), Info.InputDescs.Num()));
				}
				const FSpeechFeatures& F = Call->Features;
				OutInputs.SetNum(Info.InputDescs.Num());
				OutInputs[Index] = FPipelineTensor::FromFloat({ 1, (int64)F.NumFrames, (int64)F.NumMelBins }, F.Data);
				Call->Features = FSpeechFeatures();
				if (!OutInputs[Index].IsValid())
				{
					return FPipelineResult::Fail(EPipelineError::Internal, TEXT("could not build the input_features tensor"));
				}
				return FPipelineResult::Ok();
			};
			Stages.Postprocess = [Call](const FPipelineModelInfo& Info, TArray<FPipelineTensor>&& Outputs) -> FPipelineResult
			{
				if (!Outputs.IsValidIndex(Info.Bindings.MainOutput))
				{
					return FPipelineResult::Fail(EPipelineError::InferenceFailed,
						FString::Printf(TEXT("main output index %d not among %d encoder outputs"), Info.Bindings.MainOutput, Outputs.Num()));
				}
				const FPipelineTensor& Out = Outputs[Info.Bindings.MainOutput];
				const TConstArrayView<int64> Shape = Out.GetShape();
				const bool bRankOk = Shape.Num() == 2 || (Shape.Num() == 3 && Shape[0] == 1);
				const int64 Frames = bRankOk ? Shape[Shape.Num() - 2] : 0;
				const int64 Dim = bRankOk ? Shape.Last() : 0;
				if (!Out.IsValid() || Out.GetType() != EPipelineTensorType::Float32 || !bRankOk || Frames < 1 || Dim < 1
					|| Out.Num() > MAX_int32)
				{
					return FPipelineResult::Fail(EPipelineError::InferenceFailed,
						FString::Printf(TEXT("encoder output is %s %s; expected float32 [F,D]"), LexToString(Out.GetType()), *Out.DescribeShape()));
				}
				const TConstArrayView64<float> Data = Out.GetFloatData();
				for (const float V : Data)
				{
					if (!FMath::IsFinite(V))
					{
						return FPipelineResult::Fail(EPipelineError::InferenceFailed, TEXT("encoder output contains NaN or Inf"));
					}
				}
				Call->Encoder = TArray<float>(Data.GetData(), (int32)Data.Num());
				Call->NumFrames = (int32)Frames;
				Call->EncoderDim = (int32)Dim;
				return FPipelineResult::Ok();
			};
			return Stages;
		}

		static TSharedRef<FSpeechDecodeSession, ESPMode::ThreadSafe> MakeSession(FSpeechCall& Call)
		{
			return MakeShared<FSpeechDecodeSession, ESPMode::ThreadSafe>(Call.Decoder, Call.Speech, MoveTemp(Call.Encoder),
				Call.NumFrames, Call.EncoderDim);
		}

		/** C75 step 5: ids, text and word times. Any thread (the tokenizer is thread-safe per object). */
		static FPipelineResult Finalize(FSpeechCall& Call, const FSpeechDecodeSession& Session, FSpeechTranscript& Out)
		{
			Out = FSpeechTranscript();
			UPipelineTokenizer* Tok = Call.State.IsValid() ? Call.State->Tokenizer.Get() : nullptr;
			if (!Tok)
			{
				return FPipelineResult::Fail(EPipelineError::Internal, TEXT("tokenizer is gone"));
			}
			const FPipelineSpeechSettings& S = Call.Speech;
			const double FrameSec = (double)S.SubsamplingFactor * (double)S.HopLength / (double)FMath::Max(1, S.SampleRate);

			const TArray<FSpeechDecodeStep>& Steps = Session.GetSteps();
			TArray<FSpeechDecodeStep> Emitted;
			TArray<int32> Ids;
			for (const FSpeechDecodeStep& St : Steps)
			{
				if (St.Token != S.BlankTokenId)
				{
					Emitted.Add(St);
					Ids.Add(St.Token);
				}
			}

			FString Text;
			if (!Tok->Decode(Ids, true, Text))
			{
				return FPipelineResult::Fail(EPipelineError::PreprocessFailed,
					FString::Printf(TEXT("token decoding failed: %s"), *Tok->GetLastError()));
			}
			Out.Text = Text.TrimStartAndEnd();

			int32 WordBegin = 0;
			for (int32 i = 0; i <= Emitted.Num(); ++i)
			{
				bool bStartsWord = (i == Emitted.Num());
				if (!bStartsWord && i > 0)
				{
					FString Piece;
					if (!Tok->IdToToken(Emitted[i].Token, Piece))
					{
						return FPipelineResult::Fail(EPipelineError::PreprocessFailed,
							FString::Printf(TEXT("IdToToken(%d) failed: %s"), Emitted[i].Token, *Tok->GetLastError()));
					}
					bStartsWord = Piece.Len() > 0 && Piece[0] == WordStartMarker;
				}
				if (!bStartsWord || i == 0)
				{
					continue;
				}
				// Word = tokens [WordBegin, i).
				FString WordText;
				if (!Tok->Decode(TConstArrayView<int32>(Ids.GetData() + WordBegin, i - WordBegin), true, WordText))
				{
					return FPipelineResult::Fail(EPipelineError::PreprocessFailed,
						FString::Printf(TEXT("token decoding failed: %s"), *Tok->GetLastError()));
				}
				WordText.TrimStartAndEndInline();
				if (!WordText.IsEmpty())
				{
					FSpeechWord& W = Out.Words.AddDefaulted_GetRef();
					W.Word = MoveTemp(WordText);
					W.StartSeconds = (float)(Emitted[WordBegin].Frame * FrameSec);
					W.EndSeconds = (float)((Emitted[i - 1].Frame + Emitted[i - 1].Duration) * FrameSec);
				}
				WordBegin = i;
			}

			FPipelineResult R = Call.EncoderResult;
			R.bSuccess = true;
			R.Error = EPipelineError::None;
			const double DecoderMs = Session.GetDecoderMs();
			R.InferenceMs = (float)((double)Call.EncoderResult.InferenceMs + DecoderMs);
			Out.InferenceMs = R.InferenceMs;
			Out.AudioSeconds = Call.AudioSeconds;
			UE_LOG(LogModelPipelinesSpeech, Log, TEXT("USpeechToTextPipeline: %.2f s audio, encoder %.1f ms (%d frames), decoder %d steps %.1f us/step"),
				Call.AudioSeconds, Call.EncoderResult.InferenceMs, Call.NumFrames, Steps.Num(),
				Steps.Num() > 0 ? DecoderMs * 1000.0 / Steps.Num() : 0.0);
			return R;
		}

		/** Game thread: calls OnDone once. */
		static void Deliver(FSpeechCall& Call, const FPipelineResult& Result, const FSpeechTranscript& Transcript)
		{
			FOnSpeechTranscribed Callback = MoveTemp(Call.OnDone);
			Call.OnDone = nullptr;
			if (Callback)
			{
				Callback(Result, Result.bSuccess ? Transcript : FSpeechTranscript());
			}
		}

		static void DeliverFailure(FSpeechCall& Call, const FPipelineResult& Result)
		{
			Deliver(Call, Result, FSpeechTranscript());
		}

		/** Success without inference (empty input, silence, too short). */
		static FSpeechTranscript EmptyTranscript(const FSpeechCall& Call)
		{
			FSpeechTranscript T;
			T.AudioSeconds = Call.AudioSeconds;
			return T;
		}

		/** Delivers on a later game-thread tick. */
		static void DeliverLater(FSpeechCallRef Call, const FPipelineResult& Result, const FSpeechTranscript& Transcript)
		{
			AsyncTask(ENamedThreads::GameThread, [Call = MoveTemp(Call), Result, Transcript]() mutable
			{
				Deliver(*Call, Result, Transcript);
			});
		}

		/** Game thread: encoder Complete -> decode session on the executor -> Finalize -> OnDone. */
		static void OnEncoderComplete(const FSpeechCallRef& Call, const FPipelineResult& Result)
		{
			if (!Result.bSuccess)
			{
				DeliverFailure(*Call, Result);
				return;
			}
			Call->EncoderResult = Result;
			const TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor = FPipelineExecutor::TryGetShared();
			if (!Executor.IsValid())
			{
				DeliverFailure(*Call, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor has shut down")));
				return;
			}
			const TSharedRef<FSpeechDecodeSession, ESPMode::ThreadSafe> Session = MakeSession(*Call);
			FPipelineSessionCallbacks Callbacks;
			Callbacks.OnComplete = [Call, Session](const FPipelineResult& DecodeResult)
			{
				if (!DecodeResult.bSuccess)
				{
					DeliverFailure(*Call, DecodeResult);
					return;
				}
				FSpeechTranscript Transcript;
				const FPipelineResult Final = Finalize(*Call, *Session, Transcript);
				Deliver(*Call, Final, Transcript);
			};
			// No Owner: the call must complete even when the pipeline is destroyed meanwhile (C75).
			const FPipelineJobId Id = Executor->EnqueueSession(Session, MoveTemp(Callbacks), nullptr);
			if (Id == 0)
			{
				DeliverFailure(*Call, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor has shut down")));
			}
		}
	}

	FSpeechCall::~FSpeechCall()
	{
		if (!OnDone)
		{
			return; // normal path: Deliver already took the callback (or a blocking call never had one)
		}
		if (IsEngineExitRequested())
		{
			// The game thread may never run another task: no callback (as Core's FCallState). OnDone dies here.
			return;
		}
		// Dropped before delivery: the executor shut down with the decode session (or a stage job) queued, or a posted
		// task was discarded. Hand the callback to a fresh holder and deliver Cancelled on a later game-thread tick.
		// If that task is dropped too, the holder's destructor lands back here (at engine exit, the branch above).
		const FSpeechCallRef Orphan = MakeShared<FSpeechCall, ESPMode::ThreadSafe>();
		Orphan->OnDone = MoveTemp(OnDone);
		OnDone = nullptr;
		Orphan->AudioSeconds = AudioSeconds;
		PipelinePrivate::DeliverLater(Orphan,
			FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor shut down before this call completed")),
			FSpeechTranscript());
	}
}

USpeechToTextPipeline::USpeechToTextPipeline()
{
	PipelineTask = EPipelineTask::SpeechToText;
}

FPipelineResult USpeechToTextPipeline::OnInitialize(UPipelineModelPackage* InPackage)
{
	using namespace UE::ModelPipelines::Speech;

	if (!InPackage)
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("package is null"));
	}
	if (InPackage->TokenizerJson.IsEmpty())
	{
		const FPipelineResult R = FPipelineResult::Fail(EPipelineError::PreprocessFailed,
			FString::Printf(TEXT("package '%s' has no TokenizerJson"), *InPackage->GetName()));
		UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechToTextPipeline: %s"), *R.Message);
		return R;
	}
	const TSharedRef<FLogMelExtractor, ESPMode::ThreadSafe> Extractor = MakeShared<FLogMelExtractor, ESPMode::ThreadSafe>(GetSpeechSettings());
	if (!Extractor->IsValid())
	{
		FString Err;
		FSpeechFeatures Dummy;
		Extractor->Extract(TConstArrayView<float>(), Dummy, Err);
		const FPipelineResult R = FPipelineResult::Fail(EPipelineError::InvalidArgument,
			FString::Printf(TEXT("package '%s' has invalid speech settings: %s"), *InPackage->GetName(), *Err));
		UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechToTextPipeline: %s"), *R.Message);
		return R;
	}

	UPipelineTokenizer* NewTokenizer = NewObject<UPipelineTokenizer>(GetTransientPackage());
	if (!NewTokenizer || !NewTokenizer->InitializeTokenizerFromJson(InPackage->TokenizerJson))
	{
		const FString Err = NewTokenizer ? NewTokenizer->GetLastError() : FString(TEXT("could not create the tokenizer object"));
		if (NewTokenizer)
		{
			NewTokenizer->MarkAsGarbage();
		}
		const FPipelineResult R = FPipelineResult::Fail(EPipelineError::PreprocessFailed,
			FString::Printf(TEXT("tokenizer of package '%s' did not load: %s"), *InPackage->GetName(), *Err));
		UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechToTextPipeline: %s"), *R.Message);
		return R;
	}

	const FSpeechStatePtr State = MakeShared<FSpeechPipelineState, ESPMode::ThreadSafe>();
	State->Tokenizer.Reset(NewTokenizer);
	State->Extractor = Extractor;
	Tokenizer = NewTokenizer;
	SpeechState = State;
	return FPipelineResult::Ok();
}

void USpeechToTextPipeline::Transcribe(TArray<float> Samples, int32 SampleRate, int32 NumChannels, FOnSpeechTranscribed OnDone)
{
	using namespace UE::ModelPipelines::Speech;
	using namespace UE::ModelPipelines::Speech::PipelinePrivate;

	if (!OnDone)
	{
		UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechToTextPipeline::Transcribe: null callback, nothing runs"));
		return;
	}

	const FSpeechCallRef Call = MakeShared<FSpeechCall, ESPMode::ThreadSafe>();
	Call->Samples = MoveTemp(Samples);
	Call->SampleRate = SampleRate;
	Call->NumChannels = NumChannels;
	Call->OnDone = MoveTemp(OnDone);

	const bool bGameThread = IsInGameThread();
	FPipelineResult Check = bGameThread ? ValidateCall(IsReady(), Call->Samples, SampleRate, NumChannels)
		: FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("USpeechToTextPipeline::Transcribe must be called on the game thread"));
	if (Check.bSuccess)
	{
		Check = PrepareCall(*Call, SpeechState, GetSpeechSettings(), GetDecoderModel());
	}
	if (!Check.bSuccess)
	{
		DeliverLater(Call, Check, FSpeechTranscript());
		return;
	}
	if (Call->Samples.Num() == 0)
	{
		DeliverLater(Call, FPipelineResult::Ok(), EmptyTranscript(*Call));
		return;
	}

	TWeakObjectPtr<USpeechToTextPipeline> WeakThis(this);
	UE::Tasks::Launch(TEXT("ModelPipelinesSpeechFrontEnd"), [Call = FSpeechCallRef(Call), WeakThis]() mutable
	{
		const FPipelineResult FrontEnd = RunFrontEnd(*Call);
		AsyncTask(ENamedThreads::GameThread, [Call = MoveTemp(Call), WeakThis, FrontEnd]()
		{
			if (!FrontEnd.bSuccess)
			{
				DeliverFailure(*Call, FrontEnd);
				return;
			}
			if (Call->bGated)
			{
				Deliver(*Call, FPipelineResult::Ok(), EmptyTranscript(*Call));
				return;
			}
			USpeechToTextPipeline* Pipeline = WeakThis.Get();
			if (!Pipeline)
			{
				DeliverFailure(*Call, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the pipeline was destroyed before inference")));
				return;
			}
			Pipeline->StartEncoder(Call);
		});
	});
}

void USpeechToTextPipeline::TranscribeSoundWave(USoundWave* Sound, FOnSpeechTranscribed OnDone)
{
	using namespace UE::ModelPipelines::Speech;
	using namespace UE::ModelPipelines::Speech::PipelinePrivate;

	if (!OnDone)
	{
		UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechToTextPipeline::TranscribeSoundWave: null callback, nothing runs"));
		return;
	}

	// Holds OnDone until the decode ends; its destructor delivers Cancelled if a task is dropped at executor shutdown.
	const FSpeechCallRef Holder = MakeShared<FSpeechCall, ESPMode::ThreadSafe>();
	Holder->OnDone = MoveTemp(OnDone);

	// C76 rule 1, in order. ValidateCall with an empty mono 16 kHz clip reports only the readiness check.
	FPipelineResult Check = FPipelineResult::Ok();
	if (!IsInGameThread())
	{
		Check = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("USpeechToTextPipeline::TranscribeSoundWave must be called on the game thread"));
	}
	else if (!IsReady())
	{
		Check = ValidateCall(false, TConstArrayView<float>(), 16000, 1);
	}
	else if (!Sound)
	{
		Check = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("Sound is null"));
	}
	else if (Sound->IsProcedurallyGenerated())
	{
		Check = FPipelineResult::Fail(EPipelineError::InvalidArgument,
			FString::Printf(TEXT("'%s': procedural sound waves have no stored audio"), *Sound->GetName()));
	}
	else if ((double)Sound->Duration > MaxSecondsPerCall)
	{
		Check = FPipelineResult::Fail(EPipelineError::InvalidArgument,
			FString::Printf(TEXT("sound wave '%s' is %.2f s; at most 35 s per call until long-audio support"), *Sound->GetName(), Sound->Duration));
	}
	if (!Check.bSuccess)
	{
		DeliverLater(Holder, Check, FSpeechTranscript());
		return;
	}

	// C76 rule 2: cheap game-thread capture (RawPCMData copy or the thread-safe snapshot); the wave stays alive until the decode ends.
	TSharedRef<FSoundWaveSource, ESPMode::ThreadSafe> Source = MakeShared<FSoundWaveSource, ESPMode::ThreadSafe>();
	CaptureSoundWaveSource(*Sound, *Source);
	TStrongObjectPtr<USoundWave> KeepAlive(Sound);

	TWeakObjectPtr<USpeechToTextPipeline> WeakThis(this);
	UE::Tasks::Launch(TEXT("ModelPipelinesSpeechSoundWaveDecode"),
		[Holder = FSpeechCallRef(Holder), Source = MoveTemp(Source), KeepAlive = MoveTemp(KeepAlive), WeakThis]() mutable
	{
		int32 Rate = 0;
		int32 Channels = 0;
		const FPipelineResult Decoded = DecodeSoundWaveSource(*Source, Holder->Samples, Rate, Channels);
		AsyncTask(ENamedThreads::GameThread, [Holder = MoveTemp(Holder), KeepAlive = MoveTemp(KeepAlive), WeakThis, Decoded, Rate, Channels]() mutable
		{
			KeepAlive.Reset(); // released on the game thread
			if (!Decoded.bSuccess)
			{
				DeliverFailure(*Holder, Decoded);
				return;
			}
			USpeechToTextPipeline* Pipeline = WeakThis.Get();
			if (!Pipeline)
			{
				DeliverFailure(*Holder, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the pipeline was destroyed before inference")));
				return;
			}
			// C76 rule 4: exactly the Transcribe path (C75) on the decoded samples.
			FOnSpeechTranscribed Callback = MoveTemp(Holder->OnDone);
			Holder->OnDone = nullptr;
			Pipeline->Transcribe(MoveTemp(Holder->Samples), Rate, Channels, MoveTemp(Callback));
		});
	}, UE::Tasks::ETaskPriority::BackgroundNormal);  // the streamed-chunk poll sleeps; keep it off foreground workers
}

void USpeechToTextPipeline::StartEncoder(const TSharedRef<UE::ModelPipelines::Speech::FSpeechCall, ESPMode::ThreadSafe>& Call)
{
	using namespace UE::ModelPipelines::Speech::PipelinePrivate;

	FPipelineStages Stages = MakeEncoderStages(Call);
	Stages.Complete = [Call](const FPipelineResult& Result) { OnEncoderComplete(Call, Result); };
	RunStages(MoveTemp(Stages));
}

FPipelineResult USpeechToTextPipeline::TranscribeBlocking(TConstArrayView<float> Samples, int32 SampleRate, int32 NumChannels, FSpeechTranscript& Out)
{
	using namespace UE::ModelPipelines::Speech;
	using namespace UE::ModelPipelines::Speech::PipelinePrivate;

	Out = FSpeechTranscript();
	FPipelineResult Check = ValidateCall(IsReady(), Samples, SampleRate, NumChannels);
	if (!Check.bSuccess)
	{
		return Check;
	}
	const FSpeechCallRef Call = MakeShared<FSpeechCall, ESPMode::ThreadSafe>();
	Call->Samples = TArray<float>(Samples.GetData(), Samples.Num());
	Call->SampleRate = SampleRate;
	Call->NumChannels = NumChannels;
	Check = PrepareCall(*Call, SpeechState, GetSpeechSettings(), GetDecoderModel());
	if (!Check.bSuccess)
	{
		return Check;
	}
	if (Call->Samples.Num() == 0)
	{
		Out = EmptyTranscript(*Call);
		return FPipelineResult::Ok();
	}

	const FPipelineResult FrontEnd = RunFrontEnd(*Call);
	if (!FrontEnd.bSuccess)
	{
		return FrontEnd;
	}
	if (Call->bGated)
	{
		Out = EmptyTranscript(*Call);
		return FPipelineResult::Ok();
	}

	const FPipelineResult Enc = RunStagesBlocking(MakeEncoderStages(Call));
	if (!Enc.bSuccess)
	{
		return Enc;
	}
	Call->EncoderResult = Enc;

	const TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor = FPipelineExecutor::TryGetShared();
	if (!Executor.IsValid())
	{
		return FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor has shut down"));
	}
	const TSharedRef<FSpeechDecodeSession, ESPMode::ThreadSafe> Session = MakeSession(*Call);
	TArray<FPipelineSessionProgress> Progress;
	const FPipelineResult Dec = Executor->RunSessionBlocking(Session, Progress);
	if (!Dec.bSuccess)
	{
		return Dec;
	}
	FSpeechTranscript Transcript;
	const FPipelineResult Final = Finalize(*Call, *Session, Transcript);
	if (Final.bSuccess)
	{
		Out = MoveTemp(Transcript);
	}
	return Final;
}
