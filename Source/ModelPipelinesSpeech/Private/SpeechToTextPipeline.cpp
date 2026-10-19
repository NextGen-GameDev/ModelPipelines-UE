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
#include "SpeechEngineSession.h"
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

		/** Async calls: starts the current window's encoder on the pipeline; false when the pipeline was destroyed. Holds a
		    weak pointer only (made inside USpeechToTextPipeline::Transcribe, which may call the private StartEncoder). */
		TFunction<bool(const TSharedRef<FSpeechCall, ESPMode::ThreadSafe>&)> StartEncoderFn;

		/** C77: the 16 kHz mono signal of the whole call and its windows; WindowIndex is the window being processed. */
		TArray<float> Mono;
		TArray<FSpeechWindow> Windows;
		int32 WindowIndex = 0;

		/** C77a stitching state (non-empty trimmed window texts, global words, summed inference time). */
		TArray<FString> WindowTexts;
		TArray<FSpeechWord> Words;
		double TotalInferenceMs = 0.0;
		bool bAnyWindowRan = false;
		FPipelineResult LastWindowResult;

		/** Front end output of the current window. bGated: answered without inference (silence or too short). */
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
		constexpr double MaxSecondsPerCall = 600.0;
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
					FString::Printf(TEXT("audio is %.2f s; at most 600 s per call"), Seconds));
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

		/** C75 step 2 head (any thread): the non-finite scan and the C73 mono mix. Frees Samples on success. */
		static FPipelineResult ScanAndMixToMono(FSpeechCall& Call, TArray<float>& Mono)
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
			if (!MixToMono(Call.Samples, Call.NumChannels, Mono, Err))
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument, Err);
			}
			Call.Samples.Empty();
			return FPipelineResult::Ok();
		}

		/** C75 step 2 (any thread): mono, resample, silence gate, log-mel. Sets bGated or Features. Frees Samples. */
		static FPipelineResult RunFrontEnd(FSpeechCall& Call)
		{
			TArray<float> Mono;
			const FPipelineResult Mixed = ScanAndMixToMono(Call, Mono);
			if (!Mixed.bSuccess)
			{
				return Mixed;
			}
			FString Err;
			TArray<float> Resampled;
			if (!Resample(Mono, Call.SampleRate, Call.Speech.SampleRate, Resampled, Err))
			{
				return FPipelineResult::Fail(EPipelineError::PreprocessFailed, FString::Printf(TEXT("resampling failed: %s"), *Err));
			}
			Mono.Empty();

			// C77: plan the windows on the 16 kHz mono signal (one window [0, N) up to 35 s; none for an empty signal).
			Call.Mono = MoveTemp(Resampled);
			PlanSpeechWindows(Call.Mono, Call.Speech.SampleRate, Call.Windows);
			Call.WindowIndex = 0;
			return FPipelineResult::Ok();
		}

		/** C75 step 2 tail for the current window (any thread): its own silence gate and log-mel. Sets bGated or Features. */
		static FPipelineResult PrepareWindow(FSpeechCall& Call)
		{
			Call.bGated = false;
			Call.Features = FSpeechFeatures();
			Call.Encoder.Empty();
			Call.NumFrames = 0;
			Call.EncoderDim = 0;
			if (!Call.Windows.IsValidIndex(Call.WindowIndex))
			{
				return FPipelineResult::Fail(EPipelineError::Internal, TEXT("window index out of range"));
			}
			const FSpeechWindow& W = Call.Windows[Call.WindowIndex];
			const TConstArrayView<float> Samples(Call.Mono.GetData() + W.Start, W.End - W.Start);

			const bool bGateOn = Call.Speech.SilenceThresholdDb > -100.f;
			if (bGateOn && ComputeRmsDbfs(Samples) < Call.Speech.SilenceThresholdDb)
			{
				Call.bGated = true;
				return FPipelineResult::Ok();
			}
			if (NumValidFrames(Samples.Num(), Call.Speech) < 2)
			{
				Call.bGated = true;
				return FPipelineResult::Ok();
			}
			FString Err;
			if (!Call.State->Extractor->Extract(Samples, Call.Features, Err))
			{
				return FPipelineResult::Fail(EPipelineError::PreprocessFailed, FString::Printf(TEXT("feature extraction failed: %s"), *Err));
			}
			return FPipelineResult::Ok();
		}

		/** C77a: "window k/n: " prefix on a failing window of a multi-window call (single-window calls are unchanged). */
		static FPipelineResult WindowFailure(const FSpeechCall& Call, FPipelineResult R)
		{
			if (Call.Windows.Num() > 1)
			{
				R.Message = FString::Printf(TEXT("window %d/%d: %s"), Call.WindowIndex + 1, Call.Windows.Num(), *R.Message);
			}
			return R;
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
			// C77a: global times = window-local + window start / R (0 for a single window: unchanged).
			const double WindowOffsetSec = Call.Windows.IsValidIndex(Call.WindowIndex)
				? (double)Call.Windows[Call.WindowIndex].Start / (double)FMath::Max(1, S.SampleRate) : 0.0;

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
					W.StartSeconds = (float)((double)Emitted[WordBegin].Frame * FrameSec + WindowOffsetSec);
					W.EndSeconds = (float)((double)(Emitted[i - 1].Frame + Emitted[i - 1].Duration) * FrameSec + WindowOffsetSec);
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

		/** C77a: adds one decoded window to the stitched result. */
		static void AccumulateWindow(FSpeechCall& Call, const FPipelineResult& WindowResult, FSpeechTranscript&& WindowTranscript)
		{
			Call.bAnyWindowRan = true;
			Call.LastWindowResult = WindowResult;
			Call.TotalInferenceMs += (double)WindowResult.InferenceMs;
			FString Text = WindowTranscript.Text.TrimStartAndEnd();
			if (!Text.IsEmpty())
			{
				Call.WindowTexts.Add(MoveTemp(Text));
			}
			Call.Words.Append(MoveTemp(WindowTranscript.Words));
		}

		/** C77a: the stitched result once every window is done. No window ran -> success, empty, InferenceMs 0. */
		static FPipelineResult BuildFinal(FSpeechCall& Call, FSpeechTranscript& Out)
		{
			Out = EmptyTranscript(Call);
			Call.Mono.Empty();
			if (!Call.bAnyWindowRan)
			{
				return FPipelineResult::Ok();
			}
			FPipelineResult R = Call.LastWindowResult;
			R.InferenceMs = (float)Call.TotalInferenceMs;
			Out.Text = FString::Join(Call.WindowTexts, TEXT(" "));
			Out.Words = MoveTemp(Call.Words);
			Out.InferenceMs = R.InferenceMs;
			return R;
		}

		static void ProcessNextWindow(const FSpeechCallRef& Call);

		/** Game thread, after the current window's front end ran: gated -> next window, else that window's encoder. */
		static void OnWindowFrontEndDone(const FSpeechCallRef& Call, const FPipelineResult& FrontEnd)
		{
			if (!FrontEnd.bSuccess)
			{
				DeliverFailure(*Call, WindowFailure(*Call, FrontEnd));
				return;
			}
			if (Call->bGated)
			{
				++Call->WindowIndex;
				ProcessNextWindow(Call);
				return;
			}
			if (!Call->StartEncoderFn || !Call->StartEncoderFn(Call))
			{
				DeliverFailure(*Call, WindowFailure(*Call,
					FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the pipeline was destroyed before inference"))));
			}
		}

		/** Game thread: front end of the next window on a worker, or the stitched result when every window is done. */
		static void ProcessNextWindow(const FSpeechCallRef& Call)
		{
			if (!Call->Windows.IsValidIndex(Call->WindowIndex))
			{
				FSpeechTranscript Transcript;
				const FPipelineResult Final = BuildFinal(*Call, Transcript);
				Deliver(*Call, Final, Transcript);
				return;
			}
			UE::Tasks::Launch(TEXT("ModelPipelinesSpeechWindow"), [Call = FSpeechCallRef(Call)]() mutable
			{
				const FPipelineResult FrontEnd = PrepareWindow(*Call);
				AsyncTask(ENamedThreads::GameThread, [Call = MoveTemp(Call), FrontEnd]()
				{
					OnWindowFrontEndDone(Call, FrontEnd);
				});
			});
		}

		/** Game thread: encoder Complete -> decode session on the executor -> Finalize -> next window (OnDone after the last). */
		static void OnEncoderComplete(const FSpeechCallRef& Call, const FPipelineResult& Result)
		{
			if (!Result.bSuccess)
			{
				DeliverFailure(*Call, WindowFailure(*Call, Result));
				return;
			}
			Call->EncoderResult = Result;
			const TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor = FPipelineExecutor::TryGetShared();
			if (!Executor.IsValid())
			{
				DeliverFailure(*Call, WindowFailure(*Call, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor has shut down"))));
				return;
			}
			const TSharedRef<FSpeechDecodeSession, ESPMode::ThreadSafe> Session = MakeSession(*Call);
			FPipelineSessionCallbacks Callbacks;
			Callbacks.OnComplete = [Call, Session](const FPipelineResult& DecodeResult)
			{
				if (!DecodeResult.bSuccess)
				{
					DeliverFailure(*Call, WindowFailure(*Call, DecodeResult));
					return;
				}
				FSpeechTranscript Transcript;
				const FPipelineResult WindowResult = Finalize(*Call, *Session, Transcript);
				if (!WindowResult.bSuccess)
				{
					DeliverFailure(*Call, WindowFailure(*Call, WindowResult));
					return;
				}
				AccumulateWindow(*Call, WindowResult, MoveTemp(Transcript));
				++Call->WindowIndex;
				ProcessNextWindow(Call);
			};
			// No Owner: the call must complete even when the pipeline is destroyed meanwhile (C75).
			const FPipelineJobId Id = Executor->EnqueueSession(Session, MoveTemp(Callbacks), nullptr);
			if (Id == 0)
			{
				DeliverFailure(*Call, WindowFailure(*Call, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor has shut down"))));
			}
		}
	}

	namespace PipelinePrivate
	{
		/** C81a: the call's settings and duration for the engine path (no tokenizer, no decoder). */
		static void PrepareEngineCall(FSpeechCall& Call, const FPipelineSpeechSettings& Speech)
		{
			Call.Speech = Speech;
			Call.AudioSeconds = Call.NumChannels > 0 && Call.SampleRate > 0
				? (float)((double)(Call.Samples.Num() / Call.NumChannels) / (double)Call.SampleRate) : 0.f;
		}

		/** C81a rule 2: silence gate on the whole mono signal (same gate switch as the NNE windows). True -> empty success. */
		static bool IsSilentForEngine(const FSpeechCall& Call, TConstArrayView<float> Mono)
		{
			if (Mono.Num() == 0)
			{
				return true;
			}
			const bool bGateOn = Call.Speech.SilenceThresholdDb > -100.f;
			return bGateOn && ComputeRmsDbfs(Mono) < Call.Speech.SilenceThresholdDb;
		}

		/** Game thread: one engine session on the executor (C61); OnDone after it completes. Call->Mono is moved into it. */
		static void StartEngineSession(const FSpeechCallRef& Call, const TSharedPtr<IPipelineModel>& Model)
		{
			const TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor = FPipelineExecutor::TryGetShared();
			if (!Executor.IsValid())
			{
				DeliverFailure(*Call, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor has shut down")));
				return;
			}
			const TSharedRef<FSpeechEngineSession, ESPMode::ThreadSafe> Session =
				MakeShared<FSpeechEngineSession, ESPMode::ThreadSafe>(Model, MoveTemp(Call->Mono), Call->SampleRate);
			FPipelineSessionCallbacks Callbacks;
			Callbacks.OnComplete = [Call, Session](const FPipelineResult& Result)
			{
				if (!Result.bSuccess)
				{
					DeliverFailure(*Call, Result);
					return;
				}
				FSpeechTranscript Transcript = Session->GetTranscript();
				Transcript.AudioSeconds = Call->AudioSeconds;
				FPipelineResult Final = Result;
				Final.InferenceMs = Transcript.InferenceMs;
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
	bUseSpeechEngine = false;
	if (InPackage->GetModelFormat() == EPipelineModelFormat::Phonon)
	{
		// C81a rule 1: no tokenizer and no decoder; the base class prepared the model through the Phonon backend.
		if (!GetModel().IsValid())
		{
			return FPipelineResult::Fail(EPipelineError::Internal,
				FString::Printf(TEXT("package '%s': the Phonon model was not prepared"), *InPackage->GetName()));
		}
		Tokenizer = nullptr;
		SpeechState.Reset();
		bUseSpeechEngine = true;
		return FPipelineResult::Ok();
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
		if (bUseSpeechEngine)
		{
			PrepareEngineCall(*Call, GetSpeechSettings());
		}
		else
		{
			Check = PrepareCall(*Call, SpeechState, GetSpeechSettings(), GetDecoderModel());
		}
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
	if (bUseSpeechEngine)
	{
		TranscribeWithEngine(Call);
		return;
	}

	TWeakObjectPtr<USpeechToTextPipeline> WeakThis(this);
	Call->StartEncoderFn = [WeakThis](const FSpeechCallRef& WindowCall) -> bool
	{
		USpeechToTextPipeline* Pipeline = WeakThis.Get();
		if (!Pipeline)
		{
			return false;
		}
		Pipeline->StartEncoder(WindowCall);
		return true;
	};
	// Whole-call front end (scan, mono, resample, window plan) and the first window's gate + features in one worker task.
	UE::Tasks::Launch(TEXT("ModelPipelinesSpeechFrontEnd"), [Call = FSpeechCallRef(Call)]() mutable
	{
		FPipelineResult FrontEnd = RunFrontEnd(*Call);
		const bool bHasWindow = FrontEnd.bSuccess && Call->Windows.Num() > 0;
		if (bHasWindow)
		{
			FrontEnd = PrepareWindow(*Call);
		}
		AsyncTask(ENamedThreads::GameThread, [Call = MoveTemp(Call), FrontEnd, bHasWindow]()
		{
			if (!FrontEnd.bSuccess && !bHasWindow)
			{
				DeliverFailure(*Call, FrontEnd);
				return;
			}
			if (!bHasWindow)
			{
				ProcessNextWindow(Call);   // no window (signal resampled to nothing): empty success
				return;
			}
			OnWindowFrontEndDone(Call, FrontEnd);
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
			FString::Printf(TEXT("sound wave '%s' is %.2f s; at most 600 s per call"), *Sound->GetName(), Sound->Duration));
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
	if (bUseSpeechEngine)
	{
		PrepareEngineCall(*Call, GetSpeechSettings());
		if (Call->Samples.Num() == 0)
		{
			Out = EmptyTranscript(*Call);
			return FPipelineResult::Ok();
		}
		return TranscribeWithEngineBlocking(Call, Out);
	}
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

	// C77: the windows one after another; each window's features / encoder output are freed before the next.
	for (Call->WindowIndex = 0; Call->WindowIndex < Call->Windows.Num(); ++Call->WindowIndex)
	{
		const FPipelineResult Prep = PrepareWindow(*Call);
		if (!Prep.bSuccess)
		{
			return WindowFailure(*Call, Prep);
		}
		if (Call->bGated)
		{
			continue;
		}
		const FPipelineResult Enc = RunStagesBlocking(MakeEncoderStages(Call));
		if (!Enc.bSuccess)
		{
			return WindowFailure(*Call, Enc);
		}
		Call->EncoderResult = Enc;

		const TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor = FPipelineExecutor::TryGetShared();
		if (!Executor.IsValid())
		{
			return WindowFailure(*Call, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor has shut down")));
		}
		const TSharedRef<FSpeechDecodeSession, ESPMode::ThreadSafe> Session = MakeSession(*Call);
		TArray<FPipelineSessionProgress> Progress;
		const FPipelineResult Dec = Executor->RunSessionBlocking(Session, Progress);
		if (!Dec.bSuccess)
		{
			return WindowFailure(*Call, Dec);
		}
		FSpeechTranscript WindowTranscript;
		const FPipelineResult WindowResult = Finalize(*Call, *Session, WindowTranscript);
		if (!WindowResult.bSuccess)
		{
			return WindowFailure(*Call, WindowResult);
		}
		AccumulateWindow(*Call, WindowResult, MoveTemp(WindowTranscript));
	}

	FSpeechTranscript Transcript;
	const FPipelineResult Final = BuildFinal(*Call, Transcript);
	if (Final.bSuccess)
	{
		Out = MoveTemp(Transcript);
	}
	return Final;
}

void USpeechToTextPipeline::TranscribeWithEngine(const TSharedRef<UE::ModelPipelines::Speech::FSpeechCall, ESPMode::ThreadSafe>& Call)
{
	using namespace UE::ModelPipelines::Speech;
	using namespace UE::ModelPipelines::Speech::PipelinePrivate;

	TSharedPtr<IPipelineModel> Model = GetModel();
	TWeakObjectPtr<USpeechToTextPipeline> WeakThis(this);
	// C81a rule 2: non-finite scan, mono mix and the whole-call silence gate on a worker; no resampling (the engine does it).
	UE::Tasks::Launch(TEXT("ModelPipelinesSpeechEngineFrontEnd"), [Call = FSpeechCallRef(Call), Model = MoveTemp(Model), WeakThis]() mutable
	{
		TArray<float> Mono;
		const FPipelineResult FrontEnd = ScanAndMixToMono(*Call, Mono);
		const bool bGated = FrontEnd.bSuccess && IsSilentForEngine(*Call, Mono);
		if (FrontEnd.bSuccess && !bGated)
		{
			Call->Mono = MoveTemp(Mono);
		}
		AsyncTask(ENamedThreads::GameThread, [Call = MoveTemp(Call), Model = MoveTemp(Model), WeakThis, FrontEnd, bGated]()
		{
			if (!FrontEnd.bSuccess)
			{
				DeliverFailure(*Call, FrontEnd);
				return;
			}
			if (bGated)
			{
				Deliver(*Call, FPipelineResult::Ok(), EmptyTranscript(*Call));
				return;
			}
			if (!WeakThis.IsValid())
			{
				DeliverFailure(*Call, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the pipeline was destroyed before inference")));
				return;
			}
			StartEngineSession(Call, Model);
		});
	});
}

FPipelineResult USpeechToTextPipeline::TranscribeWithEngineBlocking(const TSharedRef<UE::ModelPipelines::Speech::FSpeechCall, ESPMode::ThreadSafe>& Call,
	FSpeechTranscript& Out)
{
	using namespace UE::ModelPipelines::Speech;
	using namespace UE::ModelPipelines::Speech::PipelinePrivate;

	TArray<float> Mono;
	const FPipelineResult FrontEnd = ScanAndMixToMono(*Call, Mono);
	if (!FrontEnd.bSuccess)
	{
		return FrontEnd;
	}
	if (IsSilentForEngine(*Call, Mono))
	{
		Out = EmptyTranscript(*Call);
		return FPipelineResult::Ok();
	}
	const TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor = FPipelineExecutor::TryGetShared();
	if (!Executor.IsValid())
	{
		return FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor has shut down"));
	}
	const TSharedRef<FSpeechEngineSession, ESPMode::ThreadSafe> Session =
		MakeShared<FSpeechEngineSession, ESPMode::ThreadSafe>(GetModel(), MoveTemp(Mono), Call->SampleRate);
	TArray<FPipelineSessionProgress> Progress;
	FPipelineResult R = Executor->RunSessionBlocking(Session, Progress);
	if (!R.bSuccess)
	{
		return R;
	}
	Out = Session->GetTranscript();
	Out.AudioSeconds = Call->AudioSeconds;
	R.InferenceMs = Out.InferenceMs;
	return R;
}

TSharedPtr<IPipelineModel> USpeechToTextPipeline::GetSpeechEngineModel() const
{
	return bUseSpeechEngine ? GetModel() : TSharedPtr<IPipelineModel>();
}

USpeechStream* USpeechToTextPipeline::StartStream(const FSpeechStreamSettings& Settings, FPipelineResult& OutResult)
{
	if (!IsInGameThread())
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("StartStream must be called on the game thread"));
		return nullptr;
	}
	if (!IsReady() || !GetModel().IsValid())
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("the speech-to-text pipeline is not ready (create it from a package first)"));
		return nullptr;
	}
	const TSharedPtr<IPipelineModel> EngineModel = GetSpeechEngineModel();
	FSpeechStreamSettings Clamped = Settings;
	// Non-finite settings (Blueprint input) fall back to the defaults BEFORE clamping (FMath::Clamp passes NaN through).
	const FSpeechStreamSettings Defaults;
	if (!FMath::IsFinite(Clamped.EndOfSpeechHangoverSeconds)) { Clamped.EndOfSpeechHangoverSeconds = Defaults.EndOfSpeechHangoverSeconds; }
	if (!FMath::IsFinite(Clamped.PartialIntervalSeconds)) { Clamped.PartialIntervalSeconds = Defaults.PartialIntervalSeconds; }
	if (!FMath::IsFinite(Clamped.SpeechThresholdDb)) { Clamped.SpeechThresholdDb = Defaults.SpeechThresholdDb; }
	if (!FMath::IsFinite(Clamped.MaxUtteranceSeconds)) { Clamped.MaxUtteranceSeconds = Defaults.MaxUtteranceSeconds; }
	Clamped.EndOfSpeechHangoverSeconds = FMath::Clamp(Clamped.EndOfSpeechHangoverSeconds, 0.1f, 2.f);
	Clamped.PartialIntervalSeconds = FMath::Clamp(Clamped.PartialIntervalSeconds, 0.1f, 5.f);
	Clamped.SpeechThresholdDb = FMath::Clamp(Clamped.SpeechThresholdDb, 3.f, 40.f);
	Clamped.MaxUtteranceSeconds = FMath::Clamp(Clamped.MaxUtteranceSeconds, 2.f, 120.f);
	if (!FMath::IsFinite(Clamped.MergeWindowSeconds)) { Clamped.MergeWindowSeconds = Defaults.MergeWindowSeconds; }
	if (!FMath::IsFinite(Clamped.MinSpeechSeconds)) { Clamped.MinSpeechSeconds = Defaults.MinSpeechSeconds; }
	Clamped.MergeWindowSeconds = FMath::Clamp(Clamped.MergeWindowSeconds, 0.f, 3.f);
	Clamped.MinSpeechSeconds = FMath::Clamp(Clamped.MinSpeechSeconds, 0.f, 2.f);

	USpeechStream* Stream = NewObject<USpeechStream>(GetTransientPackage());
	Stream->Start(this, Clamped, EngineModel);
	OutResult = FPipelineResult::Ok();
	return Stream;
}
