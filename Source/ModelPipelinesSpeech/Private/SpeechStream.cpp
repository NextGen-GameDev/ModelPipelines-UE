// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechStream.h"

#include "Containers/Queue.h"
#include "Containers/Ticker.h"
#include "HAL/CriticalSection.h"
#include "HAL/PlatformTime.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/ScopeExit.h"
#include "Misc/ScopeLock.h"
#include "Tasks/Task.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "PipelineBackend.h"
#include "PipelineExecutor.h"
#include "PipelineSession.h"
#include "PipelineSpeechDsp.h"
#include "PipelineSpeechLog.h"
#include "SpeechEngine.h"
#include "SpeechToTextPipeline.h"
#include "SpeechVad.h"

#include <atomic>

#include UE_INLINE_GENERATED_CPP_BY_NAME(SpeechStream)

namespace UE::ModelPipelines::Speech
{
	namespace StreamPrivate
	{
		constexpr int32 StreamRate = 16000;
		constexpr int64 MaxQueuedSamples = 10 * StreamRate;   // rule 7: 10 s of queued audio
		constexpr int64 IdleHistorySamples = StreamRate;      // kept before the current position while not speaking

		/** One pushed chunk (producer side). */
		struct FStreamChunk
		{
			TArray<float> Samples;
			int32 SampleRate = 0;
			int32 NumChannels = 1;
			int64 Samples16k = 0;
		};

		/** One front-end output: 16 kHz mono audio and the VAD events it completed. */
		struct FFrontEndItem
		{
			TArray<float> Audio;
			TArray<FSpeechVadEvent> Events;
			int64 SpeechSamplesNow = 0;   // WP9R.7: speech frames (samples) of the VAD's current utterance after this audio
		};

		enum class EStreamOp : uint8
		{
			Open,
			Feed,
			Partial,
			Flush,
			Reopen, // WP9R.7: discards the engine stream (a join that turned out too long) and opens a fresh one
			Close   // releases the engine stream on the worker (Finish / destroy)
		};

		/** The engine stream, created and used only on the executor worker (one session at a time). */
		struct FEngineStreamHolder
		{
			TSharedPtr<ISpeechEngineStream> Stream;
		};

		/** One engine-stream call as a single-step executor session (rule 4: every engine call runs through FPipelineExecutor). */
		class FSpeechStreamOpSession final : public IPipelineSession
		{
		public:
			FSpeechStreamOpSession(TSharedPtr<IPipelineModel> InModel, TSharedRef<FEngineStreamHolder, ESPMode::ThreadSafe> InHolder,
				EStreamOp InOp, TArray<float>&& InAudio)
				: Model(MoveTemp(InModel)), Holder(InHolder), Op(InOp), Audio(MoveTemp(InAudio))
			{
			}

			virtual TSharedPtr<IPipelineModel> GetModel() const override { return Model; }

			virtual FPipelineResult Begin() override
			{
				if (Op == EStreamOp::Close)
				{
					return FPipelineResult::Ok();
				}
				if (!Model.IsValid())
				{
					return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("the pipeline has no model"));
				}
				if (!Model->IsInitialized())
				{
					const FPipelineResult Init = Model->Initialize();
					if (!Init.bSuccess)
					{
						return Init;
					}
				}
				if (Model->AsSpeechEngine() == nullptr)
				{
					return FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("the model has no speech engine"));
				}
				Device = Model->GetDevice();
				return FPipelineResult::Ok();
			}

			virtual EPipelineStep Step(TArray<FPipelineSessionProgress>& OutProgress, FPipelineResult& OutResult) override
			{
				if (Op == EStreamOp::Close)
				{
					Holder->Stream.Reset();
					return EPipelineStep::Finished;
				}
				const double T0 = FPlatformTime::Seconds();
				FPipelineResult R = Run();
				InferenceMs = (float)((FPlatformTime::Seconds() - T0) * 1000.0);
				Audio.Empty();
				if (!R.bSuccess)
				{
					if (R.Error == EPipelineError::None)
					{
						R.Error = EPipelineError::InferenceFailed;
					}
					R.Message = FString::Printf(TEXT("engine stream: %s"), *R.Message);
					OutResult = MoveTemp(R);
					return EPipelineStep::Failed;
				}
				return EPipelineStep::Finished;
			}

			virtual void End(EPipelineSessionEnd Reason, FPipelineResult& OutFinal) override
			{
				Audio.Empty();
				if (Reason == EPipelineSessionEnd::Finished)
				{
					OutFinal = FPipelineResult::Ok();
					OutFinal.Device = Device;
					OutFinal.InferenceMs = InferenceMs;
				}
			}

			int64 GetAudioNum() const { return Audio.Num(); }

			// Read on the game thread only after the session completed.
			TArray<FSpeechTranscript> Finals;
			FSpeechTranscript PartialOut;
			float InferenceMs = 0.f;

		private:
			FPipelineResult Run()
			{
				ISpeechEngine* Engine = Model.IsValid() ? Model->AsSpeechEngine() : nullptr;
				if (!Engine)
				{
					return FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("the model has no speech engine"));
				}
				if (Op == EStreamOp::Reopen)
				{
					Holder->Stream.Reset();
				}
				if (Op == EStreamOp::Open || Op == EStreamOp::Reopen)
				{
					if (!Holder->Stream.IsValid())
					{
						FPipelineResult R = FPipelineResult::Ok();
						Holder->Stream = Engine->OpenStream(StreamRate, R);
						if (!R.bSuccess || !Holder->Stream.IsValid())
						{
							Holder->Stream.Reset();
							return R.bSuccess ? FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("the engine opened no stream")) : R;
						}
					}
					return FPipelineResult::Ok();
				}
				ISpeechEngineStream* Stream = Holder->Stream.Get();
				if (!Stream)
				{
					return FPipelineResult::Fail(EPipelineError::Internal, TEXT("the engine stream is not open"));
				}
				switch (Op)
				{
				case EStreamOp::Feed:
				{
					if (Audio.Num() > 0)
					{
						const FPipelineResult R = Stream->Feed(Audio);
						if (!R.bSuccess)
						{
							return R;
						}
					}
					return Stream->TakeFinals(Finals);
				}
				case EStreamOp::Flush:
				{
					if (Audio.Num() > 0)
					{
						const FPipelineResult R = Stream->Feed(Audio);
						if (!R.bSuccess)
						{
							return R;
						}
					}
					TArray<FSpeechTranscript> Cut;
					const FPipelineResult T = Stream->TakeFinals(Cut);
					if (!T.bSuccess)
					{
						return T;
					}
					const FPipelineResult R = Stream->Flush(Finals);
					Cut.Append(MoveTemp(Finals));
					Finals = MoveTemp(Cut);
					return R;
				}
				case EStreamOp::Partial:
					return Stream->Partial(PartialOut);
				default:
					return FPipelineResult::Ok();
				}
			}

			TSharedPtr<IPipelineModel> Model;
			TSharedRef<FEngineStreamHolder, ESPMode::ThreadSafe> Holder;
			EStreamOp Op;
			TArray<float> Audio;
			EPipelineDevice Device = EPipelineDevice::CPU;
		};

		struct FQueuedOp
		{
			EStreamOp Op = EStreamOp::Feed;
			TArray<float> Audio;
			int32 Utterance = -1;
			int32 Epoch = 0;   // engine-stream generation (WP9R.7: a Reopen starts a new one)
			bool bLive = true; // C83 am.2 (2): false for re-fed turn audio and fallback audio (not counted by back-pressure)
		};

		/** One ended utterance waiting for its final. */
		struct FUtteranceRecord
		{
			int64 Start = 0;
			int64 End = 0;
			double EngineOffsetSeconds = 0.0;   // stream clock minus engine-stream clock (Phonon)
			double EndDecisionTime = 0.0;
			bool bReady = false;
			bool bBroadcastEmpty = false;   // WP9R.7: an utterance below MinSpeechSeconds: its empty final is broadcast
			bool bReplaces = false;         // WP9R.7: joins the previous real utterance (bReplacesPrevious)
			int32 TurnId = -1;
			int32 Epoch = 0;
			FSpeechTranscript Final;
		};
	}

	using namespace StreamPrivate;

	/** Shared state of one USpeechStream. Producer fields: any thread; front-end fields: one task at a time; rest: game thread. */
	class FSpeechStreamImpl : public TSharedFromThis<FSpeechStreamImpl, ESPMode::ThreadSafe>
	{
	public:
		explicit FSpeechStreamImpl(const FSpeechVadSettings& VadSettings) : Vad(VadSettings) {}

		// ---- immutable after Start ----
		FSpeechStreamSettings Settings;
		TWeakObjectPtr<USpeechStream> Owner;
		TSharedPtr<IPipelineModel> EngineModel;   // set: Phonon engine path
		bool bPartialsEnabled = false;

		// ---- producer (any thread) ----
		TQueue<FStreamChunk, EQueueMode::Mpsc> Input;
		std::atomic<int64> Queued16k{ 0 };
		std::atomic<int32> QueuedChunks{ 0 };
		std::atomic<bool> bStopped{ false };
		std::atomic<bool> bWarnedRate{ false };
		std::atomic<bool> bWarnedChannels{ false };
		std::atomic<bool> bWarnedNonFinite{ false };

		// ---- front end (one task at a time) ----
		std::atomic<bool> bFrontEndBusy{ false };
		FSpeechVad Vad;
		std::atomic<bool> bWarnedBackPressure{ false };   // shared by the front end and the game thread (one Warning per stream)
		std::atomic<int64> DroppedSamples16k{ 0 };
		FCriticalSection OutLock;
		TArray<FFrontEndItem> Out;

		// ---- game thread ----
		FTSTicker::FDelegateHandle TickerHandle;
		bool bStopRequested = false;
		bool bFinished = false;
		bool bSpeaking = false;
		TArray<float> History;
		int64 HistoryStart = 0;
		int64 UtteranceStart = 0;
		int64 Cursor = 0;            // next sample to feed (Phonon)
		int32 UtteranceIndex = -1;   // current / last utterance
		int64 SamplesSincePartial = 0;
		FString LastPartialText;
		bool bPartialOutstanding = false;
		bool bOpenQueued = false;
		int64 FedSamples = 0;        // total fed into the engine stream
		double EngineOffsetSeconds = 0.0;
		TMap<int32, TArray<FSpeechTranscript>> CutFinals;   // per utterance: finals the engine cut by itself before its Flush
		TArray<FQueuedOp> Ops;
		bool bOpInFlight = false;
		int64 InFlightSamples = 0;   // audio of the op running on the executor (back-pressure, rule 7)
		TMap<int32, FUtteranceRecord> Utterances;
		int32 NextFinal = 0;
		// WP9R.7 join and correct (C83)
		int64 MergeWindowSamples = 0;
		int64 MinSpeechSamples = 0;
		int64 MaxTurnSamples = 0;
		bool bConfirmed = false;        // the current utterance reached MinSpeechSeconds of speech frames
		int64 CurFirstSpeech = 0;       // first speech frame of the current utterance (before the pre-roll)
		bool bTentativeJoin = false;    // Phonon: the current utterance re-fed the turn audio from TurnStart
		bool bTurnValid = false;
		int64 TurnStart = 0;            // first sample of the turn (pre-roll included)
		int64 TurnEnd = 0;              // SpeechEnded sample of the turn's last real utterance
		int32 TurnCounter = 0;
		int32 LastTextTurn = -1;        // turn of the last broadcast final with text
		bool bTurnPending = false;      // C83 am.1: the turn so far holds only utterances below MinSpeechSeconds (no final with text yet)
		int64 CurSpeechSamples = 0;     // speech frames (samples) of the current utterance so far
		/** C83 am.1 (4): NNE finals run one Transcribe at a time; a queued final superseded by a later ended utterance of the same turn
		    is delivered empty without an engine call. */
		struct FNneFinalReq
		{
			int32 Utt = 0;
			int32 TurnId = -1;
			TArray<float> Audio;
		};
		TArray<FNneFinalReq> NneQueue;
		bool bNneInFlight = false;
		int32 Epoch = 0;
		TSharedRef<FEngineStreamHolder, ESPMode::ThreadSafe> Holder = MakeShared<FEngineStreamHolder, ESPMode::ThreadSafe>();

		int64 HistoryEnd() const { return HistoryStart + History.Num(); }

		// =========================================================================================================
		// Producer
		// =========================================================================================================
		void Push(TConstArrayView<float> Interleaved, int32 SampleRate, int32 NumChannels)
		{
			if (bStopped.load())
			{
				return;
			}
			if (SampleRate < 1000 || SampleRate > 384000)
			{
				if (!bWarnedRate.exchange(true))
				{
					UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechStream::PushAudio: sample rate %d is outside 1000..384000; chunk dropped (further bad-rate chunks are dropped silently)"), SampleRate);
				}
				return;
			}
			if (NumChannels < 1 || Interleaved.Num() % NumChannels != 0)
			{
				if (!bWarnedChannels.exchange(true))
				{
					UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechStream::PushAudio: %d samples do not fit %d channels; chunk dropped (further bad-channel chunks are dropped silently)"), Interleaved.Num(), NumChannels);
				}
				return;
			}
			if (Interleaved.Num() == 0)
			{
				return;
			}
			for (const float S : Interleaved)
			{
				if (!FMath::IsFinite(S))
				{
					if (!bWarnedNonFinite.exchange(true))
					{
						UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechStream::PushAudio: non-finite sample; chunk dropped (further such chunks are dropped silently)"));
					}
					return;
				}
			}
			FStreamChunk Chunk;
			Chunk.Samples = TArray<float>(Interleaved.GetData(), Interleaved.Num());
			Chunk.SampleRate = SampleRate;
			Chunk.NumChannels = NumChannels;
			Chunk.Samples16k = FMath::Max<int64>(1, (int64)(Interleaved.Num() / NumChannels) * StreamRate / SampleRate);
			Queued16k.fetch_add(Chunk.Samples16k);
			QueuedChunks.fetch_add(1);
			Input.Enqueue(MoveTemp(Chunk));
			// Fix round 1 (latency): start the front end at once instead of waiting for the next tick.
			LaunchFrontEndIfIdle();
		}

		/** Rule 7 bookkeeping (any thread): one Warning per stream, a Verbose running total. */
		void NoteBackPressureDrop(int64 Samples)
		{
			const int64 Total = DroppedSamples16k.fetch_add(Samples) + Samples;
			if (!bWarnedBackPressure.exchange(true))
			{
				UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechStream: more than 10 s of audio queued (consumer too slow); dropping the oldest audio"));
			}
			UE_LOG(LogModelPipelinesSpeech, Verbose, TEXT("USpeechStream: back-pressure dropped %.2f s so far"), (double)Total / StreamRate);
		}

		/** Game thread, Phonon path (rule 7): audio still waiting for the engine (queued ops + the op in flight) counts against the
		    10 s limit together with the front-end queue; the oldest not yet enqueued Feed/Flush audio is dropped. */
		void EnforceEngineBackPressure()
		{
			if (!EngineModel.IsValid())
			{
				return;
			}
			// C83 am.2 (2): only LIVE audio not yet handed to the engine counts; re-fed turn audio and fallback audio do not.
			int64 Pending = InFlightSamples + FMath::Max<int64>(0, Queued16k.load());
			for (const FQueuedOp& Op : Ops)
			{
				Pending += Op.bLive ? Op.Audio.Num() : 0;
			}
			int64 Excess = Pending - MaxQueuedSamples;
			for (int32 i = 0; i < Ops.Num() && Excess > 0; ++i)
			{
				FQueuedOp& Op = Ops[i];
				if (!Op.bLive)
				{
					continue;
				}
				const int32 N = (int32)FMath::Min<int64>(Excess, Op.Audio.Num());
				if (N <= 0)
				{
					continue;
				}
				Op.Audio.RemoveAt(0, N, EAllowShrinking::No);
				Excess -= N;
				NoteBackPressureDrop(N);
				// The engine stream never sees the dropped audio: its clock runs N samples behind the stream clock from here on.
				// Only the engine stream (epoch) that would have received the audio is affected.
				const double Shift = (double)N / StreamRate;
				for (TPair<int32, FUtteranceRecord>& Pair : Utterances)
				{
					if (Pair.Key >= Op.Utterance && Pair.Value.Epoch == Op.Epoch)
					{
						Pair.Value.EngineOffsetSeconds += Shift;
					}
				}
				if (Op.Epoch == Epoch)
				{
					FedSamples -= N;
					if (UtteranceIndex >= Op.Utterance)
					{
						EngineOffsetSeconds += Shift;
					}
				}
			}
		}

		// =========================================================================================================
		// Front end (background task)
		// =========================================================================================================
		void LaunchFrontEndIfIdle()
		{
			if (!bFrontEndBusy.exchange(true))
			{
				UE::Tasks::Launch(TEXT("ModelPipelinesSpeechStreamFrontEnd"), [Self = AsShared()]() { Self->RunFrontEnd(); });
			}
		}

		void RunFrontEnd()
		{
			for (;;)
			{
				RunFrontEndOnce();
				bFrontEndBusy.store(false);
				// A chunk pushed between the last Dequeue and the store above saw the task busy: take it now.
				if (Input.IsEmpty() || bFrontEndBusy.exchange(true))
				{
					return;
				}
			}
		}

		void RunFrontEndOnce()
		{
			FStreamChunk Chunk;
			while (Input.Dequeue(Chunk))
			{
				QueuedChunks.fetch_sub(1);
				const int64 Before = Queued16k.fetch_sub(Chunk.Samples16k);
				if (Before - Chunk.Samples16k >= MaxQueuedSamples)
				{
					// Rule 7: more than 10 s queued behind this chunk: drop the oldest audio.
					NoteBackPressureDrop(Chunk.Samples16k);
					continue;
				}
				FString Error;
				TArray<float> Mono;
				if (Chunk.NumChannels > 1)
				{
					if (!MixToMono(Chunk.Samples, Chunk.NumChannels, Mono, Error))
					{
						continue;
					}
				}
				else
				{
					Mono = MoveTemp(Chunk.Samples);
				}
				FFrontEndItem Item;
				if (Chunk.SampleRate != StreamRate)
				{
					if (!Resample(Mono, Chunk.SampleRate, StreamRate, Item.Audio, Error))
					{
						continue;
					}
				}
				else
				{
					Item.Audio = MoveTemp(Mono);
				}
				Vad.Process(Item.Audio, Item.Events);
				Item.SpeechSamplesNow = Vad.GetSpeechSamples();
				FScopeLock Lock(&OutLock);
				Out.Add(MoveTemp(Item));
			}
		}

		// =========================================================================================================
		// Game thread
		// =========================================================================================================
		bool Tick(float)
		{
			USpeechStream* S = Owner.Get();
			if (!S || bFinished)
			{
				TickerHandle.Reset();
				return false;
			}
			if (!FPipelineExecutor::TryGetShared().IsValid())
			{
				Finish(FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor shut down")));
				TickerHandle.Reset();
				return false;
			}

			// Fix round 1 (latency): start the front end before consuming its output.
			if (QueuedChunks.load() > 0)
			{
				LaunchFrontEndIfIdle();
			}
			TArray<FFrontEndItem> Items;
			{
				FScopeLock Lock(&OutLock);
				Items = MoveTemp(Out);
				Out.Reset();
			}
			for (FFrontEndItem& Item : Items)
			{
				ProcessItem(Item);
				if (bFinished)
				{
					break;
				}
			}
			if (!bFinished)
			{
				EnforceEngineBackPressure();
			}
			if (!bFinished && QueuedChunks.load() > 0)
			{
				LaunchFrontEndIfIdle();
			}
			else if (!bFinished && bStopRequested && !bFrontEndBusy.load() && QueuedChunks.load() == 0 && IsOutEmpty())
			{
				// Everything pushed before Stop() is processed: end the current utterance now (rule 6, no hangover wait).
				if (bSpeaking)
				{
					// Stop flushes as before WP9R.7: the cut utterance gets its engine call even below MinSpeechSeconds.
					Confirm(MinSpeechSamples, HistoryEnd());
					FeedUpTo(HistoryEnd());
					EndUtterance(HistoryEnd());
				}
				PumpOps();
				if (!bFinished && Ops.Num() == 0 && !bOpInFlight && NextFinal > UtteranceIndex)
				{
					if (DroppedSamples16k.load() > 0)
					{
						UE_LOG(LogModelPipelinesSpeech, Log, TEXT("USpeechStream: back-pressure dropped %.2f s of audio in total"), (double)DroppedSamples16k.load() / StreamRate);
					}
					Finish(FPipelineResult::Ok());
				}
			}
			if (!bFinished)
			{
				PumpOps();
			}
			if (bFinished)
			{
				TickerHandle.Reset();
				return false;
			}
			return true;
		}

		bool IsOutEmpty()
		{
			FScopeLock Lock(&OutLock);
			return Out.Num() == 0;
		}

		void ProcessItem(FFrontEndItem& Item)
		{
			History.Append(MoveTemp(Item.Audio));
			for (const FSpeechVadEvent& E : Item.Events)
			{
				if (E.Type == ESpeechVadEventType::Started)
				{
					if (!bSpeaking)
					{
						BeginUtterance(E.SampleIndex, E.FirstSpeechSample, E.SpeechSamples);
					}
				}
				else if (bSpeaking)
				{
					Confirm(E.SpeechSamples, E.SampleIndex);
					FeedUpTo(E.SampleIndex);
					EndUtterance(E.SampleIndex);
				}
				if (bFinished || !Owner.IsValid())
				{
					return;
				}
			}
			if (bSpeaking)
			{
				Confirm(Item.SpeechSamplesNow, HistoryEnd());
				FeedUpTo(HistoryEnd());
				MaybeRequestPartial();
			}
			// WP9R.7: a turn that can no longer be joined (no new start can fall inside the window, or any new end would exceed
			// MaxUtteranceSeconds) releases its audio.
			if (bTurnValid && ((!bSpeaking && HistoryEnd() - 4 * FSpeechVad::FrameSamples - TurnEnd > MergeWindowSamples)
				|| HistoryEnd() - FSpeechVad::FrameSamples - TurnStart > MaxTurnSamples))
			{
				bTurnValid = false;
			}
			// Keep what is still needed (the current utterance from its start: NNE for its Transcribe, Phonon for a fallback re-feed
			// of a join that grew too long; the turn audio while it can still be joined) plus a second of history for the next pre-roll.
			int64 Needed = bSpeaking ? FMath::Min(UtteranceStart, Cursor) : HistoryEnd();
			if (bTurnValid && MergeWindowSamples > 0)
			{
				Needed = FMath::Min(Needed, TurnStart);
			}
			const int64 Keep = FMath::Min(Needed, HistoryEnd() - IdleHistorySamples);
			if (Keep > HistoryStart)
			{
				History.RemoveAt(0, (int32)(Keep - HistoryStart), EAllowShrinking::No);
				HistoryStart = Keep;
			}
		}

		void BeginUtterance(int64 Start, int64 FirstSpeech, int64 SpeechSamples)
		{
			Start = FMath::Clamp(Start, HistoryStart, HistoryEnd());
			FirstSpeech = FMath::Clamp(FirstSpeech, Start, HistoryEnd());
			if (MergeWindowSamples > 0 && bTurnValid && FirstSpeech - TurnEnd <= MergeWindowSamples)
			{
				// WP9R.7: speech inside the merge window: the joined audio is contiguous, so the pre-roll is not clamped at the previous
				// end (also when the join fails on MaxUtteranceSeconds and this utterance starts a new turn).
				Start = FMath::Clamp(FirstSpeech - FSpeechVad::PreRollSamples, HistoryStart, Start);
			}
			bSpeaking = true;
			UtteranceStart = Start;
			Cursor = Start;
			++UtteranceIndex;
			SamplesSincePartial = 0;
			LastPartialText.Reset();
			EngineOffsetSeconds = (double)(Start - FedSamples) / StreamRate;
			bConfirmed = false;
			bTentativeJoin = false;
			CurSpeechSamples = 0;
			CurFirstSpeech = FMath::Max(FirstSpeech, Start);
			if (EngineModel.IsValid() && !bOpenQueued)
			{
				bOpenQueued = true;
				Ops.Add({ EStreamOp::Open, TArray<float>(), UtteranceIndex, Epoch });
			}
			if (USpeechStream* S = Owner.Get())
			{
				S->OnSpeechStarted.Broadcast();
			}
			if (!bFinished && Owner.IsValid() && bSpeaking)
			{
				Confirm(SpeechSamples, HistoryEnd());
			}
		}

		/** WP9R.7: true if an utterance whose first speech frame is FirstSpeech and that ends at End joins the current turn. */
		bool CanJoin(int64 FirstSpeech, int64 End) const
		{
			return MergeWindowSamples > 0 && bTurnValid && TurnStart >= HistoryStart
				&& FirstSpeech - TurnEnd <= MergeWindowSamples && End - TurnStart <= MaxTurnSamples;
		}

		/** WP9R.7: the current utterance may reach the engine: it has MinSpeechSeconds of speech frames, or (C83 am.1) it joins an open
		    turn whatever its length. Phonon: nothing was fed before this point; a joining utterance (or one continuing a pending turn)
		    re-feeds the engine stream from the turn start (the engine clock restarts there), any other feeds from its start. */
		void Confirm(int64 SpeechSamples, int64 At)
		{
			if (!bSpeaking)
			{
				return;
			}
			CurSpeechSamples = FMath::Max(CurSpeechSamples, SpeechSamples);
			if (bConfirmed)
			{
				return;
			}
			const int64 Now = FMath::Max(At, UtteranceStart);
			const bool bJoinsOpenTurn = bTurnValid && !bTurnPending && CanJoin(CurFirstSpeech, Now);
			if (CurSpeechSamples < MinSpeechSamples && !bJoinsOpenTurn)
			{
				return;   // a pending-turn start keeps the gate
			}
			bConfirmed = true;
			if (!EngineModel.IsValid())
			{
				return;
			}
			bTentativeJoin = CanJoin(CurFirstSpeech, Now);
			const int64 From = bTentativeJoin ? TurnStart : UtteranceStart;
			Cursor = From;
			EngineOffsetSeconds = (double)(From - FedSamples) / StreamRate;
			if (bTentativeJoin)
			{
				// C83 am.2 (2): the re-fed turn audio (turn start .. now) goes in its own op that back-pressure does not count.
				FeedUpTo(Now, false);
			}
		}

		void FeedUpTo(int64 End, bool bLive = true)
		{
			if (EngineModel.IsValid() && !bConfirmed)
			{
				return;   // WP9R.7 (MinSpeechSeconds): nothing reaches the engine before the utterance has enough speech
			}
			End = FMath::Min(End, HistoryEnd());
			if (End <= Cursor)
			{
				return;
			}
			const int64 From = FMath::Max(Cursor, HistoryStart);
			const int32 N = (int32)(End - From);
			Cursor = End;
			if (!EngineModel.IsValid() || N <= 0)
			{
				return;
			}
			SamplesSincePartial += N;
			FedSamples += N;
			const float* Data = History.GetData() + (From - HistoryStart);
			if (Ops.Num() > 0 && Ops.Last().Op == EStreamOp::Feed && Ops.Last().Utterance == UtteranceIndex && Ops.Last().Epoch == Epoch
				&& Ops.Last().bLive == bLive)
			{
				Ops.Last().Audio.Append(Data, N);
			}
			else
			{
				Ops.Add({ EStreamOp::Feed, TArray<float>(Data, N), UtteranceIndex, Epoch, bLive });
			}
		}

		void MaybeRequestPartial()
		{
			if (!bPartialsEnabled || bPartialOutstanding || bStopRequested
				|| SamplesSincePartial < (int64)FMath::RoundToDouble((double)Settings.PartialIntervalSeconds * StreamRate))
			{
				return;
			}
			SamplesSincePartial = 0;
			bPartialOutstanding = true;
			Ops.Add({ EStreamOp::Partial, TArray<float>(), UtteranceIndex, Epoch });
		}

		void EndUtterance(int64 End)
		{
			End = FMath::Clamp(End, UtteranceStart, HistoryEnd());
			bSpeaking = false;
			const int32 Utt = UtteranceIndex;
			FUtteranceRecord& Rec = Utterances.Add(Utt);
			Rec.Start = UtteranceStart;
			Rec.End = End;
			Rec.EndDecisionTime = FPlatformTime::Seconds();

			const bool bShort = CurSpeechSamples < MinSpeechSamples;
			const bool bOpenTurn = bTurnValid && !bTurnPending;
			bool bJoin = false;
			bool bFallback = false;
			if (EngineModel.IsValid())
			{
				bJoin = bTentativeJoin && TurnStart >= HistoryStart && End - TurnStart <= MaxTurnSamples;
				bFallback = bTentativeJoin && !bJoin;
			}
			else
			{
				bJoin = CanJoin(CurFirstSpeech, End);
			}
			bTentativeJoin = false;
			const bool bJoinOpen = bJoin && bOpenTurn;

			if (bShort && !bJoinOpen)
			{
				// C83 am.1 (2): below MinSpeechSeconds and not joining an open turn: an empty final without an engine call. It extends a
				// pending turn it starts inside the window of, or opens a new pending turn (a real utterance inside the window then starts
				// its turn here; otherwise it is dropped as noise).
				if (bTurnValid && bTurnPending && CanJoin(CurFirstSpeech, End))
				{
					TurnEnd = End;
				}
				else
				{
					bTurnValid = MergeWindowSamples > 0;
					bTurnPending = true;
					TurnStart = UtteranceStart;
					TurnEnd = End;
				}
				if (EngineModel.IsValid() && bConfirmed)
				{
					// Phonon: it was fed for a join that turned out too long: discard what the engine stream got (its queued feeds
					// are dropped: the Reopen discards them anyway).
					for (int32 i = Ops.Num() - 1; i >= 0; --i)
					{
						if (Ops[i].Op == EStreamOp::Feed && Ops[i].Utterance == Utt && Ops[i].Epoch == Epoch)
						{
							Ops.RemoveAt(i);
						}
						else if (Ops[i].Op == EStreamOp::Partial)
						{
							Ops.RemoveAt(i);
							bPartialOutstanding = false;
						}
					}
					++Epoch;
					Ops.Add({ EStreamOp::Reopen, TArray<float>(), Utt, Epoch });
					FedSamples = 0;
					Cursor = End;
				}
				Rec.bBroadcastEmpty = true;
				Rec.EngineOffsetSeconds = EngineOffsetSeconds;
				Rec.Epoch = Epoch;
				if (USpeechStream* S = Owner.Get())
				{
					S->OnSpeechEnded.Broadcast();
				}
				if (!bFinished && Owner.IsValid())
				{
					CompleteFinal(Utt, FSpeechTranscript());
				}
				return;
			}

			if (bJoinOpen)
			{
				Rec.Start = TurnStart;
				Rec.bReplaces = true;
				TurnEnd = End;
			}
			else if (bJoin)
			{
				// C83 am.1 (2): a real utterance continuing a pending turn: a normal first final covering the short utterance(s) too.
				Rec.Start = TurnStart;
				bTurnPending = false;
				TurnEnd = End;
				++TurnCounter;
			}
			else
			{
				bTurnValid = true;
				bTurnPending = false;
				TurnStart = UtteranceStart;
				TurnEnd = End;
				++TurnCounter;
			}
			Rec.TurnId = TurnCounter;

			if (EngineModel.IsValid())
			{
				// Queued partials of this utterance are dropped; the remaining audio goes with the Flush.
				for (int32 i = Ops.Num() - 1; i >= 0; --i)
				{
					if (Ops[i].Op == EStreamOp::Partial)
					{
						Ops.RemoveAt(i);
						bPartialOutstanding = false;
					}
				}
				if (bFallback)
				{
					// WP9R.7: the joined turn would exceed MaxUtteranceSeconds: discard what the engine stream got for this turn, open a
					// fresh stream (its clock starts at 0) and give it this utterance alone (a new turn).
					for (int32 i = Ops.Num() - 1; i >= 0; --i)
					{
						if (Ops[i].Op == EStreamOp::Feed && Ops[i].Utterance == Utt && Ops[i].Epoch == Epoch)
						{
							Ops.RemoveAt(i);   // fed to the stream the Reopen discards
						}
					}
					++Epoch;
					Ops.Add({ EStreamOp::Reopen, TArray<float>(), Utt, Epoch });
					const int64 From = FMath::Max(UtteranceStart, HistoryStart);
					TArray<float> Audio;
					if (End > From)
					{
						Audio.Append(History.GetData() + (From - HistoryStart), (int32)(End - From));
					}
					EngineOffsetSeconds = (double)From / StreamRate;
					FedSamples = Audio.Num();
					Cursor = End;
					Ops.Add({ EStreamOp::Flush, MoveTemp(Audio), Utt, Epoch, false });   // C83 am.2 (2): not counted by back-pressure
				}
				else if (Ops.Num() > 0 && Ops.Last().Op == EStreamOp::Feed && Ops.Last().Epoch == Epoch)
				{
					Ops.Last().Op = EStreamOp::Flush;
					Ops.Last().Utterance = Utt;
				}
				else
				{
					Ops.Add({ EStreamOp::Flush, TArray<float>(), Utt, Epoch });
				}
			}
			Rec.EngineOffsetSeconds = EngineOffsetSeconds;
			Rec.Epoch = Epoch;
			const int64 FinalStart = Rec.Start;
			if (!EngineModel.IsValid())
			{
				StartNneFinal(Utt, FinalStart, End);
			}

			if (!bFinished && Owner.IsValid())
			{
				if (USpeechStream* S = Owner.Get())
				{
					S->OnSpeechEnded.Broadcast();
				}
			}
		}

		/** Stage A: the utterance's 16 kHz audio through one USpeechToTextPipeline::Transcribe. */
		void StartNneFinal(int32 Utt, int64 Start, int64 End)
		{
			USpeechStream* S = Owner.Get();
			USpeechToTextPipeline* Pipe = S ? S->Pipeline.Get() : nullptr;
			const int64 From = FMath::Max(Start, HistoryStart);
			TArray<float> Audio;
			if (End > From)
			{
				Audio.Append(History.GetData() + (From - HistoryStart), (int32)(End - From));
			}
			if (!Pipe || Audio.Num() == 0)
			{
				CompleteFinal(Utt, FSpeechTranscript());
				return;
			}
			const FUtteranceRecord* Rec = Utterances.Find(Utt);
			const int32 TurnId = Rec ? Rec->TurnId : -1;
			if (TurnId >= 0)
			{
				for (FNneFinalReq& Queued : NneQueue)
				{
					if (Queued.TurnId == TurnId)
					{
						Queued.Audio.Empty();   // C83 am.2 (4): superseded; PumpNne delivers it empty without an engine call
					}
				}
			}
			FNneFinalReq& Req = NneQueue.AddDefaulted_GetRef();
			Req.Utt = Utt;
			Req.TurnId = TurnId;
			Req.Audio = MoveTemp(Audio);
			PumpNne();
		}

		/** Issues the next queued NNE final (one Transcribe at a time). C83 am.1 (4): a final whose turn already has a later ended
		    utterance queued is delivered empty without an engine call (the later correction covers the turn). */
		void PumpNne()
		{
			while (!bNneInFlight && !bFinished && Owner.IsValid() && NneQueue.Num() > 0)
			{
				FNneFinalReq Req = MoveTemp(NneQueue[0]);
				NneQueue.RemoveAt(0);
				bool bSuperseded = false;
				for (const FNneFinalReq& Later : NneQueue)
				{
					bSuperseded |= Req.TurnId >= 0 && Later.TurnId == Req.TurnId;
				}
				USpeechStream* S = Owner.Get();
				USpeechToTextPipeline* Pipe = S ? S->Pipeline.Get() : nullptr;
				if (bSuperseded || !Pipe)
				{
					if (FUtteranceRecord* Rec = Utterances.Find(Req.Utt))
					{
						Rec->bBroadcastEmpty = bSuperseded;
						Rec->bReplaces = false;
					}
					CompleteFinal(Req.Utt, FSpeechTranscript());
					continue;
				}
				IssueNneTranscribe(Pipe, Req.Utt, MoveTemp(Req.Audio));
			}
		}

		void IssueNneTranscribe(USpeechToTextPipeline* Pipe, int32 Utt, TArray<float>&& Audio)
		{
			bNneInFlight = true;
			TWeakPtr<FSpeechStreamImpl, ESPMode::ThreadSafe> Weak = AsShared();
			const TWeakObjectPtr<USpeechStream> WeakOwner = Owner;
			Pipe->Transcribe(MoveTemp(Audio), StreamRate, 1, [Weak, WeakOwner, Utt](const FPipelineResult& Result, const FSpeechTranscript& Transcript)
			{
				const TSharedPtr<FSpeechStreamImpl, ESPMode::ThreadSafe> Self = Weak.Pin();
				if (!Self.IsValid())
				{
					return;
				}
				Self->bNneInFlight = false;
				if (!WeakOwner.IsValid() || Self->bFinished)
				{
					return;
				}
				ON_SCOPE_EXIT
				{
					if (!Self->bFinished && WeakOwner.IsValid())
					{
						Self->PumpNne();
					}
				};
				if (!Result.bSuccess)
				{
					if (Result.Error == EPipelineError::Cancelled)
					{
						Self->Finish(Result);
						return;
					}
					UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechStream: the final of utterance %d failed: %s"), Utt, *Result.Message);
					Self->CompleteFinal(Utt, FSpeechTranscript());
					return;
				}
				FSpeechTranscript T = Transcript;
				if (T.InferenceMs <= 0.f)
				{
					T.InferenceMs = Result.InferenceMs;
				}
				Self->CompleteFinal(Utt, MoveTemp(T));
			});
		}

		void PumpOps()
		{
			if (bOpInFlight || bFinished || Ops.Num() == 0)
			{
				return;
			}
			FQueuedOp Op = MoveTemp(Ops[0]);
			Ops.RemoveAt(0);
			const TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor = FPipelineExecutor::TryGetShared();
			if (!Executor.IsValid())
			{
				Finish(FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor shut down")));
				return;
			}
			const TSharedRef<FSpeechStreamOpSession, ESPMode::ThreadSafe> Session =
				MakeShared<FSpeechStreamOpSession, ESPMode::ThreadSafe>(EngineModel, Holder, Op.Op, MoveTemp(Op.Audio));
			TWeakPtr<FSpeechStreamImpl, ESPMode::ThreadSafe> Weak = AsShared();
			FPipelineSessionCallbacks Callbacks;
			const EStreamOp Kind = Op.Op;
			const int32 Utt = Op.Utterance;
			Callbacks.OnComplete = [Weak, Session, Kind, Utt](const FPipelineResult& Result)
			{
				const TSharedPtr<FSpeechStreamImpl, ESPMode::ThreadSafe> Self = Weak.Pin();
				if (Self.IsValid())
				{
					Self->OnOpComplete(Kind, Utt, Result, *Session);
				}
			};
			bOpInFlight = true;
			InFlightSamples = Op.bLive ? Session->GetAudioNum() : 0;   // C83 am.2 (2)
			if (Executor->EnqueueSession(Session, MoveTemp(Callbacks), Owner.Get()) == 0)
			{
				bOpInFlight = false;
				Finish(FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor shut down")));
			}
		}

		void OnOpComplete(EStreamOp Kind, int32 Utt, const FPipelineResult& Result, FSpeechStreamOpSession& Session)
		{
			bOpInFlight = false;
			InFlightSamples = 0;
			if (bFinished || !Owner.IsValid())
			{
				return;
			}
			if (Kind == EStreamOp::Partial)
			{
				bPartialOutstanding = false;
				if (!Result.bSuccess)
				{
					UE_LOG(LogModelPipelinesSpeech, Verbose, TEXT("USpeechStream: partial failed: %s"), *Result.Message);
				}
				else if (bSpeaking && Utt == UtteranceIndex && !Session.PartialOut.Text.Equals(LastPartialText, ESearchCase::CaseSensitive))
				{
					LastPartialText = Session.PartialOut.Text;
					FSpeechTranscript T = Session.PartialOut;
					OffsetWords(T, EngineOffsetSeconds);
					if (USpeechStream* S = Owner.Get())
					{
						S->OnPartial.Broadcast(T);
					}
				}
			}
			else if (!Result.bSuccess)
			{
				Finish(Result);
				return;
			}
			else if (Kind == EStreamOp::Reopen)
			{
				CutFinals.Remove(Utt);   // finals the discarded stream cut for the abandoned join
			}
			else if (Kind == EStreamOp::Feed)
			{
				if (Session.Finals.Num() > 0)
				{
					CutFinals.FindOrAdd(Utt).Append(MoveTemp(Session.Finals));
				}
			}
			else if (Kind == EStreamOp::Flush)
			{
				TArray<FSpeechTranscript> All;
				CutFinals.RemoveAndCopyValue(Utt, All);
				All.Append(MoveTemp(Session.Finals));
				FSpeechTranscript Merged;
				for (FSpeechTranscript& T : All)
				{
					const FString Text = T.Text.TrimStartAndEnd();
					if (!Text.IsEmpty())
					{
						Merged.Text = Merged.Text.IsEmpty() ? Text : Merged.Text + TEXT(" ") + Text;
					}
					Merged.Words.Append(T.Words);
					Merged.InferenceMs += T.InferenceMs;
				}
				if (Merged.InferenceMs <= 0.f)
				{
					Merged.InferenceMs = Session.InferenceMs;
				}
				CompleteFinal(Utt, MoveTemp(Merged));
			}
			if (!bFinished)
			{
				PumpOps();
			}
		}

		static void OffsetWords(FSpeechTranscript& T, double Seconds)
		{
			for (FSpeechWord& W : T.Words)
			{
				W.StartSeconds += (float)Seconds;
				W.EndSeconds += (float)Seconds;
			}
		}

		/** Stores the final of utterance Utt and broadcasts every final that is next in order. */
		void CompleteFinal(int32 Utt, FSpeechTranscript&& Final)
		{
			FUtteranceRecord* Rec = Utterances.Find(Utt);
			if (!Rec)
			{
				return;
			}
			const double Offset = EngineModel.IsValid() ? Rec->EngineOffsetSeconds : (double)Rec->Start / StreamRate;
			OffsetWords(Final, Offset);
			Final.AudioSeconds = (float)((double)(Rec->End - Rec->Start) / StreamRate);
			Rec->Final = MoveTemp(Final);
			Rec->bReady = true;
			while (FUtteranceRecord* Next = Utterances.Find(NextFinal))
			{
				if (!Next->bReady)
				{
					break;
				}
				FUtteranceRecord Done = MoveTemp(*Next);
				Utterances.Remove(NextFinal);
				++NextFinal;
				const double Ms = (FPlatformTime::Seconds() - Done.EndDecisionTime) * 1000.0;
				UE_LOG(LogModelPipelinesSpeech, Verbose, TEXT("USpeechStream: final of utterance %d [%.2f..%.2f s] turn %d joined %d below-min-speech %d '%s'"), NextFinal - 1, Done.Start / 16000.0, Done.End / 16000.0,
					Done.TurnId, (int32)Done.bReplaces, (int32)Done.bBroadcastEmpty, *Done.Final.Text);
				if (Done.Final.Text.IsEmpty())
				{
					if (!Done.bBroadcastEmpty)
					{
						UE_LOG(LogModelPipelinesSpeech, Verbose, TEXT("USpeechStream: empty final (not broadcast), speech end -> result %.0f ms"), Ms);
						continue;
					}
					Done.Final.bReplacesPrevious = false;   // WP9R.7: below MinSpeechSeconds (no engine call)
				}
				else
				{
					// WP9R.7: a correction replaces the previous final only if that final (same turn) was broadcast with text.
					Done.Final.bReplacesPrevious = Done.bReplaces && Done.TurnId == LastTextTurn;
					LastTextTurn = Done.TurnId;
				}
				UE_LOG(LogModelPipelinesSpeech, Verbose, TEXT("final: speech end → OnFinal %.0f ms (inference %.0f ms)"), Ms, Done.Final.InferenceMs);
				if (USpeechStream* S = Owner.Get())
				{
					S->OnFinal.Broadcast(Done.Final);
				}
				if (bFinished || !Owner.IsValid())
				{
					return;
				}
			}
		}

		void Finish(const FPipelineResult& Result)
		{
			if (bFinished)
			{
				return;
			}
			bFinished = true;
			bStopRequested = true;
			bStopped.store(true);
			Ops.Reset();
			NneQueue.Reset();
			CutFinals.Reset();
			CloseEngineStream();
			if (TickerHandle.IsValid())
			{
				FTSTicker::GetCoreTicker().RemoveTicker(TickerHandle);
				TickerHandle.Reset();
			}
			if (!Result.bSuccess)
			{
				UE_LOG(LogModelPipelinesSpeech, Log, TEXT("USpeechStream stopped: %s"), *Result.Message);
			}
			if (USpeechStream* S = Owner.Get())
			{
				S->OnStopped.Broadcast(Result);
			}
		}

		/** Fix round 1: the engine stream is released on the executor worker, behind any op still running there (one session at a
		    time). Without an executor (shutdown) the worker is gone and it is released here. */
		void CloseEngineStream()
		{
			if (!bOpenQueued)
			{
				return;
			}
			bOpenQueued = false;
			const TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor = FPipelineExecutor::TryGetShared();
			if (Executor.IsValid())
			{
				const TSharedRef<FSpeechStreamOpSession, ESPMode::ThreadSafe> Session =
					MakeShared<FSpeechStreamOpSession, ESPMode::ThreadSafe>(EngineModel, Holder, EStreamOp::Close, TArray<float>());
				FPipelineSessionCallbacks Callbacks;
				Callbacks.OnComplete = [](const FPipelineResult&) {};
				if (Executor->EnqueueSession(Session, MoveTemp(Callbacks), nullptr) != 0)
				{
					return;
				}
			}
			if (!bOpInFlight)
			{
				Holder->Stream.Reset();
			}
			// else: the op in flight holds the holder; the stream is released with the last reference to it.
		}

		/** The UObject is being destroyed: no events from now on. */
		void Shutdown()
		{
			const bool bWasFinished = bFinished;
			bFinished = true;
			bStopped.store(true);
			Ops.Reset();
			NneQueue.Reset();
			if (!bWasFinished)
			{
				CloseEngineStream();
			}
			if (TickerHandle.IsValid())
			{
				FTSTicker::GetCoreTicker().RemoveTicker(TickerHandle);
				TickerHandle.Reset();
			}
		}
	};
}

// =====================================================================================================================
// USpeechStream
// =====================================================================================================================

void USpeechStream::Start(USpeechToTextPipeline* InPipeline, const FSpeechStreamSettings& InSettings, TSharedPtr<IPipelineModel> InEngineModel)
{
	using namespace UE::ModelPipelines::Speech;
	Pipeline = InPipeline;
	FSpeechVadSettings Vad;
	Vad.HangoverSeconds = InSettings.EndOfSpeechHangoverSeconds;
	Vad.ThresholdDb = InSettings.SpeechThresholdDb;
	Vad.MaxUtteranceSeconds = InSettings.MaxUtteranceSeconds;
	Impl = MakeShared<FSpeechStreamImpl, ESPMode::ThreadSafe>(Vad);
	Impl->Settings = InSettings;
	Impl->Owner = this;
	Impl->MergeWindowSamples = FMath::Max<int64>(0, (int64)FMath::RoundToDouble((double)InSettings.MergeWindowSeconds * 16000.0));
	Impl->MinSpeechSamples = FMath::Max<int64>(0, (int64)FMath::RoundToDouble((double)InSettings.MinSpeechSeconds * 16000.0));
	// C83 am.1 (3): a turn's audio is at most min(MaxUtteranceSeconds, 12 s).
	Impl->MaxTurnSamples = FMath::Max<int64>(FSpeechVad::FrameSamples, (int64)FMath::RoundToDouble(FMath::Min((double)InSettings.MaxUtteranceSeconds, 12.0) * 16000.0));
	Impl->EngineModel = InEngineModel;
	if (InSettings.bPartials)
	{
		if (InEngineModel.IsValid())
		{
			const ISpeechEngine* Engine = InEngineModel->AsSpeechEngine();
			Impl->bPartialsEnabled = Engine != nullptr && Engine->SupportsPartials();
		}
		else
		{
			UE_LOG(LogModelPipelinesSpeech, Log, TEXT("USpeechStream: partial transcripts need the Phonon engine (ModelPipelines Generative); this stream delivers finals only"));
		}
	}
	Impl->TickerHandle = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateSP(Impl.ToSharedRef(), &FSpeechStreamImpl::Tick), 0.f);
}

void USpeechStream::PushAudio(TConstArrayView<float> Interleaved, int32 SampleRate, int32 NumChannels)
{
	if (Impl.IsValid())
	{
		Impl->Push(Interleaved, SampleRate, NumChannels);
	}
}

void USpeechStream::PushAudioArray(const TArray<float>& Interleaved, int32 SampleRate, int32 NumChannels)
{
	PushAudio(TConstArrayView<float>(Interleaved), SampleRate, NumChannels);
}

void USpeechStream::Stop()
{
	if (Impl.IsValid() && !Impl->bStopRequested)
	{
		Impl->bStopRequested = true;
		Impl->bStopped.store(true);
	}
}

bool USpeechStream::IsRunning() const
{
	return Impl.IsValid() && !Impl->bStopRequested && !Impl->bFinished;
}

bool USpeechStream::IsSpeaking() const
{
	return Impl.IsValid() && Impl->bSpeaking;
}

void USpeechStream::BeginDestroy()
{
	if (Impl.IsValid())
	{
		Impl->Shutdown();
	}
	Super::BeginDestroy();
}
