// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9R.3: USpeechStream (energy VAD, finals on both engines, partials on the Phonon engine). Written from docs/briefs/WP9R.3.md only.
// Layers: (1) FSpeechVad as a pure state machine, (2) streams on the fake Phonon engine (SpeechFakeEngine.h), (3) the tiny NNE
// pipeline (no partials + Info log), (4) the real stage-A pipeline on a golden recording.
// Text comparisons are CASE-SENSITIVE (FString == ignores case in Unreal). NaN/Inf through std::numeric_limits.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Containers/Ticker.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "Math/RandomStream.h"
#include "Misc/OutputDevice.h"
#include "Misc/OutputDeviceRedirector.h"
#include "Misc/ScopeLock.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineModelPackage.h"
#include "PipelineTypes.h"
#include "PipelineSpeechLog.h"
#include "SpeechPipelineLibrary.h"
#include "SpeechStream.h"
#include "SpeechToTextPipeline.h"
#include "SpeechTypes.h"
#include "SpeechVad.h"
#include "SpeechFakeEngine.h"
#include "SpeechStreamTestReceiver.h"
#include "SpeechTestData.h"

#include <limits>

using namespace UE::ModelPipelines::SpeechTests;

namespace SpeechStreamSpecPrivate
{
	// ---- ASSUMED FSpeechVad API (the brief names SpeechVad.h/.cpp but not its interface; the lead adapts this one block) ----
	//   UE::ModelPipelines::Speech::FSpeechVadSettings { float HangoverSeconds = 0.3f; float ThresholdDb = 12.f; float MaxUtteranceSeconds = 30.f; }
	//   UE::ModelPipelines::Speech::FSpeechVad(const FSpeechVadSettings&)
	//   void Process(TConstArrayView<float> Mono16k, TArray<FSpeechVadEvent>& OutEvents)   // any chunk size; appends events
	//   bool IsSpeaking() const;  float GetNoiseFloorDb() const;
	//   FSpeechVadEvent { ESpeechVadEventType Type (Started/Ended); int64 SampleIndex; }  stream clock in 16 kHz samples.
	//   Started.SampleIndex = first speech frame start - 0.3 s pre-roll (clamped at 0); Ended.SampleIndex = end of the
	//   last speech frame + hangover (the end of the audio that goes into the final).
	using UE::ModelPipelines::Speech::FSpeechVad;
	using UE::ModelPipelines::Speech::FSpeechVadSettings;
	using UE::ModelPipelines::Speech::FSpeechVadEvent;
	using UE::ModelPipelines::Speech::ESpeechVadEventType;

	constexpr int32 Frame = 320;       // 20 ms at 16 kHz
	constexpr int32 PreRoll = 4800;    // 0.3 s
	constexpr double Wait = 60.0;
	const FTimespan LatentLimit = FTimespan::FromSeconds(400);

	static void Silence(TArray<float>& A, int32 Frames) { A.AddZeroed(Frames * Frame); }

	/** Tone of Frames 20 ms frames, RMS dBFS = 20*log10(Amp/sqrt2). */
	static void Tone(TArray<float>& A, int32 Frames, double Amp, int32 Rate = 16000, double Freq = 440.0)
	{
		TArray<float> T;
		MakeToneSamples(T, Frames * Rate / 50, Rate, Freq, Amp);
		A.Append(T);
	}

	static double AmpForDb(double Db) { return FMath::Sqrt(2.0) * FMath::Pow(10.0, Db / 20.0); }

	static TArray<FSpeechVadEvent> RunVad(const TArray<float>& A, const FSpeechVadSettings& S, int32 Chunk = 0)
	{
		FSpeechVad Vad(S);
		TArray<FSpeechVadEvent> Ev;
		if (Chunk <= 0) { Vad.Process(A, Ev); return Ev; }
		for (int32 i = 0; i < A.Num(); i += Chunk)
		{
			Vad.Process(TConstArrayView<float>(A.GetData() + i, FMath::Min(Chunk, A.Num() - i)), Ev);
		}
		return Ev;
	}

	static int32 CountType(const TArray<FSpeechVadEvent>& Ev, ESpeechVadEventType T)
	{
		int32 N = 0;
		for (const FSpeechVadEvent& E : Ev) { N += (E.Type == T) ? 1 : 0; }
		return N;
	}

	static FString DescribeEvents(const TArray<FSpeechVadEvent>& Ev)
	{
		FString S;
		for (const FSpeechVadEvent& E : Ev) { S += FString::Printf(TEXT("%s@%lld "), E.Type == ESpeechVadEventType::Started ? TEXT("S") : TEXT("E"), (long long)E.SampleIndex); }
		return S.IsEmpty() ? TEXT("(none)") : S;
	}

	// ---- log capture (any verbosity, any thread) ----
	class FLogCapture : public FOutputDevice
	{
	public:
		FLogCapture() { GLog->AddOutputDevice(this); }
		virtual ~FLogCapture() override { GLog->RemoveOutputDevice(this); }
		virtual bool CanBeUsedOnAnyThread() const override { return true; }
		virtual void Serialize(const TCHAR* V, ELogVerbosity::Type Verbosity, const FName& Category) override
		{
			FScopeLock L(&Lock);
			if (Category == FName(TEXT("LogModelPipelinesSpeech")))
			{
				Lines.Add(FString(V));
				Verbosities.Add((uint8)Verbosity);
			}
		}
		int32 CountWarnings() const
		{
			FScopeLock L(&Lock);
			int32 N = 0;
			for (uint8 V : Verbosities) { N += (V == (uint8)ELogVerbosity::Warning) ? 1 : 0; }
			return N;
		}
		int32 CountContaining(const TCHAR* Needle) const
		{
			FScopeLock L(&Lock);
			int32 N = 0;
			for (const FString& S : Lines) { N += S.Contains(Needle, ESearchCase::IgnoreCase) ? 1 : 0; }
			return N;
		}
		mutable FCriticalSection Lock;
		TArray<FString> Lines;
		TArray<uint8> Verbosities;
	};

	// ---- waiting ----
	// Lead fix 2026-10-18 (WP9R.3 builder finding): a chain of WaitFor/Grace steps calls Done once, when its LAST step finished.
	// Each step counts itself in GPendingSteps while scheduled; a Then that schedules the next step keeps the count above zero.
	// (Before: every WaitFor called Done right after Then, ending the test while its checks were still scheduled.)
	static int32 GPendingSteps = 0;
	static void FinishStep(const FDoneDelegate& Done)
	{
		if (--GPendingSteps <= 0) { GPendingSteps = 0; Done.ExecuteIfBound(); }
	}

	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Pred, const FString& What, double Timeout, const FDoneDelegate& Done, TFunction<void()> Then)
	{
		const double Start = FPlatformTime::Seconds();
		++GPendingSteps;
		FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([T, Pred, What, Timeout, Done, Then, Start](float) -> bool
		{
			if (Pred())
			{
				Then();
				FinishStep(Done);
				return false;
			}
			if (FPlatformTime::Seconds() - Start > Timeout)
			{
				T->AddError(FString::Printf(TEXT("timed out after %.0f s waiting for: %s"), Timeout, *What));
				GPendingSteps = 0;
				Done.ExecuteIfBound();
				return false;
			}
			return true;
		}), 0.01f);
	}

	/** Runs Then after Seconds of ticking (events arriving in the meantime are recorded by the receivers). */
	static void Grace(double Seconds, const FDoneDelegate& Done, TFunction<void()> Then)
	{
		const double Start = FPlatformTime::Seconds();
		++GPendingSteps;
		FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Seconds, Done, Then, Start](float) -> bool
		{
			if (FPlatformTime::Seconds() - Start < Seconds) { return true; }
			Then();
			FinishStep(Done);
			return false;
		}), 0.01f);
	}

	/** Pushes Audio from the game thread, PerTick 20 ms chunks per 50 ms tick (4x real time at PerTick=8), then OnPushed. */
	static void PushPaced(USpeechStream* Stream, const TArray<float>& Audio, int32 Rate, int32 Channels, int32 PerTick, TFunction<void()> OnPushed)
	{
		const int32 ChunkFrames = Rate / 50;
		const TSharedRef<TArray<float>> A = MakeShared<TArray<float>>(Audio);
		const TSharedRef<int32> Pos = MakeShared<int32>(0);
		const TWeakObjectPtr<USpeechStream> Weak(Stream);
		FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Weak, A, Pos, Rate, Channels, PerTick, ChunkFrames, OnPushed](float) -> bool
		{
			USpeechStream* S = Weak.Get();
			for (int32 k = 0; k < PerTick && *Pos < A->Num(); ++k)
			{
				const int32 N = FMath::Min(ChunkFrames * Channels, A->Num() - *Pos);
				if (S) { S->PushAudio(TConstArrayView<float>(A->GetData() + *Pos, N), Rate, Channels); }
				*Pos += N;
			}
			if (*Pos >= A->Num()) { OnPushed(); return false; }
			return true;
		}), 0.05f);
	}

	/** NumThreads pool tasks push 20 ms chunks in strict order (a ticket makes chunk i wait for chunk i-1), chunk i on thread i % N. */
	struct FThreadedPush
	{
		TArray<float> Audio;
		int32 Rate = 16000;
		int32 Channels = 1;
		FThreadSafeCounter Ticket;
		FThreadSafeCounter Finished;
	};

	static void PushThreaded(USpeechStream* Stream, const TArray<float>& Audio, int32 Rate, int32 Channels, int32 NumThreads, const TSharedRef<FThreadedPush, ESPMode::ThreadSafe>& P)
	{
		P->Audio = Audio;
		P->Rate = Rate;
		P->Channels = Channels;
		const int32 PerChunk = (Rate / 50) * Channels;
		const int32 NumChunks = (Audio.Num() + PerChunk - 1) / PerChunk;
		for (int32 T = 0; T < NumThreads; ++T)
		{
			Async(EAsyncExecution::ThreadPool, [P, Stream, T, NumThreads, NumChunks, PerChunk]()
			{
				for (int32 i = T; i < NumChunks; i += NumThreads)
				{
					const double Start = FPlatformTime::Seconds();
					while (P->Ticket.GetValue() != i && FPlatformTime::Seconds() - Start < 30.0) { FPlatformProcess::Sleep(0.f); }
					const int32 From = i * PerChunk;
					Stream->PushAudio(TConstArrayView<float>(P->Audio.GetData() + From, FMath::Min(PerChunk, P->Audio.Num() - From)), P->Rate, P->Channels);
					P->Ticket.Increment();
				}
				P->Finished.Increment();
			});
		}
	}

	/** Lead-in silence, then NumBursts x (burst tone, gap silence). Frames of 20 ms at 16 kHz-equivalent timing; Rate scales the samples. */
	static void Script(TArray<float>& A, int32 Rate, int32 NumBursts, int32 BurstFrames, int32 GapFrames, double Amp = 0.2, int32 LeadFrames = 25)
	{
		A.Reset();
		TArray<float> Z;
		const int32 PerFrame = Rate / 50;
		A.AddZeroed(LeadFrames * PerFrame);
		for (int32 b = 0; b < NumBursts; ++b)
		{
			Tone(A, BurstFrames, Amp, Rate);
			A.AddZeroed(GapFrames * PerFrame);
		}
	}

	static void Interleave(const TArray<float>& Mono, int32 Channels, TArray<float>& Out)
	{
		Out.Reset();
		Out.Reserve(Mono.Num() * Channels);
		for (float V : Mono) { for (int32 c = 0; c < Channels; ++c) { Out.Add(V); } }
	}

	/** Grammar of rule 1 for NumUtterances utterances: Started < Partial* < Ended < Final per utterance, Stopped exactly once and last. */
	static void CheckGrammar(FAutomationTestBase* T, const USpeechStreamTestReceiver* R, int32 NumUtterances, bool bPartialsAllowed)
	{
		const TArray<FSpeechStreamTestRecord> St = R->Of(ESpeechStreamTestEvent::Started);
		const TArray<FSpeechStreamTestRecord> En = R->Of(ESpeechStreamTestEvent::Ended);
		const TArray<FSpeechStreamTestRecord> Fi = R->Of(ESpeechStreamTestEvent::Final);
		T->TestEqual(TEXT("OnSpeechStarted count"), St.Num(), NumUtterances);
		T->TestEqual(TEXT("OnSpeechEnded count"), En.Num(), NumUtterances);
		T->TestEqual(TEXT("OnFinal count (one per utterance)"), Fi.Num(), NumUtterances);
		T->TestEqual(TEXT("OnStopped exactly once"), R->Count(ESpeechStreamTestEvent::Stopped), 1);
		T->TestTrue(TEXT("every event on the game thread"), R->AllOnGameThread());
		if (R->Records.Num() > 0)
		{
			T->TestTrue(TEXT("OnStopped is the last event"), R->Records.Last().Type == ESpeechStreamTestEvent::Stopped);
		}
		if (St.Num() == NumUtterances && En.Num() == NumUtterances && Fi.Num() == NumUtterances)
		{
			for (int32 i = 0; i < NumUtterances; ++i)
			{
				T->TestTrue(*FString::Printf(TEXT("utt %d: Started before Ended"), i), St[i].Seq < En[i].Seq);
				T->TestTrue(*FString::Printf(TEXT("utt %d: Ended before Final"), i), En[i].Seq < Fi[i].Seq);
				if (i > 0) { T->TestTrue(*FString::Printf(TEXT("utt %d: starts after utt %d ended"), i, i - 1), St[i].Seq > En[i - 1].Seq); }
			}
			for (const FSpeechStreamTestRecord& P : R->Of(ESpeechStreamTestEvent::Partial))
			{
				bool bInside = false;
				for (int32 i = 0; i < NumUtterances; ++i) { bInside |= (P.Seq > St[i].Seq && P.Seq < En[i].Seq); }
				T->TestTrue(*FString::Printf(TEXT("partial (seq %d) lies between a Started and its Ended"), P.Seq), bInside);
			}
		}
		if (!bPartialsAllowed) { T->TestEqual(TEXT("no OnPartial"), R->Count(ESpeechStreamTestEvent::Partial), 0); }
	}

	static void Words(const FString& Text, TArray<FString>& Out)
	{
		TArray<FString> Raw;
		Text.ParseIntoArrayWS(Raw);
		for (FString W : Raw)
		{
			FString Clean;
			for (TCHAR C : W.ToLower()) { if (FChar::IsAlnum(C) || C == TEXT('\'')) { Clean.AppendChar(C); } }
			if (!Clean.IsEmpty()) { Out.Add(Clean); }
		}
	}

	static int32 EditDistance(const TArray<FString>& A, const TArray<FString>& B)
	{
		TArray<int32> Prev, Cur;
		for (int32 j = 0; j <= B.Num(); ++j) { Prev.Add(j); }
		Cur.SetNum(B.Num() + 1);
		for (int32 i = 1; i <= A.Num(); ++i)
		{
			Cur[0] = i;
			for (int32 j = 1; j <= B.Num(); ++j)
			{
				Cur[j] = FMath::Min3(Prev[j - 1] + (A[i - 1].Equals(B[j - 1], ESearchCase::CaseSensitive) ? 0 : 1), Prev[j] + 1, Cur[j - 1] + 1);
			}
			Prev = Cur;
		}
		return Prev[B.Num()];
	}
}

using namespace SpeechStreamSpecPrivate;

BEGIN_DEFINE_SPEC(FSpeechStreamSpec, "Pipelines.Speech.Stream",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	TSharedPtr<FFakeEngineState, ESPMode::ThreadSafe> State;
	TSharedPtr<FFakePhononBackend> Backend;
	bool bRegistered = false;
	TStrongObjectPtr<UPipelineModelPackage> Pkg;
	TStrongObjectPtr<USpeechToTextPipeline> Pipe;
	TStrongObjectPtr<USpeechStream> Stream;
	TStrongObjectPtr<USpeechStreamTestReceiver> Recv;
	TSharedPtr<FLogCapture> Log;

	/** StartStream on Pipe with the receiver bound. Null (and an error) when it fails. */
	USpeechStream* Open(const FSpeechStreamSettings& S)
	{
		if (!Pipe.IsValid()) { return nullptr; }
		FPipelineResult R;
		USpeechStream* St = Pipe->StartStream(S, R);
		if (!St)
		{
			AddError(FString::Printf(TEXT("StartStream failed: error %d '%s'"), (int32)R.Error, *R.Message));
			return nullptr;
		}
		Bind(St);
		return St;
	}
	void Bind(USpeechStream* St)
	{
		Stream.Reset(St);
		Recv.Reset(NewObject<USpeechStreamTestReceiver>(GetTransientPackage()));
		St->OnSpeechStarted.AddDynamic(Recv.Get(), &USpeechStreamTestReceiver::HandleStarted);
		St->OnPartial.AddDynamic(Recv.Get(), &USpeechStreamTestReceiver::HandlePartial);
		St->OnFinal.AddDynamic(Recv.Get(), &USpeechStreamTestReceiver::HandleFinal);
		St->OnSpeechEnded.AddDynamic(Recv.Get(), &USpeechStreamTestReceiver::HandleEnded);
		St->OnStopped.AddDynamic(Recv.Get(), &USpeechStreamTestReceiver::HandleStopped);
	}
	/**
	 * Pushes Audio (Threads > 0: from that many pool threads in strict chunk order; Threads == 0: from the game thread, PerTick
	 * 20 ms chunks per 50 ms tick), then Stop() (bStop), waits for OnStopped, a 0.4 s grace for stray events, then Check.
	 */
	void Drive(const TArray<float>& Audio, int32 Rate, int32 Ch, int32 Threads, int32 PerTick, bool bStop, const FDoneDelegate& Done, TFunction<void()> Check)
	{
		USpeechStream* St = Stream.Get();
		USpeechStreamTestReceiver* Rc = Recv.Get();
		auto AfterPush = [this, Rc, bStop, Done, Check]()
		{
			if (bStop && Stream.IsValid()) { Stream->Stop(); }
			WaitFor(this, [Rc, bStop]() { return !bStop || Rc->Count(ESpeechStreamTestEvent::Stopped) >= 1; }, TEXT("OnStopped"), Wait, Done, [this, Done, Check]()
			{
				Grace(0.4, Done, Check);
			});
		};
		if (Threads > 0)
		{
			const TSharedRef<FThreadedPush, ESPMode::ThreadSafe> P = MakeShared<FThreadedPush, ESPMode::ThreadSafe>();
			PushThreaded(St, Audio, Rate, Ch, Threads, P);
			WaitFor(this, [P, Threads]() { return P->Finished.GetValue() == Threads; }, TEXT("the push threads"), Wait, Done, AfterPush);
		}
		else
		{
			PushPaced(St, Audio, Rate, Ch, PerTick, AfterPush);
		}
	}
END_DEFINE_SPEC(FSpeechStreamSpec)

void FSpeechStreamSpec::Define()
{
	// =================================================================================================================
	// 1. VAD, pure
	// =================================================================================================================
	Describe("Vad", [this]()
	{
		It("StartIsPreRolledAndEndFollowsHangover", [this]()
		{
			// 1 s silence (50 frames), 0.5 s tone at -23 dBFS (25 frames), 1 s silence. Start = first speech frame (sample 16000)
			// minus the 0.3 s pre-roll = 11200 exactly. End = end of the last speech frame (24000) + 0.3 s hangover = 28800 (+-1 frame).
			// A VAD without pre-roll would start at 16000 (or 16960 with the 3-frame delay), one with no hangover would end at 24000.
			TArray<float> A; Silence(A, 50); Tone(A, 25, 0.1); Silence(A, 50);
			const TArray<FSpeechVadEvent> Ev = RunVad(A, FSpeechVadSettings());
			TestEqual(*FString::Printf(TEXT("2 events, got %s"), *DescribeEvents(Ev)), Ev.Num(), 2);
			if (Ev.Num() == 2)
			{
				TestTrue(TEXT("first is Started"), Ev[0].Type == ESpeechVadEventType::Started);
				TestEqual(TEXT("Started sample = 16000 - 4800 (pre-roll)"), (int32)Ev[0].SampleIndex, 11200);
				TestTrue(TEXT("second is Ended"), Ev[1].Type == ESpeechVadEventType::Ended);
				TestTrue(*FString::Printf(TEXT("Ended sample %lld within 1 frame of 28800"), (long long)Ev[1].SampleIndex), FMath::Abs((int32)Ev[1].SampleIndex - 28800) <= Frame);
			}
		});

		It("BurstAtStreamStartClampsPreRollAtZero", [this]()
		{
			// Speech from sample 0: the pre-roll would be negative. Started must be 0 (a bug would wrap to a huge value or -4800).
			TArray<float> A; Tone(A, 25, 0.1); Silence(A, 50);
			const TArray<FSpeechVadEvent> Ev = RunVad(A, FSpeechVadSettings());
			TestTrue(*FString::Printf(TEXT("first event Started@0, got %s"), *DescribeEvents(Ev)), Ev.Num() >= 1 && Ev[0].Type == ESpeechVadEventType::Started && Ev[0].SampleIndex == 0);
		});

		It("TwoSpeechFramesAreNotEnoughButThreeAre", [this]()
		{
			// Speech starts after 3 CONSECUTIVE speech frames. 2 frames (40 ms click) must give no event, 3 frames must give Started.
			// 2+gap+2 is also silent: the count of consecutive frames restarts.
			FSpeechVadSettings S;
			TArray<float> Two; Silence(Two, 50); Tone(Two, 2, 0.1); Silence(Two, 60);
			TestEqual(TEXT("2-frame click: no events"), RunVad(Two, S).Num(), 0);
			TArray<float> Split; Silence(Split, 50); Tone(Split, 2, 0.1); Silence(Split, 1); Tone(Split, 2, 0.1); Silence(Split, 60);
			TestEqual(TEXT("2+1 silent+2 frames: no events"), RunVad(Split, S).Num(), 0);
			TArray<float> Three; Silence(Three, 50); Tone(Three, 3, 0.1); Silence(Three, 60);
			const TArray<FSpeechVadEvent> Ev = RunVad(Three, S);
			TestEqual(*FString::Printf(TEXT("3-frame burst: Started+Ended, got %s"), *DescribeEvents(Ev)), Ev.Num(), 2);
			if (Ev.Num() >= 1) { TestEqual(TEXT("Started = 16000 - 4800"), (int32)Ev[0].SampleIndex, 11200); }
		});

		It("HangoverSettingMovesTheEnd", [this]()
		{
			// Same audio, hangover 0.2 vs 0.8: the End moves by 0.6 s = 9600 samples (+-1 frame). A fixed hangover fails.
			TArray<float> A; Silence(A, 25); Tone(A, 25, 0.1); Silence(A, 100);
			FSpeechVadSettings Short; Short.HangoverSeconds = 0.2f;
			FSpeechVadSettings Long; Long.HangoverSeconds = 0.8f;
			const TArray<FSpeechVadEvent> E1 = RunVad(A, Short);
			const TArray<FSpeechVadEvent> E2 = RunVad(A, Long);
			if (E1.Num() == 2 && E2.Num() == 2)
			{
				const int32 D = (int32)(E2[1].SampleIndex - E1[1].SampleIndex);
				TestTrue(*FString::Printf(TEXT("End moved by %d samples, expected 9600 +-320"), D), FMath::Abs(D - 9600) <= Frame);
			}
			else { AddError(FString::Printf(TEXT("expected 2 events each: %s | %s"), *DescribeEvents(E1), *DescribeEvents(E2))); }
		});

		It("ToneBelowMinus55DbfsIsNeverSpeech", [this]()
		{
			// -57 dBFS is above the initial floor (-70) + 12 dB = -58 but below the absolute -55 dBFS gate: no speech.
			// Without the gate this fires. A -50 dBFS tone (above both) is speech.
			TArray<float> Quiet; Tone(Quiet, 100, AmpForDb(-57.0)); Silence(Quiet, 20);
			TestEqual(TEXT("-57 dBFS for 2 s: no events"), RunVad(Quiet, FSpeechVadSettings()).Num(), 0);
			TArray<float> Louder; Silence(Louder, 25); Tone(Louder, 50, AmpForDb(-45.0)); Silence(Louder, 60);
			TestEqual(TEXT("-45 dBFS burst: detected"), CountType(RunVad(Louder, FSpeechVadSettings()), ESpeechVadEventType::Started), 1);
		});

		It("MaxUtteranceCutsAndANewOneStartsRightAway", [this]()
		{
			// 5 s of continuous tone, MaxUtteranceSeconds 2. The first utterance ends at ~2 s (32000), another starts at once, so
			// there are at least 2 Started and every Ended-Started span is at most 2 s (+ one frame, + the hangover is not added to a cut).
			FSpeechVadSettings S; S.MaxUtteranceSeconds = 2.f;
			TArray<float> A; Tone(A, 250, 0.1); Silence(A, 50);
			const TArray<FSpeechVadEvent> Ev = RunVad(A, S);
			TestTrue(*FString::Printf(TEXT(">= 2 Started and equal Ended count, got %s"), *DescribeEvents(Ev)),
				CountType(Ev, ESpeechVadEventType::Started) >= 2 && CountType(Ev, ESpeechVadEventType::Started) == CountType(Ev, ESpeechVadEventType::Ended));
			if (Ev.Num() >= 2)
			{
				TestTrue(*FString::Printf(TEXT("first End %lld within 2 frames of 32000"), (long long)Ev[1].SampleIndex), FMath::Abs((int32)Ev[1].SampleIndex - 32000) <= 2 * Frame);
			}
			for (int32 i = 0; i + 1 < Ev.Num(); i += 2)
			{
				TestTrue(*FString::Printf(TEXT("utterance %d is <= 2 s + 1 frame long"), i / 2), Ev[i + 1].SampleIndex - Ev[i].SampleIndex <= 32000 + Frame);
			}
		});

		It("ChunkSizeDoesNotChangeTheEvents", [this]()
		{
			// The VAD is a streaming state machine: feeding 1000-sample, 7-sample and 20 ms chunks must equal one big call.
			// A frame buffer that resets per Process call would shift or lose events.
			TArray<float> A; Silence(A, 30); Tone(A, 30, 0.1); Silence(A, 60); Tone(A, 20, 0.15); Silence(A, 60);
			const TArray<FSpeechVadEvent> Whole = RunVad(A, FSpeechVadSettings());
			for (int32 Chunk : { 1000, 7, 320, 4801 })
			{
				const TArray<FSpeechVadEvent> Ev = RunVad(A, FSpeechVadSettings(), Chunk);
				bool bSame = Ev.Num() == Whole.Num();
				for (int32 i = 0; bSame && i < Ev.Num(); ++i) { bSame = Ev[i].Type == Whole[i].Type && Ev[i].SampleIndex == Whole[i].SampleIndex; }
				TestTrue(*FString::Printf(TEXT("chunk %d: %s == whole %s"), Chunk, *DescribeEvents(Ev), *DescribeEvents(Whole)), bSame);
			}
			TestEqual(TEXT("two utterances in the sequence"), CountType(Whole, ESpeechVadEventType::Started), 2);
		});

		It("ConstantNoiseAtMinus40DbfsIsEventuallyTheNoiseFloor", [this]()
		{
			// -40 dBFS noise for 70 s. The floor rises 0.5 dB/s from -70, so the noise counts as speech for a while (a legal start,
			// cut by MaxUtterance), then stops being speech (> ~36 s) while the floor sits near the noise level. Afterwards a -23 dBFS
			// tone still starts an utterance, but the noise alone never does: no Started in the last 20 s of noise.
			FRandomStream Rnd(1234);
			TArray<float> A;
			const double Peak = FMath::Sqrt(3.0) * FMath::Pow(10.0, -40.0 / 20.0);   // uniform noise: RMS = peak / sqrt3
			for (int32 i = 0; i < 70 * 16000; ++i) { A.Add((float)((Rnd.FRand() * 2.0 - 1.0) * Peak)); }
			FSpeechVadSettings S;
			FSpeechVad Vad(S);
			TArray<FSpeechVadEvent> Ev;
			Vad.Process(A, Ev);
			int32 LateStarts = 0;
			for (const FSpeechVadEvent& E : Ev) { if (E.Type == ESpeechVadEventType::Started && E.SampleIndex > 50 * 16000) { ++LateStarts; } }
			TestEqual(*FString::Printf(TEXT("no Started after 50 s of constant noise: %s"), *DescribeEvents(Ev)), LateStarts, 0);
			TestFalse(TEXT("not speaking at the end of the noise"), Vad.IsSpeaking());
			TestTrue(*FString::Printf(TEXT("noise floor %.1f dB adapted up to within [-55, -40]"), Vad.GetNoiseFloorDb()),
				Vad.GetNoiseFloorDb() >= -55.f && Vad.GetNoiseFloorDb() <= -40.f);
			TArray<float> Burst; Tone(Burst, 25, 0.1); Silence(Burst, 40);
			TArray<FSpeechVadEvent> Ev2;
			Vad.Process(Burst, Ev2);
			TestEqual(*FString::Printf(TEXT("a -23 dBFS tone over the -40 dBFS noise floor is speech: %s"), *DescribeEvents(Ev2)), CountType(Ev2, ESpeechVadEventType::Started), 1);
		});

		It("SilenceKeepsTheFloorLow", [this]()
		{
			FSpeechVad Vad((FSpeechVadSettings()));
			TArray<float> A; Silence(A, 100);
			TArray<FSpeechVadEvent> Ev;
			Vad.Process(A, Ev);
			TestTrue(*FString::Printf(TEXT("floor %.1f dB <= -69 after 2 s of silence"), Vad.GetNoiseFloorDb()), Vad.GetNoiseFloorDb() <= -69.f);
			TestEqual(TEXT("no events"), Ev.Num(), 0);
			TArray<float> T; Tone(T, 10, 0.1);
			TArray<FSpeechVadEvent> Ev2;
			Vad.Process(T, Ev2);
			TestTrue(TEXT("IsSpeaking true inside a burst"), Vad.IsSpeaking());
		});
	});

	// =================================================================================================================
	// 2. Streams on the fake Phonon engine
	// =================================================================================================================
	Describe("FakePhonon", [this]()
	{
		BeforeEach([this]()
		{
			State = MakeShared<FFakeEngineState, ESPMode::ThreadSafe>();
			State->bStreaming = true;
			Backend = MakeShared<FFakePhononBackend>(State.ToSharedRef());
			FModelPipelinesCoreModule::Get().RegisterBackend(Backend.ToSharedRef());
			bRegistered = true;
			Log = MakeShared<FLogCapture>();
			Pkg.Reset(MakeFakePhononPackage());
			FPipelineResult R;
			Pipe.Reset(USpeechPipelineLibrary::CreateSpeechToTextPipeline(Pkg.Get(), R));
			if (!Pipe.IsValid()) { AddError(FString::Printf(TEXT("CreateSpeechToTextPipeline on the fake Phonon package failed: %s"), *R.Message)); }
		});

		AfterEach([this]()
		{
			Stream.Reset();
			Recv.Reset();
			Pipe.Reset();
			Pkg.Reset();
			Log.Reset();
			if (bRegistered && Backend.IsValid()) { FModelPipelinesCoreModule::Get().UnregisterBackend(Backend.ToSharedRef()); bRegistered = false; }
			Backend.Reset();
			State.Reset();
		});

		It("StartStreamOnAPipelineThatIsNotReadyFails", [this]()
		{
			// A bare pipeline object never initialised from a package: Null + failure result, no crash.
			USpeechToTextPipeline* Bare = NewObject<USpeechToTextPipeline>(GetTransientPackage());
			FPipelineResult R;
			USpeechStream* S = Bare->StartStream(FSpeechStreamSettings(), R);
			TestNull(TEXT("no stream"), S);
			TestFalse(TEXT("OutResult is a failure"), R.bSuccess);
			TestFalse(TEXT("with a message"), R.Message.IsEmpty());
		});

		It("OutOfRangeSettingsAreClampedNotRejected", [this]()
		{
			if (!Pipe.IsValid()) { return; }
			FSpeechStreamSettings S;
			S.EndOfSpeechHangoverSeconds = 0.f; S.PartialIntervalSeconds = 100.f; S.SpeechThresholdDb = -5.f; S.MaxUtteranceSeconds = 100000.f;
			USpeechStream* St = Open(S);
			if (!St) { return; }
			TestTrue(TEXT("running with clamped settings"), St->IsRunning());
			TestFalse(TEXT("not speaking before any audio"), St->IsSpeaking());
			St->Stop();
		});

		LatentIt("FourThreadsPushThreeBurstsEventOrderFinalsAndStopOnce", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 0.5 s lead, 3 x (1 s tone, 1.2 s gap) pushed from 4 threads faster than real time (strictly ordered by a ticket).
			// bPartials=false: no Partial call may reach the fake stream. Expect Started/Ended/Final x3, finals "utt 0..2" in order
			// (case-sensitive), OnStopped once and last, all on the game thread, ONE fake stream at 16 kHz, never touched from the
			// game thread or by two threads at once. A stream that dropped chunks racing from several threads loses a burst.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = false;
			if (!Open(S)) { Done.ExecuteIfBound(); return; }
			TArray<float> A; Script(A, 16000, 3, 50, 60);
			Drive(A, 16000, 1, 4, 0, true, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 3, false);
				const TArray<FSpeechStreamTestRecord> Fi = Recv->Of(ESpeechStreamTestEvent::Final);
				for (int32 i = 0; i < Fi.Num(); ++i)
				{
					TestTrue(*FString::Printf(TEXT("final %d text is utt %d, got %s"), i, i, *Fi[i].Transcript.Text), Fi[i].Transcript.Text.Equals(FString::Printf(TEXT("utt %d"), i), ESearchCase::CaseSensitive));
				}
				FScopeLock L(&State->Lock);
				TestEqual(TEXT("fake stream opened once"), State->OpenStreamCount, 1);
				TestEqual(TEXT("fake stream rate 16000"), State->LastStreamRate, 16000);
				TestEqual(TEXT("no Partial calls with bPartials=false"), State->PartialCalls, 0);
				TestEqual(TEXT("calls never overlap (one thread at a time per stream)"), State->MaxInFlight, 1);
				TestFalse(TEXT("engine stream never called on the game thread"), State->bFeedOnGameThread);
				TestEqual(TEXT("3 flushes with audio"), State->FlushesWithAudio, 3);
				TestFalse(TEXT("IsRunning false after OnStopped"), Stream->IsRunning());
			});
		});

		LatentIt("FinalAudioIsPreRollPlusBurstPlusHangover", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// One exactly 1.0 s tone burst. The audio flushed into the engine is pre-roll 0.3 + 1.0 + hangover 0.2 (default since 2026-10-18) = 24000 samples;
			// accept [19000, 29000]. Fewer means the pre-roll/onset was cut off, many more means silence was fed for the whole stream.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = false;
			if (!Open(S)) { Done.ExecuteIfBound(); return; }
			TArray<float> A; Script(A, 16000, 1, 50, 60);
			Drive(A, 16000, 1, 1, 0, true, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 1, false);
				FScopeLock L(&State->Lock);
				if (State->FlushPendingSamples.Num() == 1)
				{
					const int64 N = State->FlushPendingSamples[0];
					TestTrue(*FString::Printf(TEXT("flushed %lld samples, expected ~24000 in [19000, 29000]"), (long long)N), N >= 19000 && N <= 29000);
				}
				else { AddError(FString::Printf(TEXT("expected exactly 1 flush with audio, got %d"), State->FlushPendingSamples.Num())); }
			});
		});

		LatentIt("StereoAt44100IsResampledAndStillGivesTwoFinals", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Front end: MixToMono + Resample(44100 -> 16000). 2 bursts at 44.1 kHz stereo. The engine stream must run at 16 kHz and
			// get ~25600 samples per utterance; a front end that skipped the resample would hand it 2.75x as many.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = false;
			if (!Open(S)) { Done.ExecuteIfBound(); return; }
			TArray<float> Mono, Inter; Script(Mono, 44100, 2, 50, 60);
			Interleave(Mono, 2, Inter);
			Drive(Inter, 44100, 2, 1, 0, true, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 2, false);
				FScopeLock L(&State->Lock);
				TestEqual(TEXT("engine stream rate 16000"), State->LastStreamRate, 16000);
				if (State->FlushPendingSamples.Num() == 2)
				{
					TestTrue(*FString::Printf(TEXT("flushed %lld samples (16 kHz count ~25600)"), (long long)State->FlushPendingSamples[0]),
						State->FlushPendingSamples[0] >= 19000 && State->FlushPendingSamples[0] <= 29000);
				}
			});
		});

		LatentIt("RateChangeBetweenChunksIsAllowed", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// First burst pushed at 16 kHz, the second at 48 kHz mono on the same stream (resampler reset): both transcribed.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = false;
			USpeechStream* St = Open(S);
			if (!St) { Done.ExecuteIfBound(); return; }
			TArray<float> A, B; Script(A, 16000, 1, 50, 60); Script(B, 48000, 1, 50, 60, 0.2, 0);
			const TWeakObjectPtr<USpeechStream> Weak(St);
			PushPaced(St, A, 16000, 1, 8, [this, Weak, B, Done]()
			{
				PushPaced(Weak.Get(), B, 48000, 1, 8, [this, Done]()
				{
					Stream->Stop();
					WaitFor(this, [this]() { return Recv->Count(ESpeechStreamTestEvent::Stopped) >= 1; }, TEXT("OnStopped"), Wait, Done, [this, Done]()
					{
						Grace(0.4, Done, [this]() { CheckGrammar(this, Recv.Get(), 2, false); });
					});
				});
			});
		});

		LatentIt("PartialsComeEveryIntervalAndOnlyWhenTheTextChanged", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 3 s of speech at 4x real time, interval 0.5 s. The fake answers alpha for its first 3 partial calls and beta after.
			// Calls are made (2..8) and OnPartial fires only on change: never the same text twice in a row, and fewer broadcasts
			// than calls once there are more than 3 calls (a stream that broadcast every partial would repeat alpha).
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			State->PartialText = [](int32 N) { return N < 3 ? FString(TEXT("alpha")) : FString(TEXT("beta")); };
			FSpeechStreamSettings S; S.bPartials = true; S.PartialIntervalSeconds = 0.5f;
			if (!Open(S)) { Done.ExecuteIfBound(); return; }
			TArray<float> A; Script(A, 16000, 1, 150, 60);
			Drive(A, 16000, 1, 0, 8, true, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 1, true);
				int32 Calls = 0;
				{ FScopeLock L(&State->Lock); Calls = State->PartialCalls; }
				TestTrue(*FString::Printf(TEXT("partial calls %d in [2, 8] for 3 s of speech at 0.5 s"), Calls), Calls >= 2 && Calls <= 8);
				const TArray<FSpeechStreamTestRecord> P = Recv->Of(ESpeechStreamTestEvent::Partial);
				TestTrue(*FString::Printf(TEXT("at least one OnPartial (%d)"), P.Num()), P.Num() >= 1);
				if (Calls > 3) { TestTrue(*FString::Printf(TEXT("OnPartial %d < calls %d (unchanged texts are not broadcast)"), P.Num(), Calls), P.Num() < Calls); }
				for (int32 i = 0; i + 1 < P.Num(); ++i)
				{
					TestFalse(*FString::Printf(TEXT("partial %d repeats the previous text %s"), i + 1, *P[i].Transcript.Text), P[i].Transcript.Text.Equals(P[i + 1].Transcript.Text, ESearchCase::CaseSensitive));
				}
				if (P.Num() >= 1) { TestTrue(TEXT("first partial is alpha"), P[0].Transcript.Text.Equals(TEXT("alpha"), ESearchCase::CaseSensitive)); }
			});
		});

		LatentIt("SlowPartialsNeverRunTwoAtOnceOrQueueUp", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Partial takes 0.4 s, interval 0.1 s, 2.5 s of speech pushed in ~0.4 s. Never more than one Partial in flight and no
			// overlap with Feed/Flush; and the stream SKIPS partials while one is running (25 would be queued otherwise): <= 12 calls.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			State->PartialDelaySeconds = 0.4;
			FSpeechStreamSettings S; S.bPartials = true; S.PartialIntervalSeconds = 0.1f;
			if (!Open(S)) { Done.ExecuteIfBound(); return; }
			TArray<float> A; Script(A, 16000, 1, 125, 60);
			Drive(A, 16000, 1, 0, 25, true, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 1, true);
				FScopeLock L(&State->Lock);
				TestEqual(TEXT("max Partial calls in flight"), State->MaxPartialInFlight, 1);
				TestEqual(TEXT("max stream calls in flight"), State->MaxInFlight, 1);
				TestTrue(*FString::Printf(TEXT("partial calls %d in [1, 12]"), State->PartialCalls), State->PartialCalls >= 1 && State->PartialCalls <= 12);
			});
		});

		LatentIt("EngineWithoutPartialsGivesNoPartialsButFinals", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// SupportsPartials() false: bPartials=true must not call Partial nor broadcast OnPartial; the final still arrives.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			State->bSupportsPartials = false;
			FSpeechStreamSettings S; S.bPartials = true; S.PartialIntervalSeconds = 0.1f;
			if (!Open(S)) { Done.ExecuteIfBound(); return; }
			TArray<float> A; Script(A, 16000, 1, 100, 60);
			Drive(A, 16000, 1, 0, 8, true, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 1, false);
				FScopeLock L(&State->Lock);
				TestEqual(TEXT("Partial never called"), State->PartialCalls, 0);
			});
		});

		LatentIt("EmptyFinalsAreNotBroadcast", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// The engine answers Text "" for the utterance: no OnFinal (OnSpeechEnded still fires), OnStopped once.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			State->bEmptyFinalText = true;
			FSpeechStreamSettings S; S.bPartials = false;
			if (!Open(S)) { Done.ExecuteIfBound(); return; }
			TArray<float> A; Script(A, 16000, 1, 50, 60);
			Drive(A, 16000, 1, 0, 8, true, Done, [this]()
			{
				TestEqual(TEXT("OnSpeechEnded"), Recv->Count(ESpeechStreamTestEvent::Ended), 1);
				TestEqual(TEXT("OnFinal count (empty text not broadcast)"), Recv->Count(ESpeechStreamTestEvent::Final), 0);
				TestEqual(TEXT("OnStopped once"), Recv->Count(ESpeechStreamTestEvent::Stopped), 1);
			});
		});

		LatentIt("StopMidUtteranceDeliversTheFinalThenStoppedAndLaterAudioIsIgnored", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 2 s of tone, never silent. Wait until IsSpeaking, then Stop(): the cut utterance gets its final (utt 0), then
			// OnStopped(Ok) once. Stop ends the utterance immediately (no hangover wait): Stop -> Stopped is well under 5 s.
			// A burst pushed after Stop must produce no events at all.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = false;
			USpeechStream* St = Open(S);
			if (!St) { Done.ExecuteIfBound(); return; }
			TArray<float> A; Tone(A, 100, 0.2);
			PushPaced(St, A, 16000, 1, 8, []() {});
			WaitFor(this, [this]() { return Stream->IsSpeaking() && Recv->Count(ESpeechStreamTestEvent::Started) >= 1; }, TEXT("IsSpeaking / OnSpeechStarted"), Wait, Done, [this, Done]()
			{
				const double T0 = FPlatformTime::Seconds();
				Stream->Stop();
				WaitFor(this, [this]() { return Recv->Count(ESpeechStreamTestEvent::Stopped) >= 1; }, TEXT("OnStopped"), Wait, Done, [this, Done, T0]()
				{
					const double Dt = FPlatformTime::Seconds() - T0;
					TestTrue(*FString::Printf(TEXT("Stop -> OnStopped took %.2f s (< 5 s)"), Dt), Dt < 5.0);
					const int32 Before = Recv->Records.Num();
					TArray<float> B; Tone(B, 100, 0.2);
					Stream->PushAudio(B, 16000, 1);
					TestFalse(TEXT("IsRunning false after Stop"), Stream->IsRunning());
					Grace(0.8, Done, [this, Before]()
					{
						TestEqual(TEXT("no event for audio pushed after Stop"), Recv->Records.Num(), Before);
						TestEqual(TEXT("OnSpeechStarted"), Recv->Count(ESpeechStreamTestEvent::Started), 1);
						TestEqual(TEXT("OnSpeechEnded"), Recv->Count(ESpeechStreamTestEvent::Ended), 1);
						const TArray<FSpeechStreamTestRecord> Fi = Recv->Of(ESpeechStreamTestEvent::Final);
						TestEqual(TEXT("final for the cut utterance"), Fi.Num(), 1);
						if (Fi.Num() == 1) { TestTrue(TEXT("final text is utt 0"), Fi[0].Transcript.Text.Equals(TEXT("utt 0"), ESearchCase::CaseSensitive)); }
						const TArray<FSpeechStreamTestRecord> Sp = Recv->Of(ESpeechStreamTestEvent::Stopped);
						if (Sp.Num() == 1)
						{
							TestTrue(*FString::Printf(TEXT("OnStopped result is Ok (%s)"), *Sp[0].Result.Message), Sp[0].Result.bSuccess);
							if (Fi.Num() == 1) { TestTrue(TEXT("Final before Stopped"), Fi[0].Seq < Sp[0].Seq); }
						}
						TestTrue(TEXT("all on the game thread"), Recv->AllOnGameThread());
					});
				});
			});
		});

		LatentIt("StopWithoutAnyAudioStopsOnceWithNoFinal", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			if (!Open(FSpeechStreamSettings())) { Done.ExecuteIfBound(); return; }
			Stream->Stop();
			Stream->Stop();   // a second Stop is harmless
			WaitFor(this, [this]() { return Recv->Count(ESpeechStreamTestEvent::Stopped) >= 1; }, TEXT("OnStopped"), Wait, Done, [this, Done]()
			{
				Grace(0.4, Done, [this]()
				{
					TestEqual(TEXT("OnStopped exactly once even after two Stop calls"), Recv->Count(ESpeechStreamTestEvent::Stopped), 1);
					TestEqual(TEXT("no other events"), Recv->Records.Num(), 1);
				});
			});
		});

		LatentIt("BadChunksAreDroppedWithOneWarningPerKindAndTheStreamContinues", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Rate 0, 0 channels and non-finite samples (NaN, Inf), pushed twice each, then a normal burst. Each kind: dropped and
			// ONE Warning (7 bad pushes -> exactly 3 warnings from LogModelPipelinesSpeech). A stream that crashed, stalled or let
			// a NaN into the VAD would not deliver the later final.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = false;
			USpeechStream* St = Open(S);
			if (!St) { Done.ExecuteIfBound(); return; }
			TArray<float> Good; MakeToneSamples(Good, 1600, 16000, 440.0, 0.2);
			TArray<float> WithNan = Good; WithNan[10] = std::numeric_limits<float>::quiet_NaN();
			TArray<float> WithInf = Good; WithInf[5] = std::numeric_limits<float>::infinity();
			for (int32 Rep = 0; Rep < 2; ++Rep)
			{
				St->PushAudio(Good, 0, 1);
				St->PushAudio(Good, 16000, 0);
				St->PushAudio(WithNan, 16000, 1);
			}
			St->PushAudio(WithInf, 16000, 1);
			TArray<float> A; Script(A, 16000, 1, 50, 60);
			Drive(A, 16000, 1, 0, 8, true, Done, [this]()
			{
				TestEqual(TEXT("exactly one Warning per bad-chunk kind (3 kinds)"), Log->CountWarnings(), 3);
				CheckGrammar(this, Recv.Get(), 1, false);
				FScopeLock L(&State->Lock);
				TestEqual(TEXT("exactly one utterance reached the engine"), State->FlushesWithAudio, 1);
			});
		});

		LatentIt("BackPressureDropsOldestAudioWithAWarningAndStopStillCompletes", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// The engine is slow (Feed sleeps 0.3 s) and 40 s of continuous speech is pushed in 100 ms chunks as fast as possible.
			// Queued audio exceeds 10 s: oldest audio dropped with a Warning (>= 1 warning; timing dependent), the producer never
			// blocks (push loop < 5 s) and the stream still ends with OnStopped(Ok).
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			State->FeedDelaySeconds = 0.3;
			FSpeechStreamSettings S; S.bPartials = false; S.MaxUtteranceSeconds = 120.f;
			USpeechStream* St = Open(S);
			if (!St) { Done.ExecuteIfBound(); return; }
			// Lead fix 2026-10-18: a constant tone becomes the noise floor after ~3 s (150-frame min tracking), so the VAD ended the
			// utterance and the engine never queued 10 s; the test only passed when the front end lagged. Every 10th chunk is now
			// 0.1 s of silence: shorter than the hangover (one utterance continues), and it keeps the floor low, so the engine path
			// must queue > 10 s and drop deterministically.
			TArray<float> Chunk; MakeToneSamples(Chunk, 1600, 16000, 440.0, 0.2);
			TArray<float> Gap; Gap.SetNumZeroed(1600);
			const double T0 = FPlatformTime::Seconds();
			for (int32 i = 0; i < 400; ++i) { St->PushAudio(i % 10 == 9 ? Gap : Chunk, 16000, 1); }
			const double PushSeconds = FPlatformTime::Seconds() - T0;
			TestTrue(*FString::Printf(TEXT("producer did not block: 40 s of audio pushed in %.2f s"), PushSeconds), PushSeconds < 5.0);
			St->Stop();
			State->FeedDelaySeconds = 0.0;
			WaitFor(this, [this]() { return Recv->Count(ESpeechStreamTestEvent::Stopped) >= 1; }, TEXT("OnStopped"), 180.0, Done, [this]()
			{
				TestTrue(*FString::Printf(TEXT("back-pressure Warning logged (%d warnings)"), Log->CountWarnings()), Log->CountWarnings() >= 1);
				const TArray<FSpeechStreamTestRecord> Sp = Recv->Of(ESpeechStreamTestEvent::Stopped);
				TestEqual(TEXT("OnStopped once"), Sp.Num(), 1);
				if (Sp.Num() == 1) { TestTrue(*FString::Printf(TEXT("OnStopped Ok (%s)"), *Sp[0].Result.Message), Sp[0].Result.bSuccess); }
			});
		});

		LatentIt("DestroyingTheStreamMidUtteranceStopsAllEvents", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Push a long tone, wait for OnSpeechStarted, drop the only strong ref and GC. No crash, and the receiver sees no event
			// after the GC.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			State->FeedDelaySeconds = 0.05;
			FSpeechStreamSettings S; S.bPartials = true; S.PartialIntervalSeconds = 0.1f;
			USpeechStream* St = Open(S);
			if (!St) { Done.ExecuteIfBound(); return; }
			TArray<float> A; Tone(A, 250, 0.2);
			PushPaced(St, A, 16000, 1, 4, []() {});
			WaitFor(this, [this]() { return Recv->Count(ESpeechStreamTestEvent::Started) >= 1; }, TEXT("OnSpeechStarted"), Wait, Done, [this, Done]()
			{
				Stream.Reset();
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
				const int32 After = Recv->Records.Num();
				Grace(1.5, Done, [this, After]() { TestEqual(TEXT("no event after the stream was destroyed"), Recv->Records.Num(), After); });
			});
		});

		LatentIt("DestroyingThePipelineMidUtteranceIsSafe", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Drop pipeline and package refs while speaking, GC, then drop the stream and GC again: no crash, no events after the
			// stream died (the stream may keep the pipeline alive until then; both outcomes are legal).
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			State->FeedDelaySeconds = 0.05;
			FSpeechStreamSettings S; S.bPartials = true; S.PartialIntervalSeconds = 0.1f;
			USpeechStream* St = Open(S);
			if (!St) { Done.ExecuteIfBound(); return; }
			TArray<float> A; Tone(A, 250, 0.2);
			PushPaced(St, A, 16000, 1, 4, []() {});
			WaitFor(this, [this]() { return Recv->Count(ESpeechStreamTestEvent::Started) >= 1; }, TEXT("OnSpeechStarted"), Wait, Done, [this, Done]()
			{
				Pipe.Reset();
				Pkg.Reset();
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
				Grace(0.5, Done, [this, Done]()
				{
					Stream.Reset();
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					const int32 After = Recv->Records.Num();
					Grace(1.5, Done, [this, After]() { TestEqual(TEXT("no event after stream and pipeline died"), Recv->Records.Num(), After); });
				});
			});
		});
	});

	// =================================================================================================================
	// 3. NNE-type pipeline: no partials + Info log once
	// =================================================================================================================
	Describe("NnePipeline", [this]()
	{
		AfterEach([this]()
		{
			Stream.Reset();
			Recv.Reset();
			Pipe.Reset();
			Log.Reset();
		});

		LatentIt("NneStreamLogsThePhononHintOnceAndNeverBroadcastsPartials", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Tiny NNE package (toy sizes; the transcript is not checked). bPartials=true on an engine without partials: one Info
			// line "partial transcripts need the Phonon engine" per stream (2 bursts must not log it twice), no OnPartial.
			Log = MakeShared<FLogCapture>();
			FString Err;
			USpeechToTextPipeline* P = GetSharedTinySttPipeline(Err);
			if (!P) { AddError(Err); Done.ExecuteIfBound(); return; }
			Pipe.Reset(P);
			FSpeechStreamSettings S; S.bPartials = true; S.PartialIntervalSeconds = 0.1f;
			if (!Open(S)) { Done.ExecuteIfBound(); return; }
			TArray<float> A; Script(A, 16000, 2, 50, 60);
			Drive(A, 16000, 1, 0, 8, true, Done, [this]()
			{
				TestEqual(TEXT("no OnPartial on the NNE engine"), Recv->Count(ESpeechStreamTestEvent::Partial), 0);
				TestEqual(TEXT("OnSpeechStarted x2"), Recv->Count(ESpeechStreamTestEvent::Started), 2);
				TestEqual(TEXT("OnSpeechEnded x2"), Recv->Count(ESpeechStreamTestEvent::Ended), 2);
				TestEqual(TEXT("OnStopped once"), Recv->Count(ESpeechStreamTestEvent::Stopped), 1);
				TestEqual(TEXT("the Phonon hint is logged exactly once per stream"), Log->CountContaining(TEXT("partial transcripts need the Phonon engine")), 1);
			});
		});
	});

	// =================================================================================================================
	// 4. Real stage-A pipeline on a golden recording
	// =================================================================================================================
	Describe("RealAudio", [this]()
	{
		AfterEach([this]()
		{
			Stream.Reset();
			Recv.Reset();
			Pipe.Reset();
		});

		LatentIt("LongLibriSpeechAt4xRealTimeWerAtMost6PercentWithLatencyReport", FTimespan::FromSeconds(900), [this](const FDoneDelegate& Done)
		{
			// long_61_70968_45s (45 s of LibriSpeech) pushed in 20 ms pieces at 4x real time through the shared CPU Phonon pipeline
			// (NNE stage A). The endpointed finals joined must match the golden words with WER <= 6 % (spike measured 2.6-3 %).
			// Also reports the speech-end -> OnFinal latency per utterance (min / median / max).
			const FSttLongGoldens* G = nullptr;
			FString Err;
			if (!LoadSttLongGoldens(G, Err)) { AddError(Err); Done.ExecuteIfBound(); return; }
			const FSttLongRecording* Rec = G->Find(TEXT("long_61_70968_45s"));
			if (!Rec) { AddError(TEXT("long_61_70968_45s missing from long_goldens.json; run Tools/make_stt_long_goldens.py")); Done.ExecuteIfBound(); return; }
			TArray<float> Audio;
			if (!LoadLongAudio(*G, *Rec, Audio, Err)) { AddError(Err); Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetSharedCpuSttPipeline(Err);
			if (!P) { AddError(Err); Done.ExecuteIfBound(); return; }
			Pipe.Reset(P);
			if (!Open(FSpeechStreamSettings())) { Done.ExecuteIfBound(); return; }
			// 4x real time: 20 ms pieces, 8 per 50 ms tick.
			Drive(Audio, 16000, 1, 0, 8, true, Done, [this, Rec]()
			{
				const TArray<FSpeechStreamTestRecord> Fi = Recv->Of(ESpeechStreamTestEvent::Final);
				const TArray<FSpeechStreamTestRecord> En = Recv->Of(ESpeechStreamTestEvent::Ended);
				TestEqual(TEXT("OnStopped once"), Recv->Count(ESpeechStreamTestEvent::Stopped), 1);
				TestTrue(TEXT("every event on the game thread"), Recv->AllOnGameThread());
				TestTrue(*FString::Printf(TEXT("at least 2 finals for 45 s of speech (got %d)"), Fi.Num()), Fi.Num() >= 2);
				FString Joined;
				for (const FSpeechStreamTestRecord& F : Fi) { Joined += F.Transcript.Text + TEXT(" "); }
				TArray<FString> Got, Gold;
				Words(Joined, Got);
				for (const FSttLongWord& W : Rec->Words) { Words(W.Word, Gold); }
				const int32 Dist = EditDistance(Gold, Got);
				const double Wer = Gold.Num() > 0 ? (double)Dist / (double)Gold.Num() : 1.0;
				AddInfo(FString::Printf(TEXT("WP9R.3 real audio: %d finals, %d words vs golden %d, edit distance %d, WER %.2f %%"), Fi.Num(), Got.Num(), Gold.Num(), Dist, Wer * 100.0));
				TestTrue(*FString::Printf(TEXT("WER %.2f %% <= 6 %%"), Wer * 100.0), Wer <= 0.06);
				TArray<double> Lat;
				for (int32 i = 0; i < FMath::Min(En.Num(), Fi.Num()); ++i) { Lat.Add((Fi[i].TimeSeconds - En[i].TimeSeconds) * 1000.0); }
				Lat.Sort();
				if (Lat.Num() > 0)
				{
					AddInfo(FString::Printf(TEXT("WP9R.3 final latency speech end -> OnFinal over %d utterances: min %.0f ms, median %.0f ms, max %.0f ms"), Lat.Num(), Lat[0], Lat[Lat.Num() / 2], Lat.Last()));
					TestTrue(TEXT("every OnFinal arrives after its OnSpeechEnded"), Lat[0] >= 0.0);
				}
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
