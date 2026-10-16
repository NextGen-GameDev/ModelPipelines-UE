// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9S.5c2 / contracts C77 + C77a: calls up to 600 s are cut at quiet points into 25-35 s windows, each window is transcribed like a
// C75 call, and the results are stitched. Numbers are checked against <dev>/goldens/stt/long/long_goldens.json (Python reference,
// Tools/long_audio.py + ONNX Runtime on the same fp16w graphs), never against the implementation.
// Missing goldens or models FAIL these tests with the path and the generating command. Text compares are CASE-SENSITIVE.
// Memory: ONE shared CPU Phonon pipeline (SpeechTestData). The "destroyed mid-call" case is defined LAST: it releases that
// pipeline (the next user rebuilds it).

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "HAL/PlatformMemory.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "Math/RandomStream.h"
#include "UObject/StrongObjectPtr.h"
#include "PipelineSpeechDsp.h"
#include "PipelineTypes.h"
#include "SpeechToTextPipeline.h"
#include "SpeechTypes.h"
#include "SpeechTestData.h"

#include <limits>

using namespace UE::ModelPipelines::Speech;
using namespace UE::ModelPipelines::SpeechTests;

namespace SpeechLongAudioSpecPrivate
{
	// 309 s of audio takes about 25 s on CPU; six recordings plus model load fit comfortably. Every wait is bounded.
	const FTimespan LatentLimit = FTimespan::FromSeconds(1800);
	constexpr double CallTimeoutSeconds = 900.0;
	constexpr double WorkerTimeoutSeconds = 1500.0;
	constexpr double GraceSeconds = 0.5;      // after the last expected callback: a duplicate callback would show up here
	constexpr double TimeTolerance = 1e-4;
	constexpr int32 Rate = 16000;

	static const TCHAR* const LibriIds[6] = {
		TEXT("long_61_70968_45s"), TEXT("long_121_121726_60s"), TEXT("long_237_126133_90s"),
		TEXT("long_260_123286_120s"), TEXT("long_672_122797_180s"), TEXT("long_1188_133604_300s") };
	static const TCHAR* const AllIds[8] = {
		TEXT("long_61_70968_45s"), TEXT("long_121_121726_60s"), TEXT("long_237_126133_90s"),
		TEXT("long_260_123286_120s"), TEXT("long_672_122797_180s"), TEXT("long_1188_133604_300s"),
		TEXT("silence_60s"), TEXT("gap_70s") };

	static bool SameText(const FString& A, const FString& B) { return A.Equals(B, ESearchCase::CaseSensitive); }

	static FString FirstDiff(const FString& Expected, const FString& Got)
	{
		const int32 N = FMath::Min(Expected.Len(), Got.Len());
		int32 I = 0;
		while (I < N && Expected[I] == Got[I]) { ++I; }
		const int32 From = FMath::Max(0, I - 12);
		return FString::Printf(TEXT("at char %d (expected len %d, got len %d): expected '%s' got '%s'"), I, Expected.Len(), Got.Len(), *Expected.Mid(From, 32), *Got.Mid(From, 32));
	}

	static bool LoadLongGoldens(FAutomationTestBase& T, const FSttLongGoldens*& OutG)
	{
		FString Err;
		if (!LoadSttLongGoldens(OutG, Err)) { T.AddError(Err); return false; }
		return true;
	}

	static bool LoadRecording(FAutomationTestBase& T, const FString& Id, const FSttLongGoldens*& OutG, const FSttLongRecording*& OutRec, TArray<float>& OutAudio)
	{
		if (!LoadLongGoldens(T, OutG)) { return false; }
		OutRec = OutG->Find(Id);
		if (!OutRec) { T.AddError(FString::Printf(TEXT("golden recording '%s' missing from long_goldens.json"), *Id)); return false; }
		FString Err;
		if (!LoadLongAudio(*OutG, *OutRec, OutAudio, Err)) { T.AddError(Err); return false; }
		return true;
	}

	static USpeechToTextPipeline* GetPipeline(FAutomationTestBase& T)
	{
		FString Err;
		USpeechToTextPipeline* P = GetSharedCpuSttPipeline(Err);
		if (!P) { T.AddError(Err); }
		return P;
	}

	/** Words equal text (case-sensitive) and times within 1e-4 s of (Offset + golden time). Empty OutWhy when equal. */
	static bool WordsMatch(const FSpeechTranscript& Tr, const TArray<FSttLongWord>& Gold, double Offset, FString& OutWhy, double& OutWorstDt)
	{
		OutWorstDt = 0.0;
		if (Tr.Words.Num() != Gold.Num())
		{
			OutWhy = FString::Printf(TEXT("word count %d vs golden %d"), Tr.Words.Num(), Gold.Num());
			return false;
		}
		for (int32 i = 0; i < Gold.Num(); ++i)
		{
			if (!SameText(Tr.Words[i].Word, Gold[i].Word))
			{
				OutWhy = FString::Printf(TEXT("word %d is '%s', golden '%s'"), i, *Tr.Words[i].Word, *Gold[i].Word);
				return false;
			}
			const double DS = FMath::Abs((double)Tr.Words[i].StartSeconds - (Gold[i].Start + Offset));
			const double DE = FMath::Abs((double)Tr.Words[i].EndSeconds - (Gold[i].End + Offset));
			OutWorstDt = FMath::Max(OutWorstDt, FMath::Max(DS, DE));
			if (DS > TimeTolerance || DE > TimeTolerance)
			{
				OutWhy = FString::Printf(TEXT("word %d '%s' times %.5f..%.5f, golden %.5f..%.5f (worst dt so far %.6f)"), i, *Tr.Words[i].Word,
					Tr.Words[i].StartSeconds, Tr.Words[i].EndSeconds, Gold[i].Start + Offset, Gold[i].End + Offset, OutWorstDt);
				return false;
			}
		}
		return true;
	}

	static int32 WordEditDistance(const TArray<FString>& A, const TArray<FString>& B)
	{
		TArray<int32> Prev, Cur;
		for (int32 j = 0; j <= B.Num(); ++j) { Prev.Add(j); }
		Cur.SetNum(B.Num() + 1);
		for (int32 i = 1; i <= A.Num(); ++i)
		{
			Cur[0] = i;
			for (int32 j = 1; j <= B.Num(); ++j)
			{
				const int32 Sub = Prev[j - 1] + (SameText(A[i - 1], B[j - 1]) ? 0 : 1);
				Cur[j] = FMath::Min3(Sub, Prev[j] + 1, Cur[j - 1] + 1);
			}
			Prev = Cur;
		}
		return Prev[B.Num()];
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Latent plumbing (callbacks and workers touch only TSharedRef state, so a timed-out test cannot leave dangling captures)

	struct FCall
	{
		FThreadSafeCounter Count;
		bool bInCall = false;
		bool bAnyInCall = false;
		bool bAllOnGameThread = true;
		double IssuedSeconds = 0.0;
		double FirstSeenSeconds = 0.0;
		FPipelineResult Result;
		FSpeechTranscript Transcript;
	};

	static void Issue(USpeechToTextPipeline* P, TArray<float> Samples, int32 SampleRate, int32 Channels, const TSharedRef<FCall>& C)
	{
		C->bInCall = true;
		C->IssuedSeconds = FPlatformTime::Seconds();
		P->Transcribe(MoveTemp(Samples), SampleRate, Channels, [C](const FPipelineResult& R, const FSpeechTranscript& Tr)
		{
			if (!IsInGameThread()) { C->bAllOnGameThread = false; }
			if (C->bInCall) { C->bAnyInCall = true; }
			if (C->Count.Increment() == 1)
			{
				C->Result = R;
				C->Transcript = Tr;
				C->FirstSeenSeconds = FPlatformTime::Seconds();
			}
		});
		C->bInCall = false;
	}

	/** Ticker wait: all calls fired once (+ grace for a duplicate); then once-only / game-thread / not-in-call asserts and Check. */
	static void WaitCalls(FAutomationTestBase* T, TArray<TSharedRef<FCall>> Calls, const FDoneDelegate& Done, TFunction<void()> Check,
		double TimeoutSeconds = CallTimeoutSeconds)
	{
		const double Start = FPlatformTime::Seconds();
		const TSharedRef<double> AllSeen = MakeShared<double>(0.0);
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechLongWait"), 0.02f, [T, Calls, Start, AllSeen, DoneCopy, Check, TimeoutSeconds](float)
		{
			const double Now = FPlatformTime::Seconds();
			int32 Arrived = 0;
			for (const TSharedRef<FCall>& C : Calls) { Arrived += C->Count.GetValue() > 0 ? 1 : 0; }
			if (Arrived == Calls.Num() && *AllSeen == 0.0) { *AllSeen = Now; }
			if ((*AllSeen > 0.0 && (Now - *AllSeen) > GraceSeconds) || (Now - Start) > TimeoutSeconds)
			{
				if (Arrived != Calls.Num())
				{
					T->AddError(FString::Printf(TEXT("only %d of %d callbacks fired within %.0f s (callback never fired / hang?)"), Arrived, Calls.Num(), TimeoutSeconds));
				}
				else
				{
					for (int32 i = 0; i < Calls.Num(); ++i)
					{
						T->TestEqual(*FString::Printf(TEXT("call %d: callback fired exactly once"), i), Calls[i]->Count.GetValue(), 1);
						T->TestTrue(*FString::Printf(TEXT("call %d: callback ran on the game thread"), i), Calls[i]->bAllOnGameThread);
						T->TestFalse(*FString::Printf(TEXT("call %d: callback must not run inside Transcribe"), i), Calls[i]->bAnyInCall);
					}
					if (Check) { Check(); }
				}
				DoneCopy.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	static void RunWorker(FAutomationTestBase* T, const FDoneDelegate& Done, TFunction<void()> Work, TFunction<void()> Then)
	{
		const TSharedRef<TFuture<void>> F = MakeShared<TFuture<void>>(Async(EAsyncExecution::Thread, MoveTemp(Work)));
		const double Start = FPlatformTime::Seconds();
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechLongWorker"), 0.02f, [T, F, Start, DoneCopy, Then](float)
		{
			if (F->IsReady())
			{
				if (Then) { Then(); } else { DoneCopy.ExecuteIfBound(); }
				return false;
			}
			if ((FPlatformTime::Seconds() - Start) > WorkerTimeoutSeconds)
			{
				T->AddError(FString::Printf(TEXT("worker thread did not finish within %.0f s (hang?)"), WorkerTimeoutSeconds));
				DoneCopy.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	/** Shared checks for a stitched transcript against a golden recording. */
	static void CheckAgainstGolden(FAutomationTestBase* T, const FString& Label, const FPipelineResult& R, const FSpeechTranscript& Tr, const FSttLongRecording& Rec)
	{
		T->TestTrue(*FString::Printf(TEXT("%s: success (%s)"), *Label, *R.Message), R.bSuccess);
		T->TestTrue(*FString::Printf(TEXT("%s: Text equal to golden (case-sensitive): %s"), *Label, SameText(Tr.Text, Rec.Text) ? TEXT("equal") : *FirstDiff(Rec.Text, Tr.Text)), SameText(Tr.Text, Rec.Text));
		FString Why;
		double WorstDt = 0.0;
		const bool bWords = WordsMatch(Tr, Rec.Words, 0.0, Why, WorstDt);
		T->TestTrue(*FString::Printf(TEXT("%s: Words equal in text and times within 1e-4 s (%s; worst dt %.7f s)"), *Label, *Why, WorstDt), bWords);
		T->TestTrue(*FString::Printf(TEXT("%s: AudioSeconds %.4f == %.4f"), *Label, Tr.AudioSeconds, (double)Rec.NumSamples / Rate),
			FMath::Abs((double)Tr.AudioSeconds - (double)Rec.NumSamples / Rate) < 1e-3);
		// Text must be the trimmed window texts joined by single spaces: no leading/trailing blank, no doubled space.
		T->TestTrue(*FString::Printf(TEXT("%s: Text has no leading/trailing space and no double space"), *Label),
			Tr.Text == Tr.Text.TrimStartAndEnd() && !Tr.Text.Contains(TEXT("  ")));
	}

	/** One async Transcribe of a golden recording, parity against the golden. */
	static void ScenarioRecording(FAutomationTestBase* T, const FString& Id, const FDoneDelegate& Done)
	{
		const FSttLongGoldens* G = nullptr;
		const FSttLongRecording* Rec = nullptr;
		TArray<float> Audio;
		if (!LoadRecording(*T, Id, G, Rec, Audio)) { Done.ExecuteIfBound(); return; }
		USpeechToTextPipeline* P = GetPipeline(*T);
		if (!P) { Done.ExecuteIfBound(); return; }
		const TSharedRef<FCall> C = MakeShared<FCall>();
		Issue(P, MoveTemp(Audio), Rate, 1, C);
		T->TestEqual(TEXT("nothing delivered inside the call"), C->Count.GetValue(), 0);
		WaitCalls(T, { C }, Done, [T, C, Rec, Id]()
		{
			CheckAgainstGolden(T, Id, C->Result, C->Transcript, *Rec);
			const double Wall = C->FirstSeenSeconds - C->IssuedSeconds;
			const double AudioS = (double)Rec->NumSamples / Rate;
			T->AddInfo(FString::Printf(TEXT("%s: %d golden windows, audio %.1f s, wall %.2f s, RTF %.3f, InferenceMs %.0f, PeakUsedPhysical %.0f MB"),
				*Id, Rec->Windows.Num(), AudioS, Wall, Wall / FMath::Max(AudioS, 1e-9), C->Transcript.InferenceMs,
				(double)FPlatformMemory::GetStats().PeakUsedPhysical / (1024.0 * 1024.0)));
			if (Rec->Kind == TEXT("librispeech"))
			{
				T->TestTrue(*FString::Printf(TEXT("%s: InferenceMs %.1f > 0 (windows really ran)"), *Id, C->Transcript.InferenceMs), C->Transcript.InferenceMs > 0.f);
				T->TestTrue(TEXT("Text not empty"), !C->Transcript.Text.IsEmpty());
			}
		});
	}

	static TArray<float> MakeZeros(int32 N) { TArray<float> A; A.Init(0.f, N); return A; }

	/** Plan a signal and check the invariants every plan must satisfy: contiguous cover, lengths in [25,35] s (last >= 5 s). */
	static void CheckPlanInvariants(FAutomationTestBase* T, const TCHAR* Label, const TArray<FSpeechWindow>& W, int32 N, int32 SampleRate)
	{
		if (W.Num() == 0) { T->AddError(FString::Printf(TEXT("%s: no windows"), Label)); return; }
		T->TestEqual(*FString::Printf(TEXT("%s: first window starts at 0"), Label), W[0].Start, 0);
		T->TestEqual(*FString::Printf(TEXT("%s: last window ends at N"), Label), W.Last().End, N);
		for (int32 i = 0; i < W.Num(); ++i)
		{
			if (i > 0) { T->TestEqual(*FString::Printf(TEXT("%s: window %d starts where %d ends"), Label, i, i - 1), W[i].Start, W[i - 1].End); }
			const int32 Len = W[i].End - W[i].Start;
			if (i + 1 < W.Num())
			{
				T->TestTrue(*FString::Printf(TEXT("%s: window %d length %d in [25 s, 35 s]"), Label, i, Len), Len >= 25 * SampleRate && Len <= 35 * SampleRate);
			}
			else
			{
				T->TestTrue(*FString::Printf(TEXT("%s: last window length %d in [5 s, 35 s]"), Label, Len), Len >= 5 * SampleRate && Len <= 35 * SampleRate);
			}
		}
	}
}

BEGIN_DEFINE_SPEC(FSpeechLongAudioSpec, "Pipelines.Speech.LongAudio",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
END_DEFINE_SPEC(FSpeechLongAudioSpec)

void FSpeechLongAudioSpec::Define()
{
	using namespace SpeechLongAudioSpecPrivate;

	// ----------------------------------------------------------------------------------------------------------------
	Describe("PlanUnit", [this]()
	{
		It("EmptyInputGivesZeroWindows", [this]()
		{
			TArray<FSpeechWindow> W;
			W.Add(FSpeechWindow());   // stale content must be cleared
			PlanSpeechWindows(TConstArrayView<float>(), Rate, W);
			TestEqual(TEXT("windows for N = 0"), W.Num(), 0);
		});

		It("ExactlyMaxIsOneWindow", [this]()
		{
			// Loop condition is N - start > MAX: 35.0 s exactly is one window [0, N) (an off-by-one cuts it).
			const TArray<float> Z = MakeZeros(35 * Rate);
			TArray<FSpeechWindow> W;
			PlanSpeechWindows(Z, Rate, W);
			TestEqual(TEXT("window count"), W.Num(), 1);
			if (W.Num() == 1) { TestEqual(TEXT("Start"), W[0].Start, 0); TestEqual(TEXT("End"), W[0].End, 35 * Rate); }
		});

		It("OneSampleOverMaxCutsAt25sOnZeros", [this]()
		{
			// N = 35 s + 1: hi = min(35 s, N - 5 s) = 30 s + 1, lo = 25 s. All-zero energy ties everywhere, so the smallest i wins:
			// the cut is exactly 25 s (400000), inside [25 s, 30 s + 1]. A last-tie or a cut at hi would show here.
			const TArray<float> Z = MakeZeros(35 * Rate + 1);
			TArray<FSpeechWindow> W;
			PlanSpeechWindows(Z, Rate, W);
			TestEqual(TEXT("window count"), W.Num(), 2);
			if (W.Num() == 2)
			{
				TestTrue(*FString::Printf(TEXT("cut %d within [400000, 480001]"), W[0].End), W[0].End >= 25 * Rate && W[0].End <= 30 * Rate + 1);
				TestEqual(TEXT("earliest tie: cut = 25 s"), W[0].End, 25 * Rate);
				TestEqual(TEXT("second window starts at the cut"), W[1].Start, W[0].End);
				TestEqual(TEXT("second window ends at N"), W[1].End, 35 * Rate + 1);
			}
		});

		It("AllZero600sCutsEvery25s", [this]()
		{
			// Earliest tie each time: window k starts at k * 25 s, 23 cuts, then the 25 s remainder (24 windows).
			const TArray<float> Z = MakeZeros(600 * Rate);
			TArray<FSpeechWindow> W;
			PlanSpeechWindows(Z, Rate, W);
			TestEqual(TEXT("window count"), W.Num(), 24);
			for (int32 i = 0; i + 1 < W.Num(); ++i)
			{
				TestEqual(*FString::Printf(TEXT("window %d End = start + 25 s"), i), W[i].End, W[i].Start + 25 * Rate);
			}
			CheckPlanInvariants(this, TEXT("zeros 600 s"), W, 600 * Rate, Rate);
		});

		It("CutLandsOnTheFirstSilentCentreInsideAGap", [this]()
		{
			// 70 s of noise (amplitude 0.3) with 1.0 s of exact zeros at 31.0..32.0 s. For the first cut, lo = 25 s, hi = min(35 s, 70 - 5) = 35 s.
			// S(i) = 0 only when all 20 frames of the 200 ms window lie in the gap, i.e. H*(i-10) >= 496000: the smallest such i is 3110, cut = 497600.
			// A cut at a mere local minimum, or the end of the zero run (the largest tie), would differ. Remaining 38.9 s > 35 s, so a 3rd window exists.
			FRandomStream Rnd(1234);
			TArray<float> X;
			X.SetNumUninitialized(70 * Rate);
			for (int32 i = 0; i < X.Num(); ++i) { X[i] = 0.3f * (Rnd.FRand() * 2.f - 1.f) + (i % 2 ? 1e-3f : -1e-3f); }
			for (int32 i = 31 * Rate; i < 32 * Rate; ++i) { X[i] = 0.f; }
			TArray<FSpeechWindow> W;
			PlanSpeechWindows(X, Rate, W);
			TestEqual(TEXT("window count"), W.Num(), 3);
			if (W.Num() >= 1) { TestEqual(TEXT("first cut"), W[0].End, 497600); }
			CheckPlanInvariants(this, TEXT("noise+gap 70 s"), W, 70 * Rate, Rate);
		});

		It("LimitsScaleWithTheSampleRate", [this]()
		{
			// SampleRate 8000: MAX = 280000 samples. 280000 is one window; 280001 two, earliest-tie cut at 25 s * 8000 = 200000.
			TArray<FSpeechWindow> W;
			PlanSpeechWindows(MakeZeros(280000), 8000, W);
			TestEqual(TEXT("N = 35 s at 8 kHz: windows"), W.Num(), 1);
			PlanSpeechWindows(MakeZeros(280001), 8000, W);
			TestEqual(TEXT("N = 35 s + 1 at 8 kHz: windows"), W.Num(), 2);
			if (W.Num() == 2) { TestEqual(TEXT("cut at 25 s of 8 kHz"), W[0].End, 200000); }
		});

		It("ShortInputIsOneWindow", [this]()
		{
			TArray<FSpeechWindow> W;
			PlanSpeechWindows(MakeZeros(1), Rate, W);
			TestEqual(TEXT("N = 1: windows"), W.Num(), 1);
			if (W.Num() == 1) { TestEqual(TEXT("End"), W[0].End, 1); }
		});
	});

	// ----------------------------------------------------------------------------------------------------------------
	Describe("PlanGolden", [this]()
	{
		for (int32 k = 0; k < 8; ++k)
		{
			const FString Id = AllIds[k];
			It(*(FString(TEXT("WindowsEqualGolden_")) + Id), [this, Id]()
			{
				// Integer Start/End must equal the Python reference (float64 sequential prefix sums, ties to the smallest i). A float32
				// prefix sum, a parallel reduction or a different centre/half-width moves cuts on real speech.
				const FSttLongGoldens* G = nullptr;
				const FSttLongRecording* Rec = nullptr;
				TArray<float> Audio;
				if (!LoadRecording(*this, Id, G, Rec, Audio)) { return; }
				TArray<FSpeechWindow> W;
				PlanSpeechWindows(Audio, Rate, W);
				TestEqual(*FString::Printf(TEXT("%s: window count"), *Id), W.Num(), Rec->Windows.Num());
				int32 Mismatches = 0;
				for (int32 i = 0; i < FMath::Min(W.Num(), Rec->Windows.Num()); ++i)
				{
					if (W[i].Start != Rec->Windows[i].Start || W[i].End != Rec->Windows[i].End)
					{
						++Mismatches;
						AddError(FString::Printf(TEXT("%s window %d: ours [%d, %d), golden [%d, %d)"), *Id, i, W[i].Start, W[i].End, Rec->Windows[i].Start, Rec->Windows[i].End));
					}
				}
				TestEqual(*FString::Printf(TEXT("%s: mismatching windows"), *Id), Mismatches, 0);
				CheckPlanInvariants(this, *Id, W, Audio.Num(), Rate);
			});
		}
	});

	// ----------------------------------------------------------------------------------------------------------------
	Describe("Transcribe", [this]()
	{
		for (int32 k = 0; k < 6; ++k)
		{
			const FString Id = LibriIds[k];
			LatentIt(*(FString(TEXT("MatchesGolden_")) + Id), LatentLimit, [this, Id](const FDoneDelegate& Done)
			{
				// Text (case-sensitive) and words (text, times within 1e-4 s) equal the golden stitch of per-window ONNX decodes.
				// A stitch that forgot the window offset in word times, joined without a space, kept padding from the previous
				// window's features, or normalised features over the whole clip would break equality here.
				ScenarioRecording(this, Id, Done);
			});
		}

		LatentIt("Silence60sIsEmptySuccessWithNoInference", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Both windows are gated: empty text, no words, InferenceMs exactly 0, AudioSeconds 60 (the input duration, not the processed part).
			const FSttLongGoldens* G = nullptr; const FSttLongRecording* Rec = nullptr; TArray<float> Audio;
			if (!LoadRecording(*this, TEXT("silence_60s"), G, Rec, Audio)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(P, MoveTemp(Audio), Rate, 1, C);
			WaitCalls(this, { C }, Done, [this, C]()
			{
				TestTrue(*FString::Printf(TEXT("success (%s)"), *C->Result.Message), C->Result.bSuccess);
				TestTrue(TEXT("Text empty"), C->Transcript.Text.IsEmpty());
				TestEqual(TEXT("Words"), C->Transcript.Words.Num(), 0);
				TestEqual(TEXT("InferenceMs (no window ran)"), C->Transcript.InferenceMs, 0.f);
				TestTrue(*FString::Printf(TEXT("AudioSeconds %.3f == 60"), C->Transcript.AudioSeconds), FMath::Abs(C->Transcript.AudioSeconds - 60.f) < 1e-3f);
			});
		});

		LatentIt("Gap70sMatchesGolden", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Speech, a gap of silence, speech: the window containing mostly silence is decoded on its own (own RMS gate and
			// normalisation); Text must equal the golden and carry no double space from an empty window.
			ScenarioRecording(this, TEXT("gap_70s"), Done);
		});

		LatentIt("BlockingEqualsGoldenOn60sRecording", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// TranscribeBlocking handles windows too (worker thread; never the game thread).
			const FSttLongGoldens* G = nullptr; const FSttLongRecording* Rec = nullptr; TArray<float> Audio;
			if (!LoadRecording(*this, TEXT("long_121_121726_60s"), G, Rec, Audio)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			struct FBlock { TArray<float> Audio; FPipelineResult R; FSpeechTranscript T; };
			const TSharedRef<FBlock> S = MakeShared<FBlock>();
			S->Audio = MoveTemp(Audio);
			RunWorker(this, Done, [S, P]() { S->R = P->TranscribeBlocking(S->Audio, Rate, 1, S->T); },
				[this, S, Rec, Done]()
				{
					CheckAgainstGolden(this, TEXT("blocking long_121_121726_60s"), S->R, S->T, *Rec);
					Done.ExecuteIfBound();
				});
		});

		LatentIt("EachGoldenWindowDecodesAloneLikeAC75Call", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Per-window token decode: slice each golden window of gap_70s and long_61_70968_45s out of the audio and run the
			// plain <= 35 s path on it (no windowing involved: each slice is <= 35 s). Its words equal the golden window words
			// (window-local times, 1e-4 s) and its text equals the golden window text. Pins "each window is exactly a C75 call".
			const FSttLongGoldens* G = nullptr;
			if (!LoadLongGoldens(*this, G)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			struct FSlice { FString Label; TArray<float> Audio; const FSttLongWindow* Win = nullptr; FPipelineResult R; FSpeechTranscript T; };
			struct FSlices { TArray<FSlice> Items; };
			const TSharedRef<FSlices> S = MakeShared<FSlices>();
			for (const TCHAR* Id : { TEXT("long_61_70968_45s"), TEXT("gap_70s") })
			{
				const FSttLongRecording* Rec = G->Find(Id);
				TArray<float> Audio;
				FString Err;
				if (!Rec || !LoadLongAudio(*G, *Rec, Audio, Err)) { AddError(Rec ? Err : FString::Printf(TEXT("golden '%s' missing"), Id)); Done.ExecuteIfBound(); return; }
				for (int32 w = 0; w < Rec->Windows.Num(); ++w)
				{
					const FSttLongWindow& Win = Rec->Windows[w];
					if (Win.Start < 0 || Win.End > Audio.Num() || Win.End <= Win.Start || Win.End - Win.Start > 35 * Rate) { AddError(TEXT("golden window out of range")); Done.ExecuteIfBound(); return; }
					FSlice Item;
					Item.Label = FString::Printf(TEXT("%s window %d"), Id, w);
					Item.Audio.Append(Audio.GetData() + Win.Start, Win.End - Win.Start);
					Item.Win = &Win;
					S->Items.Add(MoveTemp(Item));
				}
			}
			RunWorker(this, Done, [S, P]()
			{
				for (FSlice& Sl : S->Items) { Sl.R = P->TranscribeBlocking(Sl.Audio, Rate, 1, Sl.T); }
			}, [this, S, Done]()
			{
				for (const FSlice& Sl : S->Items)
				{
					TestTrue(*FString::Printf(TEXT("%s: success (%s)"), *Sl.Label, *Sl.R.Message), Sl.R.bSuccess);
					const FString Gold = Sl.Win->Text.TrimStartAndEnd();
					TestTrue(*FString::Printf(TEXT("%s: text equal to golden window text (%s)"), *Sl.Label, *FirstDiff(Gold, Sl.T.Text)), SameText(Sl.T.Text.TrimStartAndEnd(), Gold));
					FString Why; double Dt = 0.0;
					TestTrue(*FString::Printf(TEXT("%s: words equal, window-local times (%s)"), *Sl.Label, *Why), WordsMatch(Sl.T, Sl.Win->Words, 0.0, Why, Dt));
				}
				Done.ExecuteIfBound();
			});
		});

		LatentIt("Resampled44kStereo47sIsWithinTwoWordsOfGolden", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// The 47.6 s recording linearly resampled to 44.1 kHz and duplicated to stereo: mix-down + resample happen BEFORE the
			// planner, so the result is a close (not bit-equal) transcript. Word edit distance vs golden words <= 2; reported.
			const FSttLongGoldens* G = nullptr; const FSttLongRecording* Rec = nullptr; TArray<float> Audio;
			if (!LoadRecording(*this, TEXT("long_61_70968_45s"), G, Rec, Audio)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			const int32 M = FMath::RoundToInt((double)Audio.Num() * 44100.0 / 16000.0);
			TArray<float> Stereo;
			Stereo.SetNumUninitialized(M * 2);
			for (int32 j = 0; j < M; ++j)
			{
				const double T = (double)j * 16000.0 / 44100.0;
				const int32 I0 = FMath::Min((int32)T, Audio.Num() - 1);
				const int32 I1 = FMath::Min(I0 + 1, Audio.Num() - 1);
				const float V = (float)(Audio[I0] + (T - I0) * (Audio[I1] - Audio[I0]));
				Stereo[2 * j] = V;
				Stereo[2 * j + 1] = V;
			}
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(P, MoveTemp(Stereo), 44100, 2, C);
			WaitCalls(this, { C }, Done, [this, C, Rec]()
			{
				TestTrue(*FString::Printf(TEXT("success (%s)"), *C->Result.Message), C->Result.bSuccess);
				TArray<FString> Got, Gold;
				for (const FSpeechWord& W : C->Transcript.Words) { Got.Add(W.Word); }
				for (const FSttLongWord& W : Rec->Words) { Gold.Add(W.Word); }
				const int32 Dist = WordEditDistance(Gold, Got);
				AddInfo(FString::Printf(TEXT("44.1 kHz stereo: %d words vs golden %d, word edit distance %d"), Got.Num(), Gold.Num(), Dist));
				TestTrue(*FString::Printf(TEXT("word edit distance %d <= 2"), Dist), Dist <= 2);
				TestTrue(*FString::Printf(TEXT("AudioSeconds %.3f ~ %.3f (input frames / input rate)"), C->Transcript.AudioSeconds, Rec->NumSamples / 16000.0),
					FMath::Abs((double)C->Transcript.AudioSeconds - Rec->NumSamples / 16000.0) < 0.01);
			});
		});

		LatentIt("ThreeLongCallsInOneFrameEachCallbackOnceAndEqualGolden", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Three long calls queued before any tick: each callback exactly once, each transcript equal to its own golden
			// (cross-talk between calls' window state, or a shared decode session, would swap or corrupt texts).
			const FSttLongGoldens* G = nullptr;
			if (!LoadLongGoldens(*this, G)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			TArray<TSharedRef<FCall>> Calls;
			TArray<const FSttLongRecording*> Recs;
			for (const TCHAR* Id : { TEXT("long_61_70968_45s"), TEXT("long_121_121726_60s"), TEXT("gap_70s") })
			{
				const FSttLongRecording* Rec = G->Find(Id);
				TArray<float> Audio;
				FString Err;
				if (!Rec || !LoadLongAudio(*G, *Rec, Audio, Err)) { AddError(Rec ? Err : FString::Printf(TEXT("golden '%s' missing"), Id)); Done.ExecuteIfBound(); return; }
				const TSharedRef<FCall> C = MakeShared<FCall>();
				Calls.Add(C);
				Recs.Add(Rec);
				Issue(P, MoveTemp(Audio), Rate, 1, C);
			}
			WaitCalls(this, Calls, Done, [this, Calls, Recs]()
			{
				for (int32 i = 0; i < Calls.Num(); ++i)
				{
					CheckAgainstGolden(this, Recs[i]->Id, Calls[i]->Result, Calls[i]->Transcript, *Recs[i]);
				}
			});
		});
	});

	// ----------------------------------------------------------------------------------------------------------------
	Describe("Limits", [this]()
	{
		LatentIt("Exactly600sOfZerosIsAcceptedAndEmpty", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Cap is "> 600 s". 24 gated windows: success, empty, InferenceMs 0 (no model run). Also 35 s + 1 sample of zeros (the old cap) succeeds.
			USpeechToTextPipeline* P = GetPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FCall> A = MakeShared<FCall>();
			const TSharedRef<FCall> B = MakeShared<FCall>();
			Issue(P, MakeZeros(600 * Rate), Rate, 1, A);
			Issue(P, MakeZeros(35 * Rate + 1), Rate, 1, B);
			WaitCalls(this, { A, B }, Done, [this, A, B]()
			{
				const TCHAR* Names[2] = { TEXT("600.0 s"), TEXT("35 s + 1 sample") };
				const TSharedRef<FCall> Cs[2] = { A, B };
				for (int32 i = 0; i < 2; ++i)
				{
					TestTrue(*FString::Printf(TEXT("%s: success (%s)"), Names[i], *Cs[i]->Result.Message), Cs[i]->Result.bSuccess);
					TestTrue(*FString::Printf(TEXT("%s: empty transcript"), Names[i]), Cs[i]->Transcript.Text.IsEmpty() && Cs[i]->Transcript.Words.Num() == 0);
					TestEqual(*FString::Printf(TEXT("%s: InferenceMs"), Names[i]), Cs[i]->Transcript.InferenceMs, 0.f);
				}
				TestTrue(*FString::Printf(TEXT("AudioSeconds %.3f == 600"), A->Transcript.AudioSeconds), FMath::Abs(A->Transcript.AudioSeconds - 600.f) < 1e-2f);
			});
		});

		LatentIt("Over600sFailsInvalidArgumentBeforeInference", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 600 s + 1 frame at 16 kHz mono, and 600.5 s at 8 kHz stereo (the cap is on duration after the input rate, not on samples).
			// InvalidArgument, message contains "at most 600 s", empty transcript, delivered later (not inside the call).
			USpeechToTextPipeline* P = GetPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FCall> A = MakeShared<FCall>();
			const TSharedRef<FCall> B = MakeShared<FCall>();
			Issue(P, MakeZeros(600 * Rate + 1), Rate, 1, A);
			Issue(P, MakeZeros((int32)(600.5 * 8000.0) * 2), 8000, 2, B);
			WaitCalls(this, { A, B }, Done, [this, A, B]()
			{
				const TSharedRef<FCall> Cs[2] = { A, B };
				for (int32 i = 0; i < 2; ++i)
				{
					TestFalse(*FString::Printf(TEXT("call %d: bSuccess"), i), Cs[i]->Result.bSuccess);
					TestTrue(*FString::Printf(TEXT("call %d: Error InvalidArgument (got %d)"), i, (int32)Cs[i]->Result.Error), Cs[i]->Result.Error == EPipelineError::InvalidArgument);
					TestTrue(*FString::Printf(TEXT("call %d: message contains 'at most 600 s' (got '%s')"), i, *Cs[i]->Result.Message),
						Cs[i]->Result.Message.Contains(TEXT("at most 600 s"), ESearchCase::CaseSensitive));
					TestTrue(*FString::Printf(TEXT("call %d: empty transcript"), i), Cs[i]->Transcript.Text.IsEmpty() && Cs[i]->Transcript.Words.Num() == 0);
				}
			});
		});
	});

	// ----------------------------------------------------------------------------------------------------------------
	// MUST stay last: releases the shared CPU pipeline.
	Describe("Lifetime", [this]()
	{
		LatentIt("PipelineDestroyedMidCallDeliversExactlyOneCallback", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 120 s recording (5 windows), pipeline released and garbage collected about 1 s after the call (typically after the
			// first window): exactly one callback, either Cancelled (non-empty message) or a full success equal to the golden.
			// A callback that is dropped, fired per window, or fired from a destroyed object's destructor and again later fails here.
			const FSttLongGoldens* G = nullptr; const FSttLongRecording* Rec = nullptr; TArray<float> Audio;
			if (!LoadRecording(*this, TEXT("long_260_123286_120s"), G, Rec, Audio)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(P, MoveTemp(Audio), Rate, 1, C);
			const double IssuedAt = FPlatformTime::Seconds();
			const TSharedRef<bool> bReleased = MakeShared<bool>(false);
			FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechLongRelease"), 0.02f, [IssuedAt, bReleased, C](float)
			{
				if (FPlatformTime::Seconds() - IssuedAt < 1.0 && C->Count.GetValue() == 0) { return true; }
				*bReleased = (C->Count.GetValue() == 0);   // false when the whole call already finished: still a valid outcome
				if (*bReleased) { ReleaseSharedCpuSttPipeline(); }
				return false;
			});
			WaitCalls(this, { C }, Done, [this, C, Rec, bReleased]()
			{
				AddInfo(FString::Printf(TEXT("pipeline released while the call was pending: %s; result: %s (%s)"), *bReleased ? TEXT("yes") : TEXT("no, call had finished"),
					C->Result.bSuccess ? TEXT("success") : TEXT("failure"), *C->Result.Message));
				if (C->Result.bSuccess)
				{
					CheckAgainstGolden(this, TEXT("destroyed-mid-call success"), C->Result, C->Transcript, *Rec);
				}
				else
				{
					TestTrue(*FString::Printf(TEXT("failure is Cancelled (got %d: %s)"), (int32)C->Result.Error, *C->Result.Message), C->Result.Error == EPipelineError::Cancelled);
					TestFalse(TEXT("Cancelled carries a message"), C->Result.Message.IsEmpty());
					TestTrue(TEXT("Cancelled: empty transcript"), C->Transcript.Text.IsEmpty() && C->Transcript.Words.Num() == 0);
				}
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
