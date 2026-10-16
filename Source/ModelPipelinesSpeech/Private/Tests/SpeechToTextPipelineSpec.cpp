// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9S.5a / contract C75: USpeechToTextPipeline (clips <= 35 s). Numbers are checked against the Python goldens
// (<dev>/goldens/stt/stt_goldens.json: ONNX Runtime reference of the same fp16w graphs), never against the implementation's output.
// Missing goldens or model folders FAIL these tests (never skip) with the path and the generating command.
//
// Text comparisons are CASE-SENSITIVE everywhere (FString == / TestEqual(FString) ignore case in Unreal).
// The public API exposes no per-step trace, so "greedy steps equal" (acceptance 1/2) is checked through its observable
// consequence: the words with their start/end times (= first token's t and last token's t + d) equal onnx.word_frames.
// A clip whose words and times all match has, for every emitted token, the same (k, t, d) at the word boundaries; we report it
// as "steps-equal proxy". See the NEEDS lines of the report.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "HAL/PlatformMemory.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "Misc/CoreDelegates.h"
#include "Misc/ScopeLock.h"
#include "NNEModelData.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineExecutor.h"
#include "PipelineModelPackage.h"
#include "PipelineNNEBackend.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "SpeechPipelineLibrary.h"
#include "SpeechToTextPipeline.h"
#include "SpeechTypes.h"
#include "SpeechTestData.h"

#include <limits>

using namespace UE::ModelPipelines::SpeechTests;

namespace SpeechPipelineSpecPrivate
{
	// Cold model load + 40 encoder runs on CPU (~0.5 s each) + decode: generous, but every wait is bounded.
	const FTimespan LatentLimit = FTimespan::FromSeconds(1500);
	constexpr double WorkerTimeoutSeconds = 1400.0;
	constexpr double CallTimeoutSeconds = 300.0;
	constexpr double GraceSeconds = 0.4;      // after the last expected callback: a duplicate callback would show up here
	constexpr double FrameSeconds = 0.08;     // SubsamplingFactor 8 * HopLength 160 / 16000
	constexpr double TimeTolerance = 1e-4;
	constexpr double TwoPi = 6.283185307179586476925286766559;

	// ------------------------------------------------------------------------------------------------------------------
	// Shared pipelines (one per device per process; the model package itself is shared by SpeechTestData)

	struct FPipelineHolder
	{
		TStrongObjectPtr<USpeechToTextPipeline> Cpu;
		TStrongObjectPtr<USpeechToTextPipeline> Gpu;
		TStrongObjectPtr<USpeechToTextPipeline> Tiny;
		TStrongObjectPtr<UPipelineModelPackage> WrongTaskPackage;
	};

	static FPipelineHolder* GHolder = nullptr;

	static FPipelineHolder& Holder()
	{
		if (!GHolder)
		{
			GHolder = new FPipelineHolder(); // never deleted; released at engine pre-exit
			FCoreDelegates::OnEnginePreExit.AddLambda([]()
			{
				if (GHolder)
				{
					GHolder->Cpu.Reset();
					GHolder->Gpu.Reset();
					GHolder->Tiny.Reset();
					GHolder->WrongTaskPackage.Reset();
				}
			});
		}
		return *GHolder;
	}

	/** A NEW pipeline on the shared Phonon package (not cached). Null + error on failure. */
	static USpeechToTextPipeline* NewPhononPipeline(FAutomationTestBase& T, EPipelineDevice Device)
	{
		UPipelineModelPackage* Pkg = nullptr;
		FString Err;
		if (!GetSharedPhononPackage(Device, Pkg, Err))
		{
			T.AddError(Err);
			return nullptr;
		}
		FPipelineResult R;
		USpeechToTextPipeline* P = USpeechPipelineLibrary::CreateSpeechToTextPipeline(Pkg, R);
		if (!P)
		{
			T.AddError(FString::Printf(TEXT("CreateSpeechToTextPipeline failed on the Phonon package: %s"), *R.Message));
		}
		return P;
	}

	static USpeechToTextPipeline* GetSharedPipeline(FAutomationTestBase& T, EPipelineDevice Device)
	{
		TStrongObjectPtr<USpeechToTextPipeline>& Slot = (Device == EPipelineDevice::GPU) ? Holder().Gpu : Holder().Cpu;
		if (!Slot.IsValid())
		{
			Slot.Reset(NewPhononPipeline(T, Device));
		}
		return Slot.Get();
	}

	static USpeechToTextPipeline* GetTinyPipeline(FAutomationTestBase& T)
	{
		TStrongObjectPtr<USpeechToTextPipeline>& Slot = Holder().Tiny;
		if (!Slot.IsValid())
		{
			UPipelineModelPackage* Pkg = nullptr;
			FString Err;
			if (!GetSharedTinySttPackage(Pkg, Err))
			{
				T.AddError(Err);
				return nullptr;
			}
			FPipelineResult R;
			USpeechToTextPipeline* P = USpeechPipelineLibrary::CreateSpeechToTextPipeline(Pkg, R);
			if (!P)
			{
				T.AddError(FString::Printf(TEXT("CreateSpeechToTextPipeline failed on the tiny package: %s"), *R.Message));
				return nullptr;
			}
			Slot.Reset(P);
		}
		return Slot.Get();
	}

	/** True = skip. GPU tests run headless (DirectML works under -nullrhi), so availability of the GPU interface decides. */
	static bool GpuSkip(FAutomationTestBase& T)
	{
		FPipelineNNEBackend Backend;
		if (Backend.IsDeviceAvailable(EPipelineDevice::GPU))
		{
			return false;
		}
		if (IsGpuRequired())
		{
			T.AddError(TEXT("GPU not available but MODELPIPELINES_REQUIRE_GPU=1"));
		}
		else
		{
			T.AddInfo(TEXT("SKIPPED: GPU not available (set MODELPIPELINES_REQUIRE_GPU=1 to make this a failure)"));
		}
		return true;
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Small helpers

	static bool SameText(const FString& A, const FString& B)
	{
		return A.Equals(B, ESearchCase::CaseSensitive);
	}

	/** "at char N: expected '...' got '...'" for the first difference (case-sensitive). */
	static FString FirstDiff(const FString& Expected, const FString& Got)
	{
		const int32 N = FMath::Min(Expected.Len(), Got.Len());
		int32 I = 0;
		while (I < N && Expected[I] == Got[I]) { ++I; }
		const int32 From = FMath::Max(0, I - 12);
		return FString::Printf(TEXT("at char %d: expected '%s' got '%s'"), I, *Expected.Mid(From, 32), *Got.Mid(From, 32));
	}

	/** Words equal onnx.word_frames: text (case-sensitive) and start/end = frames * 0.08 within 1e-4 s. Empty OutWhy when equal. */
	static bool WordsMatch(const FSpeechTranscript& Tr, const FSttClip& Clip, FString& OutWhy)
	{
		if (Tr.Words.Num() != Clip.Words.Num())
		{
			OutWhy = FString::Printf(TEXT("word count %d vs golden %d"), Tr.Words.Num(), Clip.Words.Num());
			return false;
		}
		for (int32 i = 0; i < Clip.Words.Num(); ++i)
		{
			const FSpeechWord& W = Tr.Words[i];
			const FSttGoldWord& G = Clip.Words[i];
			if (!SameText(W.Word, G.Word))
			{
				OutWhy = FString::Printf(TEXT("word %d is '%s', golden '%s'"), i, *W.Word, *G.Word);
				return false;
			}
			const double ES = G.StartFrame * FrameSeconds;
			const double EE = G.EndFrame * FrameSeconds;
			if (FMath::Abs((double)W.StartSeconds - ES) > TimeTolerance || FMath::Abs((double)W.EndSeconds - EE) > TimeTolerance)
			{
				OutWhy = FString::Printf(TEXT("word %d '%s' times %.5f..%.5f, golden %.5f..%.5f"), i, *W.Word, W.StartSeconds, W.EndSeconds, ES, EE);
				return false;
			}
		}
		return true;
	}

	static bool LoadGoldens(FAutomationTestBase& T, const FSttGoldens*& OutG)
	{
		FString Err;
		if (!LoadSttGoldens(OutG, Err))
		{
			T.AddError(Err);
			return false;
		}
		return true;
	}

	static bool LoadAudio(FAutomationTestBase& T, const FSttGoldens& G, const FSttClip& C, TArray<float>& Out)
	{
		FString Err;
		if (!LoadClipAudio(G, C, Out, Err))
		{
			T.AddError(Err);
			return false;
		}
		return true;
	}

	static TArray<float> MakeSine(double Freq, int32 Rate, int32 N, double Amp)
	{
		TArray<float> S;
		S.SetNumUninitialized(N);
		for (int32 i = 0; i < N; ++i)
		{
			S[i] = (float)(Amp * FMath::Sin(TwoPi * Freq * (double)i / (double)Rate));
		}
		return S;
	}

	/** Amplitude of a sine with the given RMS in dBFS (RMS = A / sqrt 2). */
	static double AmpForRmsDb(double Db)
	{
		return FMath::Pow(10.0, Db / 20.0) * 1.41421356237309505;
	}

	/** 16 kHz mono -> 44.1 kHz stereo, linear interpolation, both channels equal, interleaved. */
	static TArray<float> To44k1Stereo(const TArray<float>& In)
	{
		const int32 N = In.Num();
		const int32 M = FMath::RoundToInt((double)N * 44100.0 / 16000.0);
		TArray<float> Out;
		Out.SetNumUninitialized(M * 2);
		for (int32 i = 0; i < M; ++i)
		{
			const double T = (double)i * 16000.0 / 44100.0;
			const int32 I0 = FMath::Min((int32)FMath::FloorToDouble(T), N - 1);
			const int32 I1 = FMath::Min(I0 + 1, N - 1);
			const double F = T - (double)I0;
			const float V = (float)((double)In[I0] * (1.0 - F) + (double)In[I1] * F);
			Out[2 * i] = V;
			Out[2 * i + 1] = V;
		}
		return Out;
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Blocking reference cache (CPU pipeline, 16 kHz mono): the "blocking result" the async and resampling tests compare to

	static FCriticalSection GRefLock;
	static TMap<FString, FSpeechTranscript> GRefs;

	static FPipelineResult BlockingReference(USpeechToTextPipeline* P, const FString& Id, const TArray<float>& Audio, FSpeechTranscript& Out)
	{
		{
			FScopeLock Lock(&GRefLock);
			if (const FSpeechTranscript* Found = GRefs.Find(Id))
			{
				Out = *Found;
				return FPipelineResult::Ok();
			}
		}
		FPipelineResult R = P->TranscribeBlocking(Audio, 16000, 1, Out);
		if (R.bSuccess)
		{
			FScopeLock Lock(&GRefLock);
			GRefs.Add(Id, Out);
		}
		return R;
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Latent plumbing. Everything a callback or worker touches is a TSharedRef captured by value: if a test times out and
	// returns, late callbacks still only touch live memory.

	struct FCallState
	{
		FThreadSafeCounter Count;
		bool bInCall = false;
		bool bAnyInCall = false;
		bool bAllOnGameThread = true;
		int32 Order = 0;                 // sequence number when the first callback fired (if a sequence counter was given)
		double FirstSeenSeconds = 0.0;
		FPipelineResult Result;
		FSpeechTranscript Transcript;
	};

	static FOnSpeechTranscribed MakeCallback(const TSharedRef<FCallState>& C, TSharedPtr<FThreadSafeCounter> Seq = nullptr)
	{
		return [C, Seq](const FPipelineResult& R, const FSpeechTranscript& Tr)
		{
			if (!IsInGameThread()) { C->bAllOnGameThread = false; }
			if (C->bInCall) { C->bAnyInCall = true; }
			if (C->Count.Increment() == 1)
			{
				C->Result = R;
				C->Transcript = Tr;
				C->FirstSeenSeconds = FPlatformTime::Seconds();
				if (Seq.IsValid()) { C->Order = Seq->Increment(); }
			}
		};
	}

	/** Issues Transcribe on the game thread; flags a callback that runs before Transcribe returns. */
	static void Issue(USpeechToTextPipeline* P, TArray<float> Samples, int32 Rate, int32 Channels, const TSharedRef<FCallState>& C,
		TSharedPtr<FThreadSafeCounter> Seq = nullptr)
	{
		C->bInCall = true;
		P->Transcribe(MoveTemp(Samples), Rate, Channels, MakeCallback(C, Seq));
		C->bInCall = false;
	}

	/**
	 * Waits (ticker, never blocking) until every call fired once, plus GraceSeconds to catch a second callback; then asserts
	 * exactly once / game thread / not inside Transcribe and runs Check on the game thread. Fails with a message on timeout.
	 */
	static void WaitCalls(FAutomationTestBase* T, TArray<TSharedRef<FCallState>> Calls, const FDoneDelegate& Done, TFunction<void()> Check,
		double TimeoutSeconds = CallTimeoutSeconds)
	{
		const double Start = FPlatformTime::Seconds();
		const TSharedRef<double> AllSeen = MakeShared<double>(0.0);
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechPipelineSpecWait"), 0.02f, [T, Calls, Start, AllSeen, DoneCopy, Check, TimeoutSeconds](float)
		{
			const double Now = FPlatformTime::Seconds();
			int32 Arrived = 0;
			for (const TSharedRef<FCallState>& C : Calls) { Arrived += C->Count.GetValue() > 0 ? 1 : 0; }
			if (Arrived == Calls.Num() && *AllSeen == 0.0) { *AllSeen = Now; }
			const bool bTimeout = (Now - Start) > TimeoutSeconds;
			if ((*AllSeen > 0.0 && (Now - *AllSeen) > GraceSeconds) || bTimeout)
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

	/** Work runs on its own thread; Then runs on the game thread when it finishes (Then must call Done). Timeout -> error + Done. */
	static void RunWorker(FAutomationTestBase* T, const FDoneDelegate& Done, TFunction<void()> Work, TFunction<void()> Then)
	{
		const TSharedRef<TFuture<void>> F = MakeShared<TFuture<void>>(Async(EAsyncExecution::Thread, MoveTemp(Work)));
		const double Start = FPlatformTime::Seconds();
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechPipelineSpecWorker"), 0.02f, [T, F, Start, DoneCopy, Then](float)
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

	// ------------------------------------------------------------------------------------------------------------------
	// Acceptance 1 and 2: parity over the 40 visible clips

	struct FParity
	{
		TArray<const FSttClip*> Clips;
		TArray<TArray<float>> Audio;
		TArray<FPipelineResult> Results;
		TArray<FSpeechTranscript> Transcripts;
		double WallSeconds = 0.0;
	};

	static void StartParity(FAutomationTestBase* T, const FDoneDelegate& Done, USpeechToTextPipeline* Pipe, EPipelineDevice Expect,
		const TCHAR* Label, int32 MinTextEqual, int32 MinProxyEqual)
	{
		const FSttGoldens* G = nullptr;
		if (!Pipe || !LoadGoldens(*T, G)) { Done.ExecuteIfBound(); return; }
		const TSharedRef<FParity> S = MakeShared<FParity>();
		S->Clips = G->VisibleClips();
		if (S->Clips.Num() != 40)
		{
			T->AddError(FString::Printf(TEXT("%s: expected 40 visible clips in the goldens, found %d"), Label, S->Clips.Num()));
			Done.ExecuteIfBound();
			return;
		}
		S->Audio.SetNum(40);
		for (int32 i = 0; i < 40; ++i)
		{
			if (!S->Clips[i]->bHasOnnx || S->Clips[i]->Words.Num() == 0)
			{
				T->AddError(FString::Printf(TEXT("%s: clip %s has no onnx block in the goldens (regenerate with make_stt_goldens.py)"), Label, *S->Clips[i]->Id));
				Done.ExecuteIfBound();
				return;
			}
			if (!LoadAudio(*T, *G, *S->Clips[i], S->Audio[i])) { Done.ExecuteIfBound(); return; }
		}
		S->Results.SetNum(40);
		S->Transcripts.SetNum(40);
		const FString LabelStr = Label;
		FDoneDelegate DoneCopy = Done;

		RunWorker(T, Done, [S, Pipe]()
		{
			const double T0 = FPlatformTime::Seconds();
			for (int32 i = 0; i < 40; ++i)
			{
				S->Results[i] = Pipe->TranscribeBlocking(S->Audio[i], 16000, 1, S->Transcripts[i]);
			}
			S->WallSeconds = FPlatformTime::Seconds() - T0;
		}, [T, S, DoneCopy, LabelStr, Expect, MinTextEqual, MinProxyEqual]()
		{
			int32 TextEqual = 0, TextEqualNoCase = 0, ProxyEqual = 0, Failed = 0, WrongDevice = 0, BadDuration = 0, NoTiming = 0;
			double SumInferenceMs = 0.0, SumAudioSeconds = 0.0, WorstMs = 0.0;
			for (int32 i = 0; i < 40; ++i)
			{
				const FSttClip& C = *S->Clips[i];
				const FPipelineResult& R = S->Results[i];
				const FSpeechTranscript& Tr = S->Transcripts[i];
				if (!R.bSuccess)
				{
					++Failed;
					T->AddError(FString::Printf(TEXT("%s: clip %s failed: error %d, %s"), *LabelStr, *C.Id, (int32)R.Error, *R.Message));
					continue;
				}
				if (R.Device != Expect) { ++WrongDevice; }
				if (R.InferenceMs <= 0.f || Tr.InferenceMs <= 0.f) { ++NoTiming; }
				if (FMath::Abs((double)Tr.AudioSeconds - (double)C.NumSamples / 16000.0) > 1e-3) { ++BadDuration; }
				SumInferenceMs += Tr.InferenceMs;
				SumAudioSeconds += (double)C.NumSamples / 16000.0;
				WorstMs = FMath::Max(WorstMs, (double)Tr.InferenceMs);

				const bool bText = SameText(Tr.Text, C.OnnxText);
				if (bText) { ++TextEqual; }
				else if (Tr.Text.Equals(C.OnnxText, ESearchCase::IgnoreCase)) { ++TextEqualNoCase; }
				FString Why;
				const bool bWords = WordsMatch(Tr, C, Why);
				if (bWords) { ++ProxyEqual; }
				if (!bText || !bWords)
				{
					T->AddInfo(FString::Printf(TEXT("%s MISMATCH clip %s: text %s%s; words %s"), *LabelStr, *C.Id,
						bText ? TEXT("equal") : TEXT("DIFFERS "), bText ? TEXT("") : *FirstDiff(C.OnnxText, Tr.Text),
						bWords ? TEXT("equal") : *Why));
				}
			}
			T->TestEqual(*FString::Printf(TEXT("%s: clips that failed"), *LabelStr), Failed, 0);
			T->TestEqual(*FString::Printf(TEXT("%s: results whose Device != %d"), *LabelStr, (int32)Expect), WrongDevice, 0);
			T->TestEqual(*FString::Printf(TEXT("%s: clips with InferenceMs <= 0"), *LabelStr), NoTiming, 0);
			T->TestEqual(*FString::Printf(TEXT("%s: clips with AudioSeconds != samples/16000"), *LabelStr), BadDuration, 0);
			T->TestEqual(*FString::Printf(TEXT("%s: text differing from onnx.text only by letter case (case-sensitive compare)"), *LabelStr), TextEqualNoCase, 0);
			T->TestTrue(*FString::Printf(TEXT("%s: Text == onnx.text (case-sensitive) on %d/40, need >= %d"), *LabelStr, TextEqual, MinTextEqual), TextEqual >= MinTextEqual);
			if (MinProxyEqual >= 0)
			{
				T->TestTrue(*FString::Printf(TEXT("%s: words+times == onnx.word_frames (steps-equal proxy) on %d/40, need >= %d"), *LabelStr, ProxyEqual, MinProxyEqual), ProxyEqual >= MinProxyEqual);
			}
			// Item 8 (reported, not gated)
			const double Rtf = SumAudioSeconds > 0.0 ? (SumInferenceMs / 1000.0) / SumAudioSeconds : 0.0;
			T->AddInfo(FString::Printf(TEXT("%s REPORT: text equal %d/40, steps-equal proxy (words+times) %d/40, total audio %.1f s, wall %.1f s, "
				"mean InferenceMs %.1f, worst %.1f, RTF (InferenceMs/audio) %.4f, peak physical memory %.0f MB"),
				*LabelStr, TextEqual, ProxyEqual, SumAudioSeconds, S->WallSeconds, SumInferenceMs / 40.0, WorstMs, Rtf,
				(double)FPlatformMemory::GetStats().PeakUsedPhysical / (1024.0 * 1024.0)));
			DoneCopy.ExecuteIfBound();
		});
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Acceptance 5: error cases

	enum class EErrCase : int32
	{
		Rate999, Rate384001, ZeroChannels, NegativeChannels, RaggedLength, OneNaN, MixedNonFinite, Over600s, Over600sStereo44k, NotReady, Count
	};

	static const TCHAR* ErrCaseName(EErrCase C)
	{
		switch (C)
		{
		case EErrCase::Rate999: return TEXT("Rate999");
		case EErrCase::Rate384001: return TEXT("Rate384001");
		case EErrCase::ZeroChannels: return TEXT("ZeroChannels");
		case EErrCase::NegativeChannels: return TEXT("NegativeChannels");
		case EErrCase::RaggedLength: return TEXT("RaggedLength");
		case EErrCase::OneNaN: return TEXT("OneNaN");
		case EErrCase::MixedNonFinite: return TEXT("ThreeNonFinite");
		case EErrCase::Over600s: return TEXT("Over600s");
		case EErrCase::Over600sStereo44k: return TEXT("Over600sStereo44k");
		case EErrCase::NotReady: return TEXT("NotReady");
		default: return TEXT("?");
		}
	}

	/** Builds the call; AnyOf = substrings of which the message must contain at least one (case-insensitive). */
	static void BuildErrCase(EErrCase Case, TArray<float>& Samples, int32& Rate, int32& Channels, TArray<FString>& AnyOf)
	{
		Rate = 16000;
		Channels = 1;
		Samples = MakeSine(440.0, 16000, 16000, 0.1);
		switch (Case)
		{
		case EErrCase::Rate999: Rate = 999; AnyOf = { TEXT("999") }; break;
		case EErrCase::Rate384001: Rate = 384001; AnyOf = { TEXT("384001") }; break;
		case EErrCase::ZeroChannels: Channels = 0; AnyOf = { TEXT("channel") }; break;
		case EErrCase::NegativeChannels: Channels = -2; AnyOf = { TEXT("channel") }; break;
		case EErrCase::RaggedLength: Channels = 2; Samples.SetNum(16001); AnyOf = { TEXT("channel"), TEXT("multiple"), TEXT("length") }; break;
		case EErrCase::OneNaN: Samples[777] = std::numeric_limits<float>::quiet_NaN(); AnyOf = { TEXT("1 non-finite") }; break;
		case EErrCase::MixedNonFinite:
			Samples[10] = std::numeric_limits<float>::quiet_NaN();
			Samples[20] = std::numeric_limits<float>::infinity();
			Samples[30] = -std::numeric_limits<float>::infinity();
			AnyOf = { TEXT("3 non-finite") };
			break;
		case EErrCase::Over600s: Samples.Init(0.f, 9600001); AnyOf = { TEXT("at most 600 s") }; break;   // 600 s + 1 frame at 16 kHz mono (WP9S.5c2 limit)
		case EErrCase::Over600sStereo44k: Rate = 44100; Channels = 2; Samples.Init(0.f, (int32)(600.5 * 44100.0) * 2); AnyOf = { TEXT("at most 600 s") }; break;   // 600.5 s stereo
		case EErrCase::NotReady: AnyOf.Reset(); break;
		default: break;
		}
	}
}

BEGIN_DEFINE_SPEC(FSpeechToTextPipelineSpec, "Pipelines.Speech.Pipeline",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
END_DEFINE_SPEC(FSpeechToTextPipelineSpec)

void FSpeechToTextPipelineSpec::Define()
{
	using namespace SpeechPipelineSpecPrivate;

	// ----------------------------------------------------------------------------------------------------------------
	Describe("Parity", [this]()
	{
		LatentIt("CpuAgainstGoldens", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 1: 40 clips via TranscribeBlocking, 16 kHz mono, CPU. Text equal (case-sensitive) on >= 39/40; the
			// steps-equal proxy (words and 0.08 s-grid times) on >= 38/40. A front end that is slightly off (wrong pre-emphasis,
			// frame count, normalisation) flips tokens; a decoder that mishandles the duration-0 rule or the state update
			// after blanks shifts word times; a lower-cased or trimmed-differently decode fails the case-sensitive compare.
			StartParity(this, Done, GetSharedPipeline(*this, EPipelineDevice::CPU), EPipelineDevice::CPU, TEXT("CPU"), 39, 38);
		});

		LatentIt("GpuAgainstGoldens", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 2: encoder on DirectML (decoder stays on CPU). Text equal on >= 38/40; Result.Device must be GPU.
			// Skipped with an AddInfo when no GPU interface exists; FAILS when MODELPIPELINES_REQUIRE_GPU=1 and there is none.
			if (GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
			StartParity(this, Done, GetSharedPipeline(*this, EPipelineDevice::GPU), EPipelineDevice::GPU, TEXT("GPU"), 38, -1);
		});
	});

	// ----------------------------------------------------------------------------------------------------------------
	Describe("Gate", [this]()
	{
		LatentIt("SilenceAndLengthThresholds", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 3. silence_0p5s: success, empty text, InferenceMs 0 (no inference). A 1 kHz sine at -59 dBFS RMS is above
			// the -60 dB gate: success and InferenceMs > 0; at -61 dBFS it is below: empty and InferenceMs 0 (a gate that used
			// peak instead of RMS, or compared in the wrong direction, flips one of the two). short_319 (1 valid frame) -> empty
			// success; short_320 (2 frames) -> success.
			const FSttGoldens* G = nullptr;
			if (!LoadGoldens(*this, G)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetSharedPipeline(*this, EPipelineDevice::CPU);
			if (!P) { Done.ExecuteIfBound(); return; }
			struct FRun { FPipelineResult R; FSpeechTranscript T; };
			struct FGate { TArray<FString> Names; TArray<TArray<float>> Audio; TArray<FRun> Runs; double Rms320 = -200.0; bool bHasRms320 = false; };
			const TSharedRef<FGate> S = MakeShared<FGate>();
			for (const TCHAR* Id : { TEXT("silence_0p5s"), TEXT("short_319"), TEXT("short_320") })
			{
				const FSttClip* C = G->Find(Id);
				if (!C) { AddError(FString::Printf(TEXT("golden clip '%s' missing"), Id)); Done.ExecuteIfBound(); return; }
				TArray<float> A;
				if (!LoadAudio(*this, *G, *C, A)) { Done.ExecuteIfBound(); return; }
				if (FString(Id) == TEXT("short_320")) { S->bHasRms320 = C->bHasRms; S->Rms320 = C->RmsDbfs; }
				S->Names.Add(Id);
				S->Audio.Add(MoveTemp(A));
			}
			S->Names.Add(TEXT("sine1k_-59dB"));
			S->Audio.Add(MakeSine(1000.0, 16000, 32000, AmpForRmsDb(-59.0)));
			S->Names.Add(TEXT("sine1k_-61dB"));
			S->Audio.Add(MakeSine(1000.0, 16000, 32000, AmpForRmsDb(-61.0)));
			S->Runs.SetNum(S->Audio.Num());
			FDoneDelegate DoneCopy = Done;
			RunWorker(this, Done, [S, P]()
			{
				for (int32 i = 0; i < S->Audio.Num(); ++i)
				{
					S->Runs[i].R = P->TranscribeBlocking(S->Audio[i], 16000, 1, S->Runs[i].T);
				}
			}, [this, S, DoneCopy]()
			{
				for (int32 i = 0; i < S->Runs.Num(); ++i)
				{
					TestTrue(*FString::Printf(TEXT("%s: success (%s)"), *S->Names[i], *S->Runs[i].R.Message), S->Runs[i].R.bSuccess);
				}
				const FRun& Sil = S->Runs[0];
				TestTrue(TEXT("silence_0p5s: empty text"), Sil.T.Text.IsEmpty() && Sil.T.Words.Num() == 0);
				TestEqual(TEXT("silence_0p5s: InferenceMs (no inference ran)"), Sil.T.InferenceMs, 0.f);
				TestTrue(*FString::Printf(TEXT("silence_0p5s: AudioSeconds %.4f == 0.5"), Sil.T.AudioSeconds), FMath::Abs(Sil.T.AudioSeconds - 0.5f) < 1e-3f);
				const FRun& S319 = S->Runs[1];
				TestTrue(TEXT("short_319: empty text (too short, 1 valid frame)"), S319.T.Text.IsEmpty() && S319.T.Words.Num() == 0);
				TestEqual(TEXT("short_319: InferenceMs"), S319.T.InferenceMs, 0.f);
				const FRun& S320 = S->Runs[2];
				// 320 samples = 2 valid frames: the pipeline must get past the length check. Inference runs unless the clip
				// itself is below the silence gate (then it is gated by RMS, not by length).
				if (S->bHasRms320 && S->Rms320 > -59.0)
				{
					TestTrue(*FString::Printf(TEXT("short_320: InferenceMs %.3f > 0 (rms %.1f dBFS, 2 frames reach the encoder)"), S320.T.InferenceMs, S->Rms320), S320.T.InferenceMs > 0.f);
				}
				const FRun& On = S->Runs[3];
				TestTrue(*FString::Printf(TEXT("sine -59 dBFS: InferenceMs %.3f > 0 (above the -60 dB gate, transcribed)"), On.T.InferenceMs), On.T.InferenceMs > 0.f);
				AddInfo(FString::Printf(TEXT("sine -59 dBFS transcribed as '%s' (any text accepted)"), *On.T.Text));
				const FRun& Off = S->Runs[4];
				TestTrue(*FString::Printf(TEXT("sine -61 dBFS: empty text, got '%s'"), *Off.T.Text), Off.T.Text.IsEmpty() && Off.T.Words.Num() == 0);
				TestEqual(TEXT("sine -61 dBFS: InferenceMs"), Off.T.InferenceMs, 0.f);
				DoneCopy.ExecuteIfBound();
			});
		});
	});

	// ----------------------------------------------------------------------------------------------------------------
	Describe("Resample", [this]()
	{
		LatentIt("Stereo44k1MatchesMono16k", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 4: 5 clips converted in the test to 44.1 kHz stereo (linear interpolation, both channels equal) must
			// transcribe to the same Text as the 16 kHz mono run on >= 4/5. A pipeline that ignored NumChannels (treated stereo as
			// mono = double speed) or the sample rate (feeding 44.1 kHz as 16 kHz) gets none of them right.
			const FSttGoldens* G = nullptr;
			if (!LoadGoldens(*this, G)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetSharedPipeline(*this, EPipelineDevice::CPU);
			if (!P) { Done.ExecuteIfBound(); return; }
			const TArray<const FSttClip*> Visible = G->VisibleClips();
			if (Visible.Num() < 37) { AddError(TEXT("fewer than 37 visible clips in the goldens")); Done.ExecuteIfBound(); return; }
			struct FItem { FString Id; TArray<float> Audio; TArray<float> Stereo; FSpeechTranscript Ref; FSpeechTranscript Conv; FPipelineResult RefR; FPipelineResult ConvR; };
			const TSharedRef<TArray<FItem>> Items = MakeShared<TArray<FItem>>();
			for (int32 Idx : { 0, 9, 18, 27, 36 })
			{
				FItem Item;
				Item.Id = Visible[Idx]->Id;
				if (!LoadAudio(*this, *G, *Visible[Idx], Item.Audio)) { Done.ExecuteIfBound(); return; }
				Item.Stereo = To44k1Stereo(Item.Audio);
				Items->Add(MoveTemp(Item));
			}
			FDoneDelegate DoneCopy = Done;
			RunWorker(this, Done, [Items, P]()
			{
				for (FItem& Item : *Items)
				{
					Item.RefR = BlockingReference(P, Item.Id, Item.Audio, Item.Ref);
					Item.ConvR = P->TranscribeBlocking(Item.Stereo, 44100, 2, Item.Conv);
				}
			}, [this, Items, DoneCopy]()
			{
				int32 Same = 0;
				for (const FItem& Item : *Items)
				{
					if (!TestTrue(*FString::Printf(TEXT("%s: 16k mono + 44.1k stereo both succeed (%s / %s)"), *Item.Id, *Item.RefR.Message, *Item.ConvR.Message), Item.RefR.bSuccess && Item.ConvR.bSuccess))
					{
						continue;
					}
					const bool bSame = SameText(Item.Ref.Text, Item.Conv.Text);
					Same += bSame ? 1 : 0;
					AddInfo(FString::Printf(TEXT("%s: 44.1k stereo %s the 16k result%s; AudioSeconds %.3f vs %.3f"), *Item.Id, bSame ? TEXT("equals") : TEXT("DIFFERS from"),
						bSame ? TEXT("") : *(FString(TEXT(" ")) + FirstDiff(Item.Ref.Text, Item.Conv.Text)), Item.Conv.AudioSeconds, Item.Ref.AudioSeconds));
					// AudioSeconds is the INPUT duration: frames / rate, not the resampled sample count.
					TestTrue(*FString::Printf(TEXT("%s: AudioSeconds %.4f within 0.01 s of the 16k clip's %.4f"), *Item.Id, Item.Conv.AudioSeconds, Item.Ref.AudioSeconds),
						FMath::Abs(Item.Conv.AudioSeconds - Item.Ref.AudioSeconds) < 0.01f);
				}
				TestTrue(*FString::Printf(TEXT("44.1 kHz stereo Text == 16 kHz mono Text on %d/5 clips, need >= 4"), Same), Same >= 4);
				DoneCopy.ExecuteIfBound();
			});
		});
	});

	// ----------------------------------------------------------------------------------------------------------------
	Describe("Errors", [this]()
	{
		// Acceptance 5: each case through Transcribe with a callback, delivered on a later tick (never inside Transcribe), exactly once,
		// InvalidArgument with a message naming the problem and an empty transcript. Runs on the tiny package: these paths
		// never reach inference, so no 1.3 GB model is needed (and a rejection that still ran inference would fail on shape).
		for (int32 CaseIndex = 0; CaseIndex < (int32)EErrCase::Count; ++CaseIndex)
		{
			const EErrCase Case = (EErrCase)CaseIndex;
			LatentIt(ErrCaseName(Case), LatentLimit, [this, Case](const FDoneDelegate& Done)
			{
				USpeechToTextPipeline* P = nullptr;
				if (Case == EErrCase::NotReady)
				{
					// A pipeline that never saw InitializeFromPackage: IsReady() is false, so every call fails instead of crashing.
					P = NewObject<USpeechToTextPipeline>(GetTransientPackage());
					TestFalse(TEXT("precondition: pipeline is not ready"), P->IsReady());
				}
				else
				{
					P = GetTinyPipeline(*this);
				}
				if (!P) { Done.ExecuteIfBound(); return; }
				const TStrongObjectPtr<USpeechToTextPipeline> Keep(P);
				TArray<float> Samples;
				int32 Rate = 0, Channels = 0;
				TArray<FString> AnyOf;
				BuildErrCase(Case, Samples, Rate, Channels, AnyOf);
				const TSharedRef<FCallState> C = MakeShared<FCallState>();
				Issue(P, MoveTemp(Samples), Rate, Channels, C);
				WaitCalls(this, { C }, Done, [this, C, AnyOf, Keep]()
				{
					TestFalse(TEXT("bSuccess"), C->Result.bSuccess);
					TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)C->Result.Error, *C->Result.Message), C->Result.Error == EPipelineError::InvalidArgument);
					TestFalse(TEXT("Message not empty"), C->Result.Message.IsEmpty());
					if (AnyOf.Num() > 0)
					{
						bool bNamed = false;
						for (const FString& S : AnyOf) { bNamed |= C->Result.Message.Contains(S, ESearchCase::IgnoreCase); }
						TestTrue(*FString::Printf(TEXT("message names the problem (needs one of the substrings; got: %s)"), *C->Result.Message), bNamed);
					}
					TestTrue(TEXT("transcript is empty on failure"), C->Transcript.Text.IsEmpty() && C->Transcript.Words.Num() == 0);
				});
			});
		}

		LatentIt("EmptySamplesSucceedWithEmptyTranscript", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Empty Samples is not an error: success, empty transcript, no inference.
			USpeechToTextPipeline* P = GetTinyPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FCallState> C = MakeShared<FCallState>();
			Issue(P, TArray<float>(), 16000, 1, C);
			WaitCalls(this, { C }, Done, [this, C]()
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C->Result.Message), C->Result.bSuccess);
				TestTrue(TEXT("empty transcript"), C->Transcript.Text.IsEmpty() && C->Transcript.Words.Num() == 0);
				TestEqual(TEXT("InferenceMs"), C->Transcript.InferenceMs, 0.f);
			});
		});

		LatentIt("AcceptedBoundaries", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// WP9S.5c2: the cap is now "> 600 s". Exactly 35.0 s (formerly the cap, now one window), exactly 600.0 s, rate 1000 and rate 384000 are accepted. Zero samples: the silence gate answers
			// with an empty success and no inference. An off-by-one cap or range rejects one of them with InvalidArgument.
			USpeechToTextPipeline* P = GetTinyPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			TArray<TSharedRef<FCallState>> Calls;
			for (int32 k = 0; k < 4; ++k) { Calls.Add(MakeShared<FCallState>()); }
			TArray<float> A; A.Init(0.f, 560000);
			Issue(P, MoveTemp(A), 16000, 1, Calls[0]);
			TArray<float> B; B.Init(0.f, 1000);
			Issue(P, MoveTemp(B), 1000, 1, Calls[1]);
			TArray<float> D; D.Init(0.f, 384000);
			Issue(P, MoveTemp(D), 384000, 1, Calls[2]);
			TArray<float> E; E.Init(0.f, 9600000);   // exactly 600.0 s: accepted (silence gate answers)
			Issue(P, MoveTemp(E), 16000, 1, Calls[3]);
			WaitCalls(this, Calls, Done, [this, Calls]()
			{
				const TCHAR* Names[4] = { TEXT("35.0 s at 16 kHz"), TEXT("rate 1000"), TEXT("rate 384000"), TEXT("600.0 s at 16 kHz") };
				for (int32 k = 0; k < 4; ++k)
				{
					TestTrue(*FString::Printf(TEXT("%s: success (%s)"), Names[k], *Calls[k]->Result.Message), Calls[k]->Result.bSuccess);
					TestTrue(*FString::Printf(TEXT("%s: empty transcript"), Names[k]), Calls[k]->Transcript.Text.IsEmpty());
					TestEqual(*FString::Printf(TEXT("%s: InferenceMs"), Names[k]), Calls[k]->Transcript.InferenceMs, 0.f);
				}
			});
		});

		It("NullCallbackWarnsAndRunsNothing", [this]()
		{
			// A null callback is a user bug: one Warning, nothing queued, no crash. (The exact warning text is an assumption,
			// modelled on UTextEmbeddingPipeline::EmbedAsync; see the report.)
			USpeechToTextPipeline* P = GetTinyPipeline(*this);
			if (!P) { return; }
			AddExpectedMessage(TEXT("USpeechToTextPipeline::Transcribe: null callback, nothing runs"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
			TArray<float> Samples = MakeSine(440.0, 16000, 16000, 0.1);
			P->Transcribe(MoveTemp(Samples), 16000, 1, FOnSpeechTranscribed());
			TestEqual(TEXT("GetNumPendingCalls (nothing runs)"), P->GetNumPendingCalls(), 0);
		});

		It("CreatorWrongTask", [this]()
		{
			// A TextEmbedding package must be refused by the speech creator with WrongTask, not crash and not return a pipeline.
			UPipelineModelPackage* Tiny = nullptr;
			FString Err;
			if (!GetSharedTinySttPackage(Tiny, Err)) { AddError(Err); return; }
			UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
			Holder().WrongTaskPackage.Reset(Pkg);
			Pkg->Task = EPipelineTask::TextEmbedding;
			Pkg->Model = Tiny->Model;
			Pkg->TokenizerJson = Tiny->TokenizerJson;
			Pkg->License = Tiny->License;
			Pkg->PreferredDevice = EPipelineDevice::CPU;
			FPipelineResult R;
			USpeechToTextPipeline* P = USpeechPipelineLibrary::CreateSpeechToTextPipeline(Pkg, R);
			TestNull(TEXT("pipeline"), P);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("Error WrongTask, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::WrongTask);
			TestFalse(TEXT("Message not empty"), R.Message.IsEmpty());
		});
	});

	// ----------------------------------------------------------------------------------------------------------------
	Describe("Async", [this]()
	{
		LatentIt("EightCallsInOneFrameMatchBlocking", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 6a: 8 Transcribe calls in one frame on 8 different clips: 8 callbacks, each exactly once, on the game
			// thread, results equal to the blocking ones (Text case-sensitive, plus the words). Results swapped between calls (a
			// shared session or tensor buffer) or a dropped callback fails here.
			const FSttGoldens* G = nullptr;
			if (!LoadGoldens(*this, G)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetSharedPipeline(*this, EPipelineDevice::CPU);
			if (!P) { Done.ExecuteIfBound(); return; }
			const TArray<const FSttClip*> Visible = G->VisibleClips();
			if (Visible.Num() < 37) { AddError(TEXT("fewer than 37 visible clips in the goldens")); Done.ExecuteIfBound(); return; }
			struct FItem { FString Id; TArray<float> Audio; FSpeechTranscript Ref; FPipelineResult RefR; };
			const TSharedRef<TArray<FItem>> Items = MakeShared<TArray<FItem>>();
			for (int32 Idx : { 1, 6, 11, 16, 21, 26, 31, 36 })
			{
				FItem Item;
				Item.Id = Visible[Idx]->Id;
				if (!LoadAudio(*this, *G, *Visible[Idx], Item.Audio)) { Done.ExecuteIfBound(); return; }
				Items->Add(MoveTemp(Item));
			}
			FDoneDelegate DoneCopy = Done;
			FAutomationTestBase* T = this;
			RunWorker(this, Done, [Items, P]()
			{
				for (FItem& Item : *Items) { Item.RefR = BlockingReference(P, Item.Id, Item.Audio, Item.Ref); }
			}, [T, Items, P, DoneCopy]()
			{
				TArray<TSharedRef<FCallState>> Calls;
				for (int32 i = 0; i < Items->Num(); ++i)
				{
					if (!(*Items)[i].RefR.bSuccess)
					{
						T->AddError(FString::Printf(TEXT("blocking reference for %s failed: %s"), *(*Items)[i].Id, *(*Items)[i].RefR.Message));
						DoneCopy.ExecuteIfBound();
						return;
					}
				}
				for (int32 i = 0; i < Items->Num(); ++i)
				{
					const TSharedRef<FCallState> C = MakeShared<FCallState>();
					Calls.Add(C);
					Issue(P, (*Items)[i].Audio, 16000, 1, C);   // all 8 inside this one game-thread task
				}
				WaitCalls(T, Calls, DoneCopy, [T, Items, Calls]()
				{
					for (int32 i = 0; i < Calls.Num(); ++i)
					{
						const FItem& Item = (*Items)[i];
						const FCallState& C = *Calls[i];
						T->TestTrue(*FString::Printf(TEXT("%s: success (%s)"), *Item.Id, *C.Result.Message), C.Result.bSuccess);
						T->TestTrue(*FString::Printf(TEXT("%s: async Text equals the blocking Text (case-sensitive) %s"), *Item.Id, *FirstDiff(Item.Ref.Text, C.Transcript.Text)),
							SameText(C.Transcript.Text, Item.Ref.Text));
						bool bWords = C.Transcript.Words.Num() == Item.Ref.Words.Num();
						for (int32 w = 0; bWords && w < Item.Ref.Words.Num(); ++w)
						{
							bWords = SameText(C.Transcript.Words[w].Word, Item.Ref.Words[w].Word)
								&& FMath::Abs(C.Transcript.Words[w].StartSeconds - Item.Ref.Words[w].StartSeconds) < 1e-4f
								&& FMath::Abs(C.Transcript.Words[w].EndSeconds - Item.Ref.Words[w].EndSeconds) < 1e-4f;
						}
						T->TestTrue(*FString::Printf(TEXT("%s: async words and times equal the blocking ones"), *Item.Id), bWords);
						T->TestTrue(*FString::Printf(TEXT("%s: InferenceMs > 0"), *Item.Id), C.Transcript.InferenceMs > 0.f);
					}
				});
			});
		});

		LatentIt("DestroyedWithThreeCallsInFlight", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 6b: a NEW pipeline gets 3 calls, then is MarkAsGarbage'd, released and garbage collected right away.
			// Every accepted call still gets exactly one callback (its result or Cancelled) and nothing crashes. A pipeline that
			// dropped the callback of an in-flight call, or whose background task / session touched the dead UObject, fails.
			const FSttGoldens* G = nullptr;
			if (!LoadGoldens(*this, G)) { Done.ExecuteIfBound(); return; }
			const TArray<const FSttClip*> Visible = G->VisibleClips();
			if (Visible.Num() < 3) { AddError(TEXT("fewer than 3 visible clips in the goldens")); Done.ExecuteIfBound(); return; }
			TArray<TArray<float>> Audio;
			for (int32 k = 0; k < 3; ++k)
			{
				TArray<float> A;
				if (!LoadAudio(*this, *G, *Visible[k * 5], A)) { Done.ExecuteIfBound(); return; }
				Audio.Add(MoveTemp(A));
			}
			USpeechToTextPipeline* P = NewPhononPipeline(*this, EPipelineDevice::CPU);
			if (!P) { Done.ExecuteIfBound(); return; }
			TStrongObjectPtr<USpeechToTextPipeline> Guard(P);
			TArray<TSharedRef<FCallState>> Calls;
			for (int32 k = 0; k < 3; ++k)
			{
				const TSharedRef<FCallState> C = MakeShared<FCallState>();
				Calls.Add(C);
				Issue(P, MoveTemp(Audio[k]), 16000, 1, C);
			}
			P->MarkAsGarbage();
			Guard.Reset();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			WaitCalls(this, Calls, Done, [this, Calls]()
			{
				int32 Ok = 0, Cancelled = 0;
				for (int32 k = 0; k < 3; ++k)
				{
					const FPipelineResult& R = Calls[k]->Result;
					if (R.bSuccess) { ++Ok; }
					else if (R.Error == EPipelineError::Cancelled) { ++Cancelled; }
					else { AddError(FString::Printf(TEXT("call %d: failed with error %d (%s), expected success or Cancelled"), k, (int32)R.Error, *R.Message)); }
				}
				AddInfo(FString::Printf(TEXT("destroyed pipeline: %d calls succeeded, %d cancelled"), Ok, Cancelled));
				TestEqual(TEXT("callbacks delivered (success or Cancelled)"), Ok + Cancelled, 3);
			});
		});

		LatentIt("ExecutorJobInterleavesWithLongDecode", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 7: a job on another (tiny) model, enqueued once the long clip's encoder has finished and its decode session
			// is running, must complete BEFORE the transcription does (the session yields every 32 steps and is requeued behind
			// the job). A decode that ran to the end inside one Step would make the tiny job finish second.
			const FSttGoldens* G = nullptr;
			if (!LoadGoldens(*this, G)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = GetSharedPipeline(*this, EPipelineDevice::CPU);
			if (!P) { Done.ExecuteIfBound(); return; }
			const FSttClip* Longest = nullptr;
			for (const FSttClip* C : G->VisibleClips())
			{
				if (!Longest || C->NumSamples > Longest->NumSamples) { Longest = C; }
			}
			if (!Longest) { AddError(TEXT("no visible clips in the goldens")); Done.ExecuteIfBound(); return; }
			TArray<float> Audio;
			if (!LoadAudio(*this, *G, *Longest, Audio)) { Done.ExecuteIfBound(); return; }
			if (Audio.Num() > 35 * 16000) { AddError(TEXT("longest visible clip exceeds 35 s")); Done.ExecuteIfBound(); return; }

			FString Err;
			UNNEModelData* TinyData = LoadTestModelData(TEXT("tiny_stt_encoder.onnx"), Err);
			if (!TinyData) { AddError(Err); Done.ExecuteIfBound(); return; }
			struct FInterleave
			{
				TSharedPtr<FPipelineNNEBackend> Backend;
				TSharedPtr<IPipelineModel> TinyModel;
				TSharedRef<FCallState> Transcribe = MakeShared<FCallState>();
				TSharedRef<FCallState> Tiny = MakeShared<FCallState>();
				TSharedPtr<FThreadSafeCounter> Seq = MakeShared<FThreadSafeCounter>();
				int32 Phase = 0;   // 0 waiting for the encoder job, 1 waiting for both callbacks
				FString Id;
			};
			const TSharedRef<FInterleave> S = MakeShared<FInterleave>();
			S->Id = Longest->Id;
			S->Backend = MakeShared<FPipelineNNEBackend>();
			FPipelineResult PR;
			S->TinyModel = S->Backend->PrepareModel(TinyData, EPipelineDevice::CPU, PR);
			if (!S->TinyModel.IsValid()) { AddError(FString::Printf(TEXT("could not prepare the tiny model: %s"), *PR.Message)); Done.ExecuteIfBound(); return; }

			const int64 Baseline = FPipelineExecutor::Get().GetStats().NumCompleted;
			Issue(P, MoveTemp(Audio), 16000, 1, S->Transcribe, S->Seq);

			const double Start = FPlatformTime::Seconds();
			FDoneDelegate DoneCopy = Done;
			FAutomationTestBase* T = this;
			const TWeakObjectPtr<USpeechToTextPipeline> WeakP(P);
			FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechInterleaveWait"), 0.02f, [T, S, Baseline, Start, DoneCopy, WeakP](float)
			{
				const double Now = FPlatformTime::Seconds();
				if (S->Phase == 0)
				{
					// NumCompleted alone may be observed before the encoder's Complete has run, i.e. before the decode
					// session is queued. The base decrements the pending-call counter in Finish right before Complete, on the
					// same game-thread task, and Complete enqueues the decode session synchronously. This ticker also runs on
					// the game thread, so once it sees the pending count at 0 (after the encoder job completed), Complete has
					// already run and the session is already queued: the tiny job goes in behind it.
					const USpeechToTextPipeline* PipelineNow = WeakP.Get();
					if (FPipelineExecutor::Get().GetStats().NumCompleted > Baseline && PipelineNow && PipelineNow->GetNumPendingCalls() == 0)
					{
						S->Phase = 1;
						TArray<float> Zeros;
						Zeros.Init(0.f, 4 * 8);
						TArray<FPipelineTensor> Inputs;
						Inputs.Add(FPipelineTensor::FromFloat({ 1, 4, 8 }, Zeros));
						const TSharedRef<FCallState> Tiny = S->Tiny;
						const TSharedPtr<FThreadSafeCounter> Seq = S->Seq;
						Tiny->bInCall = true;
						const FPipelineJobId Id = FPipelineExecutor::Get().Enqueue(S->TinyModel, MoveTemp(Inputs),
							[Tiny, Seq](const FPipelineResult& R, TArray<FPipelineTensor>&&)
							{
								if (!IsInGameThread()) { Tiny->bAllOnGameThread = false; }
								if (Tiny->bInCall) { Tiny->bAnyInCall = true; }
								if (Tiny->Count.Increment() == 1)
								{
									Tiny->Result = R;
									Tiny->FirstSeenSeconds = FPlatformTime::Seconds();
									Tiny->Order = Seq->Increment();
								}
							});
						Tiny->bInCall = false;
						if (Id == 0)
						{
							T->AddError(TEXT("executor refused the tiny job (shut down?)"));
							DoneCopy.ExecuteIfBound();
							return false;
						}
						// From here on WaitCalls observes both callbacks; it needs its own ticker so the logic stays in one place.
						WaitCalls(T, { S->Transcribe, S->Tiny }, DoneCopy, [T, S]()
						{
							T->TestTrue(*FString::Printf(TEXT("transcription of %s succeeded (%s)"), *S->Id, *S->Transcribe->Result.Message), S->Transcribe->Result.bSuccess);
							T->TestTrue(*FString::Printf(TEXT("tiny job succeeded (%s)"), *S->Tiny->Result.Message), S->Tiny->Result.bSuccess);
							const double GapMs = (S->Transcribe->FirstSeenSeconds - S->Tiny->FirstSeenSeconds) * 1000.0;
							T->AddInfo(FString::Printf(TEXT("tiny job finished %.0f ms before the transcription of %s (order tiny %d, transcription %d)"), GapMs, *S->Id, S->Tiny->Order, S->Transcribe->Order));
							T->TestTrue(*FString::Printf(TEXT("tiny job completed (order %d) before the decode finished (order %d)"), S->Tiny->Order, S->Transcribe->Order),
								S->Tiny->Order > 0 && S->Transcribe->Order > 0 && S->Tiny->Order < S->Transcribe->Order);
						});
						return false;
					}
				}
				if ((Now - Start) > CallTimeoutSeconds)
				{
					T->AddError(FString::Printf(TEXT("encoder job of the long clip never completed within %.0f s"), CallTimeoutSeconds));
					DoneCopy.ExecuteIfBound();
					return false;
				}
				return true;
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
