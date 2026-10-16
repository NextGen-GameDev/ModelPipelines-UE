// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9S.5b / contract C76: USpeechToTextPipeline::TranscribeSoundWave. Numbers are checked against the Python goldens
// (<dev>/goldens/stt/stt_goldens.json) and against TranscribeBlocking on the golden float audio, never against the implementation.
// Missing goldens, models or imported sound-wave assets FAIL (never skip) with the path and the way to make them.
// Text comparisons are CASE-SENSITIVE (FString == / TestEqual(FString) ignore case in Unreal).

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "HAL/PlatformTime.h"
#include "Sound/SoundWave.h"
#include "Sound/SoundWaveProcedural.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineModelPackage.h"
#include "PipelineTypes.h"
#include "SpeechPipelineLibrary.h"
#include "SpeechToTextPipeline.h"
#include "SpeechTypes.h"
#include "SpeechTestData.h"

using namespace UE::ModelPipelines::SpeechTests;

namespace SpeechSoundWaveSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(1500);
	constexpr double WaitSeconds = 600.0;
	constexpr double SettleSeconds = 0.4;      // after the last expected callback: a duplicate callback would show up here
	constexpr double FrameSeconds = 0.08;

	static const TCHAR* const ClipLong = TEXT("1089-134691-0025");   // 8.005 s
	static const TCHAR* const ClipAdpcm = TEXT("1221-135767-0010");   // 8.2 s
	static const TCHAR* const AssetPcm = TEXT("/Game/Tests/Speech/SW_Vis_Mono16k_PCM");
	static const TCHAR* const AssetBink = TEXT("/Game/Tests/Speech/SW_Vis_Stereo44k_Bink_Stream");
	static const TCHAR* const AssetAdpcm = TEXT("/Game/Tests/Speech/SW_Vis_Mono48k_ADPCM");

	// ---------------------------------------------------------------- waiting (never blocks the game thread)

	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Next)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechSoundWaveSpecWait"), 0.02f, [T, Cond, What, Done, Next, Start](float)
		{
			if (Cond())
			{
				Next();
				return false;
			}
			if ((FPlatformTime::Seconds() - Start) > WaitSeconds)
			{
				T->AddError(FString::Printf(TEXT("timed out after %.0f s waiting for: %s"), WaitSeconds, *What));
				Done.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	static void Settle(double Secs, TFunction<void()> Next)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechSoundWaveSpecSettle"), 0.02f, [Start, Secs, Next](float)
		{
			if ((FPlatformTime::Seconds() - Start) >= Secs)
			{
				Next();
				return false;
			}
			return true;
		});
	}

	static TSharedRef<TFuture<void>> RunBg(TFunction<void()> Work)
	{
		return MakeShared<TFuture<void>>(Async(EAsyncExecution::ThreadPool, MoveTemp(Work)));
	}

	// ---------------------------------------------------------------- calls

	struct FCall
	{
		FPipelineResult Result;
		FSpeechTranscript Transcript;
		int32 Count = 0;
		bool bGameThread = true;
		bool bInsideCall = false;      // set by Issue around TranscribeSoundWave
		bool bFiredInsideCall = false;
		double CallMs = 0.0;           // game-thread time of the TranscribeSoundWave call itself
	};

	static void Issue(USpeechToTextPipeline* P, USoundWave* W, const TSharedRef<FCall>& C)
	{
		C->bInsideCall = true;
		const double T0 = FPlatformTime::Seconds();
		P->TranscribeSoundWave(W, [C](const FPipelineResult& R, const FSpeechTranscript& Tr)
		{
			++C->Count;
			C->Result = R;
			C->Transcript = Tr;
			C->bGameThread = IsInGameThread();
			if (C->bInsideCall) { C->bFiredInsideCall = true; }
		});
		C->CallMs = (FPlatformTime::Seconds() - T0) * 1000.0;
		C->bInsideCall = false;
	}

	/** Waits for every call, settles (to catch a duplicate callback), checks "exactly once, game thread, not inside the call", runs Check, then Done. */
	static void WaitCalls(FAutomationTestBase* T, TArray<TSharedRef<FCall>> Calls, FDoneDelegate Done, TFunction<void()> Check)
	{
		WaitFor(T, [Calls]()
		{
			for (const TSharedRef<FCall>& C : Calls) { if (C->Count == 0) { return false; } }
			return true;
		}, TEXT("TranscribeSoundWave callback(s) never fired"), Done, [T, Calls, Done, Check]()
		{
			Settle(SettleSeconds, [T, Calls, Done, Check]()
			{
				for (int32 i = 0; i < Calls.Num(); ++i)
				{
					T->TestEqual(*FString::Printf(TEXT("call %d: callback count (exactly once)"), i), Calls[i]->Count, 1);
					T->TestTrue(*FString::Printf(TEXT("call %d: delivered on the game thread"), i), Calls[i]->bGameThread);
					T->TestFalse(*FString::Printf(TEXT("call %d: delivered inside TranscribeSoundWave (must be a later tick)"), i), Calls[i]->bFiredInsideCall);
				}
				Check();
				Done.ExecuteIfBound();
			});
		});
	}

	// ---------------------------------------------------------------- goldens, references

	static bool SameText(const FString& A, const FString& B) { return A.Equals(B, ESearchCase::CaseSensitive); }

	struct FRef
	{
		const FSttClip* Clip = nullptr;
		TArray<float> Audio;
		FPipelineResult R;
		FSpeechTranscript Tr;          // TranscribeBlocking(golden audio, 16000, 1)
	};

	static bool LoadClip(FAutomationTestBase& T, const FString& Id, const FSttClip*& OutClip, TArray<float>& OutAudio)
	{
		const FSttGoldens* G = nullptr;
		FString Err;
		if (!LoadSttGoldens(G, Err)) { T.AddError(Err); return false; }
		OutClip = G->Find(Id);
		if (!OutClip) { T.AddError(FString::Printf(TEXT("clip '%s' not in stt_goldens.json. Regenerate with Tools/make_stt_goldens.py"), *Id)); return false; }
		if (!LoadClipAudio(*G, *OutClip, OutAudio, Err)) { T.AddError(Err); return false; }
		return true;
	}

	static USpeechToTextPipeline* CpuPipeline(FAutomationTestBase& T)
	{
		FString Err;
		USpeechToTextPipeline* P = GetSharedCpuSttPipeline(Err);
		if (!P) { T.AddError(Err); }
		return P;
	}

	/** Loads the clip and computes the blocking reference on a pool thread; Next(Ref) runs on the game thread when it is ready and succeeded. */
	static void WithReference(FAutomationTestBase* T, const FString& Id, FDoneDelegate Done, TFunction<void(const TSharedRef<FRef>&, USpeechToTextPipeline*)> Next)
	{
		const TSharedRef<FRef> Ref = MakeShared<FRef>();
		if (!LoadClip(*T, Id, Ref->Clip, Ref->Audio)) { Done.ExecuteIfBound(); return; }
		USpeechToTextPipeline* P = CpuPipeline(*T);
		if (!P) { Done.ExecuteIfBound(); return; }
		const TSharedRef<TFuture<void>> Fut = RunBg([Ref, P]()
		{
			Ref->R = P->TranscribeBlocking(Ref->Audio, 16000, 1, Ref->Tr);
		});
		WaitFor(T, [Fut]() { return Fut->IsReady(); }, TEXT("blocking reference call"), Done, [T, Ref, P, Done, Next]()
		{
			if (!Ref->R.bSuccess) { T->AddError(FString::Printf(TEXT("blocking reference failed: %s"), *Ref->R.Message)); Done.ExecuteIfBound(); return; }
			if (!SameText(Ref->Tr.Text, Ref->Clip->OnnxText))
			{
				T->AddError(FString::Printf(TEXT("blocking reference text differs from the golden onnx.text (precondition, WP9S.5a): '%s' vs '%s'"), *Ref->Tr.Text, *Ref->Clip->OnnxText));
			}
			Next(Ref, P);
		});
	}

	static USoundWave* LoadWaveAsset(FAutomationTestBase& T, const TCHAR* Path)
	{
		USoundWave* W = LoadObject<USoundWave>(nullptr, Path);
		if (!W)
		{
			T.AddError(FString::Printf(TEXT("Sound wave asset '%s' not found. The lead imports it from the C71 golden WAVs (see <dev>/testdata/stt/soundwave/manifest.json); it lives in the test project Content."), Path));
		}
		return W;
	}

	/** Exact parity with the blocking call and the golden text. */
	static void CheckExact(FAutomationTestBase& T, const FCall& C, const FRef& Ref, const FString& Label)
	{
		T.TestTrue(*FString::Printf(TEXT("%s: bSuccess (%s)"), *Label, *C.Result.Message), C.Result.bSuccess);
		T.TestTrue(*FString::Printf(TEXT("%s: Text equals golden onnx.text (case-sensitive). got '%s' golden '%s'"), *Label, *C.Transcript.Text, *Ref.Clip->OnnxText), SameText(C.Transcript.Text, Ref.Clip->OnnxText));
		T.TestTrue(*FString::Printf(TEXT("%s: Text equals the blocking call's Text"), *Label), SameText(C.Transcript.Text, Ref.Tr.Text));
		if (!T.TestEqual(*FString::Printf(TEXT("%s: word count"), *Label), C.Transcript.Words.Num(), Ref.Tr.Words.Num())) { return; }
		int32 Bad = 0;
		double Worst = 0.0;
		for (int32 i = 0; i < Ref.Tr.Words.Num(); ++i)
		{
			const FSpeechWord& G = C.Transcript.Words[i];
			const FSpeechWord& W = Ref.Tr.Words[i];
			if (!SameText(G.Word, W.Word) || G.StartSeconds != W.StartSeconds || G.EndSeconds != W.EndSeconds)
			{
				++Bad;
				Worst = FMath::Max(Worst, (double)FMath::Max(FMath::Abs(G.StartSeconds - W.StartSeconds), FMath::Abs(G.EndSeconds - W.EndSeconds)));
			}
		}
		T.TestEqual(*FString::Printf(TEXT("%s: words differing from TranscribeBlocking (exact equality; worst time diff %.3g s)"), *Label, Worst), Bad, 0);
		const double Expected = (double)Ref.Clip->NumSamples / 16000.0;
		T.TestTrue(*FString::Printf(TEXT("%s: AudioSeconds %.5f within 1e-3 of %.5f"), *Label, C.Transcript.AudioSeconds, Expected), FMath::Abs((double)C.Transcript.AudioSeconds - Expected) <= 1e-3);
	}

	/** Levenshtein over words (case-sensitive) with the aligned (got, golden) index pairs of the best path. */
	static int32 AlignWords(const TArray<FString>& A, const TArray<FString>& B, TArray<TPair<int32, int32>>& OutPairs)
	{
		const int32 N = A.Num(), M = B.Num();
		TArray<int32> D;
		D.SetNumZeroed((N + 1) * (M + 1));
		auto At = [&D, M](int32 i, int32 j) -> int32& { return D[i * (M + 1) + j]; };
		for (int32 i = 0; i <= N; ++i) { At(i, 0) = i; }
		for (int32 j = 0; j <= M; ++j) { At(0, j) = j; }
		for (int32 i = 1; i <= N; ++i)
		{
			for (int32 j = 1; j <= M; ++j)
			{
				const int32 Sub = At(i - 1, j - 1) + (SameText(A[i - 1], B[j - 1]) ? 0 : 1);
				At(i, j) = FMath::Min3(Sub, At(i - 1, j) + 1, At(i, j - 1) + 1);
			}
		}
		int32 i = N, j = M;
		while (i > 0 || j > 0)
		{
			if (i > 0 && j > 0 && At(i, j) == At(i - 1, j - 1) + (SameText(A[i - 1], B[j - 1]) ? 0 : 1))
			{
				OutPairs.Add(TPair<int32, int32>(i - 1, j - 1));
				--i; --j;
			}
			else if (i > 0 && At(i, j) == At(i - 1, j) + 1) { --i; }
			else { --j; }
		}
		return At(N, M);
	}

	/** Lossy / resampled wave: at most one word differs, every aligned word starts within 0.16 s of the golden. */
	static void CheckNear(FAutomationTestBase& T, const FCall& C, const FSttClip& Clip, const FString& Label)
	{
		T.TestTrue(*FString::Printf(TEXT("%s: bSuccess (%s)"), *Label, *C.Result.Message), C.Result.bSuccess);
		TArray<FString> Got, Gold;
		for (const FSpeechWord& W : C.Transcript.Words) { Got.Add(W.Word); }
		for (const FSttGoldWord& W : Clip.Words) { Gold.Add(W.Word); }
		TArray<TPair<int32, int32>> Pairs;
		const int32 Dist = AlignWords(Got, Gold, Pairs);
		T.TestTrue(*FString::Printf(TEXT("%s: word edit distance %d <= 1. got '%s' / golden '%s'"), *Label, Dist, *C.Transcript.Text, *Clip.OnnxText), Dist <= 1);
		double WorstStart = 0.0;
		for (const TPair<int32, int32>& P : Pairs)
		{
			const double ES = Clip.Words[P.Value].StartFrame * FrameSeconds;
			WorstStart = FMath::Max(WorstStart, FMath::Abs((double)C.Transcript.Words[P.Key].StartSeconds - ES));
		}
		T.TestTrue(*FString::Printf(TEXT("%s: worst aligned StartSeconds error %.3f s <= 0.16 s"), *Label, WorstStart), WorstStart <= 0.16);
		const double Expected = (double)Clip.NumSamples / 16000.0;
		T.TestTrue(*FString::Printf(TEXT("%s: AudioSeconds %.5f within 1e-2 of %.5f"), *Label, C.Transcript.AudioSeconds, Expected), FMath::Abs((double)C.Transcript.AudioSeconds - Expected) <= 1e-2);
		T.AddInfo(FString::Printf(TEXT("%s: transcript '%s' (golden '%s'), edit distance %d, worst start error %.3f s"), *Label, *C.Transcript.Text, *Clip.OnnxText, Dist, WorstStart));
	}

	// ---------------------------------------------------------------- scenarios

	static void ScenarioAssetExact(FAutomationTestBase* T, const TCHAR* AssetPath, FDoneDelegate Done)
	{
		// Lossless PCM at 16 kHz mono: the decoded floats must be the golden samples, so the transcript is bit-for-bit the blocking one.
		// A decoder that mis-scaled int16 (/32767, no scaling), skipped frames or resampled would shift a word or a time.
		WithReference(T, ClipLong, Done, [T, AssetPath, Done](const TSharedRef<FRef>& Ref, USpeechToTextPipeline* P)
		{
			USoundWave* W = LoadWaveAsset(*T, AssetPath);
			if (!W) { Done.ExecuteIfBound(); return; }
			const TStrongObjectPtr<USoundWave> Keep(W);
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(P, W, C);
			T->TestEqual(TEXT("nothing delivered inside the call"), C->Count, 0);
			WaitCalls(T, { C }, Done, [T, C, Ref, Keep]() { CheckExact(*T, *C, *Ref, TEXT("PCM asset")); });
		});
	}

	static void ScenarioAssetNear(FAutomationTestBase* T, const TCHAR* AssetPath, const TCHAR* ClipId, FDoneDelegate Done)
	{
		const FSttClip* Clip = nullptr;
		TArray<float> Unused;
		if (!LoadClip(*T, ClipId, Clip, Unused)) { Done.ExecuteIfBound(); return; }
		USpeechToTextPipeline* P = CpuPipeline(*T);
		if (!P) { Done.ExecuteIfBound(); return; }
		USoundWave* W = LoadWaveAsset(*T, AssetPath);
		if (!W) { Done.ExecuteIfBound(); return; }
		const TStrongObjectPtr<USoundWave> Keep(W);
		const TSharedRef<FCall> C = MakeShared<FCall>();
		Issue(P, W, C);
		WaitCalls(T, { C }, Done, [T, C, Clip, Keep]() { CheckNear(*T, *C, *Clip, FString(TEXT("lossy asset"))); });
	}

	enum class EErr : uint8 { NullSound, Procedural, TooLong, Blank, NotReady, Count };

	static const TCHAR* ErrName(EErr E)
	{
		switch (E)
		{
		case EErr::NullSound: return TEXT("NullSoundFailsInvalidArgument");
		case EErr::Procedural: return TEXT("ProceduralWaveFailsInvalidArgument");
		case EErr::TooLong: return TEXT("Generated36sWaveFailsBeforeDecodeWithAtMost35s");
		case EErr::Blank: return TEXT("BlankWaveFailsPreprocessFailed");
		default: return TEXT("PipelineNotReadyFailsInvalidArgument");
		}
	}

	static void ScenarioError(FAutomationTestBase* T, EErr Case, FDoneDelegate Done)
	{
		USpeechToTextPipeline* P = nullptr;
		if (Case == EErr::NotReady)
		{
			// Never initialised: IsReady() false, so the call must fail instead of crashing.
			P = NewObject<USpeechToTextPipeline>(GetTransientPackage());
			T->TestFalse(TEXT("precondition: pipeline is not ready"), P->IsReady());
		}
		else
		{
			FString Err;
			P = GetSharedTinySttPipeline(Err);
			if (!P) { T->AddError(Err); Done.ExecuteIfBound(); return; }
		}
		const TStrongObjectPtr<USpeechToTextPipeline> KeepP(P);
		TStrongObjectPtr<USoundWave> Keep;
		FString Needle;
		EPipelineError WantError = EPipelineError::InvalidArgument;
		switch (Case)
		{
		case EErr::NullSound:
			Needle = TEXT("Sound is null");
			break;
		case EErr::Procedural:
			Keep.Reset(NewObject<USoundWaveProcedural>(GetTransientPackage()));
			Needle = TEXT("procedural sound waves have no stored audio");
			break;
		case EErr::TooLong:
		{
			// 36 s of silence: a decode-first implementation would just answer with an empty success, so only the length check can fail it.
			TArray<int16> Silence;
			Silence.SetNumZeroed(36 * 16000);
			Keep.Reset(MakeGeneratedSoundWave(Silence, 16000, 1));
			Needle = TEXT("at most 35 s");
			break;
		}
		case EErr::Blank:
		{
			// No samples, no rate, no channels: the engine reader refuses it with LogAudio warnings (any number, allowed).
			T->AddExpectedMessagePlain(TEXT("Cannot create FSoundWaveProxyReader"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			T->AddExpectedMessagePlain(TEXT("LPCM data failed to load for sound"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			T->AddExpectedMessagePlain(TEXT("because there is no source LPCM data"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			T->AddExpectedMessagePlain(TEXT("failed to parse as a wave"), ELogVerbosity::Error, EAutomationExpectedMessageFlags::Contains, -1);
			Keep.Reset(NewObject<USoundWave>(GetTransientPackage()));
			Needle = TEXT("could not decode sound wave");
			WantError = EPipelineError::PreprocessFailed;
			break;
		}
		default:
		{
			TArray<int16> Silence;
			Silence.SetNumZeroed(16000);
			Keep.Reset(MakeGeneratedSoundWave(Silence, 16000, 1));
			break;
		}
		}
		const TSharedRef<FCall> C = MakeShared<FCall>();
		Issue(P, Keep.Get(), C);
		T->TestEqual(TEXT("nothing delivered inside the call"), C->Count, 0);
		WaitCalls(T, { C }, Done, [T, C, Needle, WantError, Keep, KeepP]()
		{
			T->TestFalse(TEXT("bSuccess"), C->Result.bSuccess);
			T->TestTrue(*FString::Printf(TEXT("Error %d, expected %d (%s)"), (int32)C->Result.Error, (int32)WantError, *C->Result.Message), C->Result.Error == WantError);
			T->TestFalse(TEXT("Message not empty"), C->Result.Message.IsEmpty());
			if (!Needle.IsEmpty())
			{
				T->TestTrue(*FString::Printf(TEXT("message contains '%s'; got '%s'"), *Needle, *C->Result.Message), C->Result.Message.Contains(Needle, ESearchCase::CaseSensitive));
			}
			T->TestTrue(TEXT("transcript is empty on failure"), C->Transcript.Text.IsEmpty() && C->Transcript.Words.Num() == 0);
		});
	}
}

BEGIN_DEFINE_SPEC(FSpeechSoundWaveSpec, "Pipelines.Speech.SoundWave",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
END_DEFINE_SPEC(FSpeechSoundWaveSpec)

void FSpeechSoundWaveSpec::Define()
{
	using namespace SpeechSoundWaveSpecPrivate;

	Describe("Lossless", [this]()
	{
		LatentIt("Mono16kPcmAssetEqualsBlockingAndGolden", LatentLimit, [this](const FDoneDelegate& Done)
		{
			ScenarioAssetExact(this, AssetPcm, Done);
		});

		LatentIt("GeneratedWaveFromGoldenInt16SamplesEqualsBlockingAndGolden", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// RawPCMData path (no asset): int16 of the golden samples. Same exact equality as the PCM asset.
			WithReference(this, ClipLong, Done, [this, Done](const TSharedRef<FRef>& Ref, USpeechToTextPipeline* P)
			{
				TArray<int16> I16;
				FloatToInt16(Ref->Audio, I16);
				const TStrongObjectPtr<USoundWave> Keep(MakeGeneratedSoundWave(I16, 16000, 1));
				const TSharedRef<FCall> C = MakeShared<FCall>();
				Issue(P, Keep.Get(), C);
				WaitCalls(this, { C }, Done, [this, C, Ref, Keep]() { CheckExact(*this, *C, *Ref, TEXT("generated wave")); });
			});
		});
	});

	Describe("Lossy", [this]()
	{
		LatentIt("Stereo44kBinkStreamingNearGolden", LatentLimit, [this](const FDoneDelegate& Done)
		{
			ScenarioAssetNear(this, AssetBink, ClipLong, Done);
		});

		LatentIt("Mono48kAdpcmNearGolden", LatentLimit, [this](const FDoneDelegate& Done)
		{
			ScenarioAssetNear(this, AssetAdpcm, ClipAdpcm, Done);
		});
	});

	Describe("Errors", [this]()
	{
		for (int32 I = 0; I < (int32)EErr::Count; ++I)
		{
			const EErr Case = (EErr)I;
			LatentIt(ErrName(Case), LatentLimit, [this, Case](const FDoneDelegate& Done)
			{
				ScenarioError(this, Case, Done);
			});
		}

		It("NullCallbackWarnsAndRunsNothing", [this]()
		{
			// One Warning, nothing queued, no crash. The exact warning text is not in the contract: match the phrase "null callback".
			FString Err;
			USpeechToTextPipeline* P = GetSharedTinySttPipeline(Err);
			if (!P) { AddError(Err); return; }
			TArray<int16> Silence;
			Silence.SetNumZeroed(16000);
			const TStrongObjectPtr<USoundWave> Keep(MakeGeneratedSoundWave(Silence, 16000, 1));
			AddExpectedMessagePlain(TEXT("null callback"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
			P->TranscribeSoundWave(Keep.Get(), FOnSpeechTranscribed());
			TestEqual(TEXT("GetNumPendingCalls (nothing runs)"), P->GetNumPendingCalls(), 0);
		});
	});

	Describe("Lifetime", [this]()
	{
		LatentIt("TransientWaveCollectedRightAfterCallStillGivesTheTranscript", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Every reference to the generated wave is dropped and GC runs before the call completes. The pipeline must have copied the
			// samples (or pinned the wave) on the game thread; reading the dead UObject later would crash or give garbage text.
			WithReference(this, ClipLong, Done, [this, Done](const TSharedRef<FRef>& Ref, USpeechToTextPipeline* P)
			{
				TArray<int16> I16;
				FloatToInt16(Ref->Audio, I16);
				TStrongObjectPtr<USoundWave> Keep(MakeGeneratedSoundWave(I16, 16000, 1));
				const TSharedRef<FCall> C = MakeShared<FCall>();
				Issue(P, Keep.Get(), C);
				Keep.Reset();
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
				WaitCalls(this, { C }, Done, [this, C, Ref]() { CheckExact(*this, *C, *Ref, TEXT("collected wave")); });
			});
		});

		LatentIt("PipelineDestroyedRightAfterCallGivesCancelledOrTranscriptExactlyOnce", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// A NEW tiny pipeline (the shared Phonon one is ~4 GB) gets one call on a silent wave, then is destroyed and collected.
			// Silence is answered by the gate without inference, so success with an empty transcript is a valid outcome; the other is Cancelled.
			UPipelineModelPackage* Pkg = nullptr;
			FString Err;
			if (!GetSharedTinySttPackage(Pkg, Err)) { AddError(Err); Done.ExecuteIfBound(); return; }
			FPipelineResult R;
			USpeechToTextPipeline* P = USpeechPipelineLibrary::CreateSpeechToTextPipeline(Pkg, R);
			if (!P) { AddError(FString::Printf(TEXT("CreateSpeechToTextPipeline failed: %s"), *R.Message)); Done.ExecuteIfBound(); return; }
			TStrongObjectPtr<USpeechToTextPipeline> Guard(P);
			TArray<int16> Silence;
			Silence.SetNumZeroed(2 * 16000);
			const TStrongObjectPtr<USoundWave> Keep(MakeGeneratedSoundWave(Silence, 16000, 1));
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(P, Keep.Get(), C);
			P->MarkAsGarbage();
			Guard.Reset();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			WaitCalls(this, { C }, Done, [this, C]()
			{
				const bool bOk = C->Result.bSuccess || C->Result.Error == EPipelineError::Cancelled;
				TestTrue(*FString::Printf(TEXT("success or Cancelled; got error %d (%s)"), (int32)C->Result.Error, *C->Result.Message), bOk);
				AddInfo(FString::Printf(TEXT("destroyed pipeline: %s"), C->Result.bSuccess ? TEXT("completed") : TEXT("Cancelled")));
			});
		});
	});

	Describe("Concurrency", [this]()
	{
		LatentIt("EightCallsOnBinkStreamingWaveInOneFrameGiveIdenticalTranscripts", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 8 calls on one streaming asset in one game-thread task: 8 callbacks, each exactly once, identical transcripts (a shared decode
			// buffer or a streaming chunk race would differ), and each call returns quickly (decode runs off the game thread).
			const FSttClip* Clip = nullptr;
			TArray<float> Unused;
			if (!LoadClip(*this, ClipLong, Clip, Unused)) { Done.ExecuteIfBound(); return; }
			USpeechToTextPipeline* P = CpuPipeline(*this);
			if (!P) { Done.ExecuteIfBound(); return; }
			USoundWave* W = LoadWaveAsset(*this, AssetBink);
			if (!W) { Done.ExecuteIfBound(); return; }
			const TStrongObjectPtr<USoundWave> Keep(W);
			TArray<TSharedRef<FCall>> Calls;
			for (int32 k = 0; k < 8; ++k)
			{
				const TSharedRef<FCall> C = MakeShared<FCall>();
				Calls.Add(C);
				Issue(P, W, C);
			}
			WaitCalls(this, Calls, Done, [this, Calls, Clip, Keep]()
			{
				double MaxMs = 0.0;
				for (const TSharedRef<FCall>& C : Calls) { MaxMs = FMath::Max(MaxMs, C->CallMs); }
				AddInfo(FString::Printf(TEXT("slowest TranscribeSoundWave call on the game thread: %.2f ms"), MaxMs));
				TestTrue(*FString::Printf(TEXT("each call returns in < 20 ms on the game thread; slowest %.2f ms"), MaxMs), MaxMs < 20.0);
				CheckNear(*this, *Calls[0], *Clip, TEXT("call 0"));
				for (int32 k = 1; k < Calls.Num(); ++k)
				{
					TestTrue(*FString::Printf(TEXT("call %d: bSuccess (%s)"), k, *Calls[k]->Result.Message), Calls[k]->Result.bSuccess);
					TestTrue(*FString::Printf(TEXT("call %d: Text identical to call 0 (case-sensitive). '%s' vs '%s'"), k, *Calls[k]->Transcript.Text, *Calls[0]->Transcript.Text),
						SameText(Calls[k]->Transcript.Text, Calls[0]->Transcript.Text));
					bool bSame = Calls[k]->Transcript.Words.Num() == Calls[0]->Transcript.Words.Num();
					for (int32 w = 0; bSame && w < Calls[0]->Transcript.Words.Num(); ++w)
					{
						bSame = SameText(Calls[k]->Transcript.Words[w].Word, Calls[0]->Transcript.Words[w].Word)
							&& Calls[k]->Transcript.Words[w].StartSeconds == Calls[0]->Transcript.Words[w].StartSeconds
							&& Calls[k]->Transcript.Words[w].EndSeconds == Calls[0]->Transcript.Words[w].EndSeconds;
					}
					TestTrue(*FString::Printf(TEXT("call %d: words and times identical to call 0"), k), bSame);
				}
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
