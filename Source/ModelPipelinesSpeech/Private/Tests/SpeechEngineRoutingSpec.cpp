// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9R.2: USpeechToTextPipeline routes Phonon packages to an ISpeechEngine (C81a). A FAKE Phonon backend/model/engine
// (SpeechFakeEngine.h) is registered with FModelPipelinesCoreModule::RegisterBackend in BeforeEach and removed in AfterEach.
// Written from the brief only. Text comparisons are CASE-SENSITIVE (FString == ignores case in Unreal).
// Validation messages are compared against the NNE path (tiny package, same call): "same errors/messages as NNE".

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Containers/Ticker.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "UObject/Package.h"
#include "Sound/SoundWave.h"
#include "HAL/PlatformTLS.h"
#include "HAL/ThreadSafeBool.h"
#include "Misc/ScopeLock.h"
#include "UObject/GarbageCollection.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineModelPackage.h"
#include "PipelineTypes.h"
#include "SpeechEngine.h"
#include "SpeechPipelineLibrary.h"
#include "SpeechToTextPipeline.h"
#include "SpeechTypes.h"
#include "SpeechFakeEngine.h"
#include "SpeechTestData.h"

#include <limits>

using namespace UE::ModelPipelines::SpeechTests;

namespace SpeechEngineSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(120);
	constexpr double WaitSeconds = 60.0;
	constexpr double GraceSeconds = 0.4;   // after the last callback: a duplicate callback would show up here

	struct FCall
	{
		int32 Fires = 0;
		bool bInsideIssue = true;      // true while Transcribe() has not returned yet
		bool bFiredInsideCall = false; // a callback that ran before Transcribe returned
		bool bOnGameThread = false;
		int32 Order = -1;
		FPipelineResult Result;
		FSpeechTranscript Transcript;
	};

	static int32 GOrder = 0;

	static void Issue(USpeechToTextPipeline* P, TArray<float> Samples, int32 Rate, int32 Channels, const TSharedRef<FCall>& C)
	{
		C->bInsideIssue = true;
		P->Transcribe(MoveTemp(Samples), Rate, Channels, [C](const FPipelineResult& R, const FSpeechTranscript& T)
		{
			if (C->bInsideIssue) { C->bFiredInsideCall = true; }
			C->bOnGameThread = IsInGameThread();
			C->Order = GOrder++;
			++C->Fires;
			C->Result = R;
			C->Transcript = T;
		});
		C->bInsideIssue = false;
	}

	/** Polls on the game thread until every call fired once (or 60 s), waits a grace period for duplicates, then OnAll. Never hangs. */
	static void WaitCalls(FAutomationTestBase* T, TArray<TSharedRef<FCall>> Calls, const FDoneDelegate& Done, TFunction<void()> OnAll)
	{
		const double Start = FPlatformTime::Seconds();
		const TSharedRef<double> AllAt = MakeShared<double>(-1.0);
		FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([T, Calls, Done, OnAll, Start, AllAt](float) -> bool
		{
			const double Now = FPlatformTime::Seconds();
			bool bAll = true;
			for (const TSharedRef<FCall>& C : Calls) { bAll &= (C->Fires >= 1); }
			if (bAll && *AllAt < 0.0) { *AllAt = Now; }
			if (bAll && Now - *AllAt >= GraceSeconds)
			{
				for (int32 i = 0; i < Calls.Num(); ++i)
				{
					T->TestEqual(*FString::Printf(TEXT("call %d: OnDone fired exactly once"), i), Calls[i]->Fires, 1);
					T->TestFalse(*FString::Printf(TEXT("call %d: OnDone not fired inside Transcribe()"), i), Calls[i]->bFiredInsideCall);
					T->TestTrue(*FString::Printf(TEXT("call %d: OnDone on the game thread"), i), Calls[i]->bOnGameThread);
				}
				OnAll();
				Done.ExecuteIfBound();
				return false;
			}
			if (Now - Start > WaitSeconds)
			{
				T->AddError(TEXT("timed out (60 s) waiting for the OnDone callbacks of the fake-engine calls"));
				Done.ExecuteIfBound();
				return false;
			}
			return true;
		}), 0.01f);
	}

	static void ScriptText(FFakeEngineState& S, const FString& Text, float InferenceMs)
	{
		FScopeLock L(&S.Lock);
		S.Script = [Text, InferenceMs](TConstArrayView<float>, int32, FSpeechTranscript& Out) -> FPipelineResult
		{
			Out.Text = Text;
			FSpeechWord W1; W1.Word = TEXT("hello"); W1.StartSeconds = 0.1f; W1.EndSeconds = 0.3f;
			FSpeechWord W2; W2.Word = TEXT("World"); W2.StartSeconds = 0.4f; W2.EndSeconds = 0.6f;
			Out.Words = { W1, W2 };
			Out.InferenceMs = InferenceMs;
			return FPipelineResult::Ok();
		};
	}

	static float MaxAbsDiff(TConstArrayView<float> A, TConstArrayView<float> B)
	{
		float Worst = 0.f;
		for (int32 i = 0; i < A.Num() && i < B.Num(); ++i) { Worst = FMath::Max(Worst, FMath::Abs(A[i] - B[i])); }
		return Worst;
	}

	enum class EVal : int32 { Rate999, Rate384001, ZeroChannels, NegativeChannels, Ragged, OneNaN, OneInf, Over600s, Count };

	static const TCHAR* ValName(EVal V)
	{
		switch (V)
		{
		case EVal::Rate999: return TEXT("Rate999");
		case EVal::Rate384001: return TEXT("Rate384001");
		case EVal::ZeroChannels: return TEXT("ZeroChannels");
		case EVal::NegativeChannels: return TEXT("NegativeChannels");
		case EVal::Ragged: return TEXT("RaggedLength");
		case EVal::OneNaN: return TEXT("OneNaN");
		case EVal::OneInf: return TEXT("OneInfinity");
		case EVal::Over600s: return TEXT("Over600s");
		default: return TEXT("?");
		}
	}

	static void BuildVal(EVal V, TArray<float>& S, int32& Rate, int32& Ch)
	{
		Rate = 16000;
		Ch = 1;
		MakeToneSamples(S, 16000, 16000, 440.0, 0.1);
		switch (V)
		{
		case EVal::Rate999: Rate = 999; break;
		case EVal::Rate384001: Rate = 384001; break;
		case EVal::ZeroChannels: Ch = 0; break;
		case EVal::NegativeChannels: Ch = -2; break;
		case EVal::Ragged: Ch = 2; S.SetNum(16001); break;
		case EVal::OneNaN: S[100] = std::numeric_limits<float>::quiet_NaN(); break;
		case EVal::OneInf: S[7] = std::numeric_limits<float>::infinity(); break;
		case EVal::Over600s: S.SetNumZeroed(601 * 16000); break;
		default: break;
		}
	}
}

using namespace SpeechEngineSpecPrivate;

BEGIN_DEFINE_SPEC(FSpeechEngineRoutingSpec, "Pipelines.Speech.Engine",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	TSharedPtr<FFakeEngineState, ESPMode::ThreadSafe> State;
	TSharedPtr<FFakePhononBackend> Backend;
	bool bRegistered = false;
	TStrongObjectPtr<UPipelineModelPackage> Pkg;
	TStrongObjectPtr<USpeechToTextPipeline> Pipe;

	void RegisterFake()
	{
		if (!bRegistered && Backend.IsValid())
		{
			FModelPipelinesCoreModule::Get().RegisterBackend(Backend.ToSharedRef());
			bRegistered = true;
		}
	}
	void UnregisterFake()
	{
		if (bRegistered && Backend.IsValid())
		{
			FModelPipelinesCoreModule::Get().UnregisterBackend(Backend.ToSharedRef());
			bRegistered = false;
		}
	}
END_DEFINE_SPEC(FSpeechEngineRoutingSpec)

void FSpeechEngineRoutingSpec::Define()
{
	BeforeEach([this]()
	{
		State = MakeShared<FFakeEngineState, ESPMode::ThreadSafe>();
		Backend = MakeShared<FFakePhononBackend>(State.ToSharedRef());
		RegisterFake();
		Pkg.Reset(MakeFakePhononPackage());
		FPipelineResult R;
		Pipe.Reset(USpeechPipelineLibrary::CreateSpeechToTextPipeline(Pkg.Get(), R));
		if (!Pipe.IsValid())
		{
			AddError(FString::Printf(TEXT("CreateSpeechToTextPipeline on the fake Phonon package failed: error %d '%s'"), (int32)R.Error, *R.Message));
		}
	});

	AfterEach([this]()
	{
		Pipe.Reset();
		Pkg.Reset();
		UnregisterFake();
		Backend.Reset();
		State.Reset();
	});

	// ------------------------------------------------------------------------------------------------------------------
	Describe("Routing", [this]()
	{
		It("InitializeFromPhononPackageNeedsNoTokenizerNoDecoder", [this]()
		{
			// Rule 1: success if the base class prepared the model. A pipeline that demanded a tokenizer/decoder would fail here.
			if (!Pipe.IsValid()) { return; }
			TestTrue(TEXT("IsReady"), Pipe->IsReady());
			TestEqual(TEXT("backend prepared exactly one model"), State->PrepareCount, 1);
			TestTrue(TEXT("backend got format Phonon"), State->LastPreparedFormat == EPipelineModelFormat::Phonon);
			TestTrue(*FString::Printf(TEXT("backend got the file model's path, got '%s'"), *State->LastPreparedPath),
				State->LastPreparedPath.Equals(TEXT("C:/Models/fake-phonon"), ESearchCase::CaseSensitive));
		});

		LatentIt("AsyncGivesScriptedTranscriptMonoAtInputRateOnWorker", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 1+2. 22.05 kHz mono (not 16k): a pipeline that resampled before the engine would hand it 16k samples.
			// A pipeline running the engine on the game thread (or twice) fails the thread/count checks.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			ScriptText(*State, TEXT("hello World"), 12.5f);
			TArray<float> In;
			MakeToneSamples(In, 11025, 22050, 440.0, 0.1);
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(Pipe.Get(), In, 22050, 1, C);
			WaitCalls(this, { C }, Done, [this, C, In]()
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C->Result.Message), C->Result.bSuccess);
				TestTrue(TEXT("Text == 'hello World' (case-sensitive)"), C->Transcript.Text.Equals(TEXT("hello World"), ESearchCase::CaseSensitive));
				TestEqual(TEXT("2 words"), C->Transcript.Words.Num(), 2);
				if (C->Transcript.Words.Num() == 2)
				{
					TestTrue(TEXT("word 0 text"), C->Transcript.Words[0].Word.Equals(TEXT("hello"), ESearchCase::CaseSensitive));
					TestTrue(TEXT("word 1 text keeps its case"), C->Transcript.Words[1].Word.Equals(TEXT("World"), ESearchCase::CaseSensitive));
					TestEqual(TEXT("word 0 start"), C->Transcript.Words[0].StartSeconds, 0.1f, 1e-6f);
					TestEqual(TEXT("word 1 end"), C->Transcript.Words[1].EndSeconds, 0.6f, 1e-6f);
				}
				TestEqual(TEXT("InferenceMs is the engine's"), C->Transcript.InferenceMs, 12.5f, 1e-6f);
				TestEqual(TEXT("AudioSeconds = 11025 / 22050"), C->Transcript.AudioSeconds, 0.5f, 1e-4f);
				TestTrue(TEXT("Result.Device is the model's device (CPU)"), C->Result.Device == EPipelineDevice::CPU);
				TestEqual(TEXT("engine called exactly once"), State->NumCalls(), 1);
				if (State->NumCalls() == 1)
				{
					const FFakeEngineCall Call = State->GetCall(0);
					TestEqual(TEXT("engine rate is the INPUT rate"), Call.SampleRate, 22050);
					TestEqual(TEXT("engine got all input frames"), Call.Samples.Num(), In.Num());
					const float Worst = MaxAbsDiff(Call.Samples, In);
					TestTrue(*FString::Printf(TEXT("engine samples equal input, worst diff %g"), Worst), Worst <= 1e-7f && Call.Samples.Num() == In.Num());
					TestTrue(TEXT("engine ran on a worker, not the game thread"), Call.ThreadId != FPlatformTLS::GetCurrentThreadId());
					TestTrue(TEXT("model was initialized before the engine call"), Call.bModelInitializedAtCall);
				}
				TestTrue(TEXT("model Initialize was called"), State->InitializeCount >= 1);
			});
		});

		LatentIt("StereoInputIsMixedToMonoAtInputRate", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 1: stereo 44.1 kHz -> engine gets mono 44.1 kHz = (L+R)/2 (C73). Asymmetric channels catch "took left only"
			// (error 0.15 here) and "summed instead of averaged" (error 0.3).
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			const int32 Frames = 22050;
			TArray<float> L, R, Inter;
			MakeToneSamples(L, Frames, 44100, 300.0, 0.2);
			MakeToneSamples(R, Frames, 44100, 700.0, 0.1);
			TArray<float> Expected;
			for (int32 i = 0; i < Frames; ++i) { Inter.Add(L[i]); Inter.Add(R[i]); Expected.Add(0.5f * (L[i] + R[i])); }
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(Pipe.Get(), Inter, 44100, 2, C);
			WaitCalls(this, { C }, Done, [this, C, Expected]()
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C->Result.Message), C->Result.bSuccess);
				TestEqual(TEXT("AudioSeconds = frames / rate = 0.5"), C->Transcript.AudioSeconds, 0.5f, 1e-4f);
				TestEqual(TEXT("engine called once"), State->NumCalls(), 1);
				if (State->NumCalls() == 1)
				{
					const FFakeEngineCall Call = State->GetCall(0);
					TestEqual(TEXT("engine rate 44100"), Call.SampleRate, 44100);
					TestEqual(TEXT("engine sample count = frames (mono)"), Call.Samples.Num(), Expected.Num());
					const float Worst = MaxAbsDiff(Call.Samples, Expected);
					TestTrue(*FString::Printf(TEXT("engine samples == (L+R)/2, worst diff %g (tol 1e-6)"), Worst),
						Call.Samples.Num() == Expected.Num() && Worst <= 1e-6f);
				}
			});
		});

		LatentIt("BlockingGivesSameTranscriptOffTheGameThread", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 1 (blocking variant): TranscribeBlocking from a pool thread; the engine still runs on the executor worker,
			// i.e. neither on the game thread nor on the calling pool thread.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			ScriptText(*State, TEXT("hello World"), 3.f);
			TArray<float> In;
			MakeToneSamples(In, 8000, 8000, 440.0, 0.1);
			struct FBlk { FThreadSafeBool bDone; FPipelineResult R; FSpeechTranscript T; uint32 CallerThread = 0; };
			const TSharedRef<FBlk, ESPMode::ThreadSafe> B = MakeShared<FBlk, ESPMode::ThreadSafe>();
			USpeechToTextPipeline* P = Pipe.Get();
			Async(EAsyncExecution::ThreadPool, [B, P, In]()
			{
				B->CallerThread = FPlatformTLS::GetCurrentThreadId();
				B->R = P->TranscribeBlocking(In, 8000, 1, B->T);
				B->bDone = true;
			});
			const double Start = FPlatformTime::Seconds();
			FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([this, B, Done, Start](float) -> bool
			{
				if (!B->bDone)
				{
					if (FPlatformTime::Seconds() - Start > WaitSeconds)
					{
						AddError(TEXT("timed out waiting for TranscribeBlocking"));
						Done.ExecuteIfBound();
						return false;
					}
					return true;
				}
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *B->R.Message), B->R.bSuccess);
				TestTrue(TEXT("Text"), B->T.Text.Equals(TEXT("hello World"), ESearchCase::CaseSensitive));
				TestEqual(TEXT("AudioSeconds = 8000 / 8000"), B->T.AudioSeconds, 1.0f, 1e-4f);
				TestEqual(TEXT("InferenceMs"), B->T.InferenceMs, 3.f, 1e-6f);
				TestEqual(TEXT("engine called once"), State->NumCalls(), 1);
				if (State->NumCalls() == 1)
				{
					const FFakeEngineCall Call = State->GetCall(0);
					TestEqual(TEXT("engine rate is the input rate"), Call.SampleRate, 8000);
					TestTrue(TEXT("engine not on the game thread"), Call.ThreadId != FPlatformTLS::GetCurrentThreadId());
					TestTrue(TEXT("engine not on the blocking caller's thread"), Call.ThreadId != B->CallerThread);
				}
				Done.ExecuteIfBound();
				return false;
			}), 0.01f);
		});
	});

	// ------------------------------------------------------------------------------------------------------------------
	Describe("SilenceGate", [this]()
	{
		LatentIt("DigitalSilenceDoesNotCallTheEngine", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 3: all zeros -> success, empty text, InferenceMs 0, no engine call.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			ScriptText(*State, TEXT("must not appear"), 9.f);
			TArray<float> Z;
			Z.SetNumZeroed(16000);
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(Pipe.Get(), Z, 16000, 1, C);
			WaitCalls(this, { C }, Done, [this, C]()
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C->Result.Message), C->Result.bSuccess);
				TestTrue(TEXT("Text empty"), C->Transcript.Text.IsEmpty());
				TestEqual(TEXT("no words"), C->Transcript.Words.Num(), 0);
				TestEqual(TEXT("InferenceMs 0"), C->Transcript.InferenceMs, 0.f);
				TestEqual(TEXT("AudioSeconds still the input duration"), C->Transcript.AudioSeconds, 1.0f, 1e-4f);
				TestEqual(TEXT("engine calls"), State->NumCalls(), 0);
			});
		});

		LatentIt("QuietBelowThresholdGatedLoudAboveIsNot", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// DC 1e-4 has RMS -80 dBFS (< default -60): gated. DC 0.05 is -26 dBFS: reaches the engine. A gate that never fires,
			// or always fires, fails one of the two.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			TestEqual(TEXT("precondition: default SilenceThresholdDb -60"), Pkg->Speech.SilenceThresholdDb, -60.f);
			TArray<float> Quiet, Loud;
			Quiet.Init(1e-4f, 16000);
			Loud.Init(0.05f, 16000);
			const TSharedRef<FCall> CQ = MakeShared<FCall>();
			Issue(Pipe.Get(), Quiet, 16000, 1, CQ);
			WaitCalls(this, { CQ }, FDoneDelegate::CreateLambda([this, Loud, Done]()
			{
				TestEqual(TEXT("quiet: no engine call"), State->NumCalls(), 0);
				const TSharedRef<FCall> CL = MakeShared<FCall>();
				Issue(Pipe.Get(), Loud, 16000, 1, CL);
				WaitCalls(this, { CL }, Done, [this, CL]()
				{
					TestTrue(TEXT("loud: bSuccess"), CL->Result.bSuccess);
					TestEqual(TEXT("loud: exactly one engine call"), State->NumCalls(), 1);
				});
			}), [this, CQ]()
			{
				TestTrue(TEXT("quiet: bSuccess"), CQ->Result.bSuccess);
				TestTrue(TEXT("quiet: empty"), CQ->Transcript.Text.IsEmpty());
			});
		});

		LatentIt("GateWorksOnTheMonoMixNotOnChannels", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Rule 2: the gate is on the MONO signal. L = +0.5, R = -0.5 is loud per channel but cancels to digital silence in the
			// mix: gated (a per-channel gate would call the engine).
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			TArray<float> Inter;
			for (int32 i = 0; i < 16000; ++i) { Inter.Add(0.5f); Inter.Add(-0.5f); }
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(Pipe.Get(), Inter, 16000, 2, C);
			WaitCalls(this, { C }, Done, [this, C]()
			{
				TestTrue(TEXT("bSuccess"), C->Result.bSuccess);
				TestTrue(TEXT("Text empty"), C->Transcript.Text.IsEmpty());
				TestEqual(TEXT("no engine call"), State->NumCalls(), 0);
			});
		});

		LatentIt("EmptySamplesSucceedEmptyWithoutEngine", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// C75: empty input -> success with an empty transcript (not an error), no engine call.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(Pipe.Get(), TArray<float>(), 16000, 1, C);
			WaitCalls(this, { C }, Done, [this, C]()
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C->Result.Message), C->Result.bSuccess);
				TestTrue(TEXT("Text empty"), C->Transcript.Text.IsEmpty());
				TestEqual(TEXT("no engine call"), State->NumCalls(), 0);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------------------
	Describe("Validation", [this]()
	{
		for (int32 I = 0; I < (int32)EVal::Count; ++I)
		{
			const EVal V = (EVal)I;
			LatentIt(ValName(V), LatentLimit, [this, V](const FDoneDelegate& Done)
			{
				// Acceptance 3: same error and message as the NNE path (tiny package gets the identical call), no engine call, empty transcript.
				if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
				FString Err;
				USpeechToTextPipeline* Nne = GetSharedTinySttPipeline(Err);
				if (!Nne) { AddError(Err); Done.ExecuteIfBound(); return; }
				TArray<float> S1, S2;
				int32 Rate = 0, Ch = 0;
				BuildVal(V, S1, Rate, Ch);
				S2 = S1;
				const TSharedRef<FCall> CNne = MakeShared<FCall>();
				const TSharedRef<FCall> CPh = MakeShared<FCall>();
				Issue(Nne, MoveTemp(S1), Rate, Ch, CNne);
				Issue(Pipe.Get(), MoveTemp(S2), Rate, Ch, CPh);
				WaitCalls(this, { CNne, CPh }, Done, [this, CNne, CPh]()
				{
					TestFalse(TEXT("precondition: NNE path rejects the call"), CNne->Result.bSuccess);
					TestFalse(TEXT("Phonon path rejects the call"), CPh->Result.bSuccess);
					TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d"), (int32)CPh->Result.Error), CPh->Result.Error == EPipelineError::InvalidArgument);
					TestTrue(*FString::Printf(TEXT("same message as NNE. NNE: '%s' Phonon: '%s'"), *CNne->Result.Message, *CPh->Result.Message),
						CNne->Result.Message.Equals(CPh->Result.Message, ESearchCase::CaseSensitive));
					TestTrue(TEXT("transcript empty"), CPh->Transcript.Text.IsEmpty() && CPh->Transcript.Words.Num() == 0);
					TestEqual(TEXT("no engine call"), State->NumCalls(), 0);
				});
			});
		}

		It("NullCallbackWarnsAndNothingRuns", [this]()
		{
			// C75: null callback -> Warning (phrase "null callback", as in the existing speech specs), nothing queued, no engine call.
			if (!Pipe.IsValid()) { return; }
			TArray<float> In;
			MakeToneSamples(In, 16000, 16000, 440.0, 0.1);
			AddExpectedMessagePlain(TEXT("null callback"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
			Pipe->Transcribe(In, 16000, 1, FOnSpeechTranscribed());
			TestEqual(TEXT("GetNumPendingCalls"), Pipe->GetNumPendingCalls(), 0);
			TestEqual(TEXT("no engine call"), State->NumCalls(), 0);
		});
	});

	// ------------------------------------------------------------------------------------------------------------------
	Describe("EngineErrors", [this]()
	{
		LatentIt("EngineFailureIsReportedWithEnginePrefix", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 4: the engine's error code is kept, the message is "engine: " + the engine's message, transcript empty.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			{
				FScopeLock L(&State->Lock);
				State->Script = [](TConstArrayView<float>, int32, FSpeechTranscript& Out) -> FPipelineResult
				{
					Out.Text = TEXT("partial garbage");
					return FPipelineResult::Fail(EPipelineError::InferenceFailed, TEXT("kaboom 42"));
				};
			}
			TArray<float> In;
			MakeToneSamples(In, 16000, 16000, 440.0, 0.1);
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(Pipe.Get(), In, 16000, 1, C);
			WaitCalls(this, { C }, Done, [this, C]()
			{
				TestFalse(TEXT("bSuccess"), C->Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error InferenceFailed, got %d"), (int32)C->Result.Error), C->Result.Error == EPipelineError::InferenceFailed);
				TestTrue(*FString::Printf(TEXT("message starts with 'engine: ' (case-sensitive), got '%s'"), *C->Result.Message),
					C->Result.Message.StartsWith(TEXT("engine: "), ESearchCase::CaseSensitive));
				TestTrue(TEXT("message contains the engine's text"), C->Result.Message.Contains(TEXT("kaboom 42"), ESearchCase::CaseSensitive));
				TestTrue(TEXT("transcript empty on failure"), C->Transcript.Text.IsEmpty() && C->Transcript.Words.Num() == 0);
				TestEqual(TEXT("engine called once"), State->NumCalls(), 1);
			});
		});

		LatentIt("ModelWithoutEngineIsRuntimeUnavailable", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 4: AsSpeechEngine() == nullptr -> RuntimeUnavailable "the model has no speech engine" (substring "no speech engine").
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			{
				FScopeLock L(&State->Lock);
				State->bNoEngine = true;
			}
			TArray<float> In;
			MakeToneSamples(In, 16000, 16000, 440.0, 0.1);
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Issue(Pipe.Get(), In, 16000, 1, C);
			WaitCalls(this, { C }, Done, [this, C]()
			{
				TestFalse(TEXT("bSuccess"), C->Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error RuntimeUnavailable, got %d (%s)"), (int32)C->Result.Error, *C->Result.Message),
					C->Result.Error == EPipelineError::RuntimeUnavailable);
				TestTrue(*FString::Printf(TEXT("message names the missing engine, got '%s'"), *C->Result.Message),
					C->Result.Message.Contains(TEXT("no speech engine"), ESearchCase::IgnoreCase));
				TestTrue(TEXT("transcript empty"), C->Transcript.Text.IsEmpty());
				TestEqual(TEXT("engine calls"), State->NumCalls(), 0);
			});
		});

		LatentIt("NoPhononBackendFailsInitializeAndCallsAreNotReady", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 7: with no Phonon backend registered, InitializeFromPackage fails RuntimeUnavailable (WP9R.1) and the unready
			// pipeline answers a Transcribe with the C75 not-ready InvalidArgument (same message as a never-initialized pipeline).
			// (Lead fix 2026-10-18: shared call state + WaitCalls; the not-ready failure is delivered by a game-thread task.)
			UnregisterFake();
			if (FModelPipelinesCoreModule::Get().FindBackend(EPipelineModelFormat::Phonon) != nullptr)
			{
				// Lead 2026-10-18: the paid plugin's real Phonon backend is registered in this project; the "no backend" path can't be reached.
				AddInfo(TEXT("SKIPPED: a Phonon backend is registered by another plugin (ModelPipelines Generative)"));
				Done.Execute();
				return;
			}
			UPipelineModelPackage* P2 = MakeFakePhononPackage();
			const TStrongObjectPtr<UPipelineModelPackage> KeepPkg(P2);
			FPipelineResult R;
			USpeechToTextPipeline* P = USpeechPipelineLibrary::CreateSpeechToTextPipeline(P2, R);
			TestTrue(*FString::Printf(TEXT("create fails RuntimeUnavailable, got %d (%s)"), (int32)R.Error, *R.Message),
				!R.bSuccess && R.Error == EPipelineError::RuntimeUnavailable);
			TSharedRef<TStrongObjectPtr<USpeechToTextPipeline>> Raw = MakeShared<TStrongObjectPtr<USpeechToTextPipeline>>(NewObject<USpeechToTextPipeline>(GetTransientPackage()));
			const FPipelineResult Init = (*Raw)->InitializeFromPackage(P2);
			TestFalse(TEXT("InitializeFromPackage fails"), Init.bSuccess);
			TestTrue(TEXT("Error RuntimeUnavailable"), Init.Error == EPipelineError::RuntimeUnavailable);
			TestFalse(TEXT("IsReady false"), (*Raw)->IsReady());

			TSharedRef<TStrongObjectPtr<USpeechToTextPipeline>> Never = MakeShared<TStrongObjectPtr<USpeechToTextPipeline>>(NewObject<USpeechToTextPipeline>(GetTransientPackage()));
			TArray<float> In;
			MakeToneSamples(In, 16000, 16000, 440.0, 0.1);
			const TSharedRef<FCall> Got = MakeShared<FCall>();
			const TSharedRef<FCall> Ref = MakeShared<FCall>();
			Issue(Raw->Get(), In, 16000, 1, Got);
			Issue(Never->Get(), In, 16000, 1, Ref);
			WaitCalls(this, { Got, Ref }, Done, [this, Got, Ref, Raw, Never]()
			{
				TestTrue(*FString::Printf(TEXT("unready Transcribe: InvalidArgument, got %d (%s)"), (int32)Got->Result.Error, *Got->Result.Message),
					!Got->Result.bSuccess && Got->Result.Error == EPipelineError::InvalidArgument);
				TestTrue(*FString::Printf(TEXT("same message as a never-initialized pipeline: '%s' vs '%s'"), *Got->Result.Message, *Ref->Result.Message),
					Got->Result.Message.Equals(Ref->Result.Message, ESearchCase::CaseSensitive));
				TestEqual(TEXT("no engine call"), State->NumCalls(), 0);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------------------
	Describe("SoundWave", [this]()
	{
		LatentIt("MonoGeneratedWaveReachesEngineAsDecodedMono", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 5: int16 mono 16 kHz wave -> engine gets int16/32768 at 16000 Hz.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			TArray<float> Src;
			MakeToneSamples(Src, 12000, 16000, 500.0, 0.3);
			TArray<int16> I16;
			FloatToInt16(Src, I16);
			const TStrongObjectPtr<USoundWave> Wave(MakeGeneratedSoundWave(I16, 16000, 1));
			const TSharedRef<FCall> C = MakeShared<FCall>();
			C->bInsideIssue = true;
			Pipe->TranscribeSoundWave(Wave.Get(), [C](const FPipelineResult& R, const FSpeechTranscript& T)
			{
				if (C->bInsideIssue) { C->bFiredInsideCall = true; }
				C->bOnGameThread = IsInGameThread();
				++C->Fires; C->Result = R; C->Transcript = T;
			});
			C->bInsideIssue = false;
			WaitCalls(this, { C }, Done, [this, C, I16, Wave]()
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C->Result.Message), C->Result.bSuccess);
				TestEqual(TEXT("AudioSeconds = 12000/16000"), C->Transcript.AudioSeconds, 0.75f, 1e-3f);
				TestEqual(TEXT("engine called once"), State->NumCalls(), 1);
				if (State->NumCalls() == 1)
				{
					const FFakeEngineCall Call = State->GetCall(0);
					TestEqual(TEXT("engine rate"), Call.SampleRate, 16000);
					TestEqual(TEXT("engine sample count"), Call.Samples.Num(), I16.Num());
					float Worst = 0.f;
					for (int32 i = 0; i < I16.Num() && i < Call.Samples.Num(); ++i) { Worst = FMath::Max(Worst, FMath::Abs(Call.Samples[i] - (float)I16[i] / 32768.f)); }
					TestTrue(*FString::Printf(TEXT("engine samples == int16/32768, worst diff %g (tol 1e-5)"), Worst), Worst <= 1e-5f && Call.Samples.Num() == I16.Num());
				}
			});
		});

		LatentIt("StereoGeneratedWave24kIsMixedToMono", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Stereo wave at 24 kHz: engine gets mono 24 kHz, (L+R)/2/32768. Catches a wave path that hard-codes 16 kHz or passes interleaved audio.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			TArray<int16> Inter;
			const int32 Frames = 9000;
			for (int32 i = 0; i < Frames; ++i)
			{
				Inter.Add((int16)(8000 + (i % 100) * 10));
				Inter.Add((int16)(-2000 - (i % 50) * 20));
			}
			const TStrongObjectPtr<USoundWave> Wave(MakeGeneratedSoundWave(Inter, 24000, 2));
			const TSharedRef<FCall> C = MakeShared<FCall>();
			Pipe->TranscribeSoundWave(Wave.Get(), [C](const FPipelineResult& R, const FSpeechTranscript& T)
			{
				C->bOnGameThread = IsInGameThread();
				++C->Fires; C->Result = R; C->Transcript = T;
			});
			C->bInsideIssue = false;
			WaitCalls(this, { C }, Done, [this, C, Inter, Frames, Wave]()
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C->Result.Message), C->Result.bSuccess);
				TestEqual(TEXT("engine called once"), State->NumCalls(), 1);
				if (State->NumCalls() == 1)
				{
					const FFakeEngineCall Call = State->GetCall(0);
					TestEqual(TEXT("engine rate is the wave's 24000"), Call.SampleRate, 24000);
					TestEqual(TEXT("engine frames"), Call.Samples.Num(), Frames);
					float Worst = 0.f;
					for (int32 i = 0; i < Frames && i < Call.Samples.Num(); ++i)
					{
						const float Exp = 0.5f * ((float)Inter[2 * i] + (float)Inter[2 * i + 1]) / 32768.f;
						Worst = FMath::Max(Worst, FMath::Abs(Call.Samples[i] - Exp));
					}
					TestTrue(*FString::Printf(TEXT("engine samples == (L+R)/2/32768, worst diff %g (tol 1e-5)"), Worst), Worst <= 1e-5f && Call.Samples.Num() == Frames);
				}
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------------------
	Describe("Lifetime", [this]()
	{
		LatentIt("DestroyedWithThreeCallsQueuedEachFiresOnce", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 6a: pipeline destroyed while 3 calls are queued behind a slow engine. Each call still gets exactly one OnDone
			// (success or Cancelled) on the game thread; nothing touches the dead UObject.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			{
				FScopeLock L(&State->Lock);
				State->DelaySeconds = 0.15;
			}
			TArray<TSharedRef<FCall>> Calls;
			for (int32 k = 0; k < 3; ++k)
			{
				TArray<float> In;
				MakeToneSamples(In, 8000 + 100 * k, 16000, 440.0, 0.1);
				const TSharedRef<FCall> C = MakeShared<FCall>();
				Calls.Add(C);
				Issue(Pipe.Get(), MoveTemp(In), 16000, 1, C);
			}
			Pipe->MarkAsGarbage();
			Pipe.Reset();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			WaitCalls(this, Calls, Done, [this, Calls]()
			{
				int32 Ok = 0, Cancelled = 0;
				for (int32 k = 0; k < Calls.Num(); ++k)
				{
					const FPipelineResult& R = Calls[k]->Result;
					if (R.bSuccess) { ++Ok; }
					else if (R.Error == EPipelineError::Cancelled) { ++Cancelled; }
					else { AddError(FString::Printf(TEXT("call %d: error %d (%s); expected success or Cancelled"), k, (int32)R.Error, *R.Message)); }
				}
				AddInfo(FString::Printf(TEXT("destroyed pipeline: %d succeeded, %d cancelled"), Ok, Cancelled));
				TestEqual(TEXT("3 callbacks (success or Cancelled)"), Ok + Cancelled, 3);
				TestTrue(TEXT("engine called at most 3 times"), State->NumCalls() <= 3);
			});
		});

		LatentIt("EightCallsInOneFrameEachGetsItsOwnResult", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Acceptance 6b: 8 calls with different lengths; the engine echoes the sample count it received. Every call's transcript
			// must carry ITS OWN count (a pipeline that shared a result slot, or swapped callbacks, mixes them), each fires once.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			{
				FScopeLock L(&State->Lock);
				State->DelaySeconds = 0.01;
				State->Script = [](TConstArrayView<float> Mono, int32 Rate, FSpeechTranscript& Out) -> FPipelineResult
				{
					Out.Text = FString::Printf(TEXT("n=%d r=%d"), Mono.Num(), Rate);
					Out.InferenceMs = 1.f;
					return FPipelineResult::Ok();
				};
			}
			TArray<TSharedRef<FCall>> Calls;
			TArray<int32> Lens;
			for (int32 k = 0; k < 8; ++k)
			{
				const int32 Len = 8000 + 1000 * k;
				Lens.Add(Len);
				TArray<float> In;
				MakeToneSamples(In, Len, 16000, 300.0 + 50.0 * k, 0.1);
				const TSharedRef<FCall> C = MakeShared<FCall>();
				Calls.Add(C);
				Issue(Pipe.Get(), MoveTemp(In), 16000, 1, C);
			}
			WaitCalls(this, Calls, Done, [this, Calls, Lens]()
			{
				for (int32 k = 0; k < 8; ++k)
				{
					TestTrue(*FString::Printf(TEXT("call %d bSuccess (%s)"), k, *Calls[k]->Result.Message), Calls[k]->Result.bSuccess);
					const FString Want = FString::Printf(TEXT("n=%d r=16000"), Lens[k]);
					TestTrue(*FString::Printf(TEXT("call %d text '%s' == '%s'"), k, *Calls[k]->Transcript.Text, *Want),
						Calls[k]->Transcript.Text.Equals(Want, ESearchCase::CaseSensitive));
					TestEqual(*FString::Printf(TEXT("call %d AudioSeconds"), k), Calls[k]->Transcript.AudioSeconds, (float)Lens[k] / 16000.f, 1e-4f);
				}
				TestEqual(TEXT("engine called 8 times"), State->NumCalls(), 8);
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
