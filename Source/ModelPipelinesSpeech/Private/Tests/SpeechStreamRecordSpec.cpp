// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9R.9t: developer recording of real-time sessions (contract C85). Written from docs/briefs/WP9R.9t.md only.
// Uses the fake Phonon engine (SpeechFakeEngine.h) and synthetic tone bursts. Files are written on a background task, so every wait
// is a ticker poll with a timeout (the game thread never blocks). Text comparisons are CASE-SENSITIVE (FString == ignores case).

#include "CoreMinimal.h"   // lead: defines UE_BUILD_SHIPPING / WITH_DEV_AUTOMATION_TESTS before the #if (C4668 in BuildPlugin)

#if WITH_DEV_AUTOMATION_TESTS && !UE_BUILD_SHIPPING

#include "Misc/AutomationTest.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/FileManager.h"
#include "HAL/IConsoleManager.h"
#include "HAL/PlatformTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineModelPackage.h"
#include "PipelineTypes.h"
#include "SpeechPipelineLibrary.h"
#include "SpeechStream.h"
#include "SpeechToTextPipeline.h"
#include "SpeechTypes.h"
#include "SpeechFakeEngine.h"
#include "SpeechStreamTestReceiver.h"
#include "SpeechTestData.h"

using namespace UE::ModelPipelines::SpeechTests;

namespace SpeechStreamRecordSpecPrivate
{
	constexpr int32 Frame = 320;   // 20 ms at 16 kHz
	constexpr double Wait = 60.0;
	constexpr double FileWait = 10.0;
	const FTimespan LatentLimit = FTimespan::FromSeconds(200);

	struct FSeg { int32 Tone = 0; int32 Gap = 0; };

	static void BuildAudio(TArray<float>& A, const TArray<FSeg>& Segs, int32 LeadFrames = 25, double Amp = 0.2)
	{
		A.Reset();
		A.AddZeroed(LeadFrames * Frame);
		for (const FSeg& S : Segs)
		{
			TArray<float> T;
			MakeToneSamples(T, S.Tone * Frame, 16000, 440.0, Amp);
			A.Append(T);
			A.AddZeroed(S.Gap * Frame);
		}
	}

	/** Turns 16 kHz mono audio into 44.1 kHz stereo of the same duration (nearest-sample lookup, both channels equal). */
	static void ToStereo44k(const TArray<float>& Mono16k, TArray<float>& Out)
	{
		const int64 Frames = (int64)Mono16k.Num() * 44100 / 16000;
		Out.Reset();
		Out.Reserve((int32)(Frames * 2));
		for (int64 i = 0; i < Frames; ++i)
		{
			const int64 Src = FMath::Min<int64>(i * 16000 / 44100, Mono16k.Num() - 1);
			Out.Add(Mono16k[(int32)Src]);
			Out.Add(Mono16k[(int32)Src]);
		}
	}

	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, const FString& What, const FDoneDelegate& Done, TFunction<void()> Next, double Timeout)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechStreamRecordWait"), 0.01f, [T, Cond, What, Done, Next, Start, Timeout](float) -> bool
		{
			if (Cond()) { Next(); return false; }
			if (FPlatformTime::Seconds() - Start > Timeout)
			{
				T->AddError(FString::Printf(TEXT("timed out after %.0f s waiting for: %s"), Timeout, *What));
				Done.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	static void Settle(double Seconds, TFunction<void()> Next)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechStreamRecordSettle"), 0.01f, [Seconds, Next, Start](float) -> bool
		{
			if (FPlatformTime::Seconds() - Start < Seconds) { return true; }
			Next();
			return false;
		});
	}

	/** Pushes interleaved Audio in ChunkFloats pieces from the game thread, PerTick chunks per 50 ms tick, then OnPushed. */
	static void PushPaced(USpeechStream* Stream, const TArray<float>& Audio, int32 Rate, int32 Channels, int32 ChunkFloats, int32 PerTick, TFunction<void()> OnPushed)
	{
		const TSharedRef<TArray<float>> A = MakeShared<TArray<float>>(Audio);
		const TSharedRef<int32> Pos = MakeShared<int32>(0);
		const TWeakObjectPtr<USpeechStream> Weak(Stream);
		FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Weak, A, Pos, Rate, Channels, ChunkFloats, PerTick, OnPushed](float) -> bool
		{
			USpeechStream* S = Weak.Get();
			for (int32 k = 0; k < PerTick && *Pos < A->Num(); ++k)
			{
				const int32 N = FMath::Min(ChunkFloats, A->Num() - *Pos);
				if (S) { S->PushAudio(TConstArrayView<float>(A->GetData() + *Pos, N), Rate, Channels); }
				*Pos += N;
			}
			if (*Pos >= A->Num()) { OnPushed(); return false; }
			return true;
		}), 0.05f);
	}

	// ---- recordings directory ----
	static FString RecDir()
	{
		return FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("ModelPipelines"), TEXT("SpeechRecordings"));
	}

	static void ListFiles(TSet<FString>& Out)
	{
		TArray<FString> Found;
		IFileManager::Get().FindFiles(Found, *FPaths::Combine(RecDir(), TEXT("*.*")), true, false);
		Out.Append(Found);
	}

	struct FWav
	{
		bool bValid = false;
		FString Error;
		int32 Format = 0, Channels = 0, Rate = 0, Bits = 0;
		int32 DataBytes = 0;
		int32 NumSamples() const { return Bits > 0 ? DataBytes / (Bits / 8) : 0; }
	};

	static uint32 Le32(const TArray<uint8>& B, int32 O) { return (uint32)B[O] | ((uint32)B[O + 1] << 8) | ((uint32)B[O + 2] << 16) | ((uint32)B[O + 3] << 24); }
	static uint32 Le16(const TArray<uint8>& B, int32 O) { return (uint32)B[O] | ((uint32)B[O + 1] << 8); }

	static FWav ParseWav(const FString& Path)
	{
		FWav W;
		TArray<uint8> B;
		if (!FFileHelper::LoadFileToArray(B, *Path)) { W.Error = TEXT("cannot read file"); return W; }
		if (B.Num() < 12 || FMemory::Memcmp(B.GetData(), "RIFF", 4) != 0 || FMemory::Memcmp(B.GetData() + 8, "WAVE", 4) != 0) { W.Error = TEXT("not RIFF/WAVE"); return W; }
		bool bFmt = false, bData = false;
		int32 O = 12;
		while (O + 8 <= B.Num())
		{
			const int32 Size = (int32)Le32(B, O + 4);
			if (FMemory::Memcmp(B.GetData() + O, "fmt ", 4) == 0 && O + 8 + 16 <= B.Num())
			{
				W.Format = (int32)Le16(B, O + 8);
				W.Channels = (int32)Le16(B, O + 10);
				W.Rate = (int32)Le32(B, O + 12);
				W.Bits = (int32)Le16(B, O + 22);
				bFmt = true;
			}
			else if (FMemory::Memcmp(B.GetData() + O, "data", 4) == 0)
			{
				// Tolerate a data size that overruns the file only if it is the last chunk; the real size is then what is present.
				W.DataBytes = FMath::Min(Size, B.Num() - (O + 8));
				if (W.DataBytes != Size) { W.Error = FString::Printf(TEXT("data chunk claims %d bytes, file holds %d"), Size, W.DataBytes); return W; }
				bData = true;
			}
			O += 8 + Size + (Size & 1);
		}
		W.bValid = bFmt && bData;
		if (!W.bValid) { W.Error = TEXT("missing fmt or data chunk"); }
		return W;
	}

	static TSharedPtr<FJsonObject> ParseJson(const FString& Path, FString& Err)
	{
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path)) { Err = TEXT("cannot read file"); return nullptr; }
		TSharedPtr<FJsonObject> Root;
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root) || !Root.IsValid()) { Err = TEXT("not valid JSON"); return nullptr; }
		return Root;
	}

	static IConsoleVariable* Cvar(const TCHAR* Name) { return IConsoleManager::Get().FindConsoleVariable(Name); }
}

using namespace SpeechStreamRecordSpecPrivate;

BEGIN_DEFINE_SPEC(FSpeechStreamRecordSpec, "Pipelines.Speech.Stream.Record",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	TSharedPtr<FFakeEngineState, ESPMode::ThreadSafe> State;
	TSharedPtr<FFakePhononBackend> Backend;
	bool bRegistered = false;
	TStrongObjectPtr<UPipelineModelPackage> Pkg;
	TStrongObjectPtr<USpeechToTextPipeline> Pipe;
	TStrongObjectPtr<USpeechStream> Stream;
	TStrongObjectPtr<USpeechStreamTestReceiver> Recv;
	TSet<FString> Before;
	bool bSavedRec = false;
	int32 OldRecord = 0;
	float OldMax = 1200.f;

	void SetupFake()
	{
		State = MakeShared<FFakeEngineState, ESPMode::ThreadSafe>();
		State->bStreaming = true;
		Backend = MakeShared<FFakePhononBackend>(State.ToSharedRef());
		FModelPipelinesCoreModule::Get().RegisterBackend(Backend.ToSharedRef());
		bRegistered = true;
		Pkg.Reset(MakeFakePhononPackage());
		FPipelineResult R;
		Pipe.Reset(USpeechPipelineLibrary::CreateSpeechToTextPipeline(Pkg.Get(), R));
		if (!Pipe.IsValid()) { AddError(FString::Printf(TEXT("CreateSpeechToTextPipeline on the fake package failed: %s"), *R.Message)); }
	}

	void TeardownFake()
	{
		Stream.Reset();
		Recv.Reset();
		Pipe.Reset();
		Pkg.Reset();
		if (bRegistered && Backend.IsValid()) { FModelPipelinesCoreModule::Get().UnregisterBackend(Backend.ToSharedRef()); bRegistered = false; }
		Backend.Reset();
		State.Reset();
	}

	void SetCvars(int32 Record, float MaxSeconds)
	{
		if (IConsoleVariable* R = Cvar(TEXT("mp.Speech.RecordStreams"))) { R->Set(Record, ECVF_SetByCode); }
		if (IConsoleVariable* M = Cvar(TEXT("mp.Speech.RecordMaxSeconds"))) { M->Set(MaxSeconds, ECVF_SetByCode); }
	}

	/** Files that were not in the directory when the test began. */
	void NewFiles(TArray<FString>& Wavs, TArray<FString>& Jsons)
	{
		TSet<FString> Now;
		ListFiles(Now);
		for (const FString& F : Now)
		{
			if (Before.Contains(F)) { continue; }
			if (F.EndsWith(TEXT(".wav"))) { Wavs.Add(F); }
			else if (F.EndsWith(TEXT(".json"))) { Jsons.Add(F); }
		}
		Wavs.Sort();
		Jsons.Sort();
	}

	bool PairExists()
	{
		TArray<FString> W, J;
		NewFiles(W, J);
		return W.Num() >= 1 && J.Num() >= 1;
	}

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
		Stream.Reset(St);
		Recv.Reset(NewObject<USpeechStreamTestReceiver>(GetTransientPackage()));
		St->OnSpeechStarted.AddDynamic(Recv.Get(), &USpeechStreamTestReceiver::HandleStarted);
		St->OnPartial.AddDynamic(Recv.Get(), &USpeechStreamTestReceiver::HandlePartial);
		St->OnFinal.AddDynamic(Recv.Get(), &USpeechStreamTestReceiver::HandleFinal);
		St->OnSpeechEnded.AddDynamic(Recv.Get(), &USpeechStreamTestReceiver::HandleEnded);
		St->OnStopped.AddDynamic(Recv.Get(), &USpeechStreamTestReceiver::HandleStopped);
		return St;
	}

	/**
	 * Opens a stream (cvars set by the caller beforehand; AfterOpen runs right after StartStream), pushes the audio, Stop(), waits for OnStopped.
	 * bExpectFiles: waits up to 10 s for the new .wav and .json, then 1 s more so the writer is done; otherwise waits 3 s and expects none.
	 * Then Check() and Done.
	 */
	void Run(const FSpeechStreamSettings& S, const TArray<float>& Audio, int32 Rate, int32 Channels, int32 ChunkFloats, int32 PerTick, bool bExpectFiles,
		const FDoneDelegate& Done, TFunction<void()> AfterOpen, TFunction<void()> Check)
	{
		USpeechStream* St = Open(S);
		if (!St) { Done.ExecuteIfBound(); return; }
		if (AfterOpen) { AfterOpen(); }
		PushPaced(St, Audio, Rate, Channels, ChunkFloats, PerTick, [this, Done, Check, bExpectFiles]()
		{
			if (Stream.IsValid()) { Stream->Stop(); }
			WaitFor(this, [this]() { return Recv.IsValid() && Recv->Count(ESpeechStreamTestEvent::Stopped) >= 1; }, TEXT("OnStopped"), Done, [this, Done, Check, bExpectFiles]()
			{
				if (!bExpectFiles)
				{
					Settle(3.0, [this, Done, Check]() { Check(); Done.ExecuteIfBound(); });
					return;
				}
				WaitFor(this, [this]() { return PairExists(); }, TEXT("a new recording .wav and .json in Saved/ModelPipelines/SpeechRecordings (10 s)"), Done, [this, Done, Check]()
				{
					Settle(1.0, [this, Done, Check]() { Check(); Done.ExecuteIfBound(); });
				}, FileWait);
			}, Wait);
		});
	}
END_DEFINE_SPEC(FSpeechStreamRecordSpec)

void FSpeechStreamRecordSpec::Define()
{
	BeforeEach([this]()
	{
		IConsoleVariable* R = Cvar(TEXT("mp.Speech.RecordStreams"));
		IConsoleVariable* M = Cvar(TEXT("mp.Speech.RecordMaxSeconds"));
		bSavedRec = (R != nullptr && M != nullptr);
		if (bSavedRec) { OldRecord = R->GetInt(); OldMax = M->GetFloat(); }
		Before.Reset();
		ListFiles(Before);
	});

	AfterEach([this]()
	{
		// Restore first so no other spec records, then remove only this test's files.
		if (bSavedRec) { SetCvars(OldRecord, OldMax); }
		TeardownFake();
		TArray<FString> W, J;
		NewFiles(W, J);
		for (const FString& F : W) { IFileManager::Get().Delete(*FPaths::Combine(RecDir(), F), false, true, true); }
		for (const FString& F : J) { IFileManager::Get().Delete(*FPaths::Combine(RecDir(), F), false, true, true); }
	});

	It("TheTwoConsoleVariablesExistWithTheirDefaults", [this]()
	{
		// Case 1. Missing variables, or defaults that switch recording on (every other spec would write files), fail here.
		// Reads the live value: the BeforeEach captured the values before this test touched anything.
		IConsoleVariable* R = Cvar(TEXT("mp.Speech.RecordStreams"));
		IConsoleVariable* M = Cvar(TEXT("mp.Speech.RecordMaxSeconds"));
		if (!TestNotNull(TEXT("mp.Speech.RecordStreams exists"), R) || !TestNotNull(TEXT("mp.Speech.RecordMaxSeconds exists"), M)) { return; }
		TestEqual(TEXT("mp.Speech.RecordStreams default (as found at test start)"), OldRecord, 0);
		TestEqual(TEXT("mp.Speech.RecordMaxSeconds default (as found at test start)"), OldMax, 1200.f, 1e-3f);
	});

	LatentIt("RecordStreamsZeroWritesNothing", LatentLimit, [this](const FDoneDelegate& Done)
	{
		// Case 2. Default off: a normal two-burst session must leave the directory untouched (3 s after OnStopped).
		SetupFake();
		SetCvars(0, 1200.f);
		FSpeechStreamSettings S; S.bPartials = false;
		TArray<float> A; BuildAudio(A, { { 50, 75 }, { 50, 75 } });
		Run(S, A, 16000, 1, Frame, 8, false, Done, nullptr, [this]()
		{
			TArray<FString> W, J;
			NewFiles(W, J);
			TestEqual(TEXT("new .wav files with recording off"), W.Num(), 0);
			TestEqual(TEXT("new .json files with recording off"), J.Num(), 0);
		});
	});

	LatentIt("OneWavAndJsonPairWithExactSampleCountAndTheEventLog", LatentLimit, [this](const FDoneDelegate& Done)
	{
		// Case 3. 0.5 s lead, burst 1.0 s, gap 1.5 s, burst 1.0 s, gap 1.5 s pushed at 16 kHz mono: nothing is resampled, so the WAV
		// must hold exactly the pushed samples (a writer that dropped the last partial chunk or padded would be off). The finals in the
		// JSON must be exactly what OnFinal delivered; events must be ordered.
		SetupFake();
		SetCvars(1, 1200.f);
		FSpeechStreamSettings S; S.bPartials = false;
		TArray<float> A; BuildAudio(A, { { 50, 75 }, { 50, 75 } });
		const int32 Pushed = A.Num();
		Run(S, A, 16000, 1, Frame, 8, true, Done, nullptr, [this, Pushed]()
		{
			TArray<FString> W, J;
			NewFiles(W, J);
			if (!TestEqual(TEXT("exactly one new .wav"), W.Num(), 1) || !TestEqual(TEXT("exactly one new .json"), J.Num(), 1)) { return; }
			TestTrue(TEXT("same base name"), FPaths::GetBaseFilename(W[0]).Equals(FPaths::GetBaseFilename(J[0]), ESearchCase::CaseSensitive));
			const FWav Wav = ParseWav(FPaths::Combine(RecDir(), W[0]));
			if (!TestTrue(*FString::Printf(TEXT("WAV parses (%s)"), *Wav.Error), Wav.bValid)) { return; }
			TestEqual(TEXT("WAV format PCM"), Wav.Format, 1);
			TestEqual(TEXT("WAV channels"), Wav.Channels, 1);
			TestEqual(TEXT("WAV rate"), Wav.Rate, 16000);
			TestEqual(TEXT("WAV bits"), Wav.Bits, 16);
			TestEqual(TEXT("WAV samples equal the pushed samples"), Wav.NumSamples(), Pushed);

			FString Err;
			const TSharedPtr<FJsonObject> Root = ParseJson(FPaths::Combine(RecDir(), J[0]), Err);
			if (!TestTrue(*FString::Printf(TEXT("JSON parses (%s)"), *Err), Root.IsValid())) { return; }
			TestEqual(TEXT("version"), (int32)Root->GetNumberField(TEXT("version")), 1);
			TestTrue(TEXT("engine is Phonon"), Root->GetStringField(TEXT("engine")).Equals(TEXT("Phonon"), ESearchCase::CaseSensitive));
			TestTrue(TEXT("package field present"), Root->HasField(TEXT("package")));
			TestEqual(TEXT("sample_rate"), (int32)Root->GetNumberField(TEXT("sample_rate")), 16000);
			TestEqual(TEXT("num_samples equals pushed"), (int32)Root->GetNumberField(TEXT("num_samples")), Pushed);
			TestFalse(TEXT("truncated"), Root->GetBoolField(TEXT("truncated")));
			TestTrue(TEXT("started_utc present"), Root->HasField(TEXT("started_utc")));
			const TSharedPtr<FJsonObject>* Set = nullptr;
			if (TestTrue(TEXT("settings object"), Root->TryGetObjectField(TEXT("settings"), Set)))
			{
				TestEqual(TEXT("settings.hangover"), (float)(*Set)->GetNumberField(TEXT("hangover")), 0.2f, 1e-4f);
				TestEqual(TEXT("settings.merge_window"), (float)(*Set)->GetNumberField(TEXT("merge_window")), 1.0f, 1e-4f);
			}

			const TArray<TSharedPtr<FJsonValue>>* Ev = nullptr;
			if (!TestTrue(TEXT("events array"), Root->TryGetArrayField(TEXT("events"), Ev))) { return; }
			TArray<FString> Types, FinalTexts;
			TArray<bool> FinalRepl;
			double LastT = -1.0; double LastSample = -1.0;
			bool bOrdered = true, bInRange = true;
			int32 NumSamples = (int32)Root->GetNumberField(TEXT("num_samples"));
			for (const TSharedPtr<FJsonValue>& V : *Ev)
			{
				const TSharedPtr<FJsonObject>* O = nullptr;
				if (!V.IsValid() || !V->TryGetObject(O)) { AddError(TEXT("event is not an object")); continue; }
				const FString Type = (*O)->GetStringField(TEXT("type"));
				Types.Add(Type);
				const double T = (*O)->GetNumberField(TEXT("t"));
				const double Smp = (*O)->GetNumberField(TEXT("sample"));
				bOrdered &= (T >= LastT && Smp >= LastSample);
				bInRange &= (Smp <= (double)NumSamples);
				LastT = T; LastSample = Smp;
				if (Type.Equals(TEXT("final"), ESearchCase::CaseSensitive))
				{
					FinalTexts.Add((*O)->GetStringField(TEXT("text")));
					FinalRepl.Add((*O)->GetBoolField(TEXT("replaces")));
				}
				if (Type.Equals(TEXT("stopped"), ESearchCase::CaseSensitive)) { TestTrue(TEXT("stopped event has ok true"), (*O)->GetBoolField(TEXT("ok"))); }
			}
			TestTrue(TEXT("events: t and sample non-decreasing"), bOrdered);
			TestTrue(TEXT("events: every sample <= num_samples"), bInRange);
			if (Types.Num() > 0) { TestTrue(TEXT("last event is stopped"), Types.Last().Equals(TEXT("stopped"), ESearchCase::CaseSensitive)); }
			// per-utterance order: the k-th started < k-th ended < k-th final
			TArray<int32> IS, IE, IF;
			for (int32 i = 0; i < Types.Num(); ++i)
			{
				if (Types[i] == TEXT("started")) { IS.Add(i); }
				else if (Types[i] == TEXT("ended")) { IE.Add(i); }
				else if (Types[i] == TEXT("final")) { IF.Add(i); }
			}
			TestEqual(TEXT("two utterances recorded as started"), IS.Num(), 2);
			TestEqual(TEXT("ended count equals started count"), IE.Num(), IS.Num());
			TestEqual(TEXT("final count equals started count"), IF.Num(), IS.Num());
			if (IS.Num() == IE.Num() && IE.Num() == IF.Num())
			{
				for (int32 k = 0; k < IS.Num(); ++k)
				{
					TestTrue(*FString::Printf(TEXT("utterance %d: started < ended < final"), k), IS[k] < IE[k] && IE[k] < IF[k]);
				}
			}
			const TArray<FSpeechStreamTestRecord> Fi = Recv->Of(ESpeechStreamTestEvent::Final);
			if (TestEqual(TEXT("JSON finals vs OnFinal count"), FinalTexts.Num(), Fi.Num()))
			{
				for (int32 k = 0; k < Fi.Num(); ++k)
				{
					TestTrue(*FString::Printf(TEXT("final %d text '%s' equals the delegate's '%s'"), k, *FinalTexts[k], *Fi[k].Transcript.Text),
						FinalTexts[k].Equals(Fi[k].Transcript.Text, ESearchCase::CaseSensitive));
					TestEqual(*FString::Printf(TEXT("final %d replaces"), k), FinalRepl[k], Fi[k].Transcript.bReplacesPrevious);
				}
			}
		});
	});

	LatentIt("StereoFortyFourOneIsResampledToSixteenKilohertzMono", LatentLimit, [this](const FDoneDelegate& Done)
	{
		// Case 4. The recording is the audio AFTER mix + resample: pushing 44.1 kHz stereo must give a 16 kHz mono WAV of about the same
		// duration. Recording the raw input would give 44.1 kHz / 2 channels or ~2.76x the samples.
		SetupFake();
		SetCvars(1, 1200.f);
		FSpeechStreamSettings S; S.bPartials = false;
		TArray<float> Mono, Stereo;
		BuildAudio(Mono, { { 50, 75 }, { 50, 75 } });
		ToStereo44k(Mono, Stereo);
		const int32 Expected = Mono.Num();
		Run(S, Stereo, 44100, 2, 882, 8, true, Done, nullptr, [this, Expected]()
		{
			TArray<FString> W, J;
			NewFiles(W, J);
			if (!TestEqual(TEXT("exactly one new .wav"), W.Num(), 1)) { return; }
			const FWav Wav = ParseWav(FPaths::Combine(RecDir(), W[0]));
			if (!TestTrue(*FString::Printf(TEXT("WAV parses (%s)"), *Wav.Error), Wav.bValid)) { return; }
			TestEqual(TEXT("WAV channels"), Wav.Channels, 1);
			TestEqual(TEXT("WAV rate"), Wav.Rate, 16000);
			TestEqual(TEXT("WAV bits"), Wav.Bits, 16);
			const double Ratio = (double)Wav.NumSamples() / (double)Expected;
			TestTrue(*FString::Printf(TEXT("WAV has %d samples, expected %d +-1 %% (ratio %.4f)"), Wav.NumSamples(), Expected, Ratio), FMath::Abs(Ratio - 1.0) <= 0.01);
		});
	});

	LatentIt("RecordMaxSecondsTruncatesTheAudioButNotTheEvents", LatentLimit, [this](const FDoneDelegate& Done)
	{
		// Case 5. Cap 2 s, 5 s pushed (tone from the first sample so the stream has a Started event): WAV holds exactly 32000 samples,
		// truncated is true, and the event log is still written (a started and the final stopped).
		SetupFake();
		SetCvars(1, 2.f);
		FSpeechStreamSettings S; S.bPartials = false;
		TArray<float> A; BuildAudio(A, { { 100, 150 } }, 0);
		Run(S, A, 16000, 1, Frame, 8, true, Done, nullptr, [this]()
		{
			TArray<FString> W, J;
			NewFiles(W, J);
			if (!TestEqual(TEXT("exactly one new .wav"), W.Num(), 1) || !TestEqual(TEXT("exactly one new .json"), J.Num(), 1)) { return; }
			const FWav Wav = ParseWav(FPaths::Combine(RecDir(), W[0]));
			if (!TestTrue(*FString::Printf(TEXT("WAV parses (%s)"), *Wav.Error), Wav.bValid)) { return; }
			TestEqual(TEXT("WAV samples capped at 2 s"), Wav.NumSamples(), 32000);
			FString Err;
			const TSharedPtr<FJsonObject> Root = ParseJson(FPaths::Combine(RecDir(), J[0]), Err);
			if (!TestTrue(*FString::Printf(TEXT("JSON parses (%s)"), *Err), Root.IsValid())) { return; }
			TestTrue(TEXT("truncated is true"), Root->GetBoolField(TEXT("truncated")));
			TestEqual(TEXT("num_samples matches the WAV"), (int32)Root->GetNumberField(TEXT("num_samples")), Wav.NumSamples());
			const TArray<TSharedPtr<FJsonValue>>* Ev = nullptr;
			if (!TestTrue(TEXT("events array"), Root->TryGetArrayField(TEXT("events"), Ev))) { return; }
			bool bStarted = false, bStopped = false;
			for (const TSharedPtr<FJsonValue>& V : *Ev)
			{
				const TSharedPtr<FJsonObject>* O = nullptr;
				if (!V.IsValid() || !V->TryGetObject(O)) { continue; }
				const FString Type = (*O)->GetStringField(TEXT("type"));
				bStarted |= Type.Equals(TEXT("started"), ESearchCase::CaseSensitive);
				bStopped |= Type.Equals(TEXT("stopped"), ESearchCase::CaseSensitive);
			}
			TestTrue(TEXT("a started event is recorded although the audio is truncated"), bStarted);
			TestTrue(TEXT("a stopped event is recorded although the audio is truncated"), bStopped);
		});
	});

	LatentIt("EnablingRecordingAfterStartStreamDoesNotRecordThatStream", LatentLimit, [this](const FDoneDelegate& Done)
	{
		// Case 6. The variable is read when the stream STARTS. Flipping it to 1 right after StartStream must leave no files; an
		// implementation that checked the variable at Finish would write them.
		SetupFake();
		SetCvars(0, 1200.f);
		FSpeechStreamSettings S; S.bPartials = false;
		TArray<float> A; BuildAudio(A, { { 50, 75 }, { 50, 75 } });
		Run(S, A, 16000, 1, Frame, 8, false, Done, [this]() { SetCvars(1, 1200.f); }, [this]()
		{
			TArray<FString> W, J;
			NewFiles(W, J);
			TestEqual(TEXT("new .wav files for a stream started with recording off"), W.Num(), 0);
			TestEqual(TEXT("new .json files for a stream started with recording off"), J.Num(), 0);
		});
	});

	LatentIt("DestroyingTheStreamMidUtteranceWhileRecordingDoesNotCrash", LatentLimit, [this](const FDoneDelegate& Done)
	{
		// Case 7. Drop the only strong reference mid-utterance and collect: no crash, no hang. Whether a file pair appears is not
		// asserted; if one does, its JSON must parse.
		SetupFake();
		SetCvars(1, 1200.f);
		FSpeechStreamSettings S; S.bPartials = false;
		USpeechStream* St = Open(S);
		if (!St) { Done.ExecuteIfBound(); return; }
		TArray<float> A; BuildAudio(A, { { 250, 0 } }, 0);
		PushPaced(St, A, 16000, 1, Frame, 4, []() {});
		WaitFor(this, [this]() { return Recv.IsValid() && Recv->Count(ESpeechStreamTestEvent::Started) >= 1; }, TEXT("OnSpeechStarted"), Done, [this, Done]()
		{
			Stream.Reset();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			Settle(3.0, [this, Done]()
			{
				TArray<FString> W, J;
				NewFiles(W, J);
				for (const FString& F : J)
				{
					FString Err;
					const TSharedPtr<FJsonObject> Root = ParseJson(FPaths::Combine(RecDir(), F), Err);
					TestTrue(*FString::Printf(TEXT("%s, if written, parses (%s)"), *F, *Err), Root.IsValid());
				}
				Done.ExecuteIfBound();
			});
		}, Wait);
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS && !UE_BUILD_SHIPPING
