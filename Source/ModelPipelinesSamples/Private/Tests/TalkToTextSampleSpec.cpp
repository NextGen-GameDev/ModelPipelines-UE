// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformTime.h"
#include "Misc/CoreDelegates.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "NNEModelData.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineModelPackage.h"
#include "PipelineTypes.h"
#include "SpeechStream.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesTalkToTextSample.h"
#include "SamplesTestData.h"
#include "TalkToTextTestReceiver.h"

using namespace UE::ModelPipelines::SamplesTests;

namespace TalkToTextSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(420);
	/** Loading the 1.3 GB model and creating the pipeline on the CPU can take a while. */
	constexpr double ReadySeconds = 200.0;
	constexpr double RecognitionSeconds = 150.0;
	constexpr float SampleRate = 16000.f;
	constexpr int32 PieceSamples = 320;
	/** Audio is pushed at this multiple of real time. */
	constexpr double PushSpeed = 4.0;
	/** Never push more than this many pieces in one tick (a stalled frame must not become a burst: back-pressure drops audio above 10 s queued). */
	constexpr int32 MaxPiecesPerTick = 25;
	constexpr double PushedSecondsFull = 20.0;
	constexpr int32 TailSamples = 24000; // 1.5 s of silence so the last utterance ends by the VAD
	const TCHAR* const RecordingId = TEXT("long_61_70968_45s");
	const TCHAR* const StageAHint = TEXT("Fetch the model with: py Tools/fetch_phonon2.py (see Docs/Samples.md), then export stage A into .pipelines-dev/models/stt/phonon-2-onnx/fp16w/.");
	const TCHAR* const GoldenHint = TEXT("Generate with: python Tools/make_stt_long_goldens.py");

	// ---------------------------------------------------------------- shared test data (loaded once per run)

	static TStrongObjectPtr<UNNEModelData> GEncoder;
	static TStrongObjectPtr<UNNEModelData> GDecoder;
	static FString GTokenizerJson;
	static bool bExitHooked = false;

	static TArray<float> GAudio;        // first 20 s of the recording
	static TArray<FString> GRefWords20; // normalised reference words with end <= 20 s
	static bool bAudioLoaded = false;

	static UNNEModelData* MakeModelData(const FString& File, FString& OutError)
	{
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *File) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("cannot read %s. %s"), *File, StageAHint);
			return nullptr;
		}
		UNNEModelData* Data = NewObject<UNNEModelData>(GetTransientPackage());
		Data->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), Bytes.Num()));
		return Data;
	}

	/** Loads (once) the stage A model files. false + AddError on failure. */
	static bool EnsureModelData(FAutomationTestBase& T)
	{
		if (GEncoder.IsValid() && GDecoder.IsValid() && !GTokenizerJson.IsEmpty()) { return true; }
		const FString Dir = FPaths::Combine(GetDevDir(), TEXT("models/stt/phonon-2-onnx/fp16w"));
		FString Err;
		UNNEModelData* Enc = MakeModelData(FPaths::Combine(Dir, TEXT("encoder.onnx")), Err);
		if (!Enc) { T.AddError(Err); return false; }
		UNNEModelData* Dec = MakeModelData(FPaths::Combine(Dir, TEXT("decoder_joint.onnx")), Err);
		if (!Dec) { T.AddError(Err); return false; }
		FString Tok;
		if (!FFileHelper::LoadFileToString(Tok, *FPaths::Combine(Dir, TEXT("tokenizer.json"))) || Tok.IsEmpty())
		{
			T.AddError(FString::Printf(TEXT("cannot read %s/tokenizer.json. %s"), *Dir, StageAHint));
			return false;
		}
		GEncoder.Reset(Enc);
		GDecoder.Reset(Dec);
		GTokenizerJson = Tok;
		if (!bExitHooked)
		{
			bExitHooked = true;
			FCoreDelegates::OnEnginePreExit.AddLambda([]() { GEncoder.Reset(); GDecoder.Reset(); });
		}
		return true;
	}

	/** A fresh transient package sharing the loaded stage A model data, with the given task. */
	static UPipelineModelPackage* MakeStageAPackage(EPipelineTask Task)
	{
		UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
		Pkg->Task = Task;
		Pkg->Model = GEncoder.Get();
		Pkg->DecoderModel = GDecoder.Get();
		Pkg->TokenizerJson = GTokenizerJson;
		Pkg->PreferredDevice = EPipelineDevice::CPU;
		Pkg->License.SpdxId = TEXT("CC-BY-4.0");
		Pkg->License.SourceUrl = TEXT("https://huggingface.co/");
		Pkg->License.Attribution = TEXT("Phonon 2 (test package)");
		return Pkg;
	}

	/** Upper case, letters/digits/apostrophes only, split on everything else. */
	static void NormaliseWords(const FString& In, TArray<FString>& Out)
	{
		FString Cur;
		for (const TCHAR C : In)
		{
			if (FChar::IsAlnum(C) || C == TEXT('\''))
			{
				Cur.AppendChar(FChar::ToUpper(C));
			}
			else if (!Cur.IsEmpty())
			{
				Out.Add(Cur);
				Cur.Reset();
			}
		}
		if (!Cur.IsEmpty()) { Out.Add(Cur); }
	}

	static int32 EditDistance(const TArray<FString>& A, const TArray<FString>& B)
	{
		TArray<int32> Prev, Cur;
		Prev.SetNumUninitialized(B.Num() + 1);
		Cur.SetNumUninitialized(B.Num() + 1);
		for (int32 j = 0; j <= B.Num(); ++j) { Prev[j] = j; }
		for (int32 i = 1; i <= A.Num(); ++i)
		{
			Cur[0] = i;
			for (int32 j = 1; j <= B.Num(); ++j)
			{
				const int32 Sub = Prev[j - 1] + (A[i - 1].Equals(B[j - 1], ESearchCase::CaseSensitive) ? 0 : 1);
				Cur[j] = FMath::Min3(Sub, Prev[j] + 1, Cur[j - 1] + 1);
			}
			Swap(Prev, Cur);
		}
		return Prev[B.Num()];
	}

	/** Loads (once) the first 20 s of the recording and the reference words from the long golden. */
	static bool EnsureAudio(FAutomationTestBase& T)
	{
		if (bAudioLoaded) { return true; }
		const FString Dir = FPaths::Combine(GetDevDir(), TEXT("goldens/stt/long"));
		const FString JsonPath = FPaths::Combine(Dir, TEXT("long_goldens.json"));
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *JsonPath))
		{
			T.AddError(FString::Printf(TEXT("missing %s. %s"), *JsonPath, GoldenHint));
			return false;
		}
		TSharedPtr<FJsonObject> Root;
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root) || !Root.IsValid())
		{
			T.AddError(FString::Printf(TEXT("%s is not valid JSON. %s"), *JsonPath, GoldenHint));
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>* Recs = nullptr;
		if (!Root->TryGetArrayField(TEXT("recordings"), Recs)) { T.AddError(TEXT("long_goldens.json has no 'recordings'")); return false; }
		TSharedPtr<FJsonObject> Rec;
		for (const TSharedPtr<FJsonValue>& V : *Recs)
		{
			const TSharedPtr<FJsonObject>* O = nullptr;
			if (V.IsValid() && V->TryGetObject(O) && (*O)->GetStringField(TEXT("id")) == RecordingId) { Rec = *O; break; }
		}
		if (!Rec.IsValid()) { T.AddError(FString::Printf(TEXT("recording '%s' not in %s. %s"), RecordingId, *JsonPath, GoldenHint)); return false; }

		const TSharedPtr<FJsonObject> AudioObj = Rec->GetObjectField(TEXT("audio"));
		const FString AudioPath = FPaths::Combine(Dir, AudioObj->GetStringField(TEXT("file")));
		const int32 NumSamples = static_cast<int32>(AudioObj->GetNumberField(TEXT("num_samples")));
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *AudioPath) || Bytes.Num() != NumSamples * 4)
		{
			T.AddError(FString::Printf(TEXT("cannot read %s (%d bytes expected). %s"), *AudioPath, NumSamples * 4, GoldenHint));
			return false;
		}
		const int32 Want = static_cast<int32>(PushedSecondsFull * SampleRate);
		if (NumSamples < Want) { T.AddError(TEXT("recording shorter than 20 s")); return false; }
		GAudio.SetNumUninitialized(Want);
		FMemory::Memcpy(GAudio.GetData(), Bytes.GetData(), Want * sizeof(float));

		const TArray<TSharedPtr<FJsonValue>>* Words = nullptr;
		if (!Rec->TryGetArrayField(TEXT("words"), Words)) { T.AddError(TEXT("recording has no 'words'")); return false; }
		for (const TSharedPtr<FJsonValue>& W : *Words)
		{
			const TArray<TSharedPtr<FJsonValue>>* Triple = nullptr;
			if (!W.IsValid() || !W->TryGetArray(Triple) || Triple->Num() < 3) { continue; }
			if ((*Triple)[2]->AsNumber() <= PushedSecondsFull) { NormaliseWords((*Triple)[0]->AsString(), GRefWords20); }
		}
		if (GRefWords20.Num() < 20) { T.AddError(FString::Printf(TEXT("only %d reference words under 20 s"), GRefWords20.Num())); return false; }
		bAudioLoaded = true;
		return true;
	}

	// ---------------------------------------------------------------- fixture

	struct FRun : public TSharedFromThis<FRun>
	{
		FSamplesTestWorld World;
		TStrongObjectPtr<UTalkToTextTestReceiver> Rcv;
		TStrongObjectPtr<UPipelineModelPackage> Pkg;
		TWeakObjectPtr<AModelPipelinesTalkToTextSample> Actor;
		TWeakObjectPtr<USpeechStream> PushTarget;
		FTSTicker::FDelegateHandle Pusher;
		FTSTicker::FDelegateHandle Monitor;

		int32 PushedSamples = 0;
		int32 TotalToPush = 0;
		double PushStart = 0.0;
		bool bPushDone = false;
		double PushDoneTime = 0.0;

		/** Monitor ticks that saw a non-empty partial text although HasPartialText() is false (stage A: always a bug). */
		int32 PartialViolations = 0;
		FString FirstPartialViolation;
		int32 MonitorTicks = 0;

		bool Setup(FAutomationTestBase& T, EPipelineTask Task)
		{
			FString Err;
			if (!World.Create(Err)) { T.AddError(Err); return false; }
			Rcv.Reset(NewObject<UTalkToTextTestReceiver>(GetTransientPackage()));
			Pkg.Reset(MakeStageAPackage(Task));
			return true;
		}

		AModelPipelinesTalkToTextSample* Spawn(FAutomationTestBase& T, const TSoftObjectPtr<UPipelineModelPackage>& Package, TFunction<void(AModelPipelinesTalkToTextSample*)> Configure)
		{
			FString Err;
			UTalkToTextTestReceiver* R = Rcv.Get();
			AModelPipelinesSampleBase* Base = World.SpawnSample(AModelPipelinesTalkToTextSample::StaticClass(), Package, false,
				[R, Configure](AModelPipelinesSampleBase* NewActor)
				{
					AModelPipelinesTalkToTextSample* S = CastChecked<AModelPipelinesTalkToTextSample>(NewActor);
					R->Sample = S;
					S->OnFinalText.AddDynamic(R, &UTalkToTextTestReceiver::OnFinal);
					S->OnSampleResult.AddDynamic(R, &UTalkToTextTestReceiver::OnResult);
					if (Configure) { Configure(S); }
				}, Err);
			if (!Base) { T.AddError(Err); return nullptr; }
			AModelPipelinesTalkToTextSample* A = Cast<AModelPipelinesTalkToTextSample>(Base);
			Actor = A;
			TWeakPtr<FRun> Weak = AsShared();
			Monitor = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Weak](float)
			{
				const TSharedPtr<FRun> Self = Weak.Pin();
				if (!Self.IsValid()) { return false; }
				if (const AModelPipelinesTalkToTextSample* S = Self->Actor.Get())
				{
					++Self->MonitorTicks;
					if (!S->HasPartialText() && !S->GetPartialText().IsEmpty())
					{
						++Self->PartialViolations;
						if (Self->FirstPartialViolation.IsEmpty()) { Self->FirstPartialViolation = S->GetPartialText(); }
					}
				}
				return true;
			}), 0.0f);
			return A;
		}

		/** Starts pushing the first Seconds of the audio, then 1.5 s of silence, at 4x real time into the sample's current stream. */
		void StartPush(double Seconds)
		{
			AModelPipelinesTalkToTextSample* A = Actor.Get();
			PushTarget = A ? A->GetStream() : nullptr;
			const int32 SpeechSamples = FMath::Min(GAudio.Num(), (static_cast<int32>(Seconds * SampleRate) / PieceSamples) * PieceSamples);
			TotalToPush = SpeechSamples + TailSamples;
			PushedSamples = 0;
			bPushDone = false;
			PushStart = FPlatformTime::Seconds();
			TWeakPtr<FRun> Weak = AsShared();
			Pusher = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Weak, SpeechSamples](float)
			{
				const TSharedPtr<FRun> Self = Weak.Pin();
				if (!Self.IsValid()) { return false; }
				const double Now = FPlatformTime::Seconds();
				const int32 Target = FMath::Min(Self->TotalToPush, static_cast<int32>((Now - Self->PushStart) * PushSpeed * SampleRate));
				USpeechStream* S = Self->PushTarget.Get();
				int32 Pieces = 0;
				TArray<float> Piece;
				Piece.SetNumUninitialized(PieceSamples);
				while (Self->PushedSamples + PieceSamples <= Self->TotalToPush && Self->PushedSamples + PieceSamples <= Target && Pieces < MaxPiecesPerTick)
				{
					for (int32 i = 0; i < PieceSamples; ++i)
					{
						const int32 Idx = Self->PushedSamples + i;
						Piece[i] = (Idx < SpeechSamples) ? GAudio[Idx] : 0.f;
					}
					if (IsValid(S)) { S->PushAudio(TConstArrayView<float>(Piece.GetData(), PieceSamples), 16000, 1); }
					Self->PushedSamples += PieceSamples;
					++Pieces;
				}
				if (Self->PushedSamples + PieceSamples > Self->TotalToPush)
				{
					Self->bPushDone = true;
					Self->PushDoneTime = Now;
					return false;
				}
				return true;
			}), 0.0f);
		}

		/** The later of: push start/finish and the last final's arrival. */
		double LastActivityTime() const
		{
			double T = bPushDone ? PushDoneTime : PushStart;
			if (Rcv.IsValid() && Rcv->TextTimes.Num() > 0) { T = FMath::Max(T, Rcv->TextTimes.Last()); }
			return T;
		}

		void Teardown()
		{
			if (Pusher.IsValid()) { FTSTicker::GetCoreTicker().RemoveTicker(Pusher); Pusher.Reset(); }
			if (Monitor.IsValid()) { FTSTicker::GetCoreTicker().RemoveTicker(Monitor); Monitor.Reset(); }
			if (AModelPipelinesTalkToTextSample* A = Actor.Get())
			{
				A->OnFinalText.RemoveAll(Rcv.Get());
				A->OnSampleResult.RemoveAll(Rcv.Get());
			}
			World.Destroy();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		}
	};

	// ---------------------------------------------------------------- waiting (never blocks the game thread)

	/** Polls Cond on the core ticker. On timeout: AddError(What) and Done. On success: Next(), which must call Done itself. */
	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Next, double Timeout)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TalkToTextSpecWait"), 0.02f, [T, Cond, What, Done, Next, Start, Timeout](float)
		{
			if (Cond())
			{
				Next();
				return false;
			}
			if ((FPlatformTime::Seconds() - Start) > Timeout)
			{
				T->AddError(FString::Printf(TEXT("timed out after %.0f s waiting for: %s"), Timeout, *What));
				Done.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	/** Lets the game thread tick for Secs, then Next() (which must call Done itself). */
	static void Settle(double Secs, TFunction<void()> Next)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TalkToTextSpecSettle"), 0.02f, [Start, Secs, Next](float)
		{
			if ((FPlatformTime::Seconds() - Start) >= Secs)
			{
				Next();
				return false;
			}
			return true;
		});
	}

	static FString StateName(EPipelineSampleState S)
	{
		switch (S)
		{
		case EPipelineSampleState::Loading: return TEXT("Loading");
		case EPipelineSampleState::MissingPackage: return TEXT("MissingPackage");
		case EPipelineSampleState::Failed: return TEXT("Failed");
		case EPipelineSampleState::Ready: return TEXT("Ready");
		case EPipelineSampleState::Busy: return TEXT("Busy");
		}
		return TEXT("?");
	}

	/**
	 * Spawns the sample with the stage A SpeechToText package, waits until it is listening, then Next(). On any failure (data
	 * missing, spawn failure, Failed/MissingPackage state, timeout) it AddErrors and calls Done itself; Next is then not called.
	 */
	static void BeginListening(FAutomationTestBase* T, TSharedPtr<FRun>& Slot, FDoneDelegate Done, TFunction<void(AModelPipelinesTalkToTextSample*)> Configure,
		TFunction<void(TSharedRef<FRun>, AModelPipelinesTalkToTextSample*)> Next)
	{
		if (!EnsureModelData(*T) || !EnsureAudio(*T)) { Done.ExecuteIfBound(); return; }
		TSharedRef<FRun> R = MakeShared<FRun>();
		Slot = R;
		if (!R->Setup(*T, EPipelineTask::SpeechToText)) { Done.ExecuteIfBound(); return; }
		AModelPipelinesTalkToTextSample* A = R->Spawn(*T, TSoftObjectPtr<UPipelineModelPackage>(R->Pkg.Get()), Configure);
		if (!A) { Done.ExecuteIfBound(); return; }
		TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = A;
		WaitFor(T,
			[WA]()
			{
				const AModelPipelinesTalkToTextSample* S = WA.Get();
				return !S || S->IsListening() || S->GetSampleState() == EPipelineSampleState::Failed || S->GetSampleState() == EPipelineSampleState::MissingPackage;
			},
			TEXT("the sample to start listening"), Done,
			[T, R, WA, Done, Next]()
			{
				AModelPipelinesTalkToTextSample* S = WA.Get();
				if (!S || !S->IsListening())
				{
					T->AddError(FString::Printf(TEXT("sample did not start listening: state %s, status '%s'"),
						S ? *StateName(S->GetSampleState()) : TEXT("<destroyed>"), S ? *S->GetStatusText() : TEXT("")));
					Done.ExecuteIfBound();
					return;
				}
				Next(R, S);
			}, ReadySeconds);
	}

	static FString JoinTexts(const TArray<FString>& Texts)
	{
		FString Out;
		for (const FString& S : Texts) { Out += S; Out += TEXT(" "); }
		return Out;
	}
}

BEGIN_DEFINE_SPEC(FTalkToTextSampleSpec, "Pipelines.Samples.TalkToText",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	TSharedPtr<TalkToTextSpecPrivate::FRun> Run;
END_DEFINE_SPEC(FTalkToTextSampleSpec)

void FTalkToTextSampleSpec::Define()
{
	using namespace TalkToTextSpecPrivate;

	AfterEach([this]()
	{
		if (Run.IsValid())
		{
			Run->Teardown();
			Run.Reset();
		}
	});

	// ======================================================================================================
	Describe("Defaults", [this]()
	{
		It("HasTheContractDefaults", [this]()
		{
			const AModelPipelinesTalkToTextSample* CDO = GetDefault<AModelPipelinesTalkToTextSample>();
			if (!TestNotNull(TEXT("CDO"), CDO)) { return; }
			const FString PkgPath = CDO->Package.ToSoftObjectPath().ToString();
			TestTrue(*FString::Printf(TEXT("Package default is the phonon-2-fp16w importer path, was '%s'"), *PkgPath),
				PkgPath.Equals(TEXT("/Game/Models/phonon-2-fp16w.phonon-2-fp16w"), ESearchCase::CaseSensitive));
			TestTrue(TEXT("bUseMicrophone default true"), CDO->bUseMicrophone);
			TestEqual(TEXT("DeviceIndex default -1"), CDO->DeviceIndex, -1);
			TestTrue(TEXT("bAutoStart default true"), CDO->bAutoStart);
			TestEqual(TEXT("MaxLines default 8"), CDO->MaxLines, 8);
			TestEqual(TEXT("EndOfSpeechHangoverSeconds default 0.2"), CDO->StreamSettings.EndOfSpeechHangoverSeconds, 0.2f, 1e-5f);
			TestFalse(TEXT("a fresh CDO is not listening"), CDO->IsListening());
			TestNull(TEXT("a fresh CDO has no stream"), CDO->GetStream());
			TestEqual(TEXT("latency is -1 before the first final"), CDO->GetLastFinalLatencySeconds(), -1.f);
		});
	});

	// ======================================================================================================
	Describe("BadPackages", [this]()
	{
		LatentIt("MissingPackageReportsHowToFetchIt", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches a sample that stays Loading on an unresolvable path, or hides the fix: the user needs the path and the script name.
			// lead: removed (substring expectations swallow failed assertions containing the text; warnings never fail a test) // AddExpectedMessagePlain(TEXT("does-not-exist"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			TSharedRef<FRun> R = MakeShared<FRun>();
			Run = R;
			FString Err;
			if (!R->World.Create(Err)) { AddError(Err); Done.ExecuteIfBound(); return; }
			R->Rcv.Reset(NewObject<UTalkToTextTestReceiver>(GetTransientPackage()));
			const FSoftObjectPath Path(TEXT("/Game/Models/does-not-exist.does-not-exist"));
			AModelPipelinesTalkToTextSample* A = R->Spawn(*this, TSoftObjectPtr<UPipelineModelPackage>(Path), nullptr);
			if (!A) { Done.ExecuteIfBound(); return; }
			TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = A;
			WaitFor(this, [WA]() { const AModelPipelinesTalkToTextSample* S = WA.Get(); return !S || S->GetSampleState() != EPipelineSampleState::Loading; },
				TEXT("state to leave Loading"), Done, [this, R, WA, Done]()
				{
					AModelPipelinesTalkToTextSample* S = WA.Get();
					if (!TestNotNull(TEXT("actor alive"), S)) { Done.ExecuteIfBound(); return; }
					TestEqual(TEXT("state is MissingPackage"), StateName(S->GetSampleState()), StateName(EPipelineSampleState::MissingPackage));
					const FString Status = S->GetStatusText();
					TestTrue(*FString::Printf(TEXT("status contains the package path, was '%s'"), *Status), Status.Contains(TEXT("/Game/Models/does-not-exist.does-not-exist"), ESearchCase::CaseSensitive));
					TestTrue(*FString::Printf(TEXT("status names fetch_phonon2.py, was '%s'"), *Status), Status.Contains(TEXT("fetch_phonon2.py"), ESearchCase::CaseSensitive));
					S->StartListening();
					Settle(0.5, [this, R, WA, Done]()
					{
						AModelPipelinesTalkToTextSample* S2 = WA.Get();
						if (TestNotNull(TEXT("actor alive"), S2))
						{
							TestFalse(TEXT("StartListening on an unusable sample is a no-op: IsListening false"), S2->IsListening());
							TestNull(TEXT("no stream"), S2->GetStream());
							TestEqual(TEXT("still MissingPackage"), StateName(S2->GetSampleState()), StateName(EPipelineSampleState::MissingPackage));
						}
						TestEqual(TEXT("no finals"), R->Rcv->Texts.Num(), 0);
						Done.ExecuteIfBound();
					});
				}, 10.0);
		});

		LatentIt("WrongTaskPackageFails", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// A package for another task must end in Failed (the creator rejects it), never in a listening sample.
			if (!EnsureModelData(*this)) { Done.ExecuteIfBound(); return; }
			// lead: removed (substring expectations swallow failed assertions containing the text; warnings never fail a test) // AddExpectedMessagePlain(TEXT("could not create the pipeline"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			TSharedRef<FRun> R = MakeShared<FRun>();
			Run = R;
			if (!R->Setup(*this, EPipelineTask::TextClassification)) { Done.ExecuteIfBound(); return; }
			AModelPipelinesTalkToTextSample* A = R->Spawn(*this, TSoftObjectPtr<UPipelineModelPackage>(R->Pkg.Get()), nullptr);
			if (!A) { Done.ExecuteIfBound(); return; }
			TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = A;
			WaitFor(this, [WA]() { const AModelPipelinesTalkToTextSample* S = WA.Get(); return !S || S->GetSampleState() != EPipelineSampleState::Loading; },
				TEXT("state to leave Loading"), Done, [this, WA, Done]()
				{
					AModelPipelinesTalkToTextSample* S = WA.Get();
					if (TestNotNull(TEXT("actor alive"), S))
					{
						TestEqual(*FString::Printf(TEXT("state is Failed (status '%s')"), *S->GetStatusText()), StateName(S->GetSampleState()), StateName(EPipelineSampleState::Failed));
						S->StartListening();
					}
					Settle(0.5, [this, WA, Done]()
					{
						AModelPipelinesTalkToTextSample* S2 = WA.Get();
						if (TestNotNull(TEXT("actor alive"), S2))
						{
							TestFalse(TEXT("IsListening false after a wrong-task package"), S2->IsListening());
							TestNull(TEXT("no stream"), S2->GetStream());
						}
						Done.ExecuteIfBound();
					});
				}, ReadySeconds);
		});
	});

	// ======================================================================================================
	Describe("PushedAudio", [this]()
	{
		LatentIt("TranscribesTwentySecondsWithinFivePercentWer", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Stage A (no partials): finals only. Catches a sample that never appends, appends twice, reports a result without a
			// final (or the other way round), shows partial text on an engine without partials, or garbles the transcript.
			BeginListening(this, Run, Done, [](AModelPipelinesTalkToTextSample* S) { S->bUseMicrophone = false; S->StreamSettings.MergeWindowSeconds = 0.f; /* lead: pre-C83 line-count case; joins are tested in TalkToText.Join */ },
				[this, Done](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					TestNotNull(TEXT("GetStream non-null while listening"), S->GetStream());
					TestTrue(*FString::Printf(TEXT("status mentions 'pushed audio', was '%s'"), *S->GetStatusText()), S->GetStatusText().Contains(TEXT("pushed audio"), ESearchCase::IgnoreCase));
					TestFalse(TEXT("HasPartialText false for a stage A package"), S->HasPartialText());
					TestEqual(TEXT("GetPartialText empty before audio"), S->GetPartialText(), FString());
					R->StartPush(PushedSecondsFull);
					TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = S;
					WaitFor(this, [R]() { return R->bPushDone && (FPlatformTime::Seconds() - R->LastActivityTime()) > 8.0; },
						TEXT("the audio to be pushed and the last final to arrive"), Done, [this, R, WA, Done]()
						{
							const UTalkToTextTestReceiver* Rc = R->Rcv.Get();
							const int32 Finals = Rc->Texts.Num();
							TestTrue(*FString::Printf(TEXT("at least 3 finals, got %d"), Finals), Finals >= 3);
							TestEqual(TEXT("one OnSampleResult per OnFinalText"), Rc->ResultSuccess.Num(), Finals);
							for (int32 i = 0; i < Rc->ResultSuccess.Num(); ++i)
							{
								TestTrue(*FString::Printf(TEXT("OnSampleResult %d has bSuccess"), i), Rc->ResultSuccess[i]);
							}
							TestTrue(TEXT("every delivery on the game thread"), Rc->bAllOnGameThread);
							for (int32 i = 0; i < Rc->PartialAtFinal.Num(); ++i)
							{
								TestEqual(*FString::Printf(TEXT("partial text is empty at final %d"), i), Rc->PartialAtFinal[i], FString());
								TestTrue(*FString::Printf(TEXT("latency at final %d in [0,5], was %.2f"), i, Rc->LatencyAtFinal[i]), Rc->LatencyAtFinal[i] >= 0.f && Rc->LatencyAtFinal[i] <= 5.f);
							}
							TestEqual(*FString::Printf(TEXT("GetPartialText was non-empty on %d ticks (first '%s')"), R->PartialViolations, *R->FirstPartialViolation), R->PartialViolations, 0);
							TestTrue(TEXT("the monitor ticked"), R->MonitorTicks > 10);
							if (const AModelPipelinesTalkToTextSample* S2 = WA.Get())
							{
								const TArray<FString> Lines = S2->GetFinalLines();
								TestEqual(TEXT("GetFinalLines().Num() == finals (capped at 8)"), Lines.Num(), FMath::Min(Finals, 8));
								TestFalse(TEXT("HasPartialText still false"), S2->HasPartialText());
								const float Lat = S2->GetLastFinalLatencySeconds();
								TestTrue(*FString::Printf(TEXT("GetLastFinalLatencySeconds in [0,5], was %.2f"), Lat), Lat >= 0.f && Lat <= 5.f);
							}
							// WER of the recognised texts against the reference words that end within the pushed 20 s.
							TArray<FString> Hyp;
							NormaliseWords(JoinTexts(Rc->Texts), Hyp);
							const int32 Dist = EditDistance(GRefWords20, Hyp);
							const double Wer = static_cast<double>(Dist) / FMath::Max(1, GRefWords20.Num());
							TestTrue(*FString::Printf(TEXT("WER %.3f (%d edits over %d reference words) must be <= 0.05. Hypothesis: %s"), Wer, Dist, GRefWords20.Num(), *JoinTexts(Rc->Texts)), Wer <= 0.05);
							Done.ExecuteIfBound();
						}, RecognitionSeconds);
				});
		});

		LatentIt("MaxLinesTwoKeepsTheLastTwoFinalsOldestFirst", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches dropping the newest instead of the oldest, an off-by-one in the cap, or newest-first order.
			BeginListening(this, Run, Done, [](AModelPipelinesTalkToTextSample* S) { S->bUseMicrophone = false; S->StreamSettings.MergeWindowSeconds = 0.f; /* lead: pre-C83 line-count case; joins are tested in TalkToText.Join */ S->MaxLines = 2; },
				[this, Done](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					R->StartPush(PushedSecondsFull);
					TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = S;
					WaitFor(this, [R]() { return R->Rcv->Texts.Num() >= 3; }, TEXT("3 finals"), Done, [this, R, WA, Done]()
					{
						const AModelPipelinesTalkToTextSample* S2 = WA.Get();
						if (TestNotNull(TEXT("actor alive"), S2))
						{
							const TArray<FString> Lines = S2->GetFinalLines();
							const TArray<FString>& Texts = R->Rcv->Texts;
							const int32 N = Texts.Num();
							if (TestEqual(*FString::Printf(TEXT("exactly 2 lines after %d finals"), N), Lines.Num(), 2) && N >= 2)
							{
								TestTrue(*FString::Printf(TEXT("line 0 '%s' is the second-to-last final '%s'"), *Lines[0], *Texts[N - 2]), Lines[0].TrimStartAndEnd().Equals(Texts[N - 2].TrimStartAndEnd(), ESearchCase::CaseSensitive));
								TestTrue(*FString::Printf(TEXT("line 1 '%s' is the last final '%s'"), *Lines[1], *Texts[N - 1]), Lines[1].TrimStartAndEnd().Equals(Texts[N - 1].TrimStartAndEnd(), ESearchCase::CaseSensitive));
							}
						}
						Done.ExecuteIfBound();
					}, RecognitionSeconds);
				});
		});

		LatentIt("StopListeningEndsTheStreamAndSilencesFinals", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches a Stop that leaves the stream running (finals keep coming) or leaves IsListening true until the stop callback.
			BeginListening(this, Run, Done, [](AModelPipelinesTalkToTextSample* S) { S->bUseMicrophone = false; },
				[this, Done](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					R->StartPush(PushedSecondsFull);
					TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = S;
					WaitFor(this, [R]() { return R->Rcv->Texts.Num() >= 1; }, TEXT("the first final"), Done, [this, R, WA, Done]()
					{
						AModelPipelinesTalkToTextSample* S2 = WA.Get();
						if (!TestNotNull(TEXT("actor alive"), S2)) { Done.ExecuteIfBound(); return; }
						S2->StopListening();
						TestFalse(TEXT("IsListening false at once after StopListening"), S2->IsListening());
						// The pusher keeps feeding the old stream pointer (weak, only while valid): it must be ignored, not crash.
						WaitFor(this, [WA]() { const AModelPipelinesTalkToTextSample* S3 = WA.Get(); return !S3 || S3->GetStream() == nullptr; },
							TEXT("GetStream() to become null (OnStopped)"), Done, [this, R, WA, Done]()
							{
								Settle(1.0, [this, R, WA, Done]()
								{
									const int32 Baseline = R->Rcv->Texts.Num();
									Settle(3.0, [this, R, WA, Baseline, Done]()
									{
										TestEqual(TEXT("no OnFinalText after Stop + 1 s, although audio kept being pushed"), R->Rcv->Texts.Num(), Baseline);
										if (const AModelPipelinesTalkToTextSample* S4 = WA.Get())
										{
											TestFalse(TEXT("still not listening"), S4->IsListening());
										}
										Done.ExecuteIfBound();
									});
								});
							}, 5.0);
					}, RecognitionSeconds);
				});
		});
	});

	// ======================================================================================================
	Describe("Microphone", [this]()
	{
		LatentIt("MissingMicrophoneKeepsTheStreamForPushedAudio", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Device 99 cannot exist: the sample must say so but keep listening for pushed audio (failing the whole stream
			// because of the mic would make the Blueprint feed path unusable on machines without one).
			// The mic source may log a Warning; only a message containing "icrophone" is tolerated, never "." (that would swallow failures).
			// lead: removed (substring expectations swallow failed assertions containing the text; warnings never fail a test) // AddExpectedMessagePlain(TEXT("icrophone"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			BeginListening(this, Run, Done, [](AModelPipelinesTalkToTextSample* S) { S->bUseMicrophone = true; S->DeviceIndex = 99; },
				[this, Done](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = S;
					// The mic result may arrive a tick or two after IsListening; give it a few seconds.
					WaitFor(this, [WA]() { const AModelPipelinesTalkToTextSample* S2 = WA.Get(); return !S2 || S2->GetStatusText().StartsWith(TEXT("No microphone ("), ESearchCase::CaseSensitive); },
						TEXT("status starting with 'No microphone ('"), Done, [this, R, WA, Done]()
						{
							AModelPipelinesTalkToTextSample* S3 = WA.Get();
							if (!TestNotNull(TEXT("actor alive"), S3)) { Done.ExecuteIfBound(); return; }
							TestFalse(TEXT("IsMicrophoneActive false"), S3->IsMicrophoneActive());
							TestTrue(TEXT("IsListening stays true (stream kept)"), S3->IsListening());
							TestNotNull(TEXT("stream kept"), S3->GetStream());
							R->StartPush(6.0);
							WaitFor(this, [R]() { return R->Rcv->Texts.Num() >= 1; }, TEXT("at least 1 final from 6 s of pushed audio"), Done, [this, R, Done]()
							{
								TestTrue(*FString::Printf(TEXT("got %d finals"), R->Rcv->Texts.Num()), R->Rcv->Texts.Num() >= 1);
								TestEqual(TEXT("one OnSampleResult per final"), R->Rcv->ResultSuccess.Num(), R->Rcv->Texts.Num());
								Done.ExecuteIfBound();
							}, RecognitionSeconds);
						}, 20.0);
				});
		});
	});

	// ======================================================================================================
	Describe("Lifetime", [this]()
	{
		LatentIt("DestroyWhileSpeakingIsSilentAndSafe", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches a stream callback that still reaches the destroyed actor (crash after GC) or still broadcasts OnFinalText /
			// OnSampleResult for the utterance that was in flight.
			BeginListening(this, Run, Done, [](AModelPipelinesTalkToTextSample* S) { S->bUseMicrophone = false; },
				[this, Done](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					R->StartPush(PushedSecondsFull);
					TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = S;
					WaitFor(this, [WA]() { const AModelPipelinesTalkToTextSample* S2 = WA.Get(); return !S2 || S2->GetStatusText().Equals(TEXT("Hearing speech..."), ESearchCase::CaseSensitive); },
						TEXT("status 'Hearing speech...'"), Done, [this, R, WA, Done]()
						{
							AModelPipelinesTalkToTextSample* S3 = WA.Get();
							if (!TestNotNull(TEXT("actor alive"), S3)) { Done.ExecuteIfBound(); return; }
							const int32 TextsAtDestroy = R->Rcv->Texts.Num();
							const int32 ResultsAtDestroy = R->Rcv->ResultSuccess.Num();
							S3->Destroy();
							CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
							Settle(3.0, [this, R, TextsAtDestroy, ResultsAtDestroy, Done]()
							{
								TestEqual(TEXT("no OnFinalText after the actor was destroyed"), R->Rcv->Texts.Num(), TextsAtDestroy);
								TestEqual(TEXT("no OnSampleResult after the actor was destroyed"), R->Rcv->ResultSuccess.Num(), ResultsAtDestroy);
								Done.ExecuteIfBound();
							});
						}, RecognitionSeconds);
				});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
