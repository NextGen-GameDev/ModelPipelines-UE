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
#include "SpeechMicrophoneSource.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesTalkToTextSample.h"
#include "SamplesTestData.h"
#include "TalkToTextTestReceiver.h"

using namespace UE::ModelPipelines::SamplesTests;

namespace TalkToTextMicSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(420);
	constexpr double ReadySeconds = 200.0;
	constexpr double RecognitionSeconds = 150.0;
	/** The microphone (re)start result is applied within a few ticks; allow generously. */
	constexpr double MicSeconds = 20.0;
	constexpr float SampleRate = 16000.f;
	constexpr int32 PieceSamples = 320;
	constexpr double PushSpeed = 4.0;
	constexpr int32 MaxPiecesPerTick = 25;
	constexpr double PushedSecondsFull = 20.0;
	constexpr int32 TailSamples = 24000;
	const TCHAR* const RecordingId = TEXT("long_61_70968_45s");
	const TCHAR* const StageAHint = TEXT("Fetch the model with: py Tools/fetch_phonon2.py (see Docs/Samples.md), then export stage A into .pipelines-dev/models/stt/phonon-2-onnx/fp16w/.");
	const TCHAR* const GoldenHint = TEXT("Generate with: python Tools/make_stt_long_goldens.py");

	static TStrongObjectPtr<UNNEModelData> GEncoder;
	static TStrongObjectPtr<UNNEModelData> GDecoder;
	static FString GTokenizerJson;
	static bool bExitHooked = false;
	static TArray<float> GAudio;
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

	/** Loads (once) the first 20 s of the recording. */
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
		bAudioLoaded = true;
		return true;
	}

	struct FRun : public TSharedFromThis<FRun>
	{
		FSamplesTestWorld World;
		TStrongObjectPtr<UTalkToTextTestReceiver> Rcv;
		TStrongObjectPtr<UPipelineModelPackage> Pkg;
		TWeakObjectPtr<AModelPipelinesTalkToTextSample> Actor;
		TWeakObjectPtr<USpeechStream> PushTarget;
		FTSTicker::FDelegateHandle Pusher;
		int32 PushedSamples = 0;
		int32 TotalToPush = 0;
		double PushStart = 0.0;
		bool bPushDone = false;

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
			return A;
		}

		/** Pushes the first Seconds of the audio, then 1.5 s of silence, at 4x real time into the sample's CURRENT stream. */
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
					return false;
				}
				return true;
			}), 0.0f);
		}

		void Teardown()
		{
			if (Pusher.IsValid()) { FTSTicker::GetCoreTicker().RemoveTicker(Pusher); Pusher.Reset(); }
			if (AModelPipelinesTalkToTextSample* A = Actor.Get())
			{
				A->OnFinalText.RemoveAll(Rcv.Get());
				A->OnSampleResult.RemoveAll(Rcv.Get());
			}
			World.Destroy();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		}
	};

	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Next, double Timeout)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TalkToTextMicSpecWait"), 0.02f, [T, Cond, What, Done, Next, Start, Timeout](float)
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

	static void Settle(double Secs, TFunction<void()> Next)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TalkToTextMicSpecSettle"), 0.02f, [Start, Secs, Next](float)
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

	/** Spawns the stage A sample (bUseMicrophone false unless Configure changes it), waits until Ready (and listening when bWaitListening). */
	static void BeginSample(FAutomationTestBase* T, TSharedPtr<FRun>& Slot, FDoneDelegate Done, bool bWaitListening, TFunction<void(AModelPipelinesTalkToTextSample*)> Configure,
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
			[WA, bWaitListening]()
			{
				const AModelPipelinesTalkToTextSample* S = WA.Get();
				if (!S) { return true; }
				const EPipelineSampleState St = S->GetSampleState();
				if (St == EPipelineSampleState::Failed || St == EPipelineSampleState::MissingPackage) { return true; }
				return bWaitListening ? S->IsListening() : St == EPipelineSampleState::Ready;
			},
			bWaitListening ? TEXT("the sample to start listening") : TEXT("the sample to become Ready"), Done,
			[T, R, WA, Done, Next, bWaitListening]()
			{
				AModelPipelinesTalkToTextSample* S = WA.Get();
				const bool bOk = S && (bWaitListening ? S->IsListening() : S->GetSampleState() == EPipelineSampleState::Ready);
				if (!bOk)
				{
					T->AddError(FString::Printf(TEXT("sample not in the wanted state: state %s, status '%s'"),
						S ? *StateName(S->GetSampleState()) : TEXT("<destroyed>"), S ? *S->GetStatusText() : TEXT("")));
					Done.ExecuteIfBound();
					return;
				}
				Next(R, S);
			}, ReadySeconds);
	}

	static bool StatusStarts(const AModelPipelinesTalkToTextSample* S, const TCHAR* Prefix)
	{
		return S && S->GetStatusText().StartsWith(Prefix, ESearchCase::CaseSensitive);
	}

	/** The name GetMicrophoneName() should report for SetMicrophone(-1): the device marked default, "" if none is marked. */
	static FString DefaultDeviceName(const TArray<FSpeechCaptureDevice>& Devices)
	{
		for (const FSpeechCaptureDevice& D : Devices) { if (D.bIsDefault) { return D.Name; } }
		return FString();
	}
}

BEGIN_DEFINE_SPEC(FTalkToTextMicSpec, "Pipelines.Samples.TalkToText.Mic",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	TSharedPtr<TalkToTextMicSpecPrivate::FRun> Run;
END_DEFINE_SPEC(FTalkToTextMicSpec)

void FTalkToTextMicSpec::Define()
{
	using namespace TalkToTextMicSpecPrivate;

	AfterEach([this]()
	{
		if (Run.IsValid())
		{
			Run->Teardown();
			Run.Reset();
		}
	});

	// ======================================================================================================
	Describe("Enumeration", [this]()
	{
		LatentIt("GetMicrophonesEqualsTheCaptureDeviceList", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches a cached/stale or reordered list, or a different filter than USpeechMicrophoneSource: the Blueprint dropdown and the
			// DeviceIndex semantics must match the capture API exactly. Valid with zero devices too (both lists empty).
			BeginSample(this, Run, Done, false, [](AModelPipelinesTalkToTextSample* S) { S->bUseMicrophone = false; S->bAutoStart = true; },
				[this, Done](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					const TArray<FSpeechCaptureDevice> Expected = USpeechMicrophoneSource::GetCaptureDevices();
					const TArray<FSpeechCaptureDevice> Got = S->GetMicrophones();
					AddInfo(FString::Printf(TEXT("%d capture device(s) on this machine (%s branch)"), Expected.Num(), Expected.Num() > 0 ? TEXT("devices-present") : TEXT("no-devices")));
					if (TestEqual(TEXT("GetMicrophones().Num() == GetCaptureDevices().Num()"), Got.Num(), Expected.Num()))
					{
						for (int32 i = 0; i < Got.Num(); ++i)
						{
							TestTrue(*FString::Printf(TEXT("entry %d name '%s' == '%s'"), i, *Got[i].Name, *Expected[i].Name), Got[i].Name.Equals(Expected[i].Name, ESearchCase::CaseSensitive));
							TestEqual(*FString::Printf(TEXT("entry %d DeviceIndex"), i), Got[i].DeviceIndex, Expected[i].DeviceIndex);
							TestEqual(*FString::Printf(TEXT("entry %d bIsDefault"), i), Got[i].bIsDefault, Expected[i].bIsDefault);
						}
					}
					Done.ExecuteIfBound();
				});
		});
	});

	// ======================================================================================================
	Describe("WhileListening", [this]()
	{
		LatentIt("SetMicrophone99ThenDefaultKeepsTheSameStream", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches a restart that tears the stream down (new pointer, lost pushed audio), a missing-device failure that kills listening,
			// and a SetMicrophone that forgets to set DeviceIndex/bUseMicrophone.
			BeginSample(this, Run, Done, true, [](AModelPipelinesTalkToTextSample* S) { S->bUseMicrophone = false; },
				[this, Done](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					USpeechStream* const Before = S->GetStream();
					if (!TestNotNull(TEXT("stream before"), Before)) { Done.ExecuteIfBound(); return; }
					S->SetMicrophone(99);
					TestEqual(TEXT("DeviceIndex == 99"), S->DeviceIndex, 99);
					TestTrue(TEXT("bUseMicrophone true after SetMicrophone"), S->bUseMicrophone);
					TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = S;
					WaitFor(this, [WA]() { return !WA.IsValid() || StatusStarts(WA.Get(), TEXT("No microphone (")); },
						TEXT("status starting with 'No microphone ('"), Done, [this, R, WA, Before, Done]()
						{
							AModelPipelinesTalkToTextSample* S2 = WA.Get();
							if (!TestNotNull(TEXT("actor alive"), S2)) { Done.ExecuteIfBound(); return; }
							TestTrue(TEXT("IsListening stays true"), S2->IsListening());
							TestTrue(TEXT("GetStream() is the SAME stream object"), S2->GetStream() == Before);
							TestFalse(TEXT("IsMicrophoneActive false"), S2->IsMicrophoneActive());
							TestEqual(TEXT("GetMicrophoneName() empty"), S2->GetMicrophoneName(), FString());
							// Pushed audio must still work on the kept stream (6 s -> at least one final).
							R->StartPush(6.0);
							WaitFor(this, [R]() { return R->Rcv->Texts.Num() >= 1; }, TEXT("at least 1 final from 6 s of pushed audio"), Done, [this, R, WA, Before, Done]()
							{
								TestTrue(*FString::Printf(TEXT("got %d finals"), R->Rcv->Texts.Num()), R->Rcv->Texts.Num() >= 1);
								AModelPipelinesTalkToTextSample* S3 = WA.Get();
								if (!TestNotNull(TEXT("actor alive"), S3)) { Done.ExecuteIfBound(); return; }
								// Stop the pusher so it cannot interleave with the next restart.
								if (R->Pusher.IsValid()) { FTSTicker::GetCoreTicker().RemoveTicker(R->Pusher); R->Pusher.Reset(); }

								const TArray<FSpeechCaptureDevice> Devices = USpeechMicrophoneSource::GetCaptureDevices();
								AddInfo(FString::Printf(TEXT("SetMicrophone(-1): %s branch (%d devices)"), Devices.Num() > 0 ? TEXT("devices-present") : TEXT("no-devices"), Devices.Num()));
								S3->SetMicrophone(-1);
								TestEqual(TEXT("DeviceIndex == -1"), S3->DeviceIndex, -1);
								TestTrue(TEXT("bUseMicrophone true"), S3->bUseMicrophone);
								const bool bHave = Devices.Num() > 0;
								WaitFor(this, [WA, bHave]() { return !WA.IsValid() || StatusStarts(WA.Get(), bHave ? TEXT("Listening (") : TEXT("No microphone (")); },
									bHave ? TEXT("status starting with 'Listening ('") : TEXT("status starting with 'No microphone ('"), Done,
									[this, WA, Before, Devices, bHave, Done]()
									{
										AModelPipelinesTalkToTextSample* S4 = WA.Get();
										if (!TestNotNull(TEXT("actor alive"), S4)) { Done.ExecuteIfBound(); return; }
										TestTrue(TEXT("same stream pointer after SetMicrophone(-1)"), S4->GetStream() == Before);
										TestTrue(TEXT("still listening"), S4->IsListening());
										if (bHave)
										{
											TestTrue(TEXT("IsMicrophoneActive true"), S4->IsMicrophoneActive());
											const FString Want = DefaultDeviceName(Devices);
											const FString Name = S4->GetMicrophoneName();
											if (Want.IsEmpty())
											{
												TestFalse(TEXT("no device is marked default: GetMicrophoneName() non-empty"), Name.IsEmpty());
											}
											else
											{
												TestTrue(*FString::Printf(TEXT("GetMicrophoneName() '%s' == default device '%s'"), *Name, *Want), Name.Equals(Want, ESearchCase::CaseSensitive));
											}
										}
										else
										{
											TestFalse(TEXT("no devices: IsMicrophoneActive false"), S4->IsMicrophoneActive());
										}
										Done.ExecuteIfBound();
									}, MicSeconds);
							}, RecognitionSeconds);
						}, MicSeconds);
				});
		});

		LatentIt("NegativeIndexMinusFiveDoesNotCrashAndKeepsTheStream", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches an unchecked index (out-of-range device lookup -> crash) and an invalid index that ends the stream.
			BeginSample(this, Run, Done, true, [](AModelPipelinesTalkToTextSample* S) { S->bUseMicrophone = false; },
				[this, Done](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					USpeechStream* const Before = S->GetStream();
					S->SetMicrophone(-5);
					TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = S;
					WaitFor(this, [WA]() { return !WA.IsValid() || StatusStarts(WA.Get(), TEXT("No microphone (")); },
						TEXT("status starting with 'No microphone (' after SetMicrophone(-5)"), Done, [this, WA, Before, Done]()
						{
							if (AModelPipelinesTalkToTextSample* S2 = WA.Get())
							{
								TestTrue(TEXT("still listening"), S2->IsListening());
								TestTrue(TEXT("stream kept"), S2->GetStream() == Before);
								TestFalse(TEXT("IsMicrophoneActive false"), S2->IsMicrophoneActive());
							}
							else { AddError(TEXT("actor destroyed")); }
							Done.ExecuteIfBound();
						}, MicSeconds);
				});
		});

		LatentIt("IndexZeroSelectsTheFirstListedDevice", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches SetMicrophone(0) being treated as "default" (the default is not always device 0) or mapping to the wrong name.
			const TArray<FSpeechCaptureDevice> Devices = USpeechMicrophoneSource::GetCaptureDevices();
			if (Devices.Num() == 0)
			{
				AddInfo(TEXT("SKIPPED the device-0 assertions: no capture device on this machine (the no-device path is covered by the 99 / -5 cases)."));
				Done.ExecuteIfBound();
				return;
			}
			AddInfo(FString::Printf(TEXT("devices-present branch: %d device(s), expecting '%s'"), Devices.Num(), *Devices[0].Name));
			BeginSample(this, Run, Done, true, [](AModelPipelinesTalkToTextSample* S) { S->bUseMicrophone = false; },
				[this, Done, Devices](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					S->SetMicrophone(0);
					TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = S;
					WaitFor(this, [WA]() { return !WA.IsValid() || WA->IsMicrophoneActive(); }, TEXT("IsMicrophoneActive after SetMicrophone(0)"), Done,
						[this, WA, Devices, Done]()
						{
							if (AModelPipelinesTalkToTextSample* S2 = WA.Get())
							{
								const FString Name = S2->GetMicrophoneName();
								TestTrue(*FString::Printf(TEXT("GetMicrophoneName() '%s' == GetMicrophones()[0] '%s'"), *Name, *Devices[0].Name), Name.Equals(Devices[0].Name, ESearchCase::CaseSensitive));
								TestEqual(TEXT("GetMicrophones()[0] matches the capture list"), S2->GetMicrophones()[0].Name, Devices[0].Name);
								TestTrue(TEXT("still listening"), S2->IsListening());
							}
							else { AddError(TEXT("actor destroyed")); }
							Done.ExecuteIfBound();
						}, MicSeconds);
				});
		});
	});

	// ======================================================================================================
	Describe("WhileNotListening", [this]()
	{
		LatentIt("SetMicrophoneIsStoredAndAppliedAtStart", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches a SetMicrophone that starts the stream/microphone while stopped, or one that is lost at the next StartListening.
			BeginSample(this, Run, Done, false, [](AModelPipelinesTalkToTextSample* S) { S->bAutoStart = false; S->bUseMicrophone = false; },
				[this, Done](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					TestFalse(TEXT("not listening before"), S->IsListening());
					S->SetMicrophone(0);
					TestEqual(TEXT("DeviceIndex == 0"), S->DeviceIndex, 0);
					TestTrue(TEXT("bUseMicrophone true"), S->bUseMicrophone);
					TestFalse(TEXT("IsListening still false"), S->IsListening());
					TestNull(TEXT("GetStream() still null"), S->GetStream());
					TestFalse(TEXT("microphone not active"), S->IsMicrophoneActive());
					const bool bHave = USpeechMicrophoneSource::GetCaptureDevices().Num() > 0;
					AddInfo(FString::Printf(TEXT("StartListening: %s branch"), bHave ? TEXT("devices-present") : TEXT("no-devices")));
					S->StartListening();
					TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = S;
					WaitFor(this, [WA]() { return !WA.IsValid() || WA->IsListening(); }, TEXT("IsListening after StartListening"), Done,
						[this, WA, bHave, Done]()
						{
							WaitFor(this,
								[WA, bHave]() { return !WA.IsValid() || (bHave ? WA->IsMicrophoneActive() : StatusStarts(WA.Get(), TEXT("No microphone ("))); },
								bHave ? TEXT("IsMicrophoneActive after StartListening") : TEXT("status starting with 'No microphone ('"), Done,
								[this, WA, bHave, Done]()
								{
									if (AModelPipelinesTalkToTextSample* S2 = WA.Get())
									{
										TestTrue(TEXT("listening"), S2->IsListening());
										TestNotNull(TEXT("stream exists"), S2->GetStream());
										TestEqual(TEXT("IsMicrophoneActive matches the device availability"), S2->IsMicrophoneActive(), bHave);
									}
									else { AddError(TEXT("actor destroyed")); }
									Done.ExecuteIfBound();
								}, MicSeconds);
						}, ReadySeconds);
				});
		});
	});

	// ======================================================================================================
	Describe("BadPackage", [this]()
	{
		LatentIt("SetMicrophoneOnAMissingPackageIsHarmless", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches a SetMicrophone that dereferences a pipeline/stream that never existed, or that starts listening on an unusable sample.
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
				TEXT("state to leave Loading"), Done, [this, WA, Done]()
				{
					AModelPipelinesTalkToTextSample* S = WA.Get();
					if (!TestNotNull(TEXT("actor alive"), S)) { Done.ExecuteIfBound(); return; }
					S->SetMicrophone(-1);
					Settle(0.5, [this, WA, Done]()
					{
						if (AModelPipelinesTalkToTextSample* S2 = WA.Get())
						{
							TestFalse(TEXT("IsListening false"), S2->IsListening());
							TestNull(TEXT("no stream"), S2->GetStream());
							TestEqual(TEXT("state still MissingPackage"), StateName(S2->GetSampleState()), StateName(EPipelineSampleState::MissingPackage));
						}
						Done.ExecuteIfBound();
					});
				}, 10.0);
		});
	});

	// ======================================================================================================
	Describe("Lifetime", [this]()
	{
		LatentIt("DestroyRightAfterSetMicrophoneIsSafe", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Catches a capture-restart callback that still reaches the destroyed actor (crash after GC).
			BeginSample(this, Run, Done, true, [](AModelPipelinesTalkToTextSample* S) { S->bUseMicrophone = false; },
				[this, Done](TSharedRef<FRun> R, AModelPipelinesTalkToTextSample* S)
				{
					AddInfo(USpeechMicrophoneSource::GetCaptureDevices().Num() > 0 ? TEXT("devices-present branch") : TEXT("no-devices branch"));
					S->SetMicrophone(0);
					S->Destroy();
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					Settle(2.0, [this, Done]()
					{
						AddInfo(TEXT("survived 2 s of ticks after Destroy + GC"));
						Done.ExecuteIfBound();
					});
				});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
