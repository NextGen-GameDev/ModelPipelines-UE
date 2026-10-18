// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9R.4: USpeechMicrophoneSource, USpeechListenAsyncAction ("Start Listening"), StopListening, USpeechStream::PushAudioArray.
// Written from docs/briefs/WP9R.4.md only. The test machine may have no microphone and headless runs may have no capture backend:
// device-dependent checks branch on GetCaptureDevices().Num() (AddInfo says which branch ran). Text comparisons are CASE-SENSITIVE.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Ticker.h"
#include "Engine/CancellableAsyncAction.h"
#include "HAL/PlatformTime.h"
#include "Misc/OutputDevice.h"
#include "Misc/OutputDeviceRedirector.h"
#include "Misc/ScopeLock.h"
#include "UObject/Package.h"
#include "UObject/Class.h"
#include "UObject/UnrealType.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineModelPackage.h"
#include "PipelineTypes.h"
#include "SpeechPipelineLibrary.h"
#include "SpeechStream.h"
#include "SpeechToTextPipeline.h"
#include "SpeechTypes.h"
#include "SpeechMicrophoneSource.h"
#include "SpeechListenAsyncAction.h"
#include "SpeechFakeEngine.h"
#include "SpeechListenTestReceiver.h"
#include "SpeechTestData.h"

using namespace UE::ModelPipelines::SpeechTests;

namespace SpeechListenSpecPrivate
{
	// Same chain pattern as SpeechStreamSpec (lead fix 2026-10-18): Done fires once, when the LAST scheduled step finished.
	// Nested steps must be given the real Done (an unbound Done would swallow the final FinishStep and hang the test).
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
		mutable FCriticalSection Lock;
		TArray<FString> Lines;
		TArray<uint8> Verbosities;
	};

	/** Lead-in silence, then NumBursts x (tone burst, silence gap), mono at Rate. Frames are 20 ms. */
	static void Script(TArray<float>& A, int32 Rate, int32 NumBursts, int32 BurstFrames = 25, int32 GapFrames = 30, int32 LeadFrames = 25)
	{
		A.Reset();
		const int32 PerFrame = Rate / 50;
		A.AddZeroed(LeadFrames * PerFrame);
		for (int32 b = 0; b < NumBursts; ++b)
		{
			TArray<float> T;
			MakeToneSamples(T, BurstFrames * PerFrame, Rate, 440.0, 0.2);
			A.Append(T);
			A.AddZeroed(GapFrames * PerFrame);
		}
	}

	static void Interleave(const TArray<float>& Mono, int32 Channels, TArray<float>& Out)
	{
		Out.Reset();
		for (float V : Mono) { for (int32 c = 0; c < Channels; ++c) { Out.Add(V); } }
	}

	/** Pushes Audio through the Blueprint wrapper PushAudioArray from the game thread, PerTick 20 ms chunks per 50 ms tick. */
	static void PushArrayPaced(USpeechStream* Stream, const TArray<float>& Audio, int32 Rate, int32 Channels, int32 PerTick)
	{
		const int32 ChunkSamples = (Rate / 50) * Channels;
		const TSharedRef<TArray<float>> A = MakeShared<TArray<float>>(Audio);
		const TSharedRef<int32> Pos = MakeShared<int32>(0);
		const TWeakObjectPtr<USpeechStream> Weak(Stream);
		FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Weak, A, Pos, Rate, Channels, PerTick, ChunkSamples](float) -> bool
		{
			USpeechStream* S = Weak.Get();
			for (int32 k = 0; k < PerTick && *Pos < A->Num(); ++k)
			{
				const int32 N = FMath::Min(ChunkSamples, A->Num() - *Pos);
				TArray<float> Chunk(A->GetData() + *Pos, N);
				if (S) { S->PushAudioArray(Chunk, Rate, Channels); }
				*Pos += N;
			}
			return *Pos < A->Num();
		}), 0.05f);
	}

	/** Grammar: per utterance Started < Partial* < Ended < Final; Stopped (if bExpectStopped) exactly once and last. Empty string = ok. */
	static FString CheckOrder(const USpeechListenTestReceiver* R, int32 NumUtterances, bool bExpectStopped)
	{
		int32 State = 0;  // 0 outside, 1 speaking, 2 ended awaiting final
		int32 Utt = 0;
		for (int32 i = 0; i < R->Records.Num(); ++i)
		{
			switch (R->Records[i].Type)
			{
			case ESpeechListenTestEvent::Started: if (State != 0) { return FString::Printf(TEXT("Started at #%d in state %d"), i, State); } State = 1; break;
			case ESpeechListenTestEvent::Partial: if (State != 1) { return FString::Printf(TEXT("Partial at #%d in state %d"), i, State); } break;
			case ESpeechListenTestEvent::Ended: if (State != 1) { return FString::Printf(TEXT("Ended at #%d in state %d"), i, State); } State = 2; break;
			case ESpeechListenTestEvent::Final: if (State != 2) { return FString::Printf(TEXT("Final at #%d in state %d"), i, State); } State = 0; ++Utt; break;
			case ESpeechListenTestEvent::Stopped: if (i != R->Records.Num() - 1) { return FString::Printf(TEXT("Stopped at #%d is not last"), i); } break;
			default: return FString::Printf(TEXT("unexpected Failure event at #%d"), i);
			}
		}
		if (Utt != NumUtterances) { return FString::Printf(TEXT("%d finals, expected %d"), Utt, NumUtterances); }
		if (bExpectStopped && R->Count(ESpeechListenTestEvent::Stopped) != 1) { return TEXT("Stopped count != 1"); }
		return FString();
	}
}

using namespace SpeechListenSpecPrivate;

BEGIN_DEFINE_SPEC(FSpeechListenSpec, "Pipelines.Speech.Listen",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	TSharedPtr<FFakeEngineState, ESPMode::ThreadSafe> State;
	TSharedPtr<FFakePhononBackend> Backend;
	bool bRegistered = false;
	TStrongObjectPtr<UPipelineModelPackage> Pkg;
	TStrongObjectPtr<USpeechToTextPipeline> Pipe;
	TStrongObjectPtr<USpeechListenTestReceiver> Recv;
	TStrongObjectPtr<USpeechMicrophoneSource> Mic;
	TWeakObjectPtr<USpeechListenAsyncAction> Action;
	TSharedPtr<FLogCapture> Log;

	/** StartListening + bind the six pins + Activate (what the Blueprint node does). */
	USpeechListenAsyncAction* Begin(USpeechToTextPipeline* Pipeline, bool bMic, int32 DeviceIndex = -1)
	{
		USpeechListenAsyncAction* A = USpeechListenAsyncAction::StartListening(nullptr, Pipeline, FSpeechStreamSettings(), bMic, DeviceIndex);
		if (!A) { AddError(TEXT("StartListening returned null")); return nullptr; }
		Action = A;
		Recv.Reset(NewObject<USpeechListenTestReceiver>(GetTransientPackage()));
		A->OnSpeechStarted.AddDynamic(Recv.Get(), &USpeechListenTestReceiver::HandleStarted);
		A->OnPartial.AddDynamic(Recv.Get(), &USpeechListenTestReceiver::HandlePartial);
		A->OnFinal.AddDynamic(Recv.Get(), &USpeechListenTestReceiver::HandleFinal);
		A->OnSpeechEnded.AddDynamic(Recv.Get(), &USpeechListenTestReceiver::HandleEnded);
		A->OnStopped.AddDynamic(Recv.Get(), &USpeechListenTestReceiver::HandleStopped);
		A->OnFailure.AddDynamic(Recv.Get(), &USpeechListenTestReceiver::HandleFailure);
		A->Activate();
		return A;
	}

	USpeechStream* NewStream()
	{
		FPipelineResult R;
		USpeechStream* S = Pipe.IsValid() ? Pipe->StartStream(FSpeechStreamSettings(), R) : nullptr;
		if (!S) { AddError(FString::Printf(TEXT("StartStream on the fake pipeline failed: %s"), *R.Message)); }
		return S;
	}
END_DEFINE_SPEC(FSpeechListenSpec)

void FSpeechListenSpec::Define()
{
	BeforeEach([this]()
	{
		GPendingSteps = 0;
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
		if (Action.IsValid()) { Action->Cancel(); }
		Action.Reset();
		if (Mic.IsValid()) { Mic->Stop(); }
		Mic.Reset();
		Recv.Reset();
		Pipe.Reset();
		Pkg.Reset();
		Log.Reset();
		if (bRegistered && Backend.IsValid()) { FModelPipelinesCoreModule::Get().UnregisterBackend(Backend.ToSharedRef()); bRegistered = false; }
		Backend.Reset();
		State.Reset();
	});

	// =================================================================================================================
	Describe("Reflection", [this]()
	{
		It("FactoryIsInternalUseOnlyAndSixPinsShareOneSignature", [this]()
		{
			UClass* C = USpeechListenAsyncAction::StaticClass();
			TestTrue(TEXT("derives from UCancellableAsyncAction"), C->IsChildOf(UCancellableAsyncAction::StaticClass()));
			UFunction* F = C->FindFunctionByName(TEXT("StartListening"));
			if (!TestNotNull(TEXT("StartListening UFUNCTION exists"), F)) { return; }
#if WITH_METADATA   // metadata exists only in editor builds; game targets compile the tests too
			TestTrue(TEXT("BlueprintInternalUseOnly"), F->HasMetaData(TEXT("BlueprintInternalUseOnly")));
			TestEqual(TEXT("DisplayName"), F->GetMetaData(TEXT("DisplayName")), FString(TEXT("Start Listening")));
#endif
			TestTrue(TEXT("BlueprintCallable"), F->HasAnyFunctionFlags(FUNC_BlueprintCallable));

			const TCHAR* Pins[] = { TEXT("OnSpeechStarted"), TEXT("OnPartial"), TEXT("OnFinal"), TEXT("OnSpeechEnded"), TEXT("OnStopped"), TEXT("OnFailure") };
			UFunction* Sig0 = nullptr;
			for (const TCHAR* P : Pins)
			{
				FMulticastDelegateProperty* Prop = CastField<FMulticastDelegateProperty>(C->FindPropertyByName(P));
				if (!TestNotNull(*FString::Printf(TEXT("pin %s is a multicast delegate property"), P), Prop)) { continue; }
				TestTrue(*FString::Printf(TEXT("pin %s is BlueprintAssignable"), P), Prop->HasAnyPropertyFlags(CPF_BlueprintAssignable));
				if (!Sig0) { Sig0 = Prop->SignatureFunction; }
				// K2Node_BaseAsyncTask needs one signature for every pin.
				TestTrue(*FString::Printf(TEXT("pin %s uses the same signature function as the first pin"), P), Prop->SignatureFunction == Sig0);
			}
			if (Sig0) { TestEqual(TEXT("signature has Transcript + Result"), (int32)Sig0->NumParms, 2); }
		});

		It("StopListeningAndPushSpeechAudioAreBlueprintCallableWithDisplayNames", [this]()
		{
			UFunction* Stop = USpeechPipelineLibrary::StaticClass()->FindFunctionByName(TEXT("StopListening"));
			if (TestNotNull(TEXT("StopListening on USpeechPipelineLibrary"), Stop))
			{
#if WITH_METADATA
				TestEqual(TEXT("DisplayName"), Stop->GetMetaData(TEXT("DisplayName")), FString(TEXT("Stop Listening")));
#endif
				TestTrue(TEXT("static"), Stop->HasAnyFunctionFlags(FUNC_Static));
			}
			UFunction* Push = USpeechStream::StaticClass()->FindFunctionByName(TEXT("PushAudioArray"));
			if (TestNotNull(TEXT("PushAudioArray on USpeechStream"), Push))
			{
#if WITH_METADATA
				TestEqual(TEXT("DisplayName"), Push->GetMetaData(TEXT("DisplayName")), FString(TEXT("Push Speech Audio")));
#endif
				TestTrue(TEXT("BlueprintCallable"), Push->HasAnyFunctionFlags(FUNC_BlueprintCallable));
			}
			TestNotNull(TEXT("GetStream UFUNCTION"), USpeechListenAsyncAction::StaticClass()->FindFunctionByName(TEXT("GetStream")));
		});
	});

	// =================================================================================================================
	Describe("MicrophoneSource", [this]()
	{
		It("GetCaptureDevicesNeverCrashesAndIsConsistent", [this]()
		{
			const TArray<FSpeechCaptureDevice> Devs = USpeechMicrophoneSource::GetCaptureDevices();
			AddInfo(FString::Printf(TEXT("GetCaptureDevices: %d device(s)"), Devs.Num()));
			int32 Defaults = 0;
			for (const FSpeechCaptureDevice& D : Devs)
			{
				Defaults += D.bIsDefault ? 1 : 0;
				AddInfo(FString::Printf(TEXT("  #%d '%s' ch=%d rate=%d default=%d"), D.DeviceIndex, *D.Name, D.NumChannels, D.SampleRate, D.bIsDefault ? 1 : 0));
				TestTrue(TEXT("DeviceIndex >= 0"), D.DeviceIndex >= 0);
				TestTrue(TEXT("NumChannels > 0"), D.NumChannels > 0);
				TestTrue(TEXT("SampleRate > 0"), D.SampleRate > 0);
			}
			TestTrue(TEXT("at most one default device"), Defaults <= 1);
		});

		It("StartWithNullStreamIsInvalidArgument", [this]()
		{
			USpeechMicrophoneSource* M = NewObject<USpeechMicrophoneSource>(GetTransientPackage());
			Mic.Reset(M);
			FPipelineResult R;
			TestFalse(TEXT("Start(null) returns false"), M->Start(nullptr, -1, R));
			TestFalse(TEXT("OutResult is a failure"), R.bSuccess);
			TestTrue(TEXT("InvalidArgument"), R.Error == EPipelineError::InvalidArgument);
			TestFalse(TEXT("not capturing"), M->IsCapturing());
		});

		It("StartWithBadIndexIsInvalidArgumentAndStopTwiceIsANoOp", [this]()
		{
			USpeechStream* S = NewStream();
			if (!S) { return; }
			USpeechMicrophoneSource* M = NewObject<USpeechMicrophoneSource>(GetTransientPackage());
			Mic.Reset(M);
			const int32 NumDevices = USpeechMicrophoneSource::GetCaptureDevices().Num();
			for (int32 Bad : { 9999, -5 })
			{
				FPipelineResult R;
				TestFalse(*FString::Printf(TEXT("Start(index %d) returns false"), Bad), M->Start(S, Bad, R));
				TestFalse(TEXT("failure result"), R.bSuccess);
				if (NumDevices > 0)
				{
					TestTrue(*FString::Printf(TEXT("InvalidArgument for index %d (%s)"), Bad, *R.Message), R.Error == EPipelineError::InvalidArgument);
				}
				else
				{
					AddInfo(TEXT("no capture device: a bad index may answer RuntimeUnavailable or InvalidArgument"));
					TestTrue(TEXT("InvalidArgument or RuntimeUnavailable"), R.Error == EPipelineError::InvalidArgument || R.Error == EPipelineError::RuntimeUnavailable);
				}
				TestFalse(TEXT("not capturing after a failed Start"), M->IsCapturing());
			}
			M->Stop();
			M->Stop();
			TestFalse(TEXT("Stop twice, never started: no-op"), M->IsCapturing());
			TestTrue(TEXT("a failed Start does not stop the stream"), S->IsRunning());
			S->Stop();
		});

		It("NoDeviceStartIsRuntimeUnavailable", [this]()
		{
			if (USpeechMicrophoneSource::GetCaptureDevices().Num() > 0) { AddInfo(TEXT("a capture device exists: the no-device branch is not testable here (see the capture test)")); return; }
			AddInfo(TEXT("no capture device"));
			USpeechStream* S = NewStream();
			if (!S) { return; }
			USpeechMicrophoneSource* M = NewObject<USpeechMicrophoneSource>(GetTransientPackage());
			Mic.Reset(M);
			FPipelineResult R;
			TestFalse(TEXT("Start(default) returns false"), M->Start(S, -1, R));
			TestTrue(TEXT("RuntimeUnavailable"), R.Error == EPipelineError::RuntimeUnavailable);
			TestTrue(TEXT("message names the missing device"), R.Message.Contains(TEXT("no audio capture device"), ESearchCase::IgnoreCase));
			TestFalse(TEXT("not capturing"), M->IsCapturing());
			S->Stop();
		});

		LatentIt("CapturesFromTheDefaultDeviceThenStops", FTimespan::FromSeconds(60), [this](const FDoneDelegate& Done)
		{
			if (USpeechMicrophoneSource::GetCaptureDevices().Num() == 0) { AddInfo(TEXT("no capture device: capture test skipped")); Done.Execute(); return; }
			USpeechStream* S = NewStream();
			if (!S) { Done.Execute(); return; }
			USpeechMicrophoneSource* M = NewObject<USpeechMicrophoneSource>(GetTransientPackage());
			Mic.Reset(M);
			FPipelineResult R;
			if (!M->Start(S, -1, R))
			{
				AddInfo(FString::Printf(TEXT("device listed but Start failed (error %d '%s'): capture test skipped"), (int32)R.Error, *R.Message));
				S->Stop();
				Done.Execute();
				return;
			}
			TestTrue(TEXT("IsCapturing after Start"), M->IsCapturing());
			FPipelineResult R2;
			TestFalse(TEXT("second Start while capturing returns false"), M->Start(S, -1, R2));
			TestTrue(TEXT("second Start: InvalidArgument"), R2.Error == EPipelineError::InvalidArgument);
			TestTrue(TEXT("still capturing after the refused second Start"), M->IsCapturing());

			const TWeakObjectPtr<USpeechStream> WS(S);
			Grace(2.5, Done, [this, M, WS, Done]()
			{
				TestTrue(TEXT("stream still running while capturing"), WS.IsValid() && WS->IsRunning());
				int64 Fed = 0;
				{ FScopeLock L(&State->Lock); Fed = State->FedSamples; }
				// The VAD may gate a silent room, so the fed-sample count is information; the hard checks are lifecycle ones.
				AddInfo(FString::Printf(TEXT("engine received %lld samples (16 kHz) in 2.5 s; a live mic should give >= 8000"), (long long)Fed));
				M->Stop();
				TestFalse(TEXT("not capturing after Stop"), M->IsCapturing());
				TestTrue(TEXT("Stop closes the device but does not stop the stream"), WS.IsValid() && WS->IsRunning());
				M->Stop();   // twice
				Grace(0.5, Done, [this, WS, Done]()
				{
					int32 Calls0 = 0;
					{ FScopeLock L(&State->Lock); Calls0 = State->FeedCalls; }
					Grace(0.8, Done, [this, Calls0]()
					{
						int32 Calls1 = 0;
						{ FScopeLock L(&State->Lock); Calls1 = State->FeedCalls; }
						TestEqual(TEXT("no more audio reaches the engine after Stop"), Calls1, Calls0);
					});
					if (WS.IsValid()) { WS->Stop(); }
				});
			});
		});
	});

	// =================================================================================================================
	Describe("PushSpeechAudio", [this]()
	{
		It("BadArgumentsAreDroppedWithoutCrash", [this]()
		{
			USpeechStream* S = NewStream();
			if (!S) { return; }
			S->PushAudioArray(TArray<float>(), 16000, 1);
			S->PushAudioArray(TArray<float>({ 0.1f, 0.2f }), 0, 1);
			S->PushAudioArray(TArray<float>({ 0.1f, 0.2f }), 16000, 0);
			S->PushAudioArray(TArray<float>({ 0.1f, 0.2f }), -3, -1);
			TestTrue(TEXT("stream survives bad pushes"), S->IsRunning());
			S->Stop();
		});

		LatentIt("StereoAt48kGivesOneUtterance", FTimespan::FromSeconds(60), [this](const FDoneDelegate& Done)
		{
			// The wrapper must forward rate and channel count: treating 48 kHz stereo as 16 kHz mono would stretch the signal 6x and
			// break the VAD timing (the gap would be too short to end the utterance, or the count would differ).
			USpeechListenAsyncAction* A = Begin(Pipe.Get(), false);
			if (!A || !A->GetStream()) { AddError(TEXT("no stream after Activate")); Done.Execute(); return; }
			TArray<float> Mono, Stereo;
			Script(Mono, 48000, 1);
			Interleave(Mono, 2, Stereo);
			PushArrayPaced(A->GetStream(), Stereo, 48000, 2, 8);
			WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Final) >= 1; }, TEXT("one OnFinal from 48 kHz stereo"), 30.0, Done, [this]()
			{
				TestEqual(TEXT("exactly one Started"), Recv->Count(ESpeechListenTestEvent::Started), 1);
				if (Action.IsValid()) { USpeechPipelineLibrary::StopListening(Action.Get()); }
			});
		});
	});

	// =================================================================================================================
	Describe("ListenWithoutMicrophone", [this]()
	{
		LatentIt("TwoUtterancesArriveInOrderThenStopListeningFiresOnStoppedOnce", FTimespan::FromSeconds(120), [this](const FDoneDelegate& Done)
		{
			USpeechListenAsyncAction* A = Begin(Pipe.Get(), false);
			if (!A) { Done.Execute(); return; }
			USpeechStream* S = A->GetStream();
			if (!TestNotNull(TEXT("GetStream after Activate"), S)) { Done.Execute(); return; }
			TestEqual(TEXT("no event synchronously inside Activate"), Recv->Records.Num(), 0);
			TArray<float> Audio;
			Script(Audio, 16000, 2);
			PushArrayPaced(S, Audio, 16000, 1, 8);
			WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Final) >= 2; }, TEXT("two OnFinal"), 40.0, Done, [this]()
			{
				const FString Err = CheckOrder(Recv.Get(), 2, false);
				TestTrue(*FString::Printf(TEXT("pin order Started < Partial* < Ended < Final per utterance (%s)"), *Err), Err.IsEmpty());
				TestEqual(TEXT("two Started"), Recv->Count(ESpeechListenTestEvent::Started), 2);
				TestEqual(TEXT("two Ended"), Recv->Count(ESpeechListenTestEvent::Ended), 2);
				TestTrue(TEXT("at least one Partial (fake engine supports partials, bPartials default true)"), Recv->Count(ESpeechListenTestEvent::Partial) >= 1);
				const TArray<FSpeechListenTestRecord> F = Recv->Of(ESpeechListenTestEvent::Final);
				if (F.Num() >= 2)
				{
					TestTrue(TEXT("final 0 text 'utt 0'"), F[0].Transcript.Text.Equals(TEXT("utt 0"), ESearchCase::CaseSensitive));
					TestTrue(TEXT("final 1 text 'utt 1'"), F[1].Transcript.Text.Equals(TEXT("utt 1"), ESearchCase::CaseSensitive));
				}
				for (const FSpeechListenTestRecord& R : Recv->Records)
				{
					TestTrue(TEXT("every pin fires on the game thread"), R.bGameThread);
					TestTrue(TEXT("every pin carries an Ok result"), R.Result.bSuccess);
				}
				for (const FSpeechListenTestRecord& R : Recv->Of(ESpeechListenTestEvent::Started)) { TestTrue(TEXT("OnSpeechStarted transcript is empty"), R.Transcript.Text.IsEmpty()); }
				for (const FSpeechListenTestRecord& R : Recv->Of(ESpeechListenTestEvent::Ended)) { TestTrue(TEXT("OnSpeechEnded transcript is empty"), R.Transcript.Text.IsEmpty()); }
				TestEqual(TEXT("no OnStopped before Stop Listening"), Recv->Count(ESpeechListenTestEvent::Stopped), 0);
				TestEqual(TEXT("no OnFailure"), Recv->Count(ESpeechListenTestEvent::Failure), 0);
				USpeechPipelineLibrary::StopListening(Action.Get());
				USpeechPipelineLibrary::StopListening(Action.Get());   // a second call must not double-fire
			});
			WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Stopped) >= 1; }, TEXT("OnStopped"), 40.0, Done, [this, Done]()
			{
				Grace(0.5, Done, [this]()
				{
					TestEqual(TEXT("OnStopped exactly once"), Recv->Count(ESpeechListenTestEvent::Stopped), 1);
					const TArray<FSpeechListenTestRecord> St = Recv->Of(ESpeechListenTestEvent::Stopped);
					if (St.Num() == 1) { TestTrue(TEXT("stop result is Ok after a clean stop"), St[0].Result.bSuccess); }
					const FString Err = CheckOrder(Recv.Get(), 2, true);
					TestTrue(*FString::Printf(TEXT("Stopped is last, order intact (%s)"), *Err), Err.IsEmpty());
				});
			});
		});

		LatentIt("CancelEqualsStopListeningMidUtterance", FTimespan::FromSeconds(120), [this](const FDoneDelegate& Done)
		{
			// Cancel() while an utterance is open: the stream finalises it (no hangover wait); OnStopped still fires, once, last.
			USpeechListenAsyncAction* A = Begin(Pipe.Get(), false);
			if (!A || !A->GetStream()) { AddError(TEXT("no stream after Activate")); Done.Execute(); return; }
			TArray<float> Audio;
			Script(Audio, 16000, 1, 100, 0);   // 2 s tone, no trailing gap
			PushArrayPaced(A->GetStream(), Audio, 16000, 1, 8);
			WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Started) >= 1; }, TEXT("OnSpeechStarted"), 30.0, Done, [this]()
			{
				if (Action.IsValid()) { Action->Cancel(); }
			});
			WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Stopped) >= 1; }, TEXT("OnStopped after Cancel"), 30.0, Done, [this, Done]()
			{
				Grace(0.5, Done, [this]()
				{
					TestEqual(TEXT("OnStopped exactly once"), Recv->Count(ESpeechListenTestEvent::Stopped), 1);
					const FString Err = CheckOrder(Recv.Get(), Recv->Count(ESpeechListenTestEvent::Final), true);
					TestTrue(*FString::Printf(TEXT("order valid after Cancel (%s)"), *Err), Err.IsEmpty());
				});
			});
		});

		LatentIt("NullPipelineFailsOnceOnALaterTick", FTimespan::FromSeconds(60), [this](const FDoneDelegate& Done)
		{
			USpeechListenAsyncAction* A = Begin(nullptr, false);
			if (!A) { Done.Execute(); return; }
			TestEqual(TEXT("OnFailure is not fired inside Activate"), Recv->Count(ESpeechListenTestEvent::Failure), 0);
			TestNull(TEXT("GetStream is null after failure"), A->GetStream());
			WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Failure) >= 1; }, TEXT("OnFailure"), 10.0, Done, [this, Done]()
			{
				const TArray<FSpeechListenTestRecord> F = Recv->Of(ESpeechListenTestEvent::Failure);
				if (F.Num() >= 1)
				{
					TestFalse(TEXT("failure result"), F[0].Result.bSuccess);
					TestTrue(TEXT("InvalidArgument"), F[0].Result.Error == EPipelineError::InvalidArgument);
					TestTrue(TEXT("message 'Pipeline is null'"), F[0].Result.Message.Contains(TEXT("Pipeline is null"), ESearchCase::IgnoreCase));
					TestTrue(TEXT("on the game thread"), F[0].bGameThread);
				}
				Grace(0.5, Done, [this]()
				{
					TestEqual(TEXT("OnFailure exactly once"), Recv->Count(ESpeechListenTestEvent::Failure), 1);
					TestEqual(TEXT("no pin other than OnFailure fired"), Recv->Records.Num(), 1);
				});
			});
		});

		LatentIt("NotReadyPipelineFailsOnce", FTimespan::FromSeconds(60), [this](const FDoneDelegate& Done)
		{
			USpeechToTextPipeline* Bare = NewObject<USpeechToTextPipeline>(GetTransientPackage());   // never initialised
			const TSharedRef<TStrongObjectPtr<USpeechToTextPipeline>> Keep = MakeShared<TStrongObjectPtr<USpeechToTextPipeline>>(Bare);
			USpeechListenAsyncAction* A = Begin(Bare, false);
			if (!A) { Done.Execute(); return; }
			WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Failure) >= 1; }, TEXT("OnFailure for a not-ready pipeline"), 10.0, Done, [this, Keep, Done]()
			{
				const TArray<FSpeechListenTestRecord> F = Recv->Of(ESpeechListenTestEvent::Failure);
				if (F.Num() >= 1)
				{
					TestFalse(TEXT("failure result"), F[0].Result.bSuccess);
					TestFalse(TEXT("with a message"), F[0].Result.Message.IsEmpty());
				}
				Grace(0.5, Done, [this, Keep]()
				{
					TestEqual(TEXT("OnFailure exactly once"), Recv->Count(ESpeechListenTestEvent::Failure), 1);
					TestEqual(TEXT("no OnStopped after a failed start"), Recv->Count(ESpeechListenTestEvent::Stopped), 0);
				});
			});
		});

		LatentIt("SecondActivateWarnsAndChangesNothing", FTimespan::FromSeconds(60), [this](const FDoneDelegate& Done)
		{
			USpeechListenAsyncAction* A = Begin(Pipe.Get(), false);
			if (!A || !A->GetStream()) { AddError(TEXT("no stream after Activate")); Done.Execute(); return; }
			USpeechStream* S0 = A->GetStream();
			const int32 W0 = Log->CountWarnings();
			A->Activate();
			GLog->Flush();   // lead fix: the log redirector dispatches to output devices asynchronously
			TestTrue(TEXT("second Activate logs a Warning"), Log->CountWarnings() > W0);
			TestTrue(TEXT("the stream is the same object (no second StartStream)"), A->GetStream() == S0);
			{ FScopeLock L(&State->Lock); TestTrue(TEXT("engine opened at most one stream (opened lazily at the first utterance, C81c rule 4)"), State->OpenStreamCount <= 1); }
			Grace(0.3, Done, [this]()
			{
				TestEqual(TEXT("no events from a second Activate"), Recv->Records.Num(), 0);
				if (Action.IsValid()) { USpeechPipelineLibrary::StopListening(Action.Get()); }
			});
		});

		It("StopListeningWithNullIsAWarningNoOp", [this]()
		{
			const int32 W0 = Log->CountWarnings();
			USpeechPipelineLibrary::StopListening(nullptr);
			GLog->Flush();   // lead fix: the log redirector dispatches to output devices asynchronously
			TestTrue(TEXT("a Warning was logged"), Log->CountWarnings() > W0);
		});

		LatentIt("ReceiverDestroyedMidListeningDoesNotCrash", FTimespan::FromSeconds(120), [this](const FDoneDelegate& Done)
		{
			USpeechListenAsyncAction* A = Begin(Pipe.Get(), false);
			if (!A || !A->GetStream()) { AddError(TEXT("no stream after Activate")); Done.Execute(); return; }
			Recv.Reset();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);   // the listener is gone; dynamic delegates must skip it
			TArray<float> Audio;
			Script(Audio, 16000, 1);
			PushArrayPaced(A->GetStream(), Audio, 16000, 1, 8);
			Grace(6.0, Done, [this]()
			{
				if (Action.IsValid()) { USpeechPipelineLibrary::StopListening(Action.Get()); }
				AddInfo(TEXT("survived events broadcast to a destroyed listener"));
			});
		});
	});

	// =================================================================================================================
	Describe("Lifetime", [this]()
	{
		LatentIt("ActionStaysAliveAcrossGCWhileListeningAndIsReleasedAfterStopped", FTimespan::FromSeconds(120), [this](const FDoneDelegate& Done)
		{
			USpeechListenAsyncAction* A = Begin(Pipe.Get(), false);
			if (!A || !A->GetStream()) { AddError(TEXT("no stream after Activate")); Done.Execute(); return; }
			const TWeakObjectPtr<USpeechListenAsyncAction> WA(A);
			const TWeakObjectPtr<USpeechStream> WS(A->GetStream());
			const TWeakObjectPtr<USpeechToTextPipeline> WP(Pipe.Get());
			// Drop every strong reference the test holds: the action (self-rooted, no game instance) must keep stream and pipeline.
			Pipe.Reset();
			Pkg.Reset();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			TestTrue(TEXT("action survives GC while listening"), WA.IsValid());
			TestTrue(TEXT("stream survives GC while listening"), WS.IsValid());
			TestTrue(TEXT("pipeline survives GC while listening"), WP.IsValid());
			if (!WA.IsValid() || !WS.IsValid()) { Done.Execute(); return; }

			TArray<float> Audio;
			Script(Audio, 16000, 1);
			PushArrayPaced(WS.Get(), Audio, 16000, 1, 8);
			WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Final) >= 1; }, TEXT("OnFinal after GC"), 40.0, Done, [WA]()
			{
				USpeechPipelineLibrary::StopListening(WA.Get());
			});
			WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Stopped) >= 1; }, TEXT("OnStopped"), 40.0, Done, [this, WA, Done]()
			{
				Grace(0.5, Done, [this, WA]()
				{
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					TestFalse(TEXT("action released (unrooted, ready to destroy) after OnStopped"), WA.IsValid());
				});
			});
		});
	});

	// =================================================================================================================
	Describe("ListenWithMicrophone", [this]()
	{
		LatentIt("DefaultDeviceStartsOrFailsCleanlyWithRuntimeUnavailable", FTimespan::FromSeconds(60), [this](const FDoneDelegate& Done)
		{
			const bool bHasDevice = USpeechMicrophoneSource::GetCaptureDevices().Num() > 0;
			AddInfo(bHasDevice ? TEXT("capture device present: expecting a running listener") : TEXT("no capture device: expecting OnFailure RuntimeUnavailable"));
			USpeechListenAsyncAction* A = Begin(Pipe.Get(), true, -1);
			if (!A) { Done.Execute(); return; }
			if (!bHasDevice)
			{
				WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Failure) >= 1; }, TEXT("OnFailure without a microphone"), 10.0, Done, [this, Done]()
				{
					const TArray<FSpeechListenTestRecord> F = Recv->Of(ESpeechListenTestEvent::Failure);
					if (F.Num() >= 1) { TestTrue(TEXT("RuntimeUnavailable"), F[0].Result.Error == EPipelineError::RuntimeUnavailable); }
					Grace(0.3, Done, [this]() { TestEqual(TEXT("OnFailure exactly once"), Recv->Count(ESpeechListenTestEvent::Failure), 1); });
				});
				return;
			}
			Grace(0.5, Done, [this]()
			{
				if (Recv->Count(ESpeechListenTestEvent::Failure) > 0)
				{
					AddInfo(TEXT("device listed but could not be opened; the listener failed cleanly (skipped)"));
					return;
				}
				TestNotNull(TEXT("GetStream while listening on the microphone"), Action.IsValid() ? Action->GetStream() : nullptr);
				USpeechPipelineLibrary::StopListening(Action.Get());
			});
			WaitFor(this, [this]() { return Recv->Count(ESpeechListenTestEvent::Stopped) >= 1 || Recv->Count(ESpeechListenTestEvent::Failure) >= 1; },
				TEXT("OnStopped or OnFailure"), 20.0, Done, [this, Done]()
			{
				Grace(0.3, Done, [this]()
				{
					if (Recv->Count(ESpeechListenTestEvent::Failure) == 0) { TestEqual(TEXT("OnStopped exactly once"), Recv->Count(ESpeechListenTestEvent::Stopped), 1); }
				});
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
