// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9S.5b / contracts C76 + C34: the "Transcribe Speech" and "Transcribe Sound Wave" async Blueprint nodes.
// Reference results come from TranscribeBlocking on the golden audio (itself checked against the golden onnx.text), never from the nodes.
// Text comparisons are CASE-SENSITIVE (FString == / TestEqual(FString) ignore case in Unreal).

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "Engine/CancellableAsyncAction.h"
#include "HAL/PlatformTime.h"
#include "Sound/SoundWave.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/UnrealType.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "SpeechPipelineAsyncActions.h"
#include "SpeechToTextPipeline.h"
#include "SpeechTypes.h"
#include "SpeechTestData.h"
#include "SpeechAsyncActionTestReceiver.h"

using namespace UE::ModelPipelines::SpeechTests;

namespace SpeechAsyncSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(1500);
	constexpr double WaitSeconds = 600.0;
	constexpr double SettleSeconds = 0.4;
	static const TCHAR* const ClipId = TEXT("1089-134691-0025");

	struct FKind
	{
		const TCHAR* Name = TEXT("");
		bool bWave = false;     // false: "Transcribe Speech" (float samples); true: "Transcribe Sound Wave"
	};

	static bool SameText(const FString& A, const FString& B) { return A.Equals(B, ESearchCase::CaseSensitive); }

	// ---------------------------------------------------------------- waiting (never blocks the game thread)

	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Next)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechAsyncSpecWait"), 0.02f, [T, Cond, What, Done, Next, Start](float)
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
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechAsyncSpecSettle"), 0.02f, [Start, Secs, Next](float)
		{
			if ((FPlatformTime::Seconds() - Start) >= Secs)
			{
				Next();
				return false;
			}
			return true;
		});
	}

	static void WaitSettled(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Check)
	{
		WaitFor(T, Cond, What, Done, [Done, Check]()
		{
			Settle(SettleSeconds, [Done, Check]()
			{
				Check();
				Done.ExecuteIfBound();
			});
		});
	}

	// ---------------------------------------------------------------- fixture: golden audio + blocking reference on the shared CPU pipeline

	struct FRef
	{
		const FSttClip* Clip = nullptr;
		TArray<float> Audio;
		FPipelineResult R;
		FSpeechTranscript Tr;
	};

	static void WithReference(FAutomationTestBase* T, FDoneDelegate Done, TFunction<void(const TSharedRef<FRef>&, USpeechToTextPipeline*)> Next)
	{
		const FSttGoldens* G = nullptr;
		FString Err;
		if (!LoadSttGoldens(G, Err)) { T->AddError(Err); Done.ExecuteIfBound(); return; }
		const TSharedRef<FRef> Ref = MakeShared<FRef>();
		Ref->Clip = G->Find(ClipId);
		if (!Ref->Clip) { T->AddError(FString::Printf(TEXT("clip '%s' not in stt_goldens.json. Regenerate with Tools/make_stt_goldens.py"), ClipId)); Done.ExecuteIfBound(); return; }
		if (!LoadClipAudio(*G, *Ref->Clip, Ref->Audio, Err)) { T->AddError(Err); Done.ExecuteIfBound(); return; }
		USpeechToTextPipeline* P = GetSharedCpuSttPipeline(Err);
		if (!P) { T->AddError(Err); Done.ExecuteIfBound(); return; }
		const TSharedRef<TFuture<void>> Fut = MakeShared<TFuture<void>>(Async(EAsyncExecution::ThreadPool, [Ref, P]()
		{
			Ref->R = P->TranscribeBlocking(Ref->Audio, 16000, 1, Ref->Tr);
		}));
		WaitFor(T, [Fut]() { return Fut->IsReady(); }, TEXT("blocking reference call"), Done, [T, Ref, P, Done, Next]()
		{
			if (!Ref->R.bSuccess) { T->AddError(FString::Printf(TEXT("blocking reference failed: %s"), *Ref->R.Message)); Done.ExecuteIfBound(); return; }
			Next(Ref, P);
		});
	}

	// ---------------------------------------------------------------- actions

	static TStrongObjectPtr<USpeechAsyncActionTestReceiver> NewReceiver()
	{
		return TStrongObjectPtr<USpeechAsyncActionTestReceiver>(NewObject<USpeechAsyncActionTestReceiver>(GetTransientPackage()));
	}

	/** Holds what the action needs alive until the test ends (the action itself keeps its own references). */
	struct FInputs
	{
		TArray<float> Samples;
		TStrongObjectPtr<USoundWave> Wave;
	};

	static void MakeInputs(const FKind& K, const TArray<float>& Audio, FInputs& Out)
	{
		if (K.bWave)
		{
			TArray<int16> I16;
			FloatToInt16(Audio, I16);
			Out.Wave.Reset(MakeGeneratedSoundWave(I16, 16000, 1));
		}
		else
		{
			Out.Samples = Audio;
		}
	}

	/** Factory with a null world context (no game instance in the automation run), pins bound to R. */
	static UCancellableAsyncAction* NewAction(const FKind& K, USpeechToTextPipeline* P, const FInputs& In, USpeechAsyncActionTestReceiver* R)
	{
		if (K.bWave)
		{
			UTranscribeSoundWaveAsyncAction* A = UTranscribeSoundWaveAsyncAction::TranscribeSoundWave(nullptr, P, In.Wave.Get());
			if (A)
			{
				A->OnSuccess.AddDynamic(R, &USpeechAsyncActionTestReceiver::OnSuccess);
				A->OnFailure.AddDynamic(R, &USpeechAsyncActionTestReceiver::OnFailure);
			}
			return A;
		}
		UTranscribeSpeechAsyncAction* A = UTranscribeSpeechAsyncAction::TranscribeSpeech(nullptr, P, In.Samples, 16000, 1);
		if (A)
		{
			A->OnSuccess.AddDynamic(R, &USpeechAsyncActionTestReceiver::OnSuccess);
			A->OnFailure.AddDynamic(R, &USpeechAsyncActionTestReceiver::OnFailure);
		}
		return A;
	}

	static void Activate(USpeechAsyncActionTestReceiver* R, UCancellableAsyncAction* A)
	{
		R->bInsideActivate = true;
		A->Activate();
		R->bInsideActivate = false;
	}

	// ---------------------------------------------------------------- checks

	static bool CheckSingleDelivery(FAutomationTestBase& T, const USpeechAsyncActionTestReceiver& R, FName Pin, FName OtherPin, const FString& Label)
	{
		T.TestEqual(*FString::Printf(TEXT("%s: %s count"), *Label, *Pin.ToString()), R.CountPin(Pin), 1);
		T.TestEqual(*FString::Printf(TEXT("%s: %s count (the other pin)"), *Label, *OtherPin.ToString()), R.CountPin(OtherPin), 0);
		if (!T.TestEqual(*FString::Printf(TEXT("%s: total deliveries (exactly once)"), *Label), R.Total(), 1)) { return false; }
		T.TestTrue(*FString::Printf(TEXT("%s: delivered on the game thread"), *Label), R.Records[0].bGameThread);
		T.TestFalse(*FString::Printf(TEXT("%s: delivered inside Activate()"), *Label), R.Records[0].bInsideActivate);
		return true;
	}

	static void CheckSuccessRecord(FAutomationTestBase& T, const FSpeechAsyncActionTestRecord& Rec, const FRef& Ref, const FString& Label)
	{
		T.TestEqual(*(Label + TEXT(": pin")), Rec.Pin.ToString(), FString(TEXT("Success")));
		T.TestTrue(*FString::Printf(TEXT("%s: Result.bSuccess (%s)"), *Label, *Rec.Result.Message), Rec.Result.bSuccess);
		T.TestTrue(*(Label + TEXT(": Result.Error None")), Rec.Result.Error == EPipelineError::None);
		const FSpeechTranscript& Got = Rec.Transcript;
		T.TestTrue(*FString::Printf(TEXT("%s: Text equals golden onnx.text (case-sensitive). got '%s' golden '%s'"), *Label, *Got.Text, *Ref.Clip->OnnxText), SameText(Got.Text, Ref.Clip->OnnxText));
		T.TestTrue(*(Label + TEXT(": Text equals the C++ call's Text")), SameText(Got.Text, Ref.Tr.Text));
		bool bWords = Got.Words.Num() == Ref.Tr.Words.Num();
		for (int32 w = 0; bWords && w < Ref.Tr.Words.Num(); ++w)
		{
			bWords = SameText(Got.Words[w].Word, Ref.Tr.Words[w].Word) && Got.Words[w].StartSeconds == Ref.Tr.Words[w].StartSeconds && Got.Words[w].EndSeconds == Ref.Tr.Words[w].EndSeconds;
		}
		T.TestTrue(*(Label + TEXT(": words and times equal the C++ call's (exact)")), bWords);
		T.TestTrue(*FString::Printf(TEXT("%s: AudioSeconds %.5f within 1e-3 of the clip length"), *Label, Got.AudioSeconds), FMath::Abs((double)Got.AudioSeconds - (double)Ref.Clip->NumSamples / 16000.0) <= 1e-3);
	}

	static void CheckFailureRecord(FAutomationTestBase& T, const FSpeechAsyncActionTestRecord& Rec, const FString& Label)
	{
		T.TestEqual(*(Label + TEXT(": pin")), Rec.Pin.ToString(), FString(TEXT("Failure")));
		T.TestFalse(*(Label + TEXT(": Result.bSuccess")), Rec.Result.bSuccess);
		T.TestTrue(*(Label + TEXT(": transcript default-constructed on failure")), Rec.Transcript.Text.IsEmpty() && Rec.Transcript.Words.Num() == 0 && Rec.Transcript.AudioSeconds == 0.f);
	}

	// ---------------------------------------------------------------- scenarios

	static void ScenarioSuccess(FAutomationTestBase* T, FKind K, bool bSecondActivate, FDoneDelegate Done)
	{
		// One action on the real package: pin, transcript equal to the C++ path and the golden text. The node adds no logic of its own.
		WithReference(T, Done, [T, K, bSecondActivate, Done](const TSharedRef<FRef>& Ref, USpeechToTextPipeline* P)
		{
			const TSharedRef<FInputs> In = MakeShared<FInputs>();
			MakeInputs(K, Ref->Audio, *In);
			const TStrongObjectPtr<USpeechAsyncActionTestReceiver> R = NewReceiver();
			UCancellableAsyncAction* A = NewAction(K, P, *In, R.Get());
			if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
			const TStrongObjectPtr<UCancellableAsyncAction> Hold(A);
			T->TestEqual(TEXT("nothing fired before Activate"), R->Total(), 0);
			if (bSecondActivate)
			{
				T->AddExpectedMessagePlain(TEXT("Activate called again"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
			}
			Activate(R.Get(), A);
			T->TestEqual(TEXT("nothing fired inside Activate"), R->Total(), 0);
			T->TestTrue(TEXT("IsActive() true after Activate"), A->IsActive());
			if (bSecondActivate) { Activate(R.Get(), A); }
			WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("OnSuccess/OnFailure never fired"), Done, [T, K, R, Hold, Ref, In, P, bSecondActivate]()
			{
				if (CheckSingleDelivery(*T, *R, TEXT("Success"), TEXT("Failure"), K.Name)) { CheckSuccessRecord(*T, R->Records[0], *Ref, K.Name); }
				T->TestFalse(TEXT("IsActive() false after delivery"), Hold->IsActive());
				if (bSecondActivate) { T->TestEqual(TEXT("double Activate: no extra call pending"), P->GetNumPendingCalls(), 0); }
			});
		});
	}

	static void ScenarioNullPipeline(FAutomationTestBase* T, FKind K, bool bCancel, FDoneDelegate Done)
	{
		// Null pipeline: OnFailure with InvalidArgument "Pipeline is null", one Warning, delivered LATER (never inside Activate).
		// With bCancel the deferred failure is suppressed by Cancel().
		const TStrongObjectPtr<USpeechAsyncActionTestReceiver> R = NewReceiver();
		T->AddExpectedMessagePlain(TEXT("Pipeline is null"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
		TArray<int16> Silence;
		Silence.SetNumZeroed(1600);
		FInputs In;
		In.Samples.SetNumZeroed(1600);
		if (K.bWave) { In.Wave.Reset(MakeGeneratedSoundWave(Silence, 16000, 1)); }
		UCancellableAsyncAction* A = NewAction(K, nullptr, In, R.Get());
		if (!T->TestNotNull(TEXT("factory never fails, even with a null pipeline"), A)) { Done.ExecuteIfBound(); return; }
		const TStrongObjectPtr<UCancellableAsyncAction> Hold(A);
		Activate(R.Get(), A);
		T->TestEqual(TEXT("null pipeline: nothing fired inside Activate"), R->Total(), 0);
		if (bCancel)
		{
			A->Cancel();
			T->TestFalse(TEXT("IsActive() false right after Cancel"), A->IsActive());
			Settle(1.0, [T, R, Done]()
			{
				T->TestEqual(TEXT("cancelled null-pipeline action: no pin fired"), R->Total(), 0);
				Done.ExecuteIfBound();
			});
			return;
		}
		T->TestTrue(TEXT("null pipeline: IsActive() true until the failure is delivered"), A->IsActive());
		WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("null-pipeline failure never fired"), Done, [T, R, Hold]()
		{
			if (!CheckSingleDelivery(*T, *R, TEXT("Failure"), TEXT("Success"), TEXT("null pipeline"))) { return; }
			const FSpeechAsyncActionTestRecord& Rec = R->Records[0];
			CheckFailureRecord(*T, Rec, TEXT("null pipeline"));
			T->TestTrue(*FString::Printf(TEXT("null pipeline: Error InvalidArgument, got %d"), (int32)Rec.Result.Error), Rec.Result.Error == EPipelineError::InvalidArgument);
			T->TestEqual(TEXT("null pipeline: message"), Rec.Result.Message, FString(TEXT("Pipeline is null")));
			T->TestFalse(TEXT("null pipeline: IsActive() false after delivery"), Hold->IsActive());
		});
	}

	static void ScenarioPipelineError(FAutomationTestBase* T, FKind K, bool bNotReady, FDoneDelegate Done)
	{
		// The pipeline's own check is passed on unchanged: bNotReady = pipeline never initialised; otherwise the tiny pipeline with an
		// input the pipeline rejects (36 s of samples / a null Sound). A node that mapped errors itself would fail here.
		USpeechToTextPipeline* P = nullptr;
		FString Err;
		if (bNotReady) { P = NewObject<USpeechToTextPipeline>(GetTransientPackage()); }
		else { P = GetSharedTinySttPipeline(Err); if (!P) { T->AddError(Err); Done.ExecuteIfBound(); return; } }
		const TStrongObjectPtr<USpeechToTextPipeline> KeepP(P);
		FInputs In;
		TArray<int16> Silence;
		if (bNotReady || K.bWave)
		{
			Silence.SetNumZeroed(16000);
			if (K.bWave && bNotReady) { In.Wave.Reset(MakeGeneratedSoundWave(Silence, 16000, 1)); }
			if (!K.bWave) { In.Samples.SetNumZeroed(16000); }
		}
		else
		{
			In.Samples.SetNumZeroed(600 * 16000 + 16000);   // 601 s: over the C77a limit
		}
		const TStrongObjectPtr<USpeechAsyncActionTestReceiver> R = NewReceiver();
		UCancellableAsyncAction* A = NewAction(K, P, In, R.Get());   // wave kind + tiny pipeline: null Sound
		if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
		Activate(R.Get(), A);
		T->TestEqual(TEXT("nothing fired inside Activate"), R->Total(), 0);
		WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("failure never fired"), Done, [T, K, R, bNotReady, KeepP]()
		{
			if (!CheckSingleDelivery(*T, *R, TEXT("Failure"), TEXT("Success"), K.Name)) { return; }
			const FSpeechAsyncActionTestRecord& Rec = R->Records[0];
			CheckFailureRecord(*T, Rec, K.Name);
			T->TestTrue(*FString::Printf(TEXT("%s: Error InvalidArgument, got %d (%s)"), K.Name, (int32)Rec.Result.Error, *Rec.Result.Message), Rec.Result.Error == EPipelineError::InvalidArgument);
			T->TestFalse(TEXT("message not empty"), Rec.Result.Message.IsEmpty());
			if (K.bWave && !bNotReady)
			{
				T->TestTrue(*FString::Printf(TEXT("message names the null Sound; got '%s'"), *Rec.Result.Message), Rec.Result.Message.Contains(TEXT("Sound is null"), ESearchCase::CaseSensitive));
			}
		});
	}

	static void ScenarioCancel(FAutomationTestBase* T, FKind K, bool bAfterDelivery, FDoneDelegate Done)
	{
		// Before delivery: no pin ever fires, the call still completes, the action is released afterwards (a second, uncancelled
		// control action on the same pipeline proves the executor is not stuck). After delivery (and twice): a no-op.
		WithReference(T, Done, [T, K, bAfterDelivery, Done](const TSharedRef<FRef>& Ref, USpeechToTextPipeline* P)
		{
			const TSharedRef<FInputs> In = MakeShared<FInputs>();
			MakeInputs(K, Ref->Audio, *In);
			const TStrongObjectPtr<USpeechAsyncActionTestReceiver> R1 = NewReceiver();
			const TStrongObjectPtr<USpeechAsyncActionTestReceiver> R2 = NewReceiver();
			UCancellableAsyncAction* A = NewAction(K, P, *In, R1.Get());
			if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
			const TWeakObjectPtr<UCancellableAsyncAction> Weak(A);
			const TStrongObjectPtr<UCancellableAsyncAction> Hold(A);
			Activate(R1.Get(), A);
			if (bAfterDelivery)
			{
				WaitSettled(T, [R1]() { return R1->Total() > 0; }, TEXT("OnSuccess never fired"), Done, [T, K, R1, Hold, Done]()
				{
					CheckSingleDelivery(*T, *R1, TEXT("Success"), TEXT("Failure"), K.Name);
					Hold->Cancel();
					Hold->Cancel();
					T->TestFalse(TEXT("IsActive() false after Cancel following delivery"), Hold->IsActive());
					Settle(SettleSeconds, [T, R1, Done]()
					{
						T->TestEqual(TEXT("Cancel after delivery: still exactly one delivery in total"), R1->Total(), 1);
						Done.ExecuteIfBound();
					});
				});
				return;
			}
			T->TestTrue(TEXT("IsActive() true after Activate"), A->IsActive());
			A->Cancel();
			T->TestFalse(TEXT("IsActive() false after Cancel"), A->IsActive());
			A->Cancel();   // second Cancel: no-op
			T->TestFalse(TEXT("IsActive() still false after a second Cancel"), A->IsActive());
			UCancellableAsyncAction* Control = NewAction(K, P, *In, R2.Get());
			if (!T->TestNotNull(TEXT("control action"), Control)) { Done.ExecuteIfBound(); return; }
			Activate(R2.Get(), Control);
			WaitSettled(T, [R2]() { return R2->Total() > 0; }, TEXT("control action never delivered"), Done, [T, K, P, R1, R2, Ref, In, Hold]()
			{
				T->TestEqual(TEXT("cancelled action: no pin fired"), R1->Total(), 0);
				if (CheckSingleDelivery(*T, *R2, TEXT("Success"), TEXT("Failure"), TEXT("control"))) { CheckSuccessRecord(*T, R2->Records[0], *Ref, TEXT("control")); }
				T->TestTrue(TEXT("cancelled action: still no pin after the control finished"), R1->Total() == 0);
				(void)Hold;
			});
		});
	}

	static void ScenarioKeepAlive(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// No test reference to the action (weak only), GC right after Activate: the action roots itself, the pin still fires correctly.
		WithReference(T, Done, [T, K, Done](const TSharedRef<FRef>& Ref, USpeechToTextPipeline* P)
		{
			const TSharedRef<FInputs> In = MakeShared<FInputs>();
			MakeInputs(K, Ref->Audio, *In);
			const TStrongObjectPtr<USpeechAsyncActionTestReceiver> R = NewReceiver();
			TWeakObjectPtr<UCancellableAsyncAction> Weak;
			{
				UCancellableAsyncAction* A = NewAction(K, P, *In, R.Get());
				if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
				Weak = A;
				Activate(R.Get(), A);
			}
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			T->TestTrue(TEXT("action survived GC while in flight (rooted)"), Weak.IsValid());
			WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("pin never fired after GC"), Done, [T, K, R, Ref, In]()
			{
				if (CheckSingleDelivery(*T, *R, TEXT("Success"), TEXT("Failure"), K.Name)) { CheckSuccessRecord(*T, R->Records[0], *Ref, K.Name); }
			});
		});
	}

	static void ScenarioListenerDestroyed(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// The bound listener is destroyed and collected while the call is in flight: no crash; a control action still delivers.
		WithReference(T, Done, [T, K, Done](const TSharedRef<FRef>& Ref, USpeechToTextPipeline* P)
		{
			const TSharedRef<FInputs> In = MakeShared<FInputs>();
			MakeInputs(K, Ref->Audio, *In);
			TStrongObjectPtr<USpeechAsyncActionTestReceiver> R1 = NewReceiver();
			const TStrongObjectPtr<USpeechAsyncActionTestReceiver> R2 = NewReceiver();
			UCancellableAsyncAction* A = NewAction(K, P, *In, R1.Get());
			UCancellableAsyncAction* Control = NewAction(K, P, *In, R2.Get());
			if (!T->TestNotNull(TEXT("factory returned an action"), A) || !T->TestNotNull(TEXT("control action"), Control)) { Done.ExecuteIfBound(); return; }
			const TWeakObjectPtr<USpeechAsyncActionTestReceiver> WeakR1(R1.Get());
			Activate(R1.Get(), A);
			Activate(R2.Get(), Control);
			R1.Reset();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			T->TestFalse(TEXT("listener was collected"), WeakR1.IsValid());
			WaitSettled(T, [R2]() { return R2->Total() > 0; }, TEXT("control action never delivered"), Done, [T, K, R2, Ref, In]()
			{
				if (CheckSingleDelivery(*T, *R2, TEXT("Success"), TEXT("Failure"), TEXT("control"))) { CheckSuccessRecord(*T, R2->Records[0], *Ref, TEXT("control")); }
			});
		});
	}

	static void ScenarioEightConcurrent(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// 8 actions in flight on one pipeline, each with its own receiver and its own input copy: each receiver gets exactly one
		// success with the correct transcript (a shared result buffer or cross-delivery would hand one receiver two results or none).
		WithReference(T, Done, [T, K, Done](const TSharedRef<FRef>& Ref, USpeechToTextPipeline* P)
		{
			const TSharedRef<TArray<TSharedRef<FInputs>>> Ins = MakeShared<TArray<TSharedRef<FInputs>>>();
			const TSharedRef<TArray<TStrongObjectPtr<USpeechAsyncActionTestReceiver>>> Rs = MakeShared<TArray<TStrongObjectPtr<USpeechAsyncActionTestReceiver>>>();
			TArray<UCancellableAsyncAction*> Actions;
			for (int32 i = 0; i < 8; ++i)
			{
				const TSharedRef<FInputs> In = MakeShared<FInputs>();
				MakeInputs(K, Ref->Audio, *In);
				Ins->Add(In);
				Rs->Add(NewReceiver());
				UCancellableAsyncAction* A = NewAction(K, P, *In, (*Rs)[i].Get());
				if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
				Actions.Add(A);
			}
			for (int32 i = 0; i < 8; ++i) { Activate((*Rs)[i].Get(), Actions[i]); }
			WaitSettled(T, [Rs]()
			{
				for (const TStrongObjectPtr<USpeechAsyncActionTestReceiver>& R : *Rs) { if (R->Total() == 0) { return false; } }
				return true;
			}, TEXT("not all 8 actions delivered"), Done, [T, K, Rs, Ref, Ins]()
			{
				for (int32 i = 0; i < 8; ++i)
				{
					const FString Label = FString::Printf(TEXT("%s action %d"), K.Name, i);
					if (CheckSingleDelivery(*T, *(*Rs)[i], TEXT("Success"), TEXT("Failure"), Label)) { CheckSuccessRecord(*T, (*Rs)[i]->Records[0], *Ref, Label); }
				}
			});
		});
	}

	// ---------------------------------------------------------------- reflection

	static void CheckFactory(FAutomationTestBase& T, UClass* Cls, const TCHAR* FuncName, const TCHAR* DisplayName, TArrayView<const TCHAR* const> Params)
	{
		UFunction* Fn = Cls->FindFunctionByName(FuncName);
		if (!T.TestNotNull(*FString::Printf(TEXT("%s::%s exists as a UFUNCTION"), *Cls->GetName(), FuncName), Fn)) { return; }
		T.TestTrue(TEXT("factory is static"), Fn->HasAnyFunctionFlags(FUNC_Static));
		T.TestTrue(TEXT("factory is BlueprintCallable"), Fn->HasAnyFunctionFlags(FUNC_BlueprintCallable));
#if WITH_METADATA
		T.TestEqual(TEXT("BlueprintInternalUseOnly value"), Fn->GetMetaData(TEXT("BlueprintInternalUseOnly")), FString(TEXT("true")));
		T.TestEqual(TEXT("meta WorldContext"), Fn->GetMetaData(TEXT("WorldContext")), FString(TEXT("WorldContextObject")));
		T.TestEqual(TEXT("meta DisplayName"), Fn->GetMetaData(TEXT("DisplayName")), FString(DisplayName));
		T.TestEqual(TEXT("meta Category"), Fn->GetMetaData(TEXT("Category")), FString(TEXT("Model Pipelines|Speech")));
#endif
		T.TestNotNull(TEXT("parameter WorldContextObject"), Fn->FindPropertyByName(TEXT("WorldContextObject")));
		for (const TCHAR* Name : Params)
		{
			T.TestNotNull(*FString::Printf(TEXT("parameter %s"), Name), Fn->FindPropertyByName(Name));
		}
		const FObjectProperty* Ret = CastField<FObjectProperty>(Fn->GetReturnProperty());
		if (T.TestNotNull(TEXT("factory has an object return value"), Ret)) { T.TestTrue(TEXT("factory returns the action class"), Ret->PropertyClass == Cls); }
	}

	static void CheckPins(FAutomationTestBase& T, UClass* Cls)
	{
		TArray<const FMulticastDelegateProperty*> Assignable;
		for (TFieldIterator<FMulticastDelegateProperty> It(Cls); It; ++It)
		{
			if (It->HasAnyPropertyFlags(CPF_BlueprintAssignable)) { Assignable.Add(*It); }
		}
		if (!T.TestEqual(*FString::Printf(TEXT("%s: number of BlueprintAssignable delegates"), *Cls->GetName()), Assignable.Num(), 2)) { return; }
		const FMulticastDelegateProperty* Ok = nullptr;
		const FMulticastDelegateProperty* Fail = nullptr;
		for (const FMulticastDelegateProperty* P : Assignable)
		{
			if (P->GetFName() == TEXT("OnSuccess")) { Ok = P; }
			if (P->GetFName() == TEXT("OnFailure")) { Fail = P; }
		}
		if (!T.TestNotNull(TEXT("property OnSuccess"), Ok) || !T.TestNotNull(TEXT("property OnFailure"), Fail)) { return; }
		T.TestTrue(TEXT("OnSuccess and OnFailure share one delegate signature"), Ok->SignatureFunction == Fail->SignatureFunction);
		if (Ok->SignatureFunction)
		{
			T.TestNotNull(TEXT("signature parameter Result"), Ok->SignatureFunction->FindPropertyByName(TEXT("Result")));
			T.TestNotNull(TEXT("signature parameter Transcript"), Ok->SignatureFunction->FindPropertyByName(TEXT("Transcript")));
			int32 NumParams = 0;
			for (TFieldIterator<FProperty> It(Ok->SignatureFunction); It; ++It) { ++NumParams; }
			T.TestEqual(TEXT("signature has exactly two parameters"), NumParams, 2);
		}
	}
}

BEGIN_DEFINE_SPEC(FSpeechPipelineAsyncActionSpec, "Pipelines.Speech.AsyncAction",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
END_DEFINE_SPEC(FSpeechPipelineAsyncActionSpec)

void FSpeechPipelineAsyncActionSpec::Define()
{
	using namespace SpeechAsyncSpecPrivate;

	TArray<FKind> Kinds;
	{
		FKind S; S.Name = TEXT("TranscribeSpeech"); S.bWave = false; Kinds.Add(S);
		FKind W; W.Name = TEXT("TranscribeSoundWave"); W.bWave = true; Kinds.Add(W);
	}

	for (const FKind& K : Kinds)
	{
		Describe(K.Name, [this, K]()
		{
			LatentIt("SuccessEqualsCppCallAndGoldenText", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioSuccess(this, K, false, Done); });
			LatentIt("SecondActivateWarnsAndDeliversOnce", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioSuccess(this, K, true, Done); });
			LatentIt("NullPipelineFailsOnceWithInvalidArgumentLaterThanActivate", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioNullPipeline(this, K, false, Done); });
			LatentIt("CancelledNullPipelineActionFiresNothing", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioNullPipeline(this, K, true, Done); });
			LatentIt("NotReadyPipelinePassesPipelineErrorThrough", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioPipelineError(this, K, true, Done); });
			LatentIt("RejectedInputPassesPipelineErrorThrough", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioPipelineError(this, K, false, Done); });
			LatentIt("CancelRightAfterActivateFiresNoPin", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioCancel(this, K, false, Done); });
			LatentIt("CancelAfterDeliveryAndDoubleCancelAreNoOps", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioCancel(this, K, true, Done); });
			LatentIt("ActionSurvivesGCWithNoReferencesHeldByTest", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioKeepAlive(this, K, Done); });
			LatentIt("ListenerDestroyedWhileInFlightIsNoCrash", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioListenerDestroyed(this, K, Done); });
			LatentIt("EightActionsOnOnePipelineEachGetsItsOwnResult", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioEightConcurrent(this, K, Done); });
		});
	}

	Describe("Reflection", [this]()
	{
		It("TranscribeSpeechFactoryMetadata", [this]()
		{
			const TCHAR* const Params[] = { TEXT("Pipeline"), TEXT("Samples"), TEXT("SampleRate"), TEXT("NumChannels") };
			CheckFactory(*this, UTranscribeSpeechAsyncAction::StaticClass(), TEXT("TranscribeSpeech"), TEXT("Transcribe Speech"), Params);
		});

		It("TranscribeSoundWaveFactoryMetadata", [this]()
		{
			const TCHAR* const Params[] = { TEXT("Pipeline"), TEXT("Sound") };
			CheckFactory(*this, UTranscribeSoundWaveAsyncAction::StaticClass(), TEXT("TranscribeSoundWave"), TEXT("Transcribe Sound Wave"), Params);
		});

		It("BothActionsHaveExactlyTwoAssignablePinsWithOneSignature", [this]()
		{
			CheckPins(*this, UTranscribeSpeechAsyncAction::StaticClass());
			CheckPins(*this, UTranscribeSoundWaveAsyncAction::StaticClass());
		});

		It("ActionsDeriveFromCancellableAsyncAction", [this]()
		{
			TestTrue(TEXT("UTranscribeSpeechAsyncAction is a UCancellableAsyncAction"), UTranscribeSpeechAsyncAction::StaticClass()->IsChildOf(UCancellableAsyncAction::StaticClass()));
			TestTrue(TEXT("UTranscribeSoundWaveAsyncAction is a UCancellableAsyncAction"), UTranscribeSoundWaveAsyncAction::StaticClass()->IsChildOf(UCancellableAsyncAction::StaticClass()));
		});

		It("FactoryDoesNoWorkAndNeverFailsWithNullInputs", [this]()
		{
			UTranscribeSpeechAsyncAction* S = UTranscribeSpeechAsyncAction::TranscribeSpeech(nullptr, nullptr, TArray<float>(), 16000, 1);
			UTranscribeSoundWaveAsyncAction* W = UTranscribeSoundWaveAsyncAction::TranscribeSoundWave(nullptr, nullptr, nullptr);
			TestNotNull(TEXT("TranscribeSpeech(null, null, ...)"), S);
			TestNotNull(TEXT("TranscribeSoundWave(null, null, null)"), W);
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);   // never activated, so nothing roots them
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
