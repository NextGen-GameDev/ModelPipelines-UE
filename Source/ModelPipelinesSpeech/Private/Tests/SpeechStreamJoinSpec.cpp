// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9R.7t: USpeechStream "join and correct" (contract C83): utterances separated by a short pause form a turn, the later final
// is the transcription of the CONTIGUOUS turn audio and has bReplacesPrevious. Written from docs/briefs/WP9R.7t.md only.
// Layers: (1) fake engines (SpeechFakeEngine.h), deterministic synthetic tone bursts; (2) real stage-A audio, WER bars.
// Text comparisons are CASE-SENSITIVE (FString == ignores case in Unreal). NaN through std::numeric_limits.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/ScopeLock.h"
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

#include <limits>

using namespace UE::ModelPipelines::SpeechTests;

namespace SpeechStreamJoinSpecPrivate
{
	constexpr int32 Frame = 320;           // 20 ms at 16 kHz
	constexpr int32 TolSamples = 960;      // +-3 frames
	constexpr double Wait = 60.0;
	const FTimespan LatentLimit = FTimespan::FromSeconds(400);
	const FTimespan RealLimit = FTimespan::FromSeconds(1800);
	const TCHAR* const ConvHint = TEXT("Generate the conversational goldens with the WP9R.7 script (Tools/make_stt_conversational_goldens.py) into <dev>/goldens/stt/conversational/.");

	/** One burst: Tone frames of tone, then Gap frames of digital silence (20 ms frames). */
	struct FSeg { int32 Tone = 0; int32 Gap = 0; };

	/** 0.5 s lead-in silence, then the segments. */
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

	// ---- waiting (the game thread is never blocked); Next must call Done itself ----
	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, const FString& What, const FDoneDelegate& Done, TFunction<void()> Next, double Timeout)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechStreamJoinWait"), 0.01f, [T, Cond, What, Done, Next, Start, Timeout](float) -> bool
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
		FTSTicker::GetCoreTicker().AddTicker(TEXT("SpeechStreamJoinSettle"), 0.01f, [Seconds, Next, Start](float) -> bool
		{
			if (FPlatformTime::Seconds() - Start < Seconds) { return true; }
			Next();
			return false;
		});
	}

	/** Pushes Audio from the game thread, PerTick 20 ms chunks per 50 ms tick (4x real time at PerTick = 8), then OnPushed. */
	static void PushPaced(USpeechStream* Stream, const TArray<float>& Audio, int32 PerTick, TFunction<void()> OnPushed)
	{
		const TSharedRef<TArray<float>> A = MakeShared<TArray<float>>(Audio);
		const TSharedRef<int32> Pos = MakeShared<int32>(0);
		const TWeakObjectPtr<USpeechStream> Weak(Stream);
		FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Weak, A, Pos, PerTick, OnPushed](float) -> bool
		{
			USpeechStream* S = Weak.Get();
			for (int32 k = 0; k < PerTick && *Pos < A->Num(); ++k)
			{
				const int32 N = FMath::Min(Frame, A->Num() - *Pos);
				if (S) { S->PushAudio(TConstArrayView<float>(A->GetData() + *Pos, N), 16000, 1); }
				*Pos += N;
			}
			if (*Pos >= A->Num()) { OnPushed(); return false; }
			return true;
		}), 0.05f);
	}

	/** Event grammar for NumUtterances utterances: Started < Partial* < Ended < Final per utterance, Stopped exactly once and last. */
	static void CheckGrammar(FAutomationTestBase* T, const USpeechStreamTestReceiver* R, int32 NumUtterances, bool bPartialsAllowed)
	{
		const TArray<FSpeechStreamTestRecord> St = R->Of(ESpeechStreamTestEvent::Started);
		const TArray<FSpeechStreamTestRecord> En = R->Of(ESpeechStreamTestEvent::Ended);
		const TArray<FSpeechStreamTestRecord> Fi = R->Of(ESpeechStreamTestEvent::Final);
		T->TestEqual(TEXT("OnSpeechStarted count"), St.Num(), NumUtterances);
		T->TestEqual(TEXT("OnSpeechEnded count"), En.Num(), NumUtterances);
		T->TestEqual(TEXT("OnFinal count (one per utterance, empty ones included)"), Fi.Num(), NumUtterances);
		T->TestEqual(TEXT("OnStopped exactly once"), R->Count(ESpeechStreamTestEvent::Stopped), 1);
		T->TestTrue(TEXT("every event on the game thread"), R->AllOnGameThread());
		if (R->Records.Num() > 0) { T->TestTrue(TEXT("OnStopped is the last event"), R->Records.Last().Type == ESpeechStreamTestEvent::Stopped); }
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

	// ---- WER (case- and punctuation-insensitive, case-sensitive compare after lowering) ----
	static void Words(const FString& Text, TArray<FString>& Out)
	{
		TArray<FString> Raw;
		Text.ParseIntoArrayWS(Raw);
		for (const FString& W : Raw)
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

	/** WER in percent points. */
	static double WerPoints(const FString& Reference, const FString& Hypothesis)
	{
		TArray<FString> Ref, Hyp;
		Words(Reference, Ref);
		Words(Hypothesis, Hyp);
		return Ref.Num() > 0 ? 100.0 * (double)EditDistance(Ref, Hyp) / (double)Ref.Num() : 100.0;
	}

	// ---- real audio: conversational goldens ----
	struct FAudioItem
	{
		FString Id;
		FString Text;
		TArray<float> Audio;
	};

	static bool LoadItem(const FString& Id, FAudioItem& Out, FString& Err)
	{
		const FString Dir = FPaths::Combine(GetDevDir(), TEXT("goldens/stt/conversational"));
		const FString JsonPath = FPaths::Combine(Dir, TEXT("conversational.json"));
		FString Json;
		if (!FFileHelper::LoadFileToString(Json, *JsonPath)) { Err = FString::Printf(TEXT("missing %s. %s"), *JsonPath, ConvHint); return false; }
		TSharedPtr<FJsonObject> Root;
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Root) || !Root.IsValid()) { Err = FString::Printf(TEXT("%s is not valid JSON. %s"), *JsonPath, ConvHint); return false; }
		const TArray<TSharedPtr<FJsonValue>>* Items = nullptr;
		if (!Root->TryGetArrayField(TEXT("items"), Items)) { Err = FString::Printf(TEXT("%s has no 'items'. %s"), *JsonPath, ConvHint); return false; }
		for (const TSharedPtr<FJsonValue>& V : *Items)
		{
			const TSharedPtr<FJsonObject>* O = nullptr;
			if (!V.IsValid() || !V->TryGetObject(O)) { continue; }
			if (!(*O)->GetStringField(TEXT("id")).Equals(Id, ESearchCase::CaseSensitive)) { continue; }
			Out.Id = Id;
			Out.Text = (*O)->GetStringField(TEXT("text"));
			const FString File = FPaths::Combine(Dir, (*O)->GetStringField(TEXT("file")));
			const int32 Num = (int32)(*O)->GetNumberField(TEXT("num_samples"));
			TArray<uint8> Bytes;
			if (!FFileHelper::LoadFileToArray(Bytes, *File) || Bytes.Num() != Num * 4)
			{
				Err = FString::Printf(TEXT("cannot read %s (%d bytes expected, got %d). %s"), *File, Num * 4, Bytes.Num(), ConvHint);
				return false;
			}
			Out.Audio.SetNumUninitialized(Num);
			FMemory::Memcpy(Out.Audio.GetData(), Bytes.GetData(), (SIZE_T)Num * sizeof(float));
			return true;
		}
		Err = FString::Printf(TEXT("item '%s' not in %s. %s"), *Id, *JsonPath, ConvHint);
		return false;
	}

	/** A stream fed from a ticker at 4x real time in 320-sample pieces, followed by 1.5 s of silence and Stop(). */
	struct FStreamJob
	{
		TStrongObjectPtr<USpeechStream> Stream;
		TStrongObjectPtr<USpeechStreamTestReceiver> Recv;
		TArray<float> Audio;
		int32 Total = 0;
		int32 Pos = 0;
		double Start = 0.0;
		bool bStopCalled = false;
		FTSTicker::FDelegateHandle Pusher;
	};

	static TSharedPtr<FStreamJob> StartJob(USpeechToTextPipeline* Pipe, const FSpeechStreamSettings& Settings, const TArray<float>& Audio, FString& Err)
	{
		FPipelineResult R;
		USpeechStream* St = Pipe->StartStream(Settings, R);
		if (!St) { Err = FString::Printf(TEXT("StartStream failed: error %d '%s'"), (int32)R.Error, *R.Message); return nullptr; }
		const TSharedRef<FStreamJob> Job = MakeShared<FStreamJob>();
		Job->Stream.Reset(St);
		Job->Recv.Reset(NewObject<USpeechStreamTestReceiver>(GetTransientPackage()));
		USpeechStreamTestReceiver* Rc = Job->Recv.Get();
		St->OnSpeechStarted.AddDynamic(Rc, &USpeechStreamTestReceiver::HandleStarted);
		St->OnPartial.AddDynamic(Rc, &USpeechStreamTestReceiver::HandlePartial);
		St->OnFinal.AddDynamic(Rc, &USpeechStreamTestReceiver::HandleFinal);
		St->OnSpeechEnded.AddDynamic(Rc, &USpeechStreamTestReceiver::HandleEnded);
		St->OnStopped.AddDynamic(Rc, &USpeechStreamTestReceiver::HandleStopped);
		Job->Audio = Audio;
		Job->Total = ((Audio.Num() + 24000 + Frame - 1) / Frame) * Frame;
		Job->Start = FPlatformTime::Seconds();
		const TWeakPtr<FStreamJob> Weak = Job;
		Job->Pusher = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Weak](float) -> bool
		{
			const TSharedPtr<FStreamJob> J = Weak.Pin();
			if (!J.IsValid()) { return false; }
			USpeechStream* S = J->Stream.Get();
			const int32 Target = (int32)FMath::Min<double>((double)J->Total, (FPlatformTime::Seconds() - J->Start) * 4.0 * 16000.0);
			TArray<float> Piece;
			Piece.SetNumZeroed(Frame);
			for (int32 Pieces = 0; Pieces < 25 && J->Pos + Frame <= J->Total && J->Pos + Frame <= Target; ++Pieces)
			{
				for (int32 i = 0; i < Frame; ++i) { Piece[i] = (J->Pos + i < J->Audio.Num()) ? J->Audio[J->Pos + i] : 0.f; }
				if (IsValid(S)) { S->PushAudio(Piece, 16000, 1); }
				J->Pos += Frame;
			}
			if (J->Pos + Frame > J->Total)
			{
				if (!J->bStopCalled && IsValid(S)) { J->bStopCalled = true; S->Stop(); }
				return false;
			}
			return true;
		}), 0.0f);
		return Job;
	}

	/** Lines as a UI applies the finals: a bReplacesPrevious final replaces the last line. */
	static void ApplyFinals(const USpeechStreamTestReceiver* R, TArray<FString>& Lines, int32& Replacements)
	{
		Lines.Reset();
		Replacements = 0;
		for (const FSpeechStreamTestRecord& F : R->Of(ESpeechStreamTestEvent::Final))
		{
			if (F.Transcript.Text.TrimStartAndEnd().IsEmpty()) { continue; }   // C83 am.1 (5): empty finals carry no text
			if (F.Transcript.bReplacesPrevious && Lines.Num() > 0) { Lines.Last() = F.Transcript.Text; ++Replacements; }
			else { Lines.Add(F.Transcript.Text); }
		}
	}

	static FString JoinLines(const TArray<FString>& Lines)
	{
		FString Out;
		for (const FString& L : Lines) { Out += L; Out += TEXT(" "); }
		return Out;
	}

	struct FOfflineResult
	{
		bool bDone = false;
		FPipelineResult Result;
		FString Text;
	};
}

using namespace SpeechStreamJoinSpecPrivate;

BEGIN_DEFINE_SPEC(FSpeechStreamJoinSpec, "Pipelines.Speech.Stream.Join",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	TSharedPtr<FFakeEngineState, ESPMode::ThreadSafe> State;
	TSharedPtr<FFakePhononBackend> Backend;
	bool bRegistered = false;
	TStrongObjectPtr<UPipelineModelPackage> Pkg;
	TStrongObjectPtr<USpeechToTextPipeline> Pipe;
	TStrongObjectPtr<USpeechStream> Stream;
	TStrongObjectPtr<USpeechStreamTestReceiver> Recv;
	TSharedPtr<FStreamJob> Job;

	/** Fake engine on a fake Phonon package (always streaming: a package with an engine always uses the engine stream; the NNE
	    Transcribe-per-final path is covered by RealAudio). Lead 2026-10-18: the "NneFake" cases check the engine stream's flushes
	    (FlushPendingSamples = audio per final) instead of Transcribe calls; final k with audio has text "utt <k>". */
	void SetupFake(bool /*bUnused*/)
	{
		State = MakeShared<FFakeEngineState, ESPMode::ThreadSafe>();
		State->bStreaming = true;
		if (false)
		{
			// The text shows the length of the audio the engine got: "n<samples>".
			State->Script = [](TConstArrayView<float> Mono, int32 Rate, FSpeechTranscript& Out) -> FPipelineResult
			{
				Out.Text = FString::Printf(TEXT("n%d"), Mono.Num());
				Out.InferenceMs = 1.f;
				return FPipelineResult::Ok();
			};
		}
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

	/** Opens a stream, pushes Audio from the game thread, Stop(), waits for OnStopped, 0.4 s grace, Check(), Done. */
	void PushAndCheck(const FSpeechStreamSettings& S, const TArray<float>& Audio, int32 PerTick, const FDoneDelegate& Done, TFunction<void()> Check)
	{
		USpeechStream* St = Open(S);
		if (!St) { Done.ExecuteIfBound(); return; }
		PushPaced(St, Audio, PerTick, [this, Done, Check]()
		{
			if (Stream.IsValid()) { Stream->Stop(); }
			WaitFor(this, [this]() { return Recv.IsValid() && Recv->Count(ESpeechStreamTestEvent::Stopped) >= 1; }, TEXT("OnStopped"), Done, [this, Done, Check]()
			{
				Settle(0.4, [this, Done, Check]()
				{
					Check();
					Done.ExecuteIfBound();
				});
			}, Wait);
		});
	}

	/** The Transcribe calls recorded by the NNE-like fake: count, and each one's sample count within Tol of Expected[i]. */
	void CheckCalls(const TCHAR* What, const TArray<int32>& Expected, int32 Tol)
	{
		FScopeLock L(&State->Lock);
		if (!TestEqual(*FString::Printf(TEXT("%s: number of engine flushes with audio"), What), State->FlushPendingSamples.Num(), Expected.Num())) { return; }
		for (int32 i = 0; i < Expected.Num(); ++i)
		{
			const int32 Got = (int32)State->FlushPendingSamples[i];
			TestTrue(*FString::Printf(TEXT("%s: call %d got %d samples, expected %d +-%d (off by %d)"), What, i, Got, Expected[i], Tol, Got - Expected[i]), FMath::Abs(Got - Expected[i]) <= Tol);
		}
	}

	/** Finals: ReplFlags[i] is the expected bReplacesPrevious; CallOf[i] is the Transcribe call that produced it (-1 = empty text, no call). */
	void CheckFinals(const TCHAR* What, const TArray<bool>& ReplFlags, const TArray<int32>& CallOf)
	{
		const TArray<FSpeechStreamTestRecord> Fi = Recv->Of(ESpeechStreamTestEvent::Final);
		if (!TestEqual(*FString::Printf(TEXT("%s: number of finals"), What), Fi.Num(), ReplFlags.Num())) { return; }
		FScopeLock L(&State->Lock);
		for (int32 i = 0; i < Fi.Num(); ++i)
		{
			TestTrue(*FString::Printf(TEXT("%s: final %d bReplacesPrevious is %d"), What, i, (int32)ReplFlags[i]), Fi[i].Transcript.bReplacesPrevious == ReplFlags[i]);
			if (CallOf[i] < 0)
			{
				TestTrue(*FString::Printf(TEXT("%s: final %d has empty text, got '%s'"), What, i, *Fi[i].Transcript.Text), Fi[i].Transcript.Text.IsEmpty());
			}
			else if (State->FlushPendingSamples.IsValidIndex(CallOf[i]))
			{
				const FString Want = FString::Printf(TEXT("utt %d"), CallOf[i]);
				TestTrue(*FString::Printf(TEXT("%s: final %d text '%s' is the transcription of call %d ('%s')"), What, i, *Fi[i].Transcript.Text, CallOf[i], *Want),
					Fi[i].Transcript.Text.Equals(Want, ESearchCase::CaseSensitive));
			}
		}
	}

	/** Two 1 s bursts with GapFrames of silence between them (and 1.5 s after), default settings but MergeWindow. */
	void RunTwoBursts(float MergeWindow, int32 GapFrames, const FDoneDelegate& Done, TFunction<void()> Check)
	{
		if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
		FSpeechStreamSettings S; S.bPartials = false; S.MergeWindowSeconds = MergeWindow;
		TArray<float> A; BuildAudio(A, { { 50, GapFrames }, { 50, 75 } });
		PushAndCheck(S, A, 8, Done, Check);
	}

	/** RealAudio: offline WER vs streaming WER on one golden item. */
	void RunItem(const FString& Id, double AllowedPoints, int32 MinReplacements, bool bAlsoNoMerge, const FDoneDelegate& Done)
	{
		const TSharedRef<FAudioItem> Item = MakeShared<FAudioItem>();
		FString Err;
		if (!LoadItem(Id, *Item, Err)) { AddError(Err); Done.ExecuteIfBound(); return; }
		USpeechToTextPipeline* P = GetSharedCpuSttPipeline(Err);
		if (!P) { AddError(Err); Done.ExecuteIfBound(); return; }
		Pipe.Reset(P);
		const TSharedRef<FOfflineResult> Off = MakeShared<FOfflineResult>();
		Pipe->Transcribe(Item->Audio, 16000, 1, [Off](const FPipelineResult& R, const FSpeechTranscript& T) { Off->Result = R; Off->Text = T.Text; Off->bDone = true; });
		WaitFor(this, [Off]() { return Off->bDone; }, FString::Printf(TEXT("offline transcription of %s"), *Id), Done, [this, Id, Item, Off, AllowedPoints, MinReplacements, bAlsoNoMerge, Done]()
		{
			if (!Off->Result.bSuccess) { AddError(FString::Printf(TEXT("%s: offline Transcribe failed: %s"), *Id, *Off->Result.Message)); Done.ExecuteIfBound(); return; }
			const double OfflineWer = WerPoints(Item->Text, Off->Text);
			FString StartErr;
			Job = StartJob(Pipe.Get(), FSpeechStreamSettings(), Item->Audio, StartErr);
			if (!Job.IsValid()) { AddError(StartErr); Done.ExecuteIfBound(); return; }
			const TSharedPtr<FStreamJob> J = Job;
			WaitFor(this, [J]() { return J->Recv->Count(ESpeechStreamTestEvent::Stopped) >= 1; }, FString::Printf(TEXT("%s streaming run to stop"), *Id), Done,
				[this, Id, Item, J, OfflineWer, AllowedPoints, MinReplacements, bAlsoNoMerge, Done]()
				{
					TArray<FString> Lines; int32 Repl = 0;
					ApplyFinals(J->Recv.Get(), Lines, Repl);
					const double StreamWer = WerPoints(Item->Text, JoinLines(Lines));
					AddInfo(FString::Printf(TEXT("WP9R.7 %s: offline WER %.2f %%, streaming WER %.2f %% (%d finals, %d replacements, %d lines)"),
						*Id, OfflineWer, StreamWer, J->Recv->Count(ESpeechStreamTestEvent::Final), Repl, Lines.Num()));
					TestTrue(*FString::Printf(TEXT("%s: streaming WER %.2f %% <= offline %.2f %% + %.1f points"), *Id, StreamWer, OfflineWer, AllowedPoints), StreamWer <= OfflineWer + AllowedPoints);
					TestTrue(*FString::Printf(TEXT("%s: %d replacement finals >= %d"), *Id, Repl, MinReplacements), Repl >= MinReplacements);
					TestTrue(TEXT("every event on the game thread"), J->Recv->AllOnGameThread());
					if (!bAlsoNoMerge) { Done.ExecuteIfBound(); return; }
					FSpeechStreamSettings S0; S0.MergeWindowSeconds = 0.f;
					FString Err2;
					Job = StartJob(Pipe.Get(), S0, Item->Audio, Err2);
					if (!Job.IsValid()) { AddError(Err2); Done.ExecuteIfBound(); return; }
					const TSharedPtr<FStreamJob> J0 = Job;
					WaitFor(this, [J0]() { return J0->Recv->Count(ESpeechStreamTestEvent::Stopped) >= 1; }, FString::Printf(TEXT("%s no-merge run to stop"), *Id), Done,
						[this, Id, Item, J0, StreamWer, Done]()
						{
							TArray<FString> L0; int32 R0 = 0;
							ApplyFinals(J0->Recv.Get(), L0, R0);
							const double Wer0 = WerPoints(Item->Text, JoinLines(L0));
							AddInfo(FString::Printf(TEXT("WP9R.7 %s with MergeWindowSeconds 0: WER %.2f %% (%d finals, %d replacements); with the default window %.2f %%. Gain %.2f points"),
								*Id, Wer0, J0->Recv->Count(ESpeechStreamTestEvent::Final), R0, StreamWer, Wer0 - StreamWer));
							TestEqual(TEXT("MergeWindowSeconds 0 never produces a replacement"), R0, 0);
							Done.ExecuteIfBound();
						}, 900.0);
				}, 900.0);
		}, 900.0);
	}
END_DEFINE_SPEC(FSpeechStreamJoinSpec)

void FSpeechStreamJoinSpec::Define()
{
	AfterEach([this]()
	{
		Job.Reset();
	});

	// =================================================================================================================
	Describe("Defaults", [this]()
	{
		It("MergeWindowAndMinSpeechHaveTheContractDefaults", [this]()
		{
			// Case 1. A struct without the new members does not compile; wrong defaults (0 would switch the feature off) fail here.
			const FSpeechStreamSettings S;
			TestEqual(TEXT("MergeWindowSeconds default (C83 am.1)"), S.MergeWindowSeconds, 1.0f, 1e-6f);
			TestEqual(TEXT("MinSpeechSeconds default (C83 am.2)"), S.MinSpeechSeconds, 0.1f, 1e-6f);
			TestEqual(TEXT("EndOfSpeechHangoverSeconds default unchanged"), S.EndOfSpeechHangoverSeconds, 0.2f, 1e-6f);
			const FSpeechTranscript T;
			TestFalse(TEXT("bReplacesPrevious defaults to false"), T.bReplacesPrevious);
		});
	});

	// =================================================================================================================
	// NNE-like fake: one Transcribe per final, text "n<samples>"
	// =================================================================================================================
	Describe("NneFake", [this]()
	{
		BeforeEach([this]() { SetupFake(false); });
		AfterEach([this]() { TeardownFake(); });

		LatentIt("ShortGapJoinsAndTheSecondCallGetsTheContiguousTurnAudio", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 2 (+ case 9 grammar). 1.0 s burst, 0.5 s gap, 1.0 s burst. Gap from the first SpeechEnded point (end + 0.2 hangover) to
			// the second start is 0.3 s < 0.8 s window: join. The second Transcribe must get pre-roll 0.3 + 1.0 + 0.5 + 1.0 + 0.2 = 3.0 s
			// = 48000 samples. An implementation that transcribed only the second utterance would send ~24000; one that dropped the
			// pre-roll or the gap audio would be off by thousands. Final 2 carries bReplacesPrevious.
			RunTwoBursts(0.8f, 25, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 2, false);
				CheckCalls(TEXT("gap 0.5 s"), { 24000, 48000 }, TolSamples);
				CheckFinals(TEXT("gap 0.5 s"), { false, true }, { 0, 1 });
			});
		});

		LatentIt("LongGapDoesNotJoin", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 3. Gap 1.2 s of silence => 1.0 s after the SpeechEnded point, > 0.8 s window: two independent finals, the second call only
			// pre-roll 0.3 + 1.0 + hangover 0.2 = 24000 samples. A join that ignored the window would send ~58000.
			RunTwoBursts(0.8f, 60, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 2, false);
				CheckCalls(TEXT("gap 1.2 s"), { 24000, 24000 }, TolSamples);
				CheckFinals(TEXT("gap 1.2 s"), { false, false }, { 0, 1 });
			});
		});

		LatentIt("ChainOfThreeBurstsJoinsIntoOneGrowingTurn", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 4 (+ case 9). Bursts 1.0 / gap 0.4 / 0.8 / gap 0.5 / 0.6. Each next start is 0.2 s and 0.3 s after the previous SpeechEnded
			// point: all join. Calls: 24000, then 0.3+1.0+0.4+0.8+0.2 = 2.7 s = 43200, then 0.3+1.0+0.4+0.8+0.5+0.6+0.2 = 3.8 s = 60800.
			// Joining only to the previous utterance (instead of the turn start) would make the third call 0.8+0.5+0.6+... ~ 31200.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = false;
			TArray<float> A; BuildAudio(A, { { 50, 20 }, { 40, 25 }, { 30, 75 } });
			PushAndCheck(S, A, 8, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 3, false);
				CheckCalls(TEXT("chain"), { 24000, 43200, 60800 }, TolSamples);
				CheckFinals(TEXT("chain"), { false, true, true }, { 0, 1, 2 });
			});
		});

		LatentIt("MergeWindowZeroIsTheOldBehaviour", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 5. The case-2 audio with MergeWindowSeconds 0: two plain finals, never a replacement, second call ~24000.
			RunTwoBursts(0.f, 25, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 2, false);
				CheckCalls(TEXT("window 0"), { 24000, 24000 }, TolSamples);
				CheckFinals(TEXT("window 0"), { false, false }, { 0, 1 });
			});
		});

		LatentIt("ALoneClickIsAnEmptyFinalWithoutAnEngineCall", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 6a. 0.12 s click (6 frames: enough for SpeechStarted, below MinSpeechSeconds 0.25): Started, Ended, Final with EMPTY text,
			// no Transcribe call, no replacement. (Unlike an engine answering "", this empty final IS broadcast by the contract.)
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = false; S.MinSpeechSeconds = 0.25f;   // lead: C83 am.2 default is 0.1; the 0.12 s click must stay 'short'
			TArray<float> A; BuildAudio(A, { { 6, 75 } });
			PushAndCheck(S, A, 8, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 1, false);
				CheckCalls(TEXT("click"), {}, TolSamples);
				CheckFinals(TEXT("click"), { false }, { -1 });
			});
		});

		LatentIt("AClickAfterATurnJoinsItAndTheNextBurstStillJoins", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Lead, C83 am.1 (2)+(4): the click starts inside the window after burst 1, so it JOINS the turn (its final is a correction
			// covering burst 1 + click = 0.3+1.0+0.3+0.12+0.2 = 1.92 s = 30720, or — if burst 3 already ended when its work would start —
			// it is coalesced: empty, false, no flush). Burst 3 joins too: its final is true and covers all (51520). The original
			// expectations below this line are superseded.
			// Case 6b. Burst 1.0 s, 0.3 s silence, click 0.12 s, 0.3 s silence, burst 1.0 s. The third utterance starts 0.72 s after burst 1
			// ended = 0.52 s after burst 1's SpeechEnded point (inside 0.8) but 0.1 s after the click's own end point. Expected: 3 finals; the
			// click's is empty and not a replacement; the third joins the FIRST: true, one call covering burst 1 + click + burst 3 =
			// 0.3 + 1.0 + 0.3 + 0.12 + 0.3 + 1.0 + 0.2 = 3.22 s = 51520 samples. Total Transcribe calls: 2 (none for the click). A click that
			// started its own turn would make the third final a non-replacement (24000 samples).
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = false; S.MinSpeechSeconds = 0.25f;   // lead: C83 am.2 default is 0.1; the 0.12 s click must stay 'short'
			TArray<float> A; BuildAudio(A, { { 50, 15 }, { 6, 15 }, { 50, 75 } });
			PushAndCheck(S, A, 8, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 3, false);
				const TArray<FSpeechStreamTestRecord> Fi = Recv->Of(ESpeechStreamTestEvent::Final);
				FScopeLock L(&State->Lock);
				if (!TestEqual(TEXT("click joins: 3 finals"), Fi.Num(), 3)) { return; }
				TestFalse(TEXT("final 0 is a first final"), Fi[0].Transcript.bReplacesPrevious);
				const bool bClickCoalesced = Fi[1].Transcript.Text.IsEmpty();
				if (bClickCoalesced)
				{
					TestFalse(TEXT("coalesced click final is not a replacement"), Fi[1].Transcript.bReplacesPrevious);
					CheckCalls(TEXT("click coalesced"), { 24000, 51520 }, TolSamples);
				}
				else
				{
					TestTrue(TEXT("click final is a correction of burst 1"), Fi[1].Transcript.bReplacesPrevious);
					CheckCalls(TEXT("click joined"), { 24000, 30720, 51520 }, TolSamples);
				}
				TestTrue(TEXT("final 2 is a correction"), Fi[2].Transcript.bReplacesPrevious);
				TestFalse(TEXT("final 2 has text"), Fi[2].Transcript.Text.IsEmpty());
			});
		});

		LatentIt("TurnAudioIsCappedAtMaxUtteranceSeconds", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 7. MaxUtteranceSeconds 3, bursts of 2.0 s with a 0.4 s gap. Joined turn audio would be 0.3+2.0+0.4+2.0+0.2 = 4.9 s > 3 s,
			// so no join: both finals plain, the second call ~ 0.3+2.0+0.2 = 2.5 s = 40000.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = false; S.MaxUtteranceSeconds = 3.f;
			TArray<float> A; BuildAudio(A, { { 100, 20 }, { 100, 75 } });
			PushAndCheck(S, A, 8, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 2, false);
				CheckCalls(TEXT("turn cap"), { 40000, 40000 }, TolSamples);
				CheckFinals(TEXT("turn cap"), { false, false }, { 0, 1 });
			});
		});

		LatentIt("MergeWindowAboveMaximumBehavesAsThree", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 10a. MergeWindowSeconds 10 is clamped to 3. Silence of 2.7 s between the bursts = 2.5 s after the SpeechEnded point: joins
			// (the second call = 0.3+1.0+2.7+1.0+0.2 = 5.2 s = 83200). The next test (3.7 s silence = 3.5 s after the end point) must NOT
			// join: an unclamped 10 s window would join it.
			RunTwoBursts(10.f, 135, Done, [this]()
			{
				CheckCalls(TEXT("window 10, gap 2.5"), { 24000, 83200 }, TolSamples);
				CheckFinals(TEXT("window 10, gap 2.5"), { false, true }, { 0, 1 });
			});
		});

		LatentIt("MergeWindowAboveMaximumStillHasALimit", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 10a (second half). 3.7 s of silence = 3.5 s after the end point > the clamped 3 s: no join even with a requested 10 s.
			RunTwoBursts(10.f, 185, Done, [this]()
			{
				CheckCalls(TEXT("window 10, gap 3.5"), { 24000, 24000 }, TolSamples);
				CheckFinals(TEXT("window 10, gap 3.5"), { false, false }, { 0, 1 });
			});
		});

		LatentIt("NegativeMergeWindowBehavesAsOff", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 10b. -1 is clamped to 0: the case-2 audio (which joins with 0.8) gives two plain finals.
			RunTwoBursts(-1.f, 25, Done, [this]()
			{
				CheckCalls(TEXT("window -1"), { 24000, 24000 }, TolSamples);
				CheckFinals(TEXT("window -1"), { false, false }, { 0, 1 });
			});
		});

		LatentIt("NanMergeWindowBehavesAsTheDefault", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 10c. NaN must not disable or break the feature: it falls back to 0.8, so the case-2 audio joins. A NaN that slipped
			// through comparisons would make every "gap <= window" false and never join (or crash).
			RunTwoBursts(std::numeric_limits<float>::quiet_NaN(), 25, Done, [this]()
			{
				CheckCalls(TEXT("window NaN"), { 24000, 48000 }, TolSamples);
				CheckFinals(TEXT("window NaN"), { false, true }, { 0, 1 });
			});
		});
	});

	// =================================================================================================================
	// Phonon-like fake (streaming + partials)
	// =================================================================================================================
	Describe("PhononFake", [this]()
	{
		BeforeEach([this]() { SetupFake(true); });
		AfterEach([this]() { TeardownFake(); });

		LatentIt("JoinRefeedsTheTurnAudioAfterTheFirstFlushAndPartialsContinue", FTimespan::FromSeconds(400), [this](const FDoneDelegate& Done)
		{
			// Case 8 (+ case 9 grammar). Case-2 audio on the streaming fake, partials on. After the first Flush (24000 samples) the engine stream
			// is fed the turn audio again from the turn start and then live audio, so the second Flush covers 48000 samples (+-960) and the total
			// fed is >= 24000 + 48000 - 960. A join that did not re-feed would flush only ~24000 for the second final. The second final has
			// bReplacesPrevious, the first not; the fake's finals are "utt 0", "utt 1". Partials arrive during the second burst.
			if (!Pipe.IsValid()) { Done.ExecuteIfBound(); return; }
			FSpeechStreamSettings S; S.bPartials = true; S.PartialIntervalSeconds = 0.2f;
			TArray<float> A; BuildAudio(A, { { 50, 25 }, { 50, 75 } });
			PushAndCheck(S, A, 4, Done, [this]()
			{
				CheckGrammar(this, Recv.Get(), 2, true);
				const TArray<FSpeechStreamTestRecord> Fi = Recv->Of(ESpeechStreamTestEvent::Final);
				if (TestEqual(TEXT("two finals"), Fi.Num(), 2))
				{
					TestFalse(TEXT("final 0 is not a replacement"), Fi[0].Transcript.bReplacesPrevious);
					TestTrue(TEXT("final 1 is a replacement"), Fi[1].Transcript.bReplacesPrevious);
					TestTrue(*FString::Printf(TEXT("final 1 text '%s' is the engine's second flush 'utt 1'"), *Fi[1].Transcript.Text), Fi[1].Transcript.Text.Equals(TEXT("utt 1"), ESearchCase::CaseSensitive));
				}
				{
					FScopeLock L(&State->Lock);
					if (TestEqual(TEXT("two flushes with audio"), State->FlushPendingSamples.Num(), 2))
					{
						const int64 First = State->FlushPendingSamples[0];
						const int64 Second = State->FlushPendingSamples[1];
						TestTrue(*FString::Printf(TEXT("first flush %lld samples, expected ~24000"), (long long)First), FMath::Abs((int64)First - 24000) <= 2 * TolSamples);
						TestTrue(*FString::Printf(TEXT("second flush %lld samples, expected the contiguous turn audio ~48000 +-960"), (long long)Second), FMath::Abs((int64)Second - 48000) <= TolSamples);
						TestTrue(*FString::Printf(TEXT("total fed %lld >= first %lld + 48000 - 960"), (long long)State->FedSamples, (long long)First),
							State->FedSamples >= First + 48000 - TolSamples);
					}
					TestEqual(TEXT("one engine stream for the whole stream"), State->OpenStreamCount, 1);
					TestFalse(TEXT("engine stream never called on the game thread"), State->bFeedOnGameThread);
					TestEqual(TEXT("engine calls never overlap"), State->MaxInFlight, 1);
				}
				const TArray<FSpeechStreamTestRecord> St = Recv->Of(ESpeechStreamTestEvent::Started);
				const TArray<FSpeechStreamTestRecord> En = Recv->Of(ESpeechStreamTestEvent::Ended);
				if (St.Num() == 2 && En.Num() == 2)
				{
					int32 InSecond = 0;
					for (const FSpeechStreamTestRecord& P : Recv->Of(ESpeechStreamTestEvent::Partial)) { InSecond += (P.Seq > St[1].Seq && P.Seq < En[1].Seq) ? 1 : 0; }
					TestTrue(*FString::Printf(TEXT("%d partial(s) during the second burst, expected >= 1"), InSecond), InSecond >= 1);
				}
			});
		});
	});

	// =================================================================================================================
	// Real audio, stage A NNE model on the CPU
	// =================================================================================================================
	Describe("RealAudio", [this]()
	{
		AfterEach([this]()
		{
			Job.Reset();
			Stream.Reset();
			Recv.Reset();
			Pipe.Reset();
		});

		LatentIt("PausedReadSpeechStreamingWerWithinOnePointOfOffline", RealLimit, [this](const FDoneDelegate& Done)
		{
			// Case 11. paused_121_121726: read speech with 18 inserted 0.3-0.7 s pauses. Without join, every pause splits a sentence and the
			// engine loses context; with join the corrected finals bring streaming within 1.0 WER point of one offline pass, and the stream
			// must have produced >= 3 replacement finals (otherwise nothing was joined). Also logs the WER with MergeWindowSeconds 0.
			RunItem(TEXT("paused_121_121726"), 1.0, 3, true, Done);
		});

		LatentIt("AmiMeetingSpeechStreamingWerWithinTwoPointsOfOffline", RealLimit, [this](const FDoneDelegate& Done)
		{
			// Case 11. ami_EN2002a_FEO072 (meeting speech, CC-BY-4.0, evaluation only): streaming <= offline + 2.0 points.
			RunItem(TEXT("ami_EN2002a_FEO072"), 2.0, 0, false, Done);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
