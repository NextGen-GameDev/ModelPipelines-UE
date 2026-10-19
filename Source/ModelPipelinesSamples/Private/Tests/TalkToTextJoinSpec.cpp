// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9R.7t: the Talk-to-Text sample applies bReplacesPrevious finals (C83): the last line is replaced, the line count is kept, and
// OnFinalText / OnSampleResult still fire for every final. Written from docs/briefs/WP9R.7t.md; setup copied from TalkToTextSampleSpec.cpp.
// The stage A model data is loaded for this one test only and released afterwards (the 1.3 GB must not stay resident next to the
// other spec's copy).

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformTime.h"
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
#include "TalkToTextJoinReceiver.h"

using namespace UE::ModelPipelines::SamplesTests;

namespace TalkToTextJoinSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(600);
	constexpr double ReadySeconds = 200.0;
	constexpr double RecognitionSeconds = 300.0;
	constexpr int32 PieceSamples = 320;
	constexpr double PushSpeed = 4.0;
	constexpr int32 MaxPiecesPerTick = 25;
	constexpr int32 SpeechSamples = 25 * 16000;     // first 25 s of the item
	constexpr int32 TailSamples = 24000;            // 1.5 s of silence
	const TCHAR* const ItemId = TEXT("paused_121_121726");
	const TCHAR* const StageAHint = TEXT("Fetch the model with: py Tools/fetch_phonon2.py (see Docs/Samples.md), then export stage A into .pipelines-dev/models/stt/phonon-2-onnx/fp16w/.");
	const TCHAR* const ConvHint = TEXT("Generate the conversational goldens with the WP9R.7 script (Tools/make_stt_conversational_goldens.py) into <dev>/goldens/stt/conversational/.");

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

	static bool LoadAudio(TArray<float>& Out, FString& Err)
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
			if (!V.IsValid() || !V->TryGetObject(O) || !(*O)->GetStringField(TEXT("id")).Equals(ItemId, ESearchCase::CaseSensitive)) { continue; }
			const FString File = FPaths::Combine(Dir, (*O)->GetStringField(TEXT("file")));
			const int32 Num = (int32)(*O)->GetNumberField(TEXT("num_samples"));
			TArray<uint8> Bytes;
			if (!FFileHelper::LoadFileToArray(Bytes, *File) || Bytes.Num() != Num * 4)
			{
				Err = FString::Printf(TEXT("cannot read %s (%d bytes expected, got %d). %s"), *File, Num * 4, Bytes.Num(), ConvHint);
				return false;
			}
			if (Num < SpeechSamples) { Err = FString::Printf(TEXT("%s has only %d samples, need %d"), *File, Num, SpeechSamples); return false; }
			Out.SetNumUninitialized(SpeechSamples);
			FMemory::Memcpy(Out.GetData(), Bytes.GetData(), (SIZE_T)SpeechSamples * sizeof(float));
			return true;
		}
		Err = FString::Printf(TEXT("item '%s' not in %s. %s"), ItemId, *JsonPath, ConvHint);
		return false;
	}

	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Next, double Timeout)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TalkToTextJoinWait"), 0.02f, [T, Cond, What, Done, Next, Start, Timeout](float)
		{
			if (Cond()) { Next(); return false; }
			if ((FPlatformTime::Seconds() - Start) > Timeout)
			{
				T->AddError(FString::Printf(TEXT("timed out after %.0f s waiting for: %s"), Timeout, *What));
				Done.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	struct FRun : public TSharedFromThis<FRun>
	{
		FSamplesTestWorld World;
		TStrongObjectPtr<UTalkToTextJoinReceiver> Rcv;
		TStrongObjectPtr<UNNEModelData> Encoder;
		TStrongObjectPtr<UNNEModelData> Decoder;
		TStrongObjectPtr<UPipelineModelPackage> Pkg;
		TWeakObjectPtr<AModelPipelinesTalkToTextSample> Actor;
		TWeakObjectPtr<USpeechStream> PushTarget;
		TArray<float> Audio;
		FTSTicker::FDelegateHandle Pusher;
		int32 TotalToPush = 0;
		int32 PushedSamples = 0;
		double PushStart = 0.0;
		int32 SeenN = -1;
		double SeenAt = 0.0;
		bool bPushDone = false;
		double PushDoneTime = 0.0;

		bool Prepare(FAutomationTestBase& T)
		{
			FString Err;
			if (!LoadAudio(Audio, Err)) { T.AddError(Err); return false; }
			const FString Dir = FPaths::Combine(GetDevDir(), TEXT("models/stt/phonon-2-onnx/fp16w"));
			UNNEModelData* Enc = MakeModelData(FPaths::Combine(Dir, TEXT("encoder.onnx")), Err);
			if (!Enc) { T.AddError(Err); return false; }
			Encoder.Reset(Enc);
			UNNEModelData* Dec = MakeModelData(FPaths::Combine(Dir, TEXT("decoder_joint.onnx")), Err);
			if (!Dec) { T.AddError(Err); return false; }
			Decoder.Reset(Dec);
			FString Tok;
			if (!FFileHelper::LoadFileToString(Tok, *FPaths::Combine(Dir, TEXT("tokenizer.json"))) || Tok.IsEmpty())
			{
				T.AddError(FString::Printf(TEXT("cannot read %s/tokenizer.json. %s"), *Dir, StageAHint));
				return false;
			}
			if (!World.Create(Err)) { T.AddError(Err); return false; }
			Rcv.Reset(NewObject<UTalkToTextJoinReceiver>(GetTransientPackage()));
			UPipelineModelPackage* P = NewObject<UPipelineModelPackage>(GetTransientPackage());
			P->Task = EPipelineTask::SpeechToText;
			P->Model = Encoder.Get();
			P->DecoderModel = Decoder.Get();
			P->TokenizerJson = Tok;
			P->PreferredDevice = EPipelineDevice::CPU;
			P->License.SpdxId = TEXT("CC-BY-4.0");
			P->License.SourceUrl = TEXT("https://huggingface.co/");
			P->License.Attribution = TEXT("Phonon 2 (test package)");
			Pkg.Reset(P);
			return true;
		}

		void StartPush()
		{
			AModelPipelinesTalkToTextSample* A = Actor.Get();
			PushTarget = A ? A->GetStream() : nullptr;
			TotalToPush = SpeechSamples + TailSamples;
			PushedSamples = 0;
			bPushDone = false;
			PushStart = FPlatformTime::Seconds();
			TWeakPtr<FRun> Weak = AsShared();
			Pusher = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Weak](float)
			{
				const TSharedPtr<FRun> Self = Weak.Pin();
				if (!Self.IsValid()) { return false; }
				const double Now = FPlatformTime::Seconds();
				const int32 Target = FMath::Min(Self->TotalToPush, static_cast<int32>((Now - Self->PushStart) * PushSpeed * 16000.0));
				USpeechStream* S = Self->PushTarget.Get();
				TArray<float> Piece;
				Piece.SetNumUninitialized(PieceSamples);
				for (int32 Pieces = 0; Pieces < MaxPiecesPerTick && Self->PushedSamples + PieceSamples <= Self->TotalToPush && Self->PushedSamples + PieceSamples <= Target; ++Pieces)
				{
					for (int32 i = 0; i < PieceSamples; ++i)
					{
						const int32 Idx = Self->PushedSamples + i;
						Piece[i] = (Idx < SpeechSamples) ? Self->Audio[Idx] : 0.f;
					}
					if (IsValid(S)) { S->PushAudio(TConstArrayView<float>(Piece.GetData(), PieceSamples), 16000, 1); }
					Self->PushedSamples += PieceSamples;
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

		void Teardown()
		{
			if (Pusher.IsValid()) { FTSTicker::GetCoreTicker().RemoveTicker(Pusher); Pusher.Reset(); }
			if (AModelPipelinesTalkToTextSample* A = Actor.Get())
			{
				A->OnFinalText.RemoveAll(Rcv.Get());
				A->OnSampleResult.RemoveAll(Rcv.Get());
			}
			World.Destroy();
			Pkg.Reset();
			Encoder.Reset();
			Decoder.Reset();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		}
	};
}

BEGIN_DEFINE_SPEC(FTalkToTextJoinSpec, "Pipelines.Samples.TalkToText.Join",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	TSharedPtr<TalkToTextJoinSpecPrivate::FRun> Run;
END_DEFINE_SPEC(FTalkToTextJoinSpec)

void FTalkToTextJoinSpec::Define()
{
	using namespace TalkToTextJoinSpecPrivate;

	AfterEach([this]()
	{
		if (Run.IsValid()) { Run->Teardown(); Run.Reset(); }
	});

	LatentIt("ReplacingFinalsReplaceTheLastLineAndEverythingStillFires", LatentLimit, [this](const FDoneDelegate& Done)
	{
		// Case 12. First 25 s of paused_121_121726 (5 pauses of 0.3-0.65 s) pushed at 4x into the sample with the stage A package.
		// With the default merge window the pauses join, so some finals carry bReplacesPrevious. Contract: such a final REPLACES the last line
		// (the line count equals the number of NON-replacing finals), every final still raises OnFinalText and one OnSampleResult.
		// A sample that appended replacements would show more lines than non-replacing finals and repeat sentence starts; one that
		// swallowed replacing finals would raise fewer OnFinalText than finals (visible as OnSampleResult mismatch).
		const TSharedRef<FRun> R = MakeShared<FRun>();
		Run = R;
		if (!R->Prepare(*this)) { Done.ExecuteIfBound(); return; }
		FString Err;
		UTalkToTextJoinReceiver* Rc = R->Rcv.Get();
		AModelPipelinesSampleBase* Base = R->World.SpawnSample(AModelPipelinesTalkToTextSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(R->Pkg.Get()), false,
			[Rc](AModelPipelinesSampleBase* NewActor)
			{
				AModelPipelinesTalkToTextSample* S = CastChecked<AModelPipelinesTalkToTextSample>(NewActor);
				S->bUseMicrophone = false;
				S->MaxLines = 100;
				S->OnFinalText.AddDynamic(Rc, &UTalkToTextJoinReceiver::OnFinal);
				S->OnSampleResult.AddDynamic(Rc, &UTalkToTextJoinReceiver::OnResult);
			}, Err);
		AModelPipelinesTalkToTextSample* A = Cast<AModelPipelinesTalkToTextSample>(Base);
		if (!A) { AddError(Err.IsEmpty() ? FString(TEXT("SpawnSample returned null")) : Err); Done.ExecuteIfBound(); return; }
		R->Actor = A;
		const TWeakObjectPtr<AModelPipelinesTalkToTextSample> WA = A;
		WaitFor(this, [WA]() { const AModelPipelinesTalkToTextSample* S = WA.Get(); return !S || S->IsListening() || S->GetSampleState() == EPipelineSampleState::Failed || S->GetSampleState() == EPipelineSampleState::MissingPackage; },
			TEXT("the sample to start listening"), Done, [this, R, WA, Done]()
			{
				AModelPipelinesTalkToTextSample* S = WA.Get();
				if (!S || !S->IsListening())
				{
					AddError(FString::Printf(TEXT("sample did not start listening, status '%s'"), S ? *S->GetStatusText() : TEXT("<destroyed>")));
					Done.ExecuteIfBound();
					return;
				}
				R->StartPush();
				// Done when everything was pushed and no final arrived for 10 s.
				WaitFor(this, [R]()
				{
					const double Now = FPlatformTime::Seconds();
					const int32 N = R->Rcv->Texts.Num();
					if (N != R->SeenN) { R->SeenN = N; R->SeenAt = Now; }
					return R->bPushDone && (Now - R->SeenAt) > 10.0 && (Now - R->PushDoneTime) > 10.0;
				}, TEXT("the audio to be pushed and the last final to arrive"), Done, [this, R, WA, Done]()
				{
					const UTalkToTextJoinReceiver* Rc2 = R->Rcv.Get();
					const int32 Finals = Rc2->Texts.Num();
					int32 NonReplacing = 0, Replacing = 0;
					for (bool b : Rc2->Replaces) { (b ? Replacing : NonReplacing) += 1; }
					AddInfo(FString::Printf(TEXT("WP9R.7 sample: %d finals, %d replacing"), Finals, Replacing));
					TestTrue(*FString::Printf(TEXT("at least 2 finals, got %d"), Finals), Finals >= 2);
					TestTrue(*FString::Printf(TEXT("at least 1 replacing final in 25 s with 5 short pauses, got %d"), Replacing), Replacing >= 1);
					TestEqual(TEXT("OnFinalText count equals the finals recorded with a flag"), Rc2->Replaces.Num(), Finals);
					TestEqual(TEXT("one OnSampleResult per OnFinalText (replacing finals included)"), Rc2->ResultSuccess.Num(), Finals);
					for (int32 i = 0; i < Rc2->ResultSuccess.Num(); ++i) { TestTrue(*FString::Printf(TEXT("OnSampleResult %d has bSuccess"), i), Rc2->ResultSuccess[i]); }
					TestTrue(TEXT("every delivery on the game thread"), Rc2->bAllOnGameThread);
					if (Rc2->Replaces.Num() > 0) { TestFalse(TEXT("the very first final cannot be a replacement"), Rc2->Replaces[0]); }
					const AModelPipelinesTalkToTextSample* S = WA.Get();
					if (!TestNotNull(TEXT("actor alive"), S)) { Done.ExecuteIfBound(); return; }
					const TArray<FString> Lines = S->GetFinalLines();
					const int32 Cap = S->MaxLines;
					TestEqual(TEXT("GetFinalLines().Num() == number of finals with bReplacesPrevious false"), Lines.Num(), FMath::Min(NonReplacing, Cap));
					// Rebuild the expected lines from the recorded finals (replacing one replaces the last line) and compare, trimmed, case-sensitive.
					TArray<FString> Expect;
					for (int32 i = 0; i < Finals; ++i)
					{
						const FString T = Rc2->Texts[i].TrimStartAndEnd();
						if (Rc2->Replaces[i] && Expect.Num() > 0) { Expect.Last() = T; } else { Expect.Add(T); }
					}
					while (Expect.Num() > Cap) { Expect.RemoveAt(0); }
					if (TestEqual(TEXT("line count equals the rebuilt line count"), Lines.Num(), Expect.Num()))
					{
						for (int32 i = 0; i < Lines.Num(); ++i)
						{
							TestTrue(*FString::Printf(TEXT("line %d '%s' equals the expected '%s'"), i, *Lines[i], *Expect[i]), Lines[i].TrimStartAndEnd().Equals(Expect[i], ESearchCase::CaseSensitive));
						}
					}
					if (Finals > 0 && Rc2->Replaces.Last() && Lines.Num() > 0)
					{
						TestTrue(*FString::Printf(TEXT("after a replacing last final, the last line '%s' is that final's trimmed text '%s'"), *Lines.Last(), *Rc2->Texts.Last().TrimStartAndEnd()),
							Lines.Last().TrimStartAndEnd().Equals(Rc2->Texts.Last().TrimStartAndEnd(), ESearchCase::CaseSensitive));
					}
					Done.ExecuteIfBound();
				}, RecognitionSeconds);
			}, ReadySeconds);
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
