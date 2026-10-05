// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Engine/CancellableAsyncAction.h"
#include "HAL/PlatformTime.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/UnrealType.h"
#include "PipelineBase.h"
#include "PipelineModelPackage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "TextClassificationPipeline.h"
#include "TextEmbeddingPipeline.h"
#include "TextPipelineAsyncActions.h"
#include "TextPipelineLibrary.h"
#include "PipelineTextTestData.h"
#include "TextAsyncActionTestReceiver.h"

using namespace UE::ModelPipelines::TextTests;

namespace TextAsyncSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(240);
	constexpr double WaitSeconds = 150.0;
	/** After the awaited pin: wait this long to catch a second (wrong) delivery. */
	constexpr double SettleSeconds = 0.4;

	static TSharedPtr<FJsonObject> GCachedRoot;

	static TSharedPtr<FJsonObject> GetGoldens(FAutomationTestBase& T)
	{
		if (!GCachedRoot.IsValid())
		{
			FString Err;
			GCachedRoot = LoadTextGoldens(Err);
			if (!GCachedRoot.IsValid()) { T.AddError(Err); }
		}
		return GCachedRoot;
	}

	/** Which action/pipeline pair a scenario runs on. */
	struct FKind
	{
		const TCHAR* Name = TEXT("");
		bool bEmbed = true;
		const TCHAR* Slug = TEXT("");
		int32 TopK = 1;
		/** Golden sentence indices to send; empty = the two fixed texts ("hello world", "Grüße aus Köln"). */
		TArray<int32> Picks;
		/** Expected TopK entries per classification, -1 = do not check. */
		int32 ExpectEntries = -1;
	};

	static FName SuccessPin(const FKind& K) { return K.bEmbed ? FName(TEXT("EmbedSuccess")) : FName(TEXT("ClassifySuccess")); }
	static FName FailurePin(const FKind& K) { return K.bEmbed ? FName(TEXT("EmbedFailure")) : FName(TEXT("ClassifyFailure")); }

	static TArray<FString> ValidTexts()
	{
		return TArray<FString>{ TEXT("hello world"), TEXT("Grüße aus Köln") };
	}

	struct FFixture
	{
		TStrongObjectPtr<UPipelineModelPackage> Package;
		TStrongObjectPtr<UTextEmbeddingPipeline> Embed;
		TStrongObjectPtr<UTextClassificationPipeline> Class;
		TArray<FString> Sentences;
		UObject* Pipe() const { return Embed.IsValid() ? static_cast<UObject*>(Embed.Get()) : static_cast<UObject*>(Class.Get()); }
		void DropPipeline() { Embed.Reset(); Class.Reset(); }
	};

	static bool MakeFixture(FAutomationTestBase& T, const FKind& K, FFixture& F)
	{
		const TSharedPtr<FJsonObject> Root = GetGoldens(T);
		if (!Root.IsValid()) { return false; }
		const TSharedPtr<FJsonObject> Model = FindModel(Root, K.Slug);
		if (!Model.IsValid())
		{
			T.AddError(FString::Printf(TEXT("Golden model '%s' not found in text_goldens.json. Regenerate with Tools/make_text_goldens.py."), K.Slug));
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>* Sentences = nullptr;
		if (Root->TryGetArrayField(TEXT("sentences"), Sentences))
		{
			for (const TSharedPtr<FJsonValue>& V : *Sentences) { F.Sentences.Add(V->AsObject()->GetStringField(TEXT("text"))); }
		}
		FString Err;
		UPipelineModelPackage* Pkg = MakePackage(Root, Model, EPipelineDevice::CPU, Err);
		if (!Pkg) { T.AddError(Err); return false; }
		F.Package.Reset(Pkg);
		FPipelineResult R;
		if (K.bEmbed)
		{
			UTextEmbeddingPipeline* P = UTextPipelineLibrary::CreateTextEmbeddingPipeline(Pkg, R);
			if (!P) { T.AddError(FString::Printf(TEXT("CreateTextEmbeddingPipeline failed: %s"), *R.Message)); return false; }
			F.Embed.Reset(P);
		}
		else
		{
			UTextClassificationPipeline* P = UTextPipelineLibrary::CreateTextClassificationPipeline(Pkg, R);
			if (!P) { T.AddError(FString::Printf(TEXT("CreateTextClassificationPipeline failed: %s"), *R.Message)); return false; }
			F.Class.Reset(P);
		}
		return true;
	}

	static TArray<FString> TextsFor(const FKind& K, const FFixture& F)
	{
		if (K.Picks.Num() == 0) { return ValidTexts(); }
		TArray<FString> Out;
		for (int32 I : K.Picks) { if (F.Sentences.IsValidIndex(I)) { Out.Add(F.Sentences[I]); } }
		return Out;
	}

	// ---------------------------------------------------------------- waiting (never blocks the game thread)

	/** Polls Cond on the core ticker. On timeout: AddError(What) and Done. On success: Next(), which must call Done itself. */
	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Next, double Timeout = WaitSeconds)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TextAsyncSpecWait"), 0.02f, [T, Cond, What, Done, Next, Start, Timeout](float)
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
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TextAsyncSpecSettle"), 0.02f, [Start, Secs, Next](float)
		{
			if ((FPlatformTime::Seconds() - Start) >= Secs)
			{
				Next();
				return false;
			}
			return true;
		});
	}

	/** Waits for Cond, settles (to catch a second delivery), runs Check, then Done. */
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

	static TSharedRef<TFuture<void>> RunBg(TFunction<void()> Work)
	{
		return MakeShared<TFuture<void>>(Async(EAsyncExecution::ThreadPool, MoveTemp(Work)));
	}

	// ---------------------------------------------------------------- reference results (the blocking C++ call)

	struct FExpected
	{
		FPipelineResult R;
		TArray<FPipelineEmbedding> E;
		TArray<FPipelineClassification> C;
	};

	static void RunBlocking(bool bEmbed, UTextEmbeddingPipeline* PE, UTextClassificationPipeline* PC, int32 TopK, const TArray<FString>& Texts, FExpected& Out)
	{
		if (bEmbed) { Out.R = PE->EmbedBlocking(Texts, Out.E); }
		else { Out.R = PC->ClassifyBlocking(Texts, TopK, Out.C); }
	}

	static TSharedRef<TFuture<void>> StartExpected(const FKind& K, UObject* Pipe, const TArray<FString>& Texts, TSharedRef<FExpected> Out)
	{
		UTextEmbeddingPipeline* PE = Cast<UTextEmbeddingPipeline>(Pipe);
		UTextClassificationPipeline* PC = Cast<UTextClassificationPipeline>(Pipe);
		const bool bEmbed = K.bEmbed;
		const int32 TopK = K.TopK;
		return RunBg([=]() { RunBlocking(bEmbed, PE, PC, TopK, Texts, *Out); });
	}

	// ---------------------------------------------------------------- actions

	static TStrongObjectPtr<UTextAsyncActionTestReceiver> NewReceiver()
	{
		return TStrongObjectPtr<UTextAsyncActionTestReceiver>(NewObject<UTextAsyncActionTestReceiver>(GetTransientPackage()));
	}

	/** Factory call with a null world context (no game instance in the automation run), pins bound to R. */
	static UCancellableAsyncAction* NewAction(const FKind& K, UObject* Pipe, UTextAsyncActionTestReceiver* R, const TArray<FString>& Texts)
	{
		if (K.bEmbed)
		{
			UEmbedTextsAsyncAction* A = UEmbedTextsAsyncAction::EmbedTexts(nullptr, Cast<UTextEmbeddingPipeline>(Pipe), Texts);
			if (A)
			{
				A->OnSuccess.AddDynamic(R, &UTextAsyncActionTestReceiver::OnEmbedSuccess);
				A->OnFailure.AddDynamic(R, &UTextAsyncActionTestReceiver::OnEmbedFailure);
			}
			return A;
		}
		UClassifyTextsAsyncAction* A = UClassifyTextsAsyncAction::ClassifyTexts(nullptr, Cast<UTextClassificationPipeline>(Pipe), Texts, K.TopK);
		if (A)
		{
			A->OnSuccess.AddDynamic(R, &UTextAsyncActionTestReceiver::OnClassifySuccess);
			A->OnFailure.AddDynamic(R, &UTextAsyncActionTestReceiver::OnClassifyFailure);
		}
		return A;
	}

	static void Activate(UTextAsyncActionTestReceiver* R, UCancellableAsyncAction* A)
	{
		R->bInsideActivate = true;
		A->Activate();
		R->bInsideActivate = false;
	}

	static void ExpectNullPipelineWarning(FAutomationTestBase& T, const FKind& K)
	{
		T.AddExpectedMessage(FString::Printf(TEXT("%s: Pipeline is null"), K.bEmbed ? TEXT("Embed Texts") : TEXT("Classify Texts")), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
	}

	static void ExpectSecondActivateWarning(FAutomationTestBase& T, const FKind& K)
	{
		T.AddExpectedMessage(FString::Printf(TEXT("%s: Activate called again on an action that was already activated or cancelled; ignored"), K.bEmbed ? TEXT("Embed Texts") : TEXT("Classify Texts")), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
	}

	// ---------------------------------------------------------------- checks

	static void CheckEmbeddingsEqual(FAutomationTestBase& T, const TArray<FPipelineEmbedding>& Got, const TArray<FPipelineEmbedding>& Want, const FString& Label)
	{
		if (!T.TestEqual(*(Label + TEXT(": embedding count")), Got.Num(), Want.Num())) { return; }
		T.TestTrue(*(Label + TEXT(": at least one embedding")), Got.Num() > 0);
		int32 Bad = 0;
		double Worst = 0.0;
		for (int32 i = 0; i < Got.Num(); ++i)
		{
			if (Got[i].Values.Num() != Want[i].Values.Num())
			{
				T.AddError(FString::Printf(TEXT("%s: embedding %d has %d values, EmbedBlocking gave %d"), *Label, i, Got[i].Values.Num(), Want[i].Values.Num()));
				continue;
			}
			T.TestTrue(*FString::Printf(TEXT("%s: embedding %d is not empty"), *Label, i), Got[i].Values.Num() > 0);
			for (int32 j = 0; j < Got[i].Values.Num(); ++j)
			{
				if (Got[i].Values[j] != Want[i].Values[j])
				{
					++Bad;
					Worst = FMath::Max(Worst, (double)FMath::Abs(Got[i].Values[j] - Want[i].Values[j]));
				}
			}
		}
		T.TestEqual(*FString::Printf(TEXT("%s: floats differing from EmbedBlocking (exact equality required; worst abs diff %.3g)"), *Label, Worst), Bad, 0);
	}

	static void CheckClassificationsEqual(FAutomationTestBase& T, const TArray<FPipelineClassification>& Got, const TArray<FPipelineClassification>& Want, const FString& Label)
	{
		if (!T.TestEqual(*(Label + TEXT(": classification count")), Got.Num(), Want.Num())) { return; }
		T.TestTrue(*(Label + TEXT(": at least one classification")), Got.Num() > 0);
		int32 Bad = 0;
		double Worst = 0.0;
		for (int32 i = 0; i < Got.Num(); ++i)
		{
			const FPipelineClassification& G = Got[i];
			const FPipelineClassification& W = Want[i];
			if (G.Label != W.Label || G.Index != W.Index || G.Score != W.Score) { ++Bad; }
			if (G.Scores.Num() != W.Scores.Num() || G.TopK.Num() != W.TopK.Num())
			{
				T.AddError(FString::Printf(TEXT("%s: text %d has %d scores / %d TopK, ClassifyBlocking gave %d / %d"), *Label, i, G.Scores.Num(), G.TopK.Num(), W.Scores.Num(), W.TopK.Num()));
				++Bad;
				continue;
			}
			for (int32 c = 0; c < G.Scores.Num(); ++c)
			{
				if (G.Scores[c] != W.Scores[c]) { ++Bad; Worst = FMath::Max(Worst, (double)FMath::Abs(G.Scores[c] - W.Scores[c])); }
			}
			for (int32 c = 0; c < G.TopK.Num(); ++c)
			{
				if (G.TopK[c].Label != W.TopK[c].Label || G.TopK[c].Index != W.TopK[c].Index || G.TopK[c].Score != W.TopK[c].Score) { ++Bad; }
			}
		}
		T.TestEqual(*FString::Printf(TEXT("%s: fields differing from ClassifyBlocking (exact equality required; worst score diff %.3g)"), *Label, Worst), Bad, 0);
	}

	/** Exactly one record in total, on the expected pin, on the game thread, not inside Activate. Returns false if the count is wrong. */
	static bool CheckSingleDelivery(FAutomationTestBase& T, const UTextAsyncActionTestReceiver& R, FName Pin, FName OtherPin, const FString& Label)
	{
		T.TestEqual(*FString::Printf(TEXT("%s: %s count"), *Label, *Pin.ToString()), R.CountPin(Pin), 1);
		T.TestEqual(*FString::Printf(TEXT("%s: %s count (the other pin)"), *Label, *OtherPin.ToString()), R.CountPin(OtherPin), 0);
		if (!T.TestEqual(*FString::Printf(TEXT("%s: total deliveries (exactly once)"), *Label), R.Total(), 1)) { return false; }
		T.TestTrue(*FString::Printf(TEXT("%s: delivered on the game thread"), *Label), R.Records[0].bGameThread);
		T.TestFalse(*FString::Printf(TEXT("%s: delivered inside Activate()"), *Label), R.Records[0].bInsideActivate);
		return true;
	}

	static void CheckSuccessRecord(FAutomationTestBase& T, const FKind& K, const FTextAsyncActionTestRecord& Rec, const FExpected& Exp, const FString& Label)
	{
		T.TestEqual(*(Label + TEXT(": pin")), Rec.Pin.ToString(), SuccessPin(K).ToString());
		T.TestTrue(*FString::Printf(TEXT("%s: Result.bSuccess (%s)"), *Label, *Rec.Result.Message), Rec.Result.bSuccess);
		T.TestTrue(*(Label + TEXT(": Result.Error None")), Rec.Result.Error == EPipelineError::None);
		T.TestTrue(*FString::Printf(TEXT("%s: Result.Device %d == %d"), *Label, (int32)Rec.Result.Device, (int32)Exp.R.Device), Rec.Result.Device == Exp.R.Device);
		if (K.bEmbed)
		{
			CheckEmbeddingsEqual(T, Rec.Embeddings, Exp.E, Label);
		}
		else
		{
			CheckClassificationsEqual(T, Rec.Classifications, Exp.C, Label);
			if (K.ExpectEntries >= 0)
			{
				for (int32 i = 0; i < Rec.Classifications.Num(); ++i)
				{
					T.TestEqual(*FString::Printf(TEXT("%s: text %d TopK entries"), *Label, i), Rec.Classifications[i].TopK.Num(), K.ExpectEntries);
				}
			}
		}
	}

	static void CheckFailureRecord(FAutomationTestBase& T, const FKind& K, const FTextAsyncActionTestRecord& Rec, const FString& Label)
	{
		T.TestEqual(*(Label + TEXT(": pin")), Rec.Pin.ToString(), FailurePin(K).ToString());
		T.TestFalse(*(Label + TEXT(": Result.bSuccess")), Rec.Result.bSuccess);
		T.TestEqual(*(Label + TEXT(": output array empty on failure")), K.bEmbed ? Rec.Embeddings.Num() : Rec.Classifications.Num(), 0);
	}

	// ---------------------------------------------------------------- scenarios (run for both action kinds)

	static void ScenarioSuccess(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// One action, the result must equal the blocking call on the same pipeline bit for bit: the node adds no logic, so any
		// reordering, truncation, dropped TopK clamping, or stale shared buffer shows up as a differing float.
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		const TArray<FString> Texts = TextsFor(K, *F);
		const TSharedRef<FExpected> Exp = MakeShared<FExpected>();
		const TSharedRef<TFuture<void>> Fut = StartExpected(K, F->Pipe(), Texts, Exp);
		WaitFor(T, [Fut]() { return Fut->IsReady(); }, TEXT("blocking reference call"), Done, [T, K, F, Texts, Exp, Done]()
		{
			if (!Exp->R.bSuccess) { T->AddError(FString::Printf(TEXT("reference blocking call failed: %s"), *Exp->R.Message)); Done.ExecuteIfBound(); return; }
			const TStrongObjectPtr<UTextAsyncActionTestReceiver> R = NewReceiver();
			UCancellableAsyncAction* A = NewAction(K, F->Pipe(), R.Get(), Texts);
			if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
			const TStrongObjectPtr<UCancellableAsyncAction> Hold(A);   // only so IsActive() can be read after delivery
			T->TestEqual(TEXT("nothing fired before Activate"), R->Total(), 0);
			Activate(R.Get(), A);
			T->TestEqual(TEXT("nothing fired inside Activate (count right after it returned)"), R->Total(), 0);
			T->TestTrue(TEXT("IsActive() true after Activate"), A->IsActive());
			WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("OnSuccess/OnFailure never fired"), Done, [T, K, R, Hold, Exp]()
			{
				if (CheckSingleDelivery(*T, *R, SuccessPin(K), FailurePin(K), K.Name))
				{
					CheckSuccessRecord(*T, K, R->Records[0], *Exp, K.Name);
				}
				T->TestFalse(TEXT("IsActive() false after delivery"), Hold->IsActive());
			});
		});
	}

	static void ScenarioNullPipeline(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// Null pipeline: failure, InvalidArgument, "Pipeline is null", empty output, delivered later (never inside Activate).
		// A node that delivered synchronously would fire inside Activate and break latent-node semantics in Blueprints.
		const TStrongObjectPtr<UTextAsyncActionTestReceiver> R = NewReceiver();
		ExpectNullPipelineWarning(*T, K);
		UCancellableAsyncAction* A = NewAction(K, nullptr, R.Get(), ValidTexts());
		if (!T->TestNotNull(TEXT("factory never fails, even with a null pipeline"), A)) { Done.ExecuteIfBound(); return; }
		const TStrongObjectPtr<UCancellableAsyncAction> Hold(A);
		Activate(R.Get(), A);
		T->TestEqual(TEXT("null pipeline: nothing fired inside Activate"), R->Total(), 0);
		T->TestTrue(TEXT("null pipeline: IsActive() true until the failure is delivered"), A->IsActive());
		WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("null-pipeline failure never fired"), Done, [T, K, R, Hold]()
		{
			if (!CheckSingleDelivery(*T, *R, FailurePin(K), SuccessPin(K), TEXT("null pipeline"))) { return; }
			const FTextAsyncActionTestRecord& Rec = R->Records[0];
			CheckFailureRecord(*T, K, Rec, TEXT("null pipeline"));
			T->TestTrue(*FString::Printf(TEXT("null pipeline: Error InvalidArgument, got %d"), (int32)Rec.Result.Error), Rec.Result.Error == EPipelineError::InvalidArgument);
			T->TestEqual(TEXT("null pipeline: message"), Rec.Result.Message, FString(TEXT("Pipeline is null")));
			T->TestFalse(TEXT("null pipeline: IsActive() false after delivery"), Hold->IsActive());
		});
	}

	static void ScenarioNullPipelineCancelled(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// The null-pipeline failure is also deferred, so Cancel() right after Activate must suppress it.
		const TStrongObjectPtr<UTextAsyncActionTestReceiver> R = NewReceiver();
		ExpectNullPipelineWarning(*T, K);
		UCancellableAsyncAction* A = NewAction(K, nullptr, R.Get(), ValidTexts());
		if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
		Activate(R.Get(), A);
		A->Cancel();
		T->TestFalse(TEXT("IsActive() false right after Cancel"), A->IsActive());
		// A no-op wait of a second: the deferred game-thread task has certainly run by then.
		Settle(1.0, [T, R, Done]()
		{
			T->TestEqual(TEXT("cancelled null-pipeline action: no pin fired"), R->Total(), 0);
			Done.ExecuteIfBound();
		});
	}

	/** Failures the pipeline itself produces (empty batch; uninitialised pipeline) pass through with the pipeline's own error. */
	static void ScenarioPipelineFailure(FAutomationTestBase* T, FKind K, bool bUninitialised, FDoneDelegate Done)
	{
		// The node must add no check of its own: Error is whatever the blocking call reports on the same pipeline and texts.
		// A node that mapped every failure to InvalidArgument, or swallowed the error, would fail here.
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (bUninitialised)
		{
			UPackage* Outer = GetTransientPackage();
			if (K.bEmbed) { F->Embed.Reset(NewObject<UTextEmbeddingPipeline>(Outer)); }
			else { F->Class.Reset(NewObject<UTextClassificationPipeline>(Outer)); }
		}
		else if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		const TArray<FString> Texts = bUninitialised ? ValidTexts() : TArray<FString>();
		const TSharedRef<FExpected> Exp = MakeShared<FExpected>();
		const TSharedRef<TFuture<void>> Fut = StartExpected(K, F->Pipe(), Texts, Exp);
		WaitFor(T, [Fut]() { return Fut->IsReady(); }, TEXT("blocking reference call"), Done, [T, K, F, Texts, Exp, bUninitialised, Done]()
		{
			if (!T->TestFalse(TEXT("reference blocking call fails"), Exp->R.bSuccess)) { Done.ExecuteIfBound(); return; }
			T->TestTrue(TEXT("reference blocking call has an error code"), Exp->R.Error != EPipelineError::None);
			if (!bUninitialised)
			{
				T->TestTrue(*FString::Printf(TEXT("empty batch is InvalidArgument, got %d"), (int32)Exp->R.Error), Exp->R.Error == EPipelineError::InvalidArgument);
			}
			const TStrongObjectPtr<UTextAsyncActionTestReceiver> R = NewReceiver();
			UCancellableAsyncAction* A = NewAction(K, F->Pipe(), R.Get(), Texts);
			if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
			Activate(R.Get(), A);
			T->TestEqual(TEXT("nothing fired inside Activate"), R->Total(), 0);
			WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("failure never fired"), Done, [T, K, R, Exp]()
			{
				if (!CheckSingleDelivery(*T, *R, FailurePin(K), SuccessPin(K), K.Name)) { return; }
				const FTextAsyncActionTestRecord& Rec = R->Records[0];
				CheckFailureRecord(*T, K, Rec, K.Name);
				T->TestTrue(*FString::Printf(TEXT("%s: Error %d passed on unchanged (pipeline said %d)"), K.Name, (int32)Rec.Result.Error, (int32)Exp->R.Error),
					Rec.Result.Error == Exp->R.Error);
			});
		});
	}

	static void ScenarioSecondActivate(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// A second Activate warns and does nothing: still exactly one delivery (a second call would deliver twice).
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		const TStrongObjectPtr<UTextAsyncActionTestReceiver> R = NewReceiver();
		UCancellableAsyncAction* A = NewAction(K, F->Pipe(), R.Get(), TextsFor(K, *F));
		if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
		ExpectSecondActivateWarning(*T, K);
		Activate(R.Get(), A);
		Activate(R.Get(), A);
		WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("OnSuccess never fired"), Done, [T, K, R, F]()
		{
			CheckSingleDelivery(*T, *R, SuccessPin(K), FailurePin(K), TEXT("double Activate"));
			UPipelineBase* Base = Cast<UPipelineBase>(F->Pipe());
			if (Base) { T->TestEqual(TEXT("double Activate: no extra call pending"), Base->GetNumPendingCalls(), 0); }
		});
	}

	static void ScenarioCancelRightAfterActivate(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// Cancel before delivery: no pin ever fires; the running call is not interrupted, its result is dropped, then the action is released.
		// Control: a second, uncancelled action on the same pipeline is queued behind it. When the control delivers, the cancelled
		// call is certainly done (the executor completes calls; it must not leak a pending call or crash).
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		const TArray<FString> Texts = TextsFor(K, *F);
		const TSharedRef<FExpected> Exp = MakeShared<FExpected>();
		const TSharedRef<TFuture<void>> Fut = StartExpected(K, F->Pipe(), Texts, Exp);
		WaitFor(T, [Fut]() { return Fut->IsReady(); }, TEXT("blocking reference call"), Done, [T, K, F, Texts, Exp, Done]()
		{
			const TStrongObjectPtr<UTextAsyncActionTestReceiver> R1 = NewReceiver();
			const TStrongObjectPtr<UTextAsyncActionTestReceiver> R2 = NewReceiver();
			UCancellableAsyncAction* A = NewAction(K, F->Pipe(), R1.Get(), Texts);
			if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
			const TWeakObjectPtr<UCancellableAsyncAction> Weak(A);
			Activate(R1.Get(), A);
			T->TestTrue(TEXT("IsActive() true after Activate"), A->IsActive());
			A->Cancel();
			T->TestFalse(TEXT("IsActive() false after Cancel"), A->IsActive());
			A->Cancel();   // second Cancel: no-op, no crash
			T->TestFalse(TEXT("IsActive() still false after a second Cancel"), A->IsActive());
			UCancellableAsyncAction* Control = NewAction(K, F->Pipe(), R2.Get(), Texts);
			if (!T->TestNotNull(TEXT("control action"), Control)) { Done.ExecuteIfBound(); return; }
			Activate(R2.Get(), Control);
			WaitSettled(T, [R2]() { return R2->Total() > 0; }, TEXT("control action never delivered (executor stuck behind the cancelled call?)"), Done, [T, K, F, R1, R2, Weak, Exp]()
			{
				T->TestEqual(TEXT("cancelled action: no pin fired"), R1->Total(), 0);
				if (CheckSingleDelivery(*T, *R2, SuccessPin(K), FailurePin(K), TEXT("control"))) { CheckSuccessRecord(*T, K, R2->Records[0], *Exp, TEXT("control")); }
				UPipelineBase* Base = Cast<UPipelineBase>(F->Pipe());
				if (Base) { T->TestEqual(TEXT("the cancelled call still completed (no pending call left)"), Base->GetNumPendingCalls(), 0); }
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
				T->TestFalse(TEXT("cancelled action is released once its result arrived (unrooted and collected)"), Weak.IsValid());
			});
		});
	}

	static void ScenarioCancelAfterDelivery(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// Cancel after delivery (and twice) is a no-op: no second pin, no crash, IsActive stays false.
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		const TStrongObjectPtr<UTextAsyncActionTestReceiver> R = NewReceiver();
		UCancellableAsyncAction* A = NewAction(K, F->Pipe(), R.Get(), TextsFor(K, *F));
		if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
		const TStrongObjectPtr<UCancellableAsyncAction> Hold(A);
		Activate(R.Get(), A);
		WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("OnSuccess never fired"), Done, [T, K, R, Hold, Done]()
		{
			CheckSingleDelivery(*T, *R, SuccessPin(K), FailurePin(K), TEXT("before Cancel"));
			T->TestFalse(TEXT("IsActive() false after delivery"), Hold->IsActive());
			Hold->Cancel();
			Hold->Cancel();
			T->TestFalse(TEXT("IsActive() false after Cancel following delivery"), Hold->IsActive());
			Settle(SettleSeconds, [T, R, Done]()
			{
				T->TestEqual(TEXT("Cancel after delivery: still exactly one delivery in total"), R->Total(), 1);
				Done.ExecuteIfBound();
			});
		});
	}

	static void ScenarioKeepAlive(FAutomationTestBase* T, FKind K, bool bDropPipeline, FDoneDelegate Done)
	{
		// The test keeps no reference to the action (weak pointer only) and collects garbage right after Activate. The action roots
		// itself until delivery, so the pin still fires with the right result. With bDropPipeline the test also lets go of the
		// pipeline: the action's UPROPERTY must keep it alive for the in-flight call.
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		const TArray<FString> Texts = TextsFor(K, *F);
		const TSharedRef<FExpected> Exp = MakeShared<FExpected>();
		const TSharedRef<TFuture<void>> Fut = StartExpected(K, F->Pipe(), Texts, Exp);
		WaitFor(T, [Fut]() { return Fut->IsReady(); }, TEXT("blocking reference call"), Done, [T, K, F, Texts, Exp, bDropPipeline, Done]()
		{
			const TStrongObjectPtr<UTextAsyncActionTestReceiver> R = NewReceiver();
			TWeakObjectPtr<UCancellableAsyncAction> Weak;
			TWeakObjectPtr<UObject> WeakPipe(F->Pipe());
			{
				UCancellableAsyncAction* A = NewAction(K, F->Pipe(), R.Get(), Texts);
				if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
				Weak = A;
				Activate(R.Get(), A);
			}
			if (bDropPipeline) { F->DropPipeline(); }
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			T->TestTrue(TEXT("action survived GC while in flight (rooted)"), Weak.IsValid());
			if (bDropPipeline) { T->TestTrue(TEXT("pipeline survived GC while the action holds it"), WeakPipe.IsValid()); }
			WaitSettled(T, [R]() { return R->Total() > 0; }, TEXT("pin never fired after GC"), Done, [T, K, R, Exp]()
			{
				if (CheckSingleDelivery(*T, *R, SuccessPin(K), FailurePin(K), TEXT("keep-alive"))) { CheckSuccessRecord(*T, K, R->Records[0], *Exp, TEXT("keep-alive")); }
			});
		});
	}

	static void ScenarioListenerDestroyed(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// The object bound to the pins is destroyed and collected while the call is in flight: the broadcast must skip it without a crash.
		// Control action (own receiver) proves the call finished; afterwards the first action must have been released.
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		const TArray<FString> Texts = TextsFor(K, *F);
		TStrongObjectPtr<UTextAsyncActionTestReceiver> R1 = NewReceiver();
		const TStrongObjectPtr<UTextAsyncActionTestReceiver> R2 = NewReceiver();
		UCancellableAsyncAction* A = NewAction(K, F->Pipe(), R1.Get(), Texts);
		if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
		const TWeakObjectPtr<UCancellableAsyncAction> Weak(A);
		const TWeakObjectPtr<UTextAsyncActionTestReceiver> WeakR1(R1.Get());
		Activate(R1.Get(), A);
		UCancellableAsyncAction* Control = NewAction(K, F->Pipe(), R2.Get(), Texts);
		if (!T->TestNotNull(TEXT("control action"), Control)) { Done.ExecuteIfBound(); return; }
		Activate(R2.Get(), Control);
		R1.Reset();
		CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		T->TestFalse(TEXT("listener was collected"), WeakR1.IsValid());
		WaitSettled(T, [R2]() { return R2->Total() > 0; }, TEXT("control action never delivered"), Done, [T, K, R2, Weak]()
		{
			CheckSingleDelivery(*T, *R2, SuccessPin(K), FailurePin(K), TEXT("control"));
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			T->TestFalse(TEXT("action is released after delivering to a dead listener"), Weak.IsValid());
		});
	}

	static void ScenarioEightConcurrent(FAutomationTestBase* T, FKind K, FDoneDelegate Done)
	{
		// 8 actions in flight on one pipeline, each with its own text and its own receiver. Receiver i must get exactly the result of
		// text i (a node that shared a result buffer, or delivered in completion order to the wrong listener, would mix them up).
		const TSharedRef<FFixture> F = MakeShared<FFixture>();
		if (!MakeFixture(*T, K, *F)) { Done.ExecuteIfBound(); return; }
		const TSharedRef<TArray<FString>> Texts = MakeShared<TArray<FString>>(TArray<FString>{
			TEXT("alpha"),
			TEXT("The quick brown fox jumps over the lazy dog."),
			TEXT("Grüße aus Köln"),
			TEXT("你好，世界"),
			TEXT("emoji \U0001F600 test"),
			TEXT("a"),
			TEXT("I really enjoyed this movie, it was wonderful and moving."),
			TEXT("Terrible, boring, a complete waste of money.") });
		const TSharedRef<TArray<FExpected>> Exp = MakeShared<TArray<FExpected>>();
		Exp->SetNum(8);
		UTextEmbeddingPipeline* PE = Cast<UTextEmbeddingPipeline>(F->Pipe());
		UTextClassificationPipeline* PC = Cast<UTextClassificationPipeline>(F->Pipe());
		const bool bEmbed = K.bEmbed;
		const int32 TopK = K.TopK;
		const TSharedRef<TFuture<void>> Fut = RunBg([=]()
		{
			for (int32 i = 0; i < 8; ++i) { RunBlocking(bEmbed, PE, PC, TopK, TArray<FString>{ (*Texts)[i] }, (*Exp)[i]); }
		});
		WaitFor(T, [Fut]() { return Fut->IsReady(); }, TEXT("blocking reference calls"), Done, [T, K, F, Texts, Exp, Done]()
		{
			TSet<FString> Signatures;
			for (int32 i = 0; i < 8; ++i)
			{
				if (!(*Exp)[i].R.bSuccess) { T->AddError(FString::Printf(TEXT("reference call %d failed: %s"), i, *(*Exp)[i].R.Message)); Done.ExecuteIfBound(); return; }
				FString Sig;
				if (K.bEmbed) { for (float V : (*Exp)[i].E[0].Values) { Sig += FString::Printf(TEXT("%.6f,"), V); } }
				else { for (float V : (*Exp)[i].C[0].Scores) { Sig += FString::Printf(TEXT("%.6f,"), V); } }
				Signatures.Add(Sig);
			}
			if (Signatures.Num() < 2) { T->AddError(TEXT("the 8 test texts give identical reference results; cross-delivery could not be detected")); }
			const TSharedRef<TArray<TStrongObjectPtr<UTextAsyncActionTestReceiver>>> Rs = MakeShared<TArray<TStrongObjectPtr<UTextAsyncActionTestReceiver>>>();
			TArray<UCancellableAsyncAction*> Actions;
			for (int32 i = 0; i < 8; ++i)
			{
				Rs->Add(NewReceiver());
				UCancellableAsyncAction* A = NewAction(K, F->Pipe(), (*Rs)[i].Get(), TArray<FString>{ (*Texts)[i] });
				if (!T->TestNotNull(TEXT("factory returned an action"), A)) { Done.ExecuteIfBound(); return; }
				Actions.Add(A);
			}
			for (int32 i = 0; i < 8; ++i) { Activate((*Rs)[i].Get(), Actions[i]); }
			WaitSettled(T, [Rs]()
			{
				for (const TStrongObjectPtr<UTextAsyncActionTestReceiver>& R : *Rs) { if (R->Total() == 0) { return false; } }
				return true;
			}, TEXT("not all 8 actions delivered"), Done, [T, K, Rs, Exp]()
			{
				for (int32 i = 0; i < 8; ++i)
				{
					const FString Label = FString::Printf(TEXT("%s action %d"), K.Name, i);
					if (CheckSingleDelivery(*T, *(*Rs)[i], SuccessPin(K), FailurePin(K), Label)) { CheckSuccessRecord(*T, K, (*Rs)[i]->Records[0], (*Exp)[i], Label); }
				}
			});
		});
	}

	static void CheckFactory(FAutomationTestBase& T, UClass* Cls, const TCHAR* FuncName, const TCHAR* DisplayName, const TCHAR* ParamName, const TCHAR* ExpectedOutputName)
	{
		UFunction* Fn = Cls->FindFunctionByName(FuncName);
		if (!T.TestNotNull(*FString::Printf(TEXT("%s::%s exists as a UFUNCTION"), *Cls->GetName(), FuncName), Fn)) { return; }
		T.TestTrue(TEXT("factory is static"), Fn->HasAnyFunctionFlags(FUNC_Static));
		T.TestTrue(TEXT("factory is BlueprintCallable"), Fn->HasAnyFunctionFlags(FUNC_BlueprintCallable));
#if WITH_METADATA
		T.TestTrue(TEXT("meta BlueprintInternalUseOnly"), Fn->HasMetaData(TEXT("BlueprintInternalUseOnly")));
		T.TestEqual(TEXT("BlueprintInternalUseOnly value"), Fn->GetMetaData(TEXT("BlueprintInternalUseOnly")), FString(TEXT("true")));
		T.TestEqual(TEXT("meta WorldContext"), Fn->GetMetaData(TEXT("WorldContext")), FString(TEXT("WorldContextObject")));
		T.TestEqual(TEXT("meta DisplayName"), Fn->GetMetaData(TEXT("DisplayName")), FString(DisplayName));
		T.TestEqual(TEXT("meta Category"), Fn->GetMetaData(TEXT("Category")), FString(TEXT("Model Pipelines|Text")));
#endif
		T.TestNotNull(TEXT("parameter WorldContextObject"), Fn->FindPropertyByName(TEXT("WorldContextObject")));
		T.TestNotNull(*FString::Printf(TEXT("parameter %s"), ParamName), Fn->FindPropertyByName(ParamName));
		T.TestNotNull(TEXT("parameter Texts"), Fn->FindPropertyByName(TEXT("Texts")));
		const FObjectProperty* Ret = CastField<FObjectProperty>(Fn->GetReturnProperty());
		if (T.TestNotNull(TEXT("factory has an object return value"), Ret)) { T.TestTrue(TEXT("factory returns the action class"), Ret->PropertyClass == Cls); }
		(void)ExpectedOutputName;
	}

	static void CheckPins(FAutomationTestBase& T, UClass* Cls, const TCHAR* OutputParamName)
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
		T.TestTrue(TEXT("OnSuccess and OnFailure share one delegate signature (required by K2Node_BaseAsyncTask)"), Ok->SignatureFunction == Fail->SignatureFunction);
		if (Ok->SignatureFunction)
		{
			T.TestNotNull(TEXT("signature parameter Result"), Ok->SignatureFunction->FindPropertyByName(TEXT("Result")));
			T.TestNotNull(*FString::Printf(TEXT("signature parameter %s"), OutputParamName), Ok->SignatureFunction->FindPropertyByName(OutputParamName));
			int32 NumParams = 0;
			for (TFieldIterator<FProperty> It(Ok->SignatureFunction); It; ++It) { ++NumParams; }
			T.TestEqual(TEXT("signature has exactly two parameters"), NumParams, 2);
		}
	}
}

BEGIN_DEFINE_SPEC(FTextPipelineAsyncActionSpec, "Pipelines.Text.AsyncAction",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FTextPipelineAsyncActionSpec)

void FTextPipelineAsyncActionSpec::Define()
{
	using namespace TextAsyncSpecPrivate;

	// Kinds used for the delivery / cancel / keep-alive scenarios: the tiny models keep them fast.
	TArray<FKind> Kinds;
	{
		FKind E; E.Name = TEXT("EmbedTexts"); E.bEmbed = true; E.Slug = TEXT("tiny_text_embedding"); E.TopK = 1;
		Kinds.Add(E);
		FKind C; C.Name = TEXT("ClassifyTexts"); C.bEmbed = false; C.Slug = TEXT("tiny_text_classification"); C.TopK = 2; C.ExpectEntries = 2;
		Kinds.Add(C);
	}

	// ------------------------------------------------------------------------------------------------------
	Describe("Success", [this, Kinds]()
	{
		for (const FKind& K : Kinds)
		{
			LatentIt(*FString::Printf(TEXT("%sEqualsBlockingCallOnSameTinyPipeline"), K.Name), LatentLimit, [this, K](const FDoneDelegate& Done)
			{
				ScenarioSuccess(this, K, Done);
			});
		}

		LatentIt("ClassifyTextsTopK2OnDistilbertEqualsBlockingCall", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Reference model, two golden sentences (a positive and a negative review), TopK 2: two label scores per text, equal to ClassifyBlocking.
			FKind K; K.Name = TEXT("ClassifyTexts distilbert"); K.bEmbed = false; K.Slug = TEXT("distilbert-sst2"); K.TopK = 2; K.Picks = { 17, 22 }; K.ExpectEntries = 2;
			ScenarioSuccess(this, K, Done);
		});

		LatentIt("ClassifyTextsTopK0ClampsToOneScorePerText", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// TopK 0 is clamped by the pipeline (to 1), not by the node: one TopK entry per text, identical to ClassifyBlocking(TopK 0).
			FKind K; K.Name = TEXT("ClassifyTexts topk0"); K.bEmbed = false; K.Slug = TEXT("distilbert-sst2"); K.TopK = 0; K.Picks = { 17, 22 }; K.ExpectEntries = 1;
			ScenarioSuccess(this, K, Done);
		});

		LatentIt("ClassifyTextsTopK5With3LabelsClampsToThree", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Upper clamp: TopK above the label count (3) gives 3 entries, passed through unchanged.
			FKind K; K.Name = TEXT("ClassifyTexts topk5"); K.bEmbed = false; K.Slug = TEXT("tiny_text_classification"); K.TopK = 5; K.ExpectEntries = 3;
			ScenarioSuccess(this, K, Done);
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Failure", [this, Kinds]()
	{
		for (const FKind& K : Kinds)
		{
			Describe(K.Name, [this, K]()
			{
				LatentIt("NullPipelineFailsOnceWithInvalidArgumentLaterThanActivate", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioNullPipeline(this, K, Done); });
				LatentIt("CancelledNullPipelineActionFiresNothing", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioNullPipelineCancelled(this, K, Done); });
				LatentIt("EmptyBatchPassesPipelineErrorThrough", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioPipelineFailure(this, K, false, Done); });
				LatentIt("UninitializedPipelinePassesPipelineErrorThrough", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioPipelineFailure(this, K, true, Done); });
				LatentIt("SecondActivateWarnsAndDeliversOnce", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioSecondActivate(this, K, Done); });
			});
		}
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Cancel", [this, Kinds]()
	{
		for (const FKind& K : Kinds)
		{
			Describe(K.Name, [this, K]()
			{
				LatentIt("CancelRightAfterActivateFiresNoPinAndReleasesAction", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioCancelRightAfterActivate(this, K, Done); });
				LatentIt("CancelAfterDeliveryAndDoubleCancelAreNoOps", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioCancelAfterDelivery(this, K, Done); });
			});
		}
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Lifetime", [this, Kinds]()
	{
		for (const FKind& K : Kinds)
		{
			Describe(K.Name, [this, K]()
			{
				LatentIt("ActionSurvivesGCWithNoReferencesHeldByTest", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioKeepAlive(this, K, false, Done); });
				LatentIt("ActionKeepsPipelineAliveWhenTestDropsIt", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioKeepAlive(this, K, true, Done); });
				LatentIt("ListenerDestroyedAndCollectedWhileInFlightIsNoCrash", LatentLimit, [this, K](const FDoneDelegate& Done) { ScenarioListenerDestroyed(this, K, Done); });
			});
		}
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Concurrency", [this, Kinds]()
	{
		for (const FKind& K : Kinds)
		{
			// Lead fix (WP5.1): tiny_text_embedding (CLS pooling) gives the same vector for every input, so the embed case
			// runs on the mean-pooled reference model, whose output depends on the text.
			FKind KC = K;
			if (KC.bEmbed) { KC.Slug = TEXT("all-MiniLM-L6-v2"); }
			LatentIt(*FString::Printf(TEXT("%sEightActionsOnOnePipelineEachGetsItsOwnResult"), K.Name), LatentLimit, [this, KC](const FDoneDelegate& Done) { ScenarioEightConcurrent(this, KC, Done); });
		}
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Reflection", [this]()
	{
		It("EmbedTextsFactoryMetadata", [this]()
		{
			CheckFactory(*this, UEmbedTextsAsyncAction::StaticClass(), TEXT("EmbedTexts"), TEXT("Embed Texts"), TEXT("Pipeline"), TEXT("Embeddings"));
		});

		It("ClassifyTextsFactoryMetadata", [this]()
		{
			CheckFactory(*this, UClassifyTextsAsyncAction::StaticClass(), TEXT("ClassifyTexts"), TEXT("Classify Texts"), TEXT("Pipeline"), TEXT("Classifications"));
			UFunction* Fn = UClassifyTextsAsyncAction::StaticClass()->FindFunctionByName(TEXT("ClassifyTexts"));
			if (Fn)
			{
				TestNotNull(TEXT("parameter TopK"), Fn->FindPropertyByName(TEXT("TopK")));
#if WITH_METADATA
				TestEqual(TEXT("TopK default meta (CPP_Default_TopK)"), Fn->GetMetaData(TEXT("CPP_Default_TopK")), FString(TEXT("1")));
#endif
			}
		});

		It("EmbedTextsHasExactlyTwoAssignablePinsWithOneSignature", [this]()
		{
			CheckPins(*this, UEmbedTextsAsyncAction::StaticClass(), TEXT("Embeddings"));
		});

		It("ClassifyTextsHasExactlyTwoAssignablePinsWithOneSignature", [this]()
		{
			CheckPins(*this, UClassifyTextsAsyncAction::StaticClass(), TEXT("Classifications"));
		});

		It("ActionsDeriveFromCancellableAsyncAction", [this]()
		{
			TestTrue(TEXT("UEmbedTextsAsyncAction is a UCancellableAsyncAction"), UEmbedTextsAsyncAction::StaticClass()->IsChildOf(UCancellableAsyncAction::StaticClass()));
			TestTrue(TEXT("UClassifyTextsAsyncAction is a UCancellableAsyncAction"), UClassifyTextsAsyncAction::StaticClass()->IsChildOf(UCancellableAsyncAction::StaticClass()));
		});

		It("FactoryDoesNoWorkAndNeverFailsWithNullInputs", [this]()
		{
			// Factory with a null world context and a null pipeline: returns an action, nothing fires, no warning from the factory itself.
			UEmbedTextsAsyncAction* E = UEmbedTextsAsyncAction::EmbedTexts(nullptr, nullptr, ValidTexts());
			UClassifyTextsAsyncAction* C = UClassifyTextsAsyncAction::ClassifyTexts(nullptr, nullptr, ValidTexts(), 1);
			TestNotNull(TEXT("EmbedTexts(null, null, ...)"), E);
			TestNotNull(TEXT("ClassifyTexts(null, null, ...)"), C);
			// Never activated: nothing roots them; collect so they do not leak into later tests.
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
