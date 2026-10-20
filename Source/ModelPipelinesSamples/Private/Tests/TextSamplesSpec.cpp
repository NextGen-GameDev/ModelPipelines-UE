// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "HAL/PlatformTime.h"
#include "UObject/Class.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineBase.h"
#include "PipelineLibrary.h"
#include "PipelineModelPackage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "TextClassificationPipeline.h"
#include "TextEmbeddingPipeline.h"
#include "TextPipelineLibrary.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesSemanticSearchSample.h"
#include "ModelPipelinesSentimentSample.h"
#include "SamplesTestData.h"
#include "SamplesTestReceiver.h"

using namespace UE::ModelPipelines::SamplesTests;

namespace TextSamplesSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(240);
	constexpr double WaitSeconds = 150.0;
	/** After the awaited result: wait this long to catch a second (wrong) delivery. */
	constexpr double SettleSeconds = 0.6;

	const TCHAR* const SearchSlug = TEXT("all-MiniLM-L6-v2");
	const TCHAR* const SentimentSlug = TEXT("distilbert-sst2");

	/** The one place where the brief fixes the exact wording (rule 1). */
	static FString MissingText(const FString& Path, const FString& Slug)
	{
		return FString::Printf(TEXT("Model package not found: %s. Run Tools/fetch_reference_models.py --only %s (see Docs/ImportModel.md), import the folder into /Game/Models, then press Play again. See Docs/Samples.md."), *Path, *Slug);
	}

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

	// ---------------------------------------------------------------- fixture

	/** Everything one test owns: the world with the sample, reference pipelines (same packages) and the result receiver. */
	struct FFix : public TSharedFromThis<FFix>
	{
		FSamplesTestWorld World;
		TStrongObjectPtr<UPipelineModelPackage> EmbedPkg;
		TStrongObjectPtr<UPipelineModelPackage> ClassPkg;
		TStrongObjectPtr<UTextEmbeddingPipeline> RefEmbed;
		TStrongObjectPtr<UTextClassificationPipeline> RefClass;
		TStrongObjectPtr<USamplesTestReceiver> Receiver;
		TWeakObjectPtr<AModelPipelinesSampleBase> Actor;

		/** Highest GetNumPendingCalls() of the sample's pipeline seen on any tick (the sample must never have 2 calls in flight). */
		int32 MaxPending = 0;
		bool bMonitorSawPipeline = false;
		FTSTicker::FDelegateHandle Monitor;

		bool Setup(FAutomationTestBase& T, bool bEmbed, bool bClass)
		{
			const TSharedPtr<FJsonObject> Root = GetGoldens(T);
			if (!Root.IsValid()) { return false; }
			for (int32 Pass = 0; Pass < 2; ++Pass)
			{
				const bool bThisEmbed = (Pass == 0);
				if (!(bThisEmbed ? bEmbed : bClass)) { continue; }
				const FString Slug = bThisEmbed ? SearchSlug : SentimentSlug;
				const TSharedPtr<FJsonObject> Model = FindModel(Root, Slug);
				if (!Model.IsValid())
				{
					T.AddError(FString::Printf(TEXT("Golden model '%s' not found in text_goldens.json. Regenerate with Tools/make_text_goldens.py."), *Slug));
					return false;
				}
				FString Err;
				UPipelineModelPackage* Pkg = MakePackage(Root, Model, EPipelineDevice::CPU, Err);
				if (!Pkg) { T.AddError(Err); return false; }
				FPipelineResult R;
				if (bThisEmbed)
				{
					EmbedPkg.Reset(Pkg);
					UTextEmbeddingPipeline* P = UTextPipelineLibrary::CreateTextEmbeddingPipeline(Pkg, R);
					if (!P) { T.AddError(FString::Printf(TEXT("CreateTextEmbeddingPipeline(%s) failed: %s"), *Slug, *R.Message)); return false; }
					RefEmbed.Reset(P);
				}
				else
				{
					ClassPkg.Reset(Pkg);
					UTextClassificationPipeline* P = UTextPipelineLibrary::CreateTextClassificationPipeline(Pkg, R);
					if (!P) { T.AddError(FString::Printf(TEXT("CreateTextClassificationPipeline(%s) failed: %s"), *Slug, *R.Message)); return false; }
					RefClass.Reset(P);
				}
			}
			FString Err;
			if (!World.Create(Err)) { T.AddError(Err); return false; }
			Receiver.Reset(NewObject<USamplesTestReceiver>(GetTransientPackage()));
			return true;
		}

		/** Spawns the sample (play begins), binds OnSampleResult to the receiver before BeginPlay, starts the pending-call monitor. */
		AModelPipelinesSampleBase* Spawn(FAutomationTestBase& T, UClass* SampleClass, const TSoftObjectPtr<UPipelineModelPackage>& Package, bool bOverlay = false)
		{
			FString Err;
			USamplesTestReceiver* Rcv = Receiver.Get();
			AModelPipelinesSampleBase* A = World.SpawnSample(SampleClass, Package, bOverlay,
				[Rcv](AModelPipelinesSampleBase* NewActor) { NewActor->OnSampleResult.AddDynamic(Rcv, &USamplesTestReceiver::OnResult); }, Err);
			if (!A) { T.AddError(Err); return nullptr; }
			Actor = A;
			TWeakPtr<FFix> Weak = AsShared();
			Monitor = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([Weak](float)
			{
				const TSharedPtr<FFix> Self = Weak.Pin();
				if (!Self.IsValid()) { return false; }
				if (UPipelineBase* Pipe = FindSamplePipeline(Self->Actor.Get()))
				{
					Self->bMonitorSawPipeline = true;
					Self->MaxPending = FMath::Max(Self->MaxPending, Pipe->GetNumPendingCalls());
				}
				return true;
			}), 0.0f);
			return A;
		}

		void Teardown()
		{
			if (Monitor.IsValid())
			{
				FTSTicker::GetCoreTicker().RemoveTicker(Monitor);
				Monitor.Reset();
			}
			if (AModelPipelinesSampleBase* A = Actor.Get())
			{
				A->OnSampleResult.RemoveAll(Receiver.Get());
			}
			World.Destroy();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
		}
	};

	static TSharedPtr<FFix> NewFix(FAutomationTestBase& T, bool bEmbed, bool bClass)
	{
		TSharedRef<FFix> F = MakeShared<FFix>();
		if (!F->Setup(T, bEmbed, bClass))
		{
			F->Teardown();
			return nullptr;
		}
		return F;
	}

	// ---------------------------------------------------------------- waiting (never blocks the game thread)

	/** Polls Cond on the core ticker. On timeout: AddError(What) and Done. On success: Next(), which must call Done itself. */
	static void WaitFor(FAutomationTestBase* T, TFunction<bool()> Cond, FString What, FDoneDelegate Done, TFunction<void()> Next, double Timeout = WaitSeconds)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TextSamplesSpecWait"), 0.02f, [T, Cond, What, Done, Next, Start, Timeout](float)
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
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TextSamplesSpecSettle"), 0.02f, [Start, Secs, Next](float)
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

	static bool StateIs(const AModelPipelinesSampleBase* A, EPipelineSampleState S) { return A && A->GetSampleState() == S; }

	static FString StateName(const AModelPipelinesSampleBase* A)
	{
		return A ? UEnum::GetValueAsString(A->GetSampleState()) : FString(TEXT("(null actor)"));
	}

	/** The sample's own "Model package not found" Warning (MissingPackage / unset package): exactly one line per test. */
	static void ExpectWarning(FAutomationTestBase& T)
	{
		T.AddExpectedMessage(TEXT("Model package not found: "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
		// Engine noise: the first load of the missing package path logs this once per process (cached afterwards), so which test sees it depends on run order.
		T.AddExpectedMessage(TEXT("LoadPackage: SkipPackage: /Game/Models/does-not-exist"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1, false);
	}

	/** The creator's rejection of a package of the wrong task, logged once by the sample ("could not create the pipeline from ...: package task is X, this pipeline needs Y"). */
	static void ExpectWrongTask(FAutomationTestBase& T)
	{
		T.AddExpectedMessage(TEXT("could not create the pipeline from [^ ]+: package task is \\w+, this pipeline needs \\w+"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
	}

	/** Engine noise that is not ours: the sample test worlds have no world context, so AActor::Destroy logs this. Count varies with how many actors each test destroys. */
	static void ExpectWorldNoContext(FAutomationTestBase& T)
	{
		T.AddExpectedMessage(TEXT("UWorld::DestroyActor: World has no context!"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1, false);
	}

	// ---------------------------------------------------------------- semantic search references

	struct FSearchRef
	{
		FPipelineResult CorpusResult;
		TArray<FPipelineEmbedding> Corpus;
		TArray<FPipelineResult> QueryResults;
		TArray<TArray<FPipelineEmbedding>> Queries;
	};

	/** Reference: the whole corpus in one EmbedBlocking call (like the sample), each query in its own call. */
	static TSharedRef<TFuture<void>> StartSearchRef(const TSharedRef<FFix>& F, const TArray<FString>& Lines, const TArray<FString>& Queries, const TSharedRef<FSearchRef>& Out)
	{
		UTextEmbeddingPipeline* P = F->RefEmbed.Get();
		return RunBg([P, Lines, Queries, Out]()
		{
			Out->CorpusResult = P->EmbedBlocking(Lines, Out->Corpus);
			for (const FString& Q : Queries)
			{
				TArray<FPipelineEmbedding>& E = Out->Queries.AddDefaulted_GetRef();
				Out->QueryResults.Add(P->EmbedBlocking(TArray<FString>{ Q }, E));
			}
		});
	}

	static bool CheckSearchRef(FAutomationTestBase& T, const FSearchRef& R, int32 NumQueries)
	{
		bool bOk = T.TestTrue(TEXT("reference corpus EmbedBlocking succeeded"), R.CorpusResult.bSuccess);
		for (int32 i = 0; i < NumQueries; ++i)
		{
			bOk &= T.TestTrue(*FString::Printf(TEXT("reference query %d EmbedBlocking succeeded and returned one embedding"), i), R.QueryResults.IsValidIndex(i) && R.QueryResults[i].bSuccess && R.Queries[i].Num() == 1);
		}
		return bOk;
	}

	static int32 FindLineWithAny(const TArray<FString>& Lines, const TArray<FString>& Keywords)
	{
		for (int32 i = 0; i < Lines.Num(); ++i)
		{
			for (const FString& K : Keywords)
			{
				if (Lines[i].Contains(K, ESearchCase::IgnoreCase)) { return i; }
			}
		}
		return INDEX_NONE;
	}

	/** Matches: K entries, descending, same indices as TopKSimilar over the reference, scores == CosineSimilarity(query, line). */
	static void CheckMatches(FAutomationTestBase& T, const TArray<FPipelineSimilarityMatch>& Got, const FSearchRef& R, int32 QueryIdx, int32 K, const FString& Label)
	{
		const FPipelineEmbedding& Q = R.Queries[QueryIdx][0];
		const TArray<FPipelineSimilarityMatch> Want = UPipelineLibrary::TopKSimilar(Q, R.Corpus, K);
		if (!T.TestEqual(*(Label + TEXT(": number of matches")), Got.Num(), Want.Num())) { return; }
		double Worst = 0.0;
		for (int32 i = 0; i < Got.Num(); ++i)
		{
			if (i > 0)
			{
				T.TestTrue(*FString::Printf(TEXT("%s: match %d (%.5f) is not above match %d (%.5f): matches must be sorted best first"), *Label, i, Got[i].Score, i - 1, Got[i - 1].Score), Got[i].Score <= Got[i - 1].Score);
			}
			T.TestEqual(*FString::Printf(TEXT("%s: match %d corpus index"), *Label, i), Got[i].Index, Want[i].Index);
			if (R.Corpus.IsValidIndex(Got[i].Index))
			{
				const float Cos = UPipelineLibrary::CosineSimilarity(Q, R.Corpus[Got[i].Index]);
				Worst = FMath::Max(Worst, (double)FMath::Abs(Cos - Got[i].Score));
			}
			else
			{
				T.AddError(FString::Printf(TEXT("%s: match %d has corpus index %d, out of range 0..%d"), *Label, i, Got[i].Index, R.Corpus.Num() - 1));
			}
		}
		// The sample batches its corpus exactly like the reference; only float noise is allowed.
		T.TestTrue(*FString::Printf(TEXT("%s: scores equal CosineSimilarity(query, line) from EmbedBlocking; worst |diff| = %.3e (limit 1e-5)"), *Label, Worst), Worst <= 1e-5);
	}

	// ---------------------------------------------------------------- sentiment references

	static float PositiveScore(const FPipelineClassification& C, const UPipelineModelPackage* Pkg)
	{
		const int32 Idx = Pkg->Labels.IndexOfByPredicate([](const FString& L) { return L.Equals(TEXT("POSITIVE"), ESearchCase::IgnoreCase); });
		if (Idx != INDEX_NONE && C.Scores.IsValidIndex(Idx)) { return C.Scores[Idx]; }
		if (C.Scores.Num() == 2) { return C.Scores[1]; }
		return C.Score;
	}

	struct FSentimentRef
	{
		FPipelineResult Result;
		TArray<FPipelineClassification> Out;
	};

	static TSharedRef<TFuture<void>> StartSentimentRef(const TSharedRef<FFix>& F, const TArray<FString>& Messages, const TSharedRef<TArray<FSentimentRef>>& Out)
	{
		UTextClassificationPipeline* P = F->RefClass.Get();
		return RunBg([P, Messages, Out]()
		{
			Out->SetNum(Messages.Num());
			for (int32 i = 0; i < Messages.Num(); ++i)
			{
				(*Out)[i].Result = P->ClassifyBlocking(TArray<FString>{ Messages[i] }, 2, (*Out)[i].Out);
			}
		});
	}

	static bool CheckSentimentRef(FAutomationTestBase& T, const TArray<FSentimentRef>& R)
	{
		bool bOk = true;
		for (int32 i = 0; i < R.Num(); ++i)
		{
			bOk &= T.TestTrue(*FString::Printf(TEXT("reference ClassifyBlocking %d succeeded with one result"), i), R[i].Result.bSuccess && R[i].Out.Num() == 1);
		}
		return bOk;
	}

	static void CheckSentiment(FAutomationTestBase& T, const AModelPipelinesSentimentSample* S, const FSentimentRef& R, const UPipelineModelPackage* Pkg, const FString& Label)
	{
		const float WantPos = PositiveScore(R.Out[0], Pkg);
		T.TestTrue(*FString::Printf(TEXT("%s: GetLastPositiveScore %.7f must equal ClassifyBlocking's %.7f exactly"), *Label, S->GetLastPositiveScore(), WantPos), S->GetLastPositiveScore() == WantPos);
		T.TestTrue(*FString::Printf(TEXT("%s: GetLastLabel '%s' must equal ClassifyBlocking's top label '%s'"), *Label, *S->GetLastLabel(), *R.Out[0].Label), S->GetLastLabel().Equals(R.Out[0].Label, ESearchCase::IgnoreCase));
	}

	static void CheckPending(FAutomationTestBase& T, const FFix& F, const FString& Label)
	{
		T.TestTrue(*FString::Printf(TEXT("%s: the pipeline monitor found the sample's pipeline (otherwise the pending-call check is vacuous)"), *Label), F.bMonitorSawPipeline);
		T.TestTrue(*FString::Printf(TEXT("%s: GetNumPendingCalls() peaked at %d; the sample must never have more than 1 call in flight"), *Label, F.MaxPending), F.MaxPending <= 1);
	}

	static const FString PosMsg = TEXT("I love this game, best quest ever!");
	static const FString NegMsg = TEXT("This is the worst, I want a refund.");
	static const FString SwordQuery = TEXT("Where can I get my sword repaired?");
	static const FString RoomQuery = TEXT("Is there a room where I can sleep tonight?");
}

BEGIN_DEFINE_SPEC(FTextSamplesSpec, "Pipelines.Samples.Text",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	TSharedPtr<TextSamplesSpecPrivate::FFix> Fix;
END_DEFINE_SPEC(FTextSamplesSpec)

void FTextSamplesSpec::Define()
{
	using namespace TextSamplesSpecPrivate;

	AfterEach([this]()
	{
		if (Fix.IsValid())
		{
			Fix->Teardown();
			Fix.Reset();
		}
	});

	// ======================================================================================================
	Describe("Search", [this]()
	{
		It("Defaults", [this]()
		{
			// Brief: default Package path, 24 lines, TopK 3. A sample pointing at the wrong slug would tell every user to import the wrong model.
			const AModelPipelinesSemanticSearchSample* CDO = GetDefault<AModelPipelinesSemanticSearchSample>();
			TestEqual(TEXT("default Package"), CDO->Package.ToSoftObjectPath().ToString(), FString(TEXT("/Game/Models/all-MiniLM-L6-v2.all-MiniLM-L6-v2")));
			TestEqual(TEXT("default DialogueLines count"), CDO->DialogueLines.Num(), 24);
			TestEqual(TEXT("default TopK"), CDO->TopK, 3);
			TestTrue(TEXT("default bShowOverlay"), CDO->bShowOverlay);
			TestTrue(TEXT("a line about repairing/forging weapons exists in the default lines (the parity test looks it up by keyword)"), FindLineWithAny(CDO->DialogueLines, { TEXT("sword"), TEXT("repair"), TEXT("forge"), TEXT("smith"), TEXT("blade"), TEXT("weapon") }) != INDEX_NONE);
			for (int32 i = 0; i < CDO->DialogueLines.Num(); ++i)
			{
				TestFalse(*FString::Printf(TEXT("DialogueLines[%d] is not empty"), i), CDO->DialogueLines[i].TrimStartAndEnd().IsEmpty());
			}
		});

		LatentIt("CorpusReady", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// The corpus is embedded once at start: IsCorpusReady turns true, state ends Ready (not stuck Busy/Loading),
			// OnSampleResult fired exactly once (the corpus call) and succeeded.
			Fix = NewFix(*this, true, false);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSemanticSearchSample* S = Cast<AModelPipelinesSemanticSearchSample>(F->Spawn(*this, AModelPipelinesSemanticSearchSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->EmbedPkg.Get())));
			if (!S) { Done.ExecuteIfBound(); return; }
			TestFalse(TEXT("corpus is not ready synchronously inside BeginPlay (loading and embedding are async)"), S->IsCorpusReady());
			WaitFor(this, [S]() { return S->IsCorpusReady(); }, TEXT("IsCorpusReady()"), Done, [this, F, S, Done]()
			{
				Settle(SettleSeconds, [this, F, S, Done]()
				{
					TestTrue(*FString::Printf(TEXT("state after the corpus is ready is Ready, was %s (status: %s)"), *StateName(S), *S->GetStatusText()), StateIs(S, EPipelineSampleState::Ready));
					TestEqual(TEXT("OnSampleResult fired once for the corpus call"), F->Receiver->Count(), 1);
					if (F->Receiver->Count() > 0) { TestTrue(*FString::Printf(TEXT("corpus call succeeded: %s"), *F->Receiver->Results[0].Message), F->Receiver->Results[0].bSuccess); }
					TestTrue(TEXT("GetLastResult is a success"), S->GetLastResult().bSuccess);
					TestTrue(TEXT("OnSampleResult arrived on the game thread"), F->Receiver->bAllOnGameThread);
					CheckPending(*this, *F, TEXT("corpus"));
					Done.ExecuteIfBound();
				});
			});
		});

		LatentIt("Parity", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Search("Where can I get my sword repaired?") -> 3 matches, sorted, top = the blacksmith/repair line, scores equal
			// CosineSimilarity(direct EmbedBlocking of the query, direct embedding of each line).
			// A sample that re-sorts wrongly, uses dot products of unnormalized vectors, or embeds the query with a different
			// text (e.g. forgets the question mark handling) fails the index/score comparison.
			Fix = NewFix(*this, true, false);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			const TArray<FString> Lines = GetDefault<AModelPipelinesSemanticSearchSample>()->DialogueLines;
			const TSharedRef<FSearchRef> Ref = MakeShared<FSearchRef>();
			const TSharedRef<TFuture<void>> Fut = StartSearchRef(F, Lines, TArray<FString>{ SwordQuery }, Ref);
			WaitFor(this, [Fut]() { return Fut->IsReady(); }, TEXT("reference EmbedBlocking calls"), Done, [this, F, Ref, Fut, Lines, Done]()
			{
				if (!CheckSearchRef(*this, *Ref, 1)) { Done.ExecuteIfBound(); return; }
				AModelPipelinesSemanticSearchSample* S = Cast<AModelPipelinesSemanticSearchSample>(F->Spawn(*this, AModelPipelinesSemanticSearchSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->EmbedPkg.Get())));
				if (!S) { Done.ExecuteIfBound(); return; }
				WaitFor(this, [S]() { return S->IsCorpusReady() && StateIs(S, EPipelineSampleState::Ready); }, TEXT("corpus ready and state Ready"), Done, [this, F, Ref, Lines, S, Done]()
				{
					S->Search(SwordQuery);
					TestTrue(*FString::Printf(TEXT("state right after Search() is Busy (callbacks never run inside the call), was %s"), *StateName(S)), StateIs(S, EPipelineSampleState::Busy));
					WaitFor(this, [F]() { return F->Receiver->Count() >= 2; }, TEXT("OnSampleResult for the query"), Done, [this, F, Ref, Lines, S, Done]()
					{
						Settle(SettleSeconds, [this, F, Ref, Lines, S, Done]()
						{
							TestEqual(TEXT("OnSampleResult fired exactly twice: corpus + query"), F->Receiver->Count(), 2);
							TestTrue(TEXT("query call succeeded"), F->Receiver->Count() >= 2 && F->Receiver->Results[1].bSuccess);
							TestTrue(*FString::Printf(TEXT("state after the query is Ready, was %s"), *StateName(S)), StateIs(S, EPipelineSampleState::Ready));
							const TArray<FPipelineSimilarityMatch> M = S->GetLastMatches();
							CheckMatches(*this, M, *Ref, 0, 3, TEXT("sword query"));
							if (M.Num() > 0 && Lines.IsValidIndex(M[0].Index))
							{
								const TArray<FString> Keys{ TEXT("sword"), TEXT("repair"), TEXT("forge"), TEXT("smith"), TEXT("blade"), TEXT("weapon") };
								const int32 KeyLine = FindLineWithAny(Lines, Keys);
								if (KeyLine == INDEX_NONE)
								{
									AddError(TEXT("no line in DialogueLines contains sword/repair/forge/smith/blade/weapon; the blacksmith/repair line is missing from the default corpus"));
								}
								else
								{
									TestTrue(*FString::Printf(TEXT("top match must be the blacksmith/repair line, was [%d] \"%s\""), M[0].Index, *Lines[M[0].Index]), FindLineWithAny({ Lines[M[0].Index] }, Keys) != INDEX_NONE);
								}
							}
							CheckPending(*this, *F, TEXT("parity"));
							Done.ExecuteIfBound();
						});
					});
				});
			});
		});

		LatentIt("QueryBeforeCorpusReady", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Search() right after BeginPlay (state Loading, corpus not embedded): the request waits in the pending slot and is
			// served after the corpus, so there are two results (corpus, query) and the matches are the right ones.
			// A sample that drops the early query leaves GetLastMatches empty; one that runs it against an empty corpus returns nothing.
			Fix = NewFix(*this, true, false);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			const TArray<FString> Lines = GetDefault<AModelPipelinesSemanticSearchSample>()->DialogueLines;
			const TSharedRef<FSearchRef> Ref = MakeShared<FSearchRef>();
			const TSharedRef<TFuture<void>> Fut = StartSearchRef(F, Lines, TArray<FString>{ SwordQuery }, Ref);
			WaitFor(this, [Fut]() { return Fut->IsReady(); }, TEXT("reference EmbedBlocking calls"), Done, [this, F, Ref, Fut, Done]()
			{
				if (!CheckSearchRef(*this, *Ref, 1)) { Done.ExecuteIfBound(); return; }
				AModelPipelinesSemanticSearchSample* S = Cast<AModelPipelinesSemanticSearchSample>(F->Spawn(*this, AModelPipelinesSemanticSearchSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->EmbedPkg.Get())));
				if (!S) { Done.ExecuteIfBound(); return; }
				S->Search(SwordQuery);
				TestFalse(TEXT("corpus is not ready yet when the early Search() returns"), S->IsCorpusReady());
				WaitFor(this, [F]() { return F->Receiver->Count() >= 2; }, TEXT("corpus result and the early query's result (2 OnSampleResult)"), Done, [this, F, Ref, S, Done]()
				{
					Settle(SettleSeconds, [this, F, Ref, S, Done]()
					{
						TestEqual(TEXT("OnSampleResult count: corpus + the one served query"), F->Receiver->Count(), 2);
						TestTrue(TEXT("both calls succeeded"), F->Receiver->Count() >= 2 && F->Receiver->Results[0].bSuccess && F->Receiver->Results[1].bSuccess);
						TestTrue(TEXT("IsCorpusReady after the early query was served"), S->IsCorpusReady());
						CheckMatches(*this, S->GetLastMatches(), *Ref, 0, 3, TEXT("early query"));
						CheckPending(*this, *F, TEXT("early query"));
						Done.ExecuteIfBound();
					});
				});
			});
		});

		LatentIt("TwoQuickQueriesLatestWins", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Two Search() calls back to back while the sample is Busy with the corpus: one pending slot, latest wins.
			// The final matches must be the room query's, no result may arrive after it, there are at most
			// corpus + 2 results (more means calls stacked), and the pipeline never saw 2 calls in flight.
			// A sample that stacks calls reports the first query last or has GetNumPendingCalls() == 2.
			Fix = NewFix(*this, true, false);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			const TArray<FString> Lines = GetDefault<AModelPipelinesSemanticSearchSample>()->DialogueLines;
			const TSharedRef<FSearchRef> Ref = MakeShared<FSearchRef>();
			const TSharedRef<TFuture<void>> Fut = StartSearchRef(F, Lines, TArray<FString>{ SwordQuery, RoomQuery }, Ref);
			WaitFor(this, [Fut]() { return Fut->IsReady(); }, TEXT("reference EmbedBlocking calls"), Done, [this, F, Ref, Fut, Done]()
			{
				if (!CheckSearchRef(*this, *Ref, 2)) { Done.ExecuteIfBound(); return; }
				AModelPipelinesSemanticSearchSample* S = Cast<AModelPipelinesSemanticSearchSample>(F->Spawn(*this, AModelPipelinesSemanticSearchSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->EmbedPkg.Get())));
				if (!S) { Done.ExecuteIfBound(); return; }
				S->Search(SwordQuery);
				S->Search(RoomQuery);
				// Wait until the corpus is ready and the sample is idle again, then give any stray extra call time to show up.
				WaitFor(this, [F, S]() { return S->IsCorpusReady() && F->Receiver->Count() >= 2 && StateIs(S, EPipelineSampleState::Ready); }, TEXT("corpus + latest query done and state Ready"), Done, [this, F, Ref, S, Done]()
				{
					Settle(1.5, [this, F, Ref, S, Done]()
					{
						TestTrue(*FString::Printf(TEXT("OnSampleResult count must be 2 or 3 (corpus, optionally the first query, the last query), was %d"), F->Receiver->Count()), F->Receiver->Count() >= 2 && F->Receiver->Count() <= 3);
						CheckMatches(*this, S->GetLastMatches(), *Ref, 1, 3, TEXT("latest query (room)"));
						TestTrue(*FString::Printf(TEXT("state is Ready after everything settled, was %s"), *StateName(S)), StateIs(S, EPipelineSampleState::Ready));
						CheckPending(*this, *F, TEXT("two quick queries"));
						Done.ExecuteIfBound();
					});
				});
			});
		});

		LatentIt("SequentialQueriesBothServed", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// A second query after the first finished is a normal call (not swallowed by a stale pending slot):
			// results = corpus + 2, matches follow the second query.
			Fix = NewFix(*this, true, false);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			const TArray<FString> Lines = GetDefault<AModelPipelinesSemanticSearchSample>()->DialogueLines;
			const TSharedRef<FSearchRef> Ref = MakeShared<FSearchRef>();
			const TSharedRef<TFuture<void>> Fut = StartSearchRef(F, Lines, TArray<FString>{ SwordQuery, RoomQuery }, Ref);
			WaitFor(this, [Fut]() { return Fut->IsReady(); }, TEXT("reference EmbedBlocking calls"), Done, [this, F, Ref, Fut, Done]()
			{
				if (!CheckSearchRef(*this, *Ref, 2)) { Done.ExecuteIfBound(); return; }
				AModelPipelinesSemanticSearchSample* S = Cast<AModelPipelinesSemanticSearchSample>(F->Spawn(*this, AModelPipelinesSemanticSearchSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->EmbedPkg.Get())));
				if (!S) { Done.ExecuteIfBound(); return; }
				WaitFor(this, [S]() { return S->IsCorpusReady() && StateIs(S, EPipelineSampleState::Ready); }, TEXT("corpus ready"), Done, [this, F, Ref, S, Done]()
				{
					S->Search(SwordQuery);
					WaitFor(this, [F]() { return F->Receiver->Count() >= 2; }, TEXT("first query result"), Done, [this, F, Ref, S, Done]()
					{
						CheckMatches(*this, S->GetLastMatches(), *Ref, 0, 3, TEXT("first query"));
						WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready again"), Done, [this, F, Ref, S, Done]()
						{
							S->Search(RoomQuery);
							WaitFor(this, [F]() { return F->Receiver->Count() >= 3; }, TEXT("second query result"), Done, [this, F, Ref, S, Done]()
							{
								Settle(SettleSeconds, [this, F, Ref, S, Done]()
								{
									TestEqual(TEXT("OnSampleResult count: corpus + 2 queries"), F->Receiver->Count(), 3);
									CheckMatches(*this, S->GetLastMatches(), *Ref, 1, 3, TEXT("second query"));
									CheckPending(*this, *F, TEXT("sequential queries"));
									Done.ExecuteIfBound();
								});
							});
						});
					});
				});
			});
		});

		LatentIt("EmptyQueryMakesNoCall", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// "" and "   \t" -> no call (no new OnSampleResult, no Busy), status says "Type a question".
			// A sample that embeds the empty string would fire OnSampleResult and set state Busy.
			Fix = NewFix(*this, true, false);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSemanticSearchSample* S = Cast<AModelPipelinesSemanticSearchSample>(F->Spawn(*this, AModelPipelinesSemanticSearchSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->EmbedPkg.Get())));
			if (!S) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [S]() { return S->IsCorpusReady() && StateIs(S, EPipelineSampleState::Ready); }, TEXT("corpus ready"), Done, [this, F, S, Done]()
			{
				const int32 Before = F->Receiver->Count();
				S->Search(TEXT(""));
				TestTrue(*FString::Printf(TEXT("state after Search(\"\") is still Ready, was %s"), *StateName(S)), StateIs(S, EPipelineSampleState::Ready));
				TestTrue(*FString::Printf(TEXT("status after Search(\"\") says 'Type a question', was '%s'"), *S->GetStatusText()), S->GetStatusText().Contains(TEXT("Type a question")));
				S->Search(TEXT("   \t "));
				TestTrue(*FString::Printf(TEXT("state after whitespace Search is still Ready, was %s"), *StateName(S)), StateIs(S, EPipelineSampleState::Ready));
				TestTrue(*FString::Printf(TEXT("status after whitespace Search says 'Type a question', was '%s'"), *S->GetStatusText()), S->GetStatusText().Contains(TEXT("Type a question")));
				Settle(SettleSeconds, [this, F, S, Before, Done]()
				{
					TestEqual(TEXT("no OnSampleResult for empty/whitespace queries"), F->Receiver->Count(), Before);
					TestEqual(TEXT("no matches stored"), S->GetLastMatches().Num(), 0);
					Done.ExecuteIfBound();
				});
			});
		});

		LatentIt("EmptyQueryBeforeCorpusReady", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// An empty query must not occupy the pending slot either: after the corpus there are exactly 1 result and no matches.
			Fix = NewFix(*this, true, false);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSemanticSearchSample* S = Cast<AModelPipelinesSemanticSearchSample>(F->Spawn(*this, AModelPipelinesSemanticSearchSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->EmbedPkg.Get())));
			if (!S) { Done.ExecuteIfBound(); return; }
			S->Search(TEXT("  "));
			WaitFor(this, [S]() { return S->IsCorpusReady() && StateIs(S, EPipelineSampleState::Ready); }, TEXT("corpus ready"), Done, [this, F, S, Done]()
			{
				Settle(SettleSeconds, [this, F, S, Done]()
				{
					TestEqual(TEXT("only the corpus call reported; the whitespace query was never queued"), F->Receiver->Count(), 1);
					TestEqual(TEXT("no matches stored"), S->GetLastMatches().Num(), 0);
					Done.ExecuteIfBound();
				});
			});
		});
	});

	// ======================================================================================================
	Describe("Sentiment", [this]()
	{
		It("Defaults", [this]()
		{
			const AModelPipelinesSentimentSample* CDO = GetDefault<AModelPipelinesSentimentSample>();
			TestEqual(TEXT("default Package"), CDO->Package.ToSoftObjectPath().ToString(), FString(TEXT("/Game/Models/distilbert-sst2.distilbert-sst2")));
			TestTrue(TEXT("default bShowOverlay"), CDO->bShowOverlay);
		});

		LatentIt("PositiveMessage", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// "I love this game, best quest ever!" -> POSITIVE, positive score > 0.9, score == ClassifyBlocking exactly.
			// A sample that reports the top score as the positive score for a NEGATIVE message would fail the negative test;
			// one that reads the wrong label index fails here.
			Fix = NewFix(*this, false, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			const TSharedRef<TArray<FSentimentRef>> Ref = MakeShared<TArray<FSentimentRef>>();
			const TSharedRef<TFuture<void>> Fut = StartSentimentRef(F, TArray<FString>{ PosMsg }, Ref);
			WaitFor(this, [Fut]() { return Fut->IsReady(); }, TEXT("reference ClassifyBlocking"), Done, [this, F, Ref, Fut, Done]()
			{
				if (!CheckSentimentRef(*this, *Ref)) { Done.ExecuteIfBound(); return; }
				AModelPipelinesSentimentSample* S = Cast<AModelPipelinesSentimentSample>(F->Spawn(*this, AModelPipelinesSentimentSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->ClassPkg.Get())));
				if (!S) { Done.ExecuteIfBound(); return; }
				WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, Ref, S, Done]()
				{
					TestEqual(TEXT("no OnSampleResult before the first Analyze"), F->Receiver->Count(), 0);
					S->Analyze(PosMsg);
					TestTrue(*FString::Printf(TEXT("state right after Analyze() is Busy, was %s"), *StateName(S)), StateIs(S, EPipelineSampleState::Busy));
					WaitFor(this, [F]() { return F->Receiver->Count() >= 1; }, TEXT("OnSampleResult for Analyze"), Done, [this, F, Ref, S, Done]()
					{
						Settle(SettleSeconds, [this, F, Ref, S, Done]()
						{
							TestEqual(TEXT("OnSampleResult fired exactly once"), F->Receiver->Count(), 1);
							TestTrue(TEXT("the call succeeded"), F->Receiver->Count() >= 1 && F->Receiver->Results[0].bSuccess);
							TestTrue(*FString::Printf(TEXT("state back to Ready, was %s"), *StateName(S)), StateIs(S, EPipelineSampleState::Ready));
							TestTrue(*FString::Printf(TEXT("label is POSITIVE, was '%s'"), *S->GetLastLabel()), S->GetLastLabel().Equals(TEXT("POSITIVE"), ESearchCase::IgnoreCase));
							TestTrue(*FString::Printf(TEXT("positive score > 0.9, was %.5f"), S->GetLastPositiveScore()), S->GetLastPositiveScore() > 0.9f);
							CheckSentiment(*this, S, (*Ref)[0], F->ClassPkg.Get(), TEXT("positive"));
							CheckPending(*this, *F, TEXT("positive"));
							Done.ExecuteIfBound();
						});
					});
				});
			});
		});

		LatentIt("NegativeMessage", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// "This is the worst, I want a refund." -> NEGATIVE, positive score < 0.1 (NOT the top score, which is ~0.99 here).
			Fix = NewFix(*this, false, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			const TSharedRef<TArray<FSentimentRef>> Ref = MakeShared<TArray<FSentimentRef>>();
			const TSharedRef<TFuture<void>> Fut = StartSentimentRef(F, TArray<FString>{ NegMsg }, Ref);
			WaitFor(this, [Fut]() { return Fut->IsReady(); }, TEXT("reference ClassifyBlocking"), Done, [this, F, Ref, Fut, Done]()
			{
				if (!CheckSentimentRef(*this, *Ref)) { Done.ExecuteIfBound(); return; }
				AModelPipelinesSentimentSample* S = Cast<AModelPipelinesSentimentSample>(F->Spawn(*this, AModelPipelinesSentimentSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->ClassPkg.Get())));
				if (!S) { Done.ExecuteIfBound(); return; }
				WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, Ref, S, Done]()
				{
					S->Analyze(NegMsg);
					WaitFor(this, [F]() { return F->Receiver->Count() >= 1; }, TEXT("OnSampleResult for Analyze"), Done, [this, F, Ref, S, Done]()
					{
						Settle(SettleSeconds, [this, F, Ref, S, Done]()
						{
							TestEqual(TEXT("OnSampleResult fired exactly once"), F->Receiver->Count(), 1);
							TestTrue(*FString::Printf(TEXT("label is NEGATIVE, was '%s'"), *S->GetLastLabel()), S->GetLastLabel().Equals(TEXT("NEGATIVE"), ESearchCase::IgnoreCase));
							TestTrue(*FString::Printf(TEXT("positive score < 0.1, was %.5f"), S->GetLastPositiveScore()), S->GetLastPositiveScore() < 0.1f);
							CheckSentiment(*this, S, (*Ref)[0], F->ClassPkg.Get(), TEXT("negative"));
							CheckPending(*this, *F, TEXT("negative"));
							Done.ExecuteIfBound();
						});
					});
				});
			});
		});

		LatentIt("NonAsciiAndEmojiMessage", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Non-ASCII + emoji must go through the tokenizer without a failure and give the same score as the direct call.
			// A sample that narrows the text to ANSI would turn it into '?' and move the score.
			const FString Msg = FString(TEXT("Das Spiel ist großartig — とても楽しい \U0001F389\U0001F60D"));
			Fix = NewFix(*this, false, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			const TSharedRef<TArray<FSentimentRef>> Ref = MakeShared<TArray<FSentimentRef>>();
			const TSharedRef<TFuture<void>> Fut = StartSentimentRef(F, TArray<FString>{ Msg }, Ref);
			WaitFor(this, [Fut]() { return Fut->IsReady(); }, TEXT("reference ClassifyBlocking"), Done, [this, F, Ref, Fut, Msg, Done]()
			{
				if (!CheckSentimentRef(*this, *Ref)) { Done.ExecuteIfBound(); return; }
				AModelPipelinesSentimentSample* S = Cast<AModelPipelinesSentimentSample>(F->Spawn(*this, AModelPipelinesSentimentSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->ClassPkg.Get())));
				if (!S) { Done.ExecuteIfBound(); return; }
				WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, Ref, S, Msg, Done]()
				{
					S->Analyze(Msg);
					WaitFor(this, [F]() { return F->Receiver->Count() >= 1; }, TEXT("OnSampleResult for Analyze"), Done, [this, F, Ref, S, Done]()
					{
						TestTrue(*FString::Printf(TEXT("call succeeded: %s"), *F->Receiver->Results[0].Message), F->Receiver->Results[0].bSuccess);
						CheckSentiment(*this, S, (*Ref)[0], F->ClassPkg.Get(), TEXT("non-ASCII"));
						TestTrue(TEXT("positive score is within 0..1"), S->GetLastPositiveScore() >= 0.f && S->GetLastPositiveScore() <= 1.f);
						Done.ExecuteIfBound();
					});
				});
			});
		});

		LatentIt("TwoQuickMessagesLatestWins", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Analyze(positive) then Analyze(negative) in the same frame: the second waits in the single pending slot (latest wins),
			// the final state is the NEGATIVE message's, at most 2 results arrive, never 2 calls in flight.
			// A sample that lets the first result land last would show POSITIVE.
			Fix = NewFix(*this, false, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			const TSharedRef<TArray<FSentimentRef>> Ref = MakeShared<TArray<FSentimentRef>>();
			const TSharedRef<TFuture<void>> Fut = StartSentimentRef(F, TArray<FString>{ PosMsg, NegMsg }, Ref);
			WaitFor(this, [Fut]() { return Fut->IsReady(); }, TEXT("reference ClassifyBlocking"), Done, [this, F, Ref, Fut, Done]()
			{
				if (!CheckSentimentRef(*this, *Ref)) { Done.ExecuteIfBound(); return; }
				AModelPipelinesSentimentSample* S = Cast<AModelPipelinesSentimentSample>(F->Spawn(*this, AModelPipelinesSentimentSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->ClassPkg.Get())));
				if (!S) { Done.ExecuteIfBound(); return; }
				WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, Ref, S, Done]()
				{
					S->Analyze(PosMsg);
					S->Analyze(NegMsg);
					TestTrue(*FString::Printf(TEXT("state is Busy with a request pending, was %s"), *StateName(S)), StateIs(S, EPipelineSampleState::Busy));
					WaitFor(this, [F, S]() { return F->Receiver->Count() >= 1 && StateIs(S, EPipelineSampleState::Ready); }, TEXT("results and state Ready"), Done, [this, F, Ref, S, Done]()
					{
						Settle(1.5, [this, F, Ref, S, Done]()
						{
							TestTrue(*FString::Printf(TEXT("1 or 2 OnSampleResult (never more), was %d"), F->Receiver->Count()), F->Receiver->Count() >= 1 && F->Receiver->Count() <= 2);
							TestTrue(*FString::Printf(TEXT("the last result is the negative message's, label was '%s'"), *S->GetLastLabel()), S->GetLastLabel().Equals(TEXT("NEGATIVE"), ESearchCase::IgnoreCase));
							CheckSentiment(*this, S, (*Ref)[1], F->ClassPkg.Get(), TEXT("latest wins"));
							CheckPending(*this, *F, TEXT("two quick messages"));
							Done.ExecuteIfBound();
						});
					});
				});
			});
		});

		LatentIt("EmptyMessageMakesNoCall", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// "" and whitespace -> no call, status "Type a message", state stays Ready, no OnSampleResult.
			Fix = NewFix(*this, false, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSentimentSample* S = Cast<AModelPipelinesSentimentSample>(F->Spawn(*this, AModelPipelinesSentimentSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->ClassPkg.Get())));
			if (!S) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, S, Done]()
			{
				S->Analyze(TEXT(""));
				TestTrue(*FString::Printf(TEXT("state after Analyze(\"\") is Ready, was %s"), *StateName(S)), StateIs(S, EPipelineSampleState::Ready));
				TestTrue(*FString::Printf(TEXT("status says 'Type a message', was '%s'"), *S->GetStatusText()), S->GetStatusText().Contains(TEXT("Type a message")));
				S->Analyze(TEXT(" \t  "));
				TestTrue(*FString::Printf(TEXT("state after whitespace Analyze is Ready, was %s"), *StateName(S)), StateIs(S, EPipelineSampleState::Ready));
				TestTrue(*FString::Printf(TEXT("status after whitespace says 'Type a message', was '%s'"), *S->GetStatusText()), S->GetStatusText().Contains(TEXT("Type a message")));
				Settle(SettleSeconds, [this, F, Done]()
				{
					TestEqual(TEXT("no OnSampleResult for empty/whitespace messages"), F->Receiver->Count(), 0);
					Done.ExecuteIfBound();
				});
			});
		});
	});

	// ======================================================================================================
	Describe("Package", [this]()
	{
		// Shared body: spawn with the given soft package, wait for the state to leave Loading, run Check, then poke the sample
		// once more (Search/Analyze) and confirm nothing ran.
		auto RunBad = [this](const FDoneDelegate& Done, bool bSearch, bool bEmbedPkgForWrongTask, const FSoftObjectPath* Path, bool bWrongTask, TFunction<void(AModelPipelinesSampleBase*)> Check)
		{
			Fix = NewFix(*this, bWrongTask && bEmbedPkgForWrongTask, bWrongTask && !bEmbedPkgForWrongTask);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			TSoftObjectPtr<UPipelineModelPackage> Soft;
			if (Path) { Soft = TSoftObjectPtr<UPipelineModelPackage>(*Path); }
			else if (bWrongTask) { Soft = TSoftObjectPtr<UPipelineModelPackage>(bEmbedPkgForWrongTask ? F->EmbedPkg.Get() : F->ClassPkg.Get()); }
			UClass* Cls = bSearch ? AModelPipelinesSemanticSearchSample::StaticClass() : AModelPipelinesSentimentSample::StaticClass();
			AModelPipelinesSampleBase* A = F->Spawn(*this, Cls, Soft);
			if (!A) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [A]() { return !StateIs(A, EPipelineSampleState::Loading); }, TEXT("state to leave Loading"), Done, [this, F, A, bSearch, Check, Done]()
			{
				Settle(SettleSeconds, [this, F, A, bSearch, Check, Done]()
				{
					Check(A);
					TestEqual(TEXT("no OnSampleResult: no call was made"), F->Receiver->Count(), 0);
					TestTrue(TEXT("no pipeline was created"), FindSamplePipeline(A) == nullptr);
					// A request on a sample without a pipeline must be a harmless no-op.
					if (bSearch) { Cast<AModelPipelinesSemanticSearchSample>(A)->Search(TEXT("Where can I get my sword repaired?")); }
					else { Cast<AModelPipelinesSentimentSample>(A)->Analyze(TEXT("great")); }
					Settle(SettleSeconds, [this, F, A, Done]()
					{
						TestEqual(TEXT("still no OnSampleResult after a request on the unusable sample"), F->Receiver->Count(), 0);
						TestFalse(TEXT("state is not Busy after a request on the unusable sample"), StateIs(A, EPipelineSampleState::Busy));
						Done.ExecuteIfBound();
					});
				});
			});
		};

		LatentIt("MissingPackageSearch", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			// Path that does not resolve -> MissingPackage + the exact text from the brief (with the all-MiniLM-L6-v2 slug), a Warning, no call.
			const FSoftObjectPath Path(TEXT("/Game/Models/does-not-exist.does-not-exist"));
			ExpectWarning(*this);
			RunBad(Done, true, false, &Path, false, [this](AModelPipelinesSampleBase* A)
			{
				TestTrue(*FString::Printf(TEXT("state is MissingPackage, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::MissingPackage));
				TestEqual(TEXT("status text"), A->GetStatusText(), MissingText(TEXT("/Game/Models/does-not-exist.does-not-exist"), SearchSlug));
			});
		});

		LatentIt("MissingPackageSentiment", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			const FSoftObjectPath Path(TEXT("/Game/Models/does-not-exist.does-not-exist"));
			ExpectWarning(*this);
			RunBad(Done, false, false, &Path, false, [this](AModelPipelinesSampleBase* A)
			{
				TestTrue(*FString::Printf(TEXT("state is MissingPackage, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::MissingPackage));
				TestEqual(TEXT("status text"), A->GetStatusText(), MissingText(TEXT("/Game/Models/does-not-exist.does-not-exist"), SentimentSlug));
			});
		});

		LatentIt("UnsetPackageSearch", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			// Package = null -> same message with the path "(none)". A sample that prints an empty path or crashes on null fails here.
			ExpectWarning(*this);
			RunBad(Done, true, false, nullptr, false, [this](AModelPipelinesSampleBase* A)
			{
				TestTrue(*FString::Printf(TEXT("state is MissingPackage, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::MissingPackage));
				TestEqual(TEXT("status text"), A->GetStatusText(), MissingText(TEXT("(none)"), SearchSlug));
			});
		});

		LatentIt("UnsetPackageSentiment", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			ExpectWarning(*this);
			RunBad(Done, false, false, nullptr, false, [this](AModelPipelinesSampleBase* A)
			{
				TestTrue(*FString::Printf(TEXT("state is MissingPackage, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::MissingPackage));
				TestEqual(TEXT("status text"), A->GetStatusText(), MissingText(TEXT("(none)"), SentimentSlug));
			});
		});

		LatentIt("WrongTaskSentiment", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			// An embedding package on the sentiment sample: the creator rejects it -> Failed, "Could not create the pipeline: <message>".
			ExpectWrongTask(*this);
			RunBad(Done, false, /*bEmbedPkgForWrongTask*/ true, nullptr, /*bWrongTask*/ true, [this](AModelPipelinesSampleBase* A)
			{
				TestTrue(*FString::Printf(TEXT("state is Failed, was %s (status: %s)"), *StateName(A), *A->GetStatusText()), StateIs(A, EPipelineSampleState::Failed));
				const FString Status = A->GetStatusText();
				TestTrue(*FString::Printf(TEXT("status starts with 'Could not create the pipeline: ', was '%s'"), *Status), Status.StartsWith(TEXT("Could not create the pipeline: ")));
				TestTrue(TEXT("the status carries a non-empty creator message"), Status.Len() > FString(TEXT("Could not create the pipeline: ")).Len());
			});
		});

		LatentIt("WrongTaskSearch", LatentLimit, [this, RunBad](const FDoneDelegate& Done)
		{
			// A classification package on the semantic-search sample -> Failed, same prefix.
			ExpectWrongTask(*this);
			RunBad(Done, true, /*bEmbedPkgForWrongTask*/ false, nullptr, /*bWrongTask*/ true, [this](AModelPipelinesSampleBase* A)
			{
				TestTrue(*FString::Printf(TEXT("state is Failed, was %s (status: %s)"), *StateName(A), *A->GetStatusText()), StateIs(A, EPipelineSampleState::Failed));
				TestTrue(*FString::Printf(TEXT("status starts with 'Could not create the pipeline: ', was '%s'"), *A->GetStatusText()), A->GetStatusText().StartsWith(TEXT("Could not create the pipeline: ")));
			});
		});
	});

	// ======================================================================================================
	Describe("Lifetime", [this]()
	{
		LatentIt("DestroyedWhileAnalyzing", LatentLimit, [this](const FDoneDelegate& Done)
		{
			ExpectWorldNoContext(*this);
			// Analyze, then destroy the actor in the same frame (the call is in flight), force a GC, and let the result arrive:
			// no crash and no OnSampleResult afterwards (rule 4, weak this).
			Fix = NewFix(*this, false, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSentimentSample* S = Cast<AModelPipelinesSentimentSample>(F->Spawn(*this, AModelPipelinesSentimentSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->ClassPkg.Get())));
			if (!S) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, S, Done]()
			{
				S->Analyze(PosMsg);
				TestTrue(TEXT("a call is in flight (state Busy) before the actor is destroyed"), StateIs(S, EPipelineSampleState::Busy));
				S->Destroy();
				Settle(0.05, [this, F, Done]()
				{
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					Settle(4.0, [this, F, Done]()
					{
						TestEqual(TEXT("no OnSampleResult after the actor was destroyed"), F->Receiver->Count(), 0);
						TestFalse(TEXT("the actor is gone"), F->Actor.IsValid());
						Done.ExecuteIfBound();
					});
				});
			});
		});

		LatentIt("DestroyedDuringCorpusEmbedding", LatentLimit, [this](const FDoneDelegate& Done)
		{
			ExpectWorldNoContext(*this);
			// Destroy the search sample while the corpus is still being embedded: no crash, no OnSampleResult, no second call.
			Fix = NewFix(*this, true, false);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSemanticSearchSample* S = Cast<AModelPipelinesSemanticSearchSample>(F->Spawn(*this, AModelPipelinesSemanticSearchSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->EmbedPkg.Get())));
			if (!S) { Done.ExecuteIfBound(); return; }
			S->Search(SwordQuery);
			// Let the async load resolve and the corpus call start (state Busy) but not finish, then destroy.
			WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Busy) || S->IsCorpusReady(); }, TEXT("corpus call started"), Done, [this, F, S, Done]()
			{
				const bool bInFlight = !S->IsCorpusReady();
				if (!bInFlight) { AddInfo(TEXT("corpus finished before the test could destroy the actor; destroying anyway (weaker check)")); }
				const int32 Before = F->Receiver->Count();
				S->Destroy();
				Settle(0.05, [this, F, Before, Done]()
				{
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					Settle(4.0, [this, F, Before, Done]()
					{
						TestEqual(TEXT("no OnSampleResult after the actor was destroyed"), F->Receiver->Count(), Before);
						Done.ExecuteIfBound();
					});
				});
			});
		});

		LatentIt("DestroyedWhileLoading", LatentLimit, [this](const FDoneDelegate& Done)
		{
			ExpectWorldNoContext(*this);
			// Destroy in the same frame as BeginPlay (async package load still pending): the load completion must not touch the dead actor.
			Fix = NewFix(*this, false, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSentimentSample* S = Cast<AModelPipelinesSentimentSample>(F->Spawn(*this, AModelPipelinesSentimentSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->ClassPkg.Get())));
			if (!S) { Done.ExecuteIfBound(); return; }
			S->Destroy();
			Settle(0.05, [this, F, Done]()
			{
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
				Settle(2.0, [this, F, Done]()
				{
					TestEqual(TEXT("no OnSampleResult"), F->Receiver->Count(), 0);
					Done.ExecuteIfBound();
				});
			});
		});
	});

	// ======================================================================================================
	Describe("Overlay", [this]()
	{
		LatentIt("ShowOverlayWithoutViewportSentiment", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// bShowOverlay = true in a world with no game viewport: no crash, no overlay, and the sample still works (reaches Ready, analyzes).
			Fix = NewFix(*this, false, true);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSentimentSample* S = Cast<AModelPipelinesSentimentSample>(F->Spawn(*this, AModelPipelinesSentimentSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->ClassPkg.Get()), /*bOverlay*/ true));
			if (!S) { Done.ExecuteIfBound(); return; }
			TestTrue(TEXT("bShowOverlay stayed true"), S->bShowOverlay);
			WaitFor(this, [S]() { return StateIs(S, EPipelineSampleState::Ready); }, TEXT("state Ready"), Done, [this, F, S, Done]()
			{
				S->Analyze(PosMsg);
				WaitFor(this, [F]() { return F->Receiver->Count() >= 1; }, TEXT("OnSampleResult"), Done, [this, F, S, Done]()
				{
					TestTrue(TEXT("Analyze succeeded without an overlay"), F->Receiver->Results[0].bSuccess);
					TestTrue(*FString::Printf(TEXT("label is POSITIVE, was '%s'"), *S->GetLastLabel()), S->GetLastLabel().Equals(TEXT("POSITIVE"), ESearchCase::IgnoreCase));
					Done.ExecuteIfBound();
				});
			});
		});

		LatentIt("ShowOverlayWithoutViewportSearch", LatentLimit, [this](const FDoneDelegate& Done)
		{
			Fix = NewFix(*this, true, false);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSemanticSearchSample* S = Cast<AModelPipelinesSemanticSearchSample>(F->Spawn(*this, AModelPipelinesSemanticSearchSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(F->EmbedPkg.Get()), /*bOverlay*/ true));
			if (!S) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [S]() { return S->IsCorpusReady() && StateIs(S, EPipelineSampleState::Ready); }, TEXT("corpus ready"), Done, [this, S, Done]()
			{
				TestEqual(TEXT("default TopK matches"), S->TopK, 3);
				Done.ExecuteIfBound();
			});
		});

		LatentIt("MissingPackageWithOverlayRequested", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// The error path with bShowOverlay = true and no viewport: still no crash.
			ExpectWarning(*this);
			Fix = NewFix(*this, false, false);
			if (!Fix.IsValid()) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFix> F = Fix.ToSharedRef();
			AModelPipelinesSampleBase* A = F->Spawn(*this, AModelPipelinesSentimentSample::StaticClass(), TSoftObjectPtr<UPipelineModelPackage>(), /*bOverlay*/ true);
			if (!A) { Done.ExecuteIfBound(); return; }
			WaitFor(this, [A]() { return !StateIs(A, EPipelineSampleState::Loading); }, TEXT("state to leave Loading"), Done, [this, A, Done]()
			{
				TestTrue(*FString::Printf(TEXT("state is MissingPackage, was %s"), *StateName(A)), StateIs(A, EPipelineSampleState::MissingPackage));
				Done.ExecuteIfBound();
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
