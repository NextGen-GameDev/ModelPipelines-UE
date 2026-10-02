// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/ParallelFor.h"
#include "Containers/Array.h"
#include "HAL/ThreadSafeCounter.h"
#include "Math/RandomStream.h"
#include "Math/UnrealMathUtility.h"
#include "PipelineLibrary.h"
#include "PipelineResults.h"

namespace PipelineLibrarySpecPrivate
{
	static FPipelineEmbedding Emb(std::initializer_list<float> L)
	{
		FPipelineEmbedding E;
		E.Values = TArray<float>(L);
		return E;
	}

	static bool NearF(float A, double E) { return FMath::Abs((double)A - E) <= 1e-6 * FMath::Max(1.0, FMath::Abs(E)); }

	/** The brief gives no Warning wording, so any single Warning-level message satisfies the expectation. */
	static void ExpectOneWarning(FAutomationTestBase& T)
	{
		T.AddExpectedMessage(TEXT("."), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
	}

	static FString Summarize(const TArray<FPipelineSimilarityMatch>& R)
	{
		FString S;
		for (const FPipelineSimilarityMatch& M : R) { S += FString::Printf(TEXT("[%d:%.7f]"), M.Index, M.Score); }
		return S;
	}
}

BEGIN_DEFINE_SPEC(FPipelineLibrarySpec, "Pipelines.Core.Library",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FPipelineLibrarySpec)

void FPipelineLibrarySpec::Define()
{
	using namespace PipelineLibrarySpecPrivate;

	Describe("CosineSimilarity", [this]()
	{
		It("identical vectors give 1", [this]()
		{
			const float S = UPipelineLibrary::CosineSimilarity(Emb({1.f, 0.f}), Emb({1.f, 0.f}));
			TestTrue(*FString::Printf(TEXT("got %.8f"), S), NearF(S, 1.0));
		});

		It("opposite vectors give -1", [this]()
		{
			const float S = UPipelineLibrary::CosineSimilarity(Emb({1.f, 2.f}), Emb({-1.f, -2.f}));
			TestTrue(*FString::Printf(TEXT("got %.8f"), S), NearF(S, -1.0));
		});

		It("length mismatch gives 0 with exactly one Warning", [this]()
		{
			ExpectOneWarning(*this);
			const float S = UPipelineLibrary::CosineSimilarity(Emb({1.f, 0.f}), Emb({1.f, 0.f, 0.f}));
			TestEqual("score", S, 0.f);
		});

		It("empty embedding gives 0 with exactly one Warning", [this]()
		{
			ExpectOneWarning(*this);
			const float S = UPipelineLibrary::CosineSimilarity(FPipelineEmbedding(), Emb({1.f, 0.f}));
			TestEqual("score", S, 0.f);
		});
	});

	Describe("TopKSimilar", [this]()
	{
		It("ranks the corpus best first: [{1,1.0},{2,0.7071068}]", [this]()
		{
			const TArray<FPipelineEmbedding> Corpus = {Emb({0.f, 1.f}), Emb({1.f, 0.f}), Emb({1.f, 1.f}), Emb({-1.f, 0.f})};
			const TArray<FPipelineSimilarityMatch> R = UPipelineLibrary::TopKSimilar(Emb({1.f, 0.f}), Corpus, 2);
			TestEqual("count", R.Num(), 2);
			if (R.Num() != 2) { return; }
			TestEqual("first index", R[0].Index, 1);
			TestTrue(*Summarize(R), NearF(R[0].Score, 1.0));
			TestEqual("second index", R[1].Index, 2);
			TestTrue(*Summarize(R), NearF(R[1].Score, 0.70710678118654752));
		});

		It("ties go to the lower index: [0,1,2]", [this]()
		{
			// {1,0} and {2,0} both score exactly 1; an unstable sort could return [1,0,2].
			const TArray<FPipelineEmbedding> Corpus = {Emb({1.f, 0.f}), Emb({2.f, 0.f}), Emb({0.f, 1.f})};
			const TArray<FPipelineSimilarityMatch> R = UPipelineLibrary::TopKSimilar(Emb({1.f, 0.f}), Corpus, 3);
			TestEqual("count", R.Num(), 3);
			if (R.Num() != 3) { return; }
			TestEqual("idx0", R[0].Index, 0);
			TestEqual("idx1", R[1].Index, 1);
			TestEqual("idx2", R[2].Index, 2);
		});

		It("K <= 0, empty corpus and empty query give empty", [this]()
		{
			const TArray<FPipelineEmbedding> Corpus = {Emb({1.f, 0.f}), Emb({0.f, 1.f})};
			TestEqual("K=0", UPipelineLibrary::TopKSimilar(Emb({1.f, 0.f}), Corpus, 0).Num(), 0);
			TestEqual("K=-1", UPipelineLibrary::TopKSimilar(Emb({1.f, 0.f}), Corpus, -1).Num(), 0);
			TestEqual("empty corpus", UPipelineLibrary::TopKSimilar(Emb({1.f, 0.f}), TArray<FPipelineEmbedding>(), 3).Num(), 0);
			TestEqual("empty query", UPipelineLibrary::TopKSimilar(FPipelineEmbedding(), Corpus, 3).Num(), 0);
		});

		It("K larger than the corpus returns every entry", [this]()
		{
			const TArray<FPipelineEmbedding> Corpus = {Emb({1.f, 0.f}), Emb({0.f, 1.f}), Emb({1.f, 1.f})};
			TestEqual("count", UPipelineLibrary::TopKSimilar(Emb({1.f, 0.f}), Corpus, 10).Num(), 3);
		});

		It("skips entries of a different length with one Warning", [this]()
		{
			// Brief: "one Warning naming the count" but no wording, so only the Warning count is checked.
			ExpectOneWarning(*this);
			const TArray<FPipelineEmbedding> Corpus = {Emb({0.f, 1.f}), Emb({1.f, 0.f, 0.f}), Emb({1.f, 0.f}), Emb({1.f, 1.f})};
			const TArray<FPipelineSimilarityMatch> R = UPipelineLibrary::TopKSimilar(Emb({1.f, 0.f}), Corpus, 10);
			TestEqual("usable entries only", R.Num(), 3);
			for (const FPipelineSimilarityMatch& M : R)
			{
				TestNotEqual("the 3-dim entry (index 1) is never returned", M.Index, 1);
			}
			if (R.Num() == 3)
			{
				TestEqual("best", R[0].Index, 2);
				TestEqual("second", R[1].Index, 3);
				TestEqual("third", R[2].Index, 0);
			}
		});

		It("8 threads on a 1000x384 corpus all get identical results", [this]()
		{
			FRandomStream Rng(777);
			TArray<FPipelineEmbedding> Corpus;
			Corpus.SetNum(1000);
			for (FPipelineEmbedding& E : Corpus)
			{
				E.Values.SetNumUninitialized(384);
				for (float& V : E.Values) { V = Rng.FRandRange(-1.f, 1.f); }
			}
			FPipelineEmbedding Query;
			Query.Values.SetNumUninitialized(384);
			for (float& V : Query.Values) { V = Rng.FRandRange(-1.f, 1.f); }

			const TArray<FPipelineSimilarityMatch> Reference = UPipelineLibrary::TopKSimilar(Query, Corpus, 10);
			TestEqual("reference size", Reference.Num(), 10);

			FThreadSafeCounter Mismatches;
			ParallelFor(8, [&](int32)
			{
				for (int32 Iter = 0; Iter < 5; ++Iter)
				{
					const TArray<FPipelineSimilarityMatch> R = UPipelineLibrary::TopKSimilar(Query, Corpus, 10);
					bool bSame = R.Num() == Reference.Num();
					for (int32 i = 0; bSame && i < R.Num(); ++i)
					{
						bSame = R[i].Index == Reference[i].Index && R[i].Score == Reference[i].Score;
					}
					if (!bSame) { Mismatches.Increment(); }
				}
			});
			TestEqual("mismatching runs", Mismatches.GetValue(), 0);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
