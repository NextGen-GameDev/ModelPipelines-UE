// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP7.11 / contract C43: calls with more than 16 texts run as sequential chunks of 16. Everything the caller can see must be
// unchanged: results in input order equal to the goldens, callback exactly once on the game thread and never inside the call,
// GetNumPendingCalls() >= 1 until the final callback, timings summed over chunks, first error ends the call with empty outputs.
// Expected values come only from the goldens (Python/torch) or from the same pipeline's own results, never from chunking internals.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineModelPackage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "TextClassificationPipeline.h"
#include "TextEmbeddingPipeline.h"
#include "TextPipelineLibrary.h"
#include "PipelineTextTestData.h"

using namespace UE::ModelPipelines::TextTests;

namespace TextChunkingSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(300);
	constexpr double CallTimeoutSeconds = 200.0;
	const TCHAR* const MiniSlug = TEXT("all-MiniLM-L6-v2");
	const TCHAR* const SstSlug = TEXT("distilbert-sst2");

	static TSharedPtr<FJsonObject> GCachedRoot;

	static TSharedPtr<FJsonObject> GetGoldens(FAutomationTestBase& T)
	{
		if (!GCachedRoot.IsValid())
		{
			FString Err;
			GCachedRoot = LoadTextGoldens(Err); // Err names the path and the generate command
			if (!GCachedRoot.IsValid()) { T.AddError(Err); }
		}
		return GCachedRoot;
	}

	struct FGoldenCls
	{
		FString Label;
		int32 Index = -1;
		TArray<double> Scores;
	};

	struct FFixture
	{
		TSharedPtr<FJsonObject> Root;
		TSharedPtr<FJsonObject> Model;
		TStrongObjectPtr<UPipelineModelPackage> Package;
		TStrongObjectPtr<UTextEmbeddingPipeline> Embedder;
		TStrongObjectPtr<UTextClassificationPipeline> Classifier;
		TArray<FString> Ids;
		TArray<FString> Texts;
		TMap<FString, TArray<double>> GoldenEmbedding;
		TMap<FString, FGoldenCls> GoldenCls;
	};

	/** Loads goldens + sentences and a CPU package; creates the embedding (bEmbedding) or the classification pipeline. */
	static bool MakeFixture(FAutomationTestBase& T, bool bEmbedding, FFixture& F, TFunction<void(UPipelineModelPackage&)> Mutate = nullptr)
	{
		F.Root = GetGoldens(T);
		if (!F.Root.IsValid()) { return false; }
		const TCHAR* Slug = bEmbedding ? MiniSlug : SstSlug;
		F.Model = FindModel(F.Root, Slug);
		if (!F.Model.IsValid())
		{
			T.AddError(FString::Printf(TEXT("Golden model '%s' not found in text_goldens.json. Regenerate with Tools/make_text_goldens.py."), Slug));
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>* Sentences = nullptr;
		if (!F.Root->TryGetArrayField(TEXT("sentences"), Sentences) || Sentences->Num() != 50)
		{
			T.AddError(TEXT("Goldens: expected 50 sentences."));
			return false;
		}
		for (const TSharedPtr<FJsonValue>& V : *Sentences)
		{
			const TSharedPtr<FJsonObject> S = V->AsObject();
			F.Ids.Add(S->GetStringField(TEXT("id")));
			F.Texts.Add(S->GetStringField(TEXT("text")));
		}
		const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
		if (F.Model->TryGetArrayField(TEXT("cases"), Cases))
		{
			for (const TSharedPtr<FJsonValue>& V : *Cases)
			{
				const TSharedPtr<FJsonObject> C = V->AsObject();
				const FString Id = C->GetStringField(TEXT("sentence_id"));
				const TArray<TSharedPtr<FJsonValue>>* Emb = nullptr;
				if (C->TryGetArrayField(TEXT("embedding"), Emb))
				{
					TArray<double>& Dst = F.GoldenEmbedding.Add(Id);
					for (const TSharedPtr<FJsonValue>& X : *Emb) { Dst.Add(X->AsNumber()); }
				}
				const TArray<TSharedPtr<FJsonValue>>* Sc = nullptr;
				if (C->TryGetArrayField(TEXT("scores"), Sc))
				{
					FGoldenCls& G = F.GoldenCls.Add(Id);
					G.Label = C->GetStringField(TEXT("label"));
					G.Index = (int32)C->GetNumberField(TEXT("index"));
					for (const TSharedPtr<FJsonValue>& X : *Sc) { G.Scores.Add(X->AsNumber()); }
				}
			}
		}
		FString Err;
		UPipelineModelPackage* Pkg = MakePackage(F.Root, F.Model, EPipelineDevice::CPU, Err);
		if (!Pkg) { T.AddError(Err); return false; }
		F.Package.Reset(Pkg);
		if (Mutate) { Mutate(*Pkg); }
		FPipelineResult R;
		if (bEmbedding)
		{
			UTextEmbeddingPipeline* P = UTextPipelineLibrary::CreateTextEmbeddingPipeline(Pkg, R);
			if (!P) { T.AddError(FString::Printf(TEXT("CreateTextEmbeddingPipeline failed: %s"), *R.Message)); return false; }
			F.Embedder.Reset(P);
		}
		else
		{
			UTextClassificationPipeline* P = UTextPipelineLibrary::CreateTextClassificationPipeline(Pkg, R);
			if (!P) { T.AddError(FString::Printf(TEXT("CreateTextClassificationPipeline failed: %s"), *R.Message)); return false; }
			F.Classifier.Reset(P);
		}
		return true;
	}

	/** Sentence indices Start .. Start+N-1 (wrapping over the 50 sentences). */
	static TArray<int32> Range(int32 N, int32 Start = 0)
	{
		TArray<int32> I;
		for (int32 k = 0; k < N; ++k) { I.Add((Start + k) % 50); }
		return I;
	}

	static TArray<FString> TextsOf(const FFixture& F, const TArray<int32>& Indices)
	{
		TArray<FString> Out;
		for (int32 I : Indices) { Out.Add(F.Texts[I]); }
		return Out;
	}

	static double Cosine(const TArray<float>& A, const TArray<double>& B)
	{
		if (A.Num() != B.Num() || A.Num() == 0) { return -2.0; }
		double Dot = 0, NA = 0, NB = 0;
		for (int32 i = 0; i < A.Num(); ++i) { Dot += (double)A[i] * B[i]; NA += (double)A[i] * A[i]; NB += B[i] * B[i]; }
		return (NA > 0 && NB > 0) ? Dot / (FMath::Sqrt(NA) * FMath::Sqrt(NB)) : -2.0;
	}

	static double CosineF(const TArray<float>& A, const TArray<float>& B)
	{
		TArray<double> D;
		for (float X : B) { D.Add(X); }
		return Cosine(A, D);
	}

	/** Embedding k must match the golden of sentence Indices[k] (cosine > MinCos). A chunk concatenated in the wrong order, a
	    chunk dropped or repeated, or padding of another chunk leaking in would make one of them miss. Reports the worst. */
	static void CheckEmbeddings(FAutomationTestBase& T, const FFixture& F, const TArray<FPipelineEmbedding>& Embs, const TArray<int32>& Indices, const TCHAR* Label)
	{
		if (!T.TestEqual(*FString::Printf(TEXT("%s: embedding count"), Label), Embs.Num(), Indices.Num())) { return; }
		double Worst = 2.0;
		FString WorstId;
		for (int32 k = 0; k < Indices.Num(); ++k)
		{
			const FString& Id = F.Ids[Indices[k]];
			const TArray<double>* G = F.GoldenEmbedding.Find(Id);
			if (!G) { T.AddError(FString::Printf(TEXT("%s: no golden for %s"), Label, *Id)); continue; }
			const double C = Cosine(Embs[k].Values, *G);
			if (C < Worst) { Worst = C; WorstId = FString::Printf(TEXT("%s at output position %d"), *Id, k); }
		}
		T.AddInfo(FString::Printf(TEXT("%s: worst cosine vs golden = %.8f (%s)"), Label, Worst, *WorstId));
		T.TestTrue(*FString::Printf(TEXT("%s: every cosine > 0.999 (worst %.8f, %s)"), Label, Worst, *WorstId), Worst > 0.999);
	}

	/** Same for classification: scores within 1e-3 of the golden, label and index equal. */
	static void CheckClassifications(FAutomationTestBase& T, const FFixture& F, const TArray<FPipelineClassification>& Res, const TArray<int32>& Indices, const TCHAR* Label)
	{
		if (!T.TestEqual(*FString::Printf(TEXT("%s: result count"), Label), Res.Num(), Indices.Num())) { return; }
		double Worst = 0.0;
		FString WorstId;
		int32 LabelBad = 0, ScoreBad = 0;
		FString FirstBad;
		for (int32 k = 0; k < Indices.Num(); ++k)
		{
			const FString& Id = F.Ids[Indices[k]];
			const FGoldenCls* G = F.GoldenCls.Find(Id);
			if (!G) { T.AddError(FString::Printf(TEXT("%s: no golden for %s"), Label, *Id)); continue; }
			if (Res[k].Scores.Num() != G->Scores.Num())
			{
				T.AddError(FString::Printf(TEXT("%s: %s has %d scores, golden %d"), Label, *Id, Res[k].Scores.Num(), G->Scores.Num()));
				continue;
			}
			for (int32 c = 0; c < G->Scores.Num(); ++c)
			{
				const double D = FMath::Abs((double)Res[k].Scores[c] - G->Scores[c]);
				if (D > Worst) { Worst = D; WorstId = FString::Printf(TEXT("%s[%d] at output position %d"), *Id, c, k); }
				if (D > 1e-3) { ++ScoreBad; }
			}
			if (Res[k].Label != G->Label || Res[k].Index != G->Index)
			{
				if (LabelBad++ == 0) { FirstBad = FString::Printf(TEXT("%s at position %d: got '%s' golden '%s'"), *Id, k, *Res[k].Label, *G->Label); }
			}
		}
		T.AddInfo(FString::Printf(TEXT("%s: max score diff vs golden = %.3g (%s)"), Label, Worst, *WorstId));
		T.TestEqual(*FString::Printf(TEXT("%s: scores off by > 1e-3 (max %.3g at %s)"), Label, Worst, *WorstId), ScoreBad, 0);
		T.TestEqual(*FString::Printf(TEXT("%s: label mismatches (first: %s)"), Label, *FirstBad), LabelBad, 0);
	}

	template <typename TOut>
	struct TCall
	{
		FThreadSafeCounter Count;
		bool bInCall = false;
		bool bAnyInCall = false;
		bool bAllOnGameThread = true;
		FPipelineResult Result;
		TArray<TOut> Out;
		int32 MinPendingBeforeCallback = MAX_int32; // sampled on every tick while no callback has fired
		int32 MaxPendingBeforeCallback = 0;
		int32 PendingSamples = 0;
		int32 PendingAfterIssue = -1;
	};

	struct FRunOptions
	{
		float TickSeconds = 0.02f;
		double PostCallbackWait = 0.3;   // keep watching this long after the first callback to catch a second one
		double MaxWait = CallTimeoutSeconds;
		bool bNoCallbackOk = false;      // timeout without a callback is reported as info and Check still runs
	};

	/**
	 * Issue(callback) must start the async call. The generic assertions (exactly once, game thread, not inside the call) run
	 * before Check. GetPending (optional) is sampled on every tick until the first callback. Never blocks; fails with a message on timeout.
	 */
	template <typename TOut>
	static void RunCall(FAutomationTestBase* T, TSharedRef<FFixture> Fix, TFunction<void(TUniqueFunction<void(const FPipelineResult&, const TArray<TOut>&)>)> Issue,
		TFunction<int32()> GetPending, const FDoneDelegate& Done, TFunction<void(TCall<TOut>&)> Check, FRunOptions Opt = FRunOptions(), TFunction<void(TCall<TOut>&)> AfterIssue = nullptr)
	{
		const TSharedRef<TCall<TOut>> C = MakeShared<TCall<TOut>>();
		C->bInCall = true;
		Issue([C](const FPipelineResult& R, const TArray<TOut>& Out)
		{
			if (!IsInGameThread()) { C->bAllOnGameThread = false; }
			if (C->bInCall) { C->bAnyInCall = true; }
			if (C->Count.Increment() == 1)
			{
				C->Result = R;
				C->Out = Out;
			}
		});
		C->bInCall = false;
		if (GetPending) { C->PendingAfterIssue = GetPending(); }
		if (AfterIssue) { AfterIssue(*C); }

		const double Start = FPlatformTime::Seconds();
		const TSharedRef<double> FirstSeen = MakeShared<double>(0.0);
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TextChunkingSpecWait"), Opt.TickSeconds, [T, Fix, C, Start, FirstSeen, DoneCopy, Check, GetPending, Opt](float)
		{
			const double Now = FPlatformTime::Seconds();
			if (C->Count.GetValue() == 0 && GetPending)
			{
				C->MinPendingBeforeCallback = FMath::Min(C->MinPendingBeforeCallback, GetPending());
				C->MaxPendingBeforeCallback = FMath::Max(C->MaxPendingBeforeCallback, GetPending());
				++C->PendingSamples;
			}
			if (C->Count.GetValue() > 0 && *FirstSeen == 0.0) { *FirstSeen = Now; }
			const bool bTimeout = (Now - Start) > Opt.MaxWait;
			if ((*FirstSeen > 0.0 && (Now - *FirstSeen) > Opt.PostCallbackWait) || bTimeout)
			{
				if (C->Count.GetValue() == 0)
				{
					if (Opt.bNoCallbackOk)
					{
						T->AddInfo(FString::Printf(TEXT("no callback within %.0f s (allowed here)"), Opt.MaxWait));
						if (Check) { Check(*C); }
					}
					else
					{
						T->AddError(FString::Printf(TEXT("callback never fired within %.0f s"), Opt.MaxWait));
					}
				}
				else
				{
					T->TestEqual(TEXT("callback fired exactly once"), C->Count.GetValue(), 1);
					T->TestTrue(TEXT("callback ran on the game thread"), C->bAllOnGameThread);
					T->TestFalse(TEXT("callback must not run inside the async call"), C->bAnyInCall);
					if (Check) { Check(*C); }
				}
				DoneCopy.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	static TFunction<void(TUniqueFunction<void(const FPipelineResult&, const TArray<FPipelineEmbedding>&)>)> EmbedIssue(UTextEmbeddingPipeline* P, const TArray<FString>& Texts)
	{
		return [P, Texts](TUniqueFunction<void(const FPipelineResult&, const TArray<FPipelineEmbedding>&)> Cb)
		{
			P->EmbedAsync(Texts, MoveTemp(Cb));
		};
	}

	static TFunction<void(TUniqueFunction<void(const FPipelineResult&, const TArray<FPipelineClassification>&)>)> ClassifyIssue(UTextClassificationPipeline* P, const TArray<FString>& Texts)
	{
		return [P, Texts](TUniqueFunction<void(const FPipelineResult&, const TArray<FPipelineClassification>&)> Cb)
		{
			P->ClassifyAsync(Texts, 2, MoveTemp(Cb));
		};
	}

	using FEmbCall = TCall<FPipelineEmbedding>;
	using FClsCall = TCall<FPipelineClassification>;

	static void RunEmbed(FAutomationTestBase* T, TSharedRef<FFixture> F, const TArray<FString>& Texts, const FDoneDelegate& Done, TFunction<void(FEmbCall&)> Check,
		FRunOptions Opt = FRunOptions(), bool bTrackPending = false, TFunction<void(FEmbCall&)> AfterIssue = nullptr)
	{
		UTextEmbeddingPipeline* P = F->Embedder.Get();
		RunCall<FPipelineEmbedding>(T, F, EmbedIssue(P, Texts), bTrackPending ? TFunction<int32()>([P]() { return P->GetNumPendingCalls(); }) : TFunction<int32()>(), Done, Check, Opt, AfterIssue);
	}

	static void RunClassify(FAutomationTestBase* T, TSharedRef<FFixture> F, const TArray<FString>& Texts, const FDoneDelegate& Done, TFunction<void(FClsCall&)> Check)
	{
		RunCall<FPipelineClassification>(T, F, ClassifyIssue(F->Classifier.Get(), Texts), TFunction<int32()>(), Done, Check);
	}

	static void RunWorker(FAutomationTestBase* T, TSharedRef<FFixture> Fix, const FDoneDelegate& Done, TFunction<void()> Work, TFunction<void()> Check)
	{
		const TSharedRef<TFuture<void>> Fut = MakeShared<TFuture<void>>(Async(EAsyncExecution::ThreadPool, MoveTemp(Work)));
		const double Start = FPlatformTime::Seconds();
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TextChunkingSpecWorker"), 0.02f, [T, Fix, Fut, Start, DoneCopy, Check](float)
		{
			if (Fut->IsReady())
			{
				if (Check) { Check(); }
				DoneCopy.ExecuteIfBound();
				return false;
			}
			if ((FPlatformTime::Seconds() - Start) > CallTimeoutSeconds)
			{
				T->AddError(TEXT("worker thread did not finish within the time limit (hang?)"));
				DoneCopy.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}
}

BEGIN_DEFINE_SPEC(FTextChunkingSpec, "Pipelines.Text.Chunking",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FTextChunkingSpec)

void FTextChunkingSpec::Define()
{
	using namespace TextChunkingSpecPrivate;

	// ------------------------------------------------------------------------------------------------------
	Describe("Parity", [this]()
	{
		for (const int32 N : {17, 33, 50})
		{
			LatentIt(*FString::Printf(TEXT("EmbeddingCpu_%d"), N), LatentLimit, [this, N](const FDoneDelegate& Done)
			{
				// N = 17 is one full chunk + a chunk of 1; 33 = 16+16+1; 50 = 16+16+16+2 (contains the truncated and the empty sentence).
				// Each embedding must equal the torch golden of ITS sentence, in input order: a chunk loop that dropped the remainder,
				// repeated a chunk, reordered chunks or let padding from the longest text of another chunk leak in misses at least one.
				const TSharedRef<FFixture> F = MakeShared<FFixture>();
				if (!MakeFixture(*this, true, *F)) { Done.ExecuteIfBound(); return; }
				const TArray<int32> Idx = Range(N);
				RunEmbed(this, F, TextsOf(*F, Idx), Done, [this, F, Idx](FEmbCall& C)
				{
					TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
					TestTrue(TEXT("Device == CPU"), C.Result.Device == EPipelineDevice::CPU);
					CheckEmbeddings(*this, *F, C.Out, Idx, TEXT("chunked embedding"));
				});
			});

			LatentIt(*FString::Printf(TEXT("ClassificationCpu_%d"), N), LatentLimit, [this, N](const FDoneDelegate& Done)
			{
				// Same for distilbert-sst2: scores within 1e-3 and labels equal to the golden, per sentence, input order.
				const TSharedRef<FFixture> F = MakeShared<FFixture>();
				if (!MakeFixture(*this, false, *F)) { Done.ExecuteIfBound(); return; }
				const TArray<int32> Idx = Range(N);
				RunClassify(this, F, TextsOf(*F, Idx), Done, [this, F, Idx](FClsCall& C)
				{
					TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
					TestTrue(TEXT("Device == CPU"), C.Result.Device == EPipelineDevice::CPU);
					CheckClassifications(*this, *F, C.Out, Idx, TEXT("chunked classification"));
				});
			});
		}

		LatentIt("EmbeddingReversedOrder_33", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Sentences 40..8 (descending, crossing the chunk borders at different sentences than the ascending run): output
			// position k must hold the golden of the k-th INPUT. Catches chunks appended back to front or sorted by length.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, true, *F)) { Done.ExecuteIfBound(); return; }
			TArray<int32> Idx;
			for (int32 k = 0; k < 33; ++k) { Idx.Add(40 - k); }
			RunEmbed(this, F, TextsOf(*F, Idx), Done, [this, F, Idx](FEmbCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				CheckEmbeddings(*this, *F, C.Out, Idx, TEXT("reversed 33"));
			});
		});

		LatentIt("ClassificationRotated_40", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Sentences 25..49 then 0..14 (40 texts, wrapping): different chunk membership than the ascending runs.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, false, *F)) { Done.ExecuteIfBound(); return; }
			const TArray<int32> Idx = Range(40, 25);
			RunClassify(this, F, TextsOf(*F, Idx), Done, [this, F, Idx](FClsCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				CheckClassifications(*this, *F, C.Out, Idx, TEXT("rotated 40"));
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Callback", [this]()
	{
		for (const int32 N : {1, 16, 17, 40})
		{
			LatentIt(*FString::Printf(TEXT("EmbeddingOnceOnGameThread_%d"), N), LatentLimit, [this, N](const FDoneDelegate& Done)
			{
				// RunCall asserts: exactly one callback (watched 0.3 s after the first, so a callback per chunk fails for N > 16),
				// on the game thread, not inside EmbedAsync. N = 16 must behave as before, 17 is the first chunked size.
				const TSharedRef<FFixture> F = MakeShared<FFixture>();
				if (!MakeFixture(*this, true, *F)) { Done.ExecuteIfBound(); return; }
				const TArray<int32> Idx = Range(N);
				RunEmbed(this, F, TextsOf(*F, Idx), Done, [this, F, Idx](FEmbCall& C)
				{
					TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
					CheckEmbeddings(*this, *F, C.Out, Idx, TEXT("embedding callback"));
				});
			});

			LatentIt(*FString::Printf(TEXT("ClassificationOnceOnGameThread_%d"), N), LatentLimit, [this, N](const FDoneDelegate& Done)
			{
				const TSharedRef<FFixture> F = MakeShared<FFixture>();
				if (!MakeFixture(*this, false, *F)) { Done.ExecuteIfBound(); return; }
				const TArray<int32> Idx = Range(N);
				RunClassify(this, F, TextsOf(*F, Idx), Done, [this, F, Idx](FClsCall& C)
				{
					TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
					CheckClassifications(*this, *F, C.Out, Idx, TEXT("classification callback"));
				});
			});
		}
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("PendingCalls", [this]()
	{
		LatentIt("StaysAtLeastOneUntilCallback_40", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 40 texts = 3 chunks. The pending count is sampled right after the call and on EVERY tick until the callback.
			// A chunk loop that finishes chunk k (count drops to 0) before issuing chunk k+1 shows 0 on some tick in between.
			// After the callback (we look 0.3 s later) the count must be back to 0 (a leaked count would make it stay >= 1).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, true, *F)) { Done.ExecuteIfBound(); return; }
			FRunOptions Opt;
			Opt.TickSeconds = 0.f; // every frame
			UTextEmbeddingPipeline* P = F->Embedder.Get();
			RunEmbed(this, F, TextsOf(*F, Range(40)), Done, [this, P](FEmbCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("pending right after the call >= 1, got %d"), C.PendingAfterIssue), C.PendingAfterIssue >= 1);
				AddInfo(FString::Printf(TEXT("sampled %d ticks before the callback, min pending = %d"), C.PendingSamples, C.MinPendingBeforeCallback));
				// C43 (amended): exactly 1 pending call for the whole chunked call (lead, review WP7.11 finding 2).
				TestTrue(*FString::Printf(TEXT("ticks sampled before the callback: %d"), C.PendingSamples), C.PendingSamples > 0);
				TestEqual(TEXT("min pending on every tick before the callback"), C.MinPendingBeforeCallback, 1);
				TestEqual(TEXT("max pending on every tick before the callback"), C.MaxPendingBeforeCallback, 1);
				TestEqual(TEXT("pending after the callback"), P->GetNumPendingCalls(), 0);
			}, Opt, /*bTrackPending*/ true);
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Timings", [this]()
	{
		LatentIt("SumOverChunks_48_vs_16", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Same 16 texts repeated 3x = 48 texts = 3 chunks of identical work. InferenceMs of the whole call is the SUM over
			// chunks, so it must be >= the InferenceMs of a single 16-text call (about 3x). A result that reports only the last
			// (or first) chunk's time would be about equal, and one that forgot to stamp it would be 0. A warm-up call comes first
			// so one-time model setup does not inflate the 16-text measurement.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, true, *F)) { Done.ExecuteIfBound(); return; }
			const TArray<FString> Sixteen = TextsOf(*F, Range(16));
			TArray<FString> FortyEight;
			for (int32 r = 0; r < 3; ++r) { FortyEight.Append(Sixteen); }
			struct FOut { FPipelineResult Small; };
			const TSharedRef<FOut> Out = MakeShared<FOut>();
			const FDoneDelegate DoneCopy = Done;
			RunEmbed(this, F, Sixteen, FDoneDelegate::CreateLambda([this, F, Sixteen, FortyEight, Out, DoneCopy]()
			{
				RunEmbed(this, F, Sixteen, FDoneDelegate::CreateLambda([this, F, FortyEight, Out, DoneCopy]()
				{
					RunEmbed(this, F, FortyEight, DoneCopy, [this, Out](FEmbCall& C)
					{
						TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
						TestEqual(TEXT("48 embeddings"), C.Out.Num(), 48);
						AddInfo(FString::Printf(TEXT("InferenceMs: 16 texts %.3f, 48 texts %.3f"), Out->Small.InferenceMs, C.Result.InferenceMs));
						TestTrue(*FString::Printf(TEXT("48-text InferenceMs %.3f > 0"), C.Result.InferenceMs), C.Result.InferenceMs > 0.f);
						TestTrue(*FString::Printf(TEXT("48-text InferenceMs %.3f >= 2 x 16-text InferenceMs %.3f (sum over 3 chunks)"), C.Result.InferenceMs, Out->Small.InferenceMs),
							C.Result.InferenceMs >= 2.0f * Out->Small.InferenceMs);
						TestTrue(*FString::Printf(TEXT("PreprocessMs %.4f > 0"), C.Result.PreprocessMs), C.Result.PreprocessMs > 0.f);
						TestTrue(*FString::Printf(TEXT("PostprocessMs %.4f > 0"), C.Result.PostprocessMs), C.Result.PostprocessMs > 0.f);
						TestTrue(TEXT("Device == CPU (set, not Auto)"), C.Result.Device == EPipelineDevice::CPU);
					});
				}), [this, Out](FEmbCall& C)
				{
					TestTrue(TEXT("16-text call bSuccess"), C.Result.bSuccess);
					Out->Small = C.Result;
					TestTrue(*FString::Printf(TEXT("16-text InferenceMs %.3f > 0"), C.Result.InferenceMs), C.Result.InferenceMs > 0.f);
				});
			}), [](FEmbCall&) {});
		});

		LatentIt("ClassificationInferenceMs_40", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Classification path also stamps all timings and the device for a chunked call.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, false, *F)) { Done.ExecuteIfBound(); return; }
			RunClassify(this, F, TextsOf(*F, Range(40)), Done, [this](FClsCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("InferenceMs %.3f > 0"), C.Result.InferenceMs), C.Result.InferenceMs > 0.f);
				TestTrue(*FString::Printf(TEXT("PreprocessMs %.4f > 0"), C.Result.PreprocessMs), C.Result.PreprocessMs > 0.f);
				TestTrue(TEXT("Device == CPU"), C.Result.Device == EPipelineDevice::CPU);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Errors", [this]()
	{
		LatentIt("FailingModelEndsCallWithEmptyOutputs_40", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// NOTE: the public API offers no input that fails only in a LATER chunk (long text is truncated, empty text is valid).
			// What can be provoked: MiniLM declared with Pooling None (rank-3 output) fails every chunk with ShapeMismatch. With 40 texts
			// (3 chunks) the call must still end after the first failing chunk with exactly one callback (RunCall asserts it, and a
			// loop that went on would fire more), the unchunked error code, and NO partial results (outputs empty, never 16 or 40 rows).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, true, *F, [](UPipelineModelPackage& P) { P.Preprocess.Pooling = EPipelinePooling::None; })) { Done.ExecuteIfBound(); return; }
			UTextEmbeddingPipeline* P = F->Embedder.Get();
			RunEmbed(this, F, TextsOf(*F, Range(40)), Done, [this, P](FEmbCall& C)
			{
				TestFalse(TEXT("bSuccess"), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error ShapeMismatch, got %d (%s)"), (int32)C.Result.Error, *C.Result.Message), C.Result.Error == EPipelineError::ShapeMismatch);
				TestEqual(TEXT("no partial embeddings"), C.Out.Num(), 0);
				TestEqual(TEXT("pending after the callback"), P->GetNumPendingCalls(), 0);
			});
		});

		LatentIt("EmptyTextListOneCallback", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Invalid call (no texts): exactly one callback, InvalidArgument, empty outputs; chunking must not turn 0 texts into 0 chunks and no callback.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, true, *F)) { Done.ExecuteIfBound(); return; }
			RunEmbed(this, F, TArray<FString>(), Done, [this](FEmbCall& C)
			{
				TestFalse(TEXT("bSuccess"), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)C.Result.Error, *C.Result.Message), C.Result.Error == EPipelineError::InvalidArgument);
				TestEqual(TEXT("no embeddings"), C.Out.Num(), 0);
			});
		});

		LatentIt("Over1024TextsOneCallback", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 1025 texts stay rejected up front (the 1024 limit is not replaced by chunking): InvalidArgument, empty, one callback, nothing run.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, false, *F)) { Done.ExecuteIfBound(); return; }
			TArray<FString> Texts;
			Texts.Init(TEXT("hello"), 1025);
			RunClassify(this, F, Texts, Done, [this](FClsCall& C)
			{
				TestFalse(TEXT("bSuccess"), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)C.Result.Error, *C.Result.Message), C.Result.Error == EPipelineError::InvalidArgument);
				TestEqual(TEXT("no results"), C.Out.Num(), 0);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Destroyed", [this]()
	{
		LatentIt("ReleasedRightAfterIssuingA64TextCall", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Mirrors Embedding.Calls.DestroyedMidCall: the pipeline is marked garbage and fully collected right after EmbedAsync
			// with 64 texts (4 chunks). Required (C43 clarified): exactly one callback within 60 s (none fails; a second one over
			// the next 2 s fails), success, and all 64 outputs equal to the goldens.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, true, *F)) { Done.ExecuteIfBound(); return; }
			UTextEmbeddingPipeline* P = F->Embedder.Get();
			const TArray<int32> Idx = Range(64);
			FRunOptions Opt;
			Opt.PostCallbackWait = 2.0;
			Opt.MaxWait = 60.0;
			Opt.bNoCallbackOk = false; // no callback within 60 s is a failure
			RunEmbed(this, F, TextsOf(*F, Idx), Done, [this, F, Idx](FEmbCall& C)
			{
				TestTrue(*FString::Printf(TEXT("released mid-call: bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				CheckEmbeddings(*this, *F, C.Out, Idx, TEXT("released mid-call"));
			}, Opt, false, [F, P](FEmbCall&)
			{
				P->MarkAsGarbage();
				F->Embedder.Reset();
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Blocking", [this]()
	{
		LatentIt("EmbedBlocking33EqualsAsync", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// EmbedBlocking with 33 texts (3 chunks inside one blocking call) vs the async result of the same pipeline and the golden.
			// Async and blocking run the same chunks, so the cosine between them must be > 0.99999.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, true, *F)) { Done.ExecuteIfBound(); return; }
			const TArray<int32> Idx = Range(33);
			const TArray<FString> Texts = TextsOf(*F, Idx);
			struct FOut { TArray<FPipelineEmbedding> Async; TArray<FPipelineEmbedding> Blocking; FPipelineResult BlockRes; };
			const TSharedRef<FOut> Out = MakeShared<FOut>();
			UTextEmbeddingPipeline* P = F->Embedder.Get();
			const FDoneDelegate DoneCopy = Done;
			RunEmbed(this, F, Texts, FDoneDelegate::CreateLambda([this, F, Texts, Out, P, Idx, DoneCopy]()
			{
				RunWorker(this, F, DoneCopy, [Texts, Out, P]() { Out->BlockRes = P->EmbedBlocking(Texts, Out->Blocking); }, [this, F, Out, Idx]()
				{
					TestTrue(*FString::Printf(TEXT("blocking bSuccess (%s)"), *Out->BlockRes.Message), Out->BlockRes.bSuccess);
					CheckEmbeddings(*this, *F, Out->Blocking, Idx, TEXT("blocking 33"));
					if (!TestEqual(TEXT("blocking count == async count"), Out->Blocking.Num(), Out->Async.Num())) { return; }
					double Worst = 2.0;
					for (int32 k = 0; k < Out->Async.Num(); ++k) { Worst = FMath::Min(Worst, CosineF(Out->Blocking[k].Values, Out->Async[k].Values)); }
					AddInfo(FString::Printf(TEXT("blocking vs async worst cosine = %.9f"), Worst));
					TestTrue(*FString::Printf(TEXT("blocking == async, worst cosine %.9f > 0.99999"), Worst), Worst > 0.99999);
				});
			}), [this, Out](FEmbCall& C)
			{
				TestTrue(*FString::Printf(TEXT("async bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				Out->Async = C.Out;
			});
		});

		LatentIt("ClassifyBlocking33EqualsAsync", LatentLimit, [this](const FDoneDelegate& Done)
		{
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, false, *F)) { Done.ExecuteIfBound(); return; }
			const TArray<int32> Idx = Range(33);
			const TArray<FString> Texts = TextsOf(*F, Idx);
			struct FOut { TArray<FPipelineClassification> Async; TArray<FPipelineClassification> Blocking; FPipelineResult BlockRes; };
			const TSharedRef<FOut> Out = MakeShared<FOut>();
			UTextClassificationPipeline* P = F->Classifier.Get();
			const FDoneDelegate DoneCopy = Done;
			RunClassify(this, F, Texts, FDoneDelegate::CreateLambda([this, F, Texts, Out, P, Idx, DoneCopy]()
			{
				RunWorker(this, F, DoneCopy, [Texts, Out, P]() { Out->BlockRes = P->ClassifyBlocking(Texts, 2, Out->Blocking); }, [this, F, Out, Idx]()
				{
					TestTrue(*FString::Printf(TEXT("blocking bSuccess (%s)"), *Out->BlockRes.Message), Out->BlockRes.bSuccess);
					CheckClassifications(*this, *F, Out->Blocking, Idx, TEXT("blocking 33"));
					if (!TestEqual(TEXT("blocking count == async count"), Out->Blocking.Num(), Out->Async.Num())) { return; }
					double Worst = 0.0;
					int32 LabelDiff = 0;
					for (int32 k = 0; k < Out->Async.Num(); ++k)
					{
						if (Out->Blocking[k].Label != Out->Async[k].Label) { ++LabelDiff; }
						if (Out->Blocking[k].Scores.Num() != Out->Async[k].Scores.Num()) { ++LabelDiff; continue; }
						for (int32 c = 0; c < Out->Async[k].Scores.Num(); ++c) { Worst = FMath::Max(Worst, (double)FMath::Abs(Out->Blocking[k].Scores[c] - Out->Async[k].Scores[c])); }
					}
					AddInfo(FString::Printf(TEXT("blocking vs async: max score diff %.3g, label differences %d"), Worst, LabelDiff));
					TestEqual(TEXT("blocking and async labels equal"), LabelDiff, 0);
					TestTrue(*FString::Printf(TEXT("blocking == async, max score diff %.3g <= 1e-5"), Worst), Worst <= 1e-5);
				});
			}), [this, Out](FClsCall& C)
			{
				TestTrue(*FString::Printf(TEXT("async bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				Out->Async = C.Out;
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
