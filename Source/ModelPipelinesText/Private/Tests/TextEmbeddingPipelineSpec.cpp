// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "UObject/GarbageCollection.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineBackend.h"
#include "PipelineLibrary.h"
#include "PipelineModelPackage.h"
#include "PipelineNNEBackend.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "TextEmbeddingPipeline.h"
#include "TextPipelineLibrary.h"
#include "PipelineTextTestData.h"

using namespace UE::ModelPipelines::TextTests;

namespace TextEmbeddingSpecPrivate
{
	const FTimespan LatentLimit = FTimespan::FromSeconds(240);
	constexpr double CallTimeoutSeconds = 150.0;
	const TCHAR* const MiniSlug = TEXT("all-MiniLM-L6-v2");
	const TCHAR* const TinySlug = TEXT("tiny_text_embedding");

	/** Goldens parse once per run; they are several MB. */
	static TSharedPtr<FJsonObject> GCachedRoot;

	static TSharedPtr<FJsonObject> GetGoldens(FAutomationTestBase& T)
	{
		if (!GCachedRoot.IsValid())
		{
			FString Err;
			GCachedRoot = LoadTextGoldens(Err);
			if (!GCachedRoot.IsValid())
			{
				T.AddError(Err);
			}
		}
		return GCachedRoot;
	}

	struct FFixture
	{
		TSharedPtr<FJsonObject> Root;
		TSharedPtr<FJsonObject> Model;
		TStrongObjectPtr<UPipelineModelPackage> Package;
		TStrongObjectPtr<UTextEmbeddingPipeline> Pipeline;
		TSharedPtr<FPipelineNNEBackend> Backend; // kept alive for the pipeline's lifetime (bogus-runtime test)
		TArray<FString> Ids;
		TArray<FString> Texts;
		TMap<FString, TArray<double>> GoldenEmbedding;
	};

	static FString ToHex(const FString& S)
	{
		FTCHARToUTF8 U(*S);
		FString Out;
		for (int32 i = 0; i < U.Length(); ++i)
		{
			Out += FString::Printf(TEXT("%02x"), (uint8)U.Get()[i]);
		}
		return Out;
	}

	/** Loads goldens, sentences and a package. bCreatePipeline: use the creator. Mutate may change the package first. */
	static bool MakeFixture(FAutomationTestBase& T, const FString& Slug, EPipelineDevice Device, FFixture& F, bool bCreatePipeline = true,
		TFunction<void(UPipelineModelPackage&)> Mutate = nullptr)
	{
		F.Root = GetGoldens(T);
		if (!F.Root.IsValid()) { return false; }
		F.Model = FindModel(F.Root, Slug);
		if (!F.Model.IsValid())
		{
			T.AddError(FString::Printf(TEXT("Golden model '%s' not found in text_goldens.json. Regenerate with Tools/make_text_goldens.py."), *Slug));
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
			const FString Text = S->GetStringField(TEXT("text"));
			// Guards the JSON reader: emoji / CJK must be the same UTF-8 bytes the Python side used.
			if (!ToHex(Text).Equals(S->GetStringField(TEXT("text_utf8_hex")), ESearchCase::IgnoreCase))
			{
				T.AddError(FString::Printf(TEXT("Sentence %s: UTF-8 bytes differ from the golden text_utf8_hex."), *S->GetStringField(TEXT("id"))));
			}
			F.Ids.Add(S->GetStringField(TEXT("id")));
			F.Texts.Add(Text);
		}
		const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
		if (F.Model->TryGetArrayField(TEXT("cases"), Cases))
		{
			for (const TSharedPtr<FJsonValue>& V : *Cases)
			{
				const TSharedPtr<FJsonObject> C = V->AsObject();
				const TArray<TSharedPtr<FJsonValue>>* Emb = nullptr;
				if (C->TryGetArrayField(TEXT("embedding"), Emb))
				{
					TArray<double>& Dst = F.GoldenEmbedding.Add(C->GetStringField(TEXT("sentence_id")));
					for (const TSharedPtr<FJsonValue>& X : *Emb) { Dst.Add(X->AsNumber()); }
				}
			}
		}

		FString Err;
		UPipelineModelPackage* Pkg = MakePackage(F.Root, F.Model, Device, Err);
		if (!Pkg)
		{
			T.AddError(Err);
			return false;
		}
		F.Package.Reset(Pkg);
		if (Mutate) { Mutate(*Pkg); }
		if (bCreatePipeline)
		{
			FPipelineResult R;
			UTextEmbeddingPipeline* P = UTextPipelineLibrary::CreateTextEmbeddingPipeline(Pkg, R);
			if (!P)
			{
				T.AddError(FString::Printf(TEXT("CreateTextEmbeddingPipeline failed: %s"), *R.Message));
				return false;
			}
			F.Pipeline.Reset(P);
		}
		return true;
	}

	/** Cosine in double; -2 if lengths differ or a norm is zero. */
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

	static TArray<FString> Pick(const FFixture& F, std::initializer_list<int32> Indices)
	{
		TArray<FString> Out;
		for (int32 I : Indices) { Out.Add(F.Texts[I]); }
		return Out;
	}

	/** Compares embeddings (one per index in Indices) with the golden; reports the worst cosine. Returns the worst. */
	static double CheckAgainstGolden(FAutomationTestBase& T, const FFixture& F, const TArray<FPipelineEmbedding>& Embs, const TArray<int32>& Indices,
		const TCHAR* Label, int32 ExpectedDim, double MinCosine)
	{
		if (!T.TestEqual(*FString::Printf(TEXT("%s: embedding count"), Label), Embs.Num(), Indices.Num())) { return -2.0; }
		double Worst = 2.0;
		FString WorstId;
		for (int32 k = 0; k < Indices.Num(); ++k)
		{
			const FString& Id = F.Ids[Indices[k]];
			const TArray<double>* G = F.GoldenEmbedding.Find(Id);
			if (!G) { T.AddError(FString::Printf(TEXT("%s: no golden for %s"), Label, *Id)); continue; }
			if (Embs[k].Values.Num() != ExpectedDim)
			{
				T.AddError(FString::Printf(TEXT("%s: %s has length %d, expected %d"), Label, *Id, Embs[k].Values.Num(), ExpectedDim));
				continue;
			}
			const double C = Cosine(Embs[k].Values, *G);
			if (C < Worst) { Worst = C; WorstId = Id; }
		}
		T.AddInfo(FString::Printf(TEXT("%s: worst cosine vs golden = %.8f (sentence %s)"), Label, Worst, *WorstId));
		T.TestTrue(*FString::Printf(TEXT("%s: every cosine > %.4f (worst %.8f at %s)"), Label, MinCosine, Worst, *WorstId), Worst > MinCosine);
		if (Worst <= 0.9999 && Worst > MinCosine)
		{
			T.AddWarning(FString::Printf(TEXT("%s: worst cosine %.8f is below 0.9999, expected ~1e-6 differences; suspect padding / special tokens / truncation."), Label, Worst));
		}
		return Worst;
	}

	static TArray<int32> AllIndices()
	{
		TArray<int32> I;
		for (int32 k = 0; k < 50; ++k) { I.Add(k); }
		return I;
	}

	struct FCall
	{
		FThreadSafeCounter Count;
		bool bInCall = false;
		bool bAnyInCall = false;
		bool bAllOnGameThread = true;
		FPipelineResult Result;
		TArray<FPipelineEmbedding> Embs;
	};

	/**
	 * Issues EmbedAsync on the game thread and waits (ticker, never blocking) for the callback plus 0.3 s to catch a second
	 * callback. Fails with a message instead of hanging. Check runs on the game thread after the generic assertions
	 * (exactly once, on the game thread, never inside the call). AfterIssue runs right after EmbedAsync returns.
	 */
	static void RunCall(FAutomationTestBase* T, TSharedRef<FFixture> Fix, UTextEmbeddingPipeline* Pipe, const TArray<FString>& Texts,
		const FDoneDelegate& Done, TFunction<void(FCall&)> Check, TFunction<void()> AfterIssue = nullptr)
	{
		const TSharedRef<FCall> C = MakeShared<FCall>();
		C->bInCall = true;
		Pipe->EmbedAsync(Texts, [C](const FPipelineResult& R, const TArray<FPipelineEmbedding>& E)
		{
			if (!IsInGameThread()) { C->bAllOnGameThread = false; }
			if (C->bInCall) { C->bAnyInCall = true; }
			if (C->Count.Increment() == 1)
			{
				C->Result = R;
				C->Embs = E;
			}
		});
		C->bInCall = false;
		if (AfterIssue) { AfterIssue(); }

		const double Start = FPlatformTime::Seconds();
		const TSharedRef<double> FirstSeen = MakeShared<double>(0.0);
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TextEmbeddingSpecWait"), 0.02f, [T, Fix, C, Start, FirstSeen, DoneCopy, Check](float)
		{
			const double Now = FPlatformTime::Seconds();
			if (C->Count.GetValue() > 0 && *FirstSeen == 0.0) { *FirstSeen = Now; }
			const bool bTimeout = (Now - Start) > CallTimeoutSeconds;
			if ((*FirstSeen > 0.0 && (Now - *FirstSeen) > 0.3) || bTimeout)
			{
				if (C->Count.GetValue() == 0)
				{
					T->AddError(FString::Printf(TEXT("callback never fired within %.0f s"), CallTimeoutSeconds));
				}
				else
				{
					T->TestEqual(TEXT("callback fired exactly once"), C->Count.GetValue(), 1);
					T->TestTrue(TEXT("callback ran on the game thread"), C->bAllOnGameThread);
					T->TestFalse(TEXT("callback must not run inside EmbedAsync"), C->bAnyInCall);
					if (Check) { Check(*C); }
				}
				DoneCopy.ExecuteIfBound();
				return false;
			}
			return true;
		});
	}

	/** Runs Work on a pool thread; Check runs on the game thread when it finishes. Times out with an error. */
	static void RunWorker(FAutomationTestBase* T, TSharedRef<FFixture> Fix, const FDoneDelegate& Done, TFunction<void()> Work, TFunction<void()> Check)
	{
		const TSharedRef<TFuture<void>> F = MakeShared<TFuture<void>>(Async(EAsyncExecution::ThreadPool, MoveTemp(Work)));
		const double Start = FPlatformTime::Seconds();
		FDoneDelegate DoneCopy = Done;
		FTSTicker::GetCoreTicker().AddTicker(TEXT("TextEmbeddingSpecWorker"), 0.02f, [T, Fix, F, Start, DoneCopy, Check](float)
		{
			if (F->IsReady())
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

	/** True = skip. GPU tests run headless (DirectML works under -nullrhi), so availability of the GPU interface decides. */
	static bool GpuSkip(FAutomationTestBase& T)
	{
		FPipelineNNEBackend Backend;
		if (Backend.IsDeviceAvailable(EPipelineDevice::GPU)) { return false; }
		if (IsGpuRequired())
		{
			T.AddError(TEXT("GPU not available but MODELPIPELINES_REQUIRE_GPU=1"));
		}
		else
		{
			T.AddInfo(TEXT("GPU not available: skipped"));
		}
		return true;
	}

	static void ExpectOneWarning(FAutomationTestBase& T)
	{
		T.AddExpectedMessage(TEXT("UTextEmbeddingPipeline::EmbedAsync: null callback, nothing runs"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
	}

	static void CheckParityResult(FAutomationTestBase& T, const FFixture& F, FCall& C, EPipelineDevice ExpectedDevice, const TCHAR* Label)
	{
		T.TestTrue(*FString::Printf(TEXT("%s: bSuccess (%s)"), Label, *C.Result.Message), C.Result.bSuccess);
		T.TestTrue(*FString::Printf(TEXT("%s: Result.Device == %d, got %d"), Label, (int32)ExpectedDevice, (int32)C.Result.Device), C.Result.Device == ExpectedDevice);
		CheckAgainstGolden(T, F, C.Embs, AllIndices(), Label, 384, 0.999);
	}
}

BEGIN_DEFINE_SPEC(FTextEmbeddingPipelineSpec, "Pipelines.Text.Embedding",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FTextEmbeddingPipelineSpec)

void FTextEmbeddingPipelineSpec::Define()
{
	using namespace TextEmbeddingSpecPrivate;

	// ------------------------------------------------------------------------------------------------------
	Describe("Parity", [this]()
	{
		LatentIt("CPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 50 golden sentences in ONE call: includes 2 truncated (> 256 tokens), empty, whitespace-only, emoji, CJK.
			// Mean pooling that counted padding, missing [CLS]/[SEP], or wrong truncation drops the worst cosine below 0.9999.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), F->Texts, Done, [this, F](FCall& C)
			{
				CheckParityResult(*this, *F, C, EPipelineDevice::CPU, TEXT("MiniLM CPU"));
			});
		});

		LatentIt("GPU", LatentLimit, [this](const FDoneDelegate& Done)
		{
			if (GpuSkip(*this)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::GPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), F->Texts, Done, [this, F](FCall& C)
			{
				CheckParityResult(*this, *F, C, EPipelineDevice::GPU, TEXT("MiniLM GPU"));
			});
		});

		LatentIt("TinyClsNormalize", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// CLS pooling + L2 normalize + token_type_ids input vs ORT golden, |a-e| <= 1e-5 + 1e-5|e|.
			// Mean pooling here, a missing token_type_ids tensor, or no normalization all break this at once.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, TinySlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), F->Texts, Done, [this, F](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				if (!TestEqual(TEXT("embedding count"), C.Embs.Num(), 50)) { return; }
				double WorstAbs = 0.0;
				int32 Bad = 0;
				FString FirstBad;
				for (int32 i = 0; i < 50; ++i)
				{
					const TArray<double>* G = F->GoldenEmbedding.Find(F->Ids[i]);
					if (!G || G->Num() != C.Embs[i].Values.Num())
					{
						AddError(FString::Printf(TEXT("%s: length %d vs golden %d"), *F->Ids[i], C.Embs[i].Values.Num(), G ? G->Num() : -1));
						continue;
					}
					for (int32 d = 0; d < G->Num(); ++d)
					{
						const double E = (*G)[d];
						const double Diff = FMath::Abs((double)C.Embs[i].Values[d] - E);
						WorstAbs = FMath::Max(WorstAbs, Diff);
						if (!(Diff <= 1e-5 + 1e-5 * FMath::Abs(E)))
						{
							if (Bad++ == 0) { FirstBad = FString::Printf(TEXT("%s[%d]: actual %.8f golden %.8f"), *F->Ids[i], d, C.Embs[i].Values[d], E); }
						}
					}
				}
				AddInfo(FString::Printf(TEXT("tiny: worst |a-e| = %.3g"), WorstAbs));
				TestEqual(*FString::Printf(TEXT("elements outside 1e-5 tolerance (worst abs %.3g, first: %s)"), WorstAbs, *FirstBad), Bad, 0);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Batching", [this]()
	{
		LatentIt("BatchInvariance", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// A sentence alone (SeqLen = its own length) vs inside the 50-batch (right-padded to the longest, 256) must agree:
			// cosine > 0.99999. Pooling that counted padding or an unmasked attention would show here for short sentences.
			// Indices: 0 short, 39 empty, 41 and 42 long/truncated.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			struct FOut { FPipelineResult Batch; TArray<FPipelineEmbedding> BatchEmbs; TMap<int32, TArray<FPipelineEmbedding>> Single; TMap<int32, FPipelineResult> SingleRes; };
			const TSharedRef<FOut> Out = MakeShared<FOut>();
			UTextEmbeddingPipeline* P = F->Pipeline.Get();
			RunWorker(this, F, Done, [F, Out, P]()
			{
				Out->Batch = P->EmbedBlocking(F->Texts, Out->BatchEmbs);
				for (int32 I : {0, 39, 41, 42})
				{
					TArray<FPipelineEmbedding> E;
					Out->SingleRes.Add(I, P->EmbedBlocking(TArray<FString>{F->Texts[I]}, E));
					Out->Single.Add(I, MoveTemp(E));
				}
			}, [this, F, Out]()
			{
				TestTrue(*FString::Printf(TEXT("batch bSuccess (%s)"), *Out->Batch.Message), Out->Batch.bSuccess);
				if (!TestEqual(TEXT("batch count"), Out->BatchEmbs.Num(), 50)) { return; }
				for (const TPair<int32, TArray<FPipelineEmbedding>>& KV : Out->Single)
				{
					const FPipelineResult& R = Out->SingleRes[KV.Key];
					if (!TestTrue(*FString::Printf(TEXT("single %d bSuccess (%s)"), KV.Key, *R.Message), R.bSuccess) || !TestEqual(TEXT("single count"), KV.Value.Num(), 1)) { continue; }
					const double C = CosineF(KV.Value[0].Values, Out->BatchEmbs[KV.Key].Values);
					AddInfo(FString::Printf(TEXT("sentence %d: single vs batch cosine = %.9f"), KV.Key, C));
					TestTrue(*FString::Printf(TEXT("sentence %d single vs batch cosine %.9f > 0.99999"), KV.Key, C), C > 0.99999);
				}
			});
		});

		LatentIt("Order", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// {A,B} and {B,A} must give swapped results; A=sentence 0 (short), B=sentence 41 (long). A pipeline that sorted by
			// length, or returned rows in tokenizer order, would not swap. Each is also checked against the golden.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			struct FOut { TArray<FPipelineEmbedding> AB, BA; FPipelineResult R1, R2; };
			const TSharedRef<FOut> Out = MakeShared<FOut>();
			UTextEmbeddingPipeline* P = F->Pipeline.Get();
			RunWorker(this, F, Done, [F, Out, P]()
			{
				Out->R1 = P->EmbedBlocking(Pick(*F, {0, 41}), Out->AB);
				Out->R2 = P->EmbedBlocking(Pick(*F, {41, 0}), Out->BA);
			}, [this, F, Out]()
			{
				TestTrue(*FString::Printf(TEXT("AB bSuccess (%s)"), *Out->R1.Message), Out->R1.bSuccess);
				TestTrue(*FString::Printf(TEXT("BA bSuccess (%s)"), *Out->R2.Message), Out->R2.bSuccess);
				if (Out->AB.Num() != 2 || Out->BA.Num() != 2) { AddError(TEXT("expected 2 embeddings from each call")); return; }
				const double C1 = CosineF(Out->AB[0].Values, Out->BA[1].Values);
				const double C2 = CosineF(Out->AB[1].Values, Out->BA[0].Values);
				const double Cross = CosineF(Out->AB[0].Values, Out->BA[0].Values);
				TestTrue(*FString::Printf(TEXT("AB[0] == BA[1], cosine %.9f"), C1), C1 > 0.99999);
				TestTrue(*FString::Printf(TEXT("AB[1] == BA[0], cosine %.9f"), C2), C2 > 0.99999);
				TestTrue(*FString::Printf(TEXT("A and B must differ, cosine %.4f"), Cross), Cross < 0.99);
				CheckAgainstGolden(*this, *F, Out->AB, TArray<int32>{0, 41}, TEXT("order AB"), 384, 0.999);
			});
		});

		LatentIt("Timings", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// All three stage timings must be filled for a real 50-batch (a pipeline that forgot to stamp one reports 0).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), F->Texts, Done, [this](FCall& C)
			{
				TestTrue(TEXT("bSuccess"), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("PreprocessMs %.4f > 0"), C.Result.PreprocessMs), C.Result.PreprocessMs > 0.f);
				TestTrue(*FString::Printf(TEXT("InferenceMs %.4f > 0"), C.Result.InferenceMs), C.Result.InferenceMs > 0.f);
				TestTrue(*FString::Printf(TEXT("PostprocessMs %.4f > 0"), C.Result.PostprocessMs), C.Result.PostprocessMs > 0.f);
			});
		});

		LatentIt("Max1024Texts", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Boundary: exactly 1024 texts is allowed (only > 1024 is rejected). An off-by-one in the limit rejects this.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			TArray<FString> Texts;
			for (int32 i = 0; i < 1024; ++i) { Texts.Add(F->Texts[i % 50]); }
			RunCall(this, F, F->Pipeline.Get(), Texts, Done, [this](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("1024 texts bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				TestEqual(TEXT("1024 embeddings"), C.Embs.Num(), 1024);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Similarity", [this]()
	{
		LatentIt("TopK", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// TopKSimilar over the Unreal embeddings vs the golden (torch) similarity: scores within 1e-3 per rank; indices equal
			// unless neighbouring golden scores are within 1e-3 (a legitimate tie flip).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), F->Texts, Done, [this, F](FCall& C)
			{
				if (!TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess) || !TestEqual(TEXT("count"), C.Embs.Num(), 50)) { return; }
				const TSharedPtr<FJsonObject>* Sim = nullptr;
				if (!F->Model->TryGetObjectField(TEXT("similarity"), Sim)) { AddError(TEXT("golden model has no similarity block")); return; }
				const int32 K = (int32)(*Sim)->GetNumberField(TEXT("k"));
				const TArray<TSharedPtr<FJsonValue>>* Queries = nullptr;
				(*Sim)->TryGetArrayField(TEXT("queries"), Queries);
				if (!Queries || Queries->Num() != 5) { AddError(TEXT("expected 5 golden queries")); return; }
				for (const TSharedPtr<FJsonValue>& QV : *Queries)
				{
					const TSharedPtr<FJsonObject> Q = QV->AsObject();
					const FString QId = Q->GetStringField(TEXT("sentence_id"));
					const int32 QIdx = F->Ids.IndexOfByKey(QId);
					if (QIdx == INDEX_NONE) { AddError(FString::Printf(TEXT("unknown query sentence %s"), *QId)); continue; }
					const TArray<FPipelineSimilarityMatch> Got = UPipelineLibrary::TopKSimilar(C.Embs[QIdx], C.Embs, K);
					const TArray<TSharedPtr<FJsonValue>>& Top = Q->GetArrayField(TEXT("top"));
					if (!TestEqual(*FString::Printf(TEXT("%s: result count"), *QId), Got.Num(), Top.Num())) { continue; }
					for (int32 r = 0; r < Top.Num(); ++r)
					{
						const TSharedPtr<FJsonObject> G = Top[r]->AsObject();
						const double GScore = G->GetNumberField(TEXT("score"));
						const int32 GIdx = (int32)G->GetNumberField(TEXT("index"));
						const double Diff = FMath::Abs((double)Got[r].Score - GScore);
						TestTrue(*FString::Printf(TEXT("%s rank %d: score %.6f vs golden %.6f (diff %.2g > 1e-3)"), *QId, r, Got[r].Score, GScore, Diff), Diff <= 1e-3);
						bool bTie = false;
						if (r > 0) { bTie |= FMath::Abs(Top[r - 1]->AsObject()->GetNumberField(TEXT("score")) - GScore) < 1e-3; }
						if (r + 1 < Top.Num()) { bTie |= FMath::Abs(Top[r + 1]->AsObject()->GetNumberField(TEXT("score")) - GScore) < 1e-3; }
						if (!bTie)
						{
							TestEqual(*FString::Printf(TEXT("%s rank %d: index"), *QId, r), Got[r].Index, GIdx);
						}
					}
				}
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Creator", [this]()
	{
		It("WrongTask", [this]()
		{
			// A TextClassification package must be refused by the embedding creator with WrongTask, not crash.
			FFixture F;
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, F, false, [](UPipelineModelPackage& P) { P.Task = EPipelineTask::TextClassification; })) { return; }
			FPipelineResult R;
			UTextEmbeddingPipeline* P = UTextPipelineLibrary::CreateTextEmbeddingPipeline(F.Package.Get(), R);
			TestNull(TEXT("pipeline"), P);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("Error WrongTask, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::WrongTask);
		});

		It("NullPackage", [this]()
		{
			FPipelineResult R;
			UTextEmbeddingPipeline* P = UTextPipelineLibrary::CreateTextEmbeddingPipeline(nullptr, R);
			TestNull(TEXT("pipeline"), P);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)R.Error, *R.Message), R.Error == EPipelineError::InvalidArgument);
		});

		It("InvalidTokenizerJson", [this]()
		{
			// Garbage tokenizer text must surface as PreprocessFailed (10) from OnInitialize, with a message, and no pipeline.
			AddExpectedMessage(TEXT("^UPipelineTokenizer::InitializeTokenizerFromJson: failed to parse tokenizer JSON"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1); // the LogModelPipelinesTokenizer line (anchored: the pipeline line below quotes it mid-message)
			AddExpectedMessage(TEXT("UTextEmbeddingPipeline: tokenizer of package"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			FFixture F;
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, F, false, [](UPipelineModelPackage& P) { P.TokenizerJson = TEXT("{not json"); })) { return; }
			FPipelineResult R;
			UTextEmbeddingPipeline* P = UTextPipelineLibrary::CreateTextEmbeddingPipeline(F.Package.Get(), R);
			TestNull(TEXT("pipeline"), P);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestEqual(*FString::Printf(TEXT("Error PreprocessFailed (10), got %d (%s)"), (int32)R.Error, *R.Message), (int32)R.Error, 10);
			TestFalse(TEXT("Message not empty"), R.Message.IsEmpty());
		});

		LatentIt("PoolingNoneOnRank3Model", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// MiniLM emits [batch, seq, 384]; declaring Pooling None contradicts it. The first call must fail with ShapeMismatch
			// (task I/O check) and return no embeddings; not crash and not silently return a wrong-shaped result.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F, true, [](UPipelineModelPackage& P) { P.Preprocess.Pooling = EPipelinePooling::None; })) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), Pick(*F, {0, 1}), Done, [this](FCall& C)
			{
				TestFalse(TEXT("bSuccess"), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error ShapeMismatch, got %d (%s)"), (int32)C.Result.Error, *C.Result.Message), C.Result.Error == EPipelineError::ShapeMismatch);
				TestEqual(TEXT("no embeddings on failure"), C.Embs.Num(), 0);
			});
		});
	});

	// ------------------------------------------------------------------------------------------------------
	Describe("Calls", [this]()
	{
		LatentIt("EmptyTextList", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Empty input: InvalidArgument delivered through the callback, later, on the game thread (RunCall asserts the rest).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, F->Pipeline.Get(), TArray<FString>(), Done, [this](FCall& C)
			{
				TestFalse(TEXT("bSuccess"), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)C.Result.Error, *C.Result.Message), C.Result.Error == EPipelineError::InvalidArgument);
				TestEqual(TEXT("no embeddings"), C.Embs.Num(), 0);
			});
		});

		LatentIt("Over1024Texts", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 1025 texts exceed MaxTextsPerCall: InvalidArgument, nothing run.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			TArray<FString> Texts;
			Texts.Init(TEXT("hello"), 1025);
			RunCall(this, F, F->Pipeline.Get(), Texts, Done, [this](FCall& C)
			{
				TestFalse(TEXT("bSuccess"), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Error InvalidArgument, got %d (%s)"), (int32)C.Result.Error, *C.Result.Message), C.Result.Error == EPipelineError::InvalidArgument);
				TestEqual(TEXT("no embeddings"), C.Embs.Num(), 0);
			});
		});

		It("NullCallbackWarnsAndRunsNothing", [this]()
		{
			// A null callback is a user bug: one Warning, no work queued (GetNumPendingCalls() == 0), no crash.
			FFixture F;
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, F)) { return; }
			ExpectOneWarning(*this);
			F.Pipeline->EmbedAsync(Pick(F, {0}), FPipelineEmbeddingCallback());
			TestEqual(TEXT("GetNumPendingCalls"), F.Pipeline->GetNumPendingCalls(), 0);
		});

		LatentIt("DestroyedMidCall", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// The pipeline is marked garbage and collected right after EmbedAsync. In-flight work must still finish and fire the
			// callback once with correct embeddings (a base that dropped the callback, or stages touching the dead UObject, fails).
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			UTextEmbeddingPipeline* P = F->Pipeline.Get();
			RunCall(this, F, P, Pick(*F, {0, 1, 2, 3, 4}), Done, [this, F](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				CheckAgainstGolden(*this, *F, C.Embs, TArray<int32>{0, 1, 2, 3, 4}, TEXT("destroyed mid-call"), 384, 0.999);
			}, [F, P]()
			{
				P->MarkAsGarbage();
				F->Pipeline.Reset();
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			});
		});

		LatentIt("EightThreadsBlocking", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// 8 threads x 5 EmbedBlocking calls on one pipeline with different 4-sentence subsets: every result correct, no crash.
			// A shared tokenizer or tensor buffer without proper locking would corrupt or crash here.
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::CPU, *F)) { Done.ExecuteIfBound(); return; }
			struct FSlot { TArray<int32> Indices; FPipelineResult R; TArray<FPipelineEmbedding> E; };
			const TSharedRef<TArray<FSlot>> Slots = MakeShared<TArray<FSlot>>();
			Slots->SetNum(40);
			for (int32 t = 0; t < 8; ++t)
			{
				for (int32 c = 0; c < 5; ++c)
				{
					for (int32 k = 0; k < 4; ++k) { (*Slots)[t * 5 + c].Indices.Add((t * 7 + c * 11 + k * 13) % 50); }
				}
			}
			UTextEmbeddingPipeline* P = F->Pipeline.Get();
			RunWorker(this, F, Done, [F, Slots, P]()
			{
				TArray<TFuture<void>> Threads;
				for (int32 t = 0; t < 8; ++t)
				{
					Threads.Add(Async(EAsyncExecution::Thread, [F, Slots, P, t]()
					{
						for (int32 c = 0; c < 5; ++c)
						{
							FSlot& S = (*Slots)[t * 5 + c];
							TArray<FString> Texts;
							for (int32 I : S.Indices) { Texts.Add(F->Texts[I]); }
							S.R = P->EmbedBlocking(Texts, S.E);
						}
					}));
				}
				for (TFuture<void>& Th : Threads) { Th.Wait(); }
			}, [this, F, Slots]()
			{
				double Worst = 2.0;
				for (int32 s = 0; s < Slots->Num(); ++s)
				{
					const FSlot& S = (*Slots)[s];
					if (!TestTrue(*FString::Printf(TEXT("slot %d bSuccess (%s)"), s, *S.R.Message), S.R.bSuccess)) { continue; }
					if (!TestEqual(*FString::Printf(TEXT("slot %d count"), s), S.E.Num(), S.Indices.Num())) { continue; }
					for (int32 k = 0; k < S.Indices.Num(); ++k)
					{
						const TArray<double>* G = F->GoldenEmbedding.Find(F->Ids[S.Indices[k]]);
						Worst = FMath::Min(Worst, G ? Cosine(S.E[k].Values, *G) : -2.0);
					}
				}
				AddInfo(FString::Printf(TEXT("8 threads x 5 calls: worst cosine %.8f"), Worst));
				TestTrue(*FString::Printf(TEXT("worst cosine %.8f > 0.999"), Worst), Worst > 0.999);
			});
		});

		LatentIt("GpuUnavailableFallsBackToCpu", LatentLimit, [this](const FDoneDelegate& Done)
		{
			// Backend whose GPU runtime name does not exist + PreferredDevice GPU: must run on CPU (Result.Device == CPU)
			// with correct results, not fail and not claim GPU.
			AddExpectedMessage(TEXT("GPU unavailable for this model, falling back to CPU: GPU runtime '[^']*' is not registered"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
			const TSharedRef<FFixture> F = MakeShared<FFixture>();
			if (!MakeFixture(*this, MiniSlug, EPipelineDevice::GPU, *F, false)) { Done.ExecuteIfBound(); return; }
			FPipelineNNEBackendConfig Cfg;
			Cfg.GpuRuntimeName = TEXT("NNERuntimeBogusDoesNotExist");
			F->Backend = MakeShared<FPipelineNNEBackend>(Cfg);
			UTextEmbeddingPipeline* P = NewObject<UTextEmbeddingPipeline>(GetTransientPackage());
			F->Pipeline.Reset(P);
			const FPipelineResult Init = P->InitializeFromPackage(F->Package.Get(), *F->Backend);
			if (!TestTrue(*FString::Printf(TEXT("InitializeFromPackage (%s)"), *Init.Message), Init.bSuccess)) { Done.ExecuteIfBound(); return; }
			RunCall(this, F, P, Pick(*F, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), Done, [this, F](FCall& C)
			{
				TestTrue(*FString::Printf(TEXT("bSuccess (%s)"), *C.Result.Message), C.Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("Device == CPU after fallback, got %d"), (int32)C.Result.Device), C.Result.Device == EPipelineDevice::CPU);
				CheckAgainstGolden(*this, *F, C.Embs, TArray<int32>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, TEXT("fallback"), 384, 0.999);
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
