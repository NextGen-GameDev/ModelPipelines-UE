// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Array.h"
#include "Containers/ArrayView.h"
#include "Math/RandomStream.h"
#include "Math/UnrealMathUtility.h"
#include "PipelineMath.h"
#include <limits>

namespace PipelineMathSpecPrivate
{
	static const float NaNF = std::numeric_limits<float>::quiet_NaN();
	static const float InfF = std::numeric_limits<float>::infinity();

	/** Brief tolerance: |a-e| <= 1e-6 * max(1,|e|). */
	static bool Near(double A, double E)
	{
		return FMath::Abs(A - E) <= 1e-6 * FMath::Max(1.0, FMath::Abs(E));
	}

	/** Compares arrays; the failure message carries the worst deviation and its index. */
	static void ExpectNear(FAutomationTestBase& T, const TCHAR* What, const TArray<float>& Actual, const TArray<double>& Expected)
	{
		if (Actual.Num() != Expected.Num())
		{
			T.AddError(FString::Printf(TEXT("%s: length %d, expected %d"), What, Actual.Num(), Expected.Num()));
			return;
		}
		double Worst = 0.0;
		int32 WorstIdx = -1;
		bool bOk = true;
		for (int32 i = 0; i < Actual.Num(); ++i)
		{
			const double Dev = FMath::Abs((double)Actual[i] - Expected[i]) / FMath::Max(1.0, FMath::Abs(Expected[i]));
			if (FMath::IsNaN(Actual[i]) || !(Dev <= 1e-6)) { bOk = false; }
			if (FMath::IsNaN(Actual[i]) || Dev > Worst || WorstIdx < 0) { Worst = FMath::IsNaN(Actual[i]) ? 1e30 : Dev; WorstIdx = i; }
		}
		if (!bOk)
		{
			T.AddError(FString::Printf(TEXT("%s: worst relative deviation %.3g at index %d (tolerance 1e-6)"), What, Worst, WorstIdx));
		}
	}

	static TArray<float> Arr(std::initializer_list<float> L) { return TArray<float>(L); }
}

BEGIN_DEFINE_SPEC(FPipelineMathSpec, "Pipelines.Core.Math",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FPipelineMathSpec)

void FPipelineMathSpec::Define()
{
	using namespace PipelineMathSpecPrivate;
	using namespace UE::ModelPipelines;

	Describe("Softmax", [this]()
	{
		It("matches the reference for {1,2,3} and sums to 1", [this]()
		{
			TArray<float> Out;
			TestTrue("returns true", Softmax(Arr({1.f, 2.f, 3.f}), Out));
			ExpectNear(*this, TEXT("values"), Out, {0.09003057, 0.24472847, 0.66524096});
			double Sum = 0.0;
			for (float V : Out) { Sum += V; }
			TestTrue(*FString::Printf(TEXT("sum %.9f within 1e-6 of 1"), Sum), FMath::Abs(Sum - 1.0) <= 1e-6);
		});

		It("does not overflow for large equal logits (catches exp without max subtraction)", [this]()
		{
			TArray<float> Out;
			TestTrue("returns true", Softmax(Arr({1000.f, 1000.f}), Out));
			ExpectNear(*this, TEXT("values"), Out, {0.5, 0.5});
		});

		It("handles very negative logits without NaN", [this]()
		{
			TArray<float> Out;
			TestTrue("returns true", Softmax(Arr({-1000.f, 0.f}), Out));
			TestEqual("length", Out.Num(), 2);
			if (Out.Num() == 2)
			{
				TestFalse("no NaN", FMath::IsNaN(Out[0]) || FMath::IsNaN(Out[1]));
				TestTrue("first ~0", Out[0] >= 0.f && Out[0] < 1e-6f);
				TestTrue("second ~1", Near(Out[1], 1.0));
			}
		});

		It("single element is 1", [this]()
		{
			TArray<float> Out;
			TestTrue("returns true", Softmax(Arr({5.f}), Out));
			TestEqual("length", Out.Num(), 1);
			if (Out.Num() == 1) { TestTrue("value 1", Near(Out[0], 1.0)); }
		});

		It("empty input fails and empties Out", [this]()
		{
			TArray<float> Out = {9.f, 9.f};
			TestFalse("returns false", Softmax(TConstArrayView<float>(), Out));
			TestEqual("Out emptied", Out.Num(), 0);
		});

		It("NaN or Inf input fails and empties a pre-filled Out", [this]()
		{
			// A softmax that propagated NaN, or left stale output, would hand garbage scores to a classifier.
			TArray<float> Out = {1.f, 2.f, 3.f};
			TestFalse("NaN returns false", Softmax(Arr({1.f, NaNF}), Out));
			TestEqual("Out emptied after NaN", Out.Num(), 0);
			Out = {1.f, 2.f, 3.f};
			TestFalse("+Inf returns false", Softmax(Arr({InfF, 0.f}), Out));
			TestEqual("Out emptied after Inf", Out.Num(), 0);
		});
	});

	Describe("Sigmoid", [this]()
	{
		It("matches the reference for {0,2,-2}", [this]()
		{
			TArray<float> Out;
			TestTrue("returns true", Sigmoid(Arr({0.f, 2.f, -2.f}), Out));
			ExpectNear(*this, TEXT("values"), Out, {0.5, 0.88079708, 0.11920292});
		});

		It("saturates finitely for +-100", [this]()
		{
			TArray<float> Out;
			TestTrue("returns true", Sigmoid(Arr({100.f, -100.f}), Out));
			TestEqual("length", Out.Num(), 2);
			if (Out.Num() == 2)
			{
				TestTrue("finite", FMath::IsFinite(Out[0]) && FMath::IsFinite(Out[1]));
				TestTrue("first ~1", Near(Out[0], 1.0));
				TestTrue("second ~0", Out[1] >= 0.f && Out[1] < 1e-6f);
			}
		});

		It("NaN input fails and empties Out", [this]()
		{
			TArray<float> Out = {1.f};
			TestFalse("returns false", Sigmoid(Arr({0.f, NaNF}), Out));
			TestEqual("Out emptied", Out.Num(), 0);
		});
	});

	Describe("TopKIndices", [this]()
	{
		It("returns the K largest, best first", [this]()
		{
			TestTrue(*FString("{0.1,0.7,0.2} k=2"), TopKIndices(Arr({0.1f, 0.7f, 0.2f}), 2) == TArray<int32>({1, 2}));
		});

		It("breaks ties by lower index", [this]()
		{
			TestTrue(*FString("{0.5,0.5,0.9} k=3"), TopKIndices(Arr({0.5f, 0.5f, 0.9f}), 3) == TArray<int32>({2, 0, 1}));
		});

		It("K <= 0 or empty input gives empty", [this]()
		{
			const TArray<float> V = Arr({0.1f, 0.7f, 0.2f});
			TestEqual("k=0", TopKIndices(V, 0).Num(), 0);
			TestEqual("k=-3", TopKIndices(V, -3).Num(), 0);
			TestEqual("empty values", TopKIndices(TConstArrayView<float>(), 2).Num(), 0);
		});

		It("K above Num is clamped", [this]()
		{
			TestTrue(*FString("{0.3,0.1} k=10"), TopKIndices(Arr({0.3f, 0.1f}), 10) == TArray<int32>({0, 1}));
		});

		It("NaN sorts last", [this]()
		{
			TestTrue(*FString("{NaN,0.2,0.1} k=3"), TopKIndices(Arr({NaNF, 0.2f, 0.1f}), 3) == TArray<int32>({1, 2, 0}));
		});
	});

	Describe("L2NormalizeInPlace", [this]()
	{
		It("normalizes {3,4} to {0.6,0.8}", [this]()
		{
			TArray<float> V = Arr({3.f, 4.f});
			L2NormalizeInPlace(V);
			ExpectNear(*this, TEXT("values"), V, {0.6, 0.8});
		});

		It("leaves a zero vector at zero (no NaN from 0/0)", [this]()
		{
			TArray<float> V = Arr({0.f, 0.f, 0.f});
			L2NormalizeInPlace(V);
			for (float X : V) { TestEqual("component is 0", X, 0.f); }
		});

		It("clamps the norm at 1e-12 like torch (tiny vector)", [this]()
		{
			// {1e-30,0}: norm 1e-30 < eps, so divide by 1e-12 -> 1e-18. Dividing by the true norm would give 1.
			TArray<float> V = Arr({1e-30f, 0.f});
			L2NormalizeInPlace(V);
			TestTrue(*FString::Printf(TEXT("V[0]=%g expected ~1e-18"), V[0]), FMath::Abs((double)V[0] - 1e-18) <= 1e-24);
			TestEqual("V[1]", V[1], 0.f);
		});

		It("empty is a no-op", [this]()
		{
			TArray<float> V;
			L2NormalizeInPlace(V);
			TestEqual("still empty", V.Num(), 0);
		});
	});

	Describe("CosineSimilarity", [this]()
	{
		It("covers the acceptance table", [this]()
		{
			TestTrue("orthogonal -> 0", Near(CosineSimilarity(Arr({1.f, 0.f}), Arr({0.f, 1.f})), 0.0));
			TestTrue("parallel -> 1", Near(CosineSimilarity(Arr({1.f, 2.f, 3.f}), Arr({2.f, 4.f, 6.f})), 1.0));
			TestTrue("opposite -> -1", Near(CosineSimilarity(Arr({1.f, 2.f}), Arr({-1.f, -2.f})), -1.0));
			TestEqual("length mismatch -> 0", CosineSimilarity(Arr({1.f, 2.f}), Arr({1.f, 2.f, 3.f})), 0.0);
			TestEqual("both empty -> 0", CosineSimilarity(TConstArrayView<float>(), TConstArrayView<float>()), 0.0);
			TestEqual("zero vector -> 0 (not NaN)", CosineSimilarity(Arr({0.f, 0.f}), Arr({1.f, 1.f})), 0.0);
		});

		It("non-trivial angle: {1,0} vs {1,1} is 1/sqrt(2)", [this]()
		{
			TestTrue("0.70710678", Near(CosineSimilarity(Arr({1.f, 0.f}), Arr({1.f, 1.f})), 0.70710678118654752));
		});
	});

	Describe("Pooling", [this]()
	{
		static const TArray<float> Hidden = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

		It("MeanPool ignores masked tokens: [2,3, 7,8]", [this]()
		{
			// Counting padding tokens would give [3,4, 9,10] instead.
			TArray<float> Out;
			TestTrue("returns true", MeanPool(Hidden, TArray<int32>({1, 1, 0, 1, 0, 0}), 2, 3, 2, Out));
			ExpectNear(*this, TEXT("values"), Out, {2, 3, 7, 8});
		});

		It("ClsPool takes token 0: [1,2, 7,8]", [this]()
		{
			TArray<float> Out;
			TestTrue("returns true", ClsPool(Hidden, 2, 3, 2, Out));
			ExpectNear(*this, TEXT("values"), Out, {1, 2, 7, 8});
		});

		It("any non-zero mask value counts as 1", [this]()
		{
			// A weighted mean (mask 2 as weight) would give [(2+6)/3...] instead of [2,3].
			TArray<float> Out;
			TestTrue("returns true", MeanPool(Hidden, TArray<int32>({2, 1, 0, 1, 0, 0}), 2, 3, 2, Out));
			ExpectNear(*this, TEXT("values"), Out, {2, 3, 7, 8});
		});

		It("a fully masked row yields zeros and leaves other rows alone", [this]()
		{
			TArray<float> Out;
			TestTrue("returns true", MeanPool(Hidden, TArray<int32>({0, 0, 0, 1, 0, 0}), 2, 3, 2, Out));
			ExpectNear(*this, TEXT("values"), Out, {0, 0, 7, 8});
		});

		It("MeanPool rejects bad sizes and empties Out", [this]()
		{
			const TArray<int32> Mask = {1, 1, 0, 1, 0, 0};
			TArray<float> Out = {1.f};
			TestFalse("hidden too short", MeanPool(TConstArrayView<float>(Hidden.GetData(), 11), Mask, 2, 3, 2, Out));
			TestEqual("emptied (hidden)", Out.Num(), 0);
			Out = {1.f};
			TestFalse("mask too short", MeanPool(Hidden, TConstArrayView<int32>(Mask.GetData(), 5), 2, 3, 2, Out));
			TestEqual("emptied (mask)", Out.Num(), 0);
			Out = {1.f};
			TestFalse("batch 0", MeanPool(TConstArrayView<float>(), TConstArrayView<int32>(), 0, 3, 2, Out));
			TestEqual("emptied (batch 0)", Out.Num(), 0);
			Out = {1.f};
			TestFalse("seq 0", MeanPool(TConstArrayView<float>(), TConstArrayView<int32>(), 2, 0, 2, Out));
			TestFalse("dim 0", MeanPool(TConstArrayView<float>(), TConstArrayView<int32>(), 2, 3, 0, Out));
			TestFalse("negative dim", MeanPool(Hidden, Mask, 2, 3, -2, Out));
			TestEqual("emptied (dim)", Out.Num(), 0);
		});

		It("ClsPool rejects bad sizes and empties Out", [this]()
		{
			TArray<float> Out = {1.f};
			TestFalse("hidden too short", ClsPool(TConstArrayView<float>(Hidden.GetData(), 11), 2, 3, 2, Out));
			TestEqual("emptied", Out.Num(), 0);
			Out = {1.f};
			TestFalse("hidden too long", ClsPool(Hidden, 1, 3, 2, Out));
			TestEqual("emptied (long)", Out.Num(), 0);
			TestFalse("batch 0", ClsPool(Hidden, 0, 3, 2, Out));
			TestFalse("seq 0", ClsPool(Hidden, 2, 0, 2, Out));
			TestFalse("dim 0", ClsPool(Hidden, 2, 3, 0, Out));
		});

		It("MeanPool of 1x256x384 random data matches a double reference", [this]()
		{
			const int32 B = 1, L = 256, D = 384;
			FRandomStream Rng(12345);
			TArray<float> H;
			H.SetNumUninitialized(B * L * D);
			for (float& X : H) { X = Rng.FRandRange(-3.f, 3.f); }
			TArray<int32> Mask;
			Mask.SetNumUninitialized(B * L);
			int32 Count = 0;
			for (int32 t = 0; t < L; ++t) { Mask[t] = (t < 200) ? 1 : 0; Count += Mask[t]; }

			TArray<float> Out;
			TestTrue("returns true", MeanPool(H, Mask, B, L, D, Out));
			TestEqual("Out size", Out.Num(), D);
			if (Out.Num() != D) { return; }
			double Worst = 0.0;
			for (int32 d = 0; d < D; ++d)
			{
				double Sum = 0.0;
				for (int32 t = 0; t < L; ++t) { Sum += (double)H[t * D + d] * (Mask[t] ? 1.0 : 0.0); }
				const double E = Sum / (double)Count;
				Worst = FMath::Max(Worst, FMath::Abs((double)Out[d] - E) / FMath::Max(1.0, FMath::Abs(E)));
			}
			TestTrue(*FString::Printf(TEXT("worst relative deviation %.3g (tolerance 1e-6)"), Worst), Worst <= 1e-6);
		});
	});

	Describe("AllFinite", [this]()
	{
		It("classifies finite, empty, NaN and Inf", [this]()
		{
			TestTrue("empty", AllFinite(TConstArrayView<float>()));
			TestTrue("{1,2}", AllFinite(Arr({1.f, 2.f})));
			TestFalse("{1,NaN}", AllFinite(Arr({1.f, NaNF})));
			TestFalse("{Inf}", AllFinite(Arr({InfF})));
			TestFalse("{-Inf}", AllFinite(Arr({-InfF})));
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
