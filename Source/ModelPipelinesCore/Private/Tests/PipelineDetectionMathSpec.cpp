// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.
// WP4.3 (docs/contracts.md C26): FPipelineDetection defaults, BoxIoU, class-wise NonMaxSuppression.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Array.h"
#include "Containers/ArrayView.h"
#include "Math/Box2D.h"
#include "PipelineResults.h"
#include "PipelineMath.h"
#include <limits>

namespace PipelineDetectionMathSpecPrivate
{
	using UE::ModelPipelines::FPipelineNmsBox;

	static const float NaNF = std::numeric_limits<float>::quiet_NaN();

	static FPipelineNmsBox Box(float X0, float Y0, float X1, float Y1, float Score = 0.f, int32 Class = 0)
	{
		FPipelineNmsBox B;
		B.X0 = X0; B.Y0 = Y0; B.X1 = X1; B.Y1 = Y1; B.Score = Score; B.ClassIndex = Class;
		return B;
	}

	static FString Str(const TArray<int32>& A)
	{
		FString S = TEXT("[");
		for (int32 i = 0; i < A.Num(); ++i) { if (i) { S += TEXT(","); } S += FString::Printf(TEXT("%d"), A[i]); }
		return S + TEXT("]");
	}

	static void ExpectIdx(FAutomationTestBase& T, const TCHAR* What, const TArray<int32>& Actual, const TArray<int32>& Expected)
	{
		T.TestTrue(FString::Printf(TEXT("%s: expected %s, got %s"), What, *Str(Expected), *Str(Actual)), Actual == Expected);
	}

	static void ExpectIoU(FAutomationTestBase& T, const TCHAR* What, const FPipelineNmsBox& A, const FPipelineNmsBox& B, double Expected, double Tol = 1e-9)
	{
		const double Got = UE::ModelPipelines::BoxIoU(A, B);
		T.TestTrue(FString::Printf(TEXT("%s: expected IoU %.12g, got %.12g (diff %.3g, tol %.3g)"), What, Expected, Got, FMath::Abs(Got - Expected), Tol),
			!FMath::IsNaN(Got) && FMath::Abs(Got - Expected) <= Tol);
	}
}

BEGIN_DEFINE_SPEC(FPipelineDetectionMathSpec, "Pipelines.Core.Detection",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FPipelineDetectionMathSpec)

void FPipelineDetectionMathSpec::Define()
{
	using namespace PipelineDetectionMathSpecPrivate;
	using UE::ModelPipelines::BoxIoU;
	using UE::ModelPipelines::NonMaxSuppression;

	Describe("FPipelineDetection", [this]()
	{
		// A default of Index 0 would claim "class 0" for an unset detection; an invalid box must not look like a real one.
		It("defaults: Index -1, Score 0, empty label, Box.bIsValid false", [this]()
		{
			FPipelineDetection D;
			TestEqual(TEXT("Index default"), D.Index, -1);
			TestEqual(TEXT("Score default"), D.Score, 0.f);
			TestTrue(TEXT("Label default empty"), D.Label.IsEmpty());
			TestFalse(TEXT("Box.bIsValid default"), D.Box.bIsValid != 0);
		});

		It("fields keep what is assigned (Box Min=(left,top), Max=(right,bottom))", [this]()
		{
			FPipelineDetection D;
			D.Label = TEXT("cat");
			D.Index = 3;
			D.Score = 0.75f;
			D.Box = FBox2D(FVector2D(0.1, 0.2), FVector2D(0.6, 0.9));
			TestEqual(TEXT("Label"), D.Label, FString(TEXT("cat")));
			TestEqual(TEXT("Index"), D.Index, 3);
			TestEqual(TEXT("Score"), D.Score, 0.75f);
			TestTrue(TEXT("Box valid"), D.Box.bIsValid != 0);
			TestEqual(TEXT("Box.Min.X"), D.Box.Min.X, 0.1);
			TestEqual(TEXT("Box.Max.Y"), D.Box.Max.Y, 0.9);
		});
	});

	Describe("BoxIoU", [this]()
	{
		// Half-overlap: inter 0.5, union 1.5. Catches IoU computed over the smaller box (0.5) or as inter/areaA.
		It("(0,0,1,1) vs (0.5,0,1.5,1) = 1/3", [this]()
		{
			ExpectIoU(*this, TEXT("half overlap"), Box(0, 0, 1, 1), Box(0.5f, 0, 1.5f, 1), 1.0 / 3.0);
		});

		It("argument order does not matter", [this]()
		{
			ExpectIoU(*this, TEXT("swapped"), Box(0.5f, 0, 1.5f, 1), Box(0, 0, 1, 1), 1.0 / 3.0);
		});

		It("identical boxes = 1", [this]()
		{
			ExpectIoU(*this, TEXT("identical"), Box(0.1f, 0.2f, 0.7f, 0.9f), Box(0.1f, 0.2f, 0.7f, 0.9f), 1.0, 1e-12);
		});

		It("disjoint boxes = 0", [this]()
		{
			ExpectIoU(*this, TEXT("disjoint"), Box(0, 0, 1, 1), Box(2, 2, 3, 3), 0.0, 0.0);
		});

		// Negative intersection width must clamp to 0, not multiply two negatives into a positive.
		It("disjoint in both axes (negative widths and heights) = 0, not a positive product", [this]()
		{
			ExpectIoU(*this, TEXT("both-axis gap"), Box(0, 0, 1, 1), Box(5, 7, 6, 8), 0.0, 0.0);
		});

		It("touching edge = 0", [this]()
		{
			ExpectIoU(*this, TEXT("touching"), Box(0, 0, 1, 1), Box(1, 0, 2, 1), 0.0, 0.0);
		});

		It("contained box: IoU = small area / big area (0.25 for a 0.5x0.5 box inside a 1x1)", [this]()
		{
			ExpectIoU(*this, TEXT("contained"), Box(0, 0, 1, 1), Box(0.25f, 0.25f, 0.75f, 0.75f), 0.25);
		});

		// Width = max(0, X1-X0) = 0 for X1 < X0; a naive negative area would give a bogus union.
		It("degenerate box (X1 < X0) vs any = 0", [this]()
		{
			ExpectIoU(*this, TEXT("degenerate vs unit"), Box(1, 0, 0, 1), Box(0, 0, 1, 1), 0.0, 0.0);
			ExpectIoU(*this, TEXT("unit vs degenerate"), Box(0, 0, 1, 1), Box(1, 0, 0, 1), 0.0, 0.0);
			ExpectIoU(*this, TEXT("degenerate Y vs unit"), Box(0, 1, 1, 0), Box(0, 0, 1, 1), 0.0, 0.0);
		});

		It("two zero-area boxes: union 0 -> 0, never NaN or a division by zero", [this]()
		{
			ExpectIoU(*this, TEXT("zero area identical"), Box(0.5f, 0.5f, 0.5f, 0.5f), Box(0.5f, 0.5f, 0.5f, 0.5f), 0.0, 0.0);
		});

		It("NaN coordinate -> 0", [this]()
		{
			ExpectIoU(*this, TEXT("NaN"), Box(NaNF, 0, 1, 1), Box(0, 0, 1, 1), 0.0, 0.0);
		});

		// A float-only implementation would be off by ~1e-8 for 0.1-based boxes; 1e-6 is the float-input noise bound.
		It("(0,0,1,1) vs (0.1,0,1.1,1) = 0.9/1.1 = 0.818181...", [this]()
		{
			ExpectIoU(*this, TEXT("shifted by 0.1"), Box(0, 0, 1, 1), Box(0.1f, 0, 1.1f, 1), 9.0 / 11.0, 1e-6);
		});
	});

	Describe("NonMaxSuppression", [this]()
	{
		// A(0,0,1,1,.9,c0), B(0.1,0,1.1,1,.8,c0) IoU 0.818, C = A's box in class 1, score .7.
		auto MakeABC = []() -> TArray<FPipelineNmsBox>
		{
			return {Box(0, 0, 1, 1, 0.9f, 0), Box(0.1f, 0, 1.1f, 1, 0.8f, 0), Box(0, 0, 1, 1, 0.7f, 1)};
		};

		// B overlaps A (same class) -> dropped; C is the same box but another class -> kept (class-wise, not global NMS).
		It("A,B,C at thr 0.5 -> [0,2] kept", [this, MakeABC]()
		{
			const TArray<FPipelineNmsBox> Boxes = MakeABC();
			ExpectIdx(*this, TEXT("thr 0.5"), NonMaxSuppression(Boxes, 0.5f, 100), {0, 2});
		});

		It("A,B,C at thr 0.9 (above IoU 0.818) -> [0,1,2] kept", [this, MakeABC]()
		{
			const TArray<FPipelineNmsBox> Boxes = MakeABC();
			ExpectIdx(*this, TEXT("thr 0.9"), NonMaxSuppression(Boxes, 0.9f, 100), {0, 1, 2});
		});

		It("thr 1 and thr 2 suppress nothing (>= 1 disables NMS), even for identical boxes", [this, MakeABC]()
		{
			const TArray<FPipelineNmsBox> Boxes = MakeABC();
			ExpectIdx(*this, TEXT("thr 1"), NonMaxSuppression(Boxes, 1.f, 100), {0, 1, 2});
			ExpectIdx(*this, TEXT("thr 2"), NonMaxSuppression(Boxes, 2.f, 100), {0, 1, 2});
			const TArray<FPipelineNmsBox> Same = {Box(0, 0, 1, 1, 0.9f), Box(0, 0, 1, 1, 0.8f)};
			ExpectIdx(*this, TEXT("identical boxes, thr 1"), NonMaxSuppression(Same, 1.f, 100), {0, 1});
		});

		It("MaxKeep 1 -> [0]; MaxKeep 2 with thr 1 -> [0,1] kept", [this, MakeABC]()
		{
			const TArray<FPipelineNmsBox> Boxes = MakeABC();
			ExpectIdx(*this, TEXT("MaxKeep 1"), NonMaxSuppression(Boxes, 0.5f, 1), {0});
			ExpectIdx(*this, TEXT("MaxKeep 2 thr 1"), NonMaxSuppression(Boxes, 1.f, 2), {0, 1});
		});

		It("MaxKeep 0 and negative -> empty", [this, MakeABC]()
		{
			const TArray<FPipelineNmsBox> Boxes = MakeABC();
			ExpectIdx(*this, TEXT("MaxKeep 0"), NonMaxSuppression(Boxes, 0.5f, 0), {});
			ExpectIdx(*this, TEXT("MaxKeep -5"), NonMaxSuppression(Boxes, 0.5f, -5), {});
		});

		It("empty input -> empty", [this]()
		{
			const TArray<FPipelineNmsBox> None;
			ExpectIdx(*this, TEXT("empty"), NonMaxSuppression(None, 0.5f, 10), {});
		});

		// Strictly greater: IoU 0.5 exactly (inter 0.5, union 1.0, all exact in binary) at thr 0.5 stays; just below drops.
		It("IoU exactly equal to the threshold is kept (0.5 vs 0.5); thr 0.49 drops it", [this]()
		{
			const TArray<FPipelineNmsBox> Boxes = {Box(0, 0, 1, 1, 0.9f), Box(0, 0, 0.5f, 1, 0.8f)};
			ExpectIdx(*this, TEXT("equal"), NonMaxSuppression(Boxes, 0.5f, 10), {0, 1});
			ExpectIdx(*this, TEXT("just below"), NonMaxSuppression(Boxes, 0.49f, 10), {0});
		});

		It("output is in visit order (score descending), not input order", [this]()
		{
			const TArray<FPipelineNmsBox> Boxes = {Box(0, 0, 1, 1, 0.5f), Box(2, 0, 3, 1, 0.9f), Box(4, 0, 5, 1, 0.7f)};
			ExpectIdx(*this, TEXT("order"), NonMaxSuppression(Boxes, 0.5f, 10), {1, 2, 0});
		});

		// The higher index is the one suppressed, so an unstable sort or a "later wins" rule is caught.
		It("score ties keep input order: identical overlapping boxes with equal score -> the lower index survives", [this]()
		{
			const TArray<FPipelineNmsBox> Boxes = {Box(0, 0, 1, 1, 0.8f), Box(0, 0, 1, 1, 0.8f), Box(0, 0, 1, 1, 0.8f)};
			ExpectIdx(*this, TEXT("tie overlap"), NonMaxSuppression(Boxes, 0.5f, 10), {0});
		});

		It("score ties among disjoint boxes keep input order [0,1,2] kept", [this]()
		{
			const TArray<FPipelineNmsBox> Boxes = {Box(0, 0, 1, 1, 0.8f), Box(2, 0, 3, 1, 0.8f), Box(4, 0, 5, 1, 0.8f)};
			ExpectIdx(*this, TEXT("tie disjoint"), NonMaxSuppression(Boxes, 0.5f, 10), {0, 1, 2});
		});

		It("NaN score is visited last (and suppressed by a kept box, even though it comes first in the input)", [this]()
		{
			const TArray<FPipelineNmsBox> Disjoint = {Box(0, 0, 1, 1, NaNF), Box(2, 0, 3, 1, 0.5f), Box(4, 0, 5, 1, 0.9f)};
			ExpectIdx(*this, TEXT("NaN last"), NonMaxSuppression(Disjoint, 0.5f, 10), {2, 1, 0});
			const TArray<FPipelineNmsBox> Overlap = {Box(0, 0, 1, 1, NaNF), Box(0, 0, 1, 1, 0.1f)};
			ExpectIdx(*this, TEXT("NaN suppressed"), NonMaxSuppression(Overlap, 0.5f, 10), {1});
			const TArray<FPipelineNmsBox> TwoNaN = {Box(0, 0, 1, 1, NaNF), Box(2, 0, 3, 1, 0.2f), Box(4, 0, 5, 1, NaNF)};
			ExpectIdx(*this, TEXT("NaNs by index"), NonMaxSuppression(TwoNaN, 0.5f, 10), {1, 0, 2});
		});

		It("NaN threshold = no suppression", [this]()
		{
			const TArray<FPipelineNmsBox> Boxes = {Box(0, 0, 1, 1, 0.9f), Box(0, 0, 1, 1, 0.8f)};
			ExpectIdx(*this, TEXT("NaN thr"), NonMaxSuppression(Boxes, NaNF, 10), {0, 1});
		});

		// Suppression needs IoU strictly > 0: touching boxes (IoU 0) survive even at thr 0, overlapping ones do not.
		It("threshold 0: any real overlap suppresses, touching edges do not", [this]()
		{
			const TArray<FPipelineNmsBox> Boxes = {Box(0, 0, 1, 1, 0.9f), Box(1, 0, 2, 1, 0.8f), Box(0.9f, 0, 0.95f, 1, 0.7f)};
			ExpectIdx(*this, TEXT("thr 0"), NonMaxSuppression(Boxes, 0.f, 10), {0, 1});
		});

		// Greedy, not transitive: B (suppressed by A) must not suppress C. A-B IoU 1/3 > 0.3, B-C 1/3, A-C 0.
		It("a suppressed box does not suppress others: A(.9),B(.8),C(.7) chain at thr 0.3 -> [0,2] kept", [this]()
		{
			const TArray<FPipelineNmsBox> Boxes = {Box(0, 0, 1, 1, 0.9f), Box(0.5f, 0, 1.5f, 1, 0.8f), Box(1, 0, 2, 1, 0.7f)};
			ExpectIdx(*this, TEXT("chain"), NonMaxSuppression(Boxes, 0.3f, 10), {0, 2});
		});

		It("different classes never suppress each other", [this]()
		{
			const TArray<FPipelineNmsBox> Boxes = {Box(0, 0, 1, 1, 0.9f, 0), Box(0, 0, 1, 1, 0.8f, 1), Box(0, 0, 1, 1, 0.7f, 2), Box(0, 0, 1, 1, 0.6f, 1)};
			ExpectIdx(*this, TEXT("classes"), NonMaxSuppression(Boxes, 0.5f, 10), {0, 1, 2});
		});

		It("MaxKeep limits the output after suppression: 4 disjoint boxes, MaxKeep 3 -> top 3 by score", [this]()
		{
			const TArray<FPipelineNmsBox> Boxes = {Box(0, 0, 1, 1, 0.1f), Box(2, 0, 3, 1, 0.4f), Box(4, 0, 5, 1, 0.3f), Box(6, 0, 7, 1, 0.2f)};
			ExpectIdx(*this, TEXT("MaxKeep 3"), NonMaxSuppression(Boxes, 0.5f, 3), {1, 2, 3});
		});

		It("input is not modified (const view) and a larger call of 500 boxes finishes with every box kept when disjoint", [this]()
		{
			TArray<FPipelineNmsBox> Boxes;
			for (int32 i = 0; i < 500; ++i) { Boxes.Add(Box((float)i * 2.f, 0, (float)i * 2.f + 1.f, 1, 1.f - 0.001f * (float)i, i % 3)); }
			const TArray<int32> Out = NonMaxSuppression(Boxes, 0.5f, 10000);
			TestEqual(TEXT("all disjoint boxes are kept"), Out.Num(), 500);
			TestEqual(TEXT("first is the highest score (index 0)"), Out.Num() ? Out[0] : -1, 0);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
