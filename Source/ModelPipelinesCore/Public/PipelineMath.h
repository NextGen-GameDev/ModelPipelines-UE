// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Containers/Array.h"
#include "Containers/ArrayView.h"

/**
 * Numeric helpers shared by the task pipelines (docs/contracts.md C18).
 * Pure functions: callable from any thread, never assert on their inputs.
 * An Out array must not alias (share memory with) the input views of the same call.
 */
namespace UE::ModelPipelines
{
	/** Numerically stable softmax (subtract the max, accumulate in double). false (Out emptied) if Logits is empty or any
	    value is NaN/Inf. */
	MODELPIPELINESCORE_API bool Softmax(TConstArrayView<float> Logits, TArray<float>& Out);

	/** 1 / (1 + exp(-x)) per element, computed in double. Same failure rule as Softmax. */
	MODELPIPELINESCORE_API bool Sigmoid(TConstArrayView<float> Logits, TArray<float>& Out);

	/** Indices of the K largest values, best first; ties -> lower index first; NaN sorts last.
	    K <= 0 or empty -> empty; K > Num -> Num. */
	MODELPIPELINESCORE_API TArray<int32> TopKIndices(TConstArrayView<float> Values, int32 K);

	/** v / max(||v||2, 1e-12), norm accumulated in double (torch.nn.functional.normalize). A zero vector stays zero. */
	MODELPIPELINESCORE_API void L2NormalizeInPlace(TArrayView<float> Values);

	/** dot / (max(||a||,1e-12) * max(||b||,1e-12)) in double. Different lengths, or either empty -> 0. */
	MODELPIPELINESCORE_API double CosineSimilarity(TConstArrayView<float> A, TConstArrayView<float> B);

	/**
	 * Hidden is row-major [Batch, SeqLen, Dim]; Mask [Batch, SeqLen] (0 = padding, any other value counts as 1).
	 * MeanPool: Out[b] = sum_t(h[b,t]*m[b,t]) / max(sum_t m[b,t], 1e-9) (sentence-transformers Pooling). A row with no
	 *           tokens -> zeros. Sums in double.
	 * ClsPool:  Out[b] = h[b,0].
	 * Out is [Batch * Dim] row-major. false (Out emptied) if Batch, SeqLen or Dim < 1, Hidden.Num() != B*L*D,
	 * Mask.Num() != B*L, or the products overflow int64.
	 */
	MODELPIPELINESCORE_API bool MeanPool(TConstArrayView<float> Hidden, TConstArrayView<int32> Mask, int32 Batch, int32 SeqLen, int32 Dim, TArray<float>& Out);
	MODELPIPELINESCORE_API bool ClsPool(TConstArrayView<float> Hidden, int32 Batch, int32 SeqLen, int32 Dim, TArray<float>& Out);

	/** true if every value is finite (empty -> true). */
	MODELPIPELINESCORE_API bool AllFinite(TConstArrayView<float> Values);

	/** One NMS candidate: corners (X0, Y0) top-left, (X1, Y1) bottom-right, its score and class. */
	struct FPipelineNmsBox
	{
		float X0 = 0;
		float Y0 = 0;
		float X1 = 0;
		float Y1 = 0;
		float Score = 0;
		int32 ClassIndex = 0;
	};

	/** Intersection over union in double. Width/height of each box = max(0, X1 - X0), max(0, Y1 - Y0);
	    inter = max(0, min(AX1,BX1) - max(AX0,BX0)) * max(0, min(AY1,BY1) - max(AY0,BY0)); union = areaA + areaB - inter;
	    union <= 0 (or NaN) -> 0. */
	MODELPIPELINESCORE_API double BoxIoU(const FPipelineNmsBox& A, const FPipelineNmsBox& B);

	/** Greedy class-wise NMS. Visit boxes by Score descending (ties: lower input index first; NaN scores last, among
	    themselves by index). A box is kept unless an already kept box with the same ClassIndex has IoU > IouThreshold
	    (strictly greater). IouThreshold >= 1 -> nothing suppressed; NaN -> treated as 1. Returns kept input indices in
	    visit order, at most MaxKeep (MaxKeep <= 0 -> empty). Empty input -> empty. */
	MODELPIPELINESCORE_API TArray<int32> NonMaxSuppression(TConstArrayView<FPipelineNmsBox> Boxes, float IouThreshold, int32 MaxKeep);
}
