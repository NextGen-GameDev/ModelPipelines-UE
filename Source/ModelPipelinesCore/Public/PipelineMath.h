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
}
