// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineMath.h"

#include "Math/UnrealMathUtility.h"

namespace UE::ModelPipelines
{
	namespace MathPrivate
	{
		static constexpr double NormEps = 1e-12;
		static constexpr double MaskEps = 1e-9;

		static double SquaredNorm(TConstArrayView<float> V)
		{
			double Sum = 0.0;
			for (const float X : V)
			{
				Sum += (double)X * (double)X;
			}
			return Sum;
		}

		/** Checks B, L, D >= 1 and computes B*L and B*L*D without int64 overflow. */
		static bool PoolSizes(int32 Batch, int32 SeqLen, int32 Dim, int64& OutBL, int64& OutBLD)
		{
			if (Batch < 1 || SeqLen < 1 || Dim < 1)
			{
				return false;
			}
			// Each factor < 2^31, so B*L < 2^62 fits; only the third multiplication can overflow.
			OutBL = (int64)Batch * (int64)SeqLen;
			if (OutBL > MAX_int64 / (int64)Dim)
			{
				return false;
			}
			OutBLD = OutBL * (int64)Dim;
			return true;
		}
	}

	bool AllFinite(TConstArrayView<float> Values)
	{
		for (const float X : Values)
		{
			if (!FMath::IsFinite(X))
			{
				return false;
			}
		}
		return true;
	}

	bool Softmax(TConstArrayView<float> Logits, TArray<float>& Out)
	{
		Out.Reset();
		if (Logits.Num() == 0 || !AllFinite(Logits))
		{
			return false;
		}

		double Max = (double)Logits[0];
		for (const float X : Logits)
		{
			Max = FMath::Max(Max, (double)X);
		}

		double Sum = 0.0;
		for (const float X : Logits)
		{
			Sum += FMath::Exp((double)X - Max);
		}
		// Sum >= 1 (the max term is exp(0)), so the division is safe.
		Out.SetNumUninitialized(Logits.Num());
		for (int32 i = 0; i < Logits.Num(); ++i)
		{
			Out[i] = (float)(FMath::Exp((double)Logits[i] - Max) / Sum);
		}
		return true;
	}

	bool Sigmoid(TConstArrayView<float> Logits, TArray<float>& Out)
	{
		Out.Reset();
		if (Logits.Num() == 0 || !AllFinite(Logits))
		{
			return false;
		}
		Out.SetNumUninitialized(Logits.Num());
		for (int32 i = 0; i < Logits.Num(); ++i)
		{
			Out[i] = (float)(1.0 / (1.0 + FMath::Exp(-(double)Logits[i])));
		}
		return true;
	}

	TArray<int32> TopKIndices(TConstArrayView<float> Values, int32 K)
	{
		TArray<int32> Result;
		if (K <= 0 || Values.Num() == 0)
		{
			return Result;
		}

		Result.SetNumUninitialized(Values.Num());
		for (int32 i = 0; i < Values.Num(); ++i)
		{
			Result[i] = i;
		}
		// Strict total order: non-NaN before NaN, larger value first, then lower index.
		Result.Sort([Values](const int32 IA, const int32 IB)
		{
			const float VA = Values[IA];
			const float VB = Values[IB];
			const bool bNaNA = FMath::IsNaN(VA);
			const bool bNaNB = FMath::IsNaN(VB);
			if (bNaNA != bNaNB)
			{
				return !bNaNA;
			}
			if (!bNaNA && VA != VB)
			{
				return VA > VB;
			}
			return IA < IB;
		});
		if (K < Result.Num())
		{
			Result.SetNum(K);
		}
		return Result;
	}

	void L2NormalizeInPlace(TArrayView<float> Values)
	{
		if (Values.Num() == 0)
		{
			return;
		}
		const double Norm = FMath::Sqrt(MathPrivate::SquaredNorm(Values));
		const double Denom = FMath::Max(Norm, MathPrivate::NormEps);
		for (float& X : Values)
		{
			X = (float)((double)X / Denom);
		}
	}

	double CosineSimilarity(TConstArrayView<float> A, TConstArrayView<float> B)
	{
		if (A.Num() == 0 || B.Num() == 0 || A.Num() != B.Num())
		{
			return 0.0;
		}
		double Dot = 0.0;
		double NormA = 0.0;
		double NormB = 0.0;
		for (int32 i = 0; i < A.Num(); ++i)
		{
			const double X = (double)A[i];
			const double Y = (double)B[i];
			Dot += X * Y;
			NormA += X * X;
			NormB += Y * Y;
		}
		const double Denom = FMath::Max(FMath::Sqrt(NormA), MathPrivate::NormEps) * FMath::Max(FMath::Sqrt(NormB), MathPrivate::NormEps);
		return Dot / Denom;
	}

	bool MeanPool(TConstArrayView<float> Hidden, TConstArrayView<int32> Mask, int32 Batch, int32 SeqLen, int32 Dim, TArray<float>& Out)
	{
		Out.Reset();
		int64 BL = 0;
		int64 BLD = 0;
		if (!MathPrivate::PoolSizes(Batch, SeqLen, Dim, BL, BLD)
			|| (int64)Hidden.Num() != BLD
			|| (int64)Mask.Num() != BL)
		{
			return false;
		}

		// Batch * Dim <= Hidden.Num(), so it fits in int32.
		Out.SetNumZeroed(Batch * Dim);
		TArray<double> Acc;
		Acc.SetNumUninitialized(Dim);
		for (int32 b = 0; b < Batch; ++b)
		{
			for (double& A : Acc)
			{
				A = 0.0;
			}
			int64 Count = 0;
			for (int32 t = 0; t < SeqLen; ++t)
			{
				if (Mask[b * SeqLen + t] == 0)
				{
					continue;
				}
				++Count;
				const float* Row = Hidden.GetData() + ((int64)b * SeqLen + t) * Dim;
				for (int32 d = 0; d < Dim; ++d)
				{
					Acc[d] += (double)Row[d];
				}
			}
			if (Count == 0)
			{
				continue; // row stays zero
			}
			const double Denom = FMath::Max((double)Count, MathPrivate::MaskEps);
			float* OutRow = Out.GetData() + (int64)b * Dim;
			for (int32 d = 0; d < Dim; ++d)
			{
				OutRow[d] = (float)(Acc[d] / Denom);
			}
		}
		return true;
	}

	bool ClsPool(TConstArrayView<float> Hidden, int32 Batch, int32 SeqLen, int32 Dim, TArray<float>& Out)
	{
		Out.Reset();
		int64 BL = 0;
		int64 BLD = 0;
		if (!MathPrivate::PoolSizes(Batch, SeqLen, Dim, BL, BLD) || (int64)Hidden.Num() != BLD)
		{
			return false;
		}

		Out.SetNumUninitialized(Batch * Dim);
		for (int32 b = 0; b < Batch; ++b)
		{
			const float* Src = Hidden.GetData() + (int64)b * SeqLen * Dim;
			FMemory::Memcpy(Out.GetData() + (int64)b * Dim, Src, sizeof(float) * Dim);
		}
		return true;
	}

	double BoxIoU(const FPipelineNmsBox& A, const FPipelineNmsBox& B)
	{
		// Every max(0, x) below is written as (x > 0 ? x : 0) so a NaN collapses to 0 or propagates into a NaN union,
		// which the final check maps to 0.
		auto Positive = [](double X) { return X > 0.0 ? X : 0.0; };
		const double AreaA = Positive((double)A.X1 - (double)A.X0) * Positive((double)A.Y1 - (double)A.Y0);
		const double AreaB = Positive((double)B.X1 - (double)B.X0) * Positive((double)B.Y1 - (double)B.Y0);
		const double InterW = Positive(FMath::Min((double)A.X1, (double)B.X1) - FMath::Max((double)A.X0, (double)B.X0));
		const double InterH = Positive(FMath::Min((double)A.Y1, (double)B.Y1) - FMath::Max((double)A.Y0, (double)B.Y0));
		const double Inter = InterW * InterH;
		const double Union = AreaA + AreaB - Inter;
		if (!(Union > 0.0) || FMath::IsNaN(Inter))
		{
			return 0.0;
		}
		const double IoU = Inter / Union;
		return FMath::IsFinite(IoU) ? IoU : 0.0;
	}

	TArray<int32> NonMaxSuppression(TConstArrayView<FPipelineNmsBox> Boxes, float IouThreshold, int32 MaxKeep)
	{
		TArray<int32> Kept;
		if (MaxKeep <= 0 || Boxes.Num() == 0)
		{
			return Kept;
		}

		TArray<int32> Order;
		Order.SetNumUninitialized(Boxes.Num());
		for (int32 i = 0; i < Boxes.Num(); ++i)
		{
			Order[i] = i;
		}
		// Strict total order: non-NaN before NaN, higher score first, then lower index.
		Order.Sort([Boxes](const int32 IA, const int32 IB)
		{
			const float SA = Boxes[IA].Score;
			const float SB = Boxes[IB].Score;
			const bool bNaNA = FMath::IsNaN(SA);
			const bool bNaNB = FMath::IsNaN(SB);
			if (bNaNA != bNaNB)
			{
				return !bNaNA;
			}
			if (!bNaNA && SA != SB)
			{
				return SA > SB;
			}
			return IA < IB;
		});

		// NaN or >= 1: nothing is suppressed (IoU never exceeds 1).
		const bool bSuppress = IouThreshold < 1.f;
		const double Threshold = (double)IouThreshold;
		Kept.Reserve(FMath::Min(MaxKeep, Boxes.Num()));
		for (const int32 Candidate : Order)
		{
			if (Kept.Num() >= MaxKeep)
			{
				break;
			}
			bool bDrop = false;
			if (bSuppress)
			{
				const FPipelineNmsBox& C = Boxes[Candidate];
				for (const int32 K : Kept)
				{
					const FPipelineNmsBox& KB = Boxes[K];
					if (KB.ClassIndex == C.ClassIndex && BoxIoU(KB, C) > Threshold)
					{
						bDrop = true;
						break;
					}
				}
			}
			if (!bDrop)
			{
				Kept.Add(Candidate);
			}
		}
		return Kept;
	}
}
