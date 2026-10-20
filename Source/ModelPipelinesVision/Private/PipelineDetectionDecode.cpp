// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "PipelineDetectionDecode.h"

#include "Math/Box2D.h"
#include "Math/UnrealMathUtility.h"
#include "PipelineMath.h"

namespace UE::ModelPipelines::Vision
{
	namespace DecodePrivate
	{
		/** One (query, class) candidate. Key = q * C' + c: the tie-break (lower first). */
		struct FCandidate
		{
			float Score = 0.f;
			int32 Query = 0;
			int32 Class = 0;
			int64 Key = 0;
		};

		/** Score descending, ties lower Key first. Scores are finite here. */
		static bool CandidateBefore(const FCandidate& A, const FCandidate& B)
		{
			if (A.Score != B.Score)
			{
				return A.Score > B.Score;
			}
			return A.Key < B.Key;
		}

		static FPipelineResult Fail(EPipelineError Error, const FString& Message, TArray<FPipelineDetection>& Out)
		{
			Out.Reset();
			return FPipelineResult::Fail(Error, FString::Printf(TEXT("DecodeDetections: %s"), *Message));
		}
	}

	FPipelineResult DecodeDetections(TConstArrayView<float> Logits, int32 NumQueries, int32 NumLogitClasses,
		TConstArrayView<float> Boxes, const TArray<FString>& Labels, const FPipelineDetectionDecodeParams& Params, TArray<FPipelineDetection>& Out)
	{
		using namespace DecodePrivate;
		Out.Reset();

		// Input checks (InferenceFailed: the model's outputs don't fit).
		if (NumQueries < 1)
		{
			return Fail(EPipelineError::InferenceFailed, FString::Printf(TEXT("number of queries is %d, expected >= 1"), NumQueries), Out);
		}
		if (NumLogitClasses < 1)
		{
			return Fail(EPipelineError::InferenceFailed, FString::Printf(TEXT("number of logit columns is %d, expected >= 1"), NumLogitClasses), Out);
		}
		const int64 Q = NumQueries;
		const int64 CL = NumLogitClasses;
		if ((int64)Logits.Num() != Q * CL)
		{
			return Fail(EPipelineError::InferenceFailed, FString::Printf(TEXT("logits hold %d values, expected %lld x %lld = %lld"),
				Logits.Num(), (long long)Q, (long long)CL, (long long)(Q * CL)), Out);
		}
		if ((int64)Boxes.Num() != Q * 4)
		{
			return Fail(EPipelineError::InferenceFailed, FString::Printf(TEXT("boxes hold %d values, expected %lld x 4 = %lld"),
				Boxes.Num(), (long long)Q, (long long)(Q * 4)), Out);
		}
		const bool bSigmoid = Params.Activation == EPipelineScoreActivation::Sigmoid;
		const int32 NumClasses = bSigmoid ? NumLogitClasses : NumLogitClasses - 1;
		if (Labels.Num() != NumClasses || NumClasses < 1)
		{
			return Fail(EPipelineError::InferenceFailed, bSigmoid
				? FString::Printf(TEXT("%d logit columns but %d labels (Sigmoid needs one label per column)"), NumLogitClasses, Labels.Num())
				: FString::Printf(TEXT("%d logit columns but %d labels (Softmax needs one label per column plus a no-object column)"), NumLogitClasses, Labels.Num()), Out);
		}
		if (!AllFinite(Logits))
		{
			return Fail(EPipelineError::InferenceFailed, TEXT("logits contain NaN or Inf"), Out);
		}
		if (!AllFinite(Boxes))
		{
			return Fail(EPipelineError::InferenceFailed, TEXT("boxes contain NaN or Inf"), Out);
		}

		// Parameter checks (InvalidArgument: the caller's values).
		if (FMath::IsNaN(Params.ScoreThreshold))
		{
			return Fail(EPipelineError::InvalidArgument, TEXT("score threshold is NaN"), Out);
		}
		const float Threshold = FMath::Clamp(Params.ScoreThreshold, 0.f, 1.f);
		if (Params.MaxDetections < 1)
		{
			return Fail(EPipelineError::InvalidArgument, FString::Printf(TEXT("MaxDetections is %d, expected >= 1"), Params.MaxDetections), Out);
		}
		const FPipelineImageGeometry& G = Params.Geometry;
		const bool bAllPositive = G.CropWidth > 0 && G.CropHeight > 0 && G.ResizedWidth > 0 && G.ResizedHeight > 0;
		const bool bAllZero = G.CropWidth == 0 && G.CropHeight == 0 && G.ResizedWidth == 0 && G.ResizedHeight == 0;
		if (!bAllPositive && !bAllZero)
		{
			return Fail(EPipelineError::InvalidArgument, FString::Printf(TEXT("geometry crop %dx%d, resized %dx%d: all must be > 0 (mapping) or all 0 (identity)"),
				G.CropWidth, G.CropHeight, G.ResizedWidth, G.ResizedHeight), Out);
		}

		// 1. Scores and candidates.
		TArray<FCandidate> Candidates;
		if (bSigmoid)
		{
			for (int64 q = 0; q < Q; ++q)
			{
				const float* Row = Logits.GetData() + q * CL;
				for (int64 c = 0; c < CL; ++c)
				{
					const float S = (float)(1.0 / (1.0 + FMath::Exp(-(double)Row[c])));
					if (S > Threshold)
					{
						FCandidate& Cand = Candidates.AddDefaulted_GetRef();
						Cand.Score = S;
						Cand.Query = (int32)q;
						Cand.Class = (int32)c;
						Cand.Key = q * CL + c;
					}
				}
			}
			Candidates.Sort(&CandidateBefore);
			if (Candidates.Num() > NumQueries)
			{
				Candidates.SetNum(NumQueries, EAllowShrinking::No);   // top-Q (HF: top-Q, then threshold)
			}
		}
		else
		{
			TArray<double> P;
			P.SetNumUninitialized(NumLogitClasses);
			for (int64 q = 0; q < Q; ++q)
			{
				const float* Row = Logits.GetData() + q * CL;
				double Max = (double)Row[0];
				for (int64 c = 1; c < CL; ++c)
				{
					Max = FMath::Max(Max, (double)Row[c]);
				}
				double Sum = 0.0;
				for (int64 c = 0; c < CL; ++c)
				{
					P[c] = FMath::Exp((double)Row[c] - Max);
					Sum += P[c];
				}
				int32 Best = 0;
				for (int32 c = 1; c < NumClasses; ++c)   // the last column (no-object) is dropped
				{
					if (P[c] > P[Best])
					{
						Best = c;
					}
				}
				const float S = (float)(P[Best] / Sum);
				if (S > Threshold)
				{
					FCandidate& Cand = Candidates.AddDefaulted_GetRef();
					Cand.Score = S;
					Cand.Query = (int32)q;
					Cand.Class = Best;
					Cand.Key = q * CL + Best;
				}
			}
			Candidates.Sort(&CandidateBefore);
		}
		if (Candidates.Num() == 0)
		{
			return FPipelineResult::Ok();
		}

		// 2. Corners, mapped to the source image (unclamped).
		TArray<FPipelineNmsBox> NmsBoxes;
		NmsBoxes.SetNum(Candidates.Num());
		for (int32 i = 0; i < Candidates.Num(); ++i)
		{
			const float* B = Boxes.GetData() + (int64)Candidates[i].Query * 4;
			const float Cx = B[0], Cy = B[1], W = B[2], H = B[3];
			float X0 = Cx - 0.5f * W;
			float Y0 = Cy - 0.5f * H;
			float X1 = Cx + 0.5f * W;
			float Y1 = Cy + 0.5f * H;
			if (bAllPositive)
			{
				const float Left = (float)G.CropLeft, Top = (float)G.CropTop;
				const float CW = (float)G.CropWidth, CH = (float)G.CropHeight;
				const float RW = (float)G.ResizedWidth, RH = (float)G.ResizedHeight;
				X0 = (Left + X0 * CW) / RW;
				X1 = (Left + X1 * CW) / RW;
				Y0 = (Top + Y0 * CH) / RH;
				Y1 = (Top + Y1 * CH) / RH;
			}
			FPipelineNmsBox& N = NmsBoxes[i];
			N.X0 = X0;
			N.Y0 = Y0;
			N.X1 = X1;
			N.Y1 = Y1;
			N.Score = Candidates[i].Score;
			N.ClassIndex = Candidates[i].Class;
		}

		// 3. Class-wise NMS; 4. output in NMS order, clamped.
		const TArray<int32> Kept = NonMaxSuppression(NmsBoxes, Params.NmsIouThreshold, Params.MaxDetections);
		TArray<FPipelineDetection> Result;
		Result.Reserve(Kept.Num());
		for (const int32 k : Kept)
		{
			if (!NmsBoxes.IsValidIndex(k))
			{
				return Fail(EPipelineError::Internal, FString::Printf(TEXT("NMS returned index %d of %d"), k, NmsBoxes.Num()), Out);
			}
			const FPipelineNmsBox& N = NmsBoxes[k];
			FPipelineDetection& D = Result.AddDefaulted_GetRef();
			D.Index = N.ClassIndex;
			D.Label = Labels[N.ClassIndex];
			D.Score = N.Score;
			D.Box = FBox2D(FVector2D(FMath::Clamp(N.X0, 0.f, 1.f), FMath::Clamp(N.Y0, 0.f, 1.f)),
				FVector2D(FMath::Clamp(N.X1, 0.f, 1.f), FMath::Clamp(N.Y1, 0.f, 1.f)));
		}
		Out = MoveTemp(Result);
		return FPipelineResult::Ok();
	}
}
