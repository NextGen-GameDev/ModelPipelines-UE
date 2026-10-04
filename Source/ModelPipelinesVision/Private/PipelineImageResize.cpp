// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
//
// Bit-exact port of Pillow 12's 8-bit Image.resize (libImaging/Resample.c for the convolution filters, the affine
// nearest path of libImaging/Geometry.c ImagingScaleAffine for NEAREST). docs/contracts.md C27.
// Pure functions: no statics, no caches, any thread.

#include "PipelineImage.h"

#include <cmath>

namespace UE::ModelPipelines::Vision::Private
{
	/** Largest output side and pixel count ResizeImage accepts (C27). */
	static constexpr int32 MaxResizeSide = 65536;
	static constexpr int64 MaxResizePixels = int64(1) << 26;

	bool IsResizeSizeAllowed(int64 OutWidth, int64 OutHeight)
	{
		return OutWidth >= 1 && OutWidth <= MaxResizeSide && OutHeight >= 1 && OutHeight <= MaxResizeSide
			&& OutWidth * OutHeight <= MaxResizePixels;
	}

	bool IsValidResample(EPipelineImageResample Filter)
	{
		return (uint8)Filter <= (uint8)EPipelineImageResample::Hamming;
	}

	namespace
	{
		constexpr int32 PrecisionBits = 32 - 8 - 2; // 22, as Pillow
		constexpr double Pi = 3.14159265358979323846;

		double BoxFilter(double X)
		{
			return (X > -0.5 && X <= 0.5) ? 1.0 : 0.0;
		}

		double BilinearFilter(double X)
		{
			if (X < 0.0)
			{
				X = -X;
			}
			return X < 1.0 ? 1.0 - X : 0.0;
		}

		double HammingFilter(double X)
		{
			if (X < 0.0)
			{
				X = -X;
			}
			if (X == 0.0)
			{
				return 1.0;
			}
			if (X >= 1.0)
			{
				return 0.0;
			}
			X = X * Pi;
			// Pillow writes 0.54f / 0.46f (float literals promoted to double); bit-exactness depends on it.
			return std::sin(X) / X * ((double)0.54f + (double)0.46f * std::cos(X));
		}

		double BicubicFilter(double X)
		{
			constexpr double A = -0.5;
			if (X < 0.0)
			{
				X = -X;
			}
			if (X < 1.0)
			{
				return ((A + 2.0) * X - (A + 3.0)) * X * X + 1;
			}
			if (X < 2.0)
			{
				return (((X - 5) * X + 8) * X - 4) * A;
			}
			return 0.0;
		}

		double SincFilter(double X)
		{
			if (X == 0.0)
			{
				return 1.0;
			}
			X = X * Pi;
			return std::sin(X) / X;
		}

		double LanczosFilter(double X)
		{
			if (-3.0 <= X && X < 3.0)
			{
				return SincFilter(X) * SincFilter(X / 3);
			}
			return 0.0;
		}

		using FFilterFn = double (*)(double);

		struct FFilterDef
		{
			FFilterFn Fn = nullptr;
			double Support = 0.0;
		};

		FFilterDef GetFilter(EPipelineImageResample Filter)
		{
			switch (Filter)
			{
			case EPipelineImageResample::Box:      return { &BoxFilter, 0.5 };
			case EPipelineImageResample::Bilinear: return { &BilinearFilter, 1.0 };
			case EPipelineImageResample::Hamming:  return { &HammingFilter, 1.0 };
			case EPipelineImageResample::Bicubic:  return { &BicubicFilter, 2.0 };
			case EPipelineImageResample::Lanczos:  return { &LanczosFilter, 3.0 };
			default:                               return {};
			}
		}

		/** Fixed-point coefficients for output indices [First, First + Count) of one axis (Pillow precompute_coeffs +
		    normalize_coeffs_8bpc). Bounds holds (min, count) per index; K holds KSize weights per index. */
		struct FAxisCoeffs
		{
			TArray<int32> Bounds;
			TArray<int32> K;
			int32 KSize = 0;
		};

		void ComputeCoeffs(int32 InSize, int32 OutSize, int32 First, int32 Count, const FFilterDef& Filter, FAxisCoeffs& Out)
		{
			// in0 = 0, in1 = InSize (whole image box; float in Pillow, exact for these sizes).
			const double Scale = (double)InSize / (double)OutSize;
			const double FilterScale = Scale < 1.0 ? 1.0 : Scale;
			const double Support = Filter.Support * FilterScale;
			const int32 KSize = (int32)std::ceil(Support) * 2 + 1;
			const double Ss = 1.0 / FilterScale;

			Out.KSize = KSize;
			Out.Bounds.SetNumUninitialized(Count * 2);
			Out.K.SetNumZeroed(Count * KSize);
			TArray<double> W;
			W.SetNumUninitialized(KSize);

			for (int32 i = 0; i < Count; ++i)
			{
				const int32 XX = First + i;
				const double Center = 0.0 + (XX + 0.5) * Scale;
				double Ww = 0.0;
				int32 XMin = (int32)(Center - Support + 0.5);
				if (XMin < 0)
				{
					XMin = 0;
				}
				int32 XMax = (int32)(Center + Support + 0.5);
				if (XMax > InSize)
				{
					XMax = InSize;
				}
				XMax -= XMin;
				// XMax <= KSize by construction of Pillow's bounds; guard anyway (never write past the row).
				if (XMax > KSize)
				{
					XMax = KSize;
				}
				for (int32 X = 0; X < XMax; ++X)
				{
					const double Wx = Filter.Fn((X + XMin - Center + 0.5) * Ss);
					W[X] = Wx;
					Ww += Wx;
				}
				int32* KRow = &Out.K[i * KSize];
				for (int32 X = 0; X < XMax; ++X)
				{
					double Wx = W[X];
					if (Ww != 0.0)
					{
						Wx /= Ww;
					}
					KRow[X] = Wx < 0 ? (int32)(-0.5 + Wx * (1 << PrecisionBits)) : (int32)(0.5 + Wx * (1 << PrecisionBits));
				}
				Out.Bounds[i * 2 + 0] = XMin;
				Out.Bounds[i * 2 + 1] = XMax;
			}
		}

		FORCEINLINE uint8 Clip8(uint32 Acc)
		{
			const int32 In = (int32)Acc; // Pillow accumulates in a 32-bit int; wraparound emulated via uint32.
			if (In >= (1 << PrecisionBits << 8))
			{
				return 255;
			}
			if (In <= 0)
			{
				return 0;
			}
			return (uint8)(In >> PrecisionBits);
		}

		/** One uint8 RGB pixel of the intermediate image between the passes (Pillow stores it as uint8). */
		struct FRgb8
		{
			uint8 R;
			uint8 G;
			uint8 B;
		};
		static_assert(sizeof(FRgb8) == 3, "FRgb8 must be packed RGB");

		/** Upper bound of the intermediate buffer, in bytes (C27: bounded; the passes run in strips). */
		constexpr int64 MaxIntermediateBytes = int64(4) << 20;

		FORCEINLINE void StorePixel(FRgb8& Dst, uint32 S0, uint32 S1, uint32 S2)
		{
			Dst.R = Clip8(S0);
			Dst.G = Clip8(S1);
			Dst.B = Clip8(S2);
		}

		FORCEINLINE void StorePixel(FColor& Dst, uint32 S0, uint32 S1, uint32 S2)
		{
			Dst = FColor(Clip8(S0), Clip8(S1), Clip8(S2), 255);
		}

		/**
		 * Horizontal pass (Pillow ImagingResampleHorizontal_8bpc) for output columns [First, First + Count) of H (indices
		 * relative to the columns H was computed for). SrcRow[c - ColBase] is source column c.
		 */
		template <typename TSrc, typename TDst>
		void HorizontalRow(const TSrc* SrcRow, int32 ColBase, const FAxisCoeffs& H, int32 First, int32 Count, TDst* Dst)
		{
			const int32 KSize = H.KSize;
			for (int32 X = 0; X < Count; ++X)
			{
				const int32 Xi = First + X;
				const int32 XMin = H.Bounds[Xi * 2] - ColBase;
				const int32 XCount = H.Bounds[Xi * 2 + 1];
				const int32* K = &H.K[(int64)Xi * KSize];
				uint32 S0 = 1u << (PrecisionBits - 1);
				uint32 S1 = S0;
				uint32 S2 = S0;
				const TSrc* Src = SrcRow + XMin;
				for (int32 i = 0; i < XCount; ++i)
				{
					const uint32 W = (uint32)K[i];
					S0 += (uint32)Src[i].R * W;
					S1 += (uint32)Src[i].G * W;
					S2 += (uint32)Src[i].B * W;
				}
				StorePixel(Dst[X], S0, S1, S2);
			}
		}

		/**
		 * Vertical pass (Pillow ImagingResampleVertical_8bpc) for output row Yi of V over NumCols columns. Base points at
		 * source row RowBase, first column; Stride is the source row pitch in pixels.
		 */
		template <typename TSrc, typename TDst>
		void VerticalRow(const TSrc* Base, int64 Stride, int32 RowBase, const FAxisCoeffs& V, int32 Yi, int32 NumCols, TDst* Dst)
		{
			const int32 YMin = V.Bounds[Yi * 2] - RowBase;
			const int32 YCount = V.Bounds[Yi * 2 + 1];
			const int32* K = &V.K[(int64)Yi * V.KSize];
			const TSrc* Row0 = Base + (int64)YMin * Stride;
			for (int32 X = 0; X < NumCols; ++X)
			{
				uint32 S0 = 1u << (PrecisionBits - 1);
				uint32 S1 = S0;
				uint32 S2 = S0;
				const TSrc* P = Row0 + X;
				for (int32 i = 0; i < YCount; ++i)
				{
					const uint32 W = (uint32)K[i];
					S0 += (uint32)P->R * W;
					S1 += (uint32)P->G * W;
					S2 += (uint32)P->B * W;
					P += Stride;
				}
				StorePixel(Dst[X], S0, S1, S2);
			}
		}

		/** [First, Last) source range touched by all Count indices of C. */
		void SourceRange(const FAxisCoeffs& C, int32 Count, int32& OutFirst, int32& OutLast)
		{
			OutFirst = C.Bounds[0];
			OutLast = C.Bounds[0] + C.Bounds[1];
			for (int32 i = 1; i < Count; ++i)
			{
				OutFirst = FMath::Min(OutFirst, C.Bounds[i * 2]);
				OutLast = FMath::Max(OutLast, C.Bounds[i * 2] + C.Bounds[i * 2 + 1]);
			}
		}

		/** Strip length so that Strip * BytesPerUnit <= MaxIntermediateBytes, clamped to [1, Total]. */
		int32 StripLength(int64 BytesPerUnit, int32 Total)
		{
			const int64 Fit = MaxIntermediateBytes / FMath::Max<int64>(BytesPerUnit, 1);
			return (int32)FMath::Clamp<int64>(Fit, 1, (int64)Total);
		}

		void ResizeNearestRegion(const FPipelineImage& In, int32 OutWidth, int32 OutHeight, int32 X0, int32 Y0, int32 RW, int32 RH, FPipelineImage& Out)
		{
			// ImagingScaleAffine: xo = a[2] + a[0] * 0.5, incremented by a[0] per output pixel; COORD(v) = v < 0 ? -1 : (int)v.
			const double Ax = (double)In.Width / (double)OutWidth;
			const double Ay = (double)In.Height / (double)OutHeight;
			TArray<int32> XIn;
			XIn.SetNumUninitialized(RW);
			double Xo = 0.0 + Ax * 0.5;
			for (int32 X = 0; X < X0 + RW; ++X)
			{
				if (X >= X0)
				{
					XIn[X - X0] = FMath::Clamp(Xo < 0.0 ? -1 : (int32)Xo, 0, In.Width - 1);
				}
				Xo += Ax;
			}
			double Yo = 0.0 + Ay * 0.5;
			for (int32 Y = 0; Y < Y0 + RH; ++Y)
			{
				if (Y >= Y0)
				{
					const int32 Yi = FMath::Clamp(Yo < 0.0 ? -1 : (int32)Yo, 0, In.Height - 1);
					const FColor* SrcRow = &In.Pixels[(int64)Yi * In.Width];
					FColor* DstRow = &Out.Pixels[(int64)(Y - Y0) * RW];
					for (int32 X = 0; X < RW; ++X)
					{
						const FColor S = SrcRow[XIn[X]];
						DstRow[X] = FColor(S.R, S.G, S.B, 255);
					}
				}
				Yo += Ay;
			}
		}
	}

	/**
	 * Computes the window [X0, X0+RW) x [Y0, Y0+RH) of ResizeImage(In, OutWidth, OutHeight, Filter) into Out (RW x RH, A 255).
	 * Every output pixel depends only on its own coefficients, so the window is bit-identical to cropping the full result.
	 * Preconditions (validated by the callers): In valid, sizes allowed, filter valid, window inside the output.
	 */
	void ResizeImageRegion(const FPipelineImage& In, int32 OutWidth, int32 OutHeight, EPipelineImageResample Filter,
		int32 X0, int32 Y0, int32 RW, int32 RH, FPipelineImage& Out)
	{
		Out.Width = RW;
		Out.Height = RH;
		Out.Pixels.SetNumUninitialized(RW * RH);

		if (OutWidth == In.Width && OutHeight == In.Height)
		{
			// Pillow returns a copy.
			for (int32 Y = 0; Y < RH; ++Y)
			{
				const FColor* SrcRow = &In.Pixels[(int64)(Y + Y0) * In.Width + X0];
				FColor* DstRow = &Out.Pixels[(int64)Y * RW];
				for (int32 X = 0; X < RW; ++X)
				{
					DstRow[X] = FColor(SrcRow[X].R, SrcRow[X].G, SrcRow[X].B, 255);
				}
			}
			return;
		}

		if (Filter == EPipelineImageResample::Nearest)
		{
			ResizeNearestRegion(In, OutWidth, OutHeight, X0, Y0, RW, RH, Out);
			return;
		}

		const FFilterDef Def = GetFilter(Filter);
		const bool bNeedHorizontal = OutWidth != In.Width;
		const bool bNeedVertical = OutHeight != In.Height;

		FAxisCoeffs Horiz;
		if (bNeedHorizontal)
		{
			ComputeCoeffs(In.Width, OutWidth, X0, RW, Def, Horiz);
		}
		FAxisCoeffs Vert;
		if (bNeedVertical)
		{
			ComputeCoeffs(In.Height, OutHeight, Y0, RH, Def, Vert);
		}

		const FColor* Src = In.Pixels.GetData();
		const int64 SrcStride = In.Width;

		if (!bNeedVertical)
		{
			// Horizontal only, straight from the source rows of the window.
			for (int32 Y = 0; Y < RH; ++Y)
			{
				HorizontalRow(Src + (int64)(Y0 + Y) * SrcStride, 0, Horiz, 0, RW, &Out.Pixels[(int64)Y * RW]);
			}
			return;
		}
		if (!bNeedHorizontal)
		{
			// Vertical only, straight from the source columns of the window.
			for (int32 Y = 0; Y < RH; ++Y)
			{
				VerticalRow(Src + X0, SrcStride, 0, Vert, Y, RW, &Out.Pixels[(int64)Y * RW]);
			}
			return;
		}

		TArray<FRgb8> Temp;
		// Pillow 12 Image.resize: when height > width * 100 and the height shrinks, it resizes to (width, out_height) first,
		// then to (out_width, out_height), each stored as uint8 (C27 tall-image rule). So the vertical pass runs first.
		const bool bVerticalFirst = (int64)In.Height > (int64)In.Width * 100 && OutHeight < In.Height;
		if (bVerticalFirst)
		{
			// Vertical pass over the source columns the horizontal pass needs, in strips of output rows (each output row of
			// the horizontal pass depends only on the same intermediate row).
			int32 ColFirst = 0;
			int32 ColLast = 0;
			SourceRange(Horiz, RW, ColFirst, ColLast);
			const int32 NumCols = FMath::Max(ColLast - ColFirst, 1);
			const int32 StripRows = StripLength((int64)NumCols * (int64)sizeof(FRgb8), RH);
			Temp.SetNumUninitialized(StripRows * NumCols); // <= MaxIntermediateBytes / 3 elements
			for (int32 Strip = 0; Strip < RH; Strip += StripRows)
			{
				const int32 Rows = FMath::Min(StripRows, RH - Strip);
				for (int32 R = 0; R < Rows; ++R)
				{
					FRgb8* TempRow = Temp.GetData() + (int64)R * NumCols;
					VerticalRow(Src + ColFirst, SrcStride, 0, Vert, Strip + R, ColLast - ColFirst, TempRow);
					HorizontalRow(TempRow, ColFirst, Horiz, 0, RW, &Out.Pixels[(int64)(Strip + R) * RW]);
				}
			}
			return;
		}

		// Horizontal pass first (over the source rows the vertical pass needs), then vertical, in strips of output columns
		// (each output column depends only on its own coefficients, so the strips are bit-identical to one pass).
		int32 RowFirst = 0;
		int32 RowLast = 0;
		SourceRange(Vert, RH, RowFirst, RowLast);
		const int32 NumRows = FMath::Max(RowLast - RowFirst, 1);
		const int32 StripCols = StripLength((int64)NumRows * (int64)sizeof(FRgb8), RW);
		Temp.SetNumUninitialized(StripCols * NumRows); // <= MaxIntermediateBytes / 3 elements
		for (int32 Strip = 0; Strip < RW; Strip += StripCols)
		{
			const int32 Cols = FMath::Min(StripCols, RW - Strip);
			for (int32 R = 0; R < RowLast - RowFirst; ++R)
			{
				HorizontalRow(Src + (int64)(RowFirst + R) * SrcStride, 0, Horiz, Strip, Cols, Temp.GetData() + (int64)R * Cols);
			}
			for (int32 Y = 0; Y < RH; ++Y)
			{
				VerticalRow(Temp.GetData(), (int64)Cols, RowFirst, Vert, Y, Cols, &Out.Pixels[(int64)Y * RW + Strip]);
			}
		}
	}
}

namespace UE::ModelPipelines::Vision
{
	bool ResizeImage(const FPipelineImage& In, int32 OutWidth, int32 OutHeight, EPipelineImageResample Filter, FPipelineImage& Out)
	{
		// In and Out may be the same object: read In completely before Out is written.
		if (!In.IsValid() || !Private::IsResizeSizeAllowed(OutWidth, OutHeight) || !Private::IsValidResample(Filter))
		{
			Out = FPipelineImage();
			return false;
		}
		FPipelineImage Result;
		Private::ResizeImageRegion(In, OutWidth, OutHeight, Filter, 0, 0, OutWidth, OutHeight, Result);
		Out = MoveTemp(Result);
		return true;
	}
}
