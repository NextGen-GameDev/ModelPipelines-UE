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
			return std::sin(X) / X * (0.54 + 0.46 * std::cos(X));
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

		/** Interleaved RGB rows used between the passes. */
		struct FRgbRows
		{
			TArray<uint8> Data;
			int32 Width = 0;
			int32 FirstRow = 0; // source row index of Data's row 0
			int32 NumRows = 0;
		};

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

		FAxisCoeffs Vert;
		int32 RowFirst = Y0;
		int32 RowLast = Y0 + RH; // exclusive, source rows feeding the vertical pass (or the output rows if none)
		if (bNeedVertical)
		{
			ComputeCoeffs(In.Height, OutHeight, Y0, RH, Def, Vert);
			RowFirst = Vert.Bounds[0];
			RowLast = Vert.Bounds[(RH - 1) * 2] + Vert.Bounds[(RH - 1) * 2 + 1];
			for (int32 i = 0; i < RH; ++i)
			{
				RowFirst = FMath::Min(RowFirst, Vert.Bounds[i * 2]);
				RowLast = FMath::Max(RowLast, Vert.Bounds[i * 2] + Vert.Bounds[i * 2 + 1]);
			}
		}

		// Horizontal pass (or a plain column copy) over the needed source rows, region columns only.
		FRgbRows Temp;
		Temp.Width = RW;
		Temp.FirstRow = RowFirst;
		Temp.NumRows = FMath::Max(RowLast - RowFirst, 0);
		Temp.Data.SetNumUninitialized((int64)Temp.NumRows * RW * 3);
		if (bNeedHorizontal)
		{
			FAxisCoeffs Horiz;
			ComputeCoeffs(In.Width, OutWidth, X0, RW, Def, Horiz);
			const int32 KSize = Horiz.KSize;
			for (int32 R = 0; R < Temp.NumRows; ++R)
			{
				const FColor* SrcRow = &In.Pixels[(int64)(RowFirst + R) * In.Width];
				uint8* Dst = &Temp.Data[(int64)R * RW * 3];
				for (int32 X = 0; X < RW; ++X)
				{
					const int32 XMin = Horiz.Bounds[X * 2];
					const int32 XCount = Horiz.Bounds[X * 2 + 1];
					const int32* K = &Horiz.K[X * KSize];
					uint32 S0 = 1u << (PrecisionBits - 1);
					uint32 S1 = S0;
					uint32 S2 = S0;
					const FColor* Src = SrcRow + XMin;
					for (int32 i = 0; i < XCount; ++i)
					{
						const uint32 W = (uint32)K[i];
						S0 += (uint32)Src[i].R * W;
						S1 += (uint32)Src[i].G * W;
						S2 += (uint32)Src[i].B * W;
					}
					Dst[X * 3 + 0] = Clip8(S0);
					Dst[X * 3 + 1] = Clip8(S1);
					Dst[X * 3 + 2] = Clip8(S2);
				}
			}
		}
		else
		{
			for (int32 R = 0; R < Temp.NumRows; ++R)
			{
				const FColor* SrcRow = &In.Pixels[(int64)(RowFirst + R) * In.Width + X0];
				uint8* Dst = &Temp.Data[(int64)R * RW * 3];
				for (int32 X = 0; X < RW; ++X)
				{
					Dst[X * 3 + 0] = SrcRow[X].R;
					Dst[X * 3 + 1] = SrcRow[X].G;
					Dst[X * 3 + 2] = SrcRow[X].B;
				}
			}
		}

		if (!bNeedVertical)
		{
			// Temp rows are exactly the region rows.
			for (int32 Y = 0; Y < RH; ++Y)
			{
				const uint8* Src = &Temp.Data[(int64)Y * RW * 3];
				FColor* DstRow = &Out.Pixels[(int64)Y * RW];
				for (int32 X = 0; X < RW; ++X)
				{
					DstRow[X] = FColor(Src[X * 3 + 0], Src[X * 3 + 1], Src[X * 3 + 2], 255);
				}
			}
			return;
		}

		const int32 KSize = Vert.KSize;
		const int64 Stride = (int64)RW * 3;
		for (int32 Y = 0; Y < RH; ++Y)
		{
			const int32 YMin = Vert.Bounds[Y * 2] - RowFirst;
			const int32 YCount = Vert.Bounds[Y * 2 + 1];
			const int32* K = &Vert.K[Y * KSize];
			FColor* DstRow = &Out.Pixels[(int64)Y * RW];
			const uint8* Base = Temp.Data.GetData() + (int64)YMin * Stride;
			for (int32 X = 0; X < RW; ++X)
			{
				uint32 S0 = 1u << (PrecisionBits - 1);
				uint32 S1 = S0;
				uint32 S2 = S0;
				const uint8* P = Base + X * 3;
				for (int32 i = 0; i < YCount; ++i)
				{
					const uint32 W = (uint32)K[i];
					S0 += (uint32)P[0] * W;
					S1 += (uint32)P[1] * W;
					S2 += (uint32)P[2] * W;
					P += Stride;
				}
				DstRow[X] = FColor(Clip8(S0), Clip8(S1), Clip8(S2), 255);
			}
		}
	}
}

namespace UE::ModelPipelines::Vision
{
	bool ResizeImage(const FPipelineImage& In, int32 OutWidth, int32 OutHeight, EPipelineImageResample Filter, FPipelineImage& Out)
	{
		Out = FPipelineImage();
		if (!In.IsValid() || !Private::IsResizeSizeAllowed(OutWidth, OutHeight) || !Private::IsValidResample(Filter))
		{
			return false;
		}
		Private::ResizeImageRegion(In, OutWidth, OutHeight, Filter, 0, 0, OutWidth, OutHeight, Out);
		return true;
	}
}
