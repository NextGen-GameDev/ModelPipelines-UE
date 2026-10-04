// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
//
// FPipelineImage and the CPU preprocessing chain (docs/contracts.md C27): resize per the package's resize mode, centre crop,
// rescale, normalize, CHW float32. Pure functions, any thread.

#include "PipelineImage.h"

#include "PipelineVisionLog.h"

namespace UE::ModelPipelines::Vision::Private
{
	// Defined in PipelineImageResize.cpp.
	bool IsResizeSizeAllowed(int64 OutWidth, int64 OutHeight);
	bool IsValidResample(EPipelineImageResample Filter);
	void ResizeImageRegion(const FPipelineImage& In, int32 OutWidth, int32 OutHeight, EPipelineImageResample Filter,
		int32 X0, int32 Y0, int32 RW, int32 RH, FPipelineImage& Out);
}

bool FPipelineImage::IsValid(FString* OutReason) const
{
	auto Fail = [OutReason](FString&& Reason)
	{
		if (OutReason)
		{
			*OutReason = MoveTemp(Reason);
		}
		return false;
	};
	if (Width < 1 || Width > MaxSide)
	{
		return Fail(FString::Printf(TEXT("image Width %d is outside 1..%d"), Width, MaxSide));
	}
	if (Height < 1 || Height > MaxSide)
	{
		return Fail(FString::Printf(TEXT("image Height %d is outside 1..%d"), Height, MaxSide));
	}
	if ((int64)Pixels.Num() != (int64)Width * (int64)Height)
	{
		return Fail(FString::Printf(TEXT("image has %d pixels, expected Width*Height = %d*%d = %lld"), Pixels.Num(), Width, Height, (long long)((int64)Width * Height)));
	}
	return true;
}

namespace UE::ModelPipelines::Vision
{
	namespace
	{
		constexpr int32 MaxNetworkSide = 8192;

		bool IsFiniteDouble(double V)
		{
			return FMath::IsFinite(V);
		}

		/** C27 settings rules. Empty string = usable. */
		FString CheckSettings(const FPipelinePreprocessSettings& S)
		{
			if (S.ImageWidth < 1 || S.ImageWidth > MaxNetworkSide)
			{
				return FString::Printf(TEXT("ImageWidth %d is outside 1..%d"), S.ImageWidth, MaxNetworkSide);
			}
			if (S.ImageHeight < 1 || S.ImageHeight > MaxNetworkSide)
			{
				return FString::Printf(TEXT("ImageHeight %d is outside 1..%d"), S.ImageHeight, MaxNetworkSide);
			}
			switch (S.ResizeMode)
			{
			case EPipelineImageResize::Stretch:
				break;
			case EPipelineImageResize::ShortestEdgeCenterCrop:
			{
				const int32 Needed = FMath::Max(S.ImageWidth, S.ImageHeight);
				if (S.ResizeShortestEdge < Needed)
				{
					return FString::Printf(TEXT("ResizeShortestEdge %d is smaller than the crop's larger side %d"), S.ResizeShortestEdge, Needed);
				}
				break;
			}
			case EPipelineImageResize::StretchCenterCrop:
				if (S.ResizeWidth < S.ImageWidth || S.ResizeWidth > MaxNetworkSide)
				{
					return FString::Printf(TEXT("ResizeWidth %d is outside %d..%d (ImageWidth..%d)"), S.ResizeWidth, S.ImageWidth, MaxNetworkSide, MaxNetworkSide);
				}
				if (S.ResizeHeight < S.ImageHeight || S.ResizeHeight > MaxNetworkSide)
				{
					return FString::Printf(TEXT("ResizeHeight %d is outside %d..%d (ImageHeight..%d)"), S.ResizeHeight, S.ImageHeight, MaxNetworkSide, MaxNetworkSide);
				}
				break;
			default:
				return FString::Printf(TEXT("ResizeMode %d is not a valid EPipelineImageResize value"), (int32)S.ResizeMode);
			}
			if (!FMath::IsFinite(S.RescaleFactor) || !(S.RescaleFactor > 0.f))
			{
				return FString::Printf(TEXT("RescaleFactor %g must be finite and > 0"), (double)S.RescaleFactor);
			}
			const TCHAR* Channels[3] = { TEXT("R"), TEXT("G"), TEXT("B") };
			for (int32 c = 0; c < 3; ++c)
			{
				const double Std = S.ImageStd[c];
				if (!IsFiniteDouble(Std) || !(Std > 0.0))
				{
					return FString::Printf(TEXT("ImageStd.%s %g must be finite and > 0"), Channels[c], Std);
				}
				const double Mean = S.ImageMean[c];
				if (!IsFiniteDouble(Mean))
				{
					return FString::Printf(TEXT("ImageMean.%s %g must be finite"), Channels[c], Mean);
				}
			}
			if (!Private::IsValidResample(S.Resample))
			{
				return FString::Printf(TEXT("Resample %d is not a valid EPipelineImageResample value"), (int32)S.Resample);
			}
			return FString();
		}
	}

	FPipelineResult PreprocessImage(const FPipelineImage& In, const FPipelinePreprocessSettings& Settings,
		TArray<float>& OutCHW, FPipelineImage* OutCropped, FPipelineImageGeometry* OutGeometry)
	{
		// OutCropped may alias In (PreprocessImage(Img, S, T, &Img)): In is read completely before OutCropped is written, and
		// OutCropped is only cleared on failure paths, after the last read of In.
		OutCHW.Reset();
		if (OutGeometry)
		{
			*OutGeometry = FPipelineImageGeometry();
		}
		auto Fail = [OutCropped](EPipelineError Error, const FString& Message)
		{
			if (OutCropped)
			{
				*OutCropped = FPipelineImage();
			}
			return FPipelineResult::Fail(Error, Message);
		};

		FString Reason;
		if (!In.IsValid(&Reason))
		{
			return Fail(EPipelineError::InvalidArgument, FString::Printf(TEXT("Invalid image: %s."), *Reason));
		}
		Reason = CheckSettings(Settings);
		if (!Reason.IsEmpty())
		{
			return Fail(EPipelineError::PreprocessFailed, FString::Printf(TEXT("Image preprocess settings not usable: %s."), *Reason));
		}

		// 1. Resize size (C27). Computed in int64/double and checked before any allocation.
		const int32 W = In.Width;
		const int32 H = In.Height;
		int64 ResizedW = 0;
		int64 ResizedH = 0;
		switch (Settings.ResizeMode)
		{
		case EPipelineImageResize::Stretch:
			ResizedW = Settings.ImageWidth;
			ResizedH = Settings.ImageHeight;
			break;
		case EPipelineImageResize::ShortestEdgeCenterCrop:
		{
			const int64 S = Settings.ResizeShortestEdge;
			// Python int(S * long / short): true division of the exact integer product in double, then truncation.
			if (W <= H)
			{
				ResizedW = S;
				const double Long = (double)(S * (int64)H) / (double)W;
				ResizedH = Long > 1.0e12 ? (int64)1000000000000 : (int64)Long;
			}
			else
			{
				ResizedH = S;
				const double Long = (double)(S * (int64)W) / (double)H;
				ResizedW = Long > 1.0e12 ? (int64)1000000000000 : (int64)Long;
			}
			break;
		}
		case EPipelineImageResize::StretchCenterCrop:
		default:
			ResizedW = Settings.ResizeWidth;
			ResizedH = Settings.ResizeHeight;
			break;
		}
		if (!Private::IsResizeSizeAllowed(ResizedW, ResizedH))
		{
			return Fail(EPipelineError::PreprocessFailed, FString::Printf(
				TEXT("Image resize size %lld x %lld (from %d x %d) is outside the limits (sides 1..65536, at most 2^26 pixels)."),
				(long long)ResizedW, (long long)ResizedH, W, H));
		}

		// 3. Crop window (floor; Stretch has none).
		const int32 CropW = Settings.ImageWidth;
		const int32 CropH = Settings.ImageHeight;
		const int32 Left = Settings.ResizeMode == EPipelineImageResize::Stretch ? 0 : (int32)((ResizedW - CropW) / 2);
		const int32 Top = Settings.ResizeMode == EPipelineImageResize::Stretch ? 0 : (int32)((ResizedH - CropH) / 2);
		if (Left < 0 || Top < 0 || Left + CropW > ResizedW || Top + CropH > ResizedH)
		{
			// Unreachable after CheckSettings (resize >= crop); kept as a guard, never a crash.
			return Fail(EPipelineError::PreprocessFailed, FString::Printf(
				TEXT("Crop %d x %d does not fit the resized image %lld x %lld."), CropW, CropH, (long long)ResizedW, (long long)ResizedH));
		}

		// 2. Resize, only the crop window (bit-identical to resizing everything and cropping).
		FPipelineImage Cropped;
		Private::ResizeImageRegion(In, (int32)ResizedW, (int32)ResizedH, Settings.Resample, Left, Top, CropW, CropH, Cropped);

		// 4. Rescale + normalize, float32, planes R, G, B.
		const double Rescale = (double)Settings.RescaleFactor;
		const int32 Plane = CropW * CropH;
		OutCHW.SetNumUninitialized(3 * Plane);
		for (int32 c = 0; c < 3; ++c)
		{
			const float Mean = (float)Settings.ImageMean[c];
			const float Std = (float)Settings.ImageStd[c];
			float Lut[256];
			for (int32 p = 0; p < 256; ++p)
			{
				const float V = (float)((double)p * Rescale);
				Lut[p] = (V - Mean) / Std;
			}
			float* Dst = OutCHW.GetData() + (int64)c * Plane;
			const FColor* Src = Cropped.Pixels.GetData();
			if (c == 0)
			{
				for (int32 i = 0; i < Plane; ++i) { Dst[i] = Lut[Src[i].R]; }
			}
			else if (c == 1)
			{
				for (int32 i = 0; i < Plane; ++i) { Dst[i] = Lut[Src[i].G]; }
			}
			else
			{
				for (int32 i = 0; i < Plane; ++i) { Dst[i] = Lut[Src[i].B]; }
			}
		}

		if (OutGeometry)
		{
			OutGeometry->SourceWidth = W;
			OutGeometry->SourceHeight = H;
			OutGeometry->ResizedWidth = (int32)ResizedW;
			OutGeometry->ResizedHeight = (int32)ResizedH;
			OutGeometry->CropLeft = Left;
			OutGeometry->CropTop = Top;
			OutGeometry->CropWidth = CropW;
			OutGeometry->CropHeight = CropH;
		}
		if (OutCropped)
		{
			*OutCropped = MoveTemp(Cropped);
		}
		return FPipelineResult::Ok();
	}
}
