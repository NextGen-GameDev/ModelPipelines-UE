// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.
// WP4.5 visible tests (docs/contracts.md C27 pixel semantics, C30 goldens, C33 tolerances): FPipelineImage, the Pillow-exact
// ResizeImage and the PreprocessImage chain. Expected values for the data-driven tests come from the golden files at run time.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include <limits>
#include "Async/Async.h"
#include "Async/Future.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeBool.h"
#include "Engine/Texture2D.h"
#include "Math/RandomStream.h"
#include "UObject/StrongObjectPtr.h"
#include "PipelineImage.h"
#include "PipelineModelPackage.h"
#include "PipelineTypes.h"
#include "PipelineVisionTestData.h"

using namespace UE::ModelPipelines::VisionTests;
using UE::ModelPipelines::Vision::ResizeImage;
using UE::ModelPipelines::Vision::PreprocessImage;

namespace PipelineImageSpecPrivate
{
	const TCHAR* const FilterNames[] = { TEXT("Nearest"), TEXT("Lanczos"), TEXT("Bilinear"), TEXT("Bicubic"), TEXT("Box"), TEXT("Hamming") };
	const TCHAR* const ConfigNames[] = { TEXT("convnext_crop"), TEXT("rtdetr_stretch"), TEXT("deit_crop"), TEXT("deit_rect_crop"), TEXT("clip_crop"), TEXT("vit_stretch"), TEXT("levit_crop") };

	static TSharedPtr<FJsonObject> GCachedRoot;
	static TMap<FString, FPipelineImage> GImageCache;

	static TSharedPtr<FJsonObject> GetGoldens(FAutomationTestBase& T)
	{
		if (!GCachedRoot.IsValid())
		{
			FString Err;
			GCachedRoot = LoadVisionGoldens(Err);
			if (!GCachedRoot.IsValid())
			{
				T.AddError(Err);
			}
		}
		return GCachedRoot;
	}

	/** Golden image by id (cached; goldens are large). nullptr + AddError when missing. */
	static const FPipelineImage* GetGoldenImage(FAutomationTestBase& T, const TSharedPtr<FJsonObject>& Root, const FString& Id)
	{
		if (const FPipelineImage* Cached = GImageCache.Find(Id))
		{
			return Cached;
		}
		const TArray<TSharedPtr<FJsonValue>>* Images = nullptr;
		if (Root.IsValid() && Root->TryGetArrayField(TEXT("images"), Images))
		{
			for (const TSharedPtr<FJsonValue>& V : *Images)
			{
				const TSharedPtr<FJsonObject>& O = V->AsObject();
				if (O.IsValid() && O->GetStringField(TEXT("id")) == Id)
				{
					FPipelineImage Img;
					FString Err;
					if (!LoadPng(Root->GetStringField(TEXT("root")) / O->GetStringField(TEXT("file")), Img, Err))
					{
						T.AddError(Err);
						return nullptr;
					}
					return &GImageCache.Add(Id, MoveTemp(Img));
				}
			}
		}
		T.AddError(FString::Printf(TEXT("Golden image id '%s' not listed in vision_goldens.json."), *Id));
		return nullptr;
	}

	static FPipelineImage MakeImage(int32 W, int32 H, TFunctionRef<FColor(int32, int32)> Fn)
	{
		FPipelineImage Img;
		Img.Width = W;
		Img.Height = H;
		Img.Pixels.SetNumUninitialized(W * H);
		for (int32 y = 0; y < H; ++y)
		{
			for (int32 x = 0; x < W; ++x)
			{
				Img.Pixels[y * W + x] = Fn(x, y);
			}
		}
		return Img;
	}

	static FPipelineImage MakeNoise(int32 W, int32 H, int32 Seed, uint8 Alpha = 255)
	{
		FRandomStream Rng(Seed);
		FPipelineImage Img;
		Img.Width = W;
		Img.Height = H;
		Img.Pixels.SetNumUninitialized(W * H);
		for (FColor& C : Img.Pixels)
		{
			C = FColor((uint8)Rng.RandRange(0, 255), (uint8)Rng.RandRange(0, 255), (uint8)Rng.RandRange(0, 255), Alpha);
		}
		return Img;
	}

	static FPipelinePreprocessSettings CropSettings()
	{
		FPipelinePreprocessSettings S;
		S.ImageWidth = 224;
		S.ImageHeight = 224;
		S.ResizeMode = EPipelineImageResize::ShortestEdgeCenterCrop;
		S.ResizeShortestEdge = 256;
		S.Resample = EPipelineImageResample::Bicubic;
		S.RescaleFactor = (float)(1.0 / 255.0);
		S.ImageMean = FVector(0.485, 0.456, 0.406);
		S.ImageStd = FVector(0.229, 0.224, 0.225);
		return S;
	}

	/** Byte comparison with the first mismatch named. Alpha is compared too. */
	static bool CompareImages(const FPipelineImage& Expected, const FPipelineImage& Actual, FString& OutMsg, int32& OutMaxDiff)
	{
		OutMaxDiff = 0;
		if (Expected.Width != Actual.Width || Expected.Height != Actual.Height || Expected.Pixels.Num() != Actual.Pixels.Num())
		{
			OutMsg = FString::Printf(TEXT("size %dx%d (%d px) != expected %dx%d (%d px)"), Actual.Width, Actual.Height, Actual.Pixels.Num(), Expected.Width, Expected.Height, Expected.Pixels.Num());
			return false;
		}
		bool bEqual = true;
		for (int32 i = 0; i < Expected.Pixels.Num(); ++i)
		{
			const FColor E = Expected.Pixels[i];
			const FColor A = Actual.Pixels[i];
			const int32 D = FMath::Max(FMath::Max(FMath::Abs((int32)E.R - A.R), FMath::Abs((int32)E.G - A.G)), FMath::Max(FMath::Abs((int32)E.B - A.B), FMath::Abs((int32)E.A - A.A)));
			if (D > 0 && bEqual)
			{
				bEqual = false;
				OutMsg = FString::Printf(TEXT("first mismatch at pixel (%d,%d): got RGBA(%d,%d,%d,%d), expected (%d,%d,%d,%d)"), i % Expected.Width, i / Expected.Width, A.R, A.G, A.B, A.A, E.R, E.G, E.B, E.A);
			}
			OutMaxDiff = FMath::Max(OutMaxDiff, D);
		}
		return bEqual;
	}

	static bool ImagesEqual(const FPipelineImage& A, const FPipelineImage& B)
	{
		FString Msg;
		int32 Max = 0;
		return CompareImages(A, B, Msg, Max);
	}

	static bool SameRgb(const FPipelineImage& A, const FPipelineImage& B)
	{
		if (A.Width != B.Width || A.Height != B.Height || A.Pixels.Num() != B.Pixels.Num())
		{
			return false;
		}
		for (int32 i = 0; i < A.Pixels.Num(); ++i)
		{
			if (A.Pixels[i].R != B.Pixels[i].R || A.Pixels[i].G != B.Pixels[i].G || A.Pixels[i].B != B.Pixels[i].B)
			{
				return false;
			}
		}
		return true;
	}

	static const TArray<TSharedPtr<FJsonValue>>* GetCases(FAutomationTestBase& T, const TSharedPtr<FJsonObject>& Root, const TCHAR* Field)
	{
		const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
		if (!Root.IsValid() || !Root->TryGetArrayField(Field, Cases))
		{
			T.AddError(FString::Printf(TEXT("vision_goldens.json has no '%s' array."), Field));
			return nullptr;
		}
		return Cases;
	}

	/** Every resize_case with this filter name: ResizeImage must equal the Pillow PNG byte for byte. */
	static void RunResizeFilter(FAutomationTestBase& T, const FString& FilterName)
	{
		const TSharedPtr<FJsonObject> Root = GetGoldens(T);
		const TArray<TSharedPtr<FJsonValue>>* Cases = GetCases(T, Root, TEXT("resize_cases"));
		if (!Cases)
		{
			return;
		}
		int32 Checked = 0;
		int32 Failed = 0;
		int32 WorstDiff = 0;
		FString WorstCase;
		for (const TSharedPtr<FJsonValue>& V : *Cases)
		{
			const TSharedPtr<FJsonObject> C = V->AsObject();
			if (!C.IsValid() || C->GetStringField(TEXT("filter")) != FilterName)
			{
				continue;
			}
			const FString Id = C->GetStringField(TEXT("id"));
			EPipelineImageResample Filter;
			if (!ParseResample(FilterName, Filter))
			{
				T.AddError(FString::Printf(TEXT("[%s] unknown filter name '%s' in goldens"), *Id, *FilterName));
				++Failed;
				continue;
			}
			const FPipelineImage* Src = GetGoldenImage(T, Root, C->GetStringField(TEXT("image")));
			FPipelineImage Expected;
			FString Err;
			if (!Src || !LoadPng(Root->GetStringField(TEXT("root")) / C->GetStringField(TEXT("expected")), Expected, Err))
			{
				if (!Err.IsEmpty())
				{
					T.AddError(Err);
				}
				++Failed;
				continue;
			}
			++Checked;
			const int32 W = (int32)C->GetNumberField(TEXT("width"));
			const int32 H = (int32)C->GetNumberField(TEXT("height"));
			FPipelineImage Out;
			const bool bOk = ResizeImage(*Src, W, H, Filter, Out);
			if (!bOk)
			{
				T.AddError(FString::Printf(TEXT("[%s] %s %s %dx%d -> %dx%d: ResizeImage returned false"), *Id, *FilterName, *C->GetStringField(TEXT("image")), Src->Width, Src->Height, W, H));
				++Failed;
				continue;
			}
			FString Msg;
			int32 MaxDiff = 0;
			if (!CompareImages(Expected, Out, Msg, MaxDiff))
			{
				T.AddError(FString::Printf(TEXT("[%s] %s %s %dx%d -> %dx%d not bit-exact: %s (max byte diff %d)"), *Id, *FilterName, *C->GetStringField(TEXT("image")), Src->Width, Src->Height, W, H, *Msg, MaxDiff));
				++Failed;
			}
			if (MaxDiff > WorstDiff)
			{
				WorstDiff = MaxDiff;
				WorstCase = Id;
			}
		}
		// Bug caught: a filter with no cases silently skipped (a typo in a name) would pass with zero checks.
		T.TestEqual(*FString::Printf(TEXT("%s resize case count (C30: 10 size pairs per filter)"), *FilterName), Checked + 0, 10);
		T.AddInfo(FString::Printf(TEXT("%s: %d cases checked, %d failed, worst byte diff %d%s"), *FilterName, Checked, Failed, WorstDiff, WorstCase.IsEmpty() ? TEXT("") : *FString::Printf(TEXT(" (%s)"), *WorstCase)));
	}

	/** One preprocess golden case: cropped image bytes, geometry, tensor within 1e-5. Returns the max tensor diff via OutMax. */
	static bool RunPreprocessCase(FAutomationTestBase& T, const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& C, double& OutMaxTensorDiff)
	{
		const FString Id = C->GetStringField(TEXT("id"));
		const FString ImageId = C->GetStringField(TEXT("image"));
		const FString Config = C->GetStringField(TEXT("config"));
		const FString Tag = FString::Printf(TEXT("[%s %s %s]"), *Id, *Config, *ImageId);
		const FString GoldRoot = Root->GetStringField(TEXT("root"));

		FPipelinePreprocessSettings Settings;
		FString Err;
		if (!ParseSettings(C->GetObjectField(TEXT("settings")), Settings, Err))
		{
			T.AddError(Tag + TEXT(" ") + Err);
			return false;
		}
		const FPipelineImage* Src = GetGoldenImage(T, Root, ImageId);
		FPipelineImage ExpectedCropped;
		TArray<float> ExpectedTensor;
		if (!Src || !LoadPng(GoldRoot / C->GetStringField(TEXT("cropped")), ExpectedCropped, Err) || !LoadF32(GoldRoot / C->GetStringField(TEXT("tensor")), ExpectedTensor, Err))
		{
			if (!Err.IsEmpty())
			{
				T.AddError(Tag + TEXT(" ") + Err);
			}
			return false;
		}

		TArray<float> Tensor;
		FPipelineImage Cropped;
		FPipelineImageGeometry Geo;
		const FPipelineResult R = PreprocessImage(*Src, Settings, Tensor, &Cropped, &Geo);
		if (!R.bSuccess)
		{
			T.AddError(FString::Printf(TEXT("%s PreprocessImage failed: error %d '%s'"), *Tag, (int32)R.Error, *R.Message));
			return false;
		}
		bool bOk = true;

		// Geometry: resized [w,h], crop [left, top, w, h], source size.
		const TArray<TSharedPtr<FJsonValue>>& Resized = C->GetArrayField(TEXT("resized"));
		const TArray<TSharedPtr<FJsonValue>>& Crop = C->GetArrayField(TEXT("crop"));
		if (Resized.Num() != 2 || Crop.Num() != 4)
		{
			T.AddError(Tag + TEXT(" golden 'resized'/'crop' arrays have the wrong length"));
			return false;
		}
		const int32 Exp[8] = { (int32)Resized[0]->AsNumber(), (int32)Resized[1]->AsNumber(), (int32)Crop[0]->AsNumber(), (int32)Crop[1]->AsNumber(), (int32)Crop[2]->AsNumber(), (int32)Crop[3]->AsNumber(), Src->Width, Src->Height };
		const int32 Act[8] = { Geo.ResizedWidth, Geo.ResizedHeight, Geo.CropLeft, Geo.CropTop, Geo.CropWidth, Geo.CropHeight, Geo.SourceWidth, Geo.SourceHeight };
		if (FMemory::Memcmp(Exp, Act, sizeof(Exp)) != 0)
		{
			T.AddError(FString::Printf(TEXT("%s geometry resized %dx%d crop (%d,%d,%d,%d) source %dx%d, expected resized %dx%d crop (%d,%d,%d,%d) source %dx%d"), *Tag,
				Act[0], Act[1], Act[2], Act[3], Act[4], Act[5], Act[6], Act[7], Exp[0], Exp[1], Exp[2], Exp[3], Exp[4], Exp[5], Exp[6], Exp[7]));
			bOk = false;
		}

		// Cropped 8-bit image: every byte equal.
		FString Msg;
		int32 MaxByteDiff = 0;
		if (!CompareImages(ExpectedCropped, Cropped, Msg, MaxByteDiff))
		{
			T.AddError(FString::Printf(TEXT("%s OutCropped differs from the golden: %s (max byte diff %d)"), *Tag, *Msg, MaxByteDiff));
			bOk = false;
		}

		// Tensor shape 3 x H x W and values within 1e-5.
		const TArray<TSharedPtr<FJsonValue>>& Shape = C->GetArrayField(TEXT("tensor_shape"));
		const int32 H = Settings.ImageHeight;
		const int32 W = Settings.ImageWidth;
		if (Shape.Num() != 4 || (int32)Shape[1]->AsNumber() != 3 || (int32)Shape[2]->AsNumber() != H || (int32)Shape[3]->AsNumber() != W)
		{
			T.AddError(Tag + TEXT(" golden tensor_shape is not [1,3,ImageHeight,ImageWidth]"));
			bOk = false;
		}
		if (Tensor.Num() != 3 * H * W || ExpectedTensor.Num() != 3 * H * W)
		{
			T.AddError(FString::Printf(TEXT("%s tensor has %d floats (golden file %d), expected 3*%d*%d = %d"), *Tag, Tensor.Num(), ExpectedTensor.Num(), H, W, 3 * H * W));
			return false;
		}
		double MaxDiff = 0.0;
		int32 FirstBad = INDEX_NONE;
		for (int32 i = 0; i < Tensor.Num(); ++i)
		{
			const double D = FMath::Abs((double)Tensor[i] - (double)ExpectedTensor[i]);
			// A NaN diff must fail too (!(D <= tol)).
			if (!(D <= 1e-5) && FirstBad == INDEX_NONE)
			{
				FirstBad = i;
			}
			if (D > MaxDiff || FMath::IsNaN(D))
			{
				MaxDiff = D;
			}
		}
		OutMaxTensorDiff = FMath::Max(OutMaxTensorDiff, MaxDiff);
		if (FirstBad != INDEX_NONE)
		{
			const int32 Plane = FirstBad / (H * W);
			const int32 Rem = FirstBad % (H * W);
			T.AddError(FString::Printf(TEXT("%s tensor mismatch > 1e-5 at index %d (channel %d, y %d, x %d): got %.8g, expected %.8g; max |diff| %.3g"), *Tag, FirstBad, Plane, Rem / W, Rem % W, Tensor[FirstBad], ExpectedTensor[FirstBad], MaxDiff));
			bOk = false;
		}
		return bOk;
	}

	static void RunPreprocessConfig(FAutomationTestBase& T, const FString& Config)
	{
		const TSharedPtr<FJsonObject> Root = GetGoldens(T);
		const TArray<TSharedPtr<FJsonValue>>* Cases = GetCases(T, Root, TEXT("preprocess_cases"));
		if (!Cases)
		{
			return;
		}
		int32 Checked = 0;
		int32 Failed = 0;
		double MaxDiff = 0.0;
		for (const TSharedPtr<FJsonValue>& V : *Cases)
		{
			const TSharedPtr<FJsonObject> C = V->AsObject();
			if (!C.IsValid() || C->GetStringField(TEXT("config")) != Config)
			{
				continue;
			}
			++Checked;
			if (!RunPreprocessCase(T, Root, C, MaxDiff))
			{
				++Failed;
			}
		}
		// C30: 8 images per config, rtdetr_stretch only 4.
		T.TestEqual(*FString::Printf(TEXT("%s preprocess case count"), *Config), Checked, Config == TEXT("rtdetr_stretch") ? 4 : 8);
		T.AddInfo(FString::Printf(TEXT("%s: %d cases checked, %d failed, max tensor |diff| %.3g (limit 1e-5)"), *Config, Checked, Failed, MaxDiff));
	}

	/** Hand-written reference of the C27 step 4 arithmetic (float32), straight from the contract text. */
	static float ExpectedValue(uint8 P, const FPipelinePreprocessSettings& S, int32 Channel)
	{
		const float V = (float)((double)P * (double)S.RescaleFactor);
		const float Mean = (float)S.ImageMean[Channel];
		const float Std = (float)S.ImageStd[Channel];
		return (V - Mean) / Std;
	}

	/** Checks tensor == formula applied to the 8-bit cropped image, plane order R, G, B, row-major. */
	static bool CheckTensorFromCropped(FAutomationTestBase& T, const FString& Tag, const FPipelineImage& Cropped, const TArray<float>& Tensor, const FPipelinePreprocessSettings& S)
	{
		const int32 W = Cropped.Width;
		const int32 H = Cropped.Height;
		if (Tensor.Num() != 3 * W * H)
		{
			T.AddError(FString::Printf(TEXT("%s tensor has %d floats, expected %d"), *Tag, Tensor.Num(), 3 * W * H));
			return false;
		}
		for (int32 y = 0; y < H; ++y)
		{
			for (int32 x = 0; x < W; ++x)
			{
				const FColor P = Cropped.Pixels[y * W + x];
				const uint8 Ch[3] = { P.R, P.G, P.B };
				for (int32 c = 0; c < 3; ++c)
				{
					const float E = ExpectedValue(Ch[c], S, c);
					const float A = Tensor[c * W * H + y * W + x];
					if (!(FMath::Abs(A - E) <= 1e-6f * FMath::Max(1.f, FMath::Abs(E))))
					{
						T.AddError(FString::Printf(TEXT("%s tensor[c=%d,y=%d,x=%d] = %.8g, expected %.8g for pixel value %d"), *Tag, c, y, x, A, E, Ch[c]));
						return false;
					}
				}
			}
		}
		return true;
	}

	struct FGeometryCase
	{
		const TCHAR* Name;
		int32 SrcW, SrcH;
		EPipelineImageResize Mode;
		int32 ImgW, ImgH, Shortest, RW, RH;
		EPipelineImageResample Filter;
		int32 ExpResW, ExpResH, ExpLeft, ExpTop;
	};

	struct FThreadJob
	{
		FPipelineImage Input;
		FPipelinePreprocessSettings Settings;
		EPipelineImageResample Filter = EPipelineImageResample::Bilinear;
		int32 OutW = 0, OutH = 0;
		bool bResizeOnly = false;
		// results
		TArray<float> Tensor;
		FPipelineImage Cropped;
		FPipelineImageGeometry Geo;
		bool bSuccess = false;
		int32 Mismatches = 0;
		FThreadSafeBool bDone = false;
	};

	static void RunJobOnce(FThreadJob& J, bool bCompareToSelf)
	{
		if (J.bResizeOnly)
		{
			FPipelineImage Out;
			const bool bOk = ResizeImage(J.Input, J.OutW, J.OutH, J.Filter, Out);
			if (bCompareToSelf)
			{
				if (!bOk || !ImagesEqual(Out, J.Cropped))
				{
					++J.Mismatches;
				}
			}
			else
			{
				J.bSuccess = bOk;
				J.Cropped = MoveTemp(Out);
			}
			return;
		}
		TArray<float> Tensor;
		FPipelineImage Cropped;
		FPipelineImageGeometry Geo;
		const FPipelineResult R = PreprocessImage(J.Input, J.Settings, Tensor, &Cropped, &Geo);
		if (bCompareToSelf)
		{
			const bool bSame = R.bSuccess && Tensor.Num() == J.Tensor.Num()
				&& FMemory::Memcmp(Tensor.GetData(), J.Tensor.GetData(), Tensor.Num() * sizeof(float)) == 0
				&& ImagesEqual(Cropped, J.Cropped)
				&& Geo.ResizedWidth == J.Geo.ResizedWidth && Geo.ResizedHeight == J.Geo.ResizedHeight && Geo.CropLeft == J.Geo.CropLeft && Geo.CropTop == J.Geo.CropTop;
			if (!bSame)
			{
				++J.Mismatches;
			}
		}
		else
		{
			J.bSuccess = R.bSuccess;
			J.Tensor = MoveTemp(Tensor);
			J.Cropped = MoveTemp(Cropped);
			J.Geo = Geo;
		}
	}
}

BEGIN_DEFINE_SPEC(FPipelineImageSpec, "Pipelines.Vision.Image", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FPipelineImageSpec)

void FPipelineImageSpec::Define()
{
	using namespace PipelineImageSpecPrivate;

	// ------------------------------------------------------------------------------------------------------------------
	Describe("Goldens", [this]()
	{
		It("ImagesLoadAndMatchChecksums", [this]()
		{
			// Guards the helper and the files: if LoadPng swapped channels or dropped a row, every later test would fail confusingly.
			const TSharedPtr<FJsonObject> Root = GetGoldens(*this);
			const TArray<TSharedPtr<FJsonValue>>* Images = GetCases(*this, Root, TEXT("images"));
			if (!Images)
			{
				return;
			}
			TestTrue(TEXT("at least 10 golden images"), Images->Num() >= 10);
			for (const TSharedPtr<FJsonValue>& V : *Images)
			{
				const TSharedPtr<FJsonObject> O = V->AsObject();
				const FString Id = O->GetStringField(TEXT("id"));
				const FPipelineImage* Img = GetGoldenImage(*this, Root, Id);
				if (!Img)
				{
					continue;
				}
				TestEqual(*FString::Printf(TEXT("[%s] width"), *Id), Img->Width, (int32)O->GetNumberField(TEXT("width")));
				TestEqual(*FString::Printf(TEXT("[%s] height"), *Id), Img->Height, (int32)O->GetNumberField(TEXT("height")));
				TestTrue(*FString::Printf(TEXT("[%s] IsValid"), *Id), Img->IsValid());
				int64 Sum = 0;
				for (const FColor& C : Img->Pixels)
				{
					Sum += (int64)C.R + C.G + C.B;
				}
				TestEqual(*FString::Printf(TEXT("[%s] rgb_sum"), *Id), Sum, (int64)O->GetNumberField(TEXT("rgb_sum")));
				for (const TSharedPtr<FJsonValue>& PV : O->GetArrayField(TEXT("probes")))
				{
					const TSharedPtr<FJsonObject> P = PV->AsObject();
					const int32 X = (int32)P->GetNumberField(TEXT("x"));
					const int32 Y = (int32)P->GetNumberField(TEXT("y"));
					const TArray<TSharedPtr<FJsonValue>>& Rgb = P->GetArrayField(TEXT("rgb"));
					if (X < Img->Width && Y < Img->Height && Rgb.Num() == 3)
					{
						const FColor C = Img->Pixels[Y * Img->Width + X];
						if (C.R != (int32)Rgb[0]->AsNumber() || C.G != (int32)Rgb[1]->AsNumber() || C.B != (int32)Rgb[2]->AsNumber())
						{
							AddError(FString::Printf(TEXT("[%s] probe (%d,%d) is RGB(%d,%d,%d), golden says (%d,%d,%d)"), *Id, X, Y, C.R, C.G, C.B, (int32)Rgb[0]->AsNumber(), (int32)Rgb[1]->AsNumber(), (int32)Rgb[2]->AsNumber()));
						}
					}
				}
			}
		});

		It("CaseCounts", [this]()
		{
			const TSharedPtr<FJsonObject> Root = GetGoldens(*this);
			if (!Root.IsValid())
			{
				return;
			}
			const TArray<TSharedPtr<FJsonValue>>* Resize = GetCases(*this, Root, TEXT("resize_cases"));
			const TArray<TSharedPtr<FJsonValue>>* Pre = GetCases(*this, Root, TEXT("preprocess_cases"));
			if (!Resize || !Pre)
			{
				return;
			}
			TestEqual(TEXT("resize_cases (C30: 6 filters x 10 size pairs)"), Resize->Num(), 60);
			TestEqual(TEXT("preprocess_cases (C30: 6 configs x 8 images + rtdetr_stretch x 4)"), Pre->Num(), 52);
			// Bug caught: a config name the data-driven tests don't know would be skipped without a trace.
			int32 Known = 0;
			for (const TSharedPtr<FJsonValue>& V : *Pre)
			{
				const FString Config = V->AsObject()->GetStringField(TEXT("config"));
				for (const TCHAR* N : ConfigNames)
				{
					Known += (Config == N) ? 1 : 0;
				}
			}
			TestEqual(TEXT("preprocess cases whose config the tests cover"), Known, Pre->Num());
		});
	});

	// ------------------------------------------------------------------------------------------------------------------
	Describe("IsValid", [this]()
	{
		It("AcceptsWellFormedImages", [this]()
		{
			FPipelineImage One = MakeNoise(1, 1, 1);
			FString Reason = TEXT("untouched");
			TestTrue(TEXT("1x1 valid"), One.IsValid(&Reason));
			TestTrue(TEXT("1x1 valid with null reason"), One.IsValid());
			TestTrue(TEXT("16384x1 valid (MaxSide)"), MakeNoise(FPipelineImage::MaxSide, 1, 2).IsValid());
			TestTrue(TEXT("1x16384 valid (MaxSide)"), MakeNoise(1, FPipelineImage::MaxSide, 3).IsValid());
			TestEqual(TEXT("MaxSide"), FPipelineImage::MaxSide, 16384);
		});

		It("RejectsMalformedImagesAndNamesTheRule", [this]()
		{
			// Bug caught: IsValid that only checks Width/Height, or an off-by-one at MaxSide (16385 must fail).
			struct FBad { const TCHAR* Name; FPipelineImage Img; };
			TArray<FBad> Bad;
			{ FPipelineImage I; Bad.Add({ TEXT("default (0x0)"), I }); }
			{ FPipelineImage I; I.Width = 0; I.Height = 5; Bad.Add({ TEXT("0x5, no pixels"), I }); }
			{ FPipelineImage I; I.Width = 5; I.Height = 0; Bad.Add({ TEXT("5x0, no pixels"), I }); }
			{ FPipelineImage I = MakeNoise(4, 4, 1); I.Width = -4; Bad.Add({ TEXT("negative width"), I }); }
			{ FPipelineImage I = MakeNoise(4, 4, 1); I.Pixels.Pop(); Bad.Add({ TEXT("pixel count one short"), I }); }
			{ FPipelineImage I = MakeNoise(4, 4, 1); I.Pixels.Add(FColor::Black); Bad.Add({ TEXT("pixel count one too many"), I }); }
			{ FPipelineImage I = MakeNoise(3, 4, 1); I.Width = 4; I.Height = 3; I.Pixels.SetNum(11); Bad.Add({ TEXT("4x3 with 11 pixels"), I }); }
			Bad.Add({ TEXT("16385x1"), MakeNoise(FPipelineImage::MaxSide + 1, 1, 4) });
			Bad.Add({ TEXT("1x16385"), MakeNoise(1, FPipelineImage::MaxSide + 1, 5) });
			{ FPipelineImage I; I.Width = FPipelineImage::MaxSide; I.Height = FPipelineImage::MaxSide; Bad.Add({ TEXT("16384x16384 with no pixels"), I }); }
			for (const FBad& B : Bad)
			{
				FString Reason;
				TestFalse(*FString::Printf(TEXT("%s is invalid"), B.Name), B.Img.IsValid(&Reason));
				TestFalse(*FString::Printf(TEXT("%s: OutReason names the rule"), B.Name), Reason.IsEmpty());
				TestFalse(*FString::Printf(TEXT("%s is invalid (null reason)"), B.Name), B.Img.IsValid());
			}
		});
	});

	// ------------------------------------------------------------------------------------------------------------------
	Describe("Resize", [this]()
	{
		for (const TCHAR* Name : FilterNames)
		{
			const FString FilterName(Name);
			It(*FString::Printf(TEXT("Golden.%s"), Name), [this, FilterName]()
			{
				RunResizeFilter(*this, FilterName);
			});
		}

		It("BilinearUpscaleHandComputed", [this]()
		{
			// 2x1 [0,255] -> 4x1 bilinear, worked by hand from the C27 formulas:
			//  i0: single tap -> 0; i1: weights .75/.25 -> 255*.25 = 63.75, +.5 = 64.25 -> 64; i2: .25/.75 -> 191.25 + .5 -> 191; i3: single tap -> 255.
			// Bug caught: truncating instead of rounding the fixed-point sum (63 / 191), or not normalising weights at the border.
			const FPipelineImage In = MakeImage(2, 1, [](int32 x, int32) { return x == 0 ? FColor(0, 0, 0, 255) : FColor(255, 255, 255, 255); });
			FPipelineImage Out;
			TestTrue(TEXT("ResizeImage ok"), ResizeImage(In, 4, 1, EPipelineImageResample::Bilinear, Out));
			TestEqual(TEXT("width"), Out.Width, 4);
			TestEqual(TEXT("height"), Out.Height, 1);
			if (Out.Pixels.Num() == 4)
			{
				const int32 Expected[4] = { 0, 64, 191, 255 };
				for (int32 i = 0; i < 4; ++i)
				{
					TestEqual(*FString::Printf(TEXT("R[%d]"), i), (int32)Out.Pixels[i].R, Expected[i]);
					TestEqual(*FString::Printf(TEXT("G[%d]"), i), (int32)Out.Pixels[i].G, Expected[i]);
					TestEqual(*FString::Printf(TEXT("B[%d]"), i), (int32)Out.Pixels[i].B, Expected[i]);
					TestEqual(*FString::Printf(TEXT("A[%d] is 255"), i), (int32)Out.Pixels[i].A, 255);
				}
			}
			else
			{
				AddError(FString::Printf(TEXT("expected 4 pixels, got %d"), Out.Pixels.Num()));
			}
		});

		It("BoxDownscaleRoundsTheIntermediateRowToUint8", [this]()
		{
			// 2x2 -> 1x1 Box. Pillow runs the horizontal pass first and stores uint8, then the vertical pass.
			// R [0 100; 100 200] -> rows 50, 150 -> 100. G all 255 -> 255.
			// B [0 1; 0 0]: row0 = (0+1)/2 = .5 + .5 -> 1, row1 = 0, vertical (1+0)/2 = .5 + .5 -> 1. A single-pass float average
			// (.25 + .5 -> 0) would give 0: that is the bug this catches.
			FPipelineImage In;
			In.Width = 2;
			In.Height = 2;
			In.Pixels = { FColor(0, 255, 0, 255), FColor(100, 255, 1, 255), FColor(100, 255, 0, 255), FColor(200, 255, 0, 255) };
			FPipelineImage Out;
			TestTrue(TEXT("ResizeImage ok"), ResizeImage(In, 1, 1, EPipelineImageResample::Box, Out));
			if (Out.Pixels.Num() == 1)
			{
				TestEqual(TEXT("R"), (int32)Out.Pixels[0].R, 100);
				TestEqual(TEXT("G"), (int32)Out.Pixels[0].G, 255);
				TestEqual(TEXT("B (two-pass uint8 rounding)"), (int32)Out.Pixels[0].B, 1);
				TestEqual(TEXT("A"), (int32)Out.Pixels[0].A, 255);
			}
			else
			{
				AddError(FString::Printf(TEXT("expected 1 pixel, got %d"), Out.Pixels.Num()));
			}
		});

		It("NearestPicksPillowsSourceIndex", [this]()
		{
			// C27 Nearest: source x = (int)(0.5*scale + i*scale). 2 -> 4 (scale .5): 0,0,1,1. 4 -> 2 (scale 2): 1, 3 (not 0, 2).
			// Bug caught: floor(i*scale) (0, 2) or round-half-up variants.
			const FPipelineImage Up = MakeImage(2, 1, [](int32 x, int32) { return x == 0 ? FColor(10, 20, 30, 255) : FColor(200, 210, 220, 255); });
			FPipelineImage Out;
			TestTrue(TEXT("upscale ok"), ResizeImage(Up, 4, 1, EPipelineImageResample::Nearest, Out));
			if (Out.Pixels.Num() == 4)
			{
				const int32 Src[4] = { 0, 0, 1, 1 };
				for (int32 i = 0; i < 4; ++i)
				{
					TestEqual(*FString::Printf(TEXT("up[%d]"), i), (int64)Out.Pixels[i].ToPackedARGB(), (int64)Up.Pixels[Src[i]].ToPackedARGB());
				}
			}
			else
			{
				AddError(FString::Printf(TEXT("upscale: expected 4 pixels, got %d"), Out.Pixels.Num()));
			}
			const FPipelineImage Down = MakeImage(4, 1, [](int32 x, int32) { return FColor((uint8)(x * 50), (uint8)(x * 50 + 1), (uint8)(x * 50 + 2), 255); });
			TestTrue(TEXT("downscale ok"), ResizeImage(Down, 2, 1, EPipelineImageResample::Nearest, Out));
			if (Out.Pixels.Num() == 2)
			{
				TestEqual(TEXT("down[0] is source pixel 1"), (int32)Out.Pixels[0].R, 50);
				TestEqual(TEXT("down[1] is source pixel 3"), (int32)Out.Pixels[1].R, 150);
			}
			else
			{
				AddError(FString::Printf(TEXT("downscale: expected 2 pixels, got %d"), Out.Pixels.Num()));
			}
		});

		It("ConstantImageStaysConstantForEveryFilter", [this]()
		{
			// Normalised weights summing to 2^22 (up to fixed-point rounding) must leave a flat colour unchanged, up and down.
			// Bug caught: weights not divided by their sum, a missing 2^21 rounding term, or clamping errors at the borders.
			const FPipelineImage In = MakeImage(17, 13, [](int32, int32) { return FColor(200, 100, 3, 255); });
			const int32 Sizes[3][2] = { { 5, 31 }, { 40, 7 }, { 17, 13 } };
			for (int32 f = 0; f < 6; ++f)
			{
				for (const int32(&Sz)[2] : Sizes)
				{
					FPipelineImage Out;
					const bool bOk = ResizeImage(In, Sz[0], Sz[1], (EPipelineImageResample)f, Out);
					if (!bOk || Out.Width != Sz[0] || Out.Height != Sz[1] || Out.Pixels.Num() != Sz[0] * Sz[1])
					{
						AddError(FString::Printf(TEXT("%s 17x13 -> %dx%d: ResizeImage failed or wrong size"), FilterNames[f], Sz[0], Sz[1]));
						continue;
					}
					for (int32 i = 0; i < Out.Pixels.Num(); ++i)
					{
						if (Out.Pixels[i] != FColor(200, 100, 3, 255))
						{
							AddError(FString::Printf(TEXT("%s 17x13 -> %dx%d: pixel (%d,%d) = RGBA(%d,%d,%d,%d), expected the constant (200,100,3,255)"), FilterNames[f], Sz[0], Sz[1], i % Sz[0], i / Sz[0], Out.Pixels[i].R, Out.Pixels[i].G, Out.Pixels[i].B, Out.Pixels[i].A));
							break;
						}
					}
				}
			}
		});

		It("SameSizeIsACopy", [this]()
		{
			// Bug caught: an identity resize that still filters (Lanczos ringing) or flips rows.
			const FPipelineImage In = MakeNoise(23, 11, 77);
			for (int32 f = 0; f < 6; ++f)
			{
				FPipelineImage Out;
				TestTrue(*FString::Printf(TEXT("%s ok"), FilterNames[f]), ResizeImage(In, In.Width, In.Height, (EPipelineImageResample)f, Out));
				FString Msg;
				int32 MaxDiff = 0;
				if (!CompareImages(In, Out, Msg, MaxDiff))
				{
					AddError(FString::Printf(TEXT("%s identity resize is not a copy: %s (max diff %d)"), FilterNames[f], *Msg, MaxDiff));
				}
			}
		});

		It("OutputAlphaIs255AndInputAlphaIgnored", [this]()
		{
			// Same RGB with different alpha must resize to identical output whose A is 255 (C27). Bug caught: alpha filtered into the output.
			const FPipelineImage Opaque = MakeNoise(30, 20, 5, 255);
			FPipelineImage Transparent = Opaque;
			FRandomStream Rng(9);
			for (FColor& C : Transparent.Pixels)
			{
				C.A = (uint8)Rng.RandRange(0, 255);
			}
			for (int32 f = 0; f < 6; ++f)
			{
				FPipelineImage A, B;
				TestTrue(TEXT("opaque ok"), ResizeImage(Opaque, 11, 9, (EPipelineImageResample)f, A));
				TestTrue(TEXT("alpha-varied ok"), ResizeImage(Transparent, 11, 9, (EPipelineImageResample)f, B));
				FString Msg;
				int32 MaxDiff = 0;
				if (!CompareImages(A, B, Msg, MaxDiff))
				{
					AddError(FString::Printf(TEXT("%s: input alpha changed the output: %s"), FilterNames[f], *Msg));
				}
				for (const FColor& C : B.Pixels)
				{
					if (C.A != 255)
					{
						AddError(FString::Printf(TEXT("%s: output alpha %d, expected 255"), FilterNames[f], C.A));
						break;
					}
				}
			}
		});

		It("RejectsInvalidInputAndOutOfRangeSizes", [this]()
		{
			// Never crash and never allocate by an unvalidated size: false and Out = {}.
			const FPipelineImage Good = MakeNoise(2, 2, 1);
			auto ExpectFail = [&](const TCHAR* Name, const FPipelineImage& In, int32 W, int32 H)
			{
				FPipelineImage Out = MakeNoise(3, 3, 2); // pre-filled: must be cleared on failure
				const double T0 = FPlatformTime::Seconds();
				const bool bOk = ResizeImage(In, W, H, EPipelineImageResample::Bilinear, Out);
				const double Secs = FPlatformTime::Seconds() - T0;
				TestFalse(*FString::Printf(TEXT("%s returns false"), Name), bOk);
				TestEqual(*FString::Printf(TEXT("%s: Out.Width cleared"), Name), Out.Width, 0);
				TestEqual(*FString::Printf(TEXT("%s: Out.Height cleared"), Name), Out.Height, 0);
				TestEqual(*FString::Printf(TEXT("%s: Out.Pixels cleared"), Name), Out.Pixels.Num(), 0);
				TestTrue(*FString::Printf(TEXT("%s fails fast (%.2fs)"), Name, Secs), Secs < 2.0);
			};
			FPipelineImage Zero;
			Zero.Width = 0; Zero.Height = 5;
			ExpectFail(TEXT("0x5 input"), Zero, 4, 4);
			FPipelineImage Short = MakeNoise(4, 4, 3);
			Short.Pixels.Pop();
			ExpectFail(TEXT("pixel count mismatch"), Short, 4, 4);
			ExpectFail(TEXT("OutWidth 0"), Good, 0, 4);
			ExpectFail(TEXT("OutHeight 0"), Good, 4, 0);
			ExpectFail(TEXT("OutWidth -1"), Good, -1, 4);
			ExpectFail(TEXT("OutHeight -3"), Good, 4, -3);
			ExpectFail(TEXT("OutWidth 65537"), Good, 65537, 1);
			ExpectFail(TEXT("OutHeight 65537"), Good, 1, 65537);
			ExpectFail(TEXT("8193x8193 > 2^26 pixels"), Good, 8193, 8193);
			ExpectFail(TEXT("65536x65536"), Good, 65536, 65536);
		});

		It("AcceptsTheLargestSingleAxisSize", [this]()
		{
			// 65536 is inside the documented 1..65536 range (product 65536 is far below 2^26).
			const FPipelineImage In = MakeNoise(3, 1, 11);
			FPipelineImage Wide, Tall;
			TestTrue(TEXT("3x1 -> 65536x1"), ResizeImage(In, 65536, 1, EPipelineImageResample::Bicubic, Wide));
			TestEqual(TEXT("wide pixel count"), Wide.Pixels.Num(), 65536);
			TestEqual(TEXT("wide width"), Wide.Width, 65536);
			const FPipelineImage InT = MakeNoise(1, 3, 12);
			TestTrue(TEXT("1x3 -> 1x65536"), ResizeImage(InT, 1, 65536, EPipelineImageResample::Lanczos, Tall));
			TestEqual(TEXT("tall pixel count"), Tall.Pixels.Num(), 65536);
			TestEqual(TEXT("tall height"), Tall.Height, 65536);
		});

		It("ExtremeAspectRatiosAreExact", [this]()
		{
			// 1x16384 and 16384x1 (MaxSide) squeezed to a few pixels: first/last-row weight windows at the extremes. Compared to the
			// independent identity: a column of one colour keeps that colour, and the single-pixel axis must not be resampled.
			const FPipelineImage Col = MakeImage(1, FPipelineImage::MaxSide, [](int32, int32) { return FColor(60, 120, 180, 255); });
			FPipelineImage Out;
			TestTrue(TEXT("1x16384 -> 1x8 bicubic"), ResizeImage(Col, 1, 8, EPipelineImageResample::Bicubic, Out));
			TestEqual(TEXT("height"), Out.Height, 8);
			for (const FColor& C : Out.Pixels)
			{
				if (C != FColor(60, 120, 180, 255))
				{
					AddError(FString::Printf(TEXT("flat 1x16384 column changed to RGBA(%d,%d,%d,%d)"), C.R, C.G, C.B, C.A));
					break;
				}
			}
			const FPipelineImage Row = MakeImage(FPipelineImage::MaxSide, 1, [](int32, int32) { return FColor(60, 120, 180, 255); });
			TestTrue(TEXT("16384x1 -> 8x1 lanczos"), ResizeImage(Row, 8, 1, EPipelineImageResample::Lanczos, Out));
			TestEqual(TEXT("width"), Out.Width, 8);
			for (const FColor& C : Out.Pixels)
			{
				if (C != FColor(60, 120, 180, 255))
				{
					AddError(FString::Printf(TEXT("flat 16384x1 row changed to RGBA(%d,%d,%d,%d)"), C.R, C.G, C.B, C.A));
					break;
				}
			}
		});

		It("EightThreadsSharingOneSourceMatchTheSingleThreadedRun", [this]()
		{
			// Pure function, any thread: 8 threads read the SAME const source with different filters and sizes. Bug caught: static
			// scratch buffers or a cached coefficient table shared between calls.
			const TSharedRef<FPipelineImage> Shared = MakeShared<FPipelineImage>(MakeNoise(211, 157, 31));
			TArray<TSharedRef<FThreadJob>> Jobs;
			for (int32 i = 0; i < 8; ++i)
			{
				TSharedRef<FThreadJob> J = MakeShared<FThreadJob>();
				J->Input = *Shared;
				J->bResizeOnly = true;
				J->Filter = (EPipelineImageResample)(i % 6);
				J->OutW = 40 + 17 * i;
				J->OutH = 300 - 23 * i;
				Jobs.Add(J);
			}
			for (const TSharedRef<FThreadJob>& J : Jobs)
			{
				RunJobOnce(*J, false);
				if (!J->bSuccess)
				{
					AddError(TEXT("single-threaded reference ResizeImage failed"));
					return;
				}
			}
			const TSharedRef<FThreadSafeBool> Go = MakeShared<FThreadSafeBool>(false);
			TArray<TFuture<void>> Futures;
			for (const TSharedRef<FThreadJob>& J : Jobs)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [J, Go]()
				{
					while (!*Go) { FPlatformProcess::Sleep(0.f); }
					for (int32 Rep = 0; Rep < 6; ++Rep) { RunJobOnce(*J, true); }
					J->bDone = true;
				}));
			}
			*Go = true;
			for (int32 i = 0; i < Futures.Num(); ++i)
			{
				if (!Futures[i].WaitFor(FTimespan::FromSeconds(60)))
				{
					AddError(FString::Printf(TEXT("thread %d did not finish within 60 s"), i));
					return;
				}
				TestEqual(*FString::Printf(TEXT("thread %d: runs that differed from the single-threaded result"), i), Jobs[i]->Mismatches, 0);
			}
		});
	});

	// ------------------------------------------------------------------------------------------------------------------
	Describe("Preprocess", [this]()
	{
		for (const TCHAR* Name : ConfigNames)
		{
			const FString Config(Name);
			It(*FString::Printf(TEXT("Golden.%s"), Name), [this, Config]()
			{
				RunPreprocessConfig(*this, Config);
			});
		}

		It("WorkedExample640x480ConvNext", [this]()
		{
			// Brief's worked example: 640x480, shortest edge 256, crop 224 -> resized 341x256 ((int)(256*640/480) = 341), crop at
			// left (341-224)/2 = 58, top (256-224)/2 = 16. Geometry does not depend on the pixel values, so the image is synthetic.
			const FPipelineImage In = MakeNoise(640, 480, 640);
			TArray<float> Tensor;
			FPipelineImage Cropped;
			FPipelineImageGeometry Geo;
			const FPipelineResult R = PreprocessImage(In, CropSettings(), Tensor, &Cropped, &Geo);
			TestTrue(*FString::Printf(TEXT("success (%s)"), *R.Message), R.bSuccess);
			TestEqual(TEXT("SourceWidth"), Geo.SourceWidth, 640);
			TestEqual(TEXT("SourceHeight"), Geo.SourceHeight, 480);
			TestEqual(TEXT("ResizedWidth"), Geo.ResizedWidth, 341);
			TestEqual(TEXT("ResizedHeight"), Geo.ResizedHeight, 256);
			TestEqual(TEXT("CropLeft"), Geo.CropLeft, 58);
			TestEqual(TEXT("CropTop"), Geo.CropTop, 16);
			TestEqual(TEXT("CropWidth"), Geo.CropWidth, 224);
			TestEqual(TEXT("CropHeight"), Geo.CropHeight, 224);
			TestEqual(TEXT("cropped width"), Cropped.Width, 224);
			TestEqual(TEXT("cropped height"), Cropped.Height, 224);
			TestEqual(TEXT("tensor floats = 3*224*224"), Tensor.Num(), 3 * 224 * 224);
		});

		It("GeometryAndSelfConsistencyOnSyntheticSources", [this]()
		{
			// Beyond the goldens: odd sizes, rounding of the floor crop, non-square crops, portrait, extreme 7x1000.
			// For each case: geometry equals the C27 rules computed by hand (in the table); OutCropped equals the crop window of a
			// plain ResizeImage to the resized size (tests the crop offsets independently of the chain); the tensor equals the
			// C27 step-4 formula applied to OutCropped (tests plane order, mean/std per channel, row-major layout).
			using EM = EPipelineImageResize;
			using ER = EPipelineImageResample;
			const FGeometryCase Cases[] = {
				{ TEXT("landscape 300x200 shortest 256"), 300, 200, EM::ShortestEdgeCenterCrop, 224, 224, 256, 0, 0, ER::Bicubic, 384, 256, 80, 16 },
				{ TEXT("portrait 200x300 shortest 256"), 200, 300, EM::ShortestEdgeCenterCrop, 224, 224, 256, 0, 0, ER::Bicubic, 256, 384, 16, 80 },
				{ TEXT("301x200: 385.28 truncates, (385-224)/2 floors to 80"), 301, 200, EM::ShortestEdgeCenterCrop, 224, 224, 256, 0, 0, ER::Lanczos, 385, 256, 80, 16 },
				{ TEXT("255x256 shortest 224: 224.878 truncates to 224"), 255, 256, EM::ShortestEdgeCenterCrop, 224, 224, 224, 0, 0, ER::Bilinear, 224, 224, 0, 0 },
				{ TEXT("square 100x100 shortest 224"), 100, 100, EM::ShortestEdgeCenterCrop, 224, 224, 224, 0, 0, ER::Hamming, 224, 224, 0, 0 },
				{ TEXT("37x53 shortest 100 crop 64x80"), 37, 53, EM::ShortestEdgeCenterCrop, 64, 80, 100, 0, 0, ER::Box, 100, 143, 18, 31 },
				{ TEXT("stretch-crop 40x30 -> 260x240 crop 200x224"), 40, 30, EM::StretchCenterCrop, 200, 224, 0, 260, 240, ER::Bilinear, 260, 240, 30, 8 },
				{ TEXT("stretch 37x53 -> 20x30"), 37, 53, EM::Stretch, 20, 30, 0, 0, 0, ER::Bicubic, 20, 30, 0, 0 },
				{ TEXT("stretch 5x5 -> 7x3 nearest"), 5, 5, EM::Stretch, 7, 3, 0, 0, 0, ER::Nearest, 7, 3, 0, 0 },
				{ TEXT("tall 7x1000 shortest 224 (224x32000)"), 7, 1000, EM::ShortestEdgeCenterCrop, 224, 224, 224, 0, 0, ER::Bicubic, 224, 32000, 0, 15888 },
			};
			int32 Seed = 100;
			for (const FGeometryCase& C : Cases)
			{
				const FString Tag = FString::Printf(TEXT("[%s]"), C.Name);
				FPipelinePreprocessSettings S = CropSettings();
				S.ImageWidth = C.ImgW;
				S.ImageHeight = C.ImgH;
				S.ResizeMode = C.Mode;
				S.ResizeShortestEdge = C.Shortest;
				S.ResizeWidth = C.RW;
				S.ResizeHeight = C.RH;
				S.Resample = C.Filter;
				S.ImageMean = FVector(0.3, 0.5, 0.7); // distinct per channel: a swapped channel cannot cancel out
				S.ImageStd = FVector(0.2, 0.4, 0.8);
				const FPipelineImage In = MakeNoise(C.SrcW, C.SrcH, ++Seed);
				TArray<float> Tensor;
				FPipelineImage Cropped;
				FPipelineImageGeometry Geo;
				const FPipelineResult R = PreprocessImage(In, S, Tensor, &Cropped, &Geo);
				if (!R.bSuccess)
				{
					AddError(FString::Printf(TEXT("%s PreprocessImage failed: error %d '%s'"), *Tag, (int32)R.Error, *R.Message));
					continue;
				}
				const int32 Got[10] = { Geo.SourceWidth, Geo.SourceHeight, Geo.ResizedWidth, Geo.ResizedHeight, Geo.CropLeft, Geo.CropTop, Geo.CropWidth, Geo.CropHeight, Cropped.Width, Cropped.Height };
				const int32 Want[10] = { C.SrcW, C.SrcH, C.ExpResW, C.ExpResH, C.ExpLeft, C.ExpTop, C.ImgW, C.ImgH, C.ImgW, C.ImgH };
				if (FMemory::Memcmp(Got, Want, sizeof(Got)) != 0)
				{
					AddError(FString::Printf(TEXT("%s geometry [src %dx%d, resized %dx%d, crop (%d,%d) %dx%d, cropped %dx%d], expected [src %dx%d, resized %dx%d, crop (%d,%d) %dx%d, cropped %dx%d]"), *Tag,
						Got[0], Got[1], Got[2], Got[3], Got[4], Got[5], Got[6], Got[7], Got[8], Got[9], Want[0], Want[1], Want[2], Want[3], Want[4], Want[5], Want[6], Want[7], Want[8], Want[9]));
					continue;
				}
				// Crop window of a plain resize.
				FPipelineImage Resized;
				if (!ResizeImage(In, C.ExpResW, C.ExpResH, C.Filter, Resized))
				{
					AddError(Tag + TEXT(" reference ResizeImage failed"));
					continue;
				}
				FPipelineImage Window = MakeImage(C.ImgW, C.ImgH, [&](int32 x, int32 y) { return Resized.Pixels[(y + C.ExpTop) * Resized.Width + (x + C.ExpLeft)]; });
				FString Msg;
				int32 MaxDiff = 0;
				if (!CompareImages(Window, Cropped, Msg, MaxDiff))
				{
					AddError(FString::Printf(TEXT("%s OutCropped is not the crop window of ResizeImage: %s"), *Tag, *Msg));
					continue;
				}
				CheckTensorFromCropped(*this, Tag, Cropped, Tensor, S);
			}
		});

		It("TensorFormulaPlaneOrderMeanStdRescale", [this]()
		{
			// Same-size Stretch is a copy, so the expected tensor follows from the C27 step-4 formula alone, written out per variant.
			// Bug caught: BGR instead of RGB (FColor is stored B,G,R), HWC instead of CHW, mean/std applied to the wrong channel,
			// rescale applied after normalisation, alpha leaking into a plane.
			const FPipelineImage In = MakeImage(3, 2, [](int32 x, int32 y) { return FColor((uint8)(10 + 40 * x + y), (uint8)(250 - 30 * x - 7 * y), (uint8)(5 + 90 * y + 3 * x), (uint8)(17 * (x + y))); });
			struct FVariant { const TCHAR* Name; float Rescale; FVector Mean; FVector Std; };
			const FVariant Variants[] = {
				{ TEXT("imagenet 1/255"), (float)(1.0 / 255.0), FVector(0.485, 0.456, 0.406), FVector(0.229, 0.224, 0.225) },
				{ TEXT("rescale 1, mean 0, std 1 (raw values)"), 1.f, FVector(0, 0, 0), FVector(1, 1, 1) },
				{ TEXT("rescale 0.5, skewed mean/std"), 0.5f, FVector(10.0, -3.0, 0.25), FVector(2.0, 0.5, 8.0) },
			};
			for (const FVariant& V : Variants)
			{
				FPipelinePreprocessSettings S;
				S.ImageWidth = 3;
				S.ImageHeight = 2;
				S.ResizeMode = EPipelineImageResize::Stretch;
				S.Resample = EPipelineImageResample::Bicubic;
				S.RescaleFactor = V.Rescale;
				S.ImageMean = V.Mean;
				S.ImageStd = V.Std;
				TArray<float> Tensor;
				const FPipelineResult R = PreprocessImage(In, S, Tensor);
				const FString Tag = FString::Printf(TEXT("[%s]"), V.Name);
				if (!R.bSuccess)
				{
					AddError(FString::Printf(TEXT("%s failed: %s"), *Tag, *R.Message));
					continue;
				}
				CheckTensorFromCropped(*this, Tag, In, Tensor, S);
			}
			// One value spelled out: R plane, pixel (x=1,y=0) = 50 -> (50/255 - .485)/.229.
			FPipelinePreprocessSettings S0 = CropSettings();
			S0.ImageWidth = 3;
			S0.ImageHeight = 2;
			S0.ResizeMode = EPipelineImageResize::Stretch;
			TArray<float> T0;
			if (PreprocessImage(In, S0, T0).bSuccess && T0.Num() == 18)
			{
				const double Expected = (50.0 / 255.0 - 0.485) / 0.229;
				TestTrue(*FString::Printf(TEXT("R plane (x=1,y=0): got %.7f, expected %.7f"), T0[1], Expected), FMath::Abs((double)T0[1] - Expected) <= 1e-5);
				const double ExpectedB = ((5.0 + 90.0 * 1 + 3.0 * 2) / 255.0 - 0.406) / 0.225; // B plane, pixel (x=2,y=1)
				TestTrue(*FString::Printf(TEXT("B plane (x=2,y=1): got %.7f, expected %.7f"), T0[2 * 6 + 1 * 3 + 2], ExpectedB), FMath::Abs((double)T0[2 * 6 + 1 * 3 + 2] - ExpectedB) <= 1e-5);
			}
			else
			{
				AddError(TEXT("spelled-out check: PreprocessImage failed or wrong tensor size"));
			}
		});

		It("StretchToTheSameSizeIsACopy", [this]()
		{
			// Bug caught: a resize at the same size that filters; the cropped output must be the input bytes (alpha 255 input).
			const FPipelineImage In = MakeNoise(48, 32, 5);
			for (int32 f = 0; f < 6; ++f)
			{
				FPipelinePreprocessSettings S = CropSettings();
				S.ImageWidth = 48;
				S.ImageHeight = 32;
				S.ResizeMode = EPipelineImageResize::Stretch;
				S.Resample = (EPipelineImageResample)f;
				TArray<float> Tensor;
				FPipelineImage Cropped;
				FPipelineImageGeometry Geo;
				const FPipelineResult R = PreprocessImage(In, S, Tensor, &Cropped, &Geo);
				TestTrue(*FString::Printf(TEXT("%s success"), FilterNames[f]), R.bSuccess);
				FString Msg;
				int32 MaxDiff = 0;
				if (!CompareImages(In, Cropped, Msg, MaxDiff))
				{
					AddError(FString::Printf(TEXT("%s: Stretch to the same size is not a copy: %s"), FilterNames[f], *Msg));
				}
				TestEqual(TEXT("ResizedWidth"), Geo.ResizedWidth, 48);
				TestEqual(TEXT("ResizedHeight"), Geo.ResizedHeight, 32);
				TestEqual(TEXT("CropLeft (Stretch has no crop)"), Geo.CropLeft, 0);
				TestEqual(TEXT("CropTop (Stretch has no crop)"), Geo.CropTop, 0);
				TestEqual(TEXT("CropWidth"), Geo.CropWidth, 48);
				TestEqual(TEXT("CropHeight"), Geo.CropHeight, 32);
			}
		});

		It("OptionalOutputsMayBeNull", [this]()
		{
			// Bug caught: unconditional dereference of OutCropped / OutGeometry.
			const FPipelineImage In = MakeNoise(64, 48, 8);
			FPipelinePreprocessSettings S = CropSettings();
			S.ImageWidth = 16;
			S.ImageHeight = 16;
			S.ResizeShortestEdge = 20;
			TArray<float> A, B;
			FPipelineImage Cropped;
			FPipelineImageGeometry Geo;
			TestTrue(TEXT("no optional outputs"), PreprocessImage(In, S, A).bSuccess);
			TestTrue(TEXT("both optional outputs"), PreprocessImage(In, S, B, &Cropped, &Geo).bSuccess);
			TestTrue(TEXT("same tensor with and without the optional outputs"), A.Num() == 3 * 16 * 16 && A.Num() == B.Num() && FMemory::Memcmp(A.GetData(), B.GetData(), A.Num() * sizeof(float)) == 0);
		});

		It("AlphaIsIgnored", [this]()
		{
			// Same RGB, different A -> identical tensor and cropped RGB. Bug caught: alpha used as a 4th channel or premultiplied.
			const FPipelineImage Opaque = MakeNoise(80, 60, 21, 255);
			FPipelineImage Varied = Opaque;
			FRandomStream Rng(4);
			for (FColor& C : Varied.Pixels)
			{
				C.A = (uint8)Rng.RandRange(0, 255);
			}
			FPipelineImage Zero = Opaque;
			for (FColor& C : Zero.Pixels)
			{
				C.A = 0;
			}
			FPipelinePreprocessSettings S = CropSettings();
			S.ImageWidth = 32;
			S.ImageHeight = 32;
			S.ResizeShortestEdge = 40;
			TArray<float> T0, T1, T2;
			FPipelineImage C0, C1, C2;
			TestTrue(TEXT("opaque ok"), PreprocessImage(Opaque, S, T0, &C0).bSuccess);
			TestTrue(TEXT("varied alpha ok"), PreprocessImage(Varied, S, T1, &C1).bSuccess);
			TestTrue(TEXT("zero alpha ok"), PreprocessImage(Zero, S, T2, &C2).bSuccess);
			const bool bSame1 = T0.Num() == T1.Num() && T0.Num() > 0 && FMemory::Memcmp(T0.GetData(), T1.GetData(), T0.Num() * sizeof(float)) == 0;
			const bool bSame2 = T0.Num() == T2.Num() && T0.Num() > 0 && FMemory::Memcmp(T0.GetData(), T2.GetData(), T0.Num() * sizeof(float)) == 0;
			TestTrue(TEXT("random alpha gives the identical tensor"), bSame1);
			TestTrue(TEXT("alpha 0 gives the identical tensor"), bSame2);
			TestTrue(TEXT("cropped RGB equal (random alpha)"), SameRgb(C0, C1));
			TestTrue(TEXT("cropped RGB equal (alpha 0)"), SameRgb(C0, C2));
		});

		It("InvalidImagesReturnInvalidArgument", [this]()
		{
			// 0x5, pixel count mismatch, 16385 wide, plus a few relatives; valid settings so only the image can be at fault.
			struct FNamedImage { FString Key; FPipelineImage Value; };
			TArray<FNamedImage> Bad;
			{ FPipelineImage I; I.Width = 0; I.Height = 5; Bad.Add({ TEXT("0x5"), I }); }
			{ FPipelineImage I; Bad.Add({ TEXT("default image"), I }); }
			{ FPipelineImage I = MakeNoise(8, 8, 1); I.Pixels.Pop(); Bad.Add({ TEXT("63 pixels for 8x8"), I }); }
			{ FPipelineImage I = MakeNoise(8, 8, 1); I.Width = 9; Bad.Add({ TEXT("Width changed after filling"), I }); }
			Bad.Add({ TEXT("16385x1"), MakeNoise(FPipelineImage::MaxSide + 1, 1, 2) });
			Bad.Add({ TEXT("1x16385"), MakeNoise(1, FPipelineImage::MaxSide + 1, 3) });
			{ FPipelineImage I = MakeNoise(8, 8, 1); I.Height = -8; Bad.Add({ TEXT("negative height"), I }); }
			FPipelinePreprocessSettings S = CropSettings();
			S.ImageWidth = 4;
			S.ImageHeight = 4;
			S.ResizeShortestEdge = 8;
			for (const FNamedImage& B : Bad)
			{
				TArray<float> Tensor;
				const FPipelineResult R = PreprocessImage(B.Value, S, Tensor);
				TestFalse(*FString::Printf(TEXT("%s: bSuccess"), *B.Key), R.bSuccess);
				TestEqual(*FString::Printf(TEXT("%s: error"), *B.Key), (int32)R.Error, (int32)EPipelineError::InvalidArgument);
				TestFalse(*FString::Printf(TEXT("%s: message is not empty"), *B.Key), R.Message.IsEmpty());
			}
		});

		It("UnusableSettingsReturnPreprocessFailed", [this]()
		{
			// ValidateSettings image rules and the resize limits. Never a crash, always PreprocessFailed. When the offending number is
			// distinctive the message must contain it (C27: "naming the value").
			const FPipelineImage In = MakeNoise(64, 48, 6);
			const float NaN = std::numeric_limits<float>::quiet_NaN();
			const float Inf = std::numeric_limits<float>::infinity();
			struct FBadSettings { FString Name; FPipelinePreprocessSettings S; FString MustContain; };
			TArray<FBadSettings> Bad;
			auto Add = [&Bad](const TCHAR* Name, TFunctionRef<void(FPipelinePreprocessSettings&)> Mutate, const TCHAR* MustContain = TEXT(""))
			{
				FPipelinePreprocessSettings S = CropSettings();
				S.ImageWidth = 16;
				S.ImageHeight = 16;
				S.ResizeShortestEdge = 20;
				Mutate(S);
				Bad.Add({ Name, S, MustContain });
			};
			Add(TEXT("ImageWidth 0"), [](FPipelinePreprocessSettings& S) { S.ImageWidth = 0; });
			Add(TEXT("ImageHeight 0"), [](FPipelinePreprocessSettings& S) { S.ImageHeight = 0; });
			Add(TEXT("ImageWidth -5"), [](FPipelinePreprocessSettings& S) { S.ImageWidth = -5; }, TEXT("-5"));
			Add(TEXT("ImageWidth 9000"), [](FPipelinePreprocessSettings& S) { S.ImageWidth = 9000; S.ResizeMode = EPipelineImageResize::Stretch; }, TEXT("9000"));
			Add(TEXT("ImageHeight 8193"), [](FPipelinePreprocessSettings& S) { S.ImageHeight = 8193; S.ResizeMode = EPipelineImageResize::Stretch; }, TEXT("8193"));
			Add(TEXT("shortest edge 15 < crop 16"), [](FPipelinePreprocessSettings& S) { S.ResizeShortestEdge = 15; }, TEXT("15"));
			Add(TEXT("shortest edge below the larger crop side"), [](FPipelinePreprocessSettings& S) { S.ImageWidth = 16; S.ImageHeight = 32; S.ResizeShortestEdge = 31; }, TEXT("31"));
			Add(TEXT("shortest edge 0"), [](FPipelinePreprocessSettings& S) { S.ResizeShortestEdge = 0; });
			Add(TEXT("stretch-crop width 15 < crop 16"), [](FPipelinePreprocessSettings& S) { S.ResizeMode = EPipelineImageResize::StretchCenterCrop; S.ResizeWidth = 15; S.ResizeHeight = 40; }, TEXT("15"));
			Add(TEXT("stretch-crop height 14 < crop 16"), [](FPipelinePreprocessSettings& S) { S.ResizeMode = EPipelineImageResize::StretchCenterCrop; S.ResizeWidth = 40; S.ResizeHeight = 14; }, TEXT("14"));
			Add(TEXT("stretch-crop width 8193"), [](FPipelinePreprocessSettings& S) { S.ResizeMode = EPipelineImageResize::StretchCenterCrop; S.ResizeWidth = 8193; S.ResizeHeight = 40; }, TEXT("8193"));
			Add(TEXT("stretch-crop height 8193"), [](FPipelinePreprocessSettings& S) { S.ResizeMode = EPipelineImageResize::StretchCenterCrop; S.ResizeWidth = 40; S.ResizeHeight = 8193; }, TEXT("8193"));
			Add(TEXT("stretch-crop sizes 0"), [](FPipelinePreprocessSettings& S) { S.ResizeMode = EPipelineImageResize::StretchCenterCrop; S.ResizeWidth = 0; S.ResizeHeight = 0; });
			Add(TEXT("RescaleFactor 0"), [](FPipelinePreprocessSettings& S) { S.RescaleFactor = 0.f; });
			Add(TEXT("RescaleFactor -1"), [](FPipelinePreprocessSettings& S) { S.RescaleFactor = -1.f; });
			Add(TEXT("RescaleFactor NaN"), [NaN](FPipelinePreprocessSettings& S) { S.RescaleFactor = NaN; });
			Add(TEXT("RescaleFactor +inf"), [Inf](FPipelinePreprocessSettings& S) { S.RescaleFactor = Inf; });
			Add(TEXT("ImageStd.R 0"), [](FPipelinePreprocessSettings& S) { S.ImageStd.X = 0.0; });
			Add(TEXT("ImageStd.G negative"), [](FPipelinePreprocessSettings& S) { S.ImageStd.Y = -0.5; });
			Add(TEXT("ImageStd.B NaN"), [NaN](FPipelinePreprocessSettings& S) { S.ImageStd.Z = NaN; });
			Add(TEXT("ImageStd.B inf"), [Inf](FPipelinePreprocessSettings& S) { S.ImageStd.Z = Inf; });
			Add(TEXT("ImageMean.R NaN"), [NaN](FPipelinePreprocessSettings& S) { S.ImageMean.X = NaN; });
			Add(TEXT("ImageMean.B inf"), [Inf](FPipelinePreprocessSettings& S) { S.ImageMean.Z = Inf; });
			Add(TEXT("Resample 99"), [](FPipelinePreprocessSettings& S) { S.Resample = (EPipelineImageResample)99; }, TEXT("99"));
			Add(TEXT("ResizeMode 9"), [](FPipelinePreprocessSettings& S) { S.ResizeMode = (EPipelineImageResize)9; });
			for (const FBadSettings& B : Bad)
			{
				TArray<float> Tensor;
				const FPipelineResult R = PreprocessImage(In, B.S, Tensor);
				TestFalse(*FString::Printf(TEXT("%s: bSuccess"), *B.Name), R.bSuccess);
				TestEqual(*FString::Printf(TEXT("%s: error"), *B.Name), (int32)R.Error, (int32)EPipelineError::PreprocessFailed);
				TestFalse(*FString::Printf(TEXT("%s: message is not empty"), *B.Name), R.Message.IsEmpty());
				if (!B.MustContain.IsEmpty())
				{
					TestTrue(*FString::Printf(TEXT("%s: message names the value '%s' (was '%s')"), *B.Name, *B.MustContain, *R.Message), R.Message.Contains(B.MustContain));
				}
			}
		});

		It("ExtremeAspectRatioFailsCleanlyAndQuickly", [this]()
		{
			// 1x16384 with shortest edge 256 -> 256 x 4,194,304 in ShortestEdgeCenterCrop: over the 65536 side limit -> PreprocessFailed
			// before any allocation sized by it (a 4M x 256 FColor buffer would be 4 GB). Both orientations.
			FPipelinePreprocessSettings S = CropSettings();
			const FPipelineImage Tall = MakeImage(1, FPipelineImage::MaxSide, [](int32, int32 y) { return FColor((uint8)y, (uint8)(y >> 3), 7, 255); });
			const FPipelineImage Wide = MakeImage(FPipelineImage::MaxSide, 1, [](int32 x, int32) { return FColor((uint8)x, (uint8)(x >> 3), 7, 255); });
			const FPipelineImage* Images[2] = { &Tall, &Wide };
			const TCHAR* Names[2] = { TEXT("1x16384"), TEXT("16384x1") };
			for (int32 i = 0; i < 2; ++i)
			{
				TArray<float> Tensor;
				FPipelineImage Cropped;
				FPipelineImageGeometry Geo;
				const double T0 = FPlatformTime::Seconds();
				const FPipelineResult R = PreprocessImage(*Images[i], S, Tensor, &Cropped, &Geo);
				const double Secs = FPlatformTime::Seconds() - T0;
				TestFalse(*FString::Printf(TEXT("%s: bSuccess"), Names[i]), R.bSuccess);
				TestEqual(*FString::Printf(TEXT("%s: error"), Names[i]), (int32)R.Error, (int32)EPipelineError::PreprocessFailed);
				TestFalse(*FString::Printf(TEXT("%s: message names the size"), Names[i]), R.Message.IsEmpty());
				TestTrue(*FString::Printf(TEXT("%s: fails quickly (%.3f s)"), Names[i], Secs), Secs < 2.0);
				TestTrue(*FString::Printf(TEXT("%s: no tensor produced"), Names[i]), Tensor.Num() == 0);
			}
			// Just inside the limit: 1x100 with shortest edge 256 -> 256 x 25600, which must still work: 256 x 25600, crop the middle.
			const FPipelineImage Slim = MakeImage(1, 100, [](int32, int32 y) { return FColor((uint8)(y * 2), (uint8)(255 - y), 40, 255); });
			TArray<float> Tensor;
			FPipelineImageGeometry Geo;
			const FPipelineResult R = PreprocessImage(Slim, S, Tensor, nullptr, &Geo);
			TestTrue(*FString::Printf(TEXT("1x100 shortest 256 still works (%s)"), *R.Message), R.bSuccess);
			TestEqual(TEXT("1x100: resized width"), Geo.ResizedWidth, 256);
			TestEqual(TEXT("1x100: resized height ((int)(256*100/1))"), Geo.ResizedHeight, 25600);
			TestEqual(TEXT("1x100: crop left"), Geo.CropLeft, 16);
			TestEqual(TEXT("1x100: crop top"), Geo.CropTop, 12688);
			TestEqual(TEXT("1x100: tensor size"), Tensor.Num(), 3 * 224 * 224);
		});

		It("EightThreadsPreprocessingMatchTheSingleThreadedRun", [this]()
		{
			// 8 threads, 8 different images, filters and modes at once. Bug caught: static scratch/coefficient caches, shared output
			// buffers, thread-unsafe lazy tables. Every repeat must be bit-identical to the single-threaded reference.
			TArray<TSharedRef<FThreadJob>> Jobs;
			const int32 Sizes[8][2] = { { 333, 222 }, { 222, 333 }, { 400, 300 }, { 129, 257 }, { 640, 480 }, { 61, 700 }, { 301, 200 }, { 256, 256 } };
			for (int32 i = 0; i < 8; ++i)
			{
				TSharedRef<FThreadJob> J = MakeShared<FThreadJob>();
				J->Input = MakeNoise(Sizes[i][0], Sizes[i][1], 500 + i);
				J->Settings = CropSettings();
				J->Settings.ImageWidth = 64 + 8 * i;
				J->Settings.ImageHeight = 64;
				J->Settings.Resample = (EPipelineImageResample)(i % 6);
				if (i % 3 == 0)
				{
					J->Settings.ResizeMode = EPipelineImageResize::Stretch;
				}
				else if (i % 3 == 1)
				{
					J->Settings.ResizeMode = EPipelineImageResize::StretchCenterCrop;
					J->Settings.ResizeWidth = J->Settings.ImageWidth + 20;
					J->Settings.ResizeHeight = 96;
				}
				else
				{
					J->Settings.ResizeShortestEdge = 100 + i;
				}
				Jobs.Add(J);
			}
			for (const TSharedRef<FThreadJob>& J : Jobs)
			{
				RunJobOnce(*J, false);
				if (!J->bSuccess)
				{
					AddError(TEXT("single-threaded reference PreprocessImage failed"));
					return;
				}
			}
			const TSharedRef<FThreadSafeBool> Go = MakeShared<FThreadSafeBool>(false);
			TArray<TFuture<void>> Futures;
			for (const TSharedRef<FThreadJob>& J : Jobs)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [J, Go]()
				{
					while (!*Go) { FPlatformProcess::Sleep(0.f); }
					for (int32 Rep = 0; Rep < 5; ++Rep) { RunJobOnce(*J, true); }
					J->bDone = true;
				}));
			}
			*Go = true;
			for (int32 i = 0; i < Futures.Num(); ++i)
			{
				if (!Futures[i].WaitFor(FTimespan::FromSeconds(120)))
				{
					AddError(FString::Printf(TEXT("thread %d did not finish within 120 s"), i));
					return;
				}
				TestEqual(*FString::Printf(TEXT("thread %d: runs that differed from the single-threaded result"), i), Jobs[i]->Mismatches, 0);
			}
		});

		It("PerformanceReport1080pToCrop224", [this]()
		{
			// Not a gate (brief): 1920x1080 -> shortest edge 256 bicubic + crop 224, one core. Reported for the lead.
			const FPipelineImage In = MakeNoise(1920, 1080, 1080);
			const FPipelinePreprocessSettings S = CropSettings();
			TArray<float> Tensor;
			PreprocessImage(In, S, Tensor); // warm up
			const int32 Runs = 5;
			const double T0 = FPlatformTime::Seconds();
			bool bOk = true;
			for (int32 i = 0; i < Runs; ++i)
			{
				bOk &= PreprocessImage(In, S, Tensor).bSuccess;
			}
			const double Ms = (FPlatformTime::Seconds() - T0) * 1000.0 / Runs;
			TestTrue(TEXT("1080p preprocess succeeds"), bOk);
			AddInfo(FString::Printf(TEXT("1920x1080 -> 256 bicubic + crop 224: %.2f ms per call (target <= 40 ms, not a gate)"), Ms));
		});
	});

	// ------------------------------------------------------------------------------------------------------------------
	Describe("RHI", [this]()
	{
		It("TestHelpersTextureAndRenderTarget", [this]()
		{
			// Checks the C33 texture helpers themselves (later WPs rely on them): nullptr without an RHI, a texture/target of the
			// image's size with one. Skipped under -nullrhi unless MODELPIPELINES_REQUIRE_RHI=1 (then an error).
			const FPipelineImage In = MakeNoise(8, 6, 3);
			if (!HasRhi())
			{
				if (IsRhiRequired())
				{
					AddError(TEXT("MODELPIPELINES_REQUIRE_RHI=1 but no RHI is available (run without -nullrhi)."));
				}
				else
				{
					AddInfo(TEXT("RHI not available: skipped (the helpers must return nullptr here)"));
				}
				TestTrue(TEXT("MakeTexture returns nullptr without an RHI"), MakeTexture(In, PF_B8G8R8A8) == nullptr);
				TestTrue(TEXT("MakeRenderTarget returns nullptr without an RHI"), MakeRenderTarget(In, RTF_RGBA8) == nullptr);
				return;
			}
			const EPixelFormat Formats[3] = { PF_B8G8R8A8, PF_R8G8B8A8, PF_FloatRGBA };
			for (const EPixelFormat F : Formats)
			{
				TStrongObjectPtr<UTexture2D> Tex(MakeTexture(In, F));
				TestTrue(*FString::Printf(TEXT("MakeTexture format %d is not null"), (int32)F), Tex.IsValid());
				if (Tex.IsValid())
				{
					TestEqual(TEXT("texture width"), Tex->GetSizeX(), 8);
					TestEqual(TEXT("texture height"), Tex->GetSizeY(), 6);
				}
			}
			TStrongObjectPtr<UTextureRenderTarget2D> RT(MakeRenderTarget(In, RTF_RGBA8));
			TestTrue(TEXT("MakeRenderTarget RGBA8 is not null"), RT.IsValid());
			if (RT.IsValid())
			{
				TestEqual(TEXT("render target width"), (int32)RT->SizeX, 8);
				TestEqual(TEXT("render target height"), (int32)RT->SizeY, 6);
			}
			FPipelineImage Bad;
			TestTrue(TEXT("MakeTexture of an invalid image is nullptr"), MakeTexture(Bad, PF_B8G8R8A8) == nullptr);
			TestTrue(TEXT("MakeRenderTarget of an invalid image is nullptr"), MakeRenderTarget(Bad, RTF_RGBA8) == nullptr);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
