// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "PipelineVisionTestData.h"

#include "Dom/JsonValue.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/PlatformMisc.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Math/Float16.h"
#include "Misc/App.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "NNEModelData.h"
#include "RHI.h"
#include "RHICommandList.h"
#include "RenderingThread.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "TextureResource.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"

namespace UE::ModelPipelines::VisionTests
{
	namespace
	{
		const TCHAR* const FetchHint = TEXT("Generate the vision goldens and models with: \"D:/Micromamba/micromambaenv/envs/pipelines/python.exe\" ModelPipelines-UE/Tools/make_vision_goldens.py (after Tools/fetch_reference_models.py and Tools/make_import_test_data.py).");

		/** Bytes per pixel and packed bytes for the formats the texture helpers support. */
		bool PackPixels(const FPipelineImage& Image, EPixelFormat Format, TArray<uint8>& Out, uint32& OutBytesPerPixel)
		{
			if (!Image.IsValid())
			{
				return false;
			}
			const int64 Count = (int64)Image.Width * Image.Height;
			auto PutHalf = [&Out](float V) { const FFloat16 H(V); Out.Append(reinterpret_cast<const uint8*>(&H), sizeof(FFloat16)); };
			auto PutFloat = [&Out](float V) { Out.Append(reinterpret_cast<const uint8*>(&V), sizeof(float)); };
			switch (Format)
			{
			case PF_B8G8R8A8: OutBytesPerPixel = 4; break;
			case PF_R8G8B8A8: OutBytesPerPixel = 4; break;
			case PF_G8: OutBytesPerPixel = 1; break;
			case PF_R8G8: OutBytesPerPixel = 2; break;
			case PF_FloatRGBA: OutBytesPerPixel = 8; break;
			case PF_R16F: OutBytesPerPixel = 2; break;
			case PF_G16R16F: OutBytesPerPixel = 4; break;
			case PF_A32B32G32R32F: OutBytesPerPixel = 16; break;
			case PF_R32_FLOAT: OutBytesPerPixel = 4; break;
			case PF_G32R32F: OutBytesPerPixel = 8; break;
			default: return false;
			}
			Out.Reset();
			Out.Reserve((int32)(Count * OutBytesPerPixel));
			for (int64 i = 0; i < Count; ++i)
			{
				const FColor C = Image.Pixels[(int32)i];
				switch (Format)
				{
				case PF_B8G8R8A8: Out.Add(C.B); Out.Add(C.G); Out.Add(C.R); Out.Add(C.A); break;
				case PF_R8G8B8A8: Out.Add(C.R); Out.Add(C.G); Out.Add(C.B); Out.Add(C.A); break;
				case PF_G8: Out.Add(C.R); break;
				case PF_R8G8: Out.Add(C.R); Out.Add(C.G); break;
				case PF_FloatRGBA: PutHalf(C.R / 255.f); PutHalf(C.G / 255.f); PutHalf(C.B / 255.f); PutHalf(C.A / 255.f); break;
				case PF_R16F: PutHalf(C.R / 255.f); break;
				case PF_G16R16F: PutHalf(C.R / 255.f); PutHalf(C.G / 255.f); break;
				case PF_A32B32G32R32F: PutFloat(C.R / 255.f); PutFloat(C.G / 255.f); PutFloat(C.B / 255.f); PutFloat(C.A / 255.f); break;
				case PF_R32_FLOAT: PutFloat(C.R / 255.f); break;
				case PF_G32R32F: PutFloat(C.R / 255.f); PutFloat(C.G / 255.f); break;
				default: break;
				}
			}
			return true;
		}
	}

	FString GetDevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		if (!Env.IsEmpty())
		{
			return Env;
		}
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev"));
	}

	TSharedPtr<FJsonObject> LoadVisionGoldens(FString& OutError)
	{
		const FString Path = GetDevDir() / TEXT("goldens/vision/vision_goldens.json");
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			OutError = FString::Printf(TEXT("Missing vision goldens '%s'. %s"), *Path, FetchHint);
			return nullptr;
		}
		TSharedPtr<FJsonObject> Root;
		const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
		if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
		{
			OutError = FString::Printf(TEXT("Vision goldens '%s' are not valid JSON. %s"), *Path, FetchHint);
			return nullptr;
		}
		return Root;
	}

	bool LoadPng(const FString& AbsPath, FPipelineImage& Out, FString& OutError)
	{
		Out = FPipelineImage();
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *AbsPath) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("Missing PNG '%s'. %s"), *AbsPath, FetchHint);
			return false;
		}
		IImageWrapperModule& Module = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
		const TSharedPtr<IImageWrapper> Wrapper = Module.CreateImageWrapper(EImageFormat::PNG);
		if (!Wrapper.IsValid() || !Wrapper->SetCompressed(Bytes.GetData(), Bytes.Num()))
		{
			OutError = FString::Printf(TEXT("'%s' is not a decodable PNG."), *AbsPath);
			return false;
		}
		TArray64<uint8> Raw;
		if (!Wrapper->GetRaw(ERGBFormat::BGRA, 8, Raw))
		{
			OutError = FString::Printf(TEXT("PNG '%s' could not be decoded to 8-bit BGRA."), *AbsPath);
			return false;
		}
		const int32 W = Wrapper->GetWidth();
		const int32 H = Wrapper->GetHeight();
		if (W <= 0 || H <= 0 || Raw.Num() != (int64)W * H * 4)
		{
			OutError = FString::Printf(TEXT("PNG '%s' decoded to %lld bytes for %dx%d."), *AbsPath, (long long)Raw.Num(), W, H);
			return false;
		}
		Out.Width = W;
		Out.Height = H;
		Out.Pixels.SetNumUninitialized(W * H);
		FMemory::Memcpy(Out.Pixels.GetData(), Raw.GetData(), Raw.Num()); // FColor is B, G, R, A in memory
		return true;
	}

	bool LoadF32(const FString& AbsPath, TArray<float>& Out, FString& OutError)
	{
		Out.Reset();
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *AbsPath))
		{
			OutError = FString::Printf(TEXT("Missing tensor file '%s'. %s"), *AbsPath, FetchHint);
			return false;
		}
		if (Bytes.Num() % 4 != 0)
		{
			OutError = FString::Printf(TEXT("Tensor file '%s' has %d bytes, not a multiple of 4."), *AbsPath, Bytes.Num());
			return false;
		}
		Out.SetNumUninitialized(Bytes.Num() / 4);
		if (Bytes.Num() > 0)
		{
			FMemory::Memcpy(Out.GetData(), Bytes.GetData(), Bytes.Num()); // little endian on Win64
		}
		return true;
	}

	bool ParseResample(const FString& Name, EPipelineImageResample& Out)
	{
		static const TCHAR* const Names[] = { TEXT("Nearest"), TEXT("Lanczos"), TEXT("Bilinear"), TEXT("Bicubic"), TEXT("Box"), TEXT("Hamming") };
		for (int32 i = 0; i < 6; ++i)
		{
			if (Name == Names[i])
			{
				Out = (EPipelineImageResample)i; // PIL's values (C26)
				return true;
			}
		}
		return false;
	}

	bool ParseSettings(const TSharedPtr<FJsonObject>& S, FPipelinePreprocessSettings& Out, FString& OutError)
	{
		if (!S.IsValid())
		{
			OutError = TEXT("ParseSettings: null settings object.");
			return false;
		}
		Out.ImageWidth = (int32)S->GetNumberField(TEXT("image_width"));
		Out.ImageHeight = (int32)S->GetNumberField(TEXT("image_height"));
		const FString Mode = S->GetStringField(TEXT("resize_mode"));
		if (Mode == TEXT("Stretch")) { Out.ResizeMode = EPipelineImageResize::Stretch; }
		else if (Mode == TEXT("ShortestEdgeCenterCrop")) { Out.ResizeMode = EPipelineImageResize::ShortestEdgeCenterCrop; }
		else if (Mode == TEXT("StretchCenterCrop")) { Out.ResizeMode = EPipelineImageResize::StretchCenterCrop; }
		else
		{
			OutError = FString::Printf(TEXT("Unknown resize_mode '%s' in goldens."), *Mode);
			return false;
		}
		Out.ResizeShortestEdge = (int32)S->GetNumberField(TEXT("resize_shortest_edge"));
		Out.ResizeWidth = (int32)S->GetNumberField(TEXT("resize_width"));
		Out.ResizeHeight = (int32)S->GetNumberField(TEXT("resize_height"));
		if (!ParseResample(S->GetStringField(TEXT("resample")), Out.Resample))
		{
			OutError = FString::Printf(TEXT("Unknown resample '%s' in goldens."), *S->GetStringField(TEXT("resample")));
			return false;
		}
		Out.RescaleFactor = (float)S->GetNumberField(TEXT("rescale_factor"));
		const TArray<TSharedPtr<FJsonValue>>* Mean = nullptr;
		const TArray<TSharedPtr<FJsonValue>>* Std = nullptr;
		if (!S->TryGetArrayField(TEXT("image_mean"), Mean) || Mean->Num() != 3 || !S->TryGetArrayField(TEXT("image_std"), Std) || Std->Num() != 3)
		{
			OutError = TEXT("image_mean / image_std missing or not 3 numbers in goldens.");
			return false;
		}
		Out.ImageMean = FVector((*Mean)[0]->AsNumber(), (*Mean)[1]->AsNumber(), (*Mean)[2]->AsNumber());
		Out.ImageStd = FVector((*Std)[0]->AsNumber(), (*Std)[1]->AsNumber(), (*Std)[2]->AsNumber());
		return true;
	}

	UPipelineModelPackage* MakePackage(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Model, EPipelineDevice Device, FString& OutError)
	{
		if (!Root.IsValid() || !Model.IsValid())
		{
			OutError = TEXT("MakePackage: null goldens or model entry.");
			return nullptr;
		}
		const TSharedPtr<FJsonObject>* PackageBlock = nullptr;
		if (!Model->TryGetObjectField(TEXT("package"), PackageBlock))
		{
			OutError = TEXT("MakePackage: model entry has no \"package\" block.");
			return nullptr;
		}
		const FString Folder = Model->GetStringField(TEXT("folder"));
		const FString OnnxPath = Root->GetStringField(TEXT("models_root")) / Folder / TEXT("model.onnx");
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *OnnxPath) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("Missing model '%s'. %s"), *OnnxPath, FetchHint);
			return nullptr;
		}

		UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
		if (!LexTryParseString(Pkg->Task, *Model->GetStringField(TEXT("task"))))
		{
			OutError = FString::Printf(TEXT("Unknown task '%s' in goldens."), *Model->GetStringField(TEXT("task")));
			return nullptr;
		}
		const TSharedPtr<FJsonObject>& P = *PackageBlock;
		const TSharedPtr<FJsonObject>* Settings = nullptr;
		if (!P->TryGetObjectField(TEXT("settings"), Settings) || !ParseSettings(*Settings, Pkg->Preprocess, OutError))
		{
			if (OutError.IsEmpty())
			{
				OutError = TEXT("MakePackage: package has no \"settings\" block.");
			}
			return nullptr;
		}
		Pkg->Postprocess.ScoreActivation = P->GetStringField(TEXT("score_activation")).Equals(TEXT("Sigmoid"), ESearchCase::IgnoreCase)
			? EPipelineScoreActivation::Sigmoid : EPipelineScoreActivation::Softmax;
		double Nms = 0.7;
		if (P->TryGetNumberField(TEXT("nms_iou_threshold"), Nms))
		{
			Pkg->Postprocess.NmsIouThreshold = (float)Nms;
		}
		int32 MaxDet = 100;
		if (P->TryGetNumberField(TEXT("max_detections"), MaxDet))
		{
			Pkg->Postprocess.MaxDetections = MaxDet;
		}
		const TArray<TSharedPtr<FJsonValue>>* Labels = nullptr;
		if (P->TryGetArrayField(TEXT("labels"), Labels))
		{
			for (const TSharedPtr<FJsonValue>& L : *Labels)
			{
				Pkg->Labels.Add(L->AsString());
			}
		}

		UNNEModelData* Data = NewObject<UNNEModelData>(Pkg);
		Data->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
		Pkg->Model = Data;
		Pkg->License.SpdxId = TEXT("Apache-2.0");
		Pkg->License.SourceUrl = TEXT("https://example.invalid/test-model");
		Pkg->License.Attribution = TEXT("Test attribution");
		Pkg->PreferredDevice = Device;
		return Pkg;
	}

	bool IsGpuRequired()
	{
		return FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_GPU")) == TEXT("1");
	}

	bool IsRhiRequired()
	{
		return FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_RHI")) == TEXT("1");
	}

	bool HasRhi()
	{
		return FApp::CanEverRender() && GDynamicRHI != nullptr;
	}

	UTexture2D* MakeTexture(const FPipelineImage& Image, EPixelFormat Format)
	{
		if (!HasRhi())
		{
			return nullptr;
		}
		TArray<uint8> Bytes;
		uint32 Bpp = 0;
		if (Format != PF_B8G8R8A8 && Format != PF_R8G8B8A8 && Format != PF_FloatRGBA)
		{
			return nullptr;
		}
		if (!PackPixels(Image, Format, Bytes, Bpp))
		{
			return nullptr;
		}
		UTexture2D* Texture = UTexture2D::CreateTransient(Image.Width, Image.Height, Format, NAME_None, TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
		if (!Texture)
		{
			return nullptr;
		}
		Texture->NeverStream = true;
		Texture->UpdateResource();
		FlushRenderingCommands();
		return Texture;
	}

	UTextureRenderTarget2D* MakeRenderTarget(const FPipelineImage& Image, ETextureRenderTargetFormat Format)
	{
		if (!HasRhi() || !Image.IsValid() || Format == RTF_RGB10A2)
		{
			return nullptr;
		}
		const EPixelFormat PixelFormat = GetPixelFormatFromRenderTargetFormat(Format);
		TArray<uint8> Bytes;
		uint32 Bpp = 0;
		if (!PackPixels(Image, PixelFormat, Bytes, Bpp))
		{
			return nullptr;
		}
		UTextureRenderTarget2D* Target = NewObject<UTextureRenderTarget2D>(GetTransientPackage());
		Target->RenderTargetFormat = Format;
		Target->InitAutoFormat((uint32)Image.Width, (uint32)Image.Height);
		Target->UpdateResourceImmediate(true);
		FTextureRenderTargetResource* Resource = Target->GameThread_GetRenderTargetResource();
		if (!Resource)
		{
			return nullptr;
		}
		const uint32 Width = (uint32)Image.Width;
		const uint32 Height = (uint32)Image.Height;
		const uint32 Pitch = Width * Bpp;
		ENQUEUE_RENDER_COMMAND(VisionTestWriteRenderTarget)([Resource, Data = MoveTemp(Bytes), Width, Height, Pitch](FRHICommandListImmediate& RHICmdList)
		{
			FRHITexture* Texture = Resource->GetRenderTargetTexture();
			if (Texture)
			{
				const FUpdateTextureRegion2D Region(0, 0, 0, 0, Width, Height);
				RHICmdList.UpdateTexture2D(Texture, 0, Region, Pitch, Data.GetData());
			}
		});
		FlushRenderingCommands();
		return Target;
	}
}

#endif // WITH_DEV_AUTOMATION_TESTS
