// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"
#include "PipelineModelPackage.h"
#include "PipelineTypes.h"
#include "PipelineImage.h"
#include "PixelFormat.h"
#include "Engine/TextureRenderTarget2D.h"

class UPipelineModelPackage;
class UTexture2D;

namespace UE::ModelPipelines::VisionTests
{
	/** env MODELPIPELINES_DEV_DIR, else <ProjectDir>/../.pipelines-dev (absolute). */
	FString GetDevDir();

	/** <dev>/goldens/vision/vision_goldens.json. nullptr + OutError (path and generate command) on failure. */
	TSharedPtr<FJsonObject> LoadVisionGoldens(FString& OutError);

	/** 8-bit PNG (any colour type) -> FColor, top row first. false + OutError (with the path and the generate command) on failure. */
	bool LoadPng(const FString& AbsPath, FPipelineImage& Out, FString& OutError);

	/** Raw little-endian float32 file. false + OutError on failure (missing, size not a multiple of 4). */
	bool LoadF32(const FString& AbsPath, TArray<float>& Out, FString& OutError);

	/**
	 * Transient package from the model's "package" block (C30: task, settings, labels, score_activation, nms_iou_threshold,
	 * max_detections). Model = UNNEModelData("onnx") over <models_root>/<folder>/model.onnx. The caller keeps it alive
	 * (TStrongObjectPtr). nullptr + OutError (path and fetch command) on failure.
	 */
	UPipelineModelPackage* MakePackage(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Model, EPipelineDevice Device, FString& OutError);

	/** C30 "settings" block (snake_case names + resample enumerator name) -> settings struct. */
	bool ParseSettings(const TSharedPtr<FJsonObject>& SettingsJson, FPipelinePreprocessSettings& Out, FString& OutError);

	/** Enumerator name ("Nearest", "Lanczos", "Bilinear", "Bicubic", "Box", "Hamming") -> enum. */
	bool ParseResample(const FString& Name, EPipelineImageResample& Out);

	/** env MODELPIPELINES_REQUIRE_GPU == "1". */
	bool IsGpuRequired();

	/** env MODELPIPELINES_REQUIRE_RHI == "1". */
	bool IsRhiRequired();

	/** A real RHI exists (FApp::CanEverRender() and a dynamic RHI). false under -nullrhi. */
	bool HasRhi();

	/**
	 * Transient texture with the image's pixels (B8G8R8A8 / R8G8B8A8 / FloatRGBA with v/255 as half floats; A copied as is).
	 * NeverStream, UpdateResource called. nullptr when there is no RHI, the image is invalid or the format is unsupported.
	 */
	UTexture2D* MakeTexture(const FPipelineImage& Image, EPixelFormat Format);

	/**
	 * Render target of the image's size and Format with the pixels written by RHIUpdateTexture2D on the render thread, then
	 * FlushRenderingCommands. Fixed-point formats get the bytes as they are, float formats v/255. nullptr when there is no RHI,
	 * the image is invalid or the format is unsupported (R8 .. RGBA32f except RGB10A2). The caller keeps it alive.
	 */
	UTextureRenderTarget2D* MakeRenderTarget(const FPipelineImage& Image, ETextureRenderTargetFormat Format);
}

#endif // WITH_DEV_AUTOMATION_TESTS
