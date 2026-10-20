// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "UObject/SoftObjectPtr.h"
#include "PipelineTypes.h"

class AActor;
class AModelPipelinesSampleBase;
class UPipelineBase;
class UPipelineModelPackage;
class UTexture2D;

namespace UE::ModelPipelines::SamplesTests
{
	/** env MODELPIPELINES_DEV_DIR, else <ProjectDir>/../.pipelines-dev (absolute). */
	FString GetDevDir();

	/** <dev>/goldens/text/text_goldens.json. nullptr + OutError (with the path and the generate command) on failure. */
	TSharedPtr<FJsonObject> LoadTextGoldens(FString& OutError);

	/** The entry of Root["models"] whose "slug" equals Slug, or nullptr. */
	TSharedPtr<FJsonObject> FindModel(const TSharedPtr<FJsonObject>& Root, const FString& Slug);

	/**
	 * Transient package from the model's "package" block (task, MaxTokens, bAddSpecialTokens, Pooling, normalize,
	 * ScoreActivation, Labels). nullptr + OutError (path and fetch command) on failure. The caller keeps it alive.
	 */
	UPipelineModelPackage* MakePackage(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Model, EPipelineDevice Device, FString& OutError);

	/** The first UPipelineBase held by one of the actor's object UPROPERTYs (the sample's pipeline), or nullptr. */
	UPipelineBase* FindSamplePipeline(const AActor* Actor);

	/**
	 * A game world with no viewport, no game mode and no ticking, plus the way to start play on one actor in it.
	 * Create() -> SpawnSample() -> ... -> Destroy() (idempotent).
	 */
	struct FSamplesTestWorld
	{
		UWorld* World = nullptr;

		/** CreateWorld(Game) with scenes/physics/audio/navigation off, then InitializeActorsForPlay. false + OutError on failure. */
		bool Create(FString& OutError);

		/**
		 * Deferred spawn of SampleClass, bShowOverlay = bShowOverlay, Package = Package (an empty soft pointer = "unset"),
		 * PreBeginPlay(Actor) (bind delegates here: BeginPlay may already finish the work), FinishSpawning, then BeginPlay
		 * (DispatchBeginPlay; the world itself has no game mode, so it never begins play on its own).
		 * nullptr + OutError on failure.
		 */
		AModelPipelinesSampleBase* SpawnSample(UClass* SampleClass, const TSoftObjectPtr<UPipelineModelPackage>& Package, bool bShowOverlay,
			TFunction<void(AModelPipelinesSampleBase*)> PreBeginPlay, FString& OutError);

		/** Destroys the world (and every actor in it). Safe to call twice. */
		void Destroy();
	};

	/** env MODELPIPELINES_REQUIRE_GPU == "1". */
	bool IsGpuRequired();

	// ------------------------------------------------------------------ vision helpers (WP5.5)

	/** env MODELPIPELINES_REQUIRE_RHI == "1". */
	bool IsRhiRequired();

	/** A real RHI exists (FApp::CanEverRender() and a dynamic RHI). false under -nullrhi. */
	bool HasRhi();

	/** <dev>/goldens/vision/vision_goldens.json. nullptr + OutError (with the path and the generate command) on failure. */
	TSharedPtr<FJsonObject> LoadVisionGoldens(FString& OutError);

	/** The entry of Root["reference_models"] whose "slug" equals Slug, or nullptr. */
	TSharedPtr<FJsonObject> FindVisionModel(const TSharedPtr<FJsonObject>& Root, const FString& Slug);

	/** The "cases" entry of a vision model whose "image" equals ImageId, or nullptr. */
	TSharedPtr<FJsonObject> FindVisionCase(const TSharedPtr<FJsonObject>& Model, const FString& ImageId);

	/**
	 * Transient package from a reference model's "package" block (task, settings, labels, score_activation, nms_iou_threshold,
	 * max_detections); Model = UNNEModelData over <models_root>/<folder>/model.onnx. The caller keeps it alive.
	 * nullptr + OutError (path and fetch command) on failure.
	 */
	UPipelineModelPackage* MakeVisionPackage(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Model, EPipelineDevice Device, FString& OutError);

	/** An 8-bit image, top row first. */
	struct FVisionTestImage
	{
		int32 Width = 0;
		int32 Height = 0;
		TArray<FColor> Pixels;
		bool IsValid() const { return Width > 0 && Height > 0 && Pixels.Num() == Width * Height; }
	};

	/** Decodes the PNG of the golden image entry ImageId (reference_images, then images). false + OutError (path and command) on failure. */
	bool LoadVisionImage(const TSharedPtr<FJsonObject>& Root, const FString& ImageId, FVisionTestImage& Out, FString& OutError);

	/** Transient B8G8R8A8 texture (NeverStream, no mips) with the image; nullptr when there is no RHI or the image is invalid. */
	UTexture2D* MakeVisionTexture(const FVisionTestImage& Image);

	/**
	 * FSamplesTestWorld whose world HAS a render scene (needed for scene captures; requires a real RHI, so only call it
	 * when HasRhi()). Everything else (SpawnSample, Destroy) is inherited.
	 */
	struct FSamplesSceneWorld : public FSamplesTestWorld
	{
		bool CreateWithScene(FString& OutError);
	};
}

#endif // WITH_DEV_AUTOMATION_TESTS
