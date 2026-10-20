// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"
#include "PipelineTypes.h"

class UPipelineModelPackage;

namespace UE::ModelPipelines::TextTests
{
	/** env MODELPIPELINES_DEV_DIR, else <ProjectDir>/../.pipelines-dev (absolute). */
	FString GetDevDir();

	/** <dev>/goldens/text/text_goldens.json. nullptr + OutError (with the path and the generate command) on failure. */
	TSharedPtr<FJsonObject> LoadTextGoldens(FString& OutError);

	/** The entry of Root["models"] whose "slug" equals Slug, or nullptr. */
	TSharedPtr<FJsonObject> FindModel(const TSharedPtr<FJsonObject>& Root, const FString& Slug);

	/**
	 * Transient package from the model's "package" block (task, MaxTokens, bAddSpecialTokens, Pooling, normalize,
	 * ScoreActivation, Labels). Model = UNNEModelData("onnx") over <models_root>/<folder>/model.onnx, TokenizerJson =
	 * <models_root>/<folder>/tokenizer.json, License = test values, PreferredDevice = Device.
	 * nullptr + OutError (path and fetch command) on failure. The caller keeps it alive (TStrongObjectPtr).
	 */
	UPipelineModelPackage* MakePackage(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Model, EPipelineDevice Device, FString& OutError);

	/** env MODELPIPELINES_REQUIRE_GPU == "1". */
	bool IsGpuRequired();
}

#endif // WITH_DEV_AUTOMATION_TESTS
