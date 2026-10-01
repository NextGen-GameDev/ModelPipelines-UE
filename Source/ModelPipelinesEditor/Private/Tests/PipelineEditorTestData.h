// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"

namespace UE::ModelPipelines::EditorTests
{
	/** env MODELPIPELINES_DEV_DIR, else <ProjectDir>/../.pipelines-dev (absolute). */
	FString GetDevDir();

	/** <dev>/models. */
	FString GetModelsDir();

	/**
	 * Loads <dev>/goldens/import/import_expectations.json (C16). nullptr + OutError (path and the command to
	 * generate it) on a missing file or invalid JSON.
	 */
	TSharedPtr<FJsonObject> LoadImportExpectations(FString& OutError);

	/** The entry of Root["cases"] whose "id" equals Id, or nullptr. */
	TSharedPtr<FJsonObject> FindCase(const TSharedPtr<FJsonObject>& Root, const FString& Id);

	/** env MODELPIPELINES_REQUIRE_GPU == "1". */
	bool IsGpuRequired();

	// NOTE: void DiscardImported(const FPipelineImportResult&) (C17) is intentionally not declared yet:
	// FPipelineImportResult (C15, WP2.6) does not exist. Add it with the importer tests.
}

#endif // WITH_DEV_AUTOMATION_TESTS
