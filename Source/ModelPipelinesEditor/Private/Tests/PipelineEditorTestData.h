// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"

struct FPipelineImportResult;

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

	/**
	 * C17: after a test, make an import result disappear. Renames Package and Model (whichever are non-null) into
	 * the transient package (REN_DontCreateRedirectors | REN_NonTransactional), clears RF_Public | RF_Standalone and
	 * marks them, and their now-empty content packages, as garbage. Safe on a failed result and when called twice.
	 */
	void DiscardImported(const FPipelineImportResult& Result);
}

#endif // WITH_DEV_AUTOMATION_TESTS
