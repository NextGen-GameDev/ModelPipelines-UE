// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"
#include "Misc/AutomationTest.h"

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

	/**
	 * WP7.8a: declare the importer log line "Import '<name>': Error|Warning <Code>: ..." that a test provokes on purpose.
	 * Warning verbosity, regex anchored on the code, exact Count (one line per finding per import).
	 */
	void ExpectImportFinding(FAutomationTestBase& Test, const TCHAR* Code, bool bIsError, int32 Count = 1);

	/** WP7.8a: declare "Import of '<name>' failed: N errors" (one per failed import). */
	void ExpectImportFailed(FAutomationTestBase& Test, int32 Count = 1);

	/**
	 * WP7.8a: all importer log lines of failing to import the named bad test folder Times times: its findings (from the
	 * expectations file) plus "Import of ... failed". Known ids: wrong_io_for_task, label_mismatch, unknown_op,
	 * missing_preprocessor, missing_model, bad_config_json, no_task, bad_id2label. Unknown id: adds nothing.
	 */
	void ExpectCaseImportLogs(FAutomationTestBase& Test, const FString& Id, int32 Times = 1);

	/** WP7.8a: the three license warnings of a folder without license info (SPDX, source URL, attribution). */
	void ExpectLicenseWarnings(FAutomationTestBase& Test, bool bSpdx, bool bSource, bool bAttribution);
}

#endif // WITH_DEV_AUTOMATION_TESTS
