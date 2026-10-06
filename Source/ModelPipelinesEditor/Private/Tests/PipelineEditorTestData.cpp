// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "PipelineEditorTestData.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformMisc.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "NNEModelData.h"
#include "PipelineModelPackage.h"
#include "PipelineModelImporter.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformProcess.h"

namespace UE::ModelPipelines::EditorTests
{
	FString GetDevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		if (!Env.IsEmpty())
		{
			return Env;
		}
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev"));
	}

	FString GetModelsDir()
	{
		return GetDevDir() / TEXT("models");
	}

	TSharedPtr<FJsonObject> LoadImportExpectations(FString& OutError)
	{
		const FString Path = GetDevDir() / TEXT("goldens/import/import_expectations.json");
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			OutError = FString::Printf(TEXT("Missing import expectations '%s'. Generate them with: python Tools/make_import_test_data.py (WP2.1) into <dev>/goldens/import/ and <dev>/models/import_test/."), *Path);
			return nullptr;
		}
		TSharedPtr<FJsonObject> Root;
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
		if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
		{
			OutError = FString::Printf(TEXT("Import expectations '%s' are not valid JSON."), *Path);
			return nullptr;
		}
		return Root;
	}

	TSharedPtr<FJsonObject> FindCase(const TSharedPtr<FJsonObject>& Root, const FString& Id)
	{
		const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
		if (!Root.IsValid() || !Root->TryGetArrayField(TEXT("cases"), Cases))
		{
			return nullptr;
		}
		for (const TSharedPtr<FJsonValue>& C : *Cases)
		{
			const TSharedPtr<FJsonObject>* Obj = nullptr;
			FString CaseId;
			if (C.IsValid() && C->TryGetObject(Obj) && Obj && (*Obj)->TryGetStringField(TEXT("id"), CaseId) && CaseId == Id)
			{
				return *Obj;
			}
		}
		return nullptr;
	}

	bool IsGpuRequired()
	{
		return FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_GPU")) == TEXT("1");
	}

	namespace
	{
		void DiscardOne(UObject* Obj)
		{
			if (!Obj || !IsValid(Obj))
			{
				return;
			}
			UPackage* OldPackage = Obj->GetOutermost();
			UPackage* Transient = GetTransientPackage();
			if (OldPackage != Transient)
			{
				FAssetRegistryModule::AssetDeleted(Obj);   // the importer registered it with AssetCreated
				const FName Unique = MakeUniqueObjectName(Transient, Obj->GetClass(), Obj->GetFName());
				Obj->Rename(*Unique.ToString(), Transient, REN_DontCreateRedirectors | REN_NonTransactional);
			}
			Obj->ClearFlags(RF_Public | RF_Standalone);
			Obj->MarkAsGarbage();
			if (OldPackage && OldPackage != Transient && IsValid(OldPackage))
			{
				OldPackage->SetDirtyFlag(false);
				OldPackage->ClearFlags(RF_Public | RF_Standalone);
				OldPackage->MarkAsGarbage();
			}
		}
	}

	void DiscardImported(const FPipelineImportResult& Result)
	{
		// Both objects are renamed out of their content packages regardless of order.
		DiscardOne(Result.Package.Get());
		DiscardOne(Result.Model.Get());
	}

	void ExpectImportFinding(FAutomationTestBase& Test, const TCHAR* Code, bool bIsError, int32 Count)
	{
		Test.AddExpectedMessage(FString::Printf(TEXT("Import '.*': %s %s:"), bIsError ? TEXT("Error") : TEXT("Warning"), Code),
			ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, Count, true);
	}

	void ExpectImportFailed(FAutomationTestBase& Test, int32 Count)
	{
		Test.AddExpectedMessage(TEXT("Import of '.*' failed: \\d+ errors?"), ELogVerbosity::Warning,
			EAutomationExpectedMessageFlags::Contains, Count, true);
	}

	void ExpectCaseImportLogs(FAutomationTestBase& Test, const FString& Id, int32 Times)
	{
		if (Id == TEXT("wrong_io_for_task"))
		{
			ExpectImportFinding(Test, TEXT("MissingInput"), true, 2 * Times);
			ExpectImportFinding(Test, TEXT("UnexpectedInput"), true, Times);
			ExpectImportFinding(Test, TEXT("LabelCountMismatch"), true, Times);
			ExpectImportFinding(Test, TEXT("OutputNameFallback"), false, Times);
		}
		else if (Id == TEXT("label_mismatch")) { ExpectImportFinding(Test, TEXT("LabelCountMismatch"), true, Times); }
		else if (Id == TEXT("unknown_op")) { ExpectImportFinding(Test, TEXT("ModelCreateFailed"), true, Times); }
		else if (Id == TEXT("missing_preprocessor"))
		{
			ExpectImportFinding(Test, TEXT("MissingPreprocessorConfig"), false, Times);
			ExpectImportFinding(Test, TEXT("InputDimMismatch"), true, Times);
		}
		else if (Id == TEXT("missing_model"))
		{
			ExpectImportFinding(Test, TEXT("MissingModelFile"), true, Times);
			ExpectImportFinding(Test, TEXT("MissingTokenizer"), true, Times);
		}
		else if (Id == TEXT("bad_config_json")) { ExpectImportFinding(Test, TEXT("InvalidJson"), true, Times); }
		else if (Id == TEXT("no_task")) { ExpectImportFinding(Test, TEXT("TaskUndetected"), true, Times); }
		else if (Id == TEXT("bad_id2label")) { ExpectImportFinding(Test, TEXT("InvalidLabels"), true, Times); }
		else { return; }
		ExpectImportFailed(Test, Times);
	}

	void ExpectLicenseWarnings(FAutomationTestBase& Test, bool bSpdx, bool bSource, bool bAttribution)
	{
		if (bSpdx) { ExpectImportFinding(Test, TEXT("LicenseSpdxMissing"), false); }
		if (bSource) { ExpectImportFinding(Test, TEXT("LicenseSourceMissing"), false); }
		if (bAttribution) { ExpectImportFinding(Test, TEXT("LicenseAttributionMissing"), false); }
	}
}

#endif // WITH_DEV_AUTOMATION_TESTS
