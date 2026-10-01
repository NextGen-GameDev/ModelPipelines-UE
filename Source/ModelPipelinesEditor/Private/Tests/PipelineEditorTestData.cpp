// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "PipelineEditorTestData.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformMisc.h"

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
}

#endif // WITH_DEV_AUTOMATION_TESTS
