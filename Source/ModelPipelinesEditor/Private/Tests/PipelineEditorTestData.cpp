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
		DiscardOne(Result.DecoderModel.Get());   // WP9S.6a: SpeechToText only, null otherwise
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

	UNNEModelData* LoadTinySttModel(const FString& FileName, FString& OutError)
	{
		const FString Path = GetModelsDir() / TEXT("test") / FileName;
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *Path) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("Missing test model '%s'. Generate it with: D:/Micromamba/micromambaenv/envs/pipelines/python.exe \"D:/Game Dev/Pipelines-UE/ModelPipelines-UE/Tools/make_test_models.py\""), *Path);
			return nullptr;
		}
		UNNEModelData* Data = NewObject<UNNEModelData>(GetTransientPackage());
		Data->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
		return Data;
	}

	bool LoadFirstSttGoldenClip(TArray<float>& OutSamples, int32& OutSampleRate, FString& OutExpectedText, FString& OutClipId, FString& OutError)
	{
		const FString Dir = GetDevDir() / TEXT("goldens/stt");
		const FString Hint = TEXT("Generate with: D:/Micromamba/micromambaenv/envs/pipelines/python.exe \"D:/Game Dev/Pipelines-UE/ModelPipelines-UE/Tools/make_stt_goldens.py\"");
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *(Dir / TEXT("stt_goldens.json"))))
		{
			OutError = FString::Printf(TEXT("Missing STT goldens '%s'. %s"), *(Dir / TEXT("stt_goldens.json")), *Hint);
			return false;
		}
		TSharedPtr<FJsonObject> Root;
		if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root) || !Root.IsValid())
		{
			OutError = TEXT("stt_goldens.json is not valid JSON. ") + Hint;
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>* Clips = nullptr;
		const TSharedPtr<FJsonObject>* Settings = nullptr;
		if (!Root->TryGetArrayField(TEXT("clips"), Clips) || Clips->Num() == 0)
		{
			OutError = TEXT("stt_goldens.json has no clips. ") + Hint;
			return false;
		}
		OutSampleRate = 16000;
		if (Root->TryGetObjectField(TEXT("settings"), Settings))
		{
			double Sr = 0;
			if ((*Settings)->TryGetNumberField(TEXT("sample_rate"), Sr)) { OutSampleRate = (int32)Sr; }
		}
		const TSharedPtr<FJsonObject>* Clip = nullptr;
		const TSharedPtr<FJsonObject>* Audio = nullptr;
		const TSharedPtr<FJsonObject>* Onnx = nullptr;
		FString AudioFile;
		if (!(*Clips)[0]->TryGetObject(Clip) || !(*Clip)->TryGetObjectField(TEXT("audio"), Audio) || !(*Audio)->TryGetStringField(TEXT("file"), AudioFile)
			|| !(*Clip)->TryGetObjectField(TEXT("onnx"), Onnx) || !(*Onnx)->TryGetStringField(TEXT("text"), OutExpectedText))
		{
			OutError = TEXT("first golden clip lacks audio.file or onnx.text. ") + Hint;
			return false;
		}
		(*Clip)->TryGetStringField(TEXT("id"), OutClipId);
		TArray<uint8> Raw;
		if (!FFileHelper::LoadFileToArray(Raw, *(Dir / AudioFile)) || Raw.Num() < 4)
		{
			OutError = FString::Printf(TEXT("Missing golden audio '%s'. %s"), *(Dir / AudioFile), *Hint);
			return false;
		}
		OutSamples.SetNumUninitialized(Raw.Num() / 4);
		FMemory::Memcpy(OutSamples.GetData(), Raw.GetData(), OutSamples.Num() * 4);
		return true;
	}

	void ExpectLicenseWarnings(FAutomationTestBase& Test, bool bSpdx, bool bSource, bool bAttribution)
	{
		if (bSpdx) { ExpectImportFinding(Test, TEXT("LicenseSpdxMissing"), false); }
		if (bSource) { ExpectImportFinding(Test, TEXT("LicenseSourceMissing"), false); }
		if (bAttribution) { ExpectImportFinding(Test, TEXT("LicenseAttributionMissing"), false); }
	}
}

#endif // WITH_DEV_AUTOMATION_TESTS
