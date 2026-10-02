// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "PipelineTextTestData.h"

#include "Dom/JsonValue.h"
#include "HAL/PlatformMisc.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "NNEModelData.h"
#include "PipelineModelPackage.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"

namespace UE::ModelPipelines::TextTests
{
	namespace
	{
		const TCHAR* const FetchHint = TEXT("Generate the text goldens and models with: \"D:/Micromamba/micromambaenv/envs/pipelines/python.exe\" ModelPipelines-UE/Tools/make_text_goldens.py (after Tools/fetch_reference_models.py and Tools/make_import_test_data.py).");
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

	TSharedPtr<FJsonObject> LoadTextGoldens(FString& OutError)
	{
		const FString Path = GetDevDir() / TEXT("goldens/text/text_goldens.json");
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			OutError = FString::Printf(TEXT("Missing text goldens '%s'. %s"), *Path, FetchHint);
			return nullptr;
		}
		TSharedPtr<FJsonObject> Root;
		const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
		if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
		{
			OutError = FString::Printf(TEXT("Text goldens '%s' are not valid JSON. %s"), *Path, FetchHint);
			return nullptr;
		}
		return Root;
	}

	TSharedPtr<FJsonObject> FindModel(const TSharedPtr<FJsonObject>& Root, const FString& Slug)
	{
		if (!Root.IsValid())
		{
			return nullptr;
		}
		const TArray<TSharedPtr<FJsonValue>>* Models = nullptr;
		if (!Root->TryGetArrayField(TEXT("models"), Models))
		{
			return nullptr;
		}
		for (const TSharedPtr<FJsonValue>& V : *Models)
		{
			const TSharedPtr<FJsonObject>* Obj = nullptr;
			if (V.IsValid() && V->TryGetObject(Obj) && (*Obj)->GetStringField(TEXT("slug")) == Slug)
			{
				return *Obj;
			}
		}
		return nullptr;
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
		const FString ModelDir = Root->GetStringField(TEXT("models_root")) / Folder;
		const FString OnnxPath = ModelDir / TEXT("model.onnx");
		const FString TokPath = ModelDir / TEXT("tokenizer.json");

		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *OnnxPath) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("Missing model '%s'. %s"), *OnnxPath, FetchHint);
			return nullptr;
		}
		FString TokText;
		if (!FFileHelper::LoadFileToString(TokText, *TokPath) || TokText.IsEmpty())
		{
			OutError = FString::Printf(TEXT("Missing tokenizer '%s'. %s"), *TokPath, FetchHint);
			return nullptr;
		}

		UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
		if (!LexTryParseString(Pkg->Task, *Model->GetStringField(TEXT("task"))))
		{
			OutError = FString::Printf(TEXT("Unknown task '%s' in goldens."), *Model->GetStringField(TEXT("task")));
			return nullptr;
		}

		const TSharedPtr<FJsonObject>& P = *PackageBlock;
		Pkg->Preprocess.MaxTokens = (int32)P->GetNumberField(TEXT("max_tokens"));
		Pkg->Preprocess.bAddSpecialTokens = P->GetBoolField(TEXT("add_special_tokens"));
		Pkg->Preprocess.bNormalizeEmbeddings = P->GetBoolField(TEXT("normalize_embeddings"));
		const FString Pooling = P->GetStringField(TEXT("pooling"));
		Pkg->Preprocess.Pooling = Pooling.Equals(TEXT("CLS"), ESearchCase::IgnoreCase) ? EPipelinePooling::CLS
			: Pooling.Equals(TEXT("None"), ESearchCase::IgnoreCase) ? EPipelinePooling::None : EPipelinePooling::Mean;
		Pkg->Postprocess.ScoreActivation = P->GetStringField(TEXT("score_activation")).Equals(TEXT("Sigmoid"), ESearchCase::IgnoreCase)
			? EPipelineScoreActivation::Sigmoid : EPipelineScoreActivation::Softmax;
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
		Pkg->TokenizerJson = TokText;
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
}

#endif // WITH_DEV_AUTOMATION_TESTS
