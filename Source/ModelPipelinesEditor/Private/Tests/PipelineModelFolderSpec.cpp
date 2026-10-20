// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"
#include "Math/Vector.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "UObject/Class.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"
#include "PipelineEditorLog.h"
#include "PipelineModelFolder.h"
#include "PipelineEditorTestData.h"

namespace PipelineModelFolderSpecPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;
	using namespace UE::ModelPipelines::EditorTests;

	struct FParseOut
	{
		bool bOk = false;
		FPipelineModelFolderInfo Info;
		FPipelineValidationReport Report;
	};

	static FParseOut Parse(const FString& Folder, EPipelineTask Override = EPipelineTask::None)
	{
		FParseOut Out;
		Out.bOk = ParsePipelineModelFolder(Folder, Override, Out.Info, Out.Report);
		return Out;
	}

	static bool HasMsg(const FPipelineValidationReport& R, ECode Code, ESev Sev)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code && M.Severity == Sev)
			{
				return true;
			}
		}
		return false;
	}

	static bool HasErr(const FPipelineValidationReport& R, ECode Code) { return HasMsg(R, Code, ESev::Error); }
	static bool HasWarn(const FPipelineValidationReport& R, ECode Code) { return HasMsg(R, Code, ESev::Warning); }

	static bool Near(double A, double E)
	{
		return FMath::Abs(A - E) <= 1e-6 * FMath::Max(1.0, FMath::Abs(E));
	}

	static const TCHAR* PoolingName(EPipelinePooling P)
	{
		switch (P)
		{
		case EPipelinePooling::Mean: return TEXT("Mean");
		case EPipelinePooling::CLS: return TEXT("CLS");
		case EPipelinePooling::None: return TEXT("None");
		}
		return TEXT("?");
	}

	static const TCHAR* ResizeName(EPipelineImageResize R)
	{
		return R == EPipelineImageResize::Stretch ? TEXT("Stretch") : R == EPipelineImageResize::StretchCenterCrop ? TEXT("StretchCenterCrop") : TEXT("ShortestEdgeCenterCrop");
	}

	static const TCHAR* ActivationName(EPipelineScoreActivation A)
	{
		return A == EPipelineScoreActivation::Softmax ? TEXT("Softmax") : TEXT("Sigmoid");
	}

	static const TCHAR* DeviceName(EPipelineDevice D)
	{
		switch (D)
		{
		case EPipelineDevice::Auto: return TEXT("Auto");
		case EPipelineDevice::CPU: return TEXT("CPU");
		case EPipelineDevice::GPU: return TEXT("GPU");
		}
		return TEXT("?");
	}

	static FString ErrorNames(const FPipelineValidationReport& R)
	{
		return R.ToString();
	}

	// ---- temp folders under <Saved>/PipelinesTests/<guid>/ ----

	static FString NewTempDir(TArray<FString>& Dirs, const FString& SubFolder = FString())
	{
		FString Root = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("PipelinesTests") / FGuid::NewGuid().ToString());
		Dirs.Add(Root);
		FString Dir = SubFolder.IsEmpty() ? Root : (Root / SubFolder);
		IFileManager::Get().MakeDirectory(*Dir, true);
		return Dir;
	}

	static void Put(const FString& Dir, const FString& Rel, const FString& Text)
	{
		FFileHelper::SaveStringToFile(Text, *(Dir / Rel), FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
	}

	static void PutBytes(const FString& Dir, const FString& Rel, const TArray<uint8>& Bytes)
	{
		FFileHelper::SaveArrayToFile(Bytes, *(Dir / Rel));
	}

	/** The parser never reads model bytes, so a few bytes stand in for the ONNX file. */
	static void PutModel(const FString& Dir, const FString& Rel = TEXT("model.onnx"))
	{
		IFileManager::Get().MakeDirectory(*FPaths::GetPath(Dir / Rel), true);
		PutBytes(Dir, Rel, TArray<uint8>({0x08, 0x08, 0x12, 0x04, 'T', 'E', 'S', 'T'}));
	}

	static const TCHAR* TextTokenizerJson = TEXT(R"({"version":"1.0","note":"café 日本"})");

	static const TCHAR* TextConfigJson = TEXT(R"({"architectures":["BertForSequenceClassification"],"model_type":"bert","id2label":{"0":"neg","1":"pos"}})");

	/** TextClassification folder (architecture rule). Extra files are added by the caller. */
	static FString MakeTextFolder(TArray<FString>& Dirs, const FString& ConfigJson = TextConfigJson)
	{
		const FString Dir = NewTempDir(Dirs);
		PutModel(Dir);
		Put(Dir, TEXT("config.json"), ConfigJson);
		Put(Dir, TEXT("tokenizer.json"), TextTokenizerJson);
		return Dir;
	}

	/** ImageClassification folder with the given preprocessor_config.json text (empty = no file). */
	static FString MakeImageFolder(TArray<FString>& Dirs, const FString& PreprocessorJson)
	{
		const FString Dir = NewTempDir(Dirs);
		PutModel(Dir);
		Put(Dir, TEXT("config.json"), TEXT(R"({"architectures":["ResNetForImageClassification"],"model_type":"resnet","id2label":{"0":"a","1":"b","2":"c"}})"));
		if (!PreprocessorJson.IsEmpty())
		{
			Put(Dir, TEXT("preprocessor_config.json"), PreprocessorJson);
		}
		return Dir;
	}

	/** ObjectDetection folder with the given top-level model_type. */
	static FString MakeDetectionFolder(TArray<FString>& Dirs, const FString& ModelType)
	{
		const FString Dir = NewTempDir(Dirs);
		PutModel(Dir);
		Put(Dir, TEXT("config.json"), FString::Printf(TEXT(R"({"architectures":["XForObjectDetection"],"model_type":"%s","id2label":{"0":"a","1":"b"}})"), *ModelType));
		Put(Dir, TEXT("preprocessor_config.json"), TEXT(R"({"size":{"height":64,"width":96},"do_normalize":false})"));
		return Dir;
	}

	static void CleanUp(TArray<FString>& Dirs)
	{
		for (const FString& D : Dirs)
		{
			IFileManager::Get().DeleteDirectory(*D, false, true);
		}
		Dirs.Reset();
	}

	// ---- expectations ----

	static FString ResolveCaseFolder(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Case)
	{
		FString ModelsRoot, ImportRoot, Folder;
		Root->TryGetStringField(TEXT("models_root"), ModelsRoot);
		Root->TryGetStringField(TEXT("import_root"), ImportRoot);
		Case->TryGetStringField(TEXT("folder"), Folder);
		const FString& Base = Folder.StartsWith(TEXT("reference/")) ? ModelsRoot : ImportRoot;
		return Base / Folder;
	}

	static EPipelineTask OverrideOf(const TSharedPtr<FJsonObject>& Case)
	{
		FString Name;
		EPipelineTask T = EPipelineTask::None;
		if (Case->TryGetStringField(TEXT("task_override"), Name))
		{
			LexTryParseString(T, *Name);
		}
		return T;
	}

	static FString NormalizedFull(const FString& Path)
	{
		FString P = FPaths::ConvertRelativePathToFull(Path);
		P.ReplaceInline(TEXT("\\"), TEXT("/"));
		while (P.EndsWith(TEXT("/")))
		{
			P.LeftChopInline(1);
		}
		return P;
	}

	/** Compares one valid/validation_error case; appends one line per mismatch, tracks the worst float diff. */
	static void CheckValidCase(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Case, TArray<FString>& Fails, double& Worst)
	{
		FString Id, Folder;
		Case->TryGetStringField(TEXT("id"), Id);
		const FString Abs = ResolveCaseFolder(Root, Case);
		FParseOut P = Parse(Abs, OverrideOf(Case));
		auto Fail = [&](const FString& What) { Fails.Add(FString::Printf(TEXT("[%s] %s"), *Id, *What)); };

		if (!P.bOk || P.Report.NumErrors() != 0)
		{
			Fail(FString::Printf(TEXT("expected parse success without errors, got ok=%d: %s"), P.bOk, *ErrorNames(P.Report)));
			return;
		}

		FString S;
		Case->TryGetStringField(TEXT("task"), S);
		if (LexToString(P.Info.Task) != S) { Fail(FString::Printf(TEXT("Task %s, expected %s"), LexToString(P.Info.Task), *S)); }
		Case->TryGetStringField(TEXT("task_source"), S);
		if (P.Info.TaskSource != S) { Fail(FString::Printf(TEXT("TaskSource '%s', expected '%s'"), *P.Info.TaskSource, *S)); }
		FString ModelFile;
		Case->TryGetStringField(TEXT("model_file"), ModelFile);
		if (P.Info.ModelFile != NormalizedFull(Abs) + TEXT("/") + ModelFile)
		{
			Fail(FString::Printf(TEXT("ModelFile '%s', expected '%s/%s'"), *P.Info.ModelFile, *NormalizedFull(Abs), *ModelFile));
		}
		if (P.Info.FolderPath != NormalizedFull(Abs)) { Fail(FString::Printf(TEXT("FolderPath '%s'"), *P.Info.FolderPath)); }

		const TArray<TSharedPtr<FJsonValue>>* Labels = nullptr;
		TArray<FString> Expected;
		if (Case->TryGetArrayField(TEXT("labels"), Labels))
		{
			for (const TSharedPtr<FJsonValue>& V : *Labels) { Expected.Add(V->AsString()); }
		}
		if (P.Info.Labels != Expected)
		{
			Fail(FString::Printf(TEXT("Labels differ: got %d, expected %d (first got '%s')"), P.Info.Labels.Num(), Expected.Num(),
				P.Info.Labels.Num() ? *P.Info.Labels[0] : TEXT("")));
		}

		const TSharedPtr<FJsonObject>* Pre = nullptr;
		if (Case->TryGetObjectField(TEXT("preprocess"), Pre))
		{
			const TSharedPtr<FJsonObject>& E = *Pre;
			const FPipelinePreprocessSettings& A = P.Info.Preprocess;
			auto CheckInt = [&](const TCHAR* Key, int32 Actual)
			{
				double V = 0;
				E->TryGetNumberField(Key, V);
				if (Actual != (int32)V) { Fail(FString::Printf(TEXT("%s %d, expected %d"), Key, Actual, (int32)V)); }
			};
			auto CheckStr = [&](const TCHAR* Key, const FString& Actual)
			{
				FString V;
				E->TryGetStringField(Key, V);
				if (Actual != V) { Fail(FString::Printf(TEXT("%s '%s', expected '%s'"), Key, *Actual, *V)); }
			};
			auto CheckBool = [&](const TCHAR* Key, bool Actual)
			{
				bool V = false;
				E->TryGetBoolField(Key, V);
				if (Actual != V) { Fail(FString::Printf(TEXT("%s %d, expected %d"), Key, Actual, V)); }
			};
			auto CheckFloat = [&](const TCHAR* Key, double Actual, double Exp)
			{
				const double D = FMath::Abs(Actual - Exp);
				Worst = FMath::Max(Worst, D / FMath::Max(1.0, FMath::Abs(Exp)));
				if (!Near(Actual, Exp)) { Fail(FString::Printf(TEXT("%s %.9g, expected %.9g"), Key, Actual, Exp)); }
			};
			const EPipelineTask T = P.Info.Task;
			if (IsTextTask(T))
			{
				CheckInt(TEXT("max_tokens"), A.MaxTokens);
				CheckBool(TEXT("add_special_tokens"), A.bAddSpecialTokens);
				CheckStr(TEXT("pooling"), PoolingName(A.Pooling));
				CheckBool(TEXT("normalize_embeddings"), A.bNormalizeEmbeddings);
			}
			if (IsImageTask(T))
			{
				CheckInt(TEXT("image_width"), A.ImageWidth);
				CheckInt(TEXT("image_height"), A.ImageHeight);
				CheckStr(TEXT("resize_mode"), ResizeName(A.ResizeMode));
				CheckInt(TEXT("resize_shortest_edge"), A.ResizeShortestEdge);
				CheckInt(TEXT("resize_width"), A.ResizeWidth);
				CheckInt(TEXT("resize_height"), A.ResizeHeight);
				if (E->HasField(TEXT("resample"))) { CheckStr(TEXT("resample"), FString(LexToString(A.Resample))); } // C32 (lead, WP4.4)
				double Rs = 0;
				E->TryGetNumberField(TEXT("rescale_factor"), Rs);
				CheckFloat(TEXT("rescale_factor"), A.RescaleFactor, Rs);
				const TArray<TSharedPtr<FJsonValue>>* Mean = nullptr;
				const TArray<TSharedPtr<FJsonValue>>* Std = nullptr;
				if (E->TryGetArrayField(TEXT("image_mean"), Mean) && Mean->Num() == 3)
				{
					CheckFloat(TEXT("image_mean.x"), A.ImageMean.X, (*Mean)[0]->AsNumber());
					CheckFloat(TEXT("image_mean.y"), A.ImageMean.Y, (*Mean)[1]->AsNumber());
					CheckFloat(TEXT("image_mean.z"), A.ImageMean.Z, (*Mean)[2]->AsNumber());
				}
				if (E->TryGetArrayField(TEXT("image_std"), Std) && Std->Num() == 3)
				{
					CheckFloat(TEXT("image_std.x"), A.ImageStd.X, (*Std)[0]->AsNumber());
					CheckFloat(TEXT("image_std.y"), A.ImageStd.Y, (*Std)[1]->AsNumber());
					CheckFloat(TEXT("image_std.z"), A.ImageStd.Z, (*Std)[2]->AsNumber());
				}
			}
			if (TaskUsesLabels(T))
			{
				CheckStr(TEXT("score_activation"), ActivationName(P.Info.Postprocess.ScoreActivation));
			}
		}

		const TSharedPtr<FJsonObject>* Lic = nullptr;
		if (Case->TryGetObjectField(TEXT("license"), Lic))
		{
			FString Spdx, Url, Attr;
			(*Lic)->TryGetStringField(TEXT("spdx_id"), Spdx);
			(*Lic)->TryGetStringField(TEXT("source_url"), Url);
			(*Lic)->TryGetStringField(TEXT("attribution"), Attr);
			if (P.Info.License.SpdxId != Spdx) { Fail(FString::Printf(TEXT("License.SpdxId '%s', expected '%s'"), *P.Info.License.SpdxId, *Spdx)); }
			if (P.Info.License.SourceUrl != Url) { Fail(FString::Printf(TEXT("License.SourceUrl '%s', expected '%s'"), *P.Info.License.SourceUrl, *Url)); }
			if (P.Info.License.Attribution != Attr) { Fail(FString::Printf(TEXT("License.Attribution '%s', expected '%s'"), *P.Info.License.Attribution, *Attr)); }
		}
		FString Dev;
		Case->TryGetStringField(TEXT("preferred_device"), Dev);
		if (Dev != DeviceName(P.Info.PreferredDevice)) { Fail(FString::Printf(TEXT("PreferredDevice %s, expected %s"), DeviceName(P.Info.PreferredDevice), *Dev)); }

		// Warnings that belong to parsing must appear as Warning severity.
		const TArray<TSharedPtr<FJsonValue>>* Warns = nullptr;
		if (Case->TryGetArrayField(TEXT("warning_codes"), Warns))
		{
			static const TCHAR* ParseWarnings[] = {TEXT("MissingPreprocessorConfig"), TEXT("UnverifiedDetectionFamily"), TEXT("UnsupportedPooling"), TEXT("UnsupportedResize")};
			for (const TSharedPtr<FJsonValue>& W : *Warns)
			{
				const FString Name = W->AsString();
				bool bParsing = false;
				for (const TCHAR* PW : ParseWarnings) { bParsing |= (Name == PW); }
				if (!bParsing) { continue; }
				const int64 Val = StaticEnum<EPipelineValidationCode>()->GetValueByNameString(Name);
				if (Val == INDEX_NONE || !HasWarn(P.Report, (ECode)Val))
				{
					Fail(FString::Printf(TEXT("expected Warning %s, report: %s"), *Name, *ErrorNames(P.Report)));
				}
			}
		}
	}

	/** One-line fingerprint of everything a parse produces, for the threading test. */
	static FString Fingerprint(const FParseOut& P)
	{
		const FPipelineModelFolderInfo& I = P.Info;
		const FPipelinePreprocessSettings& S = I.Preprocess;
		return FString::Printf(TEXT("%d|%s|%s|%d|%s|%d|%s|%d|%d|%s|%d|%d|%d|%d|%.9g|%.9g,%.9g,%.9g|%.9g,%.9g,%.9g|%s|%s|%s|%s|%s|%s|%s|%s|%d|%d"),
			P.bOk, LexToString(I.Task), *I.TaskSource, I.TokenizerJson.Len(), *FString::Join(I.Labels, TEXT(",")),
			S.MaxTokens, PoolingName(S.Pooling), S.bNormalizeEmbeddings, S.bAddSpecialTokens, ResizeName(S.ResizeMode),
			S.ImageWidth, S.ImageHeight, S.ResizeShortestEdge, 0, S.RescaleFactor,
			S.ImageMean.X, S.ImageMean.Y, S.ImageMean.Z, S.ImageStd.X, S.ImageStd.Y, S.ImageStd.Z,
			ActivationName(I.Postprocess.ScoreActivation), *I.License.SpdxId, *I.License.SourceUrl, *I.License.Attribution,
			DeviceName(I.PreferredDevice), *I.ModelFile, *I.ModelType, *I.Architecture, P.Report.NumErrors(), P.Report.NumWarnings());
	}
}

BEGIN_DEFINE_SPEC(FPipelineModelFolderSpec, "Pipelines.Editor.ModelFolder",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	TArray<FString> TempDirs;
END_DEFINE_SPEC(FPipelineModelFolderSpec)

void FPipelineModelFolderSpec::Define()
{
	using namespace PipelineModelFolderSpecPrivate;

	AfterEach([this]()
	{
		CleanUp(TempDirs);
	});

	Describe("Expectations", [this]()
	{
		// Catches a parser that is only right on the visible-example folders: every case of the generated
		// expectations file is compared field by field (floats 1e-6 relative).
		It("matches every valid/validation_error case", [this]()
		{
			FString Err;
			TSharedPtr<FJsonObject> Root = LoadImportExpectations(Err);
			if (!Root.IsValid()) { AddError(Err); return; }
			const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
			if (!Root->TryGetArrayField(TEXT("cases"), Cases) || Cases->Num() == 0) { AddError(TEXT("expectations file has no cases")); return; }
			TArray<FString> Fails;
			double Worst = 0.0;
			int32 Checked = 0;
			for (const TSharedPtr<FJsonValue>& V : *Cases)
			{
				const TSharedPtr<FJsonObject>* C = nullptr;
				FString Expect, Id;
				if (!V->TryGetObject(C)) { continue; }
				(*C)->TryGetStringField(TEXT("expect"), Expect);
				(*C)->TryGetStringField(TEXT("id"), Id);
				if (Expect != TEXT("valid") && Expect != TEXT("validation_error")) { continue; }
				const FString Folder = ResolveCaseFolder(Root, *C);
				if (!IFileManager::Get().DirectoryExists(*Folder))
				{
					Fails.Add(FString::Printf(TEXT("[%s] test folder '%s' missing; generate it with: python Tools/make_import_test_data.py (and Tools/fetch_reference_models.py for reference/)"), *Id, *Folder));
					continue;
				}
				CheckValidCase(Root, *C, Fails, Worst);
				++Checked;
			}
			for (const FString& F : Fails) { AddError(F); }
			TestTrue(*FString::Printf(TEXT("checked at least 10 cases (checked %d, worst relative float diff %.3g)"), Checked, Worst), Checked >= 10);
		});

		// Catches a parser that forgets the tokenizer text, or that reads it for an image model.
		It("returns TokenizerJson for text tasks and empty for image tasks", [this]()
		{
			FString Err;
			TSharedPtr<FJsonObject> Root = LoadImportExpectations(Err);
			if (!Root.IsValid()) { AddError(Err); return; }
			const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
			Root->TryGetArrayField(TEXT("cases"), Cases);
			int32 Text = 0, Image = 0;
			for (const TSharedPtr<FJsonValue>& V : Cases ? *Cases : TArray<TSharedPtr<FJsonValue>>())
			{
				const TSharedPtr<FJsonObject>* C = nullptr;
				FString Expect, Id, Task;
				if (!V->TryGetObject(C)) { continue; }
				(*C)->TryGetStringField(TEXT("expect"), Expect);
				if (Expect == TEXT("parse_error")) { continue; }
				(*C)->TryGetStringField(TEXT("id"), Id);
				(*C)->TryGetStringField(TEXT("task"), Task);
				const FString Folder = ResolveCaseFolder(Root, *C);
				FParseOut P = Parse(Folder, OverrideOf(*C));
				if (Task.StartsWith(TEXT("Text")))
				{
					FString Expected;
					FFileHelper::LoadFileToString(Expected, *(Folder / TEXT("tokenizer.json")));
					TestTrue(*FString::Printf(TEXT("[%s] tokenizer.json present in test data"), *Id), !Expected.IsEmpty());
					TestTrue(*FString::Printf(TEXT("[%s] TokenizerJson equals file text (len %d vs %d)"), *Id, P.Info.TokenizerJson.Len(), Expected.Len()), P.Info.TokenizerJson == Expected);
					++Text;
				}
				else if (Task.StartsWith(TEXT("Image")) || Task == TEXT("ObjectDetection"))
				{
					TestTrue(*FString::Printf(TEXT("[%s] TokenizerJson empty for image task"), *Id), P.Info.TokenizerJson.IsEmpty());
					++Image;
				}
			}
			TestTrue(TEXT("had text and image cases"), Text > 0 && Image > 0);
		});

		// Catches a parser that swallows errors: each parse_error case must return false with its listed code.
		It("fails every parse_error case with the listed code", [this]()
		{
			FString Err;
			TSharedPtr<FJsonObject> Root = LoadImportExpectations(Err);
			if (!Root.IsValid()) { AddError(Err); return; }
			const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
			Root->TryGetArrayField(TEXT("cases"), Cases);
			int32 N = 0;
			for (const TSharedPtr<FJsonValue>& V : Cases ? *Cases : TArray<TSharedPtr<FJsonValue>>())
			{
				const TSharedPtr<FJsonObject>* C = nullptr;
				FString Expect, Id;
				if (!V->TryGetObject(C)) { continue; }
				(*C)->TryGetStringField(TEXT("expect"), Expect);
				if (Expect != TEXT("parse_error")) { continue; }
				(*C)->TryGetStringField(TEXT("id"), Id);
				FParseOut P = Parse(ResolveCaseFolder(Root, *C), OverrideOf(*C));
				TestFalse(*FString::Printf(TEXT("[%s] returns false"), *Id), P.bOk);
				const TArray<TSharedPtr<FJsonValue>>* Codes = nullptr;
				(*C)->TryGetArrayField(TEXT("error_codes"), Codes);
				for (const TSharedPtr<FJsonValue>& Code : Codes ? *Codes : TArray<TSharedPtr<FJsonValue>>())
				{
					const int64 Val = StaticEnum<EPipelineValidationCode>()->GetValueByNameString(Code->AsString());
					TestTrue(*FString::Printf(TEXT("[%s] has Error %s; report: %s"), *Id, *Code->AsString(), *ErrorNames(P.Report)),
						Val != INDEX_NONE && HasErr(P.Report, (ECode)Val));
					if (Id == TEXT("bad_config_json"))
					{
						bool bNamed = false;
						for (const FPipelineValidationMessage& M : P.Report.Messages) { bNamed |= (M.Code == ECode::InvalidJson && M.Message.Contains(TEXT("config.json"))); }
						TestTrue(TEXT("[bad_config_json] InvalidJson message names config.json"), bNamed);
					}
				}
				++N;
			}
			TestTrue(*FString::Printf(TEXT("had parse_error cases (%d)"), N), N >= 4);
		});

		// Hardcoded examples from the brief, independent of the JSON file.
		It("reads the reference models as the brief says", [this]()
		{
			const FString M = GetModelsDir() / TEXT("reference");
			
			if (!IFileManager::Get().DirectoryExists(*(M / TEXT("all-MiniLM-L6-v2")))) { AddError(FString::Printf(TEXT("Missing reference model folder '%s'. Generate with Tools/fetch_reference_models.py into <dev>/models/reference/."), *M)); return; }
			{
				FParseOut P = Parse(M / TEXT("all-MiniLM-L6-v2"));
				TestTrue(TEXT("MiniLM ok"), P.bOk);
				TestEqual(TEXT("MiniLM task"), (int32)P.Info.Task, (int32)EPipelineTask::TextEmbedding);
				TestEqual(TEXT("MiniLM source"), P.Info.TaskSource, FString(TEXT("modules.json")));
				TestEqual(TEXT("MiniLM max tokens (sentence_bert_config 256)"), P.Info.Preprocess.MaxTokens, 256);
				TestEqual(TEXT("MiniLM pooling"), (int32)P.Info.Preprocess.Pooling, (int32)EPipelinePooling::Mean);
				TestTrue(TEXT("MiniLM normalize (Normalize module in modules.json)"), P.Info.Preprocess.bNormalizeEmbeddings);
				TestEqual(TEXT("MiniLM labels empty"), P.Info.Labels.Num(), 0);
			}
			{
				FParseOut P = Parse(M / TEXT("resnet-50"));
				TestTrue(TEXT("resnet ok"), P.bOk);
				TestEqual(TEXT("resnet task"), (int32)P.Info.Task, (int32)EPipelineTask::ImageClassification);
				TestEqual(TEXT("resnet resize"), (int32)P.Info.Preprocess.ResizeMode, (int32)EPipelineImageResize::ShortestEdgeCenterCrop);
				TestEqual(TEXT("resnet shortest edge (int(224/0.875))"), P.Info.Preprocess.ResizeShortestEdge, 256);
				TestEqual(TEXT("resnet width"), P.Info.Preprocess.ImageWidth, 224);
				TestEqual(TEXT("resnet height"), P.Info.Preprocess.ImageHeight, 224);
				TestEqual(TEXT("resnet label count"), P.Info.Labels.Num(), 1000);
				if (P.Info.Labels.Num() > 0) { TestEqual(TEXT("resnet label 0"), P.Info.Labels[0], FString(TEXT("tench, Tinca tinca"))); }
				TestTrue(TEXT("resnet no tokenizer text"), P.Info.TokenizerJson.IsEmpty());
			}
			{
				FParseOut P = Parse(M / TEXT("rtdetr_r18vd"));
				TestTrue(TEXT("rtdetr ok"), P.bOk);
				TestEqual(TEXT("rtdetr task"), (int32)P.Info.Task, (int32)EPipelineTask::ObjectDetection);
				TestEqual(TEXT("rtdetr resize"), (int32)P.Info.Preprocess.ResizeMode, (int32)EPipelineImageResize::Stretch);
				TestEqual(TEXT("rtdetr width"), P.Info.Preprocess.ImageWidth, 640);
				TestEqual(TEXT("rtdetr height"), P.Info.Preprocess.ImageHeight, 640);
				TestTrue(TEXT("rtdetr mean 0"), P.Info.Preprocess.ImageMean.IsNearlyZero());
				TestTrue(TEXT("rtdetr std 1"), P.Info.Preprocess.ImageStd.Equals(FVector(1, 1, 1), 1e-9));
				TestEqual(TEXT("rtdetr activation"), (int32)P.Info.Postprocess.ScoreActivation, (int32)EPipelineScoreActivation::Sigmoid);
				TestEqual(TEXT("rtdetr labels"), P.Info.Labels.Num(), 80);
				TestEqual(TEXT("rtdetr model_type is top level, not backbone's rt_detr_resnet"), P.Info.ModelType, FString(TEXT("rt_detr")));
				TestFalse(TEXT("rt_detr is a known family: no UnverifiedDetectionFamily"), HasWarn(P.Report, ECode::UnverifiedDetectionFamily));
			}
			{
				FParseOut P = Parse(GetModelsDir() / TEXT("import_test/tiny_text_classification"));
				TestTrue(TEXT("tiny_text_classification ok"), P.bOk);
				TestEqual(TEXT("1e30 model_max_length falls through to max_position_embeddings 128"), P.Info.Preprocess.MaxTokens, 128);
				TestEqual(TEXT("problem_type multi_label -> Sigmoid"), (int32)P.Info.Postprocess.ScoreActivation, (int32)EPipelineScoreActivation::Sigmoid);
			}
		});
	});

	Describe("FolderErrors", [this]()
	{
		// Catches a parser that crashes or returns true on a missing path.
		It("reports FolderNotFound for a nonexistent folder", [this]()
		{
			FParseOut P = Parse(FPaths::ProjectSavedDir() / TEXT("PipelinesTests/does_not_exist_") + FGuid::NewGuid().ToString());
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::FolderNotFound));
		});

		// Catches an empty string being treated as the current directory.
		It("reports FolderNotFound for an empty string", [this]()
		{
			FParseOut P = Parse(FString());
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::FolderNotFound));
		});

		// Catches FileExists-vs-DirectoryExists confusion: a file path is not a folder.
		It("reports FolderNotFound for a path to a file", [this]()
		{
			const FString Dir = NewTempDir(TempDirs);
			PutModel(Dir);
			FParseOut P = Parse(Dir / TEXT("model.onnx"));
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::FolderNotFound));
		});

		// Catches OutInfo not being reset: stale labels from an earlier call must not survive a failed call.
		It("resets OutInfo at the start", [this]()
		{
			FPipelineModelFolderInfo Info;
			Info.Labels = {TEXT("stale")};
			Info.TokenizerJson = TEXT("stale");
			Info.Task = EPipelineTask::TextEmbedding;
			Info.Preprocess.MaxTokens = 7;
			FPipelineValidationReport R;
			const bool bOk = ParsePipelineModelFolder(TEXT(""), EPipelineTask::None, Info, R);
			TestFalse(TEXT("returns false"), bOk);
			TestEqual(TEXT("labels reset"), Info.Labels.Num(), 0);
			TestTrue(TEXT("tokenizer reset"), Info.TokenizerJson.IsEmpty());
			TestEqual(TEXT("task reset"), (int32)Info.Task, (int32)EPipelineTask::None);
			TestEqual(TEXT("max tokens reset"), Info.Preprocess.MaxTokens, FPipelinePreprocessSettings().MaxTokens);
		});

		// Catches "return true only if the report is empty": an Error already in the report is not this call's.
		It("returns true when this call added no Error even if the report already holds one", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			FPipelineValidationReport R;
			R.Add(ESev::Error, ECode::NoModel, TEXT("earlier error"));
			FPipelineModelFolderInfo Info;
			const bool bOk = ParsePipelineModelFolder(Dir, EPipelineTask::None, Info, R);
			TestTrue(*R.ToString(), bOk);
			TestEqual(TEXT("earlier message kept"), R.NumErrors(), 1);
		});

		// Catches a parser that stops at the first problem: both must be reported in one call.
		It("collects MissingModelFile and InvalidJson in one call", [this]()
		{
			const FString Dir = NewTempDir(TempDirs);
			Put(Dir, TEXT("config.json"), TEXT("{ not json"));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::MissingModelFile));
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::InvalidJson));
		});

		// Catches a 0-byte model passing the exists check.
		It("treats a 0-byte model.onnx as MissingModelFile", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			PutBytes(Dir, TEXT("model.onnx"), TArray<uint8>());
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::MissingModelFile));
		});

		// Catches an empty file being skipped as "absent": it exists, so it must fail to parse.
		It("treats a 0-byte config.json as InvalidJson", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			PutBytes(Dir, TEXT("config.json"), TArray<uint8>());
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::InvalidJson));
		});

		// Catches a crash (or acceptance) on binary garbage in a JSON file.
		It("treats 64 KB of random bytes in config.json as InvalidJson without crashing", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			TArray<uint8> Junk;
			Junk.SetNumUninitialized(64 * 1024);
			FRandomStream Rand(1234);
			for (uint8& B : Junk) { B = (uint8)Rand.RandRange(0, 255); }
			Junk[0] = 0xFF; // never a valid JSON start
			PutBytes(Dir, TEXT("config.json"), Junk);
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::InvalidJson));
		});

		// Catches a JSON array/number accepted where an object is required (rule 2).
		It("requires JSON files to be objects", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			Put(Dir, TEXT("config.json"), TEXT("[1,2,3]"));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::InvalidJson));
		});

		// Catches a missing config.json being silently accepted when the sidecar names a labelled task.
		It("reports MissingConfig for a labelled task without config.json", [this]()
		{
			const FString Dir = NewTempDir(TempDirs);
			PutModel(Dir);
			Put(Dir, TEXT("tokenizer.json"), TextTokenizerJson);
			Put(Dir, TEXT("modelpipelines.json"), TEXT(R"({"task":"TextClassification"})"));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::MissingConfig));
		});

		// Catches a text task that does not insist on tokenizer.json.
		It("reports MissingTokenizer for a text task without tokenizer.json", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			IFileManager::Get().Delete(*(Dir / TEXT("tokenizer.json")));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::MissingTokenizer));
		});

		// Catches the override being ignored (acceptance 5): override wins over architectures, source "options".
		It("lets TaskOverride beat auto-detection", [this]()
		{
			FString Err;
			TSharedPtr<FJsonObject> Root = LoadImportExpectations(Err);
			if (!Root.IsValid()) { AddError(Err); return; }
			TSharedPtr<FJsonObject> Case = FindCase(Root, TEXT("tiny_image_classification"));
			if (!Case.IsValid()) { AddError(TEXT("case tiny_image_classification missing from expectations")); return; }
			FParseOut P = Parse(ResolveCaseFolder(Root, Case), EPipelineTask::TextEmbedding);
			TestFalse(TEXT("returns false (no tokenizer.json)"), P.bOk);
			TestEqual(TEXT("Task"), (int32)P.Info.Task, (int32)EPipelineTask::TextEmbedding);
			TestEqual(TEXT("TaskSource"), P.Info.TaskSource, FString(TEXT("options")));
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::MissingTokenizer));
		});

		// Catches TaskUndetected being hidden, and sidecar beating nothing: override also beats the sidecar.
		It("lets TaskOverride beat the sidecar task", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":32})"));
			Put(Dir, TEXT("modelpipelines.json"), TEXT(R"({"task":"TextEmbedding"})"));
			FParseOut P = Parse(Dir, EPipelineTask::ImageClassification);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("Task"), (int32)P.Info.Task, (int32)EPipelineTask::ImageClassification);
			TestEqual(TEXT("TaskSource"), P.Info.TaskSource, FString(TEXT("options")));
		});

		// Catches path normalisation being skipped: backslashes and trailing slash in, clean absolute path out.
		It("normalises FolderPath and ModelFile (forward slashes, no trailing slash)", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			FString Messy = Dir;
			Messy.ReplaceInline(TEXT("/"), TEXT("\\"));
			Messy += TEXT("\\");
			FParseOut P = Parse(Messy);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("FolderPath"), P.Info.FolderPath, NormalizedFull(Dir));
			TestEqual(TEXT("ModelFile"), P.Info.ModelFile, NormalizedFull(Dir) + TEXT("/model.onnx"));
			TestFalse(TEXT("no backslash in FolderPath"), P.Info.FolderPath.Contains(TEXT("\\")));
		});

		// Catches ANSI-only path handling and label decoding: non-ASCII folder name, labels and tokenizer text.
		It("handles a non-ASCII folder path and non-ASCII labels", [this]()
		{
			const FString Dir = NewTempDir(TempDirs, TEXT("模型_café"));
			PutModel(Dir);
			Put(Dir, TEXT("tokenizer.json"), TextTokenizerJson);
			Put(Dir, TEXT("config.json"), TEXT(R"({"architectures":["BertForSequenceClassification"],"id2label":{"0":"日本","1":"café 🙂"}})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("label count"), P.Info.Labels.Num(), 2);
			if (P.Info.Labels.Num() == 2)
			{
				TestEqual(TEXT("label 0"), P.Info.Labels[0], FString(TEXT("日本")));
				TestEqual(TEXT("label 1"), P.Info.Labels[1], FString(TEXT("café 🙂")));
			}
			TestTrue(TEXT("TokenizerJson keeps the non-ASCII text"), P.Info.TokenizerJson.Contains(TEXT("café")));
		});
	});

	Describe("Labels", [this]()
	{
		// Catches id2label {} being accepted as "has labels".
		It("treats an empty id2label as MissingLabels", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs, TEXT(R"({"architectures":["BertForSequenceClassification"],"id2label":{}})"));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::MissingLabels));
		});

		// Catches a missing id2label key passing for a labelled task.
		It("reports MissingLabels when id2label is absent", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs, TEXT(R"({"architectures":["BertForSequenceClassification"]})"));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::MissingLabels));
		});

		// Catches labels taken in map order instead of by index, and gaps not detected.
		It("reports InvalidLabels for keys 0,2,3", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs, TEXT(R"({"architectures":["BertForSequenceClassification"],"id2label":{"0":"a","2":"b","3":"c"}})"));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::InvalidLabels));
		});

		// Catches non-integer keys being skipped silently.
		It("reports InvalidLabels for a non-integer key", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs, TEXT(R"({"architectures":["BertForSequenceClassification"],"id2label":{"0":"a","one":"b"}})"));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::InvalidLabels));
		});

		// Catches sorting by string: with 11 labels "10" sorts before "2"; Labels[i] must be id2label["i"].
		It("orders labels by integer key (11 labels)", [this]()
		{
			FString Map;
			for (int32 i = 10; i >= 0; --i)
			{
				Map += FString::Printf(TEXT("%s\"%d\":\"L%d\""), Map.IsEmpty() ? TEXT("") : TEXT(","), i, i);
			}
			const FString Dir = MakeTextFolder(TempDirs, FString::Printf(TEXT(R"({"architectures":["BertForSequenceClassification"],"id2label":{%s}})"), *Map));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("count"), P.Info.Labels.Num(), 11);
			for (int32 i = 0; i < P.Info.Labels.Num(); ++i)
			{
				TestEqual(*FString::Printf(TEXT("label %d"), i), P.Info.Labels[i], FString::Printf(TEXT("L%d"), i));
			}
		});

		// Catches a wrong-typed id2label (array) being treated as absent or crashing.
		It("reports InvalidJson when id2label is not an object", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs, TEXT(R"({"architectures":["BertForSequenceClassification"],"id2label":["a","b"]})"));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::InvalidJson));
		});
	});

	Describe("Task", [this]()
	{
		// Catches a sidecar enumerator name that is not validated.
		It("reports InvalidJson for an unknown sidecar task name", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			Put(Dir, TEXT("modelpipelines.json"), TEXT(R"({"task":"Detector"})"));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::InvalidJson));
		});

		// Catches a case-sensitive sidecar task and checks that the sidecar beats architectures.
		It("reads the sidecar task case-insensitively and reports modelpipelines.json", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs); // architecture says classification
			Put(Dir, TEXT("modelpipelines.json"), TEXT(R"({"task":"textembedding","unknown_key":{"x":1}})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("Task"), (int32)P.Info.Task, (int32)EPipelineTask::TextEmbedding);
			TestEqual(TEXT("TaskSource"), P.Info.TaskSource, FString(TEXT("modelpipelines.json")));
			TestEqual(TEXT("TextEmbedding has no labels"), P.Info.Labels.Num(), 0);
		});

		// Catches modules.json being ranked below architectures (rule 3 order).
		It("prefers modules.json over architectures", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			Put(Dir, TEXT("modules.json"), TEXT("[]"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("Task"), (int32)P.Info.Task, (int32)EPipelineTask::TextEmbedding);
			TestEqual(TEXT("TaskSource"), P.Info.TaskSource, FString(TEXT("modules.json")));
		});

		// Catches "...Model" matching without a tokenizer.json (image backbones also end in Model).
		It("detects TextEmbedding from a *Model architecture only when tokenizer.json exists", [this]()
		{
			const FString Dir = NewTempDir(TempDirs);
			PutModel(Dir);
			Put(Dir, TEXT("config.json"), TEXT(R"({"architectures":["BertModel"],"model_type":"bert"})"));
			FParseOut NoTok = Parse(Dir);
			TestFalse(TEXT("no tokenizer.json -> TaskUndetected"), NoTok.bOk);
			TestTrue(*NoTok.Report.ToString(), HasErr(NoTok.Report, ECode::TaskUndetected));
			Put(Dir, TEXT("tokenizer.json"), TextTokenizerJson);
			FParseOut WithTok = Parse(Dir);
			TestTrue(*WithTok.Report.ToString(), WithTok.bOk);
			TestEqual(TEXT("Task"), (int32)WithTok.Info.Task, (int32)EPipelineTask::TextEmbedding);
			TestEqual(TEXT("TaskSource"), WithTok.Info.TaskSource, FString(TEXT("architectures")));
			TestEqual(TEXT("Architecture"), WithTok.Info.Architecture, FString(TEXT("BertModel")));
			TestEqual(TEXT("ModelType"), WithTok.Info.ModelType, FString(TEXT("bert")));
		});

		// Catches TaskUndetected not adding MissingConfig when config.json is absent.
		It("reports TaskUndetected and MissingConfig for a folder with only a model", [this]()
		{
			const FString Dir = NewTempDir(TempDirs);
			PutModel(Dir);
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::TaskUndetected));
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::MissingConfig));
		});
	});

	Describe("Text", [this]()
	{
		// Catches model_max_length being read as an integer only (64.0 is a legal JSON number here).
		It("accepts model_max_length written as 64.0", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			Put(Dir, TEXT("tokenizer_config.json"), TEXT(R"({"model_max_length": 64.0})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("MaxTokens"), P.Info.Preprocess.MaxTokens, 64);
			TestTrue(TEXT("bAddSpecialTokens"), P.Info.Preprocess.bAddSpecialTokens);
		});

		// Catches max_position_embeddings not capping the result: min(64... 512, 128) rule.
		It("caps MaxTokens at config max_position_embeddings", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs, TEXT(R"({"architectures":["BertForSequenceClassification"],"max_position_embeddings":128,"id2label":{"0":"a","1":"b"}})"));
			Put(Dir, TEXT("tokenizer_config.json"), TEXT(R"({"model_max_length": 512})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("MaxTokens"), P.Info.Preprocess.MaxTokens, 128);
		});

		// Catches the default of 512 and the sentence_bert_config priority.
		It("uses sentence_bert_config max_seq_length first and 512 when nothing is given", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			FParseOut Default = Parse(Dir);
			TestEqual(TEXT("default MaxTokens"), Default.Info.Preprocess.MaxTokens, 512);
			Put(Dir, TEXT("sentence_bert_config.json"), TEXT(R"({"max_seq_length": 96})"));
			Put(Dir, TEXT("tokenizer_config.json"), TEXT(R"({"model_max_length": 200})"));
			FParseOut P = Parse(Dir);
			TestEqual(TEXT("sentence_bert wins"), P.Info.Preprocess.MaxTokens, 96);
		});

		// Catches a model_max_length above 65536 (the 1e30 sentinel) being used.
		It("ignores a huge model_max_length", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			Put(Dir, TEXT("tokenizer_config.json"), TEXT(R"({"model_max_length": 1000000000000000019884624838656})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("MaxTokens falls through to 512"), P.Info.Preprocess.MaxTokens, 512);
		});

		static const TCHAR* PoolingModules = TEXT(R"([{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},{"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"}])");

		// Catches an unsupported pooling mode being accepted silently (rule 6: Mean + Warning).
		It("falls back to Mean with UnsupportedPooling for max pooling", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			Put(Dir, TEXT("modules.json"), PoolingModules);
			Put(Dir, TEXT("1_Pooling/config.json"), TEXT(R"({"pooling_mode_cls_token":false,"pooling_mode_mean_tokens":false,"pooling_mode_max_tokens":true})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("Pooling"), (int32)P.Info.Preprocess.Pooling, (int32)EPipelinePooling::Mean);
			TestTrue(*P.Report.ToString(), HasWarn(P.Report, ECode::UnsupportedPooling));
			TestFalse(TEXT("no Normalize entry"), P.Info.Preprocess.bNormalizeEmbeddings);
		});

		// Catches CLS being ignored when both flags are present (CLS is checked first).
		It("picks CLS pooling when pooling_mode_cls_token is true", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			Put(Dir, TEXT("modules.json"), PoolingModules);
			Put(Dir, TEXT("1_Pooling/config.json"), TEXT(R"({"pooling_mode_cls_token":true,"pooling_mode_mean_tokens":true})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("Pooling"), (int32)P.Info.Preprocess.Pooling, (int32)EPipelinePooling::CLS);
			TestFalse(TEXT("no UnsupportedPooling"), HasWarn(P.Report, ECode::UnsupportedPooling));
		});

		// Catches normalize detection reading the wrong entry.
		It("sets bNormalizeEmbeddings only when modules.json has a Normalize entry", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			Put(Dir, TEXT("modules.json"), TEXT(R"([{"idx":0,"path":"","type":"sentence_transformers.models.Transformer"},{"idx":1,"path":"2_Normalize","type":"sentence_transformers.models.Normalize"}])"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestTrue(TEXT("normalize"), P.Info.Preprocess.bNormalizeEmbeddings);
			TestEqual(TEXT("Pooling defaults to Mean without a Pooling entry"), (int32)P.Info.Preprocess.Pooling, (int32)EPipelinePooling::Mean);
		});
	});

	Describe("Image", [this]()
	{
		// Catches the crop_size path (no crop_pct) not producing ShortestEdgeCenterCrop with the crop dims.
		It("uses crop_size with do_center_crop", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":{"shortest_edge":256},"do_center_crop":true,"crop_size":{"height":224,"width":224}})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("mode"), (int32)P.Info.Preprocess.ResizeMode, (int32)EPipelineImageResize::ShortestEdgeCenterCrop);
			TestEqual(TEXT("shortest edge"), P.Info.Preprocess.ResizeShortestEdge, 256);
			TestEqual(TEXT("width"), P.Info.Preprocess.ImageWidth, 224);
			TestEqual(TEXT("height"), P.Info.Preprocess.ImageHeight, 224);
		});

		// Catches width/height swapped for a non-square crop_size.
		It("keeps width and height apart for a non-square crop_size", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":{"shortest_edge":300},"do_center_crop":true,"crop_size":{"height":200,"width":280}})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("width"), P.Info.Preprocess.ImageWidth, 280);
			TestEqual(TEXT("height"), P.Info.Preprocess.ImageHeight, 200);
			TestEqual(TEXT("shortest edge"), P.Info.Preprocess.ResizeShortestEdge, 300);
		});

		// Catches the ConvNext rule applied for S >= 384.
		It("uses Stretch for size 384 with crop_pct", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":384,"crop_pct":0.875})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("mode"), (int32)P.Info.Preprocess.ResizeMode, (int32)EPipelineImageResize::Stretch);
			TestEqual(TEXT("width"), P.Info.Preprocess.ImageWidth, 384);
			TestEqual(TEXT("height"), P.Info.Preprocess.ImageHeight, 384);
		});

		// Catches rounding instead of truncation: 100/0.3 = 333.33 -> 333; and 99/0.3 = 330.0 (double) stays 330.
		It("truncates ResizeShortestEdge = S / crop_pct", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":100,"crop_pct":0.3})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("mode"), (int32)P.Info.Preprocess.ResizeMode, (int32)EPipelineImageResize::ShortestEdgeCenterCrop);
			TestEqual(TEXT("shortest edge"), P.Info.Preprocess.ResizeShortestEdge, 333);
			TestEqual(TEXT("width"), P.Info.Preprocess.ImageWidth, 100);
			TestEqual(TEXT("height"), P.Info.Preprocess.ImageHeight, 100);

			const FString Dir2 = MakeImageFolder(TempDirs, TEXT(R"({"size":{"shortest_edge":224},"crop_pct":0.875})"));
			FParseOut P2 = Parse(Dir2);
			TestEqual(TEXT("shortest_edge object also takes the ConvNext rule: 224/0.875"), P2.Info.Preprocess.ResizeShortestEdge, 256);
		});

		// Catches do_rescale false still using the 1/255 factor.
		It("sets RescaleFactor 1.0 when do_rescale is false", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":{"height":64,"width":64},"do_rescale":false,"rescale_factor":0.5})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestTrue(*FString::Printf(TEXT("RescaleFactor %.9g"), P.Info.Preprocess.RescaleFactor), Near(P.Info.Preprocess.RescaleFactor, 1.0));
		});

		// Catches rescale_factor being ignored when do_rescale is true.
		It("uses a given rescale_factor when do_rescale is true", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":{"height":64,"width":64},"rescale_factor":0.5})"));
			FParseOut P = Parse(Dir);
			TestTrue(*FString::Printf(TEXT("RescaleFactor %.9g"), P.Info.Preprocess.RescaleFactor), Near(P.Info.Preprocess.RescaleFactor, 0.5));
		});

		// Catches normalisation not being turned off: mean 0 / std 1.
		It("uses mean 0 and std 1 when do_normalize is false", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":{"height":64,"width":96},"do_normalize":false,"image_mean":[0.5,0.5,0.5],"image_std":[0.2,0.2,0.2]})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestTrue(TEXT("mean zero"), P.Info.Preprocess.ImageMean.IsNearlyZero(1e-9));
			TestTrue(TEXT("std one"), P.Info.Preprocess.ImageStd.Equals(FVector(1, 1, 1), 1e-9));
			TestEqual(TEXT("Stretch"), (int32)P.Info.Preprocess.ResizeMode, (int32)EPipelineImageResize::Stretch);
			TestEqual(TEXT("W from width"), P.Info.Preprocess.ImageWidth, 96);
			TestEqual(TEXT("H from height"), P.Info.Preprocess.ImageHeight, 64);
		});

		// Catches custom mean/std ignored when do_normalize is true.
		It("reads image_mean and image_std", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":{"height":64,"width":64},"image_mean":[0.1,0.2,0.3],"image_std":[0.4,0.5,0.6]})"));
			FParseOut P = Parse(Dir);
			TestTrue(*FString::Printf(TEXT("mean %s"), *P.Info.Preprocess.ImageMean.ToString()), P.Info.Preprocess.ImageMean.Equals(FVector(0.1, 0.2, 0.3), 1e-6));
			TestTrue(*FString::Printf(TEXT("std %s"), *P.Info.Preprocess.ImageStd.ToString()), P.Info.Preprocess.ImageStd.Equals(FVector(0.4, 0.5, 0.6), 1e-6));
		});

		// Catches size objects with only shortest_edge silently treated as representable.
		It("uses Stretch plus UnsupportedResize for shortest_edge without crop information", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":{"shortest_edge":128}})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("mode"), (int32)P.Info.Preprocess.ResizeMode, (int32)EPipelineImageResize::Stretch);
			TestEqual(TEXT("width"), P.Info.Preprocess.ImageWidth, 128);
			TestEqual(TEXT("height"), P.Info.Preprocess.ImageHeight, 128);
			TestTrue(*P.Report.ToString(), HasWarn(P.Report, ECode::UnsupportedResize));
		});

		// Catches integer size without crop info not being a plain Stretch (and no spurious warning).
		It("uses Stretch for an integer size", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":160})"));
			FParseOut P = Parse(Dir);
			TestEqual(TEXT("mode"), (int32)P.Info.Preprocess.ResizeMode, (int32)EPipelineImageResize::Stretch);
			TestEqual(TEXT("width"), P.Info.Preprocess.ImageWidth, 160);
			TestEqual(TEXT("height"), P.Info.Preprocess.ImageHeight, 160);
			TestFalse(TEXT("no UnsupportedResize"), HasWarn(P.Report, ECode::UnsupportedResize));
		});

		// Catches a wrong-typed size being coerced or crashing.
		It("reports InvalidJson when size is a string", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, TEXT(R"({"size":"big"})"));
			FParseOut P = Parse(Dir);
			TestFalse(TEXT("returns false"), P.bOk);
			TestTrue(*P.Report.ToString(), HasErr(P.Report, ECode::InvalidJson));
			bool bNamed = false;
			for (const FPipelineValidationMessage& M : P.Report.Messages) { bNamed |= (M.Code == ECode::InvalidJson && M.Message.Contains(TEXT("preprocessor_config.json"))); }
			TestTrue(TEXT("message names preprocessor_config.json"), bNamed);
		});

		// Catches an absent preprocessor_config.json being an error, or defaults not applied.
		It("warns MissingPreprocessorConfig and uses the struct defaults", [this]()
		{
			const FString Dir = MakeImageFolder(TempDirs, FString());
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestTrue(*P.Report.ToString(), HasWarn(P.Report, ECode::MissingPreprocessorConfig));
			const FPipelinePreprocessSettings Def;
			TestEqual(TEXT("width"), P.Info.Preprocess.ImageWidth, Def.ImageWidth);
			TestEqual(TEXT("height"), P.Info.Preprocess.ImageHeight, Def.ImageHeight);
			TestEqual(TEXT("mode"), (int32)P.Info.Preprocess.ResizeMode, (int32)EPipelineImageResize::Stretch);
			TestTrue(TEXT("rescale default"), Near(P.Info.Preprocess.RescaleFactor, 1.0 / 255.0));
			TestTrue(TEXT("mean default"), P.Info.Preprocess.ImageMean.Equals(Def.ImageMean, 1e-9));
			TestTrue(TEXT("std default"), P.Info.Preprocess.ImageStd.Equals(Def.ImageStd, 1e-9));
		});
	});

	Describe("Postprocess", [this]()
	{
		auto CheckDetection = [this](const TCHAR* ModelType, EPipelineScoreActivation Expected, bool bWarn)
		{
			const FString Dir = MakeDetectionFolder(TempDirs, ModelType);
			FParseOut P = Parse(Dir);
			TestTrue(*FString::Printf(TEXT("%s: %s"), ModelType, *P.Report.ToString()), P.bOk);
			TestEqual(*FString::Printf(TEXT("%s task"), ModelType), (int32)P.Info.Task, (int32)EPipelineTask::ObjectDetection);
			TestEqual(*FString::Printf(TEXT("%s activation"), ModelType), (int32)P.Info.Postprocess.ScoreActivation, (int32)Expected);
			TestEqual(*FString::Printf(TEXT("%s UnverifiedDetectionFamily warning"), ModelType), HasWarn(P.Report, ECode::UnverifiedDetectionFamily), bWarn);
		};

		// Catches the family table being wrong for one of the known families.
		It("maps detr and yolos to Softmax", [this, CheckDetection]()
		{
			CheckDetection(TEXT("detr"), EPipelineScoreActivation::Softmax, false);
			CheckDetection(TEXT("yolos"), EPipelineScoreActivation::Softmax, false);
			CheckDetection(TEXT("table-transformer"), EPipelineScoreActivation::Softmax, false);
		});

		It("maps the RT-DETR family and deformable/conditional/dab detr to Sigmoid", [this, CheckDetection]()
		{
			CheckDetection(TEXT("rt_detr"), EPipelineScoreActivation::Sigmoid, false);
			CheckDetection(TEXT("rt_detr_v2"), EPipelineScoreActivation::Sigmoid, false);
			CheckDetection(TEXT("d_fine"), EPipelineScoreActivation::Sigmoid, false);
			CheckDetection(TEXT("deformable_detr"), EPipelineScoreActivation::Sigmoid, false);
			CheckDetection(TEXT("conditional_detr"), EPipelineScoreActivation::Sigmoid, false);
			CheckDetection(TEXT("dab_detr"), EPipelineScoreActivation::Sigmoid, false);
		});

		// Catches an unknown family silently assumed OK: Sigmoid plus a warning.
		It("maps an unknown detection family to Sigmoid with UnverifiedDetectionFamily", [this, CheckDetection]()
		{
			CheckDetection(TEXT("owlvit"), EPipelineScoreActivation::Sigmoid, true);
		});

		// Catches problem_type being ignored for classification.
		It("uses Sigmoid for multi_label_classification and Softmax otherwise", [this]()
		{
			const FString A = MakeTextFolder(TempDirs, TEXT(R"({"architectures":["BertForSequenceClassification"],"problem_type":"multi_label_classification","id2label":{"0":"a","1":"b"}})"));
			TestEqual(TEXT("multi_label"), (int32)Parse(A).Info.Postprocess.ScoreActivation, (int32)EPipelineScoreActivation::Sigmoid);
			const FString B = MakeTextFolder(TempDirs, TEXT(R"({"architectures":["BertForSequenceClassification"],"problem_type":"single_label_classification","id2label":{"0":"a","1":"b"}})"));
			TestEqual(TEXT("single_label"), (int32)Parse(B).Info.Postprocess.ScoreActivation, (int32)EPipelineScoreActivation::Softmax);
			const FString C = MakeTextFolder(TempDirs);
			TestEqual(TEXT("absent problem_type"), (int32)Parse(C).Info.Postprocess.ScoreActivation, (int32)EPipelineScoreActivation::Softmax);
		});
	});

	Describe("License and device", [this]()
	{
		auto LicenseFor = [this](const FString& Readme, FString& OutSpdx)
		{
			const FString Dir = MakeTextFolder(TempDirs);
			Put(Dir, TEXT("README.md"), Readme);
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			OutSpdx = P.Info.License.SpdxId;
			TestTrue(TEXT("source url stays empty (no sidecar)"), P.Info.License.SourceUrl.IsEmpty());
			TestTrue(TEXT("attribution stays empty (no sidecar)"), P.Info.License.Attribution.IsEmpty());
		};

		// Catches quote stripping missing.
		It("reads a quoted README front matter license", [this, LicenseFor]()
		{
			FString S;
			LicenseFor(TEXT("---\nlicense: \"Apache-2.0\"\ntags:\n- x\n---\n# Title\n"), S);
			TestEqual(TEXT("SpdxId"), S, FString(TEXT("Apache-2.0")));
		});

		// Catches missing normalisation (and CRLF line endings breaking the match).
		It("normalises bsd-3-clause, mit and apache-2.0 case-insensitively", [this, LicenseFor]()
		{
			FString S;
			LicenseFor(TEXT("---\nlicense: bsd-3-clause\n---\nText"), S);
			TestEqual(TEXT("bsd-3-clause"), S, FString(TEXT("BSD-3-Clause")));
			LicenseFor(TEXT("---\r\nlicense: MIT\r\n---\r\nText"), S);
			TestEqual(TEXT("MIT with CRLF"), S, FString(TEXT("MIT")));
			LicenseFor(TEXT("---\nlicense: 'APACHE-2.0'\n---\n"), S);
			TestEqual(TEXT("APACHE-2.0"), S, FString(TEXT("Apache-2.0")));
			LicenseFor(TEXT("---\nlicense: bsd-2-clause\n---\n"), S);
			TestEqual(TEXT("bsd-2-clause"), S, FString(TEXT("BSD-2-Clause")));
		});

		// Catches unknown licenses being normalised away or dropped.
		It("keeps other licenses as written", [this, LicenseFor]()
		{
			FString S;
			LicenseFor(TEXT("---\nlicense: other\n---\n"), S);
			TestEqual(TEXT("other"), S, FString(TEXT("other")));
			LicenseFor(TEXT("---\nlicense: cc-by-nc-4.0\n---\n"), S);
			TestEqual(TEXT("cc-by-nc-4.0"), S, FString(TEXT("cc-by-nc-4.0")));
		});

		// Catches a "license:" line in the README body being read, and a missing front matter giving a value.
		It("leaves SpdxId empty without front matter", [this, LicenseFor]()
		{
			FString S;
			LicenseFor(TEXT("# Model\n\nlicense: mit\n"), S);
			TestEqual(TEXT("body line ignored"), S, FString());
			LicenseFor(TEXT("---\ntags:\n- x\n---\nlicense: mit\n"), S);
			TestEqual(TEXT("license after closing --- ignored"), S, FString());
		});

		// Catches the README winning over the sidecar, and partial sidecar licenses.
		It("lets sidecar license fields win over the README per field", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			Put(Dir, TEXT("README.md"), TEXT("---\nlicense: mit\n---\n"));
			Put(Dir, TEXT("modelpipelines.json"), TEXT(R"({"license":{"spdx_id":"BSD-3-Clause","source_url":"https://example.invalid/x","attribution":"Café (日本), BSD"}})"));
			FParseOut P = Parse(Dir);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("SpdxId"), P.Info.License.SpdxId, FString(TEXT("BSD-3-Clause")));
			TestEqual(TEXT("SourceUrl"), P.Info.License.SourceUrl, FString(TEXT("https://example.invalid/x")));
			TestEqual(TEXT("Attribution"), P.Info.License.Attribution, FString(TEXT("Café (日本), BSD")));
		});

		// Catches the sidecar device being case-sensitive or ignored.
		It("reads preferred_device case-insensitively", [this]()
		{
			const FString Dir = MakeTextFolder(TempDirs);
			TestEqual(TEXT("default"), (int32)Parse(Dir).Info.PreferredDevice, (int32)EPipelineDevice::Auto);
			Put(Dir, TEXT("modelpipelines.json"), TEXT(R"({"preferred_device":"gpu"})"));
			TestEqual(TEXT("gpu"), (int32)Parse(Dir).Info.PreferredDevice, (int32)EPipelineDevice::GPU);
			Put(Dir, TEXT("modelpipelines.json"), TEXT(R"({"preferred_device":"CPU"})"));
			TestEqual(TEXT("CPU"), (int32)Parse(Dir).Info.PreferredDevice, (int32)EPipelineDevice::CPU);
		});
	});

	Describe("Concurrency and speed", [this]()
	{
		// Catches static mutable state in the parser: 8 threads x 50 parses of two folders must all equal the baseline.
		It("is identical and crash-free across 8 threads", [this]()
		{
			const FString Tiny = GetModelsDir() / TEXT("import_test/tiny_object_detection");
			const FString Mini = GetModelsDir() / TEXT("reference/all-MiniLM-L6-v2");
			if (!IFileManager::Get().DirectoryExists(*Tiny) || !IFileManager::Get().DirectoryExists(*Mini))
			{
				AddError(FString::Printf(TEXT("Missing test folders '%s' or '%s'. Generate them with the WP2.1 Tools scripts (make_import_test_data.py)."), *Tiny, *Mini));
				return;
			}
			const FString BaseTiny = Fingerprint(Parse(Tiny));
			const FString BaseMini = Fingerprint(Parse(Mini));
			TArray<TFuture<FString>> Futures;
			for (int32 T = 0; T < 8; ++T)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [Tiny, Mini, BaseTiny, BaseMini]() -> FString
				{
					for (int32 i = 0; i < 50; ++i)
					{
						if (Fingerprint(Parse(Tiny)) != BaseTiny) { return FString::Printf(TEXT("tiny_object_detection differed at iteration %d"), i); }
						if (Fingerprint(Parse(Mini)) != BaseMini) { return FString::Printf(TEXT("all-MiniLM-L6-v2 differed at iteration %d"), i); }
					}
					return FString();
				}));
			}
			for (int32 T = 0; T < Futures.Num(); ++T)
			{
				if (!Futures[T].WaitFor(FTimespan::FromSeconds(60)))
				{
					AddError(FString::Printf(TEXT("thread %d did not finish within 60 s"), T));
					continue;
				}
				const FString Result = Futures[T].Get();
				TestTrue(*FString::Printf(TEXT("thread %d: %s"), T, *Result), Result.IsEmpty());
			}
			TestTrue(TEXT("baselines are successful parses"), BaseTiny.StartsWith(TEXT("1|")) && BaseMini.StartsWith(TEXT("1|")));
		});

		// Catches the parser reading the 81 MB model.
		It("parses rtdetr_r18vd in under 200 ms", [this]()
		{
			const FString Dir = GetModelsDir() / TEXT("reference/rtdetr_r18vd");
			if (!IFileManager::Get().DirectoryExists(*Dir))
			{
				AddError(FString::Printf(TEXT("Missing reference model '%s'. Generate it with the WP2.1 Tools scripts into <dev>/models/reference/."), *Dir));
				return;
			}
			Parse(Dir); // warm the file cache once; the budget is for a normal call
			const double Start = FPlatformTime::Seconds();
			FParseOut P = Parse(Dir);
			const double Ms = (FPlatformTime::Seconds() - Start) * 1000.0;
			TestTrue(*P.Report.ToString(), P.bOk);
			TestTrue(*FString::Printf(TEXT("took %.1f ms (limit 200)"), Ms), Ms < 200.0);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
