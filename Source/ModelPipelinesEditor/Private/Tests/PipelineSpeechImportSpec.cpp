// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9S.6a (C78): import speech folders and validate SpeechToText packages. Written from the brief and the public headers,
// never from the implementation. Expected values come from <dev>/goldens/import/import_expectations.json (stt_* cases) and
// the tiny test graphs (C72c): encoder input_features f32 [1,-1,8] -> encoder_projected [1,-1,4]; decoder encoder_frame [1,4],
// token i64 [1], state_h/state_c [2,1,4] -> token_logits [1,6], duration_logits [1,3], state_*_out.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Containers/Ticker.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformFileManager.h"
#include "HAL/PlatformProcess.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/App.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "NNEModelData.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineBackend.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"
#include "PipelineModelFolder.h"
#include "PipelineModelValidator.h"
#include "PipelineModelImporter.h"
#include "PipelineEditorTestData.h"
// NEEDS (lead): ModelPipelinesEditor.Build.cs needs "ModelPipelinesSpeech" in PrivateDependencyModuleNames for these two.
#include "SpeechToTextPipeline.h"
#include "SpeechPipelineLibrary.h"
#include "SpeechTypes.h"

namespace PipelineSpeechImportSpecPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;
	using namespace UE::ModelPipelines::EditorTests;

	static const TCHAR* const FetchHint = TEXT("generate with: D:/Micromamba/micromambaenv/envs/pipelines/python.exe \"D:/Game Dev/Pipelines-UE/ModelPipelines-UE/Tools/make_import_test_data.py\" (stt_* cases need WP9S.6t)");

	static bool HasMsg(const FPipelineValidationReport& R, ECode Code, ESev Sev)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code && M.Severity == Sev) { return true; }
		}
		return false;
	}

	/** An Error with this code whose message contains Needle (case-sensitive). */
	static bool HasErrorContaining(const FPipelineValidationReport& R, ECode Code, const FString& Needle)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code && M.Severity == ESev::Error && M.Message.Contains(Needle, ESearchCase::CaseSensitive)) { return true; }
		}
		return false;
	}

	static bool Near(double A, double E) { return FMath::Abs(A - E) <= 1e-6 * FMath::Max(1.0, FMath::Abs(E)); }

	static bool CodeFromName(const FString& Name, ECode& Out)
	{
		const int64 V = StaticEnum<EPipelineValidationCode>()->GetValueByNameString(Name);
		if (V == INDEX_NONE) { return false; }
		Out = (ECode)V;
		return true;
	}

	static FString ResolveCaseFolder(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Case)
	{
		FString ModelsRoot, ImportRoot, Folder;
		Root->TryGetStringField(TEXT("models_root"), ModelsRoot);
		Root->TryGetStringField(TEXT("import_root"), ImportRoot);
		Case->TryGetStringField(TEXT("folder"), Folder);
		return (Folder.StartsWith(TEXT("reference/")) ? ModelsRoot : ImportRoot) / Folder;
	}

	static EPipelineTask OverrideOf(const TSharedPtr<FJsonObject>& Case)
	{
		FString Name;
		EPipelineTask T = EPipelineTask::None;
		if (Case->TryGetStringField(TEXT("task_override"), Name)) { LexTryParseString(T, *Name); }
		return T;
	}

	static TArray<FString> StringArray(const TSharedPtr<FJsonObject>& Obj, const TCHAR* Key)
	{
		TArray<FString> Out;
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (Obj->TryGetArrayField(Key, Arr)) { for (const TSharedPtr<FJsonValue>& V : *Arr) { Out.Add(V->AsString()); } }
		return Out;
	}

	/** The "speech" record of an expectations case as settings. */
	static FPipelineSpeechSettings SpeechFromJson(const TSharedPtr<FJsonObject>& Case)
	{
		FPipelineSpeechSettings S;
		const TSharedPtr<FJsonObject>* E = nullptr;
		if (!Case->TryGetObjectField(TEXT("speech"), E)) { return S; }
		double V = 0;
		if ((*E)->TryGetNumberField(TEXT("sample_rate"), V)) { S.SampleRate = (int32)V; }
		if ((*E)->TryGetNumberField(TEXT("num_mel_bins"), V)) { S.NumMelBins = (int32)V; }
		if ((*E)->TryGetNumberField(TEXT("fft_size"), V)) { S.FftSize = (int32)V; }
		if ((*E)->TryGetNumberField(TEXT("window_length"), V)) { S.WindowLength = (int32)V; }
		if ((*E)->TryGetNumberField(TEXT("hop_length"), V)) { S.HopLength = (int32)V; }
		if ((*E)->TryGetNumberField(TEXT("preemphasis"), V)) { S.Preemphasis = (float)V; }
		if ((*E)->TryGetNumberField(TEXT("blank_token_id"), V)) { S.BlankTokenId = (int32)V; }
		if ((*E)->TryGetNumberField(TEXT("max_symbols_per_step"), V)) { S.MaxSymbolsPerStep = (int32)V; }
		if ((*E)->TryGetNumberField(TEXT("subsampling_factor"), V)) { S.SubsamplingFactor = (int32)V; }
		if ((*E)->TryGetNumberField(TEXT("silence_threshold_db"), V)) { S.SilenceThresholdDb = (float)V; }
		const TArray<TSharedPtr<FJsonValue>>* D = nullptr;
		if ((*E)->TryGetArrayField(TEXT("durations"), D))
		{
			S.Durations.Reset();
			for (const TSharedPtr<FJsonValue>& X : *D) { S.Durations.Add((int32)X->AsNumber()); }
		}
		return S;
	}

	/** One line per mismatching speech field; Worst tracks the largest relative float difference. */
	static void CompareSpeech(const FPipelineSpeechSettings& A, const FPipelineSpeechSettings& E, const FString& Id, TArray<FString>& Fails, double& Worst)
	{
		auto Int = [&](const TCHAR* Name, int32 Got, int32 Exp)
		{
			if (Got != Exp) { Fails.Add(FString::Printf(TEXT("[%s] speech.%s %d, expected %d"), *Id, Name, Got, Exp)); }
		};
		auto Flt = [&](const TCHAR* Name, double Got, double Exp)
		{
			Worst = FMath::Max(Worst, FMath::Abs(Got - Exp) / FMath::Max(1.0, FMath::Abs(Exp)));
			if (!Near(Got, Exp)) { Fails.Add(FString::Printf(TEXT("[%s] speech.%s %.9g, expected %.9g"), *Id, Name, Got, Exp)); }
		};
		Int(TEXT("sample_rate"), A.SampleRate, E.SampleRate);
		Int(TEXT("num_mel_bins"), A.NumMelBins, E.NumMelBins);
		Int(TEXT("fft_size"), A.FftSize, E.FftSize);
		Int(TEXT("window_length"), A.WindowLength, E.WindowLength);
		Int(TEXT("hop_length"), A.HopLength, E.HopLength);
		Flt(TEXT("preemphasis"), A.Preemphasis, E.Preemphasis);
		Int(TEXT("blank_token_id"), A.BlankTokenId, E.BlankTokenId);
		Int(TEXT("max_symbols_per_step"), A.MaxSymbolsPerStep, E.MaxSymbolsPerStep);
		Int(TEXT("subsampling_factor"), A.SubsamplingFactor, E.SubsamplingFactor);
		Flt(TEXT("silence_threshold_db"), A.SilenceThresholdDb, E.SilenceThresholdDb);
		if (A.Durations != E.Durations)
		{
			Fails.Add(FString::Printf(TEXT("[%s] speech.durations has %d entries (first %d), expected %d (first %d)"), *Id, A.Durations.Num(),
				A.Durations.Num() ? A.Durations[0] : -1, E.Durations.Num(), E.Durations.Num() ? E.Durations[0] : -1));
		}
	}

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

	/** Loads a JSON file of a (temp) folder, applies Edit and writes it back. */
	static bool Mutate(const FString& Folder, const TCHAR* File, TFunctionRef<void(FJsonObject&)> Edit)
	{
		FString Text;
		TSharedPtr<FJsonObject> Root;
		if (!FFileHelper::LoadFileToString(Text, *(Folder / File)) || !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root) || !Root.IsValid())
		{
			return false;
		}
		Edit(*Root);
		FString Out;
		if (!FJsonSerializer::Serialize(Root.ToSharedRef(), TJsonWriterFactory<>::Create(&Out))) { return false; }
		return FFileHelper::SaveStringToFile(Out, *(Folder / File), FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
	}

	/** Sets Root[Section][Key] = Value, creating the section object when absent (Section empty: top level). */
	static void SetKey(FJsonObject& Root, const TCHAR* Section, const TCHAR* Key, const TSharedPtr<FJsonValue>& Value)
	{
		if (FCString::Strlen(Section) == 0) { Root.SetField(Key, Value); return; }
		const TSharedPtr<FJsonObject>* Sub = nullptr;
		if (Root.TryGetObjectField(Section, Sub)) { (*Sub)->SetField(Key, Value); return; }
		TSharedPtr<FJsonObject> New = MakeShared<FJsonObject>();
		New->SetField(Key, Value);
		Root.SetObjectField(Section, New);
	}

	static TSharedPtr<FJsonValue> JBool(bool B) { return MakeShared<FJsonValueBoolean>(B); }
	static TSharedPtr<FJsonValue> JNum(double D) { return MakeShared<FJsonValueNumber>(D); }
	static TSharedPtr<FJsonValue> JStr(const FString& S) { return MakeShared<FJsonValueString>(S); }

	static void DiscardObject(UObject* Obj)
	{
		if (!Obj || !IsValid(Obj)) { return; }
		UPackage* Old = Obj->GetOutermost();
		UPackage* Transient = GetTransientPackage();
		if (Old != Transient)
		{
			Obj->Rename(*MakeUniqueObjectName(Transient, Obj->GetClass(), Obj->GetFName()).ToString(), Transient, REN_DontCreateRedirectors | REN_NonTransactional);
		}
		Obj->ClearFlags(RF_Public | RF_Standalone);
		Obj->MarkAsGarbage();
		if (Old != Transient && IsValid(Old)) { Old->SetDirtyFlag(false); Old->MarkAsGarbage(); }
	}
}

BEGIN_DEFINE_SPEC(FPipelineSpeechImportSpec, "Pipelines.Editor.SpeechImport",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	FString Dest;
	FString DestGuid;
	TArray<FPipelineImportResult> Imported;
	TArray<TWeakObjectPtr<UObject>> Manual;
	TArray<TStrongObjectPtr<UObject>> Keep;
	TArray<FString> TempDirs;
	/** Real-import test only: keeps the pipeline alive until AfterEach. */
	TStrongObjectPtr<USpeechToTextPipeline> RealPipeline;

	FPipelineImportResult Run(const FPipelineImportOptions& Options);
	FPipelineImportOptions Opts(const FString& Folder, const FString& Name, bool bCheckGpu = false) const;
	bool LoadCase(const FString& Id, TSharedPtr<FJsonObject>& OutRoot, TSharedPtr<FJsonObject>& OutCase, FString& OutFolder);
	FString NewTempDir(const FString& Leaf);
	/** Copies a case folder to a temp folder; "" (and an error) on failure. */
	FString CopyCase(const FString& SourceFolder, const FString& Leaf);
	FString PkgPath(const FString& Name) const { return Dest / Name; }
	FString ObjPath(const FString& Name) const { return Dest / Name + TEXT(".") + Name; }
	FString ContentDir() const;
	void CheckParsedCase(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Case, TArray<FString>& Fails, double& Worst);
	void CheckNoAssets(const FString& Name, const TCHAR* What);
	UPipelineModelPackage* MakeTinyPackage(const TSharedPtr<FJsonObject>& Case, const FString& Folder, const TCHAR* DecoderFile);
END_DEFINE_SPEC(FPipelineSpeechImportSpec)

FPipelineImportResult FPipelineSpeechImportSpec::Run(const FPipelineImportOptions& Options)
{
	FPipelineImportResult R = ImportPipelineModelFolder(Options);
	Imported.Add(R);
	return R;
}

FPipelineImportOptions FPipelineSpeechImportSpec::Opts(const FString& Folder, const FString& Name, bool bCheckGpu) const
{
	FPipelineImportOptions O;
	O.SourceFolder = Folder;
	O.DestinationPath = Dest;
	O.AssetName = Name;
	O.bSave = false;
	O.bValidate = true;
	O.Validation.bCheckGpu = bCheckGpu;
	return O;
}

bool FPipelineSpeechImportSpec::LoadCase(const FString& Id, TSharedPtr<FJsonObject>& OutRoot, TSharedPtr<FJsonObject>& OutCase, FString& OutFolder)
{
	using namespace PipelineSpeechImportSpecPrivate;
	FString Err;
	OutRoot = LoadImportExpectations(Err);
	if (!OutRoot.IsValid()) { AddError(Err); return false; }
	OutCase = FindCase(OutRoot, Id);
	if (!OutCase.IsValid())
	{
		AddError(FString::Printf(TEXT("case '%s' missing from the import expectations; %s"), *Id, FetchHint));
		return false;
	}
	OutFolder = ResolveCaseFolder(OutRoot, OutCase);
	if (!IFileManager::Get().DirectoryExists(*OutFolder))
	{
		AddError(FString::Printf(TEXT("[%s] test model folder '%s' missing; %s"), *Id, *OutFolder, FetchHint));
		return false;
	}
	return true;
}

FString FPipelineSpeechImportSpec::NewTempDir(const FString& Leaf)
{
	const FString Root = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("PipelinesTests") / FGuid::NewGuid().ToString());
	TempDirs.Add(Root);
	const FString Dir = Root / Leaf;
	IFileManager::Get().MakeDirectory(*Dir, true);
	return Dir;
}

FString FPipelineSpeechImportSpec::CopyCase(const FString& SourceFolder, const FString& Leaf)
{
	const FString Copy = NewTempDir(Leaf);
	if (!FPlatformFileManager::Get().GetPlatformFile().CopyDirectoryTree(*Copy, *SourceFolder, true))
	{
		AddError(FString::Printf(TEXT("could not copy '%s' to '%s'"), *SourceFolder, *Copy));
		return FString();
	}
	return Copy;
}

FString FPipelineSpeechImportSpec::ContentDir() const
{
	return FPaths::ConvertRelativePathToFull(FPaths::ProjectContentDir() / TEXT("PipelinesTests") / DestGuid);
}

void FPipelineSpeechImportSpec::CheckNoAssets(const FString& Name, const TCHAR* What)
{
	for (const FString& N : {Name, Name + TEXT("_Model"), Name + TEXT("_Decoder")})
	{
		TestTrue(*FString::Printf(TEXT("[%s] no object '%s'"), What, *ObjPath(N)), FindObject<UObject>(nullptr, *ObjPath(N)) == nullptr);
		TestTrue(*FString::Printf(TEXT("[%s] no package '%s'"), What, *PkgPath(N)), FindPackage(nullptr, *PkgPath(N)) == nullptr);
	}
}

/** Compares one parsed valid/validation_error case with every field of the expectations (old fields and the two new ones). */
void FPipelineSpeechImportSpec::CheckParsedCase(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Case, TArray<FString>& Fails, double& Worst)
{
	using namespace PipelineSpeechImportSpecPrivate;
	FString Id, S;
	Case->TryGetStringField(TEXT("id"), Id);
	const FString Folder = ResolveCaseFolder(Root, Case);
	const FParseOut P = Parse(Folder, OverrideOf(Case));
	if (!P.bOk || P.Report.NumErrors() != 0)
	{
		Fails.Add(FString::Printf(TEXT("[%s] expected parse success without errors, got ok=%d: %s"), *Id, P.bOk, *P.Report.ToString()));
		return;
	}
	Case->TryGetStringField(TEXT("task"), S);
	if (!S.Equals(LexToString(P.Info.Task), ESearchCase::CaseSensitive)) { Fails.Add(FString::Printf(TEXT("[%s] Task %s, expected %s"), *Id, LexToString(P.Info.Task), *S)); }
	Case->TryGetStringField(TEXT("task_source"), S);
	if (!P.Info.TaskSource.Equals(S, ESearchCase::CaseSensitive)) { Fails.Add(FString::Printf(TEXT("[%s] TaskSource '%s', expected '%s'"), *Id, *P.Info.TaskSource, *S)); }

	FString ModelFile, DecoderFile;
	Case->TryGetStringField(TEXT("model_file"), ModelFile);
	Case->TryGetStringField(TEXT("decoder_model_file"), DecoderFile);
	const FString Full = FPaths::ConvertRelativePathToFull(Folder);
	if (!FPaths::IsSamePath(P.Info.ModelFile, Full / ModelFile)) { Fails.Add(FString::Printf(TEXT("[%s] ModelFile '%s', expected '%s'"), *Id, *P.Info.ModelFile, *(Full / ModelFile))); }
	if (DecoderFile.IsEmpty())
	{
		if (!P.Info.DecoderModelFile.IsEmpty()) { Fails.Add(FString::Printf(TEXT("[%s] DecoderModelFile '%s', expected empty for a non-speech task"), *Id, *P.Info.DecoderModelFile)); }
	}
	else if (!FPaths::IsSamePath(P.Info.DecoderModelFile, Full / DecoderFile))
	{
		Fails.Add(FString::Printf(TEXT("[%s] DecoderModelFile '%s', expected '%s'"), *Id, *P.Info.DecoderModelFile, *(Full / DecoderFile)));
	}

	CompareSpeech(P.Info.Speech, SpeechFromJson(Case), Id, Fails, Worst);

	const TSharedPtr<FJsonObject>* Lic = nullptr;
	if (Case->TryGetObjectField(TEXT("license"), Lic))
	{
		FString Spdx, Url, Attr;
		(*Lic)->TryGetStringField(TEXT("spdx_id"), Spdx);
		(*Lic)->TryGetStringField(TEXT("source_url"), Url);
		(*Lic)->TryGetStringField(TEXT("attribution"), Attr);
		if (!P.Info.License.SpdxId.Equals(Spdx, ESearchCase::CaseSensitive)) { Fails.Add(FString::Printf(TEXT("[%s] License.SpdxId '%s', expected '%s'"), *Id, *P.Info.License.SpdxId, *Spdx)); }
		if (!P.Info.License.SourceUrl.Equals(Url, ESearchCase::CaseSensitive)) { Fails.Add(FString::Printf(TEXT("[%s] License.SourceUrl '%s', expected '%s'"), *Id, *P.Info.License.SourceUrl, *Url)); }
		if (!P.Info.License.Attribution.Equals(Attr, ESearchCase::CaseSensitive)) { Fails.Add(FString::Printf(TEXT("[%s] License.Attribution '%s', expected '%s'"), *Id, *P.Info.License.Attribution, *Attr)); }
	}
	if (LexToString(P.Info.Task) == FString(TEXT("SpeechToText")))
	{
		FString Tok;
		FFileHelper::LoadFileToString(Tok, *(Folder / TEXT("tokenizer.json")));
		if (Tok.IsEmpty() || !P.Info.TokenizerJson.Equals(Tok, ESearchCase::CaseSensitive)) { Fails.Add(FString::Printf(TEXT("[%s] TokenizerJson len %d, expected file text len %d"), *Id, P.Info.TokenizerJson.Len(), Tok.Len())); }
	}
}

/** A hand-built SpeechToText package over the tiny graphs; Speech from the expectations case, tokenizer from the folder. */
UPipelineModelPackage* FPipelineSpeechImportSpec::MakeTinyPackage(const TSharedPtr<FJsonObject>& Case, const FString& Folder, const TCHAR* DecoderFile)
{
	using namespace PipelineSpeechImportSpecPrivate;
	FString Err;
	UNNEModelData* Enc = LoadTinySttModel(TEXT("tiny_stt_encoder.onnx"), Err);
	if (!Enc) { AddError(Err); return nullptr; }
	Keep.Add(TStrongObjectPtr<UObject>(Enc));
	UNNEModelData* Dec = LoadTinySttModel(DecoderFile, Err);
	if (!Dec) { AddError(Err); return nullptr; }
	Keep.Add(TStrongObjectPtr<UObject>(Dec));
	FString Tok;
	if (!FFileHelper::LoadFileToString(Tok, *(Folder / TEXT("tokenizer.json")))) { AddError(FString::Printf(TEXT("cannot read tokenizer.json in '%s'"), *Folder)); return nullptr; }
	UPipelineModelPackage* P = NewObject<UPipelineModelPackage>(GetTransientPackage());
	Keep.Add(TStrongObjectPtr<UObject>(P));
	P->Task = EPipelineTask::SpeechToText;
	P->Model = Enc;
	P->DecoderModel = Dec;
	P->TokenizerJson = Tok;
	P->Speech = SpeechFromJson(Case);
	P->PreferredDevice = EPipelineDevice::CPU;
	P->License.SpdxId = TEXT("MIT");
	P->License.SourceUrl = TEXT("https://example.invalid/tiny_stt");
	P->License.Attribution = TEXT("ModelPipelines test data");
	return P;
}

void FPipelineSpeechImportSpec::Define()
{
	using namespace PipelineSpeechImportSpecPrivate;

	BeforeEach([this]()
	{
		DestGuid = FGuid::NewGuid().ToString(EGuidFormats::Digits);
		Dest = TEXT("/Game/PipelinesTests/") + DestGuid;
	});

	AfterEach([this]()
	{
		RealPipeline.Reset();
		for (const FPipelineImportResult& R : Imported) { DiscardImported(R); }
		Imported.Reset();
		for (const TWeakObjectPtr<UObject>& O : Manual) { DiscardObject(O.Get()); }
		Manual.Reset();
		Keep.Reset();
		for (const FString& D : TempDirs) { IFileManager::Get().DeleteDirectory(*D, false, true); }
		TempDirs.Reset();
		bool bRemoved = !IFileManager::Get().DirectoryExists(*ContentDir());
		for (int32 Try = 0; Try < 20 && !bRemoved; ++Try)
		{
			bRemoved = IFileManager::Get().DeleteDirectory(*ContentDir(), false, true);
			if (!bRemoved) { FPlatformProcess::Sleep(0.05f); }
		}
		if (!bRemoved) { AddWarning(FString::Printf(TEXT("could not remove %s"), *ContentDir())); }
		IFileManager::Get().DeleteDirectory(*(FPaths::ProjectContentDir() / TEXT("PipelinesTests")), false, false);
	});

	// ------------------------------------------------------------------------------------------------ acceptance 1: parse
	Describe("Parse", [this]()
	{
		// Catches a parser that fills only the speech fields it was tested on, or that breaks an old case with the two new fields
		// (decoder_model_file must be "" and Speech the defaults for every non-speech task).
		It("matches every valid/validation_error case incl. decoder_model_file and speech (old cases too)", [this]()
		{
			FString Err;
			TSharedPtr<FJsonObject> Root = LoadImportExpectations(Err);
			if (!Root.IsValid()) { AddError(Err); return; }
			const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
			if (!Root->TryGetArrayField(TEXT("cases"), Cases) || Cases->Num() == 0) { AddError(TEXT("expectations file has no cases")); return; }
			TArray<FString> Fails;
			double Worst = 0.0;
			int32 Checked = 0, Stt = 0;
			for (const TSharedPtr<FJsonValue>& V : *Cases)
			{
				const TSharedPtr<FJsonObject>* C = nullptr;
				FString Expect, Id;
				if (!V->TryGetObject(C)) { continue; }
				(*C)->TryGetStringField(TEXT("expect"), Expect);
				(*C)->TryGetStringField(TEXT("id"), Id);
				if (Expect != TEXT("valid") && Expect != TEXT("validation_error")) { continue; }
				if (!(*C)->HasField(TEXT("speech")) || !(*C)->HasField(TEXT("decoder_model_file")))
				{
					Fails.Add(FString::Printf(TEXT("[%s] expectations record lacks speech/decoder_model_file; regenerate with WP9S.6t's make_import_test_data.py"), *Id));
					continue;
				}
				if (!IFileManager::Get().DirectoryExists(*ResolveCaseFolder(Root, *C)))
				{
					if (Id.StartsWith(TEXT("stt_"))) { Fails.Add(FString::Printf(TEXT("[%s] folder missing; %s"), *Id, FetchHint)); }
					continue; // reference/* folders are the Reference tests' business
				}
				CheckParsedCase(Root, *C, Fails, Worst);
				++Checked;
				Stt += Id.StartsWith(TEXT("stt_")) ? 1 : 0;
			}
			for (const FString& F : Fails) { AddError(F); }
			TestEqual(*FString::Printf(TEXT("4 stt_* parse cases checked (stt_tiny_arch, stt_with_model_onnx, stt_tiny_sidecar_task, stt_defaults); worst float diff %.3g"), Worst), Stt, 4);
			TestTrue(*FString::Printf(TEXT("checked at least 10 cases (checked %d)"), Checked), Checked >= 10);
		});

		// Catches a parser that falls back to model.onnx when encoder.onnx exists: stt_with_model_onnx has both and must use encoder.onnx.
		It("ignores model.onnx next to encoder.onnx (stt_with_model_onnx)", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_with_model_onnx"), Root, Case, Folder)) { return; }
			TestTrue(TEXT("test data: model.onnx is in the folder"), IFileManager::Get().FileExists(*(Folder / TEXT("model.onnx"))));
			const FParseOut P = Parse(Folder);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestTrue(*P.Info.ModelFile, P.Info.ModelFile.EndsWith(TEXT("encoder.onnx")));
			TestFalse(*P.Info.ModelFile, P.Info.ModelFile.EndsWith(TEXT("/model.onnx")));
			TestTrue(*P.Info.DecoderModelFile, P.Info.DecoderModelFile.EndsWith(TEXT("decoder_joint.onnx")));
		});

		// Catches parse errors swallowed or reported under the wrong code, and messages that do not name the culprit.
		It("fails stt_no_decoder (MissingModelFile naming decoder_joint.onnx) and stt_bad_speech_json (InvalidJson naming processor_config.json: feature_size)", [this]()
		{
			for (const TCHAR* Id : {TEXT("stt_no_decoder"), TEXT("stt_bad_speech_json")})
			{
				TSharedPtr<FJsonObject> Root, Case;
				FString Folder;
				if (!LoadCase(Id, Root, Case, Folder)) { continue; }
				const FParseOut P = Parse(Folder, OverrideOf(Case));
				TestFalse(*FString::Printf(TEXT("[%s] returns false"), Id), P.bOk);
				for (const FString& CodeName : StringArray(Case, TEXT("error_codes")))
				{
					ECode Code;
					if (!CodeFromName(CodeName, Code)) { AddError(FString::Printf(TEXT("[%s] unknown code %s"), Id, *CodeName)); continue; }
					TestTrue(*FString::Printf(TEXT("[%s] has Error %s; report:\n%s"), Id, *CodeName, *P.Report.ToString()), HasMsg(P.Report, Code, ESev::Error));
					for (const FString& Needle : StringArray(Case, TEXT("error_message_contains")))
					{
						TestTrue(*FString::Printf(TEXT("[%s] Error %s message contains '%s'; report:\n%s"), Id, *CodeName, *Needle, *P.Report.ToString()), HasErrorContaining(P.Report, Code, Needle));
					}
				}
			}
		});

		// Catches a loose task match: architectures[0] must equal ParakeetForTDT exactly (a different case or suffix is not speech).
		It("does not detect SpeechToText from a near-miss architecture name", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			for (const TCHAR* Arch : {TEXT("parakeetfortdt"), TEXT("ParakeetForTDTExtra"), TEXT("ParakeetForCTC")})
			{
				const FString Copy = CopyCase(Folder, TEXT("near_miss"));
				if (Copy.IsEmpty()) { return; }
				const FString A(Arch);
				TestTrue(TEXT("edited config.json"), Mutate(Copy, TEXT("config.json"), [&](FJsonObject& R) { R.SetArrayField(TEXT("architectures"), {JStr(A)}); }));
				const FParseOut P = Parse(Copy);
				const bool bSpeech = P.bOk && P.Info.Task == EPipelineTask::SpeechToText;
				TestFalse(*FString::Printf(TEXT("'%s' must not give SpeechToText (task %s, ok %d, source '%s')"), Arch, LexToString(P.Info.Task), P.bOk, *P.Info.TaskSource), bSpeech);
			}
		});

		// Catches a missing/empty file check that only looks at model.onnx: each speech file is required and non-empty, the message names it.
		It("reports MissingModelFile naming the file for a missing encoder.onnx and for an empty decoder_joint.onnx", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			{
				const FString Copy = CopyCase(Folder, TEXT("no_encoder"));
				if (Copy.IsEmpty()) { return; }
				IFileManager::Get().Delete(*(Copy / TEXT("encoder.onnx")));
				const FParseOut P = Parse(Copy);
				TestFalse(TEXT("no encoder.onnx: returns false"), P.bOk);
				TestTrue(*P.Report.ToString(), HasErrorContaining(P.Report, ECode::MissingModelFile, TEXT("encoder.onnx")));
			}
			{
				const FString Copy = CopyCase(Folder, TEXT("empty_decoder"));
				if (Copy.IsEmpty()) { return; }
				TestTrue(TEXT("emptied decoder_joint.onnx"), FFileHelper::SaveArrayToFile(TArray<uint8>(), *(Copy / TEXT("decoder_joint.onnx"))));
				const FParseOut P = Parse(Copy);
				TestFalse(TEXT("empty decoder_joint.onnx: returns false"), P.bOk);
				TestTrue(*P.Report.ToString(), HasErrorContaining(P.Report, ECode::MissingModelFile, TEXT("decoder_joint.onnx")));
			}
		});

		// Catches lax typing: ints reject bool and fractional numbers, durations must be a list of ints, strings are never numbers.
		// Each must give InvalidJson whose message holds "<file>: <key>".
		It("rejects wrong JSON types of speech keys with InvalidJson naming '<file>: <key>'", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			struct FBad { const TCHAR* File; const TCHAR* Section; const TCHAR* Key; TSharedPtr<FJsonValue> Value; FString Why; };
			TArray<FBad> Bads = {
				{TEXT("processor_config.json"), TEXT("feature_extractor"), TEXT("feature_size"), JBool(true), TEXT("bool for an int")},
				{TEXT("processor_config.json"), TEXT("feature_extractor"), TEXT("n_fft"), JNum(512.5), TEXT("fractional for an int")},
				{TEXT("processor_config.json"), TEXT("feature_extractor"), TEXT("sampling_rate"), JStr(TEXT("16000")), TEXT("string for an int")},
				{TEXT("processor_config.json"), TEXT("feature_extractor"), TEXT("preemphasis"), JStr(TEXT("0.97")), TEXT("string for a float")},
				{TEXT("config.json"), TEXT(""), TEXT("blank_token_id"), JStr(TEXT("5")), TEXT("string for an int")},
				{TEXT("config.json"), TEXT(""), TEXT("max_symbols_per_step"), JBool(false), TEXT("bool for an int")},
				{TEXT("config.json"), TEXT(""), TEXT("durations"), JStr(TEXT("0,1,2")), TEXT("string for a list")},
				{TEXT("config.json"), TEXT(""), TEXT("durations"), MakeShared<FJsonValueArray>(TArray<TSharedPtr<FJsonValue>>{JNum(0), JNum(1.5)}), TEXT("fractional in the list")},
				{TEXT("config.json"), TEXT("encoder_config"), TEXT("subsampling_factor"), JNum(8.25), TEXT("fractional for an int")},
			};
			for (const FBad& B : Bads)
			{
				const FString Copy = CopyCase(Folder, TEXT("bad_type"));
				if (Copy.IsEmpty()) { return; }
				TestTrue(TEXT("edited json"), Mutate(Copy, B.File, [&](FJsonObject& R) { SetKey(R, B.Section, B.Key, B.Value); }));
				const FParseOut P = Parse(Copy);
				const FString Needle = FString::Printf(TEXT("%s: %s"), B.File, B.Key);
				TestFalse(*FString::Printf(TEXT("%s (%s): returns false"), *Needle, *B.Why), P.bOk);
				TestTrue(*FString::Printf(TEXT("%s (%s): InvalidJson names it; report:\n%s"), *Needle, *B.Why, *P.Report.ToString()), HasErrorContaining(P.Report, ECode::InvalidJson, Needle));
			}
		});

		// Catches preemphasis read as float only (an int is valid: 0 disables) and a missing key being an error instead of the default.
		It("accepts an int preemphasis (0) and takes the default for a removed key", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			const FString Copy = CopyCase(Folder, TEXT("int_preemph"));
			if (Copy.IsEmpty()) { return; }
			TestTrue(TEXT("edited processor_config.json"), Mutate(Copy, TEXT("processor_config.json"), [](FJsonObject& R)
			{
				const TSharedPtr<FJsonObject>* FE = nullptr;
				if (R.TryGetObjectField(TEXT("feature_extractor"), FE))
				{
					(*FE)->SetField(TEXT("preemphasis"), JNum(0));
					(*FE)->RemoveField(TEXT("hop_length"));
				}
			}));
			const FParseOut P = Parse(Copy);
			TestTrue(*P.Report.ToString(), P.bOk);
			TestEqual(TEXT("Preemphasis from int 0"), (double)P.Info.Speech.Preemphasis, 0.0);
			TestEqual(TEXT("removed hop_length gives the default 160"), P.Info.Speech.HopLength, 160);
			TestEqual(TEXT("untouched NumMelBins still 8"), P.Info.Speech.NumMelBins, 8);
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 2: import
	Describe("Import", [this]()
	{
		// Catches an importer that drops the decoder, mixes up encoder/decoder bytes, or skips Speech/license/tokenizer.
		It("imports stt_tiny_arch as X, X_Model, X_Decoder with the expected package, bytes and a clean CPU validation", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			const FPipelineImportResult R = Run(Opts(Folder, TEXT("Stt")));
			TestTrue(*FString::Printf(TEXT("bSuccess; report:\n%s"), *R.Report.ToString()), R.bSuccess);
			TestEqual(TEXT("error count"), R.Report.NumErrors(), 0);
			TestEqual(*FString::Printf(TEXT("warning count (MIT, complete license); report:\n%s"), *R.Report.ToString()), R.Report.NumWarnings(), 0);
			if (!R.bSuccess || !R.Package || !R.Model || !R.DecoderModel) { AddError(TEXT("Package, Model or DecoderModel is null")); return; }

			TestTrue(TEXT("Task SpeechToText"), R.Package->Task == EPipelineTask::SpeechToText);
			TestTrue(TEXT("Package->Model == Model"), R.Package->Model == R.Model);
			TestTrue(TEXT("Package->DecoderModel == DecoderModel"), R.Package->DecoderModel == R.DecoderModel);
			TestTrue(TEXT("Model and DecoderModel are different objects"), R.Model != R.DecoderModel);
			TestEqual(TEXT("package path"), R.Package->GetPathName(), ObjPath(TEXT("Stt")));
			TestEqual(TEXT("encoder path"), R.Model->GetPathName(), ObjPath(TEXT("Stt_Model")));
			TestEqual(TEXT("decoder path"), R.DecoderModel->GetPathName(), ObjPath(TEXT("Stt_Decoder")));
			TestEqual(TEXT("decoder lives in package _Decoder"), R.DecoderModel->GetOutermost()->GetName(), PkgPath(TEXT("Stt_Decoder")));
			for (const UObject* Obj : {(const UObject*)R.Package.Get(), (const UObject*)R.Model.Get(), (const UObject*)R.DecoderModel.Get()})
			{
				TestTrue(*FString::Printf(TEXT("%s is RF_Public|RF_Standalone"), *Obj->GetName()), Obj->HasAllFlags(RF_Public | RF_Standalone));
			}

			TArray<FString> Fails;
			double Worst = 0.0;
			CompareSpeech(R.Package->Speech, SpeechFromJson(Case), TEXT("stt_tiny_arch"), Fails, Worst);
			for (const FString& F : Fails) { AddError(F); }
			TestTrue(*FString::Printf(TEXT("worst relative float diff %.3g <= 1e-6"), Worst), Worst <= 1e-6);

			FString Tok;
			FFileHelper::LoadFileToString(Tok, *(Folder / TEXT("tokenizer.json")));
			TestTrue(TEXT("package TokenizerJson equals tokenizer.json"), !Tok.IsEmpty() && R.Package->TokenizerJson.Equals(Tok, ESearchCase::CaseSensitive));
			TestTrue(TEXT("License.SpdxId MIT"), R.Package->License.SpdxId.Equals(TEXT("MIT"), ESearchCase::CaseSensitive));

			// Bytes: encoder.onnx -> Model, decoder_joint.onnx -> DecoderModel (a swap or a copy of one graph fails both).
			const TPair<const TCHAR*, UNNEModelData*> Pairs[] = {{TEXT("encoder.onnx"), R.Model.Get()}, {TEXT("decoder_joint.onnx"), R.DecoderModel.Get()}};
			for (const TPair<const TCHAR*, UNNEModelData*>& Pair : Pairs)
			{
				TArray64<uint8> FileBytes;
				if (!FFileHelper::LoadFileToArray(FileBytes, *(Folder / Pair.Key))) { AddError(FString::Printf(TEXT("cannot read %s"), Pair.Key)); continue; }
				const TConstArrayView64<uint8> Data = Pair.Value->GetFileData();
				TestEqual(*FString::Printf(TEXT("%s: GetFileData size"), Pair.Key), (int64)Data.Num(), (int64)FileBytes.Num());
				if (Data.Num() == FileBytes.Num()) { TestEqual(*FString::Printf(TEXT("%s: bytes identical"), Pair.Key), FMemory::Memcmp(Data.GetData(), FileBytes.GetData(), FileBytes.Num()), 0); }
				TestEqual(*FString::Printf(TEXT("%s: file type"), Pair.Key), Pair.Value->GetFileType(), FString(TEXT("onnx")));
			}

			// Validation ran the CPU smoke (encoder + decoder).
			const FPipelineDeviceCheck* Cpu = R.Report.DeviceChecks.FindByPredicate([](const FPipelineDeviceCheck& C) { return C.Device == EPipelineDevice::CPU; });
			if (!Cpu) { AddError(TEXT("no CPU DeviceCheck in the import report")); }
			else { TestTrue(TEXT("CPU bCreated"), Cpu->bCreated); TestTrue(TEXT("CPU bRan"), Cpu->bRan); }

			// bSave=false: nothing written under Content.
			TArray<FString> Files;
			IFileManager::Get().FindFilesRecursive(Files, *ContentDir(), TEXT("*.*"), true, false);
			TestEqual(TEXT("files written under Content"), Files.Num(), 0);
		});

		// Catches a sidecar-task folder importing differently from an architectures one (same tiny graphs, CC-BY sidecar).
		It("imports stt_with_model_onnx using encoder.onnx (not model.onnx) bytes", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_with_model_onnx"), Root, Case, Folder)) { return; }
			const FPipelineImportResult R = Run(Opts(Folder, TEXT("Both")));
			TestTrue(*R.Report.ToString(), R.bSuccess);
			if (!R.Model || !R.DecoderModel) { AddError(TEXT("Model or DecoderModel null")); return; }
			TestEqual(TEXT("Model bytes == encoder.onnx size"), (int64)R.Model->GetFileData().Num(), IFileManager::Get().FileSize(*(Folder / TEXT("encoder.onnx"))));
			TestEqual(TEXT("DecoderModel bytes == decoder_joint.onnx size"), (int64)R.DecoderModel->GetFileData().Num(), IFileManager::Get().FileSize(*(Folder / TEXT("decoder_joint.onnx"))));
		});

		// Catches a half import: a folder without decoder_joint.onnx must fail whole, leaving none of the three assets.
		It("fails stt_no_decoder and stt_bad_speech_json and leaves no asset of the three", [this]()
		{
			ExpectImportFinding(*this, TEXT("MissingModelFile"), true, 1);
			ExpectImportFinding(*this, TEXT("InvalidJson"), true, 1);
			ExpectImportFailed(*this, 2);
			for (const TCHAR* Id : {TEXT("stt_no_decoder"), TEXT("stt_bad_speech_json")})
			{
				TSharedPtr<FJsonObject> Root, Case;
				FString Folder;
				if (!LoadCase(Id, Root, Case, Folder)) { continue; }
				const FString Name = FString::Printf(TEXT("Bad_%s"), Id);
				const FPipelineImportResult R = Run(Opts(Folder, Name));
				TestFalse(*FString::Printf(TEXT("[%s] bSuccess false; report:\n%s"), Id, *R.Report.ToString()), R.bSuccess);
				TestTrue(*FString::Printf(TEXT("[%s] Package null"), Id), R.Package == nullptr);
				TestTrue(*FString::Printf(TEXT("[%s] Model null"), Id), R.Model == nullptr);
				TestTrue(*FString::Printf(TEXT("[%s] DecoderModel null"), Id), R.DecoderModel == nullptr);
				CheckNoAssets(Name, Id);
			}
		});

		// Catches overwrite creating new objects (stale references) or updating only the package: all three keep their pointers
		// and take the new folder's contents. The second folder is a copy with a different pre-emphasis.
		It("overwrites all three assets in place with the new folder's speech settings", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			const FString Copy = CopyCase(Folder, TEXT("second"));
			if (Copy.IsEmpty()) { return; }
			TestTrue(TEXT("edited processor_config.json"), Mutate(Copy, TEXT("processor_config.json"), [](FJsonObject& R) { SetKey(R, TEXT("feature_extractor"), TEXT("preemphasis"), JNum(0.5)); }));
			const FPipelineImportResult A = Run(Opts(Folder, TEXT("Ow")));
			if (!A.bSuccess || !A.Package || !A.Model || !A.DecoderModel) { AddError(FString::Printf(TEXT("first import failed: %s"), *A.Report.ToString())); return; }
			TestTrue(TEXT("first import: Preemphasis 0.97"), Near(A.Package->Speech.Preemphasis, 0.97));
			UPipelineModelPackage* P1 = A.Package;
			UNNEModelData* M1 = A.Model;
			UNNEModelData* D1 = A.DecoderModel;

			FPipelineImportOptions O = Opts(Copy, TEXT("Ow"));
			O.bOverwriteExisting = true;
			const FPipelineImportResult B = Run(O);
			TestTrue(*B.Report.ToString(), B.bSuccess);
			TestTrue(TEXT("same Package pointer"), B.Package.Get() == P1);
			TestTrue(TEXT("same Model pointer"), B.Model.Get() == M1);
			TestTrue(TEXT("same DecoderModel pointer"), B.DecoderModel.Get() == D1);
			if (!B.Package) { return; }
			TestTrue(TEXT("Package->Model still the encoder asset"), B.Package->Model == M1);
			TestTrue(TEXT("Package->DecoderModel still the decoder asset"), B.Package->DecoderModel == D1);
			TestTrue(*FString::Printf(TEXT("Preemphasis now 0.5, got %g"), B.Package->Speech.Preemphasis), Near(B.Package->Speech.Preemphasis, 0.5));
			TestEqual(TEXT("decoder bytes from the new folder"), (int64)D1->GetFileData().Num(), IFileManager::Get().FileSize(*(Copy / TEXT("decoder_joint.onnx"))));
		});

		// Catches an overwrite that deletes or half-replaces the old assets before the new folder is proven good.
		It("keeps all three existing assets when an overwrite from stt_no_decoder fails", [this]()
		{
			ExpectImportFinding(*this, TEXT("MissingModelFile"), true, 1);
			ExpectImportFailed(*this, 1);
			TSharedPtr<FJsonObject> Root, Case, Bad;
			FString Folder, BadFolder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			if (!LoadCase(TEXT("stt_no_decoder"), Root, Bad, BadFolder)) { return; }
			const FPipelineImportResult A = Run(Opts(Folder, TEXT("Keep")));
			if (!A.bSuccess || !A.Package || !A.Model || !A.DecoderModel) { AddError(FString::Printf(TEXT("first import failed: %s"), *A.Report.ToString())); return; }
			const int64 DecBytes = A.DecoderModel->GetFileData().Num();
			FPipelineImportOptions O = Opts(BadFolder, TEXT("Keep"));
			O.bOverwriteExisting = true;
			const FPipelineImportResult B = Run(O);
			TestFalse(*B.Report.ToString(), B.bSuccess);
			TestTrue(TEXT("package still found"), FindObject<UPipelineModelPackage>(nullptr, *ObjPath(TEXT("Keep"))) == A.Package.Get());
			TestTrue(TEXT("encoder asset still found"), FindObject<UNNEModelData>(nullptr, *ObjPath(TEXT("Keep_Model"))) == A.Model.Get());
			TestTrue(TEXT("decoder asset still found"), FindObject<UNNEModelData>(nullptr, *ObjPath(TEXT("Keep_Decoder"))) == A.DecoderModel.Get());
			TestEqual(TEXT("decoder bytes unchanged"), (int64)A.DecoderModel->GetFileData().Num(), DecBytes);
			TestTrue(TEXT("Package->DecoderModel unchanged"), A.Package->DecoderModel == A.DecoderModel);
			TestEqual(TEXT("Speech.BlankTokenId unchanged"), A.Package->Speech.BlankTokenId, 5);
		});

		// Catches the decoder asset slot being left out of the existence check: a lone X_Decoder blocks a non-overwrite import.
		It("reports AssetExists when only X_Decoder exists and leaves that object untouched", [this]()
		{
			ExpectImportFinding(*this, TEXT("AssetExists"), true, 1);
			ExpectImportFailed(*this, 1);
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			UPackage* P = CreatePackage(*PkgPath(TEXT("Lone_Decoder")));
			UNNEModelData* Existing = NewObject<UNNEModelData>(P, TEXT("Lone_Decoder"), RF_Public | RF_Standalone);
			Manual.Add(Existing);
			const FPipelineImportResult R = Run(Opts(Folder, TEXT("Lone")));
			TestFalse(*R.Report.ToString(), R.bSuccess);
			TestTrue(*R.Report.ToString(), HasMsg(R.Report, ECode::AssetExists, ESev::Error));
			TestTrue(TEXT("existing object untouched"), FindObject<UNNEModelData>(nullptr, *ObjPath(TEXT("Lone_Decoder"))) == Existing);
			TestTrue(TEXT("no Lone package asset created"), FindObject<UObject>(nullptr, *ObjPath(TEXT("Lone"))) == nullptr);
			TestTrue(TEXT("no Lone_Model asset created"), FindObject<UObject>(nullptr, *ObjPath(TEXT("Lone_Model"))) == nullptr);
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 3: validator
	Describe("Validator", [this]()
	{
		// Catches a validator that rejects a good SpeechToText package, or never runs the encoder/decoder smoke.
		It("passes the tiny encoder+decoder package on CPU with a CPU DeviceCheck that ran", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			UPipelineModelPackage* P = MakeTinyPackage(Case, Folder, TEXT("tiny_stt_decoder_joint.onnx"));
			if (!P) { return; }
			FPipelineValidateOptions O;
			O.bCheckGpu = false;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(P, O);
			TestEqual(*FString::Printf(TEXT("no errors; report:\n%s"), *R.ToString()), R.NumErrors(), 0);
			const FPipelineDeviceCheck* Cpu = R.DeviceChecks.FindByPredicate([](const FPipelineDeviceCheck& C) { return C.Device == EPipelineDevice::CPU; });
			if (!Cpu) { AddError(TEXT("no CPU DeviceCheck")); return; }
			TestTrue(TEXT("CPU bCreated"), Cpu->bCreated);
			TestTrue(TEXT("CPU bRan"), Cpu->bRan);
		});

		// Catches a validator that never inspects the decoder graph: tiny_stt_bad_decoder has a wrong `token` type, so the
		// decoder I/O check must report InputTypeMismatch, and the decoder smoke must not run on top of that.
		It("reports InputTypeMismatch for tiny_stt_bad_decoder and runs no decoder smoke", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			UPipelineModelPackage* P = MakeTinyPackage(Case, Folder, TEXT("tiny_stt_bad_decoder.onnx"));
			if (!P) { return; }
			FPipelineValidateOptions O;
			O.bCheckGpu = false;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(P, O);
			TestTrue(*FString::Printf(TEXT("Error InputTypeMismatch; report:\n%s"), *R.ToString()), HasMsg(R, ECode::InputTypeMismatch, ESev::Error));
			TestTrue(*FString::Printf(TEXT("the InputTypeMismatch message names the token input; report:\n%s"), *R.ToString()), HasErrorContaining(R, ECode::InputTypeMismatch, TEXT("token")));
			TestFalse(*FString::Printf(TEXT("no decoder smoke failure (SmokeOutputInvalid); report:\n%s"), *R.ToString()), HasMsg(R, ECode::SmokeOutputInvalid, ESev::Error));
			TestFalse(*FString::Printf(TEXT("no ModelCreateFailed (the bad decoder is valid ONNX); report:\n%s"), *R.ToString()), HasMsg(R, ECode::ModelCreateFailed, ESev::Error));
		});

		// Catches a validator that does not compare the decoder's token_logits width with BlankTokenId + 1: the tiny decoder
		// has 6 logits (blank 5); declaring blank 7 asks for 8. Some error must come back, never a pass and never a crash.
		It("fails when BlankTokenId does not match the decoder's token_logits width (blank 7 on the tiny pair)", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			UPipelineModelPackage* P = MakeTinyPackage(Case, Folder, TEXT("tiny_stt_decoder_joint.onnx"));
			if (!P) { return; }
			P->Speech.BlankTokenId = 7;
			FPipelineValidateOptions O;
			O.bCheckGpu = false;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(P, O);
			TestTrue(*FString::Printf(TEXT("at least one error; report:\n%s"), *R.ToString()), R.NumErrors() >= 1);
			TestTrue(*FString::Printf(TEXT("the error is a decoder I/O or smoke error (OutputDimMismatch or SmokeOutputInvalid); report:\n%s"), *R.ToString()),
				HasMsg(R, ECode::OutputDimMismatch, ESev::Error) || HasMsg(R, ECode::SmokeOutputInvalid, ESev::Error));
		});

		// Catches a null dereference on a SpeechToText package without a decoder: an error, never a crash or a pass.
		It("reports an error (no crash) for a SpeechToText package without a DecoderModel", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			UPipelineModelPackage* P = MakeTinyPackage(Case, Folder, TEXT("tiny_stt_decoder_joint.onnx"));
			if (!P) { return; }
			P->DecoderModel = nullptr;
			FPipelineValidateOptions O;
			O.bCheckGpu = false;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(P, O);
			TestTrue(*FString::Printf(TEXT("at least one error; report:\n%s"), *R.ToString()), R.NumErrors() >= 1);
		});

		// Catches an empty or unusable tokenizer passing: the validator encodes "Hello world" and needs a non-empty result.
		It("reports an error for a SpeechToText package with an empty TokenizerJson", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			UPipelineModelPackage* P = MakeTinyPackage(Case, Folder, TEXT("tiny_stt_decoder_joint.onnx"));
			if (!P) { return; }
			P->TokenizerJson.Empty();
			FPipelineValidateOptions O;
			O.bCheckGpu = false;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(P, O);
			TestTrue(*FString::Printf(TEXT("at least one error; report:\n%s"), *R.ToString()), R.NumErrors() >= 1);
		});

		// GPU as for other tasks: the encoder may run on the GPU, the decoder always on CPU. Skipped (with a message) when no GPU
		// runtime is available, an error when MODELPIPELINES_REQUIRE_GPU=1; never a silent pass.
		It("validates the tiny pair with bCheckGpu (encoder on GPU, decoder on CPU)", [this]()
		{
			if (!FModelPipelinesCoreModule::Get().GetDefaultBackend().IsDeviceAvailable(EPipelineDevice::GPU))
			{
				if (IsGpuRequired()) { AddError(TEXT("GPU not available but MODELPIPELINES_REQUIRE_GPU=1")); }
				else { AddInfo(TEXT("SKIPPED: no GPU runtime available (set MODELPIPELINES_REQUIRE_GPU=1 to make this a failure)")); }
				return;
			}
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("stt_tiny_arch"), Root, Case, Folder)) { return; }
			UPipelineModelPackage* P = MakeTinyPackage(Case, Folder, TEXT("tiny_stt_decoder_joint.onnx"));
			if (!P) { return; }
			FPipelineValidateOptions O;
			O.bCheckGpu = true;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(P, O);
			TestEqual(*FString::Printf(TEXT("no errors with GPU check; report:\n%s"), *R.ToString()), R.NumErrors(), 0);
			const FPipelineDeviceCheck* Gpu = R.DeviceChecks.FindByPredicate([](const FPipelineDeviceCheck& C) { return C.Device == EPipelineDevice::GPU; });
			if (!Gpu) { AddError(TEXT("no GPU DeviceCheck although the GPU runtime is available")); return; }
			TestTrue(TEXT("GPU bCreated"), Gpu->bCreated);
			TestTrue(TEXT("GPU bRan"), Gpu->bRan);
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 4: real Phonon (LAST, one per process)
	Describe("RealPhonon", [this]()
	{
		// Catches everything the tiny graphs cannot: the real 1.2 GB encoder import, CC-BY-4.0 handling, and a pipeline built from the
		// imported package producing the golden onnx transcript (case-sensitive compare). CPU validation only (bCheckGpu false).
		LatentIt("imports phonon-2-onnx/fp16w (CPU validation) and transcribes the first golden clip to its onnx.text", FTimespan::FromMinutes(25), [this](const FDoneDelegate& Done)
		{
			const FString Folder = GetModelsDir() / TEXT("stt/phonon-2-onnx/fp16w");
			if (!IFileManager::Get().FileExists(*(Folder / TEXT("encoder.onnx"))))
			{
				AddError(FString::Printf(TEXT("Phonon folder '%s' missing. Export it with: D:/Micromamba/micromambaenv/envs/pipelines-stt/python.exe \"D:/Game Dev/Pipelines-UE/ModelPipelines-UE/Tools/export_phonon2_onnx.py\", then write_phonon2_sidecar.py"), *Folder));
				Done.ExecuteIfBound();
				return;
			}
			TArray<float> Samples;
			int32 Rate = 16000;
			FString Expected, ClipId, Err;
			if (!LoadFirstSttGoldenClip(Samples, Rate, Expected, ClipId, Err)) { AddError(Err); Done.ExecuteIfBound(); return; }

			ExpectImportFinding(*this, TEXT("LicenseNotPermissive"), false, 1);
			const FPipelineImportResult R = Run(Opts(Folder, TEXT("Phonon")));
			TestTrue(*FString::Printf(TEXT("bSuccess; report:\n%s"), *R.Report.ToString()), R.bSuccess);
			TestEqual(TEXT("error count"), R.Report.NumErrors(), 0);
			TestEqual(*FString::Printf(TEXT("only the LicenseNotPermissive Warning; report:\n%s"), *R.Report.ToString()), R.Report.NumWarnings(), 1);
			TestTrue(TEXT("Warning LicenseNotPermissive"), HasMsg(R.Report, ECode::LicenseNotPermissive, ESev::Warning));
			if (!R.bSuccess || !R.Package || !R.Model || !R.DecoderModel) { AddError(TEXT("Package, Model or DecoderModel null")); Done.ExecuteIfBound(); return; }
			TestTrue(TEXT("Task SpeechToText"), R.Package->Task == EPipelineTask::SpeechToText);
			TestEqual(TEXT("NumMelBins"), R.Package->Speech.NumMelBins, 128);
			TestEqual(TEXT("BlankTokenId"), R.Package->Speech.BlankTokenId, 8192);
			TestTrue(TEXT("Durations 0..4"), R.Package->Speech.Durations == TArray<int32>({0, 1, 2, 3, 4}));
			TestTrue(TEXT("License CC-BY-4.0"), R.Package->License.SpdxId.Equals(TEXT("CC-BY-4.0"), ESearchCase::CaseSensitive));

			FPipelineResult CreateResult;
			USpeechToTextPipeline* Pipe = USpeechPipelineLibrary::CreateSpeechToTextPipeline(R.Package, CreateResult);
			if (!Pipe || !CreateResult.bSuccess)
			{
				AddError(FString::Printf(TEXT("CreateSpeechToTextPipeline failed: %s"), *CreateResult.Message));
				Done.ExecuteIfBound();
				return;
			}
			RealPipeline.Reset(Pipe);

			struct FShared
			{
				TAtomic<bool> bDone{false};
				FPipelineResult Result;
				FSpeechTranscript Transcript;
			};
			TSharedRef<FShared, ESPMode::ThreadSafe> Shared = MakeShared<FShared, ESPMode::ThreadSafe>();
			TWeakObjectPtr<USpeechToTextPipeline> WeakPipe(Pipe);
			Async(EAsyncExecution::ThreadPool, [Shared, WeakPipe, Samples = MoveTemp(Samples), Rate]()
			{
				// TranscribeBlocking is for non-game, non-executor threads; the pipeline is kept alive by RealPipeline.
				if (USpeechToTextPipeline* PipePtr = WeakPipe.Get())
				{
					Shared->Result = PipePtr->TranscribeBlocking(Samples, Rate, 1, Shared->Transcript);
				}
				Shared->bDone = true;
			});

			const double Start = FPlatformTime::Seconds();
			FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda([this, Shared, Done, Expected, ClipId, Start](float) -> bool
			{
				if (!Shared->bDone)
				{
					if (FPlatformTime::Seconds() - Start < 600.0) { return true; }
					AddError(TEXT("timeout: TranscribeBlocking did not finish within 600 s"));
					Done.ExecuteIfBound();
					return false;
				}
				TestTrue(*FString::Printf(TEXT("TranscribeBlocking succeeded: %s"), *Shared->Result.Message), Shared->Result.bSuccess);
				TestTrue(*FString::Printf(TEXT("[%s] transcript equals onnx.text (case-sensitive).\n got:      '%s'\n expected: '%s'"), *ClipId, *Shared->Transcript.Text, *Expected),
					Shared->Transcript.Text.Equals(Expected, ESearchCase::CaseSensitive));
				Done.ExecuteIfBound();
				return false;
			}), 0.05f);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
