// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformFileManager.h"
#include "HAL/PlatformProcess.h"
#include "Math/Vector.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "UObject/Class.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "ObjectTools.h"
#include "NNEModelData.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"
#include "PipelineModelValidator.h"
#include "PipelineModelImporter.h"
#include "PipelineEditorLibrary.h"
#include "PipelineEditorTestData.h"

namespace PipelineModelImporterSpecPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;
	using namespace UE::ModelPipelines::EditorTests;

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

	static int32 IndexOfCode(const FPipelineValidationReport& R, ECode Code)
	{
		for (int32 i = 0; i < R.Messages.Num(); ++i)
		{
			if (R.Messages[i].Code == Code)
			{
				return i;
			}
		}
		return INDEX_NONE;
	}

	static bool CodeFromName(const FString& Name, ECode& Out)
	{
		const int64 V = StaticEnum<EPipelineValidationCode>()->GetValueByNameString(Name);
		if (V == INDEX_NONE)
		{
			return false;
		}
		Out = (ECode)V;
		return true;
	}

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

	static TArray<FString> StringArray(const TSharedPtr<FJsonObject>& Obj, const TCHAR* Key)
	{
		TArray<FString> Out;
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (Obj->TryGetArrayField(Key, Arr))
		{
			for (const TSharedPtr<FJsonValue>& V : *Arr)
			{
				Out.Add(V->AsString());
			}
		}
		return Out;
	}

	/** Renames an object we created by hand into the transient package and drops it. */
	static void DiscardObject(UObject* Obj)
	{
		if (!Obj || !IsValid(Obj))
		{
			return;
		}
		UPackage* Old = Obj->GetOutermost();
		UPackage* Transient = GetTransientPackage();
		if (Old != Transient)
		{
			Obj->Rename(*MakeUniqueObjectName(Transient, Obj->GetClass(), Obj->GetFName()).ToString(), Transient, REN_DontCreateRedirectors | REN_NonTransactional);
		}
		Obj->ClearFlags(RF_Public | RF_Standalone);
		Obj->MarkAsGarbage();
		if (Old != Transient && IsValid(Old))
		{
			Old->SetDirtyFlag(false);
			Old->MarkAsGarbage();
		}
	}

	/**
	 * Compares every package field with one expectations case; one line per mismatch, tracks the worst relative
	 * float difference. DeviceOverride >= 0 replaces the expected PreferredDevice (Options.PreferredDevice != Auto).
	 */
	static void CheckPackageAgainstCase(const UPipelineModelPackage* Pkg, const TSharedPtr<FJsonObject>& Case, const FString& CaseFolder,
		int32 DeviceOverride, TArray<FString>& Fails, double& Worst)
	{
		FString Id;
		Case->TryGetStringField(TEXT("id"), Id);
		auto Fail = [&](const FString& What) { Fails.Add(FString::Printf(TEXT("[%s] %s"), *Id, *What)); };

		FString S;
		Case->TryGetStringField(TEXT("task"), S);
		if (S != LexToString(Pkg->Task)) { Fail(FString::Printf(TEXT("Task %s, expected %s"), LexToString(Pkg->Task), *S)); }

		const TArray<FString> Labels = StringArray(Case, TEXT("labels"));
		if (Pkg->Labels != Labels)
		{
			Fail(FString::Printf(TEXT("Labels differ: got %d, expected %d (first got '%s')"), Pkg->Labels.Num(), Labels.Num(),
				Pkg->Labels.Num() ? *Pkg->Labels[0] : TEXT("")));
		}

		const FPipelinePreprocessSettings& A = Pkg->Preprocess;
		const TSharedPtr<FJsonObject>* Pre = nullptr;
		if (Case->TryGetObjectField(TEXT("preprocess"), Pre))
		{
			const TSharedPtr<FJsonObject>& E = *Pre;
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
				Worst = FMath::Max(Worst, FMath::Abs(Actual - Exp) / FMath::Max(1.0, FMath::Abs(Exp)));
				if (!Near(Actual, Exp)) { Fail(FString::Printf(TEXT("%s %.9g, expected %.9g"), Key, Actual, Exp)); }
			};
			if (IsTextTask(Pkg->Task))
			{
				CheckInt(TEXT("max_tokens"), A.MaxTokens);
				CheckBool(TEXT("add_special_tokens"), A.bAddSpecialTokens);
				CheckStr(TEXT("pooling"), PoolingName(A.Pooling));
				CheckBool(TEXT("normalize_embeddings"), A.bNormalizeEmbeddings);
			}
			if (IsImageTask(Pkg->Task))
			{
				CheckInt(TEXT("image_width"), A.ImageWidth);
				CheckInt(TEXT("image_height"), A.ImageHeight);
				CheckStr(TEXT("resize_mode"), ResizeName(A.ResizeMode));
				CheckInt(TEXT("resize_shortest_edge"), A.ResizeShortestEdge);
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
			if (TaskUsesLabels(Pkg->Task))
			{
				CheckStr(TEXT("score_activation"), ActivationName(Pkg->Postprocess.ScoreActivation));
			}
		}

		const TSharedPtr<FJsonObject>* Lic = nullptr;
		if (Case->TryGetObjectField(TEXT("license"), Lic))
		{
			FString Spdx, Url, Attr;
			(*Lic)->TryGetStringField(TEXT("spdx_id"), Spdx);
			(*Lic)->TryGetStringField(TEXT("source_url"), Url);
			(*Lic)->TryGetStringField(TEXT("attribution"), Attr);
			if (Pkg->License.SpdxId != Spdx) { Fail(FString::Printf(TEXT("License.SpdxId '%s', expected '%s'"), *Pkg->License.SpdxId, *Spdx)); }
			if (Pkg->License.SourceUrl != Url) { Fail(FString::Printf(TEXT("License.SourceUrl '%s', expected '%s'"), *Pkg->License.SourceUrl, *Url)); }
			if (Pkg->License.Attribution != Attr) { Fail(FString::Printf(TEXT("License.Attribution '%s', expected '%s'"), *Pkg->License.Attribution, *Attr)); }
		}

		FString Dev;
		Case->TryGetStringField(TEXT("preferred_device"), Dev);
		if (DeviceOverride >= 0) { Dev = DeviceName((EPipelineDevice)DeviceOverride); }
		if (Dev != DeviceName(Pkg->PreferredDevice)) { Fail(FString::Printf(TEXT("PreferredDevice %s, expected %s"), DeviceName(Pkg->PreferredDevice), *Dev)); }

		// Text tasks carry the tokenizer.json text verbatim; image tasks carry none.
		if (IsTextTask(Pkg->Task))
		{
			FString Expected;
			FFileHelper::LoadFileToString(Expected, *(CaseFolder / TEXT("tokenizer.json")));
			if (Expected.IsEmpty()) { Fail(TEXT("test data has no tokenizer.json")); }
			else if (Pkg->TokenizerJson != Expected) { Fail(FString::Printf(TEXT("TokenizerJson len %d, expected %d (file text)"), Pkg->TokenizerJson.Len(), Expected.Len())); }
		}
		else if (!Pkg->TokenizerJson.IsEmpty())
		{
			Fail(TEXT("image task package carries a TokenizerJson"));
		}
	}
}

BEGIN_DEFINE_SPEC(FPipelineModelImporterSpec, "Pipelines.Editor.Import",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	/** /Game/PipelinesTests/<guid>: unique per test, so nothing collides with another test or an earlier run. */
	FString Dest;
	FString DestGuid;
	/** Every result of every import in the running test; discarded in AfterEach, also after a failed check. */
	TArray<FPipelineImportResult> Imported;
	/** Objects the test made by hand. */
	TArray<TWeakObjectPtr<UObject>> Manual;
	TArray<FString> TempDirs;

	FPipelineImportResult Run(const FPipelineImportOptions& Options);
	FPipelineImportOptions Opts(const FString& Folder, const FString& Name, bool bCheckGpu = false) const;
	bool LoadCase(const FString& Id, TSharedPtr<FJsonObject>& OutRoot, TSharedPtr<FJsonObject>& OutCase, FString& OutFolder);
	FString NewTempDir(const FString& Leaf);
	FString PkgPath(const FString& Name) const { return Dest / Name; }
	FString ObjPath(const FString& Name) const { return Dest / Name + TEXT(".") + Name; }
	FString ContentDir() const;
	bool NothingExistsFor(const FString& Name);
	void CheckFailedClean(const FPipelineImportResult& R, const FString& Name, const TCHAR* What);
	void CheckImportedCase(const FString& Id, const FString& Name, int32 DeviceOverride = -1);
	void ExpectErrorsFromCase(const FPipelineImportResult& R, const TSharedPtr<FJsonObject>& Case, const FString& Id);
END_DEFINE_SPEC(FPipelineModelImporterSpec)

FPipelineImportResult FPipelineModelImporterSpec::Run(const FPipelineImportOptions& Options)
{
	FPipelineImportResult R = ImportPipelineModelFolder(Options);
	Imported.Add(R);
	return R;
}

FPipelineImportOptions FPipelineModelImporterSpec::Opts(const FString& Folder, const FString& Name, bool bCheckGpu) const
{
	FPipelineImportOptions O;
	O.SourceFolder = Folder;
	O.DestinationPath = Dest;
	O.AssetName = Name;
	O.bSave = false;
	O.bValidate = true;
	O.Validation.bCheckGpu = bCheckGpu; // tiny-model tests are CPU-deterministic; GPU is the Reference tests' job
	return O;
}

bool FPipelineModelImporterSpec::LoadCase(const FString& Id, TSharedPtr<FJsonObject>& OutRoot, TSharedPtr<FJsonObject>& OutCase, FString& OutFolder)
{
	using namespace PipelineModelImporterSpecPrivate;
	FString Err;
	OutRoot = LoadImportExpectations(Err);
	if (!OutRoot.IsValid())
	{
		AddError(Err);
		return false;
	}
	OutCase = FindCase(OutRoot, Id);
	if (!OutCase.IsValid())
	{
		AddError(FString::Printf(TEXT("case '%s' missing from the import expectations; regenerate with: python Tools/make_import_test_data.py"), *Id));
		return false;
	}
	OutFolder = ResolveCaseFolder(OutRoot, OutCase);
	if (!IFileManager::Get().DirectoryExists(*OutFolder))
	{
		AddError(FString::Printf(TEXT("[%s] test model folder '%s' missing; generate with: python Tools/make_import_test_data.py (and Tools/fetch_reference_models.py for reference/)"), *Id, *OutFolder));
		return false;
	}
	return true;
}

FString FPipelineModelImporterSpec::NewTempDir(const FString& Leaf)
{
	const FString Root = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("PipelinesTests") / FGuid::NewGuid().ToString());
	TempDirs.Add(Root);
	const FString Dir = Root / Leaf;
	IFileManager::Get().MakeDirectory(*Dir, true);
	return Dir;
}

FString FPipelineModelImporterSpec::ContentDir() const
{
	return FPaths::ConvertRelativePathToFull(FPaths::ProjectContentDir() / TEXT("PipelinesTests") / DestGuid);
}

bool FPipelineModelImporterSpec::NothingExistsFor(const FString& Name)
{
	bool bNone = true;
	for (const FString& N : {Name, Name + TEXT("_Model")})
	{
		if (FindObject<UObject>(nullptr, *ObjPath(N)) != nullptr)
		{
			AddError(FString::Printf(TEXT("object '%s' exists but the import failed"), *ObjPath(N)));
			bNone = false;
		}
		if (FindPackage(nullptr, *PkgPath(N)) != nullptr)
		{
			AddError(FString::Printf(TEXT("package '%s' exists but the import failed"), *PkgPath(N)));
			bNone = false;
		}
	}
	return bNone;
}

void FPipelineModelImporterSpec::CheckFailedClean(const FPipelineImportResult& R, const FString& Name, const TCHAR* What)
{
	TestFalse(*FString::Printf(TEXT("[%s] bSuccess is false; report: %s"), What, *R.Report.ToString()), R.bSuccess);
	TestTrue(*FString::Printf(TEXT("[%s] Package is null"), What), R.Package == nullptr);
	TestTrue(*FString::Printf(TEXT("[%s] Model is null"), What), R.Model == nullptr);
	NothingExistsFor(Name);
}

void FPipelineModelImporterSpec::ExpectErrorsFromCase(const FPipelineImportResult& R, const TSharedPtr<FJsonObject>& Case, const FString& Id)
{
	using namespace PipelineModelImporterSpecPrivate;
	for (const FString& CodeName : StringArray(Case, TEXT("error_codes")))
	{
		ECode Code;
		if (!CodeFromName(CodeName, Code))
		{
			AddError(FString::Printf(TEXT("[%s] unknown code '%s' in expectations"), *Id, *CodeName));
			continue;
		}
		TestTrue(*FString::Printf(TEXT("[%s] has Error %s; report:\n%s"), *Id, *CodeName, *R.Report.ToString()), HasMsg(R.Report, Code, ESev::Error));
	}
}

/** Imports one valid case into Dest/Name with CPU-only validation and compares everything (acceptance 1). */
void FPipelineModelImporterSpec::CheckImportedCase(const FString& Id, const FString& Name, int32 DeviceOverride)
{
	using namespace PipelineModelImporterSpecPrivate;
	TSharedPtr<FJsonObject> Root, Case;
	FString Folder;
	if (!LoadCase(Id, Root, Case, Folder)) { return; }

	FPipelineImportOptions O = Opts(Folder, Name);
	if (DeviceOverride >= 0) { O.PreferredDevice = (EPipelineDevice)DeviceOverride; }
	const FPipelineImportResult R = Run(O);

	TestTrue(*FString::Printf(TEXT("[%s] bSuccess; report:\n%s"), *Id, *R.Report.ToString()), R.bSuccess);
	TestEqual(*FString::Printf(TEXT("[%s] error count"), *Id), R.Report.NumErrors(), 0);
	if (!R.bSuccess || !R.Package || !R.Model)
	{
		AddError(FString::Printf(TEXT("[%s] Package or Model is null"), *Id));
		return;
	}

	TestTrue(*FString::Printf(TEXT("[%s] Package->Model == Model"), *Id), R.Package->Model == R.Model);
	TestEqual(*FString::Printf(TEXT("[%s] package path"), *Id), R.Package->GetPathName(), ObjPath(Name));
	TestEqual(*FString::Printf(TEXT("[%s] model path"), *Id), R.Model->GetPathName(), ObjPath(Name + TEXT("_Model")));
	TestEqual(*FString::Printf(TEXT("[%s] package asset lives in package %s"), *Id, *PkgPath(Name)), R.Package->GetOutermost()->GetName(), PkgPath(Name));
	TestEqual(*FString::Printf(TEXT("[%s] model asset lives in package %s_Model"), *Id, *PkgPath(Name)), R.Model->GetOutermost()->GetName(), PkgPath(Name + TEXT("_Model")));
	for (const UObject* Obj : {(const UObject*)R.Package.Get(), (const UObject*)R.Model.Get()})
	{
		TestTrue(*FString::Printf(TEXT("[%s] %s is RF_Public|RF_Standalone"), *Id, *Obj->GetName()), Obj->HasAllFlags(RF_Public | RF_Standalone));
		TestFalse(*FString::Printf(TEXT("[%s] %s is not RF_Transient"), *Id, *Obj->GetName()), Obj->HasAnyFlags(RF_Transient));
	}

	TArray<FString> Fails;
	double Worst = 0.0;
	CheckPackageAgainstCase(R.Package, Case, Folder, DeviceOverride, Fails, Worst);
	for (const FString& F : Fails) { AddError(F); }
	TestTrue(*FString::Printf(TEXT("[%s] worst relative float diff %.3g <= 1e-6"), *Id, Worst), Worst <= 1e-6);

	// The model bytes are the .onnx file, byte for byte (catches truncation and a 32-bit size).
	FString ModelFile;
	Case->TryGetStringField(TEXT("model_file"), ModelFile);
	TArray64<uint8> FileBytes;
	if (!FFileHelper::LoadFileToArray(FileBytes, *(Folder / ModelFile)))
	{
		AddError(FString::Printf(TEXT("[%s] cannot read '%s'"), *Id, *(Folder / ModelFile)));
	}
	else
	{
		const TConstArrayView64<uint8> Data = R.Model->GetFileData();
		TestEqual(*FString::Printf(TEXT("[%s] GetFileData size == .onnx size"), *Id), (int64)Data.Num(), (int64)FileBytes.Num());
		if (Data.Num() == FileBytes.Num())
		{
			TestEqual(*FString::Printf(TEXT("[%s] model bytes identical"), *Id), FMemory::Memcmp(Data.GetData(), FileBytes.GetData(), FileBytes.Num()), 0);
		}
	}
	TestEqual(*FString::Printf(TEXT("[%s] model file type"), *Id), R.Model->GetFileType(), FString(TEXT("onnx")));

	// Warnings: exactly the expected set (CPU-only validation adds none of its own for these models).
	const TArray<FString> Warns = StringArray(Case, TEXT("warning_codes"));
	for (const FString& W : Warns)
	{
		ECode Code;
		if (!CodeFromName(W, Code)) { AddError(FString::Printf(TEXT("[%s] unknown warning code '%s'"), *Id, *W)); continue; }
		TestTrue(*FString::Printf(TEXT("[%s] has Warning %s; report:\n%s"), *Id, *W, *R.Report.ToString()), HasMsg(R.Report, Code, ESev::Warning));
	}
	TestEqual(*FString::Printf(TEXT("[%s] warning count; report:\n%s"), *Id, *R.Report.ToString()), R.Report.NumWarnings(), Warns.Num());

	// bSave=false: no file under the project's Content dir.
	TArray<FString> Files;
	IFileManager::Get().FindFilesRecursive(Files, *ContentDir(), TEXT("*.*"), true, false);
	TestEqual(*FString::Printf(TEXT("[%s] files written under %s"), *Id, *ContentDir()), Files.Num(), 0);
}

void FPipelineModelImporterSpec::Define()
{
	using namespace PipelineModelImporterSpecPrivate;

	BeforeEach([this]()
	{
		DestGuid = FGuid::NewGuid().ToString(EGuidFormats::Digits);
		Dest = TEXT("/Game/PipelinesTests/") + DestGuid;
	});

	// Every test discards what it imported, including after a failed check (checks only AddError, they never skip this).
	AfterEach([this]()
	{
		for (const FPipelineImportResult& R : Imported)
		{
			DiscardImported(R);
		}
		Imported.Reset();
		for (const TWeakObjectPtr<UObject>& O : Manual)
		{
			DiscardObject(O.Get());
		}
		Manual.Reset();
		for (const FString& D : TempDirs)
		{
			IFileManager::Get().DeleteDirectory(*D, false, true);
		}
		TempDirs.Reset();
		// Freshly saved files can stay open briefly (asset registry / directory watcher), so retry the removal.
		bool bRemoved = !IFileManager::Get().DirectoryExists(*ContentDir());
		for (int32 Try = 0; Try < 20 && !bRemoved; ++Try)
		{
			bRemoved = IFileManager::Get().DeleteDirectory(*ContentDir(), false, true);
			if (!bRemoved) { FPlatformProcess::Sleep(0.05f); }
		}
		if (!bRemoved) { AddWarning(FString::Printf(TEXT("could not remove %s"), *ContentDir())); }
		const FString TestsRoot = FPaths::ProjectContentDir() / TEXT("PipelinesTests");
		IFileManager::Get().DeleteDirectory(*TestsRoot, false, false);   // only if empty
	});

	// ------------------------------------------------------------------------------------------------ acceptance 1
	Describe("ValidModels", [this]()
	{
		// A fixed list so each model is its own test line. Catches an importer that only fills some fields, forgets
		// the RF flags, leaves the objects in the transient package, or copies a truncated model.
		static const TCHAR* Ids[] = {
			TEXT("tiny_text_embedding"), TEXT("tiny_text_classification"), TEXT("tiny_image_classification"),
			TEXT("tiny_object_detection"), TEXT("sidecar_task"), TEXT("readme_license"), TEXT("no_license")};
		for (const TCHAR* Id : Ids)
		{
			It(FString::Printf(TEXT("imports %s with every field equal to the expectations"), Id), [this, Id]()
			{
				CheckImportedCase(Id, FString::Printf(TEXT("Imp_%s"), Id));
			});
		}

		// Catches warnings being dropped from the report: no_license must carry its 3 license warnings, and the
		// import still succeeds (license problems warn, never block).
		It("keeps the three license warnings of no_license in the report", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("no_license"), Root, Case, Folder)) { return; }
			const FPipelineImportResult R = Run(Opts(Folder, TEXT("NoLic")));
			TestTrue(*R.Report.ToString(), R.bSuccess);
			TestTrue(TEXT("LicenseSpdxMissing"), HasMsg(R.Report, ECode::LicenseSpdxMissing, ESev::Warning));
			TestTrue(TEXT("LicenseSourceMissing"), HasMsg(R.Report, ECode::LicenseSourceMissing, ESev::Warning));
			TestTrue(TEXT("LicenseAttributionMissing"), HasMsg(R.Report, ECode::LicenseAttributionMissing, ESev::Warning));
			if (R.Package)
			{
				TestTrue(TEXT("License fields empty in the package"), R.Package->License.SpdxId.IsEmpty() && R.Package->License.SourceUrl.IsEmpty() && R.Package->License.Attribution.IsEmpty());
			}
		});

		// Catches the report order contract: parse/settings messages before validator messages before DeviceChecks.
		// tiny_text_embedding validates on CPU, so the CPU DeviceCheck must be there and have run.
		It("returns the CPU DeviceCheck of the validator in the report", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_text_embedding"), Root, Case, Folder)) { return; }
			const FPipelineImportResult R = Run(Opts(Folder, TEXT("Dev")));
			TestTrue(*R.Report.ToString(), R.bSuccess);
			const FPipelineDeviceCheck* Cpu = R.Report.DeviceChecks.FindByPredicate([](const FPipelineDeviceCheck& C) { return C.Device == EPipelineDevice::CPU; });
			if (!Cpu) { AddError(TEXT("no CPU DeviceCheck in the import report")); return; }
			TestTrue(TEXT("CPU bCreated"), Cpu->bCreated);
			TestTrue(TEXT("CPU bRan"), Cpu->bRan);
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 2
	Describe("DefaultName", [this]()
	{
		// Catches an importer that requires AssetName or uses the full path as the name.
		It("uses the sanitized folder name when AssetName is empty (tiny_object_detection)", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_object_detection"), Root, Case, Folder)) { return; }
			const FPipelineImportResult R = Run(Opts(Folder, FString()));
			TestTrue(*R.Report.ToString(), R.bSuccess);
			if (!R.Package || !R.Model) { AddError(TEXT("Package/Model null")); return; }
			TestEqual(TEXT("package path"), R.Package->GetPathName(), ObjPath(TEXT("tiny_object_detection")));
			TestEqual(TEXT("model path"), R.Model->GetPathName(), ObjPath(TEXT("tiny_object_detection_Model")));
		});

		// Catches no sanitizing: a folder called "my model.v2" has a space and a dot, both invalid in object names.
		It("sanitizes a folder named 'my model.v2' with ObjectTools::SanitizeObjectName", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			const FString Leaf = TEXT("my model.v2");
			const FString Copy = NewTempDir(Leaf);
			if (!FPlatformFileManager::Get().GetPlatformFile().CopyDirectoryTree(*Copy, *Folder, true))
			{
				AddError(FString::Printf(TEXT("could not copy '%s' to '%s'"), *Folder, *Copy));
				return;
			}
			const FString Expected = ObjectTools::SanitizeObjectName(Leaf);
			TestFalse(TEXT("sanitized name is non-empty"), Expected.IsEmpty());
			TestFalse(TEXT("sanitized name differs from the raw leaf"), Expected == Leaf);
			const FPipelineImportResult R = Run(Opts(Copy, FString()));
			TestTrue(*R.Report.ToString(), R.bSuccess);
			if (!R.Package || !R.Model) { AddError(TEXT("Package/Model null")); return; }
			TestEqual(TEXT("package path"), R.Package->GetPathName(), ObjPath(Expected));
			TestEqual(TEXT("model path"), R.Model->GetPathName(), ObjPath(Expected + TEXT("_Model")));
		});

		// Catches a name made only of invalid characters being rejected: it sanitizes to underscores and is legal.
		It("accepts a folder whose name sanitizes to underscores only ('!!!')", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			const FString Copy = NewTempDir(TEXT("!!!"));
			if (!FPlatformFileManager::Get().GetPlatformFile().CopyDirectoryTree(*Copy, *Folder, true))
			{
				AddError(TEXT("could not copy the test folder"));
				return;
			}
			const FString Expected = ObjectTools::SanitizeObjectName(TEXT("!!!"));
			TestEqual(TEXT("'!!!' sanitizes to '___'"), Expected, FString(TEXT("___")));
			const FPipelineImportResult R = Run(Opts(Copy, FString()));
			TestTrue(*R.Report.ToString(), R.bSuccess);
			if (R.Package) { TestEqual(TEXT("package path"), R.Package->GetPathName(), ObjPath(Expected)); }
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 3
	Describe("ValidationFailures", [this]()
	{
		static const TCHAR* Ids[] = {TEXT("wrong_io_for_task"), TEXT("label_mismatch"), TEXT("unknown_op"), TEXT("missing_preprocessor")};
		for (const TCHAR* Id : Ids)
		{
			// Catches a failed import leaving assets in the project (Noah's rule) or returning half-built objects.
			It(FString::Printf(TEXT("fails %s with the listed errors and creates nothing"), Id), [this, Id]()
			{
				// ORT logs its own errors when it rejects a model (unknown_op); those are expected.
				AddExpectedError(TEXT("ONNX Runtime"), EAutomationExpectedErrorFlags::Contains, -1);
				AddExpectedError(TEXT("Session creation failed"), EAutomationExpectedErrorFlags::Contains, -1);
				TSharedPtr<FJsonObject> Root, Case;
				FString Folder;
				if (!LoadCase(Id, Root, Case, Folder)) { return; }
				const FString Name = FString::Printf(TEXT("Bad_%s"), Id);
				const FPipelineImportResult R = Run(Opts(Folder, Name));
				CheckFailedClean(R, Name, Id);
				ExpectErrorsFromCase(R, Case, Id);
				// Warnings from the folder parse survive in the report (missing_preprocessor).
				for (const FString& W : StringArray(Case, TEXT("warning_codes")))
				{
					ECode Code;
					if (CodeFromName(W, Code))
					{
						TestTrue(*FString::Printf(TEXT("[%s] keeps Warning %s; report:\n%s"), Id, *W, *R.Report.ToString()), HasMsg(R.Report, Code, ESev::Warning));
					}
				}
			});
		}

		// Catches the report order contract: parse messages come before validator messages.
		It("orders parse warnings before validator errors (missing_preprocessor)", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("missing_preprocessor"), Root, Case, Folder)) { return; }
			const FPipelineImportResult R = Run(Opts(Folder, TEXT("Order")));
			const int32 Parse = IndexOfCode(R.Report, ECode::MissingPreprocessorConfig);
			const int32 Valid = IndexOfCode(R.Report, ECode::InputDimMismatch);
			TestTrue(*FString::Printf(TEXT("both present (parse idx %d, validator idx %d); report:\n%s"), Parse, Valid, *R.Report.ToString()), Parse != INDEX_NONE && Valid != INDEX_NONE);
			TestTrue(TEXT("MissingPreprocessorConfig precedes InputDimMismatch"), Parse < Valid);
		});

		// Catches validation being skipped when bValidate is on but only settings are wrong: wrong_io must still fail
		// even though its package settings are fine (the I/O check is what catches it).
		It("leaves the failed model's transient copies out of the asset registry paths (FindPackage null)", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("wrong_io_for_task"), Root, Case, Folder)) { return; }
			const FPipelineImportResult R = Run(Opts(Folder, TEXT("Gone")));
			CheckFailedClean(R, TEXT("Gone"), TEXT("wrong_io_for_task"));
			TArray<FString> Files;
			IFileManager::Get().FindFilesRecursive(Files, *ContentDir(), TEXT("*.*"), true, false);
			TestEqual(TEXT("no file in Content"), Files.Num(), 0);
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 4
	Describe("ParseErrors", [this]()
	{
		static const TCHAR* Ids[] = {TEXT("missing_model"), TEXT("bad_config_json"), TEXT("no_task"), TEXT("bad_id2label")};
		for (const TCHAR* Id : Ids)
		{
			// Catches the importer carrying on after a folder parse error (e.g. reading bytes of a missing model).
			It(FString::Printf(TEXT("fails %s with the parse code and creates nothing"), Id), [this, Id]()
			{
				TSharedPtr<FJsonObject> Root, Case;
				FString Folder;
				if (!LoadCase(Id, Root, Case, Folder)) { return; }
				const FString Name = FString::Printf(TEXT("Parse_%s"), Id);
				const FPipelineImportResult R = Run(Opts(Folder, Name));
				CheckFailedClean(R, Name, Id);
				ExpectErrorsFromCase(R, Case, Id);
			});
		}

		// Catches an exception/crash or a success on a nonexistent or empty source folder.
		It("fails with FolderNotFound for a missing and an empty SourceFolder", [this]()
		{
			const FPipelineImportResult A = Run(Opts(FPaths::ProjectSavedDir() / TEXT("PipelinesTests/no_such_folder_") + DestGuid, TEXT("Nope")));
			CheckFailedClean(A, TEXT("Nope"), TEXT("missing folder"));
			TestTrue(*A.Report.ToString(), HasMsg(A.Report, ECode::FolderNotFound, ESev::Error));
			const FPipelineImportResult B = Run(Opts(FString(), TEXT("Nope")));
			CheckFailedClean(B, TEXT("Nope"), TEXT("empty folder"));
			TestTrue(*B.Report.ToString(), HasMsg(B.Report, ECode::FolderNotFound, ESev::Error));
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 5
	Describe("Existing", [this]()
	{
		// Catches a second import replacing or modifying the first: same pointers, same fields, AssetExists.
		It("fails the second import to the same path with AssetExists and leaves the first untouched", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case, Case2;
			FString Folder, Folder2;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			if (!LoadCase(TEXT("no_license"), Root, Case2, Folder2)) { return; }
			const FPipelineImportResult A = Run(Opts(Folder, TEXT("Twice")));
			if (!A.bSuccess || !A.Package || !A.Model) { AddError(FString::Printf(TEXT("first import failed: %s"), *A.Report.ToString())); return; }
			const int64 ModelBytes = A.Model->GetFileData().Num();
			const FString Spdx = A.Package->License.SpdxId;
			const TArray<FString> Labels = A.Package->Labels;

			const FPipelineImportResult B = Run(Opts(Folder, TEXT("Twice")));
			TestFalse(TEXT("same folder again: bSuccess false"), B.bSuccess);
			TestTrue(*B.Report.ToString(), HasMsg(B.Report, ECode::AssetExists, ESev::Error));
			// A different folder to the same name must not overwrite either.
			const FPipelineImportResult C = Run(Opts(Folder2, TEXT("Twice")));
			TestFalse(TEXT("other folder, same name: bSuccess false"), C.bSuccess);
			TestTrue(*C.Report.ToString(), HasMsg(C.Report, ECode::AssetExists, ESev::Error));

			TestTrue(TEXT("package pointer unchanged"), FindObject<UPipelineModelPackage>(nullptr, *ObjPath(TEXT("Twice"))) == A.Package.Get());
			TestTrue(TEXT("model pointer unchanged"), FindObject<UNNEModelData>(nullptr, *ObjPath(TEXT("Twice_Model"))) == A.Model.Get());
			TestTrue(TEXT("Package->Model unchanged"), A.Package->Model == A.Model);
			TestEqual(TEXT("license unchanged"), A.Package->License.SpdxId, Spdx);
			TestTrue(TEXT("labels unchanged"), A.Package->Labels == Labels);
			TestEqual(TEXT("model bytes unchanged"), (int64)A.Model->GetFileData().Num(), ModelBytes);
		});

		// Catches overwrite creating new objects (stale references in the project) or keeping the old fields.
		It("overwrites in place: same pointers, fields from the new folder", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case, Case2;
			FString Folder, Folder2;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			if (!LoadCase(TEXT("no_license"), Root, Case2, Folder2)) { return; }
			const FPipelineImportResult A = Run(Opts(Folder, TEXT("X")));
			if (!A.bSuccess || !A.Package || !A.Model) { AddError(FString::Printf(TEXT("first import failed: %s"), *A.Report.ToString())); return; }
			TestEqual(TEXT("first import has the license"), A.Package->License.SpdxId, FString(TEXT("Apache-2.0")));
			UPipelineModelPackage* P1 = A.Package;
			UNNEModelData* M1 = A.Model;

			FPipelineImportOptions O = Opts(Folder2, TEXT("X"));
			O.bOverwriteExisting = true;
			const FPipelineImportResult B = Run(O);
			TestTrue(*B.Report.ToString(), B.bSuccess);
			TestTrue(TEXT("same Package pointer"), B.Package.Get() == P1);
			TestTrue(TEXT("same Model pointer"), B.Model.Get() == M1);
			if (!B.Package || !B.Model) { return; }
			TestTrue(TEXT("Package->Model still the model asset"), B.Package->Model == M1);
			TestTrue(TEXT("License.SpdxId now empty"), B.Package->License.SpdxId.IsEmpty());
			TestTrue(TEXT("License.SourceUrl now empty"), B.Package->License.SourceUrl.IsEmpty());
			TestTrue(TEXT("License.Attribution now empty"), B.Package->License.Attribution.IsEmpty());
			TArray<FString> Fails;
			double Worst = 0;
			CheckPackageAgainstCase(B.Package, Case2, Folder2, -1, Fails, Worst);
			for (const FString& F : Fails) { AddError(F); }
			// Model bytes come from the new folder.
			const int64 Expected = IFileManager::Get().FileSize(*(Folder2 / TEXT("model.onnx")));
			TestEqual(TEXT("model bytes from the new folder"), (int64)B.Model->GetFileData().Num(), Expected);
			TestTrue(TEXT("still public+standalone"), B.Package->HasAllFlags(RF_Public | RF_Standalone) && B.Model->HasAllFlags(RF_Public | RF_Standalone));
		});

		// Catches overwrite deleting/replacing things it should not: a failed validation of the NEW folder must
		// leave the old assets exactly as they were.
		It("keeps the existing assets when an overwrite fails validation", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case, Bad;
			FString Folder, BadFolder;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			if (!LoadCase(TEXT("wrong_io_for_task"), Root, Bad, BadFolder)) { return; }
			const FPipelineImportResult A = Run(Opts(Folder, TEXT("Keep")));
			if (!A.bSuccess || !A.Package || !A.Model) { AddError(FString::Printf(TEXT("first import failed: %s"), *A.Report.ToString())); return; }
			const TArray<FString> Labels = A.Package->Labels;
			const int64 Bytes = A.Model->GetFileData().Num();
			FPipelineImportOptions O = Opts(BadFolder, TEXT("Keep"));
			O.bOverwriteExisting = true;
			const FPipelineImportResult B = Run(O);
			TestFalse(*B.Report.ToString(), B.bSuccess);
			TestTrue(TEXT("package still found"), FindObject<UPipelineModelPackage>(nullptr, *ObjPath(TEXT("Keep"))) == A.Package.Get());
			TestTrue(TEXT("labels unchanged"), A.Package->Labels == Labels);
			TestEqual(TEXT("model bytes unchanged"), (int64)A.Model->GetFileData().Num(), Bytes);
			TestTrue(TEXT("Package->Model unchanged"), A.Package->Model == A.Model);
		});

		// Catches overwrite replacing an object of another type or half-existing pairs: never delete/replace foreign
		// objects. (a) only the model asset exists, (b) both exist but the package slot holds a UNNEModelData.
		It("reports AssetExists even with bOverwriteExisting when only one asset exists or a class differs", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			auto MakeModel = [this](const FString& Name) -> UNNEModelData*
			{
				UPackage* P = CreatePackage(*PkgPath(Name));
				UNNEModelData* M = NewObject<UNNEModelData>(P, *Name, RF_Public | RF_Standalone);
				Manual.Add(M);
				return M;
			};
			// (a) only Z_Model exists
			UNNEModelData* Only = MakeModel(TEXT("Z_Model"));
			FPipelineImportOptions Oa = Opts(Folder, TEXT("Z"));
			Oa.bOverwriteExisting = true;
			const FPipelineImportResult A = Run(Oa);
			TestFalse(*A.Report.ToString(), A.bSuccess);
			TestTrue(*A.Report.ToString(), HasMsg(A.Report, ECode::AssetExists, ESev::Error));
			TestTrue(TEXT("(a) the existing object is still there"), FindObject<UNNEModelData>(nullptr, *ObjPath(TEXT("Z_Model"))) == Only);
			TestTrue(TEXT("(a) no package asset was created"), FindObject<UObject>(nullptr, *ObjPath(TEXT("Z"))) == nullptr);

			// (b) V is a UNNEModelData, V_Model too
			UNNEModelData* WrongClass = MakeModel(TEXT("V"));
			MakeModel(TEXT("V_Model"));
			FPipelineImportOptions Ob = Opts(Folder, TEXT("V"));
			Ob.bOverwriteExisting = true;
			const FPipelineImportResult B = Run(Ob);
			TestFalse(*B.Report.ToString(), B.bSuccess);
			TestTrue(*B.Report.ToString(), HasMsg(B.Report, ECode::AssetExists, ESev::Error));
			TestTrue(TEXT("(b) the foreign-class object is untouched"), FindObject<UObject>(nullptr, *ObjPath(TEXT("V"))) == WrongClass);
			TestTrue(TEXT("(b) it is still a UNNEModelData"), WrongClass->IsA<UNNEModelData>() && IsValid(WrongClass));
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 6
	Describe("Destination", [this]()
	{
		// Catches a missing IsValidLongPackageName check: both must report InvalidDestination and create nothing.
		It("reports InvalidDestination for 'NotAPath' and '/Game/bad path!'", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			for (const TCHAR* Bad : {TEXT("NotAPath"), TEXT("/Game/bad path!")})
			{
				FPipelineImportOptions O = Opts(Folder, TEXT("Dst"));
				O.DestinationPath = Bad;
				const FPipelineImportResult R = Run(O);
				TestFalse(*FString::Printf(TEXT("'%s': bSuccess false"), Bad), R.bSuccess);
				TestTrue(*FString::Printf(TEXT("'%s': InvalidDestination; report:\n%s"), Bad, *R.Report.ToString()), HasMsg(R.Report, ECode::InvalidDestination, ESev::Error));
				TestTrue(*FString::Printf(TEXT("'%s': Package null"), Bad), R.Package == nullptr);
				TestTrue(*FString::Printf(TEXT("'%s': Model null"), Bad), R.Model == nullptr);
				TestTrue(*FString::Printf(TEXT("'%s': nothing created at the default name"), Bad), FindObject<UObject>(nullptr, *(FString(Bad) / TEXT("Dst.Dst"))) == nullptr);
			}
		});

		// Catches AssetName being used unchecked: a name with a space and '!' does not form a valid package name.
		It("reports InvalidDestination for an AssetName that is not a valid package name", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			const FPipelineImportResult R = Run(Opts(Folder, TEXT("bad name!")));
			TestFalse(*R.Report.ToString(), R.bSuccess);
			TestTrue(*R.Report.ToString(), HasMsg(R.Report, ECode::InvalidDestination, ESev::Error));
			TestTrue(TEXT("Package null"), R.Package == nullptr);
			TestTrue(TEXT("Model null"), R.Model == nullptr);
		});

		// Catches InvalidDestination being checked after the (expensive) model read or not at all: with a good
		// folder and a bad destination, the error is reported and no model was validated (no DeviceChecks).
		It("rejects the destination before validating anything", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			FPipelineImportOptions O = Opts(Folder, TEXT("Dst"));
			O.DestinationPath = TEXT("NotAPath");
			const FPipelineImportResult R = Run(O);
			TestEqual(TEXT("no DeviceChecks (validator never ran)"), R.Report.DeviceChecks.Num(), 0);
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 7
	Describe("PreferredDevice", [this]()
	{
		// Catches Options.PreferredDevice being ignored or Auto overriding the sidecar.
		It("takes the sidecar CPU with Options Auto (tiny_image_classification)", [this]()
		{
			CheckImportedCase(TEXT("tiny_image_classification"), TEXT("DevAuto"));
		});

		It("takes the Options value GPU over the sidecar CPU", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_image_classification"), Root, Case, Folder)) { return; }
			FPipelineImportOptions O = Opts(Folder, TEXT("DevGpu"));
			O.PreferredDevice = EPipelineDevice::GPU;
			const FPipelineImportResult R = Run(O);
			TestTrue(*R.Report.ToString(), R.bSuccess);
			if (R.Package) { TestEqual(TEXT("PreferredDevice"), (int32)R.Package->PreferredDevice, (int32)EPipelineDevice::GPU); }
		});

		// Catches the override working only in one direction: sidecar-less folder (Auto) with Options CPU.
		It("takes the Options value CPU over an Auto folder (tiny_text_embedding)", [this]()
		{
			CheckImportedCase(TEXT("tiny_text_embedding"), TEXT("DevCpu"), (int32)EPipelineDevice::CPU);
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 8
	Describe("TaskOverride", [this]()
	{
		// Catches Options.Task being ignored: forcing TextEmbedding on an image folder fails at the tokenizer check.
		It("fails tiny_image_classification with Task=TextEmbedding (MissingTokenizer) and creates nothing", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_image_classification"), Root, Case, Folder)) { return; }
			FPipelineImportOptions O = Opts(Folder, TEXT("Forced"));
			O.Task = EPipelineTask::TextEmbedding;
			const FPipelineImportResult R = Run(O);
			CheckFailedClean(R, TEXT("Forced"), TEXT("task override"));
			TestTrue(*R.Report.ToString(), HasMsg(R.Report, ECode::MissingTokenizer, ESev::Error));
		});

		// Catches an override that is applied to parsing but dropped from the package: same task as detected is a no-op.
		It("keeps the detected task when Task equals it", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_image_classification"), Root, Case, Folder)) { return; }
			FPipelineImportOptions O = Opts(Folder, TEXT("Same"));
			O.Task = EPipelineTask::ImageClassification;
			const FPipelineImportResult R = Run(O);
			TestTrue(*R.Report.ToString(), R.bSuccess);
			if (R.Package) { TestEqual(TEXT("Task"), (int32)R.Package->Task, (int32)EPipelineTask::ImageClassification); }
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 9
	Describe("NoValidate", [this]()
	{
		// Catches bValidate=false still running the validator (the wrong-I/O model must import) or skipping settings.
		It("imports wrong_io_for_task when bValidate is false", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("wrong_io_for_task"), Root, Case, Folder)) { return; }
			FPipelineImportOptions O = Opts(Folder, TEXT("NoVal"));
			O.bValidate = false;
			const FPipelineImportResult R = Run(O);
			TestTrue(*R.Report.ToString(), R.bSuccess);
			TestFalse(TEXT("no MissingInput (validator did not run)"), R.Report.HasCode(ECode::MissingInput));
			TestEqual(TEXT("no DeviceChecks"), R.Report.DeviceChecks.Num(), 0);
			if (!R.Package || !R.Model) { AddError(TEXT("Package/Model null")); return; }
			TestTrue(TEXT("package exists at its path"), FindObject<UPipelineModelPackage>(nullptr, *ObjPath(TEXT("NoVal"))) == R.Package.Get());
			TestTrue(TEXT("model exists at its path"), FindObject<UNNEModelData>(nullptr, *ObjPath(TEXT("NoVal_Model"))) == R.Model.Get());
		});

		// Catches ValidateSettings being gated by bValidate: settings are always checked, so the license warnings stay.
		It("still runs the settings check with bValidate false (no_license warnings)", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("no_license"), Root, Case, Folder)) { return; }
			FPipelineImportOptions O = Opts(Folder, TEXT("NoVal2"));
			O.bValidate = false;
			const FPipelineImportResult R = Run(O);
			TestTrue(*R.Report.ToString(), R.bSuccess);
			TestTrue(TEXT("LicenseSpdxMissing"), HasMsg(R.Report, ECode::LicenseSpdxMissing, ESev::Warning));
			TestEqual(TEXT("exactly the 3 license warnings"), R.Report.NumWarnings(), 3);
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 10
	Describe("Save", [this]()
	{
		// Catches bSave being ignored, files at the wrong path, a package left dirty after saving, and the
		// "exists on disk" check (a re-import after the objects are gone must still report AssetExists).
		It("writes both .uasset files with bSave and sees them on disk afterwards", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_object_detection"), Root, Case, Folder)) { return; }
			FPipelineImportOptions O = Opts(Folder, TEXT("Saved"));
			O.bSave = true;
			const FPipelineImportResult R = Run(O);
			const FString Ext = FPackageName::GetAssetPackageExtension();
			const FString PkgFile = FPackageName::LongPackageNameToFilename(PkgPath(TEXT("Saved")), Ext);
			const FString ModelFile = FPackageName::LongPackageNameToFilename(PkgPath(TEXT("Saved_Model")), Ext);
			TestTrue(*R.Report.ToString(), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("package file exists: %s"), *PkgFile), IFileManager::Get().FileExists(*PkgFile));
			TestTrue(*FString::Printf(TEXT("model file exists: %s"), *ModelFile), IFileManager::Get().FileExists(*ModelFile));
			TestTrue(TEXT("model file is not empty"), IFileManager::Get().FileSize(*ModelFile) > 0);
			if (R.Package && R.Model)
			{
				TestFalse(TEXT("package asset's package is not dirty after the save"), R.Package->GetOutermost()->IsDirty());
				TestFalse(TEXT("model asset's package is not dirty after the save"), R.Model->GetOutermost()->IsDirty());
			}

			// Drop the objects; the files on disk must still make the destination count as taken.
			DiscardImported(R);
			FPipelineImportOptions Again = Opts(Folder, TEXT("Saved"));
			const FPipelineImportResult R2 = Run(Again);
			TestFalse(TEXT("re-import onto files on disk: bSuccess false"), R2.bSuccess);
			TestTrue(*R2.Report.ToString(), HasMsg(R2.Report, ECode::AssetExists, ESev::Error));

			// Clean up the files.
			IFileManager::Get().Delete(*PkgFile, false, true);
			IFileManager::Get().Delete(*ModelFile, false, true);
			TestFalse(TEXT("package file deleted"), IFileManager::Get().FileExists(*PkgFile));
			TestFalse(TEXT("model file deleted"), IFileManager::Get().FileExists(*ModelFile));
		});
	});

	// ------------------------------------------------------------------------------------------------ edges
	Describe("Threading", [this]()
	{
		// Catches an importer that touches UObjects off the game thread: the call must refuse, not run.
		It("refuses to run off the game thread with one Error and touches nothing", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			const FPipelineImportOptions O = Opts(Folder, TEXT("OffThread"));
			TFuture<FPipelineImportResult> Future = Async(EAsyncExecution::ThreadPool, [O]() { return ImportPipelineModelFolder(O); });
			if (!Future.WaitFor(FTimespan::FromSeconds(60)))
			{
				AddError(TEXT("the off-thread import did not return within 60 s (hang)"));
				return;
			}
			const FPipelineImportResult R = Future.Get();
			Imported.Add(R);
			TestFalse(TEXT("bSuccess false"), R.bSuccess);
			TestEqual(*FString::Printf(TEXT("exactly one Error; report:\n%s"), *R.Report.ToString()), R.Report.NumErrors(), 1);
			TestTrue(TEXT("the Error has code None"), HasMsg(R.Report, ECode::None, ESev::Error));
			bool bMentionsThread = false;
			for (const FPipelineValidationMessage& M : R.Report.Messages) { bMentionsThread |= M.Message.Contains(TEXT("game thread")); }
			TestTrue(TEXT("the message says 'game thread'"), bMentionsThread);
			TestTrue(TEXT("Package null"), R.Package == nullptr);
			TestTrue(TEXT("Model null"), R.Model == nullptr);
			NothingExistsFor(TEXT("OffThread"));
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 12
	Describe("Library", [this]()
	{
		// Catches a library wrapper that drops options or builds a different result than the free function.
		It("ImportModelFolder gives the same result as ImportPipelineModelFolder", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			const FPipelineImportResult Free = Run(Opts(Folder, TEXT("LibFree")));
			const FPipelineImportResult Lib = Imported.Add_GetRef(UPipelineEditorLibrary::ImportModelFolder(Opts(Folder, TEXT("LibCall"))));
			TestTrue(*Free.Report.ToString(), Free.bSuccess);
			TestEqual(TEXT("bSuccess equal"), Lib.bSuccess, Free.bSuccess);
			TestTrue(*Lib.Report.ToString(), Lib.bSuccess && Lib.Package && Lib.Model);
			TestEqual(TEXT("report text equal"), Lib.Report.ToString(), Free.Report.ToString());
			if (Lib.Package && Free.Package)
			{
				TestEqual(TEXT("package path uses the library's name"), Lib.Package->GetPathName(), ObjPath(TEXT("LibCall")));
				TestTrue(TEXT("labels equal"), Lib.Package->Labels == Free.Package->Labels);
				TestEqual(TEXT("task equal"), (int32)Lib.Package->Task, (int32)Free.Package->Task);
				TestEqual(TEXT("max tokens equal"), Lib.Package->Preprocess.MaxTokens, Free.Package->Preprocess.MaxTokens);
				TestEqual(TEXT("tokenizer equal"), Lib.Package->TokenizerJson, Free.Package->TokenizerJson);
			}
			// A failing import also matches.
			TSharedPtr<FJsonObject> Root2, Bad;
			FString BadFolder;
			if (!LoadCase(TEXT("wrong_io_for_task"), Root2, Bad, BadFolder)) { return; }
			const FPipelineImportResult FreeBad = Run(Opts(BadFolder, TEXT("LibBad")));
			const FPipelineImportResult LibBad = Imported.Add_GetRef(UPipelineEditorLibrary::ImportModelFolder(Opts(BadFolder, TEXT("LibBad"))));
			TestFalse(TEXT("both fail"), FreeBad.bSuccess || LibBad.bSuccess);
			TestEqual(TEXT("failing report text equal"), LibBad.Report.ToString(), FreeBad.Report.ToString());
		});

		// Catches a library validator that does not forward (different options or a null check of its own).
		It("ValidateModelPackage gives the same report as ValidatePipelineModelPackage", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("tiny_text_classification"), Root, Case, Folder)) { return; }
			const FPipelineImportResult R = Run(Opts(Folder, TEXT("LibVal")));
			if (!R.Package) { AddError(FString::Printf(TEXT("import failed: %s"), *R.Report.ToString())); return; }
			FPipelineValidateOptions VO;
			VO.bCheckGpu = false;
			const FPipelineValidationReport Direct = ValidatePipelineModelPackage(R.Package, VO);
			const FPipelineValidationReport ViaLib = UPipelineEditorLibrary::ValidateModelPackage(R.Package, VO);
			TestEqual(TEXT("report text equal"), ViaLib.ToString(), Direct.ToString());
			TestEqual(TEXT("DeviceChecks count equal"), ViaLib.DeviceChecks.Num(), Direct.DeviceChecks.Num());
			TestTrue(*Direct.ToString(), Direct.IsValid());

			// Null package: the validator's NoPackage error comes through the library unchanged.
			const FPipelineValidationReport NullDirect = ValidatePipelineModelPackage(nullptr, VO);
			const FPipelineValidationReport NullLib = UPipelineEditorLibrary::ValidateModelPackage(nullptr, VO);
			TestTrue(*NullLib.ToString(), NullLib.HasError(ECode::NoPackage));
			TestEqual(TEXT("null report text equal"), NullLib.ToString(), NullDirect.ToString());
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 11
	Describe("Reference", [this]()
	{
		static const TCHAR* Ids[] = {TEXT("all-MiniLM-L6-v2"), TEXT("distilbert-sst2"), TEXT("resnet-50"), TEXT("rtdetr_r18vd")};
		for (const TCHAR* Id : Ids)
		{
			// Catches the 4 acceptance models failing or warning with default options, fields differing, or the CPU/GPU
			// smoke runs not happening. GPU is required (MODELPIPELINES_REQUIRE_GPU=1); otherwise it is reported skipped.
			It(FString::Printf(TEXT("imports %s with 0 errors, 0 warnings and CPU+GPU smoke runs"), Id), [this, Id]()
			{
				TSharedPtr<FJsonObject> Root, Case;
				FString Folder;
				if (!LoadCase(Id, Root, Case, Folder)) { return; }
				FPipelineImportOptions O = Opts(Folder, TEXT("Ref"), /*bCheckGpu*/ true); // the rest default: auto task, validation on
				const FPipelineImportResult R = Run(O);
				TestTrue(*FString::Printf(TEXT("[%s] bSuccess; report:\n%s"), Id, *R.Report.ToString()), R.bSuccess);
				TestEqual(*FString::Printf(TEXT("[%s] errors"), Id), R.Report.NumErrors(), 0);
				if (!R.Package || !R.Model) { AddError(FString::Printf(TEXT("[%s] Package or Model null"), Id)); return; }
				TestTrue(TEXT("Package->Model == Model"), R.Package->Model == R.Model);
				TestEqual(TEXT("package path"), R.Package->GetPathName(), ObjPath(TEXT("Ref")));
				TestEqual(TEXT("model path"), R.Model->GetPathName(), ObjPath(TEXT("Ref_Model")));

				TArray<FString> Fails;
				double Worst = 0;
				CheckPackageAgainstCase(R.Package, Case, Folder, -1, Fails, Worst);
				for (const FString& F : Fails) { AddError(F); }
				TestTrue(*FString::Printf(TEXT("[%s] worst relative float diff %.3g <= 1e-6"), Id, Worst), Worst <= 1e-6);
				TestEqual(*FString::Printf(TEXT("[%s] model bytes == .onnx size"), Id), (int64)R.Model->GetFileData().Num(), IFileManager::Get().FileSize(*(Folder / TEXT("model.onnx"))));

				const FPipelineDeviceCheck* Cpu = R.Report.DeviceChecks.FindByPredicate([](const FPipelineDeviceCheck& C) { return C.Device == EPipelineDevice::CPU; });
				const FPipelineDeviceCheck* Gpu = R.Report.DeviceChecks.FindByPredicate([](const FPipelineDeviceCheck& C) { return C.Device == EPipelineDevice::GPU; });
				if (!Cpu) { AddError(FString::Printf(TEXT("[%s] no CPU DeviceCheck"), Id)); }
				else
				{
					TestTrue(*FString::Printf(TEXT("[%s] CPU created and ran: %s"), Id, *Cpu->Message), Cpu->bCreated && Cpu->bRan);
				}
				const bool bGpuRan = Gpu && Gpu->bCreated && Gpu->bRan;
				if (!bGpuRan)
				{
					if (IsGpuRequired())
					{
						AddError(FString::Printf(TEXT("[%s] GPU smoke run did not happen but MODELPIPELINES_REQUIRE_GPU=1 (%s)"), Id, Gpu ? *Gpu->Message : TEXT("no GPU DeviceCheck")));
					}
					else
					{
						AddWarning(FString::Printf(TEXT("[%s] SKIPPED GPU assertions: GPU smoke run unavailable (%s). Set MODELPIPELINES_REQUIRE_GPU=1 to make this a failure."), Id, Gpu ? *Gpu->Message : TEXT("no GPU DeviceCheck")));
						return; // the 0-warning claim only holds when the GPU leg ran
					}
				}
				TestEqual(*FString::Printf(TEXT("[%s] warnings (must be 0); report:\n%s"), Id, *R.Report.ToString()), R.Report.NumWarnings(), 0);
			});
		}

		// Catches a model that is wrong for its task being imported: resnet-50 forced to ObjectDetection must fail
		// with the listed errors and leave nothing in the project.
		It("fails resnet_as_detection with MissingOutput + OutputRankMismatch and creates nothing", [this]()
		{
			TSharedPtr<FJsonObject> Root, Case;
			FString Folder;
			if (!LoadCase(TEXT("resnet_as_detection"), Root, Case, Folder)) { return; }
			FPipelineImportOptions O = Opts(Folder, TEXT("RefBad"), true);
			O.Task = OverrideOf(Case);
			TestEqual(TEXT("expectations say ObjectDetection override"), (int32)O.Task, (int32)EPipelineTask::ObjectDetection);
			const FPipelineImportResult R = Run(O);
			CheckFailedClean(R, TEXT("RefBad"), TEXT("resnet_as_detection"));
			ExpectErrorsFromCase(R, Case, TEXT("resnet_as_detection"));
			TestTrue(*R.Report.ToString(), HasMsg(R.Report, ECode::MissingOutput, ESev::Error) && HasMsg(R.Report, ECode::OutputRankMismatch, ESev::Error));
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
