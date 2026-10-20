// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "PipelineModelImporter.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "FileHelpers.h"
#include "Misc/FileHelper.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "Misc/ScopedSlowTask.h"
#include "NNEModelData.h"
#include "ObjectTools.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineEditorLog.h"
#include "PipelineModelFolder.h"
#include "PipelineModelValidator.h"

#define LOCTEXT_NAMESPACE "PipelineModelImporter"

namespace PipelineModelImporterPrivate
{
	using ESev = EPipelineValidationSeverity;
	using ECode = EPipelineValidationCode;

	/** Longest accepted long package name for either asset (FName asserts at NAME_SIZE = 1024; leave room for object paths). */
	static constexpr int32 MaxLongPackageNameLen = 200;

	/** Last path component of a folder path, trailing separators ignored. */
	static FString FolderLeafOf(const FString& Folder)
	{
		FString Normalized = Folder;
		FPaths::NormalizeDirectoryName(Normalized);
		while (Normalized.EndsWith(TEXT("/")))
		{
			Normalized.LeftChopInline(1);
		}
		return FPaths::GetCleanFilename(Normalized);
	}

	/** Copies every field of the folder info into a package (Model is set separately). */
	static void FillPackage(UPipelineModelPackage* Pkg, const FPipelineModelFolderInfo& Info, EPipelineDevice OptionsDevice, UNNEModelData* Model)
	{
		Pkg->Task = Info.Task;
		Pkg->Model = Model;
		Pkg->TokenizerJson = Info.TokenizerJson;
		Pkg->Labels = Info.Labels;
		Pkg->Preprocess = Info.Preprocess;
		Pkg->Postprocess = Info.Postprocess;
		Pkg->License = Info.License;
		Pkg->PreferredDevice = OptionsDevice != EPipelineDevice::Auto ? OptionsDevice : Info.PreferredDevice;
	}

	/** Drops a transient object built by the importer. */
	static void DropTransient(UObject* Obj)
	{
		if (Obj && IsValid(Obj))
		{
			Obj->MarkAsGarbage();
		}
	}

	/** Logs every report message and the summary line (Error severities at Warning on purpose, see the header). */
	static void LogResult(const FString& FolderLeaf, const FPipelineImportResult& Result)
	{
		for (const FPipelineValidationMessage& M : Result.Report.Messages)
		{
			const TCHAR* Sev = LexToString(M.Severity);
			const TCHAR* Code = LexToString(M.Code);
			if (M.Severity == ESev::Info)
			{
				UE_LOG(LogModelPipelinesEditor, Log, TEXT("Import '%s': %s %s: %s"), *FolderLeaf, Sev, Code, *M.Message);
			}
			else
			{
				UE_LOG(LogModelPipelinesEditor, Warning, TEXT("Import '%s': %s %s: %s"), *FolderLeaf, Sev, Code, *M.Message);
			}
		}
		if (Result.bSuccess && Result.Package)
		{
			UE_LOG(LogModelPipelinesEditor, Log, TEXT("Imported '%s' as %s (%s)"), *FolderLeaf, *Result.Package->GetOutermost()->GetName(),
				LexToString(Result.Package->Task));
		}
		else
		{
			UE_LOG(LogModelPipelinesEditor, Warning, TEXT("Import of '%s' failed: %d errors"), *FolderLeaf, Result.Report.NumErrors());
		}
	}

	/** One target asset: long package name, object path and what is there now. */
	struct FTarget
	{
		FString AssetName;
		FString LongPackageName;
		FString ObjectPath;
		UObject* Loaded = nullptr;
		bool bOnDisk = false;

		bool Exists() const { return Loaded != nullptr || bOnDisk; }
	};

	static FTarget MakeTarget(const FString& Dest, const FString& Name)
	{
		FTarget T;
		T.AssetName = Name;
		T.LongPackageName = Dest + TEXT("/") + Name;
		T.ObjectPath = T.LongPackageName + TEXT(".") + Name;
		return T;
	}

	/** Moves a transient object into a new (or found) package at Target and registers it as an asset. */
	static UPackage* MoveIntoPackage(UObject* Obj, const FTarget& Target)
	{
		UPackage* Pkg = CreatePackage(*Target.LongPackageName);
		Obj->Rename(*Target.AssetName, Pkg, REN_DontCreateRedirectors | REN_NonTransactional);
		Obj->ClearFlags(RF_Transient);
		Obj->SetFlags(RF_Public | RF_Standalone);
		FAssetRegistryModule::AssetCreated(Obj);
		Pkg->MarkPackageDirty();
		return Pkg;
	}
}

FPipelineImportResult ImportPipelineModelFolder(const FPipelineImportOptions& Options)
{
	using namespace PipelineModelImporterPrivate;

	FPipelineImportResult Result;
	if (!IsInGameThread())
	{
		// Touch nothing (no UObject, no log category state beyond a thread-safe log line).
		Result.Report.Add(ESev::Error, ECode::None, TEXT("ImportPipelineModelFolder must run on the game thread. Call it from the editor's game thread (e.g. via AsyncTask(ENamedThreads::GameThread, ...))."));
		return Result;
	}

	FString FolderLeaf = FolderLeafOf(Options.SourceFolder);
	FPipelineValidationReport& Report = Result.Report;

	auto Finish = [&FolderLeaf, &Result]() -> FPipelineImportResult
	{
		LogResult(FolderLeaf, Result);
		return Result;
	};

	FScopedSlowTask SlowTask(5.f, FText::Format(LOCTEXT("Importing", "Importing model folder '{0}'"), FText::FromString(FolderLeaf)));

	// 1. Parse the folder.
	SlowTask.EnterProgressFrame(1.f, LOCTEXT("Parsing", "Reading the model folder"));
	FPipelineModelFolderInfo Info;
	if (!ParsePipelineModelFolder(Options.SourceFolder, Options.Task, Info, Report) || !Report.IsValid())
	{
		return Finish();
	}
	if (!Info.FolderPath.IsEmpty())
	{
		FolderLeaf = FolderLeafOf(Info.FolderPath);
	}

	// 2. Destination and names.
	FString Dest = Options.DestinationPath;
	Dest.TrimStartAndEndInline();
	while (Dest.Len() > 1 && Dest.EndsWith(TEXT("/")))
	{
		Dest.LeftChopInline(1);
	}
	const FString Name = Options.AssetName.IsEmpty() ? ObjectTools::SanitizeObjectName(FolderLeaf) : Options.AssetName;
	if (Name.IsEmpty() || Name.Contains(TEXT("/")))
	{
		Report.Add(ESev::Error, ECode::InvalidDestination, FString::Printf(
			TEXT("Asset name '%s' is not usable (empty or contains '/'). Pass an AssetName made of letters, digits and '_'."), *Name.Left(MaxLongPackageNameLen)));
		return Finish();
	}
	// Length first: every later step (FindObject, FindPackage, CreatePackage) builds an FName, which asserts on very long
	// strings. Only string operations happen before this check.
	FTarget PkgTarget = MakeTarget(Dest, Name);
	FTarget ModelTarget = MakeTarget(Dest, Name + TEXT("_Model"));
	if (PkgTarget.LongPackageName.Len() > MaxLongPackageNameLen || ModelTarget.LongPackageName.Len() > MaxLongPackageNameLen)
	{
		Report.Add(ESev::Error, ECode::InvalidDestination, FString::Printf(
			TEXT("The asset path '%s...' is too long (%d characters; the limit is %d for '<DestinationPath>/<AssetName>_Model'). Use a shorter DestinationPath or AssetName."),
			*ModelTarget.LongPackageName.Left(64), ModelTarget.LongPackageName.Len(), MaxLongPackageNameLen));
		return Finish();
	}
	if (!Options.AssetName.IsEmpty())
	{
		// An explicit name is used as given, so it must be a valid object name ("None" is the null FName).
		FText XNameReason;
		if (Name.Equals(TEXT("None"), ESearchCase::IgnoreCase))
		{
			Report.Add(ESev::Error, ECode::InvalidDestination,
				TEXT("AssetName 'None' is reserved by Unreal (it is the null name). Choose another AssetName."));
			return Finish();
		}
		if (!FName::IsValidXName(Name, INVALID_OBJECTNAME_CHARACTERS, &XNameReason))
		{
			Report.Add(ESev::Error, ECode::InvalidDestination, FString::Printf(
				TEXT("AssetName '%s' is not a valid object name (%s). Use letters, digits and '_' only, or leave AssetName empty to use the folder name."),
				*Name, *XNameReason.ToString()));
			return Finish();
		}
	}
	bool bDestOk = true;
	for (const FTarget* T : {&PkgTarget, &ModelTarget})
	{
		FText Reason;
		if (!FPackageName::IsValidLongPackageName(T->LongPackageName, false, &Reason))
		{
			Report.Add(ESev::Error, ECode::InvalidDestination, FString::Printf(
				TEXT("'%s' is not a valid long package name (%s). DestinationPath must be a content path like /Game/Models and AssetName a plain name without spaces or punctuation."),
				*T->LongPackageName, *Reason.ToString()));
			bDestOk = false;
		}
	}
	if (!bDestOk)
	{
		return Finish();
	}

	// Existing assets: loaded objects or packages on disk.
	for (FTarget* T : {&PkgTarget, &ModelTarget})
	{
		T->Loaded = FindObject<UObject>(nullptr, *T->ObjectPath);
		T->bOnDisk = FPackageName::DoesPackageExist(T->LongPackageName);
	}
	// A package that is garbage (pending GC) would be handed back by CreatePackage; never reuse it.
	{
		bool bPendingGc = false;
		for (const FTarget* T : {&PkgTarget, &ModelTarget})
		{
			UPackage* Found = FindPackage(nullptr, *T->LongPackageName);
			if (Found && !IsValid(Found))
			{
				Report.Add(ESev::Error, ECode::AssetExists, FString::Printf(
					TEXT("'%s' is still in memory as a deleted package waiting for garbage collection. Run garbage collection (or restart the editor), or choose another AssetName or DestinationPath."),
					*T->LongPackageName));
				bPendingGc = true;
			}
		}
		if (bPendingGc)
		{
			return Finish();
		}
	}
	UPipelineModelPackage* ExistingPkg = nullptr;
	UNNEModelData* ExistingModel = nullptr;
	if (PkgTarget.Exists() || ModelTarget.Exists())
	{
		bool bCanOverwrite = Options.bOverwriteExisting && PkgTarget.Exists() && ModelTarget.Exists();
		if (bCanOverwrite)
		{
			UObject* PkgObj = PkgTarget.Loaded ? PkgTarget.Loaded : LoadObject<UObject>(nullptr, *PkgTarget.ObjectPath, nullptr, LOAD_NoWarn | LOAD_Quiet);
			UObject* ModelObj = ModelTarget.Loaded ? ModelTarget.Loaded : LoadObject<UObject>(nullptr, *ModelTarget.ObjectPath, nullptr, LOAD_NoWarn | LOAD_Quiet);
			ExistingPkg = Cast<UPipelineModelPackage>(PkgObj);
			ExistingModel = Cast<UNNEModelData>(ModelObj);
			bCanOverwrite = IsValid(ExistingPkg) && IsValid(ExistingModel);
		}
		if (!bCanOverwrite)
		{
			for (const FTarget* T : {&PkgTarget, &ModelTarget})
			{
				if (T->Exists())
				{
					Report.Add(ESev::Error, ECode::AssetExists, Options.bOverwriteExisting
						? FString::Printf(TEXT("'%s' exists and cannot be overwritten: overwrite needs both '%s' (UPipelineModelPackage) and '%s' (UNNEModelData) to exist with those classes. Choose another AssetName or DestinationPath."),
							*T->ObjectPath, *PkgTarget.ObjectPath, *ModelTarget.ObjectPath)
						: FString::Printf(TEXT("'%s' already exists. Set bOverwriteExisting to update it in place, or choose another AssetName or DestinationPath."),
							*T->ObjectPath));
				}
			}
			return Finish();
		}
	}

	// 3. Model bytes.
	SlowTask.EnterProgressFrame(1.f, LOCTEXT("Reading", "Reading the ONNX model"));
	TArray64<uint8> Bytes;
	if (Info.ModelFile.IsEmpty() || !FFileHelper::LoadFileToArray(Bytes, *Info.ModelFile) || Bytes.Num() == 0)
	{
		Report.Add(ESev::Error, ECode::MissingModelFile, FString::Printf(
			TEXT("Could not read the model file '%s' (missing, empty or unreadable). Check the file and its permissions."), *Info.ModelFile));
		return Finish();
	}

	// 4. Build both objects in the transient package.
	SlowTask.EnterProgressFrame(1.f, LOCTEXT("Building", "Building the model package"));
	TStrongObjectPtr<UNNEModelData> TempModel(NewObject<UNNEModelData>(GetTransientPackage(), NAME_None, RF_Transient));
	TempModel->Init(TEXT("onnx"), Bytes);
	TStrongObjectPtr<UPipelineModelPackage> TempPkg(NewObject<UPipelineModelPackage>(GetTransientPackage(), NAME_None, RF_Transient));
	FillPackage(TempPkg.Get(), Info, Options.PreferredDevice, TempModel.Get());

	// 5. Settings always; full validation when asked. The validator starts with ValidateSettings itself, so its report
	// replaces the settings-only one (no duplicated messages).
	SlowTask.EnterProgressFrame(1.f, LOCTEXT("Validating", "Validating the model"));
	if (Options.bValidate)
	{
		Report.Append(ValidatePipelineModelPackage(TempPkg.Get(), Options.Validation));
	}
	else
	{
		TempPkg->ValidateSettings(Report);
	}
	if (!Report.IsValid())
	{
		DropTransient(TempPkg.Get());
		DropTransient(TempModel.Get());
		return Finish();
	}

	// 6. Commit.
	SlowTask.EnterProgressFrame(1.f, LOCTEXT("Creating", "Creating the assets"));
	UPipelineModelPackage* FinalPkg = nullptr;
	UNNEModelData* FinalModel = nullptr;
	if (ExistingPkg && ExistingModel)
	{
		ExistingModel->Init(TEXT("onnx"), Bytes);
		FillPackage(ExistingPkg, Info, Options.PreferredDevice, ExistingModel);
		ExistingModel->MarkPackageDirty();
		ExistingPkg->MarkPackageDirty();
		DropTransient(TempPkg.Get());
		DropTransient(TempModel.Get());
		FinalPkg = ExistingPkg;
		FinalModel = ExistingModel;
	}
	else
	{
		FinalModel = TempModel.Get();
		FinalPkg = TempPkg.Get();
		MoveIntoPackage(FinalModel, ModelTarget);
		MoveIntoPackage(FinalPkg, PkgTarget);
		FinalPkg->Model = FinalModel;
	}
	Result.Package = FinalPkg;
	Result.Model = FinalModel;

	// 7. Save.
	if (Options.bSave)
	{
		const TArray<UPackage*> ToSave = {FinalPkg->GetOutermost(), FinalModel->GetOutermost()};
		if (!UEditorLoadingAndSavingUtils::SavePackages(ToSave, false))
		{
			Report.Add(ESev::Error, ECode::SaveFailed, FString::Printf(
				TEXT("Saving '%s' and '%s' failed. The assets are in memory; check that the files are writable (source control, read-only flag) and save them from the Content Browser."),
				*PkgTarget.LongPackageName, *ModelTarget.LongPackageName));
		}
	}

	Result.bSuccess = Report.IsValid();
	return Finish();
}

#undef LOCTEXT_NAMESPACE
