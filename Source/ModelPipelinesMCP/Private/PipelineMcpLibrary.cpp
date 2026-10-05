// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineMcpLibrary.h"

#include "AssetRegistry/ARFilter.h"
#include "AssetRegistry/AssetData.h"
#include "AssetRegistry/IAssetRegistry.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "NNEModelData.h"
#include "ObjectTools.h"
#include "UObject/NameTypes.h"
#include "UObject/Package.h"
#include "UObject/SoftObjectPath.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineMcpJson.h"
#include "PipelineMcpLog.h"
#include "PipelineMcpRun.h"
#include "PipelineModelFolder.h"
#include "PipelineModelImporter.h"
#include "PipelineModelPackage.h"
#include "PipelineModelValidator.h"
#include "PipelineValidation.h"

namespace PipelineMcpLibraryPrivate
{
	using ESev = EPipelineValidationSeverity;
	using ECode = EPipelineValidationCode;

	static const TCHAR* NotGameThreadError = TEXT("must be called on the game thread");
	static const TCHAR* PathError = TEXT("Path must be a content path like /Game");
	static const TCHAR* MsgNothingChanged = TEXT("Nothing was changed. Call again with confirm=true to import.");
	static const TCHAR* MsgCannotImport = TEXT("The folder can't be imported; fix the errors in report.");
	static const TCHAR* MsgAssetsExist = TEXT("Assets already exist; pass overwrite_existing=true (updates them in place) or another asset_name.");

	/** Same limit as the importer (PipelineModelImporter.cpp, MaxLongPackageNameLen): FName asserts on very long strings. */
	static constexpr int32 MaxLongPackageNameLen = PipelineMcpJson::MaxLongPackageNameLen;

	/** Last path component of a folder path, trailing separators ignored (mirrors the importer's FolderLeafOf). */
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

	/** One planned target asset (mirrors the importer's FTarget). */
	struct FTarget
	{
		FString AssetName;
		FString LongPackageName;
		FString ObjectPath;
		UObject* Loaded = nullptr;
		bool bOnDisk = false;
		/** Asset registry entry (invalid when it has none). */
		FAssetData RegistryData;

		bool Exists() const { return Loaded != nullptr || bOnDisk || RegistryData.IsValid(); }
	};

	static FTarget MakeTarget(const FString& Dest, const FString& Name)
	{
		FTarget T;
		T.AssetName = Name;
		T.LongPackageName = Dest + TEXT("/") + Name;
		T.ObjectPath = T.LongPackageName + TEXT(".") + Name;
		return T;
	}

	/** The planned import targets, computed exactly like ImportPipelineModelFolder (PipelineModelImporter.cpp:160-273). */
	struct FImportPlan
	{
		bool bNamesValid = false;
		FTarget Package;
		FTarget Model;
		bool bExists = false;
		/** Set when the assets exist, overwrite is requested, and the importer would refuse it anyway. */
		bool bCannotOverwrite = false;
	};

	/** Validates destination and names (adding InvalidDestination / AssetExists errors to Report) and checks for existing assets. Creates nothing. */
	static FImportPlan PlanImport(const FPipelineModelFolderInfo& Info, const FString& SourceFolder, const FString& DestinationPath,
		const FString& AssetName, bool bOverwriteExisting, FPipelineValidationReport& Report)
	{
		FImportPlan Plan;
		const FString FolderLeaf = FolderLeafOf(Info.FolderPath.IsEmpty() ? SourceFolder : Info.FolderPath);

		FString Dest = DestinationPath;
		Dest.TrimStartAndEndInline();
		while (Dest.Len() > 1 && Dest.EndsWith(TEXT("/")))
		{
			Dest.LeftChopInline(1);
		}
		const FString Name = AssetName.IsEmpty() ? ObjectTools::SanitizeObjectName(FolderLeaf) : AssetName;
		if (Name.IsEmpty() || Name.Contains(TEXT("/")))
		{
			Report.Add(ESev::Error, ECode::InvalidDestination, FString::Printf(
				TEXT("Asset name '%s' is not usable (empty or contains '/'). Pass an asset_name made of letters, digits and '_'."), *Name.Left(MaxLongPackageNameLen)));
			return Plan;
		}
		// Length first: only string operations happen before this check (FName construction asserts on very long strings).
		Plan.Package = MakeTarget(Dest, Name);
		Plan.Model = MakeTarget(Dest, Name + TEXT("_Model"));
		if (Plan.Package.LongPackageName.Len() > MaxLongPackageNameLen || Plan.Model.LongPackageName.Len() > MaxLongPackageNameLen)
		{
			Report.Add(ESev::Error, ECode::InvalidDestination, FString::Printf(
				TEXT("The asset path '%s...' is too long (%d characters; the limit is %d for '<destination_path>/<asset_name>_Model'). Use a shorter destination_path or asset_name."),
				*Plan.Model.LongPackageName.Left(64), Plan.Model.LongPackageName.Len(), MaxLongPackageNameLen));
			return Plan;
		}
		if (!AssetName.IsEmpty())
		{
			if (Name.Equals(TEXT("None"), ESearchCase::IgnoreCase))
			{
				Report.Add(ESev::Error, ECode::InvalidDestination,
					TEXT("asset_name 'None' is reserved by Unreal (it is the null name). Choose another asset_name."));
				return Plan;
			}
			FText XNameReason;
			if (!FName::IsValidXName(Name, INVALID_OBJECTNAME_CHARACTERS, &XNameReason))
			{
				Report.Add(ESev::Error, ECode::InvalidDestination, FString::Printf(
					TEXT("asset_name '%s' is not a valid object name (%s). Use letters, digits and '_' only, or leave asset_name empty to use the folder name."),
					*Name, *XNameReason.ToString()));
				return Plan;
			}
		}
		bool bDestOk = true;
		for (const FTarget* T : {&Plan.Package, &Plan.Model})
		{
			FText Reason;
			if (!FPackageName::IsValidLongPackageName(T->LongPackageName, false, &Reason))
			{
				Report.Add(ESev::Error, ECode::InvalidDestination, FString::Printf(
					TEXT("'%s' is not a valid long package name (%s). destination_path must be a content path like /Game/Models and asset_name a plain name without spaces or punctuation."),
					*T->LongPackageName, *Reason.ToString()));
				bDestOk = false;
			}
		}
		if (!bDestOk)
		{
			return Plan;
		}
		Plan.bNamesValid = true;

		// Existing assets: loaded objects, packages on disk, asset registry entries. Read-only queries.
		IAssetRegistry* Registry = IAssetRegistry::Get();
		for (FTarget* T : {&Plan.Package, &Plan.Model})
		{
			UObject* Found = FindObject<UObject>(nullptr, *T->ObjectPath);
			T->Loaded = IsValid(Found) ? Found : nullptr;
			T->bOnDisk = FPackageName::DoesPackageExist(T->LongPackageName);
			if (Registry)
			{
				T->RegistryData = Registry->GetAssetByObjectPath(FSoftObjectPath(T->ObjectPath));
			}
			UPackage* FoundPackage = FindPackage(nullptr, *T->LongPackageName);
			if (FoundPackage && !IsValid(FoundPackage))
			{
				Report.Add(ESev::Error, ECode::AssetExists, FString::Printf(
					TEXT("'%s' is still in memory as a deleted package waiting for garbage collection. Run garbage collection (or restart the editor), or choose another asset_name or destination_path."),
					*T->LongPackageName));
			}
		}
		Plan.bExists = Plan.Package.Exists() || Plan.Model.Exists();
		if (Plan.bExists && bOverwriteExisting)
		{
			auto IsClass = [](const FTarget& T, const UClass* Class) -> bool
			{
				if (T.Loaded)
				{
					return T.Loaded->IsA(Class);
				}
				// On disk but not loaded: trust the registry when it knows the asset (subclasses accepted, like the importer's
				// Cast<>); otherwise the importer decides.
				return !T.RegistryData.IsValid() || T.RegistryData.IsInstanceOf(Class);
			};
			const bool bCanOverwrite = Plan.Package.Exists() && Plan.Model.Exists()
				&& IsClass(Plan.Package, UPipelineModelPackage::StaticClass()) && IsClass(Plan.Model, UNNEModelData::StaticClass());
			if (!bCanOverwrite)
			{
				Plan.bCannotOverwrite = true;
				for (const FTarget* T : {&Plan.Package, &Plan.Model})
				{
					if (T->Exists())
					{
						Report.Add(ESev::Error, ECode::AssetExists, FString::Printf(
							TEXT("'%s' exists and cannot be overwritten: overwrite needs both '%s' (UPipelineModelPackage) and '%s' (UNNEModelData) to exist with those classes. Choose another asset_name or destination_path."),
							*T->ObjectPath, *Plan.Package.ObjectPath, *Plan.Model.ObjectPath));
					}
				}
			}
		}
		return Plan;
	}

	/**
	 * Resolves an object path or package name to a loaded UPipelineModelPackage; nullptr when the string is not a valid
	 * reference, nothing exists there, or the object has another class. Strings are checked before any FName is built
	 * (PipelineMcpJson::ResolveAsset).
	 */
	static UPipelineModelPackage* ResolvePackage(const FString& Arg)
	{
		return Cast<UPipelineModelPackage>(PipelineMcpJson::ResolveAsset(Arg));
	}

	/** Package entry of ListModelPackages. */
	struct FListedPackage
	{
		FString Path;
		TSharedPtr<FJsonObject> Json;
	};
}

FString UPipelineMcpLibrary::ListModelPackages(const FString& Path, bool bRecursive)
{
	using namespace PipelineMcpLibraryPrivate;

	if (!IsInGameThread())
	{
		return PipelineMcpJson::MakeError(NotGameThreadError);
	}

	FString Folder = Path;
	while (Folder.Len() > 1 && Folder.EndsWith(TEXT("/")))
	{
		Folder.LeftChopInline(1);
	}
	// A folder is valid when a package directly inside it would be a valid long package name (covers mount roots like /Game).
	if (Folder.IsEmpty() || Folder[0] != TEXT('/') || Folder.Len() > MaxLongPackageNameLen
		|| !FPackageName::IsValidLongPackageName(Folder + TEXT("/X"), true))
	{
		return PipelineMcpJson::MakeError(PathError);
	}

	TArray<FListedPackage> Listed;
	if (IAssetRegistry* Registry = IAssetRegistry::Get())
	{
		// Right after editor start the registry may still be gathering; scan the validated folder so on-disk packages are listed.
		Registry->ScanPathsSynchronous({ Folder }, /*bForceRescan*/ false);

		FARFilter Filter;
		Filter.PackagePaths.Add(FName(*Folder));
		Filter.ClassPaths.Add(UPipelineModelPackage::StaticClass()->GetClassPathName());
		Filter.bRecursivePaths = bRecursive;
		Filter.bRecursiveClasses = true;

		TArray<FAssetData> Assets;
		Registry->GetAssets(Filter, Assets);
		for (const FAssetData& Asset : Assets)
		{
			const UPipelineModelPackage* Pkg = Cast<UPipelineModelPackage>(Asset.GetAsset());
			if (!IsValid(Pkg))
			{
				continue;
			}
			FPipelineValidationReport Settings;
			Pkg->ValidateSettings(Settings);

			TSharedRef<FJsonObject> Entry = MakeShared<FJsonObject>();
			const FString ObjPath = Pkg->GetPathName();
			Entry->SetStringField(TEXT("path"), ObjPath);
			Entry->SetStringField(TEXT("task"), PipelineMcpJson::TaskToString(Pkg->Task));
			Entry->SetStringField(TEXT("preferred_device"), PipelineMcpJson::DeviceToString(Pkg->PreferredDevice));
			Entry->SetStringField(TEXT("model"), Pkg->Model ? Pkg->Model->GetPathName() : FString());
			Entry->SetNumberField(TEXT("labels"), Pkg->Labels.Num());
			Entry->SetBoolField(TEXT("has_tokenizer"), !Pkg->TokenizerJson.IsEmpty());
			Entry->SetBoolField(TEXT("valid_settings"), Settings.IsValid());
			Listed.Add({ ObjPath, Entry });
		}
	}
	Listed.Sort([](const FListedPackage& A, const FListedPackage& B) { return A.Path < B.Path; });

	TArray<TSharedPtr<FJsonValue>> Packages;
	Packages.Reserve(Listed.Num());
	for (const FListedPackage& L : Listed)
	{
		Packages.Add(MakeShared<FJsonValueObject>(L.Json));
	}
	TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
	Root->SetNumberField(TEXT("count"), Packages.Num());
	Root->SetArrayField(TEXT("packages"), Packages);
	return PipelineMcpJson::ToCondensedString(Root);
}

FString UPipelineMcpLibrary::ImportModelFolder(const FString& SourceFolder, const FString& DestinationPath, const FString& AssetName,
	bool bOverwriteExisting, bool bConfirm)
{
	using namespace PipelineMcpLibraryPrivate;

	if (!IsInGameThread())
	{
		return PipelineMcpJson::MakeError(NotGameThreadError);
	}
	if (SourceFolder.IsEmpty())
	{
		return PipelineMcpJson::MakeError(TEXT("source_folder is empty"));
	}

	// Always parse first (no UObject, no file written): gives the task and the preview report.
	FPipelineModelFolderInfo Info;
	FPipelineValidationReport ParseReport;
	const bool bParsed = ParsePipelineModelFolder(SourceFolder, EPipelineTask::None, Info, ParseReport) && ParseReport.IsValid();
	const FString TaskText = PipelineMcpJson::TaskToString(bParsed ? Info.Task : EPipelineTask::None);

	TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
	if (!bConfirm)
	{
		FPipelineValidationReport Report = ParseReport;
		const FImportPlan Plan = PlanImport(Info, SourceFolder, DestinationPath, AssetName, bOverwriteExisting, Report);

		bool bNeedsConfirm = false;
		FString Message;
		if (!Report.IsValid())
		{
			Message = MsgCannotImport;
		}
		else if (Plan.bExists && !bOverwriteExisting)
		{
			Message = MsgAssetsExist;
		}
		else
		{
			bNeedsConfirm = true;
			Message = MsgNothingChanged;
		}

		Root->SetBoolField(TEXT("success"), false);
		Root->SetBoolField(TEXT("needs_confirm"), bNeedsConfirm);
		// Always an object; empty strings when the destination or name is invalid.
		TSharedRef<FJsonObject> WouldCreate = MakeShared<FJsonObject>();
		WouldCreate->SetStringField(TEXT("package"), Plan.bNamesValid ? Plan.Package.ObjectPath : FString());
		WouldCreate->SetStringField(TEXT("model"), Plan.bNamesValid ? Plan.Model.ObjectPath : FString());
		Root->SetObjectField(TEXT("would_create"), WouldCreate);
		Root->SetBoolField(TEXT("exists"), Plan.bExists);
		Root->SetStringField(TEXT("task"), TaskText);
		Root->SetObjectField(TEXT("report"), PipelineMcpJson::ReportToJson(Report));
		Root->SetStringField(TEXT("message"), Message);
		UE_LOG(LogModelPipelinesMCP, Verbose, TEXT("ImportModelFolder preview of '%s': needs_confirm=%d, %d errors"),
			*SourceFolder.Left(MaxLongPackageNameLen), bNeedsConfirm ? 1 : 0, Report.NumErrors());
		return PipelineMcpJson::ToCondensedString(Root);
	}

	FPipelineImportOptions Options;
	Options.SourceFolder = SourceFolder;
	Options.DestinationPath = DestinationPath;
	Options.AssetName = AssetName;
	Options.Task = EPipelineTask::None;
	Options.PreferredDevice = EPipelineDevice::Auto;
	Options.bOverwriteExisting = bOverwriteExisting;
	Options.bSave = true;
	Options.bValidate = true;
	const FPipelineImportResult Result = ImportPipelineModelFolder(Options);

	const bool bSuccess = Result.bSuccess && IsValid(Result.Package) && IsValid(Result.Model);
	Root->SetBoolField(TEXT("success"), bSuccess);
	if (bSuccess)
	{
		Root->SetStringField(TEXT("package"), Result.Package->GetPathName());
		Root->SetStringField(TEXT("model"), Result.Model->GetPathName());
	}
	else
	{
		Root->SetField(TEXT("package"), MakeShared<FJsonValueNull>());
		Root->SetField(TEXT("model"), MakeShared<FJsonValueNull>());
	}
	Root->SetStringField(TEXT("task"), TaskText);
	Root->SetObjectField(TEXT("report"), PipelineMcpJson::ReportToJson(Result.Report));
	return PipelineMcpJson::ToCondensedString(Root);
}

FString UPipelineMcpLibrary::ValidateModelPackage(const FString& Package, bool bRunSmokeTest, bool bCheckGpu)
{
	using namespace PipelineMcpLibraryPrivate;

	if (!IsInGameThread())
	{
		return PipelineMcpJson::MakeError(NotGameThreadError);
	}
	const UPipelineModelPackage* Pkg = ResolvePackage(Package);
	if (!Pkg)
	{
		return PipelineMcpJson::MakeError(FString(TEXT("Model package not found: ")) + Package);
	}

	FPipelineValidateOptions Options;
	Options.bRunSmokeTest = bRunSmokeTest;
	Options.bCheckGpu = bCheckGpu;
	const FPipelineValidationReport Report = ValidatePipelineModelPackage(Pkg, Options);

	TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
	Root->SetStringField(TEXT("package"), Pkg->GetPathName());
	Root->SetStringField(TEXT("task"), PipelineMcpJson::TaskToString(Pkg->Task));
	Root->SetBoolField(TEXT("valid"), Report.IsValid());
	Root->SetNumberField(TEXT("errors"), Report.NumErrors());
	Root->SetNumberField(TEXT("warnings"), Report.NumWarnings());
	Root->SetObjectField(TEXT("report"), PipelineMcpJson::ReportToJson(Report));
	return PipelineMcpJson::ToCondensedString(Root);
}

UPipelineMcpJob* UPipelineMcpLibrary::BenchmarkModelPackage(const FString& Package, int32 Runs, const FString& Devices,
	const TArray<FString>& Texts, const FString& Image)
{
	if (!IsInGameThread())
	{
		UE_LOG(LogModelPipelinesMCP, Warning, TEXT("BenchmarkModelPackage must be called on the game thread; returning nullptr."));
		return nullptr;
	}
	return PipelineMcpRun::LaunchBenchmark(Package, Runs, Devices, Texts, Image);
}

UPipelineMcpJob* UPipelineMcpLibrary::RunPipelineSample(const FString& Package, const TArray<FString>& Texts,
	const FString& Image, int32 TopK, float ScoreThreshold)
{
	if (!IsInGameThread())
	{
		UE_LOG(LogModelPipelinesMCP, Warning, TEXT("RunPipelineSample must be called on the game thread; returning nullptr."));
		return nullptr;
	}
	return PipelineMcpRun::LaunchSample(Package, Texts, Image, TopK, ScoreThreshold);
}
