// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "AssetRegistry/AssetData.h"
#include "AssetRegistry/IAssetRegistry.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformFile.h"
#include "HAL/PlatformFileManager.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformProcess.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/Guid.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "NNEModelData.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"
#include "PipelineModelImporter.h"
#include "PipelineMcpLibrary.h"

namespace PipelineMcpLibrarySpecPrivate
{
	static const TCHAR* MsgNothing = TEXT("Nothing was changed. Call again with confirm=true to import.");
	static const TCHAR* MsgCannot = TEXT("The folder can't be imported; fix the errors in report.");
	static const TCHAR* MsgExists = TEXT("Assets already exist; pass overwrite_existing=true (updates them in place) or another asset_name.");
	static const TCHAR* MsgNotGameThread = TEXT("must be called on the game thread");

	static FString GetDevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		if (!Env.IsEmpty())
		{
			return Env;
		}
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev"));
	}

	static bool IsGpuRequired()
	{
		return FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_GPU")) == TEXT("1");
	}

	static FString GetStr(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
	{
		FString S;
		if (O.IsValid())
		{
			O->TryGetStringField(Key, S);
		}
		return S;
	}

	static bool IsNull(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
	{
		return O.IsValid() && O->HasTypedField<EJson::Null>(Key);
	}

	static TSharedPtr<FJsonObject> GetObj(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
	{
		const TSharedPtr<FJsonObject>* P = nullptr;
		if (O.IsValid() && O->TryGetObjectField(Key, P) && P)
		{
			return *P;
		}
		return nullptr;
	}

	/** Messages of a report with Severity (and Code when non-null). */
	static int32 CountMessages(const TSharedPtr<FJsonObject>& Report, const TCHAR* Severity, const TCHAR* Code = nullptr)
	{
		int32 N = 0;
		const TArray<TSharedPtr<FJsonValue>>* Msgs = nullptr;
		if (Report.IsValid() && Report->TryGetArrayField(TEXT("messages"), Msgs))
		{
			for (const TSharedPtr<FJsonValue>& V : *Msgs)
			{
				const TSharedPtr<FJsonObject>* M = nullptr;
				if (V.IsValid() && V->TryGetObject(M) && M && GetStr(*M, TEXT("severity")) == Severity
					&& (!Code || GetStr(*M, TEXT("code")) == Code))
				{
					++N;
				}
			}
		}
		return N;
	}

	/** The device_checks entry with this device name, or nullptr. */
	static TSharedPtr<FJsonObject> FindDeviceCheck(const TSharedPtr<FJsonObject>& Report, const TCHAR* Device)
	{
		const TArray<TSharedPtr<FJsonValue>>* Checks = nullptr;
		if (Report.IsValid() && Report->TryGetArrayField(TEXT("device_checks"), Checks))
		{
			for (const TSharedPtr<FJsonValue>& V : *Checks)
			{
				const TSharedPtr<FJsonObject>* C = nullptr;
				if (V.IsValid() && V->TryGetObject(C) && C && GetStr(*C, TEXT("device")) == Device)
				{
					return *C;
				}
			}
		}
		return nullptr;
	}

	static FString ObjPath(const FString& Folder, const FString& Name)
	{
		return Folder / Name + TEXT(".") + Name;
	}

	/** Renames an object into the transient package and garbage-marks it and its content package. */
	static void DiscardOne(UObject* Obj)
	{
		if (!Obj || !IsValid(Obj))
		{
			return;
		}
		UPackage* Old = Obj->GetOutermost();
		UPackage* Transient = GetTransientPackage();
		if (Old != Transient)
		{
			FAssetRegistryModule::AssetDeleted(Obj);
			Obj->Rename(*MakeUniqueObjectName(Transient, Obj->GetClass(), Obj->GetFName()).ToString(), Transient,
				REN_DontCreateRedirectors | REN_NonTransactional);
		}
		Obj->ClearFlags(RF_Public | RF_Standalone);
		Obj->MarkAsGarbage();
		if (Old && Old != Transient && IsValid(Old))
		{
			Old->SetDirtyFlag(false);
			Old->ClearFlags(RF_Public | RF_Standalone);
			Old->MarkAsGarbage();
		}
	}

	/** Runs F on a background thread and waits (timeout) for its result; false on timeout. */
	static bool RunOffGameThread(TFunction<FString()> F, FString& Out)
	{
		FEvent* Done = FPlatformProcess::GetSynchEventFromPool(true);
		TSharedRef<FString, ESPMode::ThreadSafe> Result = MakeShared<FString, ESPMode::ThreadSafe>();
		AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [F, Result, Done]()
		{
			*Result = F();
			Done->Trigger();
		});
		const bool bOk = Done->Wait(FTimespan::FromSeconds(30.0));
		if (bOk)
		{
			FPlatformProcess::ReturnSynchEventToPool(Done); // on timeout the event leaks on purpose: the task may still trigger it
			Out = *Result;
		}
		return bOk;
	}
}

BEGIN_DEFINE_SPEC(FPipelineMcpLibrarySpec, "Pipelines.MCP.Library",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	/** /Game/PipelinesTests/MCP_<guid8>: unique per test. */
	FString Dest;
	FString DestGuid;
	TArray<FString> TempDirs;

	/** <dev>/models/import_test/<Name>; AddError (with the fetch command) and false when missing. */
	bool RequireFolder(const FString& Name, FString& OutFolder);
	/** Parses Text with FJsonSerializer and checks it has no nan/inf text. nullptr (+error) otherwise. */
	TSharedPtr<FJsonObject> Parse(const FString& Text, const FString& What);
	/** Import with bSave = false (in-memory assets, registered with the asset registry). */
	FPipelineImportResult ImportUnsaved(const FString& Folder, const FString& Name);
	bool AnyObjectFor(const FString& Name) const;
	bool FileFor(const FString& Name) const;
	int32 RegistryCount() const;
	FString ContentDir() const;
	void Cleanup();
	void CheckReportShape(const TSharedPtr<FJsonObject>& Report, const FString& What);
END_DEFINE_SPEC(FPipelineMcpLibrarySpec)

bool FPipelineMcpLibrarySpec::RequireFolder(const FString& Name, FString& OutFolder)
{
	OutFolder = PipelineMcpLibrarySpecPrivate::GetDevDir() / TEXT("models/import_test") / Name;
	if (!IFileManager::Get().DirectoryExists(*OutFolder))
	{
		AddError(FString::Printf(TEXT("test model folder '%s' missing; generate with: python Tools/make_import_test_data.py"), *OutFolder));
		return false;
	}
	return true;
}

TSharedPtr<FJsonObject> FPipelineMcpLibrarySpec::Parse(const FString& Text, const FString& What)
{
	TSharedPtr<FJsonObject> Root;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		AddError(FString::Printf(TEXT("[%s] output is not a JSON object; got: %s"), *What, *Text.Left(400)));
		return nullptr;
	}
	// NaN/inf must be written as null, never as text. Plain "nan"/"inf" would match words like "Info", so look at JSON token positions.
	const FString Lower = Text.ToLower();
	static const TCHAR* Bad[] = { TEXT(":nan"), TEXT(",nan"), TEXT("[nan"), TEXT(":-nan"), TEXT("-nan("), TEXT(":inf"), TEXT(",inf"),
		TEXT("[inf"), TEXT(":-inf"), TEXT("infinity") };
	for (const TCHAR* B : Bad)
	{
		if (Lower.Contains(B))
		{
			AddError(FString::Printf(TEXT("[%s] output contains non-finite text '%s'; expected null instead. Got: %s"), *What, B, *Text.Left(400)));
		}
	}
	return Root;
}

FPipelineImportResult FPipelineMcpLibrarySpec::ImportUnsaved(const FString& Folder, const FString& Name)
{
	FPipelineImportOptions O;
	O.SourceFolder = Folder;
	O.DestinationPath = Dest;
	O.AssetName = Name;
	O.bSave = false;
	O.bValidate = true;
	O.Validation.bCheckGpu = false;
	return ImportPipelineModelFolder(O);
}

bool FPipelineMcpLibrarySpec::AnyObjectFor(const FString& Name) const
{
	using namespace PipelineMcpLibrarySpecPrivate;
	return FindObject<UObject>(nullptr, *ObjPath(Dest, Name)) != nullptr
		|| FindObject<UObject>(nullptr, *ObjPath(Dest, Name + TEXT("_Model"))) != nullptr
		|| FindPackage(nullptr, *(Dest / Name)) != nullptr
		|| FindPackage(nullptr, *(Dest / Name + TEXT("_Model"))) != nullptr;
}

bool FPipelineMcpLibrarySpec::FileFor(const FString& Name) const
{
	const FString Ext = FPackageName::GetAssetPackageExtension();
	return FPaths::FileExists(FPackageName::LongPackageNameToFilename(Dest / Name, Ext))
		|| FPaths::FileExists(FPackageName::LongPackageNameToFilename(Dest / Name + TEXT("_Model"), Ext));
}

int32 FPipelineMcpLibrarySpec::RegistryCount() const
{
	TArray<FAssetData> Assets;
	IAssetRegistry::Get()->GetAssetsByPath(FName(*Dest), Assets, true);
	return Assets.Num();
}

FString FPipelineMcpLibrarySpec::ContentDir() const
{
	return FPaths::ConvertRelativePathToFull(FPaths::ProjectContentDir() / TEXT("PipelinesTests") / (TEXT("MCP_") + DestGuid));
}

void FPipelineMcpLibrarySpec::Cleanup()
{
	using namespace PipelineMcpLibrarySpecPrivate;
	TArray<FAssetData> Assets;
	IAssetRegistry::Get()->GetAssetsByPath(FName(*Dest), Assets, true);
	TArray<UObject*> Objs;
	for (const FAssetData& A : Assets)
	{
		if (UObject* O = A.GetAsset())
		{
			Objs.Add(O);
		}
	}
	for (UObject* O : Objs)
	{
		DiscardOne(O);
	}
	// Test cleanup only: the product never deletes. Saved .uasset files of this test's own temp folder.
	IFileManager::Get().DeleteDirectory(*ContentDir(), false, true);
	for (const FString& D : TempDirs)
	{
		IFileManager::Get().DeleteDirectory(*D, false, true);
	}
	TempDirs.Reset();
}

void FPipelineMcpLibrarySpec::CheckReportShape(const TSharedPtr<FJsonObject>& Report, const FString& What)
{
	if (!Report.IsValid())
	{
		AddError(FString::Printf(TEXT("[%s] 'report' is missing or not an object"), *What));
		return;
	}
	const TArray<TSharedPtr<FJsonValue>>* Msgs = nullptr;
	if (!Report->TryGetArrayField(TEXT("messages"), Msgs))
	{
		AddError(FString::Printf(TEXT("[%s] report.messages missing"), *What));
	}
	else
	{
		for (const TSharedPtr<FJsonValue>& V : *Msgs)
		{
			const TSharedPtr<FJsonObject>* M = nullptr;
			if (!V.IsValid() || !V->TryGetObject(M) || !M)
			{
				AddError(FString::Printf(TEXT("[%s] a report message is not an object"), *What));
				continue;
			}
			const FString Sev = PipelineMcpLibrarySpecPrivate::GetStr(*M, TEXT("severity"));
			TestTrue(*FString::Printf(TEXT("[%s] message severity '%s' in {Info,Warning,Error}"), *What, *Sev),
				Sev == TEXT("Info") || Sev == TEXT("Warning") || Sev == TEXT("Error"));
			const FString Code = PipelineMcpLibrarySpecPrivate::GetStr(*M, TEXT("code"));
			TestTrue(*FString::Printf(TEXT("[%s] message code '%s' is an EPipelineValidationCode name"), *What, *Code),
				StaticEnum<EPipelineValidationCode>()->GetValueByNameString(Code) != INDEX_NONE);
			TestTrue(*FString::Printf(TEXT("[%s] message has a 'message' string"), *What), (*M)->HasTypedField<EJson::String>(TEXT("message")));
		}
	}
	const TArray<TSharedPtr<FJsonValue>>* Checks = nullptr;
	TestTrue(*FString::Printf(TEXT("[%s] report.device_checks is an array"), *What), Report->TryGetArrayField(TEXT("device_checks"), Checks));
}

void FPipelineMcpLibrarySpec::Define()
{
	using namespace PipelineMcpLibrarySpecPrivate;

	BeforeEach([this]()
	{
		DestGuid = FGuid::NewGuid().ToString(EGuidFormats::Digits).Left(8);
		Dest = FString::Printf(TEXT("/Game/PipelinesTests/MCP_%s"), *DestGuid);
	});

	AfterEach([this]()
	{
		Cleanup();
	});

	Describe("ListModelPackages", [this]()
	{
		It("lists in-memory imported packages with their fields, sorted by path", [this]()
		{
			FString F1, F2;
			if (!RequireFolder(TEXT("tiny_text_embedding"), F1) || !RequireFolder(TEXT("tiny_text_classification"), F2)) { return; }
			// Import the classification model first: a list in import order (not sorted) would put it first.
			const FPipelineImportResult R2 = ImportUnsaved(F2, TEXT("ZetaCls"));
			const FPipelineImportResult R1 = ImportUnsaved(F1, TEXT("AlphaEmb"));
			if (!TestTrue(TEXT("both test imports succeeded; report: ") + R1.Report.ToString() + R2.Report.ToString(), R1.bSuccess && R2.bSuccess)) { return; }

			const FString Json = UPipelineMcpLibrary::ListModelPackages(Dest, true);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("List"));
			if (!J.IsValid()) { return; }
			double Count = -1;
			J->TryGetNumberField(TEXT("count"), Count);
			TestEqual(*FString::Printf(TEXT("ListModelPackages(%s) count; got %s"), *Dest, *Json), (int32)Count, 2);
			const TArray<TSharedPtr<FJsonValue>>* Pkgs = nullptr;
			if (!J->TryGetArrayField(TEXT("packages"), Pkgs) || Pkgs->Num() != 2)
			{
				AddError(FString::Printf(TEXT("packages array missing or not 2 entries; got %s"), *Json));
				return;
			}
			const TSharedPtr<FJsonObject> P0 = (*Pkgs)[0]->AsObject();
			const TSharedPtr<FJsonObject> P1 = (*Pkgs)[1]->AsObject();
			TestEqual(TEXT("packages[0].path (sorted by path: AlphaEmb before ZetaCls)"), GetStr(P0, TEXT("path")), ObjPath(Dest, TEXT("AlphaEmb")));
			TestEqual(TEXT("packages[1].path"), GetStr(P1, TEXT("path")), ObjPath(Dest, TEXT("ZetaCls")));
			TestEqual(TEXT("packages[0].task"), GetStr(P0, TEXT("task")), FString(TEXT("TextEmbedding")));
			TestEqual(TEXT("packages[1].task"), GetStr(P1, TEXT("task")), FString(TEXT("TextClassification")));
			const FString Model0 = GetStr(P0, TEXT("model"));
			TestEqual(TEXT("packages[0].model"), Model0, ObjPath(Dest, TEXT("AlphaEmb_Model")));
			TestTrue(TEXT("packages[0].model ends with _Model object name"), Model0.EndsWith(TEXT("_Model")));
			bool bTok = false, bValid = false;
			TestTrue(TEXT("packages[0].has_tokenizer is a bool"), P0->TryGetBoolField(TEXT("has_tokenizer"), bTok));
			TestTrue(TEXT("packages[0].has_tokenizer true for a text task"), bTok);
			TestTrue(TEXT("packages[0].valid_settings is a bool"), P0->TryGetBoolField(TEXT("valid_settings"), bValid));
			TestTrue(TEXT("packages[0].valid_settings true for a fresh import"), bValid);
			bool bTok1 = false;
			P1->TryGetBoolField(TEXT("has_tokenizer"), bTok1);
			TestTrue(TEXT("packages[1].has_tokenizer true (TextClassification carries tokenizer.json)"), bTok1);
			const FString Dev = GetStr(P0, TEXT("preferred_device"));
			TestTrue(*FString::Printf(TEXT("preferred_device '%s' in {Auto,CPU,GPU}"), *Dev), Dev == TEXT("Auto") || Dev == TEXT("CPU") || Dev == TEXT("GPU"));
			double Labels = -1;
			TestTrue(TEXT("packages[1].labels is a number"), P1->TryGetNumberField(TEXT("labels"), Labels));
			const UPipelineModelPackage* Pkg = LoadObject<UPipelineModelPackage>(nullptr, *ObjPath(Dest, TEXT("ZetaCls")));
			if (Pkg)
			{
				TestEqual(TEXT("packages[1].labels equals the loaded package's Labels.Num()"), (int32)Labels, Pkg->Labels.Num());
			}
			else
			{
				AddError(TEXT("imported ZetaCls package not loadable"));
			}
		});

		It("respects bRecursive and the path itself", [this]()
		{
			FString F1;
			if (!RequireFolder(TEXT("tiny_text_embedding"), F1)) { return; }
			const FPipelineImportResult R = ImportUnsaved(F1, TEXT("Rec"));
			if (!TestTrue(TEXT("import succeeded"), R.bSuccess)) { return; }
			const FString Parent = TEXT("/Game/PipelinesTests");
			// Non-recursive on the parent must not see the package one folder down.
			TSharedPtr<FJsonObject> J = Parse(UPipelineMcpLibrary::ListModelPackages(Parent, false), TEXT("List nonrecursive"));
			if (J.IsValid())
			{
				const TArray<TSharedPtr<FJsonValue>>* Pkgs = nullptr;
				bool bFound = false;
				if (J->TryGetArrayField(TEXT("packages"), Pkgs))
				{
					for (const TSharedPtr<FJsonValue>& V : *Pkgs)
					{
						bFound |= GetStr(V->AsObject(), TEXT("path")) == ObjPath(Dest, TEXT("Rec"));
					}
				}
				TestFalse(TEXT("ListModelPackages(/Game/PipelinesTests, bRecursive=false) must not list a package in a subfolder"), bFound);
			}
			// Recursive on the parent does see it.
			J = Parse(UPipelineMcpLibrary::ListModelPackages(Parent, true), TEXT("List recursive"));
			if (J.IsValid())
			{
				const TArray<TSharedPtr<FJsonValue>>* Pkgs = nullptr;
				bool bFound = false;
				if (J->TryGetArrayField(TEXT("packages"), Pkgs))
				{
					for (const TSharedPtr<FJsonValue>& V : *Pkgs)
					{
						bFound |= GetStr(V->AsObject(), TEXT("path")) == ObjPath(Dest, TEXT("Rec"));
					}
				}
				TestTrue(TEXT("ListModelPackages(/Game/PipelinesTests, bRecursive=true) lists the package in the subfolder"), bFound);
			}
			// Non-recursive on its own folder does.
			J = Parse(UPipelineMcpLibrary::ListModelPackages(Dest, false), TEXT("List own folder"));
			if (J.IsValid())
			{
				double Count = -1;
				J->TryGetNumberField(TEXT("count"), Count);
				TestEqual(TEXT("ListModelPackages(own folder, false) count"), (int32)Count, 1);
			}
		});

		It("returns the path error for bad paths and an empty list for unknown ones", [this]()
		{
			const FString LongPath = TEXT("/Game/") + FString::ChrN(2000, TEXT('a'));
			const TArray<FString> Bad = { TEXT("Game"), TEXT(""), TEXT("/Game/Bad Path!*"), LongPath, TEXT("relative/path"), TEXT("/Game//x") };
			for (const FString& P : Bad)
			{
				const FString Json = UPipelineMcpLibrary::ListModelPackages(P, true);
				const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("List bad path"));
				if (!J.IsValid()) { continue; }
				TestEqual(*FString::Printf(TEXT("ListModelPackages('%s'): error text; got %s"), *P.Left(40), *Json.Left(200)),
					GetStr(J, TEXT("error")), FString(TEXT("Path must be a content path like /Game")));
			}
			const FString Missing = TEXT("/Game/DoesNotExist");
			const FString Json = UPipelineMcpLibrary::ListModelPackages(Missing, true);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("List missing"));
			if (J.IsValid())
			{
				double Count = -1;
				TestTrue(*FString::Printf(TEXT("ListModelPackages(%s) has count; got %s"), *Missing, *Json), J->TryGetNumberField(TEXT("count"), Count));
				TestEqual(TEXT("count for a nonexistent path"), (int32)Count, 0);
				const TArray<TSharedPtr<FJsonValue>>* Pkgs = nullptr;
				TestTrue(TEXT("packages is an empty array"), J->TryGetArrayField(TEXT("packages"), Pkgs) && Pkgs->Num() == 0);
				TestFalse(TEXT("no error field for a valid but empty path"), J->HasField(TEXT("error")));
			}
		});

		It("does not list or touch assets of other classes (engine content folder)", [this]()
		{
			// /Engine/EngineResources holds textures and materials but no UPipelineModelPackage.
			const FString Json = UPipelineMcpLibrary::ListModelPackages(TEXT("/Engine/EngineResources"), true);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("List engine"));
			if (J.IsValid())
			{
				double Count = -1;
				J->TryGetNumberField(TEXT("count"), Count);
				TestEqual(*FString::Printf(TEXT("count in /Engine/EngineResources; got %s"), *Json.Left(200)), (int32)Count, 0);
			}
		});
	});

	Describe("ImportModelFolder confirm=false", [this]()
	{
		It("previews tiny_image_classification and creates nothing", [this]()
		{
			FString Folder;
			if (!RequireFolder(TEXT("tiny_image_classification"), Folder)) { return; }
			const FString Json = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, TEXT(""), false, false);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Import preview"));
			if (!J.IsValid()) { return; }
			bool bSuccess = true, bNeeds = false, bExists = true;
			TestTrue(TEXT("success is a bool"), J->TryGetBoolField(TEXT("success"), bSuccess));
			TestFalse(TEXT("success false (nothing imported)"), bSuccess);
			TestTrue(TEXT("needs_confirm is a bool"), J->TryGetBoolField(TEXT("needs_confirm"), bNeeds));
			TestTrue(*FString::Printf(TEXT("needs_confirm true for a good folder; got %s"), *Json.Left(600)), bNeeds);
			TestTrue(TEXT("exists is a bool"), J->TryGetBoolField(TEXT("exists"), bExists));
			TestFalse(TEXT("exists false in a fresh temp folder"), bExists);
			TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("ImageClassification")));
			TestEqual(TEXT("message"), GetStr(J, TEXT("message")), FString(MsgNothing));
			const TSharedPtr<FJsonObject> WC = GetObj(J, TEXT("would_create"));
			TestEqual(TEXT("would_create.package"), GetStr(WC, TEXT("package")), ObjPath(Dest, TEXT("tiny_image_classification")));
			TestEqual(TEXT("would_create.model"), GetStr(WC, TEXT("model")), ObjPath(Dest, TEXT("tiny_image_classification_Model")));
			CheckReportShape(GetObj(J, TEXT("report")), TEXT("Import preview"));

			// Nothing exists: not in memory, not in the registry, not on disk.
			TestNull(TEXT("FindObject(package) after preview"), FindObject<UObject>(nullptr, *ObjPath(Dest, TEXT("tiny_image_classification"))));
			TestNull(TEXT("FindObject(model) after preview"), FindObject<UObject>(nullptr, *ObjPath(Dest, TEXT("tiny_image_classification_Model"))));
			TestNull(TEXT("FindPackage(package) after preview"), FindPackage(nullptr, *(Dest / TEXT("tiny_image_classification"))));
			TestNull(TEXT("FindPackage(model) after preview"), FindPackage(nullptr, *(Dest / TEXT("tiny_image_classification_Model"))));
			TestEqual(*FString::Printf(TEXT("asset registry entries under %s after preview"), *Dest), RegistryCount(), 0);
			TestFalse(TEXT(".uasset on disk after preview"), FileFor(TEXT("tiny_image_classification")));
			TestFalse(TEXT("content folder created on disk by the preview"), IFileManager::Get().DirectoryExists(*ContentDir()));
		});

		It("uses a given asset_name and sanitizes the folder leaf otherwise", [this]()
		{
			FString Folder;
			if (!RequireFolder(TEXT("tiny_text_classification"), Folder)) { return; }
			const TSharedPtr<FJsonObject> J = Parse(UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, TEXT("MyName"), false, false), TEXT("Import preview name"));
			if (!J.IsValid()) { return; }
			const TSharedPtr<FJsonObject> WC = GetObj(J, TEXT("would_create"));
			TestEqual(TEXT("would_create.package with AssetName=MyName"), GetStr(WC, TEXT("package")), ObjPath(Dest, TEXT("MyName")));
			TestEqual(TEXT("would_create.model with AssetName=MyName"), GetStr(WC, TEXT("model")), ObjPath(Dest, TEXT("MyName_Model")));
			TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("TextClassification")));
			TestFalse(TEXT("nothing created for MyName"), AnyObjectFor(TEXT("MyName")));
			TestEqual(TEXT("registry empty"), RegistryCount(), 0);
		});

		It("reports parse errors of missing_model without needs_confirm", [this]()
		{
			AddExpectedMessage(TEXT("missing_model"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			FString Folder;
			if (!RequireFolder(TEXT("missing_model"), Folder)) { return; }
			const FString Json = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, TEXT(""), false, false);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Import preview missing_model"));
			if (!J.IsValid()) { return; }
			bool bNeeds = true, bSuccess = true;
			TestTrue(TEXT("needs_confirm is a bool"), J->TryGetBoolField(TEXT("needs_confirm"), bNeeds));
			TestFalse(*FString::Printf(TEXT("needs_confirm false for a folder with errors; got %s"), *Json.Left(600)), bNeeds);
			TestTrue(TEXT("success is a bool"), J->TryGetBoolField(TEXT("success"), bSuccess));
			TestFalse(TEXT("success false"), bSuccess);
			TestEqual(TEXT("message"), GetStr(J, TEXT("message")), FString(MsgCannot));
			const TSharedPtr<FJsonObject> Report = GetObj(J, TEXT("report"));
			CheckReportShape(Report, TEXT("missing_model"));
			TestTrue(*FString::Printf(TEXT("report has an Error with code MissingModelFile; got %s"), *Json.Left(600)),
				CountMessages(Report, TEXT("Error"), TEXT("MissingModelFile")) >= 1);
			TestEqual(TEXT("registry empty"), RegistryCount(), 0);
		});

		It("reports an invalid destination as an InvalidDestination Error", [this]()
		{
			FString Folder;
			if (!RequireFolder(TEXT("tiny_text_embedding"), Folder)) { return; }
			const TArray<FString> BadDests = { TEXT("Game/NoLeadingSlash"), TEXT(""), TEXT("/Game/Bad Dest*?") };
			for (const FString& D : BadDests)
			{
				const FString Json = UPipelineMcpLibrary::ImportModelFolder(Folder, D, TEXT("X"), false, false);
				const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Import bad destination"));
				if (!J.IsValid()) { continue; }
				bool bNeeds = true;
				J->TryGetBoolField(TEXT("needs_confirm"), bNeeds);
				TestFalse(*FString::Printf(TEXT("needs_confirm for destination '%s'; got %s"), *D, *Json.Left(500)), bNeeds);
				TestTrue(*FString::Printf(TEXT("destination '%s': report Error InvalidDestination; got %s"), *D, *Json.Left(500)),
					CountMessages(GetObj(J, TEXT("report")), TEXT("Error"), TEXT("InvalidDestination")) >= 1);
			}
		});

		It("handles empty, missing and garbage source folders without crashing", [this]()
		{
			AddExpectedMessage(TEXT("."), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			const TSharedPtr<FJsonObject> Empty = Parse(UPipelineMcpLibrary::ImportModelFolder(TEXT(""), Dest, TEXT(""), false, false), TEXT("Import empty"));
			if (Empty.IsValid())
			{
				TestEqual(TEXT("ImportModelFolder('') error"), GetStr(Empty, TEXT("error")), FString(TEXT("source_folder is empty")));
			}
			const TSharedPtr<FJsonObject> EmptyConfirm = Parse(UPipelineMcpLibrary::ImportModelFolder(TEXT(""), Dest, TEXT(""), false, true), TEXT("Import empty confirm"));
			if (EmptyConfirm.IsValid())
			{
				TestEqual(TEXT("ImportModelFolder('', confirm=true) error"), GetStr(EmptyConfirm, TEXT("error")), FString(TEXT("source_folder is empty")));
			}
			const TArray<FString> Garbage = { TEXT("Z:/definitely/not/here"), TEXT(":::<>|?*"), FString::ChrN(5000, TEXT('x')), TEXT("   ") };
			for (const FString& G : Garbage)
			{
				for (int32 Confirm = 0; Confirm < 2; ++Confirm)
				{
					const FString Json = UPipelineMcpLibrary::ImportModelFolder(G, Dest, TEXT(""), false, Confirm == 1);
					const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Import garbage"));
					if (!J.IsValid()) { continue; }
					if (J->HasField(TEXT("error"))) { continue; } // an error object is an acceptable answer for garbage
					bool bSuccess = true, bNeeds = true;
					J->TryGetBoolField(TEXT("success"), bSuccess);
					J->TryGetBoolField(TEXT("needs_confirm"), bNeeds);
					TestFalse(*FString::Printf(TEXT("garbage folder '%s' confirm=%d: success; got %s"), *G.Left(30), Confirm, *Json.Left(300)), bSuccess);
					TestFalse(*FString::Printf(TEXT("garbage folder '%s' confirm=%d: needs_confirm; got %s"), *G.Left(30), Confirm, *Json.Left(300)), bNeeds);
				}
			}
			TestEqual(TEXT("registry empty after garbage imports"), RegistryCount(), 0);
		});
	});

	Describe("ImportModelFolder confirm=true", [this]()
	{
		It("imports tiny_text_classification, then reports exists and honours overwrite", [this]()
		{
			AddExpectedMessage(TEXT("tiny_text_classification"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			FString Folder;
			if (!RequireFolder(TEXT("tiny_text_classification"), Folder)) { return; }
			const FString PkgObj = ObjPath(Dest, TEXT("tiny_text_classification"));
			const FString ModelObj = ObjPath(Dest, TEXT("tiny_text_classification_Model"));

			const FString Json = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, TEXT(""), false, true);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Import confirm"));
			if (!J.IsValid()) { return; }
			bool bSuccess = false;
			J->TryGetBoolField(TEXT("success"), bSuccess);
			if (!TestTrue(*FString::Printf(TEXT("success true; got %s"), *Json.Left(800)), bSuccess)) { return; }
			TestEqual(TEXT("package"), GetStr(J, TEXT("package")), PkgObj);
			TestEqual(TEXT("model"), GetStr(J, TEXT("model")), ModelObj);
			TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("TextClassification")));
			CheckReportShape(GetObj(J, TEXT("report")), TEXT("Import confirm"));

			UPipelineModelPackage* Pkg = LoadObject<UPipelineModelPackage>(nullptr, *PkgObj);
			UNNEModelData* Model = LoadObject<UNNEModelData>(nullptr, *ModelObj);
			TestNotNull(TEXT("package object exists after confirm=true"), Pkg);
			TestNotNull(TEXT("model object exists after confirm=true"), Model);
			TestTrue(TEXT("package saved to disk (.uasset)"), FileFor(TEXT("tiny_text_classification")));
			if (!Pkg) { return; }
			TestFalse(TEXT("package not dirty after the saving import"), Pkg->GetOutermost()->IsDirty());

			// Same call without confirm: the assets exist now.
			const FString Json2 = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, TEXT(""), false, false);
			const TSharedPtr<FJsonObject> J2 = Parse(Json2, TEXT("Import exists"));
			if (J2.IsValid())
			{
				bool bExists = false, bNeeds = true;
				J2->TryGetBoolField(TEXT("exists"), bExists);
				J2->TryGetBoolField(TEXT("needs_confirm"), bNeeds);
				TestTrue(*FString::Printf(TEXT("exists true; got %s"), *Json2.Left(500)), bExists);
				TestFalse(TEXT("needs_confirm false without overwrite"), bNeeds);
				TestEqual(TEXT("message when assets exist"), GetStr(J2, TEXT("message")), FString(MsgExists));
			}
			// With overwrite: the normal preview.
			const FString Json3 = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, TEXT(""), true, false);
			const TSharedPtr<FJsonObject> J3 = Parse(Json3, TEXT("Import exists overwrite"));
			if (J3.IsValid())
			{
				bool bExists = false, bNeeds = false;
				J3->TryGetBoolField(TEXT("exists"), bExists);
				J3->TryGetBoolField(TEXT("needs_confirm"), bNeeds);
				TestTrue(*FString::Printf(TEXT("exists true with overwrite; got %s"), *Json3.Left(500)), bExists);
				TestTrue(TEXT("needs_confirm true with overwrite_existing"), bNeeds);
				TestEqual(TEXT("message with overwrite"), GetStr(J3, TEXT("message")), FString(MsgNothing));
			}
			TestFalse(TEXT("previews do not dirty the package"), Pkg->GetOutermost()->IsDirty());

			// confirm=true without overwrite on existing assets fails and leaves the existing ones alone.
			const FString Json4 = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, TEXT(""), false, true);
			const TSharedPtr<FJsonObject> J4 = Parse(Json4, TEXT("Import exists confirm"));
			if (J4.IsValid())
			{
				bool bOk = true;
				J4->TryGetBoolField(TEXT("success"), bOk);
				TestFalse(*FString::Printf(TEXT("confirm=true on existing assets without overwrite: success; got %s"), *Json4.Left(500)), bOk);
				TestTrue(TEXT("package is JSON null on failure"), IsNull(J4, TEXT("package")));
				TestTrue(TEXT("model is JSON null on failure"), IsNull(J4, TEXT("model")));
				TestTrue(TEXT("report has an AssetExists Error"), CountMessages(GetObj(J4, TEXT("report")), TEXT("Error"), TEXT("AssetExists")) >= 1);
			}
			TestTrue(TEXT("existing package still valid after the refused import"), IsValid(Pkg) && LoadObject<UPipelineModelPackage>(nullptr, *PkgObj) == Pkg);

			// confirm=true with overwrite updates the same objects in place.
			const FString Json5 = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, TEXT(""), true, true);
			const TSharedPtr<FJsonObject> J5 = Parse(Json5, TEXT("Import overwrite confirm"));
			if (J5.IsValid())
			{
				bool bOk = false;
				J5->TryGetBoolField(TEXT("success"), bOk);
				TestTrue(*FString::Printf(TEXT("overwrite import success; got %s"), *Json5.Left(800)), bOk);
				TestEqual(TEXT("package path after overwrite"), GetStr(J5, TEXT("package")), PkgObj);
			}
			TestTrue(TEXT("overwrite updates the same package object in place"), LoadObject<UPipelineModelPackage>(nullptr, *PkgObj) == Pkg);
		});

		It("fails cleanly on bad_config_json and leaves nothing behind", [this]()
		{
			AddExpectedMessage(TEXT("bad_config_json"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			FString Folder;
			if (!RequireFolder(TEXT("bad_config_json"), Folder)) { return; }
			const FString Json = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, TEXT(""), false, true);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Import bad_config_json"));
			if (!J.IsValid()) { return; }
			bool bSuccess = true;
			TestTrue(TEXT("success is a bool"), J->TryGetBoolField(TEXT("success"), bSuccess));
			TestFalse(*FString::Printf(TEXT("success false for bad_config_json; got %s"), *Json.Left(600)), bSuccess);
			TestTrue(TEXT("package is JSON null"), IsNull(J, TEXT("package")));
			TestTrue(TEXT("model is JSON null"), IsNull(J, TEXT("model")));
			const FString Task = GetStr(J, TEXT("task"));
			TestTrue(*FString::Printf(TEXT("task '%s' is an EPipelineTask name"), *Task), StaticEnum<EPipelineTask>()->GetValueByNameString(Task) != INDEX_NONE);
			const TSharedPtr<FJsonObject> Report = GetObj(J, TEXT("report"));
			CheckReportShape(Report, TEXT("bad_config_json"));
			TestTrue(TEXT("report has an InvalidJson Error"), CountMessages(Report, TEXT("Error"), TEXT("InvalidJson")) >= 1);
			TestFalse(TEXT("an object exists for the failed import"), AnyObjectFor(TEXT("bad_config_json")));
			TestEqual(TEXT("registry entries after the failed import"), RegistryCount(), 0);
			TestFalse(TEXT(".uasset on disk after the failed import"), FileFor(TEXT("bad_config_json")));
		});

		It("fails cleanly on missing_model with confirm=true", [this]()
		{
			AddExpectedMessage(TEXT("missing_model"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			FString Folder;
			if (!RequireFolder(TEXT("missing_model"), Folder)) { return; }
			const FString Json = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, TEXT(""), false, true);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Import missing_model confirm"));
			if (!J.IsValid()) { return; }
			bool bSuccess = true;
			J->TryGetBoolField(TEXT("success"), bSuccess);
			TestFalse(*FString::Printf(TEXT("success; got %s"), *Json.Left(500)), bSuccess);
			TestTrue(TEXT("package null"), IsNull(J, TEXT("package")));
			TestTrue(TEXT("model null"), IsNull(J, TEXT("model")));
			TestTrue(TEXT("MissingModelFile Error in report"), CountMessages(GetObj(J, TEXT("report")), TEXT("Error"), TEXT("MissingModelFile")) >= 1);
			TestEqual(TEXT("registry empty"), RegistryCount(), 0);
		});
	});

	Describe("ValidateModelPackage", [this]()
	{
		It("validates a fresh tiny package: CPU smoke run, GPU entry when required", [this]()
		{
			FString Folder;
			if (!RequireFolder(TEXT("tiny_text_embedding"), Folder)) { return; }
			const FPipelineImportResult R = ImportUnsaved(Folder, TEXT("ValEmb"));
			if (!TestTrue(TEXT("import succeeded; report: ") + R.Report.ToString(), R.bSuccess)) { return; }
			UPipelineModelPackage* Pkg = R.Package.Get();
			const bool bDirtyBefore = Pkg->GetOutermost()->IsDirty();
			const EPipelineTask TaskBefore = Pkg->Task;
			const int32 LabelsBefore = Pkg->Labels.Num();
			const FString PkgObj = ObjPath(Dest, TEXT("ValEmb"));

			const FString Json = UPipelineMcpLibrary::ValidateModelPackage(PkgObj, true, IsGpuRequired());
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Validate"));
			if (!J.IsValid()) { return; }
			TestEqual(TEXT("package"), GetStr(J, TEXT("package")), PkgObj);
			TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("TextEmbedding")));
			bool bValid = false;
			TestTrue(TEXT("valid is a bool"), J->TryGetBoolField(TEXT("valid"), bValid));
			TestTrue(*FString::Printf(TEXT("valid true; got %s"), *Json.Left(800)), bValid);
			const TSharedPtr<FJsonObject> Report = GetObj(J, TEXT("report"));
			CheckReportShape(Report, TEXT("Validate"));
			double Errors = -1, Warnings = -1;
			TestTrue(TEXT("errors is a number"), J->TryGetNumberField(TEXT("errors"), Errors));
			TestTrue(TEXT("warnings is a number"), J->TryGetNumberField(TEXT("warnings"), Warnings));
			TestEqual(TEXT("errors equals the Error messages in report"), (int32)Errors, CountMessages(Report, TEXT("Error")));
			TestEqual(TEXT("warnings equals the Warning messages in report"), (int32)Warnings, CountMessages(Report, TEXT("Warning")));
			TestEqual(TEXT("errors == 0 when valid"), (int32)Errors, 0);

			const TSharedPtr<FJsonObject> Cpu = FindDeviceCheck(Report, TEXT("CPU"));
			if (TestNotNull(*FString::Printf(TEXT("device_checks has a CPU entry; got %s"), *Json.Left(800)), Cpu.Get()))
			{
				bool bAvail = false, bCreated = false, bRan = false;
				Cpu->TryGetBoolField(TEXT("available"), bAvail);
				Cpu->TryGetBoolField(TEXT("created"), bCreated);
				Cpu->TryGetBoolField(TEXT("ran"), bRan);
				TestTrue(TEXT("CPU available"), bAvail);
				TestTrue(TEXT("CPU created"), bCreated);
				TestTrue(*FString::Printf(TEXT("CPU ran; entry message '%s'"), *GetStr(Cpu, TEXT("message"))), bRan);
				double Ms = -1;
				TestTrue(TEXT("CPU inference_ms is a finite number"), Cpu->TryGetNumberField(TEXT("inference_ms"), Ms) && FMath::IsFinite(Ms));
				TestTrue(TEXT("CPU inference_ms >= 0"), Ms >= 0.0);
			}
			if (IsGpuRequired())
			{
				const TSharedPtr<FJsonObject> Gpu = FindDeviceCheck(Report, TEXT("GPU"));
				TestNotNull(*FString::Printf(TEXT("MODELPIPELINES_REQUIRE_GPU=1: device_checks has a GPU entry; got %s"), *Json.Left(800)), Gpu.Get());
			}
			else
			{
				AddInfo(TEXT("GPU device_check assertion skipped: MODELPIPELINES_REQUIRE_GPU is not 1 (GPU entry not checked)."));
			}

			// Never modifies the asset.
			TestEqual(TEXT("package Task unchanged by validation"), (int32)Pkg->Task, (int32)TaskBefore);
			TestEqual(TEXT("package Labels unchanged by validation"), Pkg->Labels.Num(), LabelsBefore);
			TestEqual(TEXT("package dirty flag unchanged by validation"), Pkg->GetOutermost()->IsDirty(), bDirtyBefore);
		});

		It("accepts the package name form and skips smoke runs when asked", [this]()
		{
			FString Folder;
			if (!RequireFolder(TEXT("tiny_text_classification"), Folder)) { return; }
			const FPipelineImportResult R = ImportUnsaved(Folder, TEXT("ValCls"));
			if (!TestTrue(TEXT("import succeeded"), R.bSuccess)) { return; }
			// Package name only (object name = the leaf); the result reports the object path.
			const FString Json = UPipelineMcpLibrary::ValidateModelPackage(Dest / TEXT("ValCls"), false, false);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Validate pkg name"));
			if (!J.IsValid()) { return; }
			TestEqual(*FString::Printf(TEXT("package field for the package-name form; got %s"), *Json.Left(400)), GetStr(J, TEXT("package")), ObjPath(Dest, TEXT("ValCls")));
			TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("TextClassification")));
			bool bValid = false;
			J->TryGetBoolField(TEXT("valid"), bValid);
			TestTrue(TEXT("valid true with bRunSmokeTest=false"), bValid);
			const TSharedPtr<FJsonObject> Cpu = FindDeviceCheck(GetObj(J, TEXT("report")), TEXT("CPU"));
			if (Cpu.IsValid())
			{
				bool bRan = true;
				Cpu->TryGetBoolField(TEXT("ran"), bRan);
				TestFalse(TEXT("CPU entry present with bRunSmokeTest=false must not claim a smoke run"), bRan);
			}
		});

		It("returns 'Model package not found' for empty, missing, malformed and wrong-class references", [this]()
		{
			AddExpectedMessage(TEXT("Nope"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			AddExpectedMessage(TEXT("DefaultTexture"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			AddExpectedMessage(TEXT("not a path"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			AddExpectedMessage(TEXT("DoesNotExist"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			const TArray<FString> Args = {
				TEXT(""),
				TEXT("/Game/Nope"),
				TEXT("/Game/Nope.Nope"),
				TEXT("not a path"),
				TEXT("/Game/DoesNotExist/Pkg.Pkg"),
				TEXT("/Engine/EngineResources/DefaultTexture.DefaultTexture"), // a real asset of another class
				TEXT("/Engine/EngineResources/DefaultTexture"),
				TEXT("/Game/Bad Name!/x.x"),
			};
			for (const FString& A : Args)
			{
				const FString Json = UPipelineMcpLibrary::ValidateModelPackage(A, true, false);
				const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Validate not found"));
				if (!J.IsValid()) { continue; }
				TestEqual(*FString::Printf(TEXT("ValidateModelPackage('%s'): got %s"), *A, *Json.Left(300)),
					GetStr(J, TEXT("error")), FString(TEXT("Model package not found: ")) + A);
			}
		});

		It("does not crash on a 2000-char or non-ASCII reference", [this]()
		{
			AddExpectedMessage(TEXT("."), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			const TArray<FString> Args = { TEXT("/Game/") + FString::ChrN(2000, TEXT('a')), TEXT("/Game/") + FString::ChrN(2000, TEXT('b')) + TEXT(".x"),
				FString(TEXT("/Game/\u6a21\u578b/\u6a21\u578b.\u6a21\u578b")), FString::ChrN(4000, TEXT('/')) };
			for (const FString& A : Args)
			{
				const FString Json = UPipelineMcpLibrary::ValidateModelPackage(A, true, false);
				const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Validate odd reference"));
				if (!J.IsValid()) { continue; }
				TestTrue(*FString::Printf(TEXT("odd reference (len %d) gives an error field; got %s"), A.Len(), *Json.Left(200)), J->HasTypedField<EJson::String>(TEXT("error")));
			}
			// The non-ASCII argument is echoed back and round-trips through the JSON text.
			const FString NonAscii = TEXT("/Game/\u6a21\u578b/\u6a21\u578b.\u6a21\u578b");
			const TSharedPtr<FJsonObject> J = Parse(UPipelineMcpLibrary::ValidateModelPackage(NonAscii, false, false), TEXT("Validate non-ASCII"));
			if (J.IsValid() && GetStr(J, TEXT("error")).Contains(TEXT("Model package not found")))
			{
				TestTrue(TEXT("non-ASCII reference is echoed back unchanged in the error text"), GetStr(J, TEXT("error")).EndsWith(NonAscii));
			}
		});
	});

	Describe("Non-ASCII", [this]()
	{
		It("round-trips a non-ASCII asset_name through the JSON", [this]()
		{
			AddExpectedMessage(TEXT("tiny_text_classification|package was marked as deleted in editor"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, -1);
			FString Folder;
			if (!RequireFolder(TEXT("tiny_text_classification"), Folder)) { return; }
			const FString Name = TEXT("\u6a21\u578b");
			const FString Json = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, Name, false, false);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Import non-ASCII name preview"));
			if (!J.IsValid()) { return; }
			bool bNeeds = false;
			J->TryGetBoolField(TEXT("needs_confirm"), bNeeds);
			const TSharedPtr<FJsonObject> Report = GetObj(J, TEXT("report"));
			if (bNeeds)
			{
				const TSharedPtr<FJsonObject> WC = GetObj(J, TEXT("would_create"));
				TestEqual(TEXT("would_create.package keeps the non-ASCII name (round trip)"), GetStr(WC, TEXT("package")), ObjPath(Dest, Name));
				TestEqual(TEXT("would_create.model keeps the non-ASCII name"), GetStr(WC, TEXT("model")), ObjPath(Dest, Name + TEXT("_Model")));
			}
			else
			{
				// A refusal is fine only if it is explained in the report, not silent.
				TestTrue(*FString::Printf(TEXT("non-ASCII name refused without an Error in report; got %s"), *Json.Left(500)), CountMessages(Report, TEXT("Error")) >= 1);
			}
			// UTF-8 round trip of the whole output text.
			FTCHARToUTF8 Utf8(*Json);
			const FUTF8ToTCHAR Conv(Utf8.Get(), Utf8.Length());   // explicit length -> not null-terminated, so pass the length on
			const FString Back(Conv.Length(), Conv.Get());
			TestEqual(TEXT("output survives a UTF-8 round trip"), Back, Json);

			// And a real import with that name: no crash; success means the exact name, failure means a report Error.
			const FString Json2 = UPipelineMcpLibrary::ImportModelFolder(Folder, Dest, Name, false, true);
			const TSharedPtr<FJsonObject> J2 = Parse(Json2, TEXT("Import non-ASCII name confirm"));
			if (J2.IsValid())
			{
				bool bOk = false;
				J2->TryGetBoolField(TEXT("success"), bOk);
				if (bOk)
				{
					TestEqual(TEXT("package path of the non-ASCII import"), GetStr(J2, TEXT("package")), ObjPath(Dest, Name));
				}
				else
				{
					TestTrue(*FString::Printf(TEXT("failed non-ASCII import explains itself; got %s"), *Json2.Left(500)), CountMessages(GetObj(J2, TEXT("report")), TEXT("Error")) >= 1);
					TestTrue(TEXT("package is null on failure"), IsNull(J2, TEXT("package")));
				}
			}
		});

		It("accepts a source folder path with non-ASCII characters", [this]()
		{
			FString Source;
			if (!RequireFolder(TEXT("tiny_text_classification"), Source)) { return; }
			const FString Root = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("PipelinesTests") / FGuid::NewGuid().ToString());
			TempDirs.Add(Root);
			const FString Copy = Root / (FString(TEXT("\u6a21\u578b_")) + DestGuid);
			IFileManager::Get().MakeDirectory(*Copy, true);
			if (!FPlatformFileManager::Get().GetPlatformFile().CopyDirectoryTree(*Copy, *Source, true))
			{
				AddError(FString::Printf(TEXT("could not copy '%s' to '%s'"), *Source, *Copy));
				return;
			}
			const FString Json = UPipelineMcpLibrary::ImportModelFolder(Copy, Dest, TEXT("NonAsciiSrc"), false, false);
			const TSharedPtr<FJsonObject> J = Parse(Json, TEXT("Import non-ASCII folder"));
			if (!J.IsValid()) { return; }
			bool bNeeds = false;
			J->TryGetBoolField(TEXT("needs_confirm"), bNeeds);
			TestTrue(*FString::Printf(TEXT("needs_confirm true for a good folder with non-ASCII path; got %s"), *Json.Left(500)), bNeeds);
			TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("TextClassification")));
			TestEqual(TEXT("would_create.package"), GetStr(GetObj(J, TEXT("would_create")), TEXT("package")), ObjPath(Dest, TEXT("NonAsciiSrc")));
			TestEqual(TEXT("registry empty after preview"), RegistryCount(), 0);
		});
	});

	Describe("Off the game thread", [this]()
	{
		It("answers with the game-thread error from all three functions and touches nothing", [this]()
		{
			FString Folder;
			if (!RequireFolder(TEXT("tiny_text_classification"), Folder)) { return; }
			const FString DestCopy = Dest;
			struct FCase { const TCHAR* Name; TFunction<FString()> Call; };
			const TArray<FCase> Cases = {
				{ TEXT("ListModelPackages"), [DestCopy]() { return UPipelineMcpLibrary::ListModelPackages(DestCopy, true); } },
				{ TEXT("ImportModelFolder confirm=false"), [Folder, DestCopy]() { return UPipelineMcpLibrary::ImportModelFolder(Folder, DestCopy, TEXT("BG"), false, false); } },
				{ TEXT("ImportModelFolder confirm=true"), [Folder, DestCopy]() { return UPipelineMcpLibrary::ImportModelFolder(Folder, DestCopy, TEXT("BG"), false, true); } },
				{ TEXT("ValidateModelPackage"), []() { return UPipelineMcpLibrary::ValidateModelPackage(TEXT("/Game/Nope.Nope"), true, true); } },
				{ TEXT("ValidateModelPackage empty"), []() { return UPipelineMcpLibrary::ValidateModelPackage(TEXT(""), false, false); } },
			};
			for (const FCase& C : Cases)
			{
				FString Json;
				if (!RunOffGameThread(C.Call, Json))
				{
					AddError(FString::Printf(TEXT("%s from a background thread did not return within 30 s"), C.Name));
					continue;
				}
				const TSharedPtr<FJsonObject> J = Parse(Json, C.Name);
				if (J.IsValid())
				{
					TestEqual(*FString::Printf(TEXT("%s off the game thread: error; got %s"), C.Name, *Json.Left(300)), GetStr(J, TEXT("error")), FString(MsgNotGameThread));
				}
			}
			TestFalse(TEXT("an asset was created by a background-thread call"), AnyObjectFor(TEXT("BG")));
			TestEqual(TEXT("registry empty after background-thread calls"), RegistryCount(), 0);
			TestFalse(TEXT(".uasset on disk after background-thread confirm=true"), FileFor(TEXT("BG")));
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
