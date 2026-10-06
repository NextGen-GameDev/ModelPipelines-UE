// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/UnrealString.h"
#include "Dom/JsonObject.h"
#include "HAL/FileManager.h"
#include "Misc/Guid.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "Internationalization/Text.h"
#include "ToolMenu.h"
#include "ToolMenus.h"
#include "ToolMenuSection.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"
#include "PipelineModelImporter.h"
#include "PipelineImportMenu.h"
#include "PipelineEditorTestData.h"

namespace PipelineImportMenuSpecPrivate
{
	using namespace UE::ModelPipelines::Editor;
	using namespace UE::ModelPipelines::EditorTests;

	/** Number of entries named ImportEntryName in section ImportSectionName of the folder context menu; -1 if no menu/section. */
	static int32 CountEntries(UToolMenus* Menus)
	{
		if (!Menus)
		{
			return -1;
		}
		UToolMenu* Menu = Menus->FindMenu(ImportMenuName);
		if (!Menu)
		{
			return -1;
		}
		FToolMenuSection* Section = Menu->FindSection(ImportSectionName);
		if (!Section)
		{
			return -1;
		}
		int32 Count = 0;
		for (const FToolMenuEntry& Block : Section->Blocks)
		{
			if (Block.Name == ImportEntryName)
			{
				++Count;
			}
		}
		return Count;
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
}

BEGIN_DEFINE_SPEC(FPipelineImportMenuSpec, "Pipelines.Editor.Menu",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	/** Entry count before the test touched anything (-1 = no menu/section); the state restored in AfterEach. */
	int32 CountAtStart = -1;
	FString DestGuid;
	FString Dest;
	TArray<FPipelineImportResult> Imported;
	bool LoadCaseFolder(const FString& Id, FString& OutFolder);
END_DEFINE_SPEC(FPipelineImportMenuSpec)

bool FPipelineImportMenuSpec::LoadCaseFolder(const FString& Id, FString& OutFolder)
{
	FString Err;
	TSharedPtr<FJsonObject> Root = UE::ModelPipelines::EditorTests::LoadImportExpectations(Err);
	if (!Root.IsValid())
	{
		AddError(Err);
		return false;
	}
	TSharedPtr<FJsonObject> Case = UE::ModelPipelines::EditorTests::FindCase(Root, Id);
	if (!Case.IsValid())
	{
		AddError(FString::Printf(TEXT("case '%s' missing from the import expectations; regenerate with: python Tools/make_import_test_data.py"), *Id));
		return false;
	}
	OutFolder = PipelineImportMenuSpecPrivate::ResolveCaseFolder(Root, Case);
	if (!IFileManager::Get().DirectoryExists(*OutFolder))
	{
		AddError(FString::Printf(TEXT("[%s] test model folder '%s' missing; generate with: python Tools/make_import_test_data.py"), *Id, *OutFolder));
		return false;
	}
	return true;
}

void FPipelineImportMenuSpec::Define()
{
	using namespace PipelineImportMenuSpecPrivate;

	BeforeEach([this]()
	{
		CountAtStart = CountEntries(UToolMenus::TryGet());
		DestGuid = FGuid::NewGuid().ToString(EGuidFormats::Digits);
		Dest = FString::Printf(TEXT("/Game/PipelinesTests/%s"), *DestGuid);
		Imported.Reset();
	});

	// Always runs, also after a failed test: discard imports and put the menu back the way we found it.
	AfterEach([this]()
	{
		for (const FPipelineImportResult& R : Imported)
		{
			DiscardImported(R);
		}
		Imported.Reset();
		const int32 Now = CountEntries(UToolMenus::TryGet());
		if (CountAtStart >= 1 && Now < 1)
		{
			RegisterImportMenu();
		}
		else if (CountAtStart < 1 && Now >= 1)
		{
			UnregisterImportMenu();
		}
	});

	// ------------------------------------------------------------------------------------------------ acceptance 1
	Describe("Registration", [this]()
	{
		// Catches an entry that is never registered (StartupModule forgot RegisterImportMenu, or a wrong menu/section/entry name).
		It("has the Import entry in the folder context menu section after registration", [this]()
		{
			UToolMenus* Menus = UToolMenus::TryGet();
			if (!TestNotNull(TEXT("UToolMenus exists in the editor"), Menus)) { return; }
			RegisterImportMenu(); // idempotent; covers -nullrhi runs where startup callbacks may not have fired
			UToolMenu* Menu = Menus->FindMenu(ImportMenuName);
			if (!TestNotNull(TEXT("menu ContentBrowser.FolderContextMenu"), Menu)) { return; }
			FToolMenuSection* Section = Menu->FindSection(ImportSectionName);
			if (!TestNotNull(TEXT("section ModelPipelines"), Section)) { return; }
			TestNotNull(TEXT("entry ModelPipelines.ImportModelFolder in the section"), Section->FindEntry(ImportEntryName));
		});

		// Catches a non-idempotent register that adds a second copy each time (the user would see two menu items).
		It("keeps exactly one entry after RegisterImportMenu is called twice more", [this]()
		{
			if (!TestNotNull(TEXT("UToolMenus exists"), UToolMenus::TryGet())) { return; }
			RegisterImportMenu();
			RegisterImportMenu();
			RegisterImportMenu();
			TestEqual(TEXT("entry count in the section"), CountEntries(UToolMenus::TryGet()), 1);
		});

		// Catches label/tooltip drift from the spec; the strings are what the user reads.
		It("labels the entry 'Import Hugging Face Model Folder...' with the specified tooltip", [this]()
		{
			UToolMenus* Menus = UToolMenus::TryGet();
			if (!TestNotNull(TEXT("UToolMenus exists"), Menus)) { return; }
			RegisterImportMenu();
			UToolMenu* Menu = Menus->FindMenu(ImportMenuName);
			if (!TestNotNull(TEXT("menu"), Menu)) { return; }
			FToolMenuSection* Section = Menu->FindSection(ImportSectionName);
			if (!TestNotNull(TEXT("section"), Section)) { return; }
			FToolMenuEntry* Entry = Section->FindEntry(ImportEntryName);
			if (!TestNotNull(TEXT("entry"), Entry)) { return; }
			if (Entry->Label.IsSet())
			{
				TestEqual(TEXT("label"), Entry->Label.Get().ToString(), FString(TEXT("Import Hugging Face Model Folder...")));
			}
			else
			{
				AddError(TEXT("entry has no explicit Label"));
			}
			if (Entry->ToolTip.IsSet())
			{
				TestEqual(TEXT("tooltip"), Entry->ToolTip.Get().ToString(),
					FString(TEXT("Import a folder with model.onnx, config.json and tokenizer/preprocessor files as a Model Pipelines package")));
			}
			else
			{
				AddError(TEXT("entry has no explicit ToolTip"));
			}
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 2
	Describe("Unregistration", [this]()
	{
		// Catches UnregisterOwner using the wrong owner (entry stays) and a register that cannot re-add after unregister.
		It("removes the entry on UnregisterImportMenu and adds it back on RegisterImportMenu", [this]()
		{
			if (!TestNotNull(TEXT("UToolMenus exists"), UToolMenus::TryGet())) { return; }
			RegisterImportMenu();
			TestEqual(TEXT("one entry before unregistering"), CountEntries(UToolMenus::TryGet()), 1);
			UnregisterImportMenu();
			TestTrue(TEXT("entry (or its section/menu) gone after unregister"), CountEntries(UToolMenus::TryGet()) <= 0);
			if (UToolMenu* Menu = UToolMenus::Get()->FindMenu(ImportMenuName))
			{
				const FToolMenuSection* S = Menu->FindSection(ImportSectionName);   // UToolMenu::FindEntry is private (ToolMenu.h:164)
				TestTrue(TEXT("no entry after unregister"), S == nullptr || S->FindEntry(ImportEntryName) == nullptr);
			}
			RegisterImportMenu();
			TestEqual(TEXT("exactly one entry after re-register"), CountEntries(UToolMenus::TryGet()), 1);
		});

		// Catches a crash or assert when unregistering something that is not registered.
		It("tolerates UnregisterImportMenu twice in a row", [this]()
		{
			UnregisterImportMenu();
			UnregisterImportMenu();
			TestTrue(TEXT("entry absent"), CountEntries(UToolMenus::TryGet()) <= 0);
			RegisterImportMenu();
			TestEqual(TEXT("one entry after re-register"), CountEntries(UToolMenus::TryGet()), 1);
		});
	});

	// ------------------------------------------------------------------------------------------------ acceptance 3
	Describe("ImportFolderWithProgress", [this]()
	{
		// Catches wrong destination handling (asset landing in /Game root or under a different name) and a success path
		// that opens a dialog or toast although bInteractive is false (that would hang this unattended run).
		It("imports tiny_text_classification into the destination as <dest>/tiny_text_classification", [this]()
		{
			FString Folder;
			if (!LoadCaseFolder(TEXT("tiny_text_classification"), Folder)) { return; }
			const FPipelineImportResult R = ImportFolderWithProgress(Folder, Dest, false, TOptional<bool>(false));
			Imported.Add(R);
			TestTrue(*FString::Printf(TEXT("import succeeds; report:\n%s"), *R.Report.ToString()), R.bSuccess);
			if (!R.Package || !R.Model)
			{
				AddError(TEXT("Package or Model is null on success"));
				return;
			}
			const FString ObjPath = FString::Printf(TEXT("%s/tiny_text_classification.tiny_text_classification"), *Dest);
			TestEqual(TEXT("package object path"), R.Package->GetPathName(), ObjPath);
			TestEqual(TEXT("package long name"), R.Package->GetOutermost()->GetName(),
				FString::Printf(TEXT("%s/tiny_text_classification"), *Dest));
			TestEqual(TEXT("error count in the report"), R.Report.NumErrors(), 0);
		});

		// Catches bSaveOverride being ignored: with save false nothing may be written under Content.
		It("does not write files when bSaveOverride is false", [this]()
		{
			FString Folder;
			if (!LoadCaseFolder(TEXT("tiny_text_classification"), Folder)) { return; }
			const FPipelineImportResult R = ImportFolderWithProgress(Folder, Dest, false, TOptional<bool>(false));
			Imported.Add(R);
			TestTrue(*R.Report.ToString(), R.bSuccess);
			const FString Dir = FPaths::ConvertRelativePathToFull(FPaths::ProjectContentDir() / TEXT("PipelinesTests") / DestGuid);
			TArray<FString> Files;
			IFileManager::Get().FindFilesRecursive(Files, *Dir, TEXT("*.*"), true, false);
			TestEqual(TEXT("files under the destination on disk"), Files.Num(), 0);
		});

		// Catches validation being skipped by the menu path: the wrong-IO model must be rejected with MissingInput,
		// and a failed import must leave no package, model or file behind.
		It("rejects wrong_io_for_task with MissingInput and creates nothing", [this]()
		{
			UE::ModelPipelines::EditorTests::ExpectCaseImportLogs(*this, TEXT("wrong_io_for_task"), 1);
			FString Folder;
			if (!LoadCaseFolder(TEXT("wrong_io_for_task"), Folder)) { return; }
			const FPipelineImportResult R = ImportFolderWithProgress(Folder, Dest, false, TOptional<bool>(false));
			Imported.Add(R);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(*FString::Printf(TEXT("report has MissingInput; report:\n%s"), *R.Report.ToString()),
				R.Report.HasCode(EPipelineValidationCode::MissingInput));
			TestNull(TEXT("Package is null"), R.Package.Get());
			TestNull(TEXT("Model is null"), R.Model.Get());
			const FString Leaf = FPaths::GetCleanFilename(Folder);
			const FString LongName = Dest / Leaf;
			TestNull(TEXT("package not in memory"), FindPackage(nullptr, *LongName));
			const FString Dir = FPaths::ConvertRelativePathToFull(FPaths::ProjectContentDir() / TEXT("PipelinesTests") / DestGuid);
			TArray<FString> Files;
			IFileManager::Get().FindFilesRecursive(Files, *Dir, TEXT("*.*"), true, false);
			TestEqual(TEXT("files on disk"), Files.Num(), 0);
		});

		// Catches a crash on a bad path typed by nobody but reachable through the dialog (a folder deleted meanwhile):
		// must come back as a failed result, not a check() or a modal dialog.
		It("fails cleanly for a source folder that does not exist", [this]()
		{
			UE::ModelPipelines::EditorTests::ExpectImportFinding(*this, TEXT("FolderNotFound"), true, 1);
			UE::ModelPipelines::EditorTests::ExpectImportFailed(*this, 1);
			const FString Missing = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("PipelinesTests") / DestGuid / TEXT("does_not_exist"));
			const FPipelineImportResult R = ImportFolderWithProgress(Missing, Dest, false, TOptional<bool>(false));
			Imported.Add(R);
			TestFalse(TEXT("bSuccess"), R.bSuccess);
			TestTrue(TEXT("report has at least one message explaining why"), R.Report.Messages.Num() > 0);
			TestNull(TEXT("Package is null"), R.Package.Get());
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
