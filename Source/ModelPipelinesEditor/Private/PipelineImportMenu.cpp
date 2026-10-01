// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineImportMenu.h"

#include "ContentBrowserMenuContexts.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Framework/Application/SlateApplication.h"
#include "Framework/Notifications/NotificationManager.h"
#include "Misc/MessageDialog.h"
#include "Misc/Paths.h"
#include "Misc/ScopedSlowTask.h"
#include "PipelineEditorLog.h"
#include "PipelineModelPackage.h"
#include "Textures/SlateIcon.h"
#include "ToolMenu.h"
#include "ToolMenuSection.h"
#include "ToolMenus.h"
#include "Widgets/Notifications/SNotificationList.h"

#define LOCTEXT_NAMESPACE "ModelPipelinesImportMenu"

namespace UE::ModelPipelines::Editor
{
	namespace ImportMenuPrivate
	{
		/** Owner of every menu entry this module adds; UnregisterImportMenu removes everything with this owner. */
		static FToolMenuOwner GetOwner()
		{
			return FToolMenuOwner(FName(TEXT("ModelPipelinesEditor")));
		}

		/** Handle of a deferred startup callback (only set while UToolMenus is not ready yet). */
		static FDelegateHandle GPendingStartupHandle;

		/** True when the clicked folder context allows creating assets: modifiable and at least one asset (not class) path. */
		static bool CanImportInto(const UContentBrowserFolderContext* FolderContext)
		{
			return FolderContext && FolderContext->bCanBeModified && FolderContext->NumAssetPaths > 0;
		}

		/** Menu can-execute: the entry is disabled for read-only folders and class-only selections. */
		static bool CanExecuteImport(const FToolMenuContext& Context)
		{
			return CanImportInto(Context.FindContext<UContentBrowserFolderContext>());
		}

		/** Menu action: clicked folder -> directory dialog -> import. Does nothing when there is no folder or the dialog is cancelled. */
		static void ExecuteImport(const FToolMenuContext& Context)
		{
			const UContentBrowserFolderContext* FolderContext = Context.FindContext<UContentBrowserFolderContext>();
			if (!CanImportInto(FolderContext) || FolderContext->GetSelectedPackagePaths().Num() == 0)
			{
				return;
			}
			const FString DestinationPath = FolderContext->GetSelectedPackagePaths()[0];

			IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
			if (!DesktopPlatform)
			{
				UE_LOG(LogModelPipelinesEditor, Warning, TEXT("Import Hugging Face Model Folder: no desktop platform to open a folder dialog."));
				return;
			}
			const void* ParentWindow = FSlateApplication::IsInitialized()
				? FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr)
				: nullptr;
			FString Folder;
			const bool bPicked = DesktopPlatform->OpenDirectoryDialog(ParentWindow,
				LOCTEXT("PickFolderTitle", "Choose a Hugging Face model folder").ToString(), FPaths::ProjectDir(), Folder);
			if (!bPicked || Folder.IsEmpty())
			{
				return;
			}
			ImportFolderWithProgress(Folder, DestinationPath, true);
		}

		/** Adds the section and entry. AddEntry replaces an entry of the same name, so repeated calls keep one entry. */
		static void AddEntries()
		{
			UToolMenus* Menus = UToolMenus::TryGet();
			if (!Menus)
			{
				return;
			}
			FToolMenuOwnerScoped OwnerScoped(GetOwner());
			UToolMenu* Menu = Menus->ExtendMenu(ImportMenuName);
			if (!Menu)
			{
				return;
			}
			FToolMenuSection& Section = Menu->FindOrAddSection(ImportSectionName, LOCTEXT("SectionLabel", "Model Pipelines"));
			FToolUIAction Action(FToolMenuExecuteAction::CreateStatic(&ExecuteImport));
			Action.CanExecuteAction = FToolMenuCanExecuteAction::CreateStatic(&CanExecuteImport);
			Section.AddEntry(FToolMenuEntry::InitMenuEntry(
				ImportEntryName,
				LOCTEXT("ImportLabel", "Import Hugging Face Model Folder..."),
				LOCTEXT("ImportToolTip", "Import a folder with model.onnx, config.json and tokenizer/preprocessor files as a Model Pipelines package"),
				FSlateIcon(),
				Action));
		}
	}

	void RegisterImportMenu()
	{
		using namespace ImportMenuPrivate;
		if (GPendingStartupHandle.IsValid())
		{
			UToolMenus::UnRegisterStartupCallback(GPendingStartupHandle);
			GPendingStartupHandle.Reset();
		}
		// Runs AddEntries now when the menu UI is up, otherwise after engine init.
		GPendingStartupHandle = UToolMenus::RegisterStartupCallback(FSimpleMulticastDelegate::FDelegate::CreateStatic(&AddEntries));
		// Without menu UI (e.g. -nullrhi) the callback is deferred; the menu data still exists, so add the entry directly.
		if (GPendingStartupHandle.IsValid() && UToolMenus::TryGet())
		{
			AddEntries();
		}
	}

	void UnregisterImportMenu()
	{
		using namespace ImportMenuPrivate;
		if (GPendingStartupHandle.IsValid())
		{
			UToolMenus::UnRegisterStartupCallback(GPendingStartupHandle);
			GPendingStartupHandle.Reset();
		}
		UToolMenus::UnregisterOwner(GetOwner()); // no-op when UToolMenus is gone
	}

	FPipelineImportResult ImportFolderWithProgress(const FString& SourceFolder, const FString& DestinationPath,
		bool bInteractive, TOptional<bool> bSaveOverride)
	{
		FPipelineImportOptions Options;
		Options.SourceFolder = SourceFolder;
		Options.DestinationPath = DestinationPath;
		Options.Task = EPipelineTask::None; // auto-detect
		Options.bSave = bSaveOverride.Get(true);

		FPipelineImportResult Result;
		if (!IsInGameThread())
		{
			// FScopedSlowTask and the importer are game-thread only; touch nothing.
			Result.Report.Add(EPipelineValidationSeverity::Error, EPipelineValidationCode::None,
				TEXT("ImportFolderWithProgress must run on the game thread. Call it from the editor's game thread (e.g. via AsyncTask(ENamedThreads::GameThread, ...))."));
			return Result;
		}
		{
			FScopedSlowTask SlowTask(1.f, FText::Format(LOCTEXT("Importing", "Importing model folder {0}..."),
				FText::FromString(FPaths::GetCleanFilename(SourceFolder))), bInteractive);
			if (bInteractive)
			{
				SlowTask.MakeDialog();
			}
			SlowTask.EnterProgressFrame(1.f);
			Result = ImportPipelineModelFolder(Options);
		}

		if (bInteractive)
		{
			if (Result.bSuccess && Result.Package)
			{
				FNotificationInfo Info(FText::Format(LOCTEXT("ImportSucceeded", "Imported {0}"),
					FText::FromString(Result.Package->GetPathName())));
				Info.ExpireDuration = 5.f;
				FSlateNotificationManager::Get().AddNotification(Info);
			}
			else
			{
				FMessageDialog::Open(EAppMsgType::Ok, FText::FromString(Result.Report.ToString()),
					LOCTEXT("ImportFailedTitle", "Model folder import failed"));
			}
		}
		return Result;
	}
}

#undef LOCTEXT_NAMESPACE
