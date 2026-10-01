// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Misc/Optional.h"
#include "PipelineModelImporter.h"

/** Content Browser folder context menu entry "Import Hugging Face Model Folder..." (module-private). */
namespace UE::ModelPipelines::Editor
{
	/** The Content Browser folder context menu the entry is added to. */
	inline const FName ImportMenuName = TEXT("ContentBrowser.FolderContextMenu");
	/** Section of that menu holding the entry. */
	inline const FName ImportSectionName = TEXT("ModelPipelines");
	/** Name of the menu entry. */
	inline const FName ImportEntryName = TEXT("ModelPipelines.ImportModelFolder");

	/** Registers the entry (via UToolMenus::RegisterStartupCallback, owned by the module's FToolMenuOwner). Idempotent. */
	void RegisterImportMenu();

	/** UToolMenus::UnregisterOwner for that owner. Safe if never registered or if UToolMenus is gone. */
	void UnregisterImportMenu();

	/**
	 * The entry's work after the folder dialog: imports SourceFolder into DestinationPath (a Content Browser package path
	 * such as "/Game/Models") with automatic task detection, bSave = true unless bSaveOverride is set, inside an
	 * FScopedSlowTask. Game thread only (the importer refuses otherwise).
	 * @param bInteractive false: no dialogs, progress window or toasts (tests); true: progress dialog, toast on success,
	 *                     FMessageDialog with the full report on failure.
	 */
	FPipelineImportResult ImportFolderWithProgress(const FString& SourceFolder, const FString& DestinationPath,
		bool bInteractive, TOptional<bool> bSaveOverride = TOptional<bool>());
}
