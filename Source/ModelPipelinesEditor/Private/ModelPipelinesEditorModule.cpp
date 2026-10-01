// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesEditorModule.h"

#include "Misc/CoreMisc.h"
#include "Modules/ModuleManager.h"
#include "PipelineImportMenu.h"

void FModelPipelinesEditorModule::StartupModule()
{
	// Commandlets have no Content Browser; the menu entry is only for interactive editor sessions.
	if (!IsRunningCommandlet())
	{
		UE::ModelPipelines::Editor::RegisterImportMenu();
	}
}

void FModelPipelinesEditorModule::ShutdownModule()
{
	UE::ModelPipelines::Editor::UnregisterImportMenu();
}

IMPLEMENT_MODULE(FModelPipelinesEditorModule, ModelPipelinesEditor)
