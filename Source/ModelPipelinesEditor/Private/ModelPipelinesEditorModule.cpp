// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesEditorModule.h"

#include "CoreGlobals.h"
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
