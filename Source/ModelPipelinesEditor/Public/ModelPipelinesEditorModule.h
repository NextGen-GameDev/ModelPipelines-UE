// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"

/** Editor module of Model Pipelines: model folder import, validation and the Content Browser import menu entry. */
class FModelPipelinesEditorModule : public IModuleInterface
{
public:
	/** Registers the Content Browser "Import Hugging Face Model Folder..." entry (skipped in commandlets). */
	virtual void StartupModule() override;
	/** Removes the menu entries this module added. */
	virtual void ShutdownModule() override;
};
