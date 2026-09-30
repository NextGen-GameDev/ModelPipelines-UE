// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"
#include "Templates/UniquePtr.h"
#include "PipelineExecutor.h"
#include "PipelineNNEBackend.h"

/** ModelPipelinesCore module. Owns the plugin's one FPipelineExecutor and its default backend. */
class MODELPIPELINESCORE_API FModelPipelinesCoreModule : public IModuleInterface
{
public:
	/** FModuleManager::LoadModuleChecked. Any thread once the module has been loaded on the game thread. */
	static FModelPipelinesCoreModule& Get();

	/** The executor every inference goes through. Valid between StartupModule and ShutdownModule. */
	FPipelineExecutor& GetExecutor();

	/** An FPipelineNNEBackend with the default config. Valid between StartupModule and ShutdownModule. */
	IPipelineBackend& GetDefaultBackend();

	/** Creates the executor and the default backend. */
	virtual void StartupModule() override;

	/** Calls GetExecutor().Shutdown() before destroying anything. */
	virtual void ShutdownModule() override;

private:
	TUniquePtr<FPipelineExecutor> Executor;
	TUniquePtr<FPipelineNNEBackend> DefaultBackend;
};
