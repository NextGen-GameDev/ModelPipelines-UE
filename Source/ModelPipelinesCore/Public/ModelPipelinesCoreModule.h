// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

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

	/**
	 * The loaded module, or nullptr when it is not loaded or its ShutdownModule has started. Never loads the module.
	 * Any thread.
	 */
	static FModelPipelinesCoreModule* TryGet();

	/** The executor every inference goes through. Valid between StartupModule and ShutdownModule. */
	FPipelineExecutor& GetExecutor();

	/** An FPipelineNNEBackend with the default config. Valid between StartupModule and ShutdownModule. */
	IPipelineBackend& GetDefaultBackend();

	/** Creates the executor and the default backend. */
	virtual void StartupModule() override;

	/**
	 * Clears the FPipelineExecutor::TryGetShared() slot, then calls GetExecutor().Shutdown() before releasing anything.
	 * Background holders of a TryGetShared() reference may outlive this call; the executor then dies with them.
	 */
	virtual void ShutdownModule() override;

private:
	TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor;
	TUniquePtr<FPipelineNNEBackend> DefaultBackend;
};
