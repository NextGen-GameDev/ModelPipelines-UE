// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesCoreModule.h"

#include <atomic>

#include "Modules/ModuleManager.h"

namespace
{
	/** True from the top of ShutdownModule until the next StartupModule. */
	std::atomic<bool> GModelPipelinesCoreShuttingDown{false};
}

FModelPipelinesCoreModule& FModelPipelinesCoreModule::Get()
{
	return FModuleManager::LoadModuleChecked<FModelPipelinesCoreModule>("ModelPipelinesCore");
}

FModelPipelinesCoreModule* FModelPipelinesCoreModule::TryGet()
{
	if (GModelPipelinesCoreShuttingDown.load(std::memory_order_acquire))
	{
		return nullptr;
	}
	return FModuleManager::GetModulePtr<FModelPipelinesCoreModule>("ModelPipelinesCore");
}

FPipelineExecutor& FModelPipelinesCoreModule::GetExecutor()
{
	// Programmer error only (called outside the module's lifetime); never driven by user data.
	check(Executor.IsValid());
	return *Executor;
}

IPipelineBackend& FModelPipelinesCoreModule::GetDefaultBackend()
{
	check(DefaultBackend.IsValid());
	return *DefaultBackend;
}

void FModelPipelinesCoreModule::StartupModule()
{
	GModelPipelinesCoreShuttingDown.store(false, std::memory_order_release);
	Executor = MakeUnique<FPipelineExecutor>();
	DefaultBackend = MakeUnique<FPipelineNNEBackend>();
}

void FModelPipelinesCoreModule::ShutdownModule()
{
	GModelPipelinesCoreShuttingDown.store(true, std::memory_order_release);
	if (Executor.IsValid())
	{
		Executor->Shutdown();
	}
	Executor.Reset();
	DefaultBackend.Reset();
}

IMPLEMENT_MODULE(FModelPipelinesCoreModule, ModelPipelinesCore)
