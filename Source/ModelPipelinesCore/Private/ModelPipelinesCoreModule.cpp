// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesCoreModule.h"

#include "Modules/ModuleManager.h"

FModelPipelinesCoreModule& FModelPipelinesCoreModule::Get()
{
	return FModuleManager::LoadModuleChecked<FModelPipelinesCoreModule>("ModelPipelinesCore");
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
	Executor = MakeUnique<FPipelineExecutor>();
	DefaultBackend = MakeUnique<FPipelineNNEBackend>();
}

void FModelPipelinesCoreModule::ShutdownModule()
{
	if (Executor.IsValid())
	{
		Executor->Shutdown();
	}
	Executor.Reset();
	DefaultBackend.Reset();
}

IMPLEMENT_MODULE(FModelPipelinesCoreModule, ModelPipelinesCore)
