// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesMCPModule.h"

#include "Modules/ModuleManager.h"
#include "PipelineMcpRun.h"

void FModelPipelinesMCPModule::StartupModule()
{
}

void FModelPipelinesMCPModule::ShutdownModule()
{
	// Running UPipelineMcpJobs end with an error (no broadcast while the engine exits) and release themselves.
	FPipelineMcpJobInternal::CancelAll(TEXT("cancelled: module shutting down"));
}

IMPLEMENT_MODULE(FModelPipelinesMCPModule, ModelPipelinesMCP)
