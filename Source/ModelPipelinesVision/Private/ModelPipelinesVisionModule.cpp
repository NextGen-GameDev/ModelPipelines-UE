// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesVisionModule.h"

#include "Modules/ModuleManager.h"
#include "PipelineVisionLog.h"

DEFINE_LOG_CATEGORY(LogModelPipelinesVision);

namespace UE::ModelPipelines::Vision::Private
{
	/** PipelineTextureReader.cpp: completes pending texture reads with Cancelled and waits for their callbacks. */
	void ShutdownTextureReader();
	/** PipelineTextureReader.cpp: clears the shut-down state so a restarted module can read again. */
	void StartupTextureReader();
}

void FModelPipelinesVisionModule::StartupModule()
{
	UE::ModelPipelines::Vision::Private::StartupTextureReader();
}

void FModelPipelinesVisionModule::ShutdownModule()
{
	UE::ModelPipelines::Vision::Private::ShutdownTextureReader();
}

IMPLEMENT_MODULE(FModelPipelinesVisionModule, ModelPipelinesVision)
