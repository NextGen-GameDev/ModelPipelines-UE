// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineTypes.h"
#include "PipelineLog.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(PipelineTypes)

DEFINE_LOG_CATEGORY(LogModelPipelines);

FPipelineResult FPipelineResult::Ok()
{
	FPipelineResult Result;
	Result.bSuccess = true;
	Result.Error = EPipelineError::None;
	return Result;
}

FPipelineResult FPipelineResult::Fail(EPipelineError InError, const FString& InMessage)
{
	FPipelineResult Result;
	Result.bSuccess = false;
	Result.Error = InError;
	Result.Message = InMessage;
	return Result;
}
