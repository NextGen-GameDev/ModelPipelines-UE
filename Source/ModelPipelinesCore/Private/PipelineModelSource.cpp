// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineModelSource.h"
#include "Engine/DataAsset.h"
#include "PipelineBackend.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(PipelineModelSource)

FPipelineModelSource FPipelineModelSource::FromPackage(const UPipelineModelPackage* Package)
{
	FPipelineModelSource Source;
	if (Package == nullptr)
	{
		return Source;
	}

	Source.Generation = Package->Generation;
	Source.Format = Package->GetModelFormat();
	if (Source.Format == EPipelineModelFormat::NNE)
	{
		Source.NNEModel = Package->Model.Get();
	}
	else if (Source.Format == EPipelineModelFormat::GGUF)
	{
		// GetModelFormat returned GGUF, so GgufModel is set and implements the interface.
		const IPipelineFileModel* File = Cast<IPipelineFileModel>(Package->GgufModel.Get());
		if (File != nullptr)
		{
			Source.FilePath = File->GetAbsoluteFilePath();
			Source.FileSizeBytes = File->GetFileSizeBytes();
			Source.FileSha256 = File->GetFileSha256();
		}
		else
		{
			Source.Format = EPipelineModelFormat::None;
		}
	}
	return Source;
}

TSharedPtr<IPipelineSession> IPipelineModel::CreateSession(const FPipelineSessionRequest& Request, FPipelineResult& OutResult)
{
	OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("Model has no sessions"));
	return nullptr;
}

TSharedPtr<IPipelineModel> IPipelineBackend::PrepareModelFromSource(const FPipelineModelSource& Source, EPipelineDevice Device,
	FPipelineResult& OutResult)
{
	if (Source.Format == EPipelineModelFormat::NNE)
	{
		return PrepareModel(Source.NNEModel, Device, OutResult);
	}
	OutResult = FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, FString::Printf(TEXT("backend %s cannot prepare a %s model"),
		*GetName().ToString(), Source.Format == EPipelineModelFormat::GGUF ? TEXT("GGUF") : TEXT("format None")));
	return nullptr;
}
