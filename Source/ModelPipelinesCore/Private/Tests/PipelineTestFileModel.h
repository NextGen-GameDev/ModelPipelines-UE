// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

// Not inside WITH_DEV_AUTOMATION_TESTS: UHT ignores preprocessor guards, so the class exists in every build.

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "PipelineModelSource.h"
#include "PipelineBase.h"
#include "PipelineTestFileModel.generated.h"

/** Test-only pointer asset implementing IPipelineFileModel (stands in for UPipelineGgufModel, which lives in another plugin). */
UCLASS(Transient, NotBlueprintable)
class UPipelineTestFileModel : public UDataAsset, public IPipelineFileModel
{
	GENERATED_BODY()

public:
	EPipelineModelFormat Format = EPipelineModelFormat::GGUF;
	FString FilePath = TEXT("C:/Models/test.gguf");
	int64 SizeBytes = 10;
	FString Sha256 = FString::ChrN(64, TEXT('a'));

	virtual EPipelineModelFormat GetFileFormat() const override { return Format; }
	virtual FString GetAbsoluteFilePath() const override { return FilePath; }
	virtual int64 GetFileSizeBytes() const override { return SizeBytes; }
	virtual FString GetFileSha256() const override { return Sha256; }
};

/** Concrete UDataAsset WITHOUT IPipelineFileModel (UDataAsset itself is abstract, DataAsset.h:16; added by the lead). */
UCLASS(Transient, NotBlueprintable)
class UPipelineTestPlainAsset : public UDataAsset
{
	GENERATED_BODY()
};

/** Concrete UPipelineBase whose task is TextGeneration (WP8L.3a InitializeFromPackage tests; added by the lead). */
UCLASS(Transient, NotBlueprintable)
class UPipelineTestGenerationPipeline : public UPipelineBase
{
	GENERATED_BODY()
public:
	UPipelineTestGenerationPipeline() { PipelineTask = EPipelineTask::TextGeneration; }
};
