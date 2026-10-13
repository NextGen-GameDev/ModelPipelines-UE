// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

// Not inside WITH_DEV_AUTOMATION_TESTS: UHT ignores preprocessor guards, so the classes exist in every build.

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "PipelineModelSource.h"
#include "PipelineEditorTestFileModel.generated.h"

/** Test-only pointer asset implementing IPipelineFileModel (stands in for UPipelineGgufModel, which lives in another plugin). */
UCLASS(Transient, NotBlueprintable)
class UPipelineEditorTestFileModel : public UDataAsset, public IPipelineFileModel
{
	GENERATED_BODY()

public:
	EPipelineModelFormat Format = EPipelineModelFormat::GGUF;
	FString FilePath;
	int64 SizeBytes = 10;
	FString Sha256 = FString::ChrN(64, TEXT('a'));

	virtual EPipelineModelFormat GetFileFormat() const override { return Format; }
	virtual FString GetAbsoluteFilePath() const override { return FilePath; }
	virtual int64 GetFileSizeBytes() const override { return SizeBytes; }
	virtual FString GetFileSha256() const override { return Sha256; }
};

/** Concrete UDataAsset WITHOUT IPipelineFileModel (UDataAsset itself is abstract). */
UCLASS(Transient, NotBlueprintable)
class UPipelineEditorTestPlainAsset : public UDataAsset
{
	GENERATED_BODY()
};
