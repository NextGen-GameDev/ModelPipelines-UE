// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "UObject/Interface.h"
#include "PipelineTypes.h"
#include "PipelineModelPackage.h"
#include "PipelineModelSource.generated.h"

class UNNEModelData;

/** Implemented by pointer assets that describe a loose model file (UPipelineGgufModel). docs/contracts.md C60d. */
UINTERFACE(MinimalAPI)
class UPipelineFileModel : public UInterface
{
	GENERATED_BODY()
};

class MODELPIPELINESCORE_API IPipelineFileModel
{
	GENERATED_BODY()

public:
	/** GGUF for UPipelineGgufModel. */
	virtual EPipelineModelFormat GetFileFormat() const = 0;

	/** Absolute, normalized path of the file; empty when the asset's path is invalid. */
	virtual FString GetAbsoluteFilePath() const = 0;

	/** Expected size in bytes. */
	virtual int64 GetFileSizeBytes() const = 0;

	/** Expected SHA-256, 64 lowercase hex digits. */
	virtual FString GetFileSha256() const = 0;
};

/** What a backend needs to prepare a model. Plain data; built on the game thread from a package. */
struct MODELPIPELINESCORE_API FPipelineModelSource
{
	EPipelineModelFormat Format = EPipelineModelFormat::None;

	/** Format NNE: the model data (game thread only; the backend copies what it needs in PrepareModel). */
	UNNEModelData* NNEModel = nullptr;

	/** Format GGUF: absolute, normalized path of the loose file (resolved by UPipelineGgufModel::GetAbsolutePath). */
	FString FilePath;

	/** Format GGUF: expected size in bytes (checked on every load) and SHA-256 (64 lowercase hex). The SHA is checked by
	    the editor validator and by tests, not on every runtime load, because hashing 2 GB takes seconds. */
	int64 FileSizeBytes = 0;
	FString FileSha256;

	/** Copy of Package->Generation (load-time settings the GGUF backend needs; C60d). Unused for NNE. */
	FPipelineGenerationModelSettings Generation;

	/** Game thread. From Package->Model (NNE) or Package->GgufModel (GGUF). Neither or both set -> Format None. */
	static FPipelineModelSource FromPackage(const UPipelineModelPackage* Package);
};
