// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "PipelineEditorLibrary.h"
#include "PipelineModelPackage.h"

FPipelineImportResult UPipelineEditorLibrary::ImportModelFolder(const FPipelineImportOptions& Options)
{
	return ImportPipelineModelFolder(Options);
}

FPipelineValidationReport UPipelineEditorLibrary::ValidateModelPackage(UPipelineModelPackage* Package, FPipelineValidateOptions Options)
{
	return ValidatePipelineModelPackage(Package, Options);
}
