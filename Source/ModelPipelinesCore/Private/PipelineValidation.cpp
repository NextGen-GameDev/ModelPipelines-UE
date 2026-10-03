// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineValidation.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(PipelineValidation)

const TCHAR* LexToString(EPipelineValidationCode Code)
{
	switch (Code)
	{
	case EPipelineValidationCode::None: return TEXT("None");
	case EPipelineValidationCode::NoPackage: return TEXT("NoPackage");
	case EPipelineValidationCode::NoTask: return TEXT("NoTask");
	case EPipelineValidationCode::NoModel: return TEXT("NoModel");
	case EPipelineValidationCode::NoTokenizer: return TEXT("NoTokenizer");
	case EPipelineValidationCode::InvalidMaxTokens: return TEXT("InvalidMaxTokens");
	case EPipelineValidationCode::NoLabels: return TEXT("NoLabels");
	case EPipelineValidationCode::EmptyLabel: return TEXT("EmptyLabel");
	case EPipelineValidationCode::InvalidImageSize: return TEXT("InvalidImageSize");
	case EPipelineValidationCode::InvalidNormalization: return TEXT("InvalidNormalization");
	case EPipelineValidationCode::InvalidResize: return TEXT("InvalidResize");
	case EPipelineValidationCode::LicenseSpdxMissing: return TEXT("LicenseSpdxMissing");
	case EPipelineValidationCode::LicenseSourceMissing: return TEXT("LicenseSourceMissing");
	case EPipelineValidationCode::LicenseAttributionMissing: return TEXT("LicenseAttributionMissing");
	case EPipelineValidationCode::InvalidPostprocess: return TEXT("InvalidPostprocess");
	case EPipelineValidationCode::MissingInput: return TEXT("MissingInput");
	case EPipelineValidationCode::UnexpectedInput: return TEXT("UnexpectedInput");
	case EPipelineValidationCode::InputTypeMismatch: return TEXT("InputTypeMismatch");
	case EPipelineValidationCode::InputRankMismatch: return TEXT("InputRankMismatch");
	case EPipelineValidationCode::InputDimMismatch: return TEXT("InputDimMismatch");
	case EPipelineValidationCode::MissingOutput: return TEXT("MissingOutput");
	case EPipelineValidationCode::OutputTypeMismatch: return TEXT("OutputTypeMismatch");
	case EPipelineValidationCode::OutputRankMismatch: return TEXT("OutputRankMismatch");
	case EPipelineValidationCode::OutputDimMismatch: return TEXT("OutputDimMismatch");
	case EPipelineValidationCode::LabelCountMismatch: return TEXT("LabelCountMismatch");
	case EPipelineValidationCode::OutputNameFallback: return TEXT("OutputNameFallback");
	case EPipelineValidationCode::FixedBatch: return TEXT("FixedBatch");
	case EPipelineValidationCode::FolderNotFound: return TEXT("FolderNotFound");
	case EPipelineValidationCode::MissingModelFile: return TEXT("MissingModelFile");
	case EPipelineValidationCode::InvalidJson: return TEXT("InvalidJson");
	case EPipelineValidationCode::MissingConfig: return TEXT("MissingConfig");
	case EPipelineValidationCode::MissingTokenizer: return TEXT("MissingTokenizer");
	case EPipelineValidationCode::TaskUndetected: return TEXT("TaskUndetected");
	case EPipelineValidationCode::MissingLabels: return TEXT("MissingLabels");
	case EPipelineValidationCode::InvalidLabels: return TEXT("InvalidLabels");
	case EPipelineValidationCode::MissingPreprocessorConfig: return TEXT("MissingPreprocessorConfig");
	case EPipelineValidationCode::UnsupportedPooling: return TEXT("UnsupportedPooling");
	case EPipelineValidationCode::UnsupportedResize: return TEXT("UnsupportedResize");
	case EPipelineValidationCode::UnverifiedDetectionFamily: return TEXT("UnverifiedDetectionFamily");
	case EPipelineValidationCode::TokenizerInvalid: return TEXT("TokenizerInvalid");
	case EPipelineValidationCode::RuntimeUnavailable: return TEXT("RuntimeUnavailable");
	case EPipelineValidationCode::ModelCreateFailed: return TEXT("ModelCreateFailed");
	case EPipelineValidationCode::SmokeRunFailed: return TEXT("SmokeRunFailed");
	case EPipelineValidationCode::SmokeOutputInvalid: return TEXT("SmokeOutputInvalid");
	case EPipelineValidationCode::InvalidDestination: return TEXT("InvalidDestination");
	case EPipelineValidationCode::AssetExists: return TEXT("AssetExists");
	case EPipelineValidationCode::SaveFailed: return TEXT("SaveFailed");
	default: return TEXT("Unknown");
	}
}

const TCHAR* LexToString(EPipelineValidationSeverity Severity)
{
	switch (Severity)
	{
	case EPipelineValidationSeverity::Info: return TEXT("Info");
	case EPipelineValidationSeverity::Warning: return TEXT("Warning");
	case EPipelineValidationSeverity::Error: return TEXT("Error");
	default: return TEXT("Unknown");
	}
}

void FPipelineValidationReport::Add(EPipelineValidationSeverity InSeverity, EPipelineValidationCode InCode, const FString& InMessage)
{
	FPipelineValidationMessage& Msg = Messages.AddDefaulted_GetRef();
	Msg.Severity = InSeverity;
	Msg.Code = InCode;
	Msg.Message = InMessage;
}

bool FPipelineValidationReport::IsValid() const
{
	return NumErrors() == 0;
}

int32 FPipelineValidationReport::NumErrors() const
{
	int32 Count = 0;
	for (const FPipelineValidationMessage& Msg : Messages)
	{
		if (Msg.Severity == EPipelineValidationSeverity::Error)
		{
			++Count;
		}
	}
	return Count;
}

int32 FPipelineValidationReport::NumWarnings() const
{
	int32 Count = 0;
	for (const FPipelineValidationMessage& Msg : Messages)
	{
		if (Msg.Severity == EPipelineValidationSeverity::Warning)
		{
			++Count;
		}
	}
	return Count;
}

bool FPipelineValidationReport::HasCode(EPipelineValidationCode InCode) const
{
	for (const FPipelineValidationMessage& Msg : Messages)
	{
		if (Msg.Code == InCode)
		{
			return true;
		}
	}
	return false;
}

bool FPipelineValidationReport::HasError(EPipelineValidationCode InCode) const
{
	for (const FPipelineValidationMessage& Msg : Messages)
	{
		if (Msg.Code == InCode && Msg.Severity == EPipelineValidationSeverity::Error)
		{
			return true;
		}
	}
	return false;
}

void FPipelineValidationReport::Append(const FPipelineValidationReport& Other)
{
	if (&Other == this)
	{
		// Copy first: appending an array to itself would read while the storage grows.
		const FPipelineValidationReport Copy = Other;
		Messages.Append(Copy.Messages);
		DeviceChecks.Append(Copy.DeviceChecks);
		return;
	}
	Messages.Append(Other.Messages);
	DeviceChecks.Append(Other.DeviceChecks);
}

FString FPipelineValidationReport::ToString() const
{
	FString Out;
	for (int32 Index = 0; Index < Messages.Num(); ++Index)
	{
		const FPipelineValidationMessage& Msg = Messages[Index];
		if (Index > 0)
		{
			Out += TEXT("\n");
		}
		Out += LexToString(Msg.Severity);
		Out += TEXT(" ");
		Out += LexToString(Msg.Code);
		Out += TEXT(": ");
		Out += Msg.Message;
	}
	return Out;
}
