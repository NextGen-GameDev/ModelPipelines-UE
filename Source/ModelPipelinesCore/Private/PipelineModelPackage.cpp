// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "PipelineModelPackage.h"
#include "Misc/CString.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(PipelineModelPackage)

namespace PipelineModelPackagePrivate
{
	static constexpr int32 MinMaxTokens = 1;
	static constexpr int32 MaxMaxTokens = 65536;
	static constexpr int32 MinImageSize = 1;
	static constexpr int32 MaxImageSize = 8192;
	static constexpr int32 MinMaxDetections = 1;
	static constexpr int32 MaxMaxDetections = 10000;

	static const EPipelineTask AllTasks[] = {
		EPipelineTask::None,
		EPipelineTask::TextEmbedding,
		EPipelineTask::TextClassification,
		EPipelineTask::ImageClassification,
		EPipelineTask::ObjectDetection,
	};

	static bool IsBlank(const FString& Str)
	{
		return Str.TrimStartAndEnd().IsEmpty();
	}

	/** True only for a finite value > 0 (NaN fails every comparison). */
	static bool IsPositive(double Value)
	{
		return Value > 0.0 && FMath::IsFinite(Value);
	}
}

const TCHAR* LexToString(EPipelineImageResample Value)
{
	switch (Value)
	{
	case EPipelineImageResample::Nearest: return TEXT("Nearest");
	case EPipelineImageResample::Lanczos: return TEXT("Lanczos");
	case EPipelineImageResample::Bilinear: return TEXT("Bilinear");
	case EPipelineImageResample::Bicubic: return TEXT("Bicubic");
	case EPipelineImageResample::Box: return TEXT("Box");
	case EPipelineImageResample::Hamming: return TEXT("Hamming");
	default: return TEXT("Unknown");
	}
}

const TCHAR* LexToString(EPipelineTask Task)
{
	switch (Task)
	{
	case EPipelineTask::None: return TEXT("None");
	case EPipelineTask::TextEmbedding: return TEXT("TextEmbedding");
	case EPipelineTask::TextClassification: return TEXT("TextClassification");
	case EPipelineTask::ImageClassification: return TEXT("ImageClassification");
	case EPipelineTask::ObjectDetection: return TEXT("ObjectDetection");
	default: return TEXT("Unknown");
	}
}

bool LexTryParseString(EPipelineTask& Out, const TCHAR* Str)
{
	if (Str == nullptr || *Str == TEXT('\0'))
	{
		return false;
	}
	for (const EPipelineTask Candidate : PipelineModelPackagePrivate::AllTasks)
	{
		if (FCString::Stricmp(Str, LexToString(Candidate)) == 0)
		{
			Out = Candidate;
			return true;
		}
	}
	return false;
}

bool IsTextTask(EPipelineTask Task)
{
	return Task == EPipelineTask::TextEmbedding || Task == EPipelineTask::TextClassification;
}

bool IsImageTask(EPipelineTask Task)
{
	return Task == EPipelineTask::ImageClassification || Task == EPipelineTask::ObjectDetection;
}

bool TaskUsesLabels(EPipelineTask Task)
{
	return Task == EPipelineTask::TextClassification
		|| Task == EPipelineTask::ImageClassification
		|| Task == EPipelineTask::ObjectDetection;
}

bool UPipelineModelPackage::ValidateSettings(FPipelineValidationReport& OutReport) const
{
	using namespace PipelineModelPackagePrivate;

	bool bAddedError = false;
	auto AddError = [&OutReport, &bAddedError](EPipelineValidationCode Code, const FString& Message)
	{
		OutReport.Add(EPipelineValidationSeverity::Error, Code, Message);
		bAddedError = true;
	};
	auto AddWarning = [&OutReport](EPipelineValidationCode Code, const FString& Message)
	{
		OutReport.Add(EPipelineValidationSeverity::Warning, Code, Message);
	};

	if (Task == EPipelineTask::None || !(IsTextTask(Task) || IsImageTask(Task)))
	{
		AddError(EPipelineValidationCode::NoTask,
			FString::Printf(TEXT("Task is %s; set the task the model serves"), LexToString(Task)));
		return false;
	}

	if (Model == nullptr)
	{
		AddError(EPipelineValidationCode::NoModel, TEXT("Model is empty; assign the imported NNE model data"));
	}

	if (IsTextTask(Task))
	{
		if (TokenizerJson.IsEmpty())
		{
			AddError(EPipelineValidationCode::NoTokenizer,
				FString::Printf(TEXT("TokenizerJson is empty; a %s package needs the tokenizer.json text"), LexToString(Task)));
		}
		const int32 MaxTokens = Preprocess.MaxTokens;
		if (MaxTokens < MinMaxTokens || MaxTokens > MaxMaxTokens)
		{
			AddError(EPipelineValidationCode::InvalidMaxTokens,
				FString::Printf(TEXT("MaxTokens is %d; it must be %d..%d"), MaxTokens, MinMaxTokens, MaxMaxTokens));
		}
	}

	if (TaskUsesLabels(Task))
	{
		if (Labels.IsEmpty())
		{
			AddError(EPipelineValidationCode::NoLabels,
				FString::Printf(TEXT("Labels is empty; a %s package needs one label per class"), LexToString(Task)));
		}
		else
		{
			for (int32 Index = 0; Index < Labels.Num(); ++Index)
			{
				if (IsBlank(Labels[Index]))
				{
					// Names only the first empty index (no other numbers in the message).
					AddWarning(EPipelineValidationCode::EmptyLabel, FString::Printf(TEXT("Label %d is empty"), Index));
					break;
				}
			}
		}
	}

	if (IsImageTask(Task))
	{
		const int32 Width = Preprocess.ImageWidth;
		const int32 Height = Preprocess.ImageHeight;
		const bool bWidthOk = Width >= MinImageSize && Width <= MaxImageSize;
		const bool bHeightOk = Height >= MinImageSize && Height <= MaxImageSize;
		if (!bWidthOk || !bHeightOk)
		{
			AddError(EPipelineValidationCode::InvalidImageSize,
				FString::Printf(TEXT("ImageWidth x ImageHeight is %d x %d; each must be %d..%d"),
					Width, Height, MinImageSize, MaxImageSize));
		}

		const bool bRescaleOk = IsPositive((double)Preprocess.RescaleFactor);
		const bool bStdOk = IsPositive(Preprocess.ImageStd.X) && IsPositive(Preprocess.ImageStd.Y) && IsPositive(Preprocess.ImageStd.Z);
		if (!bRescaleOk || !bStdOk)
		{
			FString Message;
			if (!bRescaleOk)
			{
				Message = FString::Printf(TEXT("RescaleFactor is %g; it must be > 0"), (double)Preprocess.RescaleFactor);
			}
			if (!bStdOk)
			{
				if (!Message.IsEmpty())
				{
					Message += TEXT("; ");
				}
				Message += FString::Printf(TEXT("ImageStd is (%g, %g, %g); every component must be > 0"),
					Preprocess.ImageStd.X, Preprocess.ImageStd.Y, Preprocess.ImageStd.Z);
			}
			AddError(EPipelineValidationCode::InvalidNormalization, Message);
		}

		if (Preprocess.ResizeMode == EPipelineImageResize::ShortestEdgeCenterCrop)
		{
			const int32 Needed = FMath::Max(Width, Height);
			if (Preprocess.ResizeShortestEdge < Needed)
			{
				AddError(EPipelineValidationCode::InvalidResize,
					FString::Printf(TEXT("ResizeShortestEdge is %d; with ShortestEdgeCenterCrop it must be at least max(ImageWidth, ImageHeight) = %d"),
						Preprocess.ResizeShortestEdge, Needed));
			}
		}
		else if (Preprocess.ResizeMode == EPipelineImageResize::StretchCenterCrop)
		{
			const int32 RW = Preprocess.ResizeWidth;
			const int32 RH = Preprocess.ResizeHeight;
			if (RW < 1 || RH < 1 || RW > MaxImageSize || RH > MaxImageSize || RW < Width || RH < Height)
			{
				AddError(EPipelineValidationCode::InvalidResize,
					FString::Printf(TEXT("ResizeWidth x ResizeHeight is %dx%d; with StretchCenterCrop it must be at least ImageWidth x ImageHeight = %dx%d and at most %d"),
						RW, RH, Width, Height, MaxImageSize));
			}
		}
	}

	if (Task == EPipelineTask::ObjectDetection)
	{
		// > 1 is valid (disables NMS); only NaN and negative values are rejected (NaN fails the >= comparison).
		const float Thr = Postprocess.NmsIouThreshold;
		if (!(Thr >= 0.f))
		{
			AddError(EPipelineValidationCode::InvalidPostprocess,
				FString::Printf(TEXT("NmsIouThreshold is %g; it must be >= 0 (values >= 1 disable NMS)"), (double)Thr));
		}
		const int32 MaxDet = Postprocess.MaxDetections;
		if (MaxDet < MinMaxDetections || MaxDet > MaxMaxDetections)
		{
			AddError(EPipelineValidationCode::InvalidPostprocess,
				FString::Printf(TEXT("MaxDetections is %d; it must be %d..%d"), MaxDet, MinMaxDetections, MaxMaxDetections));
		}
	}

	if (IsBlank(License.SpdxId))
	{
		AddWarning(EPipelineValidationCode::LicenseSpdxMissing, TEXT("License SPDX id is empty"));
	}
	if (IsBlank(License.SourceUrl))
	{
		AddWarning(EPipelineValidationCode::LicenseSourceMissing, TEXT("License SourceUrl is empty"));
	}
	if (IsBlank(License.Attribution))
	{
		AddWarning(EPipelineValidationCode::LicenseAttributionMissing, TEXT("License Attribution is empty"));
	}

	return !bAddedError;
}
