// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineModelPackage.h"
#include "Misc/CString.h"
#include "PipelineModelSource.h"

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
		EPipelineTask::TextGeneration,
		EPipelineTask::SpeechToText,
		EPipelineTask::TextToSpeech,
	};

	/** SPDX ids accepted without a LicenseNotPermissive warning (compared case-insensitively). */
	static const TCHAR* PermissiveSpdxIds[] = {
		TEXT("Apache-2.0"),
		TEXT("MIT"),
		TEXT("BSD-2-Clause"),
		TEXT("BSD-3-Clause"),
	};

	static bool IsSha256Hex(const FString& Str)
	{
		if (Str.Len() != 64)
		{
			return false;
		}
		for (const TCHAR Ch : Str)
		{
			if (!FChar::IsHexDigit(Ch))
			{
				return false;
			}
		}
		return true;
	}

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
	case EPipelineTask::TextGeneration: return TEXT("TextGeneration");
	case EPipelineTask::SpeechToText: return TEXT("SpeechToText");
	case EPipelineTask::TextToSpeech: return TEXT("TextToSpeech");
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

bool IsGenerativeTask(EPipelineTask Task)
{
	return Task == EPipelineTask::TextGeneration
		|| Task == EPipelineTask::SpeechToText
		|| Task == EPipelineTask::TextToSpeech;
}

EPipelineModelFormat UPipelineModelPackage::GetModelFormat() const
{
	const bool bHasNne = Model != nullptr;
	const bool bHasGguf = GgufModel != nullptr;
	if (bHasNne == bHasGguf)
	{
		return EPipelineModelFormat::None;
	}
	if (bHasNne)
	{
		return EPipelineModelFormat::NNE;
	}
	const IPipelineFileModel* File = Cast<IPipelineFileModel>(GgufModel.Get());
	if (File == nullptr || File->GetFileFormat() != EPipelineModelFormat::GGUF)
	{
		return EPipelineModelFormat::None;
	}
	return EPipelineModelFormat::GGUF;
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

	if (Task == EPipelineTask::None || !(IsTextTask(Task) || IsImageTask(Task) || IsGenerativeTask(Task)))
	{
		AddError(EPipelineValidationCode::NoTask,
			FString::Printf(TEXT("Task is %s; set the task the model serves"), LexToString(Task)));
		return false;
	}

	const bool bHasNne = Model != nullptr;
	const bool bHasGguf = GgufModel != nullptr;
	if (!bHasNne && !bHasGguf)
	{
		AddError(EPipelineValidationCode::NoModel, Task == EPipelineTask::TextGeneration
			? FString(TEXT("GgufModel is empty; assign a GGUF model asset"))
			: FString(TEXT("Model is empty; assign the imported NNE model data")));
	}
	else if (bHasNne && bHasGguf)
	{
		AddError(EPipelineValidationCode::BothModelsSet, TEXT("Model and GgufModel are both set; clear one of them"));
	}
	else if (bHasNne && Task == EPipelineTask::TextGeneration)
	{
		AddError(EPipelineValidationCode::WrongModelFormat, TEXT("TextGeneration needs a GGUF model"));
	}
	else if (bHasGguf && !IsGenerativeTask(Task))
	{
		AddError(EPipelineValidationCode::WrongModelFormat, FString::Printf(TEXT("%s needs an NNE model"), LexToString(Task)));
	}

	if (bHasGguf)
	{
		const IPipelineFileModel* File = Cast<IPipelineFileModel>(GgufModel.Get());
		if (File == nullptr || File->GetFileFormat() != EPipelineModelFormat::GGUF)
		{
			AddError(EPipelineValidationCode::InvalidGgufModel,
				FString::Printf(TEXT("GgufModel %s is not a GGUF model asset"), *GgufModel->GetName()));
		}
		else
		{
			TArray<FString> Problems;
			if (File->GetAbsoluteFilePath().IsEmpty())
			{
				Problems.Add(TEXT("its RelativePath is invalid (must start with \"Models/\", end with \".gguf\", no \"..\")"));
			}
			const int64 Size = File->GetFileSizeBytes();
			if (Size <= 0)
			{
				Problems.Add(FString::Printf(TEXT("SizeBytes is %lld; it must be > 0"), Size));
			}
			if (!IsSha256Hex(File->GetFileSha256()))
			{
				Problems.Add(TEXT("Sha256 is not 64 hex digits"));
			}
			if (Problems.Num() > 0)
			{
				AddError(EPipelineValidationCode::InvalidGgufModel,
					FString::Printf(TEXT("GgufModel %s: %s"), *GgufModel->GetName(), *FString::Join(Problems, TEXT("; "))));
			}
		}
	}

	if (Task == EPipelineTask::SpeechToText)
	{
		if (bHasNne && DecoderModel == nullptr)
		{
			AddError(EPipelineValidationCode::NoDecoderModel,
				TEXT("DecoderModel is empty; a SpeechToText package needs the decoder_joint model"));
		}
		if (TokenizerJson.IsEmpty())
		{
			AddError(EPipelineValidationCode::NoTokenizer,
				FString::Printf(TEXT("TokenizerJson is empty; a %s package needs the tokenizer.json text"), LexToString(Task)));
		}

		TArray<FString> Problems;
		auto CheckPositive = [&Problems](const TCHAR* Name, int32 Value)
		{
			if (Value <= 0)
			{
				Problems.Add(FString::Printf(TEXT("%s is %d; it must be > 0"), Name, Value));
			}
		};
		CheckPositive(TEXT("SampleRate"), Speech.SampleRate);
		CheckPositive(TEXT("NumMelBins"), Speech.NumMelBins);
		CheckPositive(TEXT("FftSize"), Speech.FftSize);
		CheckPositive(TEXT("WindowLength"), Speech.WindowLength);
		CheckPositive(TEXT("HopLength"), Speech.HopLength);
		CheckPositive(TEXT("SubsamplingFactor"), Speech.SubsamplingFactor);
		CheckPositive(TEXT("MaxSymbolsPerStep"), Speech.MaxSymbolsPerStep);
		if (Speech.WindowLength > Speech.FftSize)
		{
			Problems.Add(FString::Printf(TEXT("WindowLength is %d; it must be at most FftSize (%d)"), Speech.WindowLength, Speech.FftSize));
		}
		if (!FMath::IsFinite(Speech.Preemphasis) || !(Speech.Preemphasis >= 0.f && Speech.Preemphasis < 1.f))
		{
			Problems.Add(FString::Printf(TEXT("Preemphasis is %g; it must be in [0, 1)"), (double)Speech.Preemphasis));
		}
		if (Speech.BlankTokenId < 0)
		{
			Problems.Add(FString::Printf(TEXT("BlankTokenId is %d; it must be >= 0"), Speech.BlankTokenId));
		}
		if (Speech.Durations.IsEmpty())
		{
			Problems.Add(TEXT("Durations is empty; it needs at least one value >= 1"));
		}
		else
		{
			bool bAnyNegative = false;
			bool bAnyAdvance = false;
			for (const int32 Duration : Speech.Durations)
			{
				bAnyNegative |= Duration < 0;
				bAnyAdvance |= Duration >= 1;
			}
			if (bAnyNegative)
			{
				Problems.Add(TEXT("Durations has a negative value; every value must be >= 0"));
			}
			if (!bAnyAdvance)
			{
				Problems.Add(TEXT("Durations has no value >= 1; decoding could never advance"));
			}
		}
		if (!FMath::IsFinite(Speech.SilenceThresholdDb) || Speech.SilenceThresholdDb > 0.f)
		{
			Problems.Add(FString::Printf(TEXT("SilenceThresholdDb is %g; it must be finite and <= 0 (-100 disables the gate)"),
				(double)Speech.SilenceThresholdDb));
		}
		if (Problems.Num() > 0)
		{
			AddError(EPipelineValidationCode::InvalidSpeechSettings,
				FString::Printf(TEXT("Speech settings are invalid: %s"), *FString::Join(Problems, TEXT("; "))));
		}
	}
	else if (DecoderModel != nullptr)
	{
		AddWarning(EPipelineValidationCode::UnexpectedDecoderModel,
			FString::Printf(TEXT("DecoderModel is set but %s uses one model"), LexToString(Task)));
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
	else
	{
		const FString Spdx = License.SpdxId.TrimStartAndEnd();
		bool bPermissive = false;
		for (const TCHAR* Id : PermissiveSpdxIds)
		{
			if (Spdx.Equals(Id, ESearchCase::IgnoreCase))
			{
				bPermissive = true;
				break;
			}
		}
		if (!bPermissive)
		{
			AddWarning(EPipelineValidationCode::LicenseNotPermissive,
				FString::Printf(TEXT("License %s is not Apache-2.0, MIT, BSD-2-Clause or BSD-3-Clause; check that its terms allow your use"), *Spdx));
		}
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
