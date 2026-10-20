// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "PipelineModelFolder.h"

#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/FileManager.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(PipelineModelFolder)

namespace UE::ModelPipelines::FolderPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;

	/** Deeper JSON is rejected before deserializing (protects the recursive value destructors). */
	static constexpr int32 MaxJsonDepth = 256;

	/** Outcome of reading one key. */
	enum class ERead : uint8
	{
		Absent,
		Ok,
		Invalid
	};

	/** Per-call state; nothing static and mutable. */
	struct FContext
	{
		FString Folder;
		FPipelineValidationReport& Report;

		explicit FContext(FPipelineValidationReport& InReport) : Report(InReport) {}

		void Error(ECode Code, const FString& Message) { Report.Add(ESev::Error, Code, Message); }
		void Warn(ECode Code, const FString& Message) { Report.Add(ESev::Warning, Code, Message); }
	};

	static const TCHAR* JsonTypeName(EJson Type)
	{
		switch (Type)
		{
		case EJson::None: return TEXT("nothing");
		case EJson::Null: return TEXT("null");
		case EJson::String: return TEXT("a string");
		case EJson::Number: return TEXT("a number");
		case EJson::Boolean: return TEXT("a boolean");
		case EJson::Array: return TEXT("an array");
		case EJson::Object: return TEXT("an object");
		}
		return TEXT("an unknown value");
	}

	/** Field value, or nullptr when the object is absent, the key is absent or the value is null. */
	static TSharedPtr<FJsonValue> GetValue(const TSharedPtr<FJsonObject>& Obj, const TCHAR* Key)
	{
		if (!Obj.IsValid())
		{
			return nullptr;
		}
		TSharedPtr<FJsonValue> Value = Obj->TryGetField(Key);
		if (!Value.IsValid() || Value->Type == EJson::Null || Value->Type == EJson::None)
		{
			return nullptr;
		}
		return Value;
	}

	static void TypeError(FContext& Ctx, const FString& Where, const TCHAR* Expected, const TSharedPtr<FJsonValue>& Value)
	{
		Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("%s must be %s, found %s. Fix the value in the file."),
			*Where, Expected, JsonTypeName(Value->Type)));
	}

	static bool ValueToDouble(FContext& Ctx, const TSharedPtr<FJsonValue>& Value, const FString& Where, double& Out)
	{
		double D = 0.0;
		if (Value->Type != EJson::Number || !Value->TryGetNumber(D) || !FMath::IsFinite(D))
		{
			TypeError(Ctx, Where, TEXT("a finite number"), Value);
			return false;
		}
		Out = D;
		return true;
	}

	/** Integers may be written as 64 or 64.0; the value must fit int32. */
	static bool ValueToInt(FContext& Ctx, const TSharedPtr<FJsonValue>& Value, const FString& Where, int32& Out)
	{
		double D = 0.0;
		if (!ValueToDouble(Ctx, Value, Where, D))
		{
			return false;
		}
		if (D != FMath::FloorToDouble(D) || D < (double)MIN_int32 || D > (double)MAX_int32)
		{
			Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("%s must be an integer, found %.17g. Fix the value in the file."), *Where, D));
			return false;
		}
		Out = (int32)D;
		return true;
	}

	/** "config.json: key" for a file, "modelpipelines.json: license.key" for a nested object prefix. */
	static FString Where(const TCHAR* File, const TCHAR* Key)
	{
		const FString F(File);
		return F + (F.EndsWith(TEXT(".json")) ? TEXT(": ") : TEXT(".")) + Key;
	}

	static ERead ReadNumber(FContext& Ctx, const TSharedPtr<FJsonObject>& Obj, const TCHAR* File, const TCHAR* Key, double& Out)
	{
		const TSharedPtr<FJsonValue> V = GetValue(Obj, Key);
		if (!V.IsValid())
		{
			return ERead::Absent;
		}
		return ValueToDouble(Ctx, V, Where(File, Key), Out) ? ERead::Ok : ERead::Invalid;
	}

	static ERead ReadInt(FContext& Ctx, const TSharedPtr<FJsonObject>& Obj, const TCHAR* File, const TCHAR* Key, int32& Out)
	{
		const TSharedPtr<FJsonValue> V = GetValue(Obj, Key);
		if (!V.IsValid())
		{
			return ERead::Absent;
		}
		return ValueToInt(Ctx, V, Where(File, Key), Out) ? ERead::Ok : ERead::Invalid;
	}

	static ERead ReadBool(FContext& Ctx, const TSharedPtr<FJsonObject>& Obj, const TCHAR* File, const TCHAR* Key, bool& Out)
	{
		const TSharedPtr<FJsonValue> V = GetValue(Obj, Key);
		if (!V.IsValid())
		{
			return ERead::Absent;
		}
		if (V->Type != EJson::Boolean)
		{
			TypeError(Ctx, Where(File, Key), TEXT("true or false"), V);
			return ERead::Invalid;
		}
		Out = V->AsBool();
		return ERead::Ok;
	}

	static ERead ReadString(FContext& Ctx, const TSharedPtr<FJsonObject>& Obj, const TCHAR* File, const TCHAR* Key, FString& Out)
	{
		const TSharedPtr<FJsonValue> V = GetValue(Obj, Key);
		if (!V.IsValid())
		{
			return ERead::Absent;
		}
		FString S;
		if (V->Type != EJson::String || !V->TryGetString(S))
		{
			TypeError(Ctx, Where(File, Key), TEXT("a string"), V);
			return ERead::Invalid;
		}
		Out = MoveTemp(S);
		return ERead::Ok;
	}

	static ERead ReadObject(FContext& Ctx, const TSharedPtr<FJsonObject>& Obj, const TCHAR* File, const TCHAR* Key, TSharedPtr<FJsonObject>& Out)
	{
		const TSharedPtr<FJsonValue> V = GetValue(Obj, Key);
		if (!V.IsValid())
		{
			return ERead::Absent;
		}
		if (V->Type != EJson::Object || !V->AsObject().IsValid())
		{
			TypeError(Ctx, Where(File, Key), TEXT("an object"), V);
			return ERead::Invalid;
		}
		Out = V->AsObject();
		return ERead::Ok;
	}

	/** 3 numbers (R,G,B); a single number is applied to all channels. */
	static ERead ReadVector3(FContext& Ctx, const TSharedPtr<FJsonObject>& Obj, const TCHAR* File, const TCHAR* Key, FVector& Out)
	{
		const TSharedPtr<FJsonValue> V = GetValue(Obj, Key);
		if (!V.IsValid())
		{
			return ERead::Absent;
		}
		const FString W = Where(File, Key);
		if (V->Type == EJson::Number)
		{
			double D = 0.0;
			if (!ValueToDouble(Ctx, V, W, D))
			{
				return ERead::Invalid;
			}
			Out = FVector(D, D, D);
			return ERead::Ok;
		}
		if (V->Type != EJson::Array || V->AsArray().Num() != 3)
		{
			Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("%s must be an array of 3 numbers (R,G,B), found %s%s. Fix the value in the file."),
				*W, JsonTypeName(V->Type), V->Type == EJson::Array ? *FString::Printf(TEXT(" of %d elements"), V->AsArray().Num()) : TEXT("")));
			return ERead::Invalid;
		}
		const TArray<TSharedPtr<FJsonValue>>& A = V->AsArray();
		double C[3] = {0.0, 0.0, 0.0};
		for (int32 i = 0; i < 3; ++i)
		{
			if (!A[i].IsValid() || !ValueToDouble(Ctx, A[i], FString::Printf(TEXT("%s[%d]"), *W, i), C[i]))
			{
				return ERead::Invalid;
			}
		}
		Out = FVector(C[0], C[1], C[2]);
		return ERead::Ok;
	}

	/** Maximum {/[ nesting outside strings. */
	static int32 JsonNestingDepth(const FString& Text)
	{
		int32 Depth = 0;
		int32 MaxDepth = 0;
		bool bInString = false;
		bool bEscape = false;
		for (const TCHAR Ch : Text)
		{
			if (bInString)
			{
				if (bEscape) { bEscape = false; }
				else if (Ch == TEXT('\\')) { bEscape = true; }
				else if (Ch == TEXT('"')) { bInString = false; }
				continue;
			}
			if (Ch == TEXT('"')) { bInString = true; }
			else if (Ch == TEXT('{') || Ch == TEXT('[')) { MaxDepth = FMath::Max(MaxDepth, ++Depth); }
			else if (Ch == TEXT('}') || Ch == TEXT(']')) { Depth = FMath::Max(0, Depth - 1); }
		}
		return MaxDepth;
	}

	/**
	 * Loads <Folder>/<Rel> as JSON when the file exists. bPresent tells whether it exists.
	 * Returns false (and adds InvalidJson) when it exists but cannot be read or parsed.
	 */
	static bool LoadJson(FContext& Ctx, const FString& Rel, TSharedPtr<FJsonValue>& Out, bool& bPresent)
	{
		Out.Reset();
		const FString Path = Ctx.Folder / Rel;
		bPresent = IFileManager::Get().FileExists(*Path);
		if (!bPresent)
		{
			return true;
		}
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("%s: the file could not be read. Check its permissions."), *Rel));
			return false;
		}
		if (JsonNestingDepth(Text) > MaxJsonDepth)
		{
			Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("%s: JSON is nested deeper than %d levels. Replace the file with the one from the model repository."), *Rel, MaxJsonDepth));
			return false;
		}
		TSharedRef<TJsonReader<TCHAR>> Reader = TJsonReaderFactory<TCHAR>::Create(Text);
		TSharedPtr<FJsonValue> Value;
		if (!FJsonSerializer::Deserialize(Reader, Value) || !Value.IsValid())
		{
			const FString Detail = Text.IsEmpty() ? FString(TEXT("the file is empty")) : Reader->GetErrorMessage();
			Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("%s: not valid JSON (%s). Replace the file with the one from the model repository."), *Rel, *Detail));
			return false;
		}
		Out = Value;
		return true;
	}

	/** LoadJson plus "top level must be an object". */
	static bool LoadJsonObject(FContext& Ctx, const FString& Rel, TSharedPtr<FJsonObject>& Out, bool& bPresent)
	{
		Out.Reset();
		TSharedPtr<FJsonValue> Value;
		if (!LoadJson(Ctx, Rel, Value, bPresent))
		{
			return false;
		}
		if (!bPresent)
		{
			return true;
		}
		if (Value->Type != EJson::Object || !Value->AsObject().IsValid())
		{
			Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("%s: the top level must be a JSON object, found %s."), *Rel, JsonTypeName(Value->Type)));
			return false;
		}
		Out = Value->AsObject();
		return true;
	}

	/** Python-int()-like parse of a decimal key: optional surrounding whitespace and sign, digits only. */
	static bool ParseIntegerKey(const FString& Key, int64& Out)
	{
		const FString K = Key.TrimStartAndEnd();
		int32 i = 0;
		bool bNegative = false;
		if (i < K.Len() && (K[i] == TEXT('+') || K[i] == TEXT('-')))
		{
			bNegative = K[i] == TEXT('-');
			++i;
		}
		if (i >= K.Len())
		{
			return false;
		}
		int64 V = 0;
		for (; i < K.Len(); ++i)
		{
			const TCHAR Ch = K[i];
			if (Ch < TEXT('0') || Ch > TEXT('9'))
			{
				return false;
			}
			if (V > (MAX_int64 - 9) / 10)
			{
				return false;
			}
			V = V * 10 + (Ch - TEXT('0'));
		}
		Out = bNegative ? -V : V;
		return true;
	}

	static FString NormalizeFolderPath(const FString& Folder)
	{
		FString P = FPaths::ConvertRelativePathToFull(Folder);
		P.ReplaceInline(TEXT("\\"), TEXT("/"));
		while (P.Len() > 1 && P.EndsWith(TEXT("/")))
		{
			P.LeftChopInline(1);
		}
		return P;
	}

	/** SPDX id from README.md YAML front matter, "" when none (rule 9). */
	static FString ReadmeLicense(const FString& Folder)
	{
		const FString Path = Folder / TEXT("README.md");
		if (!IFileManager::Get().FileExists(*Path))
		{
			return FString();
		}
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			return FString();
		}
		TArray<FString> Lines;
		Text.ParseIntoArrayLines(Lines, /*bCullEmpty*/ false);
		if (Lines.Num() == 0 || Lines[0].TrimEnd() != TEXT("---"))
		{
			return FString();
		}
		for (int32 i = 1; i < Lines.Num(); ++i)
		{
			const FString Line = Lines[i].TrimEnd();
			if (Line == TEXT("---"))
			{
				break;
			}
			if (!Line.StartsWith(TEXT("license:"), ESearchCase::CaseSensitive))
			{
				continue;
			}
			FString Value = Line.RightChop(8).TrimStartAndEnd();
			if (Value.Len() >= 2 && ((Value[0] == TEXT('"') && Value[Value.Len() - 1] == TEXT('"')) || (Value[0] == TEXT('\'') && Value[Value.Len() - 1] == TEXT('\''))))
			{
				Value = Value.Mid(1, Value.Len() - 2).TrimStartAndEnd();
			}
			static const TCHAR* const Canonical[] = {TEXT("Apache-2.0"), TEXT("MIT"), TEXT("BSD-2-Clause"), TEXT("BSD-3-Clause")};
			for (const TCHAR* C : Canonical)
			{
				if (Value.Equals(C, ESearchCase::IgnoreCase))
				{
					return FString(C);
				}
			}
			return Value;
		}
		return FString();
	}

	/** Image size for crop_size: integer or {height, width}. Returns Absent when the shape is not usable. */
	static ERead ReadCropSize(FContext& Ctx, const TSharedPtr<FJsonObject>& Pre, const TCHAR* File, int32& OutW, int32& OutH)
	{
		const TSharedPtr<FJsonValue> V = GetValue(Pre, TEXT("crop_size"));
		if (!V.IsValid())
		{
			return ERead::Absent;
		}
		const FString W = Where(File, TEXT("crop_size"));
		if (V->Type == EJson::Number)
		{
			int32 S = 0;
			if (!ValueToInt(Ctx, V, W, S))
			{
				return ERead::Invalid;
			}
			OutW = OutH = S;
			return ERead::Ok;
		}
		if (V->Type == EJson::Object && V->AsObject().IsValid())
		{
			const TSharedPtr<FJsonObject>& Obj = V->AsObject();
			const TSharedPtr<FJsonValue> HV = GetValue(Obj, TEXT("height"));
			const TSharedPtr<FJsonValue> WV = GetValue(Obj, TEXT("width"));
			if (!HV.IsValid() || !WV.IsValid())
			{
				return ERead::Absent;
			}
			int32 H = 0, Wd = 0;
			if (!ValueToInt(Ctx, HV, W + TEXT(".height"), H) || !ValueToInt(Ctx, WV, W + TEXT(".width"), Wd))
			{
				return ERead::Invalid;
			}
			OutW = Wd;
			OutH = H;
			return ERead::Ok;
		}
		TypeError(Ctx, W, TEXT("an integer or an object with height and width"), V);
		return ERead::Invalid;
	}

	/** C25: processors whose integer `size` means {shortest_edge: S} (union of both transformers 4.57.6 scans). */
	static bool IsShortestEdgeIntClass(const FString& Cls)
	{
		static const TCHAR* const Classes[] = {
			TEXT("BitImageProcessor"), TEXT("BlipImageProcessor"), TEXT("BridgeTowerImageProcessor"), TEXT("ChameleonImageProcessor"),
			TEXT("ChineseCLIPImageProcessor"), TEXT("CLIPImageProcessor"), TEXT("ConditionalDetrImageProcessor"), TEXT("ConvNextImageProcessor"),
			TEXT("DeepseekVLImageProcessor"), TEXT("DeformableDetrImageProcessor"), TEXT("DetaImageProcessor"), TEXT("DetrImageProcessor"),
			TEXT("EfficientLoFTRImageProcessor"), TEXT("EomtImageProcessor"), TEXT("GotOcr2ImageProcessor"), TEXT("GroundingDinoImageProcessor"),
			TEXT("InstructBlipVideoImageProcessor"), TEXT("JanusImageProcessor"), TEXT("LevitImageProcessor"), TEXT("LightGlueImageProcessor"), TEXT("LlavaImageProcessor"),
			TEXT("LlavaNextImageProcessor"), TEXT("LlavaNextVideoImageProcessor"), TEXT("LlavaOnevisionImageProcessor"), TEXT("Mask2FormerImageProcessor"),
			TEXT("MaskFormerImageProcessor"), TEXT("MobileNetV1ImageProcessor"), TEXT("MobileNetV2ImageProcessor"), TEXT("MobileViTImageProcessor"),
			TEXT("OneFormerImageProcessor"), TEXT("Ovis2ImageProcessor"), TEXT("PoolFormerImageProcessor"), TEXT("RTDetrImageProcessor"), TEXT("SuperGlueImageProcessor"),
			TEXT("SuperPointImageProcessor"), TEXT("TextNetImageProcessor"), TEXT("TvltImageProcessor"), TEXT("TvpImageProcessor"), TEXT("VideoLlavaImageProcessor"),
			TEXT("VideoMAEImageProcessor"), TEXT("ViltImageProcessor"), TEXT("ViTHybridImageProcessor"), TEXT("VivitImageProcessor"), TEXT("YolosImageProcessor"),
		};
		static_assert(UE_ARRAY_COUNT(Classes) == 44, "C25 lists 44 ShortestEdgeIntClasses");
		for (const TCHAR* C : Classes)
		{
			if (Cls.Equals(C, ESearchCase::CaseSensitive))
			{
				return true;
			}
		}
		return false;
	}

	/** String value of Key, or "" when absent or not a string (no report message: the class name is only a hint). */
	static FString StringOrEmpty(const TSharedPtr<FJsonObject>& Obj, const TCHAR* Key)
	{
		FString S;
		const TSharedPtr<FJsonValue> V = GetValue(Obj, Key);
		if (!V.IsValid() || V->Type != EJson::String || !V->TryGetString(S))
		{
			return FString();
		}
		return S;
	}

	/**
	 * C25: processor class from image_processor_type, else feature_extractor_type with FeatureExtractor -> ImageProcessor;
	 * a trailing "Fast" is removed. "" when neither key holds a string.
	 */
	static FString ProcessorClassName(const TSharedPtr<FJsonObject>& Pre)
	{
		FString Cls = StringOrEmpty(Pre, TEXT("image_processor_type"));
		if (Cls.IsEmpty())
		{
			Cls = StringOrEmpty(Pre, TEXT("feature_extractor_type"));
			static const TCHAR* const FE = TEXT("FeatureExtractor");
			if (Cls.EndsWith(FE, ESearchCase::CaseSensitive))
			{
				Cls = Cls.LeftChop(FCString::Strlen(FE)) + TEXT("ImageProcessor");
			}
		}
		if (Cls.EndsWith(TEXT("Fast"), ESearchCase::CaseSensitive))
		{
			Cls.LeftChopInline(4);
		}
		return Cls;
	}

	/**
	 * C32 rule 1: processor classes whose slow __init__ default for `resample` is not Bilinear (transformers 4.57.6;
	 * authoritative list: RESAMPLE_DEFAULTS in Tools/make_import_test_data.py). Any other class -> Bilinear.
	 */
	static EPipelineImageResample DefaultResample(const FString& Cls)
	{
		struct FEntry { const TCHAR* Cls; EPipelineImageResample Resample; };
		static const FEntry Table[] = {
			{TEXT("AriaImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("BeitImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("BitImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("BlipImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("BridgeTowerImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("CLIPImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("ChineseCLIPImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("DPTImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("DeepseekVLHybridImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("DeepseekVLImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("DeiTImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("Emu3ImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("FlavaImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("Glm4vImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("GotOcr2ImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("InstructBlipVideoImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("JanusImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("LevitImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("LlavaImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("LlavaNextImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("LlavaNextVideoImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("LlavaOnevisionImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("Ovis2ImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("OwlViTImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("PerceiverImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("PixtralImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("PoolFormerImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("PromptDepthAnythingImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("Qwen2VLImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("SegGptImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("SiglipImageProcessor"), EPipelineImageResample::Bicubic}, {TEXT("VideoLlavaImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("ViltImageProcessor"), EPipelineImageResample::Bicubic},
			{TEXT("ChameleonImageProcessor"), EPipelineImageResample::Lanczos}, {TEXT("Idefics3ImageProcessor"), EPipelineImageResample::Lanczos},
			{TEXT("SmolVLMImageProcessor"), EPipelineImageResample::Lanczos},
			{TEXT("EfficientNetImageProcessor"), EPipelineImageResample::Nearest},
		};
		static_assert(UE_ARRAY_COUNT(Table) == 37, "C32 ResampleDefaults lists 37 classes (33 Bicubic, 3 Lanczos, 1 Nearest)");
		for (const FEntry& E : Table)
		{
			if (Cls.Equals(E.Cls, ESearchCase::CaseSensitive))
			{
				return E.Resample;
			}
		}
		return EPipelineImageResample::Bilinear;
	}

	/** Short text of a JSON value for a report message (any type, never fails). */
	static FString JsonValueText(const TSharedPtr<FJsonValue>& V)
	{
		if (!V.IsValid() || V->Type == EJson::None || V->Type == EJson::Null)
		{
			return TEXT("null");
		}
		switch (V->Type)
		{
		case EJson::Number:
		{
			double D = 0.0;
			V->TryGetNumber(D);
			return FString::Printf(TEXT("%.17g"), D);
		}
		case EJson::String:
		{
			FString S;
			V->TryGetString(S);
			if (S.Len() > 64)
			{
				S = S.Left(64) + TEXT("...");
			}
			return FString::Printf(TEXT("\"%s\""), *S);
		}
		case EJson::Boolean:
		{
			bool B = false;
			V->TryGetBool(B);
			return B ? TEXT("true") : TEXT("false");
		}
		default:
			return JsonTypeName(V->Type);
		}
	}

	/** C32 rule 1: resample from the file (an exact integer 0..5), else the class default; anything else -> Bilinear + Warning. */
	static void ReadResample(FContext& Ctx, const TSharedPtr<FJsonObject>& Pre, const TCHAR* File, const FString& Cls, FPipelinePreprocessSettings& P)
	{
		if (!Pre->HasField(TEXT("resample")))
		{
			P.Resample = DefaultResample(Cls);
			return;
		}
		// Raw field: a present null counts as an invalid value, not as absent.
		const TSharedPtr<FJsonValue> V = Pre->TryGetField(TEXT("resample"));
		double D = 0.0;
		if (V.IsValid() && V->Type == EJson::Number && V->TryGetNumber(D) && FMath::IsFinite(D)
			&& D == FMath::FloorToDouble(D) && D >= 0.0 && D <= 5.0)
		{
			P.Resample = static_cast<EPipelineImageResample>((uint8)D);
			return;
		}
		P.Resample = EPipelineImageResample::Bilinear;
		Ctx.Warn(ECode::UnsupportedResize, FString::Printf(TEXT("%s: resample %s is not a PIL filter 0..5; using Bilinear"), File, *JsonValueText(V)));
	}

	/** C25 rule 2: size {height RH, width RW}; StretchCenterCrop with a usable crop, else Stretch RW x RH. */
	static void ApplyHeightWidthSize(FContext& Ctx, const TSharedPtr<FJsonObject>& Pre, const TCHAR* File, int32 RW, int32 RH, FPipelinePreprocessSettings& P)
	{
		bool bCenterCrop = false;
		ReadBool(Ctx, Pre, File, TEXT("do_center_crop"), bCenterCrop);
		int32 CropW = 0, CropH = 0;
		const ERead CropRead = bCenterCrop ? ReadCropSize(Ctx, Pre, File, CropW, CropH) : ERead::Absent;
		if (CropRead == ERead::Ok)
		{
			P.ResizeMode = EPipelineImageResize::StretchCenterCrop;
			P.ResizeWidth = RW;
			P.ResizeHeight = RH;
			P.ResizeShortestEdge = 0;
			P.ImageWidth = CropW;
			P.ImageHeight = CropH;
		}
		else if (CropRead == ERead::Absent)
		{
			P.ResizeMode = EPipelineImageResize::Stretch;
			P.ResizeWidth = 0;
			P.ResizeHeight = 0;
			P.ImageWidth = RW;
			P.ImageHeight = RH;
			if (bCenterCrop && !GetValue(Pre, TEXT("crop_size")).IsValid())
			{
				// C32 rule 3.
				Ctx.Warn(ECode::UnsupportedResize, FString::Printf(TEXT("%s: do_center_crop is set but crop_size is missing; HF would use the processor's default crop. Using %dx%d Stretch; set the crop on the package if needed."),
					File, RW, RH));
			}
		}
		// ERead::Invalid: the InvalidJson error is already in the report.
	}

	/** Rule 7 (C25 rules 1-5): image settings from preprocessor_config.json. */
	static void ReadImageSettings(FContext& Ctx, const TSharedPtr<FJsonObject>& Pre, FPipelinePreprocessSettings& P)
	{
		static const TCHAR* const File = TEXT("preprocessor_config.json");
		if (!Pre.IsValid())
		{
			Ctx.Warn(ECode::MissingPreprocessorConfig, TEXT("preprocessor_config.json not found: using 224x224 Stretch, rescale 1/255 and ImageNet mean/std. Add the file from the model repository if the model expects other preprocessing."));
			return;
		}

		const FString ProcessorCls = ProcessorClassName(Pre);
		ReadResample(Ctx, Pre, File, ProcessorCls, P);

		const TSharedPtr<FJsonValue> SizeV = GetValue(Pre, TEXT("size"));
		if (SizeV.IsValid())
		{
			int32 S = 0;
			bool bHaveS = false;
			bool bShortestEdgeObject = false;
			if (SizeV->Type == EJson::Number)
			{
				bHaveS = ValueToInt(Ctx, SizeV, Where(File, TEXT("size")), S);
			}
			else if (SizeV->Type == EJson::Object && SizeV->AsObject().IsValid())
			{
				const TSharedPtr<FJsonObject>& SizeObj = SizeV->AsObject();
				const TSharedPtr<FJsonValue> HV = GetValue(SizeObj, TEXT("height"));
				const TSharedPtr<FJsonValue> WV = GetValue(SizeObj, TEXT("width"));
				const TSharedPtr<FJsonValue> SEV = GetValue(SizeObj, TEXT("shortest_edge"));
				if (HV.IsValid() && WV.IsValid())
				{
					int32 H = 0, W = 0;
					const bool bH = ValueToInt(Ctx, HV, Where(File, TEXT("size.height")), H);
					const bool bW = ValueToInt(Ctx, WV, Where(File, TEXT("size.width")), W);
					if (bH && bW)
					{
						// Rule 2.
						ApplyHeightWidthSize(Ctx, Pre, File, W, H, P);
					}
				}
				else if (SEV.IsValid())
				{
					bHaveS = ValueToInt(Ctx, SEV, Where(File, TEXT("size.shortest_edge")), S);
					bShortestEdgeObject = true;
				}
				else
				{
					Ctx.Warn(ECode::UnsupportedResize, FString::Printf(TEXT("%s: size has neither height/width nor shortest_edge; using %dx%d Stretch. Set the image size on the package if this is wrong."),
						File, P.ImageWidth, P.ImageHeight));
				}
			}
			else
			{
				TypeError(Ctx, Where(File, TEXT("size")), TEXT("an integer or an object with height/width or shortest_edge"), SizeV);
			}

			if (bHaveS)
			{
				double CropPct = 0.0;
				const ERead CropPctRead = ReadNumber(Ctx, Pre, File, TEXT("crop_pct"), CropPct);
				if (CropPctRead == ERead::Ok)
				{
					// Rule 1 (ConvNext), unchanged.
					if (!(CropPct > 0.0))
					{
						Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("%s: crop_pct must be > 0, found %.17g. Fix the value in the file."), File, CropPct));
					}
					else if (S < 384)
					{
						// ConvNext rule: resize shortest edge to int(S / crop_pct), then center-crop S x S.
						const double R = FMath::Clamp((double)S / CropPct, (double)MIN_int32, (double)MAX_int32);
						P.ResizeMode = EPipelineImageResize::ShortestEdgeCenterCrop;
						P.ResizeShortestEdge = (int32)R;
						P.ResizeWidth = 0;
						P.ResizeHeight = 0;
						P.ImageWidth = S;
						P.ImageHeight = S;
					}
					else
					{
						P.ResizeMode = EPipelineImageResize::Stretch;
						P.ResizeWidth = 0;
						P.ResizeHeight = 0;
						P.ImageWidth = S;
						P.ImageHeight = S;
					}
				}
				else if (CropPctRead == ERead::Absent)
				{
					const FString Cls = bShortestEdgeObject ? FString() : ProcessorCls;
					if (!bShortestEdgeObject && !Cls.IsEmpty() && !IsShortestEdgeIntClass(Cls))
					{
						// Rule 3: HF's square default for an integer size, i.e. {height S, width S}.
						ApplyHeightWidthSize(Ctx, Pre, File, S, S, P);
					}
					else
					{
						// Rule 4: shortest-edge reading.
						bool bCenterCrop = false;
						ReadBool(Ctx, Pre, File, TEXT("do_center_crop"), bCenterCrop);
						int32 CropW = 0, CropH = 0;
						const ERead CropRead = bCenterCrop ? ReadCropSize(Ctx, Pre, File, CropW, CropH) : ERead::Absent;
						if (CropRead == ERead::Ok)
						{
							P.ResizeMode = EPipelineImageResize::ShortestEdgeCenterCrop;
							// C32 rule 2: LeViT resizes the shortest edge to int(256/224 * S) before the centre crop.
							P.ResizeShortestEdge = ProcessorCls.Equals(TEXT("LevitImageProcessor"), ESearchCase::CaseSensitive)
								? (int32)FMath::Clamp((256.0 / 224.0) * (double)S, (double)MIN_int32, (double)MAX_int32)
								: S;
							P.ResizeWidth = 0;
							P.ResizeHeight = 0;
							P.ImageWidth = CropW;
							P.ImageHeight = CropH;
						}
						else if (CropRead == ERead::Absent)
						{
							P.ResizeMode = EPipelineImageResize::Stretch;
							P.ResizeWidth = 0;
							P.ResizeHeight = 0;
							P.ImageWidth = S;
							P.ImageHeight = S;
							if (bShortestEdgeObject)
							{
								Ctx.Warn(ECode::UnsupportedResize, FString::Printf(TEXT("%s: size {shortest_edge: %d} without crop_pct or do_center_crop/crop_size cannot be represented; using %dx%d Stretch. Check the package's image settings."),
									File, S, S, S));
							}
							else if (!Cls.IsEmpty())
							{
								Ctx.Warn(ECode::UnsupportedResize, FString::Printf(TEXT("%s: %s resizes an integer size %d by its shortest edge (aspect kept), which cannot be represented without do_center_crop/crop_size; using %dx%d Stretch. Check the package's image settings."),
									File, *Cls, S, S, S));
							}
						}
					}
				}
			}
		}

		bool bDoRescale = true;
		ReadBool(Ctx, Pre, File, TEXT("do_rescale"), bDoRescale);
		if (bDoRescale)
		{
			double Rescale = 1.0 / 255.0;
			ReadNumber(Ctx, Pre, File, TEXT("rescale_factor"), Rescale);
			P.RescaleFactor = (float)Rescale;
		}
		else
		{
			P.RescaleFactor = 1.f;
		}

		bool bDoNormalize = true;
		ReadBool(Ctx, Pre, File, TEXT("do_normalize"), bDoNormalize);
		if (bDoNormalize)
		{
			ReadVector3(Ctx, Pre, File, TEXT("image_mean"), P.ImageMean);
			ReadVector3(Ctx, Pre, File, TEXT("image_std"), P.ImageStd);
		}
		else
		{
			P.ImageMean = FVector(0.0, 0.0, 0.0);
			P.ImageStd = FVector(1.0, 1.0, 1.0);
		}
	}

	/** Rule 5: labels from config.json id2label. */
	static void ReadLabels(FContext& Ctx, const TSharedPtr<FJsonObject>& Config, TArray<FString>& OutLabels)
	{
		static const TCHAR* const File = TEXT("config.json");
		if (!Config.IsValid())
		{
			Ctx.Error(ECode::MissingConfig, TEXT("config.json not found: it is needed for the class labels (id2label). Add the model repository's config.json to the folder."));
			return;
		}
		TSharedPtr<FJsonObject> Id2Label;
		const ERead Read = ReadObject(Ctx, Config, File, TEXT("id2label"), Id2Label);
		if (Read == ERead::Invalid)
		{
			return;
		}
		if (Read == ERead::Absent || Id2Label->Values.Num() == 0)
		{
			Ctx.Error(ECode::MissingLabels, TEXT("config.json: id2label is missing or empty; a classification or detection model needs one label per class. Add id2label {\"0\": \"...\", ...} to config.json."));
			return;
		}

		TArray<TPair<int64, FString>> Entries;
		bool bOk = true;
		for (const auto& Pair : Id2Label->Values)
		{
			const FString Key(*Pair.Key);
			int64 Id = 0;
			if (!ParseIntegerKey(Key, Id))
			{
				Ctx.Error(ECode::InvalidLabels, FString::Printf(TEXT("config.json: id2label key '%s' is not an integer; keys must be 0..N-1."), *Key));
				bOk = false;
				continue;
			}
			const TSharedPtr<FJsonValue>& V = Pair.Value;
			FString Label;
			if (!V.IsValid() || V->Type != EJson::String || !V->TryGetString(Label))
			{
				Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("config.json: id2label[\"%s\"] must be a string, found %s."),
					*Key, V.IsValid() ? JsonTypeName(V->Type) : TEXT("nothing")));
				bOk = false;
				continue;
			}
			Entries.Emplace(Id, MoveTemp(Label));
		}
		if (!bOk)
		{
			return;
		}
		Entries.Sort([](const TPair<int64, FString>& A, const TPair<int64, FString>& B) { return A.Key < B.Key; });
		bool bContiguous = true;
		for (int32 i = 0; i < Entries.Num(); ++i)
		{
			bContiguous &= (Entries[i].Key == (int64)i);
		}
		if (!bContiguous)
		{
			TArray<FString> Found;
			for (int32 i = 0; i < Entries.Num() && i < 20; ++i)
			{
				Found.Add(LexToString(Entries[i].Key));
			}
			if (Entries.Num() > 20)
			{
				Found.Add(TEXT("..."));
			}
			Ctx.Error(ECode::InvalidLabels, FString::Printf(TEXT("config.json: id2label keys must be 0..%d, found %s. Renumber the labels without gaps or duplicates."),
				Entries.Num() - 1, *FString::Join(Found, TEXT(","))));
			return;
		}
		OutLabels.Reserve(Entries.Num());
		for (TPair<int64, FString>& E : Entries)
		{
			OutLabels.Add(MoveTemp(E.Value));
		}
	}

	/** One parsed modules.json. */
	struct FModules
	{
		bool bPresent = false;
		bool bHasPooling = false;
		FString PoolingPath;
		bool bHasNormalize = false;
	};

	/** Reads modules.json (a JSON array of objects). Returns false on InvalidJson. */
	static bool LoadModules(FContext& Ctx, FModules& Out)
	{
		static const TCHAR* const File = TEXT("modules.json");
		TSharedPtr<FJsonValue> Value;
		if (!LoadJson(Ctx, File, Value, Out.bPresent))
		{
			return false;
		}
		if (!Out.bPresent)
		{
			return true;
		}
		if (Value->Type != EJson::Array)
		{
			Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("%s: the top level must be a JSON array of modules, found %s."), File, JsonTypeName(Value->Type)));
			return false;
		}
		const TArray<TSharedPtr<FJsonValue>>& Entries = Value->AsArray();
		bool bOk = true;
		for (int32 i = 0; i < Entries.Num(); ++i)
		{
			const TSharedPtr<FJsonValue>& E = Entries[i];
			if (!E.IsValid() || E->Type != EJson::Object || !E->AsObject().IsValid())
			{
				Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("%s: entry %d must be an object, found %s."), File, i, E.IsValid() ? JsonTypeName(E->Type) : TEXT("nothing")));
				bOk = false;
				continue;
			}
			const TSharedPtr<FJsonObject>& Obj = E->AsObject();
			const FString EntryWhere = FString::Printf(TEXT("%s[%d]"), File, i);
			FString Type;
			if (ReadString(Ctx, Obj, *EntryWhere, TEXT("type"), Type) == ERead::Invalid)
			{
				bOk = false;
				continue;
			}
			if (Type == TEXT("sentence_transformers.models.Normalize"))
			{
				Out.bHasNormalize = true;
			}
			else if (Type == TEXT("sentence_transformers.models.Pooling") && !Out.bHasPooling)
			{
				FString Path;
				if (ReadString(Ctx, Obj, *EntryWhere, TEXT("path"), Path) == ERead::Invalid)
				{
					bOk = false;
					continue;
				}
				Out.bHasPooling = true;
				Out.PoolingPath = Path;
			}
		}
		return bOk;
	}

	static FString PoolingConfigRel(const FModules& Modules)
	{
		FString Dir = Modules.PoolingPath;
		Dir.ReplaceInline(TEXT("\\"), TEXT("/"));
		while (Dir.EndsWith(TEXT("/")))
		{
			Dir.LeftChopInline(1);
		}
		return Dir.IsEmpty() ? FString(TEXT("config.json")) : Dir + TEXT("/config.json");
	}

	/** Rule 6: text settings. */
	static void ReadTextSettings(FContext& Ctx, const TSharedPtr<FJsonObject>& Config, const TSharedPtr<FJsonObject>& TokenizerConfig,
		const TSharedPtr<FJsonObject>& SbertConfig, const FModules& Modules, const TSharedPtr<FJsonObject>& PoolingConfig,
		const FString& PoolingRel, FPipelinePreprocessSettings& P)
	{
		int32 MaxTokens = 0;
		bool bHave = false;

		int32 SeqLen = 0;
		if (ReadInt(Ctx, SbertConfig, TEXT("sentence_bert_config.json"), TEXT("max_seq_length"), SeqLen) == ERead::Ok && SeqLen > 0)
		{
			MaxTokens = SeqLen;
			bHave = true;
		}
		if (!bHave)
		{
			double ModelMax = 0.0;
			if (ReadNumber(Ctx, TokenizerConfig, TEXT("tokenizer_config.json"), TEXT("model_max_length"), ModelMax) == ERead::Ok
				&& ModelMax > 0.0 && ModelMax <= 65536.0 && (int32)ModelMax > 0)
			{
				MaxTokens = (int32)ModelMax;
				bHave = true;
			}
		}
		int32 MaxPos = 0;
		const bool bHaveMaxPos = ReadInt(Ctx, Config, TEXT("config.json"), TEXT("max_position_embeddings"), MaxPos) == ERead::Ok && MaxPos > 0;
		if (!bHave && bHaveMaxPos)
		{
			MaxTokens = MaxPos;
			bHave = true;
		}
		if (!bHave)
		{
			MaxTokens = 512;
		}
		if (bHaveMaxPos)
		{
			MaxTokens = FMath::Min(MaxTokens, MaxPos);
		}
		P.MaxTokens = MaxTokens;
		P.bAddSpecialTokens = true;

		P.Pooling = EPipelinePooling::Mean;
		if (Modules.bHasPooling)
		{
			bool bCls = false;
			bool bMean = false;
			if (PoolingConfig.IsValid())
			{
				ReadBool(Ctx, PoolingConfig, *PoolingRel, TEXT("pooling_mode_cls_token"), bCls);
				ReadBool(Ctx, PoolingConfig, *PoolingRel, TEXT("pooling_mode_mean_tokens"), bMean);
			}
			if (bCls)
			{
				P.Pooling = EPipelinePooling::CLS;
			}
			else if (!bMean)
			{
				Ctx.Warn(ECode::UnsupportedPooling, PoolingConfig.IsValid()
					? FString::Printf(TEXT("%s: pooling mode is neither pooling_mode_cls_token nor pooling_mode_mean_tokens; Mean pooling is used, so embeddings may differ from sentence-transformers."), *PoolingRel)
					: FString::Printf(TEXT("%s not found for the Pooling module in modules.json; Mean pooling is used. Add the pooling config from the model repository."), *PoolingRel));
			}
		}
		P.bNormalizeEmbeddings = Modules.bHasNormalize;
	}

	/** Rule 8. */
	static EPipelineScoreActivation DetectionActivation(FContext& Ctx, const FString& ModelType)
	{
		static const TCHAR* const SigmoidFamilies[] = {TEXT("rt_detr"), TEXT("rt_detr_v2"), TEXT("d_fine"), TEXT("deformable_detr"), TEXT("conditional_detr"), TEXT("dab_detr")};
		static const TCHAR* const SoftmaxFamilies[] = {TEXT("detr"), TEXT("yolos"), TEXT("table-transformer")};
		for (const TCHAR* F : SigmoidFamilies)
		{
			if (ModelType == F)
			{
				return EPipelineScoreActivation::Sigmoid;
			}
		}
		for (const TCHAR* F : SoftmaxFamilies)
		{
			if (ModelType == F)
			{
				return EPipelineScoreActivation::Softmax;
			}
		}
		Ctx.Warn(ECode::UnverifiedDetectionFamily, FString::Printf(TEXT("config.json: detection model_type '%s' is not a verified family; Sigmoid scores are assumed. Check the detections, and set Postprocess.ScoreActivation on the package if they look wrong."), *ModelType));
		return EPipelineScoreActivation::Sigmoid;
	}

	static bool AddedError(const FPipelineValidationReport& Report, int32 FirstMessage)
	{
		for (int32 i = FirstMessage; i < Report.Messages.Num(); ++i)
		{
			if (Report.Messages[i].Severity == ESev::Error)
			{
				return true;
			}
		}
		return false;
	}
}

bool ParsePipelineModelFolder(const FString& Folder, EPipelineTask TaskOverride, FPipelineModelFolderInfo& OutInfo, FPipelineValidationReport& InOutReport)
{
	using namespace UE::ModelPipelines::FolderPrivate;

	OutInfo = FPipelineModelFolderInfo();
	const int32 FirstMessage = InOutReport.Messages.Num();
	FContext Ctx(InOutReport);
	IFileManager& FM = IFileManager::Get();

	// 1. Folder and model file.
	if (Folder.TrimStartAndEnd().IsEmpty())
	{
		Ctx.Error(ECode::FolderNotFound, TEXT("Model folder path is empty. Pass the folder that holds model.onnx and config.json."));
		return false;
	}
	Ctx.Folder = NormalizeFolderPath(Folder);
	OutInfo.FolderPath = Ctx.Folder;
	if (!FM.DirectoryExists(*Ctx.Folder))
	{
		Ctx.Error(ECode::FolderNotFound, FString::Printf(TEXT("Model folder '%s' does not exist or is not a folder. Pass the folder that holds model.onnx."), *Ctx.Folder));
		return false;
	}
	{
		const FString Candidates[] = {Ctx.Folder + TEXT("/model.onnx"), Ctx.Folder + TEXT("/onnx/model.onnx")};
		// C14 rule 1 is literal: the first candidate that exists is THE model file (onnx/model.onnx only when
		// model.onnx does not exist); an empty one is an error even if the other candidate exists.
		FString EmptyFile;
		for (const FString& C : Candidates)
		{
			if (FM.FileExists(*C))
			{
				if (FM.FileSize(*C) > 0)
				{
					OutInfo.ModelFile = C;
				}
				else
				{
					EmptyFile = C;
				}
				break;
			}
		}
		if (OutInfo.ModelFile.IsEmpty())
		{
			Ctx.Error(ECode::MissingModelFile, EmptyFile.IsEmpty()
				? FString::Printf(TEXT("No model file in '%s': expected model.onnx or onnx/model.onnx. Export the model to ONNX (e.g. optimum-cli export onnx) into this folder."), *Ctx.Folder)
				: FString::Printf(TEXT("Model file '%s' is empty (0 bytes). Re-export or re-download the model."), *EmptyFile));
		}
	}

	// 2. Every JSON file that exists must parse.
	bool bJsonOk = true;
	bool bPresent = false;
	TSharedPtr<FJsonObject> Sidecar, Config, TokenizerConfig, PreprocessorConfig, SbertConfig, PoolingConfig;
	bJsonOk &= LoadJsonObject(Ctx, TEXT("modelpipelines.json"), Sidecar, bPresent);
	bJsonOk &= LoadJsonObject(Ctx, TEXT("config.json"), Config, bPresent);
	const bool bHasConfig = bPresent;
	bJsonOk &= LoadJsonObject(Ctx, TEXT("tokenizer_config.json"), TokenizerConfig, bPresent);
	bJsonOk &= LoadJsonObject(Ctx, TEXT("preprocessor_config.json"), PreprocessorConfig, bPresent);
	bJsonOk &= LoadJsonObject(Ctx, TEXT("sentence_bert_config.json"), SbertConfig, bPresent);
	FModules Modules;
	bJsonOk &= LoadModules(Ctx, Modules);
	FString PoolingRel;
	if (Modules.bHasPooling)
	{
		PoolingRel = PoolingConfigRel(Modules);
		bJsonOk &= LoadJsonObject(Ctx, PoolingRel, PoolingConfig, bPresent);
	}
	if (!bJsonOk)
	{
		return false;
	}

	// Top-level config.json identity (not the nested backbone_config.model_type).
	ReadString(Ctx, Config, TEXT("config.json"), TEXT("model_type"), OutInfo.ModelType);
	{
		const TSharedPtr<FJsonValue> Arch = GetValue(Config, TEXT("architectures"));
		if (Arch.IsValid())
		{
			if (Arch->Type != EJson::Array)
			{
				TypeError(Ctx, TEXT("config.json: architectures"), TEXT("an array of strings"), Arch);
			}
			else if (Arch->AsArray().Num() > 0)
			{
				const TSharedPtr<FJsonValue>& First = Arch->AsArray()[0];
				FString Name;
				if (!First.IsValid() || First->Type != EJson::String || !First->TryGetString(Name))
				{
					Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("config.json: architectures[0] must be a string, found %s."), First.IsValid() ? JsonTypeName(First->Type) : TEXT("nothing")));
				}
				else
				{
					OutInfo.Architecture = Name;
				}
			}
		}
	}

	// 9. License (sidecar fields win, README front matter for the SPDX id).
	{
		TSharedPtr<FJsonObject> Lic;
		ReadObject(Ctx, Sidecar, TEXT("modelpipelines.json"), TEXT("license"), Lic);
		FString Spdx, Url, Attribution;
		ReadString(Ctx, Lic, TEXT("modelpipelines.json: license"), TEXT("spdx_id"), Spdx);
		ReadString(Ctx, Lic, TEXT("modelpipelines.json: license"), TEXT("source_url"), Url);
		ReadString(Ctx, Lic, TEXT("modelpipelines.json: license"), TEXT("attribution"), Attribution);
		OutInfo.License.SpdxId = Spdx.TrimStartAndEnd().IsEmpty() ? ReadmeLicense(Ctx.Folder) : Spdx;
		OutInfo.License.SourceUrl = Url;
		OutInfo.License.Attribution = Attribution;
	}

	// 10. Preferred device.
	{
		FString Device;
		if (ReadString(Ctx, Sidecar, TEXT("modelpipelines.json"), TEXT("preferred_device"), Device) == ERead::Ok)
		{
			if (Device.Equals(TEXT("Auto"), ESearchCase::IgnoreCase)) { OutInfo.PreferredDevice = EPipelineDevice::Auto; }
			else if (Device.Equals(TEXT("CPU"), ESearchCase::IgnoreCase)) { OutInfo.PreferredDevice = EPipelineDevice::CPU; }
			else if (Device.Equals(TEXT("GPU"), ESearchCase::IgnoreCase)) { OutInfo.PreferredDevice = EPipelineDevice::GPU; }
			else
			{
				Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("modelpipelines.json: preferred_device '%s' is not Auto, CPU or GPU."), *Device));
			}
		}
	}

	// 3. Task.
	EPipelineTask Task = EPipelineTask::None;
	if (TaskOverride != EPipelineTask::None)
	{
		if ((uint8)TaskOverride > (uint8)EPipelineTask::ObjectDetection)
		{
			Ctx.Error(ECode::TaskUndetected, FString::Printf(TEXT("Task override value %d is not a valid task."), (int32)(uint8)TaskOverride));
			return false;
		}
		Task = TaskOverride;
		OutInfo.TaskSource = TEXT("options");
	}
	else
	{
		const TSharedPtr<FJsonValue> TaskV = GetValue(Sidecar, TEXT("task"));
		if (TaskV.IsValid())
		{
			FString Name;
			if (TaskV->Type != EJson::String || !TaskV->TryGetString(Name))
			{
				TypeError(Ctx, TEXT("modelpipelines.json: task"), TEXT("a task name string"), TaskV);
				return false;
			}
			EPipelineTask Parsed = EPipelineTask::None;
			if (!LexTryParseString(Parsed, *Name) || Parsed == EPipelineTask::None)
			{
				Ctx.Error(ECode::InvalidJson, FString::Printf(TEXT("modelpipelines.json: task '%s' is unknown; use TextEmbedding, TextClassification, ImageClassification or ObjectDetection."), *Name));
				return false;
			}
			Task = Parsed;
			OutInfo.TaskSource = TEXT("modelpipelines.json");
		}
		else if (Modules.bPresent)
		{
			Task = EPipelineTask::TextEmbedding;
			OutInfo.TaskSource = TEXT("modules.json");
		}
		else if (!OutInfo.Architecture.IsEmpty())
		{
			const FString& A = OutInfo.Architecture;
			if (A.EndsWith(TEXT("ForSequenceClassification"), ESearchCase::CaseSensitive)) { Task = EPipelineTask::TextClassification; }
			else if (A.EndsWith(TEXT("ForImageClassification"), ESearchCase::CaseSensitive)) { Task = EPipelineTask::ImageClassification; }
			else if (A.EndsWith(TEXT("ForObjectDetection"), ESearchCase::CaseSensitive)) { Task = EPipelineTask::ObjectDetection; }
			else if (A.EndsWith(TEXT("Model"), ESearchCase::CaseSensitive) && FM.FileExists(*(Ctx.Folder / TEXT("tokenizer.json")))) { Task = EPipelineTask::TextEmbedding; }
			if (Task != EPipelineTask::None)
			{
				OutInfo.TaskSource = TEXT("architectures");
			}
		}
		if (Task == EPipelineTask::None)
		{
			Ctx.Error(ECode::TaskUndetected, OutInfo.Architecture.IsEmpty()
				? FString(TEXT("Could not detect the task: no modules.json, no sidecar task and no config.json architectures. Choose the task in the import options or add {\"task\": \"...\"} to modelpipelines.json."))
				: FString::Printf(TEXT("Could not detect the task from config.json architectures[0] '%s'. Choose the task in the import options or add {\"task\": \"...\"} to modelpipelines.json."), *OutInfo.Architecture));
			if (!bHasConfig)
			{
				Ctx.Error(ECode::MissingConfig, TEXT("config.json not found: it is needed to detect the task. Add the model repository's config.json to the folder."));
			}
			return false;
		}
	}
	OutInfo.Task = Task;

	// 4. Tokenizer.
	if (IsTextTask(Task))
	{
		const FString TokPath = Ctx.Folder / TEXT("tokenizer.json");
		if (!FM.FileExists(*TokPath))
		{
			Ctx.Error(ECode::MissingTokenizer, FString::Printf(TEXT("tokenizer.json not found, but %s needs it. Add the model repository's tokenizer.json (optimum-cli export onnx writes one)."), LexToString(Task)));
		}
		else if (!FFileHelper::LoadFileToString(OutInfo.TokenizerJson, *TokPath))
		{
			OutInfo.TokenizerJson.Reset();
			Ctx.Error(ECode::MissingTokenizer, TEXT("tokenizer.json could not be read. Check its permissions."));
		}
	}

	// 5. Labels.
	if (TaskUsesLabels(Task))
	{
		ReadLabels(Ctx, Config, OutInfo.Labels);
	}

	// 6. Text settings.
	if (IsTextTask(Task))
	{
		ReadTextSettings(Ctx, Config, TokenizerConfig, SbertConfig, Modules, PoolingConfig, PoolingRel, OutInfo.Preprocess);
	}

	// 7. Image settings.
	if (IsImageTask(Task))
	{
		ReadImageSettings(Ctx, PreprocessorConfig, OutInfo.Preprocess);
	}

	// 8. Score activation.
	if (Task == EPipelineTask::TextClassification || Task == EPipelineTask::ImageClassification)
	{
		FString ProblemType;
		ReadString(Ctx, Config, TEXT("config.json"), TEXT("problem_type"), ProblemType);
		OutInfo.Postprocess.ScoreActivation = ProblemType == TEXT("multi_label_classification")
			? EPipelineScoreActivation::Sigmoid : EPipelineScoreActivation::Softmax;
	}
	else if (Task == EPipelineTask::ObjectDetection)
	{
		OutInfo.Postprocess.ScoreActivation = DetectionActivation(Ctx, OutInfo.ModelType);
	}
	else
	{
		OutInfo.Postprocess.ScoreActivation = EPipelineScoreActivation::Softmax;
	}

	return !AddedError(InOutReport, FirstMessage);
}
