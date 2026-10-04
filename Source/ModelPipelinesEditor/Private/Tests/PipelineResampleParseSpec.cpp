// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
// WP4.4b (docs/contracts.md C32): resample filter, LeViT shortest-edge rule, crop-without-size warning.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"
#include "PipelineModelFolder.h"
#include "PipelineEditorTestData.h"

namespace PipelineResampleParseSpecPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;

	static int32 CountMsg(const FPipelineValidationReport& R, ECode Code, ESev Sev)
	{
		int32 N = 0;
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code && M.Severity == Sev) { ++N; }
		}
		return N;
	}

	static bool AnyMessageContains(const FPipelineValidationReport& R, ECode Code, ESev Sev, const TCHAR* Substr)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code && M.Severity == Sev && M.Message.Contains(Substr)) { return true; }
		}
		return false;
	}

	static const TCHAR* ResampleName(EPipelineImageResample V)
	{
		switch (V)
		{
		case EPipelineImageResample::Nearest: return TEXT("Nearest");
		case EPipelineImageResample::Lanczos: return TEXT("Lanczos");
		case EPipelineImageResample::Bilinear: return TEXT("Bilinear");
		case EPipelineImageResample::Bicubic: return TEXT("Bicubic");
		case EPipelineImageResample::Box: return TEXT("Box");
		case EPipelineImageResample::Hamming: return TEXT("Hamming");
		}
		return TEXT("?");
	}

	static const TCHAR* ResizeName(EPipelineImageResize R)
	{
		switch (R)
		{
		case EPipelineImageResize::Stretch: return TEXT("Stretch");
		case EPipelineImageResize::ShortestEdgeCenterCrop: return TEXT("ShortestEdgeCenterCrop");
		case EPipelineImageResize::StretchCenterCrop: return TEXT("StretchCenterCrop");
		}
		return TEXT("?");
	}

	static const TCHAR* ImageConfigJson = TEXT(R"({"architectures":["ViTForImageClassification"],"model_type":"vit","id2label":{"0":"a","1":"b","2":"c"}})");
}

BEGIN_DEFINE_SPEC(FPipelineResampleParseSpec, "Pipelines.Editor.ModelFolder.Resample",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	TArray<FString> TempRoots;

	/** Writes config.json and a copy of the tiny model; preprocessor_config.json only when PreprocessorJson is non-null. */
	FString MakeFolder(const FString* PreprocessorJson)
	{
		FString Root = FPaths::ConvertRelativePathToFull(FPaths::ProjectIntermediateDir() / TEXT("PipelinesResampleTests") / FGuid::NewGuid().ToString());
		TempRoots.Add(Root);
		IFileManager::Get().MakeDirectory(*Root, true);
		const FString Src = UE::ModelPipelines::EditorTests::GetModelsDir() / TEXT("import_test/tiny_image_classification/model.onnx");
		if (IFileManager::Get().Copy(*(Root / TEXT("model.onnx")), *Src) != COPY_OK)
		{
			AddError(FString::Printf(TEXT("cannot copy '%s'; generate it with: python Tools/make_import_test_data.py"), *Src));
			return FString();
		}
		FFileHelper::SaveStringToFile(PipelineResampleParseSpecPrivate::ImageConfigJson, *(Root / TEXT("config.json")), FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
		if (PreprocessorJson)
		{
			FFileHelper::SaveStringToFile(*PreprocessorJson, *(Root / TEXT("preprocessor_config.json")), FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
		}
		return Root;
	}

	/** Parses a folder built from the JSON; false (error added) if the folder could not be made or parse failed. */
	bool Parse(const FString* Json, FPipelineModelFolderInfo& Info, FPipelineValidationReport& Report, FString& Ctx)
	{
		const FString Folder = MakeFolder(Json);
		if (Folder.IsEmpty()) { return false; }
		const bool bOk = ParsePipelineModelFolder(Folder, EPipelineTask::None, Info, Report);
		Ctx = FString::Printf(TEXT("json=%s report:\n%s"), Json ? **Json : TEXT("<none>"), *Report.ToString());
		TestTrue(*(Ctx + TEXT(" parse ok")), bOk);
		TestEqual(*(Ctx + TEXT(" no errors")), Report.NumErrors(), 0);
		return bOk;
	}

	/** ViT {32,32} stretch config with a raw JSON literal for "resample" (or none when empty). */
	static FString VitWithResample(const FString& ResampleLiteral)
	{
		if (ResampleLiteral.IsEmpty())
		{
			return TEXT(R"({"image_processor_type":"ViTImageProcessor","size":{"height":32,"width":32}})");
		}
		return FString::Printf(TEXT(R"({"image_processor_type":"ViTImageProcessor","size":{"height":32,"width":32},"resample":%s})"), *ResampleLiteral);
	}

	/** Valid integer resample: maps to that enumerator with no warnings. */
	void CheckValid(const FString& Literal, EPipelineImageResample Expected)
	{
		using namespace PipelineResampleParseSpecPrivate;
		const FString Json = VitWithResample(Literal);
		FPipelineModelFolderInfo Info;
		FPipelineValidationReport Report;
		FString Ctx;
		if (!Parse(&Json, Info, Report, Ctx)) { return; }
		TestEqual(*(Ctx + TEXT(" Resample")), FString(ResampleName(Info.Preprocess.Resample)), FString(ResampleName(Expected)));
		TestEqual(*(Ctx + TEXT(" no warnings")), Report.NumWarnings(), 0);
		TestEqual(*(Ctx + TEXT(" ResizeMode stays Stretch")), FString(ResizeName(Info.Preprocess.ResizeMode)), FString(TEXT("Stretch")));
	}

	/** Invalid resample: Bilinear + exactly one UnsupportedResize Warning mentioning resample and Bilinear. */
	void CheckInvalid(const FString& Literal)
	{
		using namespace PipelineResampleParseSpecPrivate;
		const FString Json = VitWithResample(Literal);
		FPipelineModelFolderInfo Info;
		FPipelineValidationReport Report;
		FString Ctx;
		if (!Parse(&Json, Info, Report, Ctx)) { return; }
		TestEqual(*(Ctx + TEXT(" Resample falls back to Bilinear")), FString(ResampleName(Info.Preprocess.Resample)), FString(TEXT("Bilinear")));
		TestEqual(*(Ctx + TEXT(" one UnsupportedResize warning")), CountMsg(Report, ECode::UnsupportedResize, ESev::Warning), 1);
		TestEqual(*(Ctx + TEXT(" total warnings")), Report.NumWarnings(), 1);
		TestTrue(*(Ctx + TEXT(" message names resample")), AnyMessageContains(Report, ECode::UnsupportedResize, ESev::Warning, TEXT("resample")));
		TestTrue(*(Ctx + TEXT(" message says Bilinear")), AnyMessageContains(Report, ECode::UnsupportedResize, ESev::Warning, TEXT("Bilinear")));
		TestEqual(*(Ctx + TEXT(" size still Stretch 32")), FString(ResizeName(Info.Preprocess.ResizeMode)), FString(TEXT("Stretch")));
		TestEqual(*(Ctx + TEXT(" ImageWidth")), Info.Preprocess.ImageWidth, 32);
	}

	/** Absent resample with a class: expected default, no warnings. Json is the full file content. */
	void CheckDefault(const FString& Json, EPipelineImageResample Expected)
	{
		using namespace PipelineResampleParseSpecPrivate;
		FPipelineModelFolderInfo Info;
		FPipelineValidationReport Report;
		FString Ctx;
		if (!Parse(&Json, Info, Report, Ctx)) { return; }
		TestEqual(*(Ctx + TEXT(" Resample")), FString(ResampleName(Info.Preprocess.Resample)), FString(ResampleName(Expected)));
		TestEqual(*(Ctx + TEXT(" no warnings")), Report.NumWarnings(), 0);
	}
END_DEFINE_SPEC(FPipelineResampleParseSpec)

void FPipelineResampleParseSpec::Define()
{
	using namespace PipelineResampleParseSpecPrivate;

	AfterEach([this]()
	{
		for (const FString& Dir : TempRoots)
		{
			IFileManager::Get().DeleteDirectory(*Dir, false, true);
		}
		TempRoots.Reset();
	});

	Describe("Explicit", [this]()
	{
		// Each PIL value 0..5 must map to its own enumerator. Catches an off-by-one table, or a parser that only
		// knows Bilinear/Bicubic (which would give Bilinear for 0, 1, 4, 5).
		for (int32 V = 0; V <= 5; ++V)
		{
			It(*FString::Printf(TEXT("resample %d maps to the PIL enumerator with the same value"), V), [this, V]()
			{
				CheckValid(FString::FromInt(V), static_cast<EPipelineImageResample>(V));
			});
		}

		// JSON 3.0 is an integral number (Python float.is_integer()). Catches a parser that only accepts "int" tokens
		// and warns for 3.0 (Bilinear), which HF would treat as Bicubic.
		It("resample 3.0 (integral float) is Bicubic with no warning", [this]()
		{
			CheckValid(TEXT("3.0"), EPipelineImageResample::Bicubic);
		});

		// Same for 0.0: must not be confused with "absent".
		It("resample 0.0 is Nearest with no warning", [this]()
		{
			CheckValid(TEXT("0.0"), EPipelineImageResample::Nearest);
		});

		// An explicit value beats the class default (DeiT default is Bicubic).
		It("explicit resample 2 overrides the DeiT Bicubic default", [this]()
		{
			CheckDefault(TEXT(R"({"image_processor_type":"DeiTImageProcessor","size":{"height":32,"width":32},"resample":2})"), EPipelineImageResample::Bilinear);
		});

		// Explicit 0 on a class whose default is Bicubic: 0 must not be treated as "unset".
		It("explicit resample 0 on DeiT is Nearest, not the class default", [this]()
		{
			CheckDefault(TEXT(R"({"image_processor_type":"DeiTImageProcessor","size":{"height":32,"width":32},"resample":0})"), EPipelineImageResample::Nearest);
		});
	});

	Describe("Invalid", [this]()
	{
		// 3.5 is a number but not an integer: Bilinear + Warning (not truncated to Bicubic).
		It("resample 3.5 is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("3.5")); });
		// Strings are never parsed, even digit strings (HF would pass the string to PIL and fail).
		It("resample \"3\" (string) is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("\"3\"")); });
		It("resample \"bicubic\" (string) is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("\"bicubic\"")); });
		// null is present-but-invalid, not absent: must warn (and with ViT the default is Bilinear anyway).
		It("resample null is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("null")); });
		// Python excludes bool even though True == 1; a parser reading bool as number would give Bilinear silently (value 1 = Lanczos).
		It("resample true (bool) is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("true")); });
		It("resample false (bool) is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("false")); });
		// Out of range both sides.
		It("resample 6 is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("6")); });
		It("resample 7 is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("7")); });
		It("resample -1 is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("-1")); });
		// Container types must not crash and must warn.
		It("resample [3] (array) is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("[3]")); });
		It("resample {\"a\":3} (object) is Bilinear with an UnsupportedResize warning", [this]() { CheckInvalid(TEXT("{\"a\":3}")); });

		// An invalid resample on a Bicubic-default class: still Bilinear (not the class default) and warns.
		It("DeiT with resample \"x\" is Bilinear (not the class default) with a warning", [this]()
		{
			using namespace PipelineResampleParseSpecPrivate;
			const FString Json = TEXT(R"({"image_processor_type":"DeiTImageProcessor","size":{"height":32,"width":32},"resample":"x"})");
			FPipelineModelFolderInfo Info;
			FPipelineValidationReport Report;
			FString Ctx;
			if (!Parse(&Json, Info, Report, Ctx)) { return; }
			TestEqual(*(Ctx + TEXT(" Resample")), FString(ResampleName(Info.Preprocess.Resample)), FString(TEXT("Bilinear")));
			TestEqual(*(Ctx + TEXT(" one UnsupportedResize warning")), CountMsg(Report, ECode::UnsupportedResize, ESev::Warning), 1);
		});
	});

	Describe("ClassDefaults", [this]()
	{
		// DeiT's __init__ default is Bicubic (3). Catches a parser that returns Bilinear whenever resample is absent.
		It("absent resample + DeiTImageProcessor is Bicubic", [this]()
		{
			CheckDefault(TEXT(R"({"image_processor_type":"DeiTImageProcessor","size":{"height":32,"width":32}})"), EPipelineImageResample::Bicubic);
		});
		// The Fast suffix is stripped before the table lookup.
		It("absent resample + DeiTImageProcessorFast is Bicubic", [this]()
		{
			CheckDefault(TEXT(R"({"image_processor_type":"DeiTImageProcessorFast","size":{"height":32,"width":32}})"), EPipelineImageResample::Bicubic);
		});
		// Legacy key with FeatureExtractor -> ImageProcessor rewrite.
		It("absent resample + DeiTFeatureExtractor (legacy key) is Bicubic", [this]()
		{
			CheckDefault(TEXT(R"({"feature_extractor_type":"DeiTFeatureExtractor","size":{"height":32,"width":32}})"), EPipelineImageResample::Bicubic);
		});
		// ViT's default is Bilinear (2); catches a parser that returns Bicubic for every known class.
		It("absent resample + ViTImageProcessor is Bilinear", [this]()
		{
			CheckDefault(TEXT(R"({"image_processor_type":"ViTImageProcessor","size":{"height":32,"width":32}})"), EPipelineImageResample::Bilinear);
		});
		It("absent resample + no processor class is Bilinear", [this]()
		{
			CheckDefault(TEXT(R"({"size":{"height":32,"width":32}})"), EPipelineImageResample::Bilinear);
		});
		// Unknown class name is not in the table.
		It("absent resample + unknown class is Bilinear", [this]()
		{
			CheckDefault(TEXT(R"({"image_processor_type":"NoSuchImageProcessor","size":{"height":32,"width":32}})"), EPipelineImageResample::Bilinear);
		});
		// EfficientNet's default is Nearest (0), the only 0 in the table.
		It("absent resample + EfficientNetImageProcessor is Nearest", [this]()
		{
			CheckDefault(TEXT(R"({"image_processor_type":"EfficientNetImageProcessor","size":{"height":32,"width":32},"do_center_crop":false})"), EPipelineImageResample::Nearest);
		});
		// A second table entry in a different family; CLIP is Bicubic and keeps its shortest-edge+crop mode.
		It("absent resample + CLIPImageProcessor is Bicubic", [this]()
		{
			CheckDefault(TEXT(R"({"image_processor_type":"CLIPImageProcessor","size":{"shortest_edge":32},"do_center_crop":true,"crop_size":{"height":32,"width":32}})"), EPipelineImageResample::Bicubic);
		});
		// Lanczos entry (1).
		It("absent resample + Idefics3ImageProcessor is Lanczos", [this]()
		{
			using namespace PipelineResampleParseSpecPrivate;
			const FString Json = TEXT(R"({"image_processor_type":"Idefics3ImageProcessor","size":{"height":32,"width":32}})");
			FPipelineModelFolderInfo Info;
			FPipelineValidationReport Report;
			FString Ctx;
			if (!Parse(&Json, Info, Report, Ctx)) { return; }
			TestEqual(*(Ctx + TEXT(" Resample")), FString(ResampleName(Info.Preprocess.Resample)), FString(TEXT("Lanczos")));
		});
		// No preprocessor_config.json: Bilinear and only the existing MissingPreprocessorConfig warning.
		It("no preprocessor_config.json is Bilinear with only the MissingPreprocessorConfig warning", [this]()
		{
			using namespace PipelineResampleParseSpecPrivate;
			FPipelineModelFolderInfo Info;
			FPipelineValidationReport Report;
			FString Ctx;
			if (!Parse(nullptr, Info, Report, Ctx)) { return; }
			TestEqual(*(Ctx + TEXT(" Resample")), FString(ResampleName(Info.Preprocess.Resample)), FString(TEXT("Bilinear")));
			TestEqual(*(Ctx + TEXT(" MissingPreprocessorConfig warnings")), CountMsg(Report, ECode::MissingPreprocessorConfig, ESev::Warning), 1);
			TestEqual(*(Ctx + TEXT(" no UnsupportedResize")), CountMsg(Report, ECode::UnsupportedResize, ESev::Warning), 0);
		});
	});

	Describe("LeViT", [this]()
	{
		auto RunLevit = [this](const FString& Json, EPipelineImageResize ExpMode, int32 ExpW, int32 ExpH, int32 ExpShortest, int32 ExpUnsupportedWarnings)
		{
			using namespace PipelineResampleParseSpecPrivate;
			FPipelineModelFolderInfo Info;
			FPipelineValidationReport Report;
			FString Ctx;
			if (!Parse(&Json, Info, Report, Ctx)) { return; }
			TestEqual(*(Ctx + TEXT(" ResizeMode")), FString(ResizeName(Info.Preprocess.ResizeMode)), FString(ResizeName(ExpMode)));
			TestEqual(*(Ctx + TEXT(" ImageWidth")), Info.Preprocess.ImageWidth, ExpW);
			TestEqual(*(Ctx + TEXT(" ImageHeight")), Info.Preprocess.ImageHeight, ExpH);
			TestEqual(*(Ctx + TEXT(" ResizeShortestEdge")), Info.Preprocess.ResizeShortestEdge, ExpShortest);
			TestEqual(*(Ctx + TEXT(" UnsupportedResize warnings")), CountMsg(Report, ECode::UnsupportedResize, ESev::Warning), ExpUnsupportedWarnings);
			TestEqual(*(Ctx + TEXT(" Resample (LeViT default)")), FString(ResampleName(Info.Preprocess.Resample)), FString(TEXT("Bicubic")));
		};

		// HF LeViT resizes to (int)((256/224)*S) then crops: 224 -> 256. Catches a parser that uses S itself (no scaling).
		It("LevitImageProcessor int size 224 + crop 224 is ShortestEdgeCenterCrop 256 -> 224x224", [this, RunLevit]()
		{
			RunLevit(TEXT(R"({"image_processor_type":"LevitImageProcessor","size":224,"do_center_crop":true,"crop_size":{"height":224,"width":224}})"),
				EPipelineImageResize::ShortestEdgeCenterCrop, 224, 224, 256, 0);
		});

		// 40 * 256/224 = 45.71 -> truncation 45 (a rounding parser would give 46).
		It("LevitImageProcessor int size 40 + crop 32 is ShortestEdgeCenterCrop 45 -> 32x32 (truncated, not rounded)", [this, RunLevit]()
		{
			RunLevit(TEXT(R"({"image_processor_type":"LevitImageProcessor","size":40,"do_center_crop":true,"crop_size":{"height":32,"width":32}})"),
				EPipelineImageResize::ShortestEdgeCenterCrop, 32, 32, 45, 0);
		});

		// The {shortest_edge} form goes through the same class rule.
		It("LevitImageProcessor {shortest_edge 224} + crop 224 is ShortestEdgeCenterCrop 256 -> 224x224", [this, RunLevit]()
		{
			RunLevit(TEXT(R"({"image_processor_type":"LevitImageProcessor","size":{"shortest_edge":224},"do_center_crop":true,"crop_size":{"height":224,"width":224}})"),
				EPipelineImageResize::ShortestEdgeCenterCrop, 224, 224, 256, 0);
		});

		It("LevitImageProcessor {shortest_edge 40} + crop 32 is ShortestEdgeCenterCrop 45 -> 32x32", [this, RunLevit]()
		{
			RunLevit(TEXT(R"({"image_processor_type":"LevitImageProcessor","size":{"shortest_edge":40},"do_center_crop":true,"crop_size":{"height":32,"width":32}})"),
				EPipelineImageResize::ShortestEdgeCenterCrop, 32, 32, 45, 0);
		});

		// Other shortest-edge classes (CLIP) must NOT get the 256/224 scaling.
		It("CLIPImageProcessor {shortest_edge 40} + crop 32 keeps ResizeShortestEdge 40 (no LeViT scaling)", [this]()
		{
			using namespace PipelineResampleParseSpecPrivate;
			const FString Json = TEXT(R"({"image_processor_type":"CLIPImageProcessor","size":{"shortest_edge":40},"do_center_crop":true,"crop_size":{"height":32,"width":32}})");
			FPipelineModelFolderInfo Info;
			FPipelineValidationReport Report;
			FString Ctx;
			if (!Parse(&Json, Info, Report, Ctx)) { return; }
			TestEqual(*(Ctx + TEXT(" ResizeShortestEdge")), Info.Preprocess.ResizeShortestEdge, 40);
		});

		// No usable crop: unchanged behaviour = Stretch SxS plus the existing UnsupportedResize warning.
		It("LevitImageProcessor {shortest_edge 224} without crop is Stretch 224x224 with one UnsupportedResize warning", [this, RunLevit]()
		{
			RunLevit(TEXT(R"({"image_processor_type":"LevitImageProcessor","size":{"shortest_edge":224},"do_center_crop":false})"),
				EPipelineImageResize::Stretch, 224, 224, 0, 1);
		});

		It("LevitImageProcessor int 40 without crop is Stretch 40x40 with one UnsupportedResize warning", [this, RunLevit]()
		{
			RunLevit(TEXT(R"({"image_processor_type":"LevitImageProcessor","size":40,"do_center_crop":false})"),
				EPipelineImageResize::Stretch, 40, 40, 0, 1);
		});
	});

	Describe("CropWithoutSize", [this]()
	{
		// DeiT {h,w} + do_center_crop true but no crop_size: Stretch RWxRH + exactly one UnsupportedResize Warning whose
		// text says "crop_size is missing". Catches silent Stretch (no warning) or a hard error.
		It("DeiTImageProcessor {height 32,width 32} + do_center_crop without crop_size is Stretch 32x32 with a warning", [this]()
		{
			using namespace PipelineResampleParseSpecPrivate;
			const FString Json = TEXT(R"({"image_processor_type":"DeiTImageProcessor","size":{"height":32,"width":32},"do_center_crop":true,"resample":2})");
			FPipelineModelFolderInfo Info;
			FPipelineValidationReport Report;
			FString Ctx;
			if (!Parse(&Json, Info, Report, Ctx)) { return; }
			TestEqual(*(Ctx + TEXT(" ResizeMode")), FString(ResizeName(Info.Preprocess.ResizeMode)), FString(TEXT("Stretch")));
			TestEqual(*(Ctx + TEXT(" ImageWidth")), Info.Preprocess.ImageWidth, 32);
			TestEqual(*(Ctx + TEXT(" ImageHeight")), Info.Preprocess.ImageHeight, 32);
			TestEqual(*(Ctx + TEXT(" Resample")), FString(ResampleName(Info.Preprocess.Resample)), FString(TEXT("Bilinear")));
			TestEqual(*(Ctx + TEXT(" exactly one UnsupportedResize warning")), CountMsg(Report, ECode::UnsupportedResize, ESev::Warning), 1);
			TestEqual(*(Ctx + TEXT(" total warnings")), Report.NumWarnings(), 1);
			TestTrue(*(Ctx + TEXT(" message says 'crop_size is missing'")),
				AnyMessageContains(Report, ECode::UnsupportedResize, ESev::Warning, TEXT("crop_size is missing")));
			TestTrue(*(Ctx + TEXT(" message mentions do_center_crop")),
				AnyMessageContains(Report, ECode::UnsupportedResize, ESev::Warning, TEXT("do_center_crop")));
		});

		// Non-square: the message must carry RWxRH (40x48 here is width x height); Stretch keeps the sizes unswapped.
		It("ViTImageProcessor {height 48,width 40} + do_center_crop without crop_size is Stretch 40x48 with a warning", [this]()
		{
			using namespace PipelineResampleParseSpecPrivate;
			const FString Json = TEXT(R"({"image_processor_type":"ViTImageProcessor","size":{"height":48,"width":40},"do_center_crop":true})");
			FPipelineModelFolderInfo Info;
			FPipelineValidationReport Report;
			FString Ctx;
			if (!Parse(&Json, Info, Report, Ctx)) { return; }
			TestEqual(*(Ctx + TEXT(" ResizeMode")), FString(ResizeName(Info.Preprocess.ResizeMode)), FString(TEXT("Stretch")));
			TestEqual(*(Ctx + TEXT(" ImageWidth")), Info.Preprocess.ImageWidth, 40);
			TestEqual(*(Ctx + TEXT(" ImageHeight")), Info.Preprocess.ImageHeight, 48);
			TestEqual(*(Ctx + TEXT(" crop_size warning count")), CountMsg(Report, ECode::UnsupportedResize, ESev::Warning), 1);
			TestTrue(*(Ctx + TEXT(" message says 'crop_size is missing'")),
				AnyMessageContains(Report, ECode::UnsupportedResize, ESev::Warning, TEXT("crop_size is missing")));
		});

		// do_center_crop false (or absent) with {h,w}: no warning (the warning is only for the set-but-missing case).
		It("DeiTImageProcessor {height 32,width 32} with do_center_crop false has no warning", [this]()
		{
			CheckDefault(TEXT(R"({"image_processor_type":"DeiTImageProcessor","size":{"height":32,"width":32},"do_center_crop":false,"resample":3})"), EPipelineImageResample::Bicubic);
		});

		// With a crop_size present the new warning must not fire (StretchCenterCrop path).
		It("DeiTImageProcessor {height 40,width 40} + crop_size 32 has no crop_size warning", [this]()
		{
			using namespace PipelineResampleParseSpecPrivate;
			const FString Json = TEXT(R"({"image_processor_type":"DeiTImageProcessor","size":{"height":40,"width":40},"do_center_crop":true,"crop_size":{"height":32,"width":32}})");
			FPipelineModelFolderInfo Info;
			FPipelineValidationReport Report;
			FString Ctx;
			if (!Parse(&Json, Info, Report, Ctx)) { return; }
			TestEqual(*(Ctx + TEXT(" ResizeMode")), FString(ResizeName(Info.Preprocess.ResizeMode)), FString(TEXT("StretchCenterCrop")));
			TestEqual(*(Ctx + TEXT(" no warnings")), Report.NumWarnings(), 0);
			TestEqual(*(Ctx + TEXT(" Resample (DeiT default)")), FString(ResampleName(Info.Preprocess.Resample)), FString(TEXT("Bicubic")));
		});

		// Rule 4: the shortest-edge path (MobileNetV2 int, no crop) already warns; there must be exactly one warning, not two.
		It("MobileNetV2ImageProcessor int 40 without crop still has exactly one UnsupportedResize warning", [this]()
		{
			using namespace PipelineResampleParseSpecPrivate;
			const FString Json = TEXT(R"({"image_processor_type":"MobileNetV2ImageProcessor","size":40,"do_center_crop":false,"resample":2})");
			FPipelineModelFolderInfo Info;
			FPipelineValidationReport Report;
			FString Ctx;
			if (!Parse(&Json, Info, Report, Ctx)) { return; }
			TestEqual(*(Ctx + TEXT(" UnsupportedResize warnings")), CountMsg(Report, ECode::UnsupportedResize, ESev::Warning), 1);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
