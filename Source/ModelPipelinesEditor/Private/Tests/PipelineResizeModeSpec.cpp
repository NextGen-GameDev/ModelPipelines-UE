// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
// WP4.0 (docs/contracts.md C25): StretchCenterCrop folder parsing.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"
#include "PipelineModelFolder.h"
#include "PipelineModelImporter.h"
#include "PipelineEditorTestData.h"

namespace PipelineResizeModeSpecPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;
	using namespace UE::ModelPipelines::EditorTests;

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

	static bool ResizeFromName(const FString& Name, EPipelineImageResize& Out)
	{
		if (Name == TEXT("Stretch")) { Out = EPipelineImageResize::Stretch; return true; }
		if (Name == TEXT("ShortestEdgeCenterCrop")) { Out = EPipelineImageResize::ShortestEdgeCenterCrop; return true; }
		if (Name == TEXT("StretchCenterCrop")) { Out = EPipelineImageResize::StretchCenterCrop; return true; }
		return false;
	}

	static int32 CountMsg(const FPipelineValidationReport& R, ECode Code, ESev Sev)
	{
		int32 N = 0;
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code && M.Severity == Sev) { ++N; }
		}
		return N;
	}

	static const TCHAR* ImageConfigJson = TEXT(R"({"architectures":["ViTForImageClassification"],"model_type":"vit","id2label":{"0":"a","1":"b","2":"c"}})");

	struct FSynthCase
	{
		const TCHAR* Name;
		FString PreprocessorJson;
		bool bExpectParseOk;
		EPipelineImageResize Mode;
		int32 W, H, RW, RH, ShortestEdge;
		int32 ExpectedUnsupportedResizeWarnings;
	};
}

BEGIN_DEFINE_SPEC(FPipelineResizeModeSpec, "Pipelines.Editor.ModelFolder.Resize",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	TArray<FString> TempRoots;

	/** Writes config.json, a copy of the tiny image model and preprocessor_config.json; returns the folder or "" (error added). */
	FString MakeFolder(const FString& PreprocessorJson)
	{
		FString Root = FPaths::ConvertRelativePathToFull(FPaths::ProjectIntermediateDir() / TEXT("PipelinesResizeTests") / FGuid::NewGuid().ToString());
		TempRoots.Add(Root);
		IFileManager::Get().MakeDirectory(*Root, true);
		const FString Src = UE::ModelPipelines::EditorTests::GetModelsDir() / TEXT("import_test/tiny_image_classification/model.onnx");
		if (IFileManager::Get().Copy(*(Root / TEXT("model.onnx")), *Src) != COPY_OK)
		{
			AddError(FString::Printf(TEXT("cannot copy '%s'; generate it with: python Tools/make_import_test_data.py"), *Src));
			return FString();
		}
		FFileHelper::SaveStringToFile(PipelineResizeModeSpecPrivate::ImageConfigJson, *(Root / TEXT("config.json")), FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
		FFileHelper::SaveStringToFile(PreprocessorJson, *(Root / TEXT("preprocessor_config.json")), FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
		return Root;
	}

	/** Loads the case entry for Id (fails with the path and fetch command when missing). */
	bool LoadCaseAndFolder(const FString& Id, TSharedPtr<FJsonObject>& OutCase, FString& OutFolder)
	{
		using namespace UE::ModelPipelines::EditorTests;
		FString Err;
		TSharedPtr<FJsonObject> Root = LoadImportExpectations(Err);
		if (!Root.IsValid()) { AddError(Err + TEXT(" (generate with: python Tools/make_import_test_data.py)")); return false; }
		OutCase = FindCase(Root, Id);
		if (!OutCase.IsValid())
		{
			AddError(FString::Printf(TEXT("case '%s' missing from <dev>/goldens/import/import_expectations.json; regenerate with: python Tools/make_import_test_data.py"), *Id));
			return false;
		}
		FString Rel;
		OutCase->TryGetStringField(TEXT("folder"), Rel);
		OutFolder = GetModelsDir() / Rel;
		if (!IFileManager::Get().DirectoryExists(*OutFolder))
		{
			AddError(FString::Printf(TEXT("[%s] model folder '%s' missing; generate with: python Tools/make_import_test_data.py"), *Id, *OutFolder));
			return false;
		}
		return true;
	}

	void CheckSynth(const PipelineResizeModeSpecPrivate::FSynthCase& C)
	{
		using namespace PipelineResizeModeSpecPrivate;
		const FString Folder = MakeFolder(C.PreprocessorJson);
		if (Folder.IsEmpty()) { return; }
		FPipelineModelFolderInfo Info;
		FPipelineValidationReport Report;
		const bool bOk = ParsePipelineModelFolder(Folder, EPipelineTask::None, Info, Report);
		const FString Ctx = FString::Printf(TEXT("[%s] report:\n%s"), C.Name, *Report.ToString());
		TestEqual(*(Ctx + TEXT(" parse ok")), bOk, C.bExpectParseOk);
		if (!C.bExpectParseOk)
		{
			TestEqual(*(Ctx + TEXT(" one InvalidJson error")), CountMsg(Report, ECode::InvalidJson, ESev::Error), 1);
			return;
		}
		TestEqual(*(Ctx + TEXT(" ResizeMode")), FString(ResizeName(Info.Preprocess.ResizeMode)), FString(ResizeName(C.Mode)));
		TestEqual(*(Ctx + TEXT(" ImageWidth")), Info.Preprocess.ImageWidth, C.W);
		TestEqual(*(Ctx + TEXT(" ImageHeight")), Info.Preprocess.ImageHeight, C.H);
		TestEqual(*(Ctx + TEXT(" ResizeWidth")), Info.Preprocess.ResizeWidth, C.RW);
		TestEqual(*(Ctx + TEXT(" ResizeHeight")), Info.Preprocess.ResizeHeight, C.RH);
		TestEqual(*(Ctx + TEXT(" ResizeShortestEdge")), Info.Preprocess.ResizeShortestEdge, C.ShortestEdge);
		TestEqual(*(Ctx + TEXT(" UnsupportedResize warnings")), CountMsg(Report, ECode::UnsupportedResize, ESev::Warning), C.ExpectedUnsupportedResizeWarnings);
		TestEqual(*(Ctx + TEXT(" no errors")), Report.NumErrors(), 0);
	}
END_DEFINE_SPEC(FPipelineResizeModeSpec)

void FPipelineResizeModeSpec::Define()
{
	using namespace PipelineResizeModeSpecPrivate;
	using namespace UE::ModelPipelines::EditorTests;

	AfterEach([this]()
	{
		for (const FString& Dir : TempRoots)
		{
			IFileManager::Get().DeleteDirectory(*Dir, false, true);
		}
		TempRoots.Reset();
	});

	Describe("ImportTestData", [this]()
	{
		static const TCHAR* Ids[] = {TEXT("deit_style"), TEXT("deit_hw_crop"), TEXT("deit_legacy_fe"), TEXT("clip_style_crop"), TEXT("square_int_nocrop")};
		for (const TCHAR* IdRaw : Ids)
		{
			const FString Id = IdRaw;

			// Catches a parser that still maps {height,width}+crop to Stretch (model gets the wrong input size), keeps the
			// legacy shortest-edge reading for DeiT integer sizes, or leaves ResizeWidth/Height 0 for StretchCenterCrop.
			It(*FString::Printf(TEXT("parses %s to the Python-derived expectation with no warnings"), *Id), [this, Id]()
			{
				TSharedPtr<FJsonObject> Case;
				FString Folder;
				if (!LoadCaseAndFolder(Id, Case, Folder)) { return; }
				const TSharedPtr<FJsonObject>* Pre = nullptr;
				if (!Case->TryGetObjectField(TEXT("preprocess"), Pre) || !Pre)
				{
					AddError(FString::Printf(TEXT("[%s] case has no 'preprocess' block; regenerate with: python Tools/make_import_test_data.py"), *Id));
					return;
				}
				FString ModeName;
				double RW = -1, RH = -1, SE = -1, W = -1, H = -1;
				const bool bHave = (*Pre)->TryGetStringField(TEXT("resize_mode"), ModeName)
					&& (*Pre)->TryGetNumberField(TEXT("resize_width"), RW) && (*Pre)->TryGetNumberField(TEXT("resize_height"), RH)
					&& (*Pre)->TryGetNumberField(TEXT("resize_shortest_edge"), SE)
					&& (*Pre)->TryGetNumberField(TEXT("image_width"), W) && (*Pre)->TryGetNumberField(TEXT("image_height"), H);
				EPipelineImageResize Mode;
				if (!bHave || !ResizeFromName(ModeName, Mode))
				{
					AddError(FString::Printf(TEXT("[%s] expectation lacks resize_mode/resize_width/resize_height/resize_shortest_edge/image_width/image_height; regenerate with: python Tools/make_import_test_data.py"), *Id));
					return;
				}

				FPipelineModelFolderInfo Info;
				FPipelineValidationReport Report;
				const bool bOk = ParsePipelineModelFolder(Folder, EPipelineTask::None, Info, Report);
				const FString Ctx = FString::Printf(TEXT("[%s] report:\n%s"), *Id, *Report.ToString());
				TestTrue(*(Ctx + TEXT(" parse ok")), bOk);
				TestEqual(*(Ctx + TEXT(" no warnings")), Report.NumWarnings(), 0);
				TestEqual(*(Ctx + TEXT(" no errors")), Report.NumErrors(), 0);
				TestEqual(*(Ctx + TEXT(" ResizeMode")), FString(ResizeName(Info.Preprocess.ResizeMode)), FString(ResizeName(Mode)));
				TestEqual(*(Ctx + TEXT(" ResizeWidth")), Info.Preprocess.ResizeWidth, (int32)RW);
				TestEqual(*(Ctx + TEXT(" ResizeHeight")), Info.Preprocess.ResizeHeight, (int32)RH);
				TestEqual(*(Ctx + TEXT(" ResizeShortestEdge")), Info.Preprocess.ResizeShortestEdge, (int32)SE);
				TestEqual(*(Ctx + TEXT(" ImageWidth")), Info.Preprocess.ImageWidth, (int32)W);
				TestEqual(*(Ctx + TEXT(" ImageHeight")), Info.Preprocess.ImageHeight, (int32)H);
			});

			// The tiny model takes the crop size; a wrong Image size or a bad Resize size would fail validation (smoke run).
			It(*FString::Printf(TEXT("imports %s with validation: 0 errors, 0 warnings"), *Id), [this, Id]()
			{
				TSharedPtr<FJsonObject> Case;
				FString Folder;
				if (!LoadCaseAndFolder(Id, Case, Folder)) { return; }
				FPipelineImportOptions O;
				O.SourceFolder = Folder;
				O.DestinationPath = TEXT("/Game/PipelinesResizeSpec");
				O.AssetName = FString::Printf(TEXT("Resize_%s"), *Id);
				O.bSave = false;
				O.bValidate = true;
				O.Validation.bCheckGpu = false;
				FPipelineImportResult R = ImportPipelineModelFolder(O);
				TestTrue(*FString::Printf(TEXT("[%s] bSuccess; report:\n%s"), *Id, *R.Report.ToString()), R.bSuccess);
				TestEqual(*FString::Printf(TEXT("[%s] errors; report:\n%s"), *Id, *R.Report.ToString()), R.Report.NumErrors(), 0);
				TestEqual(*FString::Printf(TEXT("[%s] warnings; report:\n%s"), *Id, *R.Report.ToString()), R.Report.NumWarnings(), 0);
				DiscardImported(R);
			});
		}
	});

	Describe("Synthetic", [this]()
	{
		// BEiT squares {h,w}; do_center_crop false -> plain Stretch 30x30 (rule 2, no crop). Catches a parser that
		// invents a crop or leaves ResizeWidth/Height set.
		It("BeitImageProcessor {height 30,width 30} without crop is Stretch 30x30", [this]()
		{
			CheckSynth({TEXT("beit"), TEXT(R"({"image_processor_type":"BeitImageProcessor","size":{"height":30,"width":30},"do_center_crop":false})"),
				true, EPipelineImageResize::Stretch, 30, 30, 0, 0, 0, 0});
		});

		// Non-square crop: Image takes the crop (W40,H48); resize is 64x64. Catches swapped width/height.
		It("ViTImageProcessor int 64 + crop {height 48,width 40} is StretchCenterCrop resize 64x64 image 40x48", [this]()
		{
			CheckSynth({TEXT("vit_hw"), TEXT(R"({"image_processor_type":"ViTImageProcessor","size":64,"do_center_crop":true,"crop_size":{"height":48,"width":40}})"),
				true, EPipelineImageResize::StretchCenterCrop, 40, 48, 64, 64, 0, 0});
		});

		// "Fast" suffix must be stripped; DeiT is not a shortest-edge class, so int 50 is square (rule 3).
		It("DeiTImageProcessorFast int 50 + crop 40 is StretchCenterCrop 50x50 -> 40x40", [this]()
		{
			CheckSynth({TEXT("deit_fast"), TEXT(R"({"image_processor_type":"DeiTImageProcessorFast","size":50,"do_center_crop":true,"crop_size":40})"),
				true, EPipelineImageResize::StretchCenterCrop, 40, 40, 50, 50, 0, 0});
		});

		// MobileNetV2 is a shortest-edge class and has no crop: HF resizes by shortest edge, which we cannot represent
		// -> Stretch 50x50 with exactly one UnsupportedResize warning (rule 4).
		It("MobileNetV2ImageProcessor int 50 without crop is Stretch 50x50 with one UnsupportedResize warning", [this]()
		{
			CheckSynth({TEXT("mnv2"), TEXT(R"({"image_processor_type":"MobileNetV2ImageProcessor","size":50,"do_center_crop":false})"),
				true, EPipelineImageResize::Stretch, 50, 50, 0, 0, 0, 1});
		});

		// Empty class keeps the legacy reading (rule 4): shortest edge + crop, no warning.
		It("no processor type, int 50 + crop 40 stays ShortestEdgeCenterCrop 50 -> 40x40", [this]()
		{
			CheckSynth({TEXT("legacy"), TEXT(R"({"size":50,"do_center_crop":true,"crop_size":40})"),
				true, EPipelineImageResize::ShortestEdgeCenterCrop, 40, 40, 0, 0, 50, 0});
		});

		// Malformed crop_size under {h,w}+do_center_crop true: the existing InvalidJson behaviour, not a silent Stretch.
		It("{height 40,width 48} + do_center_crop true + crop_size \"abc\" is InvalidJson", [this]()
		{
			CheckSynth({TEXT("bad_crop"), TEXT(R"({"image_processor_type":"ViTImageProcessor","size":{"height":40,"width":48},"do_center_crop":true,"crop_size":"abc"})"),
				false, EPipelineImageResize::Stretch, 0, 0, 0, 0, 0, 0});
		});

		// Shortest-edge class with a crop keeps ShortestEdgeCenterCrop (CLIP rule), and must not set ResizeWidth/Height.
		It("CLIPImageProcessor int 56 + crop 48 is ShortestEdgeCenterCrop 56 -> 48x48 with Resize 0x0", [this]()
		{
			CheckSynth({TEXT("clip"), TEXT(R"({"image_processor_type":"CLIPImageProcessor","size":56,"do_center_crop":true,"crop_size":48})"),
				true, EPipelineImageResize::ShortestEdgeCenterCrop, 48, 48, 0, 0, 56, 0});
		});

		// Legacy key feature_extractor_type: FeatureExtractor -> ImageProcessor gives DeiTImageProcessor (rule 3).
		It("DeiTFeatureExtractor (legacy key) int 50 + crop 40 is StretchCenterCrop 50x50 -> 40x40", [this]()
		{
			CheckSynth({TEXT("deit_fe"), TEXT(R"({"feature_extractor_type":"DeiTFeatureExtractor","size":50,"do_center_crop":true,"crop_size":40})"),
				true, EPipelineImageResize::StretchCenterCrop, 40, 40, 50, 50, 0, 0});
		});

		// Class names are case-sensitive: a lower-case clip class is not in the shortest-edge table, so it is "unknown
		// non-empty" -> square rule 3 -> StretchCenterCrop.
		It("class names are case-sensitive (clipimageprocessor is not a shortest-edge class)", [this]()
		{
			CheckSynth({TEXT("case"), TEXT(R"({"image_processor_type":"clipimageprocessor","size":56,"do_center_crop":true,"crop_size":48})"),
				true, EPipelineImageResize::StretchCenterCrop, 48, 48, 56, 56, 0, 0});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
