// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Containers/Array.h"
#include "HAL/PlatformProcess.h"
#include "HAL/ThreadSafeCounter.h"
#include "Math/NumericLimits.h"
#include <limits>
#include "Math/UnrealMathUtility.h"
#include "Misc/Guid.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UnrealType.h"
#include "UObject/Class.h"
#include "NNEModelData.h"
#include "PipelineTypes.h"
#include "PipelineValidation.h"
#include "PipelineModelPackage.h"

namespace PipelinePackageSpecPrivate
{
	/** A package that passes ValidateSettings with zero messages for the given task. */
	static UPipelineModelPackage* MakeValidPackage(EPipelineTask Task)
	{
		UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
		Pkg->Task = Task;
		Pkg->Model = NewObject<UNNEModelData>(GetTransientPackage());
		Pkg->License.SpdxId = TEXT("Apache-2.0");
		Pkg->License.SourceUrl = TEXT("https://huggingface.co/x/y/tree/main");
		Pkg->License.Attribution = TEXT("Example Authors");
		if (IsTextTask(Task))
		{
			Pkg->TokenizerJson = TEXT("{}");
		}
		if (TaskUsesLabels(Task))
		{
			Pkg->Labels = {TEXT("a"), TEXT("b")};
		}
		return Pkg;
	}

	static int32 CountCode(const FPipelineValidationReport& R, EPipelineValidationCode Code)
	{
		int32 N = 0;
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code)
			{
				++N;
			}
		}
		return N;
	}

	static const FPipelineValidationMessage* FindCode(const FPipelineValidationReport& R, EPipelineValidationCode Code)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code)
			{
				return &M;
			}
		}
		return nullptr;
	}

	static const EPipelineTask AllTasks[5] = {
		EPipelineTask::None, EPipelineTask::TextEmbedding, EPipelineTask::TextClassification,
		EPipelineTask::ImageClassification, EPipelineTask::ObjectDetection};
}

BEGIN_DEFINE_SPEC(FPipelineModelPackageSpec, "Pipelines.Core.Package",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	/** Runs ValidateSettings on a fresh report; returns the report and the bool via OutOk. */
	FPipelineValidationReport Run(const UPipelineModelPackage* Pkg, bool& OutOk) const
	{
		FPipelineValidationReport R;
		OutOk = Pkg->ValidateSettings(R);
		return R;
	}

	/** Expects exactly one Error with this code and that the overall result is false. */
	void ExpectOnlyError(const UPipelineModelPackage* Pkg, EPipelineValidationCode Code, const TCHAR* What)
	{
		bool bOk = true;
		FPipelineValidationReport R = Run(Pkg, bOk);
		TestFalse(FString::Printf(TEXT("%s: ValidateSettings expected false"), What), bOk);
		TestTrue(FString::Printf(TEXT("%s: expected Error %s; report:\n%s"), What, LexToString(Code), *R.ToString()), R.HasError(Code));
		TestEqual(FString::Printf(TEXT("%s: expected exactly 1 message; report:\n%s"), What, *R.ToString()), R.Messages.Num(), 1);
	}

	/** Expects true and zero messages. */
	void ExpectClean(const UPipelineModelPackage* Pkg, const TCHAR* What)
	{
		bool bOk = false;
		FPipelineValidationReport R = Run(Pkg, bOk);
		TestTrue(FString::Printf(TEXT("%s: ValidateSettings expected true; report:\n%s"), What, *R.ToString()), bOk);
		TestEqual(FString::Printf(TEXT("%s: expected 0 messages; report:\n%s"), What, *R.ToString()), R.Messages.Num(), 0);
	}

	TArray<TStrongObjectPtr<UObject>> Keep;

	UPipelineModelPackage* Make(EPipelineTask Task)
	{
		UPipelineModelPackage* P = PipelinePackageSpecPrivate::MakeValidPackage(Task);
		Keep.Add(TStrongObjectPtr<UObject>(P));
		Keep.Add(TStrongObjectPtr<UObject>(P->Model.Get()));
		return P;
	}

END_DEFINE_SPEC(FPipelineModelPackageSpec)

void FPipelineModelPackageSpec::Define()
{
	using namespace PipelinePackageSpecPrivate;

	AfterEach([this]()
	{
		Keep.Reset();
	});

	Describe("Defaults (acceptance 1)", [this]()
	{
		It("preprocess and postprocess defaults match the contract", [this]()
		{
			const FPipelinePreprocessSettings P;
			TestEqual(TEXT("MaxTokens expected 512"), P.MaxTokens, 512);
			TestTrue(TEXT("bAddSpecialTokens expected true"), P.bAddSpecialTokens);
			TestEqual(TEXT("Pooling expected Mean"), (uint8)P.Pooling, (uint8)EPipelinePooling::Mean);
			TestFalse(TEXT("bNormalizeEmbeddings expected false"), P.bNormalizeEmbeddings);
			TestEqual(TEXT("ImageWidth expected 224"), P.ImageWidth, 224);
			TestEqual(TEXT("ImageHeight expected 224"), P.ImageHeight, 224);
			TestEqual(TEXT("ResizeMode expected Stretch"), (uint8)P.ResizeMode, (uint8)EPipelineImageResize::Stretch);
			TestEqual(TEXT("ResizeShortestEdge expected 0"), P.ResizeShortestEdge, 0);
			// A double-typed 1/255 stored in a float would still be near, but exactness to float 1.f/255.f is the contract.
			TestEqual(TEXT("RescaleFactor expected 1.f/255.f"), P.RescaleFactor, 1.f / 255.f);
			TestEqual(TEXT("ImageMean.X expected 0.485"), P.ImageMean.X, 0.485);
			TestEqual(TEXT("ImageMean.Y expected 0.456"), P.ImageMean.Y, 0.456);
			TestEqual(TEXT("ImageMean.Z expected 0.406"), P.ImageMean.Z, 0.406);
			TestEqual(TEXT("ImageStd.X expected 0.229"), P.ImageStd.X, 0.229);
			TestEqual(TEXT("ImageStd.Y expected 0.224"), P.ImageStd.Y, 0.224);
			TestEqual(TEXT("ImageStd.Z expected 0.225"), P.ImageStd.Z, 0.225);

			const FPipelinePostprocessSettings Post;
			TestEqual(TEXT("ScoreActivation expected Softmax"), (uint8)Post.ScoreActivation, (uint8)EPipelineScoreActivation::Softmax);
		});

		It("a new package is empty with Task None and PreferredDevice Auto", [this]()
		{
			UPipelineModelPackage* P = NewObject<UPipelineModelPackage>(GetTransientPackage());
			Keep.Add(TStrongObjectPtr<UObject>(P));
			TestEqual(TEXT("Task expected None"), (uint8)P->Task, (uint8)EPipelineTask::None);
			TestTrue(TEXT("Model expected null"), P->Model == nullptr);
			TestEqual(TEXT("PreferredDevice expected Auto"), (uint8)P->PreferredDevice, (uint8)EPipelineDevice::Auto);
			TestTrue(TEXT("TokenizerJson expected empty"), P->TokenizerJson.IsEmpty());
			TestEqual(TEXT("Labels expected empty"), P->Labels.Num(), 0);
			TestTrue(TEXT("License.SpdxId expected empty"), P->License.SpdxId.IsEmpty());
			TestTrue(TEXT("License.SourceUrl expected empty"), P->License.SourceUrl.IsEmpty());
			TestTrue(TEXT("License.Attribution expected empty"), P->License.Attribution.IsEmpty());
			TestEqual(TEXT("Postprocess.ScoreActivation expected Softmax"), (uint8)P->Postprocess.ScoreActivation, (uint8)EPipelineScoreActivation::Softmax);
		});
	});

	Describe("Enum values (acceptance 2)", [this]()
	{
		It("validation codes have exactly the contract numbers", [this]()
		{
			struct FPair { EPipelineValidationCode C; uint8 V; };
			const FPair Pairs[] = {
				{EPipelineValidationCode::None, 0}, {EPipelineValidationCode::NoPackage, 1}, {EPipelineValidationCode::NoTask, 2},
				{EPipelineValidationCode::NoModel, 3}, {EPipelineValidationCode::NoTokenizer, 4}, {EPipelineValidationCode::InvalidMaxTokens, 5},
				{EPipelineValidationCode::NoLabels, 6}, {EPipelineValidationCode::EmptyLabel, 7}, {EPipelineValidationCode::InvalidImageSize, 8},
				{EPipelineValidationCode::InvalidNormalization, 9}, {EPipelineValidationCode::InvalidResize, 10},
				{EPipelineValidationCode::LicenseSpdxMissing, 11}, {EPipelineValidationCode::LicenseSourceMissing, 12},
				{EPipelineValidationCode::LicenseAttributionMissing, 13},
				{EPipelineValidationCode::MissingInput, 20}, {EPipelineValidationCode::UnexpectedInput, 21},
				{EPipelineValidationCode::InputTypeMismatch, 22}, {EPipelineValidationCode::InputRankMismatch, 23},
				{EPipelineValidationCode::InputDimMismatch, 24}, {EPipelineValidationCode::MissingOutput, 25},
				{EPipelineValidationCode::OutputTypeMismatch, 26}, {EPipelineValidationCode::OutputRankMismatch, 27},
				{EPipelineValidationCode::OutputDimMismatch, 28}, {EPipelineValidationCode::LabelCountMismatch, 29},
				{EPipelineValidationCode::OutputNameFallback, 30}, {EPipelineValidationCode::FixedBatch, 31},
				{EPipelineValidationCode::FolderNotFound, 40}, {EPipelineValidationCode::MissingModelFile, 41},
				{EPipelineValidationCode::InvalidJson, 42}, {EPipelineValidationCode::MissingConfig, 43},
				{EPipelineValidationCode::MissingTokenizer, 44}, {EPipelineValidationCode::TaskUndetected, 45},
				{EPipelineValidationCode::MissingLabels, 46}, {EPipelineValidationCode::InvalidLabels, 47},
				{EPipelineValidationCode::MissingPreprocessorConfig, 48}, {EPipelineValidationCode::UnsupportedPooling, 49},
				{EPipelineValidationCode::UnsupportedResize, 50}, {EPipelineValidationCode::UnverifiedDetectionFamily, 51},
				{EPipelineValidationCode::TokenizerInvalid, 60}, {EPipelineValidationCode::RuntimeUnavailable, 61},
				{EPipelineValidationCode::ModelCreateFailed, 62}, {EPipelineValidationCode::SmokeRunFailed, 63},
				{EPipelineValidationCode::SmokeOutputInvalid, 64}, {EPipelineValidationCode::InvalidDestination, 65},
				{EPipelineValidationCode::AssetExists, 66}, {EPipelineValidationCode::SaveFailed, 67},
			};
			for (const FPair& P : Pairs)
			{
				TestEqual(FString::Printf(TEXT("code %s expected %d"), LexToString(P.C), (int32)P.V), (uint8)P.C, P.V);
			}
		});

		It("severity, task, pooling, resize, activation and device enums have the contract numbers", [this]()
		{
			TestEqual(TEXT("Info expected 0"), (uint8)EPipelineValidationSeverity::Info, (uint8)0);
			TestEqual(TEXT("Warning expected 1"), (uint8)EPipelineValidationSeverity::Warning, (uint8)1);
			TestEqual(TEXT("Error expected 2"), (uint8)EPipelineValidationSeverity::Error, (uint8)2);
			TestEqual(TEXT("None expected 0"), (uint8)EPipelineTask::None, (uint8)0);
			TestEqual(TEXT("TextEmbedding expected 1"), (uint8)EPipelineTask::TextEmbedding, (uint8)1);
			TestEqual(TEXT("TextClassification expected 2"), (uint8)EPipelineTask::TextClassification, (uint8)2);
			TestEqual(TEXT("ImageClassification expected 3"), (uint8)EPipelineTask::ImageClassification, (uint8)3);
			TestEqual(TEXT("ObjectDetection expected 4"), (uint8)EPipelineTask::ObjectDetection, (uint8)4);
			TestEqual(TEXT("Pooling Mean expected 0"), (uint8)EPipelinePooling::Mean, (uint8)0);
			TestEqual(TEXT("Pooling CLS expected 1"), (uint8)EPipelinePooling::CLS, (uint8)1);
			TestEqual(TEXT("Pooling None expected 2"), (uint8)EPipelinePooling::None, (uint8)2);
			TestEqual(TEXT("Stretch expected 0"), (uint8)EPipelineImageResize::Stretch, (uint8)0);
			TestEqual(TEXT("ShortestEdgeCenterCrop expected 1"), (uint8)EPipelineImageResize::ShortestEdgeCenterCrop, (uint8)1);
			TestEqual(TEXT("Softmax expected 0"), (uint8)EPipelineScoreActivation::Softmax, (uint8)0);
			TestEqual(TEXT("Sigmoid expected 1"), (uint8)EPipelineScoreActivation::Sigmoid, (uint8)1);
		});
	});

	Describe("Reflection (acceptance 3)", [this]()
	{
		It("TokenizerJson is hidden from the details panel; other members are editable; all are Blueprint visible", [this]()
		{
			UClass* Cls = UPipelineModelPackage::StaticClass();
			const TCHAR* Editable[] = {TEXT("Task"), TEXT("Model"), TEXT("Labels"), TEXT("Preprocess"), TEXT("Postprocess"), TEXT("License"), TEXT("PreferredDevice")};
			for (const TCHAR* Name : Editable)
			{
				const FProperty* Prop = Cls->FindPropertyByName(Name);
				if (!TestNotNull(FString::Printf(TEXT("property %s expected to exist"), Name), Prop))
				{
					continue;
				}
				TestTrue(FString::Printf(TEXT("%s expected CPF_Edit"), Name), Prop->HasAnyPropertyFlags(CPF_Edit));
				TestTrue(FString::Printf(TEXT("%s expected CPF_BlueprintVisible"), Name), Prop->HasAnyPropertyFlags(CPF_BlueprintVisible));
			}
			const FProperty* Tok = Cls->FindPropertyByName(TEXT("TokenizerJson"));
			if (TestNotNull(TEXT("property TokenizerJson expected to exist"), Tok))
			{
				// A tokenizer.json is megabytes; showing it in the details panel would freeze the editor.
				TestFalse(TEXT("TokenizerJson must NOT have CPF_Edit"), Tok->HasAnyPropertyFlags(CPF_Edit));
				TestTrue(TEXT("TokenizerJson expected CPF_BlueprintVisible"), Tok->HasAnyPropertyFlags(CPF_BlueprintVisible));
			}
		});
	});

	Describe("DuplicateObject (acceptance 4)", [this]()
	{
		It("copies every field including non-ASCII text", [this]()
		{
			UPipelineModelPackage* Src = Make(EPipelineTask::TextClassification);
			Src->TokenizerJson = TEXT("{\"x\":\"日本語 café\"}");
			Src->Labels = {TEXT("neg"), TEXT("café"), TEXT("日本")};
			Src->Preprocess.MaxTokens = 77;
			Src->Preprocess.bAddSpecialTokens = false;
			Src->Preprocess.Pooling = EPipelinePooling::CLS;
			Src->Preprocess.bNormalizeEmbeddings = true;
			Src->Preprocess.ImageWidth = 300;
			Src->Preprocess.ImageHeight = 200;
			Src->Preprocess.ResizeMode = EPipelineImageResize::ShortestEdgeCenterCrop;
			Src->Preprocess.ResizeShortestEdge = 321;
			Src->Preprocess.RescaleFactor = 0.5f;
			Src->Preprocess.ImageMean = FVector(0.1, 0.2, 0.3);
			Src->Preprocess.ImageStd = FVector(0.4, 0.5, 0.6);
			Src->Postprocess.ScoreActivation = EPipelineScoreActivation::Sigmoid;
			Src->License.SpdxId = TEXT("MIT");
			Src->License.SourceUrl = TEXT("https://example.com/é");
			Src->License.Attribution = TEXT("Zoë");
			Src->PreferredDevice = EPipelineDevice::GPU;

			UPipelineModelPackage* Dup = DuplicateObject<UPipelineModelPackage>(Src, GetTransientPackage());
			if (!TestNotNull(TEXT("DuplicateObject expected non-null"), Dup)) { return; }
			Keep.Add(TStrongObjectPtr<UObject>(Dup));

			TestTrue(TEXT("duplicate expected to be a different object"), Dup != Src);
			TestEqual(TEXT("Task"), (uint8)Dup->Task, (uint8)Src->Task);
			TestTrue(TEXT("Model pointer copied"), Dup->Model == Src->Model);
			TestEqual(TEXT("TokenizerJson (non-ASCII)"), Dup->TokenizerJson, Src->TokenizerJson);
			TestEqual(TEXT("Labels count"), Dup->Labels.Num(), 3);
			if (Dup->Labels.Num() == 3)
			{
				TestEqual(TEXT("Labels[1] expected café"), Dup->Labels[1], FString(TEXT("café")));
				TestEqual(TEXT("Labels[2] expected 日本"), Dup->Labels[2], FString(TEXT("日本")));
			}
			TestEqual(TEXT("MaxTokens"), Dup->Preprocess.MaxTokens, 77);
			TestFalse(TEXT("bAddSpecialTokens"), Dup->Preprocess.bAddSpecialTokens);
			TestEqual(TEXT("Pooling"), (uint8)Dup->Preprocess.Pooling, (uint8)EPipelinePooling::CLS);
			TestTrue(TEXT("bNormalizeEmbeddings"), Dup->Preprocess.bNormalizeEmbeddings);
			TestEqual(TEXT("ImageWidth"), Dup->Preprocess.ImageWidth, 300);
			TestEqual(TEXT("ImageHeight"), Dup->Preprocess.ImageHeight, 200);
			TestEqual(TEXT("ResizeMode"), (uint8)Dup->Preprocess.ResizeMode, (uint8)EPipelineImageResize::ShortestEdgeCenterCrop);
			TestEqual(TEXT("ResizeShortestEdge"), Dup->Preprocess.ResizeShortestEdge, 321);
			TestEqual(TEXT("RescaleFactor"), Dup->Preprocess.RescaleFactor, 0.5f);
			TestTrue(TEXT("ImageMean"), Dup->Preprocess.ImageMean == FVector(0.1, 0.2, 0.3));
			TestTrue(TEXT("ImageStd"), Dup->Preprocess.ImageStd == FVector(0.4, 0.5, 0.6));
			TestEqual(TEXT("ScoreActivation"), (uint8)Dup->Postprocess.ScoreActivation, (uint8)EPipelineScoreActivation::Sigmoid);
			TestEqual(TEXT("License.SpdxId"), Dup->License.SpdxId, FString(TEXT("MIT")));
			TestEqual(TEXT("License.SourceUrl"), Dup->License.SourceUrl, Src->License.SourceUrl);
			TestEqual(TEXT("License.Attribution"), Dup->License.Attribution, FString(TEXT("Zoë")));
			TestEqual(TEXT("PreferredDevice"), (uint8)Dup->PreferredDevice, (uint8)EPipelineDevice::GPU);
		});
	});

	Describe("Lex helpers (acceptance 5-7)", [this]()
	{
		It("LexToString of validation codes returns the enumerator name; every enumerator round-trips", [this]()
		{
			TestEqual(TEXT("InvalidResize"), FString(LexToString(EPipelineValidationCode::InvalidResize)), FString(TEXT("InvalidResize")));
			TestEqual(TEXT("None"), FString(LexToString(EPipelineValidationCode::None)), FString(TEXT("None")));
			TestEqual(TEXT("SaveFailed"), FString(LexToString(EPipelineValidationCode::SaveFailed)), FString(TEXT("SaveFailed")));

			const UEnum* Enum = StaticEnum<EPipelineValidationCode>();
			if (!TestNotNull(TEXT("StaticEnum<EPipelineValidationCode> expected"), Enum)) { return; }
			int32 Checked = 0;
			for (int32 i = 0; i < Enum->NumEnums(); ++i)
			{
				const FString Name = Enum->GetNameStringByIndex(i);
				if (Name.EndsWith(TEXT("_MAX"))) { continue; }
				const EPipelineValidationCode C = (EPipelineValidationCode)Enum->GetValueByIndex(i);
				TestEqual(FString::Printf(TEXT("round trip of value %d"), (int32)Enum->GetValueByIndex(i)), FString(LexToString(C)), Name);
				++Checked;
			}
			TestEqual(TEXT("number of enumerators checked expected 58 (46 + InvalidPostprocess, C26, + 8 GGUF codes, C60, + 3 speech codes, C72)"), Checked, 58);
		});

		It("LexToString of an out-of-range code returns Unknown without asserting", [this]()
		{
			TestEqual(TEXT("255 expected Unknown"), FString(LexToString((EPipelineValidationCode)255)), FString(TEXT("Unknown")));
			// 35 sits in a gap of the contract numbering (31 -> 40). 74 is past the last code (73, C72; 19 and 72 were gaps until C72).
			TestEqual(TEXT("35 (gap) expected Unknown"), FString(LexToString((EPipelineValidationCode)35)), FString(TEXT("Unknown")));
			TestEqual(TEXT("74 expected Unknown"), FString(LexToString((EPipelineValidationCode)74)), FString(TEXT("Unknown")));
		});

		It("LexToString of severities and tasks", [this]()
		{
			TestEqual(TEXT("Info"), FString(LexToString(EPipelineValidationSeverity::Info)), FString(TEXT("Info")));
			TestEqual(TEXT("Warning"), FString(LexToString(EPipelineValidationSeverity::Warning)), FString(TEXT("Warning")));
			TestEqual(TEXT("Error"), FString(LexToString(EPipelineValidationSeverity::Error)), FString(TEXT("Error")));
			TestEqual(TEXT("TextClassification"), FString(LexToString(EPipelineTask::TextClassification)), FString(TEXT("TextClassification")));
			TestEqual(TEXT("None"), FString(LexToString(EPipelineTask::None)), FString(TEXT("None")));
			TestEqual(TEXT("TextEmbedding"), FString(LexToString(EPipelineTask::TextEmbedding)), FString(TEXT("TextEmbedding")));
			TestEqual(TEXT("ImageClassification"), FString(LexToString(EPipelineTask::ImageClassification)), FString(TEXT("ImageClassification")));
			TestEqual(TEXT("ObjectDetection"), FString(LexToString(EPipelineTask::ObjectDetection)), FString(TEXT("ObjectDetection")));
		});

		It("LexTryParseString matches case-insensitively and leaves Out unchanged on failure", [this]()
		{
			EPipelineTask Out = EPipelineTask::ImageClassification;
			TestTrue(TEXT("textembedding expected true"), LexTryParseString(Out, TEXT("textembedding")));
			TestEqual(TEXT("textembedding -> TextEmbedding"), (uint8)Out, (uint8)EPipelineTask::TextEmbedding);
			TestTrue(TEXT("OBJECTDETECTION expected true"), LexTryParseString(Out, TEXT("OBJECTDETECTION")));
			TestEqual(TEXT("OBJECTDETECTION -> ObjectDetection"), (uint8)Out, (uint8)EPipelineTask::ObjectDetection);
			TestTrue(TEXT("None expected true"), LexTryParseString(Out, TEXT("None")));
			TestEqual(TEXT("None -> None"), (uint8)Out, (uint8)EPipelineTask::None);
			TestTrue(TEXT("TextClassification expected true"), LexTryParseString(Out, TEXT("TextClassification")));
			TestEqual(TEXT("-> TextClassification"), (uint8)Out, (uint8)EPipelineTask::TextClassification);
			TestTrue(TEXT("imageCLASSIFICATION expected true"), LexTryParseString(Out, TEXT("imageCLASSIFICATION")));
			TestEqual(TEXT("-> ImageClassification"), (uint8)Out, (uint8)EPipelineTask::ImageClassification);

			const TCHAR* Bad[] = {TEXT(""), TEXT("Text Embedding"), TEXT("Detection"), TEXT(" None"), TEXT("None "), TEXT("7")};
			for (const TCHAR* S : Bad)
			{
				EPipelineTask Sentinel = EPipelineTask::ObjectDetection;
				TestFalse(FString::Printf(TEXT("'%s' expected false"), S), LexTryParseString(Sentinel, S));
				TestEqual(FString::Printf(TEXT("'%s' must leave Out unchanged"), S), (uint8)Sentinel, (uint8)EPipelineTask::ObjectDetection);
			}
			EPipelineTask Sentinel = EPipelineTask::TextClassification;
			TestFalse(TEXT("nullptr expected false"), LexTryParseString(Sentinel, nullptr));
			TestEqual(TEXT("nullptr must leave Out unchanged"), (uint8)Sentinel, (uint8)EPipelineTask::TextClassification);
		});

		It("IsTextTask / IsImageTask / TaskUsesLabels truth table", [this]()
		{
			// Index order: None, TextEmbedding, TextClassification, ImageClassification, ObjectDetection
			const bool Text[5] = {false, true, true, false, false};
			const bool Image[5] = {false, false, false, true, true};
			const bool Labels[5] = {false, false, true, true, true};
			for (int32 i = 0; i < 5; ++i)
			{
				const FString N = LexToString(AllTasks[i]);
				TestEqual(FString::Printf(TEXT("IsTextTask(%s)"), *N), IsTextTask(AllTasks[i]), Text[i]);
				TestEqual(FString::Printf(TEXT("IsImageTask(%s)"), *N), IsImageTask(AllTasks[i]), Image[i]);
				TestEqual(FString::Printf(TEXT("TaskUsesLabels(%s)"), *N), TaskUsesLabels(AllTasks[i]), Labels[i]);
			}
		});
	});

	Describe("Report helpers (acceptance 8-9)", [this]()
	{
		It("empty report", [this]()
		{
			FPipelineValidationReport R;
			TestTrue(TEXT("IsValid expected true"), R.IsValid());
			TestEqual(TEXT("NumErrors expected 0"), R.NumErrors(), 0);
			TestEqual(TEXT("NumWarnings expected 0"), R.NumWarnings(), 0);
			TestEqual(TEXT("ToString expected empty"), R.ToString(), FString());
			TestFalse(TEXT("HasCode expected false"), R.HasCode(EPipelineValidationCode::NoTask));
			TestFalse(TEXT("HasError expected false"), R.HasError(EPipelineValidationCode::NoTask));
		});

		It("counts, HasCode vs HasError and ToString format", [this]()
		{
			FPipelineValidationReport R;
			R.Add(EPipelineValidationSeverity::Warning, EPipelineValidationCode::LicenseSpdxMissing, TEXT("a"));
			R.Add(EPipelineValidationSeverity::Error, EPipelineValidationCode::MissingInput, TEXT("b"));
			TestFalse(TEXT("IsValid expected false"), R.IsValid());
			TestEqual(TEXT("NumErrors expected 1"), R.NumErrors(), 1);
			TestEqual(TEXT("NumWarnings expected 1"), R.NumWarnings(), 1);
			TestTrue(TEXT("HasCode(LicenseSpdxMissing) expected true"), R.HasCode(EPipelineValidationCode::LicenseSpdxMissing));
			// A warning must not satisfy HasError.
			TestFalse(TEXT("HasError(LicenseSpdxMissing) expected false"), R.HasError(EPipelineValidationCode::LicenseSpdxMissing));
			TestTrue(TEXT("HasError(MissingInput) expected true"), R.HasError(EPipelineValidationCode::MissingInput));
			TestEqual(TEXT("ToString"), R.ToString(), FString(TEXT("Warning LicenseSpdxMissing: a\nError MissingInput: b")));
		});

		It("Info messages count as neither error nor warning and keep the report valid", [this]()
		{
			FPipelineValidationReport R;
			R.Add(EPipelineValidationSeverity::Info, EPipelineValidationCode::FixedBatch, TEXT("batch 1"));
			TestTrue(TEXT("IsValid expected true"), R.IsValid());
			TestEqual(TEXT("NumErrors expected 0"), R.NumErrors(), 0);
			TestEqual(TEXT("NumWarnings expected 0"), R.NumWarnings(), 0);
			TestTrue(TEXT("HasCode(FixedBatch) expected true"), R.HasCode(EPipelineValidationCode::FixedBatch));
			TestFalse(TEXT("HasError(FixedBatch) expected false"), R.HasError(EPipelineValidationCode::FixedBatch));
			TestEqual(TEXT("ToString"), R.ToString(), FString(TEXT("Info FixedBatch: batch 1")));
		});

		It("Append concatenates messages and device checks in order", [this]()
		{
			FPipelineValidationReport A, B;
			A.Add(EPipelineValidationSeverity::Error, EPipelineValidationCode::NoTask, TEXT("a1"));
			FPipelineDeviceCheck DA; DA.Device = EPipelineDevice::CPU; DA.Message = TEXT("da");
			A.DeviceChecks.Add(DA);
			B.Add(EPipelineValidationSeverity::Warning, EPipelineValidationCode::EmptyLabel, TEXT("b1"));
			B.Add(EPipelineValidationSeverity::Info, EPipelineValidationCode::FixedBatch, TEXT("b2"));
			FPipelineDeviceCheck DB; DB.Device = EPipelineDevice::GPU; DB.Message = TEXT("db");
			B.DeviceChecks.Add(DB);

			A.Append(B);
			TestEqual(TEXT("Messages expected 3"), A.Messages.Num(), 3);
			TestEqual(TEXT("DeviceChecks expected 2"), A.DeviceChecks.Num(), 2);
			if (A.Messages.Num() == 3)
			{
				TestEqual(TEXT("msg 0"), A.Messages[0].Message, FString(TEXT("a1")));
				TestEqual(TEXT("msg 1"), A.Messages[1].Message, FString(TEXT("b1")));
				TestEqual(TEXT("msg 2"), A.Messages[2].Message, FString(TEXT("b2")));
			}
			if (A.DeviceChecks.Num() == 2)
			{
				TestEqual(TEXT("devcheck 0"), A.DeviceChecks[0].Message, FString(TEXT("da")));
				TestEqual(TEXT("devcheck 1"), A.DeviceChecks[1].Message, FString(TEXT("db")));
				TestEqual(TEXT("devcheck 1 device GPU"), (uint8)A.DeviceChecks[1].Device, (uint8)EPipelineDevice::GPU);
			}
			TestEqual(TEXT("source report unchanged"), B.Messages.Num(), 2);
		});

		It("Append of an empty report changes nothing", [this]()
		{
			FPipelineValidationReport A, Empty;
			A.Add(EPipelineValidationSeverity::Error, EPipelineValidationCode::NoModel, TEXT("m"));
			const FString Before = A.ToString();
			A.Append(Empty);
			TestEqual(TEXT("ToString unchanged"), A.ToString(), Before);
			TestEqual(TEXT("DeviceChecks unchanged"), A.DeviceChecks.Num(), 0);
			Empty.Append(A);
			TestEqual(TEXT("empty.Append(A) copies A"), Empty.ToString(), Before);
		});

		It("a report appended to itself doubles without looping or crashing", [this]()
		{
			FPipelineValidationReport R;
			R.Add(EPipelineValidationSeverity::Error, EPipelineValidationCode::NoModel, TEXT("m"));
			R.Add(EPipelineValidationSeverity::Warning, EPipelineValidationCode::EmptyLabel, TEXT("w"));
			R.DeviceChecks.AddDefaulted();
			R.Append(R);
			TestEqual(TEXT("Messages expected 4"), R.Messages.Num(), 4);
			TestEqual(TEXT("DeviceChecks expected 2"), R.DeviceChecks.Num(), 2);
			TestEqual(TEXT("NumErrors expected 2"), R.NumErrors(), 2);
			TestEqual(TEXT("NumWarnings expected 2"), R.NumWarnings(), 2);
			if (R.Messages.Num() == 4)
			{
				TestEqual(TEXT("msg 2 repeats msg 0"), R.Messages[2].Message, FString(TEXT("m")));
				TestEqual(TEXT("msg 3 repeats msg 1"), R.Messages[3].Message, FString(TEXT("w")));
			}
		});
	});

	Describe("ValidateSettings (acceptance 10-18)", [this]()
	{
		It("Task None yields exactly one NoTask error even with no model and no license", [this]()
		{
			UPipelineModelPackage* P = NewObject<UPipelineModelPackage>(GetTransientPackage());
			Keep.Add(TStrongObjectPtr<UObject>(P));
			ExpectOnlyError(P, EPipelineValidationCode::NoTask, TEXT("None task"));
		});

		It("TextEmbedding with model, tokenizer and full license is clean; other tasks' fields are ignored", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextEmbedding);
			ExpectClean(P, TEXT("valid TextEmbedding"));

			// Image fields, labels and activation are irrelevant for text.
			P->Preprocess.ImageWidth = 0;
			P->Preprocess.ImageHeight = -4;
			P->Preprocess.RescaleFactor = -1.f;
			P->Preprocess.ImageStd = FVector(0, 0, 0);
			P->Preprocess.ResizeMode = EPipelineImageResize::ShortestEdgeCenterCrop;
			P->Preprocess.ResizeShortestEdge = 1;
			P->Labels = {TEXT("")};
			P->Postprocess.ScoreActivation = EPipelineScoreActivation::Sigmoid;
			ExpectClean(P, TEXT("TextEmbedding with garbage image fields and Labels {\"\"}"));
		});

		It("TextEmbedding: empty tokenizer is NoTokenizer", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextEmbedding);
			P->TokenizerJson.Empty();
			ExpectOnlyError(P, EPipelineValidationCode::NoTokenizer, TEXT("empty tokenizer"));
		});

		It("TextEmbedding: MaxTokens outside 1..65536 is InvalidMaxTokens, the bounds are fine", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextEmbedding);
			const int32 Bad[] = {0, -5, 65537, MIN_int32, MAX_int32};
			for (int32 V : Bad)
			{
				P->Preprocess.MaxTokens = V;
				ExpectOnlyError(P, EPipelineValidationCode::InvalidMaxTokens, *FString::Printf(TEXT("MaxTokens %d"), V));
			}
			const int32 Good[] = {1, 65536, 512};
			for (int32 V : Good)
			{
				P->Preprocess.MaxTokens = V;
				ExpectClean(P, *FString::Printf(TEXT("MaxTokens %d"), V));
			}
		});

		It("TextClassification: empty Labels is NoLabels", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextClassification);
			ExpectClean(P, TEXT("valid TextClassification"));
			P->Labels.Empty();
			ExpectOnlyError(P, EPipelineValidationCode::NoLabels, TEXT("no labels"));
		});

		It("TextClassification: several empty labels give one EmptyLabel warning naming index 1", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextClassification);
			P->Labels = {TEXT("neg"), TEXT(""), TEXT("pos"), TEXT("")};
			bool bOk = false;
			FPipelineValidationReport R = Run(P, bOk);
			TestTrue(TEXT("warnings do not fail: expected true"), bOk);
			TestEqual(TEXT("expected exactly 1 message"), R.Messages.Num(), 1);
			TestEqual(TEXT("expected exactly 1 EmptyLabel"), CountCode(R, EPipelineValidationCode::EmptyLabel), 1);
			const FPipelineValidationMessage* M = FindCode(R, EPipelineValidationCode::EmptyLabel);
			if (M)
			{
				TestEqual(TEXT("severity expected Warning"), (uint8)M->Severity, (uint8)EPipelineValidationSeverity::Warning);
				TestTrue(TEXT("message expected to name index 1: ") + M->Message, M->Message.Contains(TEXT("1")));
				// The second empty label (index 3) must not be the one reported.
				TestFalse(TEXT("message must not name index 3: ") + M->Message, M->Message.Contains(TEXT("3")));
			}
		});

		It("Whitespace-only label counts as empty for EmptyLabel", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ImageClassification);
			P->Labels = {TEXT("a"), TEXT("b"), TEXT("c"), TEXT("   ")};
			bool bOk = false;
			FPipelineValidationReport R = Run(P, bOk);
			TestTrue(TEXT("expected true"), bOk);
			TestEqual(TEXT("expected one EmptyLabel"), CountCode(R, EPipelineValidationCode::EmptyLabel), 1);
		});

		It("ImageClassification and ObjectDetection with defaults are clean and need no tokenizer", [this]()
		{
			for (EPipelineTask T : {EPipelineTask::ImageClassification, EPipelineTask::ObjectDetection})
			{
				UPipelineModelPackage* P = Make(T);
				P->TokenizerJson.Empty();
				ExpectClean(P, LexToString(T));
				// Text-only fields are ignored for image tasks.
				P->Preprocess.MaxTokens = 0;
				ExpectClean(P, *FString::Printf(TEXT("%s with MaxTokens 0"), LexToString(T)));
			}
		});

		It("Image tasks: labels required", [this]()
		{
			for (EPipelineTask T : {EPipelineTask::ImageClassification, EPipelineTask::ObjectDetection})
			{
				UPipelineModelPackage* P = Make(T);
				P->Labels.Empty();
				ExpectOnlyError(P, EPipelineValidationCode::NoLabels, LexToString(T));
			}
		});

		It("Image tasks: ImageWidth/ImageHeight outside 1..8192 is InvalidImageSize", [this]()
		{
			for (EPipelineTask T : {EPipelineTask::ImageClassification, EPipelineTask::ObjectDetection})
			{
				UPipelineModelPackage* P = Make(T);
				P->Preprocess.ImageWidth = 0;
				ExpectOnlyError(P, EPipelineValidationCode::InvalidImageSize, TEXT("width 0"));
				P->Preprocess.ImageWidth = 224;
				P->Preprocess.ImageHeight = 8193;
				ExpectOnlyError(P, EPipelineValidationCode::InvalidImageSize, TEXT("height 8193"));
				P->Preprocess.ImageHeight = -1;
				ExpectOnlyError(P, EPipelineValidationCode::InvalidImageSize, TEXT("height -1"));
				P->Preprocess.ImageWidth = 8192;
				P->Preprocess.ImageHeight = 8192;
				ExpectClean(P, TEXT("8192 x 8192"));
				P->Preprocess.ImageWidth = 1;
				P->Preprocess.ImageHeight = 1;
				ExpectClean(P, TEXT("1 x 1"));
			}
		});

		It("Image tasks: RescaleFactor <= 0 or any ImageStd component <= 0 is InvalidNormalization", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ImageClassification);
			P->Preprocess.RescaleFactor = 0.f;
			ExpectOnlyError(P, EPipelineValidationCode::InvalidNormalization, TEXT("rescale 0"));
			P->Preprocess.RescaleFactor = -1.f;
			ExpectOnlyError(P, EPipelineValidationCode::InvalidNormalization, TEXT("rescale -1"));
			P->Preprocess.RescaleFactor = 1.f / 255.f;
			ExpectClean(P, TEXT("rescale restored"));
			// A check that only looked at X (or only at the first zero) would miss these.
			P->Preprocess.ImageStd = FVector(0.2, 0.0, 0.3);
			ExpectOnlyError(P, EPipelineValidationCode::InvalidNormalization, TEXT("std Y 0"));
			P->Preprocess.ImageStd = FVector(0.2, 0.3, -0.1);
			ExpectOnlyError(P, EPipelineValidationCode::InvalidNormalization, TEXT("std Z negative"));
			P->Preprocess.ImageStd = FVector(0.0, 0.3, 0.3);
			ExpectOnlyError(P, EPipelineValidationCode::InvalidNormalization, TEXT("std X 0"));
		});

		It("ShortestEdgeCenterCrop requires ResizeShortestEdge >= max(width, height); Stretch ignores it", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			P->Preprocess.ResizeMode = EPipelineImageResize::ShortestEdgeCenterCrop;
			P->Preprocess.ImageWidth = 224;
			P->Preprocess.ImageHeight = 224;
			P->Preprocess.ResizeShortestEdge = 200;
			ExpectOnlyError(P, EPipelineValidationCode::InvalidResize, TEXT("200 < 224"));
			P->Preprocess.ResizeShortestEdge = 0;
			ExpectOnlyError(P, EPipelineValidationCode::InvalidResize, TEXT("0 (unset) with crop"));
			P->Preprocess.ResizeShortestEdge = 224;
			ExpectClean(P, TEXT("224 == 224"));
			P->Preprocess.ResizeShortestEdge = 256;
			ExpectClean(P, TEXT("256 > 224"));

			// The max of width and height is what counts, not just one of them.
			P->Preprocess.ImageWidth = 300;
			P->Preprocess.ImageHeight = 200;
			P->Preprocess.ResizeShortestEdge = 256;
			ExpectOnlyError(P, EPipelineValidationCode::InvalidResize, TEXT("256 < 300"));
			P->Preprocess.ResizeShortestEdge = 300;
			ExpectClean(P, TEXT("300 == max(300,200)"));
			P->Preprocess.ImageWidth = 200;
			P->Preprocess.ImageHeight = 300;
			P->Preprocess.ResizeShortestEdge = 256;
			ExpectOnlyError(P, EPipelineValidationCode::InvalidResize, TEXT("256 < 300 (height larger)"));

			P->Preprocess.ResizeMode = EPipelineImageResize::Stretch;
			P->Preprocess.ResizeShortestEdge = 1;
			ExpectClean(P, TEXT("Stretch ignores ResizeShortestEdge"));
		});

		It("Null model is a NoModel error and the other checks still run", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextEmbedding);
			P->Model = nullptr;
			ExpectOnlyError(P, EPipelineValidationCode::NoModel, TEXT("null model only"));

			P->TokenizerJson.Empty();
			bool bOk = true;
			FPipelineValidationReport R = Run(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(TEXT("NoModel expected"), R.HasError(EPipelineValidationCode::NoModel));
			TestTrue(TEXT("NoTokenizer expected too (checks must not stop at NoModel)"), R.HasError(EPipelineValidationCode::NoTokenizer));
			TestEqual(TEXT("NumErrors expected 2"), R.NumErrors(), 2);
		});

		It("License: all empty gives three warnings and still returns true", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextEmbedding);
			P->License = FPipelineModelLicense();
			bool bOk = false;
			FPipelineValidationReport R = Run(P, bOk);
			TestTrue(TEXT("warnings only: expected true"), bOk);
			TestEqual(TEXT("expected 3 messages"), R.Messages.Num(), 3);
			TestEqual(TEXT("NumWarnings expected 3"), R.NumWarnings(), 3);
			TestEqual(TEXT("NumErrors expected 0"), R.NumErrors(), 0);
			TestEqual(TEXT("one LicenseSpdxMissing"), CountCode(R, EPipelineValidationCode::LicenseSpdxMissing), 1);
			TestEqual(TEXT("one LicenseSourceMissing"), CountCode(R, EPipelineValidationCode::LicenseSourceMissing), 1);
			TestEqual(TEXT("one LicenseAttributionMissing"), CountCode(R, EPipelineValidationCode::LicenseAttributionMissing), 1);
		});

		It("License: each field is judged independently and blanks count as empty", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextEmbedding);
			P->License.SpdxId = TEXT("   ");
			bool bOk = false;
			FPipelineValidationReport R = Run(P, bOk);
			TestTrue(TEXT("expected true"), bOk);
			TestEqual(TEXT("expected 1 message"), R.Messages.Num(), 1);
			TestTrue(TEXT("blank SpdxId expected LicenseSpdxMissing"), R.HasCode(EPipelineValidationCode::LicenseSpdxMissing));

			P->License.SpdxId = TEXT("MIT");
			P->License.SourceUrl = TEXT("\t \n");
			R = Run(P, bOk);
			TestEqual(TEXT("expected 1 message"), R.Messages.Num(), 1);
			TestTrue(TEXT("blank SourceUrl expected LicenseSourceMissing"), R.HasCode(EPipelineValidationCode::LicenseSourceMissing));

			P->License.SourceUrl = TEXT("https://x");
			P->License.Attribution = TEXT(" ");
			R = Run(P, bOk);
			TestEqual(TEXT("expected 1 message"), R.Messages.Num(), 1);
			TestTrue(TEXT("blank Attribution expected LicenseAttributionMissing"), R.HasCode(EPipelineValidationCode::LicenseAttributionMissing));
		});

		It("Appends to an existing report, keeps the old message first, returns true, and does not modify the package", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextClassification);
			P->Labels = {TEXT("x"), TEXT("")};   // yields one warning
			P->License.SpdxId = TEXT("  ");        // and another
			const FString TokBefore = P->TokenizerJson;
			const TArray<FString> LabelsBefore = P->Labels;
			const FString SpdxBefore = P->License.SpdxId;
			const int32 MaxBefore = P->Preprocess.MaxTokens;
			const EPipelineTask TaskBefore = P->Task;
			const UNNEModelData* ModelBefore = P->Model.Get();

			FPipelineValidationReport R;
			R.Add(EPipelineValidationSeverity::Error, EPipelineValidationCode::FolderNotFound, TEXT("old"));
			const bool bOk = P->ValidateSettings(R);
			TestTrue(TEXT("this call added no Error: expected true even though the report holds one"), bOk);
			TestEqual(TEXT("expected old + 2 new messages"), R.Messages.Num(), 3);
			if (R.Messages.Num() >= 1)
			{
				TestEqual(TEXT("old message first"), R.Messages[0].Message, FString(TEXT("old")));
				TestEqual(TEXT("old message code"), (uint8)R.Messages[0].Code, (uint8)EPipelineValidationCode::FolderNotFound);
			}
			TestFalse(TEXT("report overall still invalid"), R.IsValid());

			TestEqual(TEXT("Task unchanged"), (uint8)P->Task, (uint8)TaskBefore);
			TestTrue(TEXT("Model unchanged"), P->Model.Get() == ModelBefore);
			TestEqual(TEXT("TokenizerJson unchanged"), P->TokenizerJson, TokBefore);
			TestTrue(TEXT("Labels unchanged"), P->Labels == LabelsBefore);
			TestEqual(TEXT("License.SpdxId unchanged"), P->License.SpdxId, SpdxBefore);
			TestEqual(TEXT("MaxTokens unchanged"), P->Preprocess.MaxTokens, MaxBefore);
		});

		It("A call that adds an error returns false even if the report already had errors; it does not clear", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextEmbedding);
			P->TokenizerJson.Empty();
			FPipelineValidationReport R;
			R.Add(EPipelineValidationSeverity::Error, EPipelineValidationCode::FolderNotFound, TEXT("old"));
			TestFalse(TEXT("expected false"), P->ValidateSettings(R));
			TestEqual(TEXT("expected old + NoTokenizer"), R.Messages.Num(), 2);
			TestFalse(TEXT("second call: still false"), P->ValidateSettings(R));
			TestEqual(TEXT("two calls append: 3 messages"), R.Messages.Num(), 3);
		});

		It("8 threads x 1000 calls on one const package give identical results", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextClassification);
			// Produces an error (no tokenizer) plus warnings (empty label, empty license).
			P->TokenizerJson.Empty();
			P->Labels = {TEXT("a"), TEXT(""), TEXT("c")};
			P->License = FPipelineModelLicense();
			FString Expected;
			{
				FPipelineValidationReport R;
				const bool bOk = P->ValidateSettings(R);
				TestFalse(TEXT("baseline expected false"), bOk);
				Expected = R.ToString();
				TestTrue(TEXT("baseline expected non-empty"), !Expected.IsEmpty());
			}

			const UPipelineModelPackage* Const = P;
			FThreadSafeCounter Mismatches;
			FThreadSafeCounter Done;
			TArray<TFuture<void>> Futures;
			for (int32 T = 0; T < 8; ++T)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [Const, &Expected, &Mismatches, &Done]()
				{
					for (int32 i = 0; i < 1000; ++i)
					{
						FPipelineValidationReport R;
						const bool bOk = Const->ValidateSettings(R);
						if (bOk || R.ToString() != Expected)
						{
							Mismatches.Increment();
						}
					}
					Done.Increment();
				}));
			}
			const double Deadline = FPlatformTime::Seconds() + 60.0;
			while (Done.GetValue() < 8 && FPlatformTime::Seconds() < Deadline)
			{
				FPlatformProcess::Sleep(0.01f);
			}
			TestEqual(TEXT("all 8 threads expected to finish within 60 s"), Done.GetValue(), 8);
			for (TFuture<void>& F : Futures) { F.Wait(); }   // always wait: the lambdas reference locals
			TestEqual(TEXT("expected 0 mismatching results across 8000 calls"), Mismatches.GetValue(), 0);
		});
	});

	Describe("Edges", [this]()
	{
		It("NaN in ImageStd or RescaleFactor is InvalidNormalization (NaN is not > 0)", [this]()
		{
			const float NaN = std::numeric_limits<float>::quiet_NaN();
			UPipelineModelPackage* P = Make(EPipelineTask::ImageClassification);
			P->Preprocess.RescaleFactor = NaN;
			ExpectOnlyError(P, EPipelineValidationCode::InvalidNormalization, TEXT("NaN rescale"));
			P->Preprocess.RescaleFactor = 1.f / 255.f;
			P->Preprocess.ImageStd = FVector(0.2, (double)NaN, 0.3);
			ExpectOnlyError(P, EPipelineValidationCode::InvalidNormalization, TEXT("NaN std Y"));
			P->Preprocess.ImageStd = FVector((double)NaN, 0.2, 0.3);
			ExpectOnlyError(P, EPipelineValidationCode::InvalidNormalization, TEXT("NaN std X"));
		});

		It("100k labels validate fine; an empty last label is reported by index", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ImageClassification);
			P->Labels.Init(FString(TEXT("label")), 100000);
			ExpectClean(P, TEXT("100k labels"));
			P->Labels.Last() = FString();
			bool bOk = false;
			FPipelineValidationReport R = Run(P, bOk);
			TestTrue(TEXT("expected true"), bOk);
			TestEqual(TEXT("expected one message"), R.Messages.Num(), 1);
			if (R.Messages.Num() == 1)
			{
				TestTrue(TEXT("message expected to name 99999: ") + R.Messages[0].Message, R.Messages[0].Message.Contains(TEXT("99999")));
			}
		});

		It("Multiple problems in an image task are all reported", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ImageClassification);
			P->Labels.Empty();
			P->Preprocess.ImageWidth = 0;
			P->Preprocess.RescaleFactor = 0.f;
			P->Model = nullptr;
			bool bOk = true;
			FPipelineValidationReport R = Run(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(TEXT("NoModel"), R.HasError(EPipelineValidationCode::NoModel));
			TestTrue(TEXT("NoLabels"), R.HasError(EPipelineValidationCode::NoLabels));
			TestTrue(TEXT("InvalidImageSize"), R.HasError(EPipelineValidationCode::InvalidImageSize));
			TestTrue(TEXT("InvalidNormalization"), R.HasError(EPipelineValidationCode::InvalidNormalization));
			TestEqual(TEXT("NumErrors expected 4"), R.NumErrors(), 4);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
