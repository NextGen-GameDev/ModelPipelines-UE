// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/FileManager.h"
#include "Misc/Guid.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineValidation.h"
#include "PipelineModelValidator.h"
#include "PipelineModelImporter.h"
#include "PipelineEditorTestData.h"

namespace PipelineWorstCaseSpecPrivate
{
	using namespace UE::ModelPipelines::EditorTests;

	static const TCHAR* GWorstCasePrefix64 = TEXT("CPU: worst-case input failed (MaxTokens 64 tokens, token id 30521)");

	static FString CaseFolder(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Case)
	{
		FString ModelsRoot, ImportRoot, Folder;
		Root->TryGetStringField(TEXT("models_root"), ModelsRoot);
		Root->TryGetStringField(TEXT("import_root"), ImportRoot);
		Case->TryGetStringField(TEXT("folder"), Folder);
		const FString& Base = Folder.StartsWith(TEXT("reference/")) ? ModelsRoot : ImportRoot;
		return Base / Folder;
	}

	static int32 CountErrors(const FPipelineValidationReport& R, EPipelineValidationCode Code)
	{
		int32 N = 0;
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Severity == EPipelineValidationSeverity::Error && M.Code == Code)
			{
				++N;
			}
		}
		return N;
	}

	static const FPipelineValidationMessage* FindError(const FPipelineValidationReport& R, EPipelineValidationCode Code)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Severity == EPipelineValidationSeverity::Error && M.Code == Code)
			{
				return &M;
			}
		}
		return nullptr;
	}
}

BEGIN_DEFINE_SPEC(FPipelineWorstCaseSpec, "Pipelines.Editor.Validator.WorstCase",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	FString Dest;
	TArray<FPipelineImportResult> Imported;

	bool LoadCaseFolder(const FString& Id, FString& OutFolder);
	FPipelineImportResult Import(const FString& Folder, const FString& Name, bool bRunSmoke);
	void CheckWorstCaseFailure(const FString& Id, int32 ExpectedMaxTokens);
END_DEFINE_SPEC(FPipelineWorstCaseSpec)

bool FPipelineWorstCaseSpec::LoadCaseFolder(const FString& Id, FString& OutFolder)
{
	using namespace PipelineWorstCaseSpecPrivate;
	FString Err;
	TSharedPtr<FJsonObject> Root = LoadImportExpectations(Err);
	if (!Root.IsValid())
	{
		AddError(Err);
		return false;
	}
	TSharedPtr<FJsonObject> Case = FindCase(Root, Id);
	if (!Case.IsValid())
	{
		AddError(FString::Printf(TEXT("case '%s' missing from the import expectations; regenerate with: python Tools/make_import_test_data.py"), *Id));
		return false;
	}
	OutFolder = CaseFolder(Root, Case);
	if (!IFileManager::Get().DirectoryExists(*OutFolder))
	{
		AddError(FString::Printf(TEXT("[%s] test model folder '%s' missing; generate with: python Tools/make_import_test_data.py"), *Id, *OutFolder));
		return false;
	}
	return true;
}

FPipelineImportResult FPipelineWorstCaseSpec::Import(const FString& Folder, const FString& Name, bool bRunSmoke)
{
	FPipelineImportOptions O;
	O.SourceFolder = Folder;
	O.DestinationPath = Dest;
	O.AssetName = Name;
	O.bSave = false;
	O.bValidate = true;
	O.Validation.bCheckGpu = false;
	O.Validation.bRunSmokeTest = bRunSmoke;
	FPipelineImportResult R = ImportPipelineModelFolder(O);
	Imported.Add(R);
	return R;
}

void FPipelineWorstCaseSpec::CheckWorstCaseFailure(const FString& Id, int32 ExpectedMaxTokens)
{
	using namespace PipelineWorstCaseSpecPrivate;
	// ORT logs its own error when it rejects the worst-case input; those lines are expected.
	// Anchored so the pattern can never swallow this test's own assertion messages (review WP3.6c).
	AddExpectedError(TEXT("^Non-zero status code returned while running"), EAutomationExpectedErrorFlags::Contains, -1);
	UE::ModelPipelines::EditorTests::ExpectImportFinding(*this, TEXT("SmokeRunFailed"), true, 1);
	UE::ModelPipelines::EditorTests::ExpectImportFailed(*this, 1);

	FString Folder;
	if (!LoadCaseFolder(Id, Folder)) { return; }
	const FString Name = FString::Printf(TEXT("WC_%s"), *Id);
	const FPipelineImportResult R = Import(Folder, Name, true);

	// Catches a validator that only runs the tiny normal smoke input (ids 0..few) and so imports a model
	// that ORT would crash on at runtime with a large id / long sequence.
	TestFalse(*FString::Printf(TEXT("[%s] import must fail; report:\n%s"), *Id, *R.Report.ToString()), R.bSuccess);
	TestEqual(*FString::Printf(TEXT("[%s] exactly one SmokeRunFailed error; report:\n%s"), *Id, *R.Report.ToString()),
		CountErrors(R.Report, EPipelineValidationCode::SmokeRunFailed), 1);
	TestEqual(*FString::Printf(TEXT("[%s] exactly one error in total (nothing else is wrong with this model); report:\n%s"), *Id, *R.Report.ToString()),
		R.Report.NumErrors(), 1);

	const FPipelineValidationMessage* M = FindError(R.Report, EPipelineValidationCode::SmokeRunFailed);
	if (M)
	{
		const FString Expected = ExpectedMaxTokens == 64 ? FString(GWorstCasePrefix64)
			: FString::Printf(TEXT("CPU: worst-case input failed (MaxTokens %d tokens, token id 30521)"), ExpectedMaxTokens);
		TestTrue(*FString::Printf(TEXT("[%s] message starts with '%s'; got: %s"), *Id, *Expected, *M->Message), M->Message.StartsWith(Expected));
		TestTrue(*FString::Printf(TEXT("[%s] message advises lowering MaxTokens / matching tokenizer; got: %s"), *Id, *M->Message),
			M->Message.Contains(TEXT("lower MaxTokens or use the matching tokenizer")));
	}

	// Failed import leaves nothing behind (importer rule).
	TestTrue(*FString::Printf(TEXT("[%s] Package is null"), *Id), R.Package == nullptr);
	TestTrue(*FString::Printf(TEXT("[%s] Model is null"), *Id), R.Model == nullptr);
	for (const FString& N : {Name, Name + TEXT("_Model")})
	{
		const FString Pkg = Dest / N;
		TestTrue(*FString::Printf(TEXT("[%s] package '%s' does not exist"), *Id, *Pkg), FindPackage(nullptr, *Pkg) == nullptr);
	}
}

void FPipelineWorstCaseSpec::Define()
{
	using namespace PipelineWorstCaseSpecPrivate;

	BeforeEach([this]()
	{
		Dest = TEXT("/Game/PipelinesTests/") + FGuid::NewGuid().ToString(EGuidFormats::Digits);
	});

	AfterEach([this]()
	{
		for (const FPipelineImportResult& R : Imported)
		{
			DiscardImported(R);
		}
		Imported.Reset();
	});

	Describe("ModelsThatCannotTakeTheWorstCase", [this]()
	{
		// small_vocab: embedding table (a few rows) smaller than the tokenizer's 30522-entry vocab. The normal smoke run
		// uses small ids and passes; only the all-(V-1) input trips the Gather bounds check. Expected id 30521.
		It("small_vocab fails with SmokeRunFailed for MaxTokens 64, token id 30521", [this]()
		{
			CheckWorstCaseFailure(TEXT("small_vocab"), 64);
		});

		// short_positions: position table shorter than MaxTokens (64). A validator that used fewer tokens than
		// MaxTokens (e.g. the normal 8-token smoke input) would let this model through.
		It("short_positions fails with SmokeRunFailed for MaxTokens 64, token id 30521", [this]()
		{
			CheckWorstCaseFailure(TEXT("short_positions"), 64);
		});

		// Direct validator call is not available without a package, so also check the report is stable when the
		// same folder is imported twice (the failure must not depend on leftover state from the first run).
		It("a second import of small_vocab fails the same way", [this]()
		{
			// Anchored so the pattern can never swallow this test's own assertion messages (review WP3.6c).
			AddExpectedError(TEXT("^Non-zero status code returned while running"), EAutomationExpectedErrorFlags::Contains, -1);
			UE::ModelPipelines::EditorTests::ExpectImportFinding(*this, TEXT("SmokeRunFailed"), true, 2);
			UE::ModelPipelines::EditorTests::ExpectImportFailed(*this, 2);
			FString Folder;
			if (!LoadCaseFolder(TEXT("small_vocab"), Folder)) { return; }
			const FPipelineImportResult A = Import(Folder, TEXT("WC_Twice"), true);
			const FPipelineImportResult B = Import(Folder, TEXT("WC_Twice"), true);
			TestFalse(TEXT("first import fails"), A.bSuccess);
			TestFalse(*FString::Printf(TEXT("second import fails (a cached success would be a bug); report:\n%s"), *B.Report.ToString()), B.bSuccess);
			TestEqual(TEXT("first: one SmokeRunFailed"), CountErrors(A.Report, EPipelineValidationCode::SmokeRunFailed), 1);
			TestEqual(TEXT("second: one SmokeRunFailed"), CountErrors(B.Report, EPipelineValidationCode::SmokeRunFailed), 1);
		});
	});

	Describe("ModelsThatTakeTheWorstCase", [this]()
	{
		// Catches false positives: a validator that always failed the worst case, or used id V instead of V-1
		// (one past the table) or more than MaxTokens tokens, would reject these correctly-sized models.
		for (const TCHAR* Id : {TEXT("tiny_text_embedding"), TEXT("tiny_text_classification")})
		{
			It(FString::Printf(TEXT("%s still imports with 0 errors and 0 warnings"), Id), [this, Id]()
			{
				FString Folder;
				if (!LoadCaseFolder(Id, Folder)) { return; }
				const FPipelineImportResult R = Import(Folder, FString::Printf(TEXT("Ok_%s"), Id), true);
				TestTrue(*FString::Printf(TEXT("[%s] import succeeds; report:\n%s"), Id, *R.Report.ToString()), R.bSuccess);
				TestEqual(*FString::Printf(TEXT("[%s] 0 errors; report:\n%s"), Id, *R.Report.ToString()), R.Report.NumErrors(), 0);
				TestEqual(*FString::Printf(TEXT("[%s] 0 warnings; report:\n%s"), Id, *R.Report.ToString()), R.Report.NumWarnings(), 0);
				TestEqual(*FString::Printf(TEXT("[%s] no SmokeRunFailed"), Id), CountErrors(R.Report, EPipelineValidationCode::SmokeRunFailed), 0);
			});
		}

		// The CPU device check reports the normal smoke run: it is still created and ran.
		It("tiny_text_embedding CPU device check shows created and ran", [this]()
		{
			FString Folder;
			if (!LoadCaseFolder(TEXT("tiny_text_embedding"), Folder)) { return; }
			const FPipelineImportResult R = Import(Folder, TEXT("Ok_DevCheck"), true);
			const FPipelineDeviceCheck* Cpu = nullptr;
			for (const FPipelineDeviceCheck& C : R.Report.DeviceChecks)
			{
				if (C.Device == EPipelineDevice::CPU) { Cpu = &C; }
			}
			if (!Cpu)
			{
				AddError(FString::Printf(TEXT("no CPU DeviceCheck in report:\n%s"), *R.Report.ToString()));
				return;
			}
			TestTrue(TEXT("CPU bCreated"), Cpu->bCreated);
			TestTrue(*FString::Printf(TEXT("CPU bRan (message '%s')"), *Cpu->Message), Cpu->bRan);
		});
	});

	Describe("SkippedRuns", [this]()
	{
		// Catches a worst-case run executed regardless of Options.bRunSmokeTest. With the smoke test off, small_vocab
		// has no inference at all, so it must not report SmokeRunFailed.
		It("bRunSmokeTest=false does not run the worst case (small_vocab has no SmokeRunFailed)", [this]()
		{
			FString Folder;
			if (!LoadCaseFolder(TEXT("small_vocab"), Folder)) { return; }
			const FPipelineImportResult R = Import(Folder, TEXT("NoSmoke_small_vocab"), false);
			TestEqual(*FString::Printf(TEXT("no SmokeRunFailed without the smoke test; report:\n%s"), *R.Report.ToString()),
				CountErrors(R.Report, EPipelineValidationCode::SmokeRunFailed), 0);
			TestEqual(*FString::Printf(TEXT("no errors at all without the smoke test; report:\n%s"), *R.Report.ToString()),
				R.Report.NumErrors(), 0);
		});

		It("bRunSmokeTest=false does not run the worst case (short_positions has no SmokeRunFailed)", [this]()
		{
			FString Folder;
			if (!LoadCaseFolder(TEXT("short_positions"), Folder)) { return; }
			const FPipelineImportResult R = Import(Folder, TEXT("NoSmoke_short_positions"), false);
			TestEqual(*FString::Printf(TEXT("no SmokeRunFailed without the smoke test; report:\n%s"), *R.Report.ToString()),
				CountErrors(R.Report, EPipelineValidationCode::SmokeRunFailed), 0);
		});

		// Image tasks never get a worst-case run (no tokenizer, no vocab): the tiny image model must import clean.
		It("image task (tiny_image_classification) imports with 0 errors, no worst-case run", [this]()
		{
			FString Folder;
			if (!LoadCaseFolder(TEXT("tiny_image_classification"), Folder)) { return; }
			const FPipelineImportResult R = Import(Folder, TEXT("Img_Ok"), true);
			TestTrue(*FString::Printf(TEXT("import succeeds; report:\n%s"), *R.Report.ToString()), R.bSuccess);
			TestEqual(*FString::Printf(TEXT("0 errors; report:\n%s"), *R.Report.ToString()), R.Report.NumErrors(), 0);
			for (const FPipelineValidationMessage& M : R.Report.Messages)
			{
				TestFalse(*FString::Printf(TEXT("no worst-case message for an image task: %s"), *M.Message), M.Message.Contains(TEXT("worst-case")));
			}
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
