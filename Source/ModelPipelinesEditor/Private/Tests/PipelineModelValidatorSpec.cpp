// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "Dom/JsonObject.h"
#include "Math/Vector.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "NNEModelData.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineNNEBackend.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"
#include "PipelineModelFolder.h"
#include "PipelineModelValidator.h"
#include "PipelineEditorTestData.h"

namespace PipelineModelValidatorSpecPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;
	using namespace UE::ModelPipelines::EditorTests;

	/** A package built from a folder; the strong pointers keep both UObjects alive for the whole test. */
	struct FTestPackage
	{
		TStrongObjectPtr<UPipelineModelPackage> Package;
		TStrongObjectPtr<UNNEModelData> ModelData;
		FString Error;   // non-empty: the package could not be built (the test fails with this)
		bool IsOk() const { return Error.IsEmpty() && Package.IsValid(); }
	};

	/** Brief preamble: ParsePipelineModelFolder -> model bytes -> UNNEModelData -> UPipelineModelPackage. */
	static FTestPackage BuildPackage(const FString& Folder, EPipelineTask Override = EPipelineTask::None)
	{
		FTestPackage Out;
		if (!FPaths::DirectoryExists(Folder))
		{
			Out.Error = FString::Printf(TEXT("Missing test model folder '%s'. Generate it with: python Tools/make_import_test_data.py (import_test/) or Tools/make_reference_models.py (reference/) into <dev>/models/."), *Folder);
			return Out;
		}
		FPipelineModelFolderInfo Info;
		FPipelineValidationReport ParseReport;
		if (!ParsePipelineModelFolder(Folder, Override, Info, ParseReport))
		{
			Out.Error = FString::Printf(TEXT("ParsePipelineModelFolder('%s') failed: %s"), *Folder, *ParseReport.ToString());
			return Out;
		}
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *Info.ModelFile) || Bytes.Num() == 0)
		{
			Out.Error = FString::Printf(TEXT("Cannot read model file '%s'."), *Info.ModelFile);
			return Out;
		}
		Out.ModelData.Reset(NewObject<UNNEModelData>(GetTransientPackage()));
		Out.ModelData->Init(TEXT("onnx"), Bytes);

		Out.Package.Reset(NewObject<UPipelineModelPackage>(GetTransientPackage()));
		UPipelineModelPackage* P = Out.Package.Get();
		P->Task = Info.Task;
		P->Model = Out.ModelData.Get();
		P->TokenizerJson = Info.TokenizerJson;
		P->Labels = Info.Labels;
		P->Preprocess = Info.Preprocess;
		P->Postprocess = Info.Postprocess;
		P->License = Info.License;
		P->PreferredDevice = Info.PreferredDevice;
		return Out;
	}

	static FString ResolveCaseFolder(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Case)
	{
		FString ModelsRoot, ImportRoot, Folder;
		Root->TryGetStringField(TEXT("models_root"), ModelsRoot);
		Root->TryGetStringField(TEXT("import_root"), ImportRoot);
		Case->TryGetStringField(TEXT("folder"), Folder);
		return (Folder.StartsWith(TEXT("reference/")) ? ModelsRoot : ImportRoot) / Folder;
	}

	/** Builds the package of an import_test case from import_expectations.json (honours task_override). */
	static FTestPackage BuildCase(const FString& CaseId)
	{
		FTestPackage Out;
		FString Err;
		TSharedPtr<FJsonObject> Root = LoadImportExpectations(Err);
		if (!Root.IsValid())
		{
			Out.Error = Err;
			return Out;
		}
		TSharedPtr<FJsonObject> Case = FindCase(Root, CaseId);
		if (!Case.IsValid())
		{
			Out.Error = FString::Printf(TEXT("Case '%s' not found in import_expectations.json; regenerate with Tools/make_import_test_data.py."), *CaseId);
			return Out;
		}
		EPipelineTask Override = EPipelineTask::None;
		FString Name;
		if (Case->TryGetStringField(TEXT("task_override"), Name))
		{
			LexTryParseString(Override, *Name);
		}
		return BuildPackage(ResolveCaseFolder(Root, Case), Override);
	}

	static FTestPackage BuildReference(const FString& Slug, EPipelineTask Override = EPipelineTask::None)
	{
		return BuildPackage(GetModelsDir() / TEXT("reference") / Slug, Override);
	}

	static const FPipelineValidationMessage* FindMsg(const FPipelineValidationReport& R, ECode Code, ESev Sev)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code && M.Severity == Sev)
			{
				return &M;
			}
		}
		return nullptr;
	}

	static int32 CountMsg(const FPipelineValidationReport& R, ECode Code, ESev Sev)
	{
		int32 N = 0;
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			N += (M.Code == Code && M.Severity == Sev) ? 1 : 0;
		}
		return N;
	}

	static const FPipelineDeviceCheck* FindCheck(const FPipelineValidationReport& R, EPipelineDevice D)
	{
		for (const FPipelineDeviceCheck& C : R.DeviceChecks)
		{
			if (C.Device == D)
			{
				return &C;
			}
		}
		return nullptr;
	}

	static int32 CountChecks(const FPipelineValidationReport& R, EPipelineDevice D)
	{
		int32 N = 0;
		for (const FPipelineDeviceCheck& C : R.DeviceChecks)
		{
			N += (C.Device == D) ? 1 : 0;
		}
		return N;
	}

	/** True when any message has a code from the later steps (>= 20: task I/O, device, smoke). */
	static bool HasLaterStepCode(const FPipelineValidationReport& R)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (static_cast<uint8>(M.Code) >= 20)
			{
				return true;
			}
		}
		return false;
	}

	/** Everything in a package that the validator must not change (acceptance 10). */
	struct FSnapshot
	{
		EPipelineTask Task;
		const UNNEModelData* Model;
		FString TokenizerJson;
		TArray<FString> Labels;
		FPipelinePreprocessSettings Pre;
		FPipelinePostprocessSettings Post;
		FPipelineModelLicense License;
		EPipelineDevice Device;

		explicit FSnapshot(const UPipelineModelPackage& P)
			: Task(P.Task), Model(P.Model), TokenizerJson(P.TokenizerJson), Labels(P.Labels), Pre(P.Preprocess)
			, Post(P.Postprocess), License(P.License), Device(P.PreferredDevice)
		{
		}

		/** Empty string when equal, else the first differing field. */
		FString Diff(const UPipelineModelPackage& P) const
		{
			if (P.Task != Task) return TEXT("Task");
			if (P.Model.Get() != Model) return TEXT("Model");
			if (P.TokenizerJson != TokenizerJson) return TEXT("TokenizerJson");
			if (P.Labels != Labels) return TEXT("Labels");
			const FPipelinePreprocessSettings& A = P.Preprocess;
			if (A.MaxTokens != Pre.MaxTokens) return TEXT("Preprocess.MaxTokens");
			if (A.bAddSpecialTokens != Pre.bAddSpecialTokens) return TEXT("Preprocess.bAddSpecialTokens");
			if (A.Pooling != Pre.Pooling) return TEXT("Preprocess.Pooling");
			if (A.bNormalizeEmbeddings != Pre.bNormalizeEmbeddings) return TEXT("Preprocess.bNormalizeEmbeddings");
			if (A.ImageWidth != Pre.ImageWidth) return TEXT("Preprocess.ImageWidth");
			if (A.ImageHeight != Pre.ImageHeight) return TEXT("Preprocess.ImageHeight");
			if (A.ResizeMode != Pre.ResizeMode) return TEXT("Preprocess.ResizeMode");
			if (A.ResizeShortestEdge != Pre.ResizeShortestEdge) return TEXT("Preprocess.ResizeShortestEdge");
			if (A.RescaleFactor != Pre.RescaleFactor) return TEXT("Preprocess.RescaleFactor");
			if (A.ImageMean != Pre.ImageMean) return TEXT("Preprocess.ImageMean");
			if (A.ImageStd != Pre.ImageStd) return TEXT("Preprocess.ImageStd");
			if (P.Postprocess.ScoreActivation != Post.ScoreActivation) return TEXT("Postprocess.ScoreActivation");
			if (P.License.SpdxId != License.SpdxId) return TEXT("License.SpdxId");
			if (P.License.SourceUrl != License.SourceUrl) return TEXT("License.SourceUrl");
			if (P.License.Attribution != License.Attribution) return TEXT("License.Attribution");
			if (P.PreferredDevice != Device) return TEXT("PreferredDevice");
			return FString();
		}
	};

	static const TCHAR* const TinyCases[] = {
		TEXT("tiny_text_embedding"), TEXT("tiny_text_classification"), TEXT("tiny_image_classification"), TEXT("tiny_object_detection") };

	static const TCHAR* const ReferenceSlugs[] = {
		TEXT("all-MiniLM-L6-v2"), TEXT("distilbert-sst2"), TEXT("resnet-50"), TEXT("rtdetr_r18vd") };
}

BEGIN_DEFINE_SPEC(FPipelineModelValidatorSpec, "Pipelines.Editor.Validator",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	/** Asserts the GPU part per the C11 GPU rule. Returns true when GPU assertions should run. */
	bool GpuGate(const FString& What)
	{
		using namespace UE::ModelPipelines::EditorTests;
		if (FModelPipelinesCoreModule::Get().GetDefaultBackend().IsDeviceAvailable(EPipelineDevice::GPU))
		{
			return true;
		}
		if (IsGpuRequired())
		{
			AddError(FString::Printf(TEXT("%s: no GPU runtime available but MODELPIPELINES_REQUIRE_GPU=1 (DirectML runtime NNERuntimeORTDml not registered)."), *What));
		}
		else
		{
			AddInfo(FString::Printf(TEXT("%s: GPU assertions SKIPPED, no GPU runtime available (set MODELPIPELINES_REQUIRE_GPU=1 to make this a failure)."), *What));
		}
		return false;
	}
END_DEFINE_SPEC(FPipelineModelValidatorSpec)

void FPipelineModelValidatorSpec::Define()
{
	using namespace PipelineModelValidatorSpecPrivate;

	// ---- acceptance 1: the four tiny packages, CPU part ----
	Describe("Tiny", [this]()
	{
		for (const TCHAR* const CaseId : TinyCases)
		{
			const FString Id(CaseId);
			It(*FString::Printf(TEXT("%s validates with 0 errors and 0 warnings; CPU entry created and ran"), *Id), [this, Id]()
			{
				FTestPackage T = BuildCase(Id);
				if (!T.IsOk()) { AddError(T.Error); return; }
				const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
				// A validator that only checked settings would leave DeviceChecks empty; one that treats Info as
				// a warning would break "0 warnings" on a machine without a GPU.
				TestTrue(*FString::Printf(TEXT("IsValid; report:\n%s"), *R.ToString()), R.IsValid());
				TestEqual(TEXT("NumErrors"), R.NumErrors(), 0);
				TestEqual(*FString::Printf(TEXT("NumWarnings; report:\n%s"), *R.ToString()), R.NumWarnings(), 0);
				TestEqual(TEXT("exactly one CPU entry"), CountChecks(R, EPipelineDevice::CPU), 1);
				if (const FPipelineDeviceCheck* C = FindCheck(R, EPipelineDevice::CPU))
				{
					TestTrue(TEXT("CPU bAvailable"), C->bAvailable);
					TestTrue(TEXT("CPU bCreated"), C->bCreated);
					TestTrue(*FString::Printf(TEXT("CPU bRan (%s)"), *C->Message), C->bRan);
					TestTrue(*FString::Printf(TEXT("CPU InferenceMs >= 0, got %f"), C->InferenceMs), C->InferenceMs >= 0.f);
				}
			});
		}
	});

	// ---- acceptance 1 (GPU) and 11: GPU entries, gated per C11 ----
	Describe("GPU", [this]()
	{
		for (const TCHAR* const CaseId : TinyCases)
		{
			const FString Id(CaseId);
			It(*FString::Printf(TEXT("%s GPU entry available, created and ran"), *Id), [this, Id]()
			{
				if (!GpuGate(Id)) { return; }
				FTestPackage T = BuildCase(Id);
				if (!T.IsOk()) { AddError(T.Error); return; }
				const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
				TestEqual(*FString::Printf(TEXT("NumWarnings (a GPU that silently fell back to CPU warns); report:\n%s"), *R.ToString()), R.NumWarnings(), 0);
				TestEqual(TEXT("exactly one GPU entry"), CountChecks(R, EPipelineDevice::GPU), 1);
				if (const FPipelineDeviceCheck* C = FindCheck(R, EPipelineDevice::GPU))
				{
					TestTrue(TEXT("GPU bAvailable"), C->bAvailable);
					TestTrue(TEXT("GPU bCreated (instance on GPU, not the CPU fallback)"), C->bCreated);
					TestTrue(*FString::Printf(TEXT("GPU bRan (%s)"), *C->Message), C->bRan);
					TestTrue(*FString::Printf(TEXT("GPU InferenceMs >= 0, got %f"), C->InferenceMs), C->InferenceMs >= 0.f);
				}
			});
		}
	});

	// ---- acceptance 2: wrong I/O ----
	Describe("WrongIO", [this]()
	{
		It("wrong_io_for_task reports MissingInput x2, UnexpectedInput, no smoke run, no GPU entry", [this]()
		{
			FTestPackage T = BuildCase(TEXT("wrong_io_for_task"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestFalse(TEXT("not valid"), R.IsValid());
			TestEqual(*FString::Printf(TEXT("MissingInput errors; report:\n%s"), *R.ToString()), CountMsg(R, ECode::MissingInput, ESev::Error), 2);
			bool bIds = false, bMask = false, bShapeText = false;
			for (const FPipelineValidationMessage& M : R.Messages)
			{
				if (M.Code != ECode::MissingInput) { continue; }
				bIds |= M.Message.Contains(TEXT("input_ids"));
				bMask |= M.Message.Contains(TEXT("attention_mask"));
				// The message lists what the model does have; "x float32 [-1,4]" is the model's actual input.
				bShapeText |= M.Message.Contains(TEXT("x float32 [-1,4]"));
			}
			TestTrue(TEXT("a MissingInput names input_ids"), bIds);
			TestTrue(TEXT("a MissingInput names attention_mask"), bMask);
			TestTrue(TEXT("a MissingInput message contains 'x float32 [-1,4]'"), bShapeText);
			const FPipelineValidationMessage* U = FindMsg(R, ECode::UnexpectedInput, ESev::Error);
			TestNotNull(TEXT("UnexpectedInput error"), U);
			if (U) { TestTrue(TEXT("UnexpectedInput names x"), U->Message.Contains(TEXT("x"))); }
			TestEqual(TEXT("exactly one CPU entry"), CountChecks(R, EPipelineDevice::CPU), 1);
			if (const FPipelineDeviceCheck* C = FindCheck(R, EPipelineDevice::CPU))
			{
				TestTrue(TEXT("CPU bCreated (the model itself is fine)"), C->bCreated);
				TestFalse(TEXT("CPU bRan (an I/O error skips the smoke run)"), C->bRan);
			}
			TestEqual(TEXT("no GPU entry (I/O error skips the GPU step)"), CountChecks(R, EPipelineDevice::GPU), 0);
			TestFalse(TEXT("no SmokeRunFailed"), R.HasCode(ECode::SmokeRunFailed));
		});
	});

	// ---- acceptance 3 ----
	Describe("IoCodes", [this]()
	{
		It("label_mismatch reports LabelCountMismatch as an Error", [this]()
		{
			FTestPackage T = BuildCase(TEXT("label_mismatch"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestFalse(TEXT("not valid"), R.IsValid());
			TestTrue(*FString::Printf(TEXT("LabelCountMismatch error; report:\n%s"), *R.ToString()), R.HasError(ECode::LabelCountMismatch));
			TestEqual(TEXT("no GPU entry after an I/O error"), CountChecks(R, EPipelineDevice::GPU), 0);
		});

		It("missing_preprocessor reports InputDimMismatch as an Error", [this]()
		{
			FTestPackage T = BuildCase(TEXT("missing_preprocessor"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestFalse(TEXT("not valid"), R.IsValid());
			TestTrue(*FString::Printf(TEXT("InputDimMismatch error; report:\n%s"), *R.ToString()), R.HasError(ECode::InputDimMismatch));
		});
	});

	// ---- acceptance 4 ----
	Describe("UnknownOp", [this]()
	{
		It("unknown_op: ModelCreateFailed error prefixed 'CPU: ', CPU not created, no GPU entry", [this]()
		{
			AddExpectedError(TEXT("ONNX Runtime"), EAutomationExpectedErrorFlags::Contains, -1);
			AddExpectedError(TEXT("Session creation failed"), EAutomationExpectedErrorFlags::Contains, -1);
			FTestPackage T = BuildCase(TEXT("unknown_op"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestFalse(TEXT("not valid"), R.IsValid());
			const FPipelineValidationMessage* M = FindMsg(R, ECode::ModelCreateFailed, ESev::Error);
			TestNotNull(*FString::Printf(TEXT("ModelCreateFailed error; report:\n%s"), *R.ToString()), M);
			if (M) { TestTrue(*FString::Printf(TEXT("message starts with 'CPU: ', got '%s'"), *M->Message), M->Message.StartsWith(TEXT("CPU: "))); }
			TestEqual(TEXT("exactly one CPU entry"), CountChecks(R, EPipelineDevice::CPU), 1);
			if (const FPipelineDeviceCheck* C = FindCheck(R, EPipelineDevice::CPU))
			{
				TestFalse(TEXT("CPU bCreated"), C->bCreated);
				TestFalse(TEXT("CPU bRan"), C->bRan);
			}
			TestEqual(TEXT("no GPU entry (CPU failure stops validation)"), CountChecks(R, EPipelineDevice::GPU), 0);
			TestFalse(TEXT("no SmokeRunFailed"), R.HasCode(ECode::SmokeRunFailed));
		});
	});

	// ---- acceptance 5 ----
	Describe("Degenerate", [this]()
	{
		It("null package: exactly one message, Error NoPackage, no device checks", [this]()
		{
			const FPipelineValidationReport R = ValidatePipelineModelPackage(nullptr);
			TestEqual(*FString::Printf(TEXT("message count; report:\n%s"), *R.ToString()), R.Messages.Num(), 1);
			TestTrue(TEXT("NoPackage error"), R.HasError(ECode::NoPackage));
			TestEqual(TEXT("NumErrors"), R.NumErrors(), 1);
			TestEqual(TEXT("DeviceChecks"), R.DeviceChecks.Num(), 0);
			TestFalse(TEXT("not valid"), R.IsValid());
		});

		It("Task None: NoTask and nothing from later steps", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_text_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			T.Package->Task = EPipelineTask::None;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestTrue(*FString::Printf(TEXT("NoTask error; report:\n%s"), *R.ToString()), R.HasError(ECode::NoTask));
			TestFalse(TEXT("no message with a code >= 20"), HasLaterStepCode(R));
			TestEqual(TEXT("DeviceChecks"), R.DeviceChecks.Num(), 0);
		});

		It("Model null: NoModel, stops with no device checks", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_text_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			T.Package->Model = nullptr;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestTrue(*FString::Printf(TEXT("NoModel error; report:\n%s"), *R.ToString()), R.HasError(ECode::NoModel));
			TestFalse(TEXT("no message with a code >= 20"), HasLaterStepCode(R));
			TestEqual(TEXT("DeviceChecks"), R.DeviceChecks.Num(), 0);
		});

		It("Model null on an image package: NoModel, no crash, no device checks", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_image_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			T.Package->Model = nullptr;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestTrue(*FString::Printf(TEXT("NoModel error; report:\n%s"), *R.ToString()), R.HasError(ECode::NoModel));
			TestEqual(TEXT("DeviceChecks"), R.DeviceChecks.Num(), 0);
		});
	});

	// ---- acceptance 6 ----
	Describe("Tokenizer", [this]()
	{
		It("TokenizerJson 'not json' -> Error TokenizerInvalid, no device checks", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_text_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			T.Package->TokenizerJson = TEXT("not json");
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestFalse(TEXT("not valid"), R.IsValid());
			TestTrue(*FString::Printf(TEXT("TokenizerInvalid error; report:\n%s"), *R.ToString()), R.HasError(ECode::TokenizerInvalid));
			TestEqual(TEXT("DeviceChecks (stops before the model is created)"), R.DeviceChecks.Num(), 0);
		});

		It("TokenizerJson '{}' (valid JSON, not a tokenizer) -> Error TokenizerInvalid, no device checks", [this]()
		{
			// Catches a check that only tests "is it JSON" instead of loading the tokenizer.
			FTestPackage T = BuildCase(TEXT("tiny_text_embedding"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			T.Package->TokenizerJson = TEXT("{}");
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestTrue(*FString::Printf(TEXT("TokenizerInvalid error; report:\n%s"), *R.ToString()), R.HasError(ECode::TokenizerInvalid));
			TestEqual(TEXT("DeviceChecks"), R.DeviceChecks.Num(), 0);
		});

		It("an image package ignores TokenizerJson (no tokenizer step)", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_image_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			T.Package->TokenizerJson = TEXT("not json");
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestFalse(*FString::Printf(TEXT("no TokenizerInvalid for an image task; report:\n%s"), *R.ToString()), R.HasCode(ECode::TokenizerInvalid));
			TestTrue(TEXT("valid"), R.IsValid());
		});
	});

	// ---- acceptance 7 ----
	Describe("Options", [this]()
	{
		It("bRunSmokeTest=false: valid, CPU (and GPU) entries created but bRan false", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_text_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			FPipelineValidateOptions O;
			O.bRunSmokeTest = false;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get(), O);
			TestTrue(*FString::Printf(TEXT("valid; report:\n%s"), *R.ToString()), R.IsValid());
			TestEqual(TEXT("CPU entries"), CountChecks(R, EPipelineDevice::CPU), 1);
			if (const FPipelineDeviceCheck* C = FindCheck(R, EPipelineDevice::CPU))
			{
				TestTrue(TEXT("CPU bCreated"), C->bCreated);
				TestFalse(TEXT("CPU bRan"), C->bRan);
			}
			if (GpuGate(TEXT("bRunSmokeTest=false")))
			{
				TestEqual(TEXT("GPU entries"), CountChecks(R, EPipelineDevice::GPU), 1);
				if (const FPipelineDeviceCheck* G = FindCheck(R, EPipelineDevice::GPU))
				{
					TestTrue(TEXT("GPU bCreated"), G->bCreated);
					TestFalse(TEXT("GPU bRan"), G->bRan);
				}
			}
		});

		It("bRunSmokeTest=false on an image model still creates the model", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_image_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			FPipelineValidateOptions O;
			O.bRunSmokeTest = false;
			O.bCheckGpu = false;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get(), O);
			TestTrue(*FString::Printf(TEXT("valid; report:\n%s"), *R.ToString()), R.IsValid());
			const FPipelineDeviceCheck* C = FindCheck(R, EPipelineDevice::CPU);
			TestNotNull(TEXT("CPU entry"), C);
			if (C) { TestTrue(TEXT("bCreated"), C->bCreated); TestFalse(TEXT("bRan"), C->bRan); }
		});

		It("bCheckGpu=false: only a CPU entry", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_text_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			FPipelineValidateOptions O;
			O.bCheckGpu = false;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get(), O);
			TestTrue(*FString::Printf(TEXT("valid; report:\n%s"), *R.ToString()), R.IsValid());
			TestEqual(TEXT("DeviceChecks count"), R.DeviceChecks.Num(), 1);
			TestEqual(TEXT("CPU entries"), CountChecks(R, EPipelineDevice::CPU), 1);
			TestEqual(TEXT("GPU entries"), CountChecks(R, EPipelineDevice::GPU), 0);
			if (const FPipelineDeviceCheck* C = FindCheck(R, EPipelineDevice::CPU)) { TestTrue(TEXT("CPU bRan"), C->bRan); }
		});
	});

	// ---- acceptance 8 ----
	Describe("Backend", [this]()
	{
		It("GpuRuntimeName 'NoSuchRuntime': still valid, Info RuntimeUnavailable 'GPU: ...', GPU not available", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_text_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			FPipelineNNEBackendConfig Cfg;
			Cfg.GpuRuntimeName = TEXT("NoSuchRuntime");
			FPipelineNNEBackend Backend(Cfg);
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get(), FPipelineValidateOptions(), Backend);
			TestTrue(*FString::Printf(TEXT("valid; report:\n%s"), *R.ToString()), R.IsValid());
			TestEqual(TEXT("NumWarnings"), R.NumWarnings(), 0);
			const FPipelineValidationMessage* M = FindMsg(R, ECode::RuntimeUnavailable, ESev::Info);
			TestNotNull(TEXT("Info RuntimeUnavailable"), M);
			if (M) { TestTrue(*FString::Printf(TEXT("message starts with 'GPU: ', got '%s'"), *M->Message), M->Message.StartsWith(TEXT("GPU: "))); }
			TestFalse(TEXT("RuntimeUnavailable is not an Error"), R.HasError(ECode::RuntimeUnavailable));
			const FPipelineDeviceCheck* G = FindCheck(R, EPipelineDevice::GPU);
			if (G)
			{
				TestFalse(TEXT("GPU entry bAvailable"), G->bAvailable);
				TestFalse(TEXT("GPU entry bCreated"), G->bCreated);
				TestFalse(TEXT("GPU entry bRan"), G->bRan);
			}
			const FPipelineDeviceCheck* C = FindCheck(R, EPipelineDevice::CPU);
			TestNotNull(TEXT("CPU entry"), C);
			if (C) { TestTrue(TEXT("CPU bRan (CPU is unaffected by the bogus GPU name)"), C->bRan); }
			// The CPU check ran on the CPU backend only; this holds on a machine without a GPU as well (C11).
		});

		It("CpuRuntimeName 'NoSuchRuntime' (and a bogus GPU): Error RuntimeUnavailable, nothing created", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_text_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			FPipelineNNEBackendConfig Cfg;
			Cfg.CpuRuntimeName = TEXT("NoSuchRuntime");
			Cfg.GpuRuntimeName = TEXT("NoSuchRuntime2");
			FPipelineNNEBackend Backend(Cfg);
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get(), FPipelineValidateOptions(), Backend);
			TestFalse(TEXT("not valid"), R.IsValid());
			TestTrue(*FString::Printf(TEXT("RuntimeUnavailable error; report:\n%s"), *R.ToString()), R.HasError(ECode::RuntimeUnavailable));
			for (const FPipelineDeviceCheck& C : R.DeviceChecks)
			{
				TestFalse(TEXT("no device check created or ran"), C.bCreated || C.bRan);
			}
			TestFalse(TEXT("no SmokeRunFailed"), R.HasCode(ECode::SmokeRunFailed));
		});

		It("bogus GPU name with an image package: valid and CPU smoke run passes", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_image_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			FPipelineNNEBackendConfig Cfg;
			Cfg.GpuRuntimeName = TEXT("NoSuchRuntime");
			FPipelineNNEBackend Backend(Cfg);
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get(), FPipelineValidateOptions(), Backend);
			TestTrue(*FString::Printf(TEXT("valid; report:\n%s"), *R.ToString()), R.IsValid());
			TestTrue(TEXT("Info RuntimeUnavailable present"), FindMsg(R, ECode::RuntimeUnavailable, ESev::Info) != nullptr);
		});
	});

	// ---- acceptance 10 ----
	Describe("Immutability", [this]()
	{
		for (const TCHAR* const CaseId : { TEXT("tiny_text_embedding"), TEXT("tiny_object_detection"), TEXT("wrong_io_for_task") })
		{
			const FString Id(CaseId);
			It(*FString::Printf(TEXT("%s is unchanged after validation"), *Id), [this, Id]()
			{
				FTestPackage T = BuildCase(Id);
				if (!T.IsOk()) { AddError(T.Error); return; }
				const FSnapshot Before(*T.Package);
				ValidatePipelineModelPackage(T.Package.Get());
				const FString Diff = Before.Diff(*T.Package);
				TestTrue(*FString::Printf(TEXT("package field changed by the validator: %s"), *Diff), Diff.IsEmpty());
			});
		}

		It("a failing package (bad tokenizer) is unchanged too", [this]()
		{
			FTestPackage T = BuildCase(TEXT("tiny_text_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			T.Package->TokenizerJson = TEXT("not json");
			const FSnapshot Before(*T.Package);
			ValidatePipelineModelPackage(T.Package.Get());
			const FString Diff = Before.Diff(*T.Package);
			TestTrue(*FString::Printf(TEXT("package field changed by the validator: %s"), *Diff), Diff.IsEmpty());
		});

		It("validating twice gives the same verdict and message count", [this]()
		{
			// Catches state leaking between calls (a cached tokenizer or model from the first call).
			FTestPackage T = BuildCase(TEXT("tiny_text_classification"));
			if (!T.IsOk()) { AddError(T.Error); return; }
			const FPipelineValidationReport A = ValidatePipelineModelPackage(T.Package.Get());
			const FPipelineValidationReport B = ValidatePipelineModelPackage(T.Package.Get());
			TestEqual(TEXT("IsValid"), A.IsValid(), B.IsValid());
			TestEqual(TEXT("Messages"), A.Messages.Num(), B.Messages.Num());
			TestEqual(TEXT("DeviceChecks"), A.DeviceChecks.Num(), B.DeviceChecks.Num());
		});
	});

	// ---- acceptance 9: reference models ----
	Describe("Reference", [this]()
	{
		for (const TCHAR* const Slug : ReferenceSlugs)
		{
			const FString Name(Slug);
			It(*FString::Printf(TEXT("%s validates: 0 warnings, CPU and GPU created and ran"), *Name), [this, Name]()
			{
				FTestPackage T = BuildReference(Name);
				if (!T.IsOk()) { AddError(T.Error); return; }
				const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
				TestTrue(*FString::Printf(TEXT("valid; report:\n%s"), *R.ToString()), R.IsValid());
				TestEqual(*FString::Printf(TEXT("NumWarnings; report:\n%s"), *R.ToString()), R.NumWarnings(), 0);
				const FPipelineDeviceCheck* C = FindCheck(R, EPipelineDevice::CPU);
				TestNotNull(TEXT("CPU entry"), C);
				if (C)
				{
					TestTrue(TEXT("CPU bAvailable"), C->bAvailable);
					TestTrue(TEXT("CPU bCreated"), C->bCreated);
					TestTrue(*FString::Printf(TEXT("CPU bRan (%s)"), *C->Message), C->bRan);
					AddInfo(FString::Printf(TEXT("%s CPU smoke run: %.2f ms"), *Name, C->InferenceMs));
				}
				if (GpuGate(Name))
				{
					const FPipelineDeviceCheck* G = FindCheck(R, EPipelineDevice::GPU);
					TestNotNull(TEXT("GPU entry"), G);
					if (G)
					{
						TestTrue(TEXT("GPU bAvailable"), G->bAvailable);
						TestTrue(TEXT("GPU bCreated"), G->bCreated);
						TestTrue(*FString::Printf(TEXT("GPU bRan (%s)"), *G->Message), G->bRan);
						AddInfo(FString::Printf(TEXT("%s GPU smoke run: %.2f ms"), *Name, G->InferenceMs));
					}
				}
			});
		}

		It("resnet-50 forced to ObjectDetection: MissingOutput pred_boxes and OutputRankMismatch logits", [this]()
		{
			FTestPackage T = BuildReference(TEXT("resnet-50"), EPipelineTask::ObjectDetection);
			if (!T.IsOk()) { AddError(T.Error); return; }
			T.Package->Postprocess.ScoreActivation = EPipelineScoreActivation::Sigmoid;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(T.Package.Get());
			TestFalse(TEXT("not valid"), R.IsValid());
			bool bBoxes = false, bRank = false;
			for (const FPipelineValidationMessage& M : R.Messages)
			{
				bBoxes |= (M.Code == ECode::MissingOutput && M.Severity == ESev::Error && M.Message.Contains(TEXT("pred_boxes")));
				bRank |= (M.Code == ECode::OutputRankMismatch && M.Severity == ESev::Error && M.Message.Contains(TEXT("logits")));
			}
			TestTrue(*FString::Printf(TEXT("MissingOutput naming pred_boxes; report:\n%s"), *R.ToString()), bBoxes);
			TestTrue(*FString::Printf(TEXT("OutputRankMismatch naming logits; report:\n%s"), *R.ToString()), bRank);
			if (const FPipelineDeviceCheck* C = FindCheck(R, EPipelineDevice::CPU))
			{
				TestTrue(TEXT("CPU bCreated"), C->bCreated);
				TestFalse(TEXT("CPU bRan (an I/O error skips the smoke run)"), C->bRan);
			}
			TestEqual(TEXT("no GPU entry"), CountChecks(R, EPipelineDevice::GPU), 0);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
