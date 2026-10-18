// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9R.1 visible specs: Phonon model format (C81a), SpeechToText file-model validation, IPipelineModel::AsSpeechEngine default.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "Engine/DataAsset.h"
#include "NNEModelData.h"
#include "PipelineBackend.h"
#include "PipelineNNEBackend.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "PipelineValidation.h"
#include "PipelineModelPackage.h"
#include "PipelineModelSource.h"
#include "PipelineBase.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineTestData.h"
#include "PipelineTestFileModel.h"
#include "PipelineBaseTestPipeline.h"

namespace PipelinePhononSpecPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;

	// Numeric values from contracts C60d / C72, so this file does not depend on the enumerator names.
	static const ECode NoModel = (ECode)3;
	static const ECode NoTokenizer = (ECode)4;
	static const ECode BothModelsSet = (ECode)15;
	static const ECode WrongModelFormat = (ECode)16;
	static const ECode InvalidGgufModel = (ECode)17;
	static const ECode NoDecoderModel = (ECode)19;
	static const ECode UnexpectedDecoderModel = (ECode)72;
	static const ECode InvalidSpeechSettings = (ECode)73;

	static int32 CountCode(const FPipelineValidationReport& R, ECode Code)
	{
		int32 N = 0;
		for (const FPipelineValidationMessage& M : R.Messages) { if (M.Code == Code) { ++N; } }
		return N;
	}

	static const FPipelineValidationMessage* FindCode(const FPipelineValidationReport& R, ECode Code)
	{
		for (const FPipelineValidationMessage& M : R.Messages) { if (M.Code == Code) { return &M; } }
		return nullptr;
	}

	/** Only the pure virtuals: AsSpeechEngine must come from the IPipelineModel default. */
	class FBareModel : public IPipelineModel
	{
	public:
		virtual FPipelineResult Initialize() override { return FPipelineResult::Ok(); }
		virtual bool IsInitialized() const override { return true; }
		virtual EPipelineDevice GetDevice() const override { return EPipelineDevice::CPU; }
		virtual TArray<FPipelineTensorDesc> GetInputDescs() const override { return {}; }
		virtual TArray<FPipelineTensorDesc> GetOutputDescs() const override { return {}; }
		virtual FPipelineResult Run(TConstArrayView<FPipelineTensor>, TArray<FPipelineTensor>& Out) override
		{
			Out.Reset();
			return FPipelineResult::Fail(EPipelineError::InferenceFailed, TEXT("bare"));
		}
	};
}

BEGIN_DEFINE_SPEC(FPipelinePhononFormatSpec, "Pipelines.Core.PhononFormat",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	TArray<TStrongObjectPtr<UObject>> Keep;

	template <typename T>
	T* Track(T* Obj)
	{
		Keep.Add(TStrongObjectPtr<UObject>(Obj));
		return Obj;
	}

	UPipelineTestFileModel* MakeFile(EPipelineModelFormat Format, int64 Size = 4096, const FString& Sha = FString::ChrN(64, TEXT('c')),
		const FString& Path = TEXT("D:/Models/phonon_weights"))
	{
		UPipelineTestFileModel* F = Track(NewObject<UPipelineTestFileModel>(GetTransientPackage()));
		F->Format = Format;
		F->FilePath = Path;
		F->SizeBytes = Size;
		F->Sha256 = Sha;
		return F;
	}

	/** Package with a task, optional NNE model, optional file model. Tokenizer left empty on purpose. */
	UPipelineModelPackage* MakePackage(EPipelineTask Task, bool bNne, UDataAsset* File)
	{
		UPipelineModelPackage* P = Track(NewObject<UPipelineModelPackage>(GetTransientPackage()));
		P->Task = Task;
		if (bNne) { P->Model = Track(NewObject<UNNEModelData>(GetTransientPackage())); }
		P->GgufModel = File;
		P->License.SpdxId = TEXT("Apache-2.0");
		P->License.SourceUrl = TEXT("https://example.com/m");
		P->License.Attribution = TEXT("Example Authors");
		if (TaskUsesLabels(Task)) { P->Labels = {TEXT("a"), TEXT("b")}; }
		return P;
	}

	FPipelineValidationReport Validate(const UPipelineModelPackage* P, bool& bOk) const
	{
		FPipelineValidationReport R;
		bOk = P->ValidateSettings(R);
		return R;
	}

	/** Expects a message with Code whose text contains Phrase (case-sensitive). */
	void ExpectMessage(const FString& What, const FPipelineValidationReport& R, EPipelineValidationCode Code, const TCHAR* Phrase)
	{
		const FPipelineValidationMessage* M = PipelinePhononSpecPrivate::FindCode(R, Code);
		if (!TestNotNull(What + FString::Printf(TEXT(": a message with code %d; report:\n"), (int32)Code) + R.ToString(), M)) { return; }
		TestTrue(What + TEXT(": message contains '") + Phrase + TEXT("', got: ") + M->Message, M->Message.Contains(Phrase, ESearchCase::CaseSensitive));
	}

END_DEFINE_SPEC(FPipelinePhononFormatSpec)

void FPipelinePhononFormatSpec::Define()
{
	using namespace PipelinePhononSpecPrivate;

	AfterEach([this]() { Keep.Reset(); });

	Describe("EPipelineModelFormat", [this]()
	{
		It("Phonon has the contract value 3 and None/NNE/GGUF keep 0/1/2", [this]()
		{
			TestEqual(TEXT("None"), (int32)EPipelineModelFormat::None, 0);
			TestEqual(TEXT("NNE"), (int32)EPipelineModelFormat::NNE, 1);
			TestEqual(TEXT("GGUF"), (int32)EPipelineModelFormat::GGUF, 2);
			TestEqual(TEXT("Phonon"), (int32)EPipelineModelFormat::Phonon, 3);
		});
	});

	Describe("GetModelFormat", [this]()
	{
		It("Model only -> NNE", [this]()
		{
			TestTrue(TEXT("NNE"), MakePackage(EPipelineTask::SpeechToText, true, nullptr)->GetModelFormat() == EPipelineModelFormat::NNE);
		});

		It("GGUF file model only -> GGUF", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile(EPipelineModelFormat::GGUF));
			TestTrue(TEXT("GGUF"), P->GetModelFormat() == EPipelineModelFormat::GGUF);
		});

		It("Phonon file model only -> Phonon (regardless of task)", [this]()
		{
			// A format check that looked at the task instead of the asset would answer None for a non-speech task.
			const EPipelineTask Tasks[3] = {EPipelineTask::SpeechToText, EPipelineTask::TextGeneration, EPipelineTask::TextEmbedding};
			for (EPipelineTask T : Tasks)
			{
				UPipelineModelPackage* P = MakePackage(T, false, MakeFile(EPipelineModelFormat::Phonon));
				TestTrue(FString::Printf(TEXT("task %s -> Phonon"), *FString(LexToString(T))), P->GetModelFormat() == EPipelineModelFormat::Phonon);
			}
		});

		It("file model with format None, with format NNE, or not an IPipelineFileModel -> None", [this]()
		{
			const EPipelineModelFormat Bad[2] = {EPipelineModelFormat::None, EPipelineModelFormat::NNE};
			for (EPipelineModelFormat F : Bad)
			{
				UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(F));
				TestTrue(FString::Printf(TEXT("file format %d -> None"), (int32)F), P->GetModelFormat() == EPipelineModelFormat::None);
			}
			UDataAsset* Plain = Track(NewObject<UPipelineTestPlainAsset>(GetTransientPackage()));
			TestTrue(TEXT("plain asset -> None"), MakePackage(EPipelineTask::SpeechToText, false, Plain)->GetModelFormat() == EPipelineModelFormat::None);
		});

		It("Model and Phonon file model both set -> None; neither -> None", [this]()
		{
			TestTrue(TEXT("both"), MakePackage(EPipelineTask::SpeechToText, true, MakeFile(EPipelineModelFormat::Phonon))->GetModelFormat() == EPipelineModelFormat::None);
			TestTrue(TEXT("neither"), MakePackage(EPipelineTask::SpeechToText, false, nullptr)->GetModelFormat() == EPipelineModelFormat::None);
		});
	});

	Describe("FromPackage", [this]()
	{
		It("Phonon fills FilePath/FileSizeBytes/FileSha256 like GGUF and leaves NNEModel null", [this]()
		{
			UPipelineTestFileModel* F = MakeFile(EPipelineModelFormat::Phonon, 987654321LL, FString::ChrN(64, TEXT('e')), TEXT("D:/w/phonon_en"));
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, F);
			const FPipelineModelSource S = FPipelineModelSource::FromPackage(P);
			TestTrue(TEXT("Format Phonon"), S.Format == EPipelineModelFormat::Phonon);
			TestEqual(TEXT("FilePath"), S.FilePath, FString(TEXT("D:/w/phonon_en")));
			TestEqual(TEXT("FileSizeBytes (> 32 bit)"), S.FileSizeBytes, (int64)987654321LL);
			TestEqual(TEXT("FileSha256"), S.FileSha256, FString::ChrN(64, TEXT('e')));
			TestTrue(TEXT("NNEModel null"), S.NNEModel == nullptr);
		});

		It("a file model with format None gives Format None", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(EPipelineModelFormat::None));
			TestTrue(TEXT("None"), FPipelineModelSource::FromPackage(P).Format == EPipelineModelFormat::None);
		});
	});

	Describe("ValidateSettings: SpeechToText + Phonon", [this]()
	{
		It("valid: no errors, no tokenizer or decoder model needed, no messages at all", [this]()
		{
			// Catches a rule that still demands NoDecoderModel(19) or NoTokenizer(4) for the one-folder format.
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(EPipelineModelFormat::Phonon));
			TestTrue(TEXT("precondition: TokenizerJson empty"), P->TokenizerJson.IsEmpty());
			bool bOk = false;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestTrue(TEXT("ValidateSettings true; report:\n") + R.ToString(), bOk);
			TestEqual(TEXT("errors: ") + R.ToString(), R.NumErrors(), 0);
			TestFalse(TEXT("no NoDecoderModel"), R.HasCode(NoDecoderModel));
			TestFalse(TEXT("no NoTokenizer"), R.HasCode(NoTokenizer));
			TestFalse(TEXT("no NoModel"), R.HasCode(NoModel));
			TestFalse(TEXT("no WrongModelFormat"), R.HasCode(WrongModelFormat));
			TestEqual(TEXT("no messages: ") + R.ToString(), R.Messages.Num(), 0);
		});

		It("with DecoderModel set: exactly one Warning 72, no error, still valid", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(EPipelineModelFormat::Phonon));
			P->DecoderModel = Track(NewObject<UNNEModelData>(GetTransientPackage()));
			bool bOk = false;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestTrue(TEXT("ValidateSettings true (warning only); report:\n") + R.ToString(), bOk);
			TestEqual(TEXT("errors"), R.NumErrors(), 0);
			TestEqual(TEXT("exactly one 72: ") + R.ToString(), CountCode(R, UnexpectedDecoderModel), 1);
			TestEqual(TEXT("one message in total: ") + R.ToString(), R.Messages.Num(), 1);
			const FPipelineValidationMessage* M = FindCode(R, UnexpectedDecoderModel);
			if (M)
			{
				TestTrue(TEXT("severity Warning"), M->Severity == ESev::Warning);
				TestTrue(TEXT("message names the weights folder: ") + M->Message, M->Message.Contains(TEXT("one weights folder"), ESearchCase::CaseSensitive));
			}
		});

		It("FPipelineSpeechSettings checks still apply (WindowLength > FftSize -> Error 73)", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(EPipelineModelFormat::Phonon));
			P->Speech.WindowLength = 600; // FftSize defaults to 512
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(TEXT("Error 73; report:\n") + R.ToString(), R.HasError(InvalidSpeechSettings));
			TestFalse(TEXT("no NoTokenizer"), R.HasCode(NoTokenizer));
		});

		It("bad path, size 0 and bad SHA are each InvalidGgufModel(17) Errors", [this]()
		{
			struct FCase { const TCHAR* Name; FString Path; int64 Size; FString Sha; };
			const FCase Cases[] = {
				{TEXT("empty path"), FString(), 4096, FString::ChrN(64, TEXT('c'))},
				{TEXT("size 0"), TEXT("D:/w/p"), 0, FString::ChrN(64, TEXT('c'))},
				{TEXT("negative size"), TEXT("D:/w/p"), -1, FString::ChrN(64, TEXT('c'))},
				{TEXT("sha too short"), TEXT("D:/w/p"), 4096, FString::ChrN(63, TEXT('c'))},
				{TEXT("sha not hex"), TEXT("D:/w/p"), 4096, FString::ChrN(64, TEXT('g'))},
			};
			for (const FCase& C : Cases)
			{
				UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(EPipelineModelFormat::Phonon, C.Size, C.Sha, C.Path));
				bool bOk = true;
				const FPipelineValidationReport R = Validate(P, bOk);
				TestFalse(FString(C.Name) + TEXT(": expected false"), bOk);
				TestTrue(FString(C.Name) + TEXT(": Error 17; report:\n") + R.ToString(), R.HasError(InvalidGgufModel));
				TestFalse(FString(C.Name) + TEXT(": not WrongModelFormat"), R.HasCode(WrongModelFormat));
			}
		});

		It("empty path message says 'its path is invalid'", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(EPipelineModelFormat::Phonon, 4096, FString::ChrN(64, TEXT('c')), FString()));
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			ExpectMessage(TEXT("empty path"), R, InvalidGgufModel, TEXT("its path is invalid"));
		});

		It("empty path + size 0 + bad SHA together: the 17 report(s) mention the path problem; not valid", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(EPipelineModelFormat::Phonon, 0, TEXT("xyz"), FString()));
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(TEXT("Error 17; report:\n") + R.ToString(), R.HasError(InvalidGgufModel));
			bool bPath = false;
			for (const FPipelineValidationMessage& M : R.Messages)
			{
				if (M.Code == InvalidGgufModel && M.Message.Contains(TEXT("its path is invalid"), ESearchCase::CaseSensitive)) { bPath = true; }
			}
			TestTrue(TEXT("a 17 message contains 'its path is invalid'; report:\n") + R.ToString(), bPath);
		});

		It("file model whose format is None (or not an IPipelineFileModel) -> Error 17 'not a GGUF or Phonon model asset'", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(EPipelineModelFormat::None));
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("None format: expected false"), bOk);
			ExpectMessage(TEXT("None format"), R, InvalidGgufModel, TEXT("is not a GGUF or Phonon model asset"));

			UDataAsset* Plain = Track(NewObject<UPipelineTestPlainAsset>(GetTransientPackage()));
			UPipelineModelPackage* P2 = MakePackage(EPipelineTask::SpeechToText, false, Plain);
			const FPipelineValidationReport R2 = Validate(P2, bOk);
			TestFalse(TEXT("plain asset: expected false"), bOk);
			ExpectMessage(TEXT("plain asset"), R2, InvalidGgufModel, TEXT("is not a GGUF or Phonon model asset"));
		});
	});

	Describe("ValidateSettings: WrongModelFormat table", [this]()
	{
		It("SpeechToText + GGUF file model -> Error 16 naming the stage-A models or a Phonon asset", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(EPipelineModelFormat::GGUF));
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(TEXT("Error 16; report:\n") + R.ToString(), R.HasError(WrongModelFormat));
			ExpectMessage(TEXT("S2T+GGUF"), R, WrongModelFormat, TEXT("SpeechToText needs the stage-A NNE models or a Phonon weights asset"));
		});

		It("TextGeneration + Phonon file model -> Error 16 'TextGeneration needs a GGUF model'", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile(EPipelineModelFormat::Phonon));
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			ExpectMessage(TEXT("TextGeneration+Phonon"), R, WrongModelFormat, TEXT("TextGeneration needs a GGUF model"));
		});

		It("TextClassification and TextEmbedding + Phonon -> Error 16 'needs an NNE model'", [this]()
		{
			const EPipelineTask Tasks[2] = {EPipelineTask::TextClassification, EPipelineTask::TextEmbedding};
			for (EPipelineTask T : Tasks)
			{
				UPipelineModelPackage* P = MakePackage(T, false, MakeFile(EPipelineModelFormat::Phonon));
				P->TokenizerJson = TEXT("{}");
				bool bOk = true;
				const FPipelineValidationReport R = Validate(P, bOk);
				TestFalse(FString(LexToString(T)) + TEXT(": expected false"), bOk);
				ExpectMessage(LexToString(T), R, WrongModelFormat, *(FString(LexToString(T)) + TEXT(" needs an NNE model")));
			}
		});

		It("TextGeneration + GGUF file model is still valid (unchanged)", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile(EPipelineModelFormat::GGUF));
			P->TokenizerJson = TEXT("{}");
			bool bOk = false;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestTrue(TEXT("valid; report:\n") + R.ToString(), bOk);
			TestEqual(TEXT("errors"), R.NumErrors(), 0);
		});

		It("SpeechToText with Model AND a Phonon file model -> Error 15 (BothModelsSet unchanged)", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, true, MakeFile(EPipelineModelFormat::Phonon));
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(TEXT("Error 15; report:\n") + R.ToString(), R.HasError(BothModelsSet));
		});
	});

	Describe("ValidateSettings: NoModel and unchanged NNE SpeechToText", [this]()
	{
		It("SpeechToText with no model at all -> NoModel with the new message", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, nullptr);
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			ExpectMessage(TEXT("S2T NoModel"), R, NoModel, TEXT("Model is empty; assign the imported NNE model data (stage A) or a Phonon weights asset in GgufModel (stage B)"));
		});

		It("other tasks with no model keep their NoModel message (no Phonon text)", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextClassification, false, nullptr);
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			const FPipelineValidationMessage* M = FindCode(R, NoModel);
			if (TestNotNull(TEXT("NoModel present; report:\n") + R.ToString(), M))
			{
				TestFalse(TEXT("message does not mention Phonon: ") + M->Message, M->Message.Contains(TEXT("Phonon"), ESearchCase::CaseSensitive));
			}
		});

		It("NNE SpeechToText: no decoder -> 19; empty tokenizer -> 4; both present -> valid", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, true, nullptr);
			bool bOk = true;
			FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("no decoder, no tokenizer: expected false"), bOk);
			TestTrue(TEXT("19 raised for NNE; report:\n") + R.ToString(), R.HasError(NoDecoderModel));
			TestTrue(TEXT("4 raised for NNE; report:\n") + R.ToString(), R.HasError(NoTokenizer));

			P->DecoderModel = Track(NewObject<UNNEModelData>(GetTransientPackage()));
			P->TokenizerJson = TEXT("x");
			R = Validate(P, bOk);
			TestTrue(TEXT("complete NNE package valid; report:\n") + R.ToString(), bOk);
			TestEqual(TEXT("no messages"), R.Messages.Num(), 0);
		});
	});

	Describe("InitializeFromPackage", [this]()
	{
		It("Phonon package with no Phonon backend -> RuntimeUnavailable naming Phonon, no crash", [this]()
		{
			if (IPipelineBackend* External = FModelPipelinesCoreModule::Get().FindBackend(EPipelineModelFormat::Phonon))
			{
				AddInfo(FString::Printf(TEXT("SKIPPED: a Phonon backend (%s) is registered by another plugin."), *External->GetName().ToString()));
				return;
			}
			UPipelineModelPackage* P = MakePackage(EPipelineTask::SpeechToText, false, MakeFile(EPipelineModelFormat::Phonon));
			UPipelineSpeechTestPipeline* Pipe = Track(NewObject<UPipelineSpeechTestPipeline>(GetTransientPackage()));
			const FPipelineResult R = Pipe->InitializeFromPackage(P);
			TestFalse(TEXT("expected failure"), R.bSuccess);
			TestTrue(FString::Printf(TEXT("expected RuntimeUnavailable, got %d '%s'"), (int32)R.Error, *R.Message), R.Error == EPipelineError::RuntimeUnavailable);
			TestTrue(TEXT("message names Phonon: ") + R.Message, R.Message.Contains(TEXT("Phonon"), ESearchCase::CaseSensitive));
		});

		It("FindBackend(Phonon) is null with Core alone (or a non-default backend when the paid plugin is loaded)", [this]()
		{
			FModelPipelinesCoreModule& M = FModelPipelinesCoreModule::Get();
			TestTrue(TEXT("the NNE default backend must not answer for Phonon"), M.FindBackend(EPipelineModelFormat::Phonon) != &M.GetDefaultBackend());
		});

		It("the NNE backend does not support Phonon", [this]()
		{
			FPipelineNNEBackend Backend;
			TestFalse(TEXT("SupportsFormat(Phonon)"), Backend.SupportsFormat(EPipelineModelFormat::Phonon));
			TestTrue(TEXT("SupportsFormat(NNE)"), Backend.SupportsFormat(EPipelineModelFormat::NNE));
			FPipelineModelSource S;
			S.Format = EPipelineModelFormat::Phonon;
			S.FilePath = TEXT("C:/none");
			FPipelineResult R;
			TSharedPtr<IPipelineModel> M = Backend.PrepareModelFromSource(S, EPipelineDevice::CPU, R);
			TestFalse(TEXT("model null"), M.IsValid());
			TestTrue(FString::Printf(TEXT("RuntimeUnavailable expected, got %d"), (int32)R.Error), R.Error == EPipelineError::RuntimeUnavailable);
		});
	});

	Describe("AsSpeechEngine", [this]()
	{
		It("default on a model with only the pure virtuals returns nullptr", [this]()
		{
			FBareModel Model;
			TestTrue(TEXT("nullptr"), Model.AsSpeechEngine() == nullptr);
			IPipelineModel& Base = Model;
			TestTrue(TEXT("nullptr via base reference"), Base.AsSpeechEngine() == nullptr);
		});

		It("an NNE model prepared by the NNE backend returns nullptr", [this]()
		{
			FString Err;
			UNNEModelData* Data = UE::ModelPipelines::Tests::LoadTestModel(TEXT("affine_dynamic.onnx"), Err);
			if (!Data)
			{
				AddError(FString::Printf(TEXT("Cannot load affine_dynamic.onnx: %s. Generate it with: "
					"D:/Micromamba/micromambaenv/envs/pipelines/python.exe \"D:/Game Dev/Pipelines-UE/ModelPipelines-UE/Tools/make_test_models.py\""), *Err));
				return;
			}
			Track(Data);
			FPipelineNNEBackend Backend;
			FPipelineResult R;
			TSharedPtr<IPipelineModel> Model = Backend.PrepareModel(Data, EPipelineDevice::CPU, R);
			if (!TestTrue(TEXT("prepare succeeded: ") + R.Message, Model.IsValid())) { return; }
			TestTrue(TEXT("AsSpeechEngine nullptr"), Model->AsSpeechEngine() == nullptr);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
