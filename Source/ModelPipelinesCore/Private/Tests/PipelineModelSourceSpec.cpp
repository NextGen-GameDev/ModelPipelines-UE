// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/Class.h"
#include "Engine/DataAsset.h"
#include "NNEModelData.h"
#include "PipelineBackend.h"
#include "PipelineNNEBackend.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "PipelineValidation.h"
#include "PipelineModelPackage.h"
#include "PipelineModelSource.h"
#include "PipelineSession.h"
#include "PipelineBase.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineTestData.h"
#include "PipelineTestFileModel.h"
#include "PipelineBaseTestPipeline.h"

namespace PipelineModelSourceSpecPrivate
{
	/** A model with only the pure virtuals, so every default of IPipelineModel is exercised. */
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

	/** Fake GGUF backend: counts PrepareModelFromSource calls and remembers the last source. Always fails the prepare. */
	class FFakeGgufBackend : public IPipelineBackend
	{
	public:
		int32 PrepareCalls = 0;
		FPipelineModelSource LastSource;

		virtual FName GetName() const override { return FName(TEXT("FakeGguf")); }
		virtual bool IsDeviceAvailable(EPipelineDevice) const override { return true; }
		virtual TSharedPtr<IPipelineModel> PrepareModel(UNNEModelData*, EPipelineDevice, FPipelineResult& Out) override
		{
			Out = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("fake: NNE not supported"));
			return nullptr;
		}
		virtual bool SupportsFormat(EPipelineModelFormat Format) const override { return Format == EPipelineModelFormat::GGUF; }
		virtual TSharedPtr<IPipelineModel> PrepareModelFromSource(const FPipelineModelSource& Source, EPipelineDevice, FPipelineResult& Out) override
		{
			++PrepareCalls;
			LastSource = Source;
			Out = FPipelineResult::Fail(EPipelineError::ModelLoadFailed, TEXT("fake backend: no real model"));
			return nullptr;
		}
	};

	static int32 CountCode(const FPipelineValidationReport& R, EPipelineValidationCode Code)
	{
		int32 N = 0;
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code) { ++N; }
		}
		return N;
	}

	static const FPipelineValidationMessage* FindCode(const FPipelineValidationReport& R, EPipelineValidationCode Code)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code) { return &M; }
		}
		return nullptr;
	}

	// Numeric values from contract C60d, so this file does not depend on the enumerator names.
	static const EPipelineValidationCode BothModelsSet = (EPipelineValidationCode)15;
	static const EPipelineValidationCode WrongModelFormat = (EPipelineValidationCode)16;
	static const EPipelineValidationCode InvalidGgufModel = (EPipelineValidationCode)17;
	static const EPipelineValidationCode LicenseNotPermissive = (EPipelineValidationCode)18;
}

BEGIN_DEFINE_SPEC(FPipelineModelSourceSpec, "Pipelines.Core.ModelSource",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	TArray<TStrongObjectPtr<UObject>> Keep;

	template <typename T>
	T* Track(T* Obj)
	{
		Keep.Add(TStrongObjectPtr<UObject>(Obj));
		return Obj;
	}

	UPipelineTestFileModel* MakeFile(int64 Size = 10, const FString& Sha = FString::ChrN(64, TEXT('a')))
	{
		UPipelineTestFileModel* F = Track(NewObject<UPipelineTestFileModel>(GetTransientPackage()));
		F->FilePath = TEXT("C:/Models/test.gguf");
		F->SizeBytes = Size;
		F->Sha256 = Sha;
		return F;
	}

	UPipelineModelPackage* MakePackage(EPipelineTask Task, bool bNne, UDataAsset* Gguf)
	{
		UPipelineModelPackage* P = Track(NewObject<UPipelineModelPackage>(GetTransientPackage()));
		P->Task = Task;
		if (bNne) { P->Model = Track(NewObject<UNNEModelData>(GetTransientPackage())); }
		P->GgufModel = Gguf;
		P->License.SpdxId = TEXT("Apache-2.0");
		P->License.SourceUrl = TEXT("https://example.com/m");
		P->License.Attribution = TEXT("Example Authors");
		if (IsTextTask(Task)) { P->TokenizerJson = TEXT("{}"); }
		if (TaskUsesLabels(Task)) { P->Labels = {TEXT("a"), TEXT("b")}; }
		return P;
	}

	FPipelineValidationReport Validate(const UPipelineModelPackage* P, bool& bOk) const
	{
		FPipelineValidationReport R;
		bOk = P->ValidateSettings(R);
		return R;
	}

	/** Backends a test registered; AfterEach unregisters them even when the test failed half way. */
	TArray<TSharedRef<IPipelineBackend>> Registered;

	void Reg(const TSharedRef<IPipelineBackend>& B)
	{
		FModelPipelinesCoreModule::Get().RegisterBackend(B);
		Registered.Add(B);
	}

END_DEFINE_SPEC(FPipelineModelSourceSpec)

void FPipelineModelSourceSpec::Define()
{
	using namespace PipelineModelSourceSpecPrivate;

	AfterEach([this]()
	{
		for (const TSharedRef<IPipelineBackend>& B : Registered)
		{
			FModelPipelinesCoreModule::Get().UnregisterBackend(B);
		}
		Registered.Reset();
		Keep.Reset();
	});

	Describe("Task enum", [this]()
	{
		It("names the three new tasks and 8 is Unknown", [this]()
		{
			TestEqual(TEXT("TextGeneration"), FString(LexToString(EPipelineTask::TextGeneration)), FString(TEXT("TextGeneration")));
			TestEqual(TEXT("SpeechToText"), FString(LexToString(EPipelineTask::SpeechToText)), FString(TEXT("SpeechToText")));
			TestEqual(TEXT("TextToSpeech"), FString(LexToString(EPipelineTask::TextToSpeech)), FString(TEXT("TextToSpeech")));
			TestEqual(TEXT("(EPipelineTask)8 expected Unknown"), FString(LexToString((EPipelineTask)8)), FString(TEXT("Unknown")));
			TestEqual(TEXT("values 5/6/7"), (int32)EPipelineTask::TextGeneration * 100 + (int32)EPipelineTask::SpeechToText * 10 + (int32)EPipelineTask::TextToSpeech, 567);
		});

		It("parses the names case-insensitively and rejects near misses", [this]()
		{
			EPipelineTask T = EPipelineTask::None;
			TestTrue(TEXT("parse textgeneration"), LexTryParseString(T, TEXT("textgeneration")));
			TestTrue(TEXT("-> TextGeneration"), T == EPipelineTask::TextGeneration);
			T = EPipelineTask::None;
			TestTrue(TEXT("parse SPEECHTOTEXT"), LexTryParseString(T, TEXT("SPEECHTOTEXT")));
			TestTrue(TEXT("-> SpeechToText"), T == EPipelineTask::SpeechToText);
			T = EPipelineTask::None;
			TestTrue(TEXT("parse TextToSpeech"), LexTryParseString(T, TEXT("TextToSpeech")));
			TestTrue(TEXT("-> TextToSpeech"), T == EPipelineTask::TextToSpeech);
			TestFalse(TEXT("near miss must not parse"), LexTryParseString(T, TEXT("TextGeneration2")));
		});

		It("TaskUsesLabels is false for 0,1,5,6,7 and true for 2,3,4 (a generative task wanting labels would fail validation)", [this]()
		{
			const bool Expect[8] = {false, false, true, true, true, false, false, false};
			for (int32 i = 0; i < 8; ++i)
			{
				TestEqual(FString::Printf(TEXT("TaskUsesLabels(%d)"), i), TaskUsesLabels((EPipelineTask)i), Expect[i]);
			}
		});

		It("IsGenerativeTask is true only for 5,6,7; IsTextTask(TextGeneration) is false", [this]()
		{
			for (int32 i = 0; i < 8; ++i)
			{
				TestEqual(FString::Printf(TEXT("IsGenerativeTask(%d)"), i), IsGenerativeTask((EPipelineTask)i), i >= 5);
			}
			TestFalse(TEXT("IsTextTask(TextGeneration) expected false"), IsTextTask(EPipelineTask::TextGeneration));
			TestFalse(TEXT("IsImageTask(TextGeneration) expected false"), IsImageTask(EPipelineTask::TextGeneration));
			TestTrue(TEXT("IsTextTask(TextEmbedding) unchanged"), IsTextTask(EPipelineTask::TextEmbedding));
		});
	});

	Describe("Error code", [this]()
	{
		It("ContextOverflow is 11 and named ContextOverflow", [this]()
		{
			TestEqual(TEXT("value"), (int32)(uint8)EPipelineError::ContextOverflow, 11);
			const UEnum* E = StaticEnum<EPipelineError>();
			if (TestNotNull(TEXT("UEnum for EPipelineError"), E))
			{
				TestEqual(TEXT("UEnum name"), E->GetNameStringByValue(11), FString(TEXT("ContextOverflow")));
			}
		});
	});

	Describe("Generation settings", [this]()
	{
		It("has the contract defaults", [this]()
		{
			const FPipelineGenerationModelSettings S;
			TestEqual(TEXT("ContextLength"), S.ContextLength, 0);
			TestEqual(TEXT("GpuLayers"), S.GpuLayers, -1);
			TestEqual(TEXT("ChatTemplateOverride"), S.ChatTemplateOverride, FString());
			TestEqual(TEXT("Threads"), S.Threads, 0);
			TestEqual(TEXT("MaxCachedConversations (amendment 1)"), S.MaxCachedConversations, 4);
		});

		It("is copied into the model source", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile());
			P->Generation.ContextLength = 2048;
			P->Generation.GpuLayers = 7;
			P->Generation.ChatTemplateOverride = TEXT("chatml");
			P->Generation.Threads = 3;
			P->Generation.MaxCachedConversations = 9;
			const FPipelineModelSource S = FPipelineModelSource::FromPackage(P);
			TestEqual(TEXT("ContextLength"), S.Generation.ContextLength, 2048);
			TestEqual(TEXT("GpuLayers"), S.Generation.GpuLayers, 7);
			TestEqual(TEXT("ChatTemplateOverride"), S.Generation.ChatTemplateOverride, FString(TEXT("chatml")));
			TestEqual(TEXT("Threads"), S.Generation.Threads, 3);
			TestEqual(TEXT("MaxCachedConversations"), S.Generation.MaxCachedConversations, 9);
		});
	});

	Describe("GetModelFormat and FromPackage", [this]()
	{
		It("NNE only -> NNE", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextEmbedding, true, nullptr);
			TestTrue(TEXT("expected NNE"), P->GetModelFormat() == EPipelineModelFormat::NNE);
			const FPipelineModelSource S = FPipelineModelSource::FromPackage(P);
			TestTrue(TEXT("source NNE"), S.Format == EPipelineModelFormat::NNE);
			TestTrue(TEXT("source NNEModel is the package's"), S.NNEModel == P->Model.Get());
		});

		It("file model only -> GGUF, and FromPackage copies path/size/sha", [this]()
		{
			UPipelineTestFileModel* F = MakeFile(12345, FString::ChrN(64, TEXT('b')));
			F->FilePath = TEXT("D:/x/y/m.gguf");
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, F);
			TestTrue(TEXT("expected GGUF"), P->GetModelFormat() == EPipelineModelFormat::GGUF);
			const FPipelineModelSource S = FPipelineModelSource::FromPackage(P);
			TestTrue(TEXT("source GGUF"), S.Format == EPipelineModelFormat::GGUF);
			TestEqual(TEXT("FilePath"), S.FilePath, FString(TEXT("D:/x/y/m.gguf")));
			TestEqual(TEXT("FileSizeBytes"), S.FileSizeBytes, (int64)12345);
			TestEqual(TEXT("FileSha256"), S.FileSha256, FString::ChrN(64, TEXT('b')));
			TestTrue(TEXT("NNEModel null for GGUF"), S.NNEModel == nullptr);
		});

		It("both set -> None; neither set -> None", [this]()
		{
			UPipelineModelPackage* Both = MakePackage(EPipelineTask::TextGeneration, true, MakeFile());
			TestTrue(TEXT("both: GetModelFormat None"), Both->GetModelFormat() == EPipelineModelFormat::None);
			TestTrue(TEXT("both: source None"), FPipelineModelSource::FromPackage(Both).Format == EPipelineModelFormat::None);
			UPipelineModelPackage* Neither = MakePackage(EPipelineTask::TextGeneration, false, nullptr);
			TestTrue(TEXT("neither: GetModelFormat None"), Neither->GetModelFormat() == EPipelineModelFormat::None);
			TestTrue(TEXT("neither: source None"), FPipelineModelSource::FromPackage(Neither).Format == EPipelineModelFormat::None);
		});

		It("GgufModel that is a plain UDataAsset (no interface) -> None", [this]()
		{
			UDataAsset* Plain = Track(NewObject<UPipelineTestPlainAsset>(GetTransientPackage()));
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, Plain);
			TestTrue(TEXT("GetModelFormat None"), P->GetModelFormat() == EPipelineModelFormat::None);
			TestTrue(TEXT("source None"), FPipelineModelSource::FromPackage(P).Format == EPipelineModelFormat::None);
		});

		It("null package -> None, no crash", [this]()
		{
			TestTrue(TEXT("expected None"), FPipelineModelSource::FromPackage(nullptr).Format == EPipelineModelFormat::None);
		});
	});

	Describe("ValidateSettings", [this]()
	{
		It("TextGeneration + valid file model + license: no Error, no tokenizer/labels errors", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile(10, FString::ChrN(64, TEXT('a'))));
			bool bOk = false;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestTrue(FString::Printf(TEXT("expected true; report:\n%s"), *R.ToString()), bOk);
			TestEqual(TEXT("error count"), R.NumErrors(), 0);
			TestFalse(TEXT("no NoTokenizer"), R.HasCode(EPipelineValidationCode::NoTokenizer));
			TestFalse(TEXT("no NoLabels"), R.HasCode(EPipelineValidationCode::NoLabels));
		});

		It("TextGeneration + NNE model -> Error 16", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, true, nullptr);
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(FString::Printf(TEXT("Error 16; report:\n%s"), *R.ToString()), R.HasError(WrongModelFormat));
		});

		It("TextEmbedding + file model -> Error 16", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextEmbedding, false, MakeFile());
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(FString::Printf(TEXT("Error 16; report:\n%s"), *R.ToString()), R.HasError(WrongModelFormat));
		});

		It("both set -> Error 15", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, true, MakeFile());
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(FString::Printf(TEXT("Error 15; report:\n%s"), *R.ToString()), R.HasError(BothModelsSet));
		});

		It("sha 'abc', 63 hex chars and 64 non-hex chars -> Error 17", [this]()
		{
			const FString Bad[3] = {TEXT("abc"), FString::ChrN(63, TEXT('a')), FString::ChrN(64, TEXT('z'))};
			for (const FString& S : Bad)
			{
				UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile(10, S));
				bool bOk = true;
				const FPipelineValidationReport R = Validate(P, bOk);
				TestFalse(FString::Printf(TEXT("sha '%s': expected false"), *S), bOk);
				TestTrue(FString::Printf(TEXT("sha '%s': Error 17; report:\n%s"), *S, *R.ToString()), R.HasError(InvalidGgufModel));
			}
		});

		It("size 0 and negative -> Error 17", [this]()
		{
			const int64 Sizes[2] = {0, -5};
			for (int64 Sz : Sizes)
			{
				UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile(Sz));
				bool bOk = true;
				const FPipelineValidationReport R = Validate(P, bOk);
				TestFalse(FString::Printf(TEXT("size %lld: expected false"), Sz), bOk);
				TestTrue(FString::Printf(TEXT("size %lld: Error 17; report:\n%s"), Sz, *R.ToString()), R.HasError(InvalidGgufModel));
			}
		});

		It("GgufModel without the interface -> Error 17", [this]()
		{
			UDataAsset* Plain = Track(NewObject<UPipelineTestPlainAsset>(GetTransientPackage()));
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, Plain);
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(FString::Printf(TEXT("Error 17; report:\n%s"), *R.ToString()), R.HasError(InvalidGgufModel));
		});

		It("neither set -> Error 3 (NoModel)", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, nullptr);
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("expected false"), bOk);
			TestTrue(FString::Printf(TEXT("Error NoModel; report:\n%s"), *R.ToString()), R.HasError(EPipelineValidationCode::NoModel));
		});
	});

	Describe("License warning (code 18)", [this]()
	{
		It("permissive SPDX ids (any case) give no code 18", [this]()
		{
			const TCHAR* Ids[] = {TEXT("Apache-2.0"), TEXT("mit"), TEXT("BSD-3-Clause"), TEXT("BSD-2-Clause"), TEXT("apache-2.0")};
			for (const TCHAR* Id : Ids)
			{
				UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile());
				P->License.SpdxId = Id;
				bool bOk = false;
				const FPipelineValidationReport R = Validate(P, bOk);
				TestTrue(FString::Printf(TEXT("%s: ok; report:\n%s"), Id, *R.ToString()), bOk);
				TestEqual(FString::Printf(TEXT("%s: no code 18"), Id), CountCode(R, LicenseNotPermissive), 0);
			}
		});

		It("non-permissive ids give exactly one Warning 18 and no Error", [this]()
		{
			const TCHAR* Ids[] = {TEXT("LLAMA-3.2-COMMUNITY"), TEXT("gemma"), TEXT("CC-BY-NC-4.0")};
			for (const TCHAR* Id : Ids)
			{
				UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile());
				P->License.SpdxId = Id;
				bool bOk = false;
				const FPipelineValidationReport R = Validate(P, bOk);
				TestTrue(FString::Printf(TEXT("%s: still valid; report:\n%s"), Id, *R.ToString()), bOk);
				TestEqual(FString::Printf(TEXT("%s: one code 18"), Id), CountCode(R, LicenseNotPermissive), 1);
				const FPipelineValidationMessage* M = FindCode(R, LicenseNotPermissive);
				if (M)
				{
					TestTrue(FString::Printf(TEXT("%s: severity Warning"), Id), M->Severity == EPipelineValidationSeverity::Warning);
				}
				TestEqual(FString::Printf(TEXT("%s: no errors"), Id), R.NumErrors(), 0);
			}
		});

		It("empty SpdxId gives the existing code 11 only, not 18", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile());
			P->License.SpdxId = FString();
			bool bOk = false;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestEqual(TEXT("code 11 once"), CountCode(R, EPipelineValidationCode::LicenseSpdxMissing), 1);
			TestEqual(TEXT("no code 18"), CountCode(R, LicenseNotPermissive), 0);
		});

		It("applies to an NNE TextEmbedding package too", [this]()
		{
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextEmbedding, true, nullptr);
			P->License.SpdxId = TEXT("gemma");
			bool bOk = false;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestTrue(FString::Printf(TEXT("valid; report:\n%s"), *R.ToString()), bOk);
			TestEqual(TEXT("one code 18"), CountCode(R, LicenseNotPermissive), 1);
		});
	});

	Describe("Backend registry", [this]()
	{
		It("FindBackend(NNE) is the default backend; FindBackend(None) is null", [this]()
		{
			FModelPipelinesCoreModule& M = FModelPipelinesCoreModule::Get();
			TestTrue(TEXT("NNE -> default backend"), M.FindBackend(EPipelineModelFormat::NNE) == &M.GetDefaultBackend());
			TestTrue(TEXT("None -> null"), M.FindBackend(EPipelineModelFormat::None) == nullptr);
		});

		It("GGUF: the registered backend, then the previous one again; a second registration wins", [this]()
		{
			FModelPipelinesCoreModule& M = FModelPipelinesCoreModule::Get();
			// Null with Core alone; the Llama backend when ModelPipelinesGenerative is enabled (lead, 2026-10-12).
			IPipelineBackend* const Before = M.FindBackend(EPipelineModelFormat::GGUF);
			TSharedRef<FFakeGgufBackend> A = MakeShared<FFakeGgufBackend>();
			TSharedRef<FFakeGgufBackend> B = MakeShared<FFakeGgufBackend>();
			Reg(A);
			TestTrue(TEXT("GGUF -> A"), M.FindBackend(EPipelineModelFormat::GGUF) == &A.Get());
			TestTrue(TEXT("NNE unaffected"), M.FindBackend(EPipelineModelFormat::NNE) == &M.GetDefaultBackend());
			Reg(B);
			TestTrue(TEXT("GGUF -> B (later wins)"), M.FindBackend(EPipelineModelFormat::GGUF) == &B.Get());
			M.UnregisterBackend(B);
			M.UnregisterBackend(A);
			Registered.Reset();
			TestTrue(TEXT("GGUF back to the backend found before (null with Core alone) after unregistering both"),
				M.FindBackend(EPipelineModelFormat::GGUF) == Before);
			TestTrue(TEXT("NNE still the default"), M.FindBackend(EPipelineModelFormat::NNE) == &M.GetDefaultBackend());
		});
	});

	Describe("InitializeFromPackage", [this]()
	{
		It("GGUF package with no GGUF backend -> RuntimeUnavailable, no crash", [this]()
		{
			if (IPipelineBackend* External = FModelPipelinesCoreModule::Get().FindBackend(EPipelineModelFormat::GGUF))
			{
				// Another plugin (ModelPipelinesGenerative) registered a GGUF backend at startup and Core can't remove it
				// here. This case runs where only the free plugin is installed (fresh-project acceptance). (Lead, 2026-10-12.)
				AddInfo(FString::Printf(TEXT("SKIPPED: a GGUF backend (%s) is registered by another plugin."), *External->GetName().ToString()));
				return;
			}
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile());
			UPipelineTestGenerationPipeline* Pipe = Track(NewObject<UPipelineTestGenerationPipeline>(GetTransientPackage()));
			const FPipelineResult R = Pipe->InitializeFromPackage(P);
			TestFalse(TEXT("expected failure"), R.bSuccess);
			TestTrue(FString::Printf(TEXT("expected RuntimeUnavailable, got %d '%s'"), (int32)R.Error, *R.Message),
				R.Error == EPipelineError::RuntimeUnavailable);
		});

		It("GGUF package with a registered backend -> PrepareModelFromSource called once with the package's source", [this]()
		{
			TSharedRef<FFakeGgufBackend> Fake = MakeShared<FFakeGgufBackend>();
			Reg(Fake);
			UPipelineTestFileModel* F = MakeFile(777, FString::ChrN(64, TEXT('c')));
			F->FilePath = TEXT("D:/m/q.gguf");
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, F);
			P->Generation.ContextLength = 1024;
			UPipelineTestGenerationPipeline* Pipe = Track(NewObject<UPipelineTestGenerationPipeline>(GetTransientPackage()));
			const FPipelineResult R = Pipe->InitializeFromPackage(P);
			TestEqual(TEXT("PrepareModelFromSource calls"), Fake->PrepareCalls, 1);
			TestTrue(TEXT("source format GGUF"), Fake->LastSource.Format == EPipelineModelFormat::GGUF);
			TestEqual(TEXT("source FilePath"), Fake->LastSource.FilePath, FString(TEXT("D:/m/q.gguf")));
			TestEqual(TEXT("source size"), Fake->LastSource.FileSizeBytes, (int64)777);
			TestEqual(TEXT("source sha"), Fake->LastSource.FileSha256, FString::ChrN(64, TEXT('c')));
			TestEqual(TEXT("source ContextLength"), Fake->LastSource.Generation.ContextLength, 1024);
			// The fake fails the prepare; that failure must come back as a result, not a crash.
			TestFalse(TEXT("fake prepare failed -> not successful"), R.bSuccess);
		});

		It("explicit-backend overload also routes through PrepareModelFromSource", [this]()
		{
			FFakeGgufBackend Fake;
			UPipelineModelPackage* P = MakePackage(EPipelineTask::TextGeneration, false, MakeFile());
			UPipelineTestGenerationPipeline* Pipe = Track(NewObject<UPipelineTestGenerationPipeline>(GetTransientPackage()));
			const FPipelineResult R = Pipe->InitializeFromPackage(P, Fake);
			TestEqual(TEXT("PrepareModelFromSource calls"), Fake.PrepareCalls, 1);
			TestFalse(TEXT("fake prepare failed -> not successful"), R.bSuccess);
		});
	});

	Describe("Backend defaults", [this]()
	{
		It("default PrepareModelFromSource with GGUF -> nullptr + RuntimeUnavailable", [this]()
		{
			FPipelineNNEBackend Backend;
			FPipelineModelSource S;
			S.Format = EPipelineModelFormat::GGUF;
			S.FilePath = TEXT("C:/none.gguf");
			FPipelineResult R;
			TSharedPtr<IPipelineModel> M = Backend.PrepareModelFromSource(S, EPipelineDevice::CPU, R);
			TestFalse(TEXT("model must be null"), M.IsValid());
			TestFalse(TEXT("result must be a failure"), R.bSuccess);
			TestTrue(FString::Printf(TEXT("RuntimeUnavailable expected, got %d"), (int32)R.Error), R.Error == EPipelineError::RuntimeUnavailable);
			TestFalse(TEXT("NNE backend does not support GGUF"), Backend.SupportsFormat(EPipelineModelFormat::GGUF));
			TestTrue(TEXT("NNE backend supports NNE"), Backend.SupportsFormat(EPipelineModelFormat::NNE));
		});

		It("default PrepareModelFromSource with NNE gives the same output as PrepareModel (affine_dynamic, [0,-0,1,-1] -> [1,1,3,-1])", [this]()
		{
			using namespace UE::ModelPipelines::Tests;
			FString Err;
			UNNEModelData* Data = LoadTestModel(TEXT("affine_dynamic.onnx"), Err);
			if (!Data) { AddError(Err); return; }
			TStrongObjectPtr<UNNEModelData> Hold(Data);
			FPipelineNNEBackend Backend;
			FPipelineModelSource S;
			S.Format = EPipelineModelFormat::NNE;
			S.NNEModel = Data;
			FPipelineResult R1, R2;
			TSharedPtr<IPipelineModel> ViaSource = Backend.PrepareModelFromSource(S, EPipelineDevice::CPU, R1);
			TSharedPtr<IPipelineModel> Direct = Backend.PrepareModel(Data, EPipelineDevice::CPU, R2);
			if (!ViaSource.IsValid() || !Direct.IsValid())
			{
				AddError(FString::Printf(TEXT("prepare failed: source '%s', direct '%s'"), *R1.Message, *R2.Message));
				return;
			}
			TArray<FPipelineTensor> In = { FPipelineTensor::FromFloat(TArray<int64>{1, 4}, TArray<float>{0.f, -0.f, 1.f, -1.f}) };
			TArray<FPipelineTensor> O1, O2;
			const FPipelineResult Run1 = ViaSource->Run(In, O1);
			const FPipelineResult Run2 = Direct->Run(In, O2);
			TestTrue(FString::Printf(TEXT("source run ok: %s"), *Run1.Message), Run1.bSuccess);
			TestTrue(FString::Printf(TEXT("direct run ok: %s"), *Run2.Message), Run2.bSuccess);
			if (O1.Num() == 1 && O2.Num() == 1 && O1[0].GetFloatData().Num() == 4 && O2[0].GetFloatData().Num() == 4)
			{
				const float E[4] = {1.f, 1.f, 3.f, -1.f};
				for (int32 i = 0; i < 4; ++i)
				{
					const float A = O1[0].GetFloatData()[i];
					const float B = O2[0].GetFloatData()[i];
					TestTrue(FString::Printf(TEXT("y[%d] via source %g, direct %g, expected %g"), i, A, B, E[i]),
						FMath::IsNearlyEqual(A, E[i], 1e-5f) && A == B);
				}
			}
			else
			{
				AddError(TEXT("unexpected output count or size from one of the models"));
			}
		});

		It("IPipelineModel defaults: CreateSession -> nullptr + InvalidArgument; GetExternalMemoryBytes 0; CancelInitialize no-op", [this]()
		{
			FBareModel Model;
			FPipelineResult R;
			FPipelineSessionRequest Req;
			Req.Kind = TEXT("TextGeneration");
			TSharedPtr<IPipelineSession> S = Model.CreateSession(Req, R);
			TestFalse(TEXT("session must be null"), S.IsValid());
			TestFalse(TEXT("result must be a failure"), R.bSuccess);
			TestTrue(FString::Printf(TEXT("InvalidArgument expected, got %d '%s'"), (int32)R.Error, *R.Message), R.Error == EPipelineError::InvalidArgument);
			TestEqual(TEXT("GetExternalMemoryBytes"), Model.GetExternalMemoryBytes(), (int64)0);
			Model.CancelInitialize();
			TestTrue(TEXT("still initialized after CancelInitialize"), Model.IsInitialized());
		});
	});
}

#endif
