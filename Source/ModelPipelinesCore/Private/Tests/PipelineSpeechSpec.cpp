// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
// WP9S.3 visible specs: SpeechToText package fields, validation (19/72/73), encoder and decoder task I/O, UPipelineBase decoder model.
// Written from the brief's contract (C72a/C72c) only.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Math/NumericLimits.h"
#include <limits>
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "NNEModelData.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "PipelineValidation.h"
#include "PipelineModelPackage.h"
#include "PipelineTaskIO.h"
#include "PipelineBase.h"
#include "PipelineBackend.h"
#include "PipelineTestData.h"
#include "PipelineBaseTestPipeline.h"

namespace PipelineSpeechSpecPrivate
{
	using EType = EPipelineTensorType;
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;

	// Numeric values from contract C72, so this file does not depend on the enumerator names.
	static const ECode NoDecoderModel = (ECode)19;
	static const ECode UnexpectedDecoderModel = (ECode)72;
	static const ECode InvalidSpeechSettings = (ECode)73;
	static const ECode NoTokenizer = (ECode)4;
	static const ECode MissingInput = (ECode)20;
	static const ECode UnexpectedInput = (ECode)21;
	static const ECode InputTypeMismatch = (ECode)22;
	static const ECode InputRankMismatch = (ECode)23;
	static const ECode InputDimMismatch = (ECode)24;
	static const ECode MissingOutput = (ECode)25;
	static const ECode OutputDimMismatch = (ECode)28;
	static const ECode OutputNameFallback = (ECode)30;

	static FPipelineTensorDesc D(const TCHAR* Name, EType Type, std::initializer_list<int64> Shape)
	{
		FPipelineTensorDesc Desc;
		Desc.Name = Name;
		Desc.Type = Type;
		Desc.bTypeSupported = true;
		Desc.Shape = TArray<int64>(Shape);
		return Desc;
	}

	static int32 CountCode(const FPipelineValidationReport& R, ECode Code)
	{
		int32 N = 0;
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code) { ++N; }
		}
		return N;
	}

	static const FPipelineValidationMessage* FindCode(const FPipelineValidationReport& R, ECode Code)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code) { return &M; }
		}
		return nullptr;
	}

	/** Settings matching the tiny test models. */
	static FPipelineSpeechSettings TinySettings()
	{
		FPipelineSpeechSettings S;
		S.NumMelBins = 8;
		S.BlankTokenId = 5;
		S.Durations = {0, 1, 2};
		return S;
	}

	static TArray<FPipelineTensorDesc> EncIn() { return {D(TEXT("input_features"), EType::Float32, {1, -1, 8})}; }
	static TArray<FPipelineTensorDesc> EncOut() { return {D(TEXT("encoder_projected"), EType::Float32, {1, -1, 4})}; }

	static TArray<FPipelineTensorDesc> DecIn(EType TokenType = EType::Float32)
	{
		// Callers pass Int64 for the good model.
		return {D(TEXT("encoder_frame"), EType::Float32, {1, 4}), D(TEXT("token"), TokenType, {1}),
			D(TEXT("state_h"), EType::Float32, {2, 1, 4}), D(TEXT("state_c"), EType::Float32, {2, 1, 4})};
	}

	static TArray<FPipelineTensorDesc> DecOut()
	{
		return {D(TEXT("token_logits"), EType::Float32, {1, 6}), D(TEXT("duration_logits"), EType::Float32, {1, 3}),
			D(TEXT("state_h_out"), EType::Float32, {2, 1, 4}), D(TEXT("state_c_out"), EType::Float32, {2, 1, 4})};
	}

	struct FEncResult
	{
		bool bOk = false;
		FPipelineTaskBindings B;
		FPipelineValidationReport R;
	};

	static FEncResult ResolveEnc(const FPipelineSpeechSettings& Speech, const TArray<FPipelineTensorDesc>& In, const TArray<FPipelineTensorDesc>& Out)
	{
		FEncResult Res;
		Res.bOk = ResolvePipelineTaskIO(EPipelineTask::SpeechToText, FPipelinePreprocessSettings(), FPipelinePostprocessSettings(), Speech, 0,
			In, Out, Res.B, Res.R);
		return Res;
	}

	struct FDecResult
	{
		bool bOk = false;
		FPipelineSpeechDecoderBindings B;
		FPipelineValidationReport R;
	};

	static FDecResult ResolveDec(const FPipelineSpeechSettings& Speech, int32 EncoderDim, const TArray<FPipelineTensorDesc>& In, const TArray<FPipelineTensorDesc>& Out)
	{
		FDecResult Res;
		// Poison the bindings: the function must reset them to -1 at entry.
		Res.B.EncoderFrame = Res.B.Token = Res.B.StateH = Res.B.StateC = 99;
		Res.B.TokenLogits = Res.B.DurationLogits = Res.B.StateHOut = Res.B.StateCOut = 99;
		Res.bOk = ResolveSpeechDecoderIO(Speech, EncoderDim, In, Out, Res.B, Res.R);
		return Res;
	}

	static TArray<FPipelineTensorDesc> GoodDecIn()
	{
		TArray<FPipelineTensorDesc> In = DecIn(EType::Int64);
		return In;
	}
}

BEGIN_DEFINE_SPEC(FPipelineSpeechSpec, "Pipelines.Core.Speech",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	TArray<TStrongObjectPtr<UObject>> Keep;

	template <typename T>
	T* Track(T* Obj)
	{
		Keep.Add(TStrongObjectPtr<UObject>(Obj));
		return Obj;
	}

	/** A valid SpeechToText package with placeholder (empty) model objects; fine for ValidateSettings. */
	UPipelineModelPackage* MakeSpeechPackage(bool bDecoder = true)
	{
		UPipelineModelPackage* P = Track(NewObject<UPipelineModelPackage>(GetTransientPackage()));
		P->Task = EPipelineTask::SpeechToText;
		P->Model = Track(NewObject<UNNEModelData>(GetTransientPackage()));
		if (bDecoder) { P->DecoderModel = Track(NewObject<UNNEModelData>(GetTransientPackage())); }
		P->TokenizerJson = TEXT("x");
		P->License.SpdxId = TEXT("Apache-2.0");
		P->License.SourceUrl = TEXT("https://example.com/m");
		P->License.Attribution = TEXT("Example Authors");
		return P;
	}

	FPipelineValidationReport Validate(const UPipelineModelPackage* P, bool& bOk) const
	{
		FPipelineValidationReport R;
		bOk = P->ValidateSettings(R);
		return R;
	}

	/** Expects exactly one 73 Error (and no other error) for a package whose Speech was edited by Edit. */
	void ExpectInvalidSpeech(const FString& What, TFunctionRef<void(FPipelineSpeechSettings&)> Edit)
	{
		using namespace PipelineSpeechSpecPrivate;
		UPipelineModelPackage* P = MakeSpeechPackage();
		Edit(P->Speech);
		bool bOk = true;
		const FPipelineValidationReport R = Validate(P, bOk);
		TestFalse(What + ": ValidateSettings returns false", bOk);
		TestEqual(What + ": exactly one InvalidSpeechSettings(73)", CountCode(R, InvalidSpeechSettings), 1);
		TestEqual(What + ": exactly one error in total", R.NumErrors(), 1);
		const FPipelineValidationMessage* M = FindCode(R, InvalidSpeechSettings);
		if (M)
		{
			TestTrue(What + ": severity Error", M->Severity == ESev::Error);
		}
	}

	/** Loads the tiny model; on failure reports the path and the command and returns null. */
	UNNEModelData* LoadTiny(const FString& File)
	{
		FString Err;
		UNNEModelData* Data = UE::ModelPipelines::Tests::LoadTestModel(File, Err);
		if (!Data)
		{
			AddError(FString::Printf(TEXT("Cannot load %s: %s. Generate it with: "
				"D:/Micromamba/micromambaenv/envs/pipelines/python.exe \"D:/Game Dev/Pipelines-UE/ModelPipelines-UE/Tools/make_test_models.py\""), *File, *Err));
			return nullptr;
		}
		return Track(Data);
	}

END_DEFINE_SPEC(FPipelineSpeechSpec)

void FPipelineSpeechSpec::Define()
{
	using namespace PipelineSpeechSpecPrivate;

	AfterEach([this]() { Keep.Reset(); });

	Describe("Defaults (acceptance 1)", [this]()
	{
		It("new package has the C72a defaults", [this]()
		{
			UPipelineModelPackage* P = Track(NewObject<UPipelineModelPackage>(GetTransientPackage()));
			const FPipelineSpeechSettings& S = P->Speech;
			TestNull(TEXT("DecoderModel defaults to null"), P->DecoderModel.Get());
			TestEqual(TEXT("SampleRate"), S.SampleRate, 16000);
			TestEqual(TEXT("NumMelBins"), S.NumMelBins, 128);
			TestEqual(TEXT("FftSize"), S.FftSize, 512);
			TestEqual(TEXT("WindowLength"), S.WindowLength, 400);
			TestEqual(TEXT("HopLength"), S.HopLength, 160);
			TestEqual(TEXT("Preemphasis"), S.Preemphasis, 0.97f);
			TestEqual(TEXT("SubsamplingFactor"), S.SubsamplingFactor, 8);
			TestEqual(TEXT("BlankTokenId"), S.BlankTokenId, 8192);
			TestEqual(TEXT("Durations"), S.Durations, TArray<int32>({0, 1, 2, 3, 4}));
			TestEqual(TEXT("MaxSymbolsPerStep"), S.MaxSymbolsPerStep, 10);
			TestEqual(TEXT("SilenceThresholdDb"), S.SilenceThresholdDb, -60.f);
		});

		It("a SpeechToText package with both models, a tokenizer and defaults has no error", [this]()
		{
			bool bOk = false;
			const FPipelineValidationReport R = Validate(MakeSpeechPackage(), bOk);
			TestTrue(TEXT("ValidateSettings expected true, report: ") + R.ToString(), bOk);
			TestEqual(TEXT("no errors: ") + R.ToString(), R.NumErrors(), 0);
			TestEqual(TEXT("no messages at all (license is filled): ") + R.ToString(), R.Messages.Num(), 0);
		});

		It("LexToString names the new codes and task helpers stay false", [this]()
		{
			TestEqual(TEXT("19"), FString(LexToString(NoDecoderModel)), FString(TEXT("NoDecoderModel")));
			TestEqual(TEXT("72"), FString(LexToString(UnexpectedDecoderModel)), FString(TEXT("UnexpectedDecoderModel")));
			TestEqual(TEXT("73"), FString(LexToString(InvalidSpeechSettings)), FString(TEXT("InvalidSpeechSettings")));
			TestFalse(TEXT("TaskUsesLabels(SpeechToText)"), TaskUsesLabels(EPipelineTask::SpeechToText));
			TestFalse(TEXT("IsTextTask(SpeechToText)"), IsTextTask(EPipelineTask::SpeechToText));
		});
	});

	Describe("Package validation", [this]()
	{
		It("NoDecoderModel: exactly one 19 Error and nothing else (acceptance 2)", [this]()
		{
			bool bOk = true;
			const FPipelineValidationReport R = Validate(MakeSpeechPackage(false), bOk);
			TestFalse(TEXT("ValidateSettings expected false"), bOk);
			TestEqual(TEXT("exactly one 19: ") + R.ToString(), CountCode(R, NoDecoderModel), 1);
			TestEqual(TEXT("exactly one message in total: ") + R.ToString(), R.Messages.Num(), 1);
			const FPipelineValidationMessage* M = FindCode(R, NoDecoderModel);
			if (M)
			{
				TestTrue(TEXT("severity Error"), M->Severity == ESev::Error);
				TestTrue(TEXT("message mentions DecoderModel: ") + M->Message, M->Message.Contains(TEXT("DecoderModel")));
			}
		});

		It("UnexpectedDecoderModel: TextEmbedding + DecoderModel gives one 72 Warning, still valid (acceptance 3)", [this]()
		{
			UPipelineModelPackage* P = Track(NewObject<UPipelineModelPackage>(GetTransientPackage()));
			P->Task = EPipelineTask::TextEmbedding;
			P->Model = Track(NewObject<UNNEModelData>(GetTransientPackage()));
			P->TokenizerJson = TEXT("{}");
			P->License.SpdxId = TEXT("MIT");
			P->License.SourceUrl = TEXT("https://example.com/m");
			P->License.Attribution = TEXT("Example Authors");
			bool bOk = false;
			FPipelineValidationReport Base = Validate(P, bOk);
			TestTrue(TEXT("control package (no DecoderModel) valid: ") + Base.ToString(), bOk);
			TestEqual(TEXT("control has no messages: ") + Base.ToString(), Base.Messages.Num(), 0);

			P->DecoderModel = Track(NewObject<UNNEModelData>(GetTransientPackage()));
			const FPipelineValidationReport R = Validate(P, bOk);
			TestTrue(TEXT("ValidateSettings expected true (warning only): ") + R.ToString(), bOk);
			TestEqual(TEXT("exactly one 72: ") + R.ToString(), CountCode(R, UnexpectedDecoderModel), 1);
			TestEqual(TEXT("exactly one message: ") + R.ToString(), R.Messages.Num(), 1);
			TestEqual(TEXT("no errors"), R.NumErrors(), 0);
			const FPipelineValidationMessage* M = FindCode(R, UnexpectedDecoderModel);
			if (M)
			{
				TestTrue(TEXT("severity Warning"), M->Severity == ESev::Warning);
			}
		});

		It("NoTokenizer: SpeechToText with empty TokenizerJson gives a 4 Error naming SpeechToText (acceptance 4)", [this]()
		{
			UPipelineModelPackage* P = MakeSpeechPackage();
			P->TokenizerJson.Reset();
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("ValidateSettings expected false"), bOk);
			TestEqual(TEXT("exactly one 4: ") + R.ToString(), CountCode(R, NoTokenizer), 1);
			TestEqual(TEXT("one error in total: ") + R.ToString(), R.NumErrors(), 1);
			const FPipelineValidationMessage* M = FindCode(R, NoTokenizer);
			if (M)
			{
				TestTrue(TEXT("severity Error"), M->Severity == ESev::Error);
				TestTrue(TEXT("message names SpeechToText: ") + M->Message, M->Message.Contains(TEXT("SpeechToText")));
			}
		});
	});

	Describe("Speech settings validation (acceptance 5)", [this]()
	{
		It("HopLength 0 + Durations {0} + Preemphasis 1.0 give ONE 73 naming all three", [this]()
		{
			UPipelineModelPackage* P = MakeSpeechPackage();
			P->Speech.HopLength = 0;
			P->Speech.Durations = {0};
			P->Speech.Preemphasis = 1.0f;
			bool bOk = true;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestFalse(TEXT("ValidateSettings expected false"), bOk);
			TestEqual(TEXT("exactly one 73 (not one per problem): ") + R.ToString(), CountCode(R, InvalidSpeechSettings), 1);
			TestEqual(TEXT("one error in total"), R.NumErrors(), 1);
			const FPipelineValidationMessage* M = FindCode(R, InvalidSpeechSettings);
			if (M)
			{
				TestTrue(TEXT("severity Error"), M->Severity == ESev::Error);
				TestTrue(TEXT("names HopLength: ") + M->Message, M->Message.Contains(TEXT("HopLength")));
				TestTrue(TEXT("names Durations: ") + M->Message, M->Message.Contains(TEXT("Durations")));
				TestTrue(TEXT("names Preemphasis: ") + M->Message, M->Message.Contains(TEXT("Preemphasis")));
			}
		});

		It("WindowLength 600 > FftSize 512", [this]()
		{
			ExpectInvalidSpeech(TEXT("WindowLength>FftSize"), [](FPipelineSpeechSettings& S) { S.WindowLength = 600; });
		});

		It("SilenceThresholdDb +3 is invalid, -100 (gate disabled) and 0 are valid", [this]()
		{
			ExpectInvalidSpeech(TEXT("SilenceThresholdDb +3"), [](FPipelineSpeechSettings& S) { S.SilenceThresholdDb = 3.f; });
			for (const float Db : {-100.f, 0.f})
			{
				UPipelineModelPackage* P = MakeSpeechPackage();
				P->Speech.SilenceThresholdDb = Db;
				bool bOk = false;
				const FPipelineValidationReport R = Validate(P, bOk);
				TestTrue(FString::Printf(TEXT("SilenceThresholdDb %g valid: %s"), Db, *R.ToString()), bOk);
				TestEqual(FString::Printf(TEXT("SilenceThresholdDb %g: no 73"), Db), CountCode(R, InvalidSpeechSettings), 0);
			}
		});

		It("non-finite Preemphasis and SilenceThresholdDb", [this]()
		{
			ExpectInvalidSpeech(TEXT("Preemphasis NaN"), [](FPipelineSpeechSettings& S) { S.Preemphasis = std::numeric_limits<float>::quiet_NaN(); });
			ExpectInvalidSpeech(TEXT("Preemphasis +Inf"), [](FPipelineSpeechSettings& S) { S.Preemphasis = std::numeric_limits<float>::infinity(); });
			ExpectInvalidSpeech(TEXT("SilenceThresholdDb NaN"), [](FPipelineSpeechSettings& S) { S.SilenceThresholdDb = std::numeric_limits<float>::quiet_NaN(); });
			ExpectInvalidSpeech(TEXT("SilenceThresholdDb -Inf"), [](FPipelineSpeechSettings& S) { S.SilenceThresholdDb = -std::numeric_limits<float>::infinity(); });
		});

		It("Preemphasis range is [0,1): 0 valid, -0.1 and 1.0 invalid", [this]()
		{
			UPipelineModelPackage* P = MakeSpeechPackage();
			P->Speech.Preemphasis = 0.f;
			bool bOk = false;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestTrue(TEXT("Preemphasis 0 valid: ") + R.ToString(), bOk);
			ExpectInvalidSpeech(TEXT("Preemphasis -0.1"), [](FPipelineSpeechSettings& S) { S.Preemphasis = -0.1f; });
			ExpectInvalidSpeech(TEXT("Preemphasis 1.0"), [](FPipelineSpeechSettings& S) { S.Preemphasis = 1.0f; });
		});

		It("each integer field <= 0 is invalid", [this]()
		{
			ExpectInvalidSpeech(TEXT("SampleRate 0"), [](FPipelineSpeechSettings& S) { S.SampleRate = 0; });
			ExpectInvalidSpeech(TEXT("NumMelBins -1"), [](FPipelineSpeechSettings& S) { S.NumMelBins = -1; });
			ExpectInvalidSpeech(TEXT("FftSize 0"), [](FPipelineSpeechSettings& S) { S.FftSize = 0; S.WindowLength = 0; });
			ExpectInvalidSpeech(TEXT("WindowLength 0"), [](FPipelineSpeechSettings& S) { S.WindowLength = 0; });
			ExpectInvalidSpeech(TEXT("HopLength -5"), [](FPipelineSpeechSettings& S) { S.HopLength = -5; });
			ExpectInvalidSpeech(TEXT("SubsamplingFactor 0"), [](FPipelineSpeechSettings& S) { S.SubsamplingFactor = 0; });
			ExpectInvalidSpeech(TEXT("MaxSymbolsPerStep 0"), [](FPipelineSpeechSettings& S) { S.MaxSymbolsPerStep = 0; });
		});

		It("BlankTokenId < 0 and bad Durations are invalid", [this]()
		{
			ExpectInvalidSpeech(TEXT("BlankTokenId -1"), [](FPipelineSpeechSettings& S) { S.BlankTokenId = -1; });
			ExpectInvalidSpeech(TEXT("Durations empty"), [](FPipelineSpeechSettings& S) { S.Durations.Reset(); });
			ExpectInvalidSpeech(TEXT("Durations has negative"), [](FPipelineSpeechSettings& S) { S.Durations = {0, -1, 2}; });
			ExpectInvalidSpeech(TEXT("Durations no value >= 1"), [](FPipelineSpeechSettings& S) { S.Durations = {0, 0}; });
			// BlankTokenId 0 and Durations {1} are the boundaries and must stay valid.
			UPipelineModelPackage* P = MakeSpeechPackage();
			P->Speech.BlankTokenId = 0;
			P->Speech.Durations = {1};
			bool bOk = false;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestTrue(TEXT("BlankTokenId 0 + Durations {1} valid: ") + R.ToString(), bOk);
		});

		It("speech settings are not checked for other tasks", [this]()
		{
			UPipelineModelPackage* P = Track(NewObject<UPipelineModelPackage>(GetTransientPackage()));
			P->Task = EPipelineTask::TextEmbedding;
			P->Model = Track(NewObject<UNNEModelData>(GetTransientPackage()));
			P->TokenizerJson = TEXT("{}");
			P->License.SpdxId = TEXT("MIT");
			P->License.SourceUrl = TEXT("https://example.com/m");
			P->License.Attribution = TEXT("Example Authors");
			P->Speech.HopLength = 0;
			bool bOk = false;
			const FPipelineValidationReport R = Validate(P, bOk);
			TestTrue(TEXT("TextEmbedding with garbage Speech still valid: ") + R.ToString(), bOk);
			TestEqual(TEXT("no 73"), CountCode(R, InvalidSpeechSettings), 0);
		});
	});

	Describe("Encoder task I/O (acceptance 6)", [this]()
	{
		It("matching descs bind input_features and encoder_projected", [this]()
		{
			const FEncResult Res = ResolveEnc(TinySettings(), EncIn(), EncOut());
			TestTrue(TEXT("returns true: ") + Res.R.ToString(), Res.bOk);
			TestEqual(TEXT("InputFeatures"), Res.B.InputFeatures, 0);
			TestEqual(TEXT("MainOutput"), Res.B.MainOutput, 0);
			TestEqual(TEXT("no messages: ") + Res.R.ToString(), Res.R.Messages.Num(), 0);
		});

		It("bindings follow desc order, not position 0", [this]()
		{
			// An extra OUTPUT (not input) is not an error; the bound output must be the named one at index 1.
			TArray<FPipelineTensorDesc> Out = {D(TEXT("other"), EType::Float32, {1, 3}), D(TEXT("encoder_projected"), EType::Float32, {-1, -1, -1})};
			const FEncResult Res = ResolveEnc(TinySettings(), EncIn(), Out);
			TestTrue(TEXT("returns true: ") + Res.R.ToString(), Res.bOk);
			TestEqual(TEXT("MainOutput index of encoder_projected"), Res.B.MainOutput, 1);
			TestEqual(TEXT("no fallback warning"), CountCode(Res.R, OutputNameFallback), 0);
		});

		It("NumMelBins 128 vs input dim 8 gives InputDimMismatch(24)", [this]()
		{
			FPipelineSpeechSettings S = TinySettings();
			S.NumMelBins = 128;
			const FEncResult Res = ResolveEnc(S, EncIn(), EncOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 24: ") + Res.R.ToString(), CountCode(Res.R, InputDimMismatch), 1);
			const FPipelineValidationMessage* M = FindCode(Res.R, InputDimMismatch);
			if (M)
			{
				TestTrue(TEXT("severity Error"), M->Severity == ESev::Error);
				TestTrue(TEXT("names the tensor: ") + M->Message, M->Message.Contains(TEXT("input_features")));
			}
		});

		It("dynamic mel dim (-1) matches any NumMelBins", [this]()
		{
			TArray<FPipelineTensorDesc> In = {D(TEXT("input_features"), EType::Float32, {-1, -1, -1})};
			FPipelineSpeechSettings S;
			S.NumMelBins = 128;
			const FEncResult Res = ResolveEnc(S, In, EncOut());
			TestTrue(TEXT("returns true: ") + Res.R.ToString(), Res.bOk);
		});

		It("an extra input gives UnexpectedInput(21)", [this]()
		{
			TArray<FPipelineTensorDesc> In = EncIn();
			In.Add(D(TEXT("length"), EType::Int64, {1}));
			const FEncResult Res = ResolveEnc(TinySettings(), In, EncOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 21: ") + Res.R.ToString(), CountCode(Res.R, UnexpectedInput), 1);
			const FPipelineValidationMessage* M = FindCode(Res.R, UnexpectedInput);
			if (M) { TestTrue(TEXT("severity Error"), M->Severity == ESev::Error); }
		});

		It("an output named x (only float rank-3) is bound with a Warning 30", [this]()
		{
			TArray<FPipelineTensorDesc> Out = {D(TEXT("x"), EType::Float32, {1, -1, 4})};
			const FEncResult Res = ResolveEnc(TinySettings(), EncIn(), Out);
			TestTrue(TEXT("returns true (warning only): ") + Res.R.ToString(), Res.bOk);
			TestEqual(TEXT("MainOutput bound to x"), Res.B.MainOutput, 0);
			TestEqual(TEXT("exactly one 30: ") + Res.R.ToString(), CountCode(Res.R, OutputNameFallback), 1);
			TestEqual(TEXT("no errors"), Res.R.NumErrors(), 0);
			const FPipelineValidationMessage* M = FindCode(Res.R, OutputNameFallback);
			if (M) { TestTrue(TEXT("severity Warning"), M->Severity == ESev::Warning); }
		});

		It("int64 input_features gives InputTypeMismatch(22)", [this]()
		{
			TArray<FPipelineTensorDesc> In = {D(TEXT("input_features"), EType::Int64, {1, -1, 8})};
			const FEncResult Res = ResolveEnc(TinySettings(), In, EncOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 22: ") + Res.R.ToString(), CountCode(Res.R, InputTypeMismatch), 1);
		});

		It("rank-2 input_features gives InputRankMismatch(23)", [this]()
		{
			TArray<FPipelineTensorDesc> In = {D(TEXT("input_features"), EType::Float32, {-1, 8})};
			const FEncResult Res = ResolveEnc(TinySettings(), In, EncOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 23: ") + Res.R.ToString(), CountCode(Res.R, InputRankMismatch), 1);
		});

		It("batch dim 2 gives InputDimMismatch(24)", [this]()
		{
			TArray<FPipelineTensorDesc> In = {D(TEXT("input_features"), EType::Float32, {2, -1, 8})};
			const FEncResult Res = ResolveEnc(TinySettings(), In, EncOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 24: ") + Res.R.ToString(), CountCode(Res.R, InputDimMismatch), 1);
		});

		It("missing input_features gives MissingInput(20)", [this]()
		{
			const FEncResult Res = ResolveEnc(TinySettings(), {}, EncOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 20: ") + Res.R.ToString(), CountCode(Res.R, MissingInput), 1);
		});

		It("no output gives MissingOutput(25)", [this]()
		{
			const FEncResult Res = ResolveEnc(TinySettings(), EncIn(), {});
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 25: ") + Res.R.ToString(), CountCode(Res.R, MissingOutput), 1);
			TestEqual(TEXT("no fallback warning"), CountCode(Res.R, OutputNameFallback), 0);
		});

		It("a rank-2 output named x is not a fallback candidate: MissingOutput(25)", [this]()
		{
			TArray<FPipelineTensorDesc> Out = {D(TEXT("x"), EType::Float32, {1, 4})};
			const FEncResult Res = ResolveEnc(TinySettings(), EncIn(), Out);
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 25: ") + Res.R.ToString(), CountCode(Res.R, MissingOutput), 1);
		});

		It("the old overload still works for other tasks (forwards with default Speech)", [this]()
		{
			TArray<FPipelineTensorDesc> In = {D(TEXT("pixel_values"), EType::Float32, {1, 3, 224, 224})};
			TArray<FPipelineTensorDesc> Out = {D(TEXT("logits"), EType::Float32, {1, 2})};
			FPipelinePreprocessSettings S;
			S.ImageWidth = 224;
			S.ImageHeight = 224;
			FPipelineTaskBindings B;
			FPipelineValidationReport R;
			const bool bOk = ResolvePipelineTaskIO(EPipelineTask::ImageClassification, S, FPipelinePostprocessSettings(), 2, In, Out, B, R);
			TestTrue(TEXT("old overload returns true: ") + R.ToString(), bOk);
		});
	});

	Describe("Decoder task I/O (acceptance 7)", [this]()
	{
		It("tiny decoder descs bind all 8 slots, in any desc order", [this]()
		{
			const FPipelineSpeechSettings S = TinySettings();
			const TArray<FPipelineTensorDesc> In = GoodDecIn();
			const TArray<FPipelineTensorDesc> Out = DecOut();

			auto Check = [this](const FString& What, const FDecResult& Res, const TArray<FPipelineTensorDesc>& I, const TArray<FPipelineTensorDesc>& O)
			{
				TestTrue(What + ": returns true: " + Res.R.ToString(), Res.bOk);
				TestEqual(What + ": no messages: " + Res.R.ToString(), Res.R.Messages.Num(), 0);
				auto Named = [&](const TArray<FPipelineTensorDesc>& Descs, int32 Idx, const TCHAR* Expected)
				{
					if (!Descs.IsValidIndex(Idx))
					{
						AddError(FString::Printf(TEXT("%s: binding for %s is %d (out of range)"), *What, Expected, Idx));
						return;
					}
					TestEqual(FString::Printf(TEXT("%s: binding points at %s"), *What, Expected), Descs[Idx].Name, FString(Expected));
				};
				Named(I, Res.B.EncoderFrame, TEXT("encoder_frame"));
				Named(I, Res.B.Token, TEXT("token"));
				Named(I, Res.B.StateH, TEXT("state_h"));
				Named(I, Res.B.StateC, TEXT("state_c"));
				Named(O, Res.B.TokenLogits, TEXT("token_logits"));
				Named(O, Res.B.DurationLogits, TEXT("duration_logits"));
				Named(O, Res.B.StateHOut, TEXT("state_h_out"));
				Named(O, Res.B.StateCOut, TEXT("state_c_out"));
			};

			Check(TEXT("natural order"), ResolveDec(S, 4, In, Out), In, Out);

			TArray<FPipelineTensorDesc> InS = {In[3], In[1], In[0], In[2]};
			TArray<FPipelineTensorDesc> OutS = {Out[2], Out[0], Out[3], Out[1]};
			Check(TEXT("shuffled"), ResolveDec(S, 4, InS, OutS), InS, OutS);
		});

		It("real tiny decoder model's descs resolve cleanly", [this]()
		{
			UNNEModelData* Dec = LoadTiny(TEXT("tiny_stt_decoder_joint.onnx"));
			UNNEModelData* Enc = LoadTiny(TEXT("tiny_stt_encoder.onnx"));
			if (!Dec || !Enc) { return; }
			UPipelineModelPackage* P = MakeSpeechPackage();
			P->Model = Enc;
			P->DecoderModel = Dec;
			P->Speech = TinySettings();
			UPipelineSpeechTestPipeline* Pipe = Track(NewObject<UPipelineSpeechTestPipeline>(GetTransientPackage()));
			const FPipelineResult Init = Pipe->InitializeFromPackage(P);
			TestTrue(FString::Printf(TEXT("init with the tiny models (error %d): %s"), (int32)Init.Error, *Init.Message), Init.bSuccess);
			if (!Init.bSuccess || !Pipe->PeekDecoderModel().IsValid()) { return; }
			const FPipelineResult DI = Pipe->PeekDecoderModel()->Initialize();
			TestTrue(FString::Printf(TEXT("decoder Initialize: %s"), *DI.Message), DI.bSuccess);
			const FDecResult Res = ResolveDec(P->Speech, 4, Pipe->PeekDecoderModel()->GetInputDescs(), Pipe->PeekDecoderModel()->GetOutputDescs());
			TestTrue(TEXT("descs of the loaded decoder resolve: ") + Res.R.ToString(), Res.bOk);
			TestEqual(TEXT("no messages: ") + Res.R.ToString(), Res.R.Messages.Num(), 0);
		});

		It("bad decoder (token float32) gives InputTypeMismatch(22) naming token", [this]()
		{
			const FDecResult Res = ResolveDec(TinySettings(), 4, DecIn(EType::Float32), DecOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 22: ") + Res.R.ToString(), CountCode(Res.R, InputTypeMismatch), 1);
			const FPipelineValidationMessage* M = FindCode(Res.R, InputTypeMismatch);
			if (M)
			{
				TestTrue(TEXT("names token: ") + M->Message, M->Message.Contains(TEXT("token")));
				TestTrue(TEXT("severity Error"), M->Severity == ESev::Error);
			}
			TestEqual(TEXT("only that error"), Res.R.NumErrors(), 1);
		});

		It("BlankTokenId 7 vs token_logits dim 6 gives OutputDimMismatch(28)", [this]()
		{
			FPipelineSpeechSettings S = TinySettings();
			S.BlankTokenId = 7;
			const FDecResult Res = ResolveDec(S, 4, GoodDecIn(), DecOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 28: ") + Res.R.ToString(), CountCode(Res.R, OutputDimMismatch), 1);
			TestEqual(TEXT("only that error"), Res.R.NumErrors(), 1);
		});

		It("Durations {0,1,2,3} vs duration_logits dim 3 gives OutputDimMismatch(28)", [this]()
		{
			FPipelineSpeechSettings S = TinySettings();
			S.Durations = {0, 1, 2, 3};
			const FDecResult Res = ResolveDec(S, 4, GoodDecIn(), DecOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 28: ") + Res.R.ToString(), CountCode(Res.R, OutputDimMismatch), 1);
			TestEqual(TEXT("only that error"), Res.R.NumErrors(), 1);
		});

		It("missing state_c gives MissingInput(20)", [this]()
		{
			TArray<FPipelineTensorDesc> In = GoodDecIn();
			In.RemoveAll([](const FPipelineTensorDesc& D) { return D.Name == TEXT("state_c"); });
			const FDecResult Res = ResolveDec(TinySettings(), 4, In, DecOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 20: ") + Res.R.ToString(), CountCode(Res.R, MissingInput), 1);
			const FPipelineValidationMessage* M = FindCode(Res.R, MissingInput);
			if (M) { TestTrue(TEXT("names state_c: ") + M->Message, M->Message.Contains(TEXT("state_c"))); }
			TestEqual(TEXT("StateC binding stays -1"), Res.B.StateC, -1);
			TestEqual(TEXT("Token binding still set (not stuck at the poison value)"), Res.B.Token, 1);
		});

		It("EncoderDim 5 vs encoder_frame dim 4 gives InputDimMismatch(24); EncoderDim 0 and -1 are not checked", [this]()
		{
			const FDecResult Bad = ResolveDec(TinySettings(), 5, GoodDecIn(), DecOut());
			TestFalse(TEXT("EncoderDim 5 returns false"), Bad.bOk);
			TestEqual(TEXT("exactly one 24: ") + Bad.R.ToString(), CountCode(Bad.R, InputDimMismatch), 1);
			for (const int32 Dim : {0, -1, 4})
			{
				const FDecResult Ok = ResolveDec(TinySettings(), Dim, GoodDecIn(), DecOut());
				TestTrue(FString::Printf(TEXT("EncoderDim %d returns true: %s"), Dim, *Ok.R.ToString()), Ok.bOk);
			}
		});

		It("state_h_out with different dims than state_h gives OutputDimMismatch(28)", [this]()
		{
			TArray<FPipelineTensorDesc> Out = DecOut();
			Out[2] = D(TEXT("state_h_out"), EType::Float32, {2, 1, 5});
			const FDecResult Res = ResolveDec(TinySettings(), 4, GoodDecIn(), Out);
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 28: ") + Res.R.ToString(), CountCode(Res.R, OutputDimMismatch), 1);
		});

		It("an unknown extra decoder input gives UnexpectedInput(21)", [this]()
		{
			TArray<FPipelineTensorDesc> In = GoodDecIn();
			In.Add(D(TEXT("mystery"), EType::Float32, {1}));
			const FDecResult Res = ResolveDec(TinySettings(), 4, In, DecOut());
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 21: ") + Res.R.ToString(), CountCode(Res.R, UnexpectedInput), 1);
		});

		It("missing token_logits gives MissingOutput(25)", [this]()
		{
			TArray<FPipelineTensorDesc> Out = DecOut();
			Out.RemoveAll([](const FPipelineTensorDesc& D) { return D.Name == TEXT("token_logits"); });
			const FDecResult Res = ResolveDec(TinySettings(), 4, GoodDecIn(), Out);
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("exactly one 25: ") + Res.R.ToString(), CountCode(Res.R, MissingOutput), 1);
		});

		It("empty descs: bindings are reset to -1 and every slot is reported missing", [this]()
		{
			const FDecResult Res = ResolveDec(TinySettings(), 4, {}, {});
			TestFalse(TEXT("returns false"), Res.bOk);
			TestEqual(TEXT("EncoderFrame"), Res.B.EncoderFrame, -1);
			TestEqual(TEXT("Token"), Res.B.Token, -1);
			TestEqual(TEXT("StateH"), Res.B.StateH, -1);
			TestEqual(TEXT("StateC"), Res.B.StateC, -1);
			TestEqual(TEXT("TokenLogits"), Res.B.TokenLogits, -1);
			TestEqual(TEXT("DurationLogits"), Res.B.DurationLogits, -1);
			TestEqual(TEXT("StateHOut"), Res.B.StateHOut, -1);
			TestEqual(TEXT("StateCOut"), Res.B.StateCOut, -1);
			TestEqual(TEXT("4 MissingInput"), CountCode(Res.R, MissingInput), 4);
			TestEqual(TEXT("4 MissingOutput"), CountCode(Res.R, MissingOutput), 4);
		});
	});

	Describe("UPipelineBase with the tiny models (acceptance 8)", [this]()
	{
		It("initializes; decoder model is valid, on CPU even with PreferredDevice GPU; Speech settings are a copy", [this]()
		{
			UNNEModelData* Enc = LoadTiny(TEXT("tiny_stt_encoder.onnx"));
			UNNEModelData* Dec = LoadTiny(TEXT("tiny_stt_decoder_joint.onnx"));
			if (!Enc || !Dec) { return; }

			UPipelineModelPackage* P = MakeSpeechPackage();
			P->Model = Enc;
			P->DecoderModel = Dec;
			P->Speech = TinySettings();
			P->Speech.MaxSymbolsPerStep = 7;
			P->PreferredDevice = EPipelineDevice::GPU;

			UPipelineSpeechTestPipeline* Pipe = Track(NewObject<UPipelineSpeechTestPipeline>(GetTransientPackage()));
			TestFalse(TEXT("GetDecoderModel null before init"), Pipe->PeekDecoderModel().IsValid());
			const FPipelineResult Init = Pipe->InitializeFromPackage(P);
			TestTrue(FString::Printf(TEXT("init (error %d): %s"), (int32)Init.Error, *Init.Message), Init.bSuccess);
			TestTrue(TEXT("IsReady"), Pipe->IsReady());
			if (!Init.bSuccess) { return; }

			const TSharedPtr<IPipelineModel> DecModel = Pipe->PeekDecoderModel();
			TestTrue(TEXT("GetDecoderModel valid"), DecModel.IsValid());
			if (DecModel.IsValid())
			{
				// Models are prepared at init and created lazily (IPipelineModel::Initialize, normally on the executor worker);
				// the tiny graphs are safe to initialize here, as PipelineNNEBackendSpec does.
				const FPipelineResult DI = DecModel->Initialize();
				TestTrue(FString::Printf(TEXT("decoder Initialize: %s"), *DI.Message), DI.bSuccess);
				TestTrue(TEXT("decoder device is CPU although PreferredDevice = GPU"), DecModel->GetDevice() == EPipelineDevice::CPU);
				TestEqual(TEXT("decoder has 4 inputs"), DecModel->GetInputDescs().Num(), 4);
				TestEqual(TEXT("decoder has 4 outputs"), DecModel->GetOutputDescs().Num(), 4);
			}
			const TSharedPtr<IPipelineModel> Main = Pipe->PeekModel();
			TestTrue(TEXT("GetModel (encoder) valid"), Main.IsValid());
			if (Main.IsValid() && DecModel.IsValid())
			{
				TestTrue(TEXT("encoder and decoder are different model objects"), Main.Get() != DecModel.Get());
				const FPipelineResult EI = Main->Initialize();
				TestTrue(FString::Printf(TEXT("encoder Initialize: %s"), *EI.Message), EI.bSuccess);
				TestEqual(TEXT("encoder has 1 input"), Main->GetInputDescs().Num(), 1);
			}

			// Copy semantics: edit the package after init.
			TestEqual(TEXT("MaxSymbolsPerStep snapshot"), Pipe->PeekSpeechSettings().MaxSymbolsPerStep, 7);
			TestEqual(TEXT("BlankTokenId snapshot"), Pipe->PeekSpeechSettings().BlankTokenId, 5);
			TestEqual(TEXT("Durations snapshot"), Pipe->PeekSpeechSettings().Durations, TArray<int32>({0, 1, 2}));
			P->Speech.MaxSymbolsPerStep = 99;
			P->Speech.BlankTokenId = 77;
			P->Speech.Durations = {0, 1};
			TestEqual(TEXT("MaxSymbolsPerStep unchanged after package edit"), Pipe->PeekSpeechSettings().MaxSymbolsPerStep, 7);
			TestEqual(TEXT("BlankTokenId unchanged after package edit"), Pipe->PeekSpeechSettings().BlankTokenId, 5);
			TestEqual(TEXT("Durations unchanged after package edit"), Pipe->PeekSpeechSettings().Durations, TArray<int32>({0, 1, 2}));
		});

		It("initializes on CPU preference too", [this]()
		{
			UNNEModelData* Enc = LoadTiny(TEXT("tiny_stt_encoder.onnx"));
			UNNEModelData* Dec = LoadTiny(TEXT("tiny_stt_decoder_joint.onnx"));
			if (!Enc || !Dec) { return; }
			UPipelineModelPackage* P = MakeSpeechPackage();
			P->Model = Enc;
			P->DecoderModel = Dec;
			P->Speech = TinySettings();
			P->PreferredDevice = EPipelineDevice::CPU;
			UPipelineSpeechTestPipeline* Pipe = Track(NewObject<UPipelineSpeechTestPipeline>(GetTransientPackage()));
			const FPipelineResult Init = Pipe->InitializeFromPackage(P);
			TestTrue(FString::Printf(TEXT("init (error %d): %s"), (int32)Init.Error, *Init.Message), Init.bSuccess);
			if (Init.bSuccess && Pipe->PeekDecoderModel().IsValid())
			{
				const FPipelineResult DI = Pipe->PeekDecoderModel()->Initialize();   // lazy creation, as in the GPU-preference test
				TestTrue(FString::Printf(TEXT("decoder Initialize: %s"), *DI.Message), DI.bSuccess);
				TestTrue(TEXT("decoder CPU"), Pipe->PeekDecoderModel()->GetDevice() == EPipelineDevice::CPU);
			}
		});

		It("a corrupt DecoderModel fails initialization with a 'DecoderModel: ' message", [this]()
		{
			UNNEModelData* Enc = LoadTiny(TEXT("tiny_stt_encoder.onnx"));
			if (!Enc) { return; }
			const char* Junk = "not an onnx model, just some text bytes 0123456789";
			TArray<uint8> Bytes;
			Bytes.Append(reinterpret_cast<const uint8*>(Junk), (int32)FCStringAnsi::Strlen(Junk));

			UPipelineModelPackage* P = MakeSpeechPackage();
			P->Model = Enc;
			P->DecoderModel = Track(UE::ModelPipelines::Tests::MakeModelFromBytes(Bytes));
			P->Speech = TinySettings();
			P->PreferredDevice = EPipelineDevice::CPU;

			UPipelineSpeechTestPipeline* Pipe = Track(NewObject<UPipelineSpeechTestPipeline>(GetTransientPackage()));
			const FPipelineResult Init = Pipe->InitializeFromPackage(P);
			TestFalse(TEXT("init expected to fail"), Init.bSuccess);
			TestTrue(TEXT("message starts with \"DecoderModel: \", got: ") + Init.Message, Init.Message.StartsWith(TEXT("DecoderModel: ")));
			TestFalse(TEXT("IsReady false"), Pipe->IsReady());
			TestFalse(TEXT("GetDecoderModel null after failed init"), Pipe->PeekDecoderModel().IsValid());
		});

		It("a wrong-signature decoder (bad_decoder) is not rejected by Core's model preparation alone", [this]()
		{
			// Brief gives no rule for init vs bad_decoder; only check that the descs are what ResolveSpeechDecoderIO reports on.
			UNNEModelData* Enc = LoadTiny(TEXT("tiny_stt_encoder.onnx"));
			UNNEModelData* Bad = LoadTiny(TEXT("tiny_stt_bad_decoder.onnx"));
			if (!Enc || !Bad) { return; }
			UPipelineModelPackage* P = MakeSpeechPackage();
			P->Model = Enc;
			P->DecoderModel = Bad;
			P->Speech = TinySettings();
			P->PreferredDevice = EPipelineDevice::CPU;
			UPipelineSpeechTestPipeline* Pipe = Track(NewObject<UPipelineSpeechTestPipeline>(GetTransientPackage()));
			const FPipelineResult Init = Pipe->InitializeFromPackage(P);
			if (Init.bSuccess && Pipe->PeekDecoderModel().IsValid())
			{
				const FPipelineResult BI = Pipe->PeekDecoderModel()->Initialize();
				TestTrue(FString::Printf(TEXT("bad decoder Initialize (valid ONNX): %s"), *BI.Message), BI.bSuccess);
				const FDecResult Res = ResolveDec(P->Speech, 4, Pipe->PeekDecoderModel()->GetInputDescs(), Pipe->PeekDecoderModel()->GetOutputDescs());
				TestFalse(TEXT("bad_decoder descs fail ResolveSpeechDecoderIO"), Res.bOk);
				TestEqual(TEXT("exactly one 22: ") + Res.R.ToString(), CountCode(Res.R, InputTypeMismatch), 1);
				const FPipelineValidationMessage* M = FindCode(Res.R, InputTypeMismatch);
				if (M) { TestTrue(TEXT("names token: ") + M->Message, M->Message.Contains(TEXT("token"))); }
			}
			else
			{
				TestFalse(TEXT("if init rejects bad_decoder, IsReady must be false"), Pipe->IsReady());
			}
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
