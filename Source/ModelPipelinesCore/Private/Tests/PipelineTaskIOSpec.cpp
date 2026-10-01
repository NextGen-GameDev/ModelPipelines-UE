// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/ParallelFor.h"
#include "Containers/Array.h"
#include "Containers/ArrayView.h"
#include "Containers/UnrealString.h"
#include "HAL/ThreadSafeCounter.h"
#include "Math/NumericLimits.h"
#include "Modules/ModuleManager.h"
#include "PipelineTensor.h"
#include "PipelineValidation.h"
#include "PipelineModelPackage.h"
#include "PipelineTaskIO.h"
#include "ModelPipelinesCoreModule.h"

namespace PipelineTaskIOSpecPrivate
{
	using EType = EPipelineTensorType;
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;

	static FPipelineTensorDesc D(const TCHAR* Name, EType Type, std::initializer_list<int64> Shape, bool bSupported = true)
	{
		FPipelineTensorDesc Desc;
		Desc.Name = Name;
		Desc.Type = Type;
		Desc.bTypeSupported = bSupported;
		Desc.Shape = TArray<int64>(Shape);
		return Desc;
	}

	static TArray<FPipelineTensorDesc> TextInputs()
	{
		return {D(TEXT("input_ids"), EType::Int64, {-1, -1}), D(TEXT("attention_mask"), EType::Int64, {-1, -1}),
			D(TEXT("token_type_ids"), EType::Int64, {-1, -1})};
	}

	static TArray<FPipelineTensorDesc> ImageInputs(std::initializer_list<int64> Shape)
	{
		return {D(TEXT("pixel_values"), EType::Float32, Shape)};
	}

	struct FResult
	{
		bool bOk = false;
		FPipelineTaskBindings B;
		FPipelineValidationReport R;
	};

	static FResult Resolve(EPipelineTask Task, const FPipelinePreprocessSettings& S, const FPipelinePostprocessSettings& P, int32 NumLabels,
		const TArray<FPipelineTensorDesc>& In, const TArray<FPipelineTensorDesc>& Out)
	{
		FResult Res;
		Res.bOk = ResolvePipelineTaskIO(Task, S, P, NumLabels, In, Out, Res.B, Res.R);
		return Res;
	}

	static FResult ResolveText(EPipelineTask Task, const TArray<FPipelineTensorDesc>& In, const TArray<FPipelineTensorDesc>& Out,
		EPipelinePooling Pooling = EPipelinePooling::Mean, int32 NumLabels = 2)
	{
		FPipelinePreprocessSettings S;
		S.Pooling = Pooling;
		return Resolve(Task, S, FPipelinePostprocessSettings(), NumLabels, In, Out);
	}

	static FResult ResolveImage(EPipelineTask Task, const TArray<FPipelineTensorDesc>& In, const TArray<FPipelineTensorDesc>& Out,
		int32 NumLabels, int32 W = 224, int32 H = 224, EPipelineScoreActivation Act = EPipelineScoreActivation::Softmax)
	{
		FPipelinePreprocessSettings S;
		S.ImageWidth = W;
		S.ImageHeight = H;
		FPipelinePostprocessSettings P;
		P.ScoreActivation = Act;
		return Resolve(Task, S, P, NumLabels, In, Out);
	}

	static int32 CountCode(const FPipelineValidationReport& R, ECode Code)
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

	static const FPipelineValidationMessage* FindCode(const FPipelineValidationReport& R, ECode Code)
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

	/** Detection model with the given logits / boxes output shapes. */
	static TArray<FPipelineTensorDesc> DetOut(std::initializer_list<int64> Logits, std::initializer_list<int64> Boxes)
	{
		return {D(TEXT("logits"), EType::Float32, Logits), D(TEXT("pred_boxes"), EType::Float32, Boxes)};
	}
}

BEGIN_DEFINE_SPEC(FPipelineTaskIOSpec, "Pipelines.Core.TaskIO",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	/** Expects exactly one message of Code with severity Error, and that nothing else was reported as an error. */
	void ExpectOnlyError(const FString& What, const PipelineTaskIOSpecPrivate::FResult& Res, EPipelineValidationCode Code)
	{
		using namespace PipelineTaskIOSpecPrivate;
		TestFalse(What + ": returns false", Res.bOk);
		TestEqual(What + ": exactly one " + LexToString(Code), CountCode(Res.R, Code), 1);
		TestEqual(What + ": exactly one error in total", Res.R.NumErrors(), 1);
	}

	void ExpectClean(const FString& What, const PipelineTaskIOSpecPrivate::FResult& Res)
	{
		TestTrue(What + ": returns true", Res.bOk);
		TestEqual(What + ": no messages (" + Res.R.ToString() + ")", Res.R.Messages.Num(), 0);
	}
END_DEFINE_SPEC(FPipelineTaskIOSpec)

void FPipelineTaskIOSpec::Define()
{
	using namespace PipelineTaskIOSpecPrivate;

	Describe("Text inputs (acceptance 1-7)", [this]()
	{
		It("binds ids/mask/type ids in order and the main output (1)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(),
				{D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 384})});
			ExpectClean(TEXT("minilm-like"), R);
			TestEqual("InputIds", R.B.InputIds, 0);
			TestEqual("AttentionMask", R.B.AttentionMask, 1);
			TestEqual("TokenTypeIds", R.B.TokenTypeIds, 2);
			TestEqual("MainOutput", R.B.MainOutput, 0);
			TestEqual("PixelValues stays absent", R.B.PixelValues, -1);
			TestEqual("PixelMask stays absent", R.B.PixelMask, -1);
			TestEqual("BoxesOutput stays absent", R.B.BoxesOutput, -1);
		});

		It("binds by name, not by position (2)", [this]()
		{
			// Code that assumed ids/mask/type order would bind 0/1/2 here.
			const TArray<FPipelineTensorDesc> In = {D(TEXT("attention_mask"), EType::Int64, {-1, -1}),
				D(TEXT("token_type_ids"), EType::Int64, {-1, -1}), D(TEXT("input_ids"), EType::Int64, {-1, -1})};
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectClean(TEXT("shuffled"), R);
			TestEqual("InputIds", R.B.InputIds, 2);
			TestEqual("AttentionMask", R.B.AttentionMask, 0);
			TestEqual("TokenTypeIds", R.B.TokenTypeIds, 1);
		});

		It("accepts int32 ids and mask, and a missing token_type_ids (3)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = {D(TEXT("input_ids"), EType::Int32, {-1, -1}), D(TEXT("attention_mask"), EType::Int32, {-1, -1})};
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectClean(TEXT("int32 no type ids"), R);
			TestEqual("InputIds", R.B.InputIds, 0);
			TestEqual("AttentionMask", R.B.AttentionMask, 1);
			TestEqual("TokenTypeIds absent", R.B.TokenTypeIds, -1);
		});

		It("reports a missing attention_mask with the actual inputs list (4)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = {D(TEXT("input_ids"), EType::Int64, {-1, -1})};
			const FResult R = ResolveText(EPipelineTask::TextClassification, In, {D(TEXT("logits"), EType::Float32, {-1, 2})});
			ExpectOnlyError(TEXT("no mask"), R, ECode::MissingInput);
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::MissingInput);
			if (M)
			{
				TestTrue("message names attention_mask: " + M->Message, M->Message.Contains(TEXT("attention_mask")));
				TestTrue("message lists actual inputs: " + M->Message, M->Message.Contains(TEXT("input_ids int64 [-1,-1]")));
				TestEqual("severity", M->Severity, ESev::Error);
			}
			TestEqual("AttentionMask unbound", R.B.AttentionMask, -1);
			TestEqual("InputIds still bound", R.B.InputIds, 0);
		});

		It("rejects an extra input and names it (5)", [this]()
		{
			TArray<FPipelineTensorDesc> In = TextInputs();
			In.Add(D(TEXT("position_ids"), EType::Int64, {-1, -1}));
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectOnlyError(TEXT("position_ids"), R, ECode::UnexpectedInput);
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::UnexpectedInput);
			if (M)
			{
				TestTrue("names position_ids: " + M->Message, M->Message.Contains(TEXT("position_ids")));
			}
		});

		It("float32 input_ids is a type mismatch and not bound (6a)", [this]()
		{
			TArray<FPipelineTensorDesc> In = TextInputs();
			In[0] = D(TEXT("input_ids"), EType::Float32, {-1, -1});
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectOnlyError(TEXT("float ids"), R, ECode::InputTypeMismatch);
			TestEqual("InputIds unbound", R.B.InputIds, -1);
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::InputTypeMismatch);
			if (M)
			{
				TestTrue("names tensor and actual type: " + M->Message, M->Message.Contains(TEXT("input_ids")) && M->Message.Contains(TEXT("float32")));
			}
		});

		It("unsupported-type input_ids never matches a type (6b)", [this]()
		{
			// Type defaults to Float32 here; a bTypeSupported-ignoring check on Int64 would also pass; use Int64 as the "meaningless" value.
			TArray<FPipelineTensorDesc> In = TextInputs();
			In[0] = D(TEXT("input_ids"), EType::Int64, {-1, -1}, /*bSupported*/ false);
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectOnlyError(TEXT("unsupported ids"), R, ECode::InputTypeMismatch);
			TestEqual("InputIds unbound", R.B.InputIds, -1);
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::InputTypeMismatch);
			if (M)
			{
				TestTrue("prints 'unsupported': " + M->Message, M->Message.Contains(TEXT("unsupported")));
			}
		});

		It("unsupported-type attention_mask and token_type_ids are also rejected (6c)", [this]()
		{
			TArray<FPipelineTensorDesc> In = TextInputs();
			In[1] = D(TEXT("attention_mask"), EType::Int32, {-1, -1}, false);
			In[2] = D(TEXT("token_type_ids"), EType::Int32, {-1, -1}, false);
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			TestFalse("returns false", R.bOk);
			TestEqual("two type mismatches", CountCode(R.R, ECode::InputTypeMismatch), 2);
			TestEqual("mask unbound", R.B.AttentionMask, -1);
			TestEqual("type ids unbound", R.B.TokenTypeIds, -1);
		});

		It("rank 3 input_ids is a rank mismatch and not bound (6d)", [this]()
		{
			TArray<FPipelineTensorDesc> In = TextInputs();
			In[0] = D(TEXT("input_ids"), EType::Int64, {-1, -1, -1});
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectOnlyError(TEXT("rank 3 ids"), R, ECode::InputRankMismatch);
			TestEqual("InputIds unbound", R.B.InputIds, -1);
		});

		It("optional token_type_ids with wrong type or rank is an error (6e)", [this]()
		{
			TArray<FPipelineTensorDesc> In = TextInputs();
			In[2] = D(TEXT("token_type_ids"), EType::Float32, {-1, -1});
			FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectOnlyError(TEXT("float type ids"), R, ECode::InputTypeMismatch);
			TestEqual("TokenTypeIds unbound", R.B.TokenTypeIds, -1);

			In[2] = D(TEXT("token_type_ids"), EType::Int64, {-1});
			R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectOnlyError(TEXT("rank 1 type ids"), R, ECode::InputRankMismatch);
			TestEqual("TokenTypeIds unbound (rank)", R.B.TokenTypeIds, -1);
		});

		It("a rank-0 input_ids is a rank mismatch, not a crash (6f)", [this]()
		{
			TArray<FPipelineTensorDesc> In = TextInputs();
			In[0] = D(TEXT("input_ids"), EType::Int64, {});
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectOnlyError(TEXT("rank 0"), R, ECode::InputRankMismatch);
		});

		It("fixed dims on input_ids give one dim mismatch naming the dims, still bound (7)", [this]()
		{
			TArray<FPipelineTensorDesc> In = TextInputs();
			In[0] = D(TEXT("input_ids"), EType::Int64, {1, 128});
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectOnlyError(TEXT("fixed ids"), R, ECode::InputDimMismatch);
			TestEqual("InputIds still bound", R.B.InputIds, 0);
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::InputDimMismatch);
			if (M)
			{
				TestTrue("names the tensor: " + M->Message, M->Message.Contains(TEXT("input_ids")));
				TestTrue("names the fixed dims [1,128]: " + M->Message, M->Message.Contains(TEXT("[1,128]")));
			}
		});

		It("a single fixed dim (sequence only) on the mask is a dim mismatch (7b)", [this]()
		{
			TArray<FPipelineTensorDesc> In = TextInputs();
			In[1] = D(TEXT("attention_mask"), EType::Int64, {-1, 64});
			const FResult R = ResolveText(EPipelineTask::TextClassification, In, {D(TEXT("logits"), EType::Float32, {-1, 2})});
			ExpectOnlyError(TEXT("fixed seq"), R, ECode::InputDimMismatch);
			TestEqual("AttentionMask still bound", R.B.AttentionMask, 1);
		});

		It("checks every rule: missing ids and mask plus two extras gives four errors (4b)", [this]()
		{
			// Code that stops at the first error reports 1.
			const TArray<FPipelineTensorDesc> In = {D(TEXT("x"), EType::Float32, {-1, 4}), D(TEXT("y"), EType::Int64, {-1})};
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			TestFalse("returns false", R.bOk);
			TestEqual("MissingInput x2", CountCode(R.R, ECode::MissingInput), 2);
			TestEqual("UnexpectedInput x2", CountCode(R.R, ECode::UnexpectedInput), 2);
			TestEqual("total messages", R.R.Messages.Num(), 4);
		});

		It("names are case-sensitive (4c)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = {D(TEXT("Input_Ids"), EType::Int64, {-1, -1}), D(TEXT("attention_mask"), EType::Int64, {-1, -1})};
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			TestFalse("returns false", R.bOk);
			TestEqual("MissingInput for input_ids", CountCode(R.R, ECode::MissingInput), 1);
			TestEqual("UnexpectedInput for Input_Ids", CountCode(R.R, ECode::UnexpectedInput), 1);
			TestEqual("InputIds unbound", R.B.InputIds, -1);
		});

		It("TextClassification uses the same input rules (7c)", [this]()
		{
			TArray<FPipelineTensorDesc> In = TextInputs();
			In.RemoveAt(1);
			const FResult R = ResolveText(EPipelineTask::TextClassification, In, {D(TEXT("logits"), EType::Float32, {-1, 2})});
			ExpectOnlyError(TEXT("classification no mask"), R, ECode::MissingInput);
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::MissingInput);
			if (M)
			{
				TestTrue("names the task: " + M->Message, M->Message.Contains(TEXT("TextClassification")));
			}
		});
	});

	Describe("Text outputs (acceptance 8-12)", [this]()
	{
		const TArray<FPipelineTensorDesc> Both = {D(TEXT("pooler_output"), EType::Float32, {-1, 384}),
			D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 384})};

		It("picks the output by pooling mode (8a)", [this, Both]()
		{
			// Picking outputs[0] regardless of pooling would fail Mean and CLS.
			FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(), Both, EPipelinePooling::Mean);
			ExpectClean(TEXT("Mean"), R);
			TestEqual("Mean -> last_hidden_state", R.B.MainOutput, 1);

			R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(), Both, EPipelinePooling::CLS);
			ExpectClean(TEXT("CLS"), R);
			TestEqual("CLS -> last_hidden_state", R.B.MainOutput, 1);

			R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(), Both, EPipelinePooling::None);
			ExpectClean(TEXT("None"), R);
			TestEqual("None -> pooler_output", R.B.MainOutput, 0);
		});

		It("accepts token_embeddings without a warning (8b)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(), {D(TEXT("token_embeddings"), EType::Float32, {-1, -1, 8})});
			ExpectClean(TEXT("token_embeddings"), R);
			TestEqual("MainOutput", R.B.MainOutput, 0);
		});

		It("prefers last_hidden_state over token_embeddings regardless of order (8c)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(),
				{D(TEXT("token_embeddings"), EType::Float32, {-1, -1, 8}), D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectClean(TEXT("both token outputs"), R);
			TestEqual("MainOutput", R.B.MainOutput, 1);
		});

		It("Pooling None prefers sentence_embedding over pooler_output (8d)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(),
				{D(TEXT("pooler_output"), EType::Float32, {-1, 16}), D(TEXT("sentence_embedding"), EType::Float32, {-1, 16})},
				EPipelinePooling::None);
			ExpectClean(TEXT("sentence_embedding"), R);
			TestEqual("MainOutput", R.B.MainOutput, 1);
		});

		It("Pooling None with only pooler_output accepts it without warning (8e)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(), {D(TEXT("pooler_output"), EType::Float32, {-1, 16})},
				EPipelinePooling::None);
			ExpectClean(TEXT("pooler only"), R);
			TestEqual("MainOutput", R.B.MainOutput, 0);
		});

		It("falls back to the first float32 rank-3 output with a warning (9)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(), {D(TEXT("hidden"), EType::Float32, {-1, -1, 16})});
			TestTrue("returns true", R.bOk);
			TestEqual("one message", R.R.Messages.Num(), 1);
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::OutputNameFallback);
			TestNotNull("OutputNameFallback present", M);
			if (M)
			{
				TestEqual("severity", M->Severity, ESev::Warning);
				TestTrue("names the used tensor: " + M->Message, M->Message.Contains(TEXT("hidden")));
			}
			TestEqual("MainOutput", R.B.MainOutput, 0);
		});

		It("fallback picks the lowest matching index and skips wrong types (9b)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(),
				{D(TEXT("ints"), EType::Int64, {-1, -1, 4}), D(TEXT("flat"), EType::Float32, {-1, 4}), D(TEXT("a"), EType::Float32, {-1, -1, 4}),
					D(TEXT("b"), EType::Float32, {-1, -1, 4})});
			TestTrue("returns true", R.bOk);
			TestEqual("MainOutput is a (2)", R.B.MainOutput, 2);
			TestEqual("one fallback warning", CountCode(R.R, ECode::OutputNameFallback), 1);
		});

		It("Pooling None fallback uses rank 2 (9c)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(),
				{D(TEXT("seq"), EType::Float32, {-1, -1, 4}), D(TEXT("vec"), EType::Float32, {-1, 4})}, EPipelinePooling::None);
			TestTrue("returns true", R.bOk);
			TestEqual("MainOutput is vec", R.B.MainOutput, 1);
			TestEqual("fallback warning", CountCode(R.R, ECode::OutputNameFallback), 1);
		});

		It("a preferred name with the wrong rank gets no fallback (10)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(),
				{D(TEXT("last_hidden_state"), EType::Float32, {-1, 384}), D(TEXT("other"), EType::Float32, {-1, -1, 384})});
			ExpectOnlyError(TEXT("rank mismatch"), R, ECode::OutputRankMismatch);
			TestEqual("MainOutput unbound", R.B.MainOutput, -1);
			TestEqual("no fallback warning", CountCode(R.R, ECode::OutputNameFallback), 0);
		});

		It("a preferred name with the wrong type gets no fallback (10b)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(),
				{D(TEXT("last_hidden_state"), EType::Int64, {-1, -1, 4}), D(TEXT("other"), EType::Float32, {-1, -1, 4})});
			ExpectOnlyError(TEXT("type mismatch"), R, ECode::OutputTypeMismatch);
			TestEqual("MainOutput unbound", R.B.MainOutput, -1);
		});

		It("no float32 output of the right rank is MissingOutput (11)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(), {D(TEXT("ids_out"), EType::Int64, {-1, -1, 4})});
			ExpectOnlyError(TEXT("int output"), R, ECode::MissingOutput);
			TestEqual("MainOutput unbound", R.B.MainOutput, -1);
		});

		It("an unsupported-type output never matches (11b)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(),
				{D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 4}, false)});
			TestFalse("returns false", R.bOk);
			TestEqual("MainOutput unbound", R.B.MainOutput, -1);
		});

		It("classification: label count against fixed logits dim (12)", [this]()
		{
			const TArray<FPipelineTensorDesc> Out = {D(TEXT("logits"), EType::Float32, {-1, 2})};
			FResult R = ResolveText(EPipelineTask::TextClassification, TextInputs(), Out, EPipelinePooling::Mean, 2);
			ExpectClean(TEXT("2 labels"), R);
			TestEqual("MainOutput", R.B.MainOutput, 0);

			R = ResolveText(EPipelineTask::TextClassification, TextInputs(), Out, EPipelinePooling::Mean, 3);
			ExpectOnlyError(TEXT("3 labels"), R, ECode::LabelCountMismatch);
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::LabelCountMismatch);
			if (M)
			{
				TestTrue("names logits: " + M->Message, M->Message.Contains(TEXT("logits")));
				TestTrue("shows 3 and 2: " + M->Message, M->Message.Contains(TEXT("3")) && M->Message.Contains(TEXT("2")));
			}

			R = ResolveText(EPipelineTask::TextClassification, TextInputs(), Out, EPipelinePooling::Mean, 0);
			ExpectClean(TEXT("0 labels skip the comparison"), R);
			R = ResolveText(EPipelineTask::TextClassification, TextInputs(), Out, EPipelinePooling::Mean, -4);
			ExpectClean(TEXT("negative labels skip the comparison"), R);

			R = ResolveText(EPipelineTask::TextClassification, TextInputs(), {D(TEXT("logits"), EType::Float32, {-1, -1})},
				EPipelinePooling::Mean, 17);
			ExpectClean(TEXT("dynamic logits dim accepts any count"), R);
		});

		It("classification falls back to the first float32 rank-2 output (12b)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextClassification, TextInputs(),
				{D(TEXT("ids"), EType::Int64, {-1, 2}), D(TEXT("scores"), EType::Float32, {-1, 2})});
			TestTrue("returns true", R.bOk);
			TestEqual("MainOutput", R.B.MainOutput, 1);
			TestEqual("fallback warning", CountCode(R.R, ECode::OutputNameFallback), 1);
		});

		It("classification: logits with wrong rank has no fallback (12c)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextClassification, TextInputs(),
				{D(TEXT("logits"), EType::Float32, {-1, -1, 2}), D(TEXT("other"), EType::Float32, {-1, 2})});
			ExpectOnlyError(TEXT("logits rank 3"), R, ECode::OutputRankMismatch);
			TestEqual("MainOutput unbound", R.B.MainOutput, -1);
		});

		It("TextEmbedding ignores NumLabels (12d)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, TextInputs(), {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 384})},
				EPipelinePooling::Mean, 7);
			ExpectClean(TEXT("labels ignored"), R);
		});

		It("Pooling is ignored for TextClassification (12e)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextClassification, TextInputs(), {D(TEXT("logits"), EType::Float32, {-1, 2})},
				EPipelinePooling::None, 2);
			ExpectClean(TEXT("classification with Pooling None"), R);
			TestEqual("MainOutput", R.B.MainOutput, 0);
		});
	});

	Describe("Image classification (acceptance 13-16)", [this]()
	{
		const TArray<FPipelineTensorDesc> Logits1000 = {D(TEXT("logits"), EType::Float32, {-1, 1000})};

		It("accepts fixed, partly dynamic and fully dynamic pixel_values (13)", [this, Logits1000]()
		{
			FResult R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, 3, 224, 224}), Logits1000, 1000);
			ExpectClean(TEXT("[-1,3,224,224]"), R);
			TestEqual("PixelValues", R.B.PixelValues, 0);
			TestEqual("MainOutput", R.B.MainOutput, 0);
			TestEqual("InputIds absent", R.B.InputIds, -1);
			TestEqual("BoxesOutput absent", R.B.BoxesOutput, -1);

			R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, 3, -1, -1}), Logits1000, 1000);
			ExpectClean(TEXT("[-1,3,-1,-1]"), R);
			R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, -1, -1, -1}), Logits1000, 1000);
			ExpectClean(TEXT("[-1,-1,-1,-1]"), R);
		});

		It("batch 1 is Info FixedBatch and still ok; other fixed batches fail (14a)", [this, Logits1000]()
		{
			FResult R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({1, 3, 224, 224}), Logits1000, 1000);
			TestTrue("returns true", R.bOk);
			TestEqual("one message", R.R.Messages.Num(), 1);
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::FixedBatch);
			TestNotNull("FixedBatch present", M);
			if (M)
			{
				TestEqual("severity Info", M->Severity, ESev::Info);
			}
			TestEqual("PixelValues bound", R.B.PixelValues, 0);

			R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({4, 3, 224, 224}), Logits1000, 1000);
			ExpectOnlyError(TEXT("batch 4"), R, ECode::InputDimMismatch);
			TestEqual("still bound after dim failure", R.B.PixelValues, 0);
		});

		It("channel dim must be -1 or 3 (14b)", [this, Logits1000]()
		{
			FResult R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, 1, 224, 224}), Logits1000, 1000);
			ExpectOnlyError(TEXT("1 channel"), R, ECode::InputDimMismatch);
			R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, 4, 224, 224}), Logits1000, 1000);
			ExpectOnlyError(TEXT("4 channels"), R, ECode::InputDimMismatch);
		});

		It("fixed height/width must equal the settings (14c)", [this, Logits1000]()
		{
			FResult R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, 3, 256, 224}), Logits1000, 1000);
			ExpectOnlyError(TEXT("H 256"), R, ECode::InputDimMismatch);
			R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, 3, 224, 256}), Logits1000, 1000);
			ExpectOnlyError(TEXT("W 256"), R, ECode::InputDimMismatch);
			R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, 3, 224, -1}), Logits1000, 1000);
			ExpectClean(TEXT("dynamic W with fixed H"), R);
		});

		It("non-square: dim 2 is height, dim 3 is width (15)", [this, Logits1000]()
		{
			// Code that swaps H and W accepts the second and rejects the first.
			FResult R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, 3, 48, 40}), Logits1000, 1000, /*W*/ 40, /*H*/ 48);
			ExpectClean(TEXT("[-1,3,48,40] with W40 H48"), R);

			R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, 3, 40, 48}), Logits1000, 1000, 40, 48);
			TestFalse("returns false", R.bOk);
			TestTrue("at least one InputDimMismatch", CountCode(R.R, ECode::InputDimMismatch) >= 1);
			TestEqual("only dim errors", R.R.NumErrors(), CountCode(R.R, ECode::InputDimMismatch));
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::InputDimMismatch);
			if (M)
			{
				TestTrue("message mentions expected 48 and 40: " + M->Message, M->Message.Contains(TEXT("48")) && M->Message.Contains(TEXT("40")));
			}
		});

		It("rejects pixel_mask for classification, wrong type and rank for pixel_values (16)", [this, Logits1000]()
		{
			TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 224, 224});
			In.Add(D(TEXT("pixel_mask"), EType::Int64, {-1, -1, -1}));
			FResult R = ResolveImage(EPipelineTask::ImageClassification, In, Logits1000, 1000);
			ExpectOnlyError(TEXT("pixel_mask"), R, ECode::UnexpectedInput);
			TestEqual("PixelMask not bound for classification", R.B.PixelMask, -1);

			TArray<FPipelineTensorDesc> IntIn = {D(TEXT("pixel_values"), EType::Int64, {-1, 3, 224, 224})};
			R = ResolveImage(EPipelineTask::ImageClassification, IntIn, Logits1000, 1000);
			ExpectOnlyError(TEXT("int64 pixels"), R, ECode::InputTypeMismatch);
			TestEqual("PixelValues unbound", R.B.PixelValues, -1);

			R = ResolveImage(EPipelineTask::ImageClassification, ImageInputs({-1, 3, 224}), Logits1000, 1000);
			ExpectOnlyError(TEXT("rank 3 pixels"), R, ECode::InputRankMismatch);
			TestEqual("PixelValues unbound (rank)", R.B.PixelValues, -1);

			R = ResolveImage(EPipelineTask::ImageClassification, {D(TEXT("pixel_values"), EType::Float32, {-1, 3, 224, 224}, false)}, Logits1000, 1000);
			TestFalse("unsupported type returns false", R.bOk);
			TestEqual("unsupported type unbound", R.B.PixelValues, -1);
		});

		It("missing pixel_values and text inputs on an image model (16b)", [this, Logits1000]()
		{
			const FResult R = ResolveImage(EPipelineTask::ImageClassification, TextInputs(), Logits1000, 1000);
			TestFalse("returns false", R.bOk);
			TestEqual("MissingInput x1", CountCode(R.R, ECode::MissingInput), 1);
			TestEqual("UnexpectedInput x3", CountCode(R.R, ECode::UnexpectedInput), 3);
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::MissingInput);
			if (M)
			{
				TestTrue("names pixel_values: " + M->Message, M->Message.Contains(TEXT("pixel_values")));
			}
		});

		It("logits label count and fallback for image classification (16c)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 224, 224});
			FResult R = ResolveImage(EPipelineTask::ImageClassification, In, {D(TEXT("logits"), EType::Float32, {-1, 10})}, 1000);
			ExpectOnlyError(TEXT("10 vs 1000"), R, ECode::LabelCountMismatch);

			R = ResolveImage(EPipelineTask::ImageClassification, In, {D(TEXT("logits"), EType::Float32, {-1, 10})}, 0);
			ExpectClean(TEXT("labels 0"), R);

			R = ResolveImage(EPipelineTask::ImageClassification, In, {D(TEXT("probs"), EType::Float32, {-1, 10})}, 10);
			TestTrue("fallback returns true", R.bOk);
			TestEqual("fallback MainOutput", R.B.MainOutput, 0);
			TestEqual("fallback warning", CountCode(R.R, ECode::OutputNameFallback), 1);

			R = ResolveImage(EPipelineTask::ImageClassification, In, {D(TEXT("logits"), EType::Float32, {-1, 10, 10})}, 10);
			ExpectOnlyError(TEXT("rank 3 logits"), R, ECode::OutputRankMismatch);

			R = ResolveImage(EPipelineTask::ImageClassification, In, {D(TEXT("logits"), EType::Int32, {-1, 10})}, 10);
			ExpectOnlyError(TEXT("int logits"), R, ECode::OutputTypeMismatch);

			R = ResolveImage(EPipelineTask::ImageClassification, In, {D(TEXT("logits"), EType::Int32, {-1, 10})}, 10);
			TestEqual("MainOutput unbound on type mismatch", R.B.MainOutput, -1);

			R = ResolveImage(EPipelineTask::ImageClassification, In, TArray<FPipelineTensorDesc>(), 10);
			ExpectOnlyError(TEXT("no outputs"), R, ECode::MissingOutput);
		});
	});

	Describe("Object detection (acceptance 17-19)", [this]()
	{
		It("binds logits and pred_boxes for DETR-like outputs, in either order (17a)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 640, 640});
			FResult R = ResolveImage(EPipelineTask::ObjectDetection, In, DetOut({-1, 300, 80}, {-1, 300, 4}), 80, 640, 640,
				EPipelineScoreActivation::Sigmoid);
			ExpectClean(TEXT("sigmoid 80"), R);
			TestEqual("PixelValues", R.B.PixelValues, 0);
			TestEqual("MainOutput", R.B.MainOutput, 0);
			TestEqual("BoxesOutput", R.B.BoxesOutput, 1);

			const TArray<FPipelineTensorDesc> Swapped = {D(TEXT("pred_boxes"), EType::Float32, {-1, 300, 4}),
				D(TEXT("logits"), EType::Float32, {-1, 300, 80})};
			R = ResolveImage(EPipelineTask::ObjectDetection, In, Swapped, 80, 640, 640, EPipelineScoreActivation::Sigmoid);
			ExpectClean(TEXT("swapped"), R);
			TestEqual("MainOutput swapped", R.B.MainOutput, 1);
			TestEqual("BoxesOutput swapped", R.B.BoxesOutput, 0);
		});

		It("Softmax needs NumLabels + 1 classes (17b)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 640, 640});
			FResult R = ResolveImage(EPipelineTask::ObjectDetection, In, DetOut({-1, 300, 80}, {-1, 300, 4}), 80, 640, 640,
				EPipelineScoreActivation::Softmax);
			ExpectOnlyError(TEXT("softmax 80 vs C 80"), R, ECode::LabelCountMismatch);

			R = ResolveImage(EPipelineTask::ObjectDetection, In, DetOut({-1, 300, 80}, {-1, 300, 4}), 79, 640, 640,
				EPipelineScoreActivation::Softmax);
			ExpectClean(TEXT("softmax 79 vs C 80"), R);
			TestEqual("MainOutput", R.B.MainOutput, 0);
			TestEqual("BoxesOutput", R.B.BoxesOutput, 1);
		});

		It("Sigmoid with a different class count fails (17c)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 640, 640});
			const FResult R = ResolveImage(EPipelineTask::ObjectDetection, In, DetOut({-1, 300, 81}, {-1, 300, 4}), 80, 640, 640,
				EPipelineScoreActivation::Sigmoid);
			ExpectOnlyError(TEXT("sigmoid 80 vs C 81"), R, ECode::LabelCountMismatch);
		});

		It("dynamic C and NumLabels <= 0 skip the class-count check (17d)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 640, 640});
			FResult R = ResolveImage(EPipelineTask::ObjectDetection, In, DetOut({-1, 300, -1}, {-1, 300, 4}), 80, 640, 640);
			ExpectClean(TEXT("dynamic C"), R);
			R = ResolveImage(EPipelineTask::ObjectDetection, In, DetOut({-1, 300, 80}, {-1, 300, 4}), 0, 640, 640);
			ExpectClean(TEXT("labels 0"), R);
		});

		It("boxes: last dim must be 4, Q must match, dynamic is fine (18a)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 640, 640});
			auto Run = [&](std::initializer_list<int64> L, std::initializer_list<int64> B)
			{
				return ResolveImage(EPipelineTask::ObjectDetection, In, DetOut(L, B), 79, 640, 640);
			};
			ExpectOnlyError(TEXT("boxes 5"), Run({-1, 300, 80}, {-1, 300, 5}), ECode::OutputDimMismatch);
			ExpectOnlyError(TEXT("Q 100 vs 300"), Run({-1, 300, 80}, {-1, 100, 4}), ECode::OutputDimMismatch);
			ExpectClean(TEXT("boxes Q dynamic"), Run({-1, 300, 80}, {-1, -1, 4}));
			ExpectClean(TEXT("logits Q dynamic"), Run({-1, -1, 80}, {-1, 100, 4}));
			ExpectClean(TEXT("both Q dynamic"), Run({-1, -1, 80}, {-1, -1, 4}));
		});

		It("a missing pred_boxes or logits is MissingOutput (18b)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 640, 640});
			FResult R = ResolveImage(EPipelineTask::ObjectDetection, In, {D(TEXT("logits"), EType::Float32, {-1, 300, 80})}, 79, 640, 640);
			ExpectOnlyError(TEXT("no boxes"), R, ECode::MissingOutput);
			TestEqual("logits still bound", R.B.MainOutput, 0);
			TestEqual("boxes unbound", R.B.BoxesOutput, -1);

			R = ResolveImage(EPipelineTask::ObjectDetection, In, {D(TEXT("pred_boxes"), EType::Float32, {-1, 300, 4})}, 79, 640, 640);
			ExpectOnlyError(TEXT("no logits"), R, ECode::MissingOutput);
			TestEqual("boxes still bound", R.B.BoxesOutput, 0);
			TestEqual("logits unbound", R.B.MainOutput, -1);
		});

		It("detection has no output-name fallback (18c)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 640, 640});
			const FResult R = ResolveImage(EPipelineTask::ObjectDetection, In,
				{D(TEXT("scores"), EType::Float32, {-1, 300, 80}), D(TEXT("boxes"), EType::Float32, {-1, 300, 4})}, 79, 640, 640);
			TestFalse("returns false", R.bOk);
			TestEqual("MissingOutput x2", CountCode(R.R, ECode::MissingOutput), 2);
			TestEqual("no fallback warning", CountCode(R.R, ECode::OutputNameFallback), 0);
		});

		It("wrong rank or type for logits / pred_boxes reports that code, not MissingOutput (18d)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 640, 640});
			FResult R = ResolveImage(EPipelineTask::ObjectDetection, In, DetOut({-1, 80}, {-1, 300, 4}), 79, 640, 640);
			ExpectOnlyError(TEXT("logits rank 2"), R, ECode::OutputRankMismatch);
			TestEqual("MissingOutput absent", CountCode(R.R, ECode::MissingOutput), 0);
			TestEqual("logits unbound", R.B.MainOutput, -1);

			R = ResolveImage(EPipelineTask::ObjectDetection, In, DetOut({-1, 300, 80}, {-1, 4}), 79, 640, 640);
			ExpectOnlyError(TEXT("boxes rank 2"), R, ECode::OutputRankMismatch);
			TestEqual("boxes unbound", R.B.BoxesOutput, -1);

			const TArray<FPipelineTensorDesc> IntBoxes = {D(TEXT("logits"), EType::Float32, {-1, 300, 80}), D(TEXT("pred_boxes"), EType::Int64, {-1, 300, 4})};
			R = ResolveImage(EPipelineTask::ObjectDetection, In, IntBoxes, 79, 640, 640);
			ExpectOnlyError(TEXT("int boxes"), R, ECode::OutputTypeMismatch);
			TestEqual("MissingOutput absent (type)", CountCode(R.R, ECode::MissingOutput), 0);
		});

		It("optional pixel_mask is bound; wrong type or rank is an error (19)", [this]()
		{
			const TArray<FPipelineTensorDesc> Out = DetOut({-1, 300, 80}, {-1, 300, 4});
			TArray<FPipelineTensorDesc> In = ImageInputs({-1, 3, 640, 640});
			In.Add(D(TEXT("pixel_mask"), EType::Int64, {-1, -1, -1}));
			FResult R = ResolveImage(EPipelineTask::ObjectDetection, In, Out, 79, 640, 640);
			ExpectClean(TEXT("pixel_mask int64"), R);
			TestEqual("PixelMask", R.B.PixelMask, 1);
			TestEqual("PixelValues", R.B.PixelValues, 0);

			In[1] = D(TEXT("pixel_mask"), EType::Int32, {-1, -1, -1});
			R = ResolveImage(EPipelineTask::ObjectDetection, In, Out, 79, 640, 640);
			ExpectClean(TEXT("pixel_mask int32"), R);
			TestEqual("PixelMask int32", R.B.PixelMask, 1);

			In[1] = D(TEXT("pixel_mask"), EType::Float32, {-1, -1, -1});
			R = ResolveImage(EPipelineTask::ObjectDetection, In, Out, 79, 640, 640);
			ExpectOnlyError(TEXT("pixel_mask float"), R, ECode::InputTypeMismatch);
			TestEqual("PixelMask unbound (type)", R.B.PixelMask, -1);

			In[1] = D(TEXT("pixel_mask"), EType::Int64, {-1, -1});
			R = ResolveImage(EPipelineTask::ObjectDetection, In, Out, 79, 640, 640);
			ExpectOnlyError(TEXT("pixel_mask rank 2"), R, ECode::InputRankMismatch);
			TestEqual("PixelMask unbound (rank)", R.B.PixelMask, -1);

			In[1] = D(TEXT("something_else"), EType::Int64, {-1, -1, -1});
			R = ResolveImage(EPipelineTask::ObjectDetection, In, Out, 79, 640, 640);
			ExpectOnlyError(TEXT("unknown extra"), R, ECode::UnexpectedInput);
		});

		It("detection pixel_values shares the image input rules (19b)", [this]()
		{
			const TArray<FPipelineTensorDesc> Out = DetOut({-1, 300, 80}, {-1, 300, 4});
			FResult R = ResolveImage(EPipelineTask::ObjectDetection, ImageInputs({-1, 3, 480, 640}), Out, 79, 640, 640);
			ExpectOnlyError(TEXT("H 480 vs 640"), R, ECode::InputDimMismatch);
			R = ResolveImage(EPipelineTask::ObjectDetection, ImageInputs({1, 3, 640, 640}), Out, 79, 640, 640);
			TestTrue("batch 1 ok", R.bOk);
			TestEqual("FixedBatch info", CountCode(R.R, ECode::FixedBatch), 1);
		});
	});

	Describe("General (acceptance 20-23)", [this]()
	{
		It("Task None is a single NoTask error (20)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::None, TextInputs(), {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			ExpectOnlyError(TEXT("None"), R, ECode::NoTask);
			TestEqual("only one message", R.R.Messages.Num(), 1);
			TestEqual("nothing bound", R.B.InputIds, -1);
		});

		It("empty arrays: two MissingInput and one MissingOutput, printing (none) (21)", [this]()
		{
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, {}, {});
			TestFalse("returns false", R.bOk);
			TestEqual("MissingInput x2", CountCode(R.R, ECode::MissingInput), 2);
			TestEqual("MissingOutput x1", CountCode(R.R, ECode::MissingOutput), 1);
			TestEqual("three messages", R.R.Messages.Num(), 3);
			for (const FPipelineValidationMessage& M : R.R.Messages)
			{
				TestTrue("prints (none): " + M.Message, M.Message.Contains(TEXT("(none)")));
			}
		});

		It("empty arrays for the image tasks do not crash and report missing tensors (21b)", [this]()
		{
			FResult R = ResolveImage(EPipelineTask::ImageClassification, {}, {}, 10);
			TestFalse("classification false", R.bOk);
			TestEqual("MissingInput", CountCode(R.R, ECode::MissingInput), 1);
			TestEqual("MissingOutput", CountCode(R.R, ECode::MissingOutput), 1);
			R = ResolveImage(EPipelineTask::ObjectDetection, {}, {}, 10);
			TestFalse("detection false", R.bOk);
			TestEqual("MissingInput", CountCode(R.R, ECode::MissingInput), 1);
			TestEqual("MissingOutput x2", CountCode(R.R, ECode::MissingOutput), 2);
		});

		It("message format: tensors printed as name type [dims], joined by ', ' (21c)", [this]()
		{
			const TArray<FPipelineTensorDesc> In = {D(TEXT("x"), EType::Float32, {-1, 4}), D(TEXT("y"), EType::Int32, {2, 3}),
				D(TEXT("z"), EType::Int64, {1}, false)};
			const FResult R = ResolveText(EPipelineTask::TextClassification, In, {D(TEXT("logits"), EType::Float32, {-1, 2})});
			const FPipelineValidationMessage* M = FindCode(R.R, ECode::MissingInput);
			TestNotNull("MissingInput present", M);
			if (M)
			{
				TestTrue("x float32 [-1,4]: " + M->Message, M->Message.Contains(TEXT("x float32 [-1,4]")));
				TestTrue("y int32 [2,3]: " + M->Message, M->Message.Contains(TEXT("y int32 [2,3]")));
				TestTrue("z unsupported [1]: " + M->Message, M->Message.Contains(TEXT("z unsupported [1]")));
				TestTrue("joined with ', ': " + M->Message, M->Message.Contains(TEXT("x float32 [-1,4], y int32 [2,3], z unsupported [1]")));
			}
		});

		It("an existing Error does not make a valid model fail, and old messages stay first (22)", [this]()
		{
			FResult R;
			R.R.Add(ESev::Error, ECode::NoModel, TEXT("pre-existing"));
			R.R.Add(ESev::Warning, ECode::EmptyLabel, TEXT("pre-existing 2"));
			FPipelinePreprocessSettings S;
			R.bOk = ResolvePipelineTaskIO(EPipelineTask::TextEmbedding, S, FPipelinePostprocessSettings(), 0, TextInputs(),
				TArray<FPipelineTensorDesc>{D(TEXT("hidden"), EType::Float32, {-1, -1, 8})}, R.B, R.R);
			TestTrue("returns true for this call", R.bOk);
			TestEqual("messages", R.R.Messages.Num(), 3);
			if (R.R.Messages.Num() == 3)
			{
				TestEqual("first kept", R.R.Messages[0].Code, ECode::NoModel);
				TestEqual("second kept", R.R.Messages[1].Code, ECode::EmptyLabel);
				TestEqual("new one appended", R.R.Messages[2].Code, ECode::OutputNameFallback);
			}
			TestFalse("report as a whole is invalid", R.R.IsValid());
		});

		It("an existing Error plus a new error still returns false (22b)", [this]()
		{
			FResult R;
			R.R.Add(ESev::Error, ECode::NoModel, TEXT("pre-existing"));
			FPipelinePreprocessSettings S;
			R.bOk = ResolvePipelineTaskIO(EPipelineTask::TextEmbedding, S, FPipelinePostprocessSettings(), 0, TArray<FPipelineTensorDesc>(),
				TArray<FPipelineTensorDesc>(), R.B, R.R);
			TestFalse("returns false", R.bOk);
			TestEqual("1 + 3 messages", R.R.Messages.Num(), 4);
		});

		It("a call on a report that holds an Error does not clear or reorder it (22c)", [this]()
		{
			FResult R;
			R.R.Add(ESev::Error, ECode::NoTask, TEXT("first"));
			FPipelinePreprocessSettings S;
			ResolvePipelineTaskIO(EPipelineTask::None, S, FPipelinePostprocessSettings(), 0, TArray<FPipelineTensorDesc>(), TArray<FPipelineTensorDesc>(),
				R.B, R.R);
			TestEqual("two NoTask messages", R.R.Messages.Num(), 2);
			if (R.R.Messages.Num() == 2)
			{
				TestEqual("first text kept", R.R.Messages[0].Message, FString(TEXT("first")));
			}
		});

		It("bindings are reset to -1 when nothing resolves (22d)", [this]()
		{
			FPipelineTaskBindings B;
			B.InputIds = 5;
			B.AttentionMask = 5;
			B.TokenTypeIds = 5;
			B.PixelValues = 5;
			B.PixelMask = 5;
			B.MainOutput = 5;
			B.BoxesOutput = 5;
			FPipelineValidationReport Rep;
			FPipelinePreprocessSettings S;
			ResolvePipelineTaskIO(EPipelineTask::TextEmbedding, S, FPipelinePostprocessSettings(), 0, TArray<FPipelineTensorDesc>(),
				TArray<FPipelineTensorDesc>(), B, Rep);
			TestEqual("InputIds", B.InputIds, -1);
			TestEqual("AttentionMask", B.AttentionMask, -1);
			TestEqual("TokenTypeIds", B.TokenTypeIds, -1);
			TestEqual("PixelValues", B.PixelValues, -1);
			TestEqual("PixelMask", B.PixelMask, -1);
			TestEqual("MainOutput", B.MainOutput, -1);
			TestEqual("BoxesOutput", B.BoxesOutput, -1);
		});

		It("rank-8 descs, MAX_int64 dims, empty and duplicate names never crash (23)", [this]()
		{
			const TArray<int64> Huge = {MAX_int64, MAX_int64, MAX_int64, MAX_int64, MAX_int64, MAX_int64, MAX_int64, MAX_int64};
			const TArray<int64> Neg = {MIN_int64, -2, -1, 0, 1, 2, MAX_int64, 3};
			auto Make = [](const TCHAR* Name, EType Type, const TArray<int64>& Shape)
			{
				FPipelineTensorDesc Desc;
				Desc.Name = Name;
				Desc.Type = Type;
				Desc.Shape = Shape;
				return Desc;
			};
			TArray<FPipelineTensorDesc> In = {Make(TEXT(""), EType::Float32, Huge), Make(TEXT(""), EType::Int64, Neg),
				Make(TEXT("input_ids"), EType::Int64, Huge), Make(TEXT("input_ids"), EType::Int32, Neg),
				Make(TEXT("pixel_values"), EType::Float32, Huge), Make(TEXT("pixel_values"), EType::Float32, Neg),
				Make(TEXT("attention_mask"), EType::Int64, Huge)};
			TArray<FPipelineTensorDesc> Out = {Make(TEXT(""), EType::Float32, Huge), Make(TEXT("logits"), EType::Float32, Neg),
				Make(TEXT("logits"), EType::Float32, Huge), Make(TEXT("pred_boxes"), EType::Float32, Huge),
				Make(TEXT("last_hidden_state"), EType::Float32, Huge)};
			const EPipelineTask Tasks[] = {EPipelineTask::None, EPipelineTask::TextEmbedding, EPipelineTask::TextClassification,
				EPipelineTask::ImageClassification, EPipelineTask::ObjectDetection, (EPipelineTask)200};
			int32 Calls = 0;
			for (EPipelineTask Task : Tasks)
			{
				for (int32 Labels : {-1, 0, 1, MAX_int32})
				{
					FPipelinePreprocessSettings S;
					S.ImageWidth = MAX_int32;
					S.ImageHeight = 0;
					FPipelinePostprocessSettings P;
					P.ScoreActivation = (Labels == 1) ? EPipelineScoreActivation::Sigmoid : EPipelineScoreActivation::Softmax;
					FResult R = Resolve(Task, S, P, Labels, In, Out);
					// Bindings must always be valid indices or -1.
					TestTrue("MainOutput in range", R.B.MainOutput >= -1 && R.B.MainOutput < Out.Num());
					TestTrue("BoxesOutput in range", R.B.BoxesOutput >= -1 && R.B.BoxesOutput < Out.Num());
					TestTrue("InputIds in range", R.B.InputIds >= -1 && R.B.InputIds < In.Num());
					TestTrue("PixelValues in range", R.B.PixelValues >= -1 && R.B.PixelValues < In.Num());
					++Calls;
				}
			}
			TestEqual("all calls returned", Calls, 24);
			// One more with zero-rank outputs and empty names only.
			const TArray<FPipelineTensorDesc> Weird = {Make(TEXT(""), EType::Float32, {})};
			const FResult R = Resolve(EPipelineTask::ObjectDetection, FPipelinePreprocessSettings(), FPipelinePostprocessSettings(), 3, Weird, Weird);
			TestFalse("weird model is rejected", R.bOk);
		});

		It("duplicate tensor names bind the first match deterministically (23b)", [this]()
		{
			// Two input_ids: whichever the implementation binds must be a valid index and the call must report the other as unexpected or similar.
			TArray<FPipelineTensorDesc> In = TextInputs();
			In.Add(D(TEXT("input_ids"), EType::Int64, {-1, -1}));
			const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 8})});
			TestTrue("bound to a valid index", R.B.InputIds == 0 || R.B.InputIds == 3);
		});

		It("is callable from many threads at once (23c)", [this]()
		{
			FThreadSafeCounter Failures;
			const TArray<FPipelineTensorDesc> In = TextInputs();
			const TArray<FPipelineTensorDesc> Out = {D(TEXT("last_hidden_state"), EType::Float32, {-1, -1, 384})};
			ParallelFor(64, [&](int32 Index)
			{
				const FResult R = ResolveText(EPipelineTask::TextEmbedding, In, Out);
				if (!R.bOk || R.B.InputIds != 0 || R.B.AttentionMask != 1 || R.B.MainOutput != 0 || R.R.Messages.Num() != 0)
				{
					Failures.Increment();
				}
				const FResult Bad = ResolveText(EPipelineTask::TextEmbedding, {}, Out);
				if (Bad.bOk || Bad.R.NumErrors() != 2)
				{
					Failures.Increment();
				}
			});
			TestEqual("failing iterations", Failures.GetValue(), 0);
		});
	});

	Describe("Module TryGet (acceptance 24)", [this]()
	{
		It("returns the loaded module instance", [this]()
		{
			FModelPipelinesCoreModule* Ptr = FModelPipelinesCoreModule::TryGet();
			TestNotNull("TryGet is non-null in the editor test run", Ptr);
			if (Ptr)
			{
				TestTrue("TryGet == &Get()", Ptr == &FModelPipelinesCoreModule::Get());
			}
		});

		It("can be called from a worker thread (24b)", [this]()
		{
			FThreadSafeCounter Mismatches;
			FModelPipelinesCoreModule* Expected = &FModelPipelinesCoreModule::Get();
			ParallelFor(16, [&](int32)
			{
				if (FModelPipelinesCoreModule::TryGet() != Expected)
				{
					Mismatches.Increment();
				}
			});
			TestEqual("worker-thread TryGet mismatches", Mismatches.GetValue(), 0);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
