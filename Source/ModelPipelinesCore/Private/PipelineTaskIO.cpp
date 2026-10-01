// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineTaskIO.h"

#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "PipelineModelPackage.h"
#include "PipelineTensor.h"
#include "PipelineValidation.h"

namespace UE::ModelPipelines::TaskIOPrivate
{
	using ESev = EPipelineValidationSeverity;
	using ECode = EPipelineValidationCode;

	/** Element type class a task tensor needs. */
	enum class ETypeKind : uint8
	{
		Float,
		Int
	};

	static bool MatchesKind(const FPipelineTensorDesc& Desc, ETypeKind Kind)
	{
		if (!Desc.bTypeSupported)
		{
			return false;
		}
		if (Kind == ETypeKind::Float)
		{
			return Desc.Type == EPipelineTensorType::Float32;
		}
		return Desc.Type == EPipelineTensorType::Int64 || Desc.Type == EPipelineTensorType::Int32;
	}

	static const TCHAR* KindToString(ETypeKind Kind)
	{
		return Kind == ETypeKind::Float ? TEXT("float32") : TEXT("int64/int32");
	}

	static FString ShapeToString(const TArray<int64>& Shape)
	{
		FString Out(TEXT("["));
		for (int32 Index = 0; Index < Shape.Num(); ++Index)
		{
			if (Index > 0)
			{
				Out += TEXT(",");
			}
			Out += FString::Printf(TEXT("%lld"), (long long)Shape[Index]);
		}
		Out += TEXT("]");
		return Out;
	}

	/** "<name> <type> [<dims>]"; an unsupported type prints "unsupported". */
	static FString DescToString(const FPipelineTensorDesc& Desc)
	{
		const TCHAR* TypeStr = Desc.bTypeSupported ? LexToString(Desc.Type) : TEXT("unsupported");
		return FString::Printf(TEXT("%s %s %s"), *Desc.Name, TypeStr, *ShapeToString(Desc.Shape));
	}

	/** Descs joined with ", "; "(none)" when empty. */
	static FString ListToString(TConstArrayView<FPipelineTensorDesc> Descs)
	{
		if (Descs.Num() == 0)
		{
			return FString(TEXT("(none)"));
		}
		FString Out;
		for (int32 Index = 0; Index < Descs.Num(); ++Index)
		{
			if (Index > 0)
			{
				Out += TEXT(", ");
			}
			Out += DescToString(Descs[Index]);
		}
		return Out;
	}

	/** Lowest index with exactly this name (case-sensitive), or -1. */
	static int32 FindByName(TConstArrayView<FPipelineTensorDesc> Descs, const TCHAR* Name)
	{
		for (int32 Index = 0; Index < Descs.Num(); ++Index)
		{
			if (Descs[Index].Name.Equals(Name, ESearchCase::CaseSensitive))
			{
				return Index;
			}
		}
		return -1;
	}

	/** One expected model input. */
	struct FInputSpec
	{
		const TCHAR* Name = nullptr;
		bool bRequired = true;
		ETypeKind Kind = ETypeKind::Int;
		int32 Rank = 2;
		int32* Binding = nullptr;
	};

	/** Shared state of one ResolvePipelineTaskIO call. */
	struct FContext
	{
		const TCHAR* TaskName = TEXT("");
		TConstArrayView<FPipelineTensorDesc> Inputs;
		TConstArrayView<FPipelineTensorDesc> Outputs;
		FPipelineValidationReport& Report;
		int32 NumErrorsAdded = 0;

		FContext(FPipelineValidationReport& InReport)
			: Report(InReport)
		{
		}

		void Error(ECode Code, const FString& Message)
		{
			Report.Add(ESev::Error, Code, Message);
			++NumErrorsAdded;
		}

		void Warning(ECode Code, const FString& Message)
		{
			Report.Add(ESev::Warning, Code, Message);
		}

		void Info(ECode Code, const FString& Message)
		{
			Report.Add(ESev::Info, Code, Message);
		}
	};

	/** Expected input names for UnexpectedInput messages, e.g. "input_ids, attention_mask, token_type_ids (optional)". */
	static FString SpecNames(TConstArrayView<FInputSpec> Specs)
	{
		FString Out;
		for (int32 Index = 0; Index < Specs.Num(); ++Index)
		{
			if (Index > 0)
			{
				Out += TEXT(", ");
			}
			Out += Specs[Index].Name;
			if (!Specs[Index].bRequired)
			{
				Out += TEXT(" (optional)");
			}
		}
		return Out;
	}

	/**
	 * Binds every spec by name, checks type and rank, and reports every input no spec claimed.
	 * DimCheck(SpecIndex, InputIndex) runs for each input that passed type and rank.
	 */
	template <typename FDimCheck>
	static void ResolveInputs(FContext& Ctx, TConstArrayView<FInputSpec> Specs, FDimCheck&& DimCheck)
	{
		TArray<bool> Claimed;
		Claimed.SetNumZeroed(Ctx.Inputs.Num());

		for (int32 SpecIndex = 0; SpecIndex < Specs.Num(); ++SpecIndex)
		{
			const FInputSpec& Spec = Specs[SpecIndex];
			const int32 Index = FindByName(Ctx.Inputs, Spec.Name);
			if (Index < 0)
			{
				if (Spec.bRequired)
				{
					Ctx.Error(ECode::MissingInput, FString::Printf(TEXT("%s needs input '%s' (%s, rank %d); the model has inputs: %s"),
						Ctx.TaskName, Spec.Name, KindToString(Spec.Kind), Spec.Rank, *ListToString(Ctx.Inputs)));
				}
				continue;
			}

			Claimed[Index] = true;
			const FPipelineTensorDesc& Desc = Ctx.Inputs[Index];
			if (!MatchesKind(Desc, Spec.Kind))
			{
				Ctx.Error(ECode::InputTypeMismatch, FString::Printf(TEXT("%s needs input '%s' as %s (rank %d); the model has %s"),
					Ctx.TaskName, Spec.Name, KindToString(Spec.Kind), Spec.Rank, *DescToString(Desc)));
				continue;
			}
			if (Desc.Shape.Num() != Spec.Rank)
			{
				Ctx.Error(ECode::InputRankMismatch, FString::Printf(TEXT("%s needs input '%s' with rank %d (%s); the model has %s (rank %d)"),
					Ctx.TaskName, Spec.Name, Spec.Rank, KindToString(Spec.Kind), *DescToString(Desc), Desc.Shape.Num()));
				continue;
			}

			*Spec.Binding = Index;
			DimCheck(SpecIndex, Index);
		}

		for (int32 Index = 0; Index < Ctx.Inputs.Num(); ++Index)
		{
			if (Claimed[Index])
			{
				continue;
			}
			const FPipelineTensorDesc& Desc = Ctx.Inputs[Index];
			const int32 First = FindByName(Ctx.Inputs, *Desc.Name);
			const bool bDuplicateOfExpected = First >= 0 && First != Index && Claimed[First];
			if (bDuplicateOfExpected)
			{
				Ctx.Error(ECode::UnexpectedInput, FString::Printf(TEXT("%s does not feed input %s (duplicate name; input %d is used); expected inputs: %s"),
					Ctx.TaskName, *DescToString(Desc), First, *SpecNames(Specs)));
			}
			else
			{
				Ctx.Error(ECode::UnexpectedInput, FString::Printf(TEXT("%s does not feed input %s; expected inputs: %s"),
					Ctx.TaskName, *DescToString(Desc), *SpecNames(Specs)));
			}
		}
	}

	/** Text inputs: every dim must be dynamic. One message per tensor. */
	static void CheckAllDynamic(FContext& Ctx, const TCHAR* Name, const FPipelineTensorDesc& Desc)
	{
		FString Offending;
		for (int32 Dim = 0; Dim < Desc.Shape.Num(); ++Dim)
		{
			if (Desc.Shape[Dim] != -1)
			{
				if (!Offending.IsEmpty())
				{
					Offending += TEXT(", ");
				}
				Offending += FString::Printf(TEXT("dim %d = %lld"), Dim, (long long)Desc.Shape[Dim]);
			}
		}
		if (!Offending.IsEmpty())
		{
			Ctx.Error(ECode::InputDimMismatch, FString::Printf(
				TEXT("%s needs input '%s' with dynamic batch and sequence dims [-1,-1]; the model has %s (fixed %s)"),
				Ctx.TaskName, Name, *DescToString(Desc), *Offending));
		}
	}

	/** pixel_values [B,3,H,W]: B -1 (1 = Info), C -1 or 3, H/W -1 or the settings. One error message per tensor. */
	static void CheckPixelDims(FContext& Ctx, const FPipelineTensorDesc& Desc, int32 ImageWidth, int32 ImageHeight)
	{
		const TArray<int64>& S = Desc.Shape;
		if (S.Num() != 4)
		{
			return;
		}
		FString Offending;
		auto AddOffending = [&Offending](int32 Dim, int64 Value, const TCHAR* What)
		{
			if (!Offending.IsEmpty())
			{
				Offending += TEXT(", ");
			}
			Offending += FString::Printf(TEXT("dim %d (%s) = %lld"), Dim, What, (long long)Value);
		};

		if (S[0] == 1)
		{
			Ctx.Info(ECode::FixedBatch, FString::Printf(TEXT("%s: input 'pixel_values' has a fixed batch of 1 (%s); images run one per inference"),
				Ctx.TaskName, *DescToString(Desc)));
		}
		else if (S[0] != -1)
		{
			AddOffending(0, S[0], TEXT("batch"));
		}
		if (S[1] != -1 && S[1] != 3)
		{
			AddOffending(1, S[1], TEXT("channels"));
		}
		if (S[2] != -1 && S[2] != (int64)ImageHeight)
		{
			AddOffending(2, S[2], TEXT("height"));
		}
		if (S[3] != -1 && S[3] != (int64)ImageWidth)
		{
			AddOffending(3, S[3], TEXT("width"));
		}
		if (!Offending.IsEmpty())
		{
			Ctx.Error(ECode::InputDimMismatch, FString::Printf(
				TEXT("%s needs input 'pixel_values' [batch -1 or 1, channels 3, height %d, width %d] (any dim may be -1); the model has %s (%s)"),
				Ctx.TaskName, ImageHeight, ImageWidth, *DescToString(Desc), *Offending));
		}
	}

	/** "'a' or 'b'". */
	static FString NamesToString(TConstArrayView<const TCHAR*> Names)
	{
		FString Out;
		for (int32 Index = 0; Index < Names.Num(); ++Index)
		{
			if (Index > 0)
			{
				Out += TEXT(" or ");
			}
			Out += FString::Printf(TEXT("'%s'"), Names[Index]);
		}
		return Out;
	}

	/**
	 * Checks one present output against float32 and Rank. Reports the mismatch and returns false on failure.
	 */
	static bool CheckOutputTypeRank(FContext& Ctx, const TCHAR* Name, const FPipelineTensorDesc& Desc, int32 Rank)
	{
		if (!MatchesKind(Desc, ETypeKind::Float))
		{
			Ctx.Error(ECode::OutputTypeMismatch, FString::Printf(TEXT("%s needs output '%s' as float32 (rank %d); the model has %s"),
				Ctx.TaskName, Name, Rank, *DescToString(Desc)));
			return false;
		}
		if (Desc.Shape.Num() != Rank)
		{
			Ctx.Error(ECode::OutputRankMismatch, FString::Printf(TEXT("%s needs output '%s' with rank %d (float32); the model has %s (rank %d)"),
				Ctx.TaskName, Name, Rank, *DescToString(Desc), Desc.Shape.Num()));
			return false;
		}
		return true;
	}

	/**
	 * Main output by preferred names, in order: the first preferred name present decides (bound if it is float32 of
	 * the rank, else a type/rank error and no fallback). No preferred name present: first unbound float32 rank-Rank
	 * output with a Warning, else MissingOutput.
	 */
	static int32 ResolveMainOutput(FContext& Ctx, TConstArrayView<const TCHAR*> Preferred, int32 Rank, const TArray<bool>& Bound)
	{
		for (const TCHAR* Name : Preferred)
		{
			const int32 Index = FindByName(Ctx.Outputs, Name);
			if (Index >= 0)
			{
				return CheckOutputTypeRank(Ctx, Name, Ctx.Outputs[Index], Rank) ? Index : -1;
			}
		}

		for (int32 Index = 0; Index < Ctx.Outputs.Num(); ++Index)
		{
			const FPipelineTensorDesc& Desc = Ctx.Outputs[Index];
			if (!Bound[Index] && MatchesKind(Desc, ETypeKind::Float) && Desc.Shape.Num() == Rank)
			{
				Ctx.Warning(ECode::OutputNameFallback, FString::Printf(TEXT("%s: no output named %s; using output %d '%s' (first float32 rank-%d output)"),
					Ctx.TaskName, *NamesToString(Preferred), Index, *DescToString(Desc), Rank));
				return Index;
			}
		}

		Ctx.Error(ECode::MissingOutput, FString::Printf(TEXT("%s needs output %s (float32, rank %d), or any float32 rank-%d output; the model has outputs: %s"),
			Ctx.TaskName, *NamesToString(Preferred), Rank, Rank, *ListToString(Ctx.Outputs)));
		return -1;
	}

	/** Classification: dim 1 of the main output, if fixed, must equal NumLabels (skipped for NumLabels <= 0). */
	static void CheckClassificationLabels(FContext& Ctx, int32 MainOutput, int32 NumLabels)
	{
		if (MainOutput < 0 || NumLabels <= 0)
		{
			return;
		}
		const FPipelineTensorDesc& Desc = Ctx.Outputs[MainOutput];
		if (Desc.Shape.Num() < 2)
		{
			return;
		}
		const int64 Classes = Desc.Shape[1];
		if (Classes != -1 && Classes != (int64)NumLabels)
		{
			Ctx.Error(ECode::LabelCountMismatch, FString::Printf(TEXT("%s: output %s has %lld classes in dim 1, but the package has %d labels"),
				Ctx.TaskName, *DescToString(Desc), (long long)Classes, NumLabels));
		}
	}

	/** Detection outputs: logits [B,Q,C] and pred_boxes [B,Q,4], no fallback. */
	static void ResolveDetectionOutputs(FContext& Ctx, EPipelineScoreActivation Activation, int32 NumLabels, FPipelineTaskBindings& B)
	{
		const TCHAR* Names[2] = {TEXT("logits"), TEXT("pred_boxes")};
		int32* Bindings[2] = {&B.MainOutput, &B.BoxesOutput};
		for (int32 Which = 0; Which < 2; ++Which)
		{
			const int32 Index = FindByName(Ctx.Outputs, Names[Which]);
			if (Index < 0)
			{
				Ctx.Error(ECode::MissingOutput, FString::Printf(TEXT("%s needs output '%s' (float32, rank 3); the model has outputs: %s"),
					Ctx.TaskName, Names[Which], *ListToString(Ctx.Outputs)));
				continue;
			}
			if (CheckOutputTypeRank(Ctx, Names[Which], Ctx.Outputs[Index], 3))
			{
				*Bindings[Which] = Index;
			}
		}

		if (B.BoxesOutput >= 0)
		{
			const FPipelineTensorDesc& Boxes = Ctx.Outputs[B.BoxesOutput];
			if (Boxes.Shape[2] != -1 && Boxes.Shape[2] != 4)
			{
				Ctx.Error(ECode::OutputDimMismatch, FString::Printf(TEXT("%s needs output 'pred_boxes' [B,Q,4] (last dim 4 or -1); the model has %s"),
					Ctx.TaskName, *DescToString(Boxes)));
			}
		}

		if (B.MainOutput >= 0 && B.BoxesOutput >= 0)
		{
			const FPipelineTensorDesc& Logits = Ctx.Outputs[B.MainOutput];
			const FPipelineTensorDesc& Boxes = Ctx.Outputs[B.BoxesOutput];
			if (Logits.Shape[1] != -1 && Boxes.Shape[1] != -1 && Logits.Shape[1] != Boxes.Shape[1])
			{
				Ctx.Error(ECode::OutputDimMismatch, FString::Printf(
					TEXT("%s needs the same query count (dim 1) in 'logits' and 'pred_boxes'; the model has %s and %s"),
					Ctx.TaskName, *DescToString(Logits), *DescToString(Boxes)));
			}
		}

		if (B.MainOutput >= 0 && NumLabels > 0)
		{
			const FPipelineTensorDesc& Logits = Ctx.Outputs[B.MainOutput];
			const int64 Classes = Logits.Shape[2];
			const bool bSigmoid = Activation == EPipelineScoreActivation::Sigmoid;
			const int64 Expected = bSigmoid ? (int64)NumLabels : (int64)NumLabels + 1;
			if (Classes != -1 && Classes != Expected)
			{
				Ctx.Error(ECode::LabelCountMismatch, FString::Printf(
					TEXT("%s: output %s has %lld classes in dim 2, but %d labels with %s need %lld%s"),
					Ctx.TaskName, *DescToString(Logits), (long long)Classes, NumLabels, bSigmoid ? TEXT("Sigmoid") : TEXT("Softmax"),
					(long long)Expected, bSigmoid ? TEXT("") : TEXT(" (labels + 1 no-object class)")));
			}
		}
	}
}

bool ResolvePipelineTaskIO(EPipelineTask Task, const FPipelinePreprocessSettings& Settings, const FPipelinePostprocessSettings& Postprocess,
	int32 NumLabels, TConstArrayView<FPipelineTensorDesc> Inputs, TConstArrayView<FPipelineTensorDesc> Outputs,
	FPipelineTaskBindings& OutBindings, FPipelineValidationReport& InOutReport)
{
	using namespace UE::ModelPipelines::TaskIOPrivate;

	OutBindings = FPipelineTaskBindings();

	FContext Ctx(InOutReport);
	Ctx.Inputs = Inputs;
	Ctx.Outputs = Outputs;

	const bool bText = IsTextTask(Task);
	const bool bImage = IsImageTask(Task);
	if (!bText && !bImage)
	{
		Ctx.Error(ECode::NoTask, FString::Printf(TEXT("Task is %s: no task I/O to resolve"), LexToString(Task)));
		return false;
	}
	Ctx.TaskName = LexToString(Task);

	TArray<bool> OutputBound;
	OutputBound.SetNumZeroed(Outputs.Num());

	if (bText)
	{
		const FInputSpec Specs[] = {
			{TEXT("input_ids"), true, ETypeKind::Int, 2, &OutBindings.InputIds},
			{TEXT("attention_mask"), true, ETypeKind::Int, 2, &OutBindings.AttentionMask},
			{TEXT("token_type_ids"), false, ETypeKind::Int, 2, &OutBindings.TokenTypeIds}};
		ResolveInputs(Ctx, Specs, [&Ctx, &Specs](int32 SpecIndex, int32 InputIndex)
		{
			CheckAllDynamic(Ctx, Specs[SpecIndex].Name, Ctx.Inputs[InputIndex]);
		});

		if (Task == EPipelineTask::TextEmbedding)
		{
			if (Settings.Pooling == EPipelinePooling::None)
			{
				const TCHAR* Preferred[] = {TEXT("sentence_embedding"), TEXT("pooler_output")};
				OutBindings.MainOutput = ResolveMainOutput(Ctx, Preferred, 2, OutputBound);
			}
			else
			{
				const TCHAR* Preferred[] = {TEXT("last_hidden_state"), TEXT("token_embeddings")};
				OutBindings.MainOutput = ResolveMainOutput(Ctx, Preferred, 3, OutputBound);
			}
		}
		else
		{
			const TCHAR* Preferred[] = {TEXT("logits")};
			OutBindings.MainOutput = ResolveMainOutput(Ctx, Preferred, 2, OutputBound);
			CheckClassificationLabels(Ctx, OutBindings.MainOutput, NumLabels);
		}
	}
	else
	{
		const bool bDetection = Task == EPipelineTask::ObjectDetection;
		TArray<FInputSpec, TInlineAllocator<2>> Specs;
		Specs.Add({TEXT("pixel_values"), true, ETypeKind::Float, 4, &OutBindings.PixelValues});
		if (bDetection)
		{
			Specs.Add({TEXT("pixel_mask"), false, ETypeKind::Int, 3, &OutBindings.PixelMask});
		}
		ResolveInputs(Ctx, Specs, [&Ctx, &Settings](int32 SpecIndex, int32 InputIndex)
		{
			if (SpecIndex == 0)
			{
				CheckPixelDims(Ctx, Ctx.Inputs[InputIndex], Settings.ImageWidth, Settings.ImageHeight);
			}
		});

		if (bDetection)
		{
			ResolveDetectionOutputs(Ctx, Postprocess.ScoreActivation, NumLabels, OutBindings);
		}
		else
		{
			const TCHAR* Preferred[] = {TEXT("logits")};
			OutBindings.MainOutput = ResolveMainOutput(Ctx, Preferred, 2, OutputBound);
			CheckClassificationLabels(Ctx, OutBindings.MainOutput, NumLabels);
		}
	}

	return Ctx.NumErrorsAdded == 0;
}
