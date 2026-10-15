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

	/** input_features [B,T,M]: B 1 or -1, T any, M NumMelBins or -1. One error message per tensor. */
	static void CheckFeatureDims(FContext& Ctx, const FPipelineTensorDesc& Desc, int32 NumMelBins)
	{
		const TArray<int64>& S = Desc.Shape;
		if (S.Num() != 3)
		{
			return;
		}
		FString Offending;
		if (S[0] != 1 && S[0] != -1)
		{
			Offending = FString::Printf(TEXT("dim 0 (batch) = %lld"), (long long)S[0]);
		}
		if (S[2] != -1 && S[2] != (int64)NumMelBins)
		{
			if (!Offending.IsEmpty())
			{
				Offending += TEXT(", ");
			}
			Offending += FString::Printf(TEXT("dim 2 (mel bins) = %lld"), (long long)S[2]);
		}
		if (!Offending.IsEmpty())
		{
			Ctx.Error(ECode::InputDimMismatch, FString::Printf(
				TEXT("%s needs input 'input_features' [batch 1 or -1, frames, mel bins %d] (mel bins may be -1); the model has %s (%s)"),
				Ctx.TaskName, NumMelBins, *DescToString(Desc), *Offending));
		}
	}

	/** True when Dim is -1 or Expected. */
	static bool DimMatches(int64 Dim, int64 Expected)
	{
		return Dim == -1 || Dim == Expected;
	}
}

bool ResolvePipelineTaskIO(EPipelineTask Task, const FPipelinePreprocessSettings& Settings, const FPipelinePostprocessSettings& Postprocess,
	int32 NumLabels, TConstArrayView<FPipelineTensorDesc> Inputs, TConstArrayView<FPipelineTensorDesc> Outputs,
	FPipelineTaskBindings& OutBindings, FPipelineValidationReport& InOutReport)
{
	return ResolvePipelineTaskIO(Task, Settings, Postprocess, FPipelineSpeechSettings(), NumLabels, Inputs, Outputs, OutBindings, InOutReport);
}

bool ResolvePipelineTaskIO(EPipelineTask Task, const FPipelinePreprocessSettings& Settings, const FPipelinePostprocessSettings& Postprocess,
	const FPipelineSpeechSettings& Speech, int32 NumLabels, TConstArrayView<FPipelineTensorDesc> Inputs,
	TConstArrayView<FPipelineTensorDesc> Outputs, FPipelineTaskBindings& OutBindings, FPipelineValidationReport& InOutReport)
{
	using namespace UE::ModelPipelines::TaskIOPrivate;

	OutBindings = FPipelineTaskBindings();

	FContext Ctx(InOutReport);
	Ctx.Inputs = Inputs;
	Ctx.Outputs = Outputs;

	const bool bText = IsTextTask(Task);
	const bool bImage = IsImageTask(Task);
	const bool bSpeech = Task == EPipelineTask::SpeechToText;
	if (!bText && !bImage && !bSpeech)
	{
		Ctx.Error(ECode::NoTask, FString::Printf(TEXT("Task is %s: no task I/O to resolve"), LexToString(Task)));
		return false;
	}
	Ctx.TaskName = LexToString(Task);

	TArray<bool> OutputBound;
	OutputBound.SetNumZeroed(Outputs.Num());

	if (bSpeech)
	{
		const FInputSpec Specs[] = {{TEXT("input_features"), true, ETypeKind::Float, 3, &OutBindings.InputFeatures}};
		ResolveInputs(Ctx, Specs, [&Ctx, &Speech](int32 SpecIndex, int32 InputIndex)
		{
			CheckFeatureDims(Ctx, Ctx.Inputs[InputIndex], Speech.NumMelBins);
		});

		const TCHAR* Preferred[] = {TEXT("encoder_projected")};
		OutBindings.MainOutput = ResolveMainOutput(Ctx, Preferred, 3, OutputBound);
		if (OutBindings.MainOutput >= 0)
		{
			const FPipelineTensorDesc& Desc = Outputs[OutBindings.MainOutput];
			if (!DimMatches(Desc.Shape[0], 1))
			{
				Ctx.Error(ECode::OutputDimMismatch, FString::Printf(
					TEXT("%s needs output '%s' [batch 1 or -1, frames, dim]; the model has %s (dim 0 (batch) = %lld)"),
					Ctx.TaskName, *Desc.Name, *DescToString(Desc), (long long)Desc.Shape[0]));
			}
		}
	}
	else if (bText)
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

bool ResolveSpeechDecoderIO(const FPipelineSpeechSettings& Speech, int32 EncoderDim, TConstArrayView<FPipelineTensorDesc> Inputs,
	TConstArrayView<FPipelineTensorDesc> Outputs, FPipelineSpeechDecoderBindings& OutBindings, FPipelineValidationReport& InOutReport)
{
	using namespace UE::ModelPipelines::TaskIOPrivate;

	OutBindings = FPipelineSpeechDecoderBindings();

	FContext Ctx(InOutReport);
	Ctx.TaskName = TEXT("SpeechToText decoder");
	Ctx.Inputs = Inputs;
	Ctx.Outputs = Outputs;

	const FInputSpec Specs[] = {
		{TEXT("encoder_frame"), true, ETypeKind::Float, 2, &OutBindings.EncoderFrame},
		{TEXT("token"), true, ETypeKind::Int, 1, &OutBindings.Token},
		{TEXT("state_h"), true, ETypeKind::Float, 3, &OutBindings.StateH},
		{TEXT("state_c"), true, ETypeKind::Float, 3, &OutBindings.StateC}};
	ResolveInputs(Ctx, Specs, [&Ctx, EncoderDim, &OutBindings](int32 SpecIndex, int32 InputIndex)
	{
		const FPipelineTensorDesc& Desc = Ctx.Inputs[InputIndex];
		if (SpecIndex == 0)
		{
			const bool bBatchOk = DimMatches(Desc.Shape[0], 1);
			const bool bDimOk = EncoderDim <= 0 || DimMatches(Desc.Shape[1], EncoderDim);
			if (!bBatchOk || !bDimOk)
			{
				const FString Expected = EncoderDim > 0 ? FString::Printf(TEXT("%d (encoder dim) or -1"), EncoderDim) : FString(TEXT("any"));
				Ctx.Error(ECode::InputDimMismatch, FString::Printf(TEXT("%s needs input 'encoder_frame' [1 or -1, %s]; the model has %s"),
					Ctx.TaskName, *Expected, *DescToString(Desc)));
			}
		}
		else if (SpecIndex == 1)
		{
			// token is int64 only (the shared Int kind also accepts int32).
			if (Desc.Type != EPipelineTensorType::Int64)
			{
				Ctx.Error(ECode::InputTypeMismatch, FString::Printf(TEXT("%s needs input 'token' as int64 (rank 1); the model has %s"),
					Ctx.TaskName, *DescToString(Desc)));
				OutBindings.Token = -1;
			}
			else if (!DimMatches(Desc.Shape[0], 1))
			{
				Ctx.Error(ECode::InputDimMismatch, FString::Printf(TEXT("%s needs input 'token' [1 or -1]; the model has %s"),
					Ctx.TaskName, *DescToString(Desc)));
			}
		}
	});

	if (OutBindings.StateH >= 0 && OutBindings.StateC >= 0 && Inputs[OutBindings.StateH].Shape != Inputs[OutBindings.StateC].Shape)
	{
		Ctx.Error(ECode::InputDimMismatch, FString::Printf(TEXT("%s needs inputs 'state_h' and 'state_c' with equal dims; the model has %s and %s"),
			Ctx.TaskName, *DescToString(Inputs[OutBindings.StateH]), *DescToString(Inputs[OutBindings.StateC])));
	}

	struct FOutputSpec
	{
		const TCHAR* Name;
		int32 Rank;
		int32* Binding;
	};
	const FOutputSpec OutSpecs[] = {
		{TEXT("token_logits"), 2, &OutBindings.TokenLogits},
		{TEXT("duration_logits"), 2, &OutBindings.DurationLogits},
		{TEXT("state_h_out"), 3, &OutBindings.StateHOut},
		{TEXT("state_c_out"), 3, &OutBindings.StateCOut}};
	for (const FOutputSpec& Spec : OutSpecs)
	{
		const int32 Index = FindByName(Outputs, Spec.Name);
		if (Index < 0)
		{
			Ctx.Error(ECode::MissingOutput, FString::Printf(TEXT("%s needs output '%s' (float32, rank %d); the model has outputs: %s"),
				Ctx.TaskName, Spec.Name, Spec.Rank, *ListToString(Outputs)));
			continue;
		}
		if (CheckOutputTypeRank(Ctx, Spec.Name, Outputs[Index], Spec.Rank))
		{
			*Spec.Binding = Index;
		}
	}

	auto CheckLogits = [&Ctx, &Outputs](int32 Index, int64 Expected, const TCHAR* What)
	{
		if (Index < 0)
		{
			return;
		}
		const FPipelineTensorDesc& Desc = Outputs[Index];
		if (!DimMatches(Desc.Shape[0], 1) || !DimMatches(Desc.Shape[1], Expected))
		{
			Ctx.Error(ECode::OutputDimMismatch, FString::Printf(TEXT("%s needs output '%s' [1 or -1, %lld (%s) or -1]; the model has %s"),
				Ctx.TaskName, *Desc.Name, (long long)Expected, What, *DescToString(Desc)));
		}
	};
	CheckLogits(OutBindings.TokenLogits, (int64)Speech.BlankTokenId + 1, TEXT("BlankTokenId + 1"));
	CheckLogits(OutBindings.DurationLogits, (int64)Speech.Durations.Num(), TEXT("Durations count"));

	auto CheckStateOut = [&Ctx, &Inputs, &Outputs](int32 OutIndex, int32 InIndex)
	{
		if (OutIndex < 0 || InIndex < 0)
		{
			return;
		}
		if (Outputs[OutIndex].Shape != Inputs[InIndex].Shape)
		{
			Ctx.Error(ECode::OutputDimMismatch, FString::Printf(TEXT("%s needs output '%s' with the dims of input %s; the model has %s"),
				Ctx.TaskName, *Outputs[OutIndex].Name, *DescToString(Inputs[InIndex]), *DescToString(Outputs[OutIndex])));
		}
	};
	CheckStateOut(OutBindings.StateHOut, OutBindings.StateH);
	CheckStateOut(OutBindings.StateCOut, OutBindings.StateC);

	return Ctx.NumErrorsAdded == 0;
}
