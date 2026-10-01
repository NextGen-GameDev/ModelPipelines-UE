// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineModelValidator.h"

#include "Containers/ArrayView.h"
#include "Math/UnrealMathUtility.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineBackend.h"
#include "PipelineExecutor.h"
#include "PipelineModelPackage.h"
#include "PipelineTaskIO.h"
#include "PipelineTensor.h"
#include "PipelineEditorLog.h"
#include "TokenizerWrapper.h"

namespace PipelineModelValidatorPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;

	static FString DescribeShape(TConstArrayView<int64> Shape)
	{
		FString Out = TEXT("[");
		for (int32 i = 0; i < Shape.Num(); ++i)
		{
			if (i > 0)
			{
				Out += TEXT(",");
			}
			Out += FString::Printf(TEXT("%lld"), Shape[i]);
		}
		Out += TEXT("]");
		return Out;
	}

	/** Builds an integer tensor of the model input's type from int64 values. False for a non-integer type. */
	static bool MakeIntTensor(const FPipelineTensorDesc& Desc, TConstArrayView<int64> Shape, TConstArrayView<int64> Values,
		FPipelineTensor& Out, FString& OutError)
	{
		if (Desc.bTypeSupported && Desc.Type == EPipelineTensorType::Int64)
		{
			Out = FPipelineTensor::FromInt64(Shape, Values);
		}
		else if (Desc.bTypeSupported && Desc.Type == EPipelineTensorType::Int32)
		{
			TArray<int32> Narrow;
			Narrow.Reserve(Values.Num());
			for (const int64 V : Values)
			{
				Narrow.Add(static_cast<int32>(V));
			}
			Out = FPipelineTensor::FromInt32(Shape, Narrow);
		}
		else
		{
			OutError = FString::Printf(TEXT("input %s is not an integer tensor (int64 or int32 expected)"), *Desc.Name);
			return false;
		}
		if (!Out.IsValid())
		{
			OutError = FString::Printf(TEXT("could not build the smoke-run tensor for input %s with shape %s"), *Desc.Name, *DescribeShape(Shape));
			return false;
		}
		return true;
	}

	/** Smoke-run inputs in model input order (brief: text ids/mask/type ids, image pixel values/mask). */
	static bool BuildSmokeInputs(const UPipelineModelPackage& Package, TConstArrayView<FPipelineTensorDesc> InputDescs,
		const FPipelineTaskBindings& B, TConstArrayView<int32> Ids, TArray<FPipelineTensor>& OutInputs, FString& OutError)
	{
		OutInputs.Reset();
		const int64 L = Ids.Num();
		const int64 W = Package.Preprocess.ImageWidth;
		const int64 H = Package.Preprocess.ImageHeight;
		for (int32 Index = 0; Index < InputDescs.Num(); ++Index)
		{
			const FPipelineTensorDesc& Desc = InputDescs[Index];
			FPipelineTensor Tensor;
			if (Index == B.InputIds)
			{
				TArray<int64> Values;
				Values.Reserve(Ids.Num());
				for (const int32 Id : Ids)
				{
					Values.Add(Id);
				}
				const int64 Shape[] = { 1, L };
				if (!MakeIntTensor(Desc, Shape, Values, Tensor, OutError))
				{
					return false;
				}
			}
			else if (Index == B.AttentionMask || Index == B.TokenTypeIds)
			{
				TArray<int64> Values;
				Values.Init(Index == B.AttentionMask ? 1 : 0, static_cast<int32>(L));
				const int64 Shape[] = { 1, L };
				if (!MakeIntTensor(Desc, Shape, Values, Tensor, OutError))
				{
					return false;
				}
			}
			else if (Index == B.PixelValues)
			{
				if (W <= 0 || H <= 0 || W > 8192 || H > 8192)
				{
					OutError = FString::Printf(TEXT("image size %lldx%lld is invalid"), W, H);
					return false;
				}
				TArray<float> Values;
				Values.Init(0.5f, static_cast<int32>(3 * H * W));
				const int64 Shape[] = { 1, 3, H, W };
				Tensor = FPipelineTensor::FromFloat(Shape, Values);
				if (!Tensor.IsValid())
				{
					OutError = FString::Printf(TEXT("could not build the smoke-run tensor for input %s with shape %s"), *Desc.Name, *DescribeShape(Shape));
					return false;
				}
			}
			else if (Index == B.PixelMask)
			{
				if (W <= 0 || H <= 0 || W > 8192 || H > 8192)
				{
					OutError = FString::Printf(TEXT("image size %lldx%lld is invalid"), W, H);
					return false;
				}
				TArray<int64> Values;
				Values.Init(1, static_cast<int32>(H * W));
				const int64 Shape[] = { 1, H, W };
				if (!MakeIntTensor(Desc, Shape, Values, Tensor, OutError))
				{
					return false;
				}
			}
			else
			{
				OutError = FString::Printf(TEXT("model input %s is not fed by the task"), *Desc.Name);
				return false;
			}
			OutInputs.Add(MoveTemp(Tensor));
		}
		return true;
	}

	/** Checks rank and each expected dim (-1 = any positive value). */
	static bool CheckShape(const FPipelineTensor& T, const TCHAR* Name, TConstArrayView<int64> Expected, FString& OutError)
	{
		const TConstArrayView<int64> Shape = T.GetShape();
		bool bOk = Shape.Num() == Expected.Num();
		for (int32 i = 0; bOk && i < Expected.Num(); ++i)
		{
			bOk = Expected[i] < 0 ? Shape[i] > 0 : Shape[i] == Expected[i];
		}
		if (!bOk)
		{
			FString ExpectedText = TEXT("[");
			for (int32 i = 0; i < Expected.Num(); ++i)
			{
				ExpectedText += (i > 0 ? TEXT(",") : TEXT(""));
				ExpectedText += Expected[i] < 0 ? FString(TEXT("N")) : FString::Printf(TEXT("%lld"), Expected[i]);
			}
			ExpectedText += TEXT("]");
			OutError = FString::Printf(TEXT("output %s has shape %s, expected %s"), Name, *T.DescribeShape(), *ExpectedText);
		}
		return bOk;
	}

	/** Every value of a float output is finite. */
	static bool CheckFinite(const FPipelineTensor& T, const TCHAR* Name, FString& OutError)
	{
		const TConstArrayView64<float> Data = T.GetFloatData();
		if (Data.Num() != T.Num())
		{
			OutError = FString::Printf(TEXT("output %s is %s, float32 expected"), Name, LexToString(T.GetType()));
			return false;
		}
		for (int64 i = 0; i < Data.Num(); ++i)
		{
			if (!FMath::IsFinite(Data[i]))
			{
				OutError = FString::Printf(TEXT("output %s has a non-finite value at flat index %lld"), Name, i);
				return false;
			}
		}
		return true;
	}

	/** Shape and finiteness checks of the smoke-run outputs (brief step 5). */
	static bool CheckSmokeOutputs(const UPipelineModelPackage& Package, const FPipelineTaskBindings& B,
		TConstArrayView<FPipelineTensorDesc> OutputDescs, int64 SeqLen, TConstArrayView<FPipelineTensor> Outputs, FString& OutError)
	{
		if (Outputs.Num() != OutputDescs.Num())
		{
			OutError = FString::Printf(TEXT("the run returned %d outputs, the model declares %d"), Outputs.Num(), OutputDescs.Num());
			return false;
		}
		if (!Outputs.IsValidIndex(B.MainOutput))
		{
			OutError = TEXT("the task's main output is not bound");
			return false;
		}
		const FPipelineTensor& Main = Outputs[B.MainOutput];
		const FString MainName = OutputDescs[B.MainOutput].Name;
		const int64 NumLabels = Package.Labels.Num();

		switch (Package.Task)
		{
		case EPipelineTask::TextClassification:
		case EPipelineTask::ImageClassification:
		{
			const int64 Expected[] = { 1, NumLabels };
			if (!CheckShape(Main, *MainName, Expected, OutError))
			{
				return false;
			}
			break;
		}
		case EPipelineTask::TextEmbedding:
		{
			if (Main.GetRank() == 3)
			{
				const int64 Expected[] = { 1, SeqLen, -1 };
				if (!CheckShape(Main, *MainName, Expected, OutError))
				{
					return false;
				}
			}
			else
			{
				const int64 Expected[] = { 1, -1 };
				if (!CheckShape(Main, *MainName, Expected, OutError))
				{
					OutError += TEXT(" (or [1,L,N])");
					return false;
				}
			}
			break;
		}
		case EPipelineTask::ObjectDetection:
		{
			if (!Outputs.IsValidIndex(B.BoxesOutput))
			{
				OutError = TEXT("the detection boxes output is not bound");
				return false;
			}
			const int64 NumClasses = NumLabels + (Package.Postprocess.ScoreActivation == EPipelineScoreActivation::Softmax ? 1 : 0);
			const int64 ExpectedLogits[] = { 1, -1, NumClasses };
			if (!CheckShape(Main, *MainName, ExpectedLogits, OutError))
			{
				return false;
			}
			const FPipelineTensor& Boxes = Outputs[B.BoxesOutput];
			const FString BoxesName = OutputDescs[B.BoxesOutput].Name;
			const int64 ExpectedBoxes[] = { 1, Main.GetShape()[1], 4 };
			if (!CheckShape(Boxes, *BoxesName, ExpectedBoxes, OutError))
			{
				return false;
			}
			if (!CheckFinite(Boxes, *BoxesName, OutError))
			{
				return false;
			}
			break;
		}
		default:
			OutError = FString::Printf(TEXT("task %s has no smoke-run check"), LexToString(Package.Task));
			return false;
		}
		return CheckFinite(Main, *MainName, OutError);
	}

	/** Outcome of one smoke run. */
	struct FSmokeOutcome
	{
		bool bRan = false;
		float InferenceMs = 0.f;
		ECode Code = ECode::None;   // SmokeRunFailed or SmokeOutputInvalid on failure
		FString Message;            // without the device prefix
	};

	static FSmokeOutcome RunSmoke(const TSharedPtr<IPipelineModel>& Model, const UPipelineModelPackage& Package,
		const FPipelineTaskBindings& B, TConstArrayView<int32> Ids)
	{
		FSmokeOutcome Out;
		const TArray<FPipelineTensorDesc> InputDescs = Model->GetInputDescs();
		const TArray<FPipelineTensorDesc> OutputDescs = Model->GetOutputDescs();

		TArray<FPipelineTensor> Inputs;
		FString Error;
		if (!BuildSmokeInputs(Package, InputDescs, B, Ids, Inputs, Error))
		{
			Out.Code = ECode::SmokeRunFailed;
			Out.Message = FString::Printf(TEXT("smoke run not started: %s"), *Error);
			return Out;
		}

		TArray<FPipelineTensor> Outputs;
		const FPipelineResult Result = FPipelineExecutor::Get().RunBlocking(Model, Inputs, Outputs);
		Out.InferenceMs = Result.InferenceMs;
		if (!Result.bSuccess)
		{
			Out.Code = ECode::SmokeRunFailed;
			Out.Message = FString::Printf(TEXT("smoke run failed: %s"), *Result.Message);
			return Out;
		}
		if (!CheckSmokeOutputs(Package, B, OutputDescs, Ids.Num(), Outputs, Error))
		{
			Out.Code = ECode::SmokeOutputInvalid;
			Out.Message = FString::Printf(TEXT("smoke run output invalid: %s"), *Error);
			return Out;
		}
		Out.bRan = true;
		return Out;
	}

	static FPipelineDeviceCheck& AddCheck(FPipelineValidationReport& Report, EPipelineDevice Device, bool bAvailable)
	{
		FPipelineDeviceCheck& Check = Report.DeviceChecks.AddDefaulted_GetRef();
		Check.Device = Device;
		Check.bAvailable = bAvailable;
		return Check;
	}
}

FPipelineValidationReport ValidatePipelineModelPackage(const UPipelineModelPackage* Package, const FPipelineValidateOptions& Options)
{
	if (!IsInGameThread())
	{
		FPipelineValidationReport Report;
		Report.Add(EPipelineValidationSeverity::Error, EPipelineValidationCode::None, TEXT("ValidatePipelineModelPackage must run on the game thread"));
		return Report;
	}
	return ValidatePipelineModelPackage(Package, Options, FModelPipelinesCoreModule::Get().GetDefaultBackend());
}

FPipelineValidationReport ValidatePipelineModelPackage(const UPipelineModelPackage* Package, const FPipelineValidateOptions& Options,
	IPipelineBackend& Backend)
{
	using namespace PipelineModelValidatorPrivate;

	FPipelineValidationReport Report;
	if (!IsInGameThread())
	{
		Report.Add(ESev::Error, ECode::None, TEXT("ValidatePipelineModelPackage must run on the game thread"));
		return Report;
	}

	// Step 1: package and settings.
	if (Package == nullptr)
	{
		Report.Add(ESev::Error, ECode::NoPackage, TEXT("No model package was given to the validator."));
		return Report;
	}
	Package->ValidateSettings(Report);
	if (Report.HasError(ECode::NoTask) || Report.HasError(ECode::NoModel) || Package->Model == nullptr)
	{
		if (!Report.HasCode(ECode::NoTask) && !Report.HasCode(ECode::NoModel))
		{
			Report.Add(ESev::Error, ECode::NoModel, TEXT("The package has no model. Re-import the ONNX model."));
		}
		return Report;
	}
	const EPipelineTask Task = Package->Task;

	// Step 2: tokenizer (text tasks only).
	TArray<int32> Ids;
	if (IsTextTask(Task))
	{
		TStrongObjectPtr<UTokenizerWrapper> Tokenizer(NewObject<UTokenizerWrapper>(GetTransientPackage()));
		if (!Tokenizer.IsValid() || !Tokenizer->InitializeTokenizerFromJson(Package->TokenizerJson))
		{
			const FString Why = Tokenizer.IsValid() ? Tokenizer->GetLastError() : FString(TEXT("could not create the tokenizer object"));
			Report.Add(ESev::Error, ECode::TokenizerInvalid, FString::Printf(
				TEXT("TokenizerJson does not load as a tokenizer: %s. Re-import the model folder with its tokenizer.json."), *Why));
			return Report;
		}
		Ids = Tokenizer->Encode(TEXT("Hello world"), Package->Preprocess.bAddSpecialTokens);
		if (Ids.Num() == 0)
		{
			Report.Add(ESev::Error, ECode::TokenizerInvalid, FString::Printf(
				TEXT("The tokenizer returns no ids for \"Hello world\" (%s). Check the package's tokenizer.json."), *Tokenizer->GetLastError()));
			return Report;
		}
		const int32 MaxTokens = FMath::Max(1, Package->Preprocess.MaxTokens);
		if (Ids.Num() > MaxTokens)
		{
			Ids.SetNum(MaxTokens);
		}
	}

	// Step 3: create the model on the CPU runtime.
	const int32 CpuIndex = Report.DeviceChecks.Num();
	AddCheck(Report, EPipelineDevice::CPU, Backend.IsDeviceAvailable(EPipelineDevice::CPU));
	if (!Report.DeviceChecks[CpuIndex].bAvailable)
	{
		const FString Msg = TEXT("CPU: no CPU runtime is registered (NNERuntimeORTCpu). Enable the NNERuntimeORT plugin.");
		Report.DeviceChecks[CpuIndex].Message = Msg;
		Report.Add(ESev::Error, ECode::RuntimeUnavailable, Msg);
		return Report;
	}
	UNNEModelData* const ModelData = Package->Model.Get();
	FPipelineResult PrepareResult;
	TSharedPtr<IPipelineModel> CpuModel = Backend.PrepareModel(ModelData, EPipelineDevice::CPU, PrepareResult);
	FPipelineResult InitResult = CpuModel.IsValid() ? CpuModel->Initialize() : PrepareResult;
	if (!CpuModel.IsValid() || !InitResult.bSuccess)
	{
		FString Reason = InitResult.Message.IsEmpty() ? FString(TEXT("the model could not be created")) : InitResult.Message;
		// The backend may wrap the device reason as "... (CPU: <reason>)"; use the text inside that parenthesis.
		const int32 InnerStart = Reason.Find(TEXT("(CPU: "), ESearchCase::CaseSensitive);
		if (InnerStart != INDEX_NONE)
		{
			const int32 TextStart = InnerStart + 6; // after "(CPU: "
			int32 TextEnd = INDEX_NONE;
			int32 Depth = 1;
			for (int32 i = TextStart; i < Reason.Len(); ++i)
			{
				if (Reason[i] == TEXT('('))
				{
					++Depth;
				}
				else if (Reason[i] == TEXT(')') && --Depth == 0)
				{
					TextEnd = i;
					break;
				}
			}
			const FString Inner = (TextEnd == INDEX_NONE ? Reason.Mid(TextStart) : Reason.Mid(TextStart, TextEnd - TextStart)).TrimStartAndEnd();
			if (!Inner.IsEmpty())
			{
				Reason = Inner;
			}
		}
		// The (remaining) message may already carry the device prefix; keep it once.
		const bool bHasPrefix = Reason.StartsWith(TEXT("CPU: "));
		const FString Msg = bHasPrefix ? Reason : FString::Printf(TEXT("CPU: %s"), *Reason);
		Report.DeviceChecks[CpuIndex].Message = Msg;
		Report.Add(ESev::Error, ECode::ModelCreateFailed, Msg);
		return Report;
	}
	Report.DeviceChecks[CpuIndex].bCreated = true;

	// Step 4: task I/O against the created model.
	const TArray<FPipelineTensorDesc> InputDescs = CpuModel->GetInputDescs();
	const TArray<FPipelineTensorDesc> OutputDescs = CpuModel->GetOutputDescs();
	FPipelineTaskBindings Bindings;
	if (!ResolvePipelineTaskIO(Task, Package->Preprocess, Package->Postprocess, Package->Labels.Num(), InputDescs, OutputDescs, Bindings, Report))
	{
		Report.DeviceChecks[CpuIndex].Message = TEXT("CPU: smoke run skipped (the model's inputs/outputs do not fit the task)");
		return Report;
	}

	// Step 5: CPU smoke run.
	if (Options.bRunSmokeTest)
	{
		const FSmokeOutcome Smoke = RunSmoke(CpuModel, *Package, Bindings, Ids);
		FPipelineDeviceCheck& Cpu = Report.DeviceChecks[CpuIndex];
		Cpu.bRan = Smoke.bRan;
		Cpu.InferenceMs = Smoke.InferenceMs;
		if (!Smoke.bRan)
		{
			const FString Msg = FString::Printf(TEXT("CPU: %s"), *Smoke.Message);
			Cpu.Message = Msg;
			Report.Add(ESev::Error, Smoke.Code, Msg);
		}
	}
	CpuModel.Reset();

	// Step 6: GPU.
	if (Options.bCheckGpu)
	{
		const int32 GpuIndex = Report.DeviceChecks.Num();
		AddCheck(Report, EPipelineDevice::GPU, Backend.IsDeviceAvailable(EPipelineDevice::GPU));
		if (!Report.DeviceChecks[GpuIndex].bAvailable)
		{
			const FString Msg = TEXT("GPU: no GPU runtime");
			Report.DeviceChecks[GpuIndex].Message = Msg;
			Report.Add(ESev::Info, ECode::RuntimeUnavailable, Msg);
		}
		else
		{
			FPipelineResult GpuPrepare;
			TSharedPtr<IPipelineModel> GpuModel = Backend.PrepareModel(ModelData, EPipelineDevice::GPU, GpuPrepare);
			const FPipelineResult GpuInit = GpuModel.IsValid() ? GpuModel->Initialize() : GpuPrepare;
			if (!GpuModel.IsValid() || !GpuInit.bSuccess || GpuModel->GetDevice() != EPipelineDevice::GPU)
			{
				FString Msg;
				if (GpuModel.IsValid() && GpuInit.bSuccess)
				{
					// Created, but the runtime placed it on the CPU.
					const FString Reason = GpuInit.Message.IsEmpty() ? FString(TEXT("the GPU runtime fell back to the CPU")) : GpuInit.Message;
					Msg = FString::Printf(TEXT("GPU: could not create the model; it runs on CPU (%s)"), *Reason);
				}
				else
				{
					const FString Reason = GpuInit.Message.IsEmpty() ? FString(TEXT("the model could not be created")) : GpuInit.Message;
					Msg = FString::Printf(TEXT("GPU: could not create the model (%s)"), *Reason);
				}
				Report.DeviceChecks[GpuIndex].Message = Msg;
				Report.Add(ESev::Warning, ECode::ModelCreateFailed, Msg);
			}
			else
			{
				Report.DeviceChecks[GpuIndex].bCreated = true;
				if (Options.bRunSmokeTest)
				{
					const FSmokeOutcome Smoke = RunSmoke(GpuModel, *Package, Bindings, Ids);
					FPipelineDeviceCheck& Gpu = Report.DeviceChecks[GpuIndex];
					Gpu.bRan = Smoke.bRan;
					Gpu.InferenceMs = Smoke.InferenceMs;
					if (!Smoke.bRan)
					{
						const FString Msg = FString::Printf(TEXT("GPU: %s"), *Smoke.Message);
						Gpu.Message = Msg;
						const ESev Sev = Package->PreferredDevice != EPipelineDevice::CPU ? ESev::Error : ESev::Warning;
						Report.Add(Sev, Smoke.Code, Msg);
					}
				}
			}
		}
	}

	UE_LOG(LogModelPipelinesEditor, Verbose, TEXT("Validated %s: %d errors, %d warnings."), *Package->GetPathName(), Report.NumErrors(), Report.NumWarnings());
	return Report;
}
