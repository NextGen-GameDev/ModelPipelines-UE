// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineModelValidator.h"

#include "Containers/ArrayView.h"
#include "Engine/EngineTypes.h"
#include "HAL/FileManager.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/Paths.h"
#include "Settings/ProjectPackagingSettings.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineBackend.h"
#include "PipelineExecutor.h"
#include "PipelineFileHash.h"
#include "PipelineModelPackage.h"
#include "PipelineModelSource.h"
#include "PipelineTaskIO.h"
#include "PipelineTensor.h"
#include "PipelineEditorLog.h"
#include "PipelineTokenizer.h"
#include "PipelineSession.h"
#include "SpeechEngine.h"
#include "SpeechTypes.h"

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
			else if (Index == B.InputFeatures)
			{
				// SpeechToText encoder (C78): 200 zero feature frames of NumMelBins.
				const int64 Mel = Package.Speech.NumMelBins;
				if (Mel <= 0 || Mel > 4096)
				{
					OutError = FString::Printf(TEXT("NumMelBins %lld is invalid"), Mel);
					return false;
				}
				const int64 Shape[] = { 1, 200, Mel };
				Tensor = FPipelineTensor::Zeros(EPipelineTensorType::Float32, Shape);
				if (!Tensor.IsValid())
				{
					OutError = FString::Printf(TEXT("could not build the smoke-run tensor for input %s with shape %s"), *Desc.Name, *DescribeShape(Shape));
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
		case EPipelineTask::SpeechToText:
		{
			// encoder_projected [1, F, D], F >= 1 (C78).
			const int64 Expected[] = { 1, -1, -1 };
			if (!CheckShape(Main, *MainName, Expected, OutError))
			{
				return false;
			}
			if (Main.GetShape()[1] < 1 || Main.GetShape()[2] < 1)
			{
				OutError = FString::Printf(TEXT("output %s has shape %s, at least one frame of at least one value expected"), *MainName, *DescribeShape(Main.GetShape()));
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
		FString Reason;             // the bare failure reason (no "smoke run ..." wording)
	};

	static FSmokeOutcome RunSmoke(const TSharedPtr<IPipelineModel>& Model, const UPipelineModelPackage& Package,
		const FPipelineTaskBindings& B, TConstArrayView<int32> Ids, TArray<FPipelineTensor>* OutOutputs = nullptr)
	{
		FSmokeOutcome Out;
		const TArray<FPipelineTensorDesc> InputDescs = Model->GetInputDescs();
		const TArray<FPipelineTensorDesc> OutputDescs = Model->GetOutputDescs();

		TArray<FPipelineTensor> Inputs;
		FString Error;
		if (!BuildSmokeInputs(Package, InputDescs, B, Ids, Inputs, Error))
		{
			Out.Code = ECode::SmokeRunFailed;
			Out.Reason = Error;
			Out.Message = FString::Printf(TEXT("smoke run not started: %s"), *Error);
			return Out;
		}

		TArray<FPipelineTensor> Outputs;
		const FPipelineResult Result = FPipelineExecutor::Get().RunBlocking(Model, Inputs, Outputs);
		Out.InferenceMs = Result.InferenceMs;
		if (!Result.bSuccess)
		{
			Out.Code = ECode::SmokeRunFailed;
			Out.Reason = Result.Message.IsEmpty() ? FString(TEXT("the run failed")) : Result.Message;
			Out.Message = FString::Printf(TEXT("smoke run failed: %s"), *Result.Message);
			return Out;
		}
		if (!CheckSmokeOutputs(Package, B, OutputDescs, Ids.Num(), Outputs, Error))
		{
			Out.Code = ECode::SmokeOutputInvalid;
			Out.Reason = Error;
			Out.Message = FString::Printf(TEXT("smoke run output invalid: %s"), *Error);
			return Out;
		}
		Out.bRan = true;
		if (OutOutputs)
		{
			*OutOutputs = MoveTemp(Outputs);
		}
		return Out;
	}

	/**
	 * SpeechToText decoder_joint check (C78), always on CPU: create, resolve its I/O, and (with an encoder smoke output)
	 * one decoder step with the first encoder frame, the blank token and zero states.
	 */
	static void ValidateSpeechDecoder(const UPipelineModelPackage& Package, IPipelineBackend& Backend, int32 EncoderDim,
		const FPipelineTensor* EncoderOut, FPipelineValidationReport& Report)
	{
		UNNEModelData* const DecoderData = Package.DecoderModel.Get();
		if (DecoderData == nullptr)
		{
			if (!Report.HasError(ECode::NoDecoderModel))
			{
				Report.Add(ESev::Error, ECode::NoDecoderModel, TEXT("DecoderModel is empty; a SpeechToText package needs the decoder_joint model. Re-import the model folder."));
			}
			return;
		}
		FPipelineResult PrepareResult;
		TSharedPtr<IPipelineModel> Decoder = Backend.PrepareModel(DecoderData, EPipelineDevice::CPU, PrepareResult);
		const FPipelineResult InitResult = Decoder.IsValid() ? Decoder->Initialize() : PrepareResult;
		if (!Decoder.IsValid() || !InitResult.bSuccess)
		{
			const FString Reason = InitResult.Message.IsEmpty() ? FString(TEXT("the model could not be created")) : InitResult.Message;
			Report.Add(ESev::Error, ECode::ModelCreateFailed, FString::Printf(
				TEXT("DecoderModel: %s. Re-export decoder_joint.onnx and re-import the model folder."), *Reason));
			return;
		}
		const TArray<FPipelineTensorDesc> InputDescs = Decoder->GetInputDescs();
		const TArray<FPipelineTensorDesc> OutputDescs = Decoder->GetOutputDescs();
		FPipelineSpeechDecoderBindings DB;
		if (!ResolveSpeechDecoderIO(Package.Speech, EncoderDim, InputDescs, OutputDescs, DB, Report))
		{
			return;
		}
		if (EncoderOut == nullptr)
		{
			return;
		}

		const TConstArrayView<int64> EncShape = EncoderOut->GetShape();
		const TConstArrayView64<float> EncData = EncoderOut->GetFloatData();
		const int64 D = EncShape.Num() == 3 ? EncShape[2] : 0;
		if (D <= 0 || EncData.Num() < D)
		{
			Report.Add(ESev::Error, ECode::SmokeOutputInvalid, TEXT("decoder: no encoder frame to feed (the encoder output is empty)."));
			return;
		}
		// Smoke-run tensors are bounded like the Mel guard: a corrupt or hostile decoder signature must not allocate gigabytes.
		constexpr int64 MaxSmokeElements = 16 * 1024 * 1024;
		if (D > MaxSmokeElements)
		{
			Report.Add(ESev::Error, ECode::SmokeRunFailed, FString::Printf(
				TEXT("decoder: encoder frame of %lld elements is too large for the smoke run (limit %lld). Check the encoder output's last dimension and re-export the model."),
				D, MaxSmokeElements));
			return;
		}
		TArray<FPipelineTensor> Inputs;
		for (int32 Index = 0; Index < InputDescs.Num(); ++Index)
		{
			const FPipelineTensorDesc& Desc = InputDescs[Index];
			FPipelineTensor Tensor;
			if (Index == DB.EncoderFrame)
			{
				const int64 Shape[] = { 1, D };
				Tensor = FPipelineTensor::FromFloat(Shape, TConstArrayView<float>(EncData.GetData(), static_cast<int32>(D)));
			}
			else if (Index == DB.Token)
			{
				const int64 Shape[] = { 1 };
				const int64 Values[] = { static_cast<int64>(Package.Speech.BlankTokenId) };
				Tensor = FPipelineTensor::FromInt64(Shape, Values);
			}
			else if (Index == DB.StateH || Index == DB.StateC)
			{
				TArray<int64> Shape = Desc.Shape;
				int64 Elements = 1;
				for (int64& Dim : Shape)
				{
					Dim = Dim < 1 ? 1 : Dim;
					// Elements <= MaxSmokeElements before the multiply, so the check below cannot overflow.
					Elements = Dim > MaxSmokeElements / Elements ? MaxSmokeElements + 1 : Elements * Dim;
				}
				if (Elements > MaxSmokeElements)
				{
					Report.Add(ESev::Error, ECode::SmokeRunFailed, FString::Printf(
						TEXT("decoder: state %s is too large for the smoke run (more than %lld elements). Check its shape in decoder_joint.onnx and re-export the model."),
						*Desc.Name, MaxSmokeElements));
					return;
				}
				Tensor = FPipelineTensor::Zeros(EPipelineTensorType::Float32, Shape);
			}
			else
			{
				Report.Add(ESev::Error, ECode::SmokeRunFailed, FString::Printf(TEXT("decoder: model input %s is not fed by the task."), *Desc.Name));
				return;
			}
			if (!Tensor.IsValid())
			{
				Report.Add(ESev::Error, ECode::SmokeRunFailed, FString::Printf(TEXT("decoder: could not build the smoke-run tensor for input %s."), *Desc.Name));
				return;
			}
			Inputs.Add(MoveTemp(Tensor));
		}
		TArray<FPipelineTensor> Outputs;
		const FPipelineResult Result = FPipelineExecutor::Get().RunBlocking(Decoder, Inputs, Outputs);
		if (!Result.bSuccess)
		{
			Report.Add(ESev::Error, ECode::SmokeRunFailed, FString::Printf(TEXT("decoder: smoke run failed: %s"),
				Result.Message.IsEmpty() ? TEXT("the run failed") : *Result.Message));
			return;
		}
		FString Error;
		bool bOk = Outputs.Num() == OutputDescs.Num() && Outputs.IsValidIndex(DB.TokenLogits) && Outputs.IsValidIndex(DB.DurationLogits);
		if (!bOk)
		{
			Error = FString::Printf(TEXT("the run returned %d outputs, the model declares %d"), Outputs.Num(), OutputDescs.Num());
		}
		else
		{
			const int64 ExpectedTokens[] = { 1, static_cast<int64>(Package.Speech.BlankTokenId) + 1 };
			const int64 ExpectedDurations[] = { 1, static_cast<int64>(Package.Speech.Durations.Num()) };
			const FString TokName = OutputDescs[DB.TokenLogits].Name;
			const FString DurName = OutputDescs[DB.DurationLogits].Name;
			bOk = CheckShape(Outputs[DB.TokenLogits], *TokName, ExpectedTokens, Error)
				&& CheckFinite(Outputs[DB.TokenLogits], *TokName, Error)
				&& CheckShape(Outputs[DB.DurationLogits], *DurName, ExpectedDurations, Error)
				&& CheckFinite(Outputs[DB.DurationLogits], *DurName, Error);
		}
		if (!bOk)
		{
			Report.Add(ESev::Error, ECode::SmokeOutputInvalid, FString::Printf(
				TEXT("decoder: smoke run output invalid: %s. Check BlankTokenId and Durations against the decoder_joint model."), *Error));
		}
	}

	static FPipelineDeviceCheck& AddCheck(FPipelineValidationReport& Report, EPipelineDevice Device, bool bAvailable)
	{
		FPipelineDeviceCheck& Check = Report.DeviceChecks.AddDefaulted_GetRef();
		Check.Device = Device;
		Check.bAvailable = bAvailable;
		return Check;
	}

	/** True when DirectoriesToAlwaysStageAsNonUFS has the Models directory ("Models" or "/Game/Models", any case). */
	static bool IsModelsStagedNonUfs()
	{
		const UProjectPackagingSettings* const Settings = GetDefault<UProjectPackagingSettings>();
		if (Settings == nullptr)
		{
			return false;
		}
		for (const FDirectoryPath& Dir : Settings->DirectoriesToAlwaysStageAsNonUFS)
		{
			FString Path = Dir.Path.TrimStartAndEnd();
			while (Path.EndsWith(TEXT("/")) || Path.EndsWith(TEXT("\\")))
			{
				Path.LeftChopInline(1);
			}
			if (Path.Equals(TEXT("Models"), ESearchCase::IgnoreCase) || Path.Equals(TEXT("/Game/Models"), ESearchCase::IgnoreCase))
			{
				return true;
			}
		}
		return false;
	}

	/** GGUF branch (C60c): loose file exists, size and SHA-256 match, and Content/Models is staged NonUFS. Never modifies anything. */
	static void ValidateGgufFile(const UPipelineModelPackage& Package, FPipelineValidationReport& Report)
	{
		if (Package.GetModelFormat() != EPipelineModelFormat::GGUF)
		{
			// Not an IPipelineFileModel, or both models set: ValidateSettings has reported it.
			return;
		}

		if (!IsModelsStagedNonUfs())
		{
			Report.Add(ESev::Warning, ECode::ModelsNotStagedNonUfs, TEXT(
				"The project does not stage Content/Models as NonUFS, so a packaged game can't open the .gguf file. Add "
				"+DirectoriesToAlwaysStageAsNonUFS=(Path=\"Models\") to [/Script/UnrealEd.ProjectPackagingSettings] in Config/DefaultGame.ini "
				"(Project Settings > Packaging > Additional Non-Asset Directories To Copy)."));
		}

		const IPipelineFileModel* const FileModel = Cast<IPipelineFileModel>(Package.GgufModel.Get());
		if (FileModel == nullptr)
		{
			return;
		}
		const FString Path = FileModel->GetAbsoluteFilePath();
		if (Path.IsEmpty())
		{
			// InvalidGgufModel was already reported by ValidateSettings.
			return;
		}

		IFileManager& FileManager = IFileManager::Get();
		if (FileManager.DirectoryExists(*Path))
		{
			Report.Add(ESev::Error, ECode::GgufFileMissing, FString::Printf(
				TEXT("The GGUF model path %s is a directory, not a file. Point the GGUF model asset at the .gguf file or re-import it."), *Path));
			return;
		}
		const int64 ActualSize = FileManager.FileSize(*Path);
		if (ActualSize < 0)
		{
			Report.Add(ESev::Error, ECode::GgufFileMissing, FString::Printf(
				TEXT("The GGUF model file %s does not exist. Copy the .gguf file there or re-import the model."), *Path));
			return;
		}
		const int64 ExpectedSize = FileModel->GetFileSizeBytes();
		if (ActualSize != ExpectedSize)
		{
			Report.Add(ESev::Error, ECode::GgufSizeMismatch, FString::Printf(
				TEXT("The GGUF model file %s is %lld bytes, the asset expects %lld bytes. The file was replaced or is incomplete; re-import the model."),
				*Path, ActualSize, ExpectedSize));
			return;
		}

		UE_LOG(LogModelPipelinesEditor, Display, TEXT("Hashing %s (%.1f MB)..."), *FPaths::GetCleanFilename(Path), ActualSize / (1024.0 * 1024.0));
		FString ActualSha;
		FString HashError;
		if (!ComputeFileSha256(Path, ActualSha, HashError))
		{
			Report.Add(ESev::Error, ECode::GgufFileMissing, FString::Printf(
				TEXT("Can't read the GGUF model file %s: %s. Close programs that lock it, check its permissions, or re-import the model."), *Path, *HashError));
			return;
		}
		const FString ExpectedSha = FileModel->GetFileSha256().TrimStartAndEnd().ToLower();
		if (!ActualSha.Equals(ExpectedSha, ESearchCase::IgnoreCase))
		{
			Report.Add(ESev::Error, ECode::GgufShaMismatch, FString::Printf(
				TEXT("The GGUF model file %s has SHA-256 %s, the asset expects %s. The file was replaced or is corrupt; re-import the model."),
				*Path, *ActualSha, *ExpectedSha));
		}
	}

	/** One ISpeechEngine::Transcribe of a fixed signal as a single-step executor session (every inference goes through the executor). */
	class FPhononSmokeSession final : public IPipelineSession
	{
	public:
		FPhononSmokeSession(TSharedPtr<IPipelineModel> InModel, TArray<float>&& InMono, int32 InSampleRate)
			: Model(MoveTemp(InModel)), Mono(MoveTemp(InMono)), SampleRate(InSampleRate)
		{
		}

		virtual TSharedPtr<IPipelineModel> GetModel() const override { return Model; }

		virtual FPipelineResult Begin() override
		{
			if (!Model.IsValid())
			{
				return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("no model"));
			}
			if (!Model->IsInitialized())
			{
				const FPipelineResult Init = Model->Initialize();
				if (!Init.bSuccess)
				{
					return Init;
				}
			}
			if (Model->AsSpeechEngine() == nullptr)
			{
				return FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("the model has no speech engine"));
			}
			return FPipelineResult::Ok();
		}

		virtual EPipelineStep Step(TArray<FPipelineSessionProgress>& OutProgress, FPipelineResult& OutResult) override
		{
			ISpeechEngine* Engine = Model.IsValid() ? Model->AsSpeechEngine() : nullptr;
			if (Engine == nullptr)
			{
				OutResult = FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("the model has no speech engine"));
				return EPipelineStep::Failed;
			}
			FSpeechTranscript Out;
			FPipelineResult R = Engine->Transcribe(Mono, SampleRate, Out);
			Mono.Empty();
			if (!R.bSuccess)
			{
				if (R.Error == EPipelineError::None)
				{
					R.Error = EPipelineError::InferenceFailed;
				}
				OutResult = MoveTemp(R);
				return EPipelineStep::Failed;
			}
			InferenceMs = Out.InferenceMs;
			return EPipelineStep::Finished;
		}

		virtual void End(EPipelineSessionEnd Reason, FPipelineResult& OutFinal) override
		{
			Mono.Empty();
			if (Reason == EPipelineSessionEnd::Finished)
			{
				OutFinal = FPipelineResult::Ok();
				OutFinal.Device = EPipelineDevice::CPU;
				OutFinal.InferenceMs = InferenceMs;
			}
		}

	private:
		TSharedPtr<IPipelineModel> Model;
		TArray<float> Mono;
		int32 SampleRate = 0;
		float InferenceMs = 0.f;
	};

	/**
	 * Phonon branch (C81b): weights folder holds weights.bin/weights.json/tokens.txt, weights.bin size and SHA-256 match the asset,
	 * Content/Models is staged NonUFS; smoke (bRunSmokeTest): the Phonon backend prepares and initializes the model on CPU and
	 * transcribes 1 s of a 440 Hz tone through the executor. Never modifies anything.
	 */
	static void ValidatePhonon(const UPipelineModelPackage& Package, const FPipelineValidateOptions& Options, FPipelineValidationReport& Report)
	{
		if (!IsModelsStagedNonUfs())
		{
			Report.Add(ESev::Warning, ECode::ModelsNotStagedNonUfs, TEXT(
				"The project does not stage Content/Models as NonUFS, so a packaged game can't open the Phonon weights folder. Add "
				"+DirectoriesToAlwaysStageAsNonUFS=(Path=\"Models\") to [/Script/UnrealEd.ProjectPackagingSettings] in Config/DefaultGame.ini "
				"(Project Settings > Packaging > Additional Non-Asset Directories To Copy)."));
		}

		const IPipelineFileModel* const FileModel = Cast<IPipelineFileModel>(Package.GgufModel.Get());
		if (FileModel == nullptr)
		{
			return;
		}
		const FString Folder = FileModel->GetAbsoluteFilePath();
		if (Folder.IsEmpty())
		{
			// ValidateSettings reported the invalid RelativePath.
			return;
		}

		IFileManager& FileManager = IFileManager::Get();
		if (!FileManager.DirectoryExists(*Folder))
		{
			Report.Add(ESev::Error, ECode::GgufFileMissing, FString::Printf(
				TEXT("Phonon weights folder %s is missing (the folder does not exist). Copy the weights folder there or re-import it with 'Import Phonon-2 Weights Folder...'."),
				*Folder));
			return;
		}
		bool bFilesOk = true;
		for (const TCHAR* Name : { TEXT("weights.bin"), TEXT("weights.json"), TEXT("tokens.txt") })
		{
			if (FileManager.FileSize(*(Folder / Name)) < 0)
			{
				Report.Add(ESev::Error, ECode::GgufFileMissing, FString::Printf(
					TEXT("Phonon weights folder %s is missing %s. Copy the complete converter output (weights.bin, weights.json, tokens.txt) there or re-import it."),
					*Folder, Name));
				bFilesOk = false;
			}
		}
		if (!bFilesOk)
		{
			return;
		}

		const FString BinPath = Folder / TEXT("weights.bin");
		const int64 ActualSize = FileManager.FileSize(*BinPath);
		const int64 ExpectedSize = FileModel->GetFileSizeBytes();
		if (ActualSize != ExpectedSize)
		{
			Report.Add(ESev::Error, ECode::GgufSizeMismatch, FString::Printf(
				TEXT("The Phonon weights file %s is %lld bytes, the asset expects %lld bytes. The file was replaced or is incomplete; re-import the weights folder."),
				*BinPath, ActualSize, ExpectedSize));
			return;
		}

		UE_LOG(LogModelPipelinesEditor, Display, TEXT("Hashing %s (%.1f MB)..."), *BinPath, ActualSize / (1024.0 * 1024.0));
		FString ActualSha;
		FString HashError;
		if (!ComputeFileSha256(BinPath, ActualSha, HashError))
		{
			Report.Add(ESev::Error, ECode::GgufFileMissing, FString::Printf(
				TEXT("Can't read the Phonon weights file %s: %s. Close programs that lock it, check its permissions, or re-import the weights folder."), *BinPath, *HashError));
			return;
		}
		const FString ExpectedSha = FileModel->GetFileSha256().TrimStartAndEnd().ToLower();
		if (!ActualSha.Equals(ExpectedSha, ESearchCase::IgnoreCase))
		{
			Report.Add(ESev::Error, ECode::GgufShaMismatch, FString::Printf(
				TEXT("The Phonon weights file %s has SHA-256 %s, the asset expects %s. The file was replaced or is corrupt; re-import the weights folder."),
				*BinPath, *ActualSha, *ExpectedSha));
			return;
		}

		if (!Options.bRunSmokeTest || !Report.IsValid())
		{
			return;
		}
		IPipelineBackend* const Backend = FModelPipelinesCoreModule::Get().FindBackend(EPipelineModelFormat::Phonon);
		if (Backend == nullptr)
		{
			Report.Add(ESev::Info, ECode::None, TEXT("no Phonon backend (enable ModelPipelines Generative); runtime checks skipped"));
			return;
		}

		const int32 CpuIndex = Report.DeviceChecks.Num();
		AddCheck(Report, EPipelineDevice::CPU, Backend->IsDeviceAvailable(EPipelineDevice::CPU));
		if (!Report.DeviceChecks[CpuIndex].bAvailable)
		{
			const FString Msg = TEXT("Phonon: the Phonon backend reports no CPU device. Reinstall or re-enable ModelPipelines Generative.");
			Report.DeviceChecks[CpuIndex].Message = Msg;
			Report.Add(ESev::Error, ECode::RuntimeUnavailable, Msg);
			return;
		}
		FPipelineResult PrepareResult;
		TSharedPtr<IPipelineModel> Model = Backend->PrepareModelFromSource(FPipelineModelSource::FromPackage(&Package), EPipelineDevice::CPU, PrepareResult);
		const FPipelineResult InitResult = Model.IsValid() ? Model->Initialize() : PrepareResult;
		if (!Model.IsValid() || !InitResult.bSuccess)
		{
			const FString Msg = FString::Printf(TEXT("Phonon: the model could not be created on CPU: %s. Re-import the weights folder."),
				InitResult.Message.IsEmpty() ? TEXT("no reason given") : *InitResult.Message);
			Report.DeviceChecks[CpuIndex].Message = Msg;
			Report.Add(ESev::Error, ECode::ModelCreateFailed, Msg);
			return;
		}
		Report.DeviceChecks[CpuIndex].bCreated = true;

		constexpr int32 SampleRate = 16000;
		TArray<float> Tone;
		Tone.SetNumUninitialized(SampleRate);
		for (int32 i = 0; i < SampleRate; ++i)
		{
			Tone[i] = 0.5f * FMath::Sin(2.0f * PI * 440.0f * static_cast<float>(i) / static_cast<float>(SampleRate));
		}
		TSharedPtr<IPipelineSession> Session = MakeShared<FPhononSmokeSession>(Model, MoveTemp(Tone), SampleRate);
		Model.Reset();
		TArray<FPipelineSessionProgress> Progress;
		const FPipelineResult SmokeResult = FPipelineExecutor::Get().RunSessionBlocking(Session, Progress);
		Session.Reset();
		FPipelineDeviceCheck& Cpu = Report.DeviceChecks[CpuIndex];
		if (!SmokeResult.bSuccess)
		{
			const FString Msg = FString::Printf(TEXT("Phonon: smoke transcription of 1 s of a 440 Hz tone failed on CPU: %s. Re-import the weights folder."),
				SmokeResult.Message.IsEmpty() ? TEXT("no reason given") : *SmokeResult.Message);
			Cpu.Message = Msg;
			Report.Add(ESev::Error, ECode::SmokeRunFailed, Msg);
			return;
		}
		Cpu.bRan = true;
		Cpu.InferenceMs = SmokeResult.InferenceMs;
		Cpu.Message = TEXT("CPU: Phonon smoke transcription ran");
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
	if (Package->Task == EPipelineTask::SpeechToText && Package->Model == nullptr && Package->GetModelFormat() == EPipelineModelFormat::Phonon)
	{
		// Phonon (SpeechToText stage B, C81b): file checks and an engine smoke; the NNE steps never apply.
		ValidatePhonon(*Package, Options, Report);
		UE_LOG(LogModelPipelinesEditor, Verbose, TEXT("Validated %s: %d errors, %d warnings."), *Package->GetPathName(), Report.NumErrors(), Report.NumWarnings());
		return Report;
	}
	if (Package->GgufModel != nullptr || Package->GetModelFormat() == EPipelineModelFormat::GGUF)
	{
		// GGUF (or a GgufModel that ValidateSettings already rejected): the NNE steps never apply.
		ValidateGgufFile(*Package, Report);
		UE_LOG(LogModelPipelinesEditor, Verbose, TEXT("Validated %s: %d errors, %d warnings."), *Package->GetPathName(), Report.NumErrors(), Report.NumWarnings());
		return Report;
	}
	if (Report.HasError(ECode::WrongModelFormat))
	{
		// E.g. a TextGeneration package with an NNE Model: the NNE steps (devices, smoke run) don't apply to it.
		UE_LOG(LogModelPipelinesEditor, Verbose, TEXT("Validated %s: %d errors, %d warnings."), *Package->GetPathName(), Report.NumErrors(), Report.NumWarnings());
		return Report;
	}
	if (Report.HasError(ECode::NoTask) || Report.HasError(ECode::NoModel) || Package->Model == nullptr)
	{
		if (!Report.HasCode(ECode::NoTask) && !Report.HasCode(ECode::NoModel))
		{
			Report.Add(ESev::Error, ECode::NoModel, TEXT("The package has no model. Re-import the ONNX model."));
		}
		else if (Package->Task != EPipelineTask::TextGeneration)
		{
			// NNE task: keep the editor's fix hint on the settings check's NoModel entry (one entry, not two).
			for (FPipelineValidationMessage& Message : Report.Messages)
			{
				if (Message.Code == ECode::NoModel && !Message.Message.Contains(TEXT("Re-import the ONNX model.")))
				{
					Message.Message += Message.Message.EndsWith(TEXT(".")) ? TEXT(" Re-import the ONNX model.") : TEXT(". Re-import the ONNX model.");
				}
			}
		}
		return Report;
	}
	const EPipelineTask Task = Package->Task;
	// An out-of-range MaxTokens is already an Error in the report. The smoke runs (and above all the worst case,
	// which allocates MaxTokens elements per input) would be unsafe with it, so they are skipped; the tokenizer,
	// CPU/GPU creation and task I/O checks still run.
	const bool bSmokeUnsafe = Report.HasError(ECode::InvalidMaxTokens);
	const bool bRunSmoke = Options.bRunSmokeTest && !bSmokeUnsafe;
	static const TCHAR* const SmokeSkippedInvalidMaxTokens = TEXT("smoke run skipped (MaxTokens is outside 1..65536; fix MaxTokens)");

	// Step 2: tokenizer (text tasks only).
	TArray<int32> Ids;
	// Kept alive for the worst-case run in step 5 (text tasks only).
	TStrongObjectPtr<UPipelineTokenizer> Tokenizer;
	if (IsTextTask(Task) || Task == EPipelineTask::SpeechToText)
	{
		Tokenizer.Reset(NewObject<UPipelineTokenizer>(GetTransientPackage()));
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
	const bool bSpeech = Task == EPipelineTask::SpeechToText;
	const bool bTaskIoOk = bSpeech
		? ResolvePipelineTaskIO(Task, Package->Preprocess, Package->Postprocess, Package->Speech, Package->Labels.Num(), InputDescs, OutputDescs, Bindings, Report)
		: ResolvePipelineTaskIO(Task, Package->Preprocess, Package->Postprocess, Package->Labels.Num(), InputDescs, OutputDescs, Bindings, Report);
	if (!bTaskIoOk)
	{
		Report.DeviceChecks[CpuIndex].Message = TEXT("CPU: smoke run skipped (the model's inputs/outputs do not fit the task)");
		return Report;
	}

	// Step 5: CPU smoke run.
	if (Options.bRunSmokeTest && bSmokeUnsafe)
	{
		Report.DeviceChecks[CpuIndex].Message = FString::Printf(TEXT("CPU: %s"), SmokeSkippedInvalidMaxTokens);
	}
	TArray<FPipelineTensor> CpuOutputs;
	bool bCpuSmokeRan = false;
	if (bRunSmoke)
	{
		const FSmokeOutcome Smoke = RunSmoke(CpuModel, *Package, Bindings, Ids, bSpeech ? &CpuOutputs : nullptr);
		bCpuSmokeRan = Smoke.bRan;
		FPipelineDeviceCheck& Cpu = Report.DeviceChecks[CpuIndex];
		Cpu.bRan = Smoke.bRan;
		Cpu.InferenceMs = Smoke.InferenceMs;
		if (!Smoke.bRan)
		{
			const FString Msg = FString::Printf(TEXT("CPU: %s"), *Smoke.Message);
			Cpu.Message = Msg;
			Report.Add(ESev::Error, Smoke.Code, Msg);
		}
		else if (IsTextTask(Task) && Tokenizer.IsValid())
		{
			// Worst case (C23c): MaxTokens tokens, all the tokenizer's highest id. ORT errors on an id past the
			// embedding table or a sequence past the position table; prove here, where errors are caught, that it doesn't.
			// Cpu.InferenceMs stays the normal smoke run's.
			const int32 VocabSize = Tokenizer->GetVocabSize();
			if (VocabSize < 1)
			{
				const FString Why = Tokenizer->GetLastError();
				Report.Add(ESev::Error, ECode::TokenizerInvalid, Why.IsEmpty()
					? FString(TEXT("The tokenizer reports no vocabulary. Check the package's tokenizer.json."))
					: FString::Printf(TEXT("The tokenizer reports no vocabulary (%s). Check the package's tokenizer.json."), *Why));
			}
			else if (Package->Preprocess.MaxTokens < 1 || Package->Preprocess.MaxTokens > 65536)
			{
				// Defence in depth: never allocate a worst-case input for an out-of-range MaxTokens.
				const FString Msg = FString::Printf(
					TEXT("CPU: worst-case input not built: MaxTokens %d is outside 1..65536. Set MaxTokens to the model's position limit."),
					Package->Preprocess.MaxTokens);
				Report.DeviceChecks[CpuIndex].Message = Msg;
				Report.Add(ESev::Error, ECode::SmokeRunFailed, Msg);
			}
			else
			{
				const int32 MaxTokens = Package->Preprocess.MaxTokens;
				const int32 TopId = VocabSize - 1;
				TArray<int32> WorstIds;
				WorstIds.Init(TopId, MaxTokens);
				const FSmokeOutcome Worst = RunSmoke(CpuModel, *Package, Bindings, WorstIds);
				if (!Worst.bRan)
				{
					const FString Msg = FString::Printf(
						TEXT("CPU: worst-case input failed (MaxTokens %d tokens, token id %d): %s. The model's vocabulary or position limit is smaller than the tokenizer or MaxTokens; lower MaxTokens or use the matching tokenizer."),
						MaxTokens, TopId, *Worst.Reason);
					Report.DeviceChecks[CpuIndex].Message = Msg;
					Report.Add(ESev::Error, ECode::SmokeRunFailed, Msg);
				}
			}
		}
	}
	CpuModel.Reset();

	// Step 5b: SpeechToText decoder_joint, always on CPU (C78).
	if (bSpeech)
	{
		int32 EncoderDim = 0;
		if (OutputDescs.IsValidIndex(Bindings.MainOutput) && OutputDescs[Bindings.MainOutput].Shape.Num() == 3
			&& OutputDescs[Bindings.MainOutput].Shape[2] > 0)
		{
			EncoderDim = static_cast<int32>(OutputDescs[Bindings.MainOutput].Shape[2]);
		}
		const FPipelineTensor* EncoderOut = bCpuSmokeRan && CpuOutputs.IsValidIndex(Bindings.MainOutput) ? &CpuOutputs[Bindings.MainOutput] : nullptr;
		if (EncoderDim <= 0 && EncoderOut && EncoderOut->GetShape().Num() == 3)
		{
			EncoderDim = static_cast<int32>(EncoderOut->GetShape()[2]);
		}
		ValidateSpeechDecoder(*Package, Backend, EncoderDim, EncoderOut, Report);
		CpuOutputs.Empty();
	}

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
				if (Options.bRunSmokeTest && bSmokeUnsafe)
				{
					Report.DeviceChecks[GpuIndex].Message = FString::Printf(TEXT("GPU: %s"), SmokeSkippedInvalidMaxTokens);
				}
				if (bRunSmoke)
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
