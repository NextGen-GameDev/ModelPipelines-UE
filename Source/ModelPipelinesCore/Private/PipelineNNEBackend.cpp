// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineNNEBackend.h"

#include "HAL/CriticalSection.h"
#include "HAL/Event.h"
#include "HAL/PlatformTime.h"
#include "Misc/ScopeLock.h"
#include "NNE.h"
#include "NNEModelData.h"
#include "NNERuntimeCPU.h"
#include "NNERuntimeGPU.h"
#include "NNERuntimeRunSync.h"
#include "NNEStatus.h"
#include "NNETypes.h"
#include "PipelineLog.h"
#include "PipelineOnnxCheck.h"

#include <atomic>

namespace UE::ModelPipelines::Private
{
namespace
{
	const TCHAR* NNEDataTypeToString(ENNETensorDataType Type)
	{
		switch (Type)
		{
		case ENNETensorDataType::None: return TEXT("none");
		case ENNETensorDataType::Char: return TEXT("char");
		case ENNETensorDataType::Boolean: return TEXT("bool");
		case ENNETensorDataType::Half: return TEXT("float16");
		case ENNETensorDataType::Float: return TEXT("float32");
		case ENNETensorDataType::Double: return TEXT("float64");
		case ENNETensorDataType::Int8: return TEXT("int8");
		case ENNETensorDataType::Int16: return TEXT("int16");
		case ENNETensorDataType::Int32: return TEXT("int32");
		case ENNETensorDataType::Int64: return TEXT("int64");
		case ENNETensorDataType::UInt8: return TEXT("uint8");
		case ENNETensorDataType::UInt16: return TEXT("uint16");
		case ENNETensorDataType::UInt32: return TEXT("uint32");
		case ENNETensorDataType::UInt64: return TEXT("uint64");
		case ENNETensorDataType::Complex64: return TEXT("complex64");
		case ENNETensorDataType::Complex128: return TEXT("complex128");
		case ENNETensorDataType::BFloat16: return TEXT("bfloat16");
		default: return TEXT("unknown");
		}
	}

	bool ToPipelineType(ENNETensorDataType Type, EPipelineTensorType& Out)
	{
		switch (Type)
		{
		case ENNETensorDataType::Float: Out = EPipelineTensorType::Float32; return true;
		case ENNETensorDataType::Int64: Out = EPipelineTensorType::Int64; return true;
		case ENNETensorDataType::Int32: Out = EPipelineTensorType::Int32; return true;
		default: return false;
		}
	}

	FString DescribeDescShape(TConstArrayView<int64> Shape)
	{
		FString S = TEXT("[");
		for (int32 i = 0; i < Shape.Num(); ++i)
		{
			if (i > 0) { S += TEXT(","); }
			S += FString::Printf(TEXT("%lld"), (long long)Shape[i]);
		}
		S += TEXT("]");
		return S;
	}

	FString DescribeDesc(const FPipelineTensorDesc& Desc)
	{
		return FString::Printf(TEXT("%s %s"), Desc.bTypeSupported ? LexToString(Desc.Type) : TEXT("<unsupported type>"), *DescribeDescShape(Desc.Shape));
	}

	FString DescribeInputShapes(TConstArrayView<FPipelineTensor> Inputs)
	{
		FString S;
		for (int32 i = 0; i < Inputs.Num(); ++i)
		{
			if (i > 0) { S += TEXT(" "); }
			S += Inputs[i].DescribeShape();
		}
		return S;
	}

	bool ShapesEqual(TConstArrayView<int64> A, TConstArrayView<int64> B)
	{
		if (A.Num() != B.Num())
		{
			return false;
		}
		for (int32 i = 0; i < A.Num(); ++i)
		{
			if (A[i] != B[i])
			{
				return false;
			}
		}
		return true;
	}

	/** Byte size of a resolved NNE shape for Type; false if it overflows or exceeds FPipelineTensor::MaxSizeInBytes. */
	bool ComputeShapeBytes(TConstArrayView<int64> Shape, EPipelineTensorType Type, uint64& OutBytes)
	{
		uint64 Bytes = (uint64)GetPipelineTensorTypeSize(Type);
		for (int64 Dim : Shape)
		{
			if (Dim < 0)
			{
				return false;
			}
			if (Dim == 0)
			{
				OutBytes = 0;
				return true;
			}
			if (Bytes > (uint64)FPipelineTensor::MaxSizeInBytes / (uint64)Dim)
			{
				return false;
			}
			Bytes *= (uint64)Dim;
		}
		OutBytes = Bytes;
		return true;
	}

	/**
	 * IPipelineModel over NNE CPU/GPU models. Created on the game thread by FPipelineNNEBackend::PrepareModel;
	 * Initialize and Run touch no UObject.
	 */
	class FPipelineNNEModel final : public IPipelineModel
	{
	public:
		FPipelineNNEModel(EPipelineDevice InRequested, TSharedPtr<UE::NNE::IModelGPU> InGpuModel, TSharedPtr<UE::NNE::IModelCPU> InCpuModel,
			const FString& InGpuRuntimeName, const FString& InCpuRuntimeName)
			: RequestedDevice(InRequested)
			, GpuModel(MoveTemp(InGpuModel))
			, CpuModel(MoveTemp(InCpuModel))
			, GpuRuntimeName(InGpuRuntimeName)
			, CpuRuntimeName(InCpuRuntimeName)
			, InitDoneEvent(EEventMode::ManualReset)
		{
		}

		virtual FPipelineResult Initialize() override
		{
			// Fast path: the result is immutable once published.
			if (bInitDone.load(std::memory_order_acquire))
			{
				return InitResult;
			}

			// InitCS only claims the attempt; instance creation runs unlocked so readers never wait on it.
			bool bClaimed = false;
			{
				FScopeLock Lock(&InitCS);
				if (!bInitAttempted)
				{
					bInitAttempted = true;
					bClaimed = true;
				}
			}
			if (!bClaimed)
			{
				// Another thread is creating the instance: wait for it to publish (only Initialize callers wait here).
				InitDoneEvent->Wait();
				return InitResult;
			}

			FPipelineResult Result = CreateInstance();

			InitResult = MoveTemp(Result);
			bInitDone.store(true, std::memory_order_release);
			InitDoneEvent->Trigger();
			return InitResult;
		}

		virtual bool IsInitialized() const override
		{
			return bInitialized.load(std::memory_order_acquire);
		}

		virtual EPipelineDevice GetDevice() const override
		{
			return bInitialized.load(std::memory_order_acquire) ? ActiveDevice : EPipelineDevice::Auto;
		}

		virtual TArray<FPipelineTensorDesc> GetInputDescs() const override
		{
			return bInitialized.load(std::memory_order_acquire) ? InputDescs : TArray<FPipelineTensorDesc>();
		}

		virtual TArray<FPipelineTensorDesc> GetOutputDescs() const override
		{
			return bInitialized.load(std::memory_order_acquire) ? OutputDescs : TArray<FPipelineTensorDesc>();
		}

		virtual FPipelineResult Run(TConstArrayView<FPipelineTensor> Inputs, TArray<FPipelineTensor>& OutOutputs) override
		{
			OutOutputs.Reset();

			const FPipelineResult Init = Initialize();
			if (!Init.bSuccess)
			{
				FPipelineResult R = Init;
				R.Device = EPipelineDevice::Auto;
				return R;
			}

			// Initialize succeeded, so Instance/descs are fixed from here on; the caller serializes Run.
			if (Inputs.Num() != InputDescs.Num())
			{
				return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
					FString::Printf(TEXT("Model expects %d input(s) but got %d"), InputDescs.Num(), Inputs.Num()));
			}

			for (int32 i = 0; i < Inputs.Num(); ++i)
			{
				const FPipelineTensor& In = Inputs[i];
				const FPipelineTensorDesc& Desc = InputDescs[i];
				if (!In.IsValid())
				{
					return FPipelineResult::Fail(EPipelineError::InvalidArgument,
						FString::Printf(TEXT("Input '%s' (#%d) is an invalid tensor"), *Desc.Name, i));
				}
				if (In.Num() == 0)
				{
					return FPipelineResult::Fail(EPipelineError::InvalidArgument,
						FString::Printf(TEXT("Input '%s' (#%d) is empty (shape %s, 0 elements)"), *Desc.Name, i, *In.DescribeShape()));
				}
			}

			for (int32 i = 0; i < Inputs.Num(); ++i)
			{
				const FPipelineTensor& In = Inputs[i];
				const FPipelineTensorDesc& Desc = InputDescs[i];
				if (!Desc.IsCompatible(In))
				{
					return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
						FString::Printf(TEXT("Input '%s' (#%d) expects %s but got %s %s"),
							*Desc.Name, i, *DescribeDesc(Desc), LexToString(In.GetType()), *In.DescribeShape()));
				}
			}

			// Concrete shapes. Dims are <= MAX_uint32 (FPipelineTensor guarantees it).
			TArray<TArray<uint32>> NewShapes;
			NewShapes.SetNum(Inputs.Num());
			for (int32 i = 0; i < Inputs.Num(); ++i)
			{
				for (int64 Dim : Inputs[i].GetShape())
				{
					NewShapes[i].Add((uint32)Dim);
				}
			}

			// Concrete outputs: SetInputTensorShapes only when the input shapes change. Dynamic outputs: on every Run,
			// because it is what clears the engine's resolved output tensors (NNERuntimeORTModel.cpp:324-326); without
			// that the engine would bind stale output shapes.
			if (!bHaveShapes || !bOutputShapesConcrete || NewShapes != LastInputShapes)
			{
				bHaveShapes = false;
				TArray<UE::NNE::FTensorShape> NNEShapes;
				NNEShapes.Reserve(NewShapes.Num());
				for (const TArray<uint32>& S : NewShapes)
				{
					NNEShapes.Add(UE::NNE::FTensorShape::Make(S));
				}
				if (Instance->SetInputTensorShapes(NNEShapes) != UE::NNE::EResultStatus::Ok)
				{
					FString Shapes;
					for (int32 i = 0; i < Inputs.Num(); ++i)
					{
						Shapes += FString::Printf(TEXT("%s'%s' %s"), i > 0 ? TEXT(", ") : TEXT(""), *InputDescs[i].Name, *Inputs[i].DescribeShape());
					}
					return FPipelineResult::Fail(EPipelineError::ShapeMismatch,
						FString::Printf(TEXT("The runtime rejected the input shapes: %s"), *Shapes));
				}
				LastInputShapes = MoveTemp(NewShapes);
				bHaveShapes = true;
				bOutputShapesConcrete = Instance->GetOutputTensorShapes().Num() == OutputDescs.Num();
			}

			TArray<UE::NNE::FTensorBindingCPU> InBindings;
			InBindings.Reserve(Inputs.Num());
			for (const FPipelineTensor& In : Inputs)
			{
				UE::NNE::FTensorBindingCPU B;
				B.Data = const_cast<void*>(In.GetRawData());
				B.SizeInBytes = (uint64)In.GetSizeInBytes();
				InBindings.Add(B);
			}

			double InferenceSeconds = 0.0;
			int32 RunSyncCalls = 0;
			TArray<FPipelineTensor> Outputs;
			TArray<UE::NNE::FTensorBindingCPU> OutBindings;

			auto CallRunSync = [&](TConstArrayView<UE::NNE::FTensorBindingCPU> Bindings) -> bool
			{
				const double T0 = FPlatformTime::Seconds();
				const UE::NNE::EResultStatus Status = Instance->RunSync(InBindings, Bindings);
				InferenceSeconds += FPlatformTime::Seconds() - T0;
				++RunSyncCalls;
				return Status == UE::NNE::EResultStatus::Ok;
			};

			auto FailRun = [this](EPipelineError Code, const FString& Message) -> FPipelineResult
			{
				bHaveShapes = false;
				bHaveCachedOutputShapes = false;
				return FPipelineResult::Fail(Code, Message);
			};

			if (bOutputShapesConcrete)
			{
				FPipelineResult AllocResult = AllocateOutputs(ReadInstanceOutputShapes(), Outputs, OutBindings);
				if (!AllocResult.bSuccess)
				{
					return FailRun(AllocResult.Error, AllocResult.Message);
				}
				if (!CallRunSync(OutBindings))
				{
					return FailRun(EPipelineError::InferenceFailed, TEXT("RunSync failed"));
				}
			}
			else
			{
				// Dynamic outputs. The engine lets ORT allocate and copies into any binding that is big enough
				// (NNERuntimeORTModel.cpp:470-476), so bind buffers sized from the last resolved shapes and run once.
				// With no cache yet, run with empty bindings to resolve the shapes.
				const bool bBoundFirstPass = bHaveCachedOutputShapes;
				if (bBoundFirstPass)
				{
					FPipelineResult AllocResult = AllocateOutputs(CachedOutputShapes, Outputs, OutBindings);
					if (!AllocResult.bSuccess)
					{
						return FailRun(AllocResult.Error, AllocResult.Message);
					}
				}
				if (!CallRunSync(OutBindings))
				{
					return FailRun(EPipelineError::InferenceFailed, TEXT("RunSync failed (dynamic-output pass)"));
				}
				if (Instance->GetOutputTensorShapes().Num() != OutputDescs.Num())
				{
					return FailRun(EPipelineError::InferenceFailed,
						FString::Printf(TEXT("The runtime resolved %d output shape(s), expected %d"), Instance->GetOutputTensorShapes().Num(), OutputDescs.Num()));
				}

				TArray<TArray<int64>> ActualShapes = ReadInstanceOutputShapes();

				// Did the engine already copy every output? Mirrors its memcpy condition (NNERuntimeORTModel.cpp:470-473).
				bool bAllFit = bBoundFirstPass;
				for (int32 i = 0; bAllFit && i < OutputDescs.Num(); ++i)
				{
					uint64 ActualBytes = 0;
					if (!ComputeShapeBytes(ActualShapes[i], OutputDescs[i].Type, ActualBytes))
					{
						bAllFit = false;
					}
					else if (ActualBytes > 0 && (OutBindings[i].Data == nullptr || OutBindings[i].SizeInBytes < ActualBytes))
					{
						bAllFit = false;
					}
				}

				if (bAllFit)
				{
					// One inference. Shrink outputs whose actual shape differs from the cached one.
					for (int32 i = 0; i < OutputDescs.Num(); ++i)
					{
						if (ShapesEqual(ActualShapes[i], Outputs[i].GetShape()))
						{
							continue;
						}
						FPipelineTensor Exact = FPipelineTensor::Zeros(OutputDescs[i].Type, ActualShapes[i]);
						if (!Exact.IsValid())
						{
							return FailRun(EPipelineError::InferenceFailed,
								FString::Printf(TEXT("Output '%s' has shape %s, which cannot be allocated"), *OutputDescs[i].Name, *DescribeDescShape(ActualShapes[i])));
						}
						if (Exact.GetSizeInBytes() > 0)
						{
							FMemory::Memcpy(Exact.GetRawData(), Outputs[i].GetRawData(), (SIZE_T)Exact.GetSizeInBytes());
						}
						Outputs[i] = MoveTemp(Exact);
					}
				}
				else
				{
					// Buffers missing or too small: allocate the actual sizes and run again. The engine's output tensors
					// are now resolved, so it binds these buffers directly.
					FPipelineResult AllocResult = AllocateOutputs(ActualShapes, Outputs, OutBindings);
					if (!AllocResult.bSuccess)
					{
						return FailRun(AllocResult.Error, AllocResult.Message);
					}
					if (!CallRunSync(OutBindings))
					{
						return FailRun(EPipelineError::InferenceFailed, TEXT("RunSync failed (bound pass)"));
					}
				}

				CachedOutputShapes = MoveTemp(ActualShapes);
				bHaveCachedOutputShapes = true;
			}

			UE_LOG(LogModelPipelines, Verbose, TEXT("NNE Run: %d RunSync call(s), inputs %s, %s outputs"),
				RunSyncCalls, *DescribeInputShapes(Inputs), bOutputShapesConcrete ? TEXT("concrete") : TEXT("dynamic"));

			OutOutputs = MoveTemp(Outputs);
			FPipelineResult R = FPipelineResult::Ok();
			R.Device = ActiveDevice;
			R.InferenceMs = (float)(InferenceSeconds * 1000.0);
			return R;
		}

	private:
		/** Runs once, on the thread that claimed the attempt, with no lock held. Publishes the instance and descs on success. */
		FPipelineResult CreateInstance()
		{
			TSharedPtr<UE::NNE::IModelInstanceRunSync> NewInstance;
			EPipelineDevice NewDevice = EPipelineDevice::Auto;
			FString GpuFailure;

			if (GpuModel.IsValid())
			{
				TSharedPtr<UE::NNE::IModelInstanceGPU> GpuInstance = GpuModel->CreateModelInstanceGPU();
				if (GpuInstance.IsValid())
				{
					NewInstance = GpuInstance;
					NewDevice = EPipelineDevice::GPU;
				}
				else
				{
					GpuFailure = FString::Printf(TEXT("runtime '%s' could not create a GPU model instance"), *GpuRuntimeName);
					GpuModel.Reset();
				}
			}

			if (!NewInstance.IsValid() && CpuModel.IsValid())
			{
				if (!GpuFailure.IsEmpty())
				{
					if (RequestedDevice == EPipelineDevice::GPU)
					{
						UE_LOG(LogModelPipelines, Warning, TEXT("GPU unavailable for this model, falling back to CPU: %s"), *GpuFailure);
					}
					else
					{
						UE_LOG(LogModelPipelines, Log, TEXT("GPU unavailable for this model, falling back to CPU: %s"), *GpuFailure);
					}
				}
				TSharedPtr<UE::NNE::IModelInstanceCPU> CpuInstance = CpuModel->CreateModelInstanceCPU();
				if (CpuInstance.IsValid())
				{
					NewInstance = CpuInstance;
					NewDevice = EPipelineDevice::CPU;
				}
			}

			if (!NewInstance.IsValid())
			{
				FString Msg = TEXT("Could not create a model instance on any allowed device");
				if (!GpuFailure.IsEmpty())
				{
					Msg += FString::Printf(TEXT(" (GPU: %s)"), *GpuFailure);
				}
				if (CpuModel.IsValid())
				{
					Msg += FString::Printf(TEXT(" (CPU: runtime '%s' could not create a CPU model instance)"), *CpuRuntimeName);
				}
				return FPipelineResult::Fail(EPipelineError::ModelLoadFailed, Msg);
			}

			TArray<FPipelineTensorDesc> NewInputs;
			TArray<FPipelineTensorDesc> NewOutputs;
			FString TypeError;
			ConvertDescs(NewInstance->GetInputTensorDescs(), TEXT("input"), NewInputs, TypeError);
			ConvertDescs(NewInstance->GetOutputTensorDescs(), TEXT("output"), NewOutputs, TypeError);
			if (!TypeError.IsEmpty())
			{
				return FPipelineResult::Fail(EPipelineError::UnsupportedType, TypeError);
			}

			Instance = MoveTemp(NewInstance);
			InputDescs = MoveTemp(NewInputs);
			OutputDescs = MoveTemp(NewOutputs);
			ActiveDevice = NewDevice;
			// Publish after the descs and device are set; readers load with acquire.
			bInitialized.store(true, std::memory_order_release);

			FPipelineResult Ok = FPipelineResult::Ok();
			Ok.Device = NewDevice;
			return Ok;
		}

		static void ConvertDescs(TConstArrayView<UE::NNE::FTensorDesc> Src, const TCHAR* Kind, TArray<FPipelineTensorDesc>& Out, FString& InOutTypeError)
		{
			Out.Reserve(Src.Num());
			for (const UE::NNE::FTensorDesc& D : Src)
			{
				FPipelineTensorDesc P;
				P.Name = D.GetName();
				P.bTypeSupported = ToPipelineType(D.GetDataType(), P.Type);
				for (int32 Dim : D.GetShape().GetData())
				{
					P.Shape.Add((int64)Dim);
				}
				if (!P.bTypeSupported && InOutTypeError.IsEmpty())
				{
					InOutTypeError = FString::Printf(TEXT("Model %s '%s' has unsupported type %s (supported: float32, int64, int32)"),
						Kind, *P.Name, NNEDataTypeToString(D.GetDataType()));
				}
				Out.Add(MoveTemp(P));
			}
		}

		/** The instance's resolved output shapes, uint32 dims converted to int64. */
		TArray<TArray<int64>> ReadInstanceOutputShapes() const
		{
			TArray<TArray<int64>> Result;
			for (const UE::NNE::FTensorShape& S : Instance->GetOutputTensorShapes())
			{
				TArray<int64>& Shape = Result.AddDefaulted_GetRef();
				for (uint32 Dim : S.GetData())
				{
					Shape.Add((int64)Dim);
				}
			}
			return Result;
		}

		/** Allocates one tensor per output with the given shapes and binds them. */
		FPipelineResult AllocateOutputs(const TArray<TArray<int64>>& Shapes, TArray<FPipelineTensor>& Outputs, TArray<UE::NNE::FTensorBindingCPU>& Bindings) const
		{
			Outputs.Reset(OutputDescs.Num());
			Bindings.Reset(OutputDescs.Num());
			if (Shapes.Num() != OutputDescs.Num())
			{
				return FPipelineResult::Fail(EPipelineError::InferenceFailed,
					FString::Printf(TEXT("The runtime reported %d output shape(s), expected %d"), Shapes.Num(), OutputDescs.Num()));
			}
			for (int32 i = 0; i < OutputDescs.Num(); ++i)
			{
				FPipelineTensor T = FPipelineTensor::Zeros(OutputDescs[i].Type, Shapes[i]);
				if (!T.IsValid())
				{
					return FPipelineResult::Fail(EPipelineError::InferenceFailed,
						FString::Printf(TEXT("Output '%s' has shape %s, which cannot be allocated"), *OutputDescs[i].Name, *DescribeDescShape(Shapes[i])));
				}
				Outputs.Add(MoveTemp(T));
			}
			for (FPipelineTensor& T : Outputs)
			{
				UE::NNE::FTensorBindingCPU B;
				B.Data = T.GetRawData();
				B.SizeInBytes = (uint64)T.GetSizeInBytes();
				Bindings.Add(B);
			}
			return FPipelineResult::Ok();
		}

		const EPipelineDevice RequestedDevice;
		TSharedPtr<UE::NNE::IModelGPU> GpuModel;
		TSharedPtr<UE::NNE::IModelCPU> CpuModel;
		const FString GpuRuntimeName;
		const FString CpuRuntimeName;

		// Initialization. InitCS guards only bInitAttempted. InitResult is written once, before bInitDone is released.
		// Instance/descs/ActiveDevice are written once, before bInitialized is released, and are immutable afterwards.
		FCriticalSection InitCS;
		bool bInitAttempted = false;
		FEventRef InitDoneEvent;
		std::atomic<bool> bInitDone{ false };
		std::atomic<bool> bInitialized{ false };
		FPipelineResult InitResult;
		EPipelineDevice ActiveDevice = EPipelineDevice::Auto;
		TSharedPtr<UE::NNE::IModelInstanceRunSync> Instance;
		TArray<FPipelineTensorDesc> InputDescs;
		TArray<FPipelineTensorDesc> OutputDescs;

		// Shape caches: used only by Run, which the caller serializes.
		TArray<TArray<uint32>> LastInputShapes;
		bool bHaveShapes = false;
		bool bOutputShapesConcrete = false;
		// Dynamic outputs: shapes resolved by the last successful Run, used to size the single-pass bindings.
		TArray<TArray<int64>> CachedOutputShapes;
		bool bHaveCachedOutputShapes = false;
	};
}
}

FPipelineNNEBackend::FPipelineNNEBackend()
{
}

FPipelineNNEBackend::FPipelineNNEBackend(const FPipelineNNEBackendConfig& InConfig)
	: Config(InConfig)
{
}

FName FPipelineNNEBackend::GetName() const
{
	return FName(TEXT("NNE"));
}

bool FPipelineNNEBackend::IsDeviceAvailable(EPipelineDevice Device) const
{
	const bool bCpu = (Device == EPipelineDevice::CPU || Device == EPipelineDevice::Auto)
		&& UE::NNE::GetRuntime<INNERuntimeCPU>(Config.CpuRuntimeName).IsValid();
	if (Device == EPipelineDevice::CPU)
	{
		return bCpu;
	}
	const bool bGpu = UE::NNE::GetRuntime<INNERuntimeGPU>(Config.GpuRuntimeName).IsValid();
	return Device == EPipelineDevice::GPU ? bGpu : (bCpu || bGpu);
}

TSharedPtr<IPipelineModel> FPipelineNNEBackend::PrepareModel(UNNEModelData* ModelData, EPipelineDevice Device, FPipelineResult& OutResult)
{
	using namespace UE::ModelPipelines::Private;

	if (!ModelData)
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("PrepareModel: ModelData is null"));
		return nullptr;
	}
	if (!IsInGameThread())
	{
		OutResult = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("PrepareModel must be called on the game thread"));
		return nullptr;
	}

	const bool bWantGpu = Device == EPipelineDevice::GPU || Device == EPipelineDevice::Auto;

	TWeakInterfacePtr<INNERuntimeCPU> CpuRuntime = UE::NNE::GetRuntime<INNERuntimeCPU>(Config.CpuRuntimeName);
	TWeakInterfacePtr<INNERuntimeGPU> GpuRuntime;
	if (bWantGpu)
	{
		GpuRuntime = UE::NNE::GetRuntime<INNERuntimeGPU>(Config.GpuRuntimeName);
	}

	if (!CpuRuntime.IsValid() && !GpuRuntime.IsValid())
	{
		const FString Msg = bWantGpu
			? FString::Printf(TEXT("No NNE runtime available: neither GPU runtime '%s' nor CPU runtime '%s' is registered"), *Config.GpuRuntimeName, *Config.CpuRuntimeName)
			: FString::Printf(TEXT("No NNE runtime available: CPU runtime '%s' is not registered"), *Config.CpuRuntimeName);
		OutResult = FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, Msg);
		return nullptr;
	}

	// Structural ONNX check before any runtime call: ORT only parses the bytes when the instance is created, and only
	// editor builds catch its exceptions there. Cooked model data has no file data, so it skips the check.
	{
		const TConstArrayView64<uint8> FileData = ModelData->GetFileData();
		if (FileData.Num() > 0)
		{
			FString Reason;
			if (!IsStructurallyValidOnnx(FileData, Reason))
			{
				OutResult = FPipelineResult::Fail(EPipelineError::ModelLoadFailed,
					FString::Printf(TEXT("model bytes are not a valid ONNX model: %s"), *Reason));
				return nullptr;
			}
		}
	}

	// All CanCreate*/Create* calls run here on the game thread: they read (and cache into) the UNNEModelData.
	TSharedPtr<UE::NNE::IModelGPU> GpuModel;
	FString GpuFailure;
	if (bWantGpu)
	{
		if (!GpuRuntime.IsValid())
		{
			GpuFailure = FString::Printf(TEXT("GPU runtime '%s' is not registered"), *Config.GpuRuntimeName);
		}
		else if (GpuRuntime->CanCreateModelGPU(ModelData) != UE::NNE::EResultStatus::Ok)
		{
			GpuFailure = FString::Printf(TEXT("GPU runtime '%s' cannot create a model from this model data"), *Config.GpuRuntimeName);
		}
		else
		{
			GpuModel = GpuRuntime->CreateModelGPU(ModelData);
			if (!GpuModel.IsValid())
			{
				GpuFailure = FString::Printf(TEXT("GPU runtime '%s' failed to create the model"), *Config.GpuRuntimeName);
			}
		}
	}

	TSharedPtr<UE::NNE::IModelCPU> CpuModel;
	FString CpuFailure;
	if (!CpuRuntime.IsValid())
	{
		CpuFailure = FString::Printf(TEXT("CPU runtime '%s' is not registered"), *Config.CpuRuntimeName);
	}
	else if (CpuRuntime->CanCreateModelCPU(ModelData) != UE::NNE::EResultStatus::Ok)
	{
		CpuFailure = FString::Printf(TEXT("CPU runtime '%s' cannot create a model from this model data"), *Config.CpuRuntimeName);
	}
	else
	{
		CpuModel = CpuRuntime->CreateModelCPU(ModelData);
		if (!CpuModel.IsValid())
		{
			CpuFailure = FString::Printf(TEXT("CPU runtime '%s' failed to create the model"), *Config.CpuRuntimeName);
		}
	}

	if (!GpuModel.IsValid() && !CpuModel.IsValid())
	{
		FString Msg = TEXT("Could not load the model on any allowed device: ");
		if (bWantGpu)
		{
			Msg += GpuFailure + TEXT("; ");
		}
		Msg += CpuFailure;
		OutResult = FPipelineResult::Fail(EPipelineError::ModelLoadFailed, Msg);
		return nullptr;
	}

	if (bWantGpu && !GpuModel.IsValid())
	{
		if (Device == EPipelineDevice::GPU)
		{
			UE_LOG(LogModelPipelines, Warning, TEXT("GPU unavailable for this model, falling back to CPU: %s"), *GpuFailure);
		}
		else
		{
			UE_LOG(LogModelPipelines, Log, TEXT("GPU unavailable for this model, falling back to CPU: %s"), *GpuFailure);
		}
	}

	OutResult = FPipelineResult::Ok();
	return MakeShared<FPipelineNNEModel>(Device, MoveTemp(GpuModel), MoveTemp(CpuModel), Config.GpuRuntimeName, Config.CpuRuntimeName);
}
