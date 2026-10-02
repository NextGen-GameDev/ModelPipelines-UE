// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineBase.h"

#include "Async/Async.h"
#include "HAL/CriticalSection.h"
#include "HAL/PlatformTime.h"
#include "Misc/ScopeLock.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "Tasks/Task.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineBackend.h"
#include "PipelineExecutor.h"
#include "PipelineLog.h"
#include "PipelineValidation.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(PipelineBase)

namespace UE::ModelPipelines::PipelineBasePrivate
{
	/** The Error lines of a validation report, one per line. */
	static FString ErrorLines(const FPipelineValidationReport& Report)
	{
		FString Lines;
		for (const FPipelineValidationMessage& M : Report.Messages)
		{
			if (M.Severity == EPipelineValidationSeverity::Error)
			{
				if (!Lines.IsEmpty())
				{
					Lines += TEXT("\n");
				}
				Lines += FString::Printf(TEXT("Error %s: %s"), LexToString(M.Code), *M.Message);
			}
		}
		return Lines;
	}
}

/**
 * Everything an in-flight call needs, without the UObject. Immutable after InitializeFromPackage except for the
 * once-resolved model info, which is written under ResolveLock and read-only afterwards.
 */
struct FPipelineRuntime
{
	TSharedPtr<IPipelineModel> Model;
	EPipelineTask Task = EPipelineTask::None;
	FPipelinePreprocessSettings PreprocessSettings;
	FPipelinePostprocessSettings PostprocessSettings;
	TArray<FString> Labels;

	FCriticalSection ResolveLock;
	bool bResolved = false;
	FPipelineResult ResolveResult;
	FPipelineModelInfo Info;

	/**
	 * Any thread except the game thread for async calls (blocking calls may use it). First call: Model->Initialize and
	 * ResolvePipelineTaskIO; the outcome is cached. OutMs = time spent here by this call on the work (0 when cached).
	 */
	FPipelineResult EnsureResolved(double& OutMs)
	{
		OutMs = 0.0;
		FScopeLock Lock(&ResolveLock);
		if (bResolved)
		{
			return ResolveResult;
		}
		const double Start = FPlatformTime::Seconds();
		FPipelineResult R;
		if (!Model.IsValid())
		{
			R = FPipelineResult::Fail(EPipelineError::Internal, TEXT("pipeline has no prepared model"));
		}
		else
		{
			R = Model->Initialize();
			if (R.bSuccess)
			{
				Info.InputDescs = Model->GetInputDescs();
				Info.OutputDescs = Model->GetOutputDescs();
				FPipelineValidationReport Report;
				const bool bOk = ResolvePipelineTaskIO(Task, PreprocessSettings, PostprocessSettings, Labels.Num(),
					Info.InputDescs, Info.OutputDescs, Info.Bindings, Report);
				if (!bOk || !Report.IsValid())
				{
					FString Lines = UE::ModelPipelines::PipelineBasePrivate::ErrorLines(Report);
					if (Lines.IsEmpty())
					{
						Lines = TEXT("model inputs/outputs do not fit the task");
					}
					R = FPipelineResult::Fail(EPipelineError::ShapeMismatch, Lines);
					R.Device = Model->GetDevice();
				}
				else
				{
					R = FPipelineResult::Ok();
				}
			}
		}
		ResolveResult = R;
		bResolved = true;
		OutMs = (FPlatformTime::Seconds() - Start) * 1000.0;
		return ResolveResult;
	}
};

namespace UE::ModelPipelines::PipelineBasePrivate
{
	/** One RunStages call in flight. Shared between the stage steps; holds no UObject. */
	struct FCallState
	{
		TSharedPtr<FPipelineRuntime, ESPMode::ThreadSafe> Runtime;
		TSharedPtr<FThreadSafeCounter, ESPMode::ThreadSafe> Pending;
		FPipelineStages Stages;
		TArray<FPipelineTensor> Outputs;

		EPipelineDevice Device = EPipelineDevice::Auto;
		double QueueMs = 0.0;
		double PreprocessMs = 0.0;
		double InferenceMs = 0.0;
		double PostprocessMs = 0.0;
	};

	using FCallStateRef = TSharedPtr<FCallState, ESPMode::ThreadSafe>;

	/** Copies the timings gathered so far into Result. A stage's own Error and Message are left unchanged. */
	static void ApplyTimings(FPipelineResult& Result, double QueueMs, double PreprocessMs, double InferenceMs,
		double PostprocessMs, EPipelineDevice Device)
	{
		Result.QueueMs = (float)QueueMs;
		Result.PreprocessMs = (float)PreprocessMs;
		Result.InferenceMs = (float)InferenceMs;
		Result.PostprocessMs = (float)PostprocessMs;
		if (Device != EPipelineDevice::Auto)
		{
			Result.Device = Device;
		}
	}

	/** Game thread. Last step of a call: counts it done, calls Complete, destroys the stage functions here. */
	static void Finish(const FCallStateRef& State, FPipelineResult Result)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Complete");
		ApplyTimings(Result, State->QueueMs, State->PreprocessMs, State->InferenceMs, State->PostprocessMs, State->Device);

		// Take the stage functions out of the shared state so they die on this thread, whoever drops the state last.
		FPipelineStages Stages = MoveTemp(State->Stages);
		State->Stages = FPipelineStages();
		State->Outputs.Empty();

		if (State->Pending.IsValid())
		{
			State->Pending->Decrement();
		}
		if (Stages.Complete)
		{
			Stages.Complete(Result);
		}
		else
		{
			UE_LOG(LogModelPipelines, Verbose, TEXT("Pipeline call finished without a Complete function (error %d)"), (int32)Result.Error);
		}
	}

	/** Any thread. Posts Finish to the game thread (never runs it inline). */
	static void PostFinish(const FCallStateRef& State, const FPipelineResult& Result)
	{
		AsyncTask(ENamedThreads::GameThread, [State, Result]()
		{
			Finish(State, Result);
		});
	}

	/** Background task. Postprocess, then Complete on the game thread. */
	static void RunPostprocess(const FCallStateRef& State)
	{
		FPipelineResult R;
		{
			TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Postprocess");
			const double Start = FPlatformTime::Seconds();
			if (State->Stages.Postprocess)
			{
				R = State->Stages.Postprocess(State->Runtime->Info, MoveTemp(State->Outputs));
			}
			else
			{
				R = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("pipeline call has no Postprocess function"));
			}
			State->PostprocessMs = (FPlatformTime::Seconds() - Start) * 1000.0;
		}
		State->Outputs.Empty();
		PostFinish(State, R.bSuccess ? FPipelineResult::Ok() : R);
	}

	/** Game thread (executor callback). Failure -> Complete now; success -> Postprocess on a background task. */
	static void OnInferenceDone(const FCallStateRef& State, const FPipelineResult& InferResult, TArray<FPipelineTensor>&& Outputs)
	{
		State->Device = InferResult.Device;
		State->QueueMs = InferResult.QueueMs;
		State->InferenceMs += InferResult.InferenceMs;
		if (!InferResult.bSuccess)
		{
			Finish(State, InferResult);
			return;
		}
		State->Outputs = MoveTemp(Outputs);
		UE::Tasks::Launch(TEXT("ModelPipelines.Postprocess"), [State]()
		{
			RunPostprocess(State);
		});
	}

	/** Background task. Resolve (first call), Preprocess, then enqueue inference. */
	static void RunPreprocessAndEnqueue(const FCallStateRef& State)
	{
		FPipelineRuntime& Rt = *State->Runtime;

		double ResolveMs = 0.0;
		const FPipelineResult Resolved = Rt.EnsureResolved(ResolveMs);
		State->InferenceMs += ResolveMs;
		if (!Resolved.bSuccess)
		{
			PostFinish(State, Resolved);
			return;
		}

		TArray<FPipelineTensor> Inputs;
		FPipelineResult R;
		{
			TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Preprocess");
			const double Start = FPlatformTime::Seconds();
			if (State->Stages.Preprocess)
			{
				R = State->Stages.Preprocess(Rt.Info, Inputs);
			}
			else
			{
				R = FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("pipeline call has no Preprocess function"));
			}
			State->PreprocessMs = (FPlatformTime::Seconds() - Start) * 1000.0;
		}
		if (!R.bSuccess)
		{
			PostFinish(State, R);
			return;
		}

		FModelPipelinesCoreModule* Module = FModelPipelinesCoreModule::TryGet();
		if (Module == nullptr)
		{
			PostFinish(State, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("ModelPipelinesCore module is not loaded")));
			return;
		}
		const FPipelineJobId Id = Module->GetExecutor().Enqueue(Rt.Model, MoveTemp(Inputs),
			[State](const FPipelineResult& InferResult, TArray<FPipelineTensor>&& Outputs)
			{
				OnInferenceDone(State, InferResult, MoveTemp(Outputs));
			},
			nullptr, 0);
		if (Id == 0)
		{
			PostFinish(State, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("the executor did not accept the inference job (shut down)")));
		}
	}

	static const FPipelinePreprocessSettings& DefaultPreprocess()
	{
		static const FPipelinePreprocessSettings Default;
		return Default;
	}

	static const FPipelinePostprocessSettings& DefaultPostprocess()
	{
		static const FPipelinePostprocessSettings Default;
		return Default;
	}

	static const TArray<FString>& DefaultLabels()
	{
		static const TArray<FString> Default;
		return Default;
	}
}

FPipelineResult UPipelineBase::InitializeFromPackage(UPipelineModelPackage* InPackage)
{
	if (!IsInGameThread())
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("InitializeFromPackage must be called on the game thread"));
	}
	FModelPipelinesCoreModule* Module = FModelPipelinesCoreModule::TryGet();
	if (Module == nullptr)
	{
		return FPipelineResult::Fail(EPipelineError::RuntimeUnavailable, TEXT("ModelPipelinesCore module is not loaded"));
	}
	return InitializeFromPackage(InPackage, Module->GetDefaultBackend());
}

FPipelineResult UPipelineBase::InitializeFromPackage(UPipelineModelPackage* InPackage, IPipelineBackend& Backend)
{
	using namespace UE::ModelPipelines::PipelineBasePrivate;

	if (!IsInGameThread())
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("InitializeFromPackage must be called on the game thread"));
	}
	if (bReady)
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("pipeline is already initialized"));
	}
	if (InPackage == nullptr)
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("null model package"));
	}
	if (InPackage->Task != PipelineTask)
	{
		return FPipelineResult::Fail(EPipelineError::WrongTask, FString::Printf(TEXT("package task is %s, this pipeline needs %s"),
			LexToString(InPackage->Task), LexToString(PipelineTask)));
	}

	FPipelineValidationReport Report;
	const bool bSettingsOk = InPackage->ValidateSettings(Report);
	if (!bSettingsOk || !Report.IsValid())
	{
		FString Lines = ErrorLines(Report);
		if (Lines.IsEmpty())
		{
			Lines = TEXT("package settings are invalid");
		}
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, Lines);
	}

	FPipelineResult PrepareResult;
	TSharedPtr<IPipelineModel> Model = Backend.PrepareModel(InPackage->Model, InPackage->PreferredDevice, PrepareResult);
	if (!Model.IsValid())
	{
		if (PrepareResult.bSuccess || PrepareResult.Error == EPipelineError::None)
		{
			PrepareResult = FPipelineResult::Fail(EPipelineError::ModelLoadFailed, TEXT("the backend returned no model"));
		}
		return PrepareResult;
	}

	TSharedPtr<FPipelineRuntime, ESPMode::ThreadSafe> NewRuntime = MakeShared<FPipelineRuntime, ESPMode::ThreadSafe>();
	NewRuntime->Model = MoveTemp(Model);
	NewRuntime->Task = PipelineTask;
	NewRuntime->PreprocessSettings = InPackage->Preprocess;
	NewRuntime->PostprocessSettings = InPackage->Postprocess;
	NewRuntime->Labels = InPackage->Labels;

	// Visible to OnInitialize (settings getters, GetPackage), removed again if it fails.
	Runtime = NewRuntime;
	Package = InPackage;

	FPipelineResult SubResult = OnInitialize(InPackage);
	if (!SubResult.bSuccess)
	{
		Runtime.Reset();
		Package = nullptr;
		if (SubResult.Error == EPipelineError::None)
		{
			SubResult.Error = EPipelineError::Internal;
		}
		return SubResult;
	}

	bReady = true;
	return FPipelineResult::Ok();
}

bool UPipelineBase::IsReady() const
{
	return bReady && Runtime.IsValid();
}

EPipelineTask UPipelineBase::GetTask() const
{
	return PipelineTask;
}

UPipelineModelPackage* UPipelineBase::GetPackage() const
{
	return Package;
}

EPipelineDevice UPipelineBase::GetDevice() const
{
	const TSharedPtr<FPipelineRuntime, ESPMode::ThreadSafe> Rt = Runtime;
	if (Rt.IsValid() && Rt->Model.IsValid() && Rt->Model->IsInitialized())
	{
		return Rt->Model->GetDevice();
	}
	return EPipelineDevice::Auto;
}

int32 UPipelineBase::GetNumPendingCalls() const
{
	return PendingCalls.IsValid() ? PendingCalls->GetValue() : 0;
}

FPipelineResult UPipelineBase::OnInitialize(UPipelineModelPackage* InPackage)
{
	return FPipelineResult::Ok();
}

void UPipelineBase::RunStages(FPipelineStages&& Stages)
{
	using namespace UE::ModelPipelines::PipelineBasePrivate;

	FCallStateRef State = MakeShared<FCallState, ESPMode::ThreadSafe>();
	State->Pending = PendingCalls;
	State->Stages = MoveTemp(Stages);
	if (State->Pending.IsValid())
	{
		State->Pending->Increment();
	}

	if (!IsInGameThread())
	{
		PostFinish(State, FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("RunStages must be called on the game thread")));
		return;
	}
	if (!IsReady())
	{
		PostFinish(State, FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("pipeline is not initialized (call InitializeFromPackage first)")));
		return;
	}

	State->Runtime = Runtime;
	UE::Tasks::Launch(TEXT("ModelPipelines.Preprocess"), [State]()
	{
		RunPreprocessAndEnqueue(State);
	});
}

FPipelineResult UPipelineBase::RunStagesBlocking(FPipelineStages&& Stages)
{
	using namespace UE::ModelPipelines::PipelineBasePrivate;

	const TSharedPtr<FPipelineRuntime, ESPMode::ThreadSafe> Rt = Runtime;
	if (!bReady || !Rt.IsValid())
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("pipeline is not initialized (call InitializeFromPackage first)"));
	}

	double QueueMs = 0.0;
	double PreprocessMs = 0.0;
	double InferenceMs = 0.0;
	double PostprocessMs = 0.0;
	EPipelineDevice Device = EPipelineDevice::Auto;

	double ResolveMs = 0.0;
	FPipelineResult Resolved = Rt->EnsureResolved(ResolveMs);
	InferenceMs += ResolveMs;
	if (!Resolved.bSuccess)
	{
		ApplyTimings(Resolved, QueueMs, PreprocessMs, InferenceMs, PostprocessMs, Device);
		return Resolved;
	}

	TArray<FPipelineTensor> Inputs;
	FPipelineResult R;
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Preprocess");
		const double Start = FPlatformTime::Seconds();
		R = Stages.Preprocess
			? Stages.Preprocess(Rt->Info, Inputs)
			: FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("pipeline call has no Preprocess function"));
		PreprocessMs = (FPlatformTime::Seconds() - Start) * 1000.0;
	}
	if (!R.bSuccess)
	{
		ApplyTimings(R, QueueMs, PreprocessMs, InferenceMs, PostprocessMs, Device);
		return R;
	}

	FModelPipelinesCoreModule* Module = FModelPipelinesCoreModule::TryGet();
	if (Module == nullptr)
	{
		R = FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("ModelPipelinesCore module is not loaded"));
		ApplyTimings(R, QueueMs, PreprocessMs, InferenceMs, PostprocessMs, Device);
		return R;
	}
	TArray<FPipelineTensor> Outputs;
	FPipelineResult InferResult = Module->GetExecutor().RunBlocking(Rt->Model, Inputs, Outputs);
	Inputs.Empty();
	QueueMs = InferResult.QueueMs;
	InferenceMs += InferResult.InferenceMs;
	Device = InferResult.Device;
	if (!InferResult.bSuccess)
	{
		ApplyTimings(InferResult, QueueMs, PreprocessMs, InferenceMs, PostprocessMs, Device);
		return InferResult;
	}

	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Postprocess");
		const double Start = FPlatformTime::Seconds();
		R = Stages.Postprocess
			? Stages.Postprocess(Rt->Info, MoveTemp(Outputs))
			: FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("pipeline call has no Postprocess function"));
		PostprocessMs = (FPlatformTime::Seconds() - Start) * 1000.0;
	}
	FPipelineResult Final = R.bSuccess ? FPipelineResult::Ok() : R;
	ApplyTimings(Final, QueueMs, PreprocessMs, InferenceMs, PostprocessMs, Device);
	return Final;
}

const FPipelinePreprocessSettings& UPipelineBase::GetPreprocessSettings() const
{
	return Runtime.IsValid() ? Runtime->PreprocessSettings : UE::ModelPipelines::PipelineBasePrivate::DefaultPreprocess();
}

const FPipelinePostprocessSettings& UPipelineBase::GetPostprocessSettings() const
{
	return Runtime.IsValid() ? Runtime->PostprocessSettings : UE::ModelPipelines::PipelineBasePrivate::DefaultPostprocess();
}

const TArray<FString>& UPipelineBase::GetLabels() const
{
	return Runtime.IsValid() ? Runtime->Labels : UE::ModelPipelines::PipelineBasePrivate::DefaultLabels();
}
