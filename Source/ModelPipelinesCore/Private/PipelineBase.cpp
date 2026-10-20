// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

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
 * once-resolved state (ResolveResult, ResolveMs, Info), which only the resolve task writes and which is read only after
 * that task has completed (a call either has it as a prerequisite or waits on it), so it needs no lock.
 */
struct FPipelineRuntime
{
	TSharedPtr<IPipelineModel> Model;
	EPipelineTask Task = EPipelineTask::None;
	FPipelinePreprocessSettings PreprocessSettings;
	FPipelinePostprocessSettings PostprocessSettings;
	TArray<FString> Labels;

	/** Guards ResolveTask creation only (held for the Launch call, never across the work). */
	FCriticalSection ResolveLock;
	/** The one resolve task; invalid until the first call creates it. Kept so later calls can depend on it. */
	UE::Tasks::FTask ResolveTask;

	/** Written by the resolve task; read-only once it completed. */
	FPipelineResult ResolveResult;
	double ResolveMs = 0.0;
	FPipelineModelInfo Info;

	/**
	 * Any thread. Returns the resolve task, launching it on the first call (bOutLaunched = true for that call only, so
	 * exactly one call adds ResolveMs to its InferenceMs). Callers either launch work with it as a prerequisite or wait
	 * on it; nobody holds a lock or occupies a worker while Model->Initialize runs.
	 */
	static UE::Tasks::FTask GetOrLaunchResolve(const TSharedPtr<FPipelineRuntime, ESPMode::ThreadSafe>& Self, bool& bOutLaunched)
	{
		bOutLaunched = false;
		FScopeLock Lock(&Self->ResolveLock);
		if (!Self->ResolveTask.IsValid())
		{
			// Weak: the runtime holds the task, so the task must not hold the runtime. Every caller holds a strong
			// reference until the task has completed, so the pin only fails if nobody is left to read the outcome.
			const TWeakPtr<FPipelineRuntime, ESPMode::ThreadSafe> Weak = Self;
			Self->ResolveTask = UE::Tasks::Launch(TEXT("ModelPipelines.Resolve"), [Weak]()
			{
				if (const TSharedPtr<FPipelineRuntime, ESPMode::ThreadSafe> Pinned = Weak.Pin())
				{
					Pinned->DoResolve();
				}
			}, UE::Tasks::ETaskPriority::BackgroundNormal);
			bOutLaunched = true;
		}
		return Self->ResolveTask;
	}

private:
	/** Resolve task body, runs once: Model->Initialize and ResolvePipelineTaskIO; the outcome is cached. */
	void DoResolve()
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Resolve");
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
		ResolveMs = (FPlatformTime::Seconds() - Start) * 1000.0;
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

		/** Set by Finish when the stages are handed over (Complete runs). Game thread only. */
		bool bFinished = false;

		FCallState() = default;
		FCallState(const FCallState&) = delete;
		FCallState& operator=(const FCallState&) = delete;

		/** Delivers Fail(Cancelled) if the call was dropped (e.g. by the executor) before Complete ran. */
		~FCallState();
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
		State->bFinished = true;
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

	FCallState::~FCallState()
	{
		if (bFinished)
		{
			return; // normal path: Finish already took the stages and ran Complete
		}
		bFinished = true;

		if (IsEngineExitRequested())
		{
			// The game thread may never run another task: no Complete. The stage functions are destroyed here, on
			// whatever thread drops the call last (accepted at engine exit only).
			if (Pending.IsValid())
			{
				Pending->Decrement();
			}
			return;
		}

		// The call was dropped before Complete (the executor shut down with the job queued, or was destroyed with the
		// job waiting for dispatch). This object is going away, so hand the stages and timings to a fresh state and
		// finish that one. Its own destructor sees bFinished once Finish ran; if its posted task is itself dropped, it
		// lands back here (at engine exit, the branch above).
		FCallStateRef Orphan = MakeShared<FCallState, ESPMode::ThreadSafe>();
		Orphan->Pending = MoveTemp(Pending);
		Orphan->Stages = MoveTemp(Stages);
		Orphan->Device = Device;
		Orphan->QueueMs = QueueMs;
		Orphan->PreprocessMs = PreprocessMs;
		Orphan->InferenceMs = InferenceMs;
		Orphan->PostprocessMs = PostprocessMs;
		const FPipelineResult Cancelled = FPipelineResult::Fail(EPipelineError::Cancelled,
			TEXT("the executor shut down before this call completed"));
		if (IsInGameThread())
		{
			Finish(Orphan, Cancelled);
		}
		else
		{
			PostFinish(Orphan, Cancelled);
		}
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
		}, UE::Tasks::ETaskPriority::BackgroundNormal);
	}

	/**
	 * Background task, launched with the runtime's resolve task as its prerequisite, so the resolve outcome is final
	 * here. Preprocess, then enqueue inference.
	 */
	static void RunPreprocessAndEnqueue(const FCallStateRef& State, bool bLaunchedResolve)
	{
		FPipelineRuntime& Rt = *State->Runtime;

		if (bLaunchedResolve)
		{
			State->InferenceMs += Rt.ResolveMs;
		}
		const FPipelineResult Resolved = Rt.ResolveResult;
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

		// A shared reference: the module may shut down on the game thread at any moment; the executor then stays alive
		// for this call and Enqueue returns 0.
		const TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor = FPipelineExecutor::TryGetShared();
		if (!Executor.IsValid())
		{
			PostFinish(State, FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("ModelPipelinesCore module is not loaded")));
			return;
		}
		const FPipelineJobId Id = Executor->Enqueue(Rt.Model, MoveTemp(Inputs),
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

	// Optional caller prerequisite (e.g. a GPU readback). An invalid handle is skipped by the task system
	// (Core/Public/Tasks/TaskPrivate.h, null prerequisites count as completed). The state no longer needs the handle.
	UE::Tasks::FTask CallerPrerequisite = MoveTemp(State->Stages.Prerequisite);
	State->Stages.Prerequisite = UE::Tasks::FTask();

	// The first call launches the resolve task (Model->Initialize + ResolvePipelineTaskIO); every call's preprocess
	// task waits for it as a prerequisite, so queued calls don't occupy a worker while the model initializes.
	bool bLaunchedResolve = false;
	UE::Tasks::FTask ResolveTask = FPipelineRuntime::GetOrLaunchResolve(Runtime, bLaunchedResolve);
	UE::Tasks::Launch(TEXT("ModelPipelines.Preprocess"), [State, bLaunchedResolve]()
	{
		RunPreprocessAndEnqueue(State, bLaunchedResolve);
	}, UE::Tasks::Prerequisites(ResolveTask, CallerPrerequisite), UE::Tasks::ETaskPriority::BackgroundNormal);
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

	// Caller prerequisite first (tests and tools only, so waiting is allowed here).
	if (Stages.Prerequisite.IsValid())
	{
		Stages.Prerequisite.Wait();
	}

	// Tests and tools only: waiting here is allowed (it may run the resolve inline if no worker picked it up yet).
	bool bLaunchedResolve = false;
	const UE::Tasks::FTask ResolveTask = FPipelineRuntime::GetOrLaunchResolve(Rt, bLaunchedResolve);
	ResolveTask.Wait();
	if (bLaunchedResolve)
	{
		InferenceMs += Rt->ResolveMs;
	}
	FPipelineResult Resolved = Rt->ResolveResult;
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

	const TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> Executor = FPipelineExecutor::TryGetShared();
	if (!Executor.IsValid())
	{
		R = FPipelineResult::Fail(EPipelineError::Cancelled, TEXT("ModelPipelinesCore module is not loaded"));
		ApplyTimings(R, QueueMs, PreprocessMs, InferenceMs, PostprocessMs, Device);
		return R;
	}
	TArray<FPipelineTensor> Outputs;
	FPipelineResult InferResult = Executor->RunBlocking(Rt->Model, Inputs, Outputs);
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
