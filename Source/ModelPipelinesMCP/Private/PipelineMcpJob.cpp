// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "PipelineMcpJob.h"

#include "Containers/Ticker.h"
#include "CoreGlobals.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "PipelineMcpJson.h"
#include "PipelineMcpLog.h"
#include "PipelineMcpRun.h"

namespace PipelineMcpJobPrivate
{
	/** Running jobs (game thread only). Each is also rooted, so the weak pointers stay valid until Finish removes them. */
	static TArray<TWeakObjectPtr<UPipelineMcpJob>> GRunningJobs;

	/** Runner of LaunchError: fails with a fixed message when started. */
	class FErrorRunner final : public FPipelineMcpRunner
	{
	public:
		explicit FErrorRunner(const FString& InMessage) : Message(InMessage) {}

		virtual void Start(UPipelineMcpJob& Job) override
		{
			FPipelineMcpJobInternal::Fail(Job, Message);
		}

	private:
		FString Message;
	};
}

bool UPipelineMcpJob::IsDone() const
{
	return bDone;
}

bool UPipelineMcpJob::Succeeded() const
{
	return bDone && bSucceeded;
}

FString UPipelineMcpJob::GetJson() const
{
	return bDone ? ResultJson : FString();
}

UPipelineMcpJob* FPipelineMcpJobInternal::Launch(const TSharedRef<FPipelineMcpRunner>& Runner)
{
	using namespace PipelineMcpJobPrivate;

	if (!IsInGameThread())
	{
		UE_LOG(LogModelPipelinesMCP, Warning, TEXT("UPipelineMcpJob can only be created on the game thread; no job was started."));
		return nullptr;
	}

	UPipelineMcpJob* Job = NewObject<UPipelineMcpJob>(GetTransientPackage(), NAME_None, RF_Transient);
	Job->Runner = Runner;
	// Kept alive while running (GC sees the root; KeepAlive is a UPROPERTY, so everything it holds is reachable too).
	Job->AddToRoot();
	GRunningJobs.Add(Job);

	// Started from the core ticker: the creating call always returns before anything can complete.
	TWeakObjectPtr<UPipelineMcpJob> WeakJob(Job);
	Job->StartTicker = FTSTicker::GetCoreTicker().AddTicker(TEXT("PipelineMcpJobStart"), 0.0f, [WeakJob](float) -> bool
	{
		UPipelineMcpJob* J = WeakJob.Get();
		if (!J || J->bDone)
		{
			return false;
		}
		J->StartTicker.Reset();
		// Local strong ref: the runner may end the job (releasing Job->Runner) inside Start.
		const TSharedPtr<FPipelineMcpRunner> R = J->Runner;
		if (!R.IsValid())
		{
			FPipelineMcpJobInternal::Fail(*J, TEXT("internal: job has no work"));
			return false;
		}
		R->Start(*J);
		return false;
	});
	return Job;
}

UPipelineMcpJob* FPipelineMcpJobInternal::LaunchError(const FString& Message)
{
	return Launch(MakeShared<PipelineMcpJobPrivate::FErrorRunner>(Message));
}

void FPipelineMcpJobInternal::KeepAlive(UPipelineMcpJob& Job, UObject* Object)
{
	if (Object && !Job.bDone)
	{
		Job.KeepAlive.AddUnique(Object);
	}
}

void FPipelineMcpJobInternal::Release(UPipelineMcpJob& Job, UObject* Object)
{
	if (Object)
	{
		Job.KeepAlive.Remove(Object);
	}
}

void FPipelineMcpJobInternal::Finish(UPipelineMcpJob& Job, bool bSuccess, const FString& Json, bool bBroadcast)
{
	using namespace PipelineMcpJobPrivate;

	if (Job.bDone)
	{
		return;
	}
	Job.bDone = true;
	Job.bSucceeded = bSuccess;
	Job.ResultJson = Json;

	if (Job.StartTicker.IsValid())
	{
		FTSTicker::RemoveTicker(Job.StartTicker);
		Job.StartTicker.Reset();
	}
	GRunningJobs.RemoveAll([&Job](const TWeakObjectPtr<UPipelineMcpJob>& W) { return !W.IsValid() || W.Get() == &Job; });

	// The runner may be the caller (Finish from inside its code); keep it alive until this function returns.
	const TSharedPtr<FPipelineMcpRunner> RunnerHold = MoveTemp(Job.Runner);
	Job.Runner.Reset();

	if (bBroadcast)
	{
		// Still rooted here: handlers may run garbage collection safely.
		Job.OnDone.Broadcast(bSuccess, Json);
	}

	Job.KeepAlive.Reset();
	if (Job.IsRooted())
	{
		Job.RemoveFromRoot();
	}
}

void FPipelineMcpJobInternal::Fail(UPipelineMcpJob& Job, const FString& Message)
{
	Finish(Job, false, PipelineMcpJson::MakeError(Message));
}

void FPipelineMcpJobInternal::CancelAll(const FString& Message)
{
	using namespace PipelineMcpJobPrivate;

	if (!UObjectInitialized())
	{
		GRunningJobs.Reset();
		return;
	}
	const bool bBroadcast = !IsEngineExitRequested();
	const FString Json = PipelineMcpJson::MakeError(Message);
	const TArray<TWeakObjectPtr<UPipelineMcpJob>> Jobs = GRunningJobs;
	for (const TWeakObjectPtr<UPipelineMcpJob>& W : Jobs)
	{
		if (UPipelineMcpJob* J = W.Get())
		{
			Finish(*J, false, Json, bBroadcast);
		}
	}
	GRunningJobs.Reset();
}
