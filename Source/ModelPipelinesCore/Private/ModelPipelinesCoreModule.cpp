// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesCoreModule.h"

#include <atomic>

#include "HAL/CriticalSection.h"
#include "Misc/ScopeLock.h"
#include "Modules/ModuleManager.h"

namespace
{
	/** True from the top of ShutdownModule until the next StartupModule. */
	std::atomic<bool> GModelPipelinesCoreShuttingDown{false};

	/** Guards GModelPipelinesCoreExecutorSlot. */
	FCriticalSection GModelPipelinesCoreExecutorSlotLock;

	/** The module's executor as handed out by FPipelineExecutor::TryGetShared(). Set in StartupModule, cleared at the
	 *  top of ShutdownModule (before the executor is shut down). */
	TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> GModelPipelinesCoreExecutorSlot;
}

// Defined here (not in PipelineExecutor.cpp) because the slot belongs to the module's lifetime.
TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> FPipelineExecutor::TryGetShared()
{
	FScopeLock Lock(&GModelPipelinesCoreExecutorSlotLock);
	return GModelPipelinesCoreExecutorSlot;
}

FModelPipelinesCoreModule& FModelPipelinesCoreModule::Get()
{
	return FModuleManager::LoadModuleChecked<FModelPipelinesCoreModule>("ModelPipelinesCore");
}

FModelPipelinesCoreModule* FModelPipelinesCoreModule::TryGet()
{
	if (GModelPipelinesCoreShuttingDown.load(std::memory_order_acquire))
	{
		return nullptr;
	}
	return FModuleManager::GetModulePtr<FModelPipelinesCoreModule>("ModelPipelinesCore");
}

FPipelineExecutor& FModelPipelinesCoreModule::GetExecutor()
{
	// Programmer error only (called outside the module's lifetime); never driven by user data.
	check(Executor.IsValid());
	return *Executor;
}

IPipelineBackend& FModelPipelinesCoreModule::GetDefaultBackend()
{
	check(DefaultBackend.IsValid());
	return *DefaultBackend;
}

void FModelPipelinesCoreModule::StartupModule()
{
	GModelPipelinesCoreShuttingDown.store(false, std::memory_order_release);
	Executor = MakeShared<FPipelineExecutor, ESPMode::ThreadSafe>();
	DefaultBackend = MakeUnique<FPipelineNNEBackend>();
	{
		FScopeLock Lock(&GModelPipelinesCoreExecutorSlotLock);
		GModelPipelinesCoreExecutorSlot = Executor;
	}
}

void FModelPipelinesCoreModule::ShutdownModule()
{
	GModelPipelinesCoreShuttingDown.store(true, std::memory_order_release);

	// 1. No new holders from here on. The slot's reference is released outside the lock.
	TSharedPtr<FPipelineExecutor, ESPMode::ThreadSafe> SlotRef;
	{
		FScopeLock Lock(&GModelPipelinesCoreExecutorSlotLock);
		Swap(SlotRef, GModelPipelinesCoreExecutorSlot);
	}
	SlotRef.Reset();

	// 2. Stop the executor (joins its worker here, on the game thread). Holders that still have a reference see
	//    Enqueue return 0; the last of them destroys the object, which after Shutdown only frees memory.
	if (Executor.IsValid())
	{
		Executor->Shutdown();
	}
	Executor.Reset();
	DefaultBackend.Reset();
}

IMPLEMENT_MODULE(FModelPipelinesCoreModule, ModelPipelinesCore)
