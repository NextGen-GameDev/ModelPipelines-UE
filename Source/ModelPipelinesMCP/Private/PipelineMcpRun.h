// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Templates/SharedPointer.h"

class UObject;
class UPipelineMcpJob;

/**
 * The work behind one UPipelineMcpJob (WP6.2). Private to ModelPipelinesMCP. Game thread only.
 * Pending pipeline callbacks hold a TSharedRef to the runner; the runner holds the job only weakly (the job keeps itself
 * alive while running) and ignores callbacks that arrive after the job ended.
 */
class FPipelineMcpRunner : public TSharedFromThis<FPipelineMcpRunner>
{
public:
	virtual ~FPipelineMcpRunner() = default;

	/** Game thread, at least one tick after the job was created, called once. Must end the job (now or from a later
	    callback) with FPipelineMcpJobInternal::Finish. */
	virtual void Start(UPipelineMcpJob& Job) = 0;
};

/** Module-internal control of UPipelineMcpJob (friend of the class). Game thread only. */
struct FPipelineMcpJobInternal
{
	/** New job, rooted until it ends, registered for module shutdown; Runner->Start runs from the core ticker on a later tick. */
	static UPipelineMcpJob* Launch(const TSharedRef<FPipelineMcpRunner>& Runner);

	/** A job that fails with {"error":"<Message>"} on a later tick. */
	static UPipelineMcpJob* LaunchError(const FString& Message);

	/** Keeps Object alive (strong reference from the job) until the job ends or Release is called. Ignored once done. */
	static void KeepAlive(UPipelineMcpJob& Job, UObject* Object);

	/** Drops the job's reference to Object (it may be collected afterwards). */
	static void Release(UPipelineMcpJob& Job, UObject* Object);

	/** Ends the job once (later calls are ignored): stores the result, broadcasts OnDone (unless bBroadcast is false), then
	    drops every reference and un-roots the job so it can be collected. */
	static void Finish(UPipelineMcpJob& Job, bool bSuccess, const FString& Json, bool bBroadcast = true);

	/** Finish with JSON {"error":"<Message>"}. */
	static void Fail(UPipelineMcpJob& Job, const FString& Message);

	/** Module shutdown: ends every running job with {"error":"<Message>"}; the broadcast is skipped when the engine is exiting. */
	static void CancelAll(const FString& Message);
};

namespace PipelineMcpRun
{
	/** Game thread. Job behind UPipelineMcpLibrary::BenchmarkModelPackage (all argument checks happen inside the job). */
	UPipelineMcpJob* LaunchBenchmark(const FString& Package, int32 Runs, const FString& Devices, const TArray<FString>& Texts, const FString& Image);

	/** Game thread. Job behind UPipelineMcpLibrary::RunPipelineSample (all argument checks happen inside the job). */
	UPipelineMcpJob* LaunchSample(const FString& Package, const TArray<FString>& Texts, const FString& Image, int32 TopK, float ScoreThreshold);
}
