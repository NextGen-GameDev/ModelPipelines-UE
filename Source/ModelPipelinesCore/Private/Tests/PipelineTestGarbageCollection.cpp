// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"
#include "UObject/GarbageCollection.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineLog.h"

/**
 * Full garbage collection after every "Pipelines." automation test.
 *
 * The editor only garbage-collects on its timer once a world has begun play (UEngine::ConditionalCollectGarbage), so a
 * headless automation run never collects between tests. Each model test releases its package and pipeline when it
 * ends, but they stay in memory until the next GC: about 1 GB per test for distilbert (file data, ORT model data,
 * session), which ran the machine out of memory (WP7.10). Collecting here frees each test's objects before the next
 * test starts, for every spec of every ModelPipelines module (visible and hidden).
 */
namespace UE::ModelPipelines::Tests::GarbageCollectionPrivate
{
	struct FCollectAfterPipelinesTests
	{
		FDelegateHandle Handle;

		FCollectAfterPipelinesTests()
		{
			Handle = FAutomationTestFramework::Get().OnTestEndEvent.AddStatic(&FCollectAfterPipelinesTests::OnTestEnd);
		}

		~FCollectAfterPipelinesTests()
		{
			FAutomationTestFramework::Get().OnTestEndEvent.Remove(Handle);
		}

		static void OnTestEnd(FAutomationTestBase* Test)
		{
			if (Test == nullptr || !IsInGameThread() || IsGarbageCollecting())
			{
				return;
			}
			if (!Test->GetTestFullName().StartsWith(TEXT("Pipelines."), ESearchCase::CaseSensitive))
			{
				return;
			}
			const double Start = FPlatformTime::Seconds();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS, true);
			UE_LOG(LogModelPipelines, Verbose, TEXT("Automation: garbage collected after %s in %.1f ms"),
				*Test->GetTestFullName(), (FPlatformTime::Seconds() - Start) * 1000.0);
		}
	};

	static FCollectAfterPipelinesTests GCollectAfterPipelinesTests;
}

#endif // WITH_DEV_AUTOMATION_TESTS
