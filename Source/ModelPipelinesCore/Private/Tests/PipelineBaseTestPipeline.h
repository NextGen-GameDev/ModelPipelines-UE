// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

// Not inside WITH_DEV_AUTOMATION_TESTS: UHT ignores preprocessor guards, so the class exists in every build.

#include "CoreMinimal.h"
#include "PipelineBase.h"
#include "PipelineBaseTestPipeline.generated.h"

/** Concrete UPipelineBase for the WP3.2 tests (TextClassification). Exposes the protected API. */
UCLASS(Transient, NotBlueprintable)
class UPipelineBaseTestPipeline : public UPipelineBase
{
	GENERATED_BODY()

public:
	UPipelineBaseTestPipeline();

	/** Calls the protected RunStages. */
	void Run(FPipelineStages&& Stages) { RunStages(MoveTemp(Stages)); }

	/** Calls the protected RunStagesBlocking. */
	FPipelineResult RunBlocking(FPipelineStages&& Stages) { return RunStagesBlocking(MoveTemp(Stages)); }

	/** Public views of the protected snapshot getters. */
	const FPipelinePreprocessSettings& PeekPreprocessSettings() const { return GetPreprocessSettings(); }
	const FPipelinePostprocessSettings& PeekPostprocessSettings() const { return GetPostprocessSettings(); }
	const TArray<FString>& PeekLabels() const { return GetLabels(); }
};
