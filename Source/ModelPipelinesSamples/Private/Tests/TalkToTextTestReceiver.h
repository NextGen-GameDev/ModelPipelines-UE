// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "HAL/PlatformTime.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "PipelineTypes.h"
#include "SpeechTypes.h"
#include "ModelPipelinesTalkToTextSample.h"
#include "TalkToTextTestReceiver.generated.h"

/**
 * Listener for AModelPipelinesTalkToTextSample::OnFinalText and AModelPipelinesSampleBase::OnSampleResult (dynamic delegates,
 * so the handlers are UFUNCTIONs). Every delivery is recorded with the time it arrived and, for finals, what the sample
 * reported at that moment (partial text, latency). Not thread-safe on purpose: an off-game-thread delivery sets bAllOnGameThread=false.
 */
UCLASS()
class UTalkToTextTestReceiver : public UObject
{
	GENERATED_BODY()

public:
	TWeakObjectPtr<AModelPipelinesTalkToTextSample> Sample;

	TArray<FString> Texts;
	TArray<double> TextTimes;
	/** sample->GetPartialText() when OnFinalText arrived. */
	TArray<FString> PartialAtFinal;
	/** sample->GetLastFinalLatencySeconds() when OnFinalText arrived. */
	TArray<float> LatencyAtFinal;

	TArray<bool> ResultSuccess;
	TArray<double> ResultTimes;

	bool bAllOnGameThread = true;

	UFUNCTION()
	void OnFinal(const FSpeechTranscript& Transcript)
	{
		Texts.Add(Transcript.Text);
		TextTimes.Add(FPlatformTime::Seconds());
		const AModelPipelinesTalkToTextSample* S = Sample.Get();
		PartialAtFinal.Add(S ? S->GetPartialText() : FString(TEXT("<sample gone>")));
		LatencyAtFinal.Add(S ? S->GetLastFinalLatencySeconds() : -99.f);
		if (!IsInGameThread()) { bAllOnGameThread = false; }
	}

	UFUNCTION()
	void OnResult(const FPipelineResult& Result)
	{
		ResultSuccess.Add(Result.bSuccess);
		ResultTimes.Add(FPlatformTime::Seconds());
		if (!IsInGameThread()) { bAllOnGameThread = false; }
	}
};
