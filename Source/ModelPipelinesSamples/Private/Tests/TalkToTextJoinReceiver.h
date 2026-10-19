// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

// WP9R.7t helper (extra file, needed because dynamic delegates bind only to UFUNCTIONs of a UCLASS in a header): records the text AND
// bReplacesPrevious of every OnFinalText, plus every OnSampleResult. UTalkToTextTestReceiver does not keep bReplacesPrevious.

#include "CoreMinimal.h"
#include "UObject/Object.h"
#include "UObject/ObjectMacros.h"
#include "PipelineResults.h"
#include "SpeechTypes.h"
#include "TalkToTextJoinReceiver.generated.h"

UCLASS()
class UTalkToTextJoinReceiver : public UObject
{
	GENERATED_BODY()

public:
	TArray<FString> Texts;
	TArray<bool> Replaces;
	TArray<bool> ResultSuccess;
	bool bAllOnGameThread = true;

	UFUNCTION()
	void OnFinal(const FSpeechTranscript& Transcript)
	{
		Texts.Add(Transcript.Text);
		Replaces.Add(Transcript.bReplacesPrevious);
		if (!IsInGameThread()) { bAllOnGameThread = false; }
	}

	UFUNCTION()
	void OnResult(const FPipelineResult& Result)
	{
		ResultSuccess.Add(Result.bSuccess);
		if (!IsInGameThread()) { bAllOnGameThread = false; }
	}
};
