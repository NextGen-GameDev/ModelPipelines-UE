// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "VisionPipelineAsyncActions.h"

#include "Async/Async.h"
#include "Engine/Texture.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "ImageClassificationPipeline.h"
#include "ObjectDetectionPipeline.h"
#include "PipelineVisionLog.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(VisionPipelineAsyncActions)

using UE::ModelPipelines::Vision::EAsyncActionState;

namespace UE::ModelPipelines::Vision::AsyncActionPrivate
{
	static FPipelineResult NullPipelineResult()
	{
		return FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("Pipeline is null"));
	}

	/** Engine convention, except that a null context is a silent no-op (the engine would log a script warning). */
	static void Register(UBlueprintAsyncActionBase* Action, const UObject* WorldContextObject)
	{
		if (WorldContextObject)
		{
			Action->RegisterWithGameInstance(WorldContextObject);
		}
	}

	/** Runs Fn on a later game-thread tick if the action still exists. */
	template <typename ActionType, typename FuncType>
	static void DeferOnGameThread(ActionType* Action, FuncType&& Fn)
	{
		AsyncTask(ENamedThreads::GameThread, [WeakAction = TWeakObjectPtr<ActionType>(Action), Fn = MoveTemp(Fn)]()
		{
			if (ActionType* Alive = WeakAction.Get())
			{
				Fn(Alive);
			}
		});
	}
}

// ---------------------------------------------------------------------------------------------------- Classify Image

UClassifyImageAsyncAction* UClassifyImageAsyncAction::ClassifyImage(UObject* WorldContextObject, UImageClassificationPipeline* Pipeline,
	UTexture* Image, int32 TopK)
{
	UClassifyImageAsyncAction* Action = NewObject<UClassifyImageAsyncAction>(GetTransientPackage());
	Action->Pipeline = Pipeline;
	Action->Image = Image;
	Action->TopK = TopK;
	UE::ModelPipelines::Vision::AsyncActionPrivate::Register(Action, WorldContextObject);
	return Action;
}

void UClassifyImageAsyncAction::Activate()
{
	using namespace UE::ModelPipelines::Vision::AsyncActionPrivate;

	if (State != EAsyncActionState::Idle)
	{
		UE_LOG(LogModelPipelinesVision, Warning, TEXT("Classify Image: Activate called again on an action that was already activated or cancelled; ignored"));
		return;
	}
	State = EAsyncActionState::Running;
	bInFlight = true;
	AddToRoot();

	if (!Pipeline)
	{
		UE_LOG(LogModelPipelinesVision, Warning, TEXT("Classify Image: Pipeline is null"));
		DeferOnGameThread(this, [](UClassifyImageAsyncAction* Action)
		{
			Action->HandleResult(NullPipelineResult(), FPipelineClassification());
		});
		return;
	}

	Pipeline->ClassifyAsync(Image, TopK, [WeakThis = TWeakObjectPtr<UClassifyImageAsyncAction>(this)](const FPipelineResult& Result, const FPipelineClassification& Classification)
	{
		if (UClassifyImageAsyncAction* Action = WeakThis.Get())
		{
			Action->HandleResult(Result, Classification);
		}
	});
}

void UClassifyImageAsyncAction::HandleResult(const FPipelineResult& Result, const FPipelineClassification& Classification)
{
	if (!bInFlight)
	{
		return;
	}
	bInFlight = false;
	if (State == EAsyncActionState::Running)
	{
		State = EAsyncActionState::Delivered;
		if (Result.bSuccess)
		{
			OnSuccess.Broadcast(Result, Classification);
		}
		else
		{
			OnFailure.Broadcast(Result, FPipelineClassification());
		}
	}
	Pipeline = nullptr;
	Image = nullptr;
	RemoveFromRoot();
	SetReadyToDestroy();
}

void UClassifyImageAsyncAction::Cancel()
{
	if (State == EAsyncActionState::Idle || State == EAsyncActionState::Running)
	{
		State = EAsyncActionState::Cancelled;
		// The base unregisters from the game instance; the root reference (and the inputs) stay until the dropped result arrives (HandleResult).
		Super::Cancel();
	}
}

bool UClassifyImageAsyncAction::IsActive() const
{
	return State == EAsyncActionState::Running;
}

bool UClassifyImageAsyncAction::ShouldBroadcastDelegates() const
{
	return IsActive();
}

// ---------------------------------------------------------------------------------------------------- Detect Objects

UDetectObjectsAsyncAction* UDetectObjectsAsyncAction::DetectObjects(UObject* WorldContextObject, UObjectDetectionPipeline* Pipeline,
	UTexture* Image, float ScoreThreshold)
{
	UDetectObjectsAsyncAction* Action = NewObject<UDetectObjectsAsyncAction>(GetTransientPackage());
	Action->Pipeline = Pipeline;
	Action->Image = Image;
	Action->ScoreThreshold = ScoreThreshold;
	UE::ModelPipelines::Vision::AsyncActionPrivate::Register(Action, WorldContextObject);
	return Action;
}

void UDetectObjectsAsyncAction::Activate()
{
	using namespace UE::ModelPipelines::Vision::AsyncActionPrivate;

	if (State != EAsyncActionState::Idle)
	{
		UE_LOG(LogModelPipelinesVision, Warning, TEXT("Detect Objects: Activate called again on an action that was already activated or cancelled; ignored"));
		return;
	}
	State = EAsyncActionState::Running;
	bInFlight = true;
	AddToRoot();

	if (!Pipeline)
	{
		UE_LOG(LogModelPipelinesVision, Warning, TEXT("Detect Objects: Pipeline is null"));
		DeferOnGameThread(this, [](UDetectObjectsAsyncAction* Action)
		{
			Action->HandleResult(NullPipelineResult(), TArray<FPipelineDetection>());
		});
		return;
	}

	Pipeline->DetectAsync(Image, ScoreThreshold, [WeakThis = TWeakObjectPtr<UDetectObjectsAsyncAction>(this)](const FPipelineResult& Result, const TArray<FPipelineDetection>& Detections)
	{
		if (UDetectObjectsAsyncAction* Action = WeakThis.Get())
		{
			Action->HandleResult(Result, Detections);
		}
	});
}

void UDetectObjectsAsyncAction::HandleResult(const FPipelineResult& Result, const TArray<FPipelineDetection>& Detections)
{
	if (!bInFlight)
	{
		return;
	}
	bInFlight = false;
	if (State == EAsyncActionState::Running)
	{
		State = EAsyncActionState::Delivered;
		if (Result.bSuccess)
		{
			OnSuccess.Broadcast(Result, Detections);
		}
		else
		{
			OnFailure.Broadcast(Result, TArray<FPipelineDetection>());
		}
	}
	Pipeline = nullptr;
	Image = nullptr;
	RemoveFromRoot();
	SetReadyToDestroy();
}

void UDetectObjectsAsyncAction::Cancel()
{
	if (State == EAsyncActionState::Idle || State == EAsyncActionState::Running)
	{
		State = EAsyncActionState::Cancelled;
		// The base unregisters from the game instance; the root reference (and the inputs) stay until the dropped result arrives (HandleResult).
		Super::Cancel();
	}
}

bool UDetectObjectsAsyncAction::IsActive() const
{
	return State == EAsyncActionState::Running;
}

bool UDetectObjectsAsyncAction::ShouldBroadcastDelegates() const
{
	return IsActive();
}
