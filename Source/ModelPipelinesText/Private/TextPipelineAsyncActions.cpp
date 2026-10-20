// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#include "TextPipelineAsyncActions.h"

#include "Async/Async.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "PipelineTextPreprocess.h"
#include "TextClassificationPipeline.h"
#include "TextEmbeddingPipeline.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(TextPipelineAsyncActions)

using UE::ModelPipelines::Text::EAsyncActionState;

namespace UE::ModelPipelines::Text::AsyncActionPrivate
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

// ---------------------------------------------------------------------------------------------------- Embed Texts

UEmbedTextsAsyncAction* UEmbedTextsAsyncAction::EmbedTexts(UObject* WorldContextObject, UTextEmbeddingPipeline* Pipeline,
	const TArray<FString>& Texts)
{
	UEmbedTextsAsyncAction* Action = NewObject<UEmbedTextsAsyncAction>(GetTransientPackage());
	Action->Pipeline = Pipeline;
	Action->Texts = Texts;
	UE::ModelPipelines::Text::AsyncActionPrivate::Register(Action, WorldContextObject);
	return Action;
}

void UEmbedTextsAsyncAction::Activate()
{
	using namespace UE::ModelPipelines::Text::AsyncActionPrivate;

	if (State != EAsyncActionState::Idle)
	{
		UE_LOG(LogModelPipelinesText, Warning, TEXT("Embed Texts: Activate called again on an action that was already activated or cancelled; ignored"));
		return;
	}
	State = EAsyncActionState::Running;
	bInFlight = true;
	AddToRoot();

	if (!Pipeline)
	{
		UE_LOG(LogModelPipelinesText, Warning, TEXT("Embed Texts: Pipeline is null"));
		DeferOnGameThread(this, [](UEmbedTextsAsyncAction* Action)
		{
			Action->HandleResult(NullPipelineResult(), TArray<FPipelineEmbedding>());
		});
		return;
	}

	Pipeline->EmbedAsync(Texts, [WeakThis = TWeakObjectPtr<UEmbedTextsAsyncAction>(this)](const FPipelineResult& Result, const TArray<FPipelineEmbedding>& Embeddings)
	{
		if (UEmbedTextsAsyncAction* Action = WeakThis.Get())
		{
			Action->HandleResult(Result, Embeddings);
		}
	});
}

void UEmbedTextsAsyncAction::HandleResult(const FPipelineResult& Result, const TArray<FPipelineEmbedding>& Embeddings)
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
			OnSuccess.Broadcast(Result, Embeddings);
		}
		else
		{
			OnFailure.Broadcast(Result, TArray<FPipelineEmbedding>());
		}
	}
	Pipeline = nullptr;
	RemoveFromRoot();
	SetReadyToDestroy();
}

void UEmbedTextsAsyncAction::Cancel()
{
	if (State == EAsyncActionState::Idle || State == EAsyncActionState::Running)
	{
		State = EAsyncActionState::Cancelled;
		// The base unregisters from the game instance; the root reference stays until the dropped result arrives (HandleResult).
		Super::Cancel();
	}
}

bool UEmbedTextsAsyncAction::IsActive() const
{
	return State == EAsyncActionState::Running;
}

bool UEmbedTextsAsyncAction::ShouldBroadcastDelegates() const
{
	return IsActive();
}

// ---------------------------------------------------------------------------------------------------- Classify Texts

UClassifyTextsAsyncAction* UClassifyTextsAsyncAction::ClassifyTexts(UObject* WorldContextObject, UTextClassificationPipeline* Pipeline,
	const TArray<FString>& Texts, int32 TopK)
{
	UClassifyTextsAsyncAction* Action = NewObject<UClassifyTextsAsyncAction>(GetTransientPackage());
	Action->Pipeline = Pipeline;
	Action->Texts = Texts;
	Action->TopK = TopK;
	UE::ModelPipelines::Text::AsyncActionPrivate::Register(Action, WorldContextObject);
	return Action;
}

void UClassifyTextsAsyncAction::Activate()
{
	using namespace UE::ModelPipelines::Text::AsyncActionPrivate;

	if (State != EAsyncActionState::Idle)
	{
		UE_LOG(LogModelPipelinesText, Warning, TEXT("Classify Texts: Activate called again on an action that was already activated or cancelled; ignored"));
		return;
	}
	State = EAsyncActionState::Running;
	bInFlight = true;
	AddToRoot();

	if (!Pipeline)
	{
		UE_LOG(LogModelPipelinesText, Warning, TEXT("Classify Texts: Pipeline is null"));
		DeferOnGameThread(this, [](UClassifyTextsAsyncAction* Action)
		{
			Action->HandleResult(NullPipelineResult(), TArray<FPipelineClassification>());
		});
		return;
	}

	Pipeline->ClassifyAsync(Texts, TopK, [WeakThis = TWeakObjectPtr<UClassifyTextsAsyncAction>(this)](const FPipelineResult& Result, const TArray<FPipelineClassification>& Classifications)
	{
		if (UClassifyTextsAsyncAction* Action = WeakThis.Get())
		{
			Action->HandleResult(Result, Classifications);
		}
	});
}

void UClassifyTextsAsyncAction::HandleResult(const FPipelineResult& Result, const TArray<FPipelineClassification>& Classifications)
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
			OnSuccess.Broadcast(Result, Classifications);
		}
		else
		{
			OnFailure.Broadcast(Result, TArray<FPipelineClassification>());
		}
	}
	Pipeline = nullptr;
	RemoveFromRoot();
	SetReadyToDestroy();
}

void UClassifyTextsAsyncAction::Cancel()
{
	if (State == EAsyncActionState::Idle || State == EAsyncActionState::Running)
	{
		State = EAsyncActionState::Cancelled;
		// The base unregisters from the game instance; the root reference stays until the dropped result arrives (HandleResult).
		Super::Cancel();
	}
}

bool UClassifyTextsAsyncAction::IsActive() const
{
	return State == EAsyncActionState::Running;
}

bool UClassifyTextsAsyncAction::ShouldBroadcastDelegates() const
{
	return IsActive();
}
