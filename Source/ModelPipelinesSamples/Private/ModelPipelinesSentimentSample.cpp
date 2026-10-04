// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesSentimentSample.h"

#include "TextClassificationPipeline.h"
#include "TextPipelineLibrary.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Notifications/SProgressBar.h"
#include "Widgets/SBoxPanel.h"
#include "SampleOverlayWidgets.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(ModelPipelinesSentimentSample)

namespace ModelPipelinesSentimentSamplePrivate
{
	/** Red (0) -> green (1). */
	static FLinearColor MeterColor(float PositiveScore)
	{
		return FMath::Lerp(FLinearColor(0.85f, 0.1f, 0.1f), FLinearColor(0.1f, 0.8f, 0.2f), FMath::Clamp(PositiveScore, 0.f, 1.f));
	}
}

AModelPipelinesSentimentSample::AModelPipelinesSentimentSample()
{
	// Where the importer puts the model by default (Tools/fetch_reference_models.py --only distilbert-sst2).
	Package = TSoftObjectPtr<UPipelineModelPackage>(FSoftObjectPath(TEXT("/Game/Models/distilbert-sst2.distilbert-sst2")));
}

// ---------------------------------------------------------------------------------------------------- API

void AModelPipelinesSentimentSample::Analyze(const FString& Message)
{
	if (Message.TrimStartAndEnd().IsEmpty())
	{
		SetStatusText(TEXT("Type a message"));
		return;
	}
	// Runs now when idle, otherwise waits in the single pending slot (latest wins).
	RunOrQueue([this, Message]() { StartAnalyzeCall(Message); });
}

float AModelPipelinesSentimentSample::GetLastPositiveScore() const
{
	return LastPositiveScore;
}

FString AModelPipelinesSentimentSample::GetLastLabel() const
{
	return LastLabel;
}

// ---------------------------------------------------------------------------------------------------- pipeline

FPipelineResult AModelPipelinesSentimentSample::CreatePipeline(UPipelineModelPackage* Loaded)
{
	FPipelineResult Result;
	Pipeline = UTextPipelineLibrary::CreateTextClassificationPipeline(Loaded, Result);
	return Result;
}

void AModelPipelinesSentimentSample::StartAnalyzeCall(const FString& Message)
{
	// TopK 2: both labels of an SST-2 model. Scores (every class, after softmax) is what the meter reads.
	Pipeline->ClassifyAsync({ Message }, /*TopK*/ 2,
		[WeakThis = TWeakObjectPtr<AModelPipelinesSentimentSample>(this), Message](const FPipelineResult& Result, const TArray<FPipelineClassification>& Results)
	{
		AModelPipelinesSentimentSample* Self = WeakThis.Get();
		if (!Self || !Self->IsPlaying())
		{
			return;   // the actor is gone or play ended: ignore the result (rule 4)
		}
		if (Result.bSuccess && Results.Num() == 1)
		{
			Self->LastLabel = Results[0].Label;
			Self->LastPositiveScore = Self->ComputePositiveScore(Results[0]);
			Self->History.Insert({ Message, Self->LastLabel, Self->LastPositiveScore }, 0);
			if (Self->History.Num() > MaxHistory)
			{
				Self->History.SetNum(MaxHistory);
			}
		}
		Self->FinishCall(Result);
	});
}

float AModelPipelinesSentimentSample::ComputePositiveScore(const FPipelineClassification& Classification) const
{
	// The pipeline's package holds the label names (index = class id).
	const UPipelineModelPackage* Bound = Pipeline ? Pipeline->GetPackage() : nullptr;
	if (Bound)
	{
		const int32 PositiveIndex = Bound->Labels.IndexOfByPredicate([](const FString& Label)
		{
			return Label.Equals(TEXT("POSITIVE"), ESearchCase::IgnoreCase);
		});
		if (Classification.Scores.IsValidIndex(PositiveIndex))
		{
			return Classification.Scores[PositiveIndex];
		}
	}
	if (Classification.Scores.Num() == 2)
	{
		return Classification.Scores[1];   // 2-label models without names: by convention label 1 is positive
	}
	return Classification.Score;
}

// ---------------------------------------------------------------------------------------------------- overlay

FString AModelPipelinesSentimentSample::GetModelSlug() const
{
	return TEXT("distilbert-sst2");
}

FString AModelPipelinesSentimentSample::GetReadyHint() const
{
	return TEXT("Ready: type a chat message and press Enter");
}

FText AModelPipelinesSentimentSample::GetOverlayTitle() const
{
	return FText::FromString(TEXT("Sentiment meter (distilbert-sst2)"));
}

TSharedPtr<SWidget> AModelPipelinesSentimentSample::GetWidgetToFocus() const
{
	return ChatBox;
}

TSharedRef<SWidget> AModelPipelinesSentimentSample::BuildOverlay()
{
	using namespace UE::ModelPipelines::Samples;
	using namespace ModelPipelinesSentimentSamplePrivate;

	ChatBox = MakeInputBox(FText::FromString(TEXT("Say something in chat...")),
		FOnTextCommitted::CreateWeakLambda(this, [this](const FText& Text, ETextCommit::Type CommitType)
		{
			if (CommitType == ETextCommit::OnEnter)
			{
				Analyze(Text.ToString());
				if (ChatBox.IsValid())
				{
					ChatBox->SetText(FText::GetEmpty());
				}
			}
		}));

	const TWeakObjectPtr<AModelPipelinesSentimentSample> WeakThis(this);
	const TAttribute<TOptional<float>> Percent = TAttribute<TOptional<float>>::CreateLambda([WeakThis]()
	{
		const AModelPipelinesSentimentSample* Self = WeakThis.Get();
		return TOptional<float>(Self && !Self->LastLabel.IsEmpty() ? Self->LastPositiveScore : 0.5f);
	});
	const TAttribute<FSlateColor> Fill = TAttribute<FSlateColor>::CreateLambda([WeakThis]()
	{
		const AModelPipelinesSentimentSample* Self = WeakThis.Get();
		return FSlateColor(MeterColor(Self && !Self->LastLabel.IsEmpty() ? Self->LastPositiveScore : 0.5f));
	});
	const TAttribute<FText> Reading = TAttribute<FText>::CreateLambda([WeakThis]()
	{
		const AModelPipelinesSentimentSample* Self = WeakThis.Get();
		if (!Self || Self->LastLabel.IsEmpty())
		{
			return FText::FromString(TEXT("No message yet"));
		}
		return FText::FromString(FString::Printf(TEXT("%s   (positive %.2f)"), *Self->LastLabel.ToUpper(), Self->LastPositiveScore));
	});

	return MakeOverlayPanel(
		SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight()
		[
			ChatBox.ToSharedRef()
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 14.f, 0.f, 4.f)
		[
			SNew(SBox)
			.HeightOverride(28.f)
			[
				SNew(SProgressBar)
				.Percent(Percent)
				.FillColorAndOpacity(Fill)
			]
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 10.f)
		[
			MakeTextLine(Reading, FLinearColor::White, 20.f, true)
		]
		+ SVerticalBox::Slot().AutoHeight()
		[
			SAssignNew(HistoryList, SVerticalBox)
		]);
}

void AModelPipelinesSentimentSample::RefreshOverlay()
{
	using namespace UE::ModelPipelines::Samples;
	using namespace ModelPipelinesSentimentSamplePrivate;
	if (!HistoryList.IsValid())
	{
		return;
	}
	HistoryList->ClearChildren();
	for (const FHistoryEntry& Entry : History)
	{
		HistoryList->AddSlot().AutoHeight().Padding(0.f, 2.f)
		[
			MakeTextLine(FText::FromString(FString::Printf(TEXT("%.2f   %s"), Entry.PositiveScore, *Entry.Message)), MeterColor(Entry.PositiveScore), 16.f)
		];
	}
}
