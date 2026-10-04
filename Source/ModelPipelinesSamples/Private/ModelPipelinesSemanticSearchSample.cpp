// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesSemanticSearchSample.h"

#include "PipelineLibrary.h"
#include "TextEmbeddingPipeline.h"
#include "TextPipelineLibrary.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/SBoxPanel.h"
#include "SampleOverlayWidgets.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(ModelPipelinesSemanticSearchSample)

AModelPipelinesSemanticSearchSample::AModelPipelinesSemanticSearchSample()
{
	// Where the importer puts the model by default (Tools/fetch_reference_models.py --only all-MiniLM-L6-v2).
	Package = TSoftObjectPtr<UPipelineModelPackage>(FSoftObjectPath(TEXT("/Game/Models/all-MiniLM-L6-v2.all-MiniLM-L6-v2")));

	// A fantasy village: one line per NPC topic, varied enough that a question finds the right speaker.
	DialogueLines = {
		TEXT("Blacksmith: Bring me your dented sword or cracked shield and I'll have it repaired by sundown."),
		TEXT("Innkeeper: We have warm rooms for the night, five silver with breakfast included."),
		TEXT("Guard: The north road is dangerous after dark; wolves and bandits have been seen near the pass."),
		TEXT("Herbalist: Mix silverleaf with honey and it will cure most fevers and stomach aches."),
		TEXT("Fisherman: The trout bite best at dawn by the old stone bridge."),
		TEXT("Baker: Fresh bread and sweet rolls come out of the oven every morning."),
		TEXT("Priest: The temple bell rings at noon; all are welcome to pray and rest in the chapel."),
		TEXT("Merchant: I buy gems, furs and old relics, and I always pay a fair price in gold."),
		TEXT("Stable master: You can rent a fast horse here if you need to travel to the capital."),
		TEXT("Farmer: Something has been stealing my sheep at night, I found giant claw marks in the mud."),
		TEXT("Mayor: The harvest festival starts tomorrow, with music, dancing and a pie contest."),
		TEXT("Librarian: The ancient maps of the ruins are kept in the archive on the second floor."),
		TEXT("Hunter: Deer tracks lead east into the forest, but stay away from the bear caves."),
		TEXT("Tavern bard: Buy me an ale and I'll sing you the ballad of the dragon slayer."),
		TEXT("Alchemist: My potions restore health and mana, but the fire resistance tonic is sold out."),
		TEXT("Captain of the watch: We are hiring brave recruits to defend the walls from the goblin raids."),
		TEXT("Old woman: My cat ran off into the cellar of the abandoned mill, could you bring her back?"),
		TEXT("Ferryman: One copper and I'll row you across the river to the eastern marshes."),
		TEXT("Tailor: I can sew you a warm cloak or patch up that torn tunic."),
		TEXT("Miner: The silver mine collapsed last week and three workers are still trapped inside."),
		TEXT("Child: Have you seen the ghost in the old lighthouse? Everyone says it glows at midnight."),
		TEXT("Wizard: Magic scrolls and spellbooks can be studied in my tower, if you can solve the riddle at the door."),
		TEXT("Banker: Your gold is safe in our vault, and we lend money at a modest interest."),
		TEXT("Cook: Tonight's stew has carrots, potatoes and rabbit, and it costs only two coins a bowl.")
	};
}

// ---------------------------------------------------------------------------------------------------- API

void AModelPipelinesSemanticSearchSample::Search(const FString& Query)
{
	const FString Trimmed = Query.TrimStartAndEnd();
	if (Trimmed.IsEmpty())
	{
		// Only a hint while usable: never replace the missing-package / failure text.
		SetInputHint(TEXT("Type a question"));
		return;
	}
	// Runs now when idle, otherwise waits in the single pending slot (e.g. while the corpus is still being embedded).
	RunOrQueue([this, Query]() { StartQueryCall(Query); });
}

TArray<FPipelineSimilarityMatch> AModelPipelinesSemanticSearchSample::GetLastMatches() const
{
	return LastMatches;
}

bool AModelPipelinesSemanticSearchSample::IsCorpusReady() const
{
	return bCorpusReady;
}

// ---------------------------------------------------------------------------------------------------- pipeline

FPipelineResult AModelPipelinesSemanticSearchSample::CreatePipeline(UPipelineModelPackage* Loaded)
{
	FPipelineResult Result;
	Pipeline = UTextPipelineLibrary::CreateTextEmbeddingPipeline(Loaded, Result);
	return Result;
}

void AModelPipelinesSemanticSearchSample::OnPipelineReady()
{
	// Embed the corpus once. The sample is Busy until it finishes, so an early Search waits for it.
	RunOrQueue([this]() { StartCorpusCall(); });
}

void AModelPipelinesSemanticSearchSample::StartCorpusCall(const FString& FollowUpQuery)
{
	CorpusLines = DialogueLines;
	if (CorpusLines.IsEmpty())
	{
		FinishCallNextTick(FPipelineResult::Fail(EPipelineError::InvalidArgument, TEXT("DialogueLines is empty: add the lines to search")));
		return;
	}

	// The callback runs on the game thread; the weak pointer skips it if the actor is gone (rule 4).
	Pipeline->EmbedAsync(CorpusLines, [WeakThis = TWeakObjectPtr<AModelPipelinesSemanticSearchSample>(this), FollowUpQuery](const FPipelineResult& Result, const TArray<FPipelineEmbedding>& Embeddings)
	{
		AModelPipelinesSemanticSearchSample* Self = WeakThis.Get();
		if (!Self || !Self->IsPlaying())
		{
			return;
		}
		if (Result.bSuccess)
		{
			Self->CorpusEmbeddings = Embeddings;
			Self->bCorpusReady = true;
		}
		Self->FinishCall(Result);
		// A retried corpus call carries the query that triggered it: run it now, unless a newer request already started from
		// the pending slot (latest wins) or the corpus failed again (the next Search retries once more).
		if (!FollowUpQuery.IsEmpty() && Self->bCorpusReady && Self->IsPlaying() && Self->GetSampleState() == EPipelineSampleState::Ready)
		{
			Self->RunOrQueue([Self, FollowUpQuery]() { Self->StartQueryCall(FollowUpQuery); });
		}
	});
}

void AModelPipelinesSemanticSearchSample::StartQueryCall(const FString& Query)
{
	if (!bCorpusReady)
	{
		// The one-time corpus call failed (or the lines were empty): retry it now, then run this query (one OnSampleResult each).
		StartCorpusCall(Query);
		return;
	}

	LastQuery = Query;
	Pipeline->EmbedAsync({ Query }, [WeakThis = TWeakObjectPtr<AModelPipelinesSemanticSearchSample>(this)](const FPipelineResult& Result, const TArray<FPipelineEmbedding>& Embeddings)
	{
		AModelPipelinesSemanticSearchSample* Self = WeakThis.Get();
		if (!Self || !Self->IsPlaying())
		{
			return;
		}
		if (Result.bSuccess && Embeddings.Num() == 1)
		{
			// Cosine similarity of the query against every line, best K first.
			Self->LastMatches = UPipelineLibrary::TopKSimilar(Embeddings[0], Self->CorpusEmbeddings, Self->TopK);
		}
		Self->FinishCall(Result);
	});
}

// ---------------------------------------------------------------------------------------------------- overlay

FString AModelPipelinesSemanticSearchSample::GetModelSlug() const
{
	return TEXT("all-MiniLM-L6-v2");
}

FString AModelPipelinesSemanticSearchSample::GetReadyHint() const
{
	return bCorpusReady ? TEXT("Ready: type a question and press Enter") : TEXT("Ready");
}

FText AModelPipelinesSemanticSearchSample::GetOverlayTitle() const
{
	return FText::FromString(TEXT("Semantic search (all-MiniLM-L6-v2)"));
}

TSharedPtr<SWidget> AModelPipelinesSemanticSearchSample::GetWidgetToFocus() const
{
	return QueryBox;
}

TSharedRef<SWidget> AModelPipelinesSemanticSearchSample::BuildOverlay()
{
	using namespace UE::ModelPipelines::Samples;

	QueryBox = MakeInputBox(FText::FromString(TEXT("Ask the villagers...")),
		FOnTextCommitted::CreateWeakLambda(this, [this](const FText& Text, ETextCommit::Type CommitType)
		{
			if (CommitType == ETextCommit::OnEnter)
			{
				Search(Text.ToString());
			}
		}));

	return MakeOverlayPanel(
		SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight()
		[
			QueryBox.ToSharedRef()
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 12.f, 0.f, 0.f)
		[
			SAssignNew(MatchList, SVerticalBox)
		]);
}

void AModelPipelinesSemanticSearchSample::RefreshOverlay()
{
	using namespace UE::ModelPipelines::Samples;
	if (!MatchList.IsValid())
	{
		return;
	}
	MatchList->ClearChildren();
	if (!LastQuery.IsEmpty())
	{
		MatchList->AddSlot().AutoHeight().Padding(0.f, 0.f, 0.f, 6.f)
		[
			MakeTextLine(FText::FromString(FString::Printf(TEXT("\"%s\""), *LastQuery)), FLinearColor(0.6f, 0.8f, 1.f))
		];
	}
	for (const FPipelineSimilarityMatch& Match : LastMatches)
	{
		const FString Line = CorpusLines.IsValidIndex(Match.Index) ? CorpusLines[Match.Index] : FString();
		MatchList->AddSlot().AutoHeight().Padding(0.f, 2.f)
		[
			MakeTextLine(FText::FromString(FString::Printf(TEXT("%.2f   %s"), Match.Score, *Line)))
		];
	}
}
