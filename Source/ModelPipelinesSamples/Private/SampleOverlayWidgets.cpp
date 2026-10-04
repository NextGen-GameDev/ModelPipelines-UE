// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SampleOverlayWidgets.h"

#include "Styling/CoreStyle.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Text/STextBlock.h"

namespace UE::ModelPipelines::Samples
{
	FSlateFontInfo SampleFont(float Size, bool bBold)
	{
		return FCoreStyle::GetDefaultFontStyle(bBold ? TEXT("Bold") : TEXT("Regular"), Size);
	}

	TSharedRef<SWidget> MakeTextLine(const TAttribute<FText>& Text, const FLinearColor& Color, float Size, bool bBold)
	{
		return SNew(STextBlock)
			.Text(Text)
			.Font(SampleFont(Size, bBold))
			.ColorAndOpacity(FSlateColor(Color))
			.AutoWrapText(true);
	}

	TSharedRef<SWidget> MakePanel(const FText& Title, const TSharedRef<SWidget>& Body, const TAttribute<FText>& Status)
	{
		// The viewport overlay fills the screen; the box pins a fixed-width panel to the top-left corner.
		return SNew(SBox)
			.HAlign(HAlign_Left)
			.VAlign(VAlign_Top)
			.Padding(FMargin(40.f))
			[
				SNew(SBox)
				.WidthOverride(760.f)
				[
					SNew(SBorder)
					.BorderImage(FCoreStyle::Get().GetBrush(TEXT("WhiteBrush")))
					.BorderBackgroundColor(FLinearColor(0.f, 0.f, 0.f, 0.72f))
					.Padding(FMargin(20.f, 16.f))
					[
						SNew(SVerticalBox)
						+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 12.f)
						[
							MakeTextLine(Title, FLinearColor::White, 24.f, true)
						]
						+ SVerticalBox::Slot().AutoHeight()
						[
							Body
						]
						+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 12.f, 0.f, 0.f)
						[
							MakeTextLine(Status, FLinearColor(0.75f, 0.75f, 0.75f), 16.f)
						]
					]
				]
			];
	}

	TSharedRef<SEditableTextBox> MakeInputBox(const FText& Hint, FOnTextCommitted OnCommitted)
	{
		return SNew(SEditableTextBox)
			.HintText(Hint)
			.Font(SampleFont(20.f))
			.ClearKeyboardFocusOnCommit(false)
			.SelectAllTextOnCommit(true)
			.OnTextCommitted(OnCommitted);
	}
}
