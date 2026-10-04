// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Fonts/SlateFontInfo.h"
#include "Misc/Attribute.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/SWidget.h"

/** Small Slate helpers shared by the sample overlays (plain Slate, readable at 1080p). */
namespace UE::ModelPipelines::Samples
{
	/** Default overlay font; Size >= 16 keeps it readable at 1080p. */
	FSlateFontInfo SampleFont(float Size = 18.f, bool bBold = false);

	/** Text line (optionally bound to a getter). */
	TSharedRef<SWidget> MakeTextLine(const TAttribute<FText>& Text, const FLinearColor& Color = FLinearColor::White, float Size = 18.f, bool bBold = false);

	/** Dark translucent panel at the top-left of the viewport: Title, Body, then the status line. */
	TSharedRef<SWidget> MakePanel(const FText& Title, const TSharedRef<SWidget>& Body, const TAttribute<FText>& Status);

	/** Single-line text box with a hint; Enter calls OnCommitted (focus stays in the box). */
	TSharedRef<SEditableTextBox> MakeInputBox(const FText& Hint, FOnTextCommitted OnCommitted);
}
