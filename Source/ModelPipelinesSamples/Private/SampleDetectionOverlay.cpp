// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SampleDetectionOverlay.h"

#include "Layout/Geometry.h"
#include "Rendering/DrawElements.h"
#include "SampleOverlayWidgets.h"

void SSampleDetectionOverlay::Construct(const FArguments& InArgs)
{
	SetVisibility(EVisibility::HitTestInvisible);
}

void SSampleDetectionOverlay::SetDetections(const TArray<FPipelineDetection>& InDetections)
{
	Detections = InDetections;
	Invalidate(EInvalidateWidgetReason::Paint);
}

FVector2D SSampleDetectionOverlay::ComputeDesiredSize(float) const
{
	// Fills whatever slot it is placed in (the whole viewport); it asks for no space of its own.
	return FVector2D::ZeroVector;
}

int32 SSampleDetectionOverlay::OnPaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry, const FSlateRect& MyCullingRect,
	FSlateWindowElementList& OutDrawElements, int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const
{
	const FVector2f Size = FVector2f(AllottedGeometry.GetLocalSize());
	if (Size.X <= 0.f || Size.Y <= 0.f)
	{
		return LayerId;
	}

	const FLinearColor BoxColor(0.1f, 1.f, 0.3f, 1.f);
	const FSlateFontInfo Font = UE::ModelPipelines::Samples::SampleFont(16.f, true);
	const FPaintGeometry Whole = AllottedGeometry.ToPaintGeometry();
	for (const FPipelineDetection& D : Detections)
	{
		// Normalized box (Min = left/top, Max = right/bottom) times the widget size; the capture has the view's aspect and FOV.
		const float X0 = FMath::Clamp((float)D.Box.Min.X, 0.f, 1.f) * Size.X;
		const float Y0 = FMath::Clamp((float)D.Box.Min.Y, 0.f, 1.f) * Size.Y;
		const float X1 = FMath::Clamp((float)D.Box.Max.X, 0.f, 1.f) * Size.X;
		const float Y1 = FMath::Clamp((float)D.Box.Max.Y, 0.f, 1.f) * Size.Y;

		TArray<FVector2f> Points;
		Points.Reserve(5);
		Points.Add(FVector2f(X0, Y0));
		Points.Add(FVector2f(X1, Y0));
		Points.Add(FVector2f(X1, Y1));
		Points.Add(FVector2f(X0, Y1));
		Points.Add(FVector2f(X0, Y0));
		FSlateDrawElement::MakeLines(OutDrawElements, LayerId, Whole, MoveTemp(Points), ESlateDrawEffect::None, BoxColor, true, 2.f);

		// "label 0.87" above the box (inside it when the box touches the top edge).
		const FString Caption = FString::Printf(TEXT("%s %.2f"), *D.Label, D.Score);
		const float TextY = (Y0 >= 24.f) ? (Y0 - 24.f) : (Y0 + 2.f);
		FSlateDrawElement::MakeText(OutDrawElements, LayerId + 1,
			AllottedGeometry.ToPaintGeometry(FVector2f(FMath::Max(Size.X - X0, 1.f), 24.f), FSlateLayoutTransform(FVector2f(X0 + 2.f, TextY))),
			Caption, Font, ESlateDrawEffect::None, BoxColor);
	}
	return LayerId + 1;
}
