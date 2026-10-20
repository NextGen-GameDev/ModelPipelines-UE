// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Widgets/DeclarativeSyntaxSupport.h"
#include "Widgets/SLeafWidget.h"
#include "PipelineResults.h"

/**
 * Full-viewport leaf widget that draws detection boxes: each box (normalized 0-1 in the capture) mapped into the player's view rect
 * (MapBoxToOverlay; the whole widget without a view rect), a 2 px outline, and "label 0.87" just above it. Not hit-testable, so game
 * input is unaffected.
 */
class SSampleDetectionOverlay : public SLeafWidget
{
public:
	SLATE_BEGIN_ARGS(SSampleDetectionOverlay) {}
	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);

	/** Boxes to draw until the next call (they belong to the frame they were computed on). ViewRect / ViewportSize: what that call
	    captured (viewport pixels); an empty rect or zero size draws over the whole widget. */
	void SetDetections(const TArray<FPipelineDetection>& InDetections, const FIntRect& InViewRect, FIntPoint InViewportSize);

	virtual int32 OnPaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry, const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements,
		int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const override;

protected:
	virtual FVector2D ComputeDesiredSize(float LayoutScaleMultiplier) const override;

private:
	TArray<FPipelineDetection> Detections;
	FIntRect ViewRect = FIntRect(0, 0, 0, 0);
	FIntPoint ViewportSize = FIntPoint::ZeroValue;
};
