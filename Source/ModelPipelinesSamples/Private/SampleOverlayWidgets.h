// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Engine/EngineTypes.h"
#include "Fonts/SlateFontInfo.h"
#include "Misc/Attribute.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/SWidget.h"

class AActor;
class AModelPipelinesSampleBase;
class UObject;
class USceneCaptureComponent2D;
class UTextureRenderTarget2D;

/** Small Slate helpers shared by the sample overlays (plain Slate, readable at 1080p), plus the vision samples' capture setup. */
namespace UE::ModelPipelines::Samples
{
	/** Default overlay font; Size >= 16 keeps it readable at 1080p. */
	FSlateFontInfo SampleFont(float Size = 18.f, bool bBold = false);

	/** Text line (optionally bound to a getter). */
	TSharedRef<SWidget> MakeTextLine(const TAttribute<FText>& Text, const FLinearColor& Color = FLinearColor::White, float Size = 18.f, bool bBold = false);

	/** Panel width (Slate units) of the text samples: no photo in the view, long result sentences read best on few lines. */
	inline constexpr float TextPanelWidth = 760.f;

	/** Panel width of the vision samples: a left column that stays clear of the photo board at the map's player start (the board
	    starts at about 30 % of the view width; 40 + 460 units end near 28 % at 1920x1080). Holds the 256-unit classifier preview. */
	inline constexpr float VisionPanelWidth = 460.f;

	/** Dark translucent panel at the top-left of the viewport (40 units from the edges, Width units wide, height follows the
	    content): Title, Body, then the status line (wraps). */
	TSharedRef<SWidget> MakePanel(const FText& Title, const TSharedRef<SWidget>& Body, const TAttribute<FText>& Status, float Width = TextPanelWidth);

	/** Status line bound to Sample->GetStatusText() (read every frame; empty once the sample is gone). */
	TAttribute<FText> MakeStatusAttribute(const AModelPipelinesSampleBase* Sample);

	/** Single-line text box with a hint; Enter calls OnCommitted (focus stays in the box). */
	TSharedRef<SEditableTextBox> MakeInputBox(const FText& Hint, FOnTextCommitted OnCommitted);

	/** On-demand capture: no capture every frame or on movement, final tonemapped LDR color (what the player sees). */
	void ConfigureSampleCapture(USceneCaptureComponent2D* Capture);

	/** New RGBA8 render target of the given size (resource created now; the capture renders into it on demand). */
	UTextureRenderTarget2D* CreateSampleCaptureTarget(UObject* Outer, int32 Width, int32 Height);

	/** Horizontal FOV (degrees) of a view with camera FOV CameraFovDeg, the given axis constraint and view size: the LEGACY MaintainYFOV
	    formula (cvar r.UseLegacyMaintainYFOVViewMatrix on, where the camera FOV is vertical under a Y constraint).
	    MaintainXFOV, or MajorAxisFOV with X >= Y: CameraFovDeg. Otherwise HFOV = 2*atan(tan(CameraFovDeg/2) * X/Y).
	    ViewSize with a zero component -> CameraFovDeg. Result clamped to [1, 170]. Used only by FallbackCaptureFov. */
	float CaptureHorizontalFov(float CameraFovDeg, EAspectRatioAxisConstraint Constraint, FIntPoint ViewSize);

	/** Capture FOV when the player's projection is unavailable. bLegacyMaintainYFov (cvar r.UseLegacyMaintainYFOVViewMatrix):
	    CaptureHorizontalFov(CameraFovDeg, Constraint, ViewSize). Otherwise CameraFovDeg (the view-info FOV is horizontal in 5.7/5.8,
	    CameraStackTypes.cpp:326). Non-finite CameraFovDeg -> 90. Result clamped to [1, 170]. */
	float FallbackCaptureFov(float CameraFovDeg, EAspectRatioAxisConstraint Constraint, FIntPoint ViewSize, bool bLegacyMaintainYFov);

	/** Render-target size for a capture of the given view rect: X = clamp(CaptureWidth, 16, 4096), Y = clamp(round(X / aspect), 1, 4096),
	    aspect = ViewRect.Width() / ViewRect.Height(); an empty rect (width or height <= 0) uses 16:9. */
	FIntPoint CaptureSizeForView(int32 CaptureWidth, const FIntRect& ViewRect);

	/** true when ViewRect is non-empty and its aspect differs from CurrentSize's aspect by more than 1 % (relative to the view's aspect).
	    CurrentSize with a zero component -> true for a non-empty rect. Empty ViewRect -> false (keep what we have). */
	bool NeedsNewCaptureTarget(FIntPoint CurrentSize, const FIntRect& ViewRect);

	/** Maps a normalized detection box (0..1 in the capture) into the overlay widget's local space. The capture shows ViewRect
	    (viewport pixels) of a viewport of ViewportSize pixels; the widget covers the whole viewport with WidgetSize local units.
	    x = (ViewRect.Min.X + nx * ViewRect.Width()) / ViewportSize.X * WidgetSize.X (same for y). Box coordinates are clamped to 0..1
	    first. An empty ViewRect or a ViewportSize with a zero component maps onto the whole widget (nx * WidgetSize.X). */
	FBox2D MapBoxToOverlay(const FBox2D& NormalizedBox, const FIntRect& ViewRect, FIntPoint ViewportSize, FVector2D WidgetSize);

	/**
	 * Copies the first player's camera location/rotation to Actor and the player's HORIZONTAL view FOV to Capture (a scene capture's
	 * FOVAngle is horizontal). The FOV is read from the local player's projection matrix (ULocalPlayer::GetProjectionData) when it
	 * succeeds; else it is FallbackCaptureFov(camera FOV, the local player's AspectRatioAxisConstraint, the game viewport size,
	 * cvar r.UseLegacyMaintainYFOVViewMatrix), i.e. the camera FOV as is unless the legacy cvar is on. false (nothing changed)
	 * without a player camera.
	 */
	bool CopyPlayerCamera(AActor* Actor, USceneCaptureComponent2D* Capture);
}
