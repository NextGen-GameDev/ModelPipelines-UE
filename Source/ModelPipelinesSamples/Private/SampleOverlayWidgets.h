// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Engine/EngineTypes.h"
#include "Fonts/SlateFontInfo.h"
#include "Misc/Attribute.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/SWidget.h"

class AActor;
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

	/** Dark translucent panel at the top-left of the viewport: Title, Body, then the status line. */
	TSharedRef<SWidget> MakePanel(const FText& Title, const TSharedRef<SWidget>& Body, const TAttribute<FText>& Status);

	/** Single-line text box with a hint; Enter calls OnCommitted (focus stays in the box). */
	TSharedRef<SEditableTextBox> MakeInputBox(const FText& Hint, FOnTextCommitted OnCommitted);

	/** On-demand capture: no capture every frame or on movement, final tonemapped LDR color (what the player sees). */
	void ConfigureSampleCapture(USceneCaptureComponent2D* Capture);

	/** New RGBA8 render target of the given size (resource created now; the capture renders into it on demand). */
	UTextureRenderTarget2D* CreateSampleCaptureTarget(UObject* Outer, int32 Width, int32 Height);

	/** Horizontal FOV (degrees) of a view with camera FOV CameraFovDeg, the given axis constraint and view size.
	    MaintainXFOV, or MajorAxisFOV with X >= Y: CameraFovDeg. Otherwise HFOV = 2*atan(tan(CameraFovDeg/2) * X/Y).
	    ViewSize with a zero component -> CameraFovDeg. Result clamped to [1, 170]. */
	float CaptureHorizontalFov(float CameraFovDeg, EAspectRatioAxisConstraint Constraint, FIntPoint ViewSize);

	/**
	 * Copies the first player's camera location/rotation to Actor and the player's HORIZONTAL view FOV to Capture (a scene capture's
	 * FOVAngle is horizontal). The FOV comes from the local player's projection (ULocalPlayer::GetProjectionData) when it succeeds,
	 * else from CaptureHorizontalFov with the local player's AspectRatioAxisConstraint and the game viewport size. Without a local
	 * player/viewport the camera FOV is used as is. false (nothing changed) without a player camera.
	 */
	bool CopyPlayerCamera(AActor* Actor, USceneCaptureComponent2D* Capture);
}
