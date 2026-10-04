// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SampleOverlayWidgets.h"

#include "Camera/PlayerCameraManager.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/GameViewportClient.h"
#include "Engine/LocalPlayer.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "GameFramework/PlayerController.h"
#include "SceneView.h"
#include "Slate/SceneViewport.h"
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

	void ConfigureSampleCapture(USceneCaptureComponent2D* Capture)
	{
		if (!Capture)
		{
			return;
		}
		// Captured only by CaptureScene() right before a pipeline call; never twice per frame for nothing.
		Capture->bCaptureEveryFrame = false;
		Capture->bCaptureOnMovement = false;
		Capture->CaptureSource = SCS_FinalColorLDR;
	}

	UTextureRenderTarget2D* CreateSampleCaptureTarget(UObject* Outer, int32 Width, int32 Height)
	{
		UTextureRenderTarget2D* Target = NewObject<UTextureRenderTarget2D>(Outer);
		Target->RenderTargetFormat = RTF_RGBA8;
		Target->ClearColor = FLinearColor::Black;
		Target->bAutoGenerateMips = false;
		// Uses RenderTargetFormat (RTF_RGBA8) and recreates the resource (UpdateResource, no flush).
		Target->InitAutoFormat((uint32)FMath::Max(1, Width), (uint32)FMath::Max(1, Height));
		return Target;
	}

	float CaptureHorizontalFov(float CameraFovDeg, EAspectRatioAxisConstraint Constraint, FIntPoint ViewSize)
	{
		constexpr float MinFov = 1.f;
		constexpr float MaxFov = 170.f;
		// The camera FOV is horizontal when X is the constrained axis (MaintainXFOV, or MajorAxisFOV on a landscape/square view),
		// and when the view size is unknown. Otherwise it is vertical: widen it by the view aspect X/Y.
		const bool bHorizontal = Constraint == AspectRatio_MaintainXFOV
			|| (Constraint == AspectRatio_MajorAxisFOV && ViewSize.X >= ViewSize.Y)
			|| ViewSize.X <= 0 || ViewSize.Y <= 0;
		if (bHorizontal || !FMath::IsFinite(CameraFovDeg))
		{
			return FMath::Clamp(FMath::IsFinite(CameraFovDeg) ? CameraFovDeg : 90.f, MinFov, MaxFov);
		}
		// A vertical FOV of 180 or more is no projection; keep tan() finite and positive before widening.
		const double HalfV = FMath::DegreesToRadians(FMath::Clamp((double)CameraFovDeg, 0.001, 179.9) * 0.5);
		const double HalfH = FMath::Atan(FMath::Tan(HalfV) * (double)ViewSize.X / (double)ViewSize.Y);
		return FMath::Clamp((float)FMath::RadiansToDegrees(2.0 * HalfH), MinFov, MaxFov);
	}

	namespace
	{
		/** Horizontal FOV of the player's actual view, read from its projection matrix (covers the cvar r.UseLegacyMaintainYFOVViewMatrix,
		    the camera's own AspectRatio, bConstrainAspectRatio and custom projections). false for no projection or an orthographic one. */
		bool PlayerProjectionHorizontalFov(ULocalPlayer* LocalPlayer, FViewport* Viewport, float& OutFovDeg)
		{
			FSceneViewProjectionData Data;
			// LocalPlayer.h:623 (5.8) / :613 (5.7); false without a viewport, a player controller or a zero size.
			if (!LocalPlayer || !Viewport || !LocalPlayer->GetProjectionData(Viewport, Data))
			{
				return false;
			}
			const FMatrix& P = Data.ProjectionMatrix;
			// Perspective (reversed Z): M[2][3] = 1, M[3][3] = 0, M[0][0] = 1 / tan(HFOV / 2).
			if (!(P.M[0][0] > 0.0) || !FMath::IsNearlyZero(P.M[3][3]) || !FMath::IsFinite(P.M[0][0]))
			{
				return false;
			}
			OutFovDeg = FMath::Clamp((float)FMath::RadiansToDegrees(2.0 * FMath::Atan(1.0 / P.M[0][0])), 1.f, 170.f);
			return true;
		}
	}

	bool CopyPlayerCamera(AActor* Actor, USceneCaptureComponent2D* Capture)
	{
		UWorld* World = Actor ? Actor->GetWorld() : nullptr;
		APlayerController* PC = World ? World->GetFirstPlayerController() : nullptr;
		APlayerCameraManager* Camera = PC ? PC->PlayerCameraManager.Get() : nullptr;
		if (!Camera)
		{
			return false;
		}
		Actor->SetActorLocationAndRotation(Camera->GetCameraLocation(), Camera->GetCameraRotation());
		if (Capture)
		{
			// A scene capture's FOVAngle is horizontal (SceneCaptureRendering.cpp, XAxisMultiplier = 1), but the camera FOV is applied
			// per the local player's AspectRatioAxisConstraint (default MaintainYFOV). Convert to the view's horizontal FOV so the
			// capture (same aspect as the viewport) frames exactly what the player sees. All read on the game thread.
			ULocalPlayer* LocalPlayer = PC->GetLocalPlayer();
			UGameViewportClient* ViewportClient = World->GetGameViewport();
			FSceneViewport* Viewport = ViewportClient ? ViewportClient->GetGameViewport() : nullptr;
			float Fov = 0.f;
			if (!PlayerProjectionHorizontalFov(LocalPlayer, Viewport, Fov))
			{
				const EAspectRatioAxisConstraint Constraint = LocalPlayer ? LocalPlayer->AspectRatioAxisConstraint.GetValue() : AspectRatio_MaintainYFOV;
				const FIntPoint ViewSize = Viewport ? Viewport->GetSizeXY() : FIntPoint::ZeroValue;
				Fov = CaptureHorizontalFov(Camera->GetFOVAngle(), Constraint, ViewSize);
			}
			Capture->FOVAngle = Fov;
		}
		return true;
	}
}
