// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "ModelPipelinesTalkToTextSample.h"

#include "HAL/PlatformTime.h"
#include "Styling/CoreStyle.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SComboBox.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Text/STextBlock.h"
#include "SpeechMicrophoneSource.h"
#include "SpeechPipelineLibrary.h"
#include "SpeechToTextPipeline.h"
#include "SampleOverlayWidgets.h"
#include "SamplesLog.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(ModelPipelinesTalkToTextSample)

AModelPipelinesTalkToTextSample::AModelPipelinesTalkToTextSample()
{
	// Where the importer puts the model by default (Tools/fetch_phonon2.py, folder phonon-2-fp16w).
	Package = TSoftObjectPtr<UPipelineModelPackage>(FSoftObjectPath(TEXT("/Game/Models/phonon-2-fp16w.phonon-2-fp16w")));
}

// ---------------------------------------------------------------------------------------------------- API

void AModelPipelinesTalkToTextSample::StartListening()
{
	if (!IsPlaying() || GetSampleState() != EPipelineSampleState::Ready || !Pipeline)
	{
		UE_LOG(LogModelPipelinesSamples, Verbose, TEXT("%s: StartListening ignored, the sample is not ready (%s)"), *GetName(), *GetStatusText());
		return;
	}
	if (bListening)
	{
		UE_LOG(LogModelPipelinesSamples, Verbose, TEXT("%s: StartListening ignored, already listening"), *GetName());
		return;
	}
	if (bWarmingUp)
	{
		UE_LOG(LogModelPipelinesSamples, Verbose, TEXT("%s: StartListening deferred until the warm-up completes"), *GetName());
		bStartAfterWarmUp = true;
		return;
	}

	// A stream still finishing after StopListening: its last events are of no interest any more.
	ReleaseStream();
	MicrophoneFailure.Reset();
	MicrophoneName.Reset();
	PartialText.Reset();
	SpeechEndedTimes.Reset();

	FPipelineResult Result;
	USpeechStream* NewStream = Pipeline->StartStream(StreamSettings, Result);
	if (!NewStream || !Result.bSuccess)
	{
		if (Result.bSuccess)
		{
			Result.bSuccess = false;
			Result.Message = TEXT("StartStream returned no stream");
		}
		FinishCall(Result);
		return;
	}

	Stream = NewStream;
	Stream->OnSpeechStarted.AddDynamic(this, &AModelPipelinesTalkToTextSample::HandleSpeechStarted);
	Stream->OnPartial.AddDynamic(this, &AModelPipelinesTalkToTextSample::HandlePartial);
	Stream->OnFinal.AddDynamic(this, &AModelPipelinesTalkToTextSample::HandleFinal);
	Stream->OnSpeechEnded.AddDynamic(this, &AModelPipelinesTalkToTextSample::HandleSpeechEnded);
	Stream->OnStopped.AddDynamic(this, &AModelPipelinesTalkToTextSample::HandleStopped);
	bListening = true;

	if (bUseMicrophone)
	{
		StartMicrophone();
	}

	SetInputHint(GetListeningStatus());
	RefreshOverlay();
}

void AModelPipelinesTalkToTextSample::StopListening()
{
	bStartAfterWarmUp = false;
	if (!bListening)
	{
		return;
	}
	bListening = false;
	StopMicrophone();
	if (Stream)
	{
		// OnStopped arrives later and drops the reference; finals still in flight are delivered until then
		// (SpeechEndedTimes stays for their latency).
		Stream->Stop();
	}
	PartialText.Reset();
	MicrophoneFailure.Reset();
	SetInputHint(GetReadyHint());
	RefreshOverlay();
}

bool AModelPipelinesTalkToTextSample::IsListening() const
{
	return bListening;
}

bool AModelPipelinesTalkToTextSample::IsMicrophoneActive() const
{
	return bListening && Microphone && Microphone->IsCapturing();
}

bool AModelPipelinesTalkToTextSample::HasPartialText() const
{
	return bPartialCapable;
}

TArray<FString> AModelPipelinesTalkToTextSample::GetFinalLines() const
{
	// MaxLines is BlueprintReadWrite: honour a lowered value at once, not only at the next final (review note).
	const int32 Cap = FMath::Max(1, MaxLines);
	if (Lines.Num() <= Cap)
	{
		return Lines;
	}
	return TArray<FString>(Lines.GetData() + (Lines.Num() - Cap), Cap);
}

FString AModelPipelinesTalkToTextSample::GetPartialText() const
{
	return PartialText;
}

float AModelPipelinesTalkToTextSample::GetLastFinalLatencySeconds() const
{
	return LastLatency;
}

USpeechStream* AModelPipelinesTalkToTextSample::GetStream() const
{
	return Stream;
}

TArray<FSpeechCaptureDevice> AModelPipelinesTalkToTextSample::GetMicrophones() const
{
	return USpeechMicrophoneSource::GetCaptureDevices();
}

void AModelPipelinesTalkToTextSample::SetMicrophone(int32 InDeviceIndex)
{
	DeviceIndex = InDeviceIndex;
	bUseMicrophone = true;
	if (!IsPlaying() || !bListening || !Stream)
	{
		// Applied by the next StartListening.
		return;
	}

	// Only the capture restarts: the stream, its delegates, the lines and the partial text stay.
	// Stop waits for the capture thread's last callback (~10 ms; up to 1 s on an unplugged device, engine behaviour).
	StopMicrophone();
	Microphone = NewObject<USpeechMicrophoneSource>(this);
	StartMicrophone();

	const bool bQuiet = !MicrophoneFailure.IsEmpty();
	SetInputHint(!bQuiet && Stream->IsSpeaking() ? FString(TEXT("Hearing speech..."))
		: !bQuiet && SpeechEndedTimes.Num() > 0 ? FString(TEXT("Transcribing...")) : GetListeningStatus());
	RefreshOverlay();
}

FString AModelPipelinesTalkToTextSample::GetMicrophoneName() const
{
	return IsMicrophoneActive() ? MicrophoneName : FString();
}

// ---------------------------------------------------------------------------------------------------- play / pipeline

void AModelPipelinesTalkToTextSample::EndPlay(const EEndPlayReason::Type Reason)
{
	bListening = false;
	bWarmingUp = false;
	bStartAfterWarmUp = false;
	StopMicrophone();
	if (Stream)
	{
		// Unbind first: Stop may finalise the utterance, and nothing of this actor may fire after EndPlay.
		USpeechStream* Old = Stream;
		ReleaseStream();
		Old->Stop();
	}
	ToggleButton.Reset();
	LinesList.Reset();
	MicCombo.Reset();
	MicOptions.Reset();
	Super::EndPlay(Reason);
}

FPipelineResult AModelPipelinesTalkToTextSample::CreatePipeline(UPipelineModelPackage* Loaded)
{
	FPipelineResult Result;
	Pipeline = USpeechPipelineLibrary::CreateSpeechToTextPipeline(Loaded, Result);
	bPartialCapable = Loaded && Loaded->GetModelFormat() == EPipelineModelFormat::Phonon;
	if (Result.bSuccess && !Pipeline)
	{
		Result.bSuccess = false;
		Result.Message = TEXT("the creator returned no pipeline");
	}
	return Result;
}

void AModelPipelinesTalkToTextSample::OnPipelineReady()
{
	bStartAfterWarmUp = bAutoStart;
	StartWarmUp();
}

void AModelPipelinesTalkToTextSample::StartWarmUp()
{
	if (!Pipeline)
	{
		return;
	}
	// One second of a quiet 300 Hz tone at 16 kHz mono. The stream reuses this pipeline's model, so the cold start
	// (session creation, first-call allocations, Phonon's lazy engine init) is paid here and not by the first final.
	constexpr int32 SampleRate = 16000;
	TArray<float> Samples;
	Samples.SetNumUninitialized(SampleRate);
	for (int32 i = 0; i < SampleRate; ++i)
	{
		Samples[i] = 0.1f * FMath::Sin(2.f * PI * 300.f * static_cast<float>(i) / static_cast<float>(SampleRate));
	}
	bWarmingUp = true;
	SetInputHint(GetReadyHint());
	RefreshOverlay();
	UE_LOG(LogModelPipelinesSamples, Verbose, TEXT("%s: warming up the model"), *GetName());
	Pipeline->Transcribe(MoveTemp(Samples), SampleRate, 1,
		[WeakThis = TWeakObjectPtr<AModelPipelinesTalkToTextSample>(this)](const FPipelineResult& Result, const FSpeechTranscript&)
		{
			if (AModelPipelinesTalkToTextSample* Self = WeakThis.Get())
			{
				Self->HandleWarmUpDone(Result);
			}
		});
}

void AModelPipelinesTalkToTextSample::HandleWarmUpDone(const FPipelineResult& Result)
{
	if (!bWarmingUp || !IsPlaying())
	{
		return;
	}
	bWarmingUp = false;
	if (!Result.bSuccess)
	{
		UE_LOG(LogModelPipelinesSamples, Verbose, TEXT("%s: warm-up failed (ignored): %s"), *GetName(), *Result.Message);
	}
	else
	{
		UE_LOG(LogModelPipelinesSamples, Verbose, TEXT("%s: warm-up done (inference %.0f ms)"), *GetName(), Result.InferenceMs);
	}
	SetInputHint(GetReadyHint());
	RefreshOverlay();
	if (bStartAfterWarmUp)
	{
		bStartAfterWarmUp = false;
		StartListening();
	}
}

// ---------------------------------------------------------------------------------------------------- stream events

void AModelPipelinesTalkToTextSample::HandleSpeechStarted()
{
	if (!IsPlaying() || !bListening)
	{
		return;
	}
	UE_LOG(LogModelPipelinesSamples, Verbose, TEXT("%s: speech started at %.3f"), *GetName(), FPlatformTime::Seconds());
	if (MicrophoneFailure.IsEmpty())
	{
		SetInputHint(TEXT("Hearing speech..."));
	}
}

void AModelPipelinesTalkToTextSample::HandlePartial(const FSpeechTranscript& Transcript)
{
	if (!IsPlaying() || !bListening)
	{
		return;
	}
	PartialText = Transcript.Text;
	RefreshOverlay();
}

void AModelPipelinesTalkToTextSample::HandleSpeechEnded()
{
	if (!IsPlaying() || !bListening)
	{
		return;
	}
	SpeechEndedTimes.Add(FPlatformTime::Seconds());
	UE_LOG(LogModelPipelinesSamples, Verbose, TEXT("%s: speech ended at %.3f (%d finals pending)"), *GetName(), SpeechEndedTimes.Last(), SpeechEndedTimes.Num());
	if (MicrophoneFailure.IsEmpty())
	{
		SetInputHint(TEXT("Transcribing..."));
	}
}

void AModelPipelinesTalkToTextSample::HandleFinal(const FSpeechTranscript& Transcript)
{
	// Finals keep coming after StopListening until OnStopped (the handlers are unbound then).
	if (!IsPlaying() || !Stream)
	{
		return;
	}
	const double Now = FPlatformTime::Seconds();
	const FString Text = Transcript.Text.TrimStartAndEnd();
	UE_LOG(LogModelPipelinesSamples, Verbose, TEXT("%s: final at %.3f (%d pending, audio %.2f s, inference %.0f ms): '%s'"), *GetName(), Now, SpeechEndedTimes.Num(),
		Transcript.AudioSeconds, Transcript.InferenceMs, *Text);
	PartialText.Reset();
	// The SpeechEnded that belongs to this final is the oldest one not consumed yet.
	double EndedTime = -1.0;
	if (SpeechEndedTimes.Num() > 0)
	{
		EndedTime = SpeechEndedTimes[0];
		SpeechEndedTimes.RemoveAt(0);
	}
	if (Text.IsEmpty())
	{
		SetInputHint(GetReadyHint());
		RefreshOverlay();
		return;
	}

	LastLatency = EndedTime >= 0.0 ? static_cast<float>(Now - EndedTime + FMath::Clamp(StreamSettings.EndOfSpeechHangoverSeconds, 0.1f, 2.f)) /* StartStream clamps the same way */ : -1.f;
	if (Transcript.bReplacesPrevious && Lines.Num() > 0)
	{
		// WP9R.7 (C83): a corrected final of the same turn replaces the last line and its latency.
		Lines.Last() = Text;
		LineLatencies.SetNum(Lines.Num());
		LineLatencies.Last() = LastLatency;
	}
	else
	{
		Lines.Add(Text);
		LineLatencies.Add(LastLatency);
	}
	const int32 Cap = FMath::Max(1, MaxLines);
	if (Lines.Num() > Cap)
	{
		const int32 Drop = Lines.Num() - Cap;
		Lines.RemoveAt(0, Drop);
		LineLatencies.RemoveAt(0, Drop);
	}

	TWeakObjectPtr<AModelPipelinesTalkToTextSample> WeakThis(this);
	OnFinalText.Broadcast(Transcript);
	if (!WeakThis.IsValid() || !IsPlaying())
	{
		return;   // a listener destroyed the actor or ended play
	}

	FPipelineResult Ok;
	Ok.bSuccess = true;
	Ok.InferenceMs = Transcript.InferenceMs;
	FinishCall(Ok);
	if (!WeakThis.IsValid() || !IsPlaying())
	{
		return;
	}
	if (bListening)
	{
		// Back to "Listening (...)", unless the next utterance is already being heard or transcribed.
		const bool bQuiet = !MicrophoneFailure.IsEmpty();
		SetInputHint(!bQuiet && Stream && Stream->IsSpeaking() ? FString(TEXT("Hearing speech..."))
			: !bQuiet && SpeechEndedTimes.Num() > 0 ? FString(TEXT("Transcribing...")) : GetListeningStatus());
	}
	RefreshOverlay();
}

void AModelPipelinesTalkToTextSample::HandleStopped(const FPipelineResult& Result)
{
	if (!IsPlaying())
	{
		return;
	}
	const bool bWasListening = bListening;
	bListening = false;
	StopMicrophone();
	ReleaseStream();
	PartialText.Reset();
	MicrophoneFailure.Reset();
	if (!Result.bSuccess)
	{
		FinishCall(Result);
	}
	else if (bWasListening)
	{
		SetInputHint(GetReadyHint());
	}
	RefreshOverlay();
}

// ---------------------------------------------------------------------------------------------------- helpers

FString AModelPipelinesTalkToTextSample::GetListeningStatus() const
{
	if (!MicrophoneFailure.IsEmpty())
	{
		return MicrophoneFailure;
	}
	if (IsMicrophoneActive())
	{
		return FString::Printf(TEXT("Listening (%s)"), *MicrophoneName);
	}
	return TEXT("Listening (pushed audio)");
}

void AModelPipelinesTalkToTextSample::ReleaseStream()
{
	if (Stream)
	{
		Stream->OnSpeechStarted.RemoveAll(this);
		Stream->OnPartial.RemoveAll(this);
		Stream->OnFinal.RemoveAll(this);
		Stream->OnSpeechEnded.RemoveAll(this);
		Stream->OnStopped.RemoveAll(this);
		Stream = nullptr;
	}
}

void AModelPipelinesTalkToTextSample::StopMicrophone()
{
	if (Microphone && Microphone->IsCapturing())
	{
		Microphone->Stop();
	}
}

void AModelPipelinesTalkToTextSample::StartMicrophone()
{
	MicrophoneName.Reset();
	MicrophoneFailure.Reset();
	if (!Stream)
	{
		return;
	}
	if (!Microphone)
	{
		Microphone = NewObject<USpeechMicrophoneSource>(this);
	}
	FPipelineResult MicResult;
	if (Microphone->Start(Stream, DeviceIndex, MicResult))
	{
		const TArray<FSpeechCaptureDevice> Devices = USpeechMicrophoneSource::GetCaptureDevices();
		if (DeviceIndex < 0)
		{
			const FSpeechCaptureDevice* Default = Devices.FindByPredicate([](const FSpeechCaptureDevice& D) { return D.bIsDefault; });
			MicrophoneName = Default ? Default->Name : FString(TEXT("default microphone"));
		}
		else
		{
			const FSpeechCaptureDevice* Found = Devices.FindByPredicate([this](const FSpeechCaptureDevice& D) { return D.DeviceIndex == DeviceIndex; });
			MicrophoneName = Found ? Found->Name
				: Devices.IsValidIndex(DeviceIndex) ? Devices[DeviceIndex].Name : FString::Printf(TEXT("device %d"), DeviceIndex);
		}
	}
	else
	{
		// The stream stays: audio pushed from Blueprint is still transcribed.
		UE_LOG(LogModelPipelinesSamples, Warning, TEXT("%s: microphone not started: %s"), *GetName(), *MicResult.Message);
		MicrophoneFailure = FString::Printf(TEXT("No microphone (%s). Audio pushed with Get Stream > Push Speech Audio is still transcribed."), *MicResult.Message);
	}
}

void AModelPipelinesTalkToTextSample::RefreshMicOptions()
{
	const TArray<FSpeechCaptureDevice> Devices = USpeechMicrophoneSource::GetCaptureDevices();
	MicOptions.Reset();
	if (Devices.Num() == 0)
	{
		TSharedPtr<FSpeechCaptureDevice> None = MakeShared<FSpeechCaptureDevice>();
		None->DeviceIndex = -2;
		None->Name = TEXT("No microphone found");
		MicOptions.Add(None);
		return;
	}
	const FSpeechCaptureDevice* Default = Devices.FindByPredicate([](const FSpeechCaptureDevice& D) { return D.bIsDefault; });
	TSharedPtr<FSpeechCaptureDevice> DefaultEntry = MakeShared<FSpeechCaptureDevice>();
	DefaultEntry->DeviceIndex = -1;
	DefaultEntry->Name = Default ? FString::Printf(TEXT("Default device (%s)"), *Default->Name) : FString(TEXT("Default device"));
	MicOptions.Add(DefaultEntry);
	for (const FSpeechCaptureDevice& D : Devices)
	{
		MicOptions.Add(MakeShared<FSpeechCaptureDevice>(D));
	}
}

FString AModelPipelinesTalkToTextSample::GetMicComboText() const
{
	if (MicOptions.Num() == 0 || (MicOptions.Num() == 1 && MicOptions[0].IsValid() && MicOptions[0]->DeviceIndex == -2))
	{
		return TEXT("No microphone found");
	}
	const int32 Wanted = DeviceIndex;   // only -1 is the default entry; other values fall through to "Device %d (not found)"
	for (const TSharedPtr<FSpeechCaptureDevice>& Option : MicOptions)
	{
		if (Option.IsValid() && Option->DeviceIndex == Wanted)
		{
			return Option->Name;
		}
	}
	return FString::Printf(TEXT("Device %d (not found)"), DeviceIndex);
}

void AModelPipelinesTalkToTextSample::ToggleListening()
{
	if (bListening)
	{
		StopListening();
	}
	else
	{
		StartListening();
	}
}

// ---------------------------------------------------------------------------------------------------- overlay

FString AModelPipelinesTalkToTextSample::GetModelSlug() const
{
	return TEXT("phonon-2-fp16w");
}

FString AModelPipelinesTalkToTextSample::GetReadyHint() const
{
	if (bListening)
	{
		return GetListeningStatus();
	}
	return bWarmingUp ? FString(TEXT("Warming up the model...")) : FString(TEXT("Ready: press Start listening"));
}

FText AModelPipelinesTalkToTextSample::GetOverlayTitle() const
{
	return FText::FromString(TEXT("Talk to Text"));
}

FString AModelPipelinesTalkToTextSample::GetMissingPackageText(const FString& PackagePath) const
{
	return FString::Printf(TEXT("Model package not found: %s. Run Plugins/ModelPipelines/Tools/fetch_phonon2.py --out <folder>, import the phonon-2-fp16w folder into /Game/Models, then press Play again. With ModelPipelines Generative you can point Package at an imported Phonon-2 weights package instead. See Docs/Samples.md."),
		*PackagePath);
}

TSharedPtr<SWidget> AModelPipelinesTalkToTextSample::GetWidgetToFocus() const
{
	return ToggleButton;
}

TSharedRef<SWidget> AModelPipelinesTalkToTextSample::BuildOverlay()
{
	using namespace UE::ModelPipelines::Samples;

	const TWeakObjectPtr<AModelPipelinesTalkToTextSample> WeakThis(this);
	const TAttribute<FText> EngineLine = TAttribute<FText>::CreateLambda([WeakThis]()
	{
		const AModelPipelinesTalkToTextSample* Self = WeakThis.Get();
		if (!Self)
		{
			return FText::GetEmpty();
		}
		return FText::FromString(Self->HasPartialText()
			? FString(TEXT("Engine: Phonon (partial text while you speak)"))
			: FString(TEXT("Engine: ONNX (final text only; partial text needs ModelPipelines Generative)")));
	});
	const TAttribute<FText> ButtonText = TAttribute<FText>::CreateLambda([WeakThis]()
	{
		const AModelPipelinesTalkToTextSample* Self = WeakThis.Get();
		return FText::FromString(Self && Self->IsListening() ? TEXT("Stop listening") : TEXT("Start listening"));
	});

	SAssignNew(ToggleButton, SButton)
		.OnClicked_Lambda([WeakThis]()
		{
			if (AModelPipelinesTalkToTextSample* Self = WeakThis.Get())
			{
				Self->ToggleListening();
			}
			return FReply::Handled();
		})
		[
			SNew(STextBlock)
			.Text(ButtonText)
			.Font(SampleFont(18.f, true))
		];

	RefreshMicOptions();
	using FMicItem = TSharedPtr<FSpeechCaptureDevice>;
	SAssignNew(MicCombo, SComboBox<FMicItem>)
		.OptionsSource(&MicOptions)
		.IsFocusable(false)   // the base focuses the button; the combo must not take it
		.OnComboBoxOpening_Lambda([WeakThis]()
		{
			// Devices plugged in during play appear.
			if (AModelPipelinesTalkToTextSample* Self = WeakThis.Get())
			{
				Self->RefreshMicOptions();
				if (Self->MicCombo.IsValid())
				{
					Self->MicCombo->RefreshOptions();
				}
			}
		})
		.OnGenerateWidget_Lambda([](FMicItem Item) -> TSharedRef<SWidget>
		{
			const bool bSelectable = Item.IsValid() && Item->DeviceIndex >= -1;
			return SNew(STextBlock)
				.Text(FText::FromString(Item.IsValid() ? Item->Name : FString()))
				.Font(SampleFont(16.f, false))
				.IsEnabled(bSelectable);
		})
		.OnSelectionChanged_Lambda([WeakThis](FMicItem Item, ESelectInfo::Type SelectInfo)
		{
			if (SelectInfo == ESelectInfo::Direct || !Item.IsValid() || Item->DeviceIndex < -1)
			{
				return;
			}
			if (AModelPipelinesTalkToTextSample* Self = WeakThis.Get())
			{
				// Re-picking the device in use is a no-op (options are rebuilt on open, so the combo always reports a change); after a
				// failure the same entry still retries.
				if (Item->DeviceIndex == Self->DeviceIndex && Self->IsMicrophoneActive())
				{
					return;
				}
				Self->SetMicrophone(Item->DeviceIndex);
			}
		})
		[
			SNew(STextBlock)
			.Text_Lambda([WeakThis]()
			{
				const AModelPipelinesTalkToTextSample* Self = WeakThis.Get();
				return Self ? FText::FromString(Self->GetMicComboText()) : FText::GetEmpty();
			})
			.Font(SampleFont(16.f, false))
		];

	TSharedRef<SWidget> Panel = MakeOverlayPanel(
		SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 8.f)
		[
			MakeTextLine(EngineLine, FLinearColor(0.75f, 0.75f, 0.75f), 16.f)
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 8.f)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0.f, 0.f, 8.f, 0.f)
			[
				MakeTextLine(FText::FromString(TEXT("Microphone:")), FLinearColor(0.75f, 0.75f, 0.75f), 16.f)
			]
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
			[
				MicCombo.ToSharedRef()
			]
		]
		+ SVerticalBox::Slot().AutoHeight()
		[
			SAssignNew(LinesList, SVerticalBox)
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 10.f, 0.f, 0.f).HAlign(HAlign_Left)
		[
			ToggleButton.ToSharedRef()
		]);
	RefreshOverlay();
	return Panel;
}

void AModelPipelinesTalkToTextSample::RefreshOverlay()
{
	using namespace UE::ModelPipelines::Samples;
	if (!LinesList.IsValid())
	{
		return;
	}
	LinesList->ClearChildren();
	for (int32 i = 0; i < Lines.Num(); ++i)
	{
		const float Latency = LineLatencies.IsValidIndex(i) ? LineLatencies[i] : -1.f;
		const FString Prefix = Latency < 0.f ? FString(TEXT("[-]")) : FString::Printf(TEXT("[%.2f s]"), Latency);
		LinesList->AddSlot().AutoHeight().Padding(0.f, 2.f)
		[
			MakeTextLine(FText::FromString(FString::Printf(TEXT("%s %s"), *Prefix, *Lines[i])), FLinearColor::White, 18.f)
		];
	}
	if (!PartialText.IsEmpty())
	{
		LinesList->AddSlot().AutoHeight().Padding(0.f, 2.f)
		[
			SNew(STextBlock)
			.Text(FText::FromString(PartialText))
			.Font(FCoreStyle::GetDefaultFontStyle(TEXT("Italic"), 18))
			.ColorAndOpacity(FSlateColor(FLinearColor(0.6f, 0.6f, 0.6f)))
			.AutoWrapText(true)
		];
	}
}
