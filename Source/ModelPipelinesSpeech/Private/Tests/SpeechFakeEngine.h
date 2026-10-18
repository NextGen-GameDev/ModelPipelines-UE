// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

// WP9R.2 test doubles: a fake Phonon backend / model / ISpeechEngine so the routing in USpeechToTextPipeline is tested without
// the paid plugin. Not inside WITH_DEV_AUTOMATION_TESTS for the UCLASS: UHT ignores preprocessor guards (same as Core's PipelineTestFileModel.h).

#include "CoreMinimal.h"
#include "Engine/DataAsset.h"
#include "HAL/CriticalSection.h"
#include "PipelineBackend.h"
#include "PipelineModelSource.h"
#include "PipelineTypes.h"
#include "SpeechEngine.h"
#include "SpeechTypes.h"
#include "SpeechFakeEngine.generated.h"

/** Test-only pointer asset: a Phonon weights folder as an IPipelineFileModel (fake path, size and sha). */
UCLASS(Transient, NotBlueprintable)
class USpeechTestPhononFileModel : public UDataAsset, public IPipelineFileModel
{
	GENERATED_BODY()

public:
	FString FilePath = TEXT("C:/Models/fake-phonon");
	int64 SizeBytes = 1234;
	FString Sha256 = FString::ChrN(64, TEXT('b'));

	virtual EPipelineModelFormat GetFileFormat() const override { return EPipelineModelFormat::Phonon; }
	virtual FString GetAbsoluteFilePath() const override { return FilePath; }
	virtual int64 GetFileSizeBytes() const override { return SizeBytes; }
	virtual FString GetFileSha256() const override { return Sha256; }
};

#if WITH_DEV_AUTOMATION_TESTS

namespace UE::ModelPipelines::SpeechTests
{
	/** One recorded ISpeechEngine::Transcribe call. */
	struct FFakeEngineCall
	{
		uint32 ThreadId = 0;
		int32 SampleRate = 0;
		TArray<float> Samples;
		bool bModelInitializedAtCall = false;
	};

	/** Shared between the backend, the model, the engine and the test (all threads). Everything under Lock. */
	struct FFakeEngineState
	{
		mutable FCriticalSection Lock;

		// Script (set before the call).
		TFunction<FPipelineResult(TConstArrayView<float> Mono, int32 Rate, FSpeechTranscript& Out)> Script;
		bool bNoEngine = false;            // AsSpeechEngine() returns nullptr
		double DelaySeconds = 0.0;         // sleep inside Transcribe (worker thread)

		// Recording.
		TArray<FFakeEngineCall> Calls;
		int32 InitializeCount = 0;
		int32 PrepareCount = 0;
		EPipelineModelFormat LastPreparedFormat = EPipelineModelFormat::None;
		FString LastPreparedPath;
		bool bModelInitialized = false;

		int32 NumCalls() const { FScopeLock S(&Lock); return Calls.Num(); }
		FFakeEngineCall GetCall(int32 Index) const { FScopeLock S(&Lock); return Calls.IsValidIndex(Index) ? Calls[Index] : FFakeEngineCall(); }
	};

	/** The scripted engine. Streams are not used by WP9R.2 (OpenStream fails). */
	class FFakeSpeechEngine : public ISpeechEngine
	{
	public:
		explicit FFakeSpeechEngine(TSharedRef<FFakeEngineState, ESPMode::ThreadSafe> InState) : State(InState) {}
		virtual FPipelineResult Transcribe(TConstArrayView<float> Mono, int32 SampleRate, FSpeechTranscript& Out) override;
		virtual TSharedPtr<ISpeechEngineStream> OpenStream(int32 SampleRate, FPipelineResult& OutResult) override;
		virtual bool SupportsPartials() const override { return true; }

	private:
		TSharedRef<FFakeEngineState, ESPMode::ThreadSafe> State;
	};

	/** Fake Phonon model: IPipelineModel with every pure virtual implemented; AsSpeechEngine() returns the scripted engine. */
	class FFakePhononModel : public IPipelineModel
	{
	public:
		explicit FFakePhononModel(TSharedRef<FFakeEngineState, ESPMode::ThreadSafe> InState);
		virtual FPipelineResult Initialize() override;
		virtual bool IsInitialized() const override;
		virtual EPipelineDevice GetDevice() const override;
		virtual TArray<FPipelineTensorDesc> GetInputDescs() const override { return TArray<FPipelineTensorDesc>(); }
		virtual TArray<FPipelineTensorDesc> GetOutputDescs() const override { return TArray<FPipelineTensorDesc>(); }
		virtual FPipelineResult Run(TConstArrayView<FPipelineTensor> Inputs, TArray<FPipelineTensor>& OutOutputs) override;
		virtual ISpeechEngine* AsSpeechEngine() override;

	private:
		TSharedRef<FFakeEngineState, ESPMode::ThreadSafe> State;
		FFakeSpeechEngine Engine;
	};

	/** Fake backend for EPipelineModelFormat::Phonon (NNE packages are refused: this is registered next to the default backend). */
	class FFakePhononBackend : public IPipelineBackend
	{
	public:
		explicit FFakePhononBackend(TSharedRef<FFakeEngineState, ESPMode::ThreadSafe> InState) : State(InState) {}
		virtual FName GetName() const override { return TEXT("FakePhonon"); }
		virtual bool IsDeviceAvailable(EPipelineDevice Device) const override { return true; }
		virtual TSharedPtr<IPipelineModel> PrepareModel(UNNEModelData* ModelData, EPipelineDevice Device, FPipelineResult& OutResult) override;
		virtual bool SupportsFormat(EPipelineModelFormat Format) const override { return Format == EPipelineModelFormat::Phonon; }
		virtual TSharedPtr<IPipelineModel> PrepareModelFromSource(const FPipelineModelSource& Source, EPipelineDevice Device,
			FPipelineResult& OutResult) override;

	private:
		TSharedRef<FFakeEngineState, ESPMode::ThreadSafe> State;
	};
}

#endif // WITH_DEV_AUTOMATION_TESTS
