// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "SpeechTestData.h"
#include "SpeechFakeEngine.h"

#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformMisc.h"
#include "Misc/CoreDelegates.h"
#include "Misc/FileHelper.h"
#include "NNEModelData.h"
#include "PipelineModelPackage.h"
#include "UObject/Package.h"
#include "Sound/SoundWave.h"
#include "SpeechPipelineLibrary.h"
#include "SpeechToTextPipeline.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace UE::ModelPipelines::SpeechTests
{
	namespace
	{
		const TCHAR* const FetchHint = TEXT("Generate the STT goldens with: \"D:/Micromamba/micromambaenv/envs/pipelines/python.exe\" ModelPipelines-UE/Tools/make_stt_goldens.py");

		FSttGoldens GCache;
		bool GLoaded = false;
		FString GLoadError;
		FCriticalSection GLock;

		bool ParseGoldens(FSttGoldens& Out, FString& OutError)
		{
			Out.Dir = GetDevDir() / TEXT("goldens/stt");
			const FString Path = Out.Dir / TEXT("stt_goldens.json");
			FString Text;
			if (!FFileHelper::LoadFileToString(Text, *Path))
			{
				OutError = FString::Printf(TEXT("Missing STT goldens '%s'. %s"), *Path, FetchHint);
				return false;
			}
			TSharedPtr<FJsonObject> Root;
			const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
			if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
			{
				OutError = FString::Printf(TEXT("STT goldens '%s' are not valid JSON. %s"), *Path, FetchHint);
				return false;
			}

			const TSharedPtr<FJsonObject>* Mel = nullptr;
			const TArray<TSharedPtr<FJsonValue>>* Shape = nullptr;
			if (!Root->TryGetObjectField(TEXT("mel_filters"), Mel) || !(*Mel)->TryGetStringField(TEXT("file"), Out.MelFile)
				|| !(*Mel)->TryGetArrayField(TEXT("shape"), Shape) || Shape->Num() != 2)
			{
				OutError = FString::Printf(TEXT("STT goldens '%s' have no valid \"mel_filters\" block (file, shape). %s"), *Path, FetchHint);
				return false;
			}
			Out.MelRows = (int32)(*Shape)[0]->AsNumber();
			Out.MelCols = (int32)(*Shape)[1]->AsNumber();

			const TArray<TSharedPtr<FJsonValue>>* Clips = nullptr;
			if (!Root->TryGetArrayField(TEXT("clips"), Clips))
			{
				OutError = FString::Printf(TEXT("STT goldens '%s' have no \"clips\" array. %s"), *Path, FetchHint);
				return false;
			}
			for (const TSharedPtr<FJsonValue>& V : *Clips)
			{
				const TSharedPtr<FJsonObject>* O = nullptr;
				if (!V.IsValid() || !V->TryGetObject(O))
				{
					continue;
				}
				FSttClip C;
				(*O)->TryGetStringField(TEXT("id"), C.Id);
				FString Kind;
				(*O)->TryGetStringField(TEXT("kind"), Kind);
				C.bSynthetic = Kind == TEXT("synthetic");

				const TSharedPtr<FJsonObject>* Audio = nullptr;
				if ((*O)->TryGetObjectField(TEXT("audio"), Audio))
				{
					(*Audio)->TryGetStringField(TEXT("file"), C.AudioFile);
					double N = 0;
					(*Audio)->TryGetNumberField(TEXT("num_samples"), N);
					C.NumSamples = (int32)N;
				}
				C.bHasRms = (*O)->TryGetNumberField(TEXT("rms_dbfs"), C.RmsDbfs);

				const TSharedPtr<FJsonObject>* Feat = nullptr;
				if ((*O)->TryGetObjectField(TEXT("features"), Feat))
				{
					C.bHasFeatures = (*Feat)->TryGetStringField(TEXT("file"), C.FeaturesFile);
					double N = 0;
					(*Feat)->TryGetNumberField(TEXT("num_frames"), N);
					C.FeatureFrames = (int32)N;
				}
				const TSharedPtr<FJsonObject>* Log = nullptr;
				if ((*O)->TryGetObjectField(TEXT("logmel"), Log))
				{
					C.bHasLogmel = (*Log)->TryGetStringField(TEXT("file"), C.LogmelFile);
					double N = 0;
					(*Log)->TryGetNumberField(TEXT("num_frames"), N);
					C.LogmelFrames = (int32)N;
				}
				const TSharedPtr<FJsonObject>* Hf = nullptr;
				if ((*O)->TryGetObjectField(TEXT("hf"), Hf))
				{
					(*Hf)->TryGetStringField(TEXT("text"), C.HfText);
				}
				const TSharedPtr<FJsonObject>* Onnx = nullptr;
				if ((*O)->TryGetObjectField(TEXT("onnx"), Onnx))
				{
					C.bHasOnnx = (*Onnx)->TryGetStringField(TEXT("text"), C.OnnxText);
					const TArray<TSharedPtr<FJsonValue>>* Steps = nullptr;
					if ((*Onnx)->TryGetArrayField(TEXT("steps"), Steps))
					{
						for (const TSharedPtr<FJsonValue>& S : *Steps)
						{
							const TArray<TSharedPtr<FJsonValue>>* Pair = nullptr;
							if (S.IsValid() && S->TryGetArray(Pair) && Pair->Num() == 2)
							{
								C.StepTokens.Add((int32)(*Pair)[0]->AsNumber());
								C.StepDurations.Add((int32)(*Pair)[1]->AsNumber());
							}
						}
					}
					const TArray<TSharedPtr<FJsonValue>>* Ids = nullptr;
					if ((*Onnx)->TryGetArrayField(TEXT("token_ids"), Ids))
					{
						for (const TSharedPtr<FJsonValue>& I : *Ids)
						{
							C.TokenIds.Add((int32)I->AsNumber());
						}
					}
					const TArray<TSharedPtr<FJsonValue>>* Words = nullptr;
					if ((*Onnx)->TryGetArrayField(TEXT("word_frames"), Words))
					{
						for (const TSharedPtr<FJsonValue>& W : *Words)
						{
							const TArray<TSharedPtr<FJsonValue>>* Triple = nullptr;
							if (W.IsValid() && W->TryGetArray(Triple) && Triple->Num() == 3)
							{
								FSttGoldWord GW;
								GW.Word = (*Triple)[0]->AsString();
								GW.StartFrame = (int32)(*Triple)[1]->AsNumber();
								GW.EndFrame = (int32)(*Triple)[2]->AsNumber();
								C.Words.Add(MoveTemp(GW));
							}
						}
					}
				}
				Out.Clips.Add(MoveTemp(C));
			}
			if (Out.Clips.Num() == 0)
			{
				OutError = FString::Printf(TEXT("STT goldens '%s' contain no clips. %s"), *Path, FetchHint);
				return false;
			}
			return true;
		}
	}

	FString GetDevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		if (!Env.IsEmpty())
		{
			return Env;
		}
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev"));
	}

	const FSttClip* FSttGoldens::Find(const FString& Id) const
	{
		return Clips.FindByPredicate([&Id](const FSttClip& C) { return C.Id == Id; });
	}

	TArray<const FSttClip*> FSttGoldens::VisibleClips() const
	{
		TArray<const FSttClip*> Out;
		for (const FSttClip& C : Clips)
		{
			if (!C.bSynthetic)
			{
				Out.Add(&C);
			}
		}
		return Out;
	}

	bool LoadSttGoldens(const FSttGoldens*& OutGoldens, FString& OutError)
	{
		FScopeLock Lock(&GLock);
		if (!GLoaded)
		{
			GLoaded = ParseGoldens(GCache, GLoadError);
			if (!GLoaded)
			{
				GCache = FSttGoldens();
			}
		}
		if (!GLoaded)
		{
			OutGoldens = nullptr;
			OutError = GLoadError;
			return false;
		}
		OutGoldens = &GCache;
		return true;
	}

	bool LoadFloatFile(const FSttGoldens& G, const FString& RelFile, int64 ExpectedCount, TArray<float>& OutData, FString& OutError)
	{
		OutData.Reset();
		const FString Path = G.Dir / RelFile;
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *Path))
		{
			OutError = FString::Printf(TEXT("Missing STT golden sidecar '%s'. %s"), *Path, FetchHint);
			return false;
		}
		if ((int64)Bytes.Num() != ExpectedCount * (int64)sizeof(float))
		{
			OutError = FString::Printf(TEXT("STT golden sidecar '%s' is %d bytes, expected %lld (%lld float32). %s"),
				*Path, Bytes.Num(), ExpectedCount * 4, ExpectedCount, FetchHint);
			return false;
		}
		OutData.SetNumUninitialized((int32)ExpectedCount);
		if (ExpectedCount > 0)
		{
			FMemory::Memcpy(OutData.GetData(), Bytes.GetData(), Bytes.Num());
		}
		return true;
	}

	bool LoadClipAudio(const FSttGoldens& G, const FSttClip& Clip, TArray<float>& OutSamples, FString& OutError)
	{
		return LoadFloatFile(G, Clip.AudioFile, Clip.NumSamples, OutSamples, OutError);
	}

	bool IsGpuRequired()
	{
		return FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_GPU")) == TEXT("1");
	}

	FString GetPhononModelDir()
	{
		return GetDevDir() / TEXT("models/stt/phonon-2-onnx/fp16w");
	}

	namespace
	{
		const TCHAR* const ModelFetchHint = TEXT("Fetch the Phonon-2 ONNX folder with: \"D:/Micromamba/micromambaenv/envs/pipelines/python.exe\" ModelPipelines-UE/Tools/export_phonon2_onnx.py (writes .pipelines-dev/models/stt/phonon-2-onnx/fp16w), and the tiny test graphs with ModelPipelines-UE/Tools/make_test_models.py");

		struct FSharedPackages
		{
			TStrongObjectPtr<UPipelineModelPackage> Phonon;
			TStrongObjectPtr<UPipelineModelPackage> Tiny;
		};

		FSharedPackages* GShared = nullptr; // heap, never deleted: the strong refs are released at OnEnginePreExit
		FDelegateHandle GPreExitHandle;

		FSharedPackages& Shared()
		{
			if (!GShared)
			{
				GShared = new FSharedPackages();
				GPreExitHandle = FCoreDelegates::OnEnginePreExit.AddLambda([]()
				{
					if (GShared)
					{
						GShared->Phonon.Reset();
						GShared->Tiny.Reset();
					}
				});
			}
			return *GShared;
		}

		void FillCommon(UPipelineModelPackage& Pkg, const FString& TokenizerText)
		{
			Pkg.Task = EPipelineTask::SpeechToText;
			Pkg.TokenizerJson = TokenizerText;
			Pkg.License.SpdxId = TEXT("cc-by-4.0");
			Pkg.License.SourceUrl = TEXT("https://huggingface.co/FermionResearch/Phonon-2");
			Pkg.License.Attribution = TEXT("Phonon-2 by Fermion Research, based on NVIDIA parakeet-tdt-0.6b-v3");
			// Pkg.Speech keeps the defaults (16000 Hz, 128 mel bins, ...).
		}
	}

	UNNEModelData* LoadTestModelData(const FString& FileName, FString& OutError)
	{
		const FString Path = GetDevDir() / TEXT("models/test") / FileName;
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *Path) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("Missing test model '%s'. %s"), *Path, ModelFetchHint);
			return nullptr;
		}
		UNNEModelData* Data = NewObject<UNNEModelData>(GetTransientPackage());
		Data->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
		return Data;
	}

	bool GetSharedPhononPackage(EPipelineDevice Device, UPipelineModelPackage*& OutPackage, FString& OutError)
	{
		OutPackage = nullptr;
		FSharedPackages& S = Shared();
		if (!S.Phonon.IsValid())
		{
			const FString Dir = GetPhononModelDir();
			const FString EncPath = Dir / TEXT("encoder.onnx");
			const FString DecPath = Dir / TEXT("decoder_joint.onnx");
			const FString TokPath = Dir / TEXT("tokenizer.json");
			FString TokText;
			if (!FFileHelper::LoadFileToString(TokText, *TokPath) || TokText.IsEmpty())
			{
				OutError = FString::Printf(TEXT("Missing tokenizer '%s'. %s"), *TokPath, ModelFetchHint);
				return false;
			}
			UNNEModelData* Dec = nullptr;
			{
				TArray<uint8> Bytes;
				if (!FFileHelper::LoadFileToArray(Bytes, *DecPath) || Bytes.Num() == 0)
				{
					OutError = FString::Printf(TEXT("Missing decoder '%s'. %s"), *DecPath, ModelFetchHint);
					return false;
				}
				Dec = NewObject<UNNEModelData>(GetTransientPackage());
				Dec->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
			}
			UNNEModelData* Enc = nullptr;
			{
				TArray<uint8> Bytes; // freed at the end of this scope, before the package is used
				if (!FFileHelper::LoadFileToArray(Bytes, *EncPath) || Bytes.Num() == 0)
				{
					OutError = FString::Printf(TEXT("Missing encoder '%s'. %s"), *EncPath, ModelFetchHint);
					return false;
				}
				Enc = NewObject<UNNEModelData>(GetTransientPackage());
				Enc->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
			}
			UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
			Enc->Rename(nullptr, Pkg);
			Dec->Rename(nullptr, Pkg);
			Pkg->Model = Enc;
			Pkg->DecoderModel = Dec;
			FillCommon(*Pkg, TokText);
			S.Phonon.Reset(Pkg);
		}
		S.Phonon->PreferredDevice = Device;
		OutPackage = S.Phonon.Get();
		return true;
	}

	bool GetSharedTinySttPackage(UPipelineModelPackage*& OutPackage, FString& OutError)
	{
		OutPackage = nullptr;
		FSharedPackages& S = Shared();
		if (!S.Tiny.IsValid())
		{
			const FString TokPath = GetPhononModelDir() / TEXT("tokenizer.json");
			FString TokText;
			if (!FFileHelper::LoadFileToString(TokText, *TokPath) || TokText.IsEmpty())
			{
				OutError = FString::Printf(TEXT("Missing tokenizer '%s'. %s"), *TokPath, ModelFetchHint);
				return false;
			}
			UNNEModelData* Enc = LoadTestModelData(TEXT("tiny_stt_encoder.onnx"), OutError);
			UNNEModelData* Dec = Enc ? LoadTestModelData(TEXT("tiny_stt_decoder_joint.onnx"), OutError) : nullptr;
			if (!Enc || !Dec)
			{
				return false;
			}
			UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
			Enc->Rename(nullptr, Pkg);
			Dec->Rename(nullptr, Pkg);
			Pkg->Model = Enc;
			Pkg->DecoderModel = Dec;
			FillCommon(*Pkg, TokText);
			Pkg->PreferredDevice = EPipelineDevice::CPU;
			S.Tiny.Reset(Pkg);
		}
		OutPackage = S.Tiny.Get();
		return true;
	}

	// ---- WP9S.5b helpers ----

	namespace
	{
		struct FSharedSttPipelines
		{
			TStrongObjectPtr<USpeechToTextPipeline> Cpu;
			TStrongObjectPtr<USpeechToTextPipeline> Tiny;
		};

		FSharedSttPipelines* GPipes = nullptr;

		FSharedSttPipelines& Pipes()
		{
			if (!GPipes)
			{
				GPipes = new FSharedSttPipelines();
				FCoreDelegates::OnEnginePreExit.AddLambda([]()
				{
					if (GPipes)
					{
						GPipes->Cpu.Reset();
						GPipes->Tiny.Reset();
					}
				});
			}
			return *GPipes;
		}
	}

	USpeechToTextPipeline* GetSharedCpuSttPipeline(FString& OutError)
	{
		FSharedSttPipelines& S = Pipes();
		if (!S.Cpu.IsValid())
		{
			UPipelineModelPackage* Pkg = nullptr;
			if (!GetSharedPhononPackage(EPipelineDevice::CPU, Pkg, OutError)) { return nullptr; }
			FPipelineResult R;
			USpeechToTextPipeline* P = USpeechPipelineLibrary::CreateSpeechToTextPipeline(Pkg, R);
			if (!P)
			{
				OutError = FString::Printf(TEXT("CreateSpeechToTextPipeline failed on the Phonon package: %s"), *R.Message);
				return nullptr;
			}
			S.Cpu.Reset(P);
		}
		return S.Cpu.Get();
	}

	USpeechToTextPipeline* GetSharedTinySttPipeline(FString& OutError)
	{
		FSharedSttPipelines& S = Pipes();
		if (!S.Tiny.IsValid())
		{
			UPipelineModelPackage* Pkg = nullptr;
			if (!GetSharedTinySttPackage(Pkg, OutError)) { return nullptr; }
			FPipelineResult R;
			USpeechToTextPipeline* P = USpeechPipelineLibrary::CreateSpeechToTextPipeline(Pkg, R);
			if (!P)
			{
				OutError = FString::Printf(TEXT("CreateSpeechToTextPipeline failed on the tiny package: %s"), *R.Message);
				return nullptr;
			}
			S.Tiny.Reset(P);
		}
		return S.Tiny.Get();
	}

	void FloatToInt16(TConstArrayView<float> In, TArray<int16>& Out)
	{
		Out.SetNumUninitialized(In.Num());
		for (int32 i = 0; i < In.Num(); ++i)
		{
			Out[i] = (int16)FMath::Clamp(FMath::RoundToInt(In[i] * 32768.0f), -32768, 32767);
		}
	}

	USoundWave* MakeGeneratedSoundWave(const TArray<int16>& Interleaved, int32 SampleRate, int32 NumChannels)
	{
		USoundWave* W = NewObject<USoundWave>(GetTransientPackage());
		const int32 N = Interleaved.Num();
		const int32 Frames = NumChannels > 0 ? N / NumChannels : 0;
		W->SetSampleRate((uint32)SampleRate);
		W->NumChannels = NumChannels;
		W->Duration = (float)Frames / (float)SampleRate;
		W->TotalSamples = (float)N;
		W->RawPCMDataSize = N * (int32)sizeof(int16);
		W->RawPCMData = (uint8*)FMemory::Malloc(W->RawPCMDataSize);
		FMemory::Memcpy(W->RawPCMData, Interleaved.GetData(), W->RawPCMDataSize);
		return W;
	}

	// ---- WP9S.5c2 ----
	namespace
	{
		const TCHAR* const LongFetchHint = TEXT("Generate the long-audio STT goldens with: \"D:/Micromamba/micromambaenv/envs/pipelines-stt/python.exe\" ModelPipelines-UE/Tools/make_stt_long_goldens.py");

		FSttLongGoldens GLongCache;
		bool GLongLoaded = false;
		FString GLongError;
		FCriticalSection GLongLock;

		bool ParseLongWords(const TSharedPtr<FJsonObject>& O, TArray<FSttLongWord>& Out)
		{
			const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
			if (!O->TryGetArrayField(TEXT("words"), Arr)) { return false; }
			for (const TSharedPtr<FJsonValue>& V : *Arr)
			{
				const TArray<TSharedPtr<FJsonValue>>* Triple = nullptr;
				if (!V.IsValid() || !V->TryGetArray(Triple) || Triple->Num() != 3) { return false; }
				FSttLongWord W;
				W.Word = (*Triple)[0]->AsString();
				W.Start = (*Triple)[1]->AsNumber();
				W.End = (*Triple)[2]->AsNumber();
				Out.Add(MoveTemp(W));
			}
			return true;
		}

		bool ParseLongGoldens(FSttLongGoldens& Out, FString& OutError)
		{
			Out.Dir = GetDevDir() / TEXT("goldens/stt/long");
			const FString Path = Out.Dir / TEXT("long_goldens.json");
			FString Text;
			if (!FFileHelper::LoadFileToString(Text, *Path))
			{
				OutError = FString::Printf(TEXT("Missing long-audio STT goldens '%s'. %s"), *Path, LongFetchHint);
				return false;
			}
			TSharedPtr<FJsonObject> Root;
			const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
			const TArray<TSharedPtr<FJsonValue>>* Recs = nullptr;
			if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid() || !Root->TryGetArrayField(TEXT("recordings"), Recs))
			{
				OutError = FString::Printf(TEXT("Long-audio STT goldens '%s' are not valid JSON or have no \"recordings\" array. %s"), *Path, LongFetchHint);
				return false;
			}
			for (const TSharedPtr<FJsonValue>& V : *Recs)
			{
				const TSharedPtr<FJsonObject>* RO = nullptr;
				if (!V.IsValid() || !V->TryGetObject(RO)) { continue; }
				FSttLongRecording R;
				const TSharedPtr<FJsonObject>* Audio = nullptr;
				const TArray<TSharedPtr<FJsonValue>>* Wins = nullptr;
				double N = 0.0;
				if (!(*RO)->TryGetStringField(TEXT("id"), R.Id) || !(*RO)->TryGetObjectField(TEXT("audio"), Audio)
					|| !(*Audio)->TryGetStringField(TEXT("file"), R.AudioFile) || !(*Audio)->TryGetNumberField(TEXT("num_samples"), N)
					|| !(*RO)->TryGetStringField(TEXT("text"), R.Text) || !(*RO)->TryGetArrayField(TEXT("windows"), Wins)
					|| !ParseLongWords(*RO, R.Words))
				{
					OutError = FString::Printf(TEXT("Long-audio STT goldens '%s': a recording lacks id/audio/text/windows/words. %s"), *Path, LongFetchHint);
					return false;
				}
				R.NumSamples = (int32)N;
				(*RO)->TryGetStringField(TEXT("kind"), R.Kind);
				(*RO)->TryGetStringField(TEXT("ref"), R.Ref);
				for (const TSharedPtr<FJsonValue>& WV : *Wins)
				{
					const TSharedPtr<FJsonObject>* WO = nullptr;
					const TArray<TSharedPtr<FJsonValue>>* Ids = nullptr;
					double S = 0.0, E = 0.0;
					FSttLongWindow W;
					if (!WV.IsValid() || !WV->TryGetObject(WO) || !(*WO)->TryGetNumberField(TEXT("start"), S) || !(*WO)->TryGetNumberField(TEXT("end"), E)
						|| !(*WO)->TryGetBoolField(TEXT("gated"), W.bGated) || !(*WO)->TryGetNumberField(TEXT("rms_dbfs"), W.RmsDbfs)
						|| !(*WO)->TryGetStringField(TEXT("text"), W.Text) || !(*WO)->TryGetArrayField(TEXT("token_ids"), Ids) || !ParseLongWords(*WO, W.Words))
					{
						OutError = FString::Printf(TEXT("Long-audio STT goldens '%s': a window of '%s' is malformed. %s"), *Path, *R.Id, LongFetchHint);
						return false;
					}
					W.Start = (int32)S;
					W.End = (int32)E;
					for (const TSharedPtr<FJsonValue>& T : *Ids) { W.TokenIds.Add((int32)T->AsNumber()); }
					R.Windows.Add(MoveTemp(W));
				}
				Out.Recordings.Add(MoveTemp(R));
			}
			if (Out.Recordings.Num() == 0)
			{
				OutError = FString::Printf(TEXT("Long-audio STT goldens '%s' hold no recordings. %s"), *Path, LongFetchHint);
				return false;
			}
			return true;
		}
	}

	const FSttLongRecording* FSttLongGoldens::Find(const FString& Id) const
	{
		for (const FSttLongRecording& R : Recordings)
		{
			if (R.Id.Equals(Id, ESearchCase::CaseSensitive)) { return &R; }
		}
		return nullptr;
	}

	void ReleaseSharedCpuSttPipeline()
	{
		Pipes().Cpu.Reset();
		CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
	}

	bool LoadSttLongGoldens(const FSttLongGoldens*& OutGoldens, FString& OutError)
	{
		FScopeLock Lock(&GLongLock);
		if (!GLongLoaded)
		{
			GLongLoaded = ParseLongGoldens(GLongCache, GLongError);
			if (!GLongLoaded) { GLongCache = FSttLongGoldens(); }
		}
		if (!GLongLoaded)
		{
			OutGoldens = nullptr;
			OutError = GLongError;
			return false;
		}
		OutGoldens = &GLongCache;
		return true;
	}

	bool LoadLongAudio(const FSttLongGoldens& G, const FSttLongRecording& Rec, TArray<float>& OutSamples, FString& OutError)
	{
		OutSamples.Reset();
		const FString Path = G.Dir / Rec.AudioFile;
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *Path))
		{
			OutError = FString::Printf(TEXT("Missing long-audio sidecar '%s'. %s"), *Path, LongFetchHint);
			return false;
		}
		if ((int64)Bytes.Num() != (int64)Rec.NumSamples * 4)
		{
			OutError = FString::Printf(TEXT("Long-audio sidecar '%s' is %d bytes, expected %lld. %s"), *Path, Bytes.Num(), (int64)Rec.NumSamples * 4, LongFetchHint);
			return false;
		}
		OutSamples.SetNumUninitialized(Rec.NumSamples);
		if (Rec.NumSamples > 0) { FMemory::Memcpy(OutSamples.GetData(), Bytes.GetData(), Bytes.Num()); }
		return true;
	}

	UPipelineModelPackage* MakeFakePhononPackage()
	{
		UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
		Pkg->Task = EPipelineTask::SpeechToText;
		Pkg->PreferredDevice = EPipelineDevice::CPU;
		Pkg->GgufModel = NewObject<USpeechTestPhononFileModel>(GetTransientPackage());
		return Pkg;
	}

	void MakeToneSamples(TArray<float>& Out, int32 NumFrames, int32 Rate, double Freq, double Amp)
	{
		Out.SetNumUninitialized(FMath::Max(NumFrames, 0));
		for (int32 i = 0; i < Out.Num(); ++i)
		{
			Out[i] = (float)(Amp * FMath::Sin(2.0 * UE_DOUBLE_PI * Freq * (double)i / (double)Rate));
		}
	}
}

#endif // WITH_DEV_AUTOMATION_TESTS
