// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "SpeechTestData.h"

#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformMisc.h"
#include "Misc/FileHelper.h"
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
}

#endif // WITH_DEV_AUTOMATION_TESTS
