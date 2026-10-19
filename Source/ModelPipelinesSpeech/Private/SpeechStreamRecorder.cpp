// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "SpeechStreamRecorder.h"

#if !UE_BUILD_SHIPPING

#include "SpeechStream.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/FileManager.h"
#include "HAL/IConsoleManager.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Tasks/Task.h"
#include "PipelineSpeechLog.h"
#include "SpeechTypes.h"

#include <atomic>

namespace UE::ModelPipelines::Speech
{
	namespace RecorderPrivate
	{
		static TAutoConsoleVariable<int32> CVarRecordStreams(
			TEXT("mp.Speech.RecordStreams"), 0,
			TEXT("1: every USpeechStream started while this is 1 saves its 16 kHz audio (.wav) and its events (.json) to Saved/ModelPipelines/SpeechRecordings. Development builds only."),
			ECVF_Default);

		static TAutoConsoleVariable<float> CVarRecordMaxSeconds(
			TEXT("mp.Speech.RecordMaxSeconds"), 1200.f,
			TEXT("Longest audio kept per recorded speech stream, in seconds (1..3600). Audio beyond is dropped and the JSON says truncated."),
			ECVF_Default);

		static std::atomic<int32> RecordingCounter{ 0 };

		struct FWriteJob
		{
			FString Directory;
			FString BaseName;
			TArray<float> Audio;
			FString Json;
		};

		static void WriteLe16(TArray<uint8>& B, uint16 V) { B.Add((uint8)(V & 0xFF)); B.Add((uint8)(V >> 8)); }
		static void WriteLe32(TArray<uint8>& B, uint32 V) { for (int32 i = 0; i < 4; ++i) { B.Add((uint8)((V >> (8 * i)) & 0xFF)); } }
		static void WriteTag(TArray<uint8>& B, const char* Tag) { B.Append((const uint8*)Tag, 4); }

		static void Write(FWriteJob& Job)
		{
			if (!IFileManager::Get().MakeDirectory(*Job.Directory, true) && !IFileManager::Get().DirectoryExists(*Job.Directory))
			{
				UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechStream: cannot create the recording directory %s"), *Job.Directory);
				return;
			}
			const int64 DataBytes64 = (int64)Job.Audio.Num() * 2;
			if (DataBytes64 > (int64)MAX_int32 - 64)
			{
				UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechStream: recording too large to write as WAV"));
				return;
			}
			const uint32 DataBytes = (uint32)DataBytes64;
			TArray<uint8> Wav;
			Wav.Reserve(44 + (int32)DataBytes);
			WriteTag(Wav, "RIFF");
			WriteLe32(Wav, 36 + DataBytes);
			WriteTag(Wav, "WAVE");
			WriteTag(Wav, "fmt ");
			WriteLe32(Wav, 16);
			WriteLe16(Wav, 1);              // PCM
			WriteLe16(Wav, 1);              // mono
			WriteLe32(Wav, 16000);          // sample rate
			WriteLe32(Wav, 16000 * 2);      // byte rate
			WriteLe16(Wav, 2);              // block align
			WriteLe16(Wav, 16);             // bits per sample
			WriteTag(Wav, "data");
			WriteLe32(Wav, DataBytes);
			for (const float X : Job.Audio)
			{
				const float C = FMath::IsFinite(X) ? FMath::Clamp(X, -1.f, 1.f) : 0.f;
				const int32 S = FMath::Clamp(FMath::RoundToInt(C * 32767.f), -32768, 32767);
				WriteLe16(Wav, (uint16)(int16)S);
			}
			const FString WavPath = FPaths::Combine(Job.Directory, Job.BaseName + TEXT(".wav"));
			const FString JsonPath = FPaths::Combine(Job.Directory, Job.BaseName + TEXT(".json"));
			if (!FFileHelper::SaveArrayToFile(Wav, *WavPath))
			{
				UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechStream: could not write the recording %s"), *WavPath);
				return;
			}
			if (!FFileHelper::SaveStringToFile(Job.Json, *JsonPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
			{
				UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechStream: could not write the recording %s"), *JsonPath);
				return;
			}
			UE_LOG(LogModelPipelinesSpeech, Log, TEXT("USpeechStream: recorded %.1f s to %s"), Job.Audio.Num() / 16000.0, *WavPath);
		}
	}

	TUniquePtr<FSpeechStreamRecorder> FSpeechStreamRecorder::CreateIfEnabled(bool bPhonon, const FString& PackagePath, const FSpeechStreamSettings& Settings, bool bPartialsEnabled)
	{
		using namespace RecorderPrivate;
		if (CVarRecordStreams.GetValueOnGameThread() != 1)
		{
			return nullptr;
		}
		TUniquePtr<FSpeechStreamRecorder> R = MakeUnique<FSpeechStreamRecorder>();
		R->bPhonon = bPhonon;
		R->Package = PackagePath;
		R->Hangover = Settings.EndOfSpeechHangoverSeconds;
		R->PartialInterval = Settings.PartialIntervalSeconds;
		R->bPartials = bPartialsEnabled;
		R->ThresholdDb = Settings.SpeechThresholdDb;
		R->MaxUtterance = Settings.MaxUtteranceSeconds;
		R->MergeWindow = Settings.MergeWindowSeconds;
		R->MinSpeech = Settings.MinSpeechSeconds;
		float MaxSeconds = CVarRecordMaxSeconds.GetValueOnGameThread();
		if (!FMath::IsFinite(MaxSeconds)) { MaxSeconds = 1200.f; }
		MaxSeconds = FMath::Clamp(MaxSeconds, 1.f, 3600.f);
		R->MaxSamples = (int64)FMath::RoundToDouble((double)MaxSeconds * 16000.0);
		R->StartSeconds = FPlatformTime::Seconds();
		R->StartedUtc = FDateTime::UtcNow();
		return R;
	}

	void FSpeechStreamRecorder::AppendAudio(const TArray<float>& In)
	{
		if (bDone || In.Num() == 0)
		{
			return;
		}
		const int64 Room = MaxSamples - Audio.Num();
		if (Room <= 0)
		{
			bTruncated = true;
			return;
		}
		if ((int64)In.Num() > Room)
		{
			Audio.Append(In.GetData(), (int32)Room);
			bTruncated = true;
		}
		else
		{
			Audio.Append(In);
		}
	}

	FSpeechStreamRecorder::FEvent& FSpeechStreamRecorder::AddEvent(const TCHAR* Type)
	{
		FEvent& E = Events.AddDefaulted_GetRef();
		E.Type = Type;
		E.T = FPlatformTime::Seconds() - StartSeconds;
		E.Sample = Audio.Num();
		return E;
	}

	void FSpeechStreamRecorder::AddPartial(const FString& Text)
	{
		FEvent& E = AddEvent(TEXT("partial"));
		E.Text = Text;
		E.bHasText = true;
	}

	void FSpeechStreamRecorder::AddFinal(const FString& Text, bool bReplaces, float InferenceMs)
	{
		FEvent& E = AddEvent(TEXT("final"));
		E.Text = Text;
		E.bHasText = true;
		E.bHasFinal = true;
		E.bReplaces = bReplaces;
		E.InferenceMs = InferenceMs;
	}

	void FSpeechStreamRecorder::AddStopped(bool bOk, const FString& Message)
	{
		FEvent& E = AddEvent(TEXT("stopped"));
		E.bHasStopped = true;
		E.bOk = bOk;
		E.Message = Message;
	}

	void FSpeechStreamRecorder::Finish()
	{
		using namespace RecorderPrivate;
		if (bDone)
		{
			return;
		}
		bDone = true;

		// The JSON is built here (small); the WAV conversion and all file IO run on the background task.
		const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
		Root->SetNumberField(TEXT("version"), 1);
		Root->SetStringField(TEXT("engine"), bPhonon ? TEXT("Phonon") : TEXT("NNE"));
		Root->SetStringField(TEXT("package"), Package);
		Root->SetNumberField(TEXT("sample_rate"), 16000);
		Root->SetNumberField(TEXT("num_samples"), (double)Audio.Num());
		Root->SetBoolField(TEXT("truncated"), bTruncated);
		Root->SetStringField(TEXT("started_utc"), StartedUtc.ToIso8601());
		const TSharedPtr<FJsonObject> Set = MakeShared<FJsonObject>();
		Set->SetNumberField(TEXT("hangover"), Hangover);
		Set->SetNumberField(TEXT("partial_interval"), PartialInterval);
		Set->SetBoolField(TEXT("partials"), bPartials);
		Set->SetNumberField(TEXT("threshold_db"), ThresholdDb);
		Set->SetNumberField(TEXT("max_utterance"), MaxUtterance);
		Set->SetNumberField(TEXT("merge_window"), MergeWindow);
		Set->SetNumberField(TEXT("min_speech"), MinSpeech);
		Root->SetObjectField(TEXT("settings"), Set);
		TArray<TSharedPtr<FJsonValue>> EventValues;
		EventValues.Reserve(Events.Num());
		for (const FEvent& E : Events)
		{
			const TSharedPtr<FJsonObject> O = MakeShared<FJsonObject>();
			O->SetStringField(TEXT("type"), E.Type);
			O->SetNumberField(TEXT("t"), E.T);
			O->SetNumberField(TEXT("sample"), (double)E.Sample);
			if (E.bHasText) { O->SetStringField(TEXT("text"), E.Text); }
			if (E.bHasFinal)
			{
				O->SetBoolField(TEXT("replaces"), E.bReplaces);
				O->SetNumberField(TEXT("inference_ms"), E.InferenceMs);
			}
			if (E.bHasStopped)
			{
				O->SetBoolField(TEXT("ok"), E.bOk);
				O->SetStringField(TEXT("message"), E.Message);
			}
			EventValues.Add(MakeShared<FJsonValueObject>(O));
		}
		Root->SetArrayField(TEXT("events"), EventValues);

		TSharedRef<FWriteJob> Job = MakeShared<FWriteJob>();
		const TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer = TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Job->Json);
		if (!FJsonSerializer::Serialize(Root, Writer))
		{
			UE_LOG(LogModelPipelinesSpeech, Warning, TEXT("USpeechStream: could not serialize the recording JSON"));
			Audio.Empty();
			Events.Empty();
			return;
		}
		Job->Directory = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("ModelPipelines"), TEXT("SpeechRecordings"));
		Job->BaseName = FString::Printf(TEXT("%s-%u-%d"), *StartedUtc.ToString(TEXT("%Y%m%d-%H%M%S")), FPlatformProcess::GetCurrentProcessId(), RecordingCounter.fetch_add(1) + 1);   // PID: two processes, same second
		Job->Audio = MoveTemp(Audio);
		Events.Empty();

		UE::Tasks::Launch(UE_SOURCE_LOCATION, [Job]() { Write(*Job); });
	}
}

#endif // !UE_BUILD_SHIPPING
