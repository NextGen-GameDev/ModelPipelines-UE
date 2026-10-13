// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineMcpRun.h"

#include "Containers/Ticker.h"
#include "Engine/Texture.h"
#include "HAL/PlatformMemory.h"
#include "HAL/PlatformTime.h"
#include "Math/UnrealMathUtility.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "ImageClassificationPipeline.h"
#include "ModelPipelinesCoreModule.h"
#include "ObjectDetectionPipeline.h"
#include "PipelineBackend.h"
#include "PipelineBase.h"
#include "PipelineImage.h"
#include "PipelineMcpJob.h"
#include "PipelineMcpHardware.h"
#include "PipelineMcpJson.h"
#include "PipelineMcpLog.h"
#include "PipelineModelPackage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "TextClassificationPipeline.h"
#include "TextEmbeddingPipeline.h"
#include "TextPipelineLibrary.h"
#include "VisionPipelineLibrary.h"

namespace PipelineMcpRunPrivate
{
	static constexpr int32 MaxTexts = 64;
	static constexpr int32 MinRuns = 1;
	static constexpr int32 MaxRuns = 1000;
	static const TCHAR* DefaultText = TEXT("The quick brown fox jumps over the lazy dog.");
	static const TCHAR* TextsCountError = TEXT("texts: give 1 to 64 strings");

	using FJsonArray = TArray<TSharedPtr<FJsonValue>>;

	/** Median of the values (mean of the two middle ones for an even count); 0 for none. */
	static double Median(TArray<double> Values)
	{
		const int32 N = Values.Num();
		if (N == 0)
		{
			return 0.0;
		}
		Values.Sort();
		return (N % 2 == 1) ? Values[N / 2] : 0.5 * (Values[N / 2 - 1] + Values[N / 2]);
	}

	/** Nearest-rank 90th percentile: the ceil(0.9*n)-th smallest value (integer arithmetic); 0 for none. */
	static double P90(TArray<double> Values)
	{
		const int32 N = Values.Num();
		if (N == 0)
		{
			return 0.0;
		}
		Values.Sort();
		const int32 Rank = FMath::Clamp((9 * N + 9) / 10, 1, N);
		return Values[Rank - 1];
	}

	static double MinOf(const TArray<double>& Values)
	{
		double M = Values.Num() > 0 ? Values[0] : 0.0;
		for (double V : Values) { M = FMath::Min(M, V); }
		return M;
	}

	static double MaxOf(const TArray<double>& Values)
	{
		double M = Values.Num() > 0 ? Values[0] : 0.0;
		for (double V : Values) { M = FMath::Max(M, V); }
		return M;
	}

	/** {"median"} or, with bFull, {"median","p90","min","max"}. */
	static TSharedRef<FJsonObject> StatsToJson(const TArray<double>& Values, bool bFull)
	{
		TSharedRef<FJsonObject> Obj = MakeShared<FJsonObject>();
		Obj->SetField(TEXT("median"), PipelineMcpJson::FiniteOrNull(Median(Values)));
		if (bFull)
		{
			Obj->SetField(TEXT("p90"), PipelineMcpJson::FiniteOrNull(P90(Values)));
			Obj->SetField(TEXT("min"), PipelineMcpJson::FiniteOrNull(MinOf(Values)));
			Obj->SetField(TEXT("max"), PipelineMcpJson::FiniteOrNull(MaxOf(Values)));
		}
		return Obj;
	}

	/** Process-wide used physical memory in MiB (approximate). */
	static double UsedPhysicalMiB()
	{
		return static_cast<double>(FPlatformMemory::GetStats().UsedPhysical) / (1024.0 * 1024.0);
	}

	/** "<EPipelineError>: <message>". */
	static FString CallError(const FPipelineResult& Result)
	{
		return FString::Printf(TEXT("%s: %s"), *PipelineMcpJson::ErrorToString(Result.Error), *Result.Message);
	}

	static UPipelineModelPackage* ResolvePackage(const FString& Arg)
	{
		return Cast<UPipelineModelPackage>(PipelineMcpJson::ResolveAsset(Arg));
	}

	static UTexture* ResolveTexture(const FString& Arg)
	{
		return Cast<UTexture>(PipelineMcpJson::ResolveAsset(Arg));
	}

	static FString PackageNotFound(const FString& Arg)
	{
		return FString(TEXT("Model package not found: ")) + Arg;
	}

	static FString TextureNotFound(const FString& Arg)
	{
		return FString(TEXT("Image not found or not a texture: ")) + Arg;
	}

	/** Game thread. The task's pipeline for Package (Create*Pipeline), or nullptr with OutResult saying why. */
	static UPipelineBase* CreatePipeline(EPipelineTask Task, UPipelineModelPackage* Package, FPipelineResult& OutResult)
	{
		switch (Task)
		{
		case EPipelineTask::TextEmbedding:       return UTextPipelineLibrary::CreateTextEmbeddingPipeline(Package, OutResult);
		case EPipelineTask::TextClassification:  return UTextPipelineLibrary::CreateTextClassificationPipeline(Package, OutResult);
		case EPipelineTask::ImageClassification: return UVisionPipelineLibrary::CreateImageClassificationPipeline(Package, OutResult);
		case EPipelineTask::ObjectDetection:     return UVisionPipelineLibrary::CreateObjectDetectionPipeline(Package, OutResult);
		default:
			OutResult = FPipelineResult::Fail(EPipelineError::WrongTask, FString::Printf(TEXT("task %s has no runnable pipeline in this plugin"), LexToString(Task)));
			return nullptr;
		}
	}

	static TSharedRef<FJsonObject> LabelScoreToJson(const FPipelineLabelScore& L)
	{
		TSharedRef<FJsonObject> Obj = MakeShared<FJsonObject>();
		Obj->SetStringField(TEXT("label"), L.Label);
		Obj->SetNumberField(TEXT("index"), L.Index);
		Obj->SetField(TEXT("score"), PipelineMcpJson::FiniteOrNull(L.Score));
		return Obj;
	}

	static FJsonArray TopToJson(const FPipelineClassification& C)
	{
		FJsonArray Top;
		Top.Reserve(C.TopK.Num());
		for (const FPipelineLabelScore& L : C.TopK)
		{
			Top.Add(MakeShared<FJsonValueObject>(LabelScoreToJson(L)));
		}
		return Top;
	}

	/** Base of the two runners: weak job access, pipeline creation and the keep-alive bookkeeping. */
	class FRunnerBase : public FPipelineMcpRunner
	{
	public:
		virtual ~FRunnerBase() override
		{
			// The job released its runner (ended or cancelled at shutdown): stop any texture wait.
			RemoveTextureWait();
		}

	protected:
		TWeakObjectPtr<UPipelineMcpJob> WeakJob;

		/** Poll interval and limit of the texture-compilation wait (C38 amendment). */
		static constexpr float TextureWaitInterval = 0.1f;
		static constexpr double TextureWaitLimitSeconds = 60.0;

		/** Pending texture-compilation wait (empty when none). */
		FTSTicker::FDelegateHandle TextureWaitHandle;

		void RemoveTextureWait()
		{
			if (TextureWaitHandle.IsValid())
			{
				FTSTicker::RemoveTicker(TextureWaitHandle);
				TextureWaitHandle.Reset();
			}
		}

		/**
		 * Game thread. Runs Continue once Tex has finished its async editor build (directly when it is not compiling),
		 * polling from the core ticker so the game thread is never blocked. Fails the job after TextureWaitLimitSeconds.
		 * Continue only runs while the job is live and the runner exists; it must not capture the runner strongly.
		 */
		void WaitForTexture(UTexture* Tex, TUniqueFunction<void()> Continue)
		{
			if (!Tex || !Tex->IsCompiling())
			{
				Continue();
				return;
			}
			RemoveTextureWait();
			const TWeakPtr<FPipelineMcpRunner> WeakSelf = AsShared();
			const TWeakObjectPtr<UTexture> WeakTex(Tex);
			const FString TexName = Tex->GetName();
			const double WaitStart = FPlatformTime::Seconds();
			TextureWaitHandle = FTSTicker::GetCoreTicker().AddTicker(TEXT("PipelineMcpTextureWait"), TextureWaitInterval,
				[WeakSelf, WeakTex, TexName, WaitStart, Continue = MoveTemp(Continue)](float) mutable -> bool
				{
					const TSharedPtr<FPipelineMcpRunner> Pinned = WeakSelf.Pin();
					if (!Pinned.IsValid())
					{
						return false;
					}
					FRunnerBase* Self = static_cast<FRunnerBase*>(Pinned.Get());
					if (!Self->LiveJob())
					{
						Self->TextureWaitHandle.Reset();
						return false;
					}
					UTexture* T = WeakTex.Get();
					if (!T)
					{
						Self->TextureWaitHandle.Reset();
						Self->FailJob(TEXT("internal: the texture was released while waiting for it to compile"));
						return false;
					}
					if (T->IsCompiling())
					{
						if (FPlatformTime::Seconds() - WaitStart >= TextureWaitLimitSeconds)
						{
							Self->TextureWaitHandle.Reset();
							Self->FailJob(FString::Printf(TEXT("texture '%s' is still compiling after %.0f s"), *TexName, TextureWaitLimitSeconds));
							return false;
						}
						return true;
					}
					Self->TextureWaitHandle.Reset();
					Continue();
					return false;
				});
		}

		/** The job while it is still running; nullptr once it ended (late callbacks are dropped). */
		UPipelineMcpJob* LiveJob() const
		{
			UPipelineMcpJob* J = WeakJob.Get();
			return (J && !J->IsDone()) ? J : nullptr;
		}

		void FailJob(const FString& Message)
		{
			if (UPipelineMcpJob* J = LiveJob())
			{
				FPipelineMcpJobInternal::Fail(*J, Message);
			}
		}

		void SucceedJob(const TSharedRef<FJsonObject>& Root)
		{
			if (UPipelineMcpJob* J = LiveJob())
			{
				FPipelineMcpJobInternal::Finish(*J, true, PipelineMcpJson::ToCondensedString(Root));
			}
		}
	};

	// -----------------------------------------------------------------------------------------------------------------
	/** BenchmarkModelPackage: per device (CPU then GPU) a transient duplicate of the package, a pipeline, Runs + 1 sequential calls. */
	class FBenchmarkRunner final : public FRunnerBase
	{
	public:
		FBenchmarkRunner(const FString& InPackage, int32 InRuns, const FString& InDevices, const TArray<FString>& InTexts, const FString& InImage)
			: PackageArg(InPackage), Runs(InRuns), DevicesArg(InDevices), TextsArg(InTexts), ImageArg(InImage)
		{
		}

		virtual void Start(UPipelineMcpJob& Job) override
		{
			WeakJob = &Job;

			UPipelineModelPackage* Pkg = ResolvePackage(PackageArg);
			if (!Pkg)
			{
				FailJob(PackageNotFound(PackageArg));
				return;
			}
			if (Runs < MinRuns || Runs > MaxRuns)
			{
				FailJob(TEXT("runs must be 1 to 1000"));
				return;
			}
			if (DevicesArg.Equals(TEXT("cpu"), ESearchCase::IgnoreCase))
			{
				Devices = { EPipelineDevice::CPU };
			}
			else if (DevicesArg.Equals(TEXT("gpu"), ESearchCase::IgnoreCase))
			{
				Devices = { EPipelineDevice::GPU };
			}
			else if (DevicesArg.Equals(TEXT("both"), ESearchCase::IgnoreCase))
			{
				Devices = { EPipelineDevice::CPU, EPipelineDevice::GPU };
			}
			else
			{
				FailJob(TEXT("devices must be cpu, gpu or both"));
				return;
			}
			Task = Pkg->Task;
			if (IsTextTask(Task))
			{
				if (TextsArg.Num() > MaxTexts)
				{
					FailJob(FString::Printf(TEXT("texts: give at most %d strings (got %d)"), MaxTexts, TextsArg.Num()));
					return;
				}
				if (TextsArg.Num() > 0)
				{
					InputTexts = TextsArg;
					InputDesc = FString::Printf(TEXT("texts (%d)"), TextsArg.Num());
				}
				else
				{
					InputTexts = { DefaultText };
					InputDesc = TEXT("default text");
				}
			}
			else if (IsImageTask(Task))
			{
				if (!ImageArg.IsEmpty())
				{
					UTexture* Tex = ResolveTexture(ImageArg);
					if (!Tex)
					{
						FailJob(TextureNotFound(ImageArg));
						return;
					}
					Texture = Tex;
					FPipelineMcpJobInternal::KeepAlive(Job, Tex);
					InputDesc = FString(TEXT("image ")) + Tex->GetPathName();
				}
				else
				{
					const int32 W = Pkg->Preprocess.ImageWidth;
					const int32 H = Pkg->Preprocess.ImageHeight;
					// Cap the game-thread fill well below FPipelineImage::MaxSide (16384^2 = 1 GiB).
					constexpr int32 MaxSyntheticSide = 4096;
					if (W > MaxSyntheticSide || H > MaxSyntheticSide)
					{
						FailJob(FString::Printf(
							TEXT("image size %dx%d is too large for a synthetic benchmark input (max %dx%d); pass image"),
							W, H, MaxSyntheticSide, MaxSyntheticSide));
						return;
					}
					Synthetic.Width = W;
					Synthetic.Height = H;
					if (W > 0 && H > 0 && W <= FPipelineImage::MaxSide && H <= FPipelineImage::MaxSide)
					{
						Synthetic.Pixels.Init(FColor(128, 128, 128, 255), W * H);
					}
					// An unusable size is left to the pipeline, which reports it per call.
					InputDesc = FString::Printf(TEXT("synthetic %dx%d"), W, H);
				}
			}
			else
			{
				FailJob(FString::Printf(TEXT("task %s has no runnable pipeline in this plugin"), LexToString(Task)));
				return;
			}

			Source = Pkg;
			FPipelineMcpJobInternal::KeepAlive(Job, Pkg);
			Root = MakeShared<FJsonObject>();
			Root->SetStringField(TEXT("package"), Pkg->GetPathName());
			Root->SetStringField(TEXT("task"), PipelineMcpJson::TaskToString(Task));
			Root->SetNumberField(TEXT("runs"), Runs);
			Root->SetStringField(TEXT("input"), InputDesc);
			Root->SetStringField(TEXT("engine_version"), PipelineMcpHardware::EngineVersion());
			Root->SetStringField(TEXT("gpu_adapter"), PipelineMcpHardware::GpuAdapter());
			UE_LOG(LogModelPipelinesMCP, Log, TEXT("Benchmark %s: engine_version '%s', gpu_adapter '%s'"), *Pkg->GetPathName(),
				*Root->GetStringField(TEXT("engine_version")), *Root->GetStringField(TEXT("gpu_adapter")));
			DeviceIndex = 0;
			// A just-imported texture may still be compiling: wait once, before the first device (not timed).
			WaitForTexture(Texture.Get(), [this]() { StartDevice(); });
		}

	private:
		// Arguments.
		FString PackageArg;
		int32 Runs = 0;
		FString DevicesArg;
		TArray<FString> TextsArg;
		FString ImageArg;

		// Resolved input (Source and Texture are kept alive by the job).
		TWeakObjectPtr<UPipelineModelPackage> Source;
		EPipelineTask Task = EPipelineTask::None;
		TArray<EPipelineDevice> Devices;
		TArray<FString> InputTexts;
		TWeakObjectPtr<UTexture> Texture;
		FPipelineImage Synthetic;
		FString InputDesc;
		TSharedPtr<FJsonObject> Root;
		FJsonArray DeviceEntries;

		// Current device.
		int32 DeviceIndex = 0;
		TSharedPtr<FJsonObject> Entry;
		TWeakObjectPtr<UPipelineModelPackage> Duplicate;
		TWeakObjectPtr<UPipelineBase> Pipeline;
		int32 CallIndex = 0;
		double CallStart = 0.0;
		bool bHaveFirst = false;
		double FirstMs = 0.0;
		double MemBefore = 0.0;
		double MemAfterWarmup = 0.0;
		bool bAnySuccess = false;
		EPipelineDevice LastDevice = EPipelineDevice::Auto;
		TArray<double> InferenceMs, PreprocessMs, PostprocessMs, TotalMs;

		/** Sets up Devices[DeviceIndex] (or finishes the job after the last one). Entries that end without a call loop move on directly. */
		void StartDevice()
		{
			UPipelineMcpJob* Job = LiveJob();
			if (!Job)
			{
				return;
			}
			if (DeviceIndex >= Devices.Num())
			{
				Root->SetArrayField(TEXT("devices"), DeviceEntries);
				SucceedJob(Root.ToSharedRef());
				return;
			}
			const EPipelineDevice Device = Devices[DeviceIndex];
			Entry = MakeShared<FJsonObject>();
			Entry->SetStringField(TEXT("device"), PipelineMcpJson::DeviceToString(Device));
			CallIndex = 0;
			bHaveFirst = false;
			FirstMs = 0.0;
			bAnySuccess = false;
			LastDevice = EPipelineDevice::Auto;
			InferenceMs.Reset();
			PreprocessMs.Reset();
			PostprocessMs.Reset();
			TotalMs.Reset();

			FModelPipelinesCoreModule* Core = FModelPipelinesCoreModule::TryGet();
			if (!Core)
			{
				FailJob(TEXT("ModelPipelinesCore is not loaded"));
				return;
			}
			if (Device == EPipelineDevice::GPU && !Core->GetDefaultBackend().IsDeviceAvailable(EPipelineDevice::GPU))
			{
				Entry->SetBoolField(TEXT("available"), false);
				EndDevice(/*bMeasured*/ false);
				return;
			}
			Entry->SetBoolField(TEXT("available"), true);

			UPipelineModelPackage* Src = Source.Get();
			if (!Src)
			{
				FailJob(TEXT("internal: the model package was released while the benchmark ran"));
				return;
			}
			MemBefore = UsedPhysicalMiB();

			// Transient duplicate with the device to measure; the asset itself is never modified.
			UPackage* Transient = GetTransientPackage();
			const FName DupName = MakeUniqueObjectName(Transient, UPipelineModelPackage::StaticClass(),
				FName(*FString::Printf(TEXT("McpBenchmark_%s"), *PipelineMcpJson::DeviceToString(Device))));
			UPipelineModelPackage* Dup = DuplicateObject<UPipelineModelPackage>(Src, Transient, DupName);
			if (!Dup)
			{
				Entry->SetStringField(TEXT("error"), TEXT("could not duplicate the model package"));
				EndDevice(false);
				return;
			}
			Dup->ClearFlags(RF_Public | RF_Standalone | RF_Transactional);
			Dup->SetFlags(RF_Transient);
			Dup->PreferredDevice = Device;
			Duplicate = Dup;
			FPipelineMcpJobInternal::KeepAlive(*Job, Dup);

			FPipelineResult Created;
			UPipelineBase* P = CreatePipeline(Task, Dup, Created);
			if (!P)
			{
				Entry->SetStringField(TEXT("error"), Created.Message.IsEmpty() ? CallError(Created) : Created.Message);
				EndDevice(false);
				return;
			}
			Pipeline = P;
			FPipelineMcpJobInternal::KeepAlive(*Job, P);
			IssueCall();
		}

		/** Starts call number CallIndex (0 = warm-up). Its callback arrives on a later tick. */
		void IssueCall()
		{
			UPipelineBase* P = Pipeline.Get();
			if (!LiveJob())
			{
				return;
			}
			if (!P)
			{
				Entry->SetStringField(TEXT("error"), FString::Printf(TEXT("call %d: Internal: the pipeline was released"), CallIndex + 1));
				EndDevice(true);
				return;
			}
			const TSharedRef<FBenchmarkRunner> Self = StaticCastSharedRef<FBenchmarkRunner>(AsShared());
			CallStart = FPlatformTime::Seconds();
			switch (Task)
			{
			case EPipelineTask::TextEmbedding:
				CastChecked<UTextEmbeddingPipeline>(P)->EmbedAsync(InputTexts,
					[Self](const FPipelineResult& R, const TArray<FPipelineEmbedding>&) { Self->OnCallDone(R); });
				break;
			case EPipelineTask::TextClassification:
				CastChecked<UTextClassificationPipeline>(P)->ClassifyAsync(InputTexts, 1,
					[Self](const FPipelineResult& R, const TArray<FPipelineClassification>&) { Self->OnCallDone(R); });
				break;
			case EPipelineTask::ImageClassification:
				if (UTexture* Tex = Texture.Get())
				{
					CastChecked<UImageClassificationPipeline>(P)->ClassifyAsync(Tex, 5,
						[Self](const FPipelineResult& R, const FPipelineClassification&) { Self->OnCallDone(R); });
				}
				else
				{
					CastChecked<UImageClassificationPipeline>(P)->ClassifyImageAsync(Synthetic, 5,
						[Self](const FPipelineResult& R, const FPipelineClassification&) { Self->OnCallDone(R); });
				}
				break;
			case EPipelineTask::ObjectDetection:
				if (UTexture* Tex = Texture.Get())
				{
					CastChecked<UObjectDetectionPipeline>(P)->DetectAsync(Tex, 0.5f,
						[Self](const FPipelineResult& R, const TArray<FPipelineDetection>&) { Self->OnCallDone(R); });
				}
				else
				{
					CastChecked<UObjectDetectionPipeline>(P)->DetectImageAsync(Synthetic, 0.5f,
						[Self](const FPipelineResult& R, const TArray<FPipelineDetection>&) { Self->OnCallDone(R); });
				}
				break;
			default:
				Entry->SetStringField(TEXT("error"), FString::Printf(TEXT("task %s has no runnable pipeline in this plugin"), LexToString(Task)));
				EndDevice(true);
				break;
			}
		}

		void OnCallDone(const FPipelineResult& R)
		{
			if (!LiveJob())
			{
				return;
			}
			const double WallMs = (FPlatformTime::Seconds() - CallStart) * 1000.0;
			if (!R.bSuccess)
			{
				// Call numbers count the warm-up as call 1.
				Entry->SetStringField(TEXT("error"), FString::Printf(TEXT("call %d: %s"), CallIndex + 1, *CallError(R)));
				EndDevice(true);
				return;
			}
			bAnySuccess = true;
			LastDevice = R.Device;
			if (CallIndex == 0)
			{
				bHaveFirst = true;
				FirstMs = WallMs;
				MemAfterWarmup = UsedPhysicalMiB();
			}
			else
			{
				InferenceMs.Add(R.InferenceMs);
				PreprocessMs.Add(R.PreprocessMs);
				PostprocessMs.Add(R.PostprocessMs);
				TotalMs.Add(WallMs);
			}
			++CallIndex;
			if (CallIndex > Runs)
			{
				EndDevice(true);
			}
			else
			{
				IssueCall();
			}
		}

		/** Writes the statistics of the current device into its entry, releases its pipeline and duplicate, moves on. */
		void EndDevice(bool bMeasured)
		{
			UPipelineMcpJob* Job = LiveJob();
			if (!Job)
			{
				return;
			}
			if (bMeasured)
			{
				if (bAnySuccess)
				{
					Entry->SetStringField(TEXT("actual_device"), PipelineMcpJson::DeviceToString(LastDevice));
				}
				if (bHaveFirst)
				{
					Entry->SetField(TEXT("first_ms"), PipelineMcpJson::FiniteOrNull(FirstMs));
				}
				if (InferenceMs.Num() > 0)
				{
					Entry->SetObjectField(TEXT("inference_ms"), StatsToJson(InferenceMs, true));
					Entry->SetObjectField(TEXT("preprocess_ms"), StatsToJson(PreprocessMs, false));
					Entry->SetObjectField(TEXT("postprocess_ms"), StatsToJson(PostprocessMs, false));
					Entry->SetObjectField(TEXT("total_ms"), StatsToJson(TotalMs, true));
				}
				if (bHaveFirst)
				{
					TSharedRef<FJsonObject> Mem = MakeShared<FJsonObject>();
					Mem->SetField(TEXT("before"), PipelineMcpJson::FiniteOrNull(MemBefore));
					Mem->SetField(TEXT("after_warmup"), PipelineMcpJson::FiniteOrNull(MemAfterWarmup));
					Mem->SetField(TEXT("delta"), PipelineMcpJson::FiniteOrNull(MemAfterWarmup - MemBefore));
					Entry->SetObjectField(TEXT("memory_mb"), Mem);
				}
			}
			DeviceEntries.Add(MakeShared<FJsonValueObject>(Entry));
			Entry.Reset();

			// The next device is measured without this one's pipeline being held (no call of it is pending any more).
			FPipelineMcpJobInternal::Release(*Job, Pipeline.Get());
			FPipelineMcpJobInternal::Release(*Job, Duplicate.Get());
			Pipeline.Reset();
			Duplicate.Reset();

			++DeviceIndex;
			StartDevice();
		}
	};

	// -----------------------------------------------------------------------------------------------------------------
	/** RunPipelineSample: one call with the package's own PreferredDevice, readable result. */
	class FSampleRunner final : public FRunnerBase
	{
	public:
		FSampleRunner(const FString& InPackage, const TArray<FString>& InTexts, const FString& InImage, int32 InTopK, float InScoreThreshold)
			: PackageArg(InPackage), Texts(InTexts), ImageArg(InImage), TopK(InTopK), ScoreThreshold(InScoreThreshold)
		{
		}

		virtual void Start(UPipelineMcpJob& Job) override
		{
			WeakJob = &Job;

			UPipelineModelPackage* Pkg = ResolvePackage(PackageArg);
			if (!Pkg)
			{
				FailJob(PackageNotFound(PackageArg));
				return;
			}
			Task = Pkg->Task;
			const FString TaskName = PipelineMcpJson::TaskToString(Task);
			if (IsTextTask(Task))
			{
				if (Texts.Num() == 0)
				{
					FailJob(TaskName + TEXT(" needs texts"));
					return;
				}
				if (Texts.Num() > MaxTexts)
				{
					FailJob(TextsCountError);
					return;
				}
			}
			else if (IsImageTask(Task))
			{
				if (ImageArg.IsEmpty())
				{
					FailJob(TaskName + TEXT(" needs image (a texture asset path)"));
					return;
				}
				if (Task == EPipelineTask::ObjectDetection && !(ScoreThreshold >= 0.0f && ScoreThreshold <= 1.0f))
				{
					FailJob(TEXT("score_threshold must be 0 to 1"));
					return;
				}
				UTexture* Tex = ResolveTexture(ImageArg);
				if (!Tex)
				{
					FailJob(TextureNotFound(ImageArg));
					return;
				}
				Texture = Tex;
				FPipelineMcpJobInternal::KeepAlive(Job, Tex);
			}
			else
			{
				FailJob(FString::Printf(TEXT("task %s has no runnable pipeline in this plugin"), LexToString(Task)));
				return;
			}

			FPipelineMcpJobInternal::KeepAlive(Job, Pkg);
			FPipelineResult Created;
			UPipelineBase* P = CreatePipeline(Task, Pkg, Created);
			if (!P)
			{
				FailJob(CallError(Created));
				return;
			}
			FPipelineMcpJobInternal::KeepAlive(Job, P);
			const int32 NumLabels = FMath::Max(1, Pkg->Labels.Num());
			const int32 K = FMath::Clamp(TopK, 1, NumLabels);

			// Text tasks have no Texture, so they never wait. P is kept alive by the job.
			Pipeline = P;
			WaitForTexture(Texture.Get(), [this, K]() { IssueCall(K); });
		}

	private:
		FString PackageArg;
		TArray<FString> Texts;
		FString ImageArg;
		int32 TopK = 1;
		float ScoreThreshold = 0.5f;
		EPipelineTask Task = EPipelineTask::None;
		TWeakObjectPtr<UTexture> Texture;
		TWeakObjectPtr<UPipelineBase> Pipeline;

		/** Issues the single pipeline call (after any texture wait). */
		void IssueCall(int32 K)
		{
			if (!LiveJob())
			{
				return;
			}
			UPipelineBase* P = Pipeline.Get();
			if (!P)
			{
				FailJob(TEXT("internal: the pipeline was released before the call"));
				return;
			}
			const TSharedRef<FSampleRunner> Self = StaticCastSharedRef<FSampleRunner>(AsShared());
			switch (Task)
			{
			case EPipelineTask::TextEmbedding:
				CastChecked<UTextEmbeddingPipeline>(P)->EmbedAsync(Texts,
					[Self](const FPipelineResult& R, const TArray<FPipelineEmbedding>& E) { Self->OnEmbeddings(R, E); });
				break;
			case EPipelineTask::TextClassification:
				CastChecked<UTextClassificationPipeline>(P)->ClassifyAsync(Texts, K,
					[Self](const FPipelineResult& R, const TArray<FPipelineClassification>& C) { Self->OnTextClassification(R, C); });
				break;
			case EPipelineTask::ImageClassification:
				CastChecked<UImageClassificationPipeline>(P)->ClassifyAsync(Texture.Get(), K,
					[Self](const FPipelineResult& R, const FPipelineClassification& C) { Self->OnImageClassification(R, C); });
				break;
			case EPipelineTask::ObjectDetection:
				CastChecked<UObjectDetectionPipeline>(P)->DetectAsync(Texture.Get(), ScoreThreshold,
					[Self](const FPipelineResult& R, const TArray<FPipelineDetection>& D) { Self->OnDetections(R, D); });
				break;
			default:
				FailJob(FString::Printf(TEXT("task %s has no runnable pipeline in this plugin"), LexToString(Task)));
				break;
			}
		}

		TSharedRef<FJsonObject> MakeRoot(const FPipelineResult& R) const
		{
			TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
			Root->SetStringField(TEXT("task"), PipelineMcpJson::TaskToString(Task));
			Root->SetObjectField(TEXT("timings"), PipelineMcpJson::TimingsToJson(R));
			return Root;
		}

		FString ImagePath() const
		{
			const UTexture* Tex = Texture.Get();
			return Tex ? Tex->GetPathName() : ImageArg;
		}

		void OnEmbeddings(const FPipelineResult& R, const TArray<FPipelineEmbedding>& Embeddings)
		{
			if (!LiveJob())
			{
				return;
			}
			if (!R.bSuccess)
			{
				FailJob(CallError(R));
				return;
			}
			TSharedRef<FJsonObject> Root = MakeRoot(R);
			TArray<double> Norms;
			FJsonArray Items;
			for (int32 i = 0; i < Embeddings.Num(); ++i)
			{
				const TArray<float>& V = Embeddings[i].Values;
				double SumSq = 0.0;
				for (float X : V) { SumSq += static_cast<double>(X) * static_cast<double>(X); }
				const double Norm = FMath::Sqrt(SumSq);
				Norms.Add(Norm);

				TSharedRef<FJsonObject> Item = MakeShared<FJsonObject>();
				Item->SetStringField(TEXT("text"), Texts.IsValidIndex(i) ? Texts[i] : FString());
				Item->SetNumberField(TEXT("dim"), V.Num());
				Item->SetField(TEXT("norm"), PipelineMcpJson::FiniteOrNull(Norm));
				FJsonArray Head;
				for (int32 j = 0; j < FMath::Min(8, V.Num()); ++j)
				{
					Head.Add(PipelineMcpJson::FiniteOrNull(V[j]));
				}
				Item->SetArrayField(TEXT("head"), Head);
				Items.Add(MakeShared<FJsonValueObject>(Item));
			}
			Root->SetArrayField(TEXT("embeddings"), Items);

			if (Embeddings.Num() >= 2)
			{
				FJsonArray Rows;
				for (int32 a = 0; a < Embeddings.Num(); ++a)
				{
					FJsonArray Row;
					for (int32 b = 0; b < Embeddings.Num(); ++b)
					{
						const TArray<float>& A = Embeddings[a].Values;
						const TArray<float>& B = Embeddings[b].Values;
						double Dot = 0.0;
						const int32 N = FMath::Min(A.Num(), B.Num());
						for (int32 k = 0; k < N; ++k) { Dot += static_cast<double>(A[k]) * static_cast<double>(B[k]); }
						const double Den = Norms[a] * Norms[b];
						const double Cos = (Norms[a] > 0.0 && Norms[b] > 0.0) ? Dot / Den : 0.0;
						Row.Add(PipelineMcpJson::FiniteOrNull(FMath::RoundToDouble(Cos * 1e6) / 1e6));
					}
					Rows.Add(MakeShared<FJsonValueArray>(Row));
				}
				Root->SetArrayField(TEXT("similarity"), Rows);
			}
			SucceedJob(Root);
		}

		void OnTextClassification(const FPipelineResult& R, const TArray<FPipelineClassification>& Results)
		{
			if (!LiveJob())
			{
				return;
			}
			if (!R.bSuccess)
			{
				FailJob(CallError(R));
				return;
			}
			TSharedRef<FJsonObject> Root = MakeRoot(R);
			FJsonArray Items;
			for (int32 i = 0; i < Results.Num(); ++i)
			{
				TSharedRef<FJsonObject> Item = MakeShared<FJsonObject>();
				Item->SetStringField(TEXT("text"), Texts.IsValidIndex(i) ? Texts[i] : FString());
				Item->SetArrayField(TEXT("top"), TopToJson(Results[i]));
				Items.Add(MakeShared<FJsonValueObject>(Item));
			}
			Root->SetArrayField(TEXT("results"), Items);
			SucceedJob(Root);
		}

		void OnImageClassification(const FPipelineResult& R, const FPipelineClassification& C)
		{
			if (!LiveJob())
			{
				return;
			}
			if (!R.bSuccess)
			{
				FailJob(CallError(R));
				return;
			}
			TSharedRef<FJsonObject> Root = MakeRoot(R);
			Root->SetStringField(TEXT("image"), ImagePath());
			Root->SetArrayField(TEXT("top"), TopToJson(C));
			SucceedJob(Root);
		}

		void OnDetections(const FPipelineResult& R, const TArray<FPipelineDetection>& Detections)
		{
			if (!LiveJob())
			{
				return;
			}
			if (!R.bSuccess)
			{
				FailJob(CallError(R));
				return;
			}
			TArray<FPipelineDetection> Sorted = Detections;
			Sorted.StableSort([](const FPipelineDetection& A, const FPipelineDetection& B) { return A.Score > B.Score; });

			TSharedRef<FJsonObject> Root = MakeRoot(R);
			Root->SetStringField(TEXT("image"), ImagePath());
			FJsonArray Items;
			for (const FPipelineDetection& D : Sorted)
			{
				TSharedRef<FJsonObject> Item = MakeShared<FJsonObject>();
				Item->SetStringField(TEXT("label"), D.Label);
				Item->SetNumberField(TEXT("index"), D.Index);
				Item->SetField(TEXT("score"), PipelineMcpJson::FiniteOrNull(D.Score));
				FJsonArray Box;
				Box.Add(PipelineMcpJson::FiniteOrNull(D.Box.Min.X));
				Box.Add(PipelineMcpJson::FiniteOrNull(D.Box.Min.Y));
				Box.Add(PipelineMcpJson::FiniteOrNull(D.Box.Max.X));
				Box.Add(PipelineMcpJson::FiniteOrNull(D.Box.Max.Y));
				Item->SetArrayField(TEXT("box"), Box);
				Items.Add(MakeShared<FJsonValueObject>(Item));
			}
			Root->SetArrayField(TEXT("detections"), Items);
			SucceedJob(Root);
		}
	};
}

namespace PipelineMcpRun
{
	UPipelineMcpJob* LaunchBenchmark(const FString& Package, int32 Runs, const FString& Devices, const TArray<FString>& Texts, const FString& Image)
	{
		return FPipelineMcpJobInternal::Launch(MakeShared<PipelineMcpRunPrivate::FBenchmarkRunner>(Package, Runs, Devices, Texts, Image));
	}

	UPipelineMcpJob* LaunchSample(const FString& Package, const TArray<FString>& Texts, const FString& Image, int32 TopK, float ScoreThreshold)
	{
		return FPipelineMcpJobInternal::Launch(MakeShared<PipelineMcpRunPrivate::FSampleRunner>(Package, Texts, Image, TopK, ScoreThreshold));
	}
}
