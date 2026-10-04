// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
// WP4.6 visible tests (docs/contracts.md C28): ReadTexturePixelsAsync. Headless cases are Pipelines.Vision.Readback.*, the
// cases that need a real RHI are Pipelines.Vision.Readback.RHI.* (skipped under -nullrhi unless MODELPIPELINES_REQUIRE_RHI=1).
// The readback needs game-thread ticks to progress, so no test ever blocks the game thread: every wait is a latent step
// driven by an FTSTicker poll with a timeout.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include <atomic>
#include "Async/Async.h"
#include "Containers/Ticker.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureCube.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/PlatformTLS.h"
#include "Math/RandomStream.h"
#include "RHIGlobals.h"
#include "Tasks/Task.h"
#include "UObject/GarbageCollection.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineImage.h"
#include "PipelineTextureReader.h"
#include "PipelineTypes.h"
#include "PipelineVisionTestData.h"

using namespace UE::ModelPipelines::VisionTests;
using UE::ModelPipelines::Vision::ReadTexturePixelsAsync;
using UE::ModelPipelines::Vision::GetImageReadStats;
using UE::ModelPipelines::Vision::ResetImageReadStats;
using UE::ModelPipelines::Vision::FPipelineImageReadStats;

namespace PipelineReadbackSpecPrivate
{
	/** The C28 message for "no RHI". */
	static const TCHAR* const NoRhiMessage = TEXT("texture input needs a GPU RHI (the editor runs with -nullrhi); pass an FPipelineImage instead");

	// ------------------------------------------------------------------------------------------------------------------
	// Capture of one read: everything the callback saw, written on whatever thread ran it, read on the game thread.
	// ------------------------------------------------------------------------------------------------------------------
	struct FCapture
	{
		FCriticalSection Lock;
		FPipelineResult Result;
		FPipelineImage Image;
		UE::Tasks::FTask Task;
		std::atomic<int32> Calls{ 0 };
		std::atomic<bool> bTaskSet{ false };
		std::atomic<bool> bInCallback{ false };
		std::atomic<bool> bCallbackReturned{ false };
		std::atomic<bool> bSawGameThread{ false };
		std::atomic<bool> bSawCallerThread{ false };
		std::atomic<bool> bTaskDoneWhileCallbackRunning{ false };
		std::atomic<bool> bContinuationRan{ false };
		std::atomic<bool> bContinuationSawReturned{ false };
		uint32 CallerThreadId = 0;
		int32 CallbackSleepMs = 0;
		double CallMs = 0.0;
		int32 Id = 0;

		void OnCallback(const FPipelineResult& R, FPipelineImage&& Img)
		{
			Calls.fetch_add(1);
			bInCallback = true;
			if (IsInGameThread())
			{
				bSawGameThread = true;
			}
			if (FPlatformTLS::GetCurrentThreadId() == CallerThreadId)
			{
				bSawCallerThread = true;
			}
			{
				FScopeLock L(&Lock);
				Result = R;
				Image = MoveTemp(Img);
			}
			if (CallbackSleepMs > 0)
			{
				FPlatformProcess::Sleep(CallbackSleepMs / 1000.f);
			}
			bCallbackReturned = true;
			bInCallback = false;
		}

		/** Game thread, once per tick: catches a returned task that completes while OnRead is still running. */
		void Poll()
		{
			if (bInCallback && bTaskSet)
			{
				FScopeLock L(&Lock);
				if (Task.IsValid() && Task.IsCompleted() && bInCallback)
				{
					bTaskDoneWhileCallbackRunning = true;
				}
			}
		}

		bool IsFinished()
		{
			if (Calls.load() < 1 || !bCallbackReturned || !bTaskSet)
			{
				return false;
			}
			FScopeLock L(&Lock);
			return Task.IsValid() && Task.IsCompleted();
		}
	};
	using FCapturePtr = TSharedPtr<FCapture, ESPMode::ThreadSafe>;

	static std::atomic<int32> GNextCaptureId{ 1 };

	/** Game thread (or any thread for the not-on-game-thread test). Calls ReadTexturePixelsAsync and records the timing. */
	static FCapturePtr StartRead(UTexture* Tex, int32 CallbackSleepMs = 0)
	{
		FCapturePtr Cap = MakeShared<FCapture, ESPMode::ThreadSafe>();
		Cap->Id = GNextCaptureId.fetch_add(1);
		Cap->CallerThreadId = FPlatformTLS::GetCurrentThreadId();
		Cap->CallbackSleepMs = CallbackSleepMs;
		const double T0 = FPlatformTime::Seconds();
		UE::Tasks::FTask Task = ReadTexturePixelsAsync(Tex, [Cap](const FPipelineResult& R, FPipelineImage&& Img)
		{
			Cap->OnCallback(R, MoveTemp(Img));
		});
		Cap->CallMs = (FPlatformTime::Seconds() - T0) * 1000.0;
		{
			FScopeLock L(&Cap->Lock);
			Cap->Task = Task;
		}
		Cap->bTaskSet = true;
		return Cap;
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Latent waiting (game-thread ticker, never blocks)
	// ------------------------------------------------------------------------------------------------------------------
	struct FWaitState
	{
		double Start = 0.0;
		double DoneAt = -1.0;
	};

	/** Polls until every capture finished (callback returned and task completed) or TimeoutSec passed; then SettleSec more (to
	 *  catch a second callback), then OnDone. A timeout is an error naming how many reads are still pending. */
	static void WaitForAll(FAutomationTestBase* T, const TArray<FCapturePtr>& Caps, double TimeoutSec, double SettleSec, TFunction<void()> OnDone)
	{
		TSharedRef<FWaitState> St = MakeShared<FWaitState>();
		St->Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("PipelineReadbackSpecWait"), 0.f, [T, Caps, TimeoutSec, SettleSec, St, OnDone = MoveTemp(OnDone)](float) -> bool
		{
			int32 Pending = 0;
			for (const FCapturePtr& C : Caps)
			{
				C->Poll();
				if (!C->IsFinished())
				{
					++Pending;
				}
			}
			const double Now = FPlatformTime::Seconds();
			if (Pending == 0)
			{
				if (St->DoneAt < 0.0)
				{
					St->DoneAt = Now;
				}
				if (Now - St->DoneAt >= SettleSec)
				{
					OnDone();
					return false;
				}
				return true;
			}
			if (Now - St->Start > TimeoutSec)
			{
				T->AddError(FString::Printf(TEXT("Timed out after %.1f s: %d of %d reads still have no finished callback + completed task (callback never fired, or the returned task never completed)."),
					TimeoutSec, Pending, Caps.Num()));
				OnDone();
				return false;
			}
			return true;
		});
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Verification
	// ------------------------------------------------------------------------------------------------------------------

	/** Threading / lifecycle rules that hold for every read, success or failure. */
	static void VerifyLifecycleImpl(FAutomationTestBase& T, const FCapturePtr& C, const FString& Label)
	{
		T.TestEqual(*FString::Printf(TEXT("%s: OnRead call count (exactly once, also after settling)"), *Label), C->Calls.load(), 1);
		T.TestFalse(*FString::Printf(TEXT("%s: OnRead must not run on the game thread"), *Label), C->bSawGameThread.load());
		T.TestFalse(*FString::Printf(TEXT("%s: OnRead must not run on the calling thread (never inside the call)"), *Label), C->bSawCallerThread.load());
		T.TestFalse(*FString::Printf(TEXT("%s: the returned task must complete only after OnRead returned"), *Label), C->bTaskDoneWhileCallbackRunning.load());
		FScopeLock L(&C->Lock);
		T.TestTrue(*FString::Printf(TEXT("%s: the call returned a valid task"), *Label), C->Task.IsValid());
		T.TestTrue(*FString::Printf(TEXT("%s: the returned task completed"), *Label), C->Task.IsValid() && C->Task.IsCompleted());
	}

	static void VerifyFailure(FAutomationTestBase& T, const FCapturePtr& C, EPipelineError Expected, const FString& Label)
	{
		VerifyLifecycleImpl(T, C, Label);
		FScopeLock L(&C->Lock);
		T.TestFalse(*FString::Printf(TEXT("%s: Result.bSuccess is false"), *Label), C->Result.bSuccess);
		T.TestEqual(*FString::Printf(TEXT("%s: Result.Error"), *Label), (int32)C->Result.Error, (int32)Expected);
		T.TestFalse(*FString::Printf(TEXT("%s: failure carries a message"), *Label), C->Result.Message.IsEmpty());
		T.TestEqual(*FString::Printf(TEXT("%s: failure image is empty (Width)"), *Label), C->Image.Width, 0);
		T.TestEqual(*FString::Printf(TEXT("%s: failure image is empty (Height)"), *Label), C->Image.Height, 0);
		T.TestEqual(*FString::Printf(TEXT("%s: failure image is empty (Pixels)"), *Label), C->Image.Pixels.Num(), 0);
	}

	/** RGB equality, worst channel difference reported. A is ignored (C27). */
	static void CompareRgb(FAutomationTestBase& T, const FPipelineImage& Got, const FPipelineImage& Want, const FString& Label)
	{
		if (Got.Width != Want.Width || Got.Height != Want.Height || Got.Pixels.Num() != Want.Pixels.Num())
		{
			T.AddError(FString::Printf(TEXT("%s: image size %dx%d (%d pixels), expected %dx%d (%d pixels)"), *Label,
				Got.Width, Got.Height, Got.Pixels.Num(), Want.Width, Want.Height, Want.Pixels.Num()));
			return;
		}
		int64 Mismatches = 0;
		int32 MaxDiff = 0;
		int32 FirstIndex = -1;
		for (int32 i = 0; i < Got.Pixels.Num(); ++i)
		{
			const FColor& A = Got.Pixels[i];
			const FColor& B = Want.Pixels[i];
			const int32 D = FMath::Max3(FMath::Abs((int32)A.R - (int32)B.R), FMath::Abs((int32)A.G - (int32)B.G), FMath::Abs((int32)A.B - (int32)B.B));
			if (D != 0)
			{
				++Mismatches;
				MaxDiff = FMath::Max(MaxDiff, D);
				if (FirstIndex < 0)
				{
					FirstIndex = i;
				}
			}
		}
		if (Mismatches != 0)
		{
			const FColor& A = Got.Pixels[FirstIndex];
			const FColor& B = Want.Pixels[FirstIndex];
			T.AddError(FString::Printf(TEXT("%s: %lld of %d pixels differ (worst channel diff %d); first at (%d,%d): got RGB(%d,%d,%d) expected RGB(%d,%d,%d). Exact equality is required."),
				*Label, Mismatches, Got.Pixels.Num(), MaxDiff, FirstIndex % Got.Width, FirstIndex / Got.Width, A.R, A.G, A.B, B.R, B.G, B.B));
		}
	}

	static void VerifyExact(FAutomationTestBase& T, const FCapturePtr& C, const FPipelineImage& Want, const FString& Label)
	{
		VerifyLifecycleImpl(T, C, Label);
		FScopeLock L(&C->Lock);
		T.TestTrue(*FString::Printf(TEXT("%s: read succeeded (message: %s)"), *Label, *C->Result.Message), C->Result.bSuccess);
		if (C->Result.bSuccess)
		{
			T.TestEqual(*FString::Printf(TEXT("%s: success has Error None"), *Label), (int32)C->Result.Error, (int32)EPipelineError::None);
			CompareRgb(T, C->Image, Want, Label);
		}
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Test data
	// ------------------------------------------------------------------------------------------------------------------
	static TSharedPtr<FJsonObject> GCachedRoot;
	static TMap<FString, TSharedPtr<FPipelineImage>> GImageCache;

	/** Golden image (s02 300x200, s04 37x53) by id; cached copy. nullptr + AddError (path and generate command) when missing. */
	static TSharedPtr<FPipelineImage> GetGoldenImage(FAutomationTestBase& T, const FString& Id)
	{
		if (const TSharedPtr<FPipelineImage>* Cached = GImageCache.Find(Id))
		{
			return *Cached;
		}
		if (!GCachedRoot.IsValid())
		{
			FString Err;
			GCachedRoot = LoadVisionGoldens(Err);
			if (!GCachedRoot.IsValid())
			{
				T.AddError(Err);
				return nullptr;
			}
		}
		const TArray<TSharedPtr<FJsonValue>>* Images = nullptr;
		if (GCachedRoot->TryGetArrayField(TEXT("images"), Images))
		{
			for (const TSharedPtr<FJsonValue>& V : *Images)
			{
				const TSharedPtr<FJsonObject>& O = V->AsObject();
				if (O.IsValid() && O->GetStringField(TEXT("id")) == Id)
				{
					TSharedPtr<FPipelineImage> Img = MakeShared<FPipelineImage>();
					FString Err;
					if (!LoadPng(GCachedRoot->GetStringField(TEXT("root")) / O->GetStringField(TEXT("file")), *Img, Err))
					{
						T.AddError(Err);
						return nullptr;
					}
					GImageCache.Add(Id, Img);
					return Img;
				}
			}
		}
		T.AddError(FString::Printf(TEXT("Golden image id '%s' not listed in vision_goldens.json."), *Id));
		return nullptr;
	}

	static FPipelineImage MakeNoise(int32 W, int32 H, int32 Seed, uint8 Alpha = 255)
	{
		FRandomStream Rng(Seed);
		FPipelineImage Img;
		Img.Width = W;
		Img.Height = H;
		Img.Pixels.SetNumUninitialized(W * H);
		for (FColor& C : Img.Pixels)
		{
			C = FColor((uint8)Rng.RandRange(0, 255), (uint8)Rng.RandRange(0, 255), (uint8)Rng.RandRange(0, 255), Alpha);
		}
		return Img;
	}

	/** false + info/error when there is no RHI (error only when MODELPIPELINES_REQUIRE_RHI=1). */
	static bool RequireRhi(FAutomationTestBase& T)
	{
		if (HasRhi())
		{
			return true;
		}
		if (IsRhiRequired())
		{
			T.AddError(TEXT("MODELPIPELINES_REQUIRE_RHI=1 but no RHI is available (run without -nullrhi)."));
		}
		else
		{
			T.AddInfo(TEXT("RHI not available: skipped"));
		}
		return false;
	}

	/** Keeps UObjects alive until the test finished (released on the game thread). */
	using FKeepAlive = TSharedRef<TArray<TStrongObjectPtr<UObject>>>;

	struct FTexCase
	{
		const TCHAR* Name;
		bool bRenderTarget;
		EPixelFormat Format;                 // textures
		ETextureRenderTargetFormat Rtf;      // render targets
	};

	static const FTexCase GTexCases[] =
	{
		{ TEXT("Texture_B8G8R8A8"),  false, PF_B8G8R8A8,  RTF_RGBA8 },
		{ TEXT("Texture_R8G8B8A8"),  false, PF_R8G8B8A8,  RTF_RGBA8 },
		{ TEXT("Texture_FloatRGBA"), false, PF_FloatRGBA, RTF_RGBA16f },
		{ TEXT("RT_RGBA8"),          true,  PF_B8G8R8A8,  RTF_RGBA8 },
		{ TEXT("RT_RGBA8_SRGB"),     true,  PF_B8G8R8A8,  RTF_RGBA8_SRGB },
		{ TEXT("RT_RGBA16f"),        true,  PF_FloatRGBA, RTF_RGBA16f },
		{ TEXT("RT_RGBA32f"),        true,  PF_A32B32G32R32F, RTF_RGBA32f },
	};

	static UTexture* MakeSource(const FTexCase& Case, const FPipelineImage& Image)
	{
		if (Case.bRenderTarget)
		{
			return MakeRenderTarget(Image, Case.Rtf);
		}
		return MakeTexture(Image, Case.Format);
	}

	/** Ticker-driven sequential reads of a 1080p render target: one warm-up read, then 20 measured reads. */
	struct FPerfState
	{
		TStrongObjectPtr<UObject> Target;
		FPipelineImage Source;
		FCapturePtr Current;
		int32 Index = -1;            // -1 = warm-up
		bool bNeedSample = false;
		TArray<double> LastMs;
		TArray<double> CallMs;
		FCapturePtr LastCapture;
		double Start = 0.0;
		double ReadStart = 0.0;
	};
}

using namespace PipelineReadbackSpecPrivate;

BEGIN_DEFINE_SPEC(FPipelineReadbackSpec, "Pipelines.Vision.Readback", EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FPipelineReadbackSpec)

void FPipelineReadbackSpec::Define()
{
	// ======================================================================================================================
	// Headless cases (valid with or without an RHI unless noted)
	// ======================================================================================================================

	LatentIt("NullTextureFailsInvalidArgument", FTimespan::FromSeconds(20), [this](const FDoneDelegate& Done)
	{
		// A null texture must be reported through OnRead (InvalidArgument), not crash, not return an invalid task.
		// Checked before the RHI check, so it holds under -nullrhi too.
		ResetImageReadStats();
		TArray<FCapturePtr> Caps = { StartRead(nullptr) };
		WaitForAll(this, Caps, 10.0, 0.2, [this, Caps, Done]()
		{
			VerifyFailure(*this, Caps[0], EPipelineError::InvalidArgument, TEXT("null texture"));
			Done.Execute();
		});
	});

	LatentIt("CubeTextureFailsInvalidArgument", FTimespan::FromSeconds(20), [this](const FDoneDelegate& Done)
	{
		// Only UTexture2D / UTextureRenderTarget2D are readable. A UTextureCube must fail as InvalidArgument and never be
		// dereferenced as a 2D texture (which would read garbage or crash).
		TStrongObjectPtr<UObject> Keep(NewObject<UTextureCube>(GetTransientPackage()));
		TArray<FCapturePtr> Caps = { StartRead(Cast<UTexture>(Keep.Get())) };
		WaitForAll(this, Caps, 10.0, 0.2, [this, Caps, Keep, Done]()
		{
			VerifyFailure(*this, Caps[0], EPipelineError::InvalidArgument, TEXT("cube texture"));
			Done.Execute();
		});
	});

	LatentIt("NotOnGameThreadFailsInvalidArgument", FTimespan::FromSeconds(30), [this](const FDoneDelegate& Done)
	{
		// ReadTexturePixelsAsync is game-thread only. Called from a worker thread with a real 2D texture it must fail with
		// InvalidArgument (not touch the UObject off the game thread). The texture is created here on the game thread.
		TStrongObjectPtr<UObject> Keep(UTexture2D::CreateTransient(4, 4, PF_B8G8R8A8));
		if (!Keep.IsValid())
		{
			AddError(TEXT("UTexture2D::CreateTransient returned null."));
			Done.Execute();
			return;
		}
		TSharedRef<FCapturePtr, ESPMode::ThreadSafe> Slot = MakeShared<FCapturePtr, ESPMode::ThreadSafe>();
		TSharedRef<std::atomic<bool>, ESPMode::ThreadSafe> Started = MakeShared<std::atomic<bool>, ESPMode::ThreadSafe>(false);
		UTexture* Tex = Cast<UTexture>(Keep.Get());
		Async(EAsyncExecution::Thread, [Slot, Started, Tex]()
		{
			*Slot = StartRead(Tex);
			Started->store(true);
		});
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("PipelineReadbackSpecWorkerCall"), 0.f, [this, Slot, Started, Keep, Done, Start](float) -> bool
		{
			if (!Started->load())
			{
				if (FPlatformTime::Seconds() - Start > 10.0)
				{
					AddError(TEXT("The worker-thread call to ReadTexturePixelsAsync never returned (it must not block or wait for the game thread)."));
					Done.Execute();
					return false;
				}
				return true;
			}
			TArray<FCapturePtr> Caps = { *Slot };
			WaitForAll(this, Caps, 10.0, 0.2, [this, Caps, Keep, Done]()
			{
				VerifyFailure(*this, Caps[0], EPipelineError::InvalidArgument, TEXT("call from a worker thread"));
				Done.Execute();
			});
			return false;
		});
	});

	LatentIt("NullRhiTextureFailsPreprocessFailedWithC28Message", FTimespan::FromSeconds(30), [this](const FDoneDelegate& Done)
	{
		// Under -nullrhi (GUsingNullRHI or no initialized RHI) every 2D texture and render target must fail with
		// PreprocessFailed and the exact C28 message that tells the user to pass an FPipelineImage instead.
		if (!(GUsingNullRHI || !GIsRHIInitialized))
		{
			AddInfo(TEXT("A real RHI is present: the nullrhi error cases do not apply, skipped (run with -nullrhi to cover them)"));
			Done.Execute();
			return;
		}
		ResetImageReadStats();
		TStrongObjectPtr<UObject> Tex(UTexture2D::CreateTransient(8, 8, PF_B8G8R8A8));
		UTextureRenderTarget2D* RtRaw = NewObject<UTextureRenderTarget2D>(GetTransientPackage());
		RtRaw->RenderTargetFormat = RTF_RGBA8;
		RtRaw->InitAutoFormat(8, 8);
		TStrongObjectPtr<UObject> Rt(RtRaw);
		if (!Tex.IsValid())
		{
			AddError(TEXT("UTexture2D::CreateTransient returned null under nullrhi."));
			Done.Execute();
			return;
		}
		TArray<FCapturePtr> Caps = { StartRead(Cast<UTexture>(Tex.Get())), StartRead(RtRaw) };
		WaitForAll(this, Caps, 10.0, 0.3, [this, Caps, Tex, Rt, Done]()
		{
			const TCHAR* Labels[2] = { TEXT("UTexture2D under nullrhi"), TEXT("UTextureRenderTarget2D under nullrhi") };
			for (int32 i = 0; i < 2; ++i)
			{
				VerifyFailure(*this, Caps[i], EPipelineError::PreprocessFailed, Labels[i]);
				FScopeLock L(&Caps[i]->Lock);
				TestEqual(*FString::Printf(TEXT("%s: exact C28 message"), Labels[i]), Caps[i]->Result.Message, FString(NoRhiMessage));
			}
			const FPipelineImageReadStats S = GetImageReadStats();
			TestEqual(TEXT("stats: NumFailed counts both nullrhi reads"), S.NumFailed, (int64)2);
			TestEqual(TEXT("stats: NumSucceeded stays 0"), S.NumSucceeded, (int64)0);
			Done.Execute();
		});
	});

	LatentIt("FailuresAreCountedInStats", FTimespan::FromSeconds(30), [this](const FDoneDelegate& Done)
	{
		// null + cube are failures: after Reset, three failing reads must show NumFailed == 3, NumSucceeded == 0 and
		// NumStarted >= NumFailed (a failure cannot happen without being counted as started).
		ResetImageReadStats();
		const FPipelineImageReadStats Zero = GetImageReadStats();
		TestEqual(TEXT("ResetImageReadStats zeroes NumStarted"), Zero.NumStarted, (int64)0);
		TestEqual(TEXT("ResetImageReadStats zeroes NumFailed"), Zero.NumFailed, (int64)0);
		TestEqual(TEXT("ResetImageReadStats zeroes NumSucceeded"), Zero.NumSucceeded, (int64)0);
		TStrongObjectPtr<UObject> Cube(NewObject<UTextureCube>(GetTransientPackage()));
		TArray<FCapturePtr> Caps = { StartRead(nullptr), StartRead(Cast<UTexture>(Cube.Get())), StartRead(nullptr) };
		WaitForAll(this, Caps, 10.0, 0.3, [this, Caps, Cube, Done]()
		{
			const FPipelineImageReadStats S = GetImageReadStats();
			TestEqual(TEXT("NumFailed after 3 failing reads"), S.NumFailed, (int64)3);
			TestEqual(TEXT("NumSucceeded after 3 failing reads"), S.NumSucceeded, (int64)0);
			TestTrue(TEXT("NumStarted >= NumFailed"), S.NumStarted >= S.NumFailed);
			Done.Execute();
		});
	});

	LatentIt("ManyCallsEachCallbackExactlyOnceOffGameThread", FTimespan::FromSeconds(60), [this](const FDoneDelegate& Done)
	{
		// 50 immediate failures queued in one game-thread frame. Each OnRead must run exactly once (never twice, never
		// zero times), on a background thread, never on the stack of the call (this catches an implementation that invokes
		// the callback synchronously on the fast failure path). Waits 0.5 s afterwards so a late duplicate would be seen.
		TArray<FCapturePtr> Caps;
		for (int32 i = 0; i < 50; ++i)
		{
			Caps.Add(StartRead(nullptr));
		}
		WaitForAll(this, Caps, 20.0, 0.5, [this, Caps, Done]()
		{
			for (int32 i = 0; i < Caps.Num(); ++i)
			{
				VerifyFailure(*this, Caps[i], EPipelineError::InvalidArgument, FString::Printf(TEXT("call %d"), i));
			}
			Done.Execute();
		});
	});

	LatentIt("ReturnedTaskCompletesOnlyAfterCallbackReturned", FTimespan::FromSeconds(30), [this](const FDoneDelegate& Done)
	{
		// The callback sleeps 250 ms. The returned task must not report completed while OnRead is still running (a task
		// that completes when the result is ready, not when OnRead returned, would be caught by the poll in WaitForAll).
		FCapturePtr Cap = StartRead(nullptr, 250);
		TArray<FCapturePtr> Caps = { Cap };
		WaitForAll(this, Caps, 15.0, 0.2, [this, Cap, Done]()
		{
			VerifyFailure(*this, Cap, EPipelineError::InvalidArgument, TEXT("slow callback"));
			TestTrue(TEXT("callback finished before the test observed the task complete"), Cap->bCallbackReturned.load());
			Done.Execute();
		});
	});

	LatentIt("ReturnedTaskWorksAsPrerequisite", FTimespan::FromSeconds(30), [this](const FDoneDelegate& Done)
	{
		// FPipelineStages::Prerequisite use: a task launched with the returned task as prerequisite must run only after
		// OnRead returned (callback sleeps 200 ms to make a too-early continuation visible).
		FCapturePtr Cap = StartRead(nullptr, 200);
		UE::Tasks::FTask Cont;
		{
			FScopeLock L(&Cap->Lock);
			Cont = UE::Tasks::Launch(TEXT("PipelineReadbackSpec.Continuation"), [Cap]()
			{
				Cap->bContinuationSawReturned = Cap->bCallbackReturned.load();
				Cap->bContinuationRan = true;
			}, UE::Tasks::Prerequisites(Cap->Task));
		}
		TArray<FCapturePtr> Caps = { Cap };
		WaitForAll(this, Caps, 15.0, 0.1, [this, Cap, Cont, Done]()
		{
			const double Start = FPlatformTime::Seconds();
			// Cont is tiny; it was released by the task finishing. Poll for it without blocking the game thread.
			FTSTicker::GetCoreTicker().AddTicker(TEXT("PipelineReadbackSpecCont"), 0.f, [this, Cap, Cont, Done, Start](float) -> bool
			{
				if (!Cont.IsCompleted())
				{
					if (FPlatformTime::Seconds() - Start > 10.0)
					{
						AddError(TEXT("The continuation task with the returned task as prerequisite never ran."));
						Done.Execute();
						return false;
					}
					return true;
				}
				TestTrue(TEXT("continuation ran"), Cap->bContinuationRan.load());
				TestTrue(TEXT("continuation ran only after OnRead returned"), Cap->bContinuationSawReturned.load());
				Done.Execute();
				return false;
			});
		});
	});

	// ======================================================================================================================
	// RHI cases
	// ======================================================================================================================
	Describe("RHI", [this]()
	{
		// ---- exact bytes for each format ------------------------------------------------------------------------------
		for (const FTexCase& Case : GTexCases)
		{
			LatentIt(FString::Printf(TEXT("%s_ExactRgbFromGoldens"), Case.Name), FTimespan::FromSeconds(60), [this, &Case](const FDoneDelegate& Done)
			{
				// The pixels read back from the GPU must equal the source bytes exactly (RGB) for both golden images (s02
				// 300x200: pitch != width*4 padding is likely, s04 37x53: odd sizes). Catches ignoring the Lock row pitch (skewed
				// rows), swapped R/B, an off-by-one in the float -> 8 bit rounding (round(v*255), not truncation),
				// and a wrong image size (the RHI mip-0 extent).
				if (!RequireRhi(*this))
				{
					Done.Execute();
					return;
				}
				const TCHAR* Ids[2] = { TEXT("s02"), TEXT("s04") };
				FKeepAlive Keep = MakeShared<TArray<TStrongObjectPtr<UObject>>>();
				TArray<FCapturePtr> Caps;
				TArray<TSharedPtr<FPipelineImage>> Sources;
				ResetImageReadStats();
				for (const TCHAR* Id : Ids)
				{
					TSharedPtr<FPipelineImage> Src = GetGoldenImage(*this, Id);
					if (!Src.IsValid())
					{
						Done.Execute();
						return;
					}
					UTexture* Tex = MakeSource(Case, *Src);
					if (!Tex)
					{
						AddError(FString::Printf(TEXT("Could not create the %s source for golden %s."), Case.Name, Id));
						Done.Execute();
						return;
					}
					Keep->Add(TStrongObjectPtr<UObject>(Tex));
					Sources.Add(Src);
					Caps.Add(StartRead(Tex));
				}
				WaitForAll(this, Caps, 30.0, 0.2, [this, Caps, Sources, Keep, Done, &Case]()
				{
					const TCHAR* Names[2] = { TEXT("s02"), TEXT("s04") };
					for (int32 i = 0; i < Caps.Num(); ++i)
					{
						VerifyExact(*this, Caps[i], *Sources[i], FString::Printf(TEXT("%s %s"), Case.Name, Names[i]));
					}
					const FPipelineImageReadStats S = GetImageReadStats();
					TestEqual(TEXT("stats: NumSucceeded"), S.NumSucceeded, (int64)Caps.Num());
					TestEqual(TEXT("stats: NumFailed"), S.NumFailed, (int64)0);
					Done.Execute();
				});
			});
		}

		LatentIt("GrayRenderTargetReplicatesToRgb", FTimespan::FromSeconds(30), [this](const FDoneDelegate& Done)
		{
			// RTF_R8 is stored as PF_G8: gray must come back as R = G = B = value. Source pixels are gray (R=G=B) so it does
			// not matter which channel the helper packs. Catches a G8 read that fills only R (or only G).
			if (!RequireRhi(*this))
			{
				Done.Execute();
				return;
			}
			FPipelineImage Src;
			Src.Width = 17;
			Src.Height = 11;
			Src.Pixels.SetNumUninitialized(17 * 11);
			for (int32 i = 0; i < Src.Pixels.Num(); ++i)
			{
				const uint8 V = (uint8)((i * 37 + 11) & 0xFF);
				Src.Pixels[i] = FColor(V, V, V, 255);
			}
			UTexture* Tex = MakeRenderTarget(Src, RTF_R8);
			if (!Tex)
			{
				AddError(TEXT("MakeRenderTarget(RTF_R8) returned null with an RHI present."));
				Done.Execute();
				return;
			}
			FKeepAlive Keep = MakeShared<TArray<TStrongObjectPtr<UObject>>>();
			Keep->Add(TStrongObjectPtr<UObject>(Tex));
			TArray<FCapturePtr> Caps = { StartRead(Tex) };
			WaitForAll(this, Caps, 20.0, 0.2, [this, Caps, Src, Keep, Done]()
			{
				VerifyExact(*this, Caps[0], Src, TEXT("RTF_R8 gray"));
				Done.Execute();
			});
		});

		// ---- 1080p game-thread cost -----------------------------------------------------------------------------------
		const TCHAR* const PerfNames[2] = { TEXT("Perf1080p_RT_RGBA8"), TEXT("Perf1080p_RT_RGBA16f") };
		const ETextureRenderTargetFormat PerfFormats[2] = { RTF_RGBA8, RTF_RGBA16f };
		for (int32 PerfIdx = 0; PerfIdx < 2; ++PerfIdx)
		{
			const FString PerfName = PerfNames[PerfIdx];
			const ETextureRenderTargetFormat PerfFormat = PerfFormats[PerfIdx];
			LatentIt(*PerfName, FTimespan::FromSeconds(240), [this, PerfName, PerfFormat](const FDoneDelegate& Done)
			{
				// 1920x1080 render target, 20 reads one after another (after one warm-up read that is not counted, because the
				// first call pays one-time costs such as ticker registration). Gates: MaxGameThreadMsPerRead < 1 ms (the
				// PLAN M4 acceptance: no game-thread wait) and every ReadTexturePixelsAsync call itself < 0.5 ms. A read that
				// waits for the GPU on the game thread (FlushRenderingCommands, Lock on the game thread) costs several ms here.
				if (!RequireRhi(*this))
				{
					Done.Execute();
					return;
				}
				TSharedRef<FPerfState> St = MakeShared<FPerfState>();
				St->Source = MakeNoise(1920, 1080, 4242);
				UTexture* Tex = MakeRenderTarget(St->Source, PerfFormat);
				if (!Tex)
				{
					AddError(TEXT("MakeRenderTarget returned null for the 1080p target with an RHI present."));
					Done.Execute();
					return;
				}
				St->Target.Reset(Tex);
				St->Start = FPlatformTime::Seconds();
				FTSTicker::GetCoreTicker().AddTicker(TEXT("PipelineReadbackSpecPerf"), 0.f, [this, St, PerfName, Done, Tex](float) -> bool
				{
					const double Now = FPlatformTime::Seconds();
					if (Now - St->Start > 200.0)
					{
						AddError(FString::Printf(TEXT("%s: timed out after 200 s at read index %d."), *PerfName, St->Index));
						Done.Execute();
						return false;
					}
					if (St->bNeedSample)
					{
						St->bNeedSample = false;
						if (St->Index >= 0)
						{
							St->LastMs.Add(GetImageReadStats().LastGameThreadMsPerRead);
						}
						else
						{
							ResetImageReadStats(); // warm-up done: measure only the 20 reads
						}
						++St->Index;
						if (St->Index >= 20)
						{
							const FPipelineImageReadStats S = GetImageReadStats();
							TArray<double> Sorted = St->LastMs;
							Sorted.Sort();
							const double Median = Sorted.Num() ? Sorted[Sorted.Num() / 2] : 0.0;
							double MaxCall = 0.0;
							for (const double C : St->CallMs)
							{
								MaxCall = FMath::Max(MaxCall, C);
							}
							AddInfo(FString::Printf(TEXT("%s: game-thread ms per read: max %.3f, median %.3f; max ReadbackWaitMs %.2f; max call %.3f ms (measured by the test)"),
								*PerfName, S.MaxGameThreadMsPerRead, Median, S.MaxReadbackWaitMs, MaxCall));
							TestTrue(*FString::Printf(TEXT("%s: MaxGameThreadMsPerRead %.3f ms must be < 1 ms"), *PerfName, S.MaxGameThreadMsPerRead), S.MaxGameThreadMsPerRead < 1.0);
							TestTrue(*FString::Printf(TEXT("%s: the ReadTexturePixelsAsync call itself, worst %.3f ms, must be < 0.5 ms"), *PerfName, MaxCall), MaxCall < 0.5);
							TestEqual(*FString::Printf(TEXT("%s: NumStarted"), *PerfName), S.NumStarted, (int64)20);
							TestEqual(*FString::Printf(TEXT("%s: NumSucceeded"), *PerfName), S.NumSucceeded, (int64)20);
							TestEqual(*FString::Printf(TEXT("%s: NumFailed"), *PerfName), S.NumFailed, (int64)0);
							if (St->LastCapture.IsValid())
							{
								VerifyExact(*this, St->LastCapture, St->Source, PerfName + TEXT(" last read"));
							}
							Done.Execute();
							return false;
						}
					}
					if (!St->Current.IsValid())
					{
						St->Current = StartRead(Tex);
						if (St->Index >= 0)
						{
							St->CallMs.Add(St->Current->CallMs);
						}
						return true;
					}
					St->Current->Poll();
					if (!St->Current->IsFinished())
					{
						return true;
					}
					St->LastCapture = St->Current;
					St->Current.Reset();
					St->bNeedSample = true; // sample the stats one tick later, after the reader had time to publish them
					return true;
				});
			});
		}

		// ---- concurrency ----------------------------------------------------------------------------------------------
		LatentIt("EightReadsInFlightAllCorrect", FTimespan::FromSeconds(60), [this](const FDoneDelegate& Done)
		{
			// 8 different textures (mixed textures / render targets / formats / sizes, distinct noise) are read in the same
			// game-thread frame. Each result must equal its own source: catches state shared between reads (a single static
			// readback object, results delivered to the wrong callback, a ticker that finishes only the first read).
			if (!RequireRhi(*this))
			{
				Done.Execute();
				return;
			}
			FKeepAlive Keep = MakeShared<TArray<TStrongObjectPtr<UObject>>>();
			TArray<FCapturePtr> Caps;
			TSharedRef<TArray<FPipelineImage>> Sources = MakeShared<TArray<FPipelineImage>>();
			Sources->Reserve(8);
			ResetImageReadStats();
			for (int32 i = 0; i < 8; ++i)
			{
				Sources->Add(MakeNoise(16 + i * 13, 9 + i * 7, 100 + i));
			}
			for (int32 i = 0; i < 8; ++i)
			{
				UTexture* Tex = MakeSource(GTexCases[i % UE_ARRAY_COUNT(GTexCases)], (*Sources)[i]);
				if (!Tex)
				{
					AddError(FString::Printf(TEXT("Could not create source texture %d."), i));
					Done.Execute();
					return;
				}
				Keep->Add(TStrongObjectPtr<UObject>(Tex));
				Caps.Add(StartRead(Tex));
			}
			WaitForAll(this, Caps, 30.0, 0.3, [this, Caps, Sources, Keep, Done]()
			{
				for (int32 i = 0; i < Caps.Num(); ++i)
				{
					VerifyExact(*this, Caps[i], (*Sources)[i], FString::Printf(TEXT("concurrent read %d (%s)"), i, GTexCases[i % UE_ARRAY_COUNT(GTexCases)].Name));
				}
				const FPipelineImageReadStats S = GetImageReadStats();
				TestEqual(TEXT("stats: NumStarted"), S.NumStarted, (int64)8);
				TestEqual(TEXT("stats: NumSucceeded"), S.NumSucceeded, (int64)8);
				Done.Execute();
			});
		});

		LatentIt("SameTextureReadTwiceBothCorrect", FTimespan::FromSeconds(30), [this](const FDoneDelegate& Done)
		{
			// Two reads of the same texture in one frame: both callbacks fire once with the same pixels. Catches per-texture
			// state (a "read already pending" cache) that would drop or merge the second read.
			if (!RequireRhi(*this))
			{
				Done.Execute();
				return;
			}
			const FPipelineImage Src = MakeNoise(64, 48, 7);
			UTexture* Tex = MakeRenderTarget(Src, RTF_RGBA8);
			if (!Tex)
			{
				AddError(TEXT("MakeRenderTarget returned null."));
				Done.Execute();
				return;
			}
			FKeepAlive Keep = MakeShared<TArray<TStrongObjectPtr<UObject>>>();
			Keep->Add(TStrongObjectPtr<UObject>(Tex));
			TArray<FCapturePtr> Caps = { StartRead(Tex), StartRead(Tex) };
			WaitForAll(this, Caps, 20.0, 0.3, [this, Caps, Src, Keep, Done]()
			{
				VerifyExact(*this, Caps[0], Src, TEXT("first read of the shared texture"));
				VerifyExact(*this, Caps[1], Src, TEXT("second read of the shared texture"));
				Done.Execute();
			});
		});

		// ---- lifetime -------------------------------------------------------------------------------------------------
		LatentIt("TextureGarbageCollectedRightAfterCall", FTimespan::FromSeconds(60), [this](const FDoneDelegate& Done)
		{
			// Nothing references the texture after the call: it is marked garbage and a full GC runs immediately, before the
			// render command / readback progressed. The read must still complete once (exact pixels, or a clean error),
			// never crash and never hang. A reader that dereferences the UObject after the call would crash or read freed memory.
			if (!RequireRhi(*this))
			{
				Done.Execute();
				return;
			}
			const FPipelineImage Src = MakeNoise(120, 80, 21);
			UTexture2D* Tex = MakeTexture(Src, PF_B8G8R8A8);
			if (!Tex)
			{
				AddError(TEXT("MakeTexture returned null."));
				Done.Execute();
				return;
			}
			FCapturePtr Cap = StartRead(Tex);
			Tex->MarkAsGarbage();
			Tex = nullptr;
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			TArray<FCapturePtr> Caps = { Cap };
			WaitForAll(this, Caps, 30.0, 0.2, [this, Cap, Src, Done]()
			{
				bool bOk = false;
				{
					FScopeLock L(&Cap->Lock);
					bOk = Cap->Result.bSuccess;
				}
				if (bOk)
				{
					VerifyExact(*this, Cap, Src, TEXT("texture destroyed after the call (read succeeded)"));
				}
				else
				{
					VerifyLifecycleImpl(*this, Cap, TEXT("texture destroyed after the call (clean error)"));
					FScopeLock L(&Cap->Lock);
					TestTrue(TEXT("a failed read after GC has an error code and a message"), Cap->Result.Error != EPipelineError::None && !Cap->Result.Message.IsEmpty());
					AddInfo(FString::Printf(TEXT("read after GC ended as a clean error: %s"), *Cap->Result.Message));
				}
				Done.Execute();
			});
		});

		LatentIt("RenderTargetGarbageCollectedRightAfterCall", FTimespan::FromSeconds(60), [this](const FDoneDelegate& Done)
		{
			// Same for a render target (its resource is released on the render thread when the UObject dies; the reader must
			// already hold the RHI texture reference).
			if (!RequireRhi(*this))
			{
				Done.Execute();
				return;
			}
			const FPipelineImage Src = MakeNoise(96, 64, 22);
			UTextureRenderTarget2D* Rt = MakeRenderTarget(Src, RTF_RGBA8);
			if (!Rt)
			{
				AddError(TEXT("MakeRenderTarget returned null."));
				Done.Execute();
				return;
			}
			FCapturePtr Cap = StartRead(Rt);
			Rt->MarkAsGarbage();
			Rt = nullptr;
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			TArray<FCapturePtr> Caps = { Cap };
			WaitForAll(this, Caps, 30.0, 0.2, [this, Cap, Src, Done]()
			{
				bool bOk = false;
				{
					FScopeLock L(&Cap->Lock);
					bOk = Cap->Result.bSuccess;
				}
				if (bOk)
				{
					VerifyExact(*this, Cap, Src, TEXT("render target destroyed after the call (read succeeded)"));
				}
				else
				{
					VerifyLifecycleImpl(*this, Cap, TEXT("render target destroyed after the call (clean error)"));
					FScopeLock L(&Cap->Lock);
					TestTrue(TEXT("a failed read after GC has an error code and a message"), Cap->Result.Error != EPipelineError::None && !Cap->Result.Message.IsEmpty());
					AddInfo(FString::Printf(TEXT("read after GC ended as a clean error: %s"), *Cap->Result.Message));
				}
				Done.Execute();
			});
		});

		// ---- rejected formats / inputs --------------------------------------------------------------------------------
		LatentIt("CompressedDxt1TextureFailsPreprocessFailedNamingFormat", FTimespan::FromSeconds(30), [this](const FDoneDelegate& Done)
		{
			// A transient PF_DXT1 4x4 texture is not in the readable-format table: PreprocessFailed whose message names
			// the format (DXT1) and the fix (CompressionSettings = UserInterface2D, or a render target). Catches a reader
			// that copies block-compressed bytes as if they were RGBA.
			if (!RequireRhi(*this))
			{
				Done.Execute();
				return;
			}
			ResetImageReadStats();
			UTexture2D* Tex = UTexture2D::CreateTransient(4, 4, PF_DXT1);
			if (!Tex)
			{
				AddError(TEXT("UTexture2D::CreateTransient(4,4,PF_DXT1) returned null."));
				Done.Execute();
				return;
			}
			Tex->NeverStream = true;
			Tex->UpdateResource();
			TStrongObjectPtr<UObject> Keep(Tex);
			TArray<FCapturePtr> Caps = { StartRead(Tex) };
			WaitForAll(this, Caps, 20.0, 0.3, [this, Caps, Keep, Done]()
			{
				VerifyFailure(*this, Caps[0], EPipelineError::PreprocessFailed, TEXT("PF_DXT1 texture"));
				FString Message;
				{
					FScopeLock L(&Caps[0]->Lock);
					Message = Caps[0]->Result.Message;
				}
				TestTrue(*FString::Printf(TEXT("message names the format DXT1: '%s'"), *Message), Message.Contains(TEXT("DXT1"), ESearchCase::IgnoreCase));
				TestTrue(*FString::Printf(TEXT("message names the fix (UserInterface2D): '%s'"), *Message), Message.Contains(TEXT("UserInterface2D")));
				const FPipelineImageReadStats S = GetImageReadStats();
				TestEqual(TEXT("stats: the rejected read counts as failed"), S.NumFailed, (int64)1);
				Done.Execute();
			});
		});

		LatentIt("NullAndCubeStillInvalidArgumentWithRhi", FTimespan::FromSeconds(30), [this](const FDoneDelegate& Done)
		{
			// With a real RHI the argument checks must still win over everything else: null and non-2D input are
			// InvalidArgument (not a timeout, not a crash while looking up a render resource).
			if (!RequireRhi(*this))
			{
				Done.Execute();
				return;
			}
			TStrongObjectPtr<UObject> Cube(NewObject<UTextureCube>(GetTransientPackage()));
			TArray<FCapturePtr> Caps = { StartRead(nullptr), StartRead(Cast<UTexture>(Cube.Get())) };
			WaitForAll(this, Caps, 10.0, 0.2, [this, Caps, Cube, Done]()
			{
				VerifyFailure(*this, Caps[0], EPipelineError::InvalidArgument, TEXT("null texture with RHI"));
				VerifyFailure(*this, Caps[1], EPipelineError::InvalidArgument, TEXT("cube texture with RHI"));
				Done.Execute();
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
