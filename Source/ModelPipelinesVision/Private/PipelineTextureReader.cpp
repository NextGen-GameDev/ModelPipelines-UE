// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
// Texture / render-target readback without blocking the game thread (docs/contracts.md C28).
//
// Flow per read:
//   game thread  ReadTexturePixelsAsync: validate, grab FTextureResource*, launch the delivery task (prerequisite: the
//                read's ReadyEvent), enqueue the render command that starts the GPU copy, register the poll ticker.
//   render       start: take Resource->TextureRHI, check the RHI desc, create FRHIGPUTextureReadback, EnqueueCopy (mip 0).
//   game thread  core ticker, every tick: enqueue one render command that polls every pending read; finish reads whose
//                copy completed / failed / timed out (stats, then trigger ReadyEvent).
//   render       poll: IsReady -> Lock -> copy rows (pitch from Lock) -> Unlock -> delete readback.
//   background   delivery task: convert bytes -> FPipelineImage, call OnRead. The delivery task is the returned task, so
//                it completes only after OnRead returned.

#include "PipelineTextureReader.h"

#include <atomic>
#include "Containers/Ticker.h"
#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/PlatformTime.h"
#include "Math/Float16.h"
#include "Misc/CoreDelegates.h"
#include "PixelFormat.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"
#include "RenderingThread.h"
#include "RHICommandList.h"
#include "RHIGlobals.h"
#include "RHIGPUReadback.h"
#include "RHIResources.h"
#include "TextureResource.h"
#include "PipelineVisionLog.h"

namespace UE::ModelPipelines::Vision::Private
{
	static const TCHAR* const NoRhiMessage = TEXT("texture input needs a GPU RHI (the editor runs with -nullrhi); pass an FPipelineImage instead");
	static constexpr double ReadTimeoutSeconds = 5.0;
	static constexpr double ShutdownWaitSeconds = 10.0;

	enum EReadPhase : int32
	{
		Phase_Pending = 0,
		/** Render thread copied the bytes. */
		Phase_Copied = 1,
		/** Render thread found an error (RtError / RtMessage). */
		Phase_RtFailed = 2,
		/** Game thread finished it (timed out or cancelled); the render thread only releases its objects. */
		Phase_GameFinished = 3,
	};

	/** One read. Shared by the game thread, render commands and the delivery task. */
	struct FReadState
	{
		// ---- set on the game thread before any other thread sees the state ----
		/** Used only by the first render command (render commands run in order: the resource is still alive there). */
		FTextureResource* Resource = nullptr;
		EPixelFormat Format = PF_Unknown;
		FPipelineImageReadCallback OnRead;
		UE::Tasks::FTaskEvent ReadyEvent{ TEXT("ModelPipelines.Vision.ReadReady") };
		double StartSeconds = 0.0;

		// ---- game thread only ----
		double GameThreadMs = 0.0;

		// ---- render thread only ----
		FRHIGPUTextureReadback* Readback = nullptr;
		FTextureRHIRef TextureRHI;

		// ---- written by the render thread before Phase leaves Pending ----
		int32 Width = 0;
		int32 Height = 0;
		TArray64<uint8> Bytes;
		double CopiedSeconds = 0.0;
		EPipelineError RtError = EPipelineError::None;
		FString RtMessage;

		std::atomic<int32> Phase{ Phase_Pending };

		// ---- written before ReadyEvent is triggered (or before the delivery task is launched), read by the delivery task ----
		FPipelineResult Final;
	};
	using FReadStatePtr = TSharedPtr<FReadState, ESPMode::ThreadSafe>;

	// ------------------------------------------------------------------------------------------------------------------
	// Globals
	// ------------------------------------------------------------------------------------------------------------------
	static FCriticalSection GStatsLock;
	static FPipelineImageReadStats GStats;

	/** Delivery tasks not known to be complete (waited for at shutdown so no module code runs after unload). */
	static FCriticalSection GTasksLock;
	static TArray<UE::Tasks::FTask> GOutstanding;

	// Game thread only.
	static TArray<FReadStatePtr> GPending;
	static FTSTicker::FDelegateHandle GTickerHandle;
	static bool GTickerRegistered = false;
	static FDelegateHandle GPreExitHandle;
	static std::atomic<bool> GShutDown{ false };

	static void RecordGameThreadMs(double Ms)
	{
		FScopeLock Lock(&GStatsLock);
		GStats.LastGameThreadMsPerRead = Ms;
		GStats.MaxGameThreadMsPerRead = FMath::Max(GStats.MaxGameThreadMsPerRead, Ms);
	}

	static void RecordWaitMs(double Ms)
	{
		FScopeLock Lock(&GStatsLock);
		GStats.MaxReadbackWaitMs = FMath::Max(GStats.MaxReadbackWaitMs, Ms);
	}

	static void TrackTask(const UE::Tasks::FTask& Task)
	{
		FScopeLock Lock(&GTasksLock);
		GOutstanding.RemoveAllSwap([](const UE::Tasks::FTask& T) { return T.IsCompleted(); });
		GOutstanding.Add(Task);
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Formats
	// ------------------------------------------------------------------------------------------------------------------
	/** Bytes per pixel of a readable format, 0 when the format is not readable. */
	static int32 ReadableBytesPerPixel(EPixelFormat Format)
	{
		switch (Format)
		{
		case PF_B8G8R8A8:       return 4;
		case PF_R8G8B8A8:       return 4;
		case PF_G8:             return 1;
		case PF_FloatRGBA:      return 8;
		case PF_A32B32G32R32F:  return 16;
		case PF_A2B10G10R10:    return 4;
		default:                return 0;
		}
	}

	static const TCHAR* FormatName(EPixelFormat Format)
	{
		return (Format >= 0 && Format < PF_MAX && GPixelFormats[Format].Name) ? GPixelFormats[Format].Name : TEXT("Unknown");
	}

	/** round(clamp(v, 0, 1) * 255), NaN -> 0. */
	static FORCEINLINE uint8 UnitFloatToByte(float V)
	{
		if (!(V > 0.0f))
		{
			return 0;   // also NaN
		}
		if (V >= 1.0f)
		{
			return 255;
		}
		return (uint8)FMath::FloorToInt32(V * 255.0f + 0.5f);
	}

	/** Tightly packed rows (Width * bytes per pixel) -> FPipelineImage. */
	static bool ConvertBytes(EPixelFormat Format, int32 Width, int32 Height, const TArray64<uint8>& Bytes, FPipelineImage& Out, FString& OutError)
	{
		const int32 Bpp = ReadableBytesPerPixel(Format);
		const int64 NumPixels = (int64)Width * Height;
		if (Bpp == 0 || Width <= 0 || Height <= 0 || Bytes.Num() != NumPixels * Bpp)
		{
			OutError = FString::Printf(TEXT("texture readback produced %lld bytes for a %dx%d %s image"), Bytes.Num(), Width, Height, FormatName(Format));
			return false;
		}
		Out.Width = Width;
		Out.Height = Height;
		Out.Pixels.SetNumUninitialized((int32)NumPixels);
		FColor* Dst = Out.Pixels.GetData();
		const uint8* Src = Bytes.GetData();
		switch (Format)
		{
		case PF_B8G8R8A8:
			for (int64 i = 0; i < NumPixels; ++i, Src += 4)
			{
				Dst[i] = FColor(Src[2], Src[1], Src[0], Src[3]);
			}
			break;
		case PF_R8G8B8A8:
			for (int64 i = 0; i < NumPixels; ++i, Src += 4)
			{
				Dst[i] = FColor(Src[0], Src[1], Src[2], Src[3]);
			}
			break;
		case PF_G8:
			for (int64 i = 0; i < NumPixels; ++i, ++Src)
			{
				Dst[i] = FColor(Src[0], Src[0], Src[0], 255);
			}
			break;
		case PF_FloatRGBA:
			for (int64 i = 0; i < NumPixels; ++i, Src += 8)
			{
				FFloat16 H[4];
				FMemory::Memcpy(H, Src, 8);
				Dst[i] = FColor(UnitFloatToByte(H[0].GetFloat()), UnitFloatToByte(H[1].GetFloat()), UnitFloatToByte(H[2].GetFloat()), UnitFloatToByte(H[3].GetFloat()));
			}
			break;
		case PF_A32B32G32R32F:
			for (int64 i = 0; i < NumPixels; ++i, Src += 16)
			{
				float F[4];
				FMemory::Memcpy(F, Src, 16);
				Dst[i] = FColor(UnitFloatToByte(F[0]), UnitFloatToByte(F[1]), UnitFloatToByte(F[2]), UnitFloatToByte(F[3]));
			}
			break;
		case PF_A2B10G10R10:
			for (int64 i = 0; i < NumPixels; ++i, Src += 4)
			{
				uint32 P;
				FMemory::Memcpy(&P, Src, 4);
				const uint32 R = P & 0x3FF;
				const uint32 G = (P >> 10) & 0x3FF;
				const uint32 B = (P >> 20) & 0x3FF;
				const uint32 A = (P >> 30) & 0x3;
				// round(v10 * 255 / 1023) in integers (no ties: 1023 is odd).
				Dst[i] = FColor((uint8)((R * 255 + 511) / 1023), (uint8)((G * 255 + 511) / 1023), (uint8)((B * 255 + 511) / 1023), (uint8)(A * 85));
			}
			break;
		default:
			OutError = FString::Printf(TEXT("texture pixel format %s is not readable"), FormatName(Format));
			return false;
		}
		return true;
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Delivery (background)
	// ------------------------------------------------------------------------------------------------------------------
	static void Deliver(FReadState& State)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Vision.Convert");
		FPipelineResult Result = State.Final;
		FPipelineImage Image;
		if (Result.bSuccess)
		{
			FString Error;
			if (!ConvertBytes(State.Format, State.Width, State.Height, State.Bytes, Image, Error))
			{
				Image = FPipelineImage();
				Result = FPipelineResult::Fail(EPipelineError::PreprocessFailed, Error);
			}
			// Only on success: after a timeout the render thread may still be writing Bytes (freed with the state).
			State.Bytes.Empty();
		}
		{
			FScopeLock Lock(&GStatsLock);
			if (Result.bSuccess)
			{
				++GStats.NumSucceeded;
			}
			else
			{
				++GStats.NumFailed;
			}
		}
		if (State.OnRead)
		{
			State.OnRead(Result, MoveTemp(Image));
		}
		State.OnRead = nullptr;
	}

	/** Launches the delivery task: it runs once ReadyEvent is triggered. bReadyNow: trigger it now (immediate failures;
	 *  an FTaskEvent must be triggered before it is destroyed). */
	static UE::Tasks::FTask LaunchDelivery(const FReadStatePtr& State, bool bReadyNow)
	{
		UE::Tasks::FTask Task = UE::Tasks::Launch(TEXT("ModelPipelines.Vision.ReadDeliver"), [State]() { Deliver(*State); },
			UE::Tasks::Prerequisites(State->ReadyEvent), UE::Tasks::ETaskPriority::Normal, UE::Tasks::EExtendedTaskPriority::None,
			UE::Tasks::ETaskFlags::DoNotRunInsideBusyWait);
		TrackTask(Task);
		if (bReadyNow)
		{
			State->ReadyEvent.Trigger();
		}
		return Task;
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Render thread
	// ------------------------------------------------------------------------------------------------------------------
	static void ReleaseRenderObjects(FReadState& State)
	{
		if (State.Readback)
		{
			delete State.Readback;
			State.Readback = nullptr;
		}
		State.TextureRHI.SafeRelease();
	}

	static void FailOnRenderThread(FReadState& State, EPipelineError Error, const FString& Message)
	{
		ReleaseRenderObjects(State);
		State.RtError = Error;
		State.RtMessage = Message;
		int32 Expected = Phase_Pending;
		State.Phase.compare_exchange_strong(Expected, Phase_RtFailed);
	}

	static void StartCopyOnRenderThread(FRHICommandListImmediate& RHICmdList, FReadState& State)
	{
		FTextureResource* Resource = State.Resource;
		State.Resource = nullptr;
		if (State.Phase.load() != Phase_Pending)
		{
			return;
		}
		FTextureRHIRef Texture = Resource ? Resource->TextureRHI : FTextureRHIRef();
		if (!Texture.IsValid())
		{
			FailOnRenderThread(State, EPipelineError::InvalidArgument, TEXT("texture has no RHI resource (not initialized, or released)"));
			return;
		}
		const FRHITextureDesc& Desc = Texture->GetDesc();
		if (Desc.Dimension != ETextureDimension::Texture2D)
		{
			FailOnRenderThread(State, EPipelineError::PreprocessFailed, TEXT("texture's RHI resource is not a 2D texture"));
			return;
		}
		if (Desc.NumSamples > 1)
		{
			FailOnRenderThread(State, EPipelineError::PreprocessFailed, TEXT("multisampled textures cannot be read back"));
			return;
		}
		if (Desc.Format != State.Format)
		{
			FailOnRenderThread(State, EPipelineError::PreprocessFailed, FString::Printf(TEXT("texture's RHI format %s differs from its pixel format %s"),
				FormatName(Desc.Format), FormatName(State.Format)));
			return;
		}
		const int32 Width = Desc.Extent.X;
		const int32 Height = Desc.Extent.Y;
		if (Width < 1 || Height < 1 || Width > FPipelineImage::MaxSide || Height > FPipelineImage::MaxSide)
		{
			FailOnRenderThread(State, EPipelineError::PreprocessFailed, FString::Printf(TEXT("texture size %dx%d is outside 1..%d"), Width, Height, FPipelineImage::MaxSide));
			return;
		}
		State.Width = Width;
		State.Height = Height;
		State.TextureRHI = Texture;
		State.Readback = new FRHIGPUTextureReadback(TEXT("ModelPipelines.Vision.Readback"));
		RHICmdList.Transition(FRHITransitionInfo(Texture.GetReference(), ERHIAccess::Unknown, ERHIAccess::CopySrc));
		State.Readback->EnqueueCopy(RHICmdList, Texture.GetReference());
		RHICmdList.Transition(FRHITransitionInfo(Texture.GetReference(), ERHIAccess::CopySrc, ERHIAccess::SRVMask));
	}

	static void PollOnRenderThread(const TArray<FReadStatePtr>& States)
	{
		for (const FReadStatePtr& StatePtr : States)
		{
			FReadState& State = *StatePtr;
			if (State.Phase.load() != Phase_Pending)
			{
				ReleaseRenderObjects(State);
				continue;
			}
			if (!State.Readback || !State.Readback->IsReady())
			{
				continue;
			}
			TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Vision.ReadbackRT");
			const int32 Bpp = ReadableBytesPerPixel(State.Format);
			int32 PitchPixels = 0;
			int32 BufferHeight = 0;
			const uint8* Src = static_cast<const uint8*>(State.Readback->Lock(PitchPixels, &BufferHeight));
			if (!Src)
			{
				FailOnRenderThread(State, EPipelineError::PreprocessFailed, TEXT("GPU readback returned no data"));
				continue;
			}
			if (PitchPixels < State.Width || (BufferHeight > 0 && BufferHeight < State.Height))
			{
				State.Readback->Unlock();
				FailOnRenderThread(State, EPipelineError::PreprocessFailed, FString::Printf(TEXT("GPU readback buffer %dx%d is smaller than the texture %dx%d"),
					PitchPixels, BufferHeight, State.Width, State.Height));
				continue;
			}
			const int64 RowBytes = (int64)State.Width * Bpp;
			const int64 PitchBytes = (int64)PitchPixels * Bpp;
			State.Bytes.SetNumUninitialized(RowBytes * State.Height);
			uint8* Dst = State.Bytes.GetData();
			for (int32 Y = 0; Y < State.Height; ++Y)
			{
				FMemory::Memcpy(Dst + Y * RowBytes, Src + Y * PitchBytes, RowBytes);
			}
			State.Readback->Unlock();
			ReleaseRenderObjects(State);
			State.CopiedSeconds = FPlatformTime::Seconds();
			int32 Expected = Phase_Pending;
			State.Phase.compare_exchange_strong(Expected, Phase_Copied);
		}
	}

	static void EnqueuePoll(TArray<FReadStatePtr>&& States)
	{
		ENQUEUE_RENDER_COMMAND(ModelPipelinesVisionReadPoll)([States = MoveTemp(States)](FRHICommandListImmediate& RHICmdList)
		{
			PollOnRenderThread(States);
		});
	}

	// ------------------------------------------------------------------------------------------------------------------
	// Game thread
	// ------------------------------------------------------------------------------------------------------------------
	static void CancelAll(const TCHAR* Why);

	static bool Tick(float)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Vision.ReadPoll");
		const double T0 = FPlatformTime::Seconds();
		TArray<FReadStatePtr> Finished;
		TArray<FReadStatePtr> StillPending;
		TArray<FReadStatePtr> ToRenderThread;
		for (const FReadStatePtr& S : GPending)
		{
			int32 Phase = S->Phase.load();
			if (Phase == Phase_Pending && T0 - S->StartSeconds > ReadTimeoutSeconds)
			{
				int32 Expected = Phase_Pending;
				if (S->Phase.compare_exchange_strong(Expected, Phase_GameFinished))
				{
					S->Final = FPipelineResult::Fail(EPipelineError::PreprocessFailed, TEXT("GPU readback timed out (no data after 5 s)"));
					Finished.Add(S);
					ToRenderThread.Add(S);   // the render thread releases the readback
					continue;
				}
				Phase = Expected;
			}
			if (Phase == Phase_Copied)
			{
				S->Final = FPipelineResult::Ok();
				Finished.Add(S);
			}
			else if (Phase == Phase_RtFailed)
			{
				S->Final = FPipelineResult::Fail(S->RtError, S->RtMessage);
				Finished.Add(S);
			}
			else
			{
				StillPending.Add(S);
				ToRenderThread.Add(S);
			}
		}
		if (ToRenderThread.Num() > 0)
		{
			EnqueuePoll(MoveTemp(ToRenderThread));
		}
		GPending = MoveTemp(StillPending);

		const int32 NumTouched = Finished.Num() + GPending.Num();
		const double ShareMs = NumTouched > 0 ? (FPlatformTime::Seconds() - T0) * 1000.0 / NumTouched : 0.0;
		for (const FReadStatePtr& S : GPending)
		{
			S->GameThreadMs += ShareMs;
		}
		for (const FReadStatePtr& S : Finished)
		{
			S->GameThreadMs += ShareMs;
			RecordGameThreadMs(S->GameThreadMs);
			if (S->Final.bSuccess)
			{
				RecordWaitMs((S->CopiedSeconds - S->StartSeconds) * 1000.0);
			}
			S->ReadyEvent.Trigger();
		}

		if (GPending.Num() == 0)
		{
			GTickerRegistered = false;
			GTickerHandle.Reset();
			return false;
		}
		return true;
	}

	static void EnsureTicker()
	{
		if (!GPreExitHandle.IsValid())
		{
			// Engine exit: cancel while the render thread and the RHI are still alive.
			GPreExitHandle = FCoreDelegates::OnEnginePreExit.AddStatic(&CancelAll, TEXT("engine exit with the read pending"));
		}
		if (!GTickerRegistered)
		{
			GTickerRegistered = true;
			GTickerHandle = FTSTicker::GetCoreTicker().AddTicker(TEXT("ModelPipelines.Vision.ReadPoll"), 0.0f, [](float DeltaTime) { return Tick(DeltaTime); });
		}
	}

	/** Game thread. Completes every pending read with Cancelled; waits for delivery tasks so no module code runs later. */
	static void CancelAll(const TCHAR* Why)
	{
		GShutDown = true;
		if (GTickerRegistered)
		{
			FTSTicker::GetCoreTicker().RemoveTicker(GTickerHandle);
			GTickerRegistered = false;
			GTickerHandle.Reset();
		}
		TArray<FReadStatePtr> Cancelled = MoveTemp(GPending);
		GPending.Reset();
		for (const FReadStatePtr& S : Cancelled)
		{
			S->Phase.store(Phase_GameFinished);
			S->Final = FPipelineResult::Fail(EPipelineError::Cancelled, FString::Printf(TEXT("texture read cancelled: %s"), Why));
		}
		if (Cancelled.Num() > 0)
		{
			if (GIsRHIInitialized && !GUsingNullRHI)
			{
				TArray<FReadStatePtr> ForRender = Cancelled;
				EnqueuePoll(MoveTemp(ForRender));   // Phase is GameFinished: only releases the readback objects
				FlushRenderingCommands();
			}
			else
			{
				// The RHI is gone: releasing GPU objects now is unsafe. Leak the states on purpose.
				for (const FReadStatePtr& S : Cancelled)
				{
					new FReadStatePtr(S);
				}
				UE_LOG(LogModelPipelinesVision, Warning, TEXT("%d texture reads were pending after the RHI shut down; their GPU objects are leaked."), Cancelled.Num());
			}
			for (const FReadStatePtr& S : Cancelled)
			{
				S->ReadyEvent.Trigger();
			}
		}
		TArray<UE::Tasks::FTask> Tasks;
		{
			FScopeLock Lock(&GTasksLock);
			Tasks = MoveTemp(GOutstanding);
			GOutstanding.Reset();
		}
		if (Tasks.Num() > 0 && !UE::Tasks::Wait(Tasks, FTimespan::FromSeconds(ShutdownWaitSeconds)))
		{
			UE_LOG(LogModelPipelinesVision, Warning, TEXT("Texture read callbacks still running after %.0f s at shutdown."), ShutdownWaitSeconds);
		}
	}

	void ShutdownTextureReader()
	{
		CancelAll(TEXT("module shutdown with the read pending"));
		if (GPreExitHandle.IsValid())
		{
			FCoreDelegates::OnEnginePreExit.Remove(GPreExitHandle);
			GPreExitHandle.Reset();
		}
	}
}

namespace UE::ModelPipelines::Vision
{
	using namespace UE::ModelPipelines::Vision::Private;

	UE::Tasks::FTask ReadTexturePixelsAsync(UTexture* Texture, FPipelineImageReadCallback OnRead)
	{
		TRACE_CPUPROFILER_EVENT_SCOPE_STR("ModelPipelines.Vision.ReadStart");
		const double T0 = FPlatformTime::Seconds();
		const bool bGameThread = IsInGameThread();

		FReadStatePtr State = MakeShared<FReadState, ESPMode::ThreadSafe>();
		State->OnRead = MoveTemp(OnRead);
		State->StartSeconds = T0;
		{
			FScopeLock Lock(&GStatsLock);
			++GStats.NumStarted;
		}

		auto FailNow = [&State, T0, bGameThread](EPipelineError Error, const FString& Message) -> UE::Tasks::FTask
		{
			State->Final = FPipelineResult::Fail(Error, Message);
			UE::Tasks::FTask Task = LaunchDelivery(State, true);
			if (bGameThread)
			{
				RecordGameThreadMs((FPlatformTime::Seconds() - T0) * 1000.0);
			}
			return Task;
		};

		if (!Texture)
		{
			return FailNow(EPipelineError::InvalidArgument, TEXT("ReadTexturePixelsAsync: Texture is null"));
		}
		if (!bGameThread)
		{
			return FailNow(EPipelineError::InvalidArgument, TEXT("ReadTexturePixelsAsync must be called on the game thread"));
		}
		if (GShutDown.load())
		{
			return FailNow(EPipelineError::Cancelled, TEXT("ReadTexturePixelsAsync: the ModelPipelinesVision module is shut down"));
		}
		UTexture2D* Texture2D = Cast<UTexture2D>(Texture);
		UTextureRenderTarget2D* RenderTarget = Cast<UTextureRenderTarget2D>(Texture);
		if (!Texture2D && !RenderTarget)
		{
			return FailNow(EPipelineError::InvalidArgument, FString::Printf(TEXT("ReadTexturePixelsAsync: '%s' is a %s; only UTexture2D and UTextureRenderTarget2D can be read"),
				*Texture->GetName(), *Texture->GetClass()->GetName()));
		}
		if (GUsingNullRHI || !GIsRHIInitialized)
		{
			return FailNow(EPipelineError::PreprocessFailed, NoRhiMessage);
		}
		if (Texture->IsCurrentlyVirtualTextured())
		{
			return FailNow(EPipelineError::PreprocessFailed, FString::Printf(TEXT("texture '%s' is a virtual texture and cannot be read back; disable Virtual Texture Streaming on it, or use a render target"),
				*Texture->GetName()));
		}
		const EPixelFormat Format = Texture2D ? Texture2D->GetPixelFormat(0) : RenderTarget->GetFormat();
		if (ReadableBytesPerPixel(Format) == 0)
		{
			return FailNow(EPipelineError::PreprocessFailed, FString::Printf(
				TEXT("texture '%s' has pixel format %s, which cannot be read back (compressed or unsupported); set CompressionSettings = UserInterface2D on the texture, or use a render target (RGBA8, RGBA16f, RGBA32f, R8)"),
				*Texture->GetName(), FormatName(Format)));
		}
		FTextureResource* Resource = Texture->GetResource();
		if (!Resource)
		{
			return FailNow(EPipelineError::InvalidArgument, FString::Printf(TEXT("texture '%s' has no render resource"), *Texture->GetName()));
		}

		State->Resource = Resource;
		State->Format = Format;
		UE::Tasks::FTask Task = LaunchDelivery(State, false);
		ENQUEUE_RENDER_COMMAND(ModelPipelinesVisionReadStart)([State](FRHICommandListImmediate& RHICmdList)
		{
			StartCopyOnRenderThread(RHICmdList, *State);
		});
		GPending.Add(State);
		EnsureTicker();
		State->GameThreadMs += (FPlatformTime::Seconds() - T0) * 1000.0;
		return Task;
	}

	FPipelineImageReadStats GetImageReadStats()
	{
		FScopeLock Lock(&GStatsLock);
		return GStats;
	}

	void ResetImageReadStats()
	{
		FScopeLock Lock(&GStatsLock);
		GStats = FPipelineImageReadStats();
	}
}
