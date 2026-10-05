// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "NNEModelData.h"
#include "AssetRegistry/AssetData.h"
#include "AssetRegistry/IAssetRegistry.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Containers/Array.h"
#include "Containers/Ticker.h"
#include "Containers/UnrealString.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineModelPackage.h"
#include "PipelineModelImporter.h"
#include "Engine/Texture2D.h"
#include "Math/RandomStream.h"
#include "TextureCompiler.h"
#include "Misc/App.h"
#include "PipelineMcpJob.h"
#include "PipelineMcpRun.h"
#include "PipelineMcpLibrary.h"
#include "PipelineMcpTestReceiver.h"

namespace PipelineMcpJobSpecPrivate
{
	static const TCHAR* DefaultTexturePath = TEXT("/Engine/EngineResources/DefaultTexture.DefaultTexture");
	static const TCHAR* DefaultTexturePackage = TEXT("/Engine/EngineResources/DefaultTexture");

	static FString GetDevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		if (!Env.IsEmpty())
		{
			return Env;
		}
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev"));
	}

	static bool IsGpuRequired()
	{
		return FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_GPU")) == TEXT("1");
	}

	static FString GetStr(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
	{
		FString S;
		if (O.IsValid())
		{
			O->TryGetStringField(Key, S);
		}
		return S;
	}

	static bool GetNum(const TSharedPtr<FJsonObject>& O, const TCHAR* Key, double& Out)
	{
		return O.IsValid() && O->TryGetNumberField(Key, Out);
	}

	static TSharedPtr<FJsonObject> GetObj(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
	{
		const TSharedPtr<FJsonObject>* P = nullptr;
		if (O.IsValid() && O->TryGetObjectField(Key, P) && P)
		{
			return *P;
		}
		return nullptr;
	}

	static TArray<TSharedPtr<FJsonValue>> GetArr(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
	{
		const TArray<TSharedPtr<FJsonValue>>* P = nullptr;
		if (O.IsValid() && O->TryGetArrayField(Key, P) && P)
		{
			return *P;
		}
		return TArray<TSharedPtr<FJsonValue>>();
	}

	static FString ObjPath(const FString& Folder, const FString& Name)
	{
		return Folder / Name + TEXT(".") + Name;
	}

	static TSharedPtr<FJsonObject> FindDevice(const TSharedPtr<FJsonObject>& J, const TCHAR* Device)
	{
		for (const TSharedPtr<FJsonValue>& V : GetArr(J, TEXT("devices")))
		{
			const TSharedPtr<FJsonObject> E = V.IsValid() ? V->AsObject() : nullptr;
			if (GetStr(E, TEXT("device")) == Device)
			{
				return E;
			}
		}
		return nullptr;
	}

	static void DiscardOne(UObject* Obj)
	{
		if (!Obj || !IsValid(Obj))
		{
			return;
		}
		UPackage* Old = Obj->GetOutermost();
		UPackage* Transient = GetTransientPackage();
		if (Old != Transient)
		{
			FAssetRegistryModule::AssetDeleted(Obj);
			Obj->Rename(*MakeUniqueObjectName(Transient, Obj->GetClass(), Obj->GetFName()).ToString(), Transient,
				REN_DontCreateRedirectors | REN_NonTransactional);
		}
		Obj->ClearFlags(RF_Public | RF_Standalone);
		Obj->MarkAsGarbage();
		if (Old && Old != Transient && IsValid(Old))
		{
			Old->SetDirtyFlag(false);
			Old->ClearFlags(RF_Public | RF_Standalone);
			Old->MarkAsGarbage();
		}
	}

	static bool RunOffGameThread(TFunction<FString()> F, FString& Out)
	{
		FEvent* Done = FPlatformProcess::GetSynchEventFromPool(true);
		TSharedRef<FString, ESPMode::ThreadSafe> Result = MakeShared<FString, ESPMode::ThreadSafe>();
		AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [F, Result, Done]()
		{
			*Result = F();
			Done->Trigger();
		});
		const bool bOk = Done->Wait(FTimespan::FromSeconds(30.0));
		if (bOk)
		{
			FPlatformProcess::ReturnSynchEventToPool(Done); // on timeout the event leaks on purpose: the task may still trigger it
			Out = *Result;
		}
		return bOk;
	}

	/** Polls Pred from the core ticker (game thread, never blocks) until it is true or TimeoutSec passed; then OnEnd(bTimedOut). */
	static void PollUntil(TFunction<bool()> Pred, double TimeoutSec, TFunction<void(bool)> OnEnd)
	{
		const double Start = FPlatformTime::Seconds();
		FTSTicker::GetCoreTicker().AddTicker(TEXT("PipelineMcpJobSpecPoll"), 0.0f, [Pred, Start, TimeoutSec, OnEnd](float)
		{
			if (Pred())
			{
				OnEnd(false);
				return false;
			}
			if (FPlatformTime::Seconds() - Start > TimeoutSec)
			{
				OnEnd(true);
				return false;
			}
			return true;
		});
	}

	/** One job under observation: an OnDone receiver bound right after the creating call returned. */
	struct FWatch
	{
		TStrongObjectPtr<UPipelineMcpTestReceiver> Recv;
		TWeakObjectPtr<UPipelineMcpJob> Job;
		bool bNull = true;
		bool bDoneAtCreate = true;
		bool bSucceededAtCreate = true;
		FString JsonAtCreate = TEXT("<unset>");
		FString Label;

		static TSharedRef<FWatch> Make(UPipelineMcpJob* J, const FString& InLabel)
		{
			TSharedRef<FWatch> W = MakeShared<FWatch>();
			W->Label = InLabel;
			W->Recv = TStrongObjectPtr<UPipelineMcpTestReceiver>(NewObject<UPipelineMcpTestReceiver>(GetTransientPackage()));
			if (J)
			{
				W->bNull = false;
				W->Job = J;
				W->bDoneAtCreate = J->IsDone();
				W->bSucceededAtCreate = J->Succeeded();
				W->JsonAtCreate = J->GetJson();
				J->OnDone.AddDynamic(W->Recv.Get(), &UPipelineMcpTestReceiver::OnDone);
			}
			return W;
		}

		bool Ok() const { return Recv->Successes.Num() > 0 && Recv->Successes[0]; }
		FString Json() const { return Recv->Jsons.Num() > 0 ? Recv->Jsons[0] : FString(); }
	};
}

/** Helpers of Describe("TextureWait") (WP6.6). */
namespace PipelineMcpJobSpecTexWait
{
	/**
	 * A transient texture whose async editor build is in flight: 8192x8192 random BGRA8 source (random, so the derived data cache
	 * cannot hit) compressed as BC7. Pattern per Texture.h:36-40 (PreEditChange / change / PostEditChange); Source.Init must be inside
	 * PreEdit/PostEdit (Texture.h:216). UTexture::IsCompiling() is Texture.h:2045 (IsDefaultTexture): for UTexture2D it is true until
	 * the async work is complete (Texture2D.cpp:530).
	 *
	 * Only works with a renderer. UTexture::CachePlatformData does nothing when FApp::CanEverRender() is false
	 * (TextureDerivedData.cpp:3668), and CanEverRender() is false under -nullrhi (App.h:405): no async task is created, the
	 * texture is never compiling. MakeCase (below) skips loudly in that case.
	 * The DDC2 build does not run on the texture thread pool and a 4096x4096 BC7 build takes only about 0.1-0.3 s, so a
	 * smaller texture can finish before the check; 8192x8192 leaves a wide margin.
	 */
	static UTexture2D* MakeBigTexture()
	{
		const int32 Size = 8192;
		// Finish what earlier tests left running so this build does not compete with them.
		FTextureCompilingManager::Get().FinishAllCompilation();
		UTexture2D* Tex = NewObject<UTexture2D>(GetTransientPackage(),
			MakeUniqueObjectName(GetTransientPackage(), UTexture2D::StaticClass(), TEXT("T_McpWait")), RF_Transient);
		TArray<uint8> Pixels;
		Pixels.SetNumUninitialized((int64)Size * Size * 4);
		// Unique per texture and per run (a fixed seed repeats the data, the derived data cache then serves a finished build).
		// xorshift32 over 4-byte words: no flat blocks for BC7 to shortcut, and fast enough to fill 256 MB.
		uint32 State = GetTypeHash(FGuid::NewGuid()) | 1u;
		uint32* Words = reinterpret_cast<uint32*>(Pixels.GetData());
		for (int64 I = 0, N = (int64)Size * Size; I < N; ++I)
		{
			State ^= State << 13; State ^= State >> 17; State ^= State << 5;
			Words[I] = State;
		}
		Tex->PreEditChange(nullptr);
		Tex->Source.Init(Size, Size, 1, 1, TSF_BGRA8, Pixels.GetData());
		Tex->CompressionSettings = TC_BC7;
		Tex->SRGB = true;
		Tex->PostEditChange();
		return Tex;
	}
	struct FTexCase
	{
		TWeakObjectPtr<UTexture2D> Tex;
		FString Path;
		bool IsCompiling() const { const UTexture2D* T = Tex.Get(); return T && T->IsCompiling(); }
	};

	/** Sampled from a core ticker the first tick after OnDone fired (the texture never starts compiling again). */
	struct FDoneSample
	{
		bool bSampled = false;
		bool bTextureCompiling = true;
		int32 TicksUntilDone = -1;
	};

	static TSharedRef<FDoneSample> SampleAtDone(const TSharedRef<PipelineMcpJobSpecPrivate::FWatch>& W, const TSharedPtr<FTexCase>& T)
	{
		TSharedRef<FDoneSample> S = MakeShared<FDoneSample>();
		TSharedRef<int32> Ticks = MakeShared<int32>(0);
		FTSTicker::GetCoreTicker().AddTicker(TEXT("PipelineMcpJobSpecTexSample"), 0.0f, [W, T, S, Ticks](float)
		{
			++(*Ticks);
			if (W->Recv->Count >= 1)
			{
				S->bSampled = true;
				S->bTextureCompiling = T->IsCompiling();
				S->TicksUntilDone = *Ticks;
				return false;
			}
			return *Ticks < 20000; // far beyond any test timeout, only a leak guard
		});
		return S;
	}
}

BEGIN_DEFINE_SPEC(FPipelineMcpJobSpec, "Pipelines.MCP.Library",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	FString Dest;
	FString DestGuid;
	/** Imported assets, kept alive for the whole test. */
	TArray<TStrongObjectPtr<UObject>> Keep;

	/** <dev>/models/<Rel>; AddError (with the fetch command) and false when missing. */
	bool RequireFolder(const FString& Rel, FString& OutFolder);
	/** Imports <dev>/models/<Rel> unsaved (bValidate false) as <Dest>/<Name>; returns the package and its object path. */
	UPipelineModelPackage* ImportPkg(const FString& Rel, const FString& Name, FString& OutObjPath);
	TSharedPtr<FJsonObject> Parse(const FString& Text, const FString& What);
	void Cleanup();
	/** Waits (ticker, timeout) until every watch got a broadcast, settles Settle seconds, runs the common per-job checks, Check, then Done. */
	void AwaitAll(TArray<TSharedRef<PipelineMcpJobSpecPrivate::FWatch>> Watches, const FDoneDelegate& Done, TFunction<void()> Check,
		double Settle = 0.3, double Timeout = 100.0);
	void CheckStats(const TSharedPtr<FJsonObject>& Entry, const TCHAR* Key, bool bFull, const FString& Ctx);
	/** A failed job whose error text is exactly Expected. */
	void ExpectError(const TSharedRef<PipelineMcpJobSpecPrivate::FWatch>& W, const FString& Expected);
END_DEFINE_SPEC(FPipelineMcpJobSpec)

bool FPipelineMcpJobSpec::RequireFolder(const FString& Rel, FString& OutFolder)
{
	OutFolder = PipelineMcpJobSpecPrivate::GetDevDir() / TEXT("models") / Rel;
	if (!IFileManager::Get().DirectoryExists(*OutFolder))
	{
		const FString Cmd = Rel.StartsWith(TEXT("reference")) ? TEXT("python Tools/fetch_reference_models.py") : TEXT("python Tools/make_import_test_data.py");
		AddError(FString::Printf(TEXT("test model folder '%s' missing; generate with: %s (Python env: D:\\Micromamba\\micromambaenv\\envs\\pipelines\\python.exe)"), *OutFolder, *Cmd));
		return false;
	}
	return true;
}

UPipelineModelPackage* FPipelineMcpJobSpec::ImportPkg(const FString& Rel, const FString& Name, FString& OutObjPath)
{
	FString Folder;
	if (!RequireFolder(Rel, Folder))
	{
		return nullptr;
	}
	FPipelineImportOptions O;
	O.SourceFolder = Folder;
	O.DestinationPath = Dest;
	O.AssetName = Name;
	O.bSave = false;
	O.bValidate = false;
	const FPipelineImportResult R = ImportPipelineModelFolder(O);
	if (!R.bSuccess || !R.Package.Get())
	{
		AddError(FString::Printf(TEXT("import of '%s' as %s/%s failed; report: %s"), *Folder, *Dest, *Name, *R.Report.ToString()));
		return nullptr;
	}
	UPipelineModelPackage* Pkg = R.Package.Get();
	Keep.Add(TStrongObjectPtr<UObject>(Pkg));
	if (R.Model.Get())
	{
		Keep.Add(TStrongObjectPtr<UObject>(R.Model.Get()));
	}
	OutObjPath = PipelineMcpJobSpecPrivate::ObjPath(Dest, Name);
	return Pkg;
}

TSharedPtr<FJsonObject> FPipelineMcpJobSpec::Parse(const FString& Text, const FString& What)
{
	TSharedPtr<FJsonObject> Root;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		AddError(FString::Printf(TEXT("[%s] output is not a JSON object; got: %s"), *What, *Text.Left(400)));
		return nullptr;
	}
	// NaN/inf must be written as null, never as text (token positions, so words like "Info" don't match).
	const FString Lower = Text.ToLower();
	static const TCHAR* Bad[] = { TEXT(":nan"), TEXT(",nan"), TEXT("[nan"), TEXT(":-nan"), TEXT("-nan("), TEXT(":inf"), TEXT(",inf"),
		TEXT("[inf"), TEXT(":-inf"), TEXT("infinity") };
	for (const TCHAR* B : Bad)
	{
		if (Lower.Contains(B))
		{
			AddError(FString::Printf(TEXT("[%s] output contains non-finite text '%s'; expected null instead. Got: %s"), *What, B, *Text.Left(400)));
		}
	}
	return Root;
}

void FPipelineMcpJobSpec::Cleanup()
{
	using namespace PipelineMcpJobSpecPrivate;
	TArray<FAssetData> Assets;
	IAssetRegistry::Get()->GetAssetsByPath(FName(*Dest), Assets, true);
	TArray<UObject*> Objs;
	for (const FAssetData& A : Assets)
	{
		if (UObject* O = A.GetAsset())
		{
			Objs.Add(O);
		}
	}
	Keep.Reset();
	for (UObject* O : Objs)
	{
		DiscardOne(O);
	}
	// Test cleanup only: the product never deletes. Nothing is saved by these tests, but remove a stray folder if one exists.
	IFileManager::Get().DeleteDirectory(*FPaths::ConvertRelativePathToFull(FPaths::ProjectContentDir() / TEXT("PipelinesTests") / (TEXT("MCP_") + DestGuid)), false, true);
}

void FPipelineMcpJobSpec::AwaitAll(TArray<TSharedRef<PipelineMcpJobSpecPrivate::FWatch>> Watches, const FDoneDelegate& Done,
	TFunction<void()> Check, double Settle, double Timeout)
{
	using namespace PipelineMcpJobSpecPrivate;
	for (const TSharedRef<FWatch>& W : Watches)
	{
		if (W->bNull)
		{
			AddError(FString::Printf(TEXT("[%s] the library call returned nullptr on the game thread; expected a job object"), *W->Label));
			Done.ExecuteIfBound();
			return;
		}
	}
	PollUntil([Watches]()
	{
		for (const TSharedRef<FWatch>& W : Watches)
		{
			if (W->Recv->Count < 1) { return false; }
		}
		return true;
	}, Timeout, [this, Watches, Done, Check, Settle](bool bTimedOut)
	{
		if (bTimedOut)
		{
			for (const TSharedRef<FWatch>& W : Watches)
			{
				if (W->Recv->Count < 1)
				{
					AddError(FString::Printf(TEXT("[%s] OnDone never fired (timeout); expected exactly one broadcast"), *W->Label));
				}
			}
			Done.ExecuteIfBound();
			return;
		}
		const double T0 = FPlatformTime::Seconds();
		PollUntil([T0, Settle]() { return FPlatformTime::Seconds() - T0 >= Settle; }, Settle + 5.0, [this, Watches, Done, Check](bool)
		{
			for (const TSharedRef<FWatch>& W : Watches)
			{
				const FString L = W->Label;
				TestEqual(*FString::Printf(TEXT("[%s] OnDone broadcast count; got %d, expected 1 (exactly once)"), *L, W->Recv->Count), W->Recv->Count, 1);
				TestTrue(*FString::Printf(TEXT("[%s] OnDone broadcast on the game thread"), *L), W->Recv->bAllOnGameThread);
				TestFalse(*FString::Printf(TEXT("[%s] IsDone() right after the creating call returned; got true, expected false (completion is deferred)"), *L), W->bDoneAtCreate);
				TestFalse(*FString::Printf(TEXT("[%s] Succeeded() before done; got true, expected false"), *L), W->bSucceededAtCreate);
				TestEqual(*FString::Printf(TEXT("[%s] GetJson() before done; got '%s', expected empty"), *L, *W->JsonAtCreate), W->JsonAtCreate, FString());
				if (UPipelineMcpJob* J = W->Job.Get())
				{
					TestTrue(*FString::Printf(TEXT("[%s] IsDone() after OnDone; got false, expected true"), *L), J->IsDone());
					TestEqual(*FString::Printf(TEXT("[%s] Succeeded() equals the bSuccess broadcast"), *L), J->Succeeded(), W->Ok());
					TestEqual(*FString::Printf(TEXT("[%s] GetJson() equals the Json broadcast"), *L), J->GetJson(), W->Json());
				}
			}
			Check();
			Done.ExecuteIfBound();
		});
	});
}

void FPipelineMcpJobSpec::CheckStats(const TSharedPtr<FJsonObject>& Entry, const TCHAR* Key, bool bFull, const FString& Ctx)
{
	using namespace PipelineMcpJobSpecPrivate;
	const TSharedPtr<FJsonObject> S = GetObj(Entry, Key);
	if (!TestNotNull(*FString::Printf(TEXT("%s: '%s' object present"), *Ctx, Key), S.Get()))
	{
		return;
	}
	double Med = -1, P90 = -1, Min = -1, Max = -1;
	TestTrue(*FString::Printf(TEXT("%s: %s.median is a number"), *Ctx, Key), GetNum(S, TEXT("median"), Med));
	if (bFull)
	{
		TestTrue(*FString::Printf(TEXT("%s: %s.p90 is a number"), *Ctx, Key), GetNum(S, TEXT("p90"), P90));
		TestTrue(*FString::Printf(TEXT("%s: %s.min is a number"), *Ctx, Key), GetNum(S, TEXT("min"), Min));
		TestTrue(*FString::Printf(TEXT("%s: %s.max is a number"), *Ctx, Key), GetNum(S, TEXT("max"), Max));
		TestTrue(*FString::Printf(TEXT("%s: %s p90 %.4f >= median %.4f"), *Ctx, Key, P90, Med), P90 >= Med);
		TestTrue(*FString::Printf(TEXT("%s: %s min %.4f <= median %.4f <= max %.4f"), *Ctx, Key, Min, Med, Max), Min <= Med && Med <= Max);
		TestTrue(*FString::Printf(TEXT("%s: %s p90 %.4f <= max %.4f"), *Ctx, Key, P90, Max), P90 <= Max);
	}
}

void FPipelineMcpJobSpec::ExpectError(const TSharedRef<PipelineMcpJobSpecPrivate::FWatch>& W, const FString& Expected)
{
	using namespace PipelineMcpJobSpecPrivate;
	TestFalse(*FString::Printf(TEXT("[%s] bSuccess; got true with %s, expected false"), *W->Label, *W->Json().Left(300)), W->Ok());
	const TSharedPtr<FJsonObject> J = Parse(W->Json(), W->Label);
	if (J.IsValid())
	{
		TestEqual(*FString::Printf(TEXT("[%s] error text; got %s, expected '%s'"), *W->Label, *W->Json().Left(300), *Expected), GetStr(J, TEXT("error")), Expected);
	}
}

void FPipelineMcpJobSpec::Define()
{
	using namespace PipelineMcpJobSpecPrivate;

	BeforeEach([this]()
	{
		DestGuid = FGuid::NewGuid().ToString(EGuidFormats::Digits).Left(8);
		Dest = FString::Printf(TEXT("/Game/PipelinesTests/MCP_%s"), *DestGuid);
		Keep.Reset();
	});

	AfterEach([this]()
	{
		Cleanup();
	});

	const FTimespan LongTimeout = FTimespan::FromSeconds(180.0);

	// ---------------------------------------------------------------------------------------------------------------------
	Describe("Job", [this, LongTimeout]()
	{
		LatentIt("defers an immediate error: IsDone false after the call, OnDone once later with the exact error (Runs 0)", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString Pkg;
			if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), Pkg)) { Done.ExecuteIfBound(); return; }
			// A completion inside the call (before the caller can bind) is the bug: AwaitAll reports bDoneAtCreate.
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Pkg, 0, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("Runs 0"));
			AwaitAll({ W }, Done, [this, W]()
			{
				ExpectError(W, TEXT("runs must be 1 to 1000"));
			});
		});

		LatentIt("fires OnDone exactly once: 1 broadcast over 2 s after completion (success job and error job)", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString Pkg;
			if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), Pkg)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FWatch> Ok = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Pkg, TArray<FString>{ TEXT("hello world") }, TEXT(""), 3, 0.5f), TEXT("sample success"));
			const TSharedRef<FWatch> Bad = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Pkg, 1001, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("runs 1001"));
			AwaitAll({ Ok, Bad }, Done, [this, Ok, Bad]()
			{
				TestTrue(*FString::Printf(TEXT("sample job succeeded; got %s"), *Ok->Json().Left(300)), Ok->Ok());
				ExpectError(Bad, TEXT("runs must be 1 to 1000"));
			}, /*Settle*/ 2.0);
		});

		LatentIt("GC while running does not kill the job; after completion the job is collectable", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString Pkg;
			if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), Pkg)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Pkg, 60, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("GC job"));
			if (W->bNull) { AddError(TEXT("BenchmarkModelPackage returned nullptr")); Done.ExecuteIfBound(); return; }
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS); // right after creation, before any tick
			TSharedRef<int32> Ticks = MakeShared<int32>(0);
			PollUntil([this, W, Ticks]()
			{
				if (W->Recv->Count >= 1) { return true; }
				if (!W->Job.IsValid())
				{
					AddError(TEXT("[GC job] the running job was garbage-collected before it completed; it must keep itself alive"));
					return true;
				}
				if ((++(*Ticks) % 3) == 0) { CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS); }
				return false;
			}, 100.0, [this, W, Done](bool)
			{
				AwaitAll({ W }, Done, [this, W]()
				{
					TestTrue(*FString::Printf(TEXT("[GC job] succeeded despite GC while running; got %s"), *W->Json().Left(300)), W->Ok());
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
					TestFalse(TEXT("[GC job] job still alive after completion and two GCs; expected collectable (it must release itself after broadcasting)"), W->Job.IsValid());
				});
			});
		});

		LatentIt("8 jobs at once (mixed samples, benchmarks, errors) each complete once with independent results", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString Emb, Cls;
			if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), Emb)) { Done.ExecuteIfBound(); return; }
			if (!ImportPkg(TEXT("import_test/tiny_text_classification"), TEXT("Cls"), Cls)) { Done.ExecuteIfBound(); return; }
			TArray<FString> Many;
			for (int32 i = 0; i < 65; ++i) { Many.Add(FString::Printf(TEXT("t%d"), i)); }
			TArray<TSharedRef<FWatch>> Ws;
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Emb, TArray<FString>{ TEXT("alpha") }, TEXT(""), 3, 0.5f), TEXT("0 emb alpha")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Emb, TArray<FString>{ TEXT("beta"), TEXT("gamma"), TEXT("delta") }, TEXT(""), 3, 0.5f), TEXT("1 emb x3")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Cls, TArray<FString>{ TEXT("hello world") }, TEXT(""), 2, 0.5f), TEXT("2 cls")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Emb, 3, TEXT("cpu"), TArray<FString>{ TEXT("a"), TEXT("b") }, TEXT("")), TEXT("3 bench emb texts")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Cls, 2, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("4 bench cls default")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Emb, 0, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("5 runs 0")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Cls, 2, TEXT("tpu"), TArray<FString>(), TEXT("")), TEXT("6 devices tpu")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Emb, Many, TEXT(""), 3, 0.5f), TEXT("7 sample 65 texts")));
			AwaitAll(Ws, Done, [this, Ws]()
			{
				const TSharedPtr<FJsonObject> J0 = Parse(Ws[0]->Json(), Ws[0]->Label);
				TestTrue(*FString::Printf(TEXT("job 0 success; got %s"), *Ws[0]->Json().Left(300)), Ws[0]->Ok());
				const TArray<TSharedPtr<FJsonValue>> E0 = GetArr(J0, TEXT("embeddings"));
				TestEqual(TEXT("job 0 embeddings count (1 text)"), E0.Num(), 1);
				if (E0.Num() == 1) { TestEqual(TEXT("job 0 embeddings[0].text (not another job's text)"), GetStr(E0[0]->AsObject(), TEXT("text")), FString(TEXT("alpha"))); }

				const TSharedPtr<FJsonObject> J1 = Parse(Ws[1]->Json(), Ws[1]->Label);
				TestTrue(*FString::Printf(TEXT("job 1 success; got %s"), *Ws[1]->Json().Left(300)), Ws[1]->Ok());
				const TArray<TSharedPtr<FJsonValue>> E1 = GetArr(J1, TEXT("embeddings"));
				TestEqual(TEXT("job 1 embeddings count (3 texts)"), E1.Num(), 3);
				if (E1.Num() == 3) { TestEqual(TEXT("job 1 embeddings[2].text"), GetStr(E1[2]->AsObject(), TEXT("text")), FString(TEXT("delta"))); }
				TestEqual(TEXT("job 1 similarity rows"), GetArr(J1, TEXT("similarity")).Num(), 3);

				const TSharedPtr<FJsonObject> J2 = Parse(Ws[2]->Json(), Ws[2]->Label);
				TestTrue(*FString::Printf(TEXT("job 2 success; got %s"), *Ws[2]->Json().Left(300)), Ws[2]->Ok());
				TestEqual(TEXT("job 2 task"), GetStr(J2, TEXT("task")), FString(TEXT("TextClassification")));
				const TArray<TSharedPtr<FJsonValue>> R2 = GetArr(J2, TEXT("results"));
				TestEqual(TEXT("job 2 results count"), R2.Num(), 1);
				if (R2.Num() == 1) { TestEqual(TEXT("job 2 results[0].text"), GetStr(R2[0]->AsObject(), TEXT("text")), FString(TEXT("hello world"))); }

				const TSharedPtr<FJsonObject> J3 = Parse(Ws[3]->Json(), Ws[3]->Label);
				TestTrue(*FString::Printf(TEXT("job 3 success; got %s"), *Ws[3]->Json().Left(300)), Ws[3]->Ok());
				TestEqual(TEXT("job 3 input"), GetStr(J3, TEXT("input")), FString(TEXT("texts (2)")));
				double Runs3 = -1; GetNum(J3, TEXT("runs"), Runs3);
				TestEqual(TEXT("job 3 runs"), (int32)Runs3, 3);

				const TSharedPtr<FJsonObject> J4 = Parse(Ws[4]->Json(), Ws[4]->Label);
				TestTrue(*FString::Printf(TEXT("job 4 success; got %s"), *Ws[4]->Json().Left(300)), Ws[4]->Ok());
				TestEqual(TEXT("job 4 input"), GetStr(J4, TEXT("input")), FString(TEXT("default text")));
				TestEqual(TEXT("job 4 task"), GetStr(J4, TEXT("task")), FString(TEXT("TextClassification")));
				double Runs4 = -1; GetNum(J4, TEXT("runs"), Runs4);
				TestEqual(TEXT("job 4 runs"), (int32)Runs4, 2);

				ExpectError(Ws[5], TEXT("runs must be 1 to 1000"));
				ExpectError(Ws[6], TEXT("devices must be cpu, gpu or both"));
				ExpectError(Ws[7], TEXT("texts: give 1 to 64 strings"));
			});
		});

		It("returns nullptr and logs a Warning when BenchmarkModelPackage / RunPipelineSample are called off the game thread", [this]()
		{
			AddExpectedMessage(TEXT("BenchmarkModelPackage must be called on the game thread; returning nullptr."), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			AddExpectedMessage(TEXT("RunPipelineSample must be called on the game thread; returning nullptr."), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			const TArray<TPair<FString, TFunction<FString()>>> Cases = {
				{ TEXT("BenchmarkModelPackage"), []() { return FString(UPipelineMcpLibrary::BenchmarkModelPackage(TEXT("/Game/Nope.Nope"), 3, TEXT("cpu"), TArray<FString>(), TEXT("")) ? TEXT("object") : TEXT("null")); } },
				{ TEXT("RunPipelineSample"), []() { return FString(UPipelineMcpLibrary::RunPipelineSample(TEXT("/Game/Nope.Nope"), TArray<FString>{ TEXT("x") }, TEXT(""), 1, 0.5f) ? TEXT("object") : TEXT("null")); } },
			};
			for (const TPair<FString, TFunction<FString()>>& C : Cases)
			{
				FString Out;
				if (!RunOffGameThread(C.Value, Out))
				{
					AddError(FString::Printf(TEXT("%s from a background thread did not return within 30 s"), *C.Key));
					continue;
				}
				TestEqual(*FString::Printf(TEXT("%s off the game thread; got %s, expected null"), *C.Key, *Out), Out, FString(TEXT("null")));
			}
		});

		// -----------------------------------------------------------------------------------------------------------------
		// WP6.6: image jobs wait (ticker, no blocking) for a texture whose async editor build is still running.
		Describe("TextureWait", [this, LongTimeout]()
		{
			using namespace PipelineMcpJobSpecTexWait;

			// Builds the compiling texture and asserts the setup worked (a setup that produced a finished texture fails loudly, it
			// never passes vacuously). Call it right before the job call: import the packages first.
			auto MakeCase = [this]() -> TSharedPtr<FTexCase>
			{
				if (!FApp::CanEverRender())
				{
					// -nullrhi: no texture build can start, so there is no compiling texture to wait for (see MakeBigTexture). A test that went
					// on would not test the wait, so it stops here. Not silent: Info always, an error when the run demands it.
					const FString Msg = TEXT("TextureWait SKIPPED: FApp::CanEverRender() is false (-nullrhi), so no texture can be compiling. Run without -nullrhi to execute it (set MODELPIPELINES_REQUIRE_TEXTURE_WAIT=1 to make this skip an error).");
					if (FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_TEXTURE_WAIT")) == TEXT("1"))
					{
						AddError(Msg);
					}
					else
					{
						AddInfo(Msg);
					}
					return nullptr;
				}
				UTexture2D* Tex = MakeBigTexture();
				if (!Tex)
				{
					AddError(TEXT("test setup: NewObject<UTexture2D> returned null"));
					return nullptr;
				}
				Keep.Add(TStrongObjectPtr<UObject>(Tex));
				TSharedRef<FTexCase> C = MakeShared<FTexCase>();
				C->Tex = Tex;
				C->Path = Tex->GetPathName();
				if (!Tex->IsCompiling())
				{
					AddError(FString::Printf(TEXT("test setup: texture '%s' is not compiling right after Source.Init + PostEditChange (did the build finish inside PostEditChange, or is Editor.AsyncTextureCompilation 0?); the TextureWait tests need a compiling texture"), *C->Path));
					return nullptr;
				}
				return C;
			};
			// The job's own error string for a texture that never finished is "texture '<name>' is still compiling after 60 s";
			// the pre-fix error is "...is still compiling; retry when it has finished". Both contain this text.
			auto NoStill = [this](const TSharedRef<FWatch>& W)
			{
				TestFalse(*FString::Printf(TEXT("[%s] result contains 'still compiling' (the job should have waited for the texture, not reported it); got %s"), *W->Label, *W->Json().Left(500)),
					W->Json().Contains(TEXT("still compiling")));
			};

			auto CheckSample = [this, NoStill](const TSharedRef<FWatch>& W, const TSharedRef<FDoneSample>& S)
			{
				NoStill(W);
				TestTrue(*FString::Printf(TEXT("[%s] the post-OnDone sampler ran"), *W->Label), S->bSampled);
				// A job that did not wait (or gave up) completes while the build is still running.
				TestFalse(*FString::Printf(TEXT("[%s] texture IsCompiling() at OnDone; got true, expected false (the job must finish only after the texture finished)"), *W->Label), S->bTextureCompiling);
			};

			LatentIt("1 sample ImageClassification on a compiling texture: one OnDone, texture finished by then, no 'still compiling' error", LongTimeout, [this, MakeCase, CheckSample](const FDoneDelegate& Done)
			{
				FString Pkg;
				if (!ImportPkg(TEXT("import_test/tiny_image_classification"), TEXT("ImgCls"), Pkg)) { Done.ExecuteIfBound(); return; }
				const TSharedPtr<FTexCase> T = MakeCase();
				if (!T.IsValid()) { Done.ExecuteIfBound(); return; }
				// Success, or (under -nullrhi) the reader's no-RHI PreprocessFailed error are both fine: only the wait is under test.
				const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Pkg, TArray<FString>(), T->Path, 3, 0.5f), TEXT("sample cls compiling"));
				const TSharedRef<FDoneSample> S = SampleAtDone(W, T);
				AwaitAll({ W }, Done, [W, S, CheckSample]()
				{
					CheckSample(W, S); // AwaitAll already checked: exactly one broadcast, game thread, not done at create
				});
			});

			LatentIt("2 benchmark (cpu, Runs 2) on a compiling texture: input 'image <path>', CPU entry has no 'still compiling' error", LongTimeout, [this, MakeCase, CheckSample](const FDoneDelegate& Done)
			{
				FString Pkg;
				if (!ImportPkg(TEXT("import_test/tiny_image_classification"), TEXT("ImgCls"), Pkg)) { Done.ExecuteIfBound(); return; }
				const TSharedPtr<FTexCase> T = MakeCase();
				if (!T.IsValid()) { Done.ExecuteIfBound(); return; }
				const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Pkg, 2, TEXT("cpu"), TArray<FString>(), T->Path), TEXT("bench cls compiling"));
				const TSharedRef<FDoneSample> S = SampleAtDone(W, T);
				AwaitAll({ W }, Done, [this, W, S, T, CheckSample]()
				{
					CheckSample(W, S);
					const FString Js = W->Json().Left(1200);
					TestTrue(*FString::Printf(TEXT("benchmark job succeeded (a per-device error entry is still a success); got %s"), *Js), W->Ok());
					const TSharedPtr<FJsonObject> J = Parse(W->Json(), W->Label);
					TestEqual(*FString::Printf(TEXT("input; got %s"), *Js), GetStr(J, TEXT("input")), FString(TEXT("image ")) + T->Path);
					const TSharedPtr<FJsonObject> Cpu = FindDevice(J, TEXT("CPU"));
					if (TestNotNull(*FString::Printf(TEXT("CPU entry; got %s"), *Js), Cpu.Get()))
					{
						TestFalse(*FString::Printf(TEXT("CPU entry error contains 'still compiling'; got '%s'"), *GetStr(Cpu, TEXT("error"))),
							GetStr(Cpu, TEXT("error")).Contains(TEXT("still compiling")));
					}
				});
			});

			LatentIt("3 sample ObjectDetection (score_threshold 0.5) on a compiling texture: same as 1", LongTimeout, [this, MakeCase, CheckSample](const FDoneDelegate& Done)
			{
				FString Pkg;
				if (!ImportPkg(TEXT("import_test/tiny_object_detection"), TEXT("ImgDet"), Pkg)) { Done.ExecuteIfBound(); return; }
				const TSharedPtr<FTexCase> T = MakeCase();
				if (!T.IsValid()) { Done.ExecuteIfBound(); return; }
				const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Pkg, TArray<FString>(), T->Path, 3, 0.5f), TEXT("sample det compiling"));
				const TSharedRef<FDoneSample> S = SampleAtDone(W, T);
				AwaitAll({ W }, Done, [W, S, CheckSample]()
				{
					CheckSample(W, S);
				});
			});

			LatentIt("4 argument error is not delayed: compiling texture + score_threshold 1.5 fails with the exact error within 2 ticks", LongTimeout, [this, MakeCase](const FDoneDelegate& Done)
			{
				FString Pkg;
				if (!ImportPkg(TEXT("import_test/tiny_object_detection"), TEXT("ImgDet"), Pkg)) { Done.ExecuteIfBound(); return; }
				const TSharedPtr<FTexCase> T = MakeCase();
				if (!T.IsValid()) { Done.ExecuteIfBound(); return; }
				const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Pkg, TArray<FString>(), T->Path, 3, 1.5f), TEXT("det threshold 1.5 compiling"));
				// A job that waited for the texture first would report this only after seconds (hundreds of ticks).
				const TSharedRef<FDoneSample> S = SampleAtDone(W, T);
				AwaitAll({ W }, Done, [this, W, S]()
				{
					ExpectError(W, TEXT("score_threshold must be 0 to 1"));
					TestTrue(*FString::Printf(TEXT("[%s] ticks until OnDone; got %d, expected 1 or 2 (no wait)"), *W->Label, S->TicksUntilDone), S->TicksUntilDone >= 1 && S->TicksUntilDone <= 2);
				});
			});

			LatentIt("5 text task with an Image naming a compiling texture does not wait: input stays 'default text', done while still compiling", LongTimeout, [this, MakeCase, NoStill](const FDoneDelegate& Done)
			{
				FString Pkg;
				if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), Pkg)) { Done.ExecuteIfBound(); return; }
				const TSharedPtr<FTexCase> T = MakeCase();
				if (!T.IsValid()) { Done.ExecuteIfBound(); return; }
				const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Pkg, 1, TEXT("cpu"), TArray<FString>(), T->Path), TEXT("text bench + compiling image"));
				const TSharedRef<FDoneSample> S = SampleAtDone(W, T);
				AwaitAll({ W }, Done, [this, W, S, NoStill]()
				{
					NoStill(W);
					const FString Js = W->Json().Left(1200);
					TestTrue(*FString::Printf(TEXT("job succeeded; got %s"), *Js), W->Ok());
					TestEqual(*FString::Printf(TEXT("input; got %s"), *Js), GetStr(Parse(W->Json(), W->Label), TEXT("input")), FString(TEXT("default text")));
					// A one-run CPU benchmark of a tiny text model finishes long before a texture that is never released; a job that waited for the
					// unrelated texture would never complete.
					TestTrue(*FString::Printf(TEXT("[%s] the texture was still compiling at OnDone (the text job must not wait for it); got compiling=%d after %d ticks"), *W->Label, S->bTextureCompiling ? 1 : 0, S->TicksUntilDone), S->bTextureCompiling);
				});
			});

			LatentIt("6 CancelAll while waiting: OnDone once with error 'x', then silence for 2 s (no crash)", LongTimeout, [this, MakeCase](const FDoneDelegate& Done)
			{
				FString Pkg;
				if (!ImportPkg(TEXT("import_test/tiny_image_classification"), TEXT("ImgCls"), Pkg)) { Done.ExecuteIfBound(); return; }
				const TSharedPtr<FTexCase> T = MakeCase();
				if (!T.IsValid()) { Done.ExecuteIfBound(); return; }
				const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Pkg, TArray<FString>(), T->Path, 3, 0.5f), TEXT("cancel while waiting"));
				if (W->bNull) { AddError(TEXT("RunPipelineSample returned nullptr")); Done.ExecuteIfBound(); return; }
				TSharedRef<int32> Ticks = MakeShared<int32>(0);
				// Tick 1 runs the job's Start (it begins waiting); the cancel comes on tick 2, while the build is still running.
				FTSTicker::GetCoreTicker().AddTicker(TEXT("PipelineMcpJobSpecCancel"), 0.0f, [this, W, T, Ticks, Done](float)
				{
					if (++(*Ticks) < 2)
					{
						return true;
					}
					TestTrue(TEXT("[cancel while waiting] the texture was still compiling at the cancel (else the test did not cover the wait)"), T->IsCompiling());
					TestEqual(TEXT("[cancel while waiting] OnDone count before the cancel (the job must still be waiting)"), W->Recv->Count, 0);
					FPipelineMcpJobInternal::CancelAll(TEXT("x"));
					const double T0 = FPlatformTime::Seconds();
					PollUntil([T0]() { return FPlatformTime::Seconds() - T0 >= 2.0; }, 10.0, [this, W, Done](bool)
					{
						TestEqual(*FString::Printf(TEXT("[cancel while waiting] OnDone count after 2 s; got %d, expected 1 (a late continuation or timeout must not fire again)"), W->Recv->Count), W->Recv->Count, 1);
						TestTrue(TEXT("[cancel while waiting] OnDone on the game thread"), W->Recv->bAllOnGameThread);
						ExpectError(W, TEXT("x"));
						CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS, true); // the runner (and its ticker) must be gone without a crash
						Done.ExecuteIfBound();
					});
					return false;
				});
			});

			LatentIt("7 GC (full purge) every tick while waiting: the job survives and completes once as in 1", LongTimeout, [this, MakeCase, CheckSample](const FDoneDelegate& Done)
			{
				FString Pkg;
				if (!ImportPkg(TEXT("import_test/tiny_image_classification"), TEXT("ImgCls"), Pkg)) { Done.ExecuteIfBound(); return; }
				const TSharedPtr<FTexCase> T = MakeCase();
				if (!T.IsValid()) { Done.ExecuteIfBound(); return; }
				const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Pkg, TArray<FString>(), T->Path, 3, 0.5f), TEXT("GC while waiting"));
				if (W->bNull) { AddError(TEXT("RunPipelineSample returned nullptr")); Done.ExecuteIfBound(); return; }
				const TSharedRef<FDoneSample> S = SampleAtDone(W, T);
				TSharedRef<int32> GcCount = MakeShared<int32>(0);
				PollUntil([this, W, GcCount]()
				{
					if (W->Recv->Count >= 1) { return true; }
					if (!W->Job.IsValid())
					{
						AddError(TEXT("[GC while waiting] the job was garbage-collected before it completed; it must keep itself alive while waiting"));
						return true;
					}
					CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS, true);
					++(*GcCount);
					return false;
				}, 100.0, [this, W, S, GcCount, Done, CheckSample](bool)
				{
					TestTrue(*FString::Printf(TEXT("[GC while waiting] GCs run during the wait; got %d, expected at least 1"), *GcCount), *GcCount >= 1);
					AwaitAll({ W }, Done, [W, S, CheckSample]()
					{
						CheckSample(W, S);
					});
				});
			});

			LatentIt("8 eight jobs at once on one compiling texture (mixed samples/benchmarks): each completes once, none 'still compiling'", LongTimeout, [this, MakeCase, NoStill](const FDoneDelegate& Done)
			{
				FString Cls, Det;
				if (!ImportPkg(TEXT("import_test/tiny_image_classification"), TEXT("ImgCls"), Cls)) { Done.ExecuteIfBound(); return; }
				if (!ImportPkg(TEXT("import_test/tiny_object_detection"), TEXT("ImgDet"), Det)) { Done.ExecuteIfBound(); return; }
				const TSharedPtr<FTexCase> T = MakeCase();
				if (!T.IsValid()) { Done.ExecuteIfBound(); return; }
				TArray<TSharedRef<FWatch>> Ws;
				Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Cls, TArray<FString>(), T->Path, 3, 0.5f), TEXT("0 sample cls")));
				Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Det, TArray<FString>(), T->Path, 3, 0.5f), TEXT("1 sample det")));
				Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Cls, 1, TEXT("cpu"), TArray<FString>(), T->Path), TEXT("2 bench cls")));
				Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Det, 1, TEXT("cpu"), TArray<FString>(), T->Path), TEXT("3 bench det")));
				Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Cls, TArray<FString>(), T->Path, 1, 0.5f), TEXT("4 sample cls topk 1")));
				Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Det, TArray<FString>(), T->Path, 3, 0.0f), TEXT("5 sample det thr 0")));
				Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Cls, 2, TEXT("cpu"), TArray<FString>(), T->Path), TEXT("6 bench cls runs 2")));
				Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Det, 2, TEXT("cpu"), TArray<FString>(), T->Path), TEXT("7 bench det runs 2")));
				TArray<TSharedRef<FDoneSample>> Ss;
				for (const TSharedRef<FWatch>& W : Ws) { Ss.Add(SampleAtDone(W, T)); }
				AwaitAll(Ws, Done, [this, Ws, Ss, NoStill]()
				{
					for (int32 i = 0; i < Ws.Num(); ++i)
					{
						NoStill(Ws[i]);
						TestFalse(*FString::Printf(TEXT("[%s] texture IsCompiling() at OnDone; got true, expected false"), *Ws[i]->Label), Ss[i]->bTextureCompiling);
						if (i == 2 || i == 3 || i == 6 || i == 7)
						{
							TestTrue(*FString::Printf(TEXT("[%s] benchmark job succeeded; got %s"), *Ws[i]->Label, *Ws[i]->Json().Left(400)), Ws[i]->Ok());
						}
					}
				});
			});
		});
	});

	// ---------------------------------------------------------------------------------------------------------------------
	Describe("Benchmark", [this, LongTimeout]()
	{
		LatentIt("tiny_text_embedding runs 5 on 'cpu': one CPU entry with sane statistics and memory; source PreferredDevice unchanged", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			UPipelineModelPackage* Pkg = ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), PkgPath);
			if (!Pkg) { Done.ExecuteIfBound(); return; }
			// A non-default device on the source: a benchmark that set PreferredDevice on the asset itself (not on a duplicate) leaves CPU here.
			Pkg->PreferredDevice = EPipelineDevice::GPU;
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(PkgPath, 5, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("bench cpu"));
			AwaitAll({ W }, Done, [this, W, Pkg, PkgPath]()
			{
				const FString Js = W->Json().Left(1200);
				TestTrue(*FString::Printf(TEXT("job succeeded; got %s"), *Js), W->Ok());
				const TSharedPtr<FJsonObject> J = Parse(W->Json(), TEXT("bench cpu"));
				TestEqual(TEXT("package"), GetStr(J, TEXT("package")), PkgPath);
				TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("TextEmbedding")));
				double Runs = -1; GetNum(J, TEXT("runs"), Runs);
				TestEqual(TEXT("runs"), (int32)Runs, 5);
				TestEqual(TEXT("input (no texts given)"), GetStr(J, TEXT("input")), FString(TEXT("default text")));
				const TArray<TSharedPtr<FJsonValue>> Devs = GetArr(J, TEXT("devices"));
				if (!TestEqual(*FString::Printf(TEXT("devices entries for 'cpu'; got %s"), *Js), Devs.Num(), 1)) { return; }
				const TSharedPtr<FJsonObject> E = Devs[0]->AsObject();
				TestEqual(TEXT("devices[0].device"), GetStr(E, TEXT("device")), FString(TEXT("CPU")));
				bool bAvail = false;
				TestTrue(TEXT("devices[0].available is a bool"), E->TryGetBoolField(TEXT("available"), bAvail));
				TestTrue(TEXT("devices[0].available"), bAvail);
				TestFalse(*FString::Printf(TEXT("devices[0] has an error field; got %s"), *Js), E->HasField(TEXT("error")));
				TestEqual(*FString::Printf(TEXT("actual_device; got %s"), *Js), GetStr(E, TEXT("actual_device")), FString(TEXT("CPU")));
				double First = -1;
				TestTrue(TEXT("first_ms is a number"), GetNum(E, TEXT("first_ms"), First));
				TestTrue(*FString::Printf(TEXT("first_ms %.4f >= 0"), First), First >= 0.0);
				CheckStats(E, TEXT("inference_ms"), true, TEXT("cpu"));
				CheckStats(E, TEXT("total_ms"), true, TEXT("cpu"));
				CheckStats(E, TEXT("preprocess_ms"), false, TEXT("cpu"));
				CheckStats(E, TEXT("postprocess_ms"), false, TEXT("cpu"));
				double InfMed = -1;
				GetNum(GetObj(E, TEXT("inference_ms")), TEXT("median"), InfMed);
				TestTrue(*FString::Printf(TEXT("inference_ms.median %.6f > 0 (a real run takes measurable time)"), InfMed), InfMed > 0.0);
				double TotMed = -1;
				GetNum(GetObj(E, TEXT("total_ms")), TEXT("median"), TotMed);
				TestTrue(*FString::Printf(TEXT("total_ms.median %.6f > 0 (game-thread wall time of a call)"), TotMed), TotMed > 0.0);

				const TSharedPtr<FJsonObject> M = GetObj(E, TEXT("memory_mb"));
				if (TestNotNull(*FString::Printf(TEXT("memory_mb present; got %s"), *Js), M.Get()))
				{
					double B = 0, A = 0, D = 0;
					TestTrue(TEXT("memory_mb.before is a number"), GetNum(M, TEXT("before"), B));
					TestTrue(TEXT("memory_mb.after_warmup is a number"), GetNum(M, TEXT("after_warmup"), A));
					TestTrue(TEXT("memory_mb.delta is a number"), GetNum(M, TEXT("delta"), D));
					TestTrue(*FString::Printf(TEXT("memory_mb.before %.3f > 0 (process used physical memory in MiB)"), B), B > 0.0);
					TestTrue(*FString::Printf(TEXT("memory_mb.delta %.3f ~ after_warmup %.3f - before %.3f (tolerance 0.05)"), D, A, B), FMath::Abs(D - (A - B)) < 0.05);
				}
				// The benchmark works on a transient duplicate: the asset keeps its device and is not modified.
				TestEqual(TEXT("source package PreferredDevice after the benchmark; expected GPU (unchanged)"), (int32)Pkg->PreferredDevice, (int32)EPipelineDevice::GPU);
			});
		});

		LatentIt("Runs 1 with the package-name form, 'CPU' upper case and 3 texts: input 'texts (3)', p90 = median = min = max (one sample)", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), PkgPath)) { Done.ExecuteIfBound(); return; }
			const FString PkgName = Dest / TEXT("Emb");
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(PkgName, 1, TEXT("CPU"), TArray<FString>{ TEXT("one"), TEXT("two"), TEXT("three") }, TEXT("")), TEXT("bench runs 1"));
			AwaitAll({ W }, Done, [this, W]()
			{
				const FString Js = W->Json().Left(1200);
				TestTrue(*FString::Printf(TEXT("job succeeded; got %s"), *Js), W->Ok());
				const TSharedPtr<FJsonObject> J = Parse(W->Json(), TEXT("bench runs 1"));
				TestEqual(*FString::Printf(TEXT("input; got %s"), *Js), GetStr(J, TEXT("input")), FString(TEXT("texts (3)")));
				const TArray<TSharedPtr<FJsonValue>> Devs = GetArr(J, TEXT("devices"));
				if (!TestEqual(TEXT("devices entries for 'CPU' (case-insensitive)"), Devs.Num(), 1)) { return; }
				const TSharedPtr<FJsonObject> S = GetObj(Devs[0]->AsObject(), TEXT("inference_ms"));
				double Med = -1, P90 = -2, Min = -3, Max = -4;
				GetNum(S, TEXT("median"), Med); GetNum(S, TEXT("p90"), P90); GetNum(S, TEXT("min"), Min); GetNum(S, TEXT("max"), Max);
				// Nearest rank with n = 1: ceil(0.9*1) = 1st value; the median of one value is that value.
				TestTrue(*FString::Printf(TEXT("Runs=1 stats equal: median %.6f p90 %.6f min %.6f max %.6f; expected all the same"), Med, P90, Min, Max),
					Med == P90 && Med == Min && Med == Max);
			});
		});

		LatentIt("devices 'both': CPU entry then GPU entry; GPU available true (REQUIRE_GPU) or a clean available:false", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), PkgPath)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(PkgPath, 3, TEXT("Both"), TArray<FString>(), TEXT("")), TEXT("bench both"));
			AwaitAll({ W }, Done, [this, W]()
			{
				const FString Js = W->Json().Left(1600);
				TestTrue(*FString::Printf(TEXT("job succeeded (an unavailable GPU is not an error); got %s"), *Js), W->Ok());
				const TSharedPtr<FJsonObject> J = Parse(W->Json(), TEXT("bench both"));
				const TArray<TSharedPtr<FJsonValue>> Devs = GetArr(J, TEXT("devices"));
				if (!TestEqual(*FString::Printf(TEXT("devices entries for 'both'; got %s"), *Js), Devs.Num(), 2)) { return; }
				TestEqual(TEXT("devices[0].device (CPU first)"), GetStr(Devs[0]->AsObject(), TEXT("device")), FString(TEXT("CPU")));
				const TSharedPtr<FJsonObject> G = Devs[1]->AsObject();
				TestEqual(TEXT("devices[1].device (GPU second)"), GetStr(G, TEXT("device")), FString(TEXT("GPU")));
				bool bAvail = false;
				TestTrue(TEXT("devices[1].available is a bool"), G->TryGetBoolField(TEXT("available"), bAvail));
				if (IsGpuRequired())
				{
					TestTrue(*FString::Printf(TEXT("MODELPIPELINES_REQUIRE_GPU=1: GPU available; got %s"), *Js), bAvail);
					TestFalse(*FString::Printf(TEXT("GPU entry has an error; got %s"), *Js), G->HasField(TEXT("error")));
					TestEqual(*FString::Printf(TEXT("GPU actual_device; got %s"), *Js), GetStr(G, TEXT("actual_device")), FString(TEXT("GPU")));
					CheckStats(G, TEXT("inference_ms"), true, TEXT("gpu"));
				}
				else if (!bAvail)
				{
					AddInfo(TEXT("GPU not available and MODELPIPELINES_REQUIRE_GPU is not 1: only the clean available:false entry was checked (GPU timing assertions skipped)."));
					TestFalse(*FString::Printf(TEXT("unavailable GPU is reported without an error field; got %s"), *Js), G->HasField(TEXT("error")));
					TestFalse(TEXT("unavailable GPU entry has no statistics"), G->HasField(TEXT("inference_ms")));
				}
				else
				{
					AddInfo(TEXT("MODELPIPELINES_REQUIRE_GPU is not 1; the GPU entry was available and only its presence was checked."));
				}
			});
		});

		LatentIt("tiny_image_classification without Image: input 'synthetic WxH' of the package preprocess size; Texts are ignored", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			UPipelineModelPackage* Pkg = ImportPkg(TEXT("import_test/tiny_image_classification"), TEXT("ImgCls"), PkgPath);
			if (!Pkg) { Done.ExecuteIfBound(); return; }
			const FString Expected = FString::Printf(TEXT("synthetic %dx%d"), Pkg->Preprocess.ImageWidth, Pkg->Preprocess.ImageHeight);
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(PkgPath, 2, TEXT("cpu"), TArray<FString>{ TEXT("ignored text") }, TEXT("")), TEXT("bench image synthetic"));
			AwaitAll({ W }, Done, [this, W, Expected]()
			{
				const FString Js = W->Json().Left(1200);
				TestTrue(*FString::Printf(TEXT("job succeeded; got %s"), *Js), W->Ok());
				const TSharedPtr<FJsonObject> J = Parse(W->Json(), TEXT("bench image synthetic"));
				TestEqual(*FString::Printf(TEXT("input; got %s"), *Js), GetStr(J, TEXT("input")), Expected);
				TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("ImageClassification")));
				const TSharedPtr<FJsonObject> Cpu = FindDevice(J, TEXT("CPU"));
				if (TestNotNull(*FString::Printf(TEXT("CPU entry; got %s"), *Js), Cpu.Get()))
				{
					TestFalse(*FString::Printf(TEXT("CPU entry error on a synthetic gray image; got %s"), *Js), Cpu->HasField(TEXT("error")));
					CheckStats(Cpu, TEXT("inference_ms"), true, TEXT("image cpu"));
				}
			});
		});

		LatentIt("tiny_object_detection without Image: input 'synthetic WxH' of the package preprocess size", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			UPipelineModelPackage* Pkg = ImportPkg(TEXT("import_test/tiny_object_detection"), TEXT("ImgDet"), PkgPath);
			if (!Pkg) { Done.ExecuteIfBound(); return; }
			const FString Expected = FString::Printf(TEXT("synthetic %dx%d"), Pkg->Preprocess.ImageWidth, Pkg->Preprocess.ImageHeight);
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(PkgPath, 2, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("bench detection synthetic"));
			AwaitAll({ W }, Done, [this, W, Expected]()
			{
				const FString Js = W->Json().Left(1200);
				TestTrue(*FString::Printf(TEXT("job succeeded; got %s"), *Js), W->Ok());
				const TSharedPtr<FJsonObject> J = Parse(W->Json(), TEXT("bench detection synthetic"));
				TestEqual(*FString::Printf(TEXT("input; got %s"), *Js), GetStr(J, TEXT("input")), Expected);
				TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("ObjectDetection")));
			});
		});

		LatentIt("image package with a texture asset (object path and package-name forms): input 'image <object path>'", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			if (!ImportPkg(TEXT("import_test/tiny_image_classification"), TEXT("ImgCls"), PkgPath)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FWatch> A = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(PkgPath, 1, TEXT("cpu"), TArray<FString>(), DefaultTexturePath), TEXT("bench texture object path"));
			const TSharedRef<FWatch> B = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(PkgPath, 1, TEXT("cpu"), TArray<FString>(), DefaultTexturePackage), TEXT("bench texture package name"));
			AwaitAll({ A, B }, Done, [this, A, B]()
			{
				// Only structure: whether the engine's default texture is readable as model input is not what is tested; a per-device
				// error entry is still a successful job.
				for (const TSharedRef<FWatch>& W : { A, B })
				{
					const FString Js = W->Json().Left(1200);
					TestTrue(*FString::Printf(TEXT("[%s] job succeeded; got %s"), *W->Label, *Js), W->Ok());
					const TSharedPtr<FJsonObject> J = Parse(W->Json(), W->Label);
					TestEqual(*FString::Printf(TEXT("[%s] input (the object path in both forms); got %s"), *W->Label, *Js), GetStr(J, TEXT("input")),
						FString(TEXT("image ")) + DefaultTexturePath);
					TestEqual(*FString::Printf(TEXT("[%s] devices entries"), *W->Label), GetArr(J, TEXT("devices")).Num(), 1);
				}
			});
		});

		LatentIt("an Image given for a text task is ignored (input stays 'default text')", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), PkgPath)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(PkgPath, 1, TEXT("cpu"), TArray<FString>(), DefaultTexturePath), TEXT("bench text + image"));
			AwaitAll({ W }, Done, [this, W]()
			{
				TestTrue(*FString::Printf(TEXT("job succeeded; got %s"), *W->Json().Left(600)), W->Ok());
				TestEqual(TEXT("input"), GetStr(Parse(W->Json(), TEXT("bench text + image")), TEXT("input")), FString(TEXT("default text")));
			});
		});

		LatentIt("errors: missing package, Runs 0 / 1001 / -5, devices 'tpu', 65 texts, Image not a texture - failed jobs with the exact messages", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString Emb, Img;
			if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), Emb)) { Done.ExecuteIfBound(); return; }
			if (!ImportPkg(TEXT("import_test/tiny_image_classification"), TEXT("ImgCls"), Img)) { Done.ExecuteIfBound(); return; }
			TArray<FString> Many;
			for (int32 i = 0; i < 65; ++i) { Many.Add(FString::Printf(TEXT("t%d"), i)); }
			TArray<TSharedRef<FWatch>> Ws;
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(TEXT("/Game/Nope.Nope"), 5, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("missing package")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Emb, 0, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("runs 0")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Emb, 1001, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("runs 1001")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Emb, -5, TEXT("cpu"), TArray<FString>(), TEXT("")), TEXT("runs -5")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Emb, 3, TEXT("tpu"), TArray<FString>(), TEXT("")), TEXT("devices tpu")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Emb, 3, TEXT("cpu"), Many, TEXT("")), TEXT("65 texts")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Img, 3, TEXT("cpu"), TArray<FString>(), TEXT("/Game/Nope.Nope")), TEXT("image not found")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::BenchmarkModelPackage(Img, 3, TEXT("cpu"), TArray<FString>(), Emb), TEXT("image is a package asset")));
			AwaitAll(Ws, Done, [this, Ws, Emb]()
			{
				ExpectError(Ws[0], TEXT("Model package not found: /Game/Nope.Nope"));
				ExpectError(Ws[1], TEXT("runs must be 1 to 1000"));
				ExpectError(Ws[2], TEXT("runs must be 1 to 1000"));
				ExpectError(Ws[3], TEXT("runs must be 1 to 1000"));
				ExpectError(Ws[4], TEXT("devices must be cpu, gpu or both"));
				TestFalse(TEXT("[65 texts] bSuccess; got true, expected false"), Ws[5]->Ok());
				TestTrue(*FString::Printf(TEXT("[65 texts] error mentions the limit 64; got %s"), *Ws[5]->Json().Left(300)), GetStr(Parse(Ws[5]->Json(), Ws[5]->Label), TEXT("error")).Contains(TEXT("64")));
				ExpectError(Ws[6], TEXT("Image not found or not a texture: /Game/Nope.Nope"));
				ExpectError(Ws[7], FString(TEXT("Image not found or not a texture: ")) + Emb);
			});
		});
	});

	// ---------------------------------------------------------------------------------------------------------------------
	Describe("Sample", [this, LongTimeout]()
	{
		LatentIt("MiniLM embeddings: dim 384, norm ~1, head of 8, similarity matrix with diagonal 1 and related pair closer", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			if (!ImportPkg(TEXT("reference/all-MiniLM-L6-v2"), TEXT("MiniLM"), PkgPath)) { Done.ExecuteIfBound(); return; }
			const TArray<FString> Texts = { TEXT("A man is eating food."), TEXT("A man is eating a piece of bread."), TEXT("The girl is carrying a baby.") };
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(PkgPath, Texts, TEXT(""), 5, 0.5f), TEXT("sample MiniLM"));
			AwaitAll({ W }, Done, [this, W, Texts]()
			{
				const FString Js = W->Json().Left(1500);
				TestTrue(*FString::Printf(TEXT("job succeeded; got %s"), *Js), W->Ok());
				const TSharedPtr<FJsonObject> J = Parse(W->Json(), TEXT("sample MiniLM"));
				TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("TextEmbedding")));
				TestNotNull(*FString::Printf(TEXT("timings object; got %s"), *Js), GetObj(J, TEXT("timings")).Get());
				const TArray<TSharedPtr<FJsonValue>> Emb = GetArr(J, TEXT("embeddings"));
				if (!TestEqual(TEXT("embeddings count"), Emb.Num(), 3)) { return; }
				for (int32 i = 0; i < 3; ++i)
				{
					const TSharedPtr<FJsonObject> E = Emb[i]->AsObject();
					TestEqual(*FString::Printf(TEXT("embeddings[%d].text"), i), GetStr(E, TEXT("text")), Texts[i]);
					double Dim = -1, Norm = -1;
					GetNum(E, TEXT("dim"), Dim);
					GetNum(E, TEXT("norm"), Norm);
					TestEqual(*FString::Printf(TEXT("embeddings[%d].dim; got %.0f, expected 384"), i, Dim), (int32)Dim, 384);
					TestEqual(*FString::Printf(TEXT("embeddings[%d].norm of a normalized sentence embedding"), i), Norm, 1.0, 1e-3);
					const TArray<TSharedPtr<FJsonValue>> Head = GetArr(E, TEXT("head"));
					TestEqual(*FString::Printf(TEXT("embeddings[%d].head length (min(8, dim))"), i), Head.Num(), 8);
					bool bNonZero = false;
					for (const TSharedPtr<FJsonValue>& V : Head) { bNonZero |= FMath::Abs(V->AsNumber()) > 1e-9; }
					TestTrue(*FString::Printf(TEXT("embeddings[%d].head is not all zeros"), i), bNonZero);
				}
				const TArray<TSharedPtr<FJsonValue>> Sim = GetArr(J, TEXT("similarity"));
				if (!TestEqual(TEXT("similarity rows"), Sim.Num(), 3)) { return; }
				double M[3][3] = {};
				for (int32 r = 0; r < 3; ++r)
				{
					const TArray<TSharedPtr<FJsonValue>>& Row = Sim[r]->AsArray();
					if (!TestEqual(*FString::Printf(TEXT("similarity[%d] columns"), r), Row.Num(), 3)) { return; }
					for (int32 c = 0; c < 3; ++c) { M[r][c] = Row[c]->AsNumber(); }
				}
				for (int32 i = 0; i < 3; ++i)
				{
					TestEqual(*FString::Printf(TEXT("similarity[%d][%d] diagonal"), i, i), M[i][i], 1.0, 1e-5);
				}
				TestTrue(*FString::Printf(TEXT("similarity[0][1] %.6f > similarity[0][2] %.6f (man eating food ~ man eating bread, not girl with a baby)"), M[0][1], M[0][2]), M[0][1] > M[0][2]);
				TestEqual(TEXT("similarity is symmetric: [0][1] vs [1][0]"), M[0][1], M[1][0], 1e-6);
				TestEqual(TEXT("similarity is symmetric: [1][2] vs [2][1]"), M[1][2], M[2][1], 1e-6);
				// Rounded to 6 decimals.
				const double Scaled = M[0][1] * 1e6;
				TestTrue(*FString::Printf(TEXT("similarity[0][1] %.9f is rounded to 6 decimals"), M[0][1]), FMath::Abs(Scaled - FMath::RoundToDouble(Scaled)) < 1e-3);
			});
		});

		LatentIt("single text embedding: no similarity matrix; head length is min(8, dim); non-ASCII/emoji text echoed unchanged", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), PkgPath)) { Done.ExecuteIfBound(); return; }
			const FString Text = FString(TEXT("caf\u00e9 \u65e5\u672c\u8a9e \U0001F600 \"quoted\" \\ back"));
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(PkgPath, TArray<FString>{ Text }, TEXT(""), 5, 0.5f), TEXT("sample single"));
			AwaitAll({ W }, Done, [this, W, Text]()
			{
				const FString Js = W->Json().Left(1200);
				TestTrue(*FString::Printf(TEXT("job succeeded; got %s"), *Js), W->Ok());
				const TSharedPtr<FJsonObject> J = Parse(W->Json(), TEXT("sample single"));
				TestFalse(*FString::Printf(TEXT("similarity present for 1 text; got %s, expected the key absent (2+ texts only)"), *Js), J->HasField(TEXT("similarity")));
				const TArray<TSharedPtr<FJsonValue>> Emb = GetArr(J, TEXT("embeddings"));
				if (!TestEqual(TEXT("embeddings count"), Emb.Num(), 1)) { return; }
				const TSharedPtr<FJsonObject> E = Emb[0]->AsObject();
				TestEqual(TEXT("embeddings[0].text round-trips through the JSON"), GetStr(E, TEXT("text")), Text);
				double Dim = -1; GetNum(E, TEXT("dim"), Dim);
				TestTrue(*FString::Printf(TEXT("dim %.0f > 0"), Dim), Dim > 0);
				TestEqual(TEXT("head length min(8, dim)"), GetArr(E, TEXT("head")).Num(), FMath::Min(8, (int32)Dim));
				FTCHARToUTF8 Utf8(*W->Json());
				const FUTF8ToTCHAR Conv(Utf8.Get(), Utf8.Length());   // explicit length -> not null-terminated
				TestEqual(TEXT("output survives a UTF-8 round trip"), FString(Conv.Length(), Conv.Get()), W->Json());
			});
		});

		LatentIt("distilbert-sst2 'I love this!' -> top[0] POSITIVE index 1 with a high score; TopK clamps to 1..label count", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			if (!ImportPkg(TEXT("reference/distilbert-sst2"), TEXT("Sst2"), PkgPath)) { Done.ExecuteIfBound(); return; }
			const TArray<FString> Texts = { TEXT("I love this!") };
			const TSharedRef<FWatch> K1 = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(PkgPath, Texts, TEXT(""), 1, 0.5f), TEXT("sst2 topk 1"));
			const TSharedRef<FWatch> K99 = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(PkgPath, Texts, TEXT(""), 99, 0.5f), TEXT("sst2 topk 99"));
			const TSharedRef<FWatch> Kneg = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(PkgPath, Texts, TEXT(""), -3, 0.5f), TEXT("sst2 topk -3"));
			AwaitAll({ K1, K99, Kneg }, Done, [this, K1, K99, Kneg]()
			{
				const FString Js = K1->Json().Left(800);
				TestTrue(*FString::Printf(TEXT("job succeeded; got %s"), *Js), K1->Ok());
				const TSharedPtr<FJsonObject> J = Parse(K1->Json(), K1->Label);
				TestEqual(TEXT("task"), GetStr(J, TEXT("task")), FString(TEXT("TextClassification")));
				const TArray<TSharedPtr<FJsonValue>> Res = GetArr(J, TEXT("results"));
				if (TestEqual(TEXT("results count"), Res.Num(), 1))
				{
					const TSharedPtr<FJsonObject> R = Res[0]->AsObject();
					TestEqual(TEXT("results[0].text"), GetStr(R, TEXT("text")), FString(TEXT("I love this!")));
					const TArray<TSharedPtr<FJsonValue>> Top = GetArr(R, TEXT("top"));
					TestEqual(*FString::Printf(TEXT("top length with TopK 1; got %s"), *Js), Top.Num(), 1);
					if (Top.Num() >= 1)
					{
						const TSharedPtr<FJsonObject> T0 = Top[0]->AsObject();
						TestEqual(TEXT("top[0].label"), GetStr(T0, TEXT("label")), FString(TEXT("POSITIVE")));
						double Idx = -1, Score = -1;
						GetNum(T0, TEXT("index"), Idx);
						GetNum(T0, TEXT("score"), Score);
						TestEqual(TEXT("top[0].index (POSITIVE is class 1 in SST-2)"), (int32)Idx, 1);
						TestTrue(*FString::Printf(TEXT("top[0].score %.6f > 0.9 for an obviously positive sentence"), Score), Score > 0.9 && Score <= 1.0 + 1e-6);
					}
				}
				// SST-2 has 2 labels: TopK 99 clamps to 2 (descending), TopK -3 clamps to 1.
				const TArray<TSharedPtr<FJsonValue>> R99 = GetArr(Parse(K99->Json(), K99->Label), TEXT("results"));
				if (R99.Num() == 1)
				{
					const TArray<TSharedPtr<FJsonValue>> Top = GetArr(R99[0]->AsObject(), TEXT("top"));
					TestEqual(*FString::Printf(TEXT("top length with TopK 99; got %s"), *K99->Json().Left(500)), Top.Num(), 2);
					if (Top.Num() == 2)
					{
						TestTrue(TEXT("top is sorted by descending score"), GetStr(Top[0]->AsObject(), TEXT("label")) == TEXT("POSITIVE")
							&& Top[0]->AsObject()->GetNumberField(TEXT("score")) >= Top[1]->AsObject()->GetNumberField(TEXT("score")));
					}
				}
				else
				{
					AddError(FString::Printf(TEXT("TopK 99: results count expected 1; got %s"), *K99->Json().Left(500)));
				}
				const TArray<TSharedPtr<FJsonValue>> RNeg = GetArr(Parse(Kneg->Json(), Kneg->Label), TEXT("results"));
				if (RNeg.Num() == 1)
				{
					TestEqual(*FString::Printf(TEXT("top length with TopK -3; got %s"), *Kneg->Json().Left(500)), GetArr(RNeg[0]->AsObject(), TEXT("top")).Num(), 1);
				}
				else
				{
					AddError(FString::Printf(TEXT("TopK -3: results count expected 1; got %s"), *Kneg->Json().Left(500)));
				}
			});
		});

		LatentIt("tiny_text_classification with several texts including non-ASCII: one result per text, texts echoed in order", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString PkgPath;
			if (!ImportPkg(TEXT("import_test/tiny_text_classification"), TEXT("Cls"), PkgPath)) { Done.ExecuteIfBound(); return; }
			const TArray<FString> Texts = { TEXT("first"), FString(TEXT("\u65e5\u672c\u8a9e \U0001F600")), TEXT("third one") };
			const TSharedRef<FWatch> W = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(PkgPath, Texts, TEXT(""), 2, 0.5f), TEXT("sample cls texts"));
			AwaitAll({ W }, Done, [this, W, Texts]()
			{
				const FString Js = W->Json().Left(1200);
				TestTrue(*FString::Printf(TEXT("job succeeded; got %s"), *Js), W->Ok());
				const TSharedPtr<FJsonObject> J = Parse(W->Json(), TEXT("sample cls texts"));
				TestNotNull(TEXT("timings object"), GetObj(J, TEXT("timings")).Get());
				const TArray<TSharedPtr<FJsonValue>> Res = GetArr(J, TEXT("results"));
				if (!TestEqual(TEXT("results count (one per text)"), Res.Num(), 3)) { return; }
				for (int32 i = 0; i < 3; ++i)
				{
					const TSharedPtr<FJsonObject> R = Res[i]->AsObject();
					TestEqual(*FString::Printf(TEXT("results[%d].text"), i), GetStr(R, TEXT("text")), Texts[i]);
					const TArray<TSharedPtr<FJsonValue>> Top = GetArr(R, TEXT("top"));
					TestTrue(*FString::Printf(TEXT("results[%d].top length %d in 1..2"), i, Top.Num()), Top.Num() >= 1 && Top.Num() <= 2);
					for (const TSharedPtr<FJsonValue>& T : Top)
					{
						const TSharedPtr<FJsonObject> To = T->AsObject();
						TestTrue(*FString::Printf(TEXT("results[%d].top entry has label, index and score"), i),
							To->HasTypedField<EJson::String>(TEXT("label")) && To->HasTypedField<EJson::Number>(TEXT("index")) && To->HasTypedField<EJson::Number>(TEXT("score")));
					}
				}
			});
		});

		LatentIt("image tasks with a texture asset: success with image/top or detections structure, or a '<EPipelineError>: <message>' failure", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString Cls, Det;
			if (!ImportPkg(TEXT("import_test/tiny_image_classification"), TEXT("ImgCls"), Cls)) { Done.ExecuteIfBound(); return; }
			if (!ImportPkg(TEXT("import_test/tiny_object_detection"), TEXT("ImgDet"), Det)) { Done.ExecuteIfBound(); return; }
			const TSharedRef<FWatch> C = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Cls, TArray<FString>(), DefaultTexturePath, 3, 0.5f), TEXT("sample image cls"));
			const TSharedRef<FWatch> D = FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Det, TArray<FString>(), DefaultTexturePath, 3, 0.0f), TEXT("sample detection"));
			AwaitAll({ C, D }, Done, [this, C, D]()
			{
				// Only structure is checked (labels of a random tiny model on an engine texture mean nothing).
				const TSharedPtr<FJsonObject> JC = Parse(C->Json(), C->Label);
				if (C->Ok())
				{
					TestEqual(TEXT("[image cls] task"), GetStr(JC, TEXT("task")), FString(TEXT("ImageClassification")));
					TestEqual(TEXT("[image cls] image (object path)"), GetStr(JC, TEXT("image")), FString(DefaultTexturePath));
					TestNotNull(TEXT("[image cls] timings"), GetObj(JC, TEXT("timings")).Get());
					const TArray<TSharedPtr<FJsonValue>> Top = GetArr(JC, TEXT("top"));
					TestTrue(*FString::Printf(TEXT("[image cls] top length %d in 1..3 (TopK 3)"), Top.Num()), Top.Num() >= 1 && Top.Num() <= 3);
					double Prev = 2.0;
					for (const TSharedPtr<FJsonValue>& T : Top)
					{
						const TSharedPtr<FJsonObject> To = T->AsObject();
						TestTrue(TEXT("[image cls] top entry has label, index and score"),
							To->HasTypedField<EJson::String>(TEXT("label")) && To->HasTypedField<EJson::Number>(TEXT("index")) && To->HasTypedField<EJson::Number>(TEXT("score")));
						double S = 0; GetNum(To, TEXT("score"), S);
						TestTrue(*FString::Printf(TEXT("[image cls] scores descending: %.6f <= previous %.6f"), S, Prev), S <= Prev);
						Prev = S;
					}
				}
				else
				{
					AddInfo(FString::Printf(TEXT("[image cls] the sample failed (texture input may not be readable headless); only the error format is checked. Got %s"), *C->Json().Left(300)));
					TestTrue(*FString::Printf(TEXT("[image cls] error '%s' has the form '<EPipelineError>: <message>'"), *GetStr(JC, TEXT("error"))), GetStr(JC, TEXT("error")).Contains(TEXT(": ")));
				}

				const TSharedPtr<FJsonObject> JD = Parse(D->Json(), D->Label);
				if (D->Ok())
				{
					TestEqual(TEXT("[detection] task"), GetStr(JD, TEXT("task")), FString(TEXT("ObjectDetection")));
					TestEqual(TEXT("[detection] image (object path)"), GetStr(JD, TEXT("image")), FString(DefaultTexturePath));
					TestNotNull(TEXT("[detection] timings"), GetObj(JD, TEXT("timings")).Get());
					TestTrue(*FString::Printf(TEXT("[detection] detections is an array; got %s"), *D->Json().Left(300)), JD->HasTypedField<EJson::Array>(TEXT("detections")));
					double Prev = 2.0;
					for (const TSharedPtr<FJsonValue>& V : GetArr(JD, TEXT("detections")))
					{
						const TSharedPtr<FJsonObject> De = V->AsObject();
						TestTrue(TEXT("[detection] entry has label, index, score"),
							De->HasTypedField<EJson::String>(TEXT("label")) && De->HasTypedField<EJson::Number>(TEXT("index")) && De->HasTypedField<EJson::Number>(TEXT("score")));
						TestEqual(TEXT("[detection] box has 4 numbers [x0,y0,x1,y1]"), GetArr(De, TEXT("box")).Num(), 4);
						double S = 0; GetNum(De, TEXT("score"), S);
						TestTrue(*FString::Printf(TEXT("[detection] score-descending: %.6f <= previous %.6f"), S, Prev), S <= Prev);
						TestTrue(*FString::Printf(TEXT("[detection] score %.6f >= threshold 0"), S), S >= 0.0);
						Prev = S;
					}
				}
				else
				{
					AddInfo(FString::Printf(TEXT("[detection] the sample failed (texture input may not be readable headless); only the error format is checked. Got %s"), *D->Json().Left(300)));
					TestTrue(*FString::Printf(TEXT("[detection] error '%s' has the form '<EPipelineError>: <message>'"), *GetStr(JD, TEXT("error"))), GetStr(JD, TEXT("error")).Contains(TEXT(": ")));
				}
			});
		});

		LatentIt("errors: missing package, missing inputs per task, 65 texts, score_threshold 1.5, bad image - failed jobs with the exact messages", LongTimeout, [this](const FDoneDelegate& Done)
		{
			FString Emb, Cls, ImgCls, Det;
			if (!ImportPkg(TEXT("import_test/tiny_text_embedding"), TEXT("Emb"), Emb)) { Done.ExecuteIfBound(); return; }
			if (!ImportPkg(TEXT("import_test/tiny_text_classification"), TEXT("Cls"), Cls)) { Done.ExecuteIfBound(); return; }
			if (!ImportPkg(TEXT("import_test/tiny_image_classification"), TEXT("ImgCls"), ImgCls)) { Done.ExecuteIfBound(); return; }
			if (!ImportPkg(TEXT("import_test/tiny_object_detection"), TEXT("ImgDet"), Det)) { Done.ExecuteIfBound(); return; }
			TArray<FString> Many;
			for (int32 i = 0; i < 65; ++i) { Many.Add(FString::Printf(TEXT("t%d"), i)); }
			TArray<TSharedRef<FWatch>> Ws;
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(TEXT("/Game/Nope.Nope"), TArray<FString>{ TEXT("x") }, TEXT(""), 3, 0.5f), TEXT("0 missing package")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(ImgCls, TArray<FString>{ TEXT("text for an image task") }, TEXT(""), 3, 0.5f), TEXT("1 image cls without image")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Det, TArray<FString>{ TEXT("text for an image task") }, TEXT(""), 3, 0.5f), TEXT("2 detection without image")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Emb, Many, TEXT(""), 3, 0.5f), TEXT("3 embedding 65 texts")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Det, TArray<FString>(), DefaultTexturePath, 3, 1.5f), TEXT("4 score_threshold 1.5")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Det, TArray<FString>(), DefaultTexturePath, 3, -0.1f), TEXT("5 score_threshold -0.1")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Emb, TArray<FString>(), TEXT(""), 3, 0.5f), TEXT("6 embedding no texts")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Cls, TArray<FString>(), TEXT(""), 3, 0.5f), TEXT("7 classification no texts")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(Cls, Many, TEXT(""), 3, 0.5f), TEXT("8 classification 65 texts")));
			Ws.Add(FWatch::Make(UPipelineMcpLibrary::RunPipelineSample(ImgCls, TArray<FString>(), TEXT("/Game/Nope.Nope"), 3, 0.5f), TEXT("9 image not found")));
			AwaitAll(Ws, Done, [this, Ws]()
			{
				ExpectError(Ws[0], TEXT("Model package not found: /Game/Nope.Nope"));
				ExpectError(Ws[1], TEXT("ImageClassification needs image (a texture asset path)"));
				ExpectError(Ws[2], TEXT("ObjectDetection needs image (a texture asset path)"));
				ExpectError(Ws[3], TEXT("texts: give 1 to 64 strings"));
				ExpectError(Ws[4], TEXT("score_threshold must be 0 to 1"));
				ExpectError(Ws[5], TEXT("score_threshold must be 0 to 1"));
				// Empty Texts: the brief names both "TextEmbedding needs texts" (missing input) and "texts: give 1 to 64 strings"; either one is accepted here.
				for (int32 i : { 6, 7 })
				{
					const FString Expected = i == 6 ? TEXT("TextEmbedding needs texts") : TEXT("TextClassification needs texts");
					TestFalse(*FString::Printf(TEXT("[%s] bSuccess; got true, expected false"), *Ws[i]->Label), Ws[i]->Ok());
					const FString Err = GetStr(Parse(Ws[i]->Json(), Ws[i]->Label), TEXT("error"));
					TestTrue(*FString::Printf(TEXT("[%s] error; got '%s', expected '%s' (or 'texts: give 1 to 64 strings')"), *Ws[i]->Label, *Err, *Expected),
						Err == Expected || Err == TEXT("texts: give 1 to 64 strings"));
				}
				ExpectError(Ws[8], TEXT("texts: give 1 to 64 strings"));
				TestFalse(TEXT("[9 image not found] bSuccess; got true, expected false"), Ws[9]->Ok());
				TestTrue(*FString::Printf(TEXT("[9 image not found] error names the image argument; got %s"), *Ws[9]->Json().Left(300)),
					GetStr(Parse(Ws[9]->Json(), Ws[9]->Label), TEXT("error")).Contains(TEXT("/Game/Nope.Nope")));
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
