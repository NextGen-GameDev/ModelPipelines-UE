// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Array.h"
#include "Containers/Ticker.h"
#include "Containers/UnrealString.h"
#include "HAL/PlatformTime.h"
#include "Misc/Guid.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UObjectGlobals.h"
#include "PipelineModelPackage.h"
#include "PipelineMcpJob.h"
#include "PipelineMcpLibrary.h"
#include "PipelineMcpTestReceiver.h"

BEGIN_DEFINE_SPEC(FPipelineMcpTextGenSpec, "Pipelines.MCP.TextGeneration",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	TStrongObjectPtr<UPackage> Pkg;
	TStrongObjectPtr<UPipelineModelPackage> Asset;
	TStrongObjectPtr<UPipelineMcpTestReceiver> Recv;
	TWeakObjectPtr<UPipelineMcpJob> Job;
	FString ObjPath;
	/** Polls (core ticker, never blocks) until the receiver got one broadcast or the timeout passed, then checks the message. */
	void AwaitAndCheck(const FDoneDelegate& Done, const TCHAR* What);
END_DEFINE_SPEC(FPipelineMcpTextGenSpec)

void FPipelineMcpTextGenSpec::AwaitAndCheck(const FDoneDelegate& Done, const TCHAR* What)
{
	const double Start = FPlatformTime::Seconds();
	FTSTicker::GetCoreTicker().AddTicker(TEXT("PipelineMcpTextGenPoll"), 0.0f, [this, Done, What, Start](float)
	{
		const bool bGot = Recv.IsValid() && Recv->Count >= 1;
		if (!bGot && FPlatformTime::Seconds() - Start < 30.0)
		{
			return true;
		}
		if (!bGot)
		{
			AddError(FString::Printf(TEXT("[%s] OnDone never fired within 30 s"), What));
		}
		else
		{
			const FString Json = Recv->Jsons.Num() > 0 ? Recv->Jsons[0] : FString();
			TestTrue(*FString::Printf(TEXT("[%s] the job fails (a TextGeneration package has no runnable pipeline in this plugin)"), What), Recv->Successes.Num() > 0 && !Recv->Successes[0]);
			TestTrue(*FString::Printf(TEXT("[%s] message names the task 'TextGeneration'; got %s"), What, *Json.Left(400)), Json.Contains(TEXT("TextGeneration")));
			TestFalse(*FString::Printf(TEXT("[%s] must not say the task is None; got %s"), What, *Json.Left(400)), Json.Contains(TEXT("task is None")));
		}
		Done.ExecuteIfBound();
		return false;
	});
}

void FPipelineMcpTextGenSpec::Define()
{
	BeforeEach([this]()
	{
		// An unsaved in-memory package, found by object path like an imported asset.
		const FString PkgName = FString::Printf(TEXT("/Game/__Scratch/WP8L6a_%s"), *FGuid::NewGuid().ToString(EGuidFormats::Digits));
		Pkg.Reset(CreatePackage(*PkgName));
		Asset.Reset(NewObject<UPipelineModelPackage>(Pkg.Get(), TEXT("TextGen"), RF_Public | RF_Standalone));
		Asset->Task = EPipelineTask::TextGeneration;   // Model and GgufModel stay null: only the task matters for the message
		ObjPath = PkgName + TEXT(".TextGen");
		Recv.Reset(NewObject<UPipelineMcpTestReceiver>(GetTransientPackage()));
	});
	AfterEach([this]()
	{
		if (Job.IsValid() && Recv.IsValid()) { Job->OnDone.RemoveAll(Recv.Get()); }
		Asset.Reset();
		Pkg.Reset();
		Recv.Reset();
	});

	LatentIt("RunPipelineSample on a TextGeneration package fails and names 'TextGeneration' (not 'None')", FTimespan::FromSeconds(60.0), [this](const FDoneDelegate& Done)
	{
		UPipelineMcpJob* J = UPipelineMcpLibrary::RunPipelineSample(ObjPath, TArray<FString>{ TEXT("hello") }, TEXT(""), 3, 0.5f);
		if (!J) { AddError(TEXT("RunPipelineSample returned nullptr on the game thread")); Done.ExecuteIfBound(); return; }
		Job = J;
		J->OnDone.AddDynamic(Recv.Get(), &UPipelineMcpTestReceiver::OnDone);
		AwaitAndCheck(Done, TEXT("sample"));
	});

	LatentIt("BenchmarkModelPackage on a TextGeneration package fails and names 'TextGeneration' (not 'None')", FTimespan::FromSeconds(60.0), [this](const FDoneDelegate& Done)
	{
		UPipelineMcpJob* J = UPipelineMcpLibrary::BenchmarkModelPackage(ObjPath, 1, TEXT("cpu"), TArray<FString>{ TEXT("hello") }, TEXT(""));
		if (!J) { AddError(TEXT("BenchmarkModelPackage returned nullptr on the game thread")); Done.ExecuteIfBound(); return; }
		Job = J;
		J->OnDone.AddDynamic(Recv.Get(), &UPipelineMcpTestReceiver::OnDone);
		AwaitAndCheck(Done, TEXT("benchmark"));
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
