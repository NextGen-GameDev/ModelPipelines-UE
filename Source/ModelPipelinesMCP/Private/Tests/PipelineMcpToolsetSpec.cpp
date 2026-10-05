// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
//
// Pipelines.MCP.Toolset.* (WP6.3, contract C39/C40): drives the Python toolset
// model_pipelines_mcp.toolset.ModelPipelinesTools through unreal.ToolsetRegistry.execute_tool, with no module dependency on
// Python. Each call writes a small .py script under <Saved>/PipelinesTests/MCP/<guid>.py, runs it with the `py` console command,
// and polls for <guid>.json ({"complete":..,"error":..,"value":..}) from an FTSTicker with a timeout.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "AssetRegistry/AssetData.h"
#include "AssetRegistry/IAssetRegistry.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Containers/Array.h"
#include "Containers/Ticker.h"
#include "Containers/UnrealString.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Engine/Engine.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformTime.h"
#include "Math/UnrealMathUtility.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"

namespace PipelineMcpToolsetSpecPrivate
{
	static const TCHAR* ToolsetName = TEXT("model_pipelines_mcp.toolset.ModelPipelinesTools");

	/** What the Python script wrote: complete flag, tool error text, and the tool value (a string value that is itself JSON
	    is parsed once more into Obj, because every tool returns the C38 JSON as a string). */
	struct FToolOut
	{
		bool bFileRead = false;
		bool bComplete = false;
		FString Error;
		TSharedPtr<FJsonValue> Value;
		TSharedPtr<FJsonObject> Obj;
		FString Raw;
	};

	static FString GetDevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		if (!Env.IsEmpty())
		{
			return Env;
		}
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev"));
	}

	static bool IsMcpRequired()
	{
		return FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_MCP")) == TEXT("1");
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

	static TSharedPtr<FJsonObject> GetObj(const TSharedPtr<FJsonObject>& O, const TCHAR* Key)
	{
		const TSharedPtr<FJsonObject>* P = nullptr;
		if (O.IsValid() && O->TryGetObjectField(Key, P) && P)
		{
			return *P;
		}
		return nullptr;
	}

	static bool GetBool(const TSharedPtr<FJsonObject>& O, const TCHAR* Key, bool bDefault = false)
	{
		bool B = bDefault;
		if (O.IsValid())
		{
			O->TryGetBoolField(Key, B);
		}
		return B;
	}

	static FString ObjPath(const FString& Folder, const FString& Name)
	{
		return Folder / Name + TEXT(".") + Name;
	}

	static FString Fwd(const FString& P)
	{
		return FPaths::ConvertRelativePathToFull(P).Replace(TEXT("\\"), TEXT("/"));
	}

	static FString Tail(const FString& S, int32 N = 500)
	{
		return S.Len() > N ? S.Left(N) + TEXT("...") : S;
	}

	static void ParseOut(const FString& Text, FToolOut& Out)
	{
		Out.Raw = Text;
		TSharedPtr<FJsonObject> Root;
		TSharedRef<TJsonReader<>> R = TJsonReaderFactory<>::Create(Text);
		if (!FJsonSerializer::Deserialize(R, Root) || !Root.IsValid())
		{
			return;
		}
		Out.bFileRead = true;
		Root->TryGetBoolField(TEXT("complete"), Out.bComplete);
		Root->TryGetStringField(TEXT("error"), Out.Error);
		Out.Value = Root->TryGetField(TEXT("value"));
		if (Out.Value.IsValid())
		{
			if (Out.Value->Type == EJson::Object)
			{
				Out.Obj = Out.Value->AsObject();
			}
			else if (Out.Value->Type == EJson::String)
			{
				TSharedRef<TJsonReader<>> R2 = TJsonReaderFactory<>::Create(Out.Value->AsString());
				TSharedPtr<FJsonObject> Inner;
				if (FJsonSerializer::Deserialize(R2, Inner) && Inner.IsValid())
				{
					Out.Obj = Inner;
				}
			}
		}
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

	// The Python runner. @OUT@ and @ARGS@ are raw file paths (forward slashes), @MODE@ is "tool" or "schema", @TOOL@ the tool name.
	static const TCHAR* ScriptTemplate = TEXT(R"PY(import json
import os
import traceback
import unreal

OUT = r"@OUT@"
ARGS = r"@ARGS@"
MODE = "@MODE@"
TOOL = "@TOOL@"
TOOLSET = "@TOOLSET@"
_state = {"written": False}


def _write(complete, error, value):
    if _state["written"]:
        return
    _state["written"] = True
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"complete": bool(complete), "error": str(error), "value": value}, f)
    os.replace(tmp, OUT)


try:
    if MODE == "schema":
        reg = unreal.ToolsetRegistry
        registered = bool(reg.is_toolset_registered(TOOLSET))
        tools = {}
        found = False
        for ts in json.loads(reg.get_all_toolset_json_schemas()):
            if ts.get("name") == TOOLSET:
                found = True
                for t in ts.get("tools", []):
                    n = t.get("name", "")
                    if n.startswith(TOOLSET + "."):
                        n = n[len(TOOLSET) + 1:]
                    props = t.get("inputSchema", {}).get("properties", {})
                    tools[n] = sorted(props.keys())
        _write(True, "", {"registered": registered, "found_in_schemas": found, "tools": tools})
    else:
        with open(ARGS, "r", encoding="utf-8") as f:
            args = f.read()
        res = unreal.ToolsetRegistry.execute_tool(TOOLSET, TOOL, args)

        def _done(r=None):
            try:
                complete = None
                for name in ("is_complete", "b_is_complete"):
                    try:
                        complete = res.get_editor_property(name)
                        break
                    except Exception:
                        pass
                vs = res.get_value_as_json_string()
                value = None
                if vs:
                    try:
                        value = json.loads(vs)
                    except Exception:
                        value = vs
                    # The registry wraps every tool value as {"returnValue": <value>} (ObjectFunctionToolCall.cpp:502-520).
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except Exception:
                            pass
                    if isinstance(value, dict) and "returnValue" in value:
                        value = value["returnValue"]
                _write(True if complete is None else complete, res.get_editor_property("error"), value)
            except Exception:
                _write(False, "callback exception: " + traceback.format_exc(), None)

        if res is None:
            _write(False, "execute_tool returned None", None)
        else:
            # A Python callable bound to a dynamic delegate lives in a proxy UObject that only a live delegate *wrapper*
            # keeps alive (PyWrapperDelegate.cpp:206-212): keep the wrapper (and res, _done) in a persistent place.
            delegate = res.on_completed
            delegate.add_callable(_done)
            import builtins
            builtins.__dict__.setdefault("_model_pipelines_mcp_test_keep", []).append((res, delegate, _done))
            # Force a GC while the call is in flight: a tool that doesn't keep its own callbacks alive never completes.
            unreal.SystemLibrary.collect_garbage()
            res.broadcast_on_completed_if_complete()
except Exception:
    _write(False, "script exception: " + traceback.format_exc(), None)
)PY");
}

BEGIN_DEFINE_SPEC(FPipelineMcpToolsetSpec, "Pipelines.MCP.Toolset",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	FString Dest;
	FString DestGuid;
	TArray<FString> TempFiles;

	/** true when PythonScriptPlugin and ToolsetRegistry are loaded; else info (skip) or, with MODELPIPELINES_REQUIRE_MCP=1, an error. */
	bool CanRun();
	/** Writes and runs the script, polls (FTSTicker) for the result json, then calls OnDone on the game thread. */
	void RunPy(const FString& Mode, const FString& Tool, const FString& ArgsJson, double TimeoutSec,
		TFunction<void(const PipelineMcpToolsetSpecPrivate::FToolOut&)> OnDone);
	void RunTool(const FString& Tool, const FString& ArgsJson, double TimeoutSec,
		TFunction<void(const PipelineMcpToolsetSpecPrivate::FToolOut&)> OnDone)
	{
		// C39 amendment (2026-10-05): `texts` is a required list in the UE 5.8 schema (an empty default is dropped); callers that
		// pass no texts send [] like an agent following the docstring would.
		FString Args = ArgsJson;
		if ((Tool == TEXT("benchmark_model_package") || Tool == TEXT("run_pipeline_sample")) && !Args.Contains(TEXT("\"texts\"")))
		{
			const int32 Open = Args.Find(TEXT("{"));
			if (Open != INDEX_NONE)
			{
				const bool bEmpty = Args.Mid(Open + 1).TrimStart().StartsWith(TEXT("}"));
				Args.InsertAt(Open + 1, bEmpty ? TEXT("\"texts\":[]") : TEXT("\"texts\":[],"));
			}
		}
		RunPy(TEXT("tool"), Tool, Args, TimeoutSec, MoveTemp(OnDone));
	}
	/** <dev>/models/<Sub>/<Name>; AddError (with the fetch command) and false when missing. */
	bool RequireFolder(const FString& Sub, const FString& Name, FString& OutFolder);
	/** import_model_folder confirm=true into Dest; calls OnOk(package object path) on success, else AddError + Done. */
	void ImportThen(const FString& Folder, const FString& AssetName, const FDoneDelegate& Done, TFunction<void(const FString&)> OnOk);
	/** Python tool errors are logged by LogPython as errors; ignore (narrowly) the ones this test provokes. */
	void IgnoreToolErrorLogs(const FString& MessageFragment);
	/** Fails with the raw output unless the script finished and the tool completed without error. */
	bool CheckOk(const PipelineMcpToolsetSpecPrivate::FToolOut& Out, const FString& What);
	/** Fails unless the tool completed with an error containing Fragment. */
	void CheckToolError(const PipelineMcpToolsetSpecPrivate::FToolOut& Out, const FString& Fragment, const FString& What);
	void Cleanup();
END_DEFINE_SPEC(FPipelineMcpToolsetSpec)

bool FPipelineMcpToolsetSpec::CanRun()
{
	using namespace PipelineMcpToolsetSpecPrivate;
	const bool bPy = FModuleManager::Get().IsModuleLoaded("PythonScriptPlugin");
	const bool bReg = FModuleManager::Get().IsModuleLoaded("ToolsetRegistry");
	if (bPy && bReg)
	{
		return true;
	}
	const FString Why = FString::Printf(TEXT("PythonScriptPlugin loaded=%d, ToolsetRegistry loaded=%d (UE 5.8 only; both plugins enabled in the test project)"),
		bPy ? 1 : 0, bReg ? 1 : 0);
	if (IsMcpRequired())
	{
		AddError(TEXT("MODELPIPELINES_REQUIRE_MCP=1 but the toolset can't be driven: ") + Why);
	}
	else
	{
		AddInfo(TEXT("SKIPPED (set MODELPIPELINES_REQUIRE_MCP=1 to make this fail): ") + Why);
	}
	return false;
}

void FPipelineMcpToolsetSpec::RunPy(const FString& Mode, const FString& Tool, const FString& ArgsJson, double TimeoutSec,
	TFunction<void(const PipelineMcpToolsetSpecPrivate::FToolOut&)> OnDone)
{
	using namespace PipelineMcpToolsetSpecPrivate;
	const FString Guid = FGuid::NewGuid().ToString(EGuidFormats::Digits);
	const FString Dir = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("PipelinesTests") / TEXT("MCP"));
	IFileManager::Get().MakeDirectory(*Dir, true);
	const FString PyPath = Dir / (Guid + TEXT(".py"));
	const FString OutPath = Dir / (Guid + TEXT(".json"));
	const FString ArgsPath = Dir / (Guid + TEXT(".args.json"));
	TempFiles.Add(PyPath);
	TempFiles.Add(OutPath);
	TempFiles.Add(OutPath + TEXT(".tmp"));
	TempFiles.Add(ArgsPath);

	FString Script = ScriptTemplate;
	Script = Script.Replace(TEXT("@OUT@"), *Fwd(OutPath)).Replace(TEXT("@ARGS@"), *Fwd(ArgsPath))
		.Replace(TEXT("@MODE@"), *Mode).Replace(TEXT("@TOOL@"), *Tool).Replace(TEXT("@TOOLSET@"), ToolsetName);
	// UTF-8 without BOM so the args (non-ASCII safe) and the script read back exactly.
	const bool bW1 = FFileHelper::SaveStringToFile(ArgsJson, *ArgsPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
	const bool bW2 = FFileHelper::SaveStringToFile(Script, *PyPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
	if (!bW1 || !bW2)
	{
		AddError(FString::Printf(TEXT("could not write the test script/args under %s"), *Dir));
		FToolOut Fail;
		OnDone(Fail);
		return;
	}

	const FString Cmd = FString::Printf(TEXT("py \"%s\""), *Fwd(PyPath));
	if (!GEngine || !GEngine->Exec(nullptr, *Cmd))
	{
		AddError(FString::Printf(TEXT("the console command was not handled (is PythonScriptPlugin's `py` command available?): %s"), *Cmd));
		FToolOut Fail;
		OnDone(Fail);
		return;
	}
	const double Started = FPlatformTime::Seconds();
	const double Deadline = Started + TimeoutSec;
	FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateLambda(
		[this, OutPath, Started, Deadline, TimeoutSec, Tool, Mode, OnDone](float) -> bool
		{
			// The registry runs the tool on a later main-thread task, so its callbacks get bound after `py` returned. A synchronous
			// full GC on every poll during the first 2 s: an async tool that doesn't keep its Python callback's delegate wrapper
			// alive loses the callback and never completes (C39, WP6.3 review).
			if (FPlatformTime::Seconds() - Started < 2.0)
			{
				CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			}
			FString Text;
			if (IFileManager::Get().FileExists(*OutPath) && FFileHelper::LoadFileToString(Text, *OutPath))
			{
				FToolOut Out;
				ParseOut(Text, Out);
				if (!Out.bFileRead)
				{
					AddError(FString::Printf(TEXT("result file for '%s' is not valid JSON: %s"), *Tool, *Tail(Text)));
				}
				OnDone(Out);
				return false;
			}
			if (FPlatformTime::Seconds() > Deadline)
			{
				AddError(FString::Printf(TEXT("timed out after %.0f s waiting for the Python result of %s '%s' (the tool never completed, or the script failed before writing; grep the run log for LogPython)"),
					TimeoutSec, *Mode, *Tool));
				FToolOut Out;
				OnDone(Out);
				return false;
			}
			return true;
		}), 0.05f);
}

bool FPipelineMcpToolsetSpec::RequireFolder(const FString& Sub, const FString& Name, FString& OutFolder)
{
	using namespace PipelineMcpToolsetSpecPrivate;
	OutFolder = FPaths::ConvertRelativePathToFull(GetDevDir() / TEXT("models") / Sub / Name);
	if (IFileManager::Get().DirectoryExists(*OutFolder))
	{
		return true;
	}
	AddError(FString::Printf(TEXT("Missing test model folder %s. Fetch with: \"D:/Micromamba/micromambaenv/envs/pipelines/python.exe\" \"D:/Game Dev/Pipelines-UE/ModelPipelines-UE/Tools/%s\""),
		*OutFolder, Sub == TEXT("reference") ? TEXT("fetch_reference_models.py") : TEXT("make_test_models.py")));
	return false;
}

void FPipelineMcpToolsetSpec::ImportThen(const FString& Folder, const FString& AssetName, const FDoneDelegate& Done, TFunction<void(const FString&)> OnOk)
{
	using namespace PipelineMcpToolsetSpecPrivate;
	const FString Args = FString::Printf(TEXT("{\"source_folder\":\"%s\",\"destination_path\":\"%s\",\"asset_name\":\"%s\",\"confirm\":true}"),
		*Folder.Replace(TEXT("\\"), TEXT("/")), *Dest, *AssetName);
	RunTool(TEXT("import_model_folder"), Args, 120.0, [this, Done, OnOk, AssetName](const FToolOut& Out)
	{
		if (!CheckOk(Out, TEXT("setup import_model_folder confirm=true")) || !Out.Obj.IsValid() || !GetBool(Out.Obj, TEXT("success")))
		{
			AddError(FString::Printf(TEXT("setup import did not succeed: %s"), *Tail(Out.Raw)));
			Done.ExecuteIfBound();
			return;
		}
		OnOk(GetStr(Out.Obj, TEXT("package")));
	});
}

void FPipelineMcpToolsetSpec::IgnoreToolErrorLogs(const FString& MessageFragment)
{
	// Occurrences < 0: silently ignored when present, no failure when absent (the Python log text is the engine's, not ours).
	AddExpectedError(MessageFragment, EAutomationExpectedErrorFlags::Contains, -1, false);
	AddExpectedError(TEXT("RuntimeError"), EAutomationExpectedErrorFlags::Contains, -1, false);
	AddExpectedError(TEXT("Traceback"), EAutomationExpectedErrorFlags::Contains, -1, false);
	AddExpectedError(TEXT("model_pipelines_mcp"), EAutomationExpectedErrorFlags::Contains, -1, false);
	AddExpectedError(TEXT("toolset_registry"), EAutomationExpectedErrorFlags::Contains, -1, false);
}

bool FPipelineMcpToolsetSpec::CheckOk(const PipelineMcpToolsetSpecPrivate::FToolOut& Out, const FString& What)
{
	using namespace PipelineMcpToolsetSpecPrivate;
	if (!Out.bFileRead)
	{
		AddError(FString::Printf(TEXT("[%s] no result from the Python script"), *What));
		return false;
	}
	if (!TestTrue(FString::Printf(TEXT("[%s] result complete; raw=%s"), *What, *Tail(Out.Raw)), Out.bComplete))
	{
		return false;
	}
	return TestEqual(FString::Printf(TEXT("[%s] no tool error; raw=%s"), *What, *Tail(Out.Raw)), Out.Error, FString());
}

void FPipelineMcpToolsetSpec::CheckToolError(const PipelineMcpToolsetSpecPrivate::FToolOut& Out, const FString& Fragment, const FString& What)
{
	using namespace PipelineMcpToolsetSpecPrivate;
	if (!Out.bFileRead)
	{
		AddError(FString::Printf(TEXT("[%s] no result from the Python script"), *What));
		return;
	}
	TestTrue(FString::Printf(TEXT("[%s] the call completes (an error must complete the async result, not leave it pending); raw=%s"), *What, *Tail(Out.Raw)), Out.bComplete);
	TestTrue(FString::Printf(TEXT("[%s] error contains '%s'; got '%s'"), *What, *Fragment, *Tail(Out.Error)), Out.Error.Contains(Fragment));
}

void FPipelineMcpToolsetSpec::Cleanup()
{
	using namespace PipelineMcpToolsetSpecPrivate;
	if (!Dest.IsEmpty() && FModuleManager::Get().IsModuleLoaded("AssetRegistry"))
	{
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
		for (UObject* O : Objs)
		{
			DiscardOne(O);
		}
		// Test cleanup only: the product never deletes. import_model_folder(confirm=true) saves .uasset files here.
		const FString ContentDir = FPaths::ConvertRelativePathToFull(FPaths::ProjectContentDir() / TEXT("PipelinesTests") / (TEXT("MCP_") + DestGuid));
		IFileManager::Get().DeleteDirectory(*ContentDir, false, true);
	}
	for (const FString& F : TempFiles)
	{
		IFileManager::Get().Delete(*F, false, true, true);
	}
	TempFiles.Reset();
}

void FPipelineMcpToolsetSpec::Define()
{
	using namespace PipelineMcpToolsetSpecPrivate;

	BeforeEach([this]()
	{
		DestGuid = FGuid::NewGuid().ToString(EGuidFormats::Digits).Left(8);
		Dest = FString::Printf(TEXT("/Game/PipelinesTests/MCP_%s"), *DestGuid);
	});

	AfterEach([this]()
	{
		Cleanup();
	});

	Describe("Registration", [this]()
	{
		LatentIt("the toolset is registered and lists exactly the five tools with the snake_case parameters", FTimespan::FromSeconds(90), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			RunPy(TEXT("schema"), TEXT(""), TEXT(""), 60.0, [this, Done](const FToolOut& Out)
			{
				// Catches: init_unreal.py not registering (toolset missing), a renamed tool, a camelCase parameter
				// (sourceFolder) or an extra tool beyond the fixed five.
				if (CheckOk(Out, TEXT("schema")) && Out.Obj.IsValid())
				{
					TestTrue(TEXT("unreal.ToolsetRegistry.is_toolset_registered(model_pipelines_mcp.toolset.ModelPipelinesTools)"), GetBool(Out.Obj, TEXT("registered")));
					TestTrue(TEXT("the toolset appears in get_all_toolset_json_schemas()"), GetBool(Out.Obj, TEXT("found_in_schemas")));
					TSharedPtr<FJsonObject> Tools = GetObj(Out.Obj, TEXT("tools"));
					if (TestTrue(TEXT("schema has a tools map"), Tools.IsValid()))
					{
						TestEqual(TEXT("exactly 5 tools"), Tools->Values.Num(), 5);
						struct FExpect { const TCHAR* Tool; TArray<FString> Params; };
						const TArray<FExpect> Expected = {
							{ TEXT("list_model_packages"), { TEXT("path"), TEXT("recursive") } },
							{ TEXT("import_model_folder"), { TEXT("asset_name"), TEXT("confirm"), TEXT("destination_path"), TEXT("overwrite_existing"), TEXT("source_folder") } },
							{ TEXT("validate_model_package"), { TEXT("check_gpu"), TEXT("package"), TEXT("run_smoke_test") } },
							{ TEXT("benchmark_model_package"), { TEXT("devices"), TEXT("image"), TEXT("package"), TEXT("runs"), TEXT("texts") } },
							{ TEXT("run_pipeline_sample"), { TEXT("image"), TEXT("package"), TEXT("score_threshold"), TEXT("texts"), TEXT("top_k") } },
						};
						for (const FExpect& E : Expected)
						{
							const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
							if (!TestTrue(FString::Printf(TEXT("tool %s exists"), E.Tool), Tools->TryGetArrayField(E.Tool, Arr) && Arr))
							{
								continue;
							}
							TArray<FString> Got;
							for (const TSharedPtr<FJsonValue>& V : *Arr)
							{
								Got.Add(V->AsString());
							}
							Got.Sort();
							TestEqual(FString::Printf(TEXT("parameters of %s (sorted)"), E.Tool), FString::Join(Got, TEXT(",")), FString::Join(E.Params, TEXT(",")));
						}
					}
				}
				Done.Execute();
			});
		});
	});

	Describe("list_model_packages", [this]()
	{
		LatentIt("an empty content folder returns the C38 JSON with count 0", FTimespan::FromSeconds(90), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			RunTool(TEXT("list_model_packages"), FString::Printf(TEXT("{\"path\":\"%s\"}"), *Dest), 60.0, [this, Done](const FToolOut& Out)
			{
				// Catches: a tool that wraps the JSON in another envelope, or forwards the wrong argument.
				if (CheckOk(Out, TEXT("list_model_packages empty")))
				{
					if (TestTrue(TEXT("value is the C38 JSON object (a string containing JSON)"), Out.Obj.IsValid()))
					{
						double Count = -1;
						TestTrue(TEXT("count is a number"), Out.Obj->TryGetNumberField(TEXT("count"), Count));
						TestEqual(TEXT("count"), (int32)Count, 0);
						const TArray<TSharedPtr<FJsonValue>>* Pkgs = nullptr;
						TestTrue(TEXT("packages is an empty array"), Out.Obj->TryGetArrayField(TEXT("packages"), Pkgs) && Pkgs->Num() == 0);
					}
				}
				Done.Execute();
			});
		});

		LatentIt("all arguments are optional: {} lists /Game", FTimespan::FromSeconds(90), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			RunTool(TEXT("list_model_packages"), TEXT("{}"), 60.0, [this, Done](const FToolOut& Out)
			{
				// Catches: a missing default (path="/Game", recursive=True) that would make the call fail with a bad-arguments error.
				if (CheckOk(Out, TEXT("list_model_packages defaults")) && TestTrue(TEXT("value is a JSON object"), Out.Obj.IsValid()))
				{
					double Count = -1;
					TestTrue(TEXT("count is a number"), Out.Obj->TryGetNumberField(TEXT("count"), Count));
					TestTrue(TEXT("count >= 0"), Count >= 0);
					const TArray<TSharedPtr<FJsonValue>>* Pkgs = nullptr;
					TestTrue(TEXT("packages is an array"), Out.Obj->TryGetArrayField(TEXT("packages"), Pkgs));
				}
				Done.Execute();
			});
		});

		LatentIt("a path that is not a content path is a tool error, not a value", FTimespan::FromSeconds(90), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			IgnoreToolErrorLogs(TEXT("Path must be a content path"));
			RunTool(TEXT("list_model_packages"), TEXT("{\"path\":\"not-a-content-path\"}"), 60.0, [this, Done](const FToolOut& Out)
			{
				// Catches: {"error":...} returned as a normal value (the agent would not see a tool error).
				CheckToolError(Out, TEXT("Path must be a content path like /Game"), TEXT("list_model_packages bad path"));
				Done.Execute();
			});
		});
	});

	Describe("import_model_folder", [this]()
	{
		LatentIt("without confirm it creates nothing and asks for confirmation", FTimespan::FromSeconds(120), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			FString Folder;
			if (!RequireFolder(TEXT("import_test"), TEXT("tiny_text_embedding"), Folder)) { Done.Execute(); return; }
			const FString Args = FString::Printf(TEXT("{\"source_folder\":\"%s\",\"destination_path\":\"%s\"}"), *Folder.Replace(TEXT("\\"), TEXT("/")), *Dest);
			RunTool(TEXT("import_model_folder"), Args, 90.0, [this, Done](const FToolOut& Out)
			{
				// Catches: confirm defaulting to True (an agent importing without asking), or snake_case args not reaching C38.
				if (CheckOk(Out, TEXT("import_model_folder no confirm")) && TestTrue(TEXT("value is a JSON object"), Out.Obj.IsValid()))
				{
					TestTrue(TEXT("needs_confirm true"), GetBool(Out.Obj, TEXT("needs_confirm")));
					TestFalse(TEXT("success false"), GetBool(Out.Obj, TEXT("success"), true));
					TestEqual(TEXT("would_create.package"), GetStr(GetObj(Out.Obj, TEXT("would_create")), TEXT("package")), ObjPath(Dest, TEXT("tiny_text_embedding")));
					TArray<FAssetData> Assets;
					IAssetRegistry::Get()->GetAssetsByPath(FName(*Dest), Assets, true);
					TestEqual(TEXT("no assets were created under the destination"), Assets.Num(), 0);
					const FString OnDisk = FPaths::ProjectContentDir() / TEXT("PipelinesTests") / (TEXT("MCP_") + DestGuid);
					TestFalse(TEXT("no folder was saved to disk"), IFileManager::Get().DirectoryExists(*OnDisk));
				}
				Done.Execute();
			});
		});

		LatentIt("asset_name is forwarded", FTimespan::FromSeconds(120), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			FString Folder;
			if (!RequireFolder(TEXT("import_test"), TEXT("tiny_text_embedding"), Folder)) { Done.Execute(); return; }
			const FString Args = FString::Printf(TEXT("{\"source_folder\":\"%s\",\"destination_path\":\"%s\",\"asset_name\":\"RenamedByTool\"}"), *Folder.Replace(TEXT("\\"), TEXT("/")), *Dest);
			RunTool(TEXT("import_model_folder"), Args, 90.0, [this, Done](const FToolOut& Out)
			{
				// Catches: asset_name dropped (the name would be the folder leaf).
				if (CheckOk(Out, TEXT("import_model_folder asset_name")) && TestTrue(TEXT("value is a JSON object"), Out.Obj.IsValid()))
				{
					TestEqual(TEXT("would_create.package uses asset_name"), GetStr(GetObj(Out.Obj, TEXT("would_create")), TEXT("package")), ObjPath(Dest, TEXT("RenamedByTool")));
				}
				Done.Execute();
			});
		});

		LatentIt("confirm=true imports and saves, validate_model_package reports valid, a repeat without overwrite is refused", FTimespan::FromSeconds(300), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			FString Folder;
			if (!RequireFolder(TEXT("import_test"), TEXT("tiny_text_embedding"), Folder)) { Done.Execute(); return; }
			ImportThen(Folder, TEXT(""), Done, [this, Done, Folder](const FString& Pkg)
			{
				TestEqual(TEXT("imported package object path"), Pkg, ObjPath(Dest, TEXT("tiny_text_embedding")));
				{
					TArray<FAssetData> Assets;
					IAssetRegistry::Get()->GetAssetsByPath(FName(*Dest), Assets, true);
					TestEqual(TEXT("the package and its model asset are registered under the destination"), Assets.Num(), 2);
				}
				const FString ValArgs = FString::Printf(TEXT("{\"package\":\"%s\",\"check_gpu\":false}"), *Pkg);
				RunTool(TEXT("validate_model_package"), ValArgs, 120.0, [this, Done, Folder, Pkg](const FToolOut& V)
				{
					// Catches: a validate tool that returns an error/invalid report for a good import, or drops the package arg.
					if (CheckOk(V, TEXT("validate_model_package")) && TestTrue(TEXT("validate value is a JSON object"), V.Obj.IsValid()))
					{
						TestTrue(*FString::Printf(TEXT("valid; got %s"), *Tail(V.Raw)), GetBool(V.Obj, TEXT("valid")));
						double Errors = -1;
						V.Obj->TryGetNumberField(TEXT("errors"), Errors);
						TestEqual(TEXT("errors"), (int32)Errors, 0);
						TestEqual(TEXT("package in the validate result"), GetStr(V.Obj, TEXT("package")), Pkg);
					}
					const FString Again = FString::Printf(TEXT("{\"source_folder\":\"%s\",\"destination_path\":\"%s\"}"), *Folder.Replace(TEXT("\\"), TEXT("/")), *Dest);
					RunTool(TEXT("import_model_folder"), Again, 90.0, [this, Done](const FToolOut& R)
					{
						// Catches: an existing package being offered for silent replacement without overwrite_existing.
						if (CheckOk(R, TEXT("import again no confirm")) && TestTrue(TEXT("value is a JSON object"), R.Obj.IsValid()))
						{
							TestTrue(TEXT("exists true"), GetBool(R.Obj, TEXT("exists")));
							TestFalse(TEXT("needs_confirm false (overwrite_existing not set)"), GetBool(R.Obj, TEXT("needs_confirm"), true));
							TestTrue(TEXT("message tells to pass overwrite_existing"), GetStr(R.Obj, TEXT("message")).Contains(TEXT("overwrite_existing")));
						}
						Done.Execute();
					});
				});
			});
		});

		LatentIt("an empty source_folder is a tool error", FTimespan::FromSeconds(90), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			IgnoreToolErrorLogs(TEXT("source"));
			RunTool(TEXT("import_model_folder"), FString::Printf(TEXT("{\"source_folder\":\"\",\"destination_path\":\"%s\"}"), *Dest), 60.0, [this, Done](const FToolOut& Out)
			{
				// Catches: {"error":...} passed through as a successful value.
				if (TestTrue(TEXT("script finished"), Out.bFileRead))
				{
					TestTrue(TEXT("tool completes"), Out.bComplete);
					TestFalse(*FString::Printf(TEXT("a tool error is reported; raw=%s"), *Tail(Out.Raw)), Out.Error.IsEmpty());
				}
				Done.Execute();
			});
		});
	});

	Describe("validate_model_package", [this]()
	{
		LatentIt("a missing package is a tool error 'Model package not found'", FTimespan::FromSeconds(90), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			IgnoreToolErrorLogs(TEXT("Model package not found"));
			RunTool(TEXT("validate_model_package"), TEXT("{\"package\":\"/Game/Nope\"}"), 60.0, [this, Done](const FToolOut& Out)
			{
				// Catches: the error JSON returned as a value, or a Python exception text instead of the C38 message.
				CheckToolError(Out, TEXT("Model package not found"), TEXT("validate_model_package /Game/Nope"));
				Done.Execute();
			});
		});
	});

	Describe("benchmark_model_package (async)", [this]()
	{
		LatentIt("runs=2 on cpu completes with one CPU entry; texts are forwarded", FTimespan::FromSeconds(420), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			FString Folder;
			if (!RequireFolder(TEXT("import_test"), TEXT("tiny_text_embedding"), Folder)) { Done.Execute(); return; }
			ImportThen(Folder, TEXT(""), Done, [this, Done](const FString& Pkg)
			{
				const FString Args = FString::Printf(TEXT("{\"package\":\"%s\",\"runs\":2,\"devices\":\"cpu\"}"), *Pkg);
				RunTool(TEXT("benchmark_model_package"), Args, 240.0, [this, Done, Pkg](const FToolOut& Out)
				{
					// Catches: a job that never completes the async result, or that benchmarks GPU despite devices="cpu".
					if (CheckOk(Out, TEXT("benchmark cpu")) && TestTrue(TEXT("value is a JSON object"), Out.Obj.IsValid()))
					{
						TestEqual(TEXT("runs"), (int32)([&]() { double R = -1; Out.Obj->TryGetNumberField(TEXT("runs"), R); return R; })(), 2);
						const TArray<TSharedPtr<FJsonValue>>* Devs = nullptr;
						if (TestTrue(TEXT("devices array"), Out.Obj->TryGetArrayField(TEXT("devices"), Devs)) && TestEqual(TEXT("exactly one device entry"), Devs->Num(), 1))
						{
							const TSharedPtr<FJsonObject>* D0 = nullptr;
							if (TestTrue(TEXT("devices[0] is an object"), (*Devs)[0]->TryGetObject(D0) && D0))
							{
								TestEqual(TEXT("devices[0].device"), GetStr(*D0, TEXT("device")), FString(TEXT("CPU")));
								TestTrue(TEXT("inference_ms.median is a number"), GetObj(*D0, TEXT("inference_ms")).IsValid() && GetObj(*D0, TEXT("inference_ms"))->HasTypedField<EJson::Number>(TEXT("median")));
							}
						}
					}
					const FString Args2 = FString::Printf(TEXT("{\"package\":\"%s\",\"runs\":1,\"devices\":\"cpu\",\"texts\":[\"first sentence\",\"zweiter Satz\"]}"), *Pkg);
					RunTool(TEXT("benchmark_model_package"), Args2, 240.0, [this, Done](const FToolOut& Out2)
					{
						// Catches: the Optional[list[str]] argument dropped on the way to C38 (input would say "default text").
						if (CheckOk(Out2, TEXT("benchmark cpu with texts")) && TestTrue(TEXT("value is a JSON object"), Out2.Obj.IsValid()))
						{
							TestEqual(TEXT("input"), GetStr(Out2.Obj, TEXT("input")), FString(TEXT("texts (2)")));
						}
						Done.Execute();
					});
				});
			});
		});

		LatentIt("runs=0 is a tool error 'runs must be 1 to 1000'", FTimespan::FromSeconds(180), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			FString Folder;
			if (!RequireFolder(TEXT("import_test"), TEXT("tiny_text_embedding"), Folder)) { Done.Execute(); return; }
			IgnoreToolErrorLogs(TEXT("runs must be 1 to 1000"));
			ImportThen(Folder, TEXT(""), Done, [this, Done](const FString& Pkg)
			{
				const FString Args = FString::Printf(TEXT("{\"package\":\"%s\",\"runs\":0,\"devices\":\"cpu\"}"), *Pkg);
				RunTool(TEXT("benchmark_model_package"), Args, 120.0, [this, Done](const FToolOut& Out)
				{
					// Catches: the failed job leaving the async result pending forever, or its error text lost.
					CheckToolError(Out, TEXT("runs must be 1 to 1000"), TEXT("benchmark runs=0"));
					Done.Execute();
				});
			});
		});

		LatentIt("a missing package is a tool error, not a hang", FTimespan::FromSeconds(120), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			IgnoreToolErrorLogs(TEXT("Model package not found"));
			RunTool(TEXT("benchmark_model_package"), TEXT("{\"package\":\"/Game/Nope\",\"runs\":1,\"devices\":\"cpu\"}"), 60.0, [this, Done](const FToolOut& Out)
			{
				// Catches: job failure not routed to result.set_error.
				CheckToolError(Out, TEXT("Model package not found"), TEXT("benchmark /Game/Nope"));
				Done.Execute();
			});
		});
	});

	Describe("run_pipeline_sample (async)", [this]()
	{
		LatentIt("two texts on all-MiniLM-L6-v2 give a 2x2 similarity matrix", FTimespan::FromSeconds(420), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			FString Folder;
			if (!RequireFolder(TEXT("reference"), TEXT("all-MiniLM-L6-v2"), Folder)) { Done.Execute(); return; }
			ImportThen(Folder, TEXT(""), Done, [this, Done](const FString& Pkg)
			{
				const FString Args = FString::Printf(TEXT("{\"package\":\"%s\",\"texts\":[\"A cat sits on the mat.\",\"A kitten rests on the rug.\"]}"), *Pkg);
				RunTool(TEXT("run_pipeline_sample"), Args, 240.0, [this, Done](const FToolOut& Out)
				{
					// Catches: texts not forwarded (matrix missing), a non-normalised diagonal, an asymmetric matrix.
					if (CheckOk(Out, TEXT("run_pipeline_sample 2 texts")) && TestTrue(TEXT("value is a JSON object"), Out.Obj.IsValid()))
					{
						TestEqual(TEXT("task"), GetStr(Out.Obj, TEXT("task")), FString(TEXT("TextEmbedding")));
						const TArray<TSharedPtr<FJsonValue>>* Emb = nullptr;
						if (TestTrue(TEXT("embeddings array"), Out.Obj->TryGetArrayField(TEXT("embeddings"), Emb)) && TestEqual(TEXT("2 embeddings"), Emb->Num(), 2))
						{
							const TSharedPtr<FJsonObject>* E0 = nullptr;
							double Dim = 0;
							if ((*Emb)[0]->TryGetObject(E0) && E0)
							{
								(*E0)->TryGetNumberField(TEXT("dim"), Dim);
							}
							TestEqual(TEXT("MiniLM embedding dim"), (int32)Dim, 384);
						}
						const TArray<TSharedPtr<FJsonValue>>* Sim = nullptr;
						if (TestTrue(TEXT("similarity matrix present"), Out.Obj->TryGetArrayField(TEXT("similarity"), Sim)) && TestEqual(TEXT("similarity rows"), Sim->Num(), 2))
						{
							double M[2][2] = { {0, 0}, {0, 0} };
							bool bShape = true;
							for (int32 R = 0; R < 2; ++R)
							{
								const TArray<TSharedPtr<FJsonValue>>* Row = nullptr;
								if (!(*Sim)[R]->TryGetArray(Row) || Row->Num() != 2)
								{
									bShape = false;
									continue;
								}
								for (int32 C = 0; C < 2; ++C)
								{
									M[R][C] = (*Row)[C]->AsNumber();
								}
							}
							if (TestTrue(TEXT("every row has 2 columns"), bShape))
							{
								TestTrue(*FString::Printf(TEXT("diagonal is 1 (unit vectors): %.6f, %.6f"), M[0][0], M[1][1]), FMath::Abs(M[0][0] - 1.0) < 1e-4 && FMath::Abs(M[1][1] - 1.0) < 1e-4);
								TestTrue(*FString::Printf(TEXT("symmetric: %.6f vs %.6f"), M[0][1], M[1][0]), FMath::Abs(M[0][1] - M[1][0]) < 1e-5);
								TestTrue(*FString::Printf(TEXT("two paraphrases are clearly similar (> 0.5): %.4f"), M[0][1]), M[0][1] > 0.5 && M[0][1] < 1.0);
							}
						}
					}
					Done.Execute();
				});
			});
		});

		LatentIt("a text-embedding package without texts is a tool error 'needs texts'", FTimespan::FromSeconds(180), [this](const FDoneDelegate& Done)
		{
			if (!CanRun()) { Done.Execute(); return; }
			FString Folder;
			if (!RequireFolder(TEXT("import_test"), TEXT("tiny_text_embedding"), Folder)) { Done.Execute(); return; }
			IgnoreToolErrorLogs(TEXT("needs texts"));
			ImportThen(Folder, TEXT(""), Done, [this, Done](const FString& Pkg)
			{
				RunTool(TEXT("run_pipeline_sample"), FString::Printf(TEXT("{\"package\":\"%s\"}"), *Pkg), 120.0, [this, Done](const FToolOut& Out)
				{
					// Catches: a missing optional list turned into [""] (a bogus embedding) instead of the C38 error.
					CheckToolError(Out, TEXT("needs texts"), TEXT("run_pipeline_sample without texts"));
					Done.Execute();
				});
			});
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
