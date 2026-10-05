// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "PipelineMcpLibrary.generated.h"

/**
 * Editor helpers behind the Model Pipelines MCP tools (docs/contracts.md C38). Each function takes plain values and returns
 * compact JSON text an AI agent can read. All functions are editor-only, game thread only (off the game thread they return
 * {"error":"must be called on the game thread"} and touch nothing) and never crash on any input.
 */
UCLASS()
class MODELPIPELINESMCP_API UPipelineMcpLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	/**
	 * Lists the UPipelineModelPackage assets under a content path (saved and in-memory ones), sorted by object path.
	 * Returns {"count":N,"packages":[{"path","task","preferred_device","model","labels","has_tokenizer","valid_settings"}]}.
	 * "model" is "" when unset; "valid_settings" is true when the package settings check has no Error (no smoke run).
	 * A path that is not a content path like /Game returns {"error":"Path must be a content path like /Game"}; an empty or
	 * unknown folder returns {"count":0,"packages":[]}. Loads each listed asset to read its fields; changes nothing.
	 * @param Path        Content folder to search, e.g. /Game or /Game/Models.
	 * @param bRecursive  Also search the sub-folders of Path.
	 */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|MCP")
	static FString ListModelPackages(const FString& Path = TEXT("/Game"), bool bRecursive = true);

	/**
	 * Imports a Hugging Face-style model folder as <DestinationPath>/<Name> (UPipelineModelPackage) and
	 * <DestinationPath>/<Name>_Model (UNNEModelData). Without bConfirm nothing is created or changed: the folder is parsed and
	 * the result describes what would be created:
	 * {"success":false,"needs_confirm":bool,"would_create":{"package","model"},"exists":bool,"task","report","message"}.
	 * needs_confirm is false when the folder has errors or the assets exist and bOverwriteExisting is false.
	 * With bConfirm the import runs (saving both packages, full validation):
	 * {"success":bool,"package":"<object path>"|null,"model":"<object path>"|null,"task","report"}. A failed import leaves
	 * nothing behind. Never deletes anything. An empty SourceFolder returns {"error":"source_folder is empty"}.
	 * @param SourceFolder        Folder on disk holding model.onnx (or onnx/model.onnx), config.json, ...
	 * @param DestinationPath     Content folder the assets go to, e.g. /Game/Models.
	 * @param AssetName           Name of the package asset; "" uses the sanitized folder name.
	 * @param bOverwriteExisting  Update existing assets of the right classes in place instead of refusing.
	 * @param bConfirm            Actually import. Without it the call only previews and changes nothing.
	 */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|MCP")
	static FString ImportModelFolder(const FString& SourceFolder, const FString& DestinationPath = TEXT("/Game/Models"),
		const FString& AssetName = TEXT(""), bool bOverwriteExisting = false, bool bConfirm = false);

	/**
	 * Validates a model package: settings, tokenizer, model creation on CPU (and GPU), task I/O and smoke runs.
	 * Returns {"package":"<object path>","task","valid":bool,"errors":N,"warnings":N,"report":{"messages","device_checks"}}.
	 * Never modifies the asset. An empty, unknown or non-package reference returns {"error":"Model package not found: <arg>"}.
	 * Blocking (runs the smoke inferences before returning).
	 * @param Package        Object path (/Game/Models/X.X) or package name (/Game/Models/X) of the UPipelineModelPackage.
	 * @param bRunSmokeTest  Run one small inference on each created device.
	 * @param bCheckGpu      Also create (and smoke-run) the model on the GPU runtime.
	 */
	UFUNCTION(BlueprintCallable, Category = "Model Pipelines|MCP")
	static FString ValidateModelPackage(const FString& Package, bool bRunSmokeTest = true, bool bCheckGpu = true);
};
