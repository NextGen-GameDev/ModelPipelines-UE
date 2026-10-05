// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "PipelineTypes.h"
#include "PipelineModelPackage.h"
#include "PipelineValidation.h"

/**
 * JSON helpers shared by the Model Pipelines MCP functions (WP6.1, WP6.2). Private to ModelPipelinesMCP.
 * Every helper is a pure function of its arguments; enum names come from the UENUM reflection data.
 */
namespace PipelineMcpJson
{
	/** UENUM value name of a task: "TextEmbedding" | ... | "None"; "Unknown" for a value outside the enum. */
	FString TaskToString(EPipelineTask Task);

	/** "Auto" | "CPU" | "GPU"; "Unknown" for a value outside the enum. */
	FString DeviceToString(EPipelineDevice Device);

	/** "Info" | "Warning" | "Error"; "Unknown" for a value outside the enum. */
	FString SeverityToString(EPipelineValidationSeverity Severity);

	/** EPipelineValidationCode value name, e.g. "MissingModelFile"; "Unknown" for a value outside the enum. */
	FString CodeToString(EPipelineValidationCode Code);

	/** A JSON number, or JSON null when Value is NaN or infinite (never "nan"/"inf" text). */
	TSharedRef<FJsonValue> FiniteOrNull(double Value);

	/** {"messages":[{"severity","code","message"}...],"device_checks":[{"device","available","created","ran","inference_ms","message"}...]} in report order. */
	TSharedRef<FJsonObject> ReportToJson(const FPipelineValidationReport& Report);

	/** {"device","queue_ms","preprocess_ms","inference_ms","postprocess_ms"} of a pipeline result. */
	TSharedRef<FJsonObject> TimingsToJson(const FPipelineResult& Result);

	/** Condensed JSON text of an object (TCondensedJsonPrintPolicy). */
	FString ToCondensedString(const TSharedRef<FJsonObject>& Object);

	/** Condensed {"error":"<Message>"}. */
	FString MakeError(const FString& Message);

	/** EPipelineError value name, e.g. "InferenceFailed"; "Unknown" for a value outside the enum. */
	FString ErrorToString(EPipelineError Error);

	/** Longest long package name accepted by the MCP helpers (same limit as the importer; FName asserts on very long strings). */
	constexpr int32 MaxLongPackageNameLen = 200;

	/** Longest accepted asset reference (object path) before any FName is built. */
	constexpr int32 MaxReferenceLen = 2 * MaxLongPackageNameLen + 1;

	/**
	 * Game thread. Resolves an object path (/Game/A/B.B) or package name (/Game/A/B, object name = last path component) to a
	 * live (IsValid) object, loading it from disk when needed; nullptr when the string is not a valid reference or nothing
	 * exists there. Every string check happens before any FName is built. Callers Cast the result to the class they need.
	 */
	UObject* ResolveAsset(const FString& Arg);
}
