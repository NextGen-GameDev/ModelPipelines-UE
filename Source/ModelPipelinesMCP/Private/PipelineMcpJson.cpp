// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineMcpJson.h"

#include "Math/UnrealMathUtility.h"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "UObject/Class.h"

namespace PipelineMcpJsonPrivate
{
	template <typename TEnum>
	static FString EnumName(TEnum Value)
	{
		const UEnum* Enum = StaticEnum<TEnum>();
		if (!Enum || !Enum->IsValidEnumValue(static_cast<int64>(Value)))
		{
			return TEXT("Unknown");
		}
		FString Name = Enum->GetNameStringByValue(static_cast<int64>(Value));
		return Name.IsEmpty() ? FString(TEXT("Unknown")) : Name;
	}
}

namespace PipelineMcpJson
{
	FString TaskToString(EPipelineTask Task)
	{
		return PipelineMcpJsonPrivate::EnumName(Task);
	}

	FString DeviceToString(EPipelineDevice Device)
	{
		return PipelineMcpJsonPrivate::EnumName(Device);
	}

	FString SeverityToString(EPipelineValidationSeverity Severity)
	{
		return PipelineMcpJsonPrivate::EnumName(Severity);
	}

	FString CodeToString(EPipelineValidationCode Code)
	{
		return PipelineMcpJsonPrivate::EnumName(Code);
	}

	TSharedRef<FJsonValue> FiniteOrNull(double Value)
	{
		if (FMath::IsFinite(Value))
		{
			return MakeShared<FJsonValueNumber>(Value);
		}
		return MakeShared<FJsonValueNull>();
	}

	TSharedRef<FJsonObject> ReportToJson(const FPipelineValidationReport& Report)
	{
		TSharedRef<FJsonObject> Obj = MakeShared<FJsonObject>();

		TArray<TSharedPtr<FJsonValue>> Messages;
		Messages.Reserve(Report.Messages.Num());
		for (const FPipelineValidationMessage& M : Report.Messages)
		{
			TSharedRef<FJsonObject> MObj = MakeShared<FJsonObject>();
			MObj->SetStringField(TEXT("severity"), SeverityToString(M.Severity));
			MObj->SetStringField(TEXT("code"), CodeToString(M.Code));
			MObj->SetStringField(TEXT("message"), M.Message);
			Messages.Add(MakeShared<FJsonValueObject>(MObj));
		}
		Obj->SetArrayField(TEXT("messages"), Messages);

		TArray<TSharedPtr<FJsonValue>> Checks;
		Checks.Reserve(Report.DeviceChecks.Num());
		for (const FPipelineDeviceCheck& C : Report.DeviceChecks)
		{
			TSharedRef<FJsonObject> CObj = MakeShared<FJsonObject>();
			CObj->SetStringField(TEXT("device"), DeviceToString(C.Device));
			CObj->SetBoolField(TEXT("available"), C.bAvailable);
			CObj->SetBoolField(TEXT("created"), C.bCreated);
			CObj->SetBoolField(TEXT("ran"), C.bRan);
			CObj->SetField(TEXT("inference_ms"), FiniteOrNull(C.InferenceMs));
			CObj->SetStringField(TEXT("message"), C.Message);
			Checks.Add(MakeShared<FJsonValueObject>(CObj));
		}
		Obj->SetArrayField(TEXT("device_checks"), Checks);
		return Obj;
	}

	TSharedRef<FJsonObject> TimingsToJson(const FPipelineResult& Result)
	{
		TSharedRef<FJsonObject> Obj = MakeShared<FJsonObject>();
		Obj->SetStringField(TEXT("device"), DeviceToString(Result.Device));
		Obj->SetField(TEXT("queue_ms"), FiniteOrNull(Result.QueueMs));
		Obj->SetField(TEXT("preprocess_ms"), FiniteOrNull(Result.PreprocessMs));
		Obj->SetField(TEXT("inference_ms"), FiniteOrNull(Result.InferenceMs));
		Obj->SetField(TEXT("postprocess_ms"), FiniteOrNull(Result.PostprocessMs));
		return Obj;
	}

	FString ToCondensedString(const TSharedRef<FJsonObject>& Object)
	{
		FString Out;
		TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer =
			TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Out);
		if (!FJsonSerializer::Serialize(Object, Writer))
		{
			return TEXT("{\"error\":\"internal: JSON serialization failed\"}");
		}
		return Out;
	}

	FString MakeError(const FString& Message)
	{
		TSharedRef<FJsonObject> Obj = MakeShared<FJsonObject>();
		Obj->SetStringField(TEXT("error"), Message);
		return ToCondensedString(Obj);
	}
}
