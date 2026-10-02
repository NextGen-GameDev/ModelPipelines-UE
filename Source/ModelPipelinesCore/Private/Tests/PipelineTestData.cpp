// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "PipelineTestData.h"
#include "NNEModelData.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformMisc.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"

namespace UE::ModelPipelines::Tests
{
	FString GetDevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		if (!Env.IsEmpty())
		{
			return Env;
		}
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev"));
	}

	UNNEModelData* LoadTestModel(const FString& FileName, FString& OutError)
	{
		const FString Path = GetDevDir() / TEXT("models/test") / FileName;
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *Path) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("Missing test model '%s'. Generate it with the WP1.1 script in Tools/ (make_test_models.py) into <dev>/models/test/."), *Path);
			return nullptr;
		}
		UNNEModelData* Data = NewObject<UNNEModelData>(GetTransientPackage());
		Data->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
		return Data;
	}

	UNNEModelData* MakeModelFromBytes(TConstArrayView<uint8> Bytes)
	{
		UNNEModelData* Data = NewObject<UNNEModelData>(GetTransientPackage());
		Data->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
		return Data;
	}

	TSharedPtr<FJsonObject> LoadTestGoldens(FString& OutError, const FString& OverridePath)
	{
		const FString Path = OverridePath.IsEmpty() ? (GetDevDir() / TEXT("goldens/test_models/test_models.json")) : OverridePath;
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			OutError = FString::Printf(TEXT("Missing goldens '%s'. Generate them with the WP1.1 script Tools/make_test_goldens.py."), *Path);
			return nullptr;
		}
		TSharedPtr<FJsonObject> Root;
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
		if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
		{
			OutError = FString::Printf(TEXT("Goldens file '%s' is not valid JSON."), *Path);
			return nullptr;
		}
		return Root;
	}

	FPipelineTensor TensorFromGoldenJson(const TSharedPtr<FJsonObject>& Obj)
	{
		if (!Obj.IsValid())
		{
			return FPipelineTensor();
		}
		FString Type;
		const TArray<TSharedPtr<FJsonValue>>* ShapeArr = nullptr;
		const TArray<TSharedPtr<FJsonValue>>* DataArr = nullptr;
		if (!Obj->TryGetStringField(TEXT("type"), Type) || !Obj->TryGetArrayField(TEXT("shape"), ShapeArr) || !Obj->TryGetArrayField(TEXT("data"), DataArr))
		{
			return FPipelineTensor();
		}
		TArray<int64> Shape;
		for (const TSharedPtr<FJsonValue>& V : *ShapeArr)
		{
			Shape.Add((int64)V->AsNumber());
		}
		if (Type == TEXT("float32"))
		{
			TArray<float> Vals;
			for (const TSharedPtr<FJsonValue>& V : *DataArr) { Vals.Add((float)V->AsNumber()); }
			return FPipelineTensor::FromFloat(Shape, Vals);
		}
		if (Type == TEXT("int64"))
		{
			TArray<int64> Vals;
			for (const TSharedPtr<FJsonValue>& V : *DataArr) { Vals.Add((int64)V->AsNumber()); }
			return FPipelineTensor::FromInt64(Shape, Vals);
		}
		if (Type == TEXT("int32"))
		{
			TArray<int32> Vals;
			for (const TSharedPtr<FJsonValue>& V : *DataArr) { Vals.Add((int32)V->AsNumber()); }
			return FPipelineTensor::FromInt32(Shape, Vals);
		}
		return FPipelineTensor();
	}

	UNNEModelData* LoadModelFile(const FString& AbsPath, FString& OutError)
	{
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *AbsPath) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("Missing model file '%s'. Fetch: run ModelPipelines-UE/Tools/make_import_test_data.py (and fetch_reference_models.py) so that <dev>/models/import_test exists."), *AbsPath);
			return nullptr;
		}
		UNNEModelData* Data = NewObject<UNNEModelData>(GetTransientPackage());
		Data->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
		return Data;
	}

	TSharedPtr<FJsonObject> LoadJsonFile(const FString& AbsPath, FString& OutError)
	{
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *AbsPath))
		{
			OutError = FString::Printf(TEXT("Missing JSON file '%s'."), *AbsPath);
			return nullptr;
		}
		TSharedPtr<FJsonObject> Root;
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
		if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
		{
			OutError = FString::Printf(TEXT("File '%s' is not valid JSON."), *AbsPath);
			return nullptr;
		}
		return Root;
	}

	TSharedPtr<FJsonObject> LoadTextGoldens(FString& OutError)
	{
		const FString Path = GetDevDir() / TEXT("goldens/text/text_goldens.json");
		TSharedPtr<FJsonObject> Root = LoadJsonFile(Path, OutError);
		if (!Root.IsValid())
		{
			OutError += TEXT(" Generate the text goldens with: \"D:/Micromamba/micromambaenv/envs/pipelines/python.exe\" ModelPipelines-UE/Tools/make_text_goldens.py");
		}
		return Root;
	}

	bool IsGpuRequired()
	{
		return FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_GPU")) == TEXT("1");
	}

	bool NearlyEqualGolden(float Actual, float Expected)
	{
		if (FMath::IsNaN(Actual) || FMath::IsNaN(Expected))
		{
			return false;
		}
		const double Diff = FMath::Abs((double)Actual - (double)Expected);
		return Diff <= 1e-6 + 1e-6 * FMath::Abs((double)Expected);
	}

	FString CompareTensorToGolden(const FPipelineTensor& Actual, const TSharedPtr<FJsonObject>& GoldenOutput, double* OutWorstAbsDiff)
	{
		if (OutWorstAbsDiff) { *OutWorstAbsDiff = 0.0; }
		const FPipelineTensor Expected = TensorFromGoldenJson(GoldenOutput);
		if (!Expected.IsValid())
		{
			return TEXT("golden output could not be parsed into a tensor");
		}
		if (!Actual.IsValid())
		{
			return TEXT("actual output tensor is invalid");
		}
		if (Actual.GetType() != Expected.GetType())
		{
			return FString::Printf(TEXT("type mismatch: actual %s, expected %s"), LexToString(Actual.GetType()), LexToString(Expected.GetType()));
		}
		if (Actual.DescribeShape() != Expected.DescribeShape())
		{
			return FString::Printf(TEXT("shape mismatch: actual %s, expected %s"), *Actual.DescribeShape(), *Expected.DescribeShape());
		}
		switch (Expected.GetType())
		{
		case EPipelineTensorType::Float32:
		{
			TConstArrayView64<float> A = Actual.GetFloatData();
			TConstArrayView64<float> E = Expected.GetFloatData();
			FString First;
			double Worst = 0.0;
			for (int64 i = 0; i < E.Num() && i < A.Num(); ++i)
			{
				const double D = FMath::Abs((double)A[i] - (double)E[i]);
				if (D > Worst) { Worst = D; }
				if (First.IsEmpty() && !NearlyEqualGolden(A[i], E[i]))
				{
					First = FString::Printf(TEXT("float mismatch at element %lld: actual %.9g, expected %.9g"), i, (double)A[i], (double)E[i]);
				}
			}
			if (OutWorstAbsDiff) { *OutWorstAbsDiff = Worst; }
			return First.IsEmpty() ? FString() : FString::Printf(TEXT("%s (worst abs diff %.9g)"), *First, Worst);
		}
		case EPipelineTensorType::Int64:
		{
			TConstArrayView64<int64> A = Actual.GetInt64Data();
			TConstArrayView64<int64> E = Expected.GetInt64Data();
			for (int64 i = 0; i < E.Num() && i < A.Num(); ++i)
			{
				if (A[i] != E[i])
				{
					return FString::Printf(TEXT("int64 mismatch at element %lld: actual %lld, expected %lld"), i, A[i], E[i]);
				}
			}
			return FString();
		}
		case EPipelineTensorType::Int32:
		{
			TConstArrayView64<int32> A = Actual.GetInt32Data();
			TConstArrayView64<int32> E = Expected.GetInt32Data();
			for (int64 i = 0; i < E.Num() && i < A.Num(); ++i)
			{
				if (A[i] != E[i])
				{
					return FString::Printf(TEXT("int32 mismatch at element %lld: actual %d, expected %d"), i, A[i], E[i]);
				}
			}
			return FString();
		}
		}
		return TEXT("unknown tensor type");
	}

	TSharedPtr<FJsonObject> FindGoldenModel(const TSharedPtr<FJsonObject>& Goldens, const FString& FileName)
	{
		const TArray<TSharedPtr<FJsonValue>>* Models = nullptr;
		if (!Goldens.IsValid() || !Goldens->TryGetArrayField(TEXT("models"), Models))
		{
			return nullptr;
		}
		for (const TSharedPtr<FJsonValue>& M : *Models)
		{
			const TSharedPtr<FJsonObject>* Obj = nullptr;
			FString File;
			if (M.IsValid() && M->TryGetObject(Obj) && (*Obj)->TryGetStringField(TEXT("file"), File) && File == FileName)
			{
				return *Obj;
			}
		}
		return nullptr;
	}
}

#endif // WITH_DEV_AUTOMATION_TESTS
