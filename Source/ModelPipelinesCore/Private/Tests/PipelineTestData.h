// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#if WITH_DEV_AUTOMATION_TESTS

#include "CoreMinimal.h"
#include "Dom/JsonObject.h"
#include "PipelineTensor.h"

class UNNEModelData;

namespace UE::ModelPipelines::Tests
{
	/** env MODELPIPELINES_DEV_DIR, else <ProjectDir>/../.pipelines-dev (absolute). */
	FString GetDevDir();

	/** Reads <dev>/models/test/<FileName> into a transient UNNEModelData ("onnx"). nullptr + OutError if missing. Caller keeps it alive. */
	UNNEModelData* LoadTestModel(const FString& FileName, FString& OutError);

	/** Transient UNNEModelData("onnx") over arbitrary bytes (for garbage-model tests). */
	UNNEModelData* MakeModelFromBytes(TConstArrayView<uint8> Bytes);

	/** Default path: <dev>/goldens/test_models/test_models.json. nullptr + OutError on missing file or bad JSON. */
	TSharedPtr<FJsonObject> LoadTestGoldens(FString& OutError, const FString& OverridePath = FString());

	/** Builds a tensor from {"type","shape","data"} (float32, int64, int32). Invalid tensor on anything unexpected. */
	FPipelineTensor TensorFromGoldenJson(const TSharedPtr<FJsonObject>& Obj);

	/** env MODELPIPELINES_REQUIRE_GPU == "1". */
	bool IsGpuRequired();

	/** |a-e| <= 1e-6 + 1e-6*|e| */
	bool NearlyEqualGolden(float Actual, float Expected);

	/**
	 * Compares an output tensor with a golden output object (type, shape, every element; ints exact).
	 * Returns an empty string on match, otherwise a description of the first mismatch.
	 * OutWorstAbsDiff (optional) receives the largest |a-e| seen among float elements.
	 */
	FString CompareTensorToGolden(const FPipelineTensor& Actual, const TSharedPtr<FJsonObject>& GoldenOutput, double* OutWorstAbsDiff = nullptr);

	/** Finds the entry of goldens["models"] whose "file" equals FileName, or nullptr. */
	TSharedPtr<FJsonObject> FindGoldenModel(const TSharedPtr<FJsonObject>& Goldens, const FString& FileName);
}

#endif // WITH_DEV_AUTOMATION_TESTS
