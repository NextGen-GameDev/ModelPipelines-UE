// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "PipelineTensor.generated.h"

/** Element type of an FPipelineTensor. Values are fixed (docs/contracts.md C7). */
UENUM(BlueprintType)
enum class EPipelineTensorType : uint8
{
	Float32 = 0,
	Int64 = 1,
	Int32 = 2
};

/** Bytes per element: 4, 8, 4. 0 for an out-of-range value. */
MODELPIPELINESCORE_API int32 GetPipelineTensorTypeSize(EPipelineTensorType Type);

/** "float32", "int64", "int32"; "unknown" for an out-of-range value. */
MODELPIPELINESCORE_API const TCHAR* LexToString(EPipelineTensorType Type);

/**
 * A shape plus a typed, flat, row-major buffer of float32, int64 or int32.
 *
 * Plain value type: copies are deep, moves leave the source invalid. Never asserts on bad input:
 * factories return an invalid tensor and log a Warning to LogModelPipelines instead.
 *
 * Thread safety: none needed for separate objects; one object must not be mutated from two threads.
 */
struct MODELPIPELINESCORE_API FPipelineTensor
{
	static constexpr int32 MaxRank = 8;
	static constexpr int64 MaxSizeInBytes = 4ll * 1024 * 1024 * 1024; // 4 GiB

	/** Invalid: IsValid() false, Num() 0, rank 0, type Float32. */
	FPipelineTensor();
	~FPipelineTensor();

	FPipelineTensor(const FPipelineTensor& Other);
	FPipelineTensor(FPipelineTensor&& Other);
	FPipelineTensor& operator=(const FPipelineTensor& Other);
	FPipelineTensor& operator=(FPipelineTensor&& Other);

	/** Factories. Invalid shape (rank > 8, a dim < 0 or > MAX_uint32, element count or byte size overflow /
	    > MaxSizeInBytes) or Values.Num() != element count -> invalid tensor plus a Warning. Never asserts.
	    Shape {} is a scalar (1 element). A dim of 0 is valid (0 elements). */
	static FPipelineTensor Zeros(EPipelineTensorType Type, TConstArrayView<int64> Shape);
	static FPipelineTensor FromFloat(TConstArrayView<int64> Shape, TConstArrayView<float> Values);
	static FPipelineTensor FromInt64(TConstArrayView<int64> Shape, TConstArrayView<int64> Values);
	static FPipelineTensor FromInt32(TConstArrayView<int64> Shape, TConstArrayView<int32> Values);

	bool IsValid() const;
	EPipelineTensorType GetType() const;

	/** Empty for a scalar and for an invalid tensor. */
	TConstArrayView<int64> GetShape() const;
	int32 GetRank() const;

	/** Element count; 1 for a scalar; 0 when invalid. */
	int64 Num() const;

	/** Num() * type size; 0 when invalid. */
	int64 GetSizeInBytes() const;

	/** Typed views: empty view when the type differs or the tensor is invalid. Never asserts. */
	TArrayView64<float> GetFloatData();
	TConstArrayView64<float> GetFloatData() const;
	TArrayView64<int64> GetInt64Data();
	TConstArrayView64<int64> GetInt64Data() const;
	TArrayView64<int32> GetInt32Data();
	TConstArrayView64<int32> GetInt32Data() const;

	/** nullptr when invalid or Num() == 0; 16-byte aligned otherwise. */
	void* GetRawData();
	const void* GetRawData() const;

	/** True if NewShape is valid and has the same element count; otherwise false and unchanged. No -1 inference. */
	bool Reshape(TConstArrayView<int64> NewShape);

	/** "[2,4]", scalar "[]", invalid "<invalid>". */
	FString DescribeShape() const;

private:
	/** Validates Shape for Type. On success writes the element count and returns true; on failure logs a
	    Warning naming Context and returns false. Allocates nothing. */
	static bool ValidateShape(EPipelineTensorType Type, TConstArrayView<int64> Shape, int64& OutNum, const TCHAR* Context);

	/** Builds a valid tensor of Type/Shape holding NumElements elements, zero-filled. Caller has validated. */
	static FPipelineTensor MakeZeroed(EPipelineTensorType Type, TConstArrayView<int64> Shape, int64 NumElements);

	/** Frees Data and resets to the invalid state. */
	void Reset();

	template <typename T>
	TArrayView64<T> GetTypedView(EPipelineTensorType Expected) const;

	/** Owned; FMemory::Malloc(bytes, 16). nullptr when invalid or 0 elements. */
	void* Data = nullptr;
	TArray<int64, TInlineAllocator<MaxRank>> Shape;
	int64 NumElements = 0;
	EPipelineTensorType Type = EPipelineTensorType::Float32;
	bool bValid = false;
};

/** Description of one model input or output. */
struct MODELPIPELINESCORE_API FPipelineTensorDesc
{
	FString Name;
	EPipelineTensorType Type = EPipelineTensorType::Float32;

	/** False when the model's type is none of the three; Type is then meaningless. */
	bool bTypeSupported = true;

	/** -1 = dynamic dim. */
	TArray<int64> Shape;

	/** True if bTypeSupported, T valid, same type, same rank, and each dim equal or desc dim == -1. */
	bool IsCompatible(const FPipelineTensor& T) const;
};
