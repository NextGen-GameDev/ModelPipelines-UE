// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineTensor.h"
#include "PipelineLog.h"
#include "HAL/UnrealMemory.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(PipelineTensor)

namespace UE::ModelPipelines::Private
{
	constexpr uint32 TensorAlignment = 16;

	FString FormatShape(TConstArrayView<int64> Shape)
	{
		FString Out(TEXT("["));
		for (int32 Index = 0; Index < Shape.Num(); ++Index)
		{
			if (Index > 0)
			{
				Out += TEXT(",");
			}
			Out += FString::Printf(TEXT("%lld"), (long long)Shape[Index]);
		}
		Out += TEXT("]");
		return Out;
	}
}

int32 GetPipelineTensorTypeSize(EPipelineTensorType Type)
{
	switch (Type)
	{
	case EPipelineTensorType::Float32: return 4;
	case EPipelineTensorType::Int64:   return 8;
	case EPipelineTensorType::Int32:   return 4;
	default:                           return 0;
	}
}

const TCHAR* LexToString(EPipelineTensorType Type)
{
	switch (Type)
	{
	case EPipelineTensorType::Float32: return TEXT("float32");
	case EPipelineTensorType::Int64:   return TEXT("int64");
	case EPipelineTensorType::Int32:   return TEXT("int32");
	default:                           return TEXT("unknown");
	}
}

FPipelineTensor::FPipelineTensor() = default;

FPipelineTensor::~FPipelineTensor()
{
	Reset();
}

FPipelineTensor::FPipelineTensor(const FPipelineTensor& Other)
{
	*this = Other;
}

FPipelineTensor::FPipelineTensor(FPipelineTensor&& Other)
{
	*this = MoveTemp(Other);
}

FPipelineTensor& FPipelineTensor::operator=(const FPipelineTensor& Other)
{
	if (this == &Other)
	{
		return *this;
	}

	Reset();
	if (!Other.bValid)
	{
		return *this;
	}

	const int64 Bytes = Other.GetSizeInBytes();
	void* NewData = nullptr;
	if (Bytes > 0 && Other.Data != nullptr)
	{
		NewData = FMemory::Malloc((SIZE_T)Bytes, UE::ModelPipelines::Private::TensorAlignment);
		if (NewData == nullptr)
		{
			UE_LOG(LogModelPipelines, Warning, TEXT("FPipelineTensor copy: allocation of %lld bytes failed; the copy is invalid."), (long long)Bytes);
			return *this;
		}
		FMemory::Memcpy(NewData, Other.Data, (SIZE_T)Bytes);
	}

	Data = NewData;
	Shape = Other.Shape;
	NumElements = Other.NumElements;
	Type = Other.Type;
	bValid = true;
	return *this;
}

FPipelineTensor& FPipelineTensor::operator=(FPipelineTensor&& Other)
{
	if (this == &Other)
	{
		return *this;
	}

	Reset();
	Data = Other.Data;
	Shape = MoveTemp(Other.Shape);
	NumElements = Other.NumElements;
	Type = Other.Type;
	bValid = Other.bValid;

	// Leave the source in the documented invalid state without freeing the moved buffer.
	Other.Data = nullptr;
	Other.Shape.Reset();
	Other.NumElements = 0;
	Other.Type = EPipelineTensorType::Float32;
	Other.bValid = false;
	return *this;
}

void FPipelineTensor::Reset()
{
	if (Data != nullptr)
	{
		FMemory::Free(Data);
		Data = nullptr;
	}
	Shape.Reset();
	NumElements = 0;
	Type = EPipelineTensorType::Float32;
	bValid = false;
}

bool FPipelineTensor::ValidateShape(EPipelineTensorType InType, TConstArrayView<int64> InShape, int64& OutNum, const TCHAR* Context)
{
	OutNum = 0;

	const int64 TypeSize = GetPipelineTensorTypeSize(InType);
	if (TypeSize <= 0)
	{
		UE_LOG(LogModelPipelines, Warning, TEXT("%s: unknown tensor type %d."), Context, (int32)InType);
		return false;
	}

	if (InShape.Num() > MaxRank)
	{
		UE_LOG(LogModelPipelines, Warning, TEXT("%s: rank %d exceeds the maximum of %d."), Context, InShape.Num(), MaxRank);
		return false;
	}

	int64 Count = 1;
	bool bOverflow = false;
	for (int32 Index = 0; Index < InShape.Num(); ++Index)
	{
		const int64 Dim = InShape[Index];
		if (Dim < 0)
		{
			UE_LOG(LogModelPipelines, Warning, TEXT("%s: dim %d is negative (%lld) in shape %s."),
				Context, Index, (long long)Dim, *UE::ModelPipelines::Private::FormatShape(InShape));
			return false;
		}
		if (Dim > (int64)MAX_uint32)
		{
			UE_LOG(LogModelPipelines, Warning, TEXT("%s: dim %d (%lld) exceeds MAX_uint32 in shape %s."),
				Context, Index, (long long)Dim, *UE::ModelPipelines::Private::FormatShape(InShape));
			return false;
		}
		// Keep checking the remaining dims for negatives even after an overflow, but stop multiplying.
		if (!bOverflow)
		{
			if (Dim != 0 && Count > MAX_int64 / Dim)
			{
				bOverflow = true;
			}
			else
			{
				Count *= Dim;
			}
		}
	}

	// A zero dim anywhere makes the element count 0, whatever the other dims are.
	for (const int64 Dim : InShape)
	{
		if (Dim == 0)
		{
			OutNum = 0;
			return true;
		}
	}

	if (bOverflow || Count > MAX_int64 / TypeSize)
	{
		UE_LOG(LogModelPipelines, Warning, TEXT("%s: element count or byte size of shape %s overflows."),
			Context, *UE::ModelPipelines::Private::FormatShape(InShape));
		return false;
	}

	const int64 Bytes = Count * TypeSize;
	if (Bytes > MaxSizeInBytes)
	{
		UE_LOG(LogModelPipelines, Warning, TEXT("%s: shape %s of type %s needs %lld bytes, over the %lld-byte limit."),
			Context, *UE::ModelPipelines::Private::FormatShape(InShape), LexToString(InType), (long long)Bytes, (long long)MaxSizeInBytes);
		return false;
	}

	OutNum = Count;
	return true;
}

FPipelineTensor FPipelineTensor::MakeZeroed(EPipelineTensorType InType, TConstArrayView<int64> InShape, int64 InNum)
{
	FPipelineTensor Result;
	const int64 Bytes = InNum * GetPipelineTensorTypeSize(InType);
	if (Bytes > 0)
	{
		Result.Data = FMemory::Malloc((SIZE_T)Bytes, UE::ModelPipelines::Private::TensorAlignment);
		if (Result.Data == nullptr)
		{
			UE_LOG(LogModelPipelines, Warning, TEXT("FPipelineTensor: allocation of %lld bytes failed; returning an invalid tensor."), (long long)Bytes);
			return FPipelineTensor();
		}
		FMemory::Memzero(Result.Data, (SIZE_T)Bytes);
	}
	Result.Shape.Append(InShape.GetData(), InShape.Num());
	Result.NumElements = InNum;
	Result.Type = InType;
	Result.bValid = true;
	return Result;
}

FPipelineTensor FPipelineTensor::Zeros(EPipelineTensorType InType, TConstArrayView<int64> InShape)
{
	int64 Count = 0;
	if (!ValidateShape(InType, InShape, Count, TEXT("FPipelineTensor::Zeros")))
	{
		return FPipelineTensor();
	}
	return MakeZeroed(InType, InShape, Count);
}

namespace UE::ModelPipelines::Private
{
	template <typename T>
	FPipelineTensor MakeFromValues(FPipelineTensor Zeroed, TConstArrayView<T> Values)
	{
		const int64 Bytes = Zeroed.GetSizeInBytes();
		if (Zeroed.IsValid() && Bytes > 0)
		{
			FMemory::Memcpy(Zeroed.GetRawData(), Values.GetData(), (SIZE_T)Bytes);
		}
		return Zeroed;
	}
}

FPipelineTensor FPipelineTensor::FromFloat(TConstArrayView<int64> InShape, TConstArrayView<float> Values)
{
	int64 Count = 0;
	if (!ValidateShape(EPipelineTensorType::Float32, InShape, Count, TEXT("FPipelineTensor::FromFloat")))
	{
		return FPipelineTensor();
	}
	if ((int64)Values.Num() != Count)
	{
		UE_LOG(LogModelPipelines, Warning, TEXT("FPipelineTensor::FromFloat: %d values for shape %s, which holds %lld elements."),
			Values.Num(), *UE::ModelPipelines::Private::FormatShape(InShape), (long long)Count);
		return FPipelineTensor();
	}
	return UE::ModelPipelines::Private::MakeFromValues<float>(MakeZeroed(EPipelineTensorType::Float32, InShape, Count), Values);
}

FPipelineTensor FPipelineTensor::FromInt64(TConstArrayView<int64> InShape, TConstArrayView<int64> Values)
{
	int64 Count = 0;
	if (!ValidateShape(EPipelineTensorType::Int64, InShape, Count, TEXT("FPipelineTensor::FromInt64")))
	{
		return FPipelineTensor();
	}
	if ((int64)Values.Num() != Count)
	{
		UE_LOG(LogModelPipelines, Warning, TEXT("FPipelineTensor::FromInt64: %d values for shape %s, which holds %lld elements."),
			Values.Num(), *UE::ModelPipelines::Private::FormatShape(InShape), (long long)Count);
		return FPipelineTensor();
	}
	return UE::ModelPipelines::Private::MakeFromValues<int64>(MakeZeroed(EPipelineTensorType::Int64, InShape, Count), Values);
}

FPipelineTensor FPipelineTensor::FromInt32(TConstArrayView<int64> InShape, TConstArrayView<int32> Values)
{
	int64 Count = 0;
	if (!ValidateShape(EPipelineTensorType::Int32, InShape, Count, TEXT("FPipelineTensor::FromInt32")))
	{
		return FPipelineTensor();
	}
	if ((int64)Values.Num() != Count)
	{
		UE_LOG(LogModelPipelines, Warning, TEXT("FPipelineTensor::FromInt32: %d values for shape %s, which holds %lld elements."),
			Values.Num(), *UE::ModelPipelines::Private::FormatShape(InShape), (long long)Count);
		return FPipelineTensor();
	}
	return UE::ModelPipelines::Private::MakeFromValues<int32>(MakeZeroed(EPipelineTensorType::Int32, InShape, Count), Values);
}

bool FPipelineTensor::IsValid() const
{
	return bValid;
}

EPipelineTensorType FPipelineTensor::GetType() const
{
	return Type;
}

TConstArrayView<int64> FPipelineTensor::GetShape() const
{
	if (!bValid)
	{
		return TConstArrayView<int64>();
	}
	return TConstArrayView<int64>(Shape.GetData(), Shape.Num());
}

int32 FPipelineTensor::GetRank() const
{
	return bValid ? Shape.Num() : 0;
}

int64 FPipelineTensor::Num() const
{
	return bValid ? NumElements : 0;
}

int64 FPipelineTensor::GetSizeInBytes() const
{
	return bValid ? NumElements * GetPipelineTensorTypeSize(Type) : 0;
}

template <typename T>
TArrayView64<T> FPipelineTensor::GetTypedView(EPipelineTensorType Expected) const
{
	if (!bValid || Type != Expected || Data == nullptr || NumElements <= 0)
	{
		return TArrayView64<T>();
	}
	return TArrayView64<T>(static_cast<T*>(Data), NumElements);
}

TArrayView64<float> FPipelineTensor::GetFloatData()
{
	return GetTypedView<float>(EPipelineTensorType::Float32);
}

TConstArrayView64<float> FPipelineTensor::GetFloatData() const
{
	return GetTypedView<float>(EPipelineTensorType::Float32);
}

TArrayView64<int64> FPipelineTensor::GetInt64Data()
{
	return GetTypedView<int64>(EPipelineTensorType::Int64);
}

TConstArrayView64<int64> FPipelineTensor::GetInt64Data() const
{
	return GetTypedView<int64>(EPipelineTensorType::Int64);
}

TArrayView64<int32> FPipelineTensor::GetInt32Data()
{
	return GetTypedView<int32>(EPipelineTensorType::Int32);
}

TConstArrayView64<int32> FPipelineTensor::GetInt32Data() const
{
	return GetTypedView<int32>(EPipelineTensorType::Int32);
}

void* FPipelineTensor::GetRawData()
{
	return (bValid && NumElements > 0) ? Data : nullptr;
}

const void* FPipelineTensor::GetRawData() const
{
	return (bValid && NumElements > 0) ? Data : nullptr;
}

bool FPipelineTensor::Reshape(TConstArrayView<int64> NewShape)
{
	if (!bValid)
	{
		return false;
	}
	int64 Count = 0;
	if (!ValidateShape(Type, NewShape, Count, TEXT("FPipelineTensor::Reshape")))
	{
		return false;
	}
	if (Count != NumElements)
	{
		return false;
	}
	Shape.Reset();
	Shape.Append(NewShape.GetData(), NewShape.Num());
	return true;
}

FString FPipelineTensor::DescribeShape() const
{
	if (!bValid)
	{
		return FString(TEXT("<invalid>"));
	}
	return UE::ModelPipelines::Private::FormatShape(TConstArrayView<int64>(Shape.GetData(), Shape.Num()));
}

bool FPipelineTensorDesc::IsCompatible(const FPipelineTensor& T) const
{
	if (!bTypeSupported || !T.IsValid() || T.GetType() != Type)
	{
		return false;
	}
	const TConstArrayView<int64> TShape = T.GetShape();
	if (TShape.Num() != Shape.Num())
	{
		return false;
	}
	for (int32 Index = 0; Index < Shape.Num(); ++Index)
	{
		if (Shape[Index] != -1 && Shape[Index] != TShape[Index])
		{
			return false;
		}
	}
	return true;
}
