// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "PipelineLog.h"

BEGIN_DEFINE_SPEC(FPipelineTensorSpec, "Pipelines.Core.Tensor",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	/** Checks the tensor is the documented "invalid" value. */
	void ExpectInvalid(const FPipelineTensor& T, const TCHAR* What)
	{
		TestFalse(FString::Printf(TEXT("%s: IsValid() should be false"), What), T.IsValid());
		TestEqual(FString::Printf(TEXT("%s: Num() should be 0"), What), T.Num(), (int64)0);
		TestEqual(FString::Printf(TEXT("%s: DescribeShape() should be <invalid>"), What), T.DescribeShape(), FString(TEXT("<invalid>")));
		TestEqual(FString::Printf(TEXT("%s: size in bytes should be 0"), What), T.GetSizeInBytes(), (int64)0);
		TestTrue(FString::Printf(TEXT("%s: raw data should be null"), What), T.GetRawData() == nullptr);
	}

END_DEFINE_SPEC(FPipelineTensorSpec)

void FPipelineTensorSpec::Define()
{
	Describe("Float tensor", [this]()
	{
		It("builds a {2,3} float tensor with correct metadata and typed views", [this]()
		{
			const TArray<int64> Shape = {2, 3};
			const TArray<float> Vals = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
			FPipelineTensor T = FPipelineTensor::FromFloat(Shape, Vals);
			TestTrue(TEXT("IsValid expected true"), T.IsValid());
			TestEqual(TEXT("rank expected 2"), T.GetRank(), 2);
			TestEqual(TEXT("Num expected 6"), T.Num(), (int64)6);
			TestEqual(TEXT("size in bytes expected 24"), T.GetSizeInBytes(), (int64)24);
			TestTrue(TEXT("type expected Float32"), T.GetType() == EPipelineTensorType::Float32);
			TestEqual(TEXT("DescribeShape expected [2,3]"), T.DescribeShape(), FString(TEXT("[2,3]")));
			TestEqual(TEXT("shape view length expected 2"), T.GetShape().Num(), 2);
			TConstArrayView64<float> F = T.GetFloatData();
			TestEqual(TEXT("float view length expected 6"), (int64)F.Num(), (int64)6);
			if (F.Num() == 6)
			{
				// Catches a row-major index mixup or a copy that dropped values.
				TestEqual(TEXT("GetFloatData()[4] expected 5"), F[4], 5.f);
				for (int32 i = 0; i < 6; ++i)
				{
					TestEqual(*FString::Printf(TEXT("value %d expected %d"), i, i + 1), F[i], (float)(i + 1));
				}
			}
			// Wrong-type views must be empty, not reinterpretations of the bytes.
			TestEqual(TEXT("int64 view of a float tensor expected empty"), (int64)T.GetInt64Data().Num(), (int64)0);
			TestEqual(TEXT("int32 view of a float tensor expected empty"), (int64)T.GetInt32Data().Num(), (int64)0);
		});
	});

	Describe("Integer tensors", [this]()
	{
		It("round-trips int64 values beyond 32 bits", [this]()
		{
			const TArray<int64> Shape = {3};
			const int64 Big = 1ll << 40;
			const TArray<int64> Vals = {-Big, 0, Big};
			FPipelineTensor T = FPipelineTensor::FromInt64(Shape, Vals);
			TestTrue(TEXT("IsValid expected true"), T.IsValid());
			TestTrue(TEXT("type expected Int64"), T.GetType() == EPipelineTensorType::Int64);
			TestEqual(TEXT("size in bytes expected 24"), T.GetSizeInBytes(), (int64)24);
			TConstArrayView64<int64> V = T.GetInt64Data();
			TestEqual(TEXT("int64 view length expected 3"), (int64)V.Num(), (int64)3);
			if (V.Num() == 3)
			{
				// Truncation to 32 bits would break these.
				TestEqual(TEXT("value 0 expected -(1<<40)"), V[0], -Big);
				TestEqual(TEXT("value 1 expected 0"), V[1], (int64)0);
				TestEqual(TEXT("value 2 expected 1<<40"), V[2], Big);
			}
			TestEqual(TEXT("float view of an int64 tensor expected empty"), (int64)T.GetFloatData().Num(), (int64)0);
			TestEqual(TEXT("int32 view of an int64 tensor expected empty"), (int64)T.GetInt32Data().Num(), (int64)0);
		});

		It("builds a {1,1,2} int32 tensor", [this]()
		{
			const TArray<int64> Shape = {1, 1, 2};
			const TArray<int32> Vals = {7, -7};
			FPipelineTensor T = FPipelineTensor::FromInt32(Shape, Vals);
			TestTrue(TEXT("IsValid expected true"), T.IsValid());
			TestEqual(TEXT("Num expected 2"), T.Num(), (int64)2);
			TestEqual(TEXT("rank expected 3"), T.GetRank(), 3);
			TestEqual(TEXT("size in bytes expected 8"), T.GetSizeInBytes(), (int64)8);
			TestEqual(TEXT("DescribeShape expected [1,1,2]"), T.DescribeShape(), FString(TEXT("[1,1,2]")));
			TConstArrayView64<int32> V = T.GetInt32Data();
			if (TestEqual(TEXT("int32 view length expected 2"), (int64)V.Num(), (int64)2))
			{
				TestEqual(TEXT("value 0 expected 7"), V[0], 7);
				TestEqual(TEXT("value 1 expected -7"), V[1], -7);
			}
		});
	});

	Describe("Scalars and empty tensors", [this]()
	{
		It("Zeros(Int32, {}) is a scalar holding 0", [this]()
		{
			FPipelineTensor T = FPipelineTensor::Zeros(EPipelineTensorType::Int32, TConstArrayView<int64>());
			TestTrue(TEXT("scalar should be valid"), T.IsValid());
			TestEqual(TEXT("scalar Num expected 1"), T.Num(), (int64)1);
			TestEqual(TEXT("scalar rank expected 0"), T.GetRank(), 0);
			TestEqual(TEXT("scalar DescribeShape expected []"), T.DescribeShape(), FString(TEXT("[]")));
			TestEqual(TEXT("scalar shape view expected empty"), T.GetShape().Num(), 0);
			TestEqual(TEXT("scalar size in bytes expected 4"), T.GetSizeInBytes(), (int64)4);
			TConstArrayView64<int32> V = T.GetInt32Data();
			if (TestEqual(TEXT("scalar int32 view length expected 1"), (int64)V.Num(), (int64)1))
			{
				TestEqual(TEXT("scalar value expected 0"), V[0], 0);
			}
		});

		It("Zeros(Float32, {4,0,3}) is valid with no elements and null raw data", [this]()
		{
			const TArray<int64> Shape = {4, 0, 3};
			FPipelineTensor T = FPipelineTensor::Zeros(EPipelineTensorType::Float32, Shape);
			TestTrue(TEXT("zero-dim tensor should be valid"), T.IsValid());
			TestEqual(TEXT("Num expected 0"), T.Num(), (int64)0);
			TestEqual(TEXT("size in bytes expected 0"), T.GetSizeInBytes(), (int64)0);
			TestTrue(TEXT("GetRawData expected nullptr for 0 elements"), T.GetRawData() == nullptr);
			TestEqual(TEXT("rank expected 3"), T.GetRank(), 3);
			TestEqual(TEXT("DescribeShape expected [4,0,3]"), T.DescribeShape(), FString(TEXT("[4,0,3]")));
			TestEqual(TEXT("float view expected empty"), (int64)T.GetFloatData().Num(), (int64)0);
		});

		It("Zeros(Float32, {3,5}) is all zeros", [this]()
		{
			const TArray<int64> Shape = {3, 5};
			FPipelineTensor T = FPipelineTensor::Zeros(EPipelineTensorType::Float32, Shape);
			TestTrue(TEXT("should be valid"), T.IsValid());
			TestEqual(TEXT("Num expected 15"), T.Num(), (int64)15);
			TConstArrayView64<float> F = T.GetFloatData();
			if (TestEqual(TEXT("float view length expected 15"), (int64)F.Num(), (int64)15))
			{
				int32 NonZero = 0;
				for (int64 i = 0; i < F.Num(); ++i)
				{
					NonZero += (F[i] != 0.f) ? 1 : 0;
				}
				TestEqual(TEXT("non-zero elements in a fresh Zeros tensor expected 0 (uninitialized memory?)"), NonZero, 0);
			}
		});

		It("Zeros works for int64", [this]()
		{
			const TArray<int64> Shape = {2, 2};
			FPipelineTensor T = FPipelineTensor::Zeros(EPipelineTensorType::Int64, Shape);
			TestTrue(TEXT("should be valid"), T.IsValid());
			TestEqual(TEXT("size in bytes expected 32"), T.GetSizeInBytes(), (int64)32);
			TConstArrayView64<int64> V = T.GetInt64Data();
			TestEqual(TEXT("int64 view length expected 4"), (int64)V.Num(), (int64)4);
			for (int64 i = 0; i < V.Num(); ++i)
			{
				TestEqual(TEXT("int64 zero element"), V[i], (int64)0);
			}
		});
	});

	Describe("Default construction", [this]()
	{
		It("default tensor is invalid, rank 0, Float32", [this]()
		{
			FPipelineTensor T;
			ExpectInvalid(T, TEXT("default"));
			TestEqual(TEXT("default rank expected 0"), T.GetRank(), 0);
			TestTrue(TEXT("default type expected Float32"), T.GetType() == EPipelineTensorType::Float32);
			TestEqual(TEXT("default shape view expected empty"), T.GetShape().Num(), 0);
			TestEqual(TEXT("default float view expected empty"), (int64)T.GetFloatData().Num(), (int64)0);
			TestEqual(TEXT("default int64 view expected empty"), (int64)T.GetInt64Data().Num(), (int64)0);
			TestEqual(TEXT("default int32 view expected empty"), (int64)T.GetInt32Data().Num(), (int64)0);
		});
	});

	Describe("Invalid inputs", [this]()
	{
		// These log Warnings to LogModelPipelines; warnings do not fail the test.
		It("rejects a value count that differs from the element count", [this]()
		{
			AddExpectedMessage(TEXT("FPipelineTensor::FromFloat: "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 2, false);
			AddExpectedMessage(TEXT("FPipelineTensor::FromInt"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 2, false);
			const TArray<int64> Shape = {2, 2};
			const TArray<float> Vals = {1.f, 2.f, 3.f};
			ExpectInvalid(FPipelineTensor::FromFloat(Shape, Vals), TEXT("FromFloat {2,2} with 3 values"));
			const TArray<float> TooMany = {1.f, 2.f, 3.f, 4.f, 5.f};
			ExpectInvalid(FPipelineTensor::FromFloat(Shape, TooMany), TEXT("FromFloat {2,2} with 5 values"));
			const TArray<int64> I64 = {1, 2, 3};
			ExpectInvalid(FPipelineTensor::FromInt64(Shape, I64), TEXT("FromInt64 {2,2} with 3 values"));
			const TArray<int32> I32 = {1};
			ExpectInvalid(FPipelineTensor::FromInt32(Shape, I32), TEXT("FromInt32 {2,2} with 1 value"));
		});

		It("rejects a negative dim", [this]()
		{
			AddExpectedMessage(TEXT("FPipelineTensor::Zeros: dim "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			AddExpectedMessage(TEXT("FPipelineTensor::FromFloat: dim "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			const TArray<int64> Shape = {2, -1};
			ExpectInvalid(FPipelineTensor::Zeros(EPipelineTensorType::Float32, Shape), TEXT("Zeros {2,-1}"));
			const TArray<float> Vals = {1.f, 2.f};
			ExpectInvalid(FPipelineTensor::FromFloat(Shape, Vals), TEXT("FromFloat {2,-1}"));
		});

		It("rejects rank 9 but accepts rank 8", [this]()
		{
			AddExpectedMessage(TEXT("FPipelineTensor::Zeros: rank "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			const TArray<int64> Rank9 = {1, 1, 1, 1, 1, 1, 1, 1, 1};
			ExpectInvalid(FPipelineTensor::Zeros(EPipelineTensorType::Float32, Rank9), TEXT("Zeros rank 9"));
			const TArray<int64> Rank8 = {1, 1, 1, 1, 1, 1, 1, 1};
			FPipelineTensor T = FPipelineTensor::Zeros(EPipelineTensorType::Float32, Rank8);
			TestTrue(TEXT("rank 8 (MaxRank) expected valid"), T.IsValid());
			TestEqual(TEXT("rank 8 expected GetRank 8"), T.GetRank(), 8);
		});

		It("rejects a dim larger than MAX_uint32", [this]()
		{
			AddExpectedMessage(TEXT("FPipelineTensor::Zeros: dim "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			const TArray<int64> Shape = {1ll << 33};
			ExpectInvalid(FPipelineTensor::Zeros(EPipelineTensorType::Int32, Shape), TEXT("Zeros {1<<33}"));
		});

		It("rejects byte-size overflow without allocating", [this]()
		{
			AddExpectedMessage(TEXT("FPipelineTensor::Zeros: element count or byte size of shape "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 2, false);
			const TArray<int64> Shape = {1ll << 31, 1ll << 31};
			ExpectInvalid(FPipelineTensor::Zeros(EPipelineTensorType::Float32, Shape), TEXT("Zeros {1<<31,1<<31}"));
			// Element count itself overflows int64 when many large dims multiply.
			const TArray<int64> Many = {4000000000ll, 4000000000ll, 4000000000ll, 4000000000ll};
			ExpectInvalid(FPipelineTensor::Zeros(EPipelineTensorType::Int64, Many), TEXT("Zeros four 4e9 dims"));
		});

		It("rejects over 4 GiB without allocating", [this]()
		{
			AddExpectedMessage(TEXT("FPipelineTensor::Zeros: shape "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			// 2^31 * 2 floats = 8 GiB. A wrapper that allocated first would hit OOM or take very long.
			const TArray<int64> Shape = {1ll << 31, 2};
			ExpectInvalid(FPipelineTensor::Zeros(EPipelineTensorType::Float32, Shape), TEXT("Zeros {1<<31,2}"));
		});
	});

	Describe("Alignment", [this]()
	{
		It("raw data is 16-byte aligned, including for a 1-element tensor", [this]()
		{
			const TArray<int64> One = {1};
			const TArray<float> OneVal = {3.f};
			FPipelineTensor A = FPipelineTensor::FromFloat(One, OneVal);
			TestTrue(TEXT("1-element raw data expected non-null"), A.GetRawData() != nullptr);
			TestEqual(TEXT("1-element raw data address % 16 expected 0"), (int64)(reinterpret_cast<UPTRINT>(A.GetRawData()) % 16), (int64)0);

			FPipelineTensor B = FPipelineTensor::Zeros(EPipelineTensorType::Int32, TConstArrayView<int64>());
			TestTrue(TEXT("scalar raw data expected non-null"), B.GetRawData() != nullptr);
			TestEqual(TEXT("scalar raw data address % 16 expected 0"), (int64)(reinterpret_cast<UPTRINT>(B.GetRawData()) % 16), (int64)0);

			const TArray<int64> Odd = {3, 5, 7};
			FPipelineTensor C = FPipelineTensor::Zeros(EPipelineTensorType::Int64, Odd);
			TestTrue(TEXT("{3,5,7} int64 raw data expected non-null"), C.GetRawData() != nullptr);
			TestEqual(TEXT("{3,5,7} raw data address % 16 expected 0"), (int64)(reinterpret_cast<UPTRINT>(C.GetRawData()) % 16), (int64)0);

			// The const overload must agree with the mutable one.
			const FPipelineTensor& CRef = C;
			TestTrue(TEXT("const and mutable GetRawData expected to match"), CRef.GetRawData() == C.GetRawData());
		});
	});

	Describe("Copy and move", [this]()
	{
		It("copies are deep", [this]()
		{
			const TArray<int64> Shape = {2, 2};
			const TArray<float> Vals = {1.f, 2.f, 3.f, 4.f};
			FPipelineTensor Original = FPipelineTensor::FromFloat(Shape, Vals);
			FPipelineTensor Copy = Original;
			TestTrue(TEXT("copy expected valid"), Copy.IsValid());
			TestTrue(TEXT("copy should not share the buffer"), Copy.GetRawData() != Original.GetRawData());
			Copy.GetFloatData()[0] = 99.f;
			TestEqual(TEXT("original[0] expected unchanged 1 after editing the copy"), Original.GetFloatData()[0], 1.f);
			TestEqual(TEXT("copy[0] expected 99"), Copy.GetFloatData()[0], 99.f);
			TestEqual(TEXT("copy[3] expected 4"), Copy.GetFloatData()[3], 4.f);

			// Copy-assignment too.
			FPipelineTensor Assigned;
			Assigned = Original;
			Assigned.GetFloatData()[1] = -5.f;
			TestEqual(TEXT("original[1] expected unchanged 2 after editing the assigned copy"), Original.GetFloatData()[1], 2.f);
			TestEqual(TEXT("assigned shape expected [2,2]"), Assigned.DescribeShape(), FString(TEXT("[2,2]")));
		});

		It("a moved-from tensor is safe to query", [this]()
		{
			const TArray<int64> Shape = {2, 2};
			const TArray<float> Vals = {1.f, 2.f, 3.f, 4.f};
			FPipelineTensor Source = FPipelineTensor::FromFloat(Shape, Vals);
			FPipelineTensor Dest = MoveTemp(Source);
			TestTrue(TEXT("move destination expected valid"), Dest.IsValid());
			TestEqual(TEXT("move destination Num expected 4"), Dest.Num(), (int64)4);
			if (Dest.GetFloatData().Num() == 4)
			{
				TestEqual(TEXT("move destination [3] expected 4"), Dest.GetFloatData()[3], 4.f);
			}
			// Only that these calls do not crash is asserted; the moved-from state is unspecified but safe.
			const bool bSrcValid = Source.IsValid();
			const int64 SrcNum = Source.Num();
			const int64 F = Source.GetFloatData().Num();
			const int64 I64 = Source.GetInt64Data().Num();
			const int64 I32 = Source.GetInt32Data().Num();
			const void* Raw = Source.GetRawData();
			const FString Desc = Source.DescribeShape();
			TestTrue(TEXT("moved-from queries completed without crashing"), true);
			if (!bSrcValid)
			{
				TestEqual(TEXT("invalid moved-from Num expected 0"), SrcNum, (int64)0);
				TestEqual(TEXT("invalid moved-from views expected empty"), F + I64 + I32, (int64)0);
				TestTrue(TEXT("invalid moved-from raw expected null"), Raw == nullptr);
			}
			(void)Desc;

			// Move-assign.
			FPipelineTensor Target;
			Target = MoveTemp(Dest);
			TestTrue(TEXT("move-assigned expected valid"), Target.IsValid());
			TestEqual(TEXT("move-assigned Num expected 4"), Target.Num(), (int64)4);
			TestEqual(TEXT("moved-from Dest IsValid call is safe"), Dest.Num() >= 0, true);
		});
	});

	Describe("Reshape", [this]()
	{
		It("reshapes to same element count and keeps the data", [this]()
		{
			const TArray<int64> Shape = {2, 3};
			const TArray<float> Vals = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
			FPipelineTensor T = FPipelineTensor::FromFloat(Shape, Vals);
			const TArray<int64> To32 = {3, 2};
			TestTrue(TEXT("{2,3} -> {3,2} expected true"), T.Reshape(To32));
			TestEqual(TEXT("shape expected [3,2]"), T.DescribeShape(), FString(TEXT("[3,2]")));
			TConstArrayView64<float> F = T.GetFloatData();
			if (TestEqual(TEXT("data length expected 6"), (int64)F.Num(), (int64)6))
			{
				for (int32 i = 0; i < 6; ++i)
				{
					TestEqual(*FString::Printf(TEXT("value %d expected %d after reshape"), i, i + 1), F[i], (float)(i + 1));
				}
			}
			const TArray<int64> Flat = {6};
			TestTrue(TEXT("{3,2} -> {6} expected true"), T.Reshape(Flat));
			TestEqual(TEXT("rank expected 1"), T.GetRank(), 1);
			TestEqual(TEXT("shape expected [6]"), T.DescribeShape(), FString(TEXT("[6]")));
		});

		It("rejects a different element count and leaves the tensor unchanged", [this]()
		{
			const TArray<int64> Shape = {2, 3};
			const TArray<float> Vals = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
			FPipelineTensor T = FPipelineTensor::FromFloat(Shape, Vals);
			const TArray<int64> Bad = {4, 2};
			TestFalse(TEXT("{2,3} -> {4,2} expected false"), T.Reshape(Bad));
			TestEqual(TEXT("shape expected unchanged [2,3]"), T.DescribeShape(), FString(TEXT("[2,3]")));
			TestEqual(TEXT("Num expected unchanged 6"), T.Num(), (int64)6);
			TestEqual(TEXT("data expected unchanged"), T.GetFloatData()[5], 6.f);
		});

		It("does not infer -1", [this]()
		{
			AddExpectedMessage(TEXT("FPipelineTensor::Reshape: dim "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			AddExpectedMessage(TEXT("FPipelineTensor::Reshape: rank "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			const TArray<int64> Shape = {2, 3};
			const TArray<float> Vals = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
			FPipelineTensor T = FPipelineTensor::FromFloat(Shape, Vals);
			const TArray<int64> Infer = {-1, 3};
			TestFalse(TEXT("{2,3} -> {-1,3} expected false"), T.Reshape(Infer));
			TestEqual(TEXT("shape expected unchanged [2,3]"), T.DescribeShape(), FString(TEXT("[2,3]")));
			// Rank above the max is also refused.
			const TArray<int64> Rank9 = {1, 1, 1, 1, 1, 1, 1, 1, 6};
			TestFalse(TEXT("reshape to rank 9 expected false"), T.Reshape(Rank9));
			TestEqual(TEXT("shape expected unchanged [2,3] after rank-9 attempt"), T.DescribeShape(), FString(TEXT("[2,3]")));
		});

		It("returns false on an invalid tensor", [this]()
		{
			AddExpectedMessage(TEXT("FPipelineTensor::Zeros: dim "), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1, false);
			FPipelineTensor T;
			const TArray<int64> Shape = {1};
			TestFalse(TEXT("reshape of default tensor expected false"), T.Reshape(Shape));
			TestFalse(TEXT("default tensor must stay invalid"), T.IsValid());
			const TArray<int64> Bad = {2, -1};
			FPipelineTensor Invalid = FPipelineTensor::Zeros(EPipelineTensorType::Float32, Bad);
			TestFalse(TEXT("reshape of factory-rejected tensor expected false"), Invalid.Reshape(Shape));
		});
	});

	Describe("FPipelineTensorDesc", [this]()
	{
		It("IsCompatible honours dynamic dims, type and rank", [this]()
		{
			FPipelineTensorDesc Desc;
			Desc.Name = TEXT("x");
			Desc.Type = EPipelineTensorType::Float32;
			Desc.bTypeSupported = true;
			Desc.Shape = {-1, 4};

			const TArray<int64> S34 = {3, 4};
			const TArray<int64> S35 = {3, 5};
			const TArray<int64> S4 = {4};
			TestTrue(TEXT("float {3,4} vs {-1,4} expected compatible"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Float32, S34)));
			TestFalse(TEXT("float {3,5} vs {-1,4} expected incompatible (fixed dim differs)"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Float32, S35)));
			TestFalse(TEXT("float {4} vs {-1,4} expected incompatible (rank differs)"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Float32, S4)));
			TestFalse(TEXT("int32 {3,4} vs float desc expected incompatible (type differs)"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Int32, S34)));
			TestFalse(TEXT("invalid tensor expected incompatible"), Desc.IsCompatible(FPipelineTensor()));

			// A dynamic dim of size 0 is still a match.
			const TArray<int64> S04 = {0, 4};
			TestTrue(TEXT("float {0,4} vs {-1,4} expected compatible"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Float32, S04)));
		});

		It("an unsupported-type desc is compatible with nothing", [this]()
		{
			FPipelineTensorDesc Desc;
			Desc.Name = TEXT("x");
			Desc.Type = EPipelineTensorType::Float32;
			Desc.bTypeSupported = false;
			Desc.Shape = {-1, 4};
			const TArray<int64> S34 = {3, 4};
			TestFalse(TEXT("bTypeSupported=false expected incompatible even for matching float"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Float32, S34)));
			TestFalse(TEXT("bTypeSupported=false expected incompatible for invalid tensor"), Desc.IsCompatible(FPipelineTensor()));
		});

		It("a fixed {2,4} desc matches only {2,4}", [this]()
		{
			FPipelineTensorDesc Desc;
			Desc.Name = TEXT("y");
			Desc.Type = EPipelineTensorType::Float32;
			Desc.Shape = {2, 4};
			const TArray<int64> S24 = {2, 4};
			const TArray<int64> S34 = {3, 4};
			const TArray<int64> S25 = {2, 5};
			TestTrue(TEXT("{2,4} expected compatible"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Float32, S24)));
			TestFalse(TEXT("{3,4} expected incompatible"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Float32, S34)));
			TestFalse(TEXT("{2,5} expected incompatible"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Float32, S25)));
		});

		It("a scalar desc matches a scalar tensor only", [this]()
		{
			FPipelineTensorDesc Desc;
			Desc.Type = EPipelineTensorType::Int64;
			TestTrue(TEXT("int64 scalar expected compatible with rank-0 desc"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Int64, TConstArrayView<int64>())));
			const TArray<int64> S1 = {1};
			TestFalse(TEXT("int64 {1} expected incompatible with rank-0 desc"), Desc.IsCompatible(FPipelineTensor::Zeros(EPipelineTensorType::Int64, S1)));
		});
	});

	Describe("Results and helpers", [this]()
	{
		It("FPipelineResult::Ok and Fail", [this]()
		{
			const FPipelineResult Ok = FPipelineResult::Ok();
			TestTrue(TEXT("Ok bSuccess expected true"), Ok.bSuccess);
			TestTrue(TEXT("Ok Error expected None"), Ok.Error == EPipelineError::None);
			TestTrue(TEXT("Ok Message expected empty"), Ok.Message.IsEmpty());

			const FPipelineResult Fail = FPipelineResult::Fail(EPipelineError::ShapeMismatch, TEXT("bad"));
			TestFalse(TEXT("Fail bSuccess expected false"), Fail.bSuccess);
			TestTrue(TEXT("Fail Error expected ShapeMismatch"), Fail.Error == EPipelineError::ShapeMismatch);
			TestEqual(TEXT("Fail Message expected bad"), Fail.Message, FString(TEXT("bad")));

			const FPipelineResult Def;
			TestFalse(TEXT("default result bSuccess expected false"), Def.bSuccess);
			TestTrue(TEXT("default result Device expected Auto"), Def.Device == EPipelineDevice::Auto);
			TestTrue(TEXT("default result Error expected None"), Def.Error == EPipelineError::None);
			TestEqual(TEXT("default QueueMs expected 0"), Def.QueueMs, 0.f);
			TestEqual(TEXT("default InferenceMs expected 0"), Def.InferenceMs, 0.f);
		});

		It("enum values match the contract", [this]()
		{
			TestEqual(TEXT("Device Auto"), (int32)EPipelineDevice::Auto, 0);
			TestEqual(TEXT("Device CPU"), (int32)EPipelineDevice::CPU, 1);
			TestEqual(TEXT("Device GPU"), (int32)EPipelineDevice::GPU, 2);
			TestEqual(TEXT("Error InvalidArgument"), (int32)EPipelineError::InvalidArgument, 1);
			TestEqual(TEXT("Error RuntimeUnavailable"), (int32)EPipelineError::RuntimeUnavailable, 4);
			TestEqual(TEXT("Error Cancelled"), (int32)EPipelineError::Cancelled, 7);
			TestEqual(TEXT("Error Internal"), (int32)EPipelineError::Internal, 9);
			TestEqual(TEXT("TensorType Float32"), (int32)EPipelineTensorType::Float32, 0);
			TestEqual(TEXT("TensorType Int64"), (int32)EPipelineTensorType::Int64, 1);
			TestEqual(TEXT("TensorType Int32"), (int32)EPipelineTensorType::Int32, 2);
		});

		It("type size and LexToString", [this]()
		{
			TestEqual(TEXT("Float32 size expected 4"), GetPipelineTensorTypeSize(EPipelineTensorType::Float32), 4);
			TestEqual(TEXT("Int64 size expected 8"), GetPipelineTensorTypeSize(EPipelineTensorType::Int64), 8);
			TestEqual(TEXT("Int32 size expected 4"), GetPipelineTensorTypeSize(EPipelineTensorType::Int32), 4);
			TestEqual(TEXT("Float32 name expected float32"), FString(LexToString(EPipelineTensorType::Float32)), FString(TEXT("float32")));
			TestEqual(TEXT("Int64 name expected int64"), FString(LexToString(EPipelineTensorType::Int64)), FString(TEXT("int64")));
			TestEqual(TEXT("Int32 name expected int32"), FString(LexToString(EPipelineTensorType::Int32)), FString(TEXT("int32")));
		});

		It("DescribeShape formats ranks and limits", [this]()
		{
			TestEqual(TEXT("MaxRank expected 8"), FPipelineTensor::MaxRank, 8);
			TestEqual(TEXT("MaxSizeInBytes expected 4 GiB"), FPipelineTensor::MaxSizeInBytes, (int64)4 * 1024 * 1024 * 1024);
			const TArray<int64> Shape = {2, 4};
			TestEqual(TEXT("expected [2,4]"), FPipelineTensor::Zeros(EPipelineTensorType::Float32, Shape).DescribeShape(), FString(TEXT("[2,4]")));
			const TArray<int64> One = {7};
			TestEqual(TEXT("expected [7]"), FPipelineTensor::Zeros(EPipelineTensorType::Int32, One).DescribeShape(), FString(TEXT("[7]")));
		});

		It("LogModelPipelines is usable", [this]()
		{
			UE_LOG(LogModelPipelines, Verbose, TEXT("PipelineTensorSpec log category check"));
			TestTrue(TEXT("logging to LogModelPipelines compiled and ran"), true);
		});
	});

	Describe("Threads", [this]()
	{
		It("8 threads build, copy and read their own tensors 1000 times", [this]()
		{
			constexpr int32 NumThreads = 8;
			TArray<TFuture<int32>> Futures;
			for (int32 ThreadIdx = 0; ThreadIdx < NumThreads; ++ThreadIdx)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [ThreadIdx]() -> int32
				{
					int32 Errors = 0;
					for (int32 Iter = 0; Iter < 1000; ++Iter)
					{
						TArray<float> Vals;
						Vals.SetNumUninitialized(1000);
						for (int32 i = 0; i < 1000; ++i)
						{
							Vals[i] = (float)(ThreadIdx * 100000 + Iter + i);
						}
						const TArray<int64> Shape = {10, 100};
						FPipelineTensor A = FPipelineTensor::FromFloat(Shape, Vals);
						FPipelineTensor B = A;
						if (!A.IsValid() || !B.IsValid() || A.Num() != 1000 || B.Num() != 1000)
						{
							++Errors;
							continue;
						}
						B.GetFloatData()[0] = -1.f;
						TConstArrayView64<float> FA = A.GetFloatData();
						TConstArrayView64<float> FB = B.GetFloatData();
						if (FA.Num() != 1000 || FB.Num() != 1000) { ++Errors; continue; }
						if (FA[0] != Vals[0] || FB[0] != -1.f) { ++Errors; }
						for (int32 i = 1; i < 1000; ++i)
						{
							if (FA[i] != Vals[i] || FB[i] != Vals[i]) { ++Errors; break; }
						}
					}
					return Errors;
				}));
			}

			for (int32 i = 0; i < NumThreads; ++i)
			{
				if (!Futures[i].WaitFor(FTimespan::FromSeconds(60)))
				{
					AddError(FString::Printf(TEXT("thread %d did not finish within 60 s"), i));
					// Cannot safely abandon running tasks; wait so locals stay valid.
					Futures[i].Wait();
				}
				TestEqual(*FString::Printf(TEXT("thread %d error count expected 0"), i), Futures[i].Get(), 0);
			}
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
