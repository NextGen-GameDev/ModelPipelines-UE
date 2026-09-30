// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineOnnxCheck.h"

namespace UE::ModelPipelines::Private
{
	namespace OnnxCheck
	{
		/** Reads a protobuf varint (at most 10 bytes) at Pos. Returns false if truncated or longer than 10 bytes. */
		static bool ReadVarint(const TConstArrayView64<uint8>& Bytes, uint64& Pos, uint64& OutValue)
		{
			const uint64 Size = (uint64)Bytes.Num();
			uint64 Value = 0;
			for (int32 i = 0; i < 10; ++i)
			{
				if (Pos >= Size)
				{
					return false;
				}
				const uint8 B = Bytes[(int64)Pos];
				++Pos;
				Value |= (uint64)(B & 0x7F) << (7 * i);
				if ((B & 0x80) == 0)
				{
					OutValue = Value;
					return true;
				}
			}
			return false;
		}
	}

	bool IsStructurallyValidOnnx(TConstArrayView64<uint8> Bytes, FString& OutReason)
	{
		using namespace OnnxCheck;

		const uint64 Size = (uint64)Bytes.Num();
		if (Size == 0)
		{
			OutReason = TEXT("the buffer is empty");
			return false;
		}

		bool bSawIrVersion = false;
		bool bSawGraph = false;
		uint64 Pos = 0;

		while (Pos < Size)
		{
			const uint64 FieldStart = Pos;
			uint64 Tag = 0;
			if (!ReadVarint(Bytes, Pos, Tag))
			{
				OutReason = FString::Printf(TEXT("truncated or overlong field tag at byte %llu"), (unsigned long long)FieldStart);
				return false;
			}
			const uint64 FieldNumber = Tag >> 3;
			const uint32 WireType = (uint32)(Tag & 7);
			if (FieldNumber == 0)
			{
				OutReason = FString::Printf(TEXT("field number 0 at byte %llu"), (unsigned long long)FieldStart);
				return false;
			}

			switch (WireType)
			{
			case 0:
			{
				uint64 Value = 0;
				if (!ReadVarint(Bytes, Pos, Value))
				{
					OutReason = FString::Printf(TEXT("truncated or overlong varint in field %llu at byte %llu"), (unsigned long long)FieldNumber, (unsigned long long)FieldStart);
					return false;
				}
				if (FieldNumber == 1)
				{
					bSawIrVersion = true;
				}
				break;
			}
			case 1:
			case 5:
			{
				const uint64 Width = WireType == 1 ? 8 : 4;
				if (Size - Pos < Width)
				{
					OutReason = FString::Printf(TEXT("fixed-width field %llu at byte %llu runs past the end"), (unsigned long long)FieldNumber, (unsigned long long)FieldStart);
					return false;
				}
				Pos += Width;
				break;
			}
			case 2:
			{
				uint64 Length = 0;
				if (!ReadVarint(Bytes, Pos, Length))
				{
					OutReason = FString::Printf(TEXT("truncated or overlong length in field %llu at byte %llu"), (unsigned long long)FieldNumber, (unsigned long long)FieldStart);
					return false;
				}
				// Pos <= Size here, so Size - Pos cannot underflow; comparing this way cannot overflow.
				if (Length > Size - Pos)
				{
					OutReason = FString::Printf(TEXT("field %llu at byte %llu declares %llu bytes but only %llu remain"),
						(unsigned long long)FieldNumber, (unsigned long long)FieldStart, (unsigned long long)Length, (unsigned long long)(Size - Pos));
					return false;
				}
				Pos += Length;
				if (FieldNumber == 7 && Length > 0)
				{
					bSawGraph = true;
				}
				break;
			}
			default:
				OutReason = FString::Printf(TEXT("unsupported wire type %u in field %llu at byte %llu"), WireType, (unsigned long long)FieldNumber, (unsigned long long)FieldStart);
				return false;
			}
		}

		if (!bSawIrVersion)
		{
			OutReason = TEXT("missing ir_version (field 1)");
			return false;
		}
		if (!bSawGraph)
		{
			OutReason = TEXT("missing or empty graph (field 7)");
			return false;
		}
		return true;
	}
}
