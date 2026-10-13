// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineFileHash.h"

#include "HAL/FileManager.h"
#include "Serialization/Archive.h"
#include "Templates/UniquePtr.h"

namespace PipelineFileHashPrivate
{
	/** SHA-256 per FIPS 180-4. Plain state, no globals: safe on any number of threads. */
	struct FSha256
	{
		uint32 H[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
		uint8 Block[64] = {};
		uint32 BlockLen = 0;
		uint64 TotalBytes = 0;

		static uint32 Rotr(uint32 X, uint32 N) { return (X >> N) | (X << (32 - N)); }

		void Compress(const uint8* P)
		{
			static constexpr uint32 K[64] = {
				0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
				0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
				0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
				0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
				0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
				0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
				0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
				0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u };

			uint32 W[64];
			for (int32 t = 0; t < 16; ++t)
			{
				W[t] = (uint32(P[4 * t]) << 24) | (uint32(P[4 * t + 1]) << 16) | (uint32(P[4 * t + 2]) << 8) | uint32(P[4 * t + 3]);
			}
			for (int32 t = 16; t < 64; ++t)
			{
				const uint32 S0 = Rotr(W[t - 15], 7) ^ Rotr(W[t - 15], 18) ^ (W[t - 15] >> 3);
				const uint32 S1 = Rotr(W[t - 2], 17) ^ Rotr(W[t - 2], 19) ^ (W[t - 2] >> 10);
				W[t] = W[t - 16] + S0 + W[t - 7] + S1;
			}
			uint32 A = H[0], B = H[1], C = H[2], D = H[3], E = H[4], F = H[5], G = H[6], Hh = H[7];
			for (int32 t = 0; t < 64; ++t)
			{
				const uint32 T1 = Hh + (Rotr(E, 6) ^ Rotr(E, 11) ^ Rotr(E, 25)) + ((E & F) ^ (~E & G)) + K[t] + W[t];
				const uint32 T2 = (Rotr(A, 2) ^ Rotr(A, 13) ^ Rotr(A, 22)) + ((A & B) ^ (A & C) ^ (B & C));
				Hh = G; G = F; F = E; E = D + T1; D = C; C = B; B = A; A = T1 + T2;
			}
			H[0] += A; H[1] += B; H[2] += C; H[3] += D; H[4] += E; H[5] += F; H[6] += G; H[7] += Hh;
		}

		void Update(const uint8* Data, int64 Len)
		{
			TotalBytes += static_cast<uint64>(Len);
			if (BlockLen > 0)
			{
				const int64 Take = FMath::Min<int64>(Len, 64 - BlockLen);
				FMemory::Memcpy(Block + BlockLen, Data, Take);
				BlockLen += static_cast<uint32>(Take);
				Data += Take;
				Len -= Take;
				if (BlockLen < 64)
				{
					return;
				}
				Compress(Block);
				BlockLen = 0;
			}
			while (Len >= 64)
			{
				Compress(Data);
				Data += 64;
				Len -= 64;
			}
			if (Len > 0)
			{
				FMemory::Memcpy(Block, Data, Len);
				BlockLen = static_cast<uint32>(Len);
			}
		}

		FString FinalHex()
		{
			const uint64 BitLen = TotalBytes * 8;
			Block[BlockLen++] = 0x80;
			if (BlockLen > 56)
			{
				FMemory::Memzero(Block + BlockLen, 64 - BlockLen);
				Compress(Block);
				BlockLen = 0;
			}
			FMemory::Memzero(Block + BlockLen, 56 - BlockLen);
			for (int32 i = 0; i < 8; ++i)
			{
				Block[56 + i] = static_cast<uint8>(BitLen >> (56 - 8 * i));
			}
			Compress(Block);
			BlockLen = 0;

			static const TCHAR* const Digits = TEXT("0123456789abcdef");
			FString Out;
			Out.Reserve(64);
			for (int32 i = 0; i < 8; ++i)
			{
				for (int32 Shift = 28; Shift >= 0; Shift -= 4)
				{
					Out.AppendChar(Digits[(H[i] >> Shift) & 0xF]);
				}
			}
			return Out;
		}
	};

	static constexpr int64 BlockSize = 4 * 1024 * 1024;
}

FString ComputeSha256Hex(TConstArrayView<uint8> Data)
{
	PipelineFileHashPrivate::FSha256 Sha;
	Sha.Update(Data.GetData(), Data.Num());
	return Sha.FinalHex();
}

bool ComputeFileSha256(const FString& AbsolutePath, FString& OutHex, FString& OutError, const FThreadSafeBool* bCancel)
{
	using namespace PipelineFileHashPrivate;
	OutHex.Reset();
	OutError.Reset();

	if (bCancel != nullptr && *bCancel)
	{
		OutError = TEXT("cancelled");
		return false;
	}
	if (AbsolutePath.IsEmpty())
	{
		OutError = TEXT("no file path was given");
		return false;
	}
	IFileManager& FileManager = IFileManager::Get();
	if (FileManager.DirectoryExists(*AbsolutePath))
	{
		OutError = FString::Printf(TEXT("'%s' is a directory, not a file"), *AbsolutePath);
		return false;
	}
	TUniquePtr<FArchive> Reader(FileManager.CreateFileReader(*AbsolutePath, FILEREAD_Silent));
	if (!Reader.IsValid())
	{
		OutError = FString::Printf(TEXT("can't open '%s' for reading (missing, locked or not a file)"), *AbsolutePath);
		return false;
	}

	const int64 Total = Reader->TotalSize();
	if (Total < 0)
	{
		OutError = FString::Printf(TEXT("can't get the size of '%s'"), *AbsolutePath);
		return false;
	}
	TArray<uint8> Buffer;
	Buffer.SetNumUninitialized(static_cast<int32>(FMath::Min<int64>(BlockSize, FMath::Max<int64>(Total, 1))));

	FSha256 Sha;
	int64 Done = 0;
	while (Done < Total)
	{
		if (bCancel != nullptr && *bCancel)
		{
			OutError = TEXT("cancelled");
			return false;
		}
		const int64 Chunk = FMath::Min<int64>(Buffer.Num(), Total - Done);
		Reader->Serialize(Buffer.GetData(), Chunk);
		if (Reader->IsError())
		{
			OutError = FString::Printf(TEXT("read error in '%s' at byte %lld"), *AbsolutePath, Done);
			return false;
		}
		Sha.Update(Buffer.GetData(), Chunk);
		Done += Chunk;
	}
	if (!Reader->Close())
	{
		OutError = FString::Printf(TEXT("read error in '%s'"), *AbsolutePath);
		return false;
	}
	OutHex = Sha.FinalHex();
	return true;
}
