// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/ParallelFor.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeBool.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "PipelineFileHash.h"

namespace PipelineFileHashSpecPrivate
{
	static FString TempDir()
	{
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Temp/WP8L6a/FileHash"));
	}

	static FString DevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		return Env.IsEmpty() ? FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev")) : Env;
	}

	static TArray<uint8> Ascii(const char* S)
	{
		TArray<uint8> Out;
		for (; *S; ++S) { Out.Add(static_cast<uint8>(*S)); }
		return Out;
	}

	/** Deterministic non-constant bytes: a block mix-up or an off-by-one at a boundary changes the hash. */
	static TArray<uint8> Pattern(int64 Num)
	{
		TArray<uint8> Out;
		Out.SetNumUninitialized(static_cast<int32>(Num));
		for (int64 i = 0; i < Num; ++i) { Out[static_cast<int32>(i)] = static_cast<uint8>((i * 31 + (i >> 12) + 7) & 0xFF); }
		return Out;
	}

	static const TCHAR* Model05 = TEXT("llm/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf");
	static const TCHAR* Model15 = TEXT("llm/qwen2.5-1.5b-instruct-q8_0/qwen2.5-1.5b-instruct-q8_0.gguf");
}

BEGIN_DEFINE_SPEC(FPipelineFileHashSpec, "Pipelines.Editor.FileHash",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	/** Writes Bytes to <TempDir>/Name; returns the absolute path ("" + AddError on failure). */
	FString WriteTemp(const FString& Name, const TArray<uint8>& Bytes);
	void ExpectMem(const TCHAR* What, const TArray<uint8>& Data, const TCHAR* Expected);
	void HashRealModel(const TCHAR* RelPath, const TCHAR* Expected);
END_DEFINE_SPEC(FPipelineFileHashSpec)

FString FPipelineFileHashSpec::WriteTemp(const FString& Name, const TArray<uint8>& Bytes)
{
	const FString Path = PipelineFileHashSpecPrivate::TempDir() / Name;
	if (!FFileHelper::SaveArrayToFile(Bytes, *Path))
	{
		AddError(FString::Printf(TEXT("could not write temp file '%s'"), *Path));
		return FString();
	}
	return Path;
}

void FPipelineFileHashSpec::ExpectMem(const TCHAR* What, const TArray<uint8>& Data, const TCHAR* Expected)
{
	const FString Got = ComputeSha256Hex(TConstArrayView<uint8>(Data));
	TestEqual(*FString::Printf(TEXT("SHA-256 of %s; got %s, expected %s"), What, *Got, Expected), Got, FString(Expected));
}

void FPipelineFileHashSpec::HashRealModel(const TCHAR* RelPath, const TCHAR* Expected)
{
	using namespace PipelineFileHashSpecPrivate;
	const FString Path = DevDir() / TEXT("models") / RelPath;
	const int64 Size = IFileManager::Get().FileSize(*Path);
	if (Size <= 0)
	{
		AddError(FString::Printf(TEXT("GGUF test model '%s' missing; it is the WP8L.1 download under .pipelines-dev/models/llm/ (see that folder's README.md)"), *Path));
		return;
	}
	FString Hex, Err;
	const double T0 = FPlatformTime::Seconds();
	const bool bOk = ComputeFileSha256(Path, Hex, Err);
	const double Secs = FPlatformTime::Seconds() - T0;
	TestTrue(*FString::Printf(TEXT("ComputeFileSha256('%s') returned false, error: %s"), *Path, *Err), bOk);
	TestEqual(*FString::Printf(TEXT("SHA-256 of %s; got %s, expected %s"), RelPath, *Hex, Expected), Hex, FString(Expected));
	AddInfo(FString::Printf(TEXT("hashed %s: %.1f MB in %.2f s = %.0f MB/s"), RelPath, Size / 1.0e6, Secs, Secs > 0.0 ? (Size / 1.0e6) / Secs : 0.0));
}

void FPipelineFileHashSpec::Define()
{
	using namespace PipelineFileHashSpecPrivate;

	BeforeEach([this]()
	{
		IFileManager::Get().DeleteDirectory(*TempDir(), false, true);
		IFileManager::Get().MakeDirectory(*TempDir(), true);
	});
	AfterEach([this]()
	{
		IFileManager::Get().DeleteDirectory(*TempDir(), false, true);
	});

	Describe("ComputeSha256Hex vectors (FIPS 180-4 / NIST examples)", [this]()
	{
		It("empty input -> e3b0c442...b855 (a hash that skipped padding would differ)", [this]()
		{
			ExpectMem(TEXT("\"\""), TArray<uint8>(), TEXT("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
		});
		It("\"abc\" -> ba7816bf...15ad", [this]()
		{
			ExpectMem(TEXT("\"abc\""), Ascii("abc"), TEXT("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
		});
		It("448-bit message (padding spills into a second block) -> 248d6a61...06c1", [this]()
		{
			ExpectMem(TEXT("abcdbcde...nopq"), Ascii("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
				TEXT("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
		});
		It("896-bit message (two full blocks of text) -> cf5b16a7...e9d1", [this]()
		{
			ExpectMem(TEXT("abcdefgh...nopqrstu"),
				Ascii("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
				TEXT("cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"));
		});
		It("1,000,000 x 'a' -> cdc76e5c...2cd0 (a 32-bit length counter or lost carry breaks it)", [this]()
		{
			TArray<uint8> Data;
			Data.Init(static_cast<uint8>('a'), 1000000);
			ExpectMem(TEXT("1,000,000 x 'a'"), Data, TEXT("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
		});
		It("result is 64 lowercase hex characters", [this]()
		{
			const FString H = ComputeSha256Hex(TConstArrayView<uint8>(Ascii("abc")));
			TestEqual(TEXT("length"), H.Len(), 64);
			TestEqual(*FString::Printf(TEXT("lower-case; got %s"), *H), H, H.ToLower());
		});
		It("is safe on 8 threads at once (no shared state)", [this]()
		{
			const TArray<uint8> Abc = Ascii("abc");
			TArray<FString> Out;
			Out.SetNum(8);
			ParallelFor(8, [&Out, &Abc](int32 i) { Out[i] = ComputeSha256Hex(TConstArrayView<uint8>(Abc)); });
			for (int32 i = 0; i < 8; ++i)
			{
				TestEqual(*FString::Printf(TEXT("thread %d; got %s"), i, *Out[i]), Out[i], FString(TEXT("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")));
			}
		});
	});

	Describe("ComputeFileSha256 on files", [this]()
	{
		It("small files match the vectors (empty file -> e3b0c442..., 'abc' -> ba7816bf...)", [this]()
		{
			FString Hex, Err;
			const FString Empty = WriteTemp(TEXT("empty.bin"), TArray<uint8>());
			TestTrue(*FString::Printf(TEXT("empty file hashes (is not an error); error: %s"), *Err), ComputeFileSha256(Empty, Hex, Err));
			TestEqual(*FString::Printf(TEXT("empty file; got %s"), *Hex), Hex, FString(TEXT("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")));
			const FString Abc = WriteTemp(TEXT("abc.bin"), Ascii("abc"));
			TestTrue(*FString::Printf(TEXT("abc file hashes; error: %s"), *Err), ComputeFileSha256(Abc, Hex, Err));
			TestEqual(*FString::Printf(TEXT("abc file; got %s"), *Hex), Hex, FString(TEXT("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")));
		});

		It("4 MiB + 1 byte (one byte spills into a second block) equals the in-memory hash", [this]()
		{
			const TArray<uint8> Data = Pattern(4 * 1024 * 1024 + 1);
			const FString Path = WriteTemp(TEXT("b4m1.bin"), Data);
			FString Hex, Err;
			TestTrue(*FString::Printf(TEXT("hash succeeds; error: %s"), *Err), ComputeFileSha256(Path, Hex, Err));
			const FString Mem = ComputeSha256Hex(TConstArrayView<uint8>(Data));
			TestEqual(*FString::Printf(TEXT("file %s vs memory %s"), *Hex, *Mem), Hex, Mem);
		});

		It("exactly 8 MiB (two full blocks, no short tail) equals the in-memory hash", [this]()
		{
			const TArray<uint8> Data = Pattern(8 * 1024 * 1024);
			const FString Path = WriteTemp(TEXT("b8m.bin"), Data);
			FString Hex, Err;
			TestTrue(*FString::Printf(TEXT("hash succeeds; error: %s"), *Err), ComputeFileSha256(Path, Hex, Err));
			const FString Mem = ComputeSha256Hex(TConstArrayView<uint8>(Data));
			TestEqual(*FString::Printf(TEXT("file %s vs memory %s"), *Hex, *Mem), Hex, Mem);
		});

		It("a missing file returns false with a non-empty error and an empty or untouched-safe hash", [this]()
		{
			FString Hex, Err;
			const FString Path = TempDir() / TEXT("does_not_exist.bin");
			TestFalse(TEXT("missing file must return false"), ComputeFileSha256(Path, Hex, Err));
			TestFalse(TEXT("error text is set"), Err.IsEmpty());
		});

		It("a directory instead of a file returns false with an error (no crash)", [this]()
		{
			FString Hex, Err;
			TestFalse(TEXT("a directory is not a readable file"), ComputeFileSha256(TempDir(), Hex, Err));
			TestFalse(TEXT("error text is set"), Err.IsEmpty());
		});

		It("cancel flag already set -> false and error exactly 'cancelled' (small and 5 MiB file)", [this]()
		{
			const FString Small = WriteTemp(TEXT("c_small.bin"), Ascii("abc"));
			const FString Big = WriteTemp(TEXT("c_big.bin"), Pattern(5 * 1024 * 1024));
			FThreadSafeBool bCancel(true);
			for (const FString& P : { Small, Big })
			{
				FString Hex, Err;
				TestFalse(*FString::Printf(TEXT("'%s' with cancel set must return false"), *P), ComputeFileSha256(P, Hex, Err, &bCancel));
				TestEqual(*FString::Printf(TEXT("error for '%s'"), *P), Err, FString(TEXT("cancelled")));
			}
		});

		It("cancel flag present but false does not disturb the result", [this]()
		{
			const FString Path = WriteTemp(TEXT("nc.bin"), Ascii("abc"));
			FThreadSafeBool bCancel(false);
			FString Hex, Err;
			TestTrue(*FString::Printf(TEXT("hash succeeds; error: %s"), *Err), ComputeFileSha256(Path, Hex, Err, &bCancel));
			TestEqual(*FString::Printf(TEXT("got %s"), *Hex), Hex, FString(TEXT("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")));
		});
	});

	Describe("real GGUF models (WP8L.1 downloads)", [this]()
	{
		It("Qwen2.5-0.5B Q8_0 -> ca59ca7f...844e; reports MB/s", [this]()
		{
			HashRealModel(Model05, TEXT("ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"));
		});
		It("Qwen2.5-1.5B Q8_0 -> d7efb072...63c8; reports MB/s", [this]()
		{
			HashRealModel(Model15, TEXT("d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8"));
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
