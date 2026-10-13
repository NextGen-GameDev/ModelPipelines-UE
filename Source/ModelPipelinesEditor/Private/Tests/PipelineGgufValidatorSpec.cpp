// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "Engine/EngineTypes.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Settings/ProjectPackagingSettings.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "NNEModelData.h"
#include "PipelineModelPackage.h"
#include "PipelineModelSource.h"
#include "PipelineValidation.h"
#include "PipelineModelValidator.h"
#include "PipelineEditorTestFileModel.h"

namespace PipelineGgufValidatorSpecPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;

	static const TCHAR* AbcSha = TEXT("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

	static FString TempDir()
	{
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Temp/WP8L6a/GgufValidator"));
	}

	static FString DevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		return Env.IsEmpty() ? FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev")) : Env;
	}

	/** A TextGeneration package pointing at a file model; the strong pointers keep both UObjects alive. */
	struct FGgufCase
	{
		TStrongObjectPtr<UPipelineModelPackage> Package;
		TStrongObjectPtr<UPipelineEditorTestFileModel> File;
	};

	static FGgufCase MakeCase(const FString& Path, int64 Size, const FString& Sha, EPipelineTask Task = EPipelineTask::TextGeneration)
	{
		FGgufCase C;
		C.File.Reset(NewObject<UPipelineEditorTestFileModel>(GetTransientPackage()));
		C.File->FilePath = Path;
		C.File->SizeBytes = Size;
		C.File->Sha256 = Sha;
		C.Package.Reset(NewObject<UPipelineModelPackage>(GetTransientPackage()));
		C.Package->Task = Task;
		C.Package->GgufModel = C.File.Get();
		C.Package->License.SpdxId = TEXT("Apache-2.0");
		C.Package->License.SourceUrl = TEXT("https://example.com/model");
		C.Package->License.Attribution = TEXT("Test");
		return C;
	}

	static bool HasCode(const FPipelineValidationReport& R, ECode Code, ESev Sev)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code && M.Severity == Sev) { return true; }
		}
		return false;
	}

	static FString MessageOf(const FPipelineValidationReport& R, ECode Code)
	{
		for (const FPipelineValidationMessage& M : R.Messages)
		{
			if (M.Code == Code) { return M.Message; }
		}
		return FString();
	}

	static int32 CountCode(const FPipelineValidationReport& R, ECode Code)
	{
		int32 N = 0;
		for (const FPipelineValidationMessage& M : R.Messages) { if (M.Code == Code) { ++N; } }
		return N;
	}

	/** Codes the NNE-only steps can raise (tokenizer, runtime, creation, smoke run, task I/O, labels). */
	static bool IsNneCode(ECode Code)
	{
		const int32 V = static_cast<int32>(Code);
		return (V >= 20 && V <= 31) || (V >= 60 && V <= 64);
	}

	/** Restores DirectoriesToAlwaysStageAsNonUFS on scope exit, also when a test fails or returns early. */
	struct FStagingGuard
	{
		TArray<FDirectoryPath> Saved;
		FStagingGuard() { Saved = GetMutableDefault<UProjectPackagingSettings>()->DirectoriesToAlwaysStageAsNonUFS; }
		~FStagingGuard() { GetMutableDefault<UProjectPackagingSettings>()->DirectoriesToAlwaysStageAsNonUFS = Saved; }
	};

	static void SetModelsStaging(bool bPresent, const TCHAR* PathValue = TEXT("Models"))
	{
		TArray<FDirectoryPath>& Dirs = GetMutableDefault<UProjectPackagingSettings>()->DirectoriesToAlwaysStageAsNonUFS;
		Dirs.RemoveAll([](const FDirectoryPath& D)
		{
			return D.Path.Equals(TEXT("Models"), ESearchCase::IgnoreCase) || D.Path.Equals(TEXT("/Game/Models"), ESearchCase::IgnoreCase);
		});
		if (bPresent)
		{
			FDirectoryPath P;
			P.Path = PathValue;
			Dirs.Add(P);
		}
	}
}

BEGIN_DEFINE_SPEC(FPipelineGgufValidatorSpec, "Pipelines.Editor.GgufValidator",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	/** Writes "abc" (3 bytes, SHA ba7816bf...) to <TempDir>/Name and returns the absolute path. */
	FString WriteAbc(const FString& Name);
	FPipelineValidationReport Validate(const PipelineGgufValidatorSpecPrivate::FGgufCase& C);
	/** Fails (with the fetch hint) and returns false when the real 1.5B model is absent. */
	bool Real15(FString& OutPath, int64& OutSize);
END_DEFINE_SPEC(FPipelineGgufValidatorSpec)

FString FPipelineGgufValidatorSpec::WriteAbc(const FString& Name)
{
	const FString Path = PipelineGgufValidatorSpecPrivate::TempDir() / Name;
	const FString Text = TEXT("abc");
	FTCHARToUTF8 Conv(*Text);
	TArray<uint8> Bytes;
	Bytes.Append(reinterpret_cast<const uint8*>(Conv.Get()), Conv.Length());
	if (!FFileHelper::SaveArrayToFile(Bytes, *Path))
	{
		AddError(FString::Printf(TEXT("could not write temp file '%s'"), *Path));
	}
	return Path;
}

FPipelineValidationReport FPipelineGgufValidatorSpec::Validate(const PipelineGgufValidatorSpecPrivate::FGgufCase& C)
{
	return ValidatePipelineModelPackage(C.Package.Get());   // default options: smoke test and GPU on, to prove they are skipped
}

bool FPipelineGgufValidatorSpec::Real15(FString& OutPath, int64& OutSize)
{
	OutPath = PipelineGgufValidatorSpecPrivate::DevDir() / TEXT("models/llm/qwen2.5-1.5b-instruct-q8_0/qwen2.5-1.5b-instruct-q8_0.gguf");
	OutSize = IFileManager::Get().FileSize(*OutPath);
	if (OutSize <= 0)
	{
		AddError(FString::Printf(TEXT("GGUF test model '%s' missing; it is the WP8L.1 download under .pipelines-dev/models/llm/ (see that folder's README.md)"), *OutPath));
		return false;
	}
	return true;
}

void FPipelineGgufValidatorSpec::Define()
{
	using namespace PipelineGgufValidatorSpecPrivate;

	BeforeEach([this]()
	{
		IFileManager::Get().DeleteDirectory(*TempDir(), false, true);
		IFileManager::Get().MakeDirectory(*TempDir(), true);
	});
	AfterEach([this]()
	{
		IFileManager::Get().DeleteDirectory(*TempDir(), false, true);
	});

	Describe("valid file", [this]()
	{
		It("no 68-70, no NoModel, no NNE messages, no device checks, no errors at all (false 'Re-import the ONNX model' was the bug)", [this]()
		{
			const FString Path = WriteAbc(TEXT("ok.gguf"));
			const FGgufCase C = MakeCase(Path, 3, AbcSha);
			const FPipelineValidationReport R = Validate(C);
			for (const FPipelineValidationMessage& M : R.Messages)
			{
				TestFalse(*FString::Printf(TEXT("unexpected GGUF-file code in a valid package: %s"), *R.ToString()),
					M.Code == ECode::GgufFileMissing || M.Code == ECode::GgufSizeMismatch || M.Code == ECode::GgufShaMismatch);
				TestFalse(*FString::Printf(TEXT("NoModel in a valid GGUF package: %s"), *R.ToString()), M.Code == ECode::NoModel);
				TestFalse(*FString::Printf(TEXT("NNE-only code %s must not appear for GGUF: %s"), LexToString(M.Code), *R.ToString()), IsNneCode(M.Code));
				TestFalse(TEXT("the message 'Re-import the ONNX model' must not appear for GGUF"), M.Message.Contains(TEXT("Re-import the ONNX model")));
			}
			TestEqual(*FString::Printf(TEXT("DeviceChecks (NNE steps skipped); report: %s"), *R.ToString()), R.DeviceChecks.Num(), 0);
			TestEqual(*FString::Printf(TEXT("NumErrors; report: %s"), *R.ToString()), R.NumErrors(), 0);
		});

		It("the real 0.5B Q8_0 model with its real size and SHA validates without 68-70", [this]()
		{
			const FString Path = DevDir() / TEXT("models/llm/qwen2.5-0.5b-instruct-q8_0/qwen2.5-0.5b-instruct-q8_0.gguf");
			const int64 Size = IFileManager::Get().FileSize(*Path);
			if (Size <= 0)
			{
				AddError(FString::Printf(TEXT("GGUF test model '%s' missing; it is the WP8L.1 download under .pipelines-dev/models/llm/"), *Path));
				return;
			}
			const FGgufCase C = MakeCase(Path, Size, TEXT("ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"));
			const FPipelineValidationReport R = Validate(C);
			TestFalse(*FString::Printf(TEXT("68; %s"), *R.ToString()), R.HasCode(ECode::GgufFileMissing));
			TestFalse(*FString::Printf(TEXT("69; %s"), *R.ToString()), R.HasCode(ECode::GgufSizeMismatch));
			TestFalse(*FString::Printf(TEXT("70; %s"), *R.ToString()), R.HasCode(ECode::GgufShaMismatch));
		});

		It("does not modify the package or the file model", [this]()
		{
			const FString Path = WriteAbc(TEXT("ro.gguf"));
			const FGgufCase C = MakeCase(Path, 3, AbcSha);
			Validate(C);
			TestEqual(TEXT("Task unchanged"), C.Package->Task, EPipelineTask::TextGeneration);
			TestTrue(TEXT("GgufModel unchanged"), C.Package->GgufModel == C.File.Get());
			TestNull(TEXT("Model still null"), C.Package->Model.Get());
			TestEqual(TEXT("file model path unchanged"), C.File->FilePath, Path);
			TestEqual(TEXT("file model size unchanged"), C.File->SizeBytes, static_cast<int64>(3));
		});
	});

	Describe("file problems", [this]()
	{
		It("missing file -> 68 Error naming the path; 69 and 70 are not also reported", [this]()
		{
			const FString Path = TempDir() / TEXT("not_there.gguf");
			const FGgufCase C = MakeCase(Path, 3, AbcSha);
			const FPipelineValidationReport R = Validate(C);
			TestTrue(*FString::Printf(TEXT("68 Error expected; %s"), *R.ToString()), HasCode(R, ECode::GgufFileMissing, ESev::Error));
			TestTrue(*FString::Printf(TEXT("68 message names the path '%s'; got '%s'"), *Path, *MessageOf(R, ECode::GgufFileMissing)),
				MessageOf(R, ECode::GgufFileMissing).Contains(Path) || MessageOf(R, ECode::GgufFileMissing).Contains(FPaths::GetCleanFilename(Path)));
			TestFalse(*FString::Printf(TEXT("no 69 for a missing file; %s"), *R.ToString()), R.HasCode(ECode::GgufSizeMismatch));
			TestFalse(*FString::Printf(TEXT("no 70 for a missing file; %s"), *R.ToString()), R.HasCode(ECode::GgufShaMismatch));
			TestFalse(TEXT("report is invalid"), R.IsValid());
		});

		It("a directory where the file should be -> 68 Error, no crash", [this]()
		{
			const FString Dir = TempDir() / TEXT("a_dir.gguf");
			IFileManager::Get().MakeDirectory(*Dir, true);
			const FGgufCase C = MakeCase(Dir, 3, AbcSha);
			const FPipelineValidationReport R = Validate(C);
			TestTrue(*FString::Printf(TEXT("68 Error expected; %s"), *R.ToString()), HasCode(R, ECode::GgufFileMissing, ESev::Error));
		});

		It("size + 1 -> 69 Error with both numbers, and the SHA is not computed (no 70)", [this]()
		{
			const FString Path = WriteAbc(TEXT("size.gguf"));
			const FGgufCase C = MakeCase(Path, 4, TEXT("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
			const FPipelineValidationReport R = Validate(C);
			TestTrue(*FString::Printf(TEXT("69 Error expected; %s"), *R.ToString()), HasCode(R, ECode::GgufSizeMismatch, ESev::Error));
			const FString Msg = MessageOf(R, ECode::GgufSizeMismatch);
			// Exact wording of the validator: "... is %lld bytes, the asset expects %lld bytes ..." (3 on disk, 4 declared).
			TestTrue(*FString::Printf(TEXT("69 message says 'is 3 bytes' (file size); got '%s'"), *Msg), Msg.Contains(TEXT("is 3 bytes")));
			TestTrue(*FString::Printf(TEXT("69 message says 'expects 4 bytes' (declared size); got '%s'"), *Msg), Msg.Contains(TEXT("expects 4 bytes")));
			TestFalse(*FString::Printf(TEXT("no 70 when the size already differs; %s"), *R.ToString()), R.HasCode(ECode::GgufShaMismatch));
		});

		It("a 0-byte file with a declared size of 10 -> 69 Error (empty file handled, no crash)", [this]()
		{
			const FString Path = TempDir() / TEXT("zero.gguf");
			FFileHelper::SaveArrayToFile(TArray<uint8>(), *Path);
			const FGgufCase C = MakeCase(Path, 10, FString::ChrN(64, TEXT('a')));
			const FPipelineValidationReport R = Validate(C);
			TestTrue(*FString::Printf(TEXT("69 Error expected; %s"), *R.ToString()), HasCode(R, ECode::GgufSizeMismatch, ESev::Error));
			TestFalse(*FString::Printf(TEXT("no 68 for an existing empty file; %s"), *R.ToString()), R.HasCode(ECode::GgufFileMissing));
		});

		It("size mismatch on the real 1.5B model returns in under 1 s (the 1.6 GB file is not hashed)", [this]()
		{
			FString Path; int64 Size = 0;
			if (!Real15(Path, Size)) { return; }
			const FGgufCase C = MakeCase(Path, Size + 1, FString::ChrN(64, TEXT('a')));
			const double T0 = FPlatformTime::Seconds();
			const FPipelineValidationReport R = Validate(C);
			const double Secs = FPlatformTime::Seconds() - T0;
			TestTrue(*FString::Printf(TEXT("69 Error expected; %s"), *R.ToString()), HasCode(R, ECode::GgufSizeMismatch, ESev::Error));
			TestTrue(*FString::Printf(TEXT("validator took %.3f s, expected < 1 s (SHA must be skipped after a size mismatch)"), Secs), Secs < 1.0);
		});

		It("wrong SHA (64 x 'a') -> 70 Error with the expected and the actual value; no 68/69", [this]()
		{
			const FString Path = WriteAbc(TEXT("sha.gguf"));
			const FString Wrong = FString::ChrN(64, TEXT('a'));
			const FGgufCase C = MakeCase(Path, 3, Wrong);
			const FPipelineValidationReport R = Validate(C);
			TestTrue(*FString::Printf(TEXT("70 Error expected; %s"), *R.ToString()), HasCode(R, ECode::GgufShaMismatch, ESev::Error));
			const FString Msg = MessageOf(R, ECode::GgufShaMismatch).ToLower();
			TestTrue(*FString::Printf(TEXT("70 message holds the declared value; got '%s'"), *Msg), Msg.Contains(Wrong));
			TestTrue(*FString::Printf(TEXT("70 message holds the computed value %s; got '%s'"), AbcSha, *Msg), Msg.Contains(FString(AbcSha)));
			TestFalse(*FString::Printf(TEXT("no 68/69; %s"), *R.ToString()), R.HasCode(ECode::GgufFileMissing) || R.HasCode(ECode::GgufSizeMismatch));
		});

		It("upper-case SHA of the right value -> no 70 (comparison is case-insensitive)", [this]()
		{
			const FString Path = WriteAbc(TEXT("upper.gguf"));
			const FGgufCase C = MakeCase(Path, 3, FString(AbcSha).ToUpper());
			const FPipelineValidationReport R = Validate(C);
			TestFalse(*FString::Printf(TEXT("70 must not be raised; %s"), *R.ToString()), R.HasCode(ECode::GgufShaMismatch));
			TestEqual(*FString::Printf(TEXT("NumErrors; %s"), *R.ToString()), R.NumErrors(), 0);
		});

		It("empty path -> InvalidGgufModel (17) from the settings check; the file branch stops (no 68)", [this]()
		{
			const FGgufCase C = MakeCase(FString(), 3, AbcSha);
			const FPipelineValidationReport R = Validate(C);
			TestTrue(*FString::Printf(TEXT("17 Error expected; %s"), *R.ToString()), HasCode(R, ECode::InvalidGgufModel, ESev::Error));
			TestFalse(*FString::Printf(TEXT("no 68 for an empty path; %s"), *R.ToString()), R.HasCode(ECode::GgufFileMissing));
		});
	});

	Describe("model presence and non-GGUF packages", [this]()
	{
		It("TextGeneration with neither model: exactly one NoModel (from the settings check) and no ONNX wording", [this]()
		{
			TStrongObjectPtr<UPipelineModelPackage> P(NewObject<UPipelineModelPackage>(GetTransientPackage()));
			P->Task = EPipelineTask::TextGeneration;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(P.Get());
			TestEqual(*FString::Printf(TEXT("NoModel count; %s"), *R.ToString()), CountCode(R, ECode::NoModel), 1);
			for (const FPipelineValidationMessage& M : R.Messages)
			{
				TestFalse(*FString::Printf(TEXT("'Re-import the ONNX model' must not appear for TextGeneration; %s"), *R.ToString()), M.Message.Contains(TEXT("Re-import the ONNX model")));
			}
		});

		It("NNE task with neither model keeps the NoModel Error with 'Re-import the ONNX model.'", [this]()
		{
			TStrongObjectPtr<UPipelineModelPackage> P(NewObject<UPipelineModelPackage>(GetTransientPackage()));
			P->Task = EPipelineTask::TextEmbedding;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(P.Get());
			TestTrue(*FString::Printf(TEXT("NoModel Error expected; %s"), *R.ToString()), HasCode(R, ECode::NoModel, ESev::Error));
			bool bWording = false;
			for (const FPipelineValidationMessage& M : R.Messages) { bWording |= M.Message.Contains(TEXT("Re-import the ONNX model.")); }
			TestTrue(*FString::Printf(TEXT("a message says 'Re-import the ONNX model.'; %s"), *R.ToString()), bWording);
		});

		// Catches a validator that, for a TextGeneration package holding an NNE Model and no GgufModel, reports 16 but keeps going
		// into the NNE steps (device checks, tokenizer / runtime / smoke-run codes) instead of returning early.
		It("TextGeneration with an NNE Model and no GgufModel -> WrongModelFormat (16) and an early return (no device checks, no NNE codes)", [this]()
		{
			TStrongObjectPtr<UNNEModelData> Nne(NewObject<UNNEModelData>(GetTransientPackage()));
			TStrongObjectPtr<UPipelineModelPackage> P(NewObject<UPipelineModelPackage>(GetTransientPackage()));
			P->Task = EPipelineTask::TextGeneration;
			P->Model = Nne.Get();
			P->License.SpdxId = TEXT("Apache-2.0");
			P->License.SourceUrl = TEXT("https://example.com/model");
			P->License.Attribution = TEXT("Test");
			const FPipelineValidationReport R = ValidatePipelineModelPackage(P.Get());   // default options: smoke test and GPU on
			TestTrue(*FString::Printf(TEXT("16 Error expected; %s"), *R.ToString()), HasCode(R, ECode::WrongModelFormat, ESev::Error));
			TestEqual(*FString::Printf(TEXT("DeviceChecks empty (NNE steps not run); %s"), *R.ToString()), R.DeviceChecks.Num(), 0);
			for (const FPipelineValidationMessage& M : R.Messages)
			{
				TestFalse(*FString::Printf(TEXT("NNE-only code %s after 16; %s"), LexToString(M.Code), *R.ToString()), IsNneCode(M.Code));
			}
			TestFalse(TEXT("report is invalid"), R.IsValid());
		});

		It("GgufModel that is not an IPipelineFileModel -> InvalidGgufModel (17), no crash, no NNE-only codes", [this]()
		{
			TStrongObjectPtr<UPipelineEditorTestPlainAsset> Plain(NewObject<UPipelineEditorTestPlainAsset>(GetTransientPackage()));
			TStrongObjectPtr<UPipelineModelPackage> P(NewObject<UPipelineModelPackage>(GetTransientPackage()));
			P->Task = EPipelineTask::TextGeneration;
			P->GgufModel = Plain.Get();
			const FPipelineValidationReport R = ValidatePipelineModelPackage(P.Get());
			TestTrue(*FString::Printf(TEXT("17 Error expected; %s"), *R.ToString()), HasCode(R, ECode::InvalidGgufModel, ESev::Error));
			for (const FPipelineValidationMessage& M : R.Messages)
			{
				TestFalse(*FString::Printf(TEXT("NNE-only code %s; %s"), LexToString(M.Code), *R.ToString()), IsNneCode(M.Code));
			}
		});
	});

	Describe("ModelsNotStagedNonUfs (71)", [this]()
	{
		It("missing 'Models' entry -> 71 Warning naming the ini line; present (Models or /Game/Models, any case) -> no 71; setting restored", [this]()
		{
			FStagingGuard Guard;
			const FString Path = WriteAbc(TEXT("stage.gguf"));
			const FGgufCase C = MakeCase(Path, 3, AbcSha);

			SetModelsStaging(false);
			FPipelineValidationReport R = Validate(C);
			TestTrue(*FString::Printf(TEXT("71 Warning expected without the Models entry; %s"), *R.ToString()), HasCode(R, ECode::ModelsNotStagedNonUfs, ESev::Warning));
			const FString Msg = MessageOf(R, ECode::ModelsNotStagedNonUfs);
			TestTrue(*FString::Printf(TEXT("71 names DirectoriesToAlwaysStageAsNonUFS and Models; got '%s'"), *Msg),
				Msg.Contains(TEXT("DirectoriesToAlwaysStageAsNonUFS")) && Msg.Contains(TEXT("Models")));
			TestEqual(*FString::Printf(TEXT("71 is only a Warning, NumErrors; %s"), *R.ToString()), R.NumErrors(), 0);

			SetModelsStaging(true, TEXT("Models"));
			R = Validate(C);
			TestFalse(*FString::Printf(TEXT("no 71 with Path=Models; %s"), *R.ToString()), R.HasCode(ECode::ModelsNotStagedNonUfs));

			SetModelsStaging(true, TEXT("/Game/Models"));
			R = Validate(C);
			TestFalse(*FString::Printf(TEXT("no 71 with Path=/Game/Models; %s"), *R.ToString()), R.HasCode(ECode::ModelsNotStagedNonUfs));

			SetModelsStaging(true, TEXT("models"));
			R = Validate(C);
			TestFalse(*FString::Printf(TEXT("no 71 with Path=models (case-insensitive); %s"), *R.ToString()), R.HasCode(ECode::ModelsNotStagedNonUfs));
		});

		It("an unrelated staging entry does not satisfy 71", [this]()
		{
			FStagingGuard Guard;
			const FString Path = WriteAbc(TEXT("stage2.gguf"));
			const FGgufCase C = MakeCase(Path, 3, AbcSha);
			SetModelsStaging(false);
			FDirectoryPath Other;
			Other.Path = TEXT("Movies");
			GetMutableDefault<UProjectPackagingSettings>()->DirectoriesToAlwaysStageAsNonUFS.Add(Other);
			const FPipelineValidationReport R = Validate(C);
			TestTrue(*FString::Printf(TEXT("71 Warning expected; %s"), *R.ToString()), HasCode(R, ECode::ModelsNotStagedNonUfs, ESev::Warning));
		});

		It("71 is also raised when the file itself is missing (it is about packaging, not the file)", [this]()
		{
			FStagingGuard Guard;
			SetModelsStaging(false);
			const FGgufCase C = MakeCase(TempDir() / TEXT("gone.gguf"), 3, AbcSha);
			const FPipelineValidationReport R = Validate(C);
			TestTrue(*FString::Printf(TEXT("68 expected; %s"), *R.ToString()), R.HasCode(ECode::GgufFileMissing));
			TestTrue(*FString::Printf(TEXT("71 expected alongside 68; %s"), *R.ToString()), HasCode(R, ECode::ModelsNotStagedNonUfs, ESev::Warning));
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
