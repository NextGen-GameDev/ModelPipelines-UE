// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
//
// WP9B5.3b visible tests: the Phonon branch of ValidatePipelineModelPackage (Pipelines.Editor.Validator.Phonon).
// Uses UPipelineEditorTestFileModel with Format = Phonon as the pointer asset (the real UPipelinePhononModel is in the paid plugin).
// Contract: folder missing / any of weights.bin, weights.json, tokens.txt missing -> GgufFileMissing (68) "Phonon weights folder <path>
// is missing <file>"; weights.bin size != asset -> 69; SHA != asset -> 70 (full validation only); Content/Models not staged NonUFS ->
// 71 Warning; no NNE steps. Smoke: no backend -> Info "no Phonon backend ..."; else one CPU DeviceCheck that ran.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Array.h"
#include "Containers/UnrealString.h"
#include "Engine/EngineTypes.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMisc.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Settings/ProjectPackagingSettings.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "PipelineModelPackage.h"
#include "PipelineModelSource.h"
#include "PipelineValidation.h"
#include "PipelineModelValidator.h"
#include "ModelPipelinesCoreModule.h"
#include "PipelineEditorTestFileModel.h"

namespace PipelinePhononValidatorSpecPrivate
{
	using ECode = EPipelineValidationCode;
	using ESev = EPipelineValidationSeverity;

	static const TCHAR* AbcSha = TEXT("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
	static const TCHAR* RealSha = TEXT("7dbcfd8ce258972cc5604f141ac7d64bbf5251c49bbc79c3e0c09b536e23fea5");

	static FString TempDir()
	{
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir() / TEXT("Temp/WP9B53/PhononValidator"));
	}
	static FString DevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		return Env.IsEmpty() ? FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev")) : Env;
	}
	static FString RealFolder() { return DevDir() / TEXT("build/m9/p2c/wk/c_weights"); }

	struct FPhononCase
	{
		TStrongObjectPtr<UPipelineModelPackage> Package;
		TStrongObjectPtr<UPipelineEditorTestFileModel> File;
	};

	static FPhononCase MakeCase(const FString& Folder, int64 Size, const FString& Sha)
	{
		FPhononCase C;
		C.File.Reset(NewObject<UPipelineEditorTestFileModel>(GetTransientPackage()));
		C.File->Format = EPipelineModelFormat::Phonon;
		C.File->FilePath = Folder;
		C.File->SizeBytes = Size;
		C.File->Sha256 = Sha;
		C.Package.Reset(NewObject<UPipelineModelPackage>(GetTransientPackage()));
		C.Package->Task = EPipelineTask::SpeechToText;
		C.Package->GgufModel = C.File.Get();
		C.Package->License.SpdxId = TEXT("CC-BY-4.0");
		C.Package->License.SourceUrl = TEXT("https://example.com/model");
		C.Package->License.Attribution = TEXT("Test");
		return C;
	}

	static bool HasCode(const FPipelineValidationReport& R, ECode Code, ESev Sev)
	{
		for (const FPipelineValidationMessage& M : R.Messages) { if (M.Code == Code && M.Severity == Sev) { return true; } }
		return false;
	}
	static FString MessageOf(const FPipelineValidationReport& R, ECode Code)
	{
		for (const FPipelineValidationMessage& M : R.Messages) { if (M.Code == Code) { return M.Message; } }
		return FString();
	}
	/** Codes the NNE-only steps can raise (tokenizer, runtime, creation, smoke run, task I/O, labels). */
	static bool IsNneCode(ECode Code)
	{
		const int32 V = static_cast<int32>(Code);
		return (V >= 20 && V <= 31) || (V >= 60 && V <= 64);
	}

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
		if (bPresent) { FDirectoryPath P; P.Path = PathValue; Dirs.Add(P); }
	}

	static FPipelineValidateOptions NoSmoke()
	{
		FPipelineValidateOptions O;
		O.bRunSmokeTest = false;
		O.bCheckGpu = false;
		return O;
	}
}

BEGIN_DEFINE_SPEC(FPipelinePhononValidatorSpec, "Pipelines.Editor.Validator.Phonon",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
	/** Writes <TempDir>/<Folder>/{weights.bin="abc", weights.json="{}", tokens.txt="a"} except the names in Skip; returns the folder. */
	FString WriteFolder(const FString& Folder, const TArray<FString>& Skip = TArray<FString>());
	void WriteText(const FString& Path, const FString& Text);
END_DEFINE_SPEC(FPipelinePhononValidatorSpec)

void FPipelinePhononValidatorSpec::WriteText(const FString& Path, const FString& Text)
{
	if (!FFileHelper::SaveStringToFile(Text, *Path, FFileHelper::EEncodingOptions::ForceAnsi))
	{
		AddError(FString::Printf(TEXT("could not write temp file '%s'"), *Path));
	}
}

FString FPipelinePhononValidatorSpec::WriteFolder(const FString& Folder, const TArray<FString>& Skip)
{
	const FString Dir = PipelinePhononValidatorSpecPrivate::TempDir() / Folder;
	IFileManager::Get().MakeDirectory(*Dir, true);
	if (!Skip.Contains(TEXT("weights.bin"))) { WriteText(Dir / TEXT("weights.bin"), TEXT("abc")); }
	if (!Skip.Contains(TEXT("weights.json"))) { WriteText(Dir / TEXT("weights.json"), TEXT("{}")); }
	if (!Skip.Contains(TEXT("tokens.txt"))) { WriteText(Dir / TEXT("tokens.txt"), TEXT("a")); }
	return Dir;
}

void FPipelinePhononValidatorSpec::Define()
{
	using namespace PipelinePhononValidatorSpecPrivate;

	BeforeEach([this]()
	{
		IFileManager::Get().DeleteDirectory(*TempDir(), false, true);
		IFileManager::Get().MakeDirectory(*TempDir(), true);
	});
	AfterEach([this]()
	{
		IFileManager::Get().DeleteDirectory(*TempDir(), false, true);
	});

	Describe("folder files", [this]()
	{
		It("a complete folder with the right size and SHA: no 68-71 errors, no NNE codes, no NoModel, zero errors (smoke off)", [this]()
		{
			// Catches a validator that sends Phonon through the NNE steps (NoModel / 'Re-import the ONNX model' / tokenizer codes).
			const FString Dir = WriteFolder(TEXT("ok"));
			const FPhononCase C = MakeCase(Dir, 3, AbcSha);
			FStagingGuard Guard; SetModelsStaging(true);
			const FPipelineValidationReport R = ValidatePipelineModelPackage(C.Package.Get(), NoSmoke());
			for (const FPipelineValidationMessage& M : R.Messages)
			{
				TestFalse(*FString::Printf(TEXT("unexpected file code in a valid folder: %s"), *R.ToString()),
					M.Code == ECode::GgufFileMissing || M.Code == ECode::GgufSizeMismatch || M.Code == ECode::GgufShaMismatch);
				TestFalse(*FString::Printf(TEXT("NoModel in a Phonon package: %s"), *R.ToString()), M.Code == ECode::NoModel);
				TestFalse(*FString::Printf(TEXT("NNE-only code %s must not appear for Phonon: %s"), LexToString(M.Code), *R.ToString()), IsNneCode(M.Code));
				TestFalse(TEXT("'Re-import the ONNX model' must not appear for Phonon"), M.Message.Contains(TEXT("Re-import the ONNX model")));
			}
			TestEqual(*FString::Printf(TEXT("NumErrors; %s"), *R.ToString()), R.NumErrors(), 0);
		});

		It("folder missing -> 68 Error naming the folder path; no 69/70", [this]()
		{
			const FString Dir = TempDir() / TEXT("not_there");
			const FPhononCase C = MakeCase(Dir, 3, AbcSha);
			const FPipelineValidationReport R = ValidatePipelineModelPackage(C.Package.Get(), NoSmoke());
			TestTrue(*FString::Printf(TEXT("68 Error expected; %s"), *R.ToString()), HasCode(R, ECode::GgufFileMissing, ESev::Error));
			const FString Msg = MessageOf(R, ECode::GgufFileMissing);
			TestTrue(*FString::Printf(TEXT("68 message names the folder (%s); got '%s'"), *Dir, *Msg), Msg.Contains(Dir) || Msg.Contains(TEXT("not_there")));
			TestFalse(*FString::Printf(TEXT("no 69; %s"), *R.ToString()), R.HasCode(ECode::GgufSizeMismatch));
			TestFalse(*FString::Printf(TEXT("no 70; %s"), *R.ToString()), R.HasCode(ECode::GgufShaMismatch));
			TestFalse(TEXT("report invalid"), R.IsValid());
		});

		for (const TCHAR* Missing : { TEXT("weights.bin"), TEXT("weights.json"), TEXT("tokens.txt") })
		{
			const FString MissingName(Missing);
			It(*FString::Printf(TEXT("%s absent -> 68 Error whose message names %s"), Missing, Missing), [this, MissingName]()
			{
				// Catches a validator that only checks weights.bin (the file the pointer asset describes).
				TArray<FString> Skip; Skip.Add(MissingName);
				const FString Dir = WriteFolder(TEXT("partial"), Skip);
				const FPhononCase C = MakeCase(Dir, 3, AbcSha);
				const FPipelineValidationReport R = ValidatePipelineModelPackage(C.Package.Get(), NoSmoke());
				TestTrue(*FString::Printf(TEXT("68 Error expected; %s"), *R.ToString()), HasCode(R, ECode::GgufFileMissing, ESev::Error));
				bool bNames = false;
				for (const FPipelineValidationMessage& M : R.Messages)
				{
					if (M.Code == ECode::GgufFileMissing && M.Message.Contains(MissingName)) { bNames = true; }
				}
				TestTrue(*FString::Printf(TEXT("a 68 message names '%s'; %s"), *MissingName, *R.ToString()), bNames);
			});
		}

		It("weights.bin size mismatch -> 69 Error with both numbers; SHA not computed (no 70)", [this]()
		{
			const FString Dir = WriteFolder(TEXT("size"));
			const FPhononCase C = MakeCase(Dir, 4, AbcSha);   // disk 3, asset 4
			const FPipelineValidationReport R = ValidatePipelineModelPackage(C.Package.Get(), NoSmoke());
			TestTrue(*FString::Printf(TEXT("69 Error expected; %s"), *R.ToString()), HasCode(R, ECode::GgufSizeMismatch, ESev::Error));
			const FString Msg = MessageOf(R, ECode::GgufSizeMismatch);
			TestTrue(*FString::Printf(TEXT("69 message holds 3 and 4; got '%s'"), *Msg), Msg.Contains(TEXT("3")) && Msg.Contains(TEXT("4")));
			TestFalse(*FString::Printf(TEXT("no 70 when size differs; %s"), *R.ToString()), R.HasCode(ECode::GgufShaMismatch));
		});

		It("wrong SHA with a full validation (default options) -> 70 Error holding both values; no 68/69", [this]()
		{
			const FString Dir = WriteFolder(TEXT("sha"));
			const FString Wrong = FString::ChrN(64, TEXT('a'));
			const FPhononCase C = MakeCase(Dir, 3, Wrong);
			// Default options = full validation (smoke on). The smoke itself may fail on the fake weights; only the file codes matter.
			const FPipelineValidationReport R = ValidatePipelineModelPackage(C.Package.Get());
			TestTrue(*FString::Printf(TEXT("70 Error expected; %s"), *R.ToString()), HasCode(R, ECode::GgufShaMismatch, ESev::Error));
			const FString Msg = MessageOf(R, ECode::GgufShaMismatch).ToLower();
			TestTrue(*FString::Printf(TEXT("70 holds the declared value; got '%s'"), *Msg), Msg.Contains(Wrong));
			TestTrue(*FString::Printf(TEXT("70 holds the computed value; got '%s'"), *Msg), Msg.Contains(FString(AbcSha)));
			TestFalse(*FString::Printf(TEXT("no 68/69; %s"), *R.ToString()), R.HasCode(ECode::GgufFileMissing) || R.HasCode(ECode::GgufSizeMismatch));
		});

		It("does not modify the package or the file model", [this]()
		{
			const FString Dir = WriteFolder(TEXT("ro"));
			const FPhononCase C = MakeCase(Dir, 3, AbcSha);
			ValidatePipelineModelPackage(C.Package.Get(), NoSmoke());
			TestEqual(TEXT("Task unchanged"), C.Package->Task, EPipelineTask::SpeechToText);
			TestTrue(TEXT("GgufModel unchanged"), C.Package->GgufModel == C.File.Get());
			TestNull(TEXT("Model still null"), C.Package->Model.Get());
			TestEqual(TEXT("size unchanged"), C.File->SizeBytes, static_cast<int64>(3));
		});
	});

	Describe("ModelsNotStagedNonUfs (71)", [this]()
	{
		It("missing 'Models' staging entry -> 71 Warning naming DirectoriesToAlwaysStageAsNonUFS (only a Warning); present -> no 71", [this]()
		{
			FStagingGuard Guard;
			const FString Dir = WriteFolder(TEXT("stage"));
			const FPhononCase C = MakeCase(Dir, 3, AbcSha);
			SetModelsStaging(false);
			FPipelineValidationReport R = ValidatePipelineModelPackage(C.Package.Get(), NoSmoke());
			TestTrue(*FString::Printf(TEXT("71 Warning expected; %s"), *R.ToString()), HasCode(R, ECode::ModelsNotStagedNonUfs, ESev::Warning));
			TestTrue(TEXT("71 names the setting"), MessageOf(R, ECode::ModelsNotStagedNonUfs).Contains(TEXT("DirectoriesToAlwaysStageAsNonUFS")));
			TestEqual(*FString::Printf(TEXT("71 alone is not an error; %s"), *R.ToString()), R.NumErrors(), 0);
			SetModelsStaging(true, TEXT("Models"));
			R = ValidatePipelineModelPackage(C.Package.Get(), NoSmoke());
			TestFalse(*FString::Printf(TEXT("no 71 with Models; %s"), *R.ToString()), R.HasCode(ECode::ModelsNotStagedNonUfs));
			SetModelsStaging(true, TEXT("/Game/Models"));
			R = ValidatePipelineModelPackage(C.Package.Get(), NoSmoke());
			TestFalse(*FString::Printf(TEXT("no 71 with /Game/Models; %s"), *R.ToString()), R.HasCode(ECode::ModelsNotStagedNonUfs));
		});

		It("71 is also raised when the folder is missing (packaging, not file, problem)", [this]()
		{
			FStagingGuard Guard;
			SetModelsStaging(false);
			const FPhononCase C = MakeCase(TempDir() / TEXT("gone"), 3, AbcSha);
			const FPipelineValidationReport R = ValidatePipelineModelPackage(C.Package.Get(), NoSmoke());
			TestTrue(*FString::Printf(TEXT("68 expected; %s"), *R.ToString()), R.HasCode(ECode::GgufFileMissing));
			TestTrue(*FString::Printf(TEXT("71 Warning expected; %s"), *R.ToString()), HasCode(R, ECode::ModelsNotStagedNonUfs, ESev::Warning));
		});
	});

	Describe("smoke", [this]()
	{
		It("no Phonon backend registered -> Info 'no Phonon backend' and no error; a backend -> exactly one CPU DeviceCheck that ran", [this]()
		{
			// Both branches are valid depending on the loaded plugins (the paid plugin is loaded in the test project, so the
			// second one is expected). The test fails if neither contract holds, e.g. a smoke that silently reports nothing.
			const FString Dir = RealFolder();
			const int64 Size = IFileManager::Get().FileSize(*(Dir / TEXT("weights.bin")));
			if (Size <= 0)
			{
				AddError(FString::Printf(TEXT("Phonon weights folder '%s' missing; see D:/Game Dev/Pipelines-UE/.pipelines-dev/build/m9/p2c (Tools/fetch_phonon2_p2c.py)"), *Dir));
				return;
			}
			FStagingGuard Guard; SetModelsStaging(true);
			const FPhononCase C = MakeCase(Dir, Size, RealSha);
			FPipelineValidateOptions O;   // smoke on
			O.bCheckGpu = false;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(C.Package.Get(), O);
			const bool bBackend = FModelPipelinesCoreModule::Get().FindBackend(EPipelineModelFormat::Phonon) != nullptr;
			if (!bBackend)
			{
				bool bInfo = false;
				for (const FPipelineValidationMessage& M : R.Messages)
				{
					if (M.Severity == ESev::Info && M.Message.Contains(TEXT("no Phonon backend"))) { bInfo = true; }
				}
				TestTrue(*FString::Printf(TEXT("Info 'no Phonon backend' expected without a backend; %s"), *R.ToString()), bInfo);
				TestEqual(*FString::Printf(TEXT("NumErrors; %s"), *R.ToString()), R.NumErrors(), 0);
				AddInfo(TEXT("no Phonon backend registered: only the no-backend path was exercised"));
				return;
			}
			TestEqual(*FString::Printf(TEXT("NumErrors with a backend; %s"), *R.ToString()), R.NumErrors(), 0);
			if (TestEqual(*FString::Printf(TEXT("one DeviceCheck; %s"), *R.ToString()), R.DeviceChecks.Num(), 1))
			{
				const FPipelineDeviceCheck& D = R.DeviceChecks[0];
				TestEqual(TEXT("device is CPU"), static_cast<int32>(D.Device), static_cast<int32>(EPipelineDevice::CPU));
				TestTrue(*FString::Printf(TEXT("created; %s"), *D.Message), D.bCreated);
				TestTrue(*FString::Printf(TEXT("smoke ran; %s"), *D.Message), D.bRan);
			}
			for (const FPipelineValidationMessage& M : R.Messages)
			{
				TestFalse(*FString::Printf(TEXT("NNE-only code %s for Phonon; %s"), LexToString(M.Code), *R.ToString()), M.Code == ECode::NoModel || M.Code == ECode::InvalidGgufModel);
			}
		});

		It("a backend whose model cannot load (fake 3-byte weights) -> Error whose message starts 'Phonon: ' (or no-backend Info), never a crash", [this]()
		{
			const FString Dir = WriteFolder(TEXT("fake"));
			const FPhononCase C = MakeCase(Dir, 3, AbcSha);
			FPipelineValidateOptions O; O.bCheckGpu = false;
			const FPipelineValidationReport R = ValidatePipelineModelPackage(C.Package.Get(), O);
			if (FModelPipelinesCoreModule::Get().FindBackend(EPipelineModelFormat::Phonon) == nullptr)
			{
				AddInfo(TEXT("no Phonon backend registered: failure path not exercised"));
				return;
			}
			bool bPrefixed = false;
			for (const FPipelineValidationMessage& M : R.Messages)
			{
				if (M.Severity == ESev::Error && M.Message.StartsWith(TEXT("Phonon: "))) { bPrefixed = true; }
			}
			TestTrue(*FString::Printf(TEXT("an Error message prefixed 'Phonon: ' expected; %s"), *R.ToString()), bPrefixed);
			TestFalse(TEXT("report invalid"), R.IsValid());
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
