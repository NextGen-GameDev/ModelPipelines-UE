// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.
// WP4.3 (docs/contracts.md C26): EPipelineImageResample, NMS settings and InvalidPostprocess validation.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Containers/Array.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/Class.h"
#include "NNEModelData.h"
#include "PipelineTypes.h"
#include "PipelineValidation.h"
#include "PipelineModelPackage.h"
#include <limits>

namespace PipelinePackageM4SpecPrivate
{
	static UPipelineModelPackage* MakeValidPackage(EPipelineTask Task)
	{
		UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
		Pkg->Task = Task;
		Pkg->Model = NewObject<UNNEModelData>(GetTransientPackage());
		Pkg->License.SpdxId = TEXT("Apache-2.0");
		Pkg->License.SourceUrl = TEXT("https://huggingface.co/x/y/tree/main");
		Pkg->License.Attribution = TEXT("Example Authors");
		if (IsTextTask(Task))
		{
			Pkg->TokenizerJson = TEXT("{}");
		}
		if (TaskUsesLabels(Task))
		{
			Pkg->Labels = {TEXT("a"), TEXT("b")};
		}
		return Pkg;
	}

	static const float NaNF = std::numeric_limits<float>::quiet_NaN();
}

BEGIN_DEFINE_SPEC(FPipelinePackageM4Spec, "Pipelines.Core.Package.M4",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	TArray<TStrongObjectPtr<UObject>> Keep;

	UPipelineModelPackage* Make(EPipelineTask Task)
	{
		UPipelineModelPackage* P = PipelinePackageM4SpecPrivate::MakeValidPackage(Task);
		Keep.Add(TStrongObjectPtr<UObject>(P));
		Keep.Add(TStrongObjectPtr<UObject>(P->Model.Get()));
		return P;
	}

	int32 CountPostprocessErrors(const FPipelineValidationReport& R) const
	{
		int32 N = 0;
		for (const auto& M : R.Messages)
		{
			if (M.Code == EPipelineValidationCode::InvalidPostprocess) { ++N; }
		}
		return N;
	}

	/** Exactly one InvalidPostprocess Error, ValidateSettings false. Optionally the report text must contain Needle. */
	void ExpectBad(const UPipelineModelPackage* P, const TCHAR* What, const TCHAR* Needle = nullptr)
	{
		FPipelineValidationReport R;
		const bool bOk = P->ValidateSettings(R);
		TestFalse(FString::Printf(TEXT("%s: ValidateSettings expected false"), What), bOk);
		TestTrue(FString::Printf(TEXT("%s: expected Error InvalidPostprocess; report:\n%s"), What, *R.ToString()), R.HasError(EPipelineValidationCode::InvalidPostprocess));
		TestEqual(FString::Printf(TEXT("%s: expected exactly 1 InvalidPostprocess message; report:\n%s"), What, *R.ToString()), CountPostprocessErrors(R), 1);
		if (Needle)
		{
			TestTrue(FString::Printf(TEXT("%s: message should name the value '%s'; report:\n%s"), What, Needle, *R.ToString()), R.ToString().Contains(Needle));
		}
	}

	void ExpectNoPostprocessError(const UPipelineModelPackage* P, const TCHAR* What)
	{
		FPipelineValidationReport R;
		P->ValidateSettings(R);
		TestFalse(FString::Printf(TEXT("%s: expected no InvalidPostprocess; report:\n%s"), What, *R.ToString()), R.HasError(EPipelineValidationCode::InvalidPostprocess));
		TestEqual(FString::Printf(TEXT("%s: expected 0 InvalidPostprocess messages"), What), CountPostprocessErrors(R), 0);
	}
END_DEFINE_SPEC(FPipelinePackageM4Spec)

void FPipelinePackageM4Spec::Define()
{
	using namespace PipelinePackageM4SpecPrivate;

	AfterEach([this]() { Keep.Reset(); });

	Describe("Resample", [this]()
	{
		It("LexToString(Bicubic) = \"Bicubic\" and every enumerator returns its own name", [this]()
		{
			TestEqual(TEXT("Bicubic"), FString(LexToString(EPipelineImageResample::Bicubic)), FString(TEXT("Bicubic")));
			TestEqual(TEXT("Nearest"), FString(LexToString(EPipelineImageResample::Nearest)), FString(TEXT("Nearest")));
			TestEqual(TEXT("Lanczos"), FString(LexToString(EPipelineImageResample::Lanczos)), FString(TEXT("Lanczos")));
			TestEqual(TEXT("Bilinear"), FString(LexToString(EPipelineImageResample::Bilinear)), FString(TEXT("Bilinear")));
			TestEqual(TEXT("Box"), FString(LexToString(EPipelineImageResample::Box)), FString(TEXT("Box")));
			TestEqual(TEXT("Hamming"), FString(LexToString(EPipelineImageResample::Hamming)), FString(TEXT("Hamming")));
		});

		It("out of range values -> \"Unknown\" (9 and 255)", [this]()
		{
			TestEqual(TEXT("9"), FString(LexToString((EPipelineImageResample)9)), FString(TEXT("Unknown")));
			TestEqual(TEXT("255"), FString(LexToString((EPipelineImageResample)255)), FString(TEXT("Unknown")));
		});

		// The numeric values are PIL's (Image.NEAREST=0, LANCZOS=1, BILINEAR=2, BICUBIC=3, BOX=4, HAMMING=5); goldens/Python depend on them.
		It("numeric values equal PIL's", [this]()
		{
			TestEqual(TEXT("Nearest"), (int32)EPipelineImageResample::Nearest, 0);
			TestEqual(TEXT("Lanczos"), (int32)EPipelineImageResample::Lanczos, 1);
			TestEqual(TEXT("Bilinear"), (int32)EPipelineImageResample::Bilinear, 2);
			TestEqual(TEXT("Bicubic"), (int32)EPipelineImageResample::Bicubic, 3);
			TestEqual(TEXT("Box"), (int32)EPipelineImageResample::Box, 4);
			TestEqual(TEXT("Hamming"), (int32)EPipelineImageResample::Hamming, 5);
		});

		It("FPipelinePreprocessSettings default Resample is Bilinear; the field is read/write on a package", [this]()
		{
			FPipelinePreprocessSettings S;
			TestTrue(TEXT("struct default Bilinear"), S.Resample == EPipelineImageResample::Bilinear);
			UPipelineModelPackage* P = Make(EPipelineTask::ImageClassification);
			TestTrue(TEXT("package default Bilinear"), P->Preprocess.Resample == EPipelineImageResample::Bilinear);
			P->Preprocess.Resample = EPipelineImageResample::Lanczos;
			TestTrue(TEXT("assigned Lanczos kept"), P->Preprocess.Resample == EPipelineImageResample::Lanczos);
		});

		It("every Resample value leaves a valid package valid (no new validation rule on it)", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ImageClassification);
			for (int32 V = 0; V <= 5; ++V)
			{
				P->Preprocess.Resample = (EPipelineImageResample)V;
				FPipelineValidationReport R;
				TestTrue(FString::Printf(TEXT("Resample %d: valid; report:\n%s"), V, *R.ToString()), P->ValidateSettings(R));
			}
		});
	});

	Describe("NMS settings defaults", [this]()
	{
		It("NmsIouThreshold 0.7, MaxDetections 100", [this]()
		{
			FPipelinePostprocessSettings S;
			TestEqual(TEXT("NmsIouThreshold"), S.NmsIouThreshold, 0.7f);
			TestEqual(TEXT("MaxDetections"), S.MaxDetections, 100);
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			TestEqual(TEXT("package NmsIouThreshold"), P->Postprocess.NmsIouThreshold, 0.7f);
			TestEqual(TEXT("package MaxDetections"), P->Postprocess.MaxDetections, 100);
		});
	});

	Describe("InvalidPostprocess", [this]()
	{
		It("LexToString(InvalidPostprocess) = \"InvalidPostprocess\", value 14 (after LicenseAttributionMissing = 13)", [this]()
		{
			TestEqual(TEXT("name"), FString(LexToString(EPipelineValidationCode::InvalidPostprocess)), FString(TEXT("InvalidPostprocess")));
			TestEqual(TEXT("value"), (int32)EPipelineValidationCode::InvalidPostprocess, 14);
			TestEqual(TEXT("previous value unchanged"), (int32)EPipelineValidationCode::LicenseAttributionMissing, 13);
		});

		// Baseline so a failure below is about the new rule, not about the fixture.
		It("valid ObjectDetection package with defaults: clean, 0 messages", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			FPipelineValidationReport R;
			TestTrue(FString::Printf(TEXT("expected valid; report:\n%s"), *R.ToString()), P->ValidateSettings(R));
			TestEqual(TEXT("0 messages"), R.Messages.Num(), 0);
		});

		It("ObjectDetection: NmsIouThreshold -0.1 -> 1 Error InvalidPostprocess naming the value", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			P->Postprocess.NmsIouThreshold = -0.1f;
			ExpectBad(P, TEXT("-0.1"), TEXT("-0.1"));
		});

		It("ObjectDetection: NmsIouThreshold NaN -> Error", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			P->Postprocess.NmsIouThreshold = NaNF;
			ExpectBad(P, TEXT("NaN threshold"));
		});

		// Rejecting only < 0 and NaN: the boundary 0 is valid (suppress any overlap).
		It("ObjectDetection: NmsIouThreshold 0 and 0.5 and 1 are valid", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			for (float V : {0.f, 0.5f, 1.f})
			{
				P->Postprocess.NmsIouThreshold = V;
				ExpectNoPostprocessError(P, *FString::Printf(TEXT("thr %g"), V));
			}
		});

		// > 1 disables NMS; a range check against [0,1] would wrongly reject it.
		It("ObjectDetection: NmsIouThreshold 1.5 -> no error (disables NMS)", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			P->Postprocess.NmsIouThreshold = 1.5f;
			FPipelineValidationReport R;
			TestTrue(FString::Printf(TEXT("expected valid; report:\n%s"), *R.ToString()), P->ValidateSettings(R));
			TestEqual(TEXT("0 messages"), R.Messages.Num(), 0);
		});

		It("ObjectDetection: MaxDetections 0 -> Error", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			P->Postprocess.MaxDetections = 0;
			ExpectBad(P, TEXT("MaxDetections 0"));
		});

		It("ObjectDetection: MaxDetections 10001 -> Error, and negative -> Error", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			P->Postprocess.MaxDetections = 10001;
			ExpectBad(P, TEXT("MaxDetections 10001"), TEXT("10001"));
			P->Postprocess.MaxDetections = -3;
			ExpectBad(P, TEXT("MaxDetections -3"));
		});

		It("ObjectDetection: MaxDetections 1 and 10000 (the bounds) are valid", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			P->Postprocess.MaxDetections = 1;
			ExpectNoPostprocessError(P, TEXT("MaxDetections 1"));
			P->Postprocess.MaxDetections = 10000;
			ExpectNoPostprocessError(P, TEXT("MaxDetections 10000"));
		});

		It("both fields bad -> 2 InvalidPostprocess errors (each field reported)", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ObjectDetection);
			P->Postprocess.NmsIouThreshold = -1.f;
			P->Postprocess.MaxDetections = 0;
			FPipelineValidationReport R;
			TestFalse(TEXT("invalid"), P->ValidateSettings(R));
			TestEqual(FString::Printf(TEXT("2 messages; report:\n%s"), *R.ToString()), CountPostprocessErrors(R), 2);
		});

		// Other tasks: these fields are not checked (an image classifier never reads them).
		It("ImageClassification with the same bad values -> no InvalidPostprocess", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::ImageClassification);
			P->Postprocess.NmsIouThreshold = -0.1f;
			ExpectNoPostprocessError(P, TEXT("thr -0.1"));
			P->Postprocess.NmsIouThreshold = NaNF;
			ExpectNoPostprocessError(P, TEXT("thr NaN"));
			P->Postprocess.NmsIouThreshold = 0.7f;
			P->Postprocess.MaxDetections = 0;
			ExpectNoPostprocessError(P, TEXT("MaxDetections 0"));
			P->Postprocess.MaxDetections = 10001;
			ExpectNoPostprocessError(P, TEXT("MaxDetections 10001"));
		});

		It("TextClassification with bad values -> no InvalidPostprocess", [this]()
		{
			UPipelineModelPackage* P = Make(EPipelineTask::TextClassification);
			P->Postprocess.NmsIouThreshold = -2.f;
			P->Postprocess.MaxDetections = -1;
			ExpectNoPostprocessError(P, TEXT("text task"));
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
