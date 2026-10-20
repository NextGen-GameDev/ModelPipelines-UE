// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.
// WP4.0 (docs/contracts.md C25): InvalidResize rule for StretchCenterCrop.

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

namespace PipelinePackageResizeSpecPrivate
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
}

BEGIN_DEFINE_SPEC(FPipelinePackageResizeSpec, "Pipelines.Core.Package.Resize",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)

	TArray<TStrongObjectPtr<UObject>> Keep;

	UPipelineModelPackage* MakeScc(EPipelineTask Task, int32 RW, int32 RH, int32 W = 224, int32 H = 224)
	{
		UPipelineModelPackage* P = PipelinePackageResizeSpecPrivate::MakeValidPackage(Task);
		Keep.Add(TStrongObjectPtr<UObject>(P));
		Keep.Add(TStrongObjectPtr<UObject>(P->Model.Get()));
		P->Preprocess.ResizeMode = EPipelineImageResize::StretchCenterCrop;
		P->Preprocess.ResizeWidth = RW;
		P->Preprocess.ResizeHeight = RH;
		P->Preprocess.ImageWidth = W;
		P->Preprocess.ImageHeight = H;
		return P;
	}

	void ExpectInvalidResize(const UPipelineModelPackage* P, const TCHAR* What)
	{
		FPipelineValidationReport R;
		const bool bOk = P->ValidateSettings(R);
		TestFalse(FString::Printf(TEXT("%s: ValidateSettings expected false"), What), bOk);
		TestTrue(FString::Printf(TEXT("%s: expected Error InvalidResize; report:\n%s"), What, *R.ToString()), R.HasError(EPipelineValidationCode::InvalidResize));
		TestEqual(FString::Printf(TEXT("%s: expected exactly 1 message; report:\n%s"), What, *R.ToString()), R.Messages.Num(), 1);
	}

	void ExpectClean(const UPipelineModelPackage* P, const TCHAR* What)
	{
		FPipelineValidationReport R;
		const bool bOk = P->ValidateSettings(R);
		TestTrue(FString::Printf(TEXT("%s: ValidateSettings expected true; report:\n%s"), What, *R.ToString()), bOk);
		TestEqual(FString::Printf(TEXT("%s: expected 0 messages; report:\n%s"), What, *R.ToString()), R.Messages.Num(), 0);
	}
END_DEFINE_SPEC(FPipelinePackageResizeSpec)

void FPipelinePackageResizeSpec::Define()
{
	AfterEach([this]() { Keep.Reset(); });

	Describe("StretchCenterCrop", [this]()
	{
		// A resize smaller than the crop would need padding (HF pads; we reject). Catches a rule that only checks the range.
		It("rejects ResizeWidth 200 < ImageWidth 224 (resize 200x256, image 224x224)", [this]()
		{
			ExpectInvalidResize(MakeScc(EPipelineTask::ImageClassification, 200, 256), TEXT("200x256"));
		});

		// Catches a rule that checks width only.
		It("rejects ResizeHeight 200 < ImageHeight 224 (resize 256x200)", [this]()
		{
			ExpectInvalidResize(MakeScc(EPipelineTask::ImageClassification, 256, 200), TEXT("256x200"));
		});

		// 0 is "unset": StretchCenterCrop without sizes must not pass.
		It("rejects ResizeWidth 0 (resize 0x256)", [this]()
		{
			ExpectInvalidResize(MakeScc(EPipelineTask::ImageClassification, 0, 256), TEXT("0x256"));
		});

		It("rejects ResizeHeight 0 (resize 256x0)", [this]()
		{
			ExpectInvalidResize(MakeScc(EPipelineTask::ImageClassification, 256, 0), TEXT("256x0"));
		});

		It("rejects negative resize (-1x256)", [this]()
		{
			ExpectInvalidResize(MakeScc(EPipelineTask::ImageClassification, -1, 256), TEXT("-1x256"));
		});

		// Upper bound 8192; with Image 224 the lower-bound check passes, so only the range check can catch it.
		It("rejects resize 9000x9000 (above 8192)", [this]()
		{
			ExpectInvalidResize(MakeScc(EPipelineTask::ImageClassification, 9000, 9000), TEXT("9000x9000"));
		});

		It("rejects only the oversized side (256x9000)", [this]()
		{
			ExpectInvalidResize(MakeScc(EPipelineTask::ImageClassification, 256, 9000), TEXT("256x9000"));
		});

		// Off by one: 224x223 against image 224x224 -> height too small.
		It("rejects resize 224x223 with image 224x224 (off by one)", [this]()
		{
			ExpectInvalidResize(MakeScc(EPipelineTask::ImageClassification, 224, 223), TEXT("224x223"));
		});

		It("accepts resize 256x256 with image 224x224", [this]()
		{
			ExpectClean(MakeScc(EPipelineTask::ImageClassification, 256, 256), TEXT("256x256"));
		});

		// Equal sizes are a no-op crop and are allowed (rule is strictly "<").
		It("accepts resize 224x224 with image 224x224", [this]()
		{
			ExpectClean(MakeScc(EPipelineTask::ImageClassification, 224, 224), TEXT("224x224"));
		});

		It("accepts the upper bound 8192x8192", [this]()
		{
			ExpectClean(MakeScc(EPipelineTask::ImageClassification, 8192, 8192), TEXT("8192x8192"));
		});

		// Non-square crop with distinct resize sides; catches swapped width/height comparisons.
		It("accepts resize 260x240 with image 200x224 and rejects 200x260 with image 224x200", [this]()
		{
			ExpectClean(MakeScc(EPipelineTask::ImageClassification, 260, 240, 200, 224), TEXT("260x240 vs 200x224"));
			ExpectInvalidResize(MakeScc(EPipelineTask::ImageClassification, 200, 260, 224, 200), TEXT("200x260 vs 224x200"));
		});

		It("applies to ObjectDetection too (resize 100x100, image 224x224)", [this]()
		{
			ExpectInvalidResize(MakeScc(EPipelineTask::ObjectDetection, 100, 100), TEXT("OD 100x100"));
		});
	});

	Describe("OtherModes", [this]()
	{
		// Stretch must ignore Resize fields, even nonsense ones.
		It("Stretch ignores ResizeWidth/ResizeHeight (0, 9000, 5)", [this]()
		{
			const int32 Vals[3][2] = {{0, 0}, {9000, 9000}, {5, 5}};
			for (const int32(&V)[2] : Vals)
			{
				UPipelineModelPackage* P = MakeScc(EPipelineTask::ImageClassification, V[0], V[1]);
				P->Preprocess.ResizeMode = EPipelineImageResize::Stretch;
				ExpectClean(P, *FString::Printf(TEXT("Stretch resize %dx%d"), V[0], V[1]));
			}
		});

		It("ShortestEdgeCenterCrop ignores ResizeWidth/ResizeHeight", [this]()
		{
			const int32 Vals[3][2] = {{0, 0}, {9000, 9000}, {5, 5}};
			for (const int32(&V)[2] : Vals)
			{
				UPipelineModelPackage* P = MakeScc(EPipelineTask::ImageClassification, V[0], V[1]);
				P->Preprocess.ResizeMode = EPipelineImageResize::ShortestEdgeCenterCrop;
				P->Preprocess.ResizeShortestEdge = 256;
				ExpectClean(P, *FString::Printf(TEXT("ShortestEdgeCenterCrop resize %dx%d"), V[0], V[1]));
			}
		});

		// Text tasks have no image settings; even a StretchCenterCrop mode with bad sizes must not produce InvalidResize.
		It("text tasks ignore the resize fields", [this]()
		{
			const EPipelineTask Tasks[2] = {EPipelineTask::TextEmbedding, EPipelineTask::TextClassification};
			for (EPipelineTask T : Tasks)
			{
				UPipelineModelPackage* P = MakeScc(T, 0, 9000);
				FPipelineValidationReport R;
				P->ValidateSettings(R);
				TestFalse(FString::Printf(TEXT("text task: no InvalidResize; report:\n%s"), *R.ToString()), R.HasError(EPipelineValidationCode::InvalidResize));
			}
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
