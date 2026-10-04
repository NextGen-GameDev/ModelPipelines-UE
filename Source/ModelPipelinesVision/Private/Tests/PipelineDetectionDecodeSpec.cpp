// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
// WP4.8a visible tests (docs/contracts.md C29 "Detection postprocess" rules 1-4, C30 tiny_models goldens, C26 NMS):
// DecodeDetections, the pure DETR-family decode. Golden expectations come from the JSON at run time; hand examples use
// numbers worked out on paper in the comments.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include <limits>
#include "Async/Async.h"
#include "Async/Future.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "HAL/PlatformTime.h"
#include "Math/RandomStream.h"
#include "PipelineDetectionDecode.h"
#include "PipelineImage.h"
#include "PipelineModelPackage.h"
#include "PipelineResults.h"
#include "PipelineTypes.h"
#include "PipelineVisionTestData.h"

using namespace UE::ModelPipelines::VisionTests;
using UE::ModelPipelines::Vision::DecodeDetections;
using UE::ModelPipelines::Vision::FPipelineDetectionDecodeParams;

namespace PipelineDetectionDecodeSpecPrivate
{
	static const float GQuietNaN = std::numeric_limits<float>::quiet_NaN();
	static const float GInf = std::numeric_limits<float>::infinity();

	struct FInput
	{
		TArray<float> Logits;
		int32 Q = 0;
		int32 C = 0;   // number of logit columns (C')
		TArray<float> Boxes;
		TArray<FString> Labels;
		FPipelineDetectionDecodeParams Params;
	};

	static FPipelineDetectionDecodeParams MakeParams(EPipelineScoreActivation Act, float Threshold, float Nms = 0.7f, int32 MaxDet = 100)
	{
		FPipelineDetectionDecodeParams P;
		P.Activation = Act;
		P.ScoreThreshold = Threshold;
		P.NmsIouThreshold = Nms;
		P.MaxDetections = MaxDet;
		return P;
	}

	static FPipelineResult Decode(const FInput& In, TArray<FPipelineDetection>& Out)
	{
		return DecodeDetections(In.Logits, In.Q, In.C, In.Boxes, In.Labels, In.Params, Out);
	}

	static double SigmoidD(double X) { return 1.0 / (1.0 + FMath::Exp(-X)); }

	/** One-query input: Sigmoid, labels L0..L(C-1), box (0.5, 0.5, 0.2, 0.2). */
	static FInput OneQuerySigmoid(const TArray<float>& Logits, float Threshold)
	{
		FInput In;
		In.Q = 1;
		In.C = Logits.Num();
		In.Logits = Logits;
		In.Boxes = { 0.5f, 0.5f, 0.2f, 0.2f };
		for (int32 i = 0; i < In.C; ++i)
		{
			In.Labels.Add(FString::Printf(TEXT("L%d"), i));
		}
		In.Params = MakeParams(EPipelineScoreActivation::Sigmoid, Threshold);
		return In;
	}

	/** Checks one detection; every mismatch becomes an error naming Ctx. Returns the worst |score/box diff|. */
	static double ExpectDet(FAutomationTestBase& T, const FString& Ctx, const FPipelineDetection& D, const FString& Label, int32 Index,
		float Score, float X0, float Y0, float X1, float Y1, double Tol)
	{
		if (D.Label != Label)
		{
			T.AddError(FString::Printf(TEXT("%s: label '%s', expected '%s'."), *Ctx, *D.Label, *Label));
		}
		if (D.Index != Index)
		{
			T.AddError(FString::Printf(TEXT("%s: index %d, expected %d."), *Ctx, D.Index, Index));
		}
		if (!D.Box.bIsValid)
		{
			T.AddError(FString::Printf(TEXT("%s: Box.bIsValid is false."), *Ctx));
		}
		const double Diffs[5] = { FMath::Abs((double)D.Score - Score), FMath::Abs((double)D.Box.Min.X - X0), FMath::Abs((double)D.Box.Min.Y - Y0),
			FMath::Abs((double)D.Box.Max.X - X1), FMath::Abs((double)D.Box.Max.Y - Y1) };
		const TCHAR* Names[5] = { TEXT("score"), TEXT("box.min.x"), TEXT("box.min.y"), TEXT("box.max.x"), TEXT("box.max.y") };
		const double Got[5] = { D.Score, D.Box.Min.X, D.Box.Min.Y, D.Box.Max.X, D.Box.Max.Y };
		const double Exp[5] = { Score, X0, Y0, X1, Y1 };
		double Worst = 0.0;
		for (int32 i = 0; i < 5; ++i)
		{
			Worst = FMath::Max(Worst, Diffs[i]);
			if (!(Diffs[i] <= Tol))
			{
				T.AddError(FString::Printf(TEXT("%s: %s = %.9g, expected %.9g (|diff| %.3g > %.3g)."), *Ctx, Names[i], Got[i], Exp[i], Diffs[i], Tol));
			}
		}
		return Worst;
	}

	static bool ExpectFail(FAutomationTestBase& T, const TCHAR* Ctx, const FInput& In, EPipelineError Expected)
	{
		TArray<FPipelineDetection> Out;
		FPipelineDetection Stale;
		Stale.Label = TEXT("stale");
		Out.Add(Stale);   // "Out is emptied on failure"
		const FPipelineResult R = Decode(In, Out);
		bool bOk = true;
		if (R.bSuccess)
		{
			T.AddError(FString::Printf(TEXT("%s: expected a failure, got success."), Ctx));
			bOk = false;
		}
		else
		{
			if (R.Error != Expected)
			{
				T.AddError(FString::Printf(TEXT("%s: error code %d, expected %d."), Ctx, (int32)R.Error, (int32)Expected));
				bOk = false;
			}
			if (R.Message.IsEmpty())
			{
				T.AddError(FString::Printf(TEXT("%s: failure message is empty (it must name the problem)."), Ctx));
				bOk = false;
			}
		}
		if (Out.Num() != 0)
		{
			T.AddError(FString::Printf(TEXT("%s: Out holds %d entries after a failure, expected it emptied."), Ctx, Out.Num()));
			bOk = false;
		}
		return bOk;
	}

	static bool ReadFloatArray(const TSharedPtr<FJsonObject>& O, const TCHAR* Field, TArray<float>& Out)
	{
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (!O.IsValid() || !O->TryGetArrayField(Field, Arr))
		{
			return false;
		}
		Out.Reset();
		for (const TSharedPtr<FJsonValue>& V : *Arr)
		{
			Out.Add((float)V->AsNumber());
		}
		return true;
	}

	static TSharedPtr<FJsonObject> GCachedRoot;

	static TSharedPtr<FJsonObject> GetGoldens(FAutomationTestBase& T)
	{
		if (!GCachedRoot.IsValid())
		{
			FString Err;
			GCachedRoot = LoadVisionGoldens(Err);
			if (!GCachedRoot.IsValid())
			{
				T.AddError(Err);
			}
		}
		return GCachedRoot;
	}

	/** Decodes every case of one tiny_models entry (by slug) and compares with its goldens. Returns the number of cases checked. */
	static int32 RunGoldenModel(FAutomationTestBase& T, const FString& Slug)
	{
		const TSharedPtr<FJsonObject> Root = GetGoldens(T);
		if (!Root.IsValid())
		{
			return 0;
		}
		const TArray<TSharedPtr<FJsonValue>>* Models = nullptr;
		if (!Root->TryGetArrayField(TEXT("tiny_models"), Models))
		{
			T.AddError(TEXT("vision_goldens.json has no tiny_models array. Regenerate: Tools/make_vision_goldens.py (see docs/contracts.md C30)."));
			return 0;
		}
		TSharedPtr<FJsonObject> Model;
		for (const TSharedPtr<FJsonValue>& V : *Models)
		{
			const TSharedPtr<FJsonObject> O = V->AsObject();
			if (O.IsValid() && O->GetStringField(TEXT("slug")) == Slug)
			{
				Model = O;
				break;
			}
		}
		if (!Model.IsValid())
		{
			T.AddError(FString::Printf(TEXT("tiny_models has no entry '%s'. Regenerate: Tools/make_vision_goldens.py (see docs/contracts.md C30)."), *Slug));
			return 0;
		}
		const TSharedPtr<FJsonObject> Pkg = Model->GetObjectField(TEXT("package"));
		const FString ActName = Pkg->GetStringField(TEXT("score_activation"));
		const EPipelineScoreActivation Act = ActName == TEXT("Sigmoid") ? EPipelineScoreActivation::Sigmoid : EPipelineScoreActivation::Softmax;
		TArray<FString> Labels;
		for (const TSharedPtr<FJsonValue>& L : Pkg->GetArrayField(TEXT("labels")))
		{
			Labels.Add(L->AsString());
		}

		int32 Checked = 0;
		double WorstAll = 0.0;
		FString WorstWhere;
		for (const TSharedPtr<FJsonValue>& CV : Model->GetArrayField(TEXT("cases")))
		{
			const TSharedPtr<FJsonObject> Case = CV->AsObject();
			const FString Ctx = FString::Printf(TEXT("%s/%s thr=%.2f nms=%.2f max=%d"), *Slug, *Case->GetStringField(TEXT("image")),
				Case->GetNumberField(TEXT("score_threshold")), Case->GetNumberField(TEXT("nms_iou_threshold")), (int32)Case->GetNumberField(TEXT("max_detections")));

			FInput In;
			TArray<float> LogitsShape, BoxShape;
			if (!ReadFloatArray(Case, TEXT("logits"), In.Logits) || !ReadFloatArray(Case, TEXT("pred_boxes"), In.Boxes)
				|| !ReadFloatArray(Case, TEXT("logits_shape"), LogitsShape) || LogitsShape.Num() != 3)
			{
				T.AddError(FString::Printf(TEXT("%s: case lacks logits / pred_boxes / logits_shape."), *Ctx));
				continue;
			}
			In.Q = (int32)LogitsShape[1];
			In.C = (int32)LogitsShape[2];
			In.Labels = Labels;
			In.Params = MakeParams(Act, (float)Case->GetNumberField(TEXT("score_threshold")), (float)Case->GetNumberField(TEXT("nms_iou_threshold")),
				(int32)Case->GetNumberField(TEXT("max_detections")));   // Geometry stays all-zero: identity

			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			if (!R.bSuccess)
			{
				T.AddError(FString::Printf(TEXT("%s: decode failed: %s"), *Ctx, *R.Message));
				continue;
			}
			const TArray<TSharedPtr<FJsonValue>>& Exp = Case->GetArrayField(TEXT("detections"));
			if (Out.Num() != Exp.Num())
			{
				T.AddError(FString::Printf(TEXT("%s: %d detections, golden has %d."), *Ctx, Out.Num(), Exp.Num()));
				++Checked;
				continue;
			}
			for (int32 i = 0; i < Exp.Num(); ++i)
			{
				const TSharedPtr<FJsonObject> E = Exp[i]->AsObject();
				const TArray<TSharedPtr<FJsonValue>>& B = E->GetArrayField(TEXT("box"));
				const double W = ExpectDet(T, FString::Printf(TEXT("%s det#%d"), *Ctx, i), Out[i], E->GetStringField(TEXT("label")),
					(int32)E->GetNumberField(TEXT("index")), (float)E->GetNumberField(TEXT("score")), (float)B[0]->AsNumber(), (float)B[1]->AsNumber(),
					(float)B[2]->AsNumber(), (float)B[3]->AsNumber(), 1e-5);
				if (W > WorstAll)
				{
					WorstAll = W;
					WorstWhere = FString::Printf(TEXT("%s det#%d"), *Ctx, i);
				}
			}
			++Checked;
		}
		T.AddInfo(FString::Printf(TEXT("%s: %d cases, worst |diff| %.3g (%s), tolerance 1e-5."), *Slug, Checked, WorstAll, *WorstWhere));
		return Checked;
	}
}

using namespace PipelineDetectionDecodeSpecPrivate;

BEGIN_DEFINE_SPEC(FPipelineDetectionDecodeVisibleSpec, "Pipelines.Vision.Detection.Decode",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FPipelineDetectionDecodeVisibleSpec)

void FPipelineDetectionDecodeVisibleSpec::Define()
{
	Describe("Golden", [this]()
	{
		// Parity against the Python numpy reference: catches any deviation in activation, ordering, box maths, NMS or clamping.
		It("designed Sigmoid model: every case matches", [this]()
		{
			const int32 N = RunGoldenModel(*this, TEXT("tiny_det_designed_sigmoid"));
			TestTrue(TEXT("at least one tiny_det_designed_sigmoid case was checked"), N > 0);
		});
		It("designed Softmax model: every case matches", [this]()
		{
			const int32 N = RunGoldenModel(*this, TEXT("tiny_det_designed_softmax"));
			TestTrue(TEXT("at least one tiny_det_designed_softmax case was checked"), N > 0);
		});
		It("tiny_object_detection: every case matches", [this]()
		{
			const int32 N = RunGoldenModel(*this, TEXT("tiny_object_detection"));
			TestTrue(TEXT("at least one tiny_object_detection case was checked"), N > 0);
		});
		It("designed goldens run the four parameter sets", [this]()
		{
			// Guards against a regenerated golden file silently dropping cases (then the parity tests prove less).
			const TSharedPtr<FJsonObject> Root = GetGoldens(*this);
			if (!Root.IsValid()) { return; }
			int32 DetModels = 0;
			for (const TSharedPtr<FJsonValue>& V : Root->GetArrayField(TEXT("tiny_models")))
			{
				const TSharedPtr<FJsonObject> O = V->AsObject();
				if (O->GetStringField(TEXT("task")) == TEXT("ObjectDetection"))
				{
					++DetModels;
					const FString Slug = O->GetStringField(TEXT("slug"));
					if (Slug.StartsWith(TEXT("tiny_det_designed")))
					{
						TestTrue(*FString::Printf(TEXT("%s has 8 cases (4 parameter sets x 2 images)"), *Slug), O->GetArrayField(TEXT("cases")).Num() == 8);
					}
				}
			}
			TestTrue(TEXT("three detection models in tiny_models"), DetModels >= 3);
		});
	});

	Describe("Hand", [this]()
	{
		It("Sigmoid logit 0 scores 0.5; strict threshold: dropped at 0.5, kept at 0.49", [this]()
		{
			// sigmoid(0) = 0.5. "score > threshold" is strict, so 0.5 is not kept at 0.5 (a >= would keep it).
			TArray<FPipelineDetection> Out;
			FPipelineResult R = Decode(OneQuerySigmoid({ 0.f }, 0.5f), Out);
			TestTrue(TEXT("threshold 0.5 succeeds"), R.bSuccess);
			TestEqual(TEXT("threshold 0.5 keeps nothing"), Out.Num(), 0);
			R = Decode(OneQuerySigmoid({ 0.f }, 0.49f), Out);
			TestTrue(TEXT("threshold 0.49 succeeds"), R.bSuccess);
			if (TestEqual(TEXT("threshold 0.49 keeps one"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("logit 0"), Out[0], TEXT("L0"), 0, 0.5f, 0.4f, 0.4f, 0.6f, 0.6f, 1e-6);
			}
		});

		It("Sigmoid score matches 1/(1+exp(-x)) for several logits", [this]()
		{
			// Catches a wrong activation (e.g. softmax over classes or tanh) on a one-class model.
			const float Logits[] = { -3.f, -0.7f, 0.25f, 2.f, 6.f };
			for (float L : Logits)
			{
				TArray<FPipelineDetection> Out;
				const FPipelineResult R = Decode(OneQuerySigmoid({ L }, 0.f), Out);
				if (TestTrue(TEXT("decode succeeds"), R.bSuccess) && TestEqual(TEXT("one detection"), Out.Num(), 1))
				{
					ExpectDet(*this, FString::Printf(TEXT("logit %g"), L), Out[0], TEXT("L0"), 0, (float)SigmoidD(L), 0.4f, 0.4f, 0.6f, 0.6f, 1e-6);
				}
			}
		});

		It("Sigmoid: a query with two classes above threshold gives two detections in score order", [this]()
		{
			// logits 1 and 2 -> 0.7311 and 0.8808; both candidates (q,c), different classes so class-wise NMS keeps both on the same box.
			// Lead fix: Q = 2 (second query far below the threshold) so the top-Q cap (C29 rule 1) allows two candidates.
			FInput In = OneQuerySigmoid({ 1.f, 2.f, -9.f, -9.f }, 0.5f);
			In.Q = 2; In.C = 2; In.Labels.SetNum(2);
			In.Boxes = { 0.5f, 0.5f, 0.2f, 0.2f, 0.1f, 0.1f, 0.1f, 0.1f };
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("two detections"), Out.Num(), 2))
			{
				ExpectDet(*this, TEXT("best"), Out[0], TEXT("L1"), 1, (float)SigmoidD(2.0), 0.4f, 0.4f, 0.6f, 0.6f, 1e-6);
				ExpectDet(*this, TEXT("second"), Out[1], TEXT("L0"), 0, (float)SigmoidD(1.0), 0.4f, 0.4f, 0.6f, 0.6f, 1e-6);
			}
		});

		It("Sigmoid ties: equal scores order by flat index q*C+c and at most Q candidates survive", [this]()
		{
			// Q=2, C=2, all logits 0 -> four candidates with score 0.5. Flat order: (0,0),(0,1),(1,0),(1,1). Top-Q keeps 2:
			// both from query 0 (class 0 then class 1), box of query 0. A decoder that sorted by class first, or kept all four, fails.
			FInput In;
			In.Q = 2; In.C = 2;
			In.Logits = { 0.f, 0.f, 0.f, 0.f };
			In.Boxes = { 0.3f, 0.3f, 0.2f, 0.2f,   0.8f, 0.8f, 0.2f, 0.2f };
			In.Labels = { TEXT("a"), TEXT("b") };
			In.Params = MakeParams(EPipelineScoreActivation::Sigmoid, 0.49f);
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("exactly Q = 2 detections"), Out.Num(), 2))
			{
				ExpectDet(*this, TEXT("first"), Out[0], TEXT("a"), 0, 0.5f, 0.2f, 0.2f, 0.4f, 0.4f, 1e-6);
				ExpectDet(*this, TEXT("second"), Out[1], TEXT("b"), 1, 0.5f, 0.2f, 0.2f, 0.4f, 0.4f, 1e-6);
			}
		});

		It("Softmax [0,0,0] over C'=3: score 1/3, class 0 (ties lower class)", [this]()
		{
			// Two labels + a no-object column. p = 1/3 each. Best of the first two is class 0 on ties.
			FInput In;
			In.Q = 1; In.C = 3;
			In.Logits = { 0.f, 0.f, 0.f };
			In.Boxes = { 0.5f, 0.5f, 0.2f, 0.2f };
			In.Labels = { TEXT("a"), TEXT("b") };
			In.Params = MakeParams(EPipelineScoreActivation::Softmax, 0.3f);
			TArray<FPipelineDetection> Out;
			FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("one detection at 0.3"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("softmax uniform"), Out[0], TEXT("a"), 0, 1.0f / 3.0f, 0.4f, 0.4f, 0.6f, 0.6f, 1e-6);
			}
			In.Params.ScoreThreshold = 0.34f;
			R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds at 0.34"), R.bSuccess);
			TestEqual(TEXT("1/3 is below 0.34"), Out.Num(), 0);
		});

		It("Softmax normalizes over all C' columns and drops only the last", [this]()
		{
			// logits [0, 3, 5]: p = exp(l)/sum = [0.00674, 0.1353, 1]/1.1421... -> class 1 wins among the first two with 0.1185.
			// Normalizing over only the first C columns would give 0.953; a decoder using the no-object column as a class would pick it.
			FInput In;
			In.Q = 1; In.C = 3;
			In.Logits = { 0.f, 3.f, 5.f };
			In.Boxes = { 0.5f, 0.5f, 0.2f, 0.2f };
			In.Labels = { TEXT("a"), TEXT("b") };
			In.Params = MakeParams(EPipelineScoreActivation::Softmax, 0.1f);
			const double Sum = FMath::Exp(0.0) + FMath::Exp(3.0) + FMath::Exp(5.0);
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("one detection"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("softmax"), Out[0], TEXT("b"), 1, (float)(FMath::Exp(3.0) / Sum), 0.4f, 0.4f, 0.6f, 0.6f, 1e-6);
			}
		});

		It("Softmax with huge logits stays finite (max-subtracted)", [this]()
		{
			// exp(1000) overflows a naive softmax to inf/inf = NaN. [1000,0,0]: p0 = 1. [0,0,1000]: object classes ~0, below any threshold.
			FInput In;
			In.Q = 1; In.C = 3;
			In.Logits = { 1000.f, 0.f, 0.f };
			In.Boxes = { 0.5f, 0.5f, 0.2f, 0.2f };
			In.Labels = { TEXT("a"), TEXT("b") };
			In.Params = MakeParams(EPipelineScoreActivation::Softmax, 0.5f);
			TArray<FPipelineDetection> Out;
			FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds for [1000,0,0]"), R.bSuccess);
			if (TestEqual(TEXT("one detection"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("[1000,0,0]"), Out[0], TEXT("a"), 0, 1.0f, 0.4f, 0.4f, 0.6f, 0.6f, 1e-6);
			}
			In.Logits = { 0.f, 0.f, 1000.f };
			R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds for [0,0,1000]"), R.bSuccess);
			TestEqual(TEXT("no-object dominates: nothing kept"), Out.Num(), 0);
		});

		It("Softmax ties between queries order by lower query index", [this]()
		{
			// Two queries with identical logits and disjoint boxes: the output order is q0 then q1.
			FInput In;
			In.Q = 2; In.C = 3;
			In.Logits = { 3.f, 0.f, 0.f,   3.f, 0.f, 0.f };
			In.Boxes = { 0.25f, 0.25f, 0.2f, 0.2f,   0.75f, 0.75f, 0.2f, 0.2f };
			In.Labels = { TEXT("a"), TEXT("b") };
			In.Params = MakeParams(EPipelineScoreActivation::Softmax, 0.5f);
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("two detections"), Out.Num(), 2))
			{
				const float P = (float)(FMath::Exp(3.0) / (FMath::Exp(3.0) + 2.0));
				ExpectDet(*this, TEXT("q0"), Out[0], TEXT("a"), 0, P, 0.15f, 0.15f, 0.35f, 0.35f, 1e-6);
				ExpectDet(*this, TEXT("q1"), Out[1], TEXT("a"), 0, P, 0.65f, 0.65f, 0.85f, 0.85f, 1e-6);
			}
		});

		It("Box centre/size converts to corners and clamps to [0,1]", [this]()
		{
			// (cx 0.5, cy 0.5, w 1.2, h 0.4): x 0.5 -/+ 0.6 = -0.1 .. 1.1 -> 0 .. 1; y 0.3 .. 0.7.
			FInput In = OneQuerySigmoid({ 3.f }, 0.5f);
			In.Boxes = { 0.5f, 0.5f, 1.2f, 0.4f };
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("one detection"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("wide box"), Out[0], TEXT("L0"), 0, (float)SigmoidD(3.0), 0.0f, 0.3f, 1.0f, 0.7f, 1e-6);
			}
		});

		It("Box outside the top/left and bottom/right clamps each side separately", [this]()
		{
			// cx 0.1 cy 0.9 w 0.5 h 0.5: x -0.15 .. 0.35 -> 0 .. 0.35; y 0.65 .. 1.15 -> 0.65 .. 1.
			FInput In = OneQuerySigmoid({ 3.f }, 0.5f);
			In.Boxes = { 0.1f, 0.9f, 0.5f, 0.5f };
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("one detection"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("corner box"), Out[0], TEXT("L0"), 0, (float)SigmoidD(3.0), 0.0f, 0.65f, 0.35f, 1.0f, 1e-6);
			}
		});

		It("Geometry: resized 341x256, crop (58,16) 224x224 maps the unit box to (58/341,16/256)-(282/341,240/256)", [this]()
		{
			// x_src = (CropLeft + x*CropWidth)/ResizedWidth. A decoder that ignores the crop returns (0,0)-(1,1);
			// one that divides by the crop size instead of the resized size is off by a large amount.
			FInput In = OneQuerySigmoid({ 3.f }, 0.5f);
			In.Boxes = { 0.5f, 0.5f, 1.0f, 1.0f };
			In.Params.Geometry.SourceWidth = 640; In.Params.Geometry.SourceHeight = 480;
			In.Params.Geometry.ResizedWidth = 341; In.Params.Geometry.ResizedHeight = 256;
			In.Params.Geometry.CropLeft = 58; In.Params.Geometry.CropTop = 16;
			In.Params.Geometry.CropWidth = 224; In.Params.Geometry.CropHeight = 224;
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("one detection"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("crop mapping"), Out[0], TEXT("L0"), 0, (float)SigmoidD(3.0), 58.f / 341.f, 16.f / 256.f, 282.f / 341.f, 240.f / 256.f, 1e-6);
			}
		});

		It("Geometry: clamping happens after the mapping, not before", [this]()
		{
			// Box x -0.2 .. 1.2 (cx 0.5, w 1.4), y 0 .. 1. Mapped x: (58 - 44.8)/341 = 0.03871 .. (58 + 268.8)/341 = 0.95836; both inside [0,1].
			// A decoder that clamped to [0,1] first would return 58/341 = 0.1701 .. 282/341 = 0.8270.
			FInput In = OneQuerySigmoid({ 3.f }, 0.5f);
			In.Boxes = { 0.5f, 0.5f, 1.4f, 1.0f };
			In.Params.Geometry.SourceWidth = 640; In.Params.Geometry.SourceHeight = 480;
			In.Params.Geometry.ResizedWidth = 341; In.Params.Geometry.ResizedHeight = 256;
			In.Params.Geometry.CropLeft = 58; In.Params.Geometry.CropTop = 16;
			In.Params.Geometry.CropWidth = 224; In.Params.Geometry.CropHeight = 224;
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("one detection"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("map then clamp"), Out[0], TEXT("L0"), 0, (float)SigmoidD(3.0), 13.2f / 341.f, 16.f / 256.f, 326.8f / 341.f, 240.f / 256.f, 1e-5);
			}
		});

		It("Geometry: an inner box maps with the crop offset on both axes", [this]()
		{
			// Box (cx 0.25, cy 0.75, w 0.2, h 0.1): x .15 .. .35, y .70 .. .80.
			// x_src = (58 + 33.6)/341 .. (58 + 78.4)/341 ; y_src = (16 + 156.8)/256 .. (16 + 179.2)/256.
			FInput In = OneQuerySigmoid({ 3.f }, 0.5f);
			In.Boxes = { 0.25f, 0.75f, 0.2f, 0.1f };
			In.Params.Geometry.SourceWidth = 640; In.Params.Geometry.SourceHeight = 480;
			In.Params.Geometry.ResizedWidth = 341; In.Params.Geometry.ResizedHeight = 256;
			In.Params.Geometry.CropLeft = 58; In.Params.Geometry.CropTop = 16;
			In.Params.Geometry.CropWidth = 224; In.Params.Geometry.CropHeight = 224;
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("one detection"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("inner box"), Out[0], TEXT("L0"), 0, (float)SigmoidD(3.0), 91.6f / 341.f, 172.8f / 256.f, 136.4f / 341.f, 195.2f / 256.f, 1e-5);
			}
		});

		It("Class-wise NMS: same class IoU 0.9 suppressed, different class kept, labels pass through (non-ASCII)", [this]()
		{
			// Boxes A (0.1,0.1)-(0.5,0.5) and B (0.12,0.1)-(0.52,0.5): inter 0.152, union 0.168, IoU 0.905 > 0.7.
			// Q=2, C=2. Case 1: both queries class 0 -> B suppressed. Case 2: q1 class 1 -> both kept.
			FInput In;
			In.Q = 2; In.C = 2;
			In.Boxes = { 0.3f, 0.3f, 0.4f, 0.4f,   0.32f, 0.3f, 0.4f, 0.4f };
			In.Labels = { TEXT("café"), TEXT("日本") };
			In.Params = MakeParams(EPipelineScoreActivation::Sigmoid, 0.5f, 0.7f);
			In.Logits = { 3.f, -10.f,   2.5f, -10.f };
			TArray<FPipelineDetection> Out;
			FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("same-class decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("same class: one survives"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("survivor"), Out[0], TEXT("café"), 0, (float)SigmoidD(3.0), 0.1f, 0.1f, 0.5f, 0.5f, 1e-6);
			}
			In.Logits = { 3.f, -10.f,   -10.f, 2.5f };
			R = Decode(In, Out);
			TestTrue(TEXT("different-class decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("different class: both kept"), Out.Num(), 2))
			{
				ExpectDet(*this, TEXT("first"), Out[0], TEXT("café"), 0, (float)SigmoidD(3.0), 0.1f, 0.1f, 0.5f, 0.5f, 1e-6);
				ExpectDet(*this, TEXT("second"), Out[1], TEXT("日本"), 1, (float)SigmoidD(2.5), 0.12f, 0.1f, 0.52f, 0.5f, 1e-6);
			}
		});

		It("NMS threshold: IoU 0.6 is kept at 0.7 and suppressed at 0.5; threshold 1.0 disables NMS", [this]()
		{
			// A (0.1,0.1)-(0.5,0.5) and B shifted by 0.1 in x: inter 0.3*0.4 = 0.12, union 0.2, IoU 0.6.
			FInput In;
			In.Q = 2; In.C = 1;
			In.Logits = { 3.f, 2.f };
			In.Boxes = { 0.3f, 0.3f, 0.4f, 0.4f,   0.4f, 0.3f, 0.4f, 0.4f };
			In.Labels = { TEXT("a") };
			TArray<FPipelineDetection> Out;
			In.Params = MakeParams(EPipelineScoreActivation::Sigmoid, 0.5f, 0.7f);
			TestTrue(TEXT("decode 0.7"), Decode(In, Out).bSuccess);
			TestEqual(TEXT("IoU 0.6 kept at nms 0.7"), Out.Num(), 2);
			In.Params.NmsIouThreshold = 0.5f;
			TestTrue(TEXT("decode 0.5"), Decode(In, Out).bSuccess);
			TestEqual(TEXT("IoU 0.6 suppressed at nms 0.5"), Out.Num(), 1);
			In.Params.NmsIouThreshold = 1.0f;
			TestTrue(TEXT("decode 1.0"), Decode(In, Out).bSuccess);
			TestEqual(TEXT("nms 1.0 disables suppression"), Out.Num(), 2);
		});

		It("NMS runs on mapped, unclamped boxes", [this]()
		{
			// A: x 0.5 .. 1.5, B: x 0.9 .. 1.9, same y (0.3 .. 0.7). Unclamped IoU = 0.6/1.4 = 0.43 > 0.3 -> B suppressed.
			// Clamped first they would be 0.5..1 and 0.9..1: IoU 0.2 < 0.3 and both would survive. The survivor is A, clamped to 0.5 .. 1.
			FInput In;
			In.Q = 2; In.C = 1;
			In.Logits = { 3.f, 2.f };
			In.Boxes = { 1.0f, 0.5f, 1.0f, 0.4f,   1.4f, 0.5f, 1.0f, 0.4f };
			In.Labels = { TEXT("a") };
			In.Params = MakeParams(EPipelineScoreActivation::Sigmoid, 0.5f, 0.3f);
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("one survivor"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("survivor"), Out[0], TEXT("a"), 0, (float)SigmoidD(3.0), 0.5f, 0.3f, 1.0f, 0.7f, 1e-6);
			}
		});

		It("MaxDetections 1 keeps only the best", [this]()
		{
			FInput In;
			In.Q = 3; In.C = 1;
			In.Logits = { 1.f, 4.f, 2.f };
			In.Boxes = { 0.2f, 0.2f, 0.1f, 0.1f,   0.5f, 0.5f, 0.1f, 0.1f,   0.8f, 0.8f, 0.1f, 0.1f };
			In.Labels = { TEXT("a") };
			In.Params = MakeParams(EPipelineScoreActivation::Sigmoid, 0.5f, 0.7f, 1);
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("one detection"), Out.Num(), 1))
			{
				ExpectDet(*this, TEXT("best"), Out[0], TEXT("a"), 0, (float)SigmoidD(4.0), 0.45f, 0.45f, 0.55f, 0.55f, 1e-6);
			}
		});

		It("Threshold 1 and above keep nothing and succeed", [this]()
		{
			// Scores are < 1 for logit 20 in double (0.999999998), as float 1.0f; "> 1" is never true. Also 2.0 clamps to 1.
			TArray<FPipelineDetection> Out;
			FPipelineResult R = Decode(OneQuerySigmoid({ 20.f }, 1.0f), Out);
			TestTrue(TEXT("threshold 1 succeeds"), R.bSuccess);
			TestEqual(TEXT("threshold 1 empty"), Out.Num(), 0);
			R = Decode(OneQuerySigmoid({ 20.f }, 2.0f), Out);
			TestTrue(TEXT("threshold 2 succeeds (clamped)"), R.bSuccess);
			TestEqual(TEXT("threshold 2 empty"), Out.Num(), 0);
		});

		It("Threshold -0.5 clamps to 0: every (q,c) is a candidate, then top-Q", [this]()
		{
			// Q=2, C=3, all scores > 0. Six candidates, top-2 by score: q1c0 (logit -0.5) then q0c0 (logit -1). Boxes are disjoint.
			FInput In;
			In.Q = 2; In.C = 3;
			In.Logits = { -1.f, -2.f, -3.f,   -0.5f, -4.f, -5.f };
			In.Boxes = { 0.25f, 0.25f, 0.2f, 0.2f,   0.75f, 0.75f, 0.2f, 0.2f };
			In.Labels = { TEXT("a"), TEXT("b"), TEXT("c") };
			In.Params = MakeParams(EPipelineScoreActivation::Sigmoid, -0.5f);
			TArray<FPipelineDetection> Out;
			const FPipelineResult R = Decode(In, Out);
			TestTrue(TEXT("decode succeeds"), R.bSuccess);
			if (TestEqual(TEXT("top-Q = 2 detections"), Out.Num(), 2))
			{
				ExpectDet(*this, TEXT("first"), Out[0], TEXT("a"), 0, (float)SigmoidD(-0.5), 0.65f, 0.65f, 0.85f, 0.85f, 1e-6);
				ExpectDet(*this, TEXT("second"), Out[1], TEXT("a"), 0, (float)SigmoidD(-1.0), 0.15f, 0.15f, 0.35f, 0.35f, 1e-6);
			}
		});

		It("Zero queries above threshold gives an empty Ok result", [this]()
		{
			TArray<FPipelineDetection> Out;
			FPipelineDetection Stale;
			Out.Add(Stale);
			const FPipelineResult R = Decode(OneQuerySigmoid({ -5.f, -6.f }, 0.5f), Out);
			TestTrue(TEXT("succeeds"), R.bSuccess);
			TestEqual(TEXT("empty, stale content cleared"), Out.Num(), 0);
		});
	});

	Describe("Errors", [this]()
	{
		It("NumQueries < 1 is InferenceFailed", [this]()
		{
			FInput In = OneQuerySigmoid({ 0.f }, 0.5f);
			In.Q = 0;
			In.Logits.Reset();
			In.Boxes.Reset();
			ExpectFail(*this, TEXT("Q = 0"), In, EPipelineError::InferenceFailed);
			In.Q = -3;
			ExpectFail(*this, TEXT("Q = -3"), In, EPipelineError::InferenceFailed);
		});

		It("NumLogitClasses < 1 is InferenceFailed", [this]()
		{
			FInput In = OneQuerySigmoid({ 0.f }, 0.5f);
			In.C = 0;
			In.Logits.Reset();
			In.Labels.Reset();
			ExpectFail(*this, TEXT("C = 0"), In, EPipelineError::InferenceFailed);
		});

		It("Logits length != Q*C' is InferenceFailed", [this]()
		{
			FInput In = OneQuerySigmoid({ 0.f, 1.f }, 0.5f);
			In.Logits.Add(2.f);   // 3 values for 1 x 2
			ExpectFail(*this, TEXT("too many logits"), In, EPipelineError::InferenceFailed);
			In.Logits = { 0.f };
			ExpectFail(*this, TEXT("too few logits"), In, EPipelineError::InferenceFailed);
		});

		It("Boxes length != Q*4 is InferenceFailed", [this]()
		{
			FInput In = OneQuerySigmoid({ 0.f }, 0.5f);
			In.Boxes = { 0.5f, 0.5f, 0.2f };
			ExpectFail(*this, TEXT("3 box values"), In, EPipelineError::InferenceFailed);
			In.Boxes = { 0.5f, 0.5f, 0.2f, 0.2f, 0.1f };
			ExpectFail(*this, TEXT("5 box values"), In, EPipelineError::InferenceFailed);
		});

		It("Sigmoid label count must equal C'", [this]()
		{
			FInput In = OneQuerySigmoid({ 0.f, 0.f, 0.f }, 0.5f);
			In.Labels.Pop();   // 2 labels for 3 columns
			ExpectFail(*this, TEXT("Sigmoid, one label short"), In, EPipelineError::InferenceFailed);
			In.Labels.Add(TEXT("x"));
			In.Labels.Add(TEXT("y"));   // 4 labels for 3 columns
			ExpectFail(*this, TEXT("Sigmoid, one label extra"), In, EPipelineError::InferenceFailed);
		});

		It("Softmax label count must equal C' - 1", [this]()
		{
			FInput In;
			In.Q = 1; In.C = 3;
			In.Logits = { 0.f, 0.f, 0.f };
			In.Boxes = { 0.5f, 0.5f, 0.2f, 0.2f };
			In.Params = MakeParams(EPipelineScoreActivation::Softmax, 0.1f);
			In.Labels = { TEXT("a"), TEXT("b"), TEXT("c") };   // Sigmoid-style count: C' labels
			ExpectFail(*this, TEXT("Softmax, C' labels"), In, EPipelineError::InferenceFailed);
			In.Labels = { TEXT("a") };
			ExpectFail(*this, TEXT("Softmax, C' - 2 labels"), In, EPipelineError::InferenceFailed);
		});

		It("Non-finite logits or boxes are InferenceFailed", [this]()
		{
			FInput In = OneQuerySigmoid({ 0.f, 1.f, 2.f }, 0.5f);
			In.Logits[1] = GQuietNaN;
			ExpectFail(*this, TEXT("NaN in one logit"), In, EPipelineError::InferenceFailed);
			In.Logits[1] = GInf;
			ExpectFail(*this, TEXT("+inf logit"), In, EPipelineError::InferenceFailed);
			In.Logits[1] = -GInf;
			ExpectFail(*this, TEXT("-inf logit"), In, EPipelineError::InferenceFailed);
			In.Logits[1] = 1.f;
			In.Boxes[2] = GQuietNaN;
			ExpectFail(*this, TEXT("NaN box value"), In, EPipelineError::InferenceFailed);
			In.Boxes[2] = GInf;
			ExpectFail(*this, TEXT("inf box value"), In, EPipelineError::InferenceFailed);
		});

		It("NaN in a logit of the last query (not just the first) is found", [this]()
		{
			// A decoder that validates only while scoring candidates, or only the first row, would skip this value.
			FInput In;
			In.Q = 3; In.C = 2;
			In.Logits = { -9.f, -9.f,  -9.f, -9.f,  -9.f, GQuietNaN };
			In.Boxes = { 0.2f, 0.2f, 0.1f, 0.1f,  0.5f, 0.5f, 0.1f, 0.1f,  0.8f, 0.8f, 0.1f, 0.1f };
			In.Labels = { TEXT("a"), TEXT("b") };
			In.Params = MakeParams(EPipelineScoreActivation::Sigmoid, 0.5f);
			ExpectFail(*this, TEXT("NaN in the last logit, nothing above threshold"), In, EPipelineError::InferenceFailed);
		});

		It("Threshold NaN is InvalidArgument", [this]()
		{
			ExpectFail(*this, TEXT("NaN threshold"), OneQuerySigmoid({ 0.f }, GQuietNaN), EPipelineError::InvalidArgument);
		});

		It("MaxDetections < 1 is InvalidArgument", [this]()
		{
			FInput In = OneQuerySigmoid({ 0.f }, 0.1f);
			In.Params.MaxDetections = 0;
			ExpectFail(*this, TEXT("MaxDetections 0"), In, EPipelineError::InvalidArgument);
			In.Params.MaxDetections = -5;
			ExpectFail(*this, TEXT("MaxDetections -5"), In, EPipelineError::InvalidArgument);
		});

		It("Partial geometry is InvalidArgument; all-zero is identity", [this]()
		{
			// All four of CropWidth/CropHeight/ResizedWidth/ResizedHeight > 0 maps; all 0 is identity; any mix is an error.
			FInput In = OneQuerySigmoid({ 3.f }, 0.5f);
			In.Params.Geometry.CropWidth = 224;
			ExpectFail(*this, TEXT("only CropWidth"), In, EPipelineError::InvalidArgument);
			In.Params.Geometry.CropHeight = 224;
			In.Params.Geometry.ResizedWidth = 341;
			ExpectFail(*this, TEXT("ResizedHeight missing"), In, EPipelineError::InvalidArgument);
			In.Params.Geometry.ResizedHeight = 256;
			In.Params.Geometry.CropWidth = 0;
			ExpectFail(*this, TEXT("CropWidth missing"), In, EPipelineError::InvalidArgument);
			In.Params.Geometry = FPipelineImageGeometry();
			In.Params.Geometry.ResizedWidth = 341;
			In.Params.Geometry.ResizedHeight = 256;
			ExpectFail(*this, TEXT("only the resized size"), In, EPipelineError::InvalidArgument);
			In.Params.Geometry = FPipelineImageGeometry();
			TArray<FPipelineDetection> Out;
			TestTrue(TEXT("all-zero geometry succeeds"), Decode(In, Out).bSuccess);
		});

		It("Negative geometry sizes are InvalidArgument", [this]()
		{
			FInput In = OneQuerySigmoid({ 3.f }, 0.5f);
			In.Params.Geometry.CropWidth = -224;
			In.Params.Geometry.CropHeight = 224;
			In.Params.Geometry.ResizedWidth = 341;
			In.Params.Geometry.ResizedHeight = 256;
			ExpectFail(*this, TEXT("negative CropWidth"), In, EPipelineError::InvalidArgument);
		});
	});

	Describe("Robustness", [this]()
	{
		It("Random 300 x 80 RT-DETR-sized input decodes in under 20 ms", [this]()
		{
			FRandomStream Rng(4818);
			FInput In;
			In.Q = 300; In.C = 80;
			In.Logits.SetNumUninitialized(300 * 80);
			for (float& L : In.Logits) { L = Rng.FRandRange(-8.f, 4.f); }
			In.Boxes.SetNumUninitialized(300 * 4);
			for (int32 q = 0; q < 300; ++q)
			{
				In.Boxes[q * 4 + 0] = Rng.FRandRange(0.1f, 0.9f);
				In.Boxes[q * 4 + 1] = Rng.FRandRange(0.1f, 0.9f);
				In.Boxes[q * 4 + 2] = Rng.FRandRange(0.02f, 0.4f);
				In.Boxes[q * 4 + 3] = Rng.FRandRange(0.02f, 0.4f);
			}
			for (int32 c = 0; c < 80; ++c) { In.Labels.Add(FString::Printf(TEXT("class%d"), c)); }
			In.Params = MakeParams(EPipelineScoreActivation::Sigmoid, 0.3f, 0.7f, 100);
			TArray<FPipelineDetection> Out;
			double Best = 1e9;
			for (int32 Run = 0; Run < 3; ++Run)   // best of 3: a cold first call or a scheduler hiccup is not the decoder's cost
			{
				const double T0 = FPlatformTime::Seconds();
				const FPipelineResult R = Decode(In, Out);
				const double Ms = (FPlatformTime::Seconds() - T0) * 1000.0;
				if (!TestTrue(TEXT("decode succeeds"), R.bSuccess)) { return; }
				Best = FMath::Min(Best, Ms);
			}
			AddInfo(FString::Printf(TEXT("300 x 80 decode: best of 3 = %.3f ms, %d detections."), Best, Out.Num()));
			TestTrue(*FString::Printf(TEXT("decode took %.3f ms, limit 20 ms"), Best), Best < 20.0);
			TestTrue(TEXT("at most MaxDetections"), Out.Num() <= 100);
			for (const FPipelineDetection& D : Out)
			{
				if (D.Box.Min.X < 0.f || D.Box.Min.Y < 0.f || D.Box.Max.X > 1.f || D.Box.Max.Y > 1.f)
				{
					AddError(TEXT("an output box lies outside [0,1]."));
					break;
				}
			}
			for (int32 i = 1; i < Out.Num(); ++i)
			{
				if (Out[i].Score > Out[i - 1].Score)
				{
					AddError(FString::Printf(TEXT("output not in descending score order at %d: %g after %g."), i, Out[i].Score, Out[i - 1].Score));
					break;
				}
			}
		});

		It("Eight threads decoding the same input get identical results", [this]()
		{
			// Pure function, any thread: shared static state (a scratch buffer) would break this.
			// Shared with the threads by value: a timed-out wait returns while threads may still read them.
			struct FShared { FInput In; TArray<FPipelineDetection> Reference; };
			const TSharedRef<FShared, ESPMode::ThreadSafe> Shared = MakeShared<FShared, ESPMode::ThreadSafe>();
			FInput& In = Shared->In;
			In.Q = 40; In.C = 6;
			FRandomStream Rng(77);
			for (int32 i = 0; i < In.Q * In.C; ++i) { In.Logits.Add(Rng.FRandRange(-4.f, 3.f)); }
			for (int32 q = 0; q < In.Q; ++q)
			{
				In.Boxes.Add(Rng.FRandRange(0.2f, 0.8f)); In.Boxes.Add(Rng.FRandRange(0.2f, 0.8f));
				In.Boxes.Add(Rng.FRandRange(0.05f, 0.3f)); In.Boxes.Add(Rng.FRandRange(0.05f, 0.3f));
			}
			for (int32 c = 0; c < In.C; ++c) { In.Labels.Add(FString::Printf(TEXT("c%d"), c)); }
			In.Params = MakeParams(EPipelineScoreActivation::Sigmoid, 0.2f, 0.5f, 50);
			TArray<FPipelineDetection>& Reference = Shared->Reference;
			if (!TestTrue(TEXT("reference decode succeeds"), Decode(In, Reference).bSuccess)) { return; }

			TArray<TFuture<bool>> Futures;
			for (int32 t = 0; t < 8; ++t)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [Shared]()
				{
					const FInput& In = Shared->In;
					const TArray<FPipelineDetection>& Reference = Shared->Reference;
					for (int32 Iter = 0; Iter < 50; ++Iter)
					{
						TArray<FPipelineDetection> Mine;
						const FPipelineResult R = Decode(In, Mine);
						if (!R.bSuccess || Mine.Num() != Reference.Num()) { return false; }
						for (int32 i = 0; i < Mine.Num(); ++i)
						{
							if (Mine[i].Index != Reference[i].Index || Mine[i].Score != Reference[i].Score
								|| Mine[i].Box.Min != Reference[i].Box.Min || Mine[i].Box.Max != Reference[i].Box.Max)
							{
								return false;
							}
						}
					}
					return true;
				}));
			}
			for (int32 t = 0; t < 8; ++t)
			{
				// Bounded wait: fail rather than hang.
				if (!Futures[t].WaitFor(FTimespan::FromSeconds(30)))
				{
					AddError(FString::Printf(TEXT("thread %d did not finish within 30 s."), t));
					return;
				}
				TestTrue(*FString::Printf(TEXT("thread %d produced the reference result"), t), Futures[t].Get());
			}
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
