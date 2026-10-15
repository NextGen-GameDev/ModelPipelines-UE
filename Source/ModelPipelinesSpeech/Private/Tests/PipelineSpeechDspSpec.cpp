// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// WP9S.4 / contract C73: speech DSP front end (mono mix, resampler, RMS, log-mel). Numbers are checked against the Python goldens
// (<dev>/goldens/stt, HF ParakeetFeatureExtractor + librosa) or against analytic signals generated here, never against the
// implementation's own output. Missing goldens FAIL these tests (they never skip) with the path and the generator command.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Async/Future.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HAL/ThreadSafeCounter.h"
#include "Math/RandomStream.h"
#include "PipelineModelPackage.h"
#include "PipelineSpeechDsp.h"
#include "SpeechTestData.h"

#include <limits>

using namespace UE::ModelPipelines::Speech;
using namespace UE::ModelPipelines::SpeechTests;

namespace PipelineSpeechDspSpecPrivate
{
	constexpr double TwoPi = 6.283185307179586476925286766559;

	static bool GetGoldens(FAutomationTestBase& T, const FSttGoldens*& OutG)
	{
		FString Err;
		if (!LoadSttGoldens(OutG, Err))
		{
			T.AddError(Err);
			return false;
		}
		return true;
	}

	static bool LoadAudio(FAutomationTestBase& T, const FSttGoldens& G, const FSttClip& C, TArray<float>& Out)
	{
		FString Err;
		if (!LoadFloatFile(G, C.AudioFile, C.NumSamples, Out, Err))
		{
			T.AddError(Err);
			return false;
		}
		return true;
	}

	static TArray<float> MakeSine(double Freq, int32 Rate, int32 N, double Amp)
	{
		TArray<float> S;
		S.SetNumUninitialized(N);
		for (int32 i = 0; i < N; ++i)
		{
			S[i] = (float)(Amp * FMath::Sin(TwoPi * Freq * (double)i / (double)Rate));
		}
		return S;
	}

	static TArray<float> MakeNoise(int32 Seed, int32 N, float Amp)
	{
		FRandomStream R(Seed);
		TArray<float> S;
		S.SetNumUninitialized(N);
		for (int32 i = 0; i < N; ++i)
		{
			S[i] = Amp * (R.FRand() * 2.f - 1.f);
		}
		return S;
	}

	static double RmsTrimmed(const TArray<float>& S, int32 Trim)
	{
		double Sum = 0.0;
		int32 Count = 0;
		for (int32 i = Trim; i < S.Num() - Trim; ++i)
		{
			Sum += (double)S[i] * (double)S[i];
			++Count;
		}
		return Count > 0 ? FMath::Sqrt(Sum / Count) : 0.0;
	}

	/** SNR (dB, computed in double) of Out against the analytic sine Amp*sin(2*pi*Freq*m/OutRate), trimming Trim samples at each end. */
	static double SineSnrDb(const TArray<float>& Out, double Freq, int32 OutRate, double Amp, int32 Trim)
	{
		double Sig = 0.0, Noise = 0.0;
		int32 Count = 0;
		for (int32 m = Trim; m < Out.Num() - Trim; ++m)
		{
			const double Ref = Amp * FMath::Sin(TwoPi * Freq * (double)m / (double)OutRate);
			const double D = (double)Out[m] - Ref;
			Sig += Ref * Ref;
			Noise += D * D;
			++Count;
		}
		if (Count == 0) { return -1000.0; }
		if (Noise <= 0.0) { return 300.0; }
		return 10.0 * FMath::LogX(10.0, Sig / Noise);
	}

	static int64 ExpectedResampleLength(int64 N, int32 InRate, int32 OutRate)
	{
		return (int64)FMath::FloorToDouble((double)N * (double)OutRate / (double)InRate + 0.5);
	}

	/** Resamples Amp*sin(Freq) and returns the SNR against the analytic output sine; checks the length rule. Returns -1000 on failure. */
	static double ResampleSineSnr(FAutomationTestBase& T, int32 InRate, int32 OutRate, double Freq, int32 NumIn)
	{
		const TArray<float> In = MakeSine(Freq, InRate, NumIn, 0.5);
		TArray<float> Out;
		Out.Add(123.f); // junk that must be replaced
		FString Err;
		if (!Resample(In, InRate, OutRate, Out, Err))
		{
			T.AddError(FString::Printf(TEXT("Resample %d->%d failed: %s"), InRate, OutRate, *Err));
			return -1000.0;
		}
		const int64 Want = ExpectedResampleLength(NumIn, InRate, OutRate);
		T.TestEqual(*FString::Printf(TEXT("Resample %d->%d of %d samples: output length rule round(N*Out/In)"), InRate, OutRate, NumIn), (int64)Out.Num(), Want);
		return SineSnrDb(Out, Freq, OutRate, 0.5, 64);
	}

	static bool BitEqual(const TArray<float>& A, const TArray<float>& B)
	{
		return A.Num() == B.Num() && (A.Num() == 0 || FMemory::Memcmp(A.GetData(), B.GetData(), A.Num() * sizeof(float)) == 0);
	}

	struct FDiff { double MaxAbs = 0.0; double SumAbs = 0.0; int64 Count = 0; double MaxRel = 0.0; };

	static void Accumulate(FDiff& D, const TArray<float>& Got, const TArray<float>& Want)
	{
		const int32 N = FMath::Min(Got.Num(), Want.Num());
		for (int32 i = 0; i < N; ++i)
		{
			const double A = FMath::Abs((double)Got[i] - (double)Want[i]);
			D.MaxAbs = FMath::Max(D.MaxAbs, A);
			D.SumAbs += A;
			++D.Count;
			if (FMath::Abs((double)Want[i]) > 1e-6)
			{
				D.MaxRel = FMath::Max(D.MaxRel, A / FMath::Abs((double)Want[i]));
			}
		}
	}

	static FPipelineSpeechSettings DefaultSettings() { return FPipelineSpeechSettings(); }
}

BEGIN_DEFINE_SPEC(FPipelineSpeechDspSpec, "Pipelines.Speech.Dsp",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
END_DEFINE_SPEC(FPipelineSpeechDspSpec)

void FPipelineSpeechDspSpec::Define()
{
	using namespace PipelineSpeechDspSpecPrivate;

	// ---- Acceptance 1 ----------------------------------------------------------------------------------------------------
	Describe("MelFilters", [this]()
	{
		It("matches mel_filters.f32 (abs <= 1e-6, rel <= 1e-5 on entries > 1e-6)", [this]()
		{
			const FSttGoldens* G = nullptr;
			if (!GetGoldens(*this, G)) { return; }
			TestEqual(TEXT("golden mel shape rows"), G->MelRows, 128);
			TestEqual(TEXT("golden mel shape cols"), G->MelCols, 257);
			TArray<float> Want;
			FString Err;
			if (!LoadFloatFile(*G, G->MelFile, (int64)G->MelRows * G->MelCols, Want, Err)) { AddError(Err); return; }

			const FLogMelExtractor Ex(DefaultSettings());
			TestTrue(TEXT("default settings are valid"), Ex.IsValid());
			const TArray<float>& Got = Ex.GetMelFilters();
			if (!TestEqual(TEXT("filter matrix element count (128*257)"), Got.Num(), Want.Num())) { return; }
			FDiff D;
			Accumulate(D, Got, Want);
			AddInfo(FString::Printf(TEXT("mel filters: max abs %.3e, max rel %.3e"), D.MaxAbs, D.MaxRel));
			// A filterbank built with htk mel, no slaney norm, or float32 accumulation errors would exceed these.
			TestTrue(*FString::Printf(TEXT("mel filters max abs %.3e <= 1e-6"), D.MaxAbs), D.MaxAbs <= 1e-6);
			TestTrue(*FString::Printf(TEXT("mel filters max rel %.3e <= 1e-5"), D.MaxRel), D.MaxRel <= 1e-5);
		});
	});

	// ---- Acceptance 2 ----------------------------------------------------------------------------------------------------
	Describe("LogMel", [this]()
	{
		It("ExtractLogMel matches logmel/* for every clip that has one (abs <= 5e-4), frames = N/160+1", [this]()
		{
			const FSttGoldens* G = nullptr;
			if (!GetGoldens(*this, G)) { return; }
			const FLogMelExtractor Ex(DefaultSettings());
			int32 Checked = 0;
			double Worst = 0.0;
			FString WorstId;
			for (const FSttClip& C : G->Clips)
			{
				if (!C.bHasLogmel) { continue; }
				TArray<float> Audio, Want;
				FString Err;
				if (!LoadAudio(*this, *G, C, Audio)) { continue; }
				if (!LoadFloatFile(*G, C.LogmelFile, (int64)C.LogmelFrames * 128, Want, Err)) { AddError(Err); continue; }
				FSpeechFeatures F;
				if (!Ex.ExtractLogMel(Audio, F, Err))
				{
					AddError(FString::Printf(TEXT("ExtractLogMel failed on '%s': %s"), *C.Id, *Err));
					continue;
				}
				++Checked;
				TestEqual(*FString::Printf(TEXT("'%s' log-mel frames = N/160+1"), *C.Id), F.NumFrames, C.NumSamples / 160 + 1);
				TestEqual(*FString::Printf(TEXT("'%s' log-mel frames match golden"), *C.Id), F.NumFrames, C.LogmelFrames);
				TestEqual(*FString::Printf(TEXT("'%s' NumMelBins"), *C.Id), F.NumMelBins, 128);
				if (F.Data.Num() != Want.Num())
				{
					AddError(FString::Printf(TEXT("'%s' log-mel has %d values, golden %d"), *C.Id, F.Data.Num(), Want.Num()));
					continue;
				}
				FDiff D;
				Accumulate(D, F.Data, Want);
				if (D.MaxAbs > Worst) { Worst = D.MaxAbs; WorstId = C.Id; }
				if (D.MaxAbs > 5e-4)
				{
					AddError(FString::Printf(TEXT("'%s' log-mel max abs diff %.3e > 5e-4"), *C.Id, D.MaxAbs));
				}
			}
			AddInfo(FString::Printf(TEXT("log-mel: %d clips, worst max abs %.3e (%s)"), Checked, Worst, *WorstId));
			TestTrue(TEXT("at least 4 clips with log-mel goldens were checked (3 speech + synthetic)"), Checked >= 4);
		});

		It("ExtractLogMel on a short signal gives N/160+1 frames (shape only)", [this]()
		{
			const FLogMelExtractor Ex(DefaultSettings());
			FSpeechFeatures F;
			FString Err;
			const TArray<float> S = MakeNoise(7, 1600, 0.1f);
			if (!TestTrue(*FString::Printf(TEXT("ExtractLogMel(1600 samples): %s"), *Err), Ex.ExtractLogMel(S, F, Err))) { return; }
			TestEqual(TEXT("frames for 1600 samples"), F.NumFrames, 11);
			TestEqual(TEXT("mel bins"), F.NumMelBins, 128);
			TestEqual(TEXT("data size"), F.Data.Num(), 11 * 128);
		});
	});

	// ---- Acceptance 3 ----------------------------------------------------------------------------------------------------
	Describe("Features", [this]()
	{
		It("Extract matches features/* for all clips with features (max abs <= 2e-3, mean abs <= 1e-4)", [this]()
		{
			const FSttGoldens* G = nullptr;
			if (!GetGoldens(*this, G)) { return; }
			const FLogMelExtractor Ex(DefaultSettings());
			int32 Checked = 0, Speech = 0;
			double WorstMax = 0.0, WorstMean = 0.0;
			FString WorstMaxId, WorstMeanId;
			for (const FSttClip& C : G->Clips)
			{
				if (!C.bHasFeatures) { continue; }
				// Lead (C73 amendment 2026-10-15): digital silence has zero variance in every bin; HF's value there is float
				// rounding / 1e-5 (-0.1599). Our contract normalises constant bins to 0 — checked in "constant input" below.
				if (C.Id == TEXT("silence_0p5s")) { continue; }
				TArray<float> Audio, Want;
				FString Err;
				if (!LoadAudio(*this, *G, C, Audio)) { continue; }
				if (!LoadFloatFile(*G, C.FeaturesFile, (int64)C.FeatureFrames * 128, Want, Err)) { AddError(Err); continue; }
				FSpeechFeatures F;
				if (!Ex.Extract(Audio, F, Err))
				{
					AddError(FString::Printf(TEXT("Extract failed on '%s': %s"), *C.Id, *Err));
					continue;
				}
				++Checked;
				if (!C.bSynthetic) { ++Speech; }
				TestEqual(*FString::Printf(TEXT("'%s' NumFrames = num_samples/160"), *C.Id), F.NumFrames, C.NumSamples / 160);
				TestEqual(*FString::Printf(TEXT("'%s' NumFrames matches golden"), *C.Id), F.NumFrames, C.FeatureFrames);
				TestEqual(*FString::Printf(TEXT("'%s' NumMelBins"), *C.Id), F.NumMelBins, 128);
				if (F.Data.Num() != Want.Num())
				{
					AddError(FString::Printf(TEXT("'%s' features have %d values, golden %d"), *C.Id, F.Data.Num(), Want.Num()));
					continue;
				}
				FDiff D;
				Accumulate(D, F.Data, Want);
				const double Mean = D.Count > 0 ? D.SumAbs / (double)D.Count : 0.0;
				if (D.MaxAbs > WorstMax) { WorstMax = D.MaxAbs; WorstMaxId = C.Id; }
				if (Mean > WorstMean) { WorstMean = Mean; WorstMeanId = C.Id; }
				if (D.MaxAbs > 2e-3) { AddError(FString::Printf(TEXT("'%s' features max abs diff %.3e > 2e-3"), *C.Id, D.MaxAbs)); }
				if (Mean > 1e-4) { AddError(FString::Printf(TEXT("'%s' features mean abs diff %.3e > 1e-4"), *C.Id, Mean)); }
			}
			AddInfo(FString::Printf(TEXT("features: %d clips (%d speech); worst max abs %.3e (%s); worst mean abs %.3e (%s)"),
				Checked, Speech, WorstMax, *WorstMaxId, WorstMean, *WorstMeanId));
			TestTrue(*FString::Printf(TEXT("expected 40 speech clips with features, checked %d"), Speech), Speech >= 40);
		});

		It("constant input (digital silence, DC) normalises to exactly 0 in every bin (C73 amendment)", [this]()
		{
			const FLogMelExtractor Ex(DefaultSettings());
			for (const float Level : { 0.0f, 0.25f })
			{
				TArray<float> Audio;
				Audio.Init(Level, 8000);
				FSpeechFeatures F;
				FString Err;
				if (!TestTrue(*FString::Printf(TEXT("Extract on constant %.2f: %s"), Level, *Err), Ex.Extract(Audio, F, Err))) { continue; }
				TestEqual(TEXT("frames"), F.NumFrames, 50);
				float MaxAbs = 0.f;
				for (const float V : F.Data) { MaxAbs = FMath::Max(MaxAbs, FMath::Abs(V)); }
				// DC 0.25 after pre-emphasis is a step then 0.0075 constant: only the first frames differ, so only require finite
				// values there; for 0.0 every bin is constant -> exactly 0.
				if (Level == 0.0f) { TestTrue(*FString::Printf(TEXT("silence features are all exactly 0 (max abs %g)"), MaxAbs), MaxAbs == 0.0f); }   // lead: exact, not TestEqual's 1e-4
				for (const float V : F.Data) { if (!FMath::IsFinite(V)) { AddError(TEXT("non-finite feature")); break; } }
			}
		});

		It("short_319 -> false 'audio too short'; short_320 -> 2 frames", [this]()
		{
			const FSttGoldens* G = nullptr;
			if (!GetGoldens(*this, G)) { return; }
			const FLogMelExtractor Ex(DefaultSettings());

			const FSttClip* C319 = G->Find(TEXT("short_319"));
			if (!C319) { AddError(TEXT("golden clip short_319 missing from stt_goldens.json")); }
			else
			{
				TArray<float> Audio;
				if (LoadAudio(*this, *G, *C319, Audio))
				{
					TestEqual(TEXT("short_319 has 319 samples"), Audio.Num(), 319);
					FSpeechFeatures F;
					FString Err;
					TestFalse(TEXT("short_319 (1 valid frame) must fail"), Ex.Extract(Audio, F, Err));
					TestTrue(*FString::Printf(TEXT("error must say 'audio too short', got '%s'"), *Err), Err.Contains(TEXT("audio too short")));
				}
			}
			const FSttClip* C320 = G->Find(TEXT("short_320"));
			if (!C320) { AddError(TEXT("golden clip short_320 missing from stt_goldens.json")); }
			else
			{
				TArray<float> Audio;
				if (LoadAudio(*this, *G, *C320, Audio))
				{
					FSpeechFeatures F;
					FString Err;
					if (TestTrue(*FString::Printf(TEXT("short_320 must succeed: %s"), *Err), Ex.Extract(Audio, F, Err)))
					{
						TestEqual(TEXT("short_320 NumFrames"), F.NumFrames, 2);
						TestEqual(TEXT("short_320 data size"), F.Data.Num(), 2 * 128);
					}
				}
			}
		});

		It("too-short inputs without goldens: empty, 1, 319 samples fail; 320 succeeds; NumValidFrames = N/160", [this]()
		{
			const FPipelineSpeechSettings S = DefaultSettings();
			const FLogMelExtractor Ex(S);
			for (const int32 N : { 0, 1, 159, 319 })
			{
				FSpeechFeatures F;
				FString Err;
				const TArray<float> Sig = MakeNoise(3, N, 0.1f);
				TestFalse(*FString::Printf(TEXT("Extract of %d samples must fail"), N), Ex.Extract(Sig, F, Err));
				TestFalse(*FString::Printf(TEXT("error for %d samples must be non-empty"), N), Err.IsEmpty());
			}
			FSpeechFeatures F;
			FString Err;
			TestTrue(TEXT("Extract of 320 samples succeeds"), Ex.Extract(MakeNoise(3, 320, 0.1f), F, Err));
			TestEqual(TEXT("320 samples -> 2 frames"), F.NumFrames, 2);

			TestEqual(TEXT("NumValidFrames(0)"), NumValidFrames(0, S), 0);
			TestEqual(TEXT("NumValidFrames(-5)"), NumValidFrames(-5, S), 0);
			TestEqual(TEXT("NumValidFrames(159)"), NumValidFrames(159, S), 0);
			TestEqual(TEXT("NumValidFrames(160)"), NumValidFrames(160, S), 1);
			TestEqual(TEXT("NumValidFrames(319)"), NumValidFrames(319, S), 1);
			TestEqual(TEXT("NumValidFrames(320)"), NumValidFrames(320, S), 2);
			TestEqual(TEXT("NumValidFrames(16000)"), NumValidFrames(16000, S), 100);
		});

		It("Extract output is normalised per mel bin (mean ~0, unbiased variance ~1) on generated noise", [this]()
		{
			// Independent of the goldens: a missing mean subtraction, dividing by T instead of T-1 mattering less, or a
			// missing normalisation altogether would all break one of these.
			const FLogMelExtractor Ex(DefaultSettings());
			FSpeechFeatures F;
			FString Err;
			const TArray<float> S = MakeNoise(99, 32000, 0.2f);
			if (!TestTrue(*FString::Printf(TEXT("Extract noise: %s"), *Err), Ex.Extract(S, F, Err))) { return; }
			TestEqual(TEXT("frames"), F.NumFrames, 200);
			if (F.Data.Num() != F.NumFrames * F.NumMelBins) { AddError(TEXT("Data size != NumFrames*NumMelBins")); return; }
			double WorstMean = 0.0, WorstVarErr = 0.0;
			for (int32 b = 0; b < F.NumMelBins; ++b)
			{
				double Sum = 0.0, SumSq = 0.0;
				for (int32 t = 0; t < F.NumFrames; ++t)
				{
					const double V = F.Data[t * F.NumMelBins + b];
					Sum += V; SumSq += V * V;
				}
				const double Mean = Sum / F.NumFrames;
				const double Var = (SumSq - F.NumFrames * Mean * Mean) / (F.NumFrames - 1);
				WorstMean = FMath::Max(WorstMean, FMath::Abs(Mean));
				WorstVarErr = FMath::Max(WorstVarErr, FMath::Abs(Var - 1.0));
			}
			AddInfo(FString::Printf(TEXT("normalisation: worst |mean| %.3e, worst |var-1| %.3e"), WorstMean, WorstVarErr));
			TestTrue(*FString::Printf(TEXT("per-bin |mean| %.3e <= 1e-4"), WorstMean), WorstMean <= 1e-4);
			TestTrue(*FString::Printf(TEXT("per-bin |variance-1| %.3e <= 1e-3"), WorstVarErr), WorstVarErr <= 1e-3);
		});
	});

	// ---- Acceptance 4 ----------------------------------------------------------------------------------------------------
	Describe("Rms", [this]()
	{
		It("ComputeRmsDbfs matches rms_dbfs for every clip (within 0.01 dB)", [this]()
		{
			const FSttGoldens* G = nullptr;
			if (!GetGoldens(*this, G)) { return; }
			int32 Checked = 0;
			double Worst = 0.0;
			FString WorstId;
			for (const FSttClip& C : G->Clips)
			{
				if (!C.bHasRms) { continue; }
				TArray<float> Audio;
				if (!LoadAudio(*this, *G, C, Audio)) { continue; }
				const float Got = ComputeRmsDbfs(Audio);
				const double D = FMath::Abs((double)Got - C.RmsDbfs);
				if (D > Worst) { Worst = D; WorstId = C.Id; }
				if (D > 0.01) { AddError(FString::Printf(TEXT("'%s' rms %.4f dBFS vs golden %.4f (diff %.4f > 0.01)"), *C.Id, Got, C.RmsDbfs, D)); }
				++Checked;
			}
			AddInfo(FString::Printf(TEXT("rms: %d clips, worst diff %.5f dB (%s)"), Checked, Worst, *WorstId));
			TestTrue(TEXT("at least 40 clips had an rms_dbfs golden"), Checked >= 40);
		});

		It("silence, empty, constants, NaN/Inf handling", [this]()
		{
			TestEqual(TEXT("empty -> -200"), ComputeRmsDbfs(TConstArrayView<float>()), -200.f);
			TArray<float> Zeros;
			Zeros.SetNumZeroed(8000);
			TestEqual(TEXT("all zeros -> -200"), ComputeRmsDbfs(Zeros), -200.f);

			TArray<float> Ones;
			Ones.Init(1.f, 100);
			TestTrue(TEXT("constant 1.0 -> 0 dBFS"), FMath::Abs(ComputeRmsDbfs(Ones)) < 1e-4f);
			TArray<float> Tenth;
			Tenth.Init(0.1f, 100);
			TestTrue(TEXT("constant 0.1 -> -20 dBFS"), FMath::Abs(ComputeRmsDbfs(Tenth) + 20.f) < 1e-3f);
			TArray<float> Neg;
			Neg.Init(-0.1f, 100);
			TestTrue(TEXT("constant -0.1 -> -20 dBFS (sign irrelevant)"), FMath::Abs(ComputeRmsDbfs(Neg) + 20.f) < 1e-3f);

			// NaN/Inf count as 0 samples AND stay in the denominator: rms = sqrt((0.25+0.25)/4).
			const float NaN = std::numeric_limits<float>::quiet_NaN();
			const float Inf = std::numeric_limits<float>::infinity();
			const TArray<float> Mixed = { NaN, 0.5f, Inf, 0.5f };
			const double Want = 20.0 * FMath::LogX(10.0, FMath::Sqrt(0.5 / 4.0));
			const float Got = ComputeRmsDbfs(Mixed);
			TestTrue(*FString::Printf(TEXT("NaN/Inf as 0: got %.4f, want %.4f dBFS"), Got, Want), FMath::IsFinite(Got) && FMath::Abs((double)Got - Want) < 1e-3);
			const TArray<float> AllBad = { NaN, -Inf, Inf };
			TestEqual(TEXT("all non-finite -> -200"), ComputeRmsDbfs(AllBad), -200.f);
		});
	});

	// ---- Acceptance 5 ----------------------------------------------------------------------------------------------------
	Describe("MixToMono", [this]()
	{
		It("stereo L=1 R=-1 -> 0", [this]()
		{
			const TArray<float> In = { 1.f, -1.f, 1.f, -1.f, 1.f, -1.f };
			TArray<float> Out = { 9.f };
			FString Err;
			if (!TestTrue(*FString::Printf(TEXT("MixToMono stereo: %s"), *Err), MixToMono(In, 2, Out, Err))) { return; }
			TestEqual(TEXT("3 mono samples"), Out.Num(), 3);
			for (float V : Out) { TestEqual(TEXT("sample is 0"), V, 0.f); }
		});

		It("3 channels are averaged", [this]()
		{
			const TArray<float> In = { 1.f, 2.f, 3.f, 4.f, 5.f, 9.f };
			TArray<float> Out;
			FString Err;
			if (!TestTrue(*FString::Printf(TEXT("MixToMono 3ch: %s"), *Err), MixToMono(In, 3, Out, Err))) { return; }
			if (!TestEqual(TEXT("2 frames"), Out.Num(), 2)) { return; }
			TestTrue(TEXT("frame 0 = 2"), FMath::Abs(Out[0] - 2.f) < 1e-6f);
			TestTrue(TEXT("frame 1 = 6"), FMath::Abs(Out[1] - 6.f) < 1e-6f);
		});

		It("mono is a bit-exact copy", [this]()
		{
			const TArray<float> In = MakeNoise(5, 1000, 0.7f);
			TArray<float> Out = { 1.f, 2.f };
			FString Err;
			if (!TestTrue(TEXT("MixToMono mono"), MixToMono(In, 1, Out, Err))) { return; }
			TestTrue(TEXT("identical copy"), BitEqual(In, Out));
		});

		It("0 or negative channels, ragged length -> false with an error", [this]()
		{
			const TArray<float> In = { 1.f, 2.f, 3.f, 4.f, 5.f };
			for (const int32 Ch : { 0, -1, 2, 3 })
			{
				TArray<float> Out;
				FString Err;
				const bool bOk = MixToMono(In, Ch, Out, Err);
				// 5 samples: ragged for 2, 3 channels; invalid for 0, -1.
				TestFalse(*FString::Printf(TEXT("NumChannels %d with 5 samples must fail"), Ch), bOk);
				TestFalse(*FString::Printf(TEXT("NumChannels %d error must be non-empty"), Ch), Err.IsEmpty());
			}
		});
	});

	// ---- Acceptance 6 ----------------------------------------------------------------------------------------------------
	Describe("Resample", [this]()
	{
		It("44100 -> 16000 of a 1 kHz sine: length rule and SNR >= 60 dB", [this]()
		{
			const double Snr = ResampleSineSnr(*this, 44100, 16000, 1000.0, 44100);
			AddInfo(FString::Printf(TEXT("44100->16000 1 kHz SNR %.2f dB"), Snr));
			TestTrue(*FString::Printf(TEXT("SNR %.2f dB >= 60"), Snr), Snr >= 60.0);
			// A non-round length exercises the rounding rule (44117 * 16000/44100 = 16005.99...).
			const double Snr2 = ResampleSineSnr(*this, 44100, 16000, 1000.0, 44117);
			AddInfo(FString::Printf(TEXT("44100->16000 (44117 samples) SNR %.2f dB"), Snr2));
			TestTrue(*FString::Printf(TEXT("SNR %.2f dB >= 60"), Snr2), Snr2 >= 60.0);
		});

		It("48000 -> 16000 of a 1 kHz sine: length rule and SNR >= 60 dB", [this]()
		{
			const double Snr = ResampleSineSnr(*this, 48000, 16000, 1000.0, 48000);
			AddInfo(FString::Printf(TEXT("48000->16000 1 kHz SNR %.2f dB"), Snr));
			TestTrue(*FString::Printf(TEXT("SNR %.2f dB >= 60"), Snr), Snr >= 60.0);
			const double Snr2 = ResampleSineSnr(*this, 48000, 16000, 1000.0, 47999);
			AddInfo(FString::Printf(TEXT("48000->16000 (47999 samples) SNR %.2f dB"), Snr2));
			TestTrue(*FString::Printf(TEXT("SNR %.2f dB >= 60"), Snr2), Snr2 >= 60.0);
		});

		It("8000 -> 16000 (upsampling) of a 1 kHz sine: SNR >= 50 dB", [this]()
		{
			const double Snr = ResampleSineSnr(*this, 8000, 16000, 1000.0, 8000);
			AddInfo(FString::Printf(TEXT("8000->16000 1 kHz SNR %.2f dB"), Snr));
			TestTrue(*FString::Printf(TEXT("SNR %.2f dB >= 50"), Snr), Snr >= 50.0);
		});

		It("anti-aliasing: 7.5 kHz tone keeps >= 0.9 RMS, 9 kHz tone is attenuated >= 40 dB (48000 -> 16000)", [this]()
		{
			const double InRms = 0.5 / FMath::Sqrt(2.0);
			{
				const TArray<float> In = MakeSine(7500.0, 48000, 48000, 0.5);
				TArray<float> Out;
				FString Err;
				if (!TestTrue(*FString::Printf(TEXT("Resample 7.5k: %s"), *Err), Resample(In, 48000, 16000, Out, Err))) { return; }
				const double Ratio = RmsTrimmed(Out, 64) / InRms;
				AddInfo(FString::Printf(TEXT("7.5 kHz tone RMS ratio %.4f"), Ratio));
				// A resampler with a too-low cutoff (e.g. a plain box/linear decimator) would drop this below 0.9.
				TestTrue(*FString::Printf(TEXT("7.5 kHz RMS ratio %.4f >= 0.9"), Ratio), Ratio >= 0.9);
			}
			{
				const TArray<float> In = MakeSine(9000.0, 48000, 48000, 0.5);
				TArray<float> Out;
				FString Err;
				if (!TestTrue(*FString::Printf(TEXT("Resample 9k: %s"), *Err), Resample(In, 48000, 16000, Out, Err))) { return; }
				const double Rms = RmsTrimmed(Out, 64);
				const double AttDb = Rms > 0.0 ? 20.0 * FMath::LogX(10.0, Rms / InRms) : -300.0;
				AddInfo(FString::Printf(TEXT("9 kHz tone attenuation %.2f dB"), AttDb));
				// 9 kHz would alias to 7 kHz at 16 kHz output; no anti-alias filter leaves it near 0 dB.
				TestTrue(*FString::Printf(TEXT("9 kHz attenuation %.2f dB <= -40"), AttDb), AttDb <= -40.0);
			}
		});

		It("equal rates -> bit-exact copy", [this]()
		{
			const TArray<float> In = MakeNoise(11, 5000, 0.9f);
			TArray<float> Out = { 5.f };
			FString Err;
			if (!TestTrue(TEXT("Resample 16000->16000"), Resample(In, 16000, 16000, Out, Err))) { return; }
			TestTrue(TEXT("output is bit-identical to input"), BitEqual(In, Out));
			TArray<float> Out2;
			if (TestTrue(TEXT("Resample 44100->44100"), Resample(In, 44100, 44100, Out2, Err)))
			{
				TestTrue(TEXT("44100 -> 44100 also a bit-exact copy"), BitEqual(In, Out2));
			}
		});

		It("rates outside 1000..384000 -> false with an error", [this]()
		{
			const TArray<float> In = MakeSine(100.0, 16000, 1000, 0.5);
			struct FCase { int32 InRate; int32 OutRate; };
			const FCase Cases[] = { {999, 16000}, {384001, 16000}, {0, 16000}, {-48000, 16000}, {16000, 999}, {16000, 384001}, {16000, 0} };
			for (const FCase& C : Cases)
			{
				TArray<float> Out;
				FString Err;
				TestFalse(*FString::Printf(TEXT("Resample %d->%d must fail"), C.InRate, C.OutRate), Resample(In, C.InRate, C.OutRate, Out, Err));
				TestFalse(*FString::Printf(TEXT("Resample %d->%d error must be non-empty"), C.InRate, C.OutRate), Err.IsEmpty());
			}
			// The boundaries themselves are valid.
			TArray<float> Out;
			FString Err;
			TestTrue(TEXT("1000 Hz input is valid"), Resample(In, 1000, 16000, Out, Err));
			TestTrue(TEXT("384000 Hz input is valid"), Resample(In, 384000, 16000, Out, Err));
		});

		It("empty input -> true with empty output", [this]()
		{
			TArray<float> Out = { 1.f, 2.f, 3.f };
			FString Err;
			TestTrue(*FString::Printf(TEXT("Resample of empty input: %s"), *Err), Resample(TConstArrayView<float>(), 44100, 16000, Out, Err));
			TestEqual(TEXT("output empty"), Out.Num(), 0);
		});
	});

	// ---- Acceptance 7 ----------------------------------------------------------------------------------------------------
	Describe("Threading", [this]()
	{
		It("8 threads sharing one FLogMelExtractor give bit-identical results to serial", [this]()
		{
			const FSttGoldens* G = nullptr;
			if (!GetGoldens(*this, G)) { return; }
			TArray<TArray<float>> Audio;
			for (const FSttClip& C : G->Clips)
			{
				if (!C.bHasFeatures || Audio.Num() >= 8) { continue; }
				TArray<float> A;
				if (LoadAudio(*this, *G, C, A)) { Audio.Add(MoveTemp(A)); }
			}
			if (!TestEqual(TEXT("8 clips available for the threading test"), Audio.Num(), 8)) { return; }

			const FLogMelExtractor Ex(DefaultSettings());
			TArray<TArray<float>> Serial;
			Serial.SetNum(8);
			for (int32 i = 0; i < 8; ++i)
			{
				FSpeechFeatures F;
				FString Err;
				if (!Ex.Extract(Audio[i], F, Err)) { AddError(FString::Printf(TEXT("serial Extract %d failed: %s"), i, *Err)); return; }
				Serial[i] = MoveTemp(F.Data);
			}

			FThreadSafeCounter Arrived;
			TArray<int32> Failures;
			Failures.Init(0, 8);
			TArray<TFuture<void>> Futures;
			for (int32 i = 0; i < 8; ++i)
			{
				Futures.Add(Async(EAsyncExecution::Thread, [i, &Ex, &Audio, &Serial, &Arrived, &Failures]()
				{
					Arrived.Increment();
					const double Start = FPlatformTime::Seconds();
					while (Arrived.GetValue() < 8 && FPlatformTime::Seconds() - Start < 10.0) { FPlatformProcess::YieldThread(); }
					for (int32 Iter = 0; Iter < 3; ++Iter)
					{
						FSpeechFeatures F;
						FString Err;
						if (!Ex.Extract(Audio[i], F, Err) || !BitEqual(F.Data, Serial[i])) { ++Failures[i]; }
					}
				}));
			}
			for (int32 i = 0; i < 8; ++i)
			{
				if (!Futures[i].WaitFor(FTimespan::FromSeconds(120)))
				{
					// Lead (review): never return while workers still reference these stack locals; report, then wait.
					AddError(FString::Printf(TEXT("thread %d did not finish within 120 s (deadlock or hang)"), i));
					Futures[i].Wait();
				}
			}
			for (int32 i = 0; i < 8; ++i)
			{
				TestEqual(*FString::Printf(TEXT("thread %d: mismatching runs vs serial (of 3)"), i), Failures[i], 0);
			}
		});
	});

	// ---- Acceptance 8 ----------------------------------------------------------------------------------------------------
	Describe("InvalidSettings", [this]()
	{
		It("invalid settings -> IsValid false, Extract/ExtractLogMel false with a message, no crash", [this]()
		{
			const TArray<float> Sig = MakeNoise(1, 16000, 0.1f);

			struct FCase { const TCHAR* Name; FPipelineSpeechSettings S; };
			TArray<FCase> Cases;
			{ FPipelineSpeechSettings S; S.FftSize = 500; Cases.Add({ TEXT("FftSize 500 (not a power of two)"), S }); }
			{ FPipelineSpeechSettings S; S.HopLength = 0; Cases.Add({ TEXT("HopLength 0"), S }); }
			{ FPipelineSpeechSettings S; S.FftSize = 500; S.HopLength = 0; Cases.Add({ TEXT("FftSize 500 and HopLength 0"), S }); }
			{ FPipelineSpeechSettings S; S.FftSize = 0; Cases.Add({ TEXT("FftSize 0"), S }); }
			{ FPipelineSpeechSettings S; S.WindowLength = 1024; Cases.Add({ TEXT("WindowLength > FftSize"), S }); }
			{ FPipelineSpeechSettings S; S.NumMelBins = 0; Cases.Add({ TEXT("NumMelBins 0"), S }); }
			{ FPipelineSpeechSettings S; S.Preemphasis = 1.f; Cases.Add({ TEXT("Preemphasis 1.0"), S }); }
			{ FPipelineSpeechSettings S; S.Preemphasis = std::numeric_limits<float>::quiet_NaN(); Cases.Add({ TEXT("Preemphasis NaN"), S }); }

			for (const FCase& C : Cases)
			{
				const FLogMelExtractor Ex(C.S);
				TestFalse(*FString::Printf(TEXT("IsValid false for %s"), C.Name), Ex.IsValid());
				FSpeechFeatures F;
				FString Err;
				TestFalse(*FString::Printf(TEXT("Extract false for %s"), C.Name), Ex.Extract(Sig, F, Err));
				TestFalse(*FString::Printf(TEXT("Extract message non-empty for %s"), C.Name), Err.IsEmpty());
				FString Err2;
				TestFalse(*FString::Printf(TEXT("ExtractLogMel false for %s"), C.Name), Ex.ExtractLogMel(Sig, F, Err2));
				TestFalse(*FString::Printf(TEXT("ExtractLogMel message non-empty for %s"), C.Name), Err2.IsEmpty());
				(void)Ex.GetMelFilters(); // must not crash
			}
		});
	});

	// ---- Acceptance 9 (report only) --------------------------------------------------------------------------------------
	Describe("Performance", [this]()
	{
		It("reports Extract time for a 30 s clip (single thread; not gated)", [this]()
		{
			const FLogMelExtractor Ex(DefaultSettings());
			const TArray<float> S = MakeNoise(2024, 16000 * 30, 0.1f);
			double Best = 1e9;
			FSpeechFeatures F;
			FString Err;
			for (int32 Run = 0; Run < 3; ++Run)
			{
				const double T0 = FPlatformTime::Seconds();
				if (!Ex.Extract(S, F, Err)) { AddError(FString::Printf(TEXT("Extract of 30 s failed: %s"), *Err)); return; }
				Best = FMath::Min(Best, (FPlatformTime::Seconds() - T0) * 1000.0);
			}
			AddInfo(FString::Printf(TEXT("Extract of 30 s clip: %.1f ms (best of 3), %d frames"), Best, F.NumFrames));
			TestEqual(TEXT("30 s -> 3000 frames"), F.NumFrames, 3000);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
