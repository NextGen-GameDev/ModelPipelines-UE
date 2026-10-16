// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "PipelineModelPackage.h"

/**
 * Speech DSP front end (contract C73): mono mix, band-limited resampling, RMS loudness and the log-mel features of
 * Hugging Face's ParakeetFeatureExtractor. Pure maths: no UObjects, no inference, no engine DSP code (own FFT and
 * resampler so every engine version behaves the same). All functions are reentrant and may run on any thread.
 */
namespace UE::ModelPipelines::Speech
{
/** Normalised log-mel features of the VALID frames, row-major [NumFrames][NumMelBins]. */
struct MODELPIPELINESSPEECH_API FSpeechFeatures
{
	TArray<float> Data;
	int32 NumFrames = 0;
	int32 NumMelBins = 0;
};

/** Interleaved -> mono by averaging channels. NumChannels < 1, or Interleaved.Num() not a multiple of it -> false + error. */
MODELPIPELINESSPEECH_API bool MixToMono(TConstArrayView<float> Interleaved, int32 NumChannels, TArray<float>& OutMono, FString& OutError);

/**
 * Band-limited (windowed-sinc, Kaiser, 80 dB stopband) resampler. Zero-delay: output sample m is the band-limited input
 * evaluated at time m / OutRate. Equal rates -> bit-exact copy. Rates outside 1000..384000 -> false + error.
 * Output length = floor(In.Num() * OutRate / InRate + 0.5) (computed in double). Empty input -> true + empty output.
 */
MODELPIPELINESSPEECH_API bool Resample(TConstArrayView<float> In, int32 InRate, int32 OutRate, TArray<float>& Out, FString& OutError);

/** 20*log10(RMS); empty or all-zero -> -200.f. Non-finite samples are treated as 0 (but still counted). */
MODELPIPELINESSPEECH_API float ComputeRmsDbfs(TConstArrayView<float> Samples);

/** Number of valid feature frames for N samples: N / HopLength (integer division). N <= 0 (or HopLength <= 0) -> 0. */
MODELPIPELINESSPEECH_API int32 NumValidFrames(int32 NumSamples, const FPipelineSpeechSettings& Settings);

/** One long-audio window (C77): [Start, End) in samples of the 16 kHz mono signal. */
struct FSpeechWindow
{
	int32 Start = 0;
	int32 End = 0;
};

/**
 * C77 plan. SampleRate is the signal's rate (16000 in practice; MIN/MAX/TAIL scale with it, H = 160 fixed).
 * Empty input -> zero windows; N <= MAX -> one window [0, N). Otherwise cuts at the quietest 200 ms (float64 prefix sums of
 * x^2, earliest minimum) between start + 25 s and start + min(35 s, remaining - 5 s). Never fails.
 */
MODELPIPELINESSPEECH_API void PlanSpeechWindows(TConstArrayView<float> Mono, int32 SampleRate, TArray<FSpeechWindow>& OutWindows);

/**
 * ParakeetFeatureExtractor (batch 1, unpadded): pre-emphasis, centred STFT with a symmetric Hann window, power spectrum,
 * slaney mel filterbank (librosa), log(mel + 2^-24), per-bin mean/variance normalisation over the valid frames.
 * Settings are validated like Core does (C72c); in addition FftSize must be a power of two (<= 65536) and
 * NumMelBins <= 4096, otherwise IsValid() is false, Extract/ExtractLogMel fail with a message and the filters are empty.
 * Construction builds the window, FFT tables and filters once; all extraction methods are const and thread-safe.
 */
class MODELPIPELINESSPEECH_API FLogMelExtractor
{
public:
	explicit FLogMelExtractor(const FPipelineSpeechSettings& Settings);

	bool IsValid() const { return bValid; }

	/** Normalised features of the N/Hop valid frames. N/Hop < 2 -> false + "audio too short". */
	bool Extract(TConstArrayView<float> Samples, FSpeechFeatures& Out, FString& OutError) const;

	/** Unnormalised log-mel (steps 1-4, all N/Hop+1 frames) — tests and diagnostics. Empty input -> false. */
	bool ExtractLogMel(TConstArrayView<float> Samples, FSpeechFeatures& Out, FString& OutError) const;

	/** [NumMelBins][FftSize/2+1] row-major; empty when !IsValid(). */
	const TArray<float>& GetMelFilters() const { return MelFilters; }

private:
	/** Log-mel (double) of frames 0..NumFrames-1, row-major [NumFrames][NumMelBins]. */
	void ComputeLogMel(TConstArrayView<float> Samples, int32 NumFrames, TArray<double>& OutLogMel) const;
	void Fft(double* Re, double* Im) const;

	bool bValid = false;
	FString InvalidReason;
	int32 NumMelBins = 0;
	int32 FftSize = 0;
	int32 NumBins = 0;
	int32 WindowLength = 0;
	int32 WindowOffset = 0;
	int32 HopLength = 0;
	float Preemphasis = 0.f;

	TArray<float> MelFilters;
	TArray<int32> MelBinStart;   // first non-zero bin per mel band
	TArray<int32> MelBinEnd;     // one past the last non-zero bin per mel band
	TArray<double> Window;       // WindowLength symmetric Hann values
	TArray<int32> BitReverse;    // FftSize
	TArray<double> TwiddleCos;   // FftSize/2
	TArray<double> TwiddleSin;   // FftSize/2
};
}
