// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineSpeechDsp.h"

namespace UE::ModelPipelines::Speech
{
namespace DspPrivate
{
	constexpr double Pi = 3.14159265358979323846264338327950288;
	constexpr int32 MinRate = 1000;
	constexpr int32 MaxRate = 384000;
	constexpr int32 MaxFftSize = 65536;
	constexpr int32 MaxMelBins = 4096;
	constexpr double LogZeroGuard = 5.9604644775390625e-08; // 2^-24
	constexpr double NormEpsilon = 1e-5;

	// Kaiser resampler design: 80 dB stopband, transition = 10% of the lower Nyquist frequency.
	constexpr double StopbandDb = 80.0;
	constexpr double TransitionFraction = 0.1;
	constexpr int64 MaxPolyphaseTable = 4 * 1024 * 1024;

	static double BesselI0(double X)
	{
		double Sum = 1.0;
		double Term = 1.0;
		const double Q = X * X * 0.25;
		for (int32 K = 1; K < 500; ++K)
		{
			Term *= Q / ((double)K * (double)K);
			Sum += Term;
			if (Term < Sum * 1e-17)
			{
				break;
			}
		}
		return Sum;
	}

	struct FKaiserKernel
	{
		double TwoFc = 1.0;       // 2 * cutoff in cycles per input sample
		double Beta = 0.0;
		double InvI0Beta = 1.0;
		int32 HalfWidth = 1;      // in input samples

		double Eval(double X) const
		{
			const double AbsX = FMath::Abs(X);
			if (AbsX >= (double)HalfWidth)
			{
				return 0.0;
			}
			const double R = X / (double)HalfWidth;
			const double W = BesselI0(Beta * FMath::Sqrt(FMath::Max(0.0, 1.0 - R * R))) * InvI0Beta;
			const double U = TwoFc * X;
			const double Sinc = (U == 0.0) ? 1.0 : FMath::Sin(Pi * U) / (Pi * U);
			return TwoFc * Sinc * W;
		}
	};

	static FKaiserKernel DesignKernel(int32 InRate, int32 OutRate)
	{
		const double NyqMin = 0.5 * (double)FMath::Min(InRate, OutRate);
		const double TransitionHz = TransitionFraction * NyqMin;
		// Downsampling: transition centred on the output Nyquist (keeps the band up to 0.95 * Nyquist).
		// Upsampling: stopband starts at the input Nyquist so no spectral images leak through.
		const double CutoffHz = (OutRate < InRate) ? NyqMin : NyqMin - 0.5 * TransitionHz;
		FKaiserKernel K;
		K.TwoFc = 2.0 * CutoffHz / (double)InRate;
		K.Beta = 0.1102 * (StopbandDb - 8.7);
		K.InvI0Beta = 1.0 / BesselI0(K.Beta);
		const double DeltaOmega = 2.0 * Pi * TransitionHz / (double)InRate;
		const double NumTaps = (StopbandDb - 7.95) / (2.285 * DeltaOmega) + 1.0;
		K.HalfWidth = FMath::Max(1, (int32)FMath::CeilToDouble(0.5 * NumTaps));
		return K;
	}

	static int64 Gcd(int64 A, int64 B)
	{
		while (B != 0)
		{
			const int64 T = A % B;
			A = B;
			B = T;
		}
		return A;
	}

	// ---- librosa slaney mel scale (float64) ----
	constexpr double MelFSp = 200.0 / 3.0;
	constexpr double MinLogHz = 1000.0;

	static double LogStep() { return FMath::Loge(6.4) / 27.0; }
	static double MinLogMel() { return (MinLogHz - 0.0) / MelFSp; }

	static double HzToMel(double F)
	{
		double Mel = (F - 0.0) / MelFSp;
		if (F >= MinLogHz)
		{
			Mel = MinLogMel() + FMath::Loge(F / MinLogHz) / LogStep();
		}
		return Mel;
	}

	static double MelToHz(double Mel)
	{
		double F = 0.0 + MelFSp * Mel;
		if (Mel >= MinLogMel())
		{
			F = MinLogHz * FMath::Exp(LogStep() * (Mel - MinLogMel()));
		}
		return F;
	}

	static bool IsPowerOfTwo(int32 V) { return V > 0 && (V & (V - 1)) == 0; }

	static bool ValidateSettings(const FPipelineSpeechSettings& S, FString& OutReason)
	{
		if (S.SampleRate <= 0 || S.NumMelBins <= 0 || S.FftSize <= 0 || S.WindowLength <= 0 || S.HopLength <= 0
			|| S.SubsamplingFactor <= 0 || S.MaxSymbolsPerStep <= 0)
		{
			OutReason = TEXT("speech settings: SampleRate, NumMelBins, FftSize, WindowLength, HopLength, SubsamplingFactor and MaxSymbolsPerStep must be > 0");
			return false;
		}
		if (S.WindowLength > S.FftSize)
		{
			OutReason = FString::Printf(TEXT("speech settings: WindowLength %d exceeds FftSize %d"), S.WindowLength, S.FftSize);
			return false;
		}
		if (!FMath::IsFinite(S.Preemphasis) || S.Preemphasis < 0.f || S.Preemphasis >= 1.f)
		{
			OutReason = TEXT("speech settings: Preemphasis must be finite and in [0, 1)");
			return false;
		}
		if (S.BlankTokenId < 0)
		{
			OutReason = TEXT("speech settings: BlankTokenId must be >= 0");
			return false;
		}
		bool bHasAdvance = false;
		for (const int32 D : S.Durations)
		{
			if (D < 0)
			{
				OutReason = TEXT("speech settings: Durations must be >= 0");
				return false;
			}
			bHasAdvance |= (D >= 1);
		}
		if (!bHasAdvance)
		{
			OutReason = TEXT("speech settings: Durations must be non-empty and contain a value >= 1");
			return false;
		}
		if (!FMath::IsFinite(S.SilenceThresholdDb) || S.SilenceThresholdDb > 0.f)
		{
			OutReason = TEXT("speech settings: SilenceThresholdDb must be finite and <= 0");
			return false;
		}
		if (!IsPowerOfTwo(S.FftSize) || S.FftSize > MaxFftSize)
		{
			OutReason = FString::Printf(TEXT("speech settings: FftSize %d must be a power of two <= %d for the log-mel extractor"), S.FftSize, MaxFftSize);
			return false;
		}
		if (S.NumMelBins > MaxMelBins)
		{
			OutReason = FString::Printf(TEXT("speech settings: NumMelBins %d exceeds %d"), S.NumMelBins, MaxMelBins);
			return false;
		}
		return true;
	}
}

bool MixToMono(TConstArrayView<float> Interleaved, int32 NumChannels, TArray<float>& OutMono, FString& OutError)
{
	if (NumChannels < 1)
	{
		OutError = FString::Printf(TEXT("MixToMono: NumChannels must be >= 1 (got %d)"), NumChannels);
		return false;
	}
	if (Interleaved.Num() % NumChannels != 0)
	{
		OutError = FString::Printf(TEXT("MixToMono: %d samples is not a multiple of %d channels"), Interleaved.Num(), NumChannels);
		return false;
	}
	TArray<float> Result;
	if (NumChannels == 1)
	{
		Result = TArray<float>(Interleaved.GetData(), Interleaved.Num());
	}
	else
	{
		const int32 NumFrames = Interleaved.Num() / NumChannels;
		Result.SetNumUninitialized(NumFrames);
		const float* Src = Interleaved.GetData();
		for (int32 F = 0; F < NumFrames; ++F)
		{
			double Sum = 0.0;
			for (int32 C = 0; C < NumChannels; ++C)
			{
				Sum += (double)Src[(int64)F * NumChannels + C];
			}
			Result[F] = (float)(Sum / (double)NumChannels);
		}
	}
	OutMono = MoveTemp(Result);
	return true;
}

bool Resample(TConstArrayView<float> In, int32 InRate, int32 OutRate, TArray<float>& Out, FString& OutError)
{
	using namespace DspPrivate;
	if (InRate < MinRate || InRate > MaxRate || OutRate < MinRate || OutRate > MaxRate)
	{
		OutError = FString::Printf(TEXT("Resample: sample rates must be in %d..%d Hz (got %d -> %d)"), MinRate, MaxRate, InRate, OutRate);
		return false;
	}
	if (InRate == OutRate)
	{
		TArray<float> Copy(In.GetData(), In.Num());
		Out = MoveTemp(Copy);
		return true;
	}
	const int64 NumIn = In.Num();
	const int64 NumOut = (int64)FMath::FloorToDouble((double)NumIn * (double)OutRate / (double)InRate + 0.5);
	if (NumOut > (int64)MAX_int32)
	{
		OutError = TEXT("Resample: output would exceed 2^31 samples");
		return false;
	}
	TArray<float> Result;
	Result.SetNumUninitialized((int32)NumOut);
	if (NumOut == 0)
	{
		Out = MoveTemp(Result);
		return true;
	}

	const FKaiserKernel K = DesignKernel(InRate, OutRate);
	const int64 G = Gcd(InRate, OutRate);
	const int64 L = OutRate / G;   // phases
	const int64 M = InRate / G;    // input step per output in units of 1/L
	const int32 HW = K.HalfWidth;
	const int32 NumTaps = 2 * HW;
	const float* Src = In.GetData();

	// Taps j = -HW+1 .. HW around k0 = floor(t); x = frac - j.
	TArray<double> Table;
	const bool bUseTable = L * (int64)NumTaps <= MaxPolyphaseTable;
	if (bUseTable)
	{
		Table.SetNumUninitialized((int32)(L * NumTaps));
		for (int64 P = 0; P < L; ++P)
		{
			const double Frac = (double)P / (double)L;
			for (int32 T = 0; T < NumTaps; ++T)
			{
				const int32 J = T - HW + 1;
				Table[(int32)(P * NumTaps + T)] = K.Eval(Frac - (double)J);
			}
		}
	}

	for (int64 Mo = 0; Mo < NumOut; ++Mo)
	{
		const int64 Num = Mo * M;
		const int64 K0 = Num / L;
		const int64 P = Num - K0 * L;
		const int64 First = K0 - HW + 1;
		const int32 TBegin = (int32)FMath::Max<int64>(0, -First);
		const int32 TEnd = (int32)FMath::Min<int64>(NumTaps, NumIn - First);
		double Acc = 0.0;
		if (bUseTable)
		{
			const double* Row = Table.GetData() + P * NumTaps;
			for (int32 T = TBegin; T < TEnd; ++T)
			{
				Acc += Row[T] * (double)Src[First + T];
			}
		}
		else
		{
			const double Frac = (double)P / (double)L;
			for (int32 T = TBegin; T < TEnd; ++T)
			{
				Acc += K.Eval(Frac - (double)(T - HW + 1)) * (double)Src[First + T];
			}
		}
		Result[(int32)Mo] = (float)Acc;
	}
	Out = MoveTemp(Result);
	return true;
}

float ComputeRmsDbfs(TConstArrayView<float> Samples)
{
	if (Samples.Num() == 0)
	{
		return -200.f;
	}
	double Sum = 0.0;
	for (const float V : Samples)
	{
		if (FMath::IsFinite(V))
		{
			Sum += (double)V * (double)V;
		}
	}
	if (Sum <= 0.0)
	{
		return -200.f;
	}
	const double Rms = FMath::Sqrt(Sum / (double)Samples.Num());
	return (float)(20.0 * FMath::LogX(10.0, Rms));
}

int32 NumValidFrames(int32 NumSamples, const FPipelineSpeechSettings& Settings)
{
	if (NumSamples <= 0 || Settings.HopLength <= 0)
	{
		return 0;
	}
	return NumSamples / Settings.HopLength;
}

FLogMelExtractor::FLogMelExtractor(const FPipelineSpeechSettings& Settings)
{
	using namespace DspPrivate;
	bValid = ValidateSettings(Settings, InvalidReason);
	if (!bValid)
	{
		return;
	}
	NumMelBins = Settings.NumMelBins;
	FftSize = Settings.FftSize;
	NumBins = FftSize / 2 + 1;
	WindowLength = Settings.WindowLength;
	WindowOffset = (FftSize - WindowLength) / 2;
	HopLength = Settings.HopLength;
	Preemphasis = Settings.Preemphasis;

	// Symmetric Hann (torch.hann_window(periodic=False)).
	Window.SetNumUninitialized(WindowLength);
	for (int32 I = 0; I < WindowLength; ++I)
	{
		Window[I] = (WindowLength == 1) ? 1.0 : 0.5 - 0.5 * FMath::Cos(2.0 * Pi * (double)I / (double)(WindowLength - 1));
	}

	// FFT tables.
	int32 Bits = 0;
	while ((1 << Bits) < FftSize)
	{
		++Bits;
	}
	BitReverse.SetNumUninitialized(FftSize);
	for (int32 I = 0; I < FftSize; ++I)
	{
		int32 R = 0;
		for (int32 B = 0; B < Bits; ++B)
		{
			R |= ((I >> B) & 1) << (Bits - 1 - B);
		}
		BitReverse[I] = R;
	}
	const int32 Half = FMath::Max(1, FftSize / 2);
	TwiddleCos.SetNumUninitialized(Half);
	TwiddleSin.SetNumUninitialized(Half);
	for (int32 I = 0; I < Half; ++I)
	{
		const double A = -2.0 * Pi * (double)I / (double)FftSize;
		TwiddleCos[I] = FMath::Cos(A);
		TwiddleSin[I] = FMath::Sin(A);
	}

	// librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, htk=False, norm="slaney"), float64 maths, float32 weights.
	const double Sr = (double)Settings.SampleRate;
	const double D = 1.0 / Sr;
	const double FreqStep = 1.0 / ((double)FftSize * D); // np.fft.rfftfreq
	TArray<double> FftFreqs;
	FftFreqs.SetNumUninitialized(NumBins);
	for (int32 J = 0; J < NumBins; ++J)
	{
		FftFreqs[J] = (double)J * FreqStep;
	}
	const int32 NumPoints = NumMelBins + 2;
	const double MinMel = HzToMel(0.0);
	const double MaxMel = HzToMel(Sr / 2.0);
	const double Step = (MaxMel - MinMel) / (double)(NumPoints - 1); // np.linspace
	TArray<double> MelF;
	MelF.SetNumUninitialized(NumPoints);
	for (int32 I = 0; I < NumPoints; ++I)
	{
		const double Mel = (I == NumPoints - 1) ? MaxMel : (double)I * Step + MinMel;
		MelF[I] = MelToHz(Mel);
	}
	MelFilters.SetNumZeroed(NumMelBins * NumBins);
	MelBinStart.SetNumUninitialized(NumMelBins);
	MelBinEnd.SetNumUninitialized(NumMelBins);
	for (int32 I = 0; I < NumMelBins; ++I)
	{
		const double FDiffLo = MelF[I + 1] - MelF[I];
		const double FDiffHi = MelF[I + 2] - MelF[I + 1];
		const double ENorm = 2.0 / (MelF[I + 2] - MelF[I]);
		int32 Start = NumBins;
		int32 End = 0;
		for (int32 J = 0; J < NumBins; ++J)
		{
			const double Lower = -(MelF[I] - FftFreqs[J]) / FDiffLo;
			const double Upper = (MelF[I + 2] - FftFreqs[J]) / FDiffHi;
			const float W32 = (float)FMath::Max(0.0, FMath::Min(Lower, Upper));
			const float Normed = (float)((double)W32 * ENorm);
			MelFilters[I * NumBins + J] = Normed;
			if (Normed != 0.f)
			{
				Start = FMath::Min(Start, J);
				End = J + 1;
			}
		}
		MelBinStart[I] = FMath::Min(Start, End);
		MelBinEnd[I] = End;
	}
}

void FLogMelExtractor::Fft(double* Re, double* Im) const
{
	for (int32 I = 0; I < FftSize; ++I)
	{
		const int32 R = BitReverse[I];
		if (R > I)
		{
			Swap(Re[I], Re[R]);
			Swap(Im[I], Im[R]);
		}
	}
	for (int32 Len = 2; Len <= FftSize; Len <<= 1)
	{
		const int32 HalfLen = Len >> 1;
		const int32 Stride = FftSize / Len;
		for (int32 Base = 0; Base < FftSize; Base += Len)
		{
			for (int32 K = 0; K < HalfLen; ++K)
			{
				const double C = TwiddleCos[K * Stride];
				const double S = TwiddleSin[K * Stride];
				const int32 A = Base + K;
				const int32 B = A + HalfLen;
				const double TRe = Re[B] * C - Im[B] * S;
				const double TIm = Re[B] * S + Im[B] * C;
				Re[B] = Re[A] - TRe;
				Im[B] = Im[A] - TIm;
				Re[A] += TRe;
				Im[A] += TIm;
			}
		}
	}
}

void FLogMelExtractor::ComputeLogMel(TConstArrayView<float> Samples, int32 NumFrames, TArray<double>& OutLogMel) const
{
	using namespace DspPrivate;
	const int32 N = Samples.Num();
	const float* X = Samples.GetData();

	// 1. Pre-emphasis in float32, as torch does on a float32 tensor.
	TArray<float> Y;
	Y.SetNumUninitialized(N);
	if (N > 0)
	{
		Y[0] = X[0];
	}
	for (int32 I = 1; I < N; ++I)
	{
		const float Scaled = Preemphasis * X[I - 1];
		Y[I] = X[I] - Scaled;
	}

	TArray<double> Re, Im, Power;
	Re.SetNumUninitialized(FftSize);
	Im.SetNumUninitialized(FftSize);
	Power.SetNumUninitialized(NumBins);
	OutLogMel.SetNumUninitialized(NumFrames * NumMelBins);
	const int64 PadLeft = FftSize / 2;

	for (int32 Frame = 0; Frame < NumFrames; ++Frame)
	{
		// 2. Centred STFT frame: padded index Frame*Hop + i maps to sample Frame*Hop + i - FftSize/2.
		FMemory::Memzero(Re.GetData(), FftSize * sizeof(double));
		FMemory::Memzero(Im.GetData(), FftSize * sizeof(double));
		const int64 FrameStart = (int64)Frame * HopLength - PadLeft + WindowOffset;
		for (int32 I = 0; I < WindowLength; ++I)
		{
			const int64 Idx = FrameStart + I;
			if (Idx >= 0 && Idx < N)
			{
				Re[WindowOffset + I] = (double)Y[(int32)Idx] * Window[I];
			}
		}
		Fft(Re.GetData(), Im.GetData());
		for (int32 B = 0; B < NumBins; ++B)
		{
			Power[B] = Re[B] * Re[B] + Im[B] * Im[B];
		}
		// 3-4. Mel projection and log.
		double* Row = OutLogMel.GetData() + (int64)Frame * NumMelBins;
		for (int32 Mel = 0; Mel < NumMelBins; ++Mel)
		{
			const float* Filter = MelFilters.GetData() + (int64)Mel * NumBins;
			double Acc = 0.0;
			for (int32 B = MelBinStart[Mel]; B < MelBinEnd[Mel]; ++B)
			{
				Acc += (double)Filter[B] * Power[B];
			}
			Row[Mel] = FMath::Loge(Acc + LogZeroGuard);
		}
	}
}

bool FLogMelExtractor::ExtractLogMel(TConstArrayView<float> Samples, FSpeechFeatures& Out, FString& OutError) const
{
	if (!bValid)
	{
		OutError = FString::Printf(TEXT("FLogMelExtractor: invalid settings (%s)"), *InvalidReason);
		return false;
	}
	if (Samples.Num() == 0)
	{
		OutError = TEXT("FLogMelExtractor: audio is empty");
		return false;
	}
	const int64 NumFrames = (int64)Samples.Num() / HopLength + 1;
	if (NumFrames * NumMelBins > (int64)MAX_int32)
	{
		OutError = TEXT("FLogMelExtractor: audio too long");
		return false;
	}
	TArray<double> LogMel;
	ComputeLogMel(Samples, (int32)NumFrames, LogMel);
	Out.Data.SetNumUninitialized(LogMel.Num());
	for (int32 I = 0; I < LogMel.Num(); ++I)
	{
		Out.Data[I] = (float)LogMel[I];
	}
	Out.NumFrames = (int32)NumFrames;
	Out.NumMelBins = NumMelBins;
	return true;
}

bool FLogMelExtractor::Extract(TConstArrayView<float> Samples, FSpeechFeatures& Out, FString& OutError) const
{
	using namespace DspPrivate;
	if (!bValid)
	{
		OutError = FString::Printf(TEXT("FLogMelExtractor: invalid settings (%s)"), *InvalidReason);
		return false;
	}
	const int32 T = Samples.Num() / HopLength;
	if (T < 2)
	{
		OutError = FString::Printf(TEXT("audio too short: %d samples give %d feature frames, need at least 2 (%d samples)"),
			Samples.Num(), T, 2 * HopLength);
		return false;
	}
	if ((int64)T * NumMelBins > (int64)MAX_int32)
	{
		OutError = TEXT("FLogMelExtractor: audio too long");
		return false;
	}
	TArray<double> LogMel;
	ComputeLogMel(Samples, T, LogMel);

	// 5. Per-bin mean and unbiased variance over the valid frames, accumulated in double (C73 amendment 1). A bin that is
	// constant over the valid frames (zero variance, e.g. digital silence) normalises to exactly 0; HF's value there is
	// float32 rounding noise divided by 1e-5 and is deliberately not matched.
	TArray<double> Mean, Denom;
	TArray<double> MinV, MaxV;
	Mean.SetNumZeroed(NumMelBins);
	Denom.SetNumZeroed(NumMelBins);
	MinV.Init(TNumericLimits<double>::Max(), NumMelBins);
	MaxV.Init(TNumericLimits<double>::Lowest(), NumMelBins);
	for (int32 F = 0; F < T; ++F)
	{
		const double* Row = LogMel.GetData() + (int64)F * NumMelBins;
		for (int32 B = 0; B < NumMelBins; ++B)
		{
			Mean[B] += Row[B];
			MinV[B] = FMath::Min(MinV[B], Row[B]);
			MaxV[B] = FMath::Max(MaxV[B], Row[B]);
		}
	}
	for (int32 B = 0; B < NumMelBins; ++B)
	{
		Mean[B] /= (double)T;
	}
	for (int32 F = 0; F < T; ++F)
	{
		const double* Row = LogMel.GetData() + (int64)F * NumMelBins;
		for (int32 B = 0; B < NumMelBins; ++B)
		{
			const double Dv = Row[B] - Mean[B];
			Denom[B] += Dv * Dv;
		}
	}
	for (int32 B = 0; B < NumMelBins; ++B)
	{
		Denom[B] = FMath::Sqrt(Denom[B] / (double)(T - 1)) + NormEpsilon;
	}
	Out.Data.SetNumUninitialized(T * NumMelBins);
	for (int32 F = 0; F < T; ++F)
	{
		const double* Row = LogMel.GetData() + (int64)F * NumMelBins;
		float* Dst = Out.Data.GetData() + (int64)F * NumMelBins;
		for (int32 B = 0; B < NumMelBins; ++B)
		{
			Dst[B] = (MinV[B] == MaxV[B]) ? 0.f : (float)((Row[B] - Mean[B]) / Denom[B]);
		}
	}
	Out.NumFrames = T;
	Out.NumMelBins = NumMelBins;
	return true;
}
}
