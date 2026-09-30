// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "Async/Async.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/OutputDevice.h"
#include "Misc/ScopeLock.h"
#include "HAL/PlatformProcess.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/WeakObjectPtrTemplates.h"
#include "UObject/UObjectGlobals.h"
#include "NNEModelData.h"
#include "PipelineBackend.h"
#include "PipelineNNEBackend.h"
#include "PipelineTensor.h"
#include "PipelineTypes.h"
#include "PipelineLog.h"
#include "PipelineTestData.h"

using namespace UE::ModelPipelines::Tests;

namespace
{
	/** A prepared model plus the UObject that must stay alive until PrepareModel returns. */
	struct FPrepared
	{
		TStrongObjectPtr<UNNEModelData> Data;
		TSharedPtr<IPipelineModel> Model;
	};

	bool PrepareFile(FAutomationTestBase& T, IPipelineBackend& Backend, const FString& File, EPipelineDevice Device, FPrepared& Out)
	{
		FString Err;
		UNNEModelData* Data = LoadTestModel(File, Err);
		if (!Data)
		{
			T.AddError(Err);
			return false;
		}
		Out.Data.Reset(Data);
		FPipelineResult R;
		Out.Model = Backend.PrepareModel(Data, Device, R);
		if (!Out.Model.IsValid() || !R.bSuccess)
		{
			T.AddError(FString::Printf(TEXT("PrepareModel(%s) failed: error %d, '%s'"), *File, (int32)R.Error, *R.Message));
			return false;
		}
		return true;
	}

	TSharedPtr<FJsonObject> LoadGoldensOrError(FAutomationTestBase& T)
	{
		FString Err;
		TSharedPtr<FJsonObject> G = LoadTestGoldens(Err);
		if (!G.IsValid())
		{
			T.AddError(Err);
		}
		return G;
	}

	TArray<TSharedPtr<FJsonObject>> GetCases(const TSharedPtr<FJsonObject>& GoldenModel)
	{
		TArray<TSharedPtr<FJsonObject>> Out;
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (GoldenModel.IsValid() && GoldenModel->TryGetArrayField(TEXT("cases"), Arr))
		{
			for (const TSharedPtr<FJsonValue>& V : *Arr)
			{
				const TSharedPtr<FJsonObject>* O = nullptr;
				if (V.IsValid() && V->TryGetObject(O)) { Out.Add(*O); }
			}
		}
		return Out;
	}

	TArray<TSharedPtr<FJsonObject>> GetObjectArray(const TSharedPtr<FJsonObject>& Parent, const TCHAR* Field)
	{
		TArray<TSharedPtr<FJsonObject>> Out;
		const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
		if (Parent.IsValid() && Parent->TryGetArrayField(Field, Arr))
		{
			for (const TSharedPtr<FJsonValue>& V : *Arr)
			{
				const TSharedPtr<FJsonObject>* O = nullptr;
				if (V.IsValid() && V->TryGetObject(O)) { Out.Add(*O); }
			}
		}
		return Out;
	}

	TArray<FPipelineTensor> CaseInputs(const TSharedPtr<FJsonObject>& Case)
	{
		TArray<FPipelineTensor> Out;
		for (const TSharedPtr<FJsonObject>& O : GetObjectArray(Case, TEXT("inputs")))
		{
			Out.Add(TensorFromGoldenJson(O));
		}
		return Out;
	}

	/** Runs one golden case on the model and compares every output. Thread-safe (no Test* calls). Returns "" on success. */
	FString RunAndCompare(IPipelineModel& Model, const TSharedPtr<FJsonObject>& Case, FPipelineResult& OutResult, TArray<FPipelineTensor>& OutOutputs)
	{
		const TArray<FPipelineTensor> Inputs = CaseInputs(Case);
		const TArray<TSharedPtr<FJsonObject>> Expected = GetObjectArray(Case, TEXT("outputs"));
		FString CaseId;
		Case->TryGetStringField(TEXT("id"), CaseId);
		for (const FPipelineTensor& In : Inputs)
		{
			if (!In.IsValid()) { return FString::Printf(TEXT("case %s: golden input could not be parsed"), *CaseId); }
		}
		OutResult = Model.Run(Inputs, OutOutputs);
		if (!OutResult.bSuccess)
		{
			return FString::Printf(TEXT("case %s: Run failed, error %d: %s"), *CaseId, (int32)OutResult.Error, *OutResult.Message);
		}
		if (OutOutputs.Num() != Expected.Num())
		{
			return FString::Printf(TEXT("case %s: got %d outputs, expected %d"), *CaseId, OutOutputs.Num(), Expected.Num());
		}
		for (int32 i = 0; i < Expected.Num(); ++i)
		{
			double Worst = 0.0;
			const FString Mismatch = CompareTensorToGolden(OutOutputs[i], Expected[i], &Worst);
			if (!Mismatch.IsEmpty())
			{
				FString Name;
				Expected[i]->TryGetStringField(TEXT("name"), Name);
				return FString::Printf(TEXT("case %s output #%d '%s': %s"), *CaseId, i, *Name, *Mismatch);
			}
		}
		return FString();
	}

	bool TensorsBitIdentical(const FPipelineTensor& A, const FPipelineTensor& B)
	{
		if (A.IsValid() != B.IsValid() || A.GetType() != B.GetType() || A.DescribeShape() != B.DescribeShape())
		{
			return false;
		}
		if (A.GetSizeInBytes() == 0) { return true; }
		return FMemory::Memcmp(A.GetRawData(), B.GetRawData(), (SIZE_T)A.GetSizeInBytes()) == 0;
	}

	/**
	 * Item 1/2: runs every case of the three golden models, in order, on one prepared model per file.
	 * ExpectedDevice is checked on every Result.
	 */
	void RunGoldenSuite(FAutomationTestBase& T, IPipelineBackend& Backend, EPipelineDevice RequestedDevice, EPipelineDevice ExpectedDevice)
	{
		TSharedPtr<FJsonObject> Goldens = LoadGoldensOrError(T);
		if (!Goldens.IsValid()) { return; }

		const TCHAR* Files[] = { TEXT("affine_dynamic.onnx"), TEXT("affine_static.onnx"), TEXT("int_ops.onnx") };
		int32 TotalCases = 0;
		for (const TCHAR* File : Files)
		{
			TSharedPtr<FJsonObject> GM = FindGoldenModel(Goldens, File);
			if (!GM.IsValid())
			{
				T.AddError(FString::Printf(TEXT("Goldens contain no model entry for %s"), File));
				continue;
			}
			FPrepared P;
			if (!PrepareFile(T, Backend, File, RequestedDevice, P)) { continue; }

			const TArray<TSharedPtr<FJsonObject>> Cases = GetCases(GM);
			T.TestTrue(FString::Printf(TEXT("%s: goldens should have at least one case"), File), Cases.Num() > 0);
			for (const TSharedPtr<FJsonObject>& Case : Cases)
			{
				++TotalCases;
				FString CaseId;
				Case->TryGetStringField(TEXT("id"), CaseId);

				FPipelineResult R1;
				TArray<FPipelineTensor> Out1;
				const FString Err1 = RunAndCompare(*P.Model, Case, R1, Out1);
				T.TestTrue(FString::Printf(TEXT("%s/%s: outputs should match goldens. %s"), File, *CaseId, *Err1), Err1.IsEmpty());
				T.TestTrue(FString::Printf(TEXT("%s/%s: Result.Device expected %d, got %d"), File, *CaseId, (int32)ExpectedDevice, (int32)R1.Device), R1.Device == ExpectedDevice);
				T.TestTrue(FString::Printf(TEXT("%s/%s: InferenceMs should be >= 0 (got %f)"), File, *CaseId, R1.InferenceMs), R1.InferenceMs >= 0.f);

				if (!Err1.IsEmpty()) { continue; }

				// Same case twice in a row takes the cached-shape path and must be bit-identical.
				FPipelineResult R2;
				TArray<FPipelineTensor> Out2;
				const FString Err2 = RunAndCompare(*P.Model, Case, R2, Out2);
				T.TestTrue(FString::Printf(TEXT("%s/%s (repeat): outputs should match goldens. %s"), File, *CaseId, *Err2), Err2.IsEmpty());
				if (Err2.IsEmpty() && Out1.Num() == Out2.Num())
				{
					for (int32 i = 0; i < Out1.Num(); ++i)
					{
						T.TestTrue(FString::Printf(TEXT("%s/%s: repeat run output #%d should be bit-identical"), File, *CaseId, i), TensorsBitIdentical(Out1[i], Out2[i]));
					}
				}
			}

			// Output names and order match the model's golden description.
			const TArray<TSharedPtr<FJsonObject>> GoldenOutputs = GetObjectArray(GM, TEXT("outputs"));
			const TArray<FPipelineTensorDesc> Descs = P.Model->GetOutputDescs();
			T.TestEqual(FString::Printf(TEXT("%s: output desc count"), File), Descs.Num(), GoldenOutputs.Num());
			for (int32 i = 0; i < Descs.Num() && i < GoldenOutputs.Num(); ++i)
			{
				FString Name;
				GoldenOutputs[i]->TryGetStringField(TEXT("name"), Name);
				T.TestEqual(FString::Printf(TEXT("%s: output #%d name"), File, i), Descs[i].Name, Name);
			}
		}
		T.TestTrue(TEXT("golden suite should have run at least 7 cases (4 + 1 + 2)"), TotalCases >= 7);
	}

	/** Counts messages from LogModelPipelines-independent text, by verbosity. */
	class FLogCapture : public FOutputDevice
	{
	public:
		virtual void Serialize(const TCHAR* V, ELogVerbosity::Type Verbosity, const FName& Category) override
		{
			if (FCString::Strstr(V, TEXT("falling back to CPU")))
			{
				FScopeLock Lock(&CS);
				if (Verbosity == ELogVerbosity::Warning) { ++Warnings; }
				else if (Verbosity == ELogVerbosity::Error || Verbosity == ELogVerbosity::Fatal) { ++Errors; }
				else { ++Others; }
			}
		}
		FCriticalSection CS;
		int32 Warnings = 0;
		int32 Errors = 0;
		int32 Others = 0;
	};

	/** Item 2 skip rule. Returns true if the test must stop here. */
	bool SkipGpu(FAutomationTestBase& T)
	{
		if (!FPipelineNNEBackend().IsDeviceAvailable(EPipelineDevice::GPU))
		{
			if (IsGpuRequired())
			{
				T.AddError(TEXT("GPU not available but MODELPIPELINES_REQUIRE_GPU=1"));
			}
			else
			{
				T.AddInfo(TEXT("GPU not available: skipped"));
			}
			return true;
		}
		return false;
	}

	/** Item 6 helper: a failing Run must return Code, leave OutOutputs empty, and not poison the model. */
	void ExpectRunFails(FAutomationTestBase& T, IPipelineModel& Model, const TCHAR* What, const TArray<FPipelineTensor>& Inputs,
		EPipelineError Code, const TCHAR* MsgContains, const TSharedPtr<FJsonObject>& GoodCase)
	{
		TArray<FPipelineTensor> Outs;
		Outs.Add(FPipelineTensor::Zeros(EPipelineTensorType::Float32, TArray<int64>{1})); // must be cleared on failure
		const FPipelineResult R = Model.Run(Inputs, Outs);
		T.TestFalse(FString::Printf(TEXT("%s: bSuccess should be false"), What), R.bSuccess);
		T.TestTrue(FString::Printf(TEXT("%s: error expected %d, got %d (%s)"), What, (int32)Code, (int32)R.Error, *R.Message), R.Error == Code);
		T.TestTrue(FString::Printf(TEXT("%s: message should be non-empty"), What), !R.Message.IsEmpty());
		if (MsgContains && *MsgContains)
		{
			T.TestTrue(FString::Printf(TEXT("%s: message should contain '%s', got '%s'"), What, MsgContains, *R.Message), R.Message.Contains(MsgContains));
		}
		T.TestEqual(FString::Printf(TEXT("%s: OutOutputs should be empty on failure"), What), Outs.Num(), 0);

		// A valid Run afterwards on the same model must still succeed.
		FPipelineResult Good;
		TArray<FPipelineTensor> GoodOuts;
		const FString Err = RunAndCompare(Model, GoodCase, Good, GoodOuts);
		T.TestTrue(FString::Printf(TEXT("%s: valid Run afterwards should still succeed. %s"), What, *Err), Err.IsEmpty());
	}

	struct FWorkerOut
	{
		FPipelineResult Init;
		FString Err;
	};

	const FTimespan WaitLimit = FTimespan::FromSeconds(120);
}

BEGIN_DEFINE_SPEC(FPipelineNNEBackendSpec, "Pipelines.Core.NNEBackend",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FPipelineNNEBackendSpec)

void FPipelineNNEBackendSpec::Define()
{
	Describe("Basics", [this]()
	{
		It("reports the name NNE and CPU/Auto available on the default backend", [this]()
		{
			FPipelineNNEBackend Backend;
			TestEqual(TEXT("GetName expected NNE"), Backend.GetName(), FName(TEXT("NNE")));
			TestTrue(TEXT("CPU should be available"), Backend.IsDeviceAvailable(EPipelineDevice::CPU));
			TestTrue(TEXT("Auto should be available (CPU exists)"), Backend.IsDeviceAvailable(EPipelineDevice::Auto));
		});
	});

	Describe("CPU goldens", [this]()
	{
		It("matches every golden case of affine_dynamic, affine_static and int_ops, twice each, in order on one model", [this]()
		{
			// Order 1 -> 3 -> 8 -> 256 re-sets shapes on one model; a stale shape cache would fail here.
			FPipelineNNEBackend Backend;
			RunGoldenSuite(*this, Backend, EPipelineDevice::CPU, EPipelineDevice::CPU);
		});

		It("gives y=[1,1,3,-1] for x=[0,-0,1,-1] (worked example)", [this]()
		{
			FPipelineNNEBackend Backend;
			FPrepared P;
			if (!PrepareFile(*this, Backend, TEXT("affine_dynamic.onnx"), EPipelineDevice::CPU, P)) { return; }
			TArray<FPipelineTensor> In = { FPipelineTensor::FromFloat(TArray<int64>{1, 4}, TArray<float>{0.f, -0.f, 1.f, -1.f}) };
			TArray<FPipelineTensor> Out;
			const FPipelineResult R = P.Model->Run(In, Out);
			TestTrue(FString::Printf(TEXT("Run should succeed: %s"), *R.Message), R.bSuccess);
			if (R.bSuccess && TestEqual(TEXT("one output expected"), Out.Num(), 1))
			{
				TestEqual(TEXT("output shape expected [1,4]"), Out[0].DescribeShape(), FString(TEXT("[1,4]")));
				TConstArrayView64<float> Y = Out[0].GetFloatData();
				if (TestEqual(TEXT("4 output values expected"), (int64)Y.Num(), (int64)4))
				{
					const float E[4] = {1.f, 1.f, 3.f, -1.f};
					for (int32 i = 0; i < 4; ++i)
					{
						TestTrue(FString::Printf(TEXT("y[%d] expected %g, got %g"), i, E[i], Y[i]), NearlyEqualGolden(Y[i], E[i]));
					}
				}
			}
		});
	});

	Describe("Fallback", [this]()
	{
		It("GPU requested with an unknown GPU runtime: CPU results, Device CPU, exactly one warning over 3 runs", [this]()
		{
			AddExpectedMessage(TEXT("falling back to CPU"), ELogVerbosity::Warning, EAutomationExpectedMessageFlags::Contains, 1);
			FPipelineNNEBackendConfig Cfg;
			Cfg.GpuRuntimeName = TEXT("NoSuchRuntime");
			FPipelineNNEBackend Backend(Cfg);
			TSharedPtr<FJsonObject> Goldens = LoadGoldensOrError(*this);
			if (!Goldens.IsValid()) { return; }
			TSharedPtr<FJsonObject> GM = FindGoldenModel(Goldens, TEXT("affine_dynamic.onnx"));
			if (!GM.IsValid()) { AddError(TEXT("no affine_dynamic golden")); return; }
			FPrepared P;
			if (!PrepareFile(*this, Backend, TEXT("affine_dynamic.onnx"), EPipelineDevice::GPU, P)) { return; }
			const TArray<TSharedPtr<FJsonObject>> Cases = GetCases(GM);
			for (int32 Run = 0; Run < 3; ++Run)
			{
				for (const TSharedPtr<FJsonObject>& Case : Cases)
				{
					FPipelineResult R;
					TArray<FPipelineTensor> Out;
					const FString Err = RunAndCompare(*P.Model, Case, R, Out);
					TestTrue(FString::Printf(TEXT("run %d: CPU fallback results should match. %s"), Run, *Err), Err.IsEmpty());
					TestTrue(FString::Printf(TEXT("run %d: Result.Device expected CPU, got %d"), Run, (int32)R.Device), R.Device == EPipelineDevice::CPU);
				}
			}
			TestTrue(TEXT("model device expected CPU after fallback"), P.Model->GetDevice() == EPipelineDevice::CPU);
		});

		It("Auto with an unknown GPU runtime: CPU results and no warning", [this]()
		{
			FLogCapture Capture;
			GLog->AddOutputDevice(&Capture);
			{
				FPipelineNNEBackendConfig Cfg;
				Cfg.GpuRuntimeName = TEXT("NoSuchRuntime");
				FPipelineNNEBackend Backend(Cfg);
				RunGoldenSuite(*this, Backend, EPipelineDevice::Auto, EPipelineDevice::CPU);
			}
			GLog->Flush();
			GLog->RemoveOutputDevice(&Capture);
			// Auto falling back is Log verbosity; a Warning here means a user who asked for Auto is nagged.
			TestEqual(TEXT("no Warning 'falling back to CPU' expected for Auto"), Capture.Warnings, 0);
			TestEqual(TEXT("no Error 'falling back to CPU' expected for Auto"), Capture.Errors, 0);
		});
	});

	Describe("Unavailable runtimes", [this]()
	{
		It("CPU requested with a bogus CPU runtime: nullptr and RuntimeUnavailable naming the runtime", [this]()
		{
			FPipelineNNEBackendConfig Cfg;
			Cfg.CpuRuntimeName = TEXT("NoCpu");
			FPipelineNNEBackend Backend(Cfg);
			TestFalse(TEXT("IsDeviceAvailable(CPU) expected false"), Backend.IsDeviceAvailable(EPipelineDevice::CPU));

			FString Err;
			TStrongObjectPtr<UNNEModelData> Data(LoadTestModel(TEXT("affine_static.onnx"), Err));
			if (!Data.IsValid()) { AddError(Err); return; }
			FPipelineResult R;
			TSharedPtr<IPipelineModel> M = Backend.PrepareModel(Data.Get(), EPipelineDevice::CPU, R);
			TestTrue(TEXT("model expected nullptr"), !M.IsValid());
			TestFalse(TEXT("bSuccess expected false"), R.bSuccess);
			TestTrue(FString::Printf(TEXT("error expected RuntimeUnavailable(4), got %d"), (int32)R.Error), R.Error == EPipelineError::RuntimeUnavailable);
			TestTrue(FString::Printf(TEXT("message should name the runtime 'NoCpu': '%s'"), *R.Message), R.Message.Contains(TEXT("NoCpu")));
		});

		It("both runtime names bogus: nothing available, GPU and Auto PrepareModel give RuntimeUnavailable", [this]()
		{
			FPipelineNNEBackendConfig Cfg;
			Cfg.CpuRuntimeName = TEXT("NoCpu");
			Cfg.GpuRuntimeName = TEXT("NoGpu");
			FPipelineNNEBackend Backend(Cfg);
			TestFalse(TEXT("CPU unavailable"), Backend.IsDeviceAvailable(EPipelineDevice::CPU));
			TestFalse(TEXT("GPU unavailable"), Backend.IsDeviceAvailable(EPipelineDevice::GPU));
			TestFalse(TEXT("Auto unavailable"), Backend.IsDeviceAvailable(EPipelineDevice::Auto));

			FString Err;
			TStrongObjectPtr<UNNEModelData> Data(LoadTestModel(TEXT("affine_static.onnx"), Err));
			if (!Data.IsValid()) { AddError(Err); return; }
			for (EPipelineDevice D : { EPipelineDevice::GPU, EPipelineDevice::Auto })
			{
				FPipelineResult R;
				TSharedPtr<IPipelineModel> M = Backend.PrepareModel(Data.Get(), D, R);
				TestTrue(FString::Printf(TEXT("device %d: model expected nullptr"), (int32)D), !M.IsValid());
				TestFalse(TEXT("bSuccess expected false"), R.bSuccess);
				TestTrue(FString::Printf(TEXT("device %d: error expected RuntimeUnavailable(4), got %d"), (int32)D, (int32)R.Error), R.Error == EPipelineError::RuntimeUnavailable);
			}
		});

		It("a bogus GPU name alone leaves CPU and Auto available and GPU unavailable", [this]()
		{
			FPipelineNNEBackendConfig Cfg;
			Cfg.GpuRuntimeName = TEXT("NoSuchRuntime");
			FPipelineNNEBackend Backend(Cfg);
			TestTrue(TEXT("CPU available"), Backend.IsDeviceAvailable(EPipelineDevice::CPU));
			TestFalse(TEXT("GPU unavailable"), Backend.IsDeviceAvailable(EPipelineDevice::GPU));
			TestTrue(TEXT("Auto available via CPU"), Backend.IsDeviceAvailable(EPipelineDevice::Auto));
		});
	});

	Describe("Bad model", [this]()
	{
		It("null ModelData gives InvalidArgument", [this]()
		{
			FPipelineNNEBackend Backend;
			FPipelineResult R;
			TSharedPtr<IPipelineModel> M = Backend.PrepareModel(nullptr, EPipelineDevice::CPU, R);
			TestTrue(TEXT("model expected nullptr"), !M.IsValid());
			TestFalse(TEXT("bSuccess expected false"), R.bSuccess);
			TestTrue(FString::Printf(TEXT("error expected InvalidArgument(1), got %d"), (int32)R.Error), R.Error == EPipelineError::InvalidArgument);
			TestTrue(TEXT("message non-empty"), !R.Message.IsEmpty());
		});

		It("garbage bytes give nullptr and ModelLoadFailed without crashing", [this]()
		{
			// The engine logs its own errors for unparsable data; those are expected, nothing else.
			AddExpectedError(TEXT("model data"), EAutomationExpectedErrorFlags::Contains, -1);
			AddExpectedError(TEXT("Cannot create"), EAutomationExpectedErrorFlags::Contains, -1);
			AddExpectedError(TEXT("ONNX"), EAutomationExpectedErrorFlags::Contains, -1);
			const char* Junk = "not an onnx model, just some text bytes 0123456789";
			TArray<uint8> Bytes;
			Bytes.Append(reinterpret_cast<const uint8*>(Junk), (int32)FCStringAnsi::Strlen(Junk));
			TStrongObjectPtr<UNNEModelData> Data(MakeModelFromBytes(Bytes));
			FPipelineNNEBackend Backend;
			FPipelineResult R;
			TSharedPtr<IPipelineModel> M = Backend.PrepareModel(Data.Get(), EPipelineDevice::CPU, R);
			TestTrue(TEXT("model expected nullptr"), !M.IsValid());
			TestFalse(TEXT("bSuccess expected false"), R.bSuccess);
			TestTrue(FString::Printf(TEXT("error expected ModelLoadFailed(5), got %d: %s"), (int32)R.Error, *R.Message), R.Error == EPipelineError::ModelLoadFailed);
			TestTrue(TEXT("message non-empty"), !R.Message.IsEmpty());
		});

		It("half_io (float16) fails Initialize and Run with UnsupportedType naming the tensor", [this]()
		{
			AddExpectedError(TEXT("model data"), EAutomationExpectedErrorFlags::Contains, -1);
			AddExpectedError(TEXT("Cannot create"), EAutomationExpectedErrorFlags::Contains, -1);
			AddExpectedError(TEXT("ONNX"), EAutomationExpectedErrorFlags::Contains, -1);
			FPipelineNNEBackend Backend;
			FString Err;
			TStrongObjectPtr<UNNEModelData> Data(LoadTestModel(TEXT("half_io.onnx"), Err));
			if (!Data.IsValid()) { AddError(Err); return; }
			FPipelineResult PR;
			TSharedPtr<IPipelineModel> M = Backend.PrepareModel(Data.Get(), EPipelineDevice::CPU, PR);
			if (!M.IsValid())
			{
				// Brief: "PrepareModel may succeed". If it refuses, it must still be a clean error.
				AddInfo(FString::Printf(TEXT("PrepareModel refused half_io (error %d: %s); Initialize checks skipped"), (int32)PR.Error, *PR.Message));
				TestFalse(TEXT("bSuccess expected false"), PR.bSuccess);
				TestTrue(TEXT("refusal should be UnsupportedType or ModelLoadFailed"), PR.Error == EPipelineError::UnsupportedType || PR.Error == EPipelineError::ModelLoadFailed);
				return;
			}
			const FPipelineResult IR = M->Initialize();
			TestFalse(TEXT("Initialize bSuccess expected false"), IR.bSuccess);
			TestTrue(FString::Printf(TEXT("Initialize error expected UnsupportedType(2), got %d: %s"), (int32)IR.Error, *IR.Message), IR.Error == EPipelineError::UnsupportedType);
			TestTrue(FString::Printf(TEXT("message should name tensor 'h': '%s'"), *IR.Message), IR.Message.Contains(TEXT("h")));
			TestFalse(TEXT("IsInitialized expected false"), M->IsInitialized());

			TArray<FPipelineTensor> Out;
			const FPipelineResult RR = M->Run(TArray<FPipelineTensor>{ FPipelineTensor::FromFloat(TArray<int64>{2}, TArray<float>{1.f, 2.f}) }, Out);
			TestFalse(TEXT("Run bSuccess expected false"), RR.bSuccess);
			TestTrue(FString::Printf(TEXT("Run error expected UnsupportedType(2), got %d: %s"), (int32)RR.Error, *RR.Message), RR.Error == EPipelineError::UnsupportedType);
			TestEqual(TEXT("OutOutputs empty"), Out.Num(), 0);
			TestFalse(TEXT("IsInitialized still false"), M->IsInitialized());
		});
	});

	Describe("Input validation", [this]()
	{
		It("affine_dynamic: bad input count, type, rank, dims, invalid and empty tensors all fail cleanly and do not poison the model", [this]()
		{
			TSharedPtr<FJsonObject> Goldens = LoadGoldensOrError(*this);
			if (!Goldens.IsValid()) { return; }
			TSharedPtr<FJsonObject> GM = FindGoldenModel(Goldens, TEXT("affine_dynamic.onnx"));
			if (!GM.IsValid()) { AddError(TEXT("no affine_dynamic golden")); return; }
			const TArray<TSharedPtr<FJsonObject>> Cases = GetCases(GM);
			if (Cases.Num() < 2) { AddError(TEXT("affine_dynamic needs >= 2 golden cases")); return; }
			const TSharedPtr<FJsonObject> Good = Cases[1];

			FPipelineNNEBackend Backend;
			FPrepared P;
			if (!PrepareFile(*this, Backend, TEXT("affine_dynamic.onnx"), EPipelineDevice::CPU, P)) { return; }
			IPipelineModel& M = *P.Model;

			// Prime the shape cache with a good run, so failures also prove the cache is not corrupted.
			{
				FPipelineResult R; TArray<FPipelineTensor> O;
				const FString E = RunAndCompare(M, Good, R, O);
				TestTrue(FString::Printf(TEXT("priming run should pass. %s"), *E), E.IsEmpty());
			}

			ExpectRunFails(*this, M, TEXT("0 inputs"), TArray<FPipelineTensor>(), EPipelineError::ShapeMismatch, nullptr, Good);

			TArray<FPipelineTensor> Two;
			Two.Add(FPipelineTensor::Zeros(EPipelineTensorType::Float32, TArray<int64>{2, 4}));
			Two.Add(FPipelineTensor::Zeros(EPipelineTensorType::Float32, TArray<int64>{2, 4}));
			ExpectRunFails(*this, M, TEXT("2 inputs"), Two, EPipelineError::ShapeMismatch, nullptr, Good);

			// Wrong dtype: the message must name the input 'x' and both types, and show -1 as -1.
			{
				TArray<FPipelineTensor> In = { FPipelineTensor::FromInt32(TArray<int64>{2, 4}, TArray<int32>{1, 2, 3, 4, 5, 6, 7, 8}) };
				ExpectRunFails(*this, M, TEXT("int32 [2,4]"), In, EPipelineError::ShapeMismatch, TEXT("x"), Good);
				TArray<FPipelineTensor> Outs;
				const FPipelineResult R = M.Run(In, Outs);
				TestTrue(FString::Printf(TEXT("message should contain 'int32': '%s'"), *R.Message), R.Message.Contains(TEXT("int32")));
				TestTrue(FString::Printf(TEXT("message should contain 'float32': '%s'"), *R.Message), R.Message.Contains(TEXT("float32")));
				TestTrue(FString::Printf(TEXT("message should contain expected shape [-1,4]: '%s'"), *R.Message), R.Message.Contains(TEXT("[-1,4]")));
				TestTrue(FString::Printf(TEXT("message should contain actual shape [2,4]: '%s'"), *R.Message), R.Message.Contains(TEXT("[2,4]")));
			}
			{
				TArray<FPipelineTensor> In = { FPipelineTensor::Zeros(EPipelineTensorType::Float32, TArray<int64>{2, 5}) };
				ExpectRunFails(*this, M, TEXT("float [2,5]"), In, EPipelineError::ShapeMismatch, TEXT("x"), Good);
			}
			{
				TArray<FPipelineTensor> In = { FPipelineTensor::Zeros(EPipelineTensorType::Float32, TArray<int64>{4}) };
				ExpectRunFails(*this, M, TEXT("float [4] (wrong rank)"), In, EPipelineError::ShapeMismatch, TEXT("x"), Good);
			}
			{
				TArray<FPipelineTensor> In = { FPipelineTensor() };
				ExpectRunFails(*this, M, TEXT("invalid tensor"), In, EPipelineError::InvalidArgument, nullptr, Good);
			}
			{
				TArray<FPipelineTensor> In = { FPipelineTensor::FromFloat(TArray<int64>{0, 4}, TArray<float>()) };
				TestTrue(TEXT("[0,4] tensor is valid but has 0 elements"), In[0].IsValid() && In[0].Num() == 0);
				ExpectRunFails(*this, M, TEXT("empty [0,4]"), In, EPipelineError::InvalidArgument, nullptr, Good);
			}
		});

		It("affine_static: float [3,4] against fixed [2,4] is ShapeMismatch, and [2,4] still works", [this]()
		{
			TSharedPtr<FJsonObject> Goldens = LoadGoldensOrError(*this);
			if (!Goldens.IsValid()) { return; }
			const TArray<TSharedPtr<FJsonObject>> Cases = GetCases(FindGoldenModel(Goldens, TEXT("affine_static.onnx")));
			if (Cases.Num() < 1) { AddError(TEXT("affine_static has no golden case")); return; }
			FPipelineNNEBackend Backend;
			FPrepared P;
			if (!PrepareFile(*this, Backend, TEXT("affine_static.onnx"), EPipelineDevice::CPU, P)) { return; }
			TArray<FPipelineTensor> In = { FPipelineTensor::Zeros(EPipelineTensorType::Float32, TArray<int64>{3, 4}) };
			ExpectRunFails(*this, *P.Model, TEXT("static [3,4]"), In, EPipelineError::ShapeMismatch, TEXT("x"), Cases[0]);
		});

		It("int_ops: inputs in swapped order (mask first) is ShapeMismatch, and the right order still works", [this]()
		{
			TSharedPtr<FJsonObject> Goldens = LoadGoldensOrError(*this);
			if (!Goldens.IsValid()) { return; }
			const TArray<TSharedPtr<FJsonObject>> Cases = GetCases(FindGoldenModel(Goldens, TEXT("int_ops.onnx")));
			if (Cases.Num() < 1) { AddError(TEXT("int_ops has no golden case")); return; }
			FPipelineNNEBackend Backend;
			FPrepared P;
			if (!PrepareFile(*this, Backend, TEXT("int_ops.onnx"), EPipelineDevice::CPU, P)) { return; }
			TArray<FPipelineTensor> Swapped = CaseInputs(Cases[0]);
			if (Swapped.Num() != 2) { AddError(TEXT("int_ops golden case should have 2 inputs")); return; }
			Swapped.Swap(0, 1);
			ExpectRunFails(*this, *P.Model, TEXT("swapped int_ops inputs"), Swapped, EPipelineError::ShapeMismatch, nullptr, Cases[0]);
		});
	});

	Describe("Descriptions", [this]()
	{
		It("before Initialize: no descs, device Auto, not initialized", [this]()
		{
			FPipelineNNEBackend Backend;
			FPrepared P;
			if (!PrepareFile(*this, Backend, TEXT("affine_dynamic.onnx"), EPipelineDevice::CPU, P)) { return; }
			TestEqual(TEXT("input descs empty before Initialize"), P.Model->GetInputDescs().Num(), 0);
			TestEqual(TEXT("output descs empty before Initialize"), P.Model->GetOutputDescs().Num(), 0);
			TestTrue(TEXT("GetDevice expected Auto before Initialize"), P.Model->GetDevice() == EPipelineDevice::Auto);
			TestFalse(TEXT("IsInitialized expected false"), P.Model->IsInitialized());
		});

		It("after Initialize on CPU: names, types and shapes of all three models; Initialize idempotent", [this]()
		{
			struct FExpect { const TCHAR* File; TArray<FPipelineTensorDesc> In; TArray<FPipelineTensorDesc> Out; };
			auto D = [](const TCHAR* Name, EPipelineTensorType Type, std::initializer_list<int64> Shape)
			{
				FPipelineTensorDesc Desc; Desc.Name = Name; Desc.Type = Type; Desc.Shape = TArray<int64>(Shape); return Desc;
			};
			TArray<FExpect> Expects;
			Expects.Add({ TEXT("affine_dynamic.onnx"), { D(TEXT("x"), EPipelineTensorType::Float32, {-1, 4}) }, { D(TEXT("y"), EPipelineTensorType::Float32, {-1, 4}) } });
			Expects.Add({ TEXT("affine_static.onnx"), { D(TEXT("x"), EPipelineTensorType::Float32, {2, 4}) }, { D(TEXT("y"), EPipelineTensorType::Float32, {2, 4}) } });
			// int_ops: Int64 ids then Int32 mask; outputs ids_plus_one (Int64), mask_half (Float32), both [-1,-1].
			Expects.Add({ TEXT("int_ops.onnx"),
				{ D(TEXT("ids"), EPipelineTensorType::Int64, {-1, -1}), D(TEXT("mask"), EPipelineTensorType::Int32, {-1, -1}) },
				{ D(TEXT("ids_plus_one"), EPipelineTensorType::Int64, {-1, -1}), D(TEXT("mask_half"), EPipelineTensorType::Float32, {-1, -1}) } });

			FPipelineNNEBackend Backend;
			for (const FExpect& E : Expects)
			{
				FPrepared P;
				if (!PrepareFile(*this, Backend, E.File, EPipelineDevice::CPU, P)) { continue; }
				const FPipelineResult R1 = P.Model->Initialize();
				TestTrue(FString::Printf(TEXT("%s: Initialize should succeed: %s"), E.File, *R1.Message), R1.bSuccess);
				const FPipelineResult R2 = P.Model->Initialize();
				TestTrue(FString::Printf(TEXT("%s: second Initialize should succeed"), E.File), R2.bSuccess);
				TestTrue(FString::Printf(TEXT("%s: IsInitialized expected true"), E.File), P.Model->IsInitialized());
				TestTrue(FString::Printf(TEXT("%s: GetDevice expected CPU"), E.File), P.Model->GetDevice() == EPipelineDevice::CPU);

				const TArray<FPipelineTensorDesc> In = P.Model->GetInputDescs();
				const TArray<FPipelineTensorDesc> Out = P.Model->GetOutputDescs();
				const TArray<FPipelineTensorDesc>* Sets[2][2] = { { &In, &E.In }, { &Out, &E.Out } };
				for (int32 S = 0; S < 2; ++S)
				{
					const TArray<FPipelineTensorDesc>& A = *Sets[S][0];
					const TArray<FPipelineTensorDesc>& X = *Sets[S][1];
					const TCHAR* Kind = S == 0 ? TEXT("input") : TEXT("output");
					if (!TestEqual(FString::Printf(TEXT("%s: %s count"), E.File, Kind), A.Num(), X.Num())) { continue; }
					for (int32 i = 0; i < A.Num(); ++i)
					{
						TestEqual(FString::Printf(TEXT("%s: %s #%d name"), E.File, Kind, i), A[i].Name, X[i].Name);
						TestTrue(FString::Printf(TEXT("%s: %s #%d type expected %s"), E.File, Kind, i, LexToString(X[i].Type)), A[i].Type == X[i].Type);
						TestTrue(FString::Printf(TEXT("%s: %s #%d should be type-supported"), E.File, Kind, i), A[i].bTypeSupported);
						TestTrue(FString::Printf(TEXT("%s: %s #%d shape differs from expected"), E.File, Kind, i), A[i].Shape == X[i].Shape);
					}
				}
			}
		});
	});

	Describe("Lifetime", [this]()
	{
		It("Initialize and Run still work after the UNNEModelData is released and garbage-collected", [this]()
		{
			TSharedPtr<FJsonObject> Goldens = LoadGoldensOrError(*this);
			if (!Goldens.IsValid()) { return; }
			const TArray<TSharedPtr<FJsonObject>> Cases = GetCases(FindGoldenModel(Goldens, TEXT("affine_dynamic.onnx")));
			if (Cases.Num() < 2) { AddError(TEXT("affine_dynamic needs >= 2 golden cases")); return; }
			FPipelineNNEBackend Backend;
			FPrepared P;
			if (!PrepareFile(*this, Backend, TEXT("affine_dynamic.onnx"), EPipelineDevice::CPU, P)) { return; }
			TWeakObjectPtr<UNNEModelData> Weak(P.Data.Get());
			P.Data.Reset();
			CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
			if (Weak.IsValid())
			{
				AddInfo(TEXT("UNNEModelData was not collected; the GC part of this test is weaker than intended"));
			}
			const FPipelineResult IR = P.Model->Initialize();
			TestTrue(FString::Printf(TEXT("Initialize after GC should succeed: %s"), *IR.Message), IR.bSuccess);
			for (const TSharedPtr<FJsonObject>& Case : Cases)
			{
				FPipelineResult R; TArray<FPipelineTensor> Out;
				const FString Err = RunAndCompare(*P.Model, Case, R, Out);
				TestTrue(FString::Printf(TEXT("Run after GC should match goldens. %s"), *Err), Err.IsEmpty());
			}
		});
	});

	Describe("Threads", [this]()
	{
		It("Initialize and Run on a worker thread give golden results (waits with a timeout)", [this]()
		{
			TSharedPtr<FJsonObject> Goldens = LoadGoldensOrError(*this);
			if (!Goldens.IsValid()) { return; }
			const TCHAR* Files[] = { TEXT("affine_dynamic.onnx"), TEXT("int_ops.onnx") };
			FPipelineNNEBackend Backend;
			for (const TCHAR* File : Files)
			{
				const TArray<TSharedPtr<FJsonObject>> Cases = GetCases(FindGoldenModel(Goldens, File));
				if (Cases.Num() < 1) { AddError(FString::Printf(TEXT("%s: no golden cases"), File)); continue; }
				FPrepared P;
				if (!PrepareFile(*this, Backend, File, EPipelineDevice::CPU, P)) { continue; }
				TSharedPtr<IPipelineModel> Model = P.Model;
				TFuture<FWorkerOut> Fut = Async(EAsyncExecution::Thread, [Model, Cases]() -> FWorkerOut
				{
					FWorkerOut Out;
					Out.Init = Model->Initialize();
					if (!Out.Init.bSuccess) { Out.Err = TEXT("Initialize failed on worker: ") + Out.Init.Message; return Out; }
					for (const TSharedPtr<FJsonObject>& Case : Cases)
					{
						FPipelineResult R; TArray<FPipelineTensor> O;
						Out.Err = RunAndCompare(*Model, Case, R, O);
						if (!Out.Err.IsEmpty()) { break; }
					}
					return Out;
				});
				if (!Fut.WaitFor(WaitLimit))
				{
					AddError(FString::Printf(TEXT("%s: worker thread did not finish within 120 s"), File));
					continue;
				}
				const FWorkerOut Res = Fut.Get();
				TestTrue(FString::Printf(TEXT("%s: worker Initialize should succeed"), File), Res.Init.bSuccess);
				TestTrue(FString::Printf(TEXT("%s: worker results should match goldens. %s"), File, *Res.Err), Res.Err.IsEmpty());
			}
		});

		It("8 models prepared on the game thread each run 50 times on their own thread, all outputs correct", [this]()
		{
			TSharedPtr<FJsonObject> Goldens = LoadGoldensOrError(*this);
			if (!Goldens.IsValid()) { return; }
			const TArray<TSharedPtr<FJsonObject>> Cases = GetCases(FindGoldenModel(Goldens, TEXT("affine_dynamic.onnx")));
			if (Cases.Num() < 1) { AddError(TEXT("affine_dynamic has no golden cases")); return; }

			const int32 NumThreads = 8;
			const int32 NumRuns = 50;
			FPipelineNNEBackend Backend;
			TArray<FPrepared> Models;
			Models.SetNum(NumThreads);
			for (int32 i = 0; i < NumThreads; ++i)
			{
				if (!PrepareFile(*this, Backend, TEXT("affine_dynamic.onnx"), EPipelineDevice::CPU, Models[i])) { return; }
			}
			TArray<TFuture<FString>> Futures;
			for (int32 i = 0; i < NumThreads; ++i)
			{
				TSharedPtr<IPipelineModel> Model = Models[i].Model;
				Futures.Add(Async(EAsyncExecution::Thread, [Model, Cases, i, NumRuns]() -> FString
				{
					const FPipelineResult IR = Model->Initialize();
					if (!IR.bSuccess) { return FString::Printf(TEXT("thread %d: Initialize failed: %s"), i, *IR.Message); }
					for (int32 r = 0; r < NumRuns; ++r)
					{
						// Different threads walk the cases at different offsets so shapes differ between models.
						const TSharedPtr<FJsonObject>& Case = Cases[(i + r) % Cases.Num()];
						FPipelineResult R; TArray<FPipelineTensor> O;
						const FString Err = RunAndCompare(*Model, Case, R, O);
						if (!Err.IsEmpty()) { return FString::Printf(TEXT("thread %d run %d: %s"), i, r, *Err); }
						if (R.Device != EPipelineDevice::CPU) { return FString::Printf(TEXT("thread %d run %d: Device %d, expected CPU"), i, r, (int32)R.Device); }
					}
					return FString();
				}));
			}
			for (int32 i = 0; i < NumThreads; ++i)
			{
				if (!Futures[i].WaitFor(WaitLimit))
				{
					AddError(FString::Printf(TEXT("thread %d did not finish within 120 s"), i));
					continue;
				}
				const FString Err = Futures[i].Get();
				TestTrue(FString::Printf(TEXT("thread %d should pass all %d runs. %s"), i, NumRuns, *Err), Err.IsEmpty());
			}
		});
	});
}

BEGIN_DEFINE_SPEC(FPipelineNNEBackendGpuSpec, "Pipelines.Core.NNEBackend.GPU",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
END_DEFINE_SPEC(FPipelineNNEBackendGpuSpec)

void FPipelineNNEBackendGpuSpec::Define()
{
	Describe("Goldens", [this]()
	{
		It("GPU device: every golden case matches within 1e-6 + 1e-6*|e| and Result.Device is GPU", [this]()
		{
			if (SkipGpu(*this)) { return; }
			FPipelineNNEBackend Backend;
			RunGoldenSuite(*this, Backend, EPipelineDevice::GPU, EPipelineDevice::GPU);
		});

		It("Auto device also runs on the GPU when it is available", [this]()
		{
			if (SkipGpu(*this)) { return; }
			FPipelineNNEBackend Backend;
			RunGoldenSuite(*this, Backend, EPipelineDevice::Auto, EPipelineDevice::GPU);
		});
	});

	Describe("Validation and threads", [this]()
	{
		It("GPU model rejects bad input cleanly and still runs afterwards; Initialize reports GPU", [this]()
		{
			if (SkipGpu(*this)) { return; }
			TSharedPtr<FJsonObject> Goldens = LoadGoldensOrError(*this);
			if (!Goldens.IsValid()) { return; }
			const TArray<TSharedPtr<FJsonObject>> Cases = GetCases(FindGoldenModel(Goldens, TEXT("affine_dynamic.onnx")));
			if (Cases.Num() < 2) { AddError(TEXT("affine_dynamic needs >= 2 golden cases")); return; }
			FPipelineNNEBackend Backend;
			FPrepared P;
			if (!PrepareFile(*this, Backend, TEXT("affine_dynamic.onnx"), EPipelineDevice::GPU, P)) { return; }
			const FPipelineResult IR = P.Model->Initialize();
			TestTrue(FString::Printf(TEXT("Initialize should succeed: %s"), *IR.Message), IR.bSuccess);
			TestTrue(TEXT("GetDevice expected GPU"), P.Model->GetDevice() == EPipelineDevice::GPU);
			TArray<FPipelineTensor> Bad = { FPipelineTensor::Zeros(EPipelineTensorType::Float32, TArray<int64>{2, 5}) };
			ExpectRunFails(*this, *P.Model, TEXT("GPU float [2,5]"), Bad, EPipelineError::ShapeMismatch, TEXT("x"), Cases[1]);
			ExpectRunFails(*this, *P.Model, TEXT("GPU 0 inputs"), TArray<FPipelineTensor>(), EPipelineError::ShapeMismatch, nullptr, Cases[1]);
		});

		It("GPU Initialize and Run on a worker thread give golden results", [this]()
		{
			if (SkipGpu(*this)) { return; }
			TSharedPtr<FJsonObject> Goldens = LoadGoldensOrError(*this);
			if (!Goldens.IsValid()) { return; }
			const TArray<TSharedPtr<FJsonObject>> Cases = GetCases(FindGoldenModel(Goldens, TEXT("int_ops.onnx")));
			if (Cases.Num() < 1) { AddError(TEXT("int_ops has no golden cases")); return; }
			FPipelineNNEBackend Backend;
			FPrepared P;
			if (!PrepareFile(*this, Backend, TEXT("int_ops.onnx"), EPipelineDevice::GPU, P)) { return; }
			TSharedPtr<IPipelineModel> Model = P.Model;
			TFuture<FWorkerOut> Fut = Async(EAsyncExecution::Thread, [Model, Cases]() -> FWorkerOut
			{
				FWorkerOut Out;
				Out.Init = Model->Initialize();
				if (!Out.Init.bSuccess) { Out.Err = TEXT("Initialize failed on worker: ") + Out.Init.Message; return Out; }
				for (const TSharedPtr<FJsonObject>& Case : Cases)
				{
					FPipelineResult R; TArray<FPipelineTensor> O;
					Out.Err = RunAndCompare(*Model, Case, R, O);
					if (Out.Err.IsEmpty() && R.Device != EPipelineDevice::GPU) { Out.Err = TEXT("Result.Device was not GPU"); }
					if (!Out.Err.IsEmpty()) { break; }
				}
				return Out;
			});
			if (!Fut.WaitFor(WaitLimit))
			{
				AddError(TEXT("GPU worker thread did not finish within 120 s"));
				return;
			}
			const FWorkerOut Res = Fut.Get();
			TestTrue(FString::Printf(TEXT("GPU worker run should match goldens. %s"), *Res.Err), Res.Init.bSuccess && Res.Err.IsEmpty());
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
