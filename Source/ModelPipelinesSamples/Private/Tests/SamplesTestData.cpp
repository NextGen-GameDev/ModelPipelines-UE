// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#if WITH_DEV_AUTOMATION_TESTS

#include "SamplesTestData.h"

#include "Dom/JsonValue.h"
#include "Engine/Engine.h"
#include "Engine/Texture2D.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Misc/App.h"
#include "Modules/ModuleManager.h"
#include "RHI.h"
#include "RenderingThread.h"
#include "Engine/WorldInitializationValues.h"
#include "GameFramework/Actor.h"
#include "HAL/PlatformMisc.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "NNEModelData.h"
#include "PipelineBase.h"
#include "PipelineModelPackage.h"
#include "ModelPipelinesSampleBase.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UObject/Package.h"
#include "UObject/UnrealType.h"
#include "UObject/UObjectGlobals.h"

namespace UE::ModelPipelines::SamplesTests
{
	namespace
	{
		const TCHAR* const FetchHint = TEXT("Generate the text goldens and models with: \"D:/Micromamba/micromambaenv/envs/pipelines/python.exe\" ModelPipelines-UE/Tools/make_text_goldens.py (after Tools/fetch_reference_models.py and Tools/make_import_test_data.py).");
	}

	FString GetDevDir()
	{
		const FString Env = FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_DEV_DIR"));
		if (!Env.IsEmpty())
		{
			return Env;
		}
		return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir() / TEXT("../.pipelines-dev"));
	}

	TSharedPtr<FJsonObject> LoadTextGoldens(FString& OutError)
	{
		const FString Path = GetDevDir() / TEXT("goldens/text/text_goldens.json");
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			OutError = FString::Printf(TEXT("Missing text goldens '%s'. %s"), *Path, FetchHint);
			return nullptr;
		}
		TSharedPtr<FJsonObject> Root;
		const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
		if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
		{
			OutError = FString::Printf(TEXT("Text goldens '%s' are not valid JSON. %s"), *Path, FetchHint);
			return nullptr;
		}
		return Root;
	}

	TSharedPtr<FJsonObject> FindModel(const TSharedPtr<FJsonObject>& Root, const FString& Slug)
	{
		if (!Root.IsValid())
		{
			return nullptr;
		}
		const TArray<TSharedPtr<FJsonValue>>* Models = nullptr;
		if (!Root->TryGetArrayField(TEXT("models"), Models))
		{
			return nullptr;
		}
		for (const TSharedPtr<FJsonValue>& V : *Models)
		{
			const TSharedPtr<FJsonObject>* Obj = nullptr;
			if (V.IsValid() && V->TryGetObject(Obj) && (*Obj)->GetStringField(TEXT("slug")) == Slug)
			{
				return *Obj;
			}
		}
		return nullptr;
	}

	UPipelineModelPackage* MakePackage(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Model, EPipelineDevice Device, FString& OutError)
	{
		if (!Root.IsValid() || !Model.IsValid())
		{
			OutError = TEXT("MakePackage: null goldens or model entry.");
			return nullptr;
		}
		const TSharedPtr<FJsonObject>* PackageBlock = nullptr;
		if (!Model->TryGetObjectField(TEXT("package"), PackageBlock))
		{
			OutError = TEXT("MakePackage: model entry has no \"package\" block.");
			return nullptr;
		}
		const FString Folder = Model->GetStringField(TEXT("folder"));
		const FString ModelDir = Root->GetStringField(TEXT("models_root")) / Folder;
		const FString OnnxPath = ModelDir / TEXT("model.onnx");
		const FString TokPath = ModelDir / TEXT("tokenizer.json");

		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *OnnxPath) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("Missing model '%s'. %s"), *OnnxPath, FetchHint);
			return nullptr;
		}
		FString TokText;
		if (!FFileHelper::LoadFileToString(TokText, *TokPath) || TokText.IsEmpty())
		{
			OutError = FString::Printf(TEXT("Missing tokenizer '%s'. %s"), *TokPath, FetchHint);
			return nullptr;
		}

		UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
		if (!LexTryParseString(Pkg->Task, *Model->GetStringField(TEXT("task"))))
		{
			OutError = FString::Printf(TEXT("Unknown task '%s' in goldens."), *Model->GetStringField(TEXT("task")));
			return nullptr;
		}

		const TSharedPtr<FJsonObject>& P = *PackageBlock;
		Pkg->Preprocess.MaxTokens = (int32)P->GetNumberField(TEXT("max_tokens"));
		Pkg->Preprocess.bAddSpecialTokens = P->GetBoolField(TEXT("add_special_tokens"));
		Pkg->Preprocess.bNormalizeEmbeddings = P->GetBoolField(TEXT("normalize_embeddings"));
		const FString Pooling = P->GetStringField(TEXT("pooling"));
		Pkg->Preprocess.Pooling = Pooling.Equals(TEXT("CLS"), ESearchCase::IgnoreCase) ? EPipelinePooling::CLS
			: Pooling.Equals(TEXT("None"), ESearchCase::IgnoreCase) ? EPipelinePooling::None : EPipelinePooling::Mean;
		Pkg->Postprocess.ScoreActivation = P->GetStringField(TEXT("score_activation")).Equals(TEXT("Sigmoid"), ESearchCase::IgnoreCase)
			? EPipelineScoreActivation::Sigmoid : EPipelineScoreActivation::Softmax;
		const TArray<TSharedPtr<FJsonValue>>* Labels = nullptr;
		if (P->TryGetArrayField(TEXT("labels"), Labels))
		{
			for (const TSharedPtr<FJsonValue>& L : *Labels)
			{
				Pkg->Labels.Add(L->AsString());
			}
		}

		UNNEModelData* Data = NewObject<UNNEModelData>(Pkg);
		Data->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
		Pkg->Model = Data;
		Pkg->TokenizerJson = TokText;
		Pkg->License.SpdxId = TEXT("Apache-2.0");
		Pkg->License.SourceUrl = TEXT("https://example.invalid/test-model");
		Pkg->License.Attribution = TEXT("Test attribution");
		Pkg->PreferredDevice = Device;
		return Pkg;
	}

	UPipelineBase* FindSamplePipeline(const AActor* Actor)
	{
		if (!IsValid(Actor))
		{
			return nullptr;
		}
		for (TFieldIterator<FObjectProperty> It(Actor->GetClass()); It; ++It)
		{
			const FObjectProperty* Prop = *It;
			if (Prop->PropertyClass && Prop->PropertyClass->IsChildOf(UPipelineBase::StaticClass()))
			{
				if (UPipelineBase* Pipe = Cast<UPipelineBase>(Prop->GetObjectPropertyValue_InContainer(Actor)))
				{
					return Pipe;
				}
			}
		}
		return nullptr;
	}

	bool FSamplesTestWorld::Create(FString& OutError)
	{
		FWorldInitializationValues IVS;
		IVS.InitializeScenes(false).AllowAudioPlayback(false).RequiresHitProxies(false).CreatePhysicsScene(false)
			.CreateNavigation(false).CreateAISystem(false).ShouldSimulatePhysics(false).EnableTraceCollision(false)
			.SetTransactional(false).CreateFXSystem(false);
		World = UWorld::CreateWorld(EWorldType::Game, /*bInformEngineOfWorld*/ false, FName(TEXT("SamplesTestWorld")), nullptr, /*bAddToRoot*/ true, ERHIFeatureLevel::Num, &IVS);
		if (!World)
		{
			OutError = TEXT("UWorld::CreateWorld(Game) returned null.");
			return false;
		}
		// Gets components and the level ready for actors; BeginPlay of the actor is dispatched by SpawnSample.
		World->InitializeActorsForPlay(FURL());
		return true;
	}

	AModelPipelinesSampleBase* FSamplesTestWorld::SpawnSample(UClass* SampleClass, const TSoftObjectPtr<UPipelineModelPackage>& Package, bool bShowOverlay,
		TFunction<void(AModelPipelinesSampleBase*)> PreBeginPlay, FString& OutError)
	{
		if (!World || !SampleClass)
		{
			OutError = TEXT("SpawnSample: no world or no class.");
			return nullptr;
		}
		AModelPipelinesSampleBase* Actor = World->SpawnActorDeferred<AModelPipelinesSampleBase>(SampleClass, FTransform::Identity);
		if (!Actor)
		{
			OutError = FString::Printf(TEXT("SpawnActorDeferred(%s) returned null."), *GetNameSafe(SampleClass));
			return nullptr;
		}
		Actor->Package = Package;
		Actor->bShowOverlay = bShowOverlay;
		if (PreBeginPlay)
		{
			PreBeginPlay(Actor);
		}
		Actor->FinishSpawning(FTransform::Identity);
		if (IsValid(Actor) && !Actor->HasActorBegunPlay())
		{
			Actor->DispatchBeginPlay();
		}
		return Actor;
	}

	void FSamplesTestWorld::Destroy()
	{
		if (World)
		{
			World->DestroyWorld(/*bInformEngineOfWorld*/ false);
			World = nullptr;
		}
	}

	bool IsGpuRequired()
	{
		return FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_GPU")) == TEXT("1");
	}

	// ------------------------------------------------------------------ vision helpers (WP5.5)

	namespace
	{
		const TCHAR* const VisionFetchHint = TEXT("Generate the vision goldens and models with: \"D:/Micromamba/micromambaenv/envs/pipelines/python.exe\" ModelPipelines-UE/Tools/make_vision_goldens.py (after Tools/fetch_reference_models.py and Tools/fetch_test_images.py).");

		bool ParseVisionSettings(const TSharedPtr<FJsonObject>& S, FPipelinePreprocessSettings& Out, FString& OutError)
		{
			if (!S.IsValid())
			{
				OutError = TEXT("ParseVisionSettings: null settings object.");
				return false;
			}
			Out.ImageWidth = (int32)S->GetNumberField(TEXT("image_width"));
			Out.ImageHeight = (int32)S->GetNumberField(TEXT("image_height"));
			const FString Mode = S->GetStringField(TEXT("resize_mode"));
			if (Mode == TEXT("Stretch")) { Out.ResizeMode = EPipelineImageResize::Stretch; }
			else if (Mode == TEXT("ShortestEdgeCenterCrop")) { Out.ResizeMode = EPipelineImageResize::ShortestEdgeCenterCrop; }
			else if (Mode == TEXT("StretchCenterCrop")) { Out.ResizeMode = EPipelineImageResize::StretchCenterCrop; }
			else
			{
				OutError = FString::Printf(TEXT("Unknown resize_mode '%s' in goldens."), *Mode);
				return false;
			}
			Out.ResizeShortestEdge = (int32)S->GetNumberField(TEXT("resize_shortest_edge"));
			Out.ResizeWidth = (int32)S->GetNumberField(TEXT("resize_width"));
			Out.ResizeHeight = (int32)S->GetNumberField(TEXT("resize_height"));
			static const TCHAR* const Names[] = { TEXT("Nearest"), TEXT("Lanczos"), TEXT("Bilinear"), TEXT("Bicubic"), TEXT("Box"), TEXT("Hamming") };
			const FString Resample = S->GetStringField(TEXT("resample"));
			bool bFound = false;
			for (int32 i = 0; i < 6; ++i)
			{
				if (Resample == Names[i]) { Out.Resample = (EPipelineImageResample)i; bFound = true; break; } // PIL's values (C26)
			}
			if (!bFound)
			{
				OutError = FString::Printf(TEXT("Unknown resample '%s' in goldens."), *Resample);
				return false;
			}
			Out.RescaleFactor = (float)S->GetNumberField(TEXT("rescale_factor"));
			const TArray<TSharedPtr<FJsonValue>>* Mean = nullptr;
			const TArray<TSharedPtr<FJsonValue>>* Std = nullptr;
			if (!S->TryGetArrayField(TEXT("image_mean"), Mean) || Mean->Num() != 3 || !S->TryGetArrayField(TEXT("image_std"), Std) || Std->Num() != 3)
			{
				OutError = TEXT("image_mean / image_std missing or not 3 numbers in goldens.");
				return false;
			}
			Out.ImageMean = FVector((*Mean)[0]->AsNumber(), (*Mean)[1]->AsNumber(), (*Mean)[2]->AsNumber());
			Out.ImageStd = FVector((*Std)[0]->AsNumber(), (*Std)[1]->AsNumber(), (*Std)[2]->AsNumber());
			return true;
		}
	}

	bool IsRhiRequired()
	{
		return FPlatformMisc::GetEnvironmentVariable(TEXT("MODELPIPELINES_REQUIRE_RHI")) == TEXT("1");
	}

	bool HasRhi()
	{
		return FApp::CanEverRender() && GDynamicRHI != nullptr;
	}

	TSharedPtr<FJsonObject> LoadVisionGoldens(FString& OutError)
	{
		const FString Path = GetDevDir() / TEXT("goldens/vision/vision_goldens.json");
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			OutError = FString::Printf(TEXT("Missing vision goldens '%s'. %s"), *Path, VisionFetchHint);
			return nullptr;
		}
		TSharedPtr<FJsonObject> Root;
		const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
		if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
		{
			OutError = FString::Printf(TEXT("Vision goldens '%s' are not valid JSON. %s"), *Path, VisionFetchHint);
			return nullptr;
		}
		return Root;
	}

	TSharedPtr<FJsonObject> FindVisionModel(const TSharedPtr<FJsonObject>& Root, const FString& Slug)
	{
		const TArray<TSharedPtr<FJsonValue>>* Models = nullptr;
		if (!Root.IsValid() || !Root->TryGetArrayField(TEXT("reference_models"), Models))
		{
			return nullptr;
		}
		for (const TSharedPtr<FJsonValue>& V : *Models)
		{
			const TSharedPtr<FJsonObject> Obj = V.IsValid() ? V->AsObject() : nullptr;
			if (Obj.IsValid() && Obj->GetStringField(TEXT("slug")) == Slug)
			{
				return Obj;
			}
		}
		return nullptr;
	}

	TSharedPtr<FJsonObject> FindVisionCase(const TSharedPtr<FJsonObject>& Model, const FString& ImageId)
	{
		const TArray<TSharedPtr<FJsonValue>>* Cases = nullptr;
		if (!Model.IsValid() || !Model->TryGetArrayField(TEXT("cases"), Cases))
		{
			return nullptr;
		}
		for (const TSharedPtr<FJsonValue>& V : *Cases)
		{
			const TSharedPtr<FJsonObject> Obj = V.IsValid() ? V->AsObject() : nullptr;
			if (Obj.IsValid() && Obj->GetStringField(TEXT("image")) == ImageId)
			{
				return Obj;
			}
		}
		return nullptr;
	}

	UPipelineModelPackage* MakeVisionPackage(const TSharedPtr<FJsonObject>& Root, const TSharedPtr<FJsonObject>& Model, EPipelineDevice Device, FString& OutError)
	{
		if (!Root.IsValid() || !Model.IsValid())
		{
			OutError = TEXT("MakeVisionPackage: null goldens or model entry.");
			return nullptr;
		}
		const TSharedPtr<FJsonObject>* PackageBlock = nullptr;
		if (!Model->TryGetObjectField(TEXT("package"), PackageBlock))
		{
			OutError = TEXT("MakeVisionPackage: model entry has no \"package\" block.");
			return nullptr;
		}
		const FString OnnxPath = Root->GetStringField(TEXT("models_root")) / Model->GetStringField(TEXT("folder")) / TEXT("model.onnx");
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *OnnxPath) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("Missing model '%s'. %s"), *OnnxPath, VisionFetchHint);
			return nullptr;
		}

		UPipelineModelPackage* Pkg = NewObject<UPipelineModelPackage>(GetTransientPackage());
		if (!LexTryParseString(Pkg->Task, *Model->GetStringField(TEXT("task"))))
		{
			OutError = FString::Printf(TEXT("Unknown task '%s' in goldens."), *Model->GetStringField(TEXT("task")));
			return nullptr;
		}
		const TSharedPtr<FJsonObject>& P = *PackageBlock;
		const TSharedPtr<FJsonObject>* Settings = nullptr;
		if (!P->TryGetObjectField(TEXT("settings"), Settings) || !ParseVisionSettings(*Settings, Pkg->Preprocess, OutError))
		{
			if (OutError.IsEmpty())
			{
				OutError = TEXT("MakeVisionPackage: package has no \"settings\" block.");
			}
			return nullptr;
		}
		Pkg->Postprocess.ScoreActivation = P->GetStringField(TEXT("score_activation")).Equals(TEXT("Sigmoid"), ESearchCase::IgnoreCase)
			? EPipelineScoreActivation::Sigmoid : EPipelineScoreActivation::Softmax;
		double Nms = 0.7;
		if (P->TryGetNumberField(TEXT("nms_iou_threshold"), Nms))
		{
			Pkg->Postprocess.NmsIouThreshold = (float)Nms;
		}
		int32 MaxDet = 100;
		if (P->TryGetNumberField(TEXT("max_detections"), MaxDet))
		{
			Pkg->Postprocess.MaxDetections = MaxDet;
		}
		const TArray<TSharedPtr<FJsonValue>>* Labels = nullptr;
		if (P->TryGetArrayField(TEXT("labels"), Labels))
		{
			for (const TSharedPtr<FJsonValue>& L : *Labels)
			{
				Pkg->Labels.Add(L->AsString());
			}
		}

		UNNEModelData* Data = NewObject<UNNEModelData>(Pkg);
		Data->Init(TEXT("onnx"), TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
		Pkg->Model = Data;
		Pkg->License.SpdxId = TEXT("Apache-2.0");
		Pkg->License.SourceUrl = TEXT("https://example.invalid/test-model");
		Pkg->License.Attribution = TEXT("Test attribution");
		Pkg->PreferredDevice = Device;
		return Pkg;
	}

	bool LoadVisionImage(const TSharedPtr<FJsonObject>& Root, const FString& ImageId, FVisionTestImage& Out, FString& OutError)
	{
		Out = FVisionTestImage();
		if (!Root.IsValid())
		{
			OutError = TEXT("LoadVisionImage: null goldens.");
			return false;
		}
		TSharedPtr<FJsonObject> Entry;
		for (const TCHAR* Section : { TEXT("reference_images"), TEXT("images") })
		{
			const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
			if (!Root->TryGetArrayField(Section, Arr)) { continue; }
			for (const TSharedPtr<FJsonValue>& V : *Arr)
			{
				const TSharedPtr<FJsonObject> O = V.IsValid() ? V->AsObject() : nullptr;
				if (O.IsValid() && O->GetStringField(TEXT("id")) == ImageId) { Entry = O; break; }
			}
			if (Entry.IsValid()) { break; }
		}
		if (!Entry.IsValid())
		{
			OutError = FString::Printf(TEXT("Golden image '%s' is not listed in reference_images/images of vision_goldens.json. %s"), *ImageId, VisionFetchHint);
			return false;
		}
		FString RootDir;
		if (!Root->TryGetStringField(TEXT("root"), RootDir) || RootDir.IsEmpty()) { RootDir = GetDevDir() / TEXT("goldens/vision"); }
		const FString AbsPath = RootDir / Entry->GetStringField(TEXT("file"));
		TArray<uint8> Bytes;
		if (!FFileHelper::LoadFileToArray(Bytes, *AbsPath) || Bytes.Num() == 0)
		{
			OutError = FString::Printf(TEXT("Missing PNG '%s'. %s"), *AbsPath, VisionFetchHint);
			return false;
		}
		IImageWrapperModule& Module = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
		const TSharedPtr<IImageWrapper> Wrapper = Module.CreateImageWrapper(EImageFormat::PNG);
		if (!Wrapper.IsValid() || !Wrapper->SetCompressed(Bytes.GetData(), Bytes.Num()))
		{
			OutError = FString::Printf(TEXT("'%s' is not a decodable PNG."), *AbsPath);
			return false;
		}
		TArray64<uint8> Raw;
		if (!Wrapper->GetRaw(ERGBFormat::BGRA, 8, Raw))
		{
			OutError = FString::Printf(TEXT("PNG '%s' could not be decoded to 8-bit BGRA."), *AbsPath);
			return false;
		}
		const int32 W = Wrapper->GetWidth();
		const int32 H = Wrapper->GetHeight();
		if (W <= 0 || H <= 0 || Raw.Num() != (int64)W * H * 4)
		{
			OutError = FString::Printf(TEXT("PNG '%s' decoded to %lld bytes for %dx%d."), *AbsPath, (long long)Raw.Num(), W, H);
			return false;
		}
		Out.Width = W;
		Out.Height = H;
		Out.Pixels.SetNumUninitialized(W * H);
		FMemory::Memcpy(Out.Pixels.GetData(), Raw.GetData(), Raw.Num()); // FColor is B, G, R, A in memory
		return true;
	}

	UTexture2D* MakeVisionTexture(const FVisionTestImage& Image)
	{
		if (!HasRhi() || !Image.IsValid())
		{
			return nullptr;
		}
		TArray<uint8> Bytes;
		Bytes.SetNumUninitialized(Image.Pixels.Num() * 4);
		for (int32 i = 0; i < Image.Pixels.Num(); ++i)
		{
			const FColor C = Image.Pixels[i];
			Bytes[i * 4 + 0] = C.B;
			Bytes[i * 4 + 1] = C.G;
			Bytes[i * 4 + 2] = C.R;
			Bytes[i * 4 + 3] = 255;
		}
		UTexture2D* Texture = UTexture2D::CreateTransient(Image.Width, Image.Height, PF_B8G8R8A8, NAME_None, TConstArrayView64<uint8>(Bytes.GetData(), (int64)Bytes.Num()));
		if (!Texture)
		{
			return nullptr;
		}
		Texture->NeverStream = true;
		Texture->UpdateResource();
		FlushRenderingCommands(); // test code: the sample under test must never do this
		return Texture;
	}

	bool FSamplesSceneWorld::CreateWithScene(FString& OutError)
	{
		FWorldInitializationValues IVS;
		IVS.InitializeScenes(true).AllowAudioPlayback(false).RequiresHitProxies(false).CreatePhysicsScene(false)
			.CreateNavigation(false).CreateAISystem(false).ShouldSimulatePhysics(false).EnableTraceCollision(false)
			.SetTransactional(false).CreateFXSystem(false);
		World = UWorld::CreateWorld(EWorldType::Game, /*bInformEngineOfWorld*/ false, FName(TEXT("SamplesSceneTestWorld")), nullptr, /*bAddToRoot*/ true, ERHIFeatureLevel::Num, &IVS);
		if (!World)
		{
			OutError = TEXT("UWorld::CreateWorld(Game) with a render scene returned null.");
			return false;
		}
		if (!World->Scene)
		{
			OutError = TEXT("The test world has no render scene (World->Scene == nullptr) although InitializeScenes(true) was requested.");
			return false;
		}
		World->InitializeActorsForPlay(FURL());
		return true;
	}
}

#endif // WITH_DEV_AUTOMATION_TESTS
