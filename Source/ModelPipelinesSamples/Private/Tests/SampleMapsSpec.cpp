// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

// Smoke test for the script-built sample maps (WP5.6 / contract C37). Loads the editor-saved maps headless without opening
// them in a viewport and checks their contents and the Samples content assets.

#if WITH_DEV_AUTOMATION_TESTS

#include "Misc/AutomationTest.h"
#include "AssetRegistry/AssetData.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "AssetRegistry/IAssetRegistry.h"
#include "Engine/DirectionalLight.h"
#include "Engine/SkyLight.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/Texture2D.h"
#include "Engine/World.h"
#include "GameFramework/GameModeBase.h"
#include "GameFramework/PlayerStart.h"
#include "GameFramework/WorldSettings.h"
#include "Components/StaticMeshComponent.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstance.h"
#include "Materials/MaterialInterface.h"
#include "UObject/Package.h"
#include "UObject/UObjectGlobals.h"
#include "UObject/GarbageCollection.h"
#include "UObject/SoftObjectPath.h"
#include "ModelPipelinesSampleBase.h"
#include "ModelPipelinesSemanticSearchSample.h"
#include "ModelPipelinesSentimentSample.h"
#include "ModelPipelinesSceneClassifierSample.h"
#include "ModelPipelinesDetectionSample.h"

namespace SampleMapsSpecPrivate
{
	struct FMapCase
	{
		const TCHAR* MapName;
		UClass* (*GetSampleClass)();
		const TCHAR* ExpectedPackage;
		/** Photo names (without MI_ prefix) of the boards expected in the map; empty for text maps. */
		TArray<FString> Boards;
	};

	static UClass* SearchClass() { return AModelPipelinesSemanticSearchSample::StaticClass(); }
	static UClass* SentimentClass() { return AModelPipelinesSentimentSample::StaticClass(); }
	static UClass* SceneClass() { return AModelPipelinesSceneClassifierSample::StaticClass(); }
	static UClass* DetectionClass() { return AModelPipelinesDetectionSample::StaticClass(); }

	static TArray<FMapCase> MakeCases()
	{
		TArray<FMapCase> C;
		C.Add({ TEXT("L_SemanticSearch"), &SearchClass, TEXT("/Game/Models/all-MiniLM-L6-v2.all-MiniLM-L6-v2"), {} });
		C.Add({ TEXT("L_Sentiment"), &SentimentClass, TEXT("/Game/Models/distilbert-sst2.distilbert-sst2"), {} });
		C.Add({ TEXT("L_SceneClassifier"), &SceneClass, TEXT("/Game/Models/resnet-50.resnet-50"),
			{ TEXT("classify_jellyfish"), TEXT("classify_soccer_ball"), TEXT("classify_violin"), TEXT("classify_elephant") } });
		C.Add({ TEXT("L_Detection"), &DetectionClass, TEXT("/Game/Models/rtdetr_r18vd.rtdetr_r18vd"),
			{ TEXT("detect_dog_walk"), TEXT("detect_street_market") } });
		return C;
	}

	static const TCHAR* const MapsDir = TEXT("/ModelPipelines/Samples/Maps/");
	static const TCHAR* const MaterialsDir = TEXT("/ModelPipelines/Samples/Materials/");
	static const TCHAR* const TexturesDir = TEXT("/ModelPipelines/Samples/Textures/");

	static const TCHAR* const AllPhotos[] = {
		TEXT("classify_jellyfish"), TEXT("classify_soccer_ball"), TEXT("classify_violin"), TEXT("classify_elephant"),
		TEXT("detect_dog_walk"), TEXT("detect_street_market") };

	/** Loads the map package and finds its world (never initialized, never added to the editor). Reports a failure and returns null. */
	static UWorld* LoadMapWorld(FAutomationTestBase& T, const FString& MapName)
	{
		const FString PackageName = FString(MapsDir) + MapName;
		UPackage* Pkg = LoadPackage(nullptr, *PackageName, LOAD_None);
		if (!Pkg)
		{
			T.AddError(FString::Printf(TEXT("LoadPackage failed for %s. The sample maps are built by ModelPipelines-UE/Tools/samples/run_build_sample_maps.ps1 (UE 5.7) and live in Content/Samples/Maps. Is the plugin mounted at /ModelPipelines/?"), *PackageName));
			return nullptr;
		}
		UWorld* World = UWorld::FindWorldInPackage(Pkg);
		if (!World)
		{
			T.AddError(FString::Printf(TEXT("Package %s loaded but contains no UWorld."), *PackageName));
			return nullptr;
		}
		return World;
	}

	template <typename T>
	static TArray<T*> ActorsOf(UWorld* World)
	{
		TArray<T*> Out;
		if (World && World->PersistentLevel)
		{
			for (AActor* A : World->PersistentLevel->Actors)
			{
				if (T* Typed = Cast<T>(A)) { Out.Add(Typed); }
			}
		}
		return Out;
	}

	static FString ListActors(UWorld* World)
	{
		TArray<FString> Names;
		if (World && World->PersistentLevel)
		{
			for (AActor* A : World->PersistentLevel->Actors)
			{
				if (A) { Names.Add(A->GetClass()->GetName()); }
			}
		}
		return FString::Join(Names, TEXT(", "));
	}

	static UMaterialInstance* GetBoardMI(AStaticMeshActor* Actor)
	{
		UStaticMeshComponent* Comp = Actor ? Actor->GetStaticMeshComponent() : nullptr;
		return Comp ? Cast<UMaterialInstance>(Comp->GetMaterial(0)) : nullptr;
	}

	static bool IsBasicPlane(AStaticMeshActor* Actor)
	{
		UStaticMeshComponent* Comp = Actor ? Actor->GetStaticMeshComponent() : nullptr;
		UStaticMesh* Mesh = Comp ? Comp->GetStaticMesh() : nullptr;
		return Mesh && Mesh->GetPathName() == TEXT("/Engine/BasicShapes/Plane.Plane");
	}

	static UTexture* GetPhotoParam(UMaterialInstance* MI)
	{
		if (!MI) { return nullptr; }
		for (const FTextureParameterValue& V : MI->TextureParameterValues)
		{
			if (V.ParameterInfo.Name == FName(TEXT("Photo"))) { return V.ParameterValue; }
		}
		return nullptr;
	}
}

BEGIN_DEFINE_SPEC(FSampleMapsSpec, "Pipelines.Samples.Maps",
	EAutomationTestFlags::EditorContext | EAutomationTestFlags::ProductFilter)
	TArray<TWeakObjectPtr<UWorld>> LoadedWorlds;
END_DEFINE_SPEC(FSampleMapsSpec)

void FSampleMapsSpec::Define()
{
	using namespace SampleMapsSpecPrivate;

	AfterEach([this]()
	{
		// The loaded worlds are plain packages, not the editor world: nothing is rooted or destroyed by hand, GC takes them.
		LoadedWorlds.Reset();
		CollectGarbage(GARBAGE_COLLECTION_KEEPFLAGS);
	});

	for (const FMapCase& Case : MakeCases())
	{
		const FString MapName = Case.MapName;
		Describe(MapName, [this, Case, MapName]()
		{
			It("Contains the expected actors and world settings", [this, Case, MapName]()
			{
				// A map missing its sample actor (or holding two) would show nothing or run two pipelines on Play; no PlayerStart
				// drops the player at the origin inside the floor; a wrong game mode removes WASD movement.
				UWorld* World = LoadMapWorld(*this, MapName);
				if (!World) { return; }
				LoadedWorlds.Add(World);
				const FString All = ListActors(World);

				const TArray<AModelPipelinesSampleBase*> Samples = ActorsOf<AModelPipelinesSampleBase>(World);
				int32 Right = 0;
				for (AModelPipelinesSampleBase* S : Samples) { if (S->GetClass() == Case.GetSampleClass()) { ++Right; } }
				TestEqual(*FString::Printf(TEXT("%s: sample actors of class %s (all sample actors: %d; actors in level: %s)"), *MapName, *Case.GetSampleClass()->GetName(), Samples.Num(), *All), Right, 1);
				TestEqual(*FString::Printf(TEXT("%s: total sample actors (any class); found %d"), *MapName, Samples.Num()), Samples.Num(), 1);

				const int32 Starts = ActorsOf<APlayerStart>(World).Num();
				TestEqual(*FString::Printf(TEXT("%s: APlayerStart count; actors: %s"), *MapName, *All), Starts, 1);
				TestTrue(*FString::Printf(TEXT("%s: has a directional light; actors: %s"), *MapName, *All), ActorsOf<ADirectionalLight>(World).Num() >= 1);
				TestTrue(*FString::Printf(TEXT("%s: has a sky light; actors: %s"), *MapName, *All), ActorsOf<ASkyLight>(World).Num() >= 1);

				AWorldSettings* WS = World->GetWorldSettings();
				if (TestNotNull(*FString::Printf(TEXT("%s: world settings"), *MapName), WS))
				{
					UClass* GM = WS->DefaultGameMode.Get();
					TestTrue(*FString::Printf(TEXT("%s: DefaultGameMode should be exactly GameModeBase, found %s"), *MapName, GM ? *GM->GetPathName() : TEXT("<null>")),
						GM == AGameModeBase::StaticClass());
				}
			});

			It("Sample actor Package equals its default", [this, Case, MapName]()
			{
				// Wrong slug in the placed actor means the sample asks the user to import the wrong model.
				UWorld* World = LoadMapWorld(*this, MapName);
				if (!World) { return; }
				LoadedWorlds.Add(World);
				AModelPipelinesSampleBase* Sample = nullptr;
				for (AModelPipelinesSampleBase* S : ActorsOf<AModelPipelinesSampleBase>(World)) { if (S->GetClass() == Case.GetSampleClass()) { Sample = S; } }
				if (!TestNotNull(*FString::Printf(TEXT("%s: sample actor present"), *MapName), Sample)) { return; }
				const FString Found = Sample->Package.ToSoftObjectPath().ToString();
				TestEqual(*FString::Printf(TEXT("%s: Package soft path; found '%s'"), *MapName, *Found), Found, FString(Case.ExpectedPackage));
			});

			if (Case.Boards.Num() > 0)
			{
				It("Has the photo boards with the right material instances", [this, Case, MapName]()
				{
					// A board with the default material (or the wrong photo) would make the classifier/detector see grey or the wrong
					// image; a missing board leaves the player looking at nothing.
					UWorld* World = LoadMapWorld(*this, MapName);
					if (!World) { return; }
					LoadedWorlds.Add(World);
					TArray<FString> Found;
					for (AStaticMeshActor* A : ActorsOf<AStaticMeshActor>(World))
					{
						UMaterialInstance* MI = GetBoardMI(A);
						if (IsBasicPlane(A) && MI && MI->GetPathName().StartsWith(MaterialsDir))
						{
							Found.Add(MI->GetName());
						}
					}
					Found.Sort();
					TArray<FString> Expected;
					for (const FString& B : Case.Boards) { Expected.Add(TEXT("MI_") + B); }
					Expected.Sort();
					TestEqual(*FString::Printf(TEXT("%s: board count; found [%s] expected [%s]"), *MapName, *FString::Join(Found, TEXT(", ")), *FString::Join(Expected, TEXT(", "))), Found.Num(), Expected.Num());
					TestTrue(*FString::Printf(TEXT("%s: board material instances; found [%s] expected [%s]"), *MapName, *FString::Join(Found, TEXT(", ")), *FString::Join(Expected, TEXT(", "))), Found == Expected);
				});
			}
		});
	}

	Describe("Assets", [this]()
	{
		It("M_SamplePhoto has a texture parameter named Photo", [this]()
		{
			// The MIs set "Photo"; if the parent material names it differently the boards render the default texture.
			UMaterial* M = LoadObject<UMaterial>(nullptr, TEXT("/ModelPipelines/Samples/Materials/M_SamplePhoto.M_SamplePhoto"));
			if (!TestNotNull(TEXT("M_SamplePhoto loads as a UMaterial"), M)) { return; }
			TArray<FMaterialParameterInfo> Infos;
			TArray<FGuid> Ids;
			M->GetAllTextureParameterInfo(Infos, Ids);
			TArray<FString> Names;
			bool bFound = false;
			for (const FMaterialParameterInfo& I : Infos) { Names.Add(I.Name.ToString()); if (I.Name == FName(TEXT("Photo"))) { bFound = true; } }
			TestTrue(*FString::Printf(TEXT("texture parameter 'Photo' exists; found [%s]"), *FString::Join(Names, TEXT(", "))), bFound);
		});

		It("Each MI sets Photo to the matching texture", [this]()
		{
			// MI_x pointing at T_y would make a classify board show the wrong photo.
			for (const TCHAR* Name : AllPhotos)
			{
				const FString MIPath = FString::Printf(TEXT("%sMI_%s.MI_%s"), MaterialsDir, Name, Name);
				UMaterialInstance* MI = LoadObject<UMaterialInstance>(nullptr, *MIPath);
				if (!TestNotNull(*FString::Printf(TEXT("%s loads"), *MIPath), MI)) { continue; }
				UTexture* Tex = GetPhotoParam(MI);
				const FString ExpectedTex = FString::Printf(TEXT("%sT_%s.T_%s"), TexturesDir, Name, Name);
				TestEqual(*FString::Printf(TEXT("%s: Photo value; found '%s'"), *MIPath, Tex ? *Tex->GetPathName() : TEXT("<none>")),
					Tex ? Tex->GetPathName() : FString(), ExpectedTex);
				TestTrue(*FString::Printf(TEXT("%s: parent is M_SamplePhoto; found '%s'"), *MIPath, MI->Parent ? *MI->Parent->GetPathName() : TEXT("<none>")),
					MI->Parent && MI->Parent->GetPathName() == TEXT("/ModelPipelines/Samples/Materials/M_SamplePhoto.M_SamplePhoto"));
			}
		});

		It("Each texture is at most 1024 px on both axes", [this]()
		{
			// The plugin ships these; oversized photos bloat the package (target < 8 MB total).
			for (const TCHAR* Name : AllPhotos)
			{
				const FString TexPath = FString::Printf(TEXT("%sT_%s.T_%s"), TexturesDir, Name, Name);
				UTexture2D* Tex = LoadObject<UTexture2D>(nullptr, *TexPath);
				if (!TestNotNull(*FString::Printf(TEXT("%s loads as UTexture2D"), *TexPath), Tex)) { continue; }
				// Lead fix (WP5.6): under -nullrhi the editor never builds platform data, so GetSizeX() is 0; the source
				// image size (editor-only data) is what ships to the cooker.
#if WITH_EDITORONLY_DATA
				const int32 W = (int32)Tex->Source.GetSizeX();
				const int32 H = (int32)Tex->Source.GetSizeY();
#else
				const int32 W = Tex->GetSizeX();
				const int32 H = Tex->GetSizeY();
#endif
				TestTrue(*FString::Printf(TEXT("%s is %dx%d, expected 1..1024 on both axes"), *TexPath, W, H), W >= 1 && H >= 1 && W <= 1024 && H <= 1024);
			}
		});

		It("Asset registry lists exactly the 17 sample assets", [this]()
		{
			// No stray engine copies or leftovers from an earlier script run: 4 maps + 6 textures + 1 material + 6 MIs.
			IAssetRegistry& AR = FAssetRegistryModule::GetRegistry();
			AR.ScanPathsSynchronous({ TEXT("/ModelPipelines/Samples") }, true);
			TArray<FAssetData> Assets;
			AR.GetAssetsByPath(FName(TEXT("/ModelPipelines/Samples")), Assets, true);

			TMap<FString, int32> ByClass;
			TArray<FString> Names;
			for (const FAssetData& A : Assets)
			{
				ByClass.FindOrAdd(A.AssetClassPath.GetAssetName().ToString())++;
				Names.Add(A.GetObjectPathString());
			}
			Names.Sort();
			const FString Listing = FString::Join(Names, TEXT("; "));
			TestEqual(*FString::Printf(TEXT("total assets under /ModelPipelines/Samples; found: %s"), *Listing), Assets.Num(), 17);
			TestEqual(*FString::Printf(TEXT("World assets; found: %s"), *Listing), ByClass.FindRef(TEXT("World")), 4);
			TestEqual(*FString::Printf(TEXT("Texture2D assets; found: %s"), *Listing), ByClass.FindRef(TEXT("Texture2D")), 6);
			TestEqual(*FString::Printf(TEXT("Material assets; found: %s"), *Listing), ByClass.FindRef(TEXT("Material")), 1);
			TestEqual(*FString::Printf(TEXT("MaterialInstanceConstant assets; found: %s"), *Listing), ByClass.FindRef(TEXT("MaterialInstanceConstant")), 6);
		});
	});
}

#endif // WITH_DEV_AUTOMATION_TESTS
