// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

using UnrealBuildTool;

public class ModelPipelinesVision : ModuleRules
{
	public ModelPipelinesVision(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "ModelPipelinesCore" });
		// RenderCore + RHI: texture / render-target GPU readback on the render thread.
		PrivateDependencyModuleNames.AddRange(new string[] { "RenderCore", "RHI" });

		// Automation tests only: read goldens (Json), build model packages (NNE), decode PNG test images (ImageWrapper).
		// Not needed in Shipping.
		if (Target.Configuration != UnrealTargetConfiguration.Shipping)
		{
			PrivateDependencyModuleNames.AddRange(new string[] { "Json", "NNE", "ImageWrapper" });
		}
	}
}
