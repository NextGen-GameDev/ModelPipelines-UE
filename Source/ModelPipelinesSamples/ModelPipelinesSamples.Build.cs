// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

using UnrealBuildTool;

public class ModelPipelinesSamples : ModuleRules
{
	public ModelPipelinesSamples(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "ModelPipelinesCore" });
		// The sample actors use the task pipelines and draw their overlays with plain Slate.
		PrivateDependencyModuleNames.AddRange(new string[] { "ModelPipelinesText", "ModelPipelinesVision", "Slate", "SlateCore", "InputCore" });

		// Automation tests only: read goldens (Json), build model packages (NNE), decode PNG test images (ImageWrapper),
		// write test render targets (RenderCore, RHI). Not needed in Shipping.
		if (Target.Configuration != UnrealTargetConfiguration.Shipping)
		{
			PrivateDependencyModuleNames.AddRange(new string[] { "Json", "NNE", "ImageWrapper", "RenderCore", "RHI" });
		}
	}
}
