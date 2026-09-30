// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

using UnrealBuildTool;

public class ModelPipelinesMCP : ModuleRules
{
	public ModelPipelinesMCP(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine" });
		PrivateDependencyModuleNames.AddRange(new string[] { "ModelPipelinesEditor" });
	}
}
