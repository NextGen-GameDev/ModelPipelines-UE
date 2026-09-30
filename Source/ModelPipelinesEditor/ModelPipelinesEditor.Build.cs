// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

using UnrealBuildTool;

public class ModelPipelinesEditor : ModuleRules
{
	public ModelPipelinesEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "ModelPipelinesCore" });
		PrivateDependencyModuleNames.AddRange(new string[] { "UnrealEd" });
	}
}
