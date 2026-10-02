// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

using UnrealBuildTool;

public class ModelPipelinesText : ModuleRules
{
	public ModelPipelinesText(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "ModelPipelinesCore" });
		PrivateDependencyModuleNames.AddRange(new string[] { "Tokenizers", "Json" }); // Json: automation tests read goldens
	}
}
