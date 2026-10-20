// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

using UnrealBuildTool;

public class ModelPipelinesCore : ModuleRules
{
	public ModelPipelinesCore(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "NNE" });
		PrivateDependencyModuleNames.AddRange(new string[] { "Projects", "Json" }); // Json: automation tests read goldens
	}
}
