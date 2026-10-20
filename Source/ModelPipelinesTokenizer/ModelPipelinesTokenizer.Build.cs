// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

using UnrealBuildTool;

public class ModelPipelinesTokenizer : ModuleRules
{
	public ModelPipelinesTokenizer(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject" });
		// The renamed lib is private: no mpt_ type appears in this module's public headers.
		PrivateDependencyModuleNames.AddRange(new string[] { "ModelPipelinesTokenizerLibrary" });
		// Automation tests only: read goldens (Json). Not needed in Shipping.
		if (Target.Configuration != UnrealTargetConfiguration.Shipping)
		{
			PrivateDependencyModuleNames.Add("Json");
		}
	}
}
