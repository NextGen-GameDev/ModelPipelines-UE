// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

using UnrealBuildTool;

public class ModelPipelinesEditor : ModuleRules
{
	public ModelPipelinesEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "ModelPipelinesCore" });
		PrivateDependencyModuleNames.AddRange(new string[] {
			"UnrealEd", "NNE", "ModelPipelinesTokenizer", "Json", "AssetRegistry", "AssetTools", "ToolMenus",
			"ContentBrowser", "DesktopPlatform", "Slate", "SlateCore", "Projects", "DeveloperToolSettings",
			"ModelPipelinesSpeech" });   // WP9S.6a: speech import tests create a USpeechToTextPipeline
	}
}
