// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

using UnrealBuildTool;

// Helpers behind the Model Pipelines MCP tools (the toolset itself is Python, Content/Python/model_pipelines_mcp).
// No dependency on ToolsetRegistry, ModelContextProtocol or PythonScriptPlugin: this module builds on every engine.
public class ModelPipelinesMCP : ModuleRules
{
	public ModelPipelinesMCP(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "ModelPipelinesCore" });
		PrivateDependencyModuleNames.AddRange(new string[] { "Json", "AssetRegistry", "UnrealEd", "NNE",
			"ModelPipelinesEditor", "ModelPipelinesText", "ModelPipelinesVision" });
	}
}
