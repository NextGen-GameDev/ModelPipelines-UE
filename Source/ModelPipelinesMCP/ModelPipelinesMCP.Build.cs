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
			"ModelPipelinesEditor", "ModelPipelinesText", "ModelPipelinesVision", "RHI" });

			// Benchmark results name the DXGI adapter DirectML uses when the RHI isn't D3D12 (PipelineMcpHardware.cpp).
			if (Target.Platform == UnrealTargetPlatform.Win64)
			{
				PublicSystemLibraries.Add("dxgi.lib");
			}
	}
}
