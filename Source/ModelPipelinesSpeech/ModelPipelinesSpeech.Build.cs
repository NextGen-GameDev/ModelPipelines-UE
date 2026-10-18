// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

using UnrealBuildTool;

public class ModelPipelinesSpeech : ModuleRules
{
	public ModelPipelinesSpeech(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "ModelPipelinesCore" });
		PrivateDependencyModuleNames.AddRange(new string[] { "ModelPipelinesTokenizer", "SignalProcessing" });

		// WP9R.4 microphone source: Audio::FAudioCapture. The backend (WASAPI on Windows) is loaded by the engine's AudioCapture
		// plugin (ModelPipelines.uplugin depends on it).
		PrivateDependencyModuleNames.Add("AudioCaptureCore");

		// Automation tests only: read goldens (Json) and build model packages (NNE). Not needed in Shipping.
		if (Target.Configuration != UnrealTargetConfiguration.Shipping)
		{
			PrivateDependencyModuleNames.AddRange(new string[] { "Json", "NNE" });
		}
	}
}
