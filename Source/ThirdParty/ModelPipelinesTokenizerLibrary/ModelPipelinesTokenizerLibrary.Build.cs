// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

using System.IO;
using UnrealBuildTool;

/** mpt_tokenizers_c.lib: the pinned tokenizers_c.lib with every symbol renamed to mpt_* (contract C41), so it links
    next to the Tokenizers plugin's tokenizers_c.lib. Static lib; its token_ids buffers are freed only by
    mpt_tokenizers_free_encode_results and handles only by mpt_tokenizers_free. */
public class ModelPipelinesTokenizerLibrary : ModuleRules
{
	public ModelPipelinesTokenizerLibrary(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;
		PublicSystemIncludePaths.Add(Path.Combine(ModuleDirectory, "Public"));

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string LibPath = Path.Combine(ModuleDirectory, "x64", "Release", "mpt_tokenizers_c.lib");
			if (!File.Exists(LibPath))
			{
				throw new BuildException(
					"ModelPipelinesTokenizerLibrary: missing " + LibPath + ". Build it from the pinned tokenizers_c.lib with: " +
					"powershell -NoProfile -ExecutionPolicy Bypass -File <Tokenizers-UE>/Scripts/MakePrefixedTokenizersLib.ps1 " +
					"-InputLib <Tokenizers-UE>/Source/ThirdParty/tokenizersLibrary/x64/Release/tokenizers_c.lib -SymbolPrefix mpt_ " +
					"-OutDir \"" + Path.Combine(ModuleDirectory, "x64", "Release") + "\" (the lib itself comes from " +
					"<Tokenizers-UE>/Scripts/BuildTokenizersLib.ps1), then copy mpt_tokenizers_c.h to " + Path.Combine(ModuleDirectory, "Public") + ".");
			}
			PublicAdditionalLibraries.Add(LibPath);

			// native-static-libs of the Rust lib (minus the CRT libs UE already links); see contract C41.
			PublicSystemLibraries.AddRange(new string[] { "ntdll.lib", "userenv.lib", "ws2_32.lib", "dbghelp.lib" });
		}
	}
}
