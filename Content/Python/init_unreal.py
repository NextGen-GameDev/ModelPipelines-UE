# Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
"""Registers the Model Pipelines MCP toolset at editor start-up when the Toolset Registry is available (UE 5.8).

Without it (UE 5.7/5.6, or the ToolsetRegistry / ModelPipelinesMCP modules not loaded) this logs one line and does nothing.
"""

import unreal


def _register_model_pipelines_tools() -> None:
    """Registers model_pipelines_mcp.toolset.ModelPipelinesTools, or logs why it was not registered. Never raises."""
    try:
        if not hasattr(unreal, "ToolsetDefinition"):
            unreal.log("Model Pipelines MCP tools not registered: the Toolset Registry plugin is not loaded")
            return
        if not hasattr(unreal, "PipelineMcpLibrary"):
            unreal.log("Model Pipelines MCP tools not registered: the ModelPipelinesMCP module is not loaded")
            return
        try:
            # pylint: disable-next=import-outside-toplevel
            import toolset_registry.registration
        except ImportError as error:
            unreal.log(f"Model Pipelines MCP tools not registered: toolset_registry is not importable ({error})")
            return

        # pylint: disable-next=import-outside-toplevel
        from model_pipelines_mcp.toolset import ModelPipelinesTools

        if not toolset_registry.registration.Registration([ModelPipelinesTools]).register():
            unreal.log("Model Pipelines MCP tools not registered: the Toolset Registry is not available")
    except Exception as error:  # pylint: disable=broad-except
        unreal.log_warning(f"Model Pipelines MCP tools not registered: {type(error).__name__}: {error}")


_register_model_pipelines_tools()
