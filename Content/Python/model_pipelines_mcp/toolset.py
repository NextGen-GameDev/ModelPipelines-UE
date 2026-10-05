# Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.
"""Model Pipelines toolset: thin Python layer over unreal.PipelineMcpLibrary (contracts C38/C39).

Imported only by init_unreal.py after it checked that unreal.ToolsetDefinition, unreal.PipelineMcpLibrary and the
toolset_registry module exist (UE 5.8 with the Toolset Registry plugin enabled).
"""

import json
from typing import Any

import unreal

import toolset_registry

# (job, async result, callback, on_done delegate wrapper) per running async tool call, keyed by id(job). Keeps the Python
# wrappers (and with them the UObjects, including the delegate's callable proxy) alive until the job completes; the entry
# is dropped on completion.
_PENDING: dict[int, tuple[Any, Any, Any, Any]] = {}


def _error_of(text: str) -> str | None:
    """Returns the "error" message of a C38 JSON string, or None when it is not an error object."""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, dict) and "error" in parsed:
        message = str(parsed.get("error") or "")
        return message or "unknown error"
    return None


def _sync_result(text: str) -> str:
    """Passes a C38 JSON string through unchanged; {"error": ...} becomes a tool error."""
    message = _error_of(text)
    if message is not None:
        raise RuntimeError(message)
    return text


def _texts_arg(texts: Any) -> list[str]:
    """texts -> a new plain list of str (None -> []). texts is a required list[str] parameter, not Optional[list[str]]:
    the UE 5.8 JSON importer cannot fill an optional array from a JSON array (JsonObjectConverter.cpp:1235-1300), and an
    empty list default is not emitted to the schema, so the registry treats the parameter as required."""
    if texts is None:
        return []
    return [str(t) for t in texts]


def _str_arg(value: Any) -> str:
    """Optional[str] -> str (None -> ""). Optional string parameters stand in for `str = ""`: the registry's schema
    generator drops empty-string defaults, which would make those parameters required."""
    return "" if value is None else str(value)


def _complete(result: Any, success: bool, text: str) -> None:
    """Completes the async result from a job outcome."""
    if success:
        result.set_value(text)
        return
    message = _error_of(text)
    result.set_error(message if message else (text or "the job failed without a message"))


def _chain_job(job: Any) -> Any:
    """Returns a ToolCallAsyncResultString that completes when the C38 job completes."""
    result = unreal.ToolCallAsyncResultString()
    if job is None:
        result.set_error("must be called on the game thread")
        return result
    if job.is_done():
        _complete(result, bool(job.succeeded()), job.get_json())
        return result

    key = id(job)

    def _on_done(success: bool, text: str) -> None:
        entry = _PENDING.pop(key, None)
        target = entry[1] if entry else result
        try:
            _complete(target, bool(success), str(text))
        except Exception as error:  # pylint: disable=broad-except
            target.set_error(f"could not complete the tool result: {error}")

    # Keep the delegate wrapper itself: add_callable binds a UPythonCallableForDelegate proxy that delegates hold only
    # weakly (PyWrapperDelegate.cpp:206-212); the live multicast wrapper reports it to GC (PyWrapperDelegate.cpp:1263,
    # PyReferenceCollector.cpp:267-275). A temporary `job.on_done` wrapper would let a GC drop the callback.
    done_delegate = job.on_done
    _PENDING[key] = (job, result, _on_done, done_delegate)
    done_delegate.add_callable(_on_done)
    return result


@unreal.uclass()
class ModelPipelinesTools(unreal.ToolsetDefinition):
    """Lists, imports, validates, benchmarks and test-runs Model Pipelines model packages (Hugging Face ONNX models
    running on Unreal's NNE). Results are compact JSON text. Only import_model_folder changes assets, and only with
    confirm=True; no tool deletes anything."""

    @toolset_registry.tool_call
    @staticmethod
    def list_model_packages(path: str = "/Game", recursive: bool = True) -> str:
        """Lists the model package assets (PipelineModelPackage) under a content folder. Changes nothing.

        Args:
            path: Content folder to search, e.g. "/Game" (default) or "/Game/Models". Must be a content path.
            recursive: Also search the sub-folders of path (default True).

        Returns:
            JSON text {"count": N, "packages": [{"path", "task", "preferred_device", "model", "labels", "has_tokenizer",
            "valid_settings"}]}, sorted by object path. "model" is "" when unset; "valid_settings" is true when the
            package settings check has no error (no inference is run). An empty or unknown folder gives count 0.
            A path that is not a content path is a tool error.
        """
        return _sync_result(unreal.PipelineMcpLibrary.list_model_packages(path, recursive))

    @toolset_registry.tool_call
    @staticmethod
    def import_model_folder(source_folder: str, destination_path: str = "/Game/Models", asset_name: str | None = None,
                            overwrite_existing: bool = False, confirm: bool = False) -> str:
        """Imports a Hugging Face-style ONNX model folder as a model package plus its NNE model asset.

        Without confirm=True nothing is created or changed: the folder is parsed and the result previews what would be
        created. Ask the user before calling again with confirm=True. Never deletes anything.

        Args:
            source_folder: Folder on disk holding model.onnx (or onnx/model.onnx), config.json and, depending on the
                task, tokenizer.json or preprocessor_config.json. Required.
            destination_path: Content folder for the new assets (default "/Game/Models").
            asset_name: Name of the package asset; omitted or "" (default) uses the sanitized folder name. The model asset is
                named <asset_name>_Model.
            overwrite_existing: Update existing assets of the right classes in place instead of refusing (default
                False).
            confirm: Actually import and save the assets (default False = preview only, nothing changes).

        Returns:
            Preview (confirm False): JSON text {"success": false, "needs_confirm", "would_create": {"package", "model"},
            "exists", "task", "report", "message"}; needs_confirm is false when the folder has errors or the assets
            exist and overwrite_existing is False. Import (confirm True): JSON text {"success", "package", "model",
            "task", "report"} with object paths (null on failure; a failed import leaves nothing behind).
            An empty source_folder is a tool error.
        """
        return _sync_result(unreal.PipelineMcpLibrary.import_model_folder(
            source_folder, destination_path, _str_arg(asset_name), overwrite_existing, confirm))

    @toolset_registry.tool_call
    @staticmethod
    def validate_model_package(package: str, run_smoke_test: bool = True, check_gpu: bool = True) -> str:
        """Validates a model package: settings, tokenizer, model creation on CPU (and GPU), task inputs/outputs and
        smoke inferences. Never modifies the asset. Blocks until the checks finish (usually well under a minute).

        Args:
            package: Object path ("/Game/Models/X.X") or package name ("/Game/Models/X") of the model package.
            run_smoke_test: Run one small inference on each device the model was created on (default True).
            check_gpu: Also create and smoke-run the model on the GPU runtime (default True).

        Returns:
            JSON text {"package", "task", "valid", "errors", "warnings", "report": {"messages", "device_checks"}}.
            An unknown package is a tool error "Model package not found: <package>".
        """
        return _sync_result(unreal.PipelineMcpLibrary.validate_model_package(package, run_smoke_test, check_gpu))

    @toolset_registry.tool_call
    @staticmethod
    # pylint: disable-next=dangerous-default-value
    def benchmark_model_package(package: str, runs: int = 20, devices: str = "both", texts: list[str] = [],
                                image: str | None = None) -> unreal.ToolCallAsyncResultString:
        """Benchmarks a model package per device without blocking the editor (the asset is never modified).

        Each device gets runs + 1 sequential calls; the first is a warm-up reported as first_ms. Times are in
        milliseconds, memory in megabytes.

        Args:
            package: Object path or package name of the model package.
            runs: Measured calls per device, 1 to 1000 (default 20).
            devices: "cpu", "gpu" or "both" (default "both", case-insensitive).
            texts: Required in the call (pass [] when there are none). Text tasks: the batch of one call, at most 64
                strings ([] = one built-in sentence). Ignored for image tasks.
            image: Image tasks: texture object path or package name (default omitted or "" = a synthetic mid-gray
                image of the model input size, at most 4096x4096). Ignored for text tasks.

        Returns:
            JSON text {"package", "task", "runs", "input", "devices": [{"device", "available", "actual_device",
            "first_ms", "inference_ms": {"median", "p90", "min", "max"}, "preprocess_ms": {"median"},
            "postprocess_ms": {"median"}, "total_ms": {"median", "p90", "min", "max"}, "memory_mb": {"before",
            "after_warmup", "delta"}}]}. An unavailable GPU is {"device": "GPU", "available": false}. Argument errors
            (unknown package, runs outside 1..1000, bad devices, too many texts, missing texture) are tool errors.
        """
        try:
            job = unreal.PipelineMcpLibrary.benchmark_model_package(package, runs, devices, _texts_arg(texts),
                                                                  _str_arg(image))
            return _chain_job(job)
        except Exception as error:  # pylint: disable=broad-except
            result = unreal.ToolCallAsyncResultString()
            result.set_error(str(error) or type(error).__name__)
            return result

    @toolset_registry.tool_call
    @staticmethod
    # pylint: disable-next=dangerous-default-value
    def run_pipeline_sample(package: str, texts: list[str] = [], image: str | None = None, top_k: int = 5,
                            score_threshold: float = 0.5) -> unreal.ToolCallAsyncResultString:
        """Runs one call of a model package (on its preferred device) and returns a readable result. Does not block.

        Args:
            package: Object path or package name of the model package.
            texts: Required in the call (pass [] for image tasks). Text tasks: 1 to 64 strings. Ignored for image
                tasks.
            image: Image tasks: texture object path or package name (required for image tasks). Ignored for text tasks.
            top_k: Classification: labels per result, clamped to 1..label count (default 5).
            score_threshold: Object detection: keep detections scoring above it, 0 to 1 (default 0.5).

        Returns:
            JSON text with "task" and "timings" (device, queue/preprocess/inference/postprocess ms) plus, by task: embeddings (dim, norm, head, and a cosine "similarity"
            matrix for 2+ texts), text classification top-k per text, image classification top-k, or detections
            (label, index, score, normalized box [x0, y0, x1, y1], highest score first). Errors are tool errors.
        """
        try:
            job = unreal.PipelineMcpLibrary.run_pipeline_sample(package, _texts_arg(texts), _str_arg(image), top_k,
                                                                score_threshold)
            return _chain_job(job)
        except Exception as error:  # pylint: disable=broad-except
            result = unreal.ToolCallAsyncResultString()
            result.set_error(str(error) or type(error).__name__)
            return result
