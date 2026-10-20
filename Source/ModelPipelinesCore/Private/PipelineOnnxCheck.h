// Copyright 2026 Holdout Labs, LLC. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"

namespace UE::ModelPipelines::Private
{
	/**
	 * Structural check of serialized ONNX ModelProto bytes (contract C8 amendment rule 1).
	 * Walks the top-level protobuf fields: wire types 0/1/2/5 only, every length bounds-checked, the walk must end
	 * exactly at the buffer end, and field 1 (ir_version, varint) and field 7 (graph, non-empty length-delimited)
	 * must be present. Does not parse nested messages and allocates nothing proportional to the size.
	 * Returns false and sets OutReason on failure.
	 */
	bool IsStructurallyValidOnnx(TConstArrayView64<uint8> Bytes, FString& OutReason);
}
