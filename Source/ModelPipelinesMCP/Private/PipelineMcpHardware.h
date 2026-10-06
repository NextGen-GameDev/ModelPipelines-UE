// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"

namespace PipelineMcpHardware
{
	/** FEngineVersion::Current().ToString(), e.g. "5.8.3-58210709+++UE5+Release-5.8". */
	FString EngineVersion();

	/**
	 * The adapter DirectML runs on: GRHIAdapterName when the RHI is D3D12 (DML then uses the RHI's device),
	 * else the description of DXGI adapter 0 (DML device index 0). "" when it can't be determined. Game thread.
	 */
	FString GpuAdapter();
}
