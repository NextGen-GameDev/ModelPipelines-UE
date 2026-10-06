// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#include "PipelineMcpHardware.h"

#include "DynamicRHI.h"
#include "Misc/EngineVersion.h"
#include "RHIGlobals.h"

#if PLATFORM_WINDOWS
#include "Windows/AllowWindowsPlatformTypes.h"
THIRD_PARTY_INCLUDES_START
#include <dxgi.h>
THIRD_PARTY_INCLUDES_END
#include "Windows/HideWindowsPlatformTypes.h"
#endif

namespace PipelineMcpHardware
{
	FString EngineVersion()
	{
		return FEngineVersion::Current().ToString();
	}

	FString GpuAdapter()
	{
		if (GDynamicRHI && RHIGetInterfaceType() == ERHIInterfaceType::D3D12 && !GRHIAdapterName.IsEmpty())
		{
			return GRHIAdapterName;
		}
#if PLATFORM_WINDOWS
		FString Result;
		IDXGIFactory1* Factory = nullptr;
		if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&Factory))) && Factory)
		{
			IDXGIAdapter1* Adapter = nullptr;
			if (SUCCEEDED(Factory->EnumAdapters1(0, &Adapter)) && Adapter)
			{
				DXGI_ADAPTER_DESC1 Desc;
				FMemory::Memzero(Desc);
				if (SUCCEEDED(Adapter->GetDesc1(&Desc)))
				{
					Desc.Description[UE_ARRAY_COUNT(Desc.Description) - 1] = 0;
					Result = FString(Desc.Description).TrimStartAndEnd();
				}
				Adapter->Release();
			}
			Factory->Release();
		}
		return Result;
#else
		return FString();
#endif
	}
}
