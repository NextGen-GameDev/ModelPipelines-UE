// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"
#include "Containers/ArrayView.h"
#include "HAL/ThreadSafeBool.h"

/** Streams the file in 4 MiB blocks and returns its SHA-256 as 64 lowercase hex. Any thread. False + OutError if the file
    can't be opened/read or bCancel becomes true (checked between blocks; OutError "cancelled"). Never loads the whole file. */
MODELPIPELINESEDITOR_API bool ComputeFileSha256(const FString& AbsolutePath, FString& OutHex, FString& OutError,
	const FThreadSafeBool* bCancel = nullptr);

/** Same over memory (tests, small buffers). */
MODELPIPELINESEDITOR_API FString ComputeSha256Hex(TConstArrayView<uint8> Data);
