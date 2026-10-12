// Copyright NextGen-GameDev. Licensed under the Apache License, Version 2.0.

#pragma once

#include "CoreMinimal.h"

/** A long-running job created by a backend model (docs/contracts.md C61; defined in WP8L.3b). */
class IPipelineSession;

/** What IPipelineModel::CreateSession is asked to create. Plain data. */
struct MODELPIPELINESCORE_API FPipelineSessionRequest
{
	/** e.g. "TextGeneration". */
	FName Kind;

	/** Defined by the module pair (C63 defines the TextGeneration one). */
	TSharedPtr<void, ESPMode::ThreadSafe> Payload;
};
