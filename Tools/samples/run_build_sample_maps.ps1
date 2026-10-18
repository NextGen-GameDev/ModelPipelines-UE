<#
Runs build_sample_maps.py inside UnrealEditor-Cmd (headless) and fails unless the script printed BUILD_SAMPLE_MAPS OK.

  powershell -NoProfile -ExecutionPolicy Bypass -File run_build_sample_maps.ps1 `
      -Engine "D:\Unreal\UE_5.7" -Project "<host>\SamplesHost.uproject" -Images "<dir with the 6 PNGs>" *> "<log>"

-Rhi null       passes -nullrhi (CRASHES in UE 5.7: EXCEPTION_INT_DIVIDE_BY_ZERO when the sample actor is spawned)
-Rhi offscreen  (default) passes -RenderOffscreen instead; this is the one that works
                for -OrientCheck, which renders)
-OrientCheck <dir>  forwards --orient-check <dir> to the script
-Only <map>[,<map>] rebuilds only those maps (e.g. -Only L_TalkToText): other maps, textures and materials stay untouched, and
                -Images is optional when the named maps need no photos. Forwarded to the script as --only (env SAMPLE_MAPS_ONLY).
Paths must not contain spaces inside the script argument string unless quoted; this runner quotes them.
Exit code: 0 only if UnrealEditor-Cmd exited 0 AND the log has the OK line AND no Python traceback.
#>
param(
    [Parameter(Mandatory = $true)] [string]$Engine,
    [Parameter(Mandatory = $true)] [string]$Project,
    [string]$Images = "",
    [ValidateSet("null", "offscreen")] [string]$Rhi = "offscreen",
    [string]$OrientCheck = "",
    [string]$Only = "",
    [string]$Log = ""
)
$ErrorActionPreference = "Stop"

$script = Join-Path $PSScriptRoot "build_sample_maps.py"
$exe = Join-Path $Engine "Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
if (-not (Test-Path $exe)) { Write-Error "not found: $exe"; exit 2 }
if (-not (Test-Path $Project)) { Write-Error "not found: $Project"; exit 2 }
if (-not (Test-Path $script)) { Write-Error "not found: $script"; exit 2 }

# The /ModelPipelines/ content mount only exists if the plugin has a Content folder when the editor starts.
$pluginContent = Join-Path (Split-Path $Project -Parent) "Plugins\ModelPipelines\Content"
if (-not (Test-Path $pluginContent)) {
    New-Item -ItemType Directory -Force -Path $pluginContent | Out-Null
    Write-Host "created $pluginContent"
}

# Paths with spaces do not survive the -ExecutePythonScript command-line parsing, so the folders go in
# environment variables (the script reads SAMPLE_MAPS_IMAGES and SAMPLE_MAPS_ORIENT_DIR as defaults).
$pyArgs = "$script"
if ($Images -ne "") { $env:SAMPLE_MAPS_IMAGES = $Images }
if ($Only -ne "") { $env:SAMPLE_MAPS_ONLY = $Only }
if ($OrientCheck -ne "") { New-Item -ItemType Directory -Force -Path $OrientCheck | Out-Null; $env:SAMPLE_MAPS_ORIENT_DIR = $OrientCheck }

if ($Log -eq "") {
    $logDir = Join-Path $PSScriptRoot "..\..\..\.pipelines-dev\logs\m5"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $Log = Join-Path (Resolve-Path $logDir) "wp5.6-editor.log"
}
$rhiArg = if ($Rhi -eq "null") { "-nullrhi" } else { "-RenderOffscreen" }

$argList = @("`"$Project`"", "-ExecutePythonScript=`"$pyArgs`"", "-unattended", "-nosplash", "-nopause", $rhiArg, "-stdout", "-FullStdOutLogOutput", "-abslog=`"$Log`"")
Write-Host "$exe $($argList -join ' ')"
$p = Start-Process -FilePath $exe -ArgumentList $argList -Wait -PassThru -NoNewWindow
Write-Host "UnrealEditor-Cmd exit code: $($p.ExitCode)"

$text = if (Test-Path $Log) { Get-Content $Log -Raw } else { "" }
$ok = $text -match "(?m)^.*BUILD_SAMPLE_MAPS OK\s*$"
$trace = $text -match "Traceback \(most recent call last\)"
if ($trace) { Write-Host "RUNNER: Python traceback found in $Log" }
if ($p.ExitCode -ne 0 -or -not $ok -or $trace) {
    Write-Host "RUNNER: FAILED (exit=$($p.ExitCode), okLine=$ok, traceback=$trace)"
    if ($p.ExitCode -ne 0) { exit $p.ExitCode } else { exit 1 }
}
Write-Host "RUNNER: OK"
exit 0
