# Sample maps: rebuild script

`build_sample_maps.py` builds the four sample maps, their photo textures and materials with editor Python.
The `.umap` and `.uasset` files are build output. Rebuild them with this script, never by hand.

## What it creates

Everything lives under `/ModelPipelines/Samples/` (plugin folder `Content/Samples/`). The script deletes and recreates that
folder on every run. It creates 17 assets:

| Asset | Count | Notes |
|---|---|---|
| `Maps/L_SemanticSearch`, `L_Sentiment`, `L_SceneClassifier`, `L_Detection` | 4 | Each has a PlayerStart, a directional light, a sky light, `GameModeBase` as the game mode override and one sample actor. |
| `Textures/T_<name>` | 6 | The PNGs made by `Tools/fetch_sample_images.py`. Each is 1024 px or smaller. |
| `Materials/M_SamplePhoto` | 1 | Unlit. A texture parameter named `Photo` feeds Emissive Color. |
| `Materials/MI_<name>` | 6 | One material instance per photo. |

The text maps (`L_SemanticSearch`, `L_Sentiment`) also get a floor plane. The vision maps get photo boards: the four
classify photos in a row on `L_SceneClassifier`, the two detect photos on `L_Detection`. Boards use `/Engine/BasicShapes/Plane`.
No engine content is copied.

## Board orientation

The board rotation is set by constants at the top of the script. `BOARD_SPIN_DEG = 90` is the value that shows the photos
upright and not mirrored (found with the orient check below). `BOARD_MIRROR` is off. If you change the board mesh or the layout,
run the orient check again.

## Engine

Rebuild in the **oldest supported engine** (UE 5.7 now, 5.6 once it is supported), so the assets also load in the newer ones.
The script only uses API names that exist in all of them. It needs the `ModelPipelines` plugin (with the samples module built)
and the Python Editor Script plugin enabled in the host project.

## Rebuild

1. Sync and build the host project in the oldest engine (`.pipelines-dev/build/m5_samples_host.sh sync`, then `build`).
2. Make sure the six PNGs exist (run `Tools/fetch_sample_images.py` if not).
3. Close every other Unreal process. Run:

```
powershell -NoProfile -ExecutionPolicy Bypass -File "<repo>\Tools\samples\run_build_sample_maps.ps1" -Engine "D:\Unreal\UE_5.7" -Project "<host>\SamplesHost.uproject" -Images "<dir with the PNGs>" *> "<log>"
```

4. The run succeeded when the exit code is 0 and the log contains `BUILD_SAMPLE_MAPS OK`.
5. Copy `<host>/Plugins/ModelPipelines/Content/Samples` into the plugin repo's `Content/Samples` (`m5_samples_host.sh copyback`).

Runner options:

- `-Rhi offscreen` (default) passes `-RenderOffscreen`. Use this one: the 5.7 editor crashes under `-nullrhi` (a divide by zero in its own actor-factory loading when the sample actor is spawned).
- `-Rhi null` passes `-nullrhi`. Kept for completeness; do not use it with 5.7.
- `-OrientCheck <dir>` renders each vision board as seen from the PlayerStart into `<dir>` (needs a rendering RHI, so not `-Rhi null`).
- `-Log <path>` sets the editor log path.

The images folder and the orient-check folder reach the script through environment variables (`SAMPLE_MAPS_IMAGES` and
`SAMPLE_MAPS_ORIENT_DIR`), because paths with spaces do not survive the `-ExecutePythonScript` command-line parsing. The runner sets
them from `-Images` and `-OrientCheck`. The script reads them as the defaults of `--images` and `--orient-check`.

## Orient check

After a rebuild with `-OrientCheck <captures dir>`, compare the renders with the source photos using the Pillow-capable
Python (the micromamba env), not the editor's Python:

```
D:\Micromamba\micromambaenv\envs\pipelines\python.exe orient_compare.py --captures <captures dir> --images <dir with the PNGs>
```

For each photo it prints how well the capture matches each flip and rotation of the source PNG. The identity transform must win on
every photo. If not, change `BOARD_SPIN_DEG` (and `BOARD_MIRROR` if needed), rebuild and check again.

## Check the inputs without Unreal

```
python build_sample_maps.py --dry-run --images <dir>
```

This validates the PNGs, prints the planned assets and board placement, and touches nothing.
