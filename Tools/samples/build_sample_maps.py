"""Build the five ModelPipelines sample maps (editor Python, UE 5.7; assets also load in 5.8).

Run inside the editor, normally through run_build_sample_maps.ps1:

    UnrealEditor-Cmd.exe <project> -ExecutePythonScript="<this file> --images <dir>" -unattended -nosplash -nullrhi

Everything under /ModelPipelines/Samples/ is deleted and recreated on every run (idempotent), except with --only:
then only the named map assets are deleted and recreated and textures, materials and other maps are left alone
(--only L_TalkToText needs no photos, so --images is optional). The runner passes --only through the SAMPLE_MAPS_ONLY
environment variable.

Outside the editor only `--dry-run` works: it validates the PNGs and prints the planned asset list.

Lines marked "# VERIFY:" are Python API names derived from the 5.7 C++ headers (UFUNCTION -> snake_case)
but not yet confirmed against `Intermediate/PythonStub/unreal.py` or a run.
"""

import argparse
import json
import math
import os
import struct
import sys
import traceback

# ---------------------------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------------------------

CONTENT_ROOT = "/ModelPipelines/Samples"
MAPS_DIR = CONTENT_ROOT + "/Maps"
TEXTURES_DIR = CONTENT_ROOT + "/Textures"
MATERIALS_DIR = CONTENT_ROOT + "/Materials"
MASTER_MATERIAL = "M_SamplePhoto"
PHOTO_PARAMETER = "Photo"  # contract C37: TextureSampleParameter2D named "Photo"

MAX_TEXTURE_PX = 1024

CLASSIFY_PHOTOS = ["classify_jellyfish", "classify_soccer_ball", "classify_violin", "classify_elephant"]
DETECT_PHOTOS = ["detect_dog_walk", "detect_street_market"]

# map name -> (python class name of the C++ sample actor, kind, photos)
MAPS = [
    ("L_SemanticSearch", "ModelPipelinesSemanticSearchSample", "text", []),
    ("L_Sentiment", "ModelPipelinesSentimentSample", "text", []),
    ("L_SceneClassifier", "ModelPipelinesSceneClassifierSample", "vision", CLASSIFY_PHOTOS),
    ("L_Detection", "ModelPipelinesDetectionSample", "vision", DETECT_PHOTOS),
    ("L_TalkToText", "ModelPipelinesTalkToTextSample", "speech", []),  # contract C82
]
# A "speech" map is laid out like a "text" map: PlayerStart, sun, sky light, floor, one sample actor, no boards.
FLOOR_KINDS = ("text", "speech")
ALL_PHOTOS = CLASSIFY_PHOTOS + DETECT_PHOTOS

# --- Layout (Unreal units; X forward, Y right, Z up; yaw 0 faces +X, yaw 90 faces +Y) ---
PLAYER_START_LOCATION = (0.0, 0.0, 100.0)
PLAYER_START_YAW = 90.0          # the player looks along +Y so that a row of boards runs along X
EYE_HEIGHT = 64.0                # DefaultPawn eye height above the PlayerStart (brief)
BOARD_DISTANCE = 350.0           # PlayerStart -> board along the view direction
BOARD_MAX_WIDTH = 400.0
BOARD_MAX_HEIGHT = 300.0         # keeps portrait photos inside a 16:9 view at FOV 90 (visible height ~394 uu at 350 uu)
BOARD_SPACING = 500.0            # centre-to-centre along X
FLOOR_SCALE = (50.0, 50.0, 1.0)  # /Engine/BasicShapes/Plane is 100x100 uu -> 5000x5000 uu floor
SAMPLE_ACTOR_LOCATION = (-200.0, 0.0, 100.0)

# /Engine/BasicShapes/Plane: 100x100 uu in local XY, normal +Z.
# Standing it up: Rotator(pitch=90, yaw=PLAYER_START_YAW) maps local +Z (normal) to -Y (towards the player at
# yaw 90), local +X to world up (+Z) and local +Y to the player's right (-X when yaw is 90).
# UNVERIFIED (part 2 checks with --orient-check): which plane UV axis points up/right and whether the image is
# mirrored. These three constants are what you change:
#   BOARD_SPIN_DEG      rotation about the plane's own normal applied BEFORE standing it up (0, 90, 180, 270).
#                       With 0 or 180 the photo WIDTH lies along local Y; with 90 or 270 along local X.
#   BOARD_MIRROR        True -> negate the scale on the axis that carries the photo width (mirror fix).
#   BOARD_STAND_PITCH   90 stands the plane up with its +Z normal facing the player.
BOARD_SPIN_DEG = 90.0
BOARD_MIRROR = False
BOARD_STAND_PITCH = 90.0

LEVEL_ACTOR_CLASSES_TO_CLEAR = [
    "DirectionalLight", "SkyLight", "SkyAtmosphere", "ExponentialHeightFog", "VolumetricCloud",
    "PlayerStart", "StaticMeshActor", "PostProcessVolume", "AtmosphericFog",
]


def log(msg):
    print("[build_sample_maps] " + str(msg), flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Pure Python helpers (testable outside the editor)
# ---------------------------------------------------------------------------------------------------------------

def png_size(path):
    """Return (width, height) from a PNG header. No third-party imports (Pillow is not in the editor's Python)."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError("not a PNG: " + path)
    return struct.unpack(">II", head[16:24])


def default_images_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    # <repo>/ModelPipelines-UE/Tools/samples -> <workspace>/.pipelines-dev/sample_images
    return os.path.normpath(os.path.join(here, "..", "..", "..", ".pipelines-dev", "sample_images"))


def select_maps(only):
    """Maps to build. `only` is a list of map names (None = all). Unknown names raise."""
    if not only:
        return list(MAPS)
    known = [m[0] for m in MAPS]
    bad = [n for n in only if n not in known]
    if bad:
        raise RuntimeError("unknown map(s) for --only: %s (known: %s)" % (", ".join(bad), ", ".join(known)))
    return [m for m in MAPS if m[0] in only]


def parse_only(values):
    """--only values: repeatable and/or comma separated. Returns a de-duplicated list, or None."""
    names = []
    for v in values or []:
        for n in str(v).replace(";", ",").split(","):
            n = n.strip()
            if n and n not in names:
                names.append(n)
    return names or None


def photos_needed(maps):
    out = []
    for _m, _c, _k, photos in maps:
        for p in photos:
            if p not in out:
                out.append(p)
    return out


def validate_images(images_dir, names=None):
    """Check the PNGs (all six, or `names`) exist and are <= MAX_TEXTURE_PX. Returns {name: (w, h, path)}."""
    out = {}
    problems = []
    for name in (ALL_PHOTOS if names is None else names):
        path = os.path.join(images_dir, name + ".png")
        if not os.path.isfile(path):
            problems.append("missing: " + path)
            continue
        try:
            w, h = png_size(path)
        except Exception as exc:  # noqa: BLE001 - report every bad file
            problems.append("%s: %s" % (path, exc))
            continue
        if max(w, h) > MAX_TEXTURE_PX:
            problems.append("%s is %dx%d, larger than %d px (run fetch_sample_images.py)" % (path, w, h, MAX_TEXTURE_PX))
            continue
        out[name] = (w, h, path)
    if problems:
        raise RuntimeError("sample images not usable:\n  " + "\n  ".join(problems))
    return out


def board_size(w_px, h_px):
    """Board (width, height) in uu: fit the photo inside BOARD_MAX_WIDTH x BOARD_MAX_HEIGHT, keeping aspect."""
    scale = min(BOARD_MAX_WIDTH / float(w_px), BOARD_MAX_HEIGHT / float(h_px))
    return w_px * scale, h_px * scale


def board_layout(photos, sizes):
    """Per photo: dict(name, center=(x,y,z), width, height, scale=(sx,sy,sz)). Boards are in a row along X.

    The player stands at PLAYER_START_LOCATION looking along +Y (yaw 90), so the right-hand side is -X.
    The first photo is on the player's left (+X), the last on the right (-X).
    """
    n = len(photos)
    spin = int(round(BOARD_SPIN_DEG)) % 360
    width_on_local_y = spin in (0, 180)
    out = []
    for i, name in enumerate(photos):
        w_uu, h_uu = board_size(sizes[name][0], sizes[name][1])
        x = ((n - 1) / 2.0 - i) * BOARD_SPACING + PLAYER_START_LOCATION[0]
        y = PLAYER_START_LOCATION[1] + BOARD_DISTANCE
        z = PLAYER_START_LOCATION[2] + EYE_HEIGHT
        sw = w_uu / 100.0 * (-1.0 if BOARD_MIRROR else 1.0)  # scale on the axis carrying the width
        sh = h_uu / 100.0
        # After standing the plane up: local X is up, local Y is right (see constants above).
        # width_on_local_y False (spin 90/270): the spun plane has its local X axis along the screen horizontal.
        scale = (sh, sw, 1.0) if width_on_local_y else (sw, sh, 1.0)
        out.append(dict(name=name, center=(x, y, z), width=w_uu, height=h_uu, scale=scale))
    return out


def planned_assets(sizes, maps=None, full=True):
    """Planned asset list. full=False (--only): the named maps only, textures and materials stay untouched."""
    items = []
    for m, cls, kind, photos in (MAPS if maps is None else maps):
        items.append("%s/%s  (map, %s, actor unreal.%s)" % (MAPS_DIR, m, kind, cls))
    if not full:
        return items
    for name in ALL_PHOTOS:
        items.append("%s/T_%s  (texture %dx%d)" % (TEXTURES_DIR, name, sizes[name][0], sizes[name][1]))
    items.append("%s/%s  (material, unlit, parameter \"%s\")" % (MATERIALS_DIR, MASTER_MATERIAL, PHOTO_PARAMETER))
    for name in ALL_PHOTOS:
        items.append("%s/MI_%s  (material instance)" % (MATERIALS_DIR, name))
    return items


# ---------------------------------------------------------------------------------------------------------------
# Editor part
# ---------------------------------------------------------------------------------------------------------------

def vec(unreal, t):
    return unreal.Vector(t[0], t[1], t[2])


def import_texture(unreal, asset_tools, name, png_path):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", png_path)
    task.set_editor_property("destination_path", TEXTURES_DIR)
    task.set_editor_property("destination_name", "T_" + name)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", False)
    asset_tools.import_asset_tasks([task])
    tex = unreal.EditorAssetLibrary.load_asset("%s/T_%s" % (TEXTURES_DIR, name))
    if tex is None:
        raise RuntimeError("texture import failed: " + png_path)
    tex.set_editor_property("srgb", True)
    # TC_DEFAULT = BC1/BC3: small on disk, any format is readable through a scene capture.
    tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_DEFAULT)  # VERIFY: enum name
    tex.set_editor_property("mip_gen_settings", unreal.TextureMipGenSettings.TMGS_FROM_TEXTURE_GROUP)  # VERIFY: enum name
    tex.set_editor_property("lod_group", unreal.TextureGroup.TEXTUREGROUP_UI)  # VERIFY: enum name
    tex.set_editor_property("max_texture_size", MAX_TEXTURE_PX)  # VERIFY: property exists (UTexture::MaxTextureSize)
    if not unreal.EditorAssetLibrary.save_loaded_asset(tex, False):
        raise RuntimeError("could not save texture T_" + name)
    return tex


def create_master_material(unreal, asset_tools):
    mel = unreal.MaterialEditingLibrary
    mat = asset_tools.create_asset(MASTER_MATERIAL, MATERIALS_DIR, unreal.Material, unreal.MaterialFactoryNew())
    if mat is None:
        raise RuntimeError("could not create " + MASTER_MATERIAL)
    # Unlit: Material::ShadingModel is a private UPROPERTY(EditAnywhere); Python exposes it as "shading_model".
    # VERIFY: set_editor_property("shading_model", ...) works and rebuilds ShadingModels (else use a post-edit call).
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)  # VERIFY: enum name MaterialShadingModel.MSM_UNLIT
    mat.set_editor_property("two_sided", True)
    expr = mel.create_material_expression(mat, unreal.MaterialExpressionTextureSampleParameter2D, -400, 0)
    expr.set_editor_property("parameter_name", PHOTO_PARAMETER)
    # Default texture for the parameter, so the master compiles on its own (engine content, referenced only).
    default_tex = unreal.EditorAssetLibrary.load_asset("/Engine/EngineResources/WhiteSquareTexture")  # VERIFY: path loads
    if default_tex is not None:
        expr.set_editor_property("texture", default_tex)
    # VERIFY: output name "RGB" for TextureSample (alternatives: "" or "Output"); and MaterialProperty.MP_EMISSIVE_COLOR
    ok = mel.connect_material_property(expr, "RGB", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    if not ok:
        raise RuntimeError("connect_material_property(RGB -> Emissive) failed")
    mel.recompile_material(mat)
    if not unreal.EditorAssetLibrary.save_loaded_asset(mat, False):
        raise RuntimeError("could not save " + MASTER_MATERIAL)
    return mat


def create_instance(unreal, asset_tools, master, name, tex):
    mel = unreal.MaterialEditingLibrary
    mi = asset_tools.create_asset("MI_" + name, MATERIALS_DIR, unreal.MaterialInstanceConstant,
                                  unreal.MaterialInstanceConstantFactoryNew())
    if mi is None:
        raise RuntimeError("could not create MI_" + name)
    mi.set_editor_property("parent", master)
    # The engine's SetMaterialInstanceTextureParameterValue always returns false (MaterialEditingLibrary.cpp:1195),
    # so ignore the result and read the value back.
    mel.set_material_instance_texture_parameter_value(mi, PHOTO_PARAMETER, tex)
    if mel.get_material_instance_texture_parameter_value(mi, PHOTO_PARAMETER) != tex:
        raise RuntimeError("could not set Photo on MI_" + name)
    mel.update_material_instance(mi)
    if not unreal.EditorAssetLibrary.save_loaded_asset(mi, False):
        raise RuntimeError("could not save MI_" + name)
    return mi


def clear_default_actors(unreal, actor_ss):
    """NewLevel may add template actors; remove the ones we create ourselves so counts are exact."""
    removed = []
    for a in actor_ss.get_all_level_actors():
        cname = a.get_class().get_name()
        if cname in LEVEL_ACTOR_CLASSES_TO_CLEAR:
            removed.append(a.get_actor_label() + ":" + cname)
            actor_ss.destroy_actor(a)
    if removed:
        log("  removed template actors: " + ", ".join(removed))


def board_rotation(unreal):
    """Rotation that stands the plane up facing the player, after spinning it about its own normal."""
    stand = unreal.Rotator(roll=0.0, pitch=BOARD_STAND_PITCH, yaw=PLAYER_START_YAW)  # VERIFY: Rotator kwargs roll/pitch/yaw
    if int(round(BOARD_SPIN_DEG)) % 360 == 0:
        return stand
    spin = unreal.Rotator(roll=0.0, pitch=0.0, yaw=BOARD_SPIN_DEG)
    return unreal.MathLibrary.compose_rotators(spin, stand)  # VERIFY: name compose_rotators (A applied first, then B)


def set_movable(unreal, actor):
    root = actor.get_editor_property("root_component")
    root.set_mobility(unreal.ComponentMobility.MOVABLE)  # VERIFY: ComponentMobility.MOVABLE, set_mobility exposed


def build_map(unreal, map_name, cls_name, kind, photos, mis, sizes, plane):
    level_ss = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    actor_ss = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    path = "%s/%s" % (MAPS_DIR, map_name)
    log("building " + path)
    if not level_ss.new_level(path):
        raise RuntimeError("new_level failed: " + path)
    clear_default_actors(unreal, actor_ss)

    # WorldSettings.DefaultGameMode = GameModeBase (DefaultPawn: WASD + mouse look).
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()  # VERIFY: subsystem + method name
    ws = world.get_world_settings()  # VERIFY: UWorld::K2_GetWorldSettings ScriptName "GetWorldSettings"
    ws.set_editor_property("default_game_mode", unreal.GameModeBase.static_class())
    if ws.get_editor_property("default_game_mode") != unreal.GameModeBase.static_class():
        raise RuntimeError("DefaultGameMode override did not stick")

    ps = actor_ss.spawn_actor_from_class(unreal.PlayerStart, vec(unreal, PLAYER_START_LOCATION),
                                         unreal.Rotator(roll=0.0, pitch=0.0, yaw=PLAYER_START_YAW))
    ps.set_actor_label("PlayerStart")

    sun = actor_ss.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 500),
                                          unreal.Rotator(roll=0.0, pitch=-50.0, yaw=30.0))
    sun.set_actor_label("Sun")
    set_movable(unreal, sun)  # static lights show nothing without a lighting build
    sky = actor_ss.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 400), unreal.Rotator())
    sky.set_actor_label("SkyLight")
    set_movable(unreal, sky)

    sample_cls = getattr(unreal, cls_name, None)
    if sample_cls is None:
        raise RuntimeError("unreal.%s not found: is the ModelPipelinesSamples module built and loaded?" % cls_name)
    sample = actor_ss.spawn_actor_from_class(sample_cls, vec(unreal, SAMPLE_ACTOR_LOCATION), unreal.Rotator())
    if sample is None:
        raise RuntimeError("could not spawn " + cls_name)
    sample.set_actor_label(cls_name)

    if kind in FLOOR_KINDS:
        floor = actor_ss.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator())
        floor.set_actor_label("Floor")
        floor.set_actor_scale3d(vec(unreal, FLOOR_SCALE))  # VERIFY: set_actor_scale3d
        floor.get_editor_property("static_mesh_component").set_static_mesh(plane)
    else:
        for b in board_layout(photos, sizes):
            board = actor_ss.spawn_actor_from_class(unreal.StaticMeshActor, vec(unreal, b["center"]),
                                                    board_rotation(unreal))
            board.set_actor_label("Board_" + b["name"])
            board.set_actor_scale3d(vec(unreal, b["scale"]))
            comp = board.get_editor_property("static_mesh_component")
            comp.set_static_mesh(plane)
            comp.set_material(0, mis[b["name"]])
            e = board.get_actor_location(), board.get_actor_rotation(), board.get_actor_scale3d()
            log("  board %s loc=%s rot=%s scale=%s" % ((b["name"],) + tuple(str(x) for x in e)))

    log("  PlayerStart loc=%s rot=%s" % (ps.get_actor_location(), ps.get_actor_rotation()))
    # Actor set check for this map: the sample actor, PlayerStart, Sun, SkyLight, and the floor or the boards.
    labels = sorted(a.get_actor_label() for a in actor_ss.get_all_level_actors())
    want = ["PlayerStart", "Sun", "SkyLight", cls_name]
    want += ["Floor"] if kind in FLOOR_KINDS else ["Board_" + n for n in photos]
    missing = [w for w in want if w not in labels]
    if missing:
        raise RuntimeError("%s is missing actors: %s (has: %s)" % (map_name, ", ".join(missing), ", ".join(labels)))
    if not level_ss.save_current_level():
        raise RuntimeError("save_current_level failed: " + path)


def orient_check(unreal, out_dir, map_name, photos, sizes):
    """Render what the PlayerStart sees (needs a real RHI, not -nullrhi) and export it as an image.

    Part 2 compares the exported image with the source PNG (flip / mirror / rotate correlation) in Python.
    Does not save the level.
    """
    os.makedirs(out_dir, exist_ok=True)
    level_ss = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    actor_ss = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not level_ss.load_level("%s/%s" % (MAPS_DIR, map_name)):
        raise RuntimeError("load_level failed for orient check")
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()  # after load_level: the old world is gone
    rt = unreal.RenderingLibrary.create_render_target2d(  # VERIFY: name create_render_target2d
        world, 768, 432, unreal.TextureRenderTargetFormat.RTF_RGBA8)  # VERIFY: enum name
    eye = unreal.Vector(PLAYER_START_LOCATION[0], PLAYER_START_LOCATION[1], PLAYER_START_LOCATION[2] + EYE_HEIGHT)
    layout = board_layout(photos, sizes)
    cap = actor_ss.spawn_actor_from_class(unreal.SceneCapture2D, eye,
                                          unreal.Rotator(roll=0.0, pitch=0.0, yaw=PLAYER_START_YAW))
    try:
        comp = cap.get_editor_property("capture_component2d")  # VERIFY: property name
        comp.set_editor_property("texture_target", rt)
        comp.set_editor_property("fov_angle", 90.0)
        comp.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)  # VERIFY: enum name
        for i, b in enumerate(layout):
            # Aim straight at board i by moving the capture along X (the view direction stays +Y).
            cap.set_actor_location(unreal.Vector(b["center"][0], eye.y, eye.z), False, False)  # VERIFY: signature
            comp.capture_scene()
            unreal.RenderingLibrary.export_render_target(world, rt, out_dir, "orient_%s_%d.png" % (map_name, i))  # non-HDR names are written as PNG
            log("  orient capture written for %s -> %s" % (b["name"], out_dir))
    finally:
        actor_ss.destroy_actor(cap)


def list_assets(unreal):
    return sorted(unreal.EditorAssetLibrary.list_assets(CONTENT_ROOT, True, False))


def run_editor(args, sizes):
    import unreal  # only exists inside the editor

    try:
        log("engine version: %s" % unreal.SystemLibrary.get_engine_version())  # VERIFY: exists in 5.7 Python
    except Exception:  # noqa: BLE001 - version line is informational
        pass

    eal = unreal.EditorAssetLibrary
    only = parse_only(args.only)
    maps = select_maps(only)
    if only:
        # Rebuild only the named maps: delete just those map assets, keep everything else.
        eal.make_directory(MAPS_DIR)
        for map_name, _c, _k, _p in maps:
            asset = "%s/%s" % (MAPS_DIR, map_name)
            if eal.does_asset_exist(asset):
                log("deleting " + asset)
                if not eal.delete_asset(asset):
                    raise RuntimeError("delete_asset failed: " + asset)
    else:
        if eal.does_directory_exist(CONTENT_ROOT):
            log("deleting " + CONTENT_ROOT)
            if not eal.delete_directory(CONTENT_ROOT):
                raise RuntimeError("delete_directory failed: " + CONTENT_ROOT)
        for d in (MAPS_DIR, TEXTURES_DIR, MATERIALS_DIR):
            if not eal.make_directory(d):
                raise RuntimeError("make_directory failed: %s (is the /ModelPipelines/ plugin Content folder mounted?)" % d)

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    plane = eal.load_asset("/Engine/BasicShapes/Plane")
    if plane is None:
        raise RuntimeError("/Engine/BasicShapes/Plane not found")

    mis = {}
    if not only:
        textures = {n: import_texture(unreal, asset_tools, n, sizes[n][2]) for n in ALL_PHOTOS}
        master = create_master_material(unreal, asset_tools)
        mis = {n: create_instance(unreal, asset_tools, master, n, textures[n]) for n in ALL_PHOTOS}
    else:
        # Photo maps rebuilt on their own reuse the existing material instances.
        for n in photos_needed(maps):
            mi = eal.load_asset("%s/MI_%s" % (MATERIALS_DIR, n))
            if mi is None:
                raise RuntimeError("MI_%s not found: run once without --only to create textures and materials" % n)
            mis[n] = mi

    for map_name, cls_name, kind, photos in maps:
        build_map(unreal, map_name, cls_name, kind, photos, mis, sizes, plane)

    if args.orient_check:
        for map_name, _cls, kind, photos in maps:
            if kind == "vision":
                orient_check(unreal, args.orient_check, map_name, photos, sizes)

    assets = list_assets(unreal)
    log("assets under %s (%d):" % (CONTENT_ROOT, len(assets)))
    for a in assets:
        log("  " + a)
    if only:
        # Verify only the rebuilt maps.
        for map_name, _c, _k, _p in maps:
            want = "%s/%s" % (MAPS_DIR, map_name)
            if not any(a.split(".")[0] == want for a in assets):
                raise RuntimeError("rebuilt map not found: " + want)
        log("rebuilt %d map(s): %s" % (len(maps), ", ".join(m[0] for m in maps)))
        return
    expected = len(MAPS) + len(ALL_PHOTOS) + 1 + len(ALL_PHOTOS)
    if len(assets) != expected:
        raise RuntimeError("expected %d assets, found %d" % (expected, len(assets)))


def main(argv):
    p = argparse.ArgumentParser(description="Build the ModelPipelines sample maps (run inside the editor).")
    p.add_argument("--images", default=os.environ.get("SAMPLE_MAPS_IMAGES") or default_images_dir(), help="folder with the six sample PNGs (optional when --only needs no photos)")
    p.add_argument("--only", action="append", metavar="MapName",
                   default=[os.environ["SAMPLE_MAPS_ONLY"]] if os.environ.get("SAMPLE_MAPS_ONLY") else [],
                   help="rebuild only these maps (repeatable or comma list); other assets stay untouched")
    p.add_argument("--dry-run", action="store_true",
                   help="validate inputs and print the planned asset list; touches no assets (works outside the editor)")
    p.add_argument("--orient-check", metavar="DIR", default=os.environ.get("SAMPLE_MAPS_ORIENT_DIR") or None,
                   help="after building, render the view of each vision board to DIR (needs a real RHI)")
    args = p.parse_args(argv)
    try:
        only = parse_only(args.only)
        maps = select_maps(only)
        sizes = validate_images(args.images, photos_needed(maps) if only else None)
        if only:
            log("only: " + ", ".join(m[0] for m in maps))
        if sizes:
            log("images: " + args.images)
        for line in planned_assets(sizes, maps, full=not only):
            log("plan: " + line)
        if args.dry_run:
            for m, _c, kind, photos in maps:
                for b in board_layout(photos, sizes):
                    log("plan: %s board %s center=%s scale=%s" % (m, b["name"], b["center"], b["scale"]))
        else:
            run_editor(args, sizes)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - the runner needs the traceback and a non-zero exit
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
    print("BUILD_SAMPLE_MAPS OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
