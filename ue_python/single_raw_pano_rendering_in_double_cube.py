import os
import unreal

# ---------- 配置区 ----------
TAG_NAME = "PanoRig"
OUTPUT_DIR = r"E:\City_smaple_rendered\py_export_12shot"


# 仍然复用同一个 RT；你当前实践里能跑就先保持这个方案
RT_REFS = {
    "rgb": "TextureRenderTarget2D'/Game/RT/RGB/RT_RGB_PX.RT_RGB_PX'",
    "normal": "TextureRenderTarget2D'/Game/RT/Normal/RT_N_PX.RT_N_PX'",
    "depth": "TextureRenderTarget2D'/Game/RT/Depth/RT_D_PX.RT_D_PX'",
}

# ------------------------------------------------------------------
# 两套 cubemap：
# 第一套：标准 6 面
# 第二套：整体 yaw 偏移 45 度的 6 面
#
# 注意：
# - 这里不再使用你之前的 (45,45,0) / (-45,45,0)
# - top / bottom 也只是保持 pitch=±90，再加 yaw=45
# - 这样几何上是完整一致的第二套 cubemap
# ------------------------------------------------------------------
POSES_12 = [
    ("00_PX",   unreal.Rotator(0,   0, 0)),
    ("01_NX",   unreal.Rotator(180, 0, 0)),
    ("02_PY",   unreal.Rotator(90,  0, 0)),
    ("03_NY",   unreal.Rotator(270, 0, 0)),
    ("04_PZ",   unreal.Rotator(0, 0,  90)),
    ("05_NZ",   unreal.Rotator(0, 0, 270)),

    ("06_PX_O", unreal.Rotator(45, 0, 0)),
    ("07_NX_O", unreal.Rotator(225, 0, 0)),
    ("08_PY_O", unreal.Rotator(135, 0, 0)),
    ("09_NY_O", unreal.Rotator(315, 0, 0)),
    ("10_PZ_O", unreal.Rotator(45, 0, 90)),
    ("11_NZ_O", unreal.Rotator(45, 0, 270)),
]

MODES = ["rgb", "normal", "depth"]


# ------------------ logging ------------------

def log(msg: str):
    unreal.log(f"[Pano12Shot] {msg}")


def warn(msg: str):
    unreal.log_warning(f"[Pano12Shot] {msg}")


def err(msg: str):
    unreal.log_error(f"[Pano12Shot] {msg}")


# ------------------ 核心助手函数 (UE 5.0 适配版) ------------------

def get_subsystems():
    return {
        "actor": unreal.get_editor_subsystem(unreal.EditorActorSubsystem),
        "editor": unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem),
    }


def get_editor_world():
    try:
        subs = get_subsystems()
        world = subs["editor"].get_editor_world()
        if world:
            return world
    except Exception as e:
        warn(f"get_editor_world via subsystem failed: {e}")

    try:
        return unreal.EditorLevelLibrary.get_editor_world()
    except Exception as e:
        raise RuntimeError(f"Cannot get editor world: {e}")


def get_all_level_actors():
    try:
        subs = get_subsystems()
        return subs["actor"].get_all_level_actors()
    except Exception as e:
        warn(f"get_all_level_actors via subsystem failed: {e}")
        try:
            return unreal.EditorLevelLibrary.get_all_level_actors()
        except Exception as e2:
            raise RuntimeError(f"Cannot get level actors: {e2}")


def find_actor_by_tag(tag_name: str):
    target_tag = unreal.Name(tag_name)
    actors = get_all_level_actors()

    for a in actors:
        try:
            if target_tag in a.tags:
                return a
        except Exception:
            continue
    return None


def spawn_capture_actor(loc, rot):
    subs = get_subsystems()
    actor_sub = subs["actor"]

    try:
        actor = actor_sub.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)
    except Exception as e:
        warn(f"spawn_actor_from_class via subsystem failed: {e}")
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D, loc, rot)

    if not actor:
        raise RuntimeError("Failed to spawn SceneCapture2D actor")

    try:
        actor.set_editor_property("is_spatially_loaded", False)
    except Exception:
        pass

    return actor


def destroy_capture_actor(actor):
    if not actor:
        return

    try:
        get_subsystems()["actor"].destroy_actor(actor)
        return
    except Exception as e:
        warn(f"destroy via subsystem failed: {e}")

    try:
        unreal.EditorLevelLibrary.destroy_actor(actor)
    except Exception as e:
        warn(f"destroy via EditorLevelLibrary failed: {e}")


def load_rt(ref: str):
    rt = unreal.load_object(None, ref)
    if not rt:
        raise RuntimeError(f"Failed to load RenderTarget: {ref}")
    return rt


def get_capture_component(actor):
    try:
        comp = actor.get_component_by_class(unreal.SceneCaptureComponent2D)
        if comp:
            return comp
    except Exception:
        pass

    try:
        comps = actor.get_components_by_class(unreal.SceneCaptureComponent2D)
        if comps:
            return comps[0]
    except Exception:
        pass

    raise RuntimeError(f"No SceneCaptureComponent2D found on actor: {actor.get_name()}")


# ------------------ 配置 ------------------

def configure_cap(cap, mode, rt):
    """
    这里保持你当前能跑通的最简逻辑。
    不强行塞一堆 UE5.0.3 实测失效的设置。
    """
    cap.texture_target = rt
    cap.capture_every_frame = False
    cap.capture_on_movement = False

    try:
        cap.set_editor_property("fov_angle", 90.0)
    except Exception:
        pass

    try:
        cap.set_editor_property("show_flag_settings", [])
    except Exception:
        pass

    if mode == "rgb":

        cap.capture_source = unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR
    elif mode == "normal":
        cap.capture_source = unreal.SceneCaptureSource.SCS_NORMAL
    elif mode == "depth":
        cap.capture_source = unreal.SceneCaptureSource.SCS_SCENE_DEPTH
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ------------------ 工具 ------------------

def make_world_rot(base_rot: unreal.Rotator, rel_rot: unreal.Rotator):
    """
    仍然采用你当前可跑的欧拉角相加方案。
    """
    return unreal.Rotator(
        base_rot.pitch + rel_rot.pitch,
        base_rot.yaw + rel_rot.yaw,
        base_rot.roll + rel_rot.roll,
    )


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def export_rt(world, rt, render_dir: str, file_name: str):
    full_path = os.path.join(render_dir, file_name)

    try:
        unreal.RenderingLibrary.export_render_target(world, rt, render_dir, file_name)
    except Exception as e1:
        warn(f"4-arg export failed: {full_path} | {e1}")
        try:
            unreal.RenderingLibrary.export_render_target(world, rt, full_path)
        except Exception as e2:
            raise RuntimeError(
                f"Export failed for {full_path}\n"
                f"4-arg error: {e1}\n"
                f"3-arg error: {e2}"
            )

    exists = os.path.exists(full_path)
    log(f"Export check: exists={exists}, path={full_path}")


# ------------------ 执行 ------------------

def render_one_pose(world, base_loc, base_rot, render_dir, suffix, rel_rot, modes):
    world_rot = make_world_rot(base_rot, rel_rot)

    log(
        f"Rendering pose={suffix} | "
        f"base_rot=({base_rot.pitch:.3f},{base_rot.yaw:.3f},{base_rot.roll:.3f}) | "
        f"rel_rot=({rel_rot.pitch:.3f},{rel_rot.yaw:.3f},{rel_rot.roll:.3f}) | "
        f"world_rot=({world_rot.pitch:.3f},{world_rot.yaw:.3f},{world_rot.roll:.3f})"
    )

    for mode in modes:
        rt = None
        temp_actor = None

        try:
            rt = load_rt(RT_REFS[mode])
            log(f"Loaded RT for mode={mode}: {rt.get_name()}")

            temp_actor = spawn_capture_actor(base_loc, world_rot)
            log(f"Spawned temp actor: {temp_actor.get_name()}")

            cap = get_capture_component(temp_actor)
            configure_cap(cap, mode, rt)

            try:
                cap.capture_scene()
            except Exception as e:
                raise RuntimeError(f"capture_scene failed for {mode}/{suffix}: {e}")

            file_name = f"{mode}_{suffix}.exr"
            export_rt(world, rt, render_dir, file_name)
            log(f"Successfully exported: {file_name}")

        except Exception as e:
            err(f"Failed on mode={mode}, pose={suffix}: {e}")

        finally:
            destroy_capture_actor(temp_actor)


def render_named_pano_set(shot_name: str, modes=None):
    if modes is None:
        modes = MODES

    world = get_editor_world()
    if not world:
        raise RuntimeError("Cannot get editor world.")

    pano_actor = find_actor_by_tag(TAG_NAME)
    if not pano_actor:
        raise RuntimeError(f"Cannot find actor with tag '{TAG_NAME}'")

    try:
        base_loc = pano_actor.get_actor_location()
    except Exception:
        base_loc = pano_actor.get_root_component().get_world_location()

    try:
        base_rot = pano_actor.get_actor_rotation()
    except Exception:
        base_rot = pano_actor.get_root_component().get_world_rotation()

    render_dir = os.path.join(OUTPUT_DIR, shot_name)
    ensure_dir(render_dir)

    log("====================================================")
    log(f"Start rendering shot: {shot_name}")
    log(f"Output dir: {render_dir}")
    log(f"Pano actor: {pano_actor.get_name()}")
    log(f"Base loc: {base_loc}")
    log(f"Base rot: {base_rot}")
    log(f"Modes: {modes}")
    log("====================================================")

    for idx, (suffix, rel_rot) in enumerate(POSES_12):
        log(f"[{idx + 1}/{len(POSES_12)}] Begin pose: {suffix}")
        render_one_pose(
            world=world,
            base_loc=base_loc,
            base_rot=base_rot,
            render_dir=render_dir,
            suffix=suffix,
            rel_rot=rel_rot,
            modes=modes,
        )

    log("====================================================")
    log(f"Finished rendering shot: {shot_name}")
    log("====================================================")


if __name__ == "__main__":
    render_named_pano_set("shot_012")