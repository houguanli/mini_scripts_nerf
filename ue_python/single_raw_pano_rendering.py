# pano_export_temp_capture_ue503.py
# UE 5.0.3
# 方案：
# - 保留场景中的 BP_PanoRig，只读取 Cap_* 的世界位姿
# - 真正 capture 时，临时生成 SceneCapture2D Actor
# - 对临时 Actor 的 capture component 进行配置
# - capture -> export -> destroy
#
# 这样可以绕开 BP 内嵌组件在 editor 中被重建成 TRASH_* 的问题

import os
import unreal

TAG_NAME   = "PanoRig"
OUTPUT_DIR = r"E:\City_smaple_rendered\py_export"
ORDER      = ["PX", "NX", "PY", "NY", "PZ", "NZ"]

CAP_NAMES = {
    "PX": "Cap_PX",
    "NX": "Cap_NX",
    "PY": "Cap_PY",
    "NY": "Cap_NY",
    "PZ": "Cap_PZ",
    "NZ": "Cap_NZ",
}

# ---------- RenderTarget assets ----------
RT_REFS_RGB = {
    "PX": "TextureRenderTarget2D'/Game/RT/RGB/RT_RGB_PX.RT_RGB_PX'",
    "NX": "TextureRenderTarget2D'/Game/RT/RGB/RT_RGB_NX.RT_RGB_NX'",
    "PY": "TextureRenderTarget2D'/Game/RT/RGB/RT_RGB_PY.RT_RGB_PY'",
    "NY": "TextureRenderTarget2D'/Game/RT/RGB/RT_RGB_NY.RT_RGB_NY'",
    "PZ": "TextureRenderTarget2D'/Game/RT/RGB/RT_RGB_PZ.RT_RGB_PZ'",
    "NZ": "TextureRenderTarget2D'/Game/RT/RGB/RT_RGB_NZ.RT_RGB_NZ'",
}

RT_REFS_N = {
    "PX": "TextureRenderTarget2D'/Game/RT/Normal/RT_N_PX.RT_N_PX'",
    "NX": "TextureRenderTarget2D'/Game/RT/Normal/RT_N_NX.RT_N_NX'",
    "PY": "TextureRenderTarget2D'/Game/RT/Normal/RT_N_PY.RT_N_PY'",
    "NY": "TextureRenderTarget2D'/Game/RT/Normal/RT_N_NY.RT_N_NY'",
    "PZ": "TextureRenderTarget2D'/Game/RT/Normal/RT_N_PZ.RT_N_PZ'",
    "NZ": "TextureRenderTarget2D'/Game/RT/Normal/RT_N_NZ.RT_N_NZ'",
}

RT_REFS_D = {
    "PX": "TextureRenderTarget2D'/Game/RT/Depth/RT_D_PX.RT_D_PX'",
    "NX": "TextureRenderTarget2D'/Game/RT/Depth/RT_D_NX.RT_D_NX'",
    "PY": "TextureRenderTarget2D'/Game/RT/Depth/RT_D_PY.RT_D_PY'",
    "NY": "TextureRenderTarget2D'/Game/RT/Depth/RT_D_NY.RT_D_NY'",
    "PZ": "TextureRenderTarget2D'/Game/RT/Depth/RT_D_PZ.RT_D_PZ'",
    "NZ": "TextureRenderTarget2D'/Game/RT/Depth/RT_D_NZ.RT_D_NZ'",
}


# ------------------ logging ------------------

def log(msg: str):
    unreal.log(f"[PanoExportTemp] {msg}")

def warn(msg: str):
    unreal.log_warning(f"[PanoExportTemp] {msg}")


# ------------------ world / actor helpers ------------------

def get_editor_world():
    try:
        ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        w = ues.get_editor_world()
        if w:
            return w
    except Exception:
        pass
    try:
        return unreal.EditorLevelLibrary.get_editor_world()
    except Exception:
        return None

def get_all_level_actors():
    try:
        eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        return eas.get_all_level_actors()
    except Exception:
        return unreal.EditorLevelLibrary.get_all_level_actors()

def find_actor_by_tag(tag: str):
    tag_name = unreal.Name(tag)
    for a in get_all_level_actors():
        try:
            if tag_name in a.tags:
                return a
        except Exception:
            continue
    return None

def get_component_by_name(actor, comp_name: str):
    """
    这里仍然从 BP_PanoRig 中找 Cap_*，
    但只读取 transform，不去改这些组件的属性。
    """
    try:
        comps = actor.get_components_by_class(unreal.SceneComponent)
    except Exception:
        return None

    # 精确匹配
    for c in comps:
        try:
            if c.get_name() == comp_name:
                return c
        except Exception:
            continue

    # 模糊匹配
    for c in comps:
        try:
            if comp_name.lower() in c.get_name().lower():
                return c
        except Exception:
            continue

    return None

def load_rt(ref: str):
    rt = unreal.load_object(None, ref)
    if not rt:
        raise RuntimeError(f"Failed to load RenderTarget: {ref}")
    return rt

def debug_print_rt(rt, label: str):
    try:
        fmt = rt.get_editor_property("render_target_format")
    except Exception:
        fmt = "N/A"
    try:
        srgb = rt.get_editor_property("srgb")
    except Exception:
        srgb = "N/A"
    try:
        sx = rt.get_editor_property("size_x")
        sy = rt.get_editor_property("size_y")
    except Exception:
        sx, sy = "N/A", "N/A"

    log(f"{label}: name={rt.get_name()}, fmt={fmt}, srgb={srgb}, size={sx}x{sy}")


# ------------------ temp capture actor helpers ------------------

def spawn_temp_capture_actor(world, location: unreal.Vector, rotation: unreal.Rotator, label: str):
    """
    在编辑器世界中临时生成一个 SceneCapture2D Actor
    """
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.SceneCapture2D,
        location,
        rotation
    )
    if not actor:
        raise RuntimeError(f"Failed to spawn temp SceneCapture2D for {label}")

    try:
        actor.set_actor_label(label)
    except Exception:
        pass

    return actor

def get_capture_component_from_actor(actor):
    comps = actor.get_components_by_class(unreal.SceneCaptureComponent2D)
    if not comps:
        raise RuntimeError(f"No SceneCaptureComponent2D found on temp actor: {actor.get_name()}")
    return comps[0]

def destroy_actor(actor):
    try:
        unreal.EditorLevelLibrary.destroy_actor(actor)
    except Exception as e:
        warn(f"Failed to destroy temp actor {actor.get_name()}: {e}")


# ------------------ capture configuration ------------------

def set_showflag(cap, flag_name: str, enabled: bool):
    try:
        sfs = list(cap.get_editor_property("show_flag_settings"))
    except Exception:
        sfs = []

    found = False
    for s in sfs:
        try:
            if str(s.show_flag_name).lower() == flag_name.lower():
                s.enabled = enabled
                found = True
                break
        except Exception:
            pass

    if not found:
        s = unreal.EngineShowFlagsSetting()
        s.show_flag_name = flag_name
        s.enabled = enabled
        sfs.append(s)

    cap.set_editor_property("show_flag_settings", sfs)

def clear_showflags(cap):
    """
    尽量避免前一轮模式残留。
    """
    try:
        cap.set_editor_property("show_flag_settings", [])
    except Exception:
        pass

def disable_auto_exposure(cap):
    pps = cap.post_process_settings
    try:
        pps.auto_exposure_method = unreal.AutoExposureMethod.AEM_MANUAL
    except Exception:
        pass
    try:
        pps.min_brightness = 1.0
        pps.max_brightness = 1.0
    except Exception:
        pass
    try:
        pps.auto_exposure_bias = 0.0
    except Exception:
        pass
    cap.post_process_settings = pps

def configure_common_capture(cap):
    cap.capture_every_frame = False
    cap.capture_on_movement = False
    cap.always_persist_rendering_state = True

def configure_rgb(cap, rt):
    """
    RGB:
    - 推荐 RT 用 RTF_RGBA16F + sRGB=False
    - 输出 exr
    """
    cap.texture_target = rt
    configure_common_capture(cap)
    clear_showflags(cap)

    cap.capture_source = unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR

    set_showflag(cap, "PostProcessing", True)
    set_showflag(cap, "Tonemapper", True)

    disable_auto_exposure(cap)

def configure_normal(cap, rt):
    """
    Normal:
    - 若项目渲染路径不支持，可能仍然黑
    """
    cap.texture_target = rt
    configure_common_capture(cap)
    clear_showflags(cap)

    cap.capture_source = unreal.SceneCaptureSource.SCS_NORMAL

    set_showflag(cap, "PostProcessing", False)
    set_showflag(cap, "Tonemapper", False)

def configure_depth(cap, rt, use_device_depth=False):
    cap.texture_target = rt
    configure_common_capture(cap)
    clear_showflags(cap)

    cap.capture_source = (
        unreal.SceneCaptureSource.SCS_DEVICE_DEPTH
        if use_device_depth else
        unreal.SceneCaptureSource.SCS_SCENE_DEPTH
    )

    set_showflag(cap, "PostProcessing", False)
    set_showflag(cap, "Tonemapper", False)

def debug_print_cap(cap, label: str):
    try:
        name = cap.get_name()
    except Exception:
        name = "<invalid>"
    try:
        source = cap.capture_source
    except Exception:
        source = "<unknown>"
    try:
        target_name = cap.texture_target.get_name() if cap.texture_target else "None"
    except Exception:
        target_name = "<invalid target>"

    log(f"{label}: cap={name}, capture_source={source}, texture_target={target_name}")


# ------------------ export ------------------

def export_rt(world, rt, out_dir: str, file_name: str):
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, file_name)

    try:
        unreal.RenderingLibrary.export_render_target(world, rt, out_dir, file_name)
    except Exception as e1:
        warn(f"4-arg export failed for {full_path}: {e1}")
        try:
            unreal.RenderingLibrary.export_render_target(world, rt, full_path)
        except Exception as e2:
            raise RuntimeError(
                f"Export failed: {full_path}\n"
                f"4-arg error: {e1}\n"
                f"3-arg error: {e2}"
            )

    exists = os.path.exists(full_path)
    log(f"Export check: exists={exists}, path={full_path}")


# ------------------ pose reading ------------------

def get_cap_pose(pano_actor, face_key: str):
    comp_name = CAP_NAMES[face_key]
    comp = get_component_by_name(pano_actor, comp_name)
    if not comp:
        raise RuntimeError(f"Cannot find reference component: {comp_name}")

    try:
        loc = comp.get_world_location()
    except Exception:
        loc = comp.get_component_location()

    try:
        rot = comp.get_world_rotation()
    except Exception:
        rot = comp.get_component_rotation()

    log(f"Reference pose {face_key}: loc={loc}, rot={rot}")
    return loc, rot


# ------------------ capture one face ------------------

def capture_one_face(world, pano_actor, face_key: str, mode: str):
    loc, rot = get_cap_pose(pano_actor, face_key)

    if mode == "rgb":
        rt = load_rt(RT_REFS_RGB[face_key])
        debug_print_rt(rt, f"RGB[{face_key}]")
        file_name = f"rgb_{face_key.lower()}.exr"
        actor_label = f"TempCap_RGB_{face_key}"
    elif mode == "normal":
        rt = load_rt(RT_REFS_N[face_key])
        debug_print_rt(rt, f"NORM[{face_key}]")
        file_name = f"normal_{face_key.lower()}.exr"
        actor_label = f"TempCap_N_{face_key}"
    elif mode == "depth":
        rt = load_rt(RT_REFS_D[face_key])
        debug_print_rt(rt, f"DEPTH[{face_key}]")
        file_name = f"depth_{face_key.lower()}.exr"
        actor_label = f"TempCap_D_{face_key}"
    else:
        raise ValueError(mode)

    temp_actor = spawn_temp_capture_actor(world, loc, rot, actor_label)
    cap = get_capture_component_from_actor(temp_actor)

    try:
        if mode == "rgb":
            configure_rgb(cap, rt)
        elif mode == "normal":
            configure_normal(cap, rt)
        elif mode == "depth":
            configure_depth(cap, rt, use_device_depth=False)

        debug_print_cap(cap, f"[{mode.upper()}][{face_key}] before capture")

        cap.capture_scene()
        log(f"Captured {mode.upper()} {face_key} via temp actor {temp_actor.get_name()} / {cap.get_name()}")

        export_rt(world, rt, OUTPUT_DIR, file_name)
        log(f"Exported: {os.path.join(OUTPUT_DIR, file_name)}")

    finally:
        destroy_actor(temp_actor)


# ------------------ main ------------------

def main():
    log("=== Start export with temporary SceneCapture2D actors (UE5.0.3) ===")

    world = get_editor_world()
    if not world:
        raise RuntimeError("Cannot get editor world.")

    pano_actor = find_actor_by_tag(TAG_NAME)
    if not pano_actor:
        raise RuntimeError(f"No actor found with tag '{TAG_NAME}'")

    log(f"Found pano actor: {pano_actor.get_name()}")

    # RGB
    for k in ORDER:
        capture_one_face(world, pano_actor, k, "rgb")

    # Normal
    for k in ORDER:
        capture_one_face(world, pano_actor, k, "normal")

    # Depth
    for k in ORDER:
        capture_one_face(world, pano_actor, k, "depth")

    log("=== Done ===")


if __name__ == "__main__":
    main()