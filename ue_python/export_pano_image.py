import os
import numpy as np
from PIL import Image
import OpenEXR
import Imath
import imageio.v3 as iio

FACE_KEYS = ["px", "nx", "py", "ny", "pz", "nz"]

# ============================
# 1) 每个面独立旋转/翻转配置
# ============================
# rot90: 0/1/2/3 -> 逆时针旋转 0/90/180/270
# flip_ud: 上下翻转
# flip_lr: 左右翻转
#
# 你现在图里最像是 “Y 方向面（py/ny）其中至少一个被上下翻了/旋转了”
# 先给一个我认为最可能的起手式：py/ny rot90=2 (180度) + flip_ud=True（等价于 rot90=2+flip_ud? 不完全等价）
# 你跑完如果更差，就把它们改回 0，再只开 flip_ud 或只改 rot90。
FACE_XFORM = {
    "px": dict(rot90=2, flip_ud=False, flip_lr=False),
    "nx": dict(rot90=2, flip_ud=False, flip_lr=False),

    # ====== 猜测重点在这两张 ======
    "py": dict(rot90=2, flip_ud=False,  flip_lr=False),
    "ny": dict(rot90=2, flip_ud=False,  flip_lr=False),

    "pz": dict(rot90=1, flip_ud=False, flip_lr=False),
    "nz": dict(rot90=3, flip_ud=False, flip_lr=False),
}


def transform_face(arr, rot90=0, flip_ud=False, flip_lr=False):
    """arr: (N,N,3) or (N,N)"""
    rot90 = int(rot90) % 4
    if rot90:
        arr = np.rot90(arr, k=rot90, axes=(0, 1))
    if flip_ud:
        arr = np.flipud(arr)
    if flip_lr:
        arr = np.fliplr(arr)
    return arr


# ============================
# 2) PNG RGB I/O
# ============================
def load_png_rgb_u8(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)

def save_png_rgb_u8(path: str, arr_u8: np.ndarray):
    Image.fromarray(arr_u8, mode="RGB").save(path)


# ============================
# 3) EXR I/O (Depth 单通道)
# ============================
def read_exr_all_channels(path: str) -> dict:
    exr = OpenEXR.InputFile(path)
    hdr = exr.header()
    dw = hdr["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    chans = list(hdr["channels"].keys())
    data = {}
    for c in chans:
        arr = np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(h, w)
        data[c] = arr
    return data

def pick_best_channel(ch_dict: dict):
    best = None
    best_var = -1.0
    for k, v in ch_dict.items():
        vv = v[np.isfinite(v)]
        if vv.size == 0:
            continue
        var = float(np.var(vv))
        if var > best_var:
            best_var = var
            best = k
    return best, best_var

def exr_to_single_depth(chs: dict, prefer_keys=("Z", "z", "Depth", "depth")):
    # 1) 常见深度通道优先
    for pk in prefer_keys:
        if pk in chs:
            return chs[pk].astype(np.float32), ("PREFER", pk)
    # 2) 否则：方差最大兜底
    k, var = pick_best_channel(chs)
    if k is None:
        raise RuntimeError("No finite channels found in EXR.")
    return chs[k].astype(np.float32), ("BESTVAR", k, var)

def write_exr_single_Y(path: str, y: np.ndarray):
    """写单通道 EXR：通道名用 'Y'"""
    assert y.ndim == 2
    h, w = y.shape
    header = OpenEXR.Header(w, h)
    header["channels"] = {"Y": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    out = OpenEXR.OutputFile(path, header)
    out.writePixels({"Y": y.astype(np.float32).tobytes()})
    out.close()

def save_preview_png_from_single(y: np.ndarray, out_png: str):
    """单通道预览：1/99 分位映射到 8-bit 灰度"""
    v = y.astype(np.float32)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        iio.imwrite(out_png, np.zeros_like(v, dtype=np.uint8))
        return

    lo, hi = np.nanpercentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    img01 = np.clip((v - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    img8 = (img01 * 255.0 + 0.5).astype(np.uint8)
    iio.imwrite(out_png, img8)


# ============================
# 4) Cubemap -> Equirect 核心（支持 RGB / 单通道）
# ============================
def _cubemap_dir_grid(out_w: int, out_h: int):
    # lon: [-pi, pi], lat: [-pi/2, pi/2]
    xs = (np.arange(out_w) + 0.5) / out_w
    ys = (np.arange(out_h) + 0.5) / out_h
    lon = (xs * 2.0 - 1.0) * np.pi
    lat = (0.5 - ys) * np.pi

    cos_lat = np.cos(lat)[:, None]
    sin_lat = np.sin(lat)[:, None]
    cos_lon = np.cos(lon)[None, :]
    sin_lon = np.sin(lon)[None, :]

    # UE-like world: X forward, Y right, Z up
    dx = cos_lat * cos_lon                 # (H,W)
    dy = cos_lat * sin_lon                 # (H,W)
    dz = sin_lat * np.ones_like(dx)        # ✅ (H,W) 修复 boolean mask 问题

    adx, ady, adz = np.abs(dx), np.abs(dy), np.abs(dz)

    mx = (adx >= ady) & (adx >= adz)
    my = (ady >  adx) & (ady >= adz)
    mz = (adz >  adx) & (adz >  ady)

    return dx, dy, dz, adx, ady, adz, mx, my, mz

def cubemap_to_equirect_rgb_u8(faces_u8: dict, out_w: int, out_h: int) -> np.ndarray:
    dx, dy, dz, adx, ady, adz, mx, my, mz = _cubemap_dir_grid(out_w, out_h)
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    def sample_face(face_key: str, u: np.ndarray, v: np.ndarray):
        face = faces_u8[face_key]
        n = face.shape[0]
        ix = ((u + 1.0) * 0.5 * (n - 1)).astype(np.int32)
        iy = ((v + 1.0) * 0.5 * (n - 1)).astype(np.int32)
        ix = np.clip(ix, 0, n - 1)
        iy = np.clip(iy, 0, n - 1)
        return face[iy, ix]

    # X faces
    m = mx & (dx > 0)   # +X
    if np.any(m):
        u = (-dy[m] / adx[m]).astype(np.float32)
        v = ( dz[m] / adx[m]).astype(np.float32)
        out[m] = sample_face("px", u, v)

    m = mx & (dx <= 0)  # -X
    if np.any(m):
        u = ( dy[m] / adx[m]).astype(np.float32)
        v = ( dz[m] / adx[m]).astype(np.float32)
        out[m] = sample_face("nx", u, v)

    # Y faces
    m = my & (dy > 0)   # +Y
    if np.any(m):
        u = ( dx[m] / ady[m]).astype(np.float32)
        v = ( dz[m] / ady[m]).astype(np.float32)
        out[m] = sample_face("py", u, v)

    m = my & (dy <= 0)  # -Y
    if np.any(m):
        u = (-dx[m] / ady[m]).astype(np.float32)
        v = ( dz[m] / ady[m]).astype(np.float32)
        out[m] = sample_face("ny", u, v)

    # Z faces
    m = mz & (dz > 0)   # +Z
    if np.any(m):
        u = ( dx[m] / adz[m]).astype(np.float32)
        v = (-dy[m] / adz[m]).astype(np.float32)
        out[m] = sample_face("pz", u, v)

    m = mz & (dz <= 0)  # -Z
    if np.any(m):
        u = ( dx[m] / adz[m]).astype(np.float32)
        v = ( dy[m] / adz[m]).astype(np.float32)
        out[m] = sample_face("nz", u, v)

    return out

def cubemap_to_equirect_single_float(faces_y: dict, out_w: int, out_h: int) -> np.ndarray:
    dx, dy, dz, adx, ady, adz, mx, my, mz = _cubemap_dir_grid(out_w, out_h)
    out = np.zeros((out_h, out_w), dtype=np.float32)

    def sample_face(face_key: str, u: np.ndarray, v: np.ndarray):
        face = faces_y[face_key]
        n = face.shape[0]
        ix = ((u + 1.0) * 0.5 * (n - 1)).astype(np.int32)
        iy = ((v + 1.0) * 0.5 * (n - 1)).astype(np.int32)
        ix = np.clip(ix, 0, n - 1)
        iy = np.clip(iy, 0, n - 1)
        return face[iy, ix]

    # X faces
    m = mx & (dx > 0)   # +X
    if np.any(m):
        u = (-dy[m] / adx[m]).astype(np.float32)
        v = ( dz[m] / adx[m]).astype(np.float32)
        out[m] = sample_face("px", u, v)

    m = mx & (dx <= 0)  # -X
    if np.any(m):
        u = ( dy[m] / adx[m]).astype(np.float32)
        v = ( dz[m] / adx[m]).astype(np.float32)
        out[m] = sample_face("nx", u, v)

    # Y faces
    m = my & (dy > 0)   # +Y
    if np.any(m):
        u = ( dx[m] / ady[m]).astype(np.float32)
        v = ( dz[m] / ady[m]).astype(np.float32)
        out[m] = sample_face("py", u, v)

    m = my & (dy <= 0)  # -Y
    if np.any(m):
        u = (-dx[m] / ady[m]).astype(np.float32)
        v = ( dz[m] / ady[m]).astype(np.float32)
        out[m] = sample_face("ny", u, v)

    # Z faces
    m = mz & (dz > 0)   # +Z
    if np.any(m):
        u = ( dx[m] / adz[m]).astype(np.float32)
        v = (-dy[m] / adz[m]).astype(np.float32)
        out[m] = sample_face("pz", u, v)

    m = mz & (dz <= 0)  # -Z
    if np.any(m):
        u = ( dx[m] / adz[m]).astype(np.float32)
        v = ( dy[m] / adz[m]).astype(np.float32)
        out[m] = sample_face("nz", u, v)

    return out


# ============================
# 5) 导出：RGB / Depth 分开
# ============================
def export_pano_rgb_from_png(in_dir: str, out_png: str, size: int = 1024, rgb_prefix="rgb"):
    faces = {}
    for k in FACE_KEYS:
        p = os.path.join(in_dir, f"{rgb_prefix}_{k}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        faces[k] = load_png_rgb_u8(p)
        faces[k] = transform_face(faces[k], **FACE_XFORM[k])

    # sanity
    n0 = faces["px"].shape[0]
    for k in FACE_KEYS:
        if faces[k].shape[:2] != (n0, n0):
            raise RuntimeError(f"RGB face size mismatch: {k}={faces[k].shape}, px={faces['px'].shape}")

    pano = cubemap_to_equirect_rgb_u8(faces, out_w=size, out_h=size // 2)
    save_png_rgb_u8(out_png, pano)
    print(f"[OK] RGB pano -> {out_png} shape={pano.shape} dtype={pano.dtype}")

def export_pano_depth_from_exr(in_dir: str, out_exr: str, size: int = 1024, depth_prefix="depth"):
    faces = {}
    info = {}

    for k in FACE_KEYS:
        p = os.path.join(in_dir, f"{depth_prefix}_{k}.exr")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        chs = read_exr_all_channels(p)
        y, sel = exr_to_single_depth(chs)
        faces[k] = y
        faces[k] = transform_face(faces[k], **FACE_XFORM[k])
        info[k] = sel

    # sanity
    n0 = faces["px"].shape[0]
    for k in FACE_KEYS:
        if faces[k].shape != (n0, n0):
            raise RuntimeError(f"Depth face size mismatch: {k}={faces[k].shape}, px={(n0,n0)}")

    print("[depth] channel selection:")
    for k in FACE_KEYS:
        print(" ", k, "->", info[k])

    pano_y = cubemap_to_equirect_single_float(faces, out_w=size, out_h=size // 2)
    write_exr_single_Y(out_exr, pano_y)
    print(f"[OK] Depth pano EXR(Y) -> {out_exr} shape={pano_y.shape} dtype={pano_y.dtype}")

    prev_png = os.path.splitext(out_exr)[0] + "_preview.png"
    save_preview_png_from_single(pano_y, prev_png)
    print(f"[OK] Depth pano preview -> {prev_png}")


if __name__ == "__main__":
    in_dir = r"E:\City_smaple_rendered\py_export"
    OUT_SIZE = 1024

    RGB_PREFIX = "rgb"      # rgb_px.png ...
    DEPTH_PREFIX = "depth"  # depth_px.exr ...

    export_pano_rgb_from_png(
        in_dir=in_dir,
        out_png=os.path.join(in_dir, f"pano_rgb_{OUT_SIZE}.png"),
        size=OUT_SIZE,
        rgb_prefix=RGB_PREFIX,
    )

    export_pano_depth_from_exr(
        in_dir=in_dir,
        out_exr=os.path.join(in_dir, f"pano_depth_{OUT_SIZE}.exr"),
        size=OUT_SIZE,
        depth_prefix=DEPTH_PREFIX,
    )

    print("[skip] normal pano not implemented yet.")