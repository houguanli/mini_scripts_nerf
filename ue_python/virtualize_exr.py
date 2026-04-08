import os
import OpenEXR
import Imath
import numpy as np
import imageio.v3 as iio
import matplotlib.cm as cm


def read_exr_all_channels(path):
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


def pick_best_channel(ch_dict):
    """
    挑一个最可能用于可视化的通道：
    - 忽略全无效通道
    - 优先选择方差较大的
    """
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


def robust_normalize(
    depth,
    pmin=1,
    pmax=99,
    invert=True,
    invalid_fill=0.0,
):
    """
    稳健归一化到 [0, 1]
    - 仅基于有限值统计 percentiles
    - invalid 区域单独 mask
    - invert=True 时更接近常见 depth 可视化：近处更亮/更暖
    """
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth)

    if not np.any(valid):
        out = np.full_like(depth, invalid_fill, dtype=np.float32)
        return out, valid, np.nan, np.nan

    v = depth[valid]
    lo, hi = np.percentile(v, [pmin, pmax])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi <= lo:
            out = np.full_like(depth, 0.0, dtype=np.float32)
            out[~valid] = invalid_fill
            return out, valid, lo, hi

    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    if invert:
        norm[valid] = 1.0 - norm[valid]

    norm[~valid] = invalid_fill
    return norm, valid, float(lo), float(hi)


def colorize_depth(
    depth01,
    valid_mask=None,
    cmap_name="turbo",
    invalid_color=(0, 0, 0),
):
    """
    将 [0,1] 深度映射到伪彩色
    """
    cmap = cm.get_cmap(cmap_name)
    color = cmap(np.clip(depth01, 0.0, 1.0))[..., :3]  # HWC, float [0,1]
    color8 = (color * 255.0 + 0.5).astype(np.uint8)

    if valid_mask is not None:
        invalid = ~valid_mask
        color8[invalid] = np.array(invalid_color, dtype=np.uint8)

    return color8


def save_gray_preview(depth01, out_png):
    img8 = (np.clip(depth01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    iio.imwrite(out_png, img8)


def save_color_preview(depth01, valid_mask, out_png, cmap_name="turbo"):
    color8 = colorize_depth(depth01, valid_mask=valid_mask, cmap_name=cmap_name)
    iio.imwrite(out_png, color8)


def visualize_exr_depth(
    exr_path,
    out_dir=None,
    preferred_channel=None,
    cmap_name="turbo",
    invert=True,
    pmin=1,
    pmax=99,
):
    chs = read_exr_all_channels(exr_path)
    print("channels:", list(chs.keys()))

    if preferred_channel is not None:
        if preferred_channel not in chs:
            raise ValueError(f"preferred_channel={preferred_channel} not found in EXR channels")
        k = preferred_channel
        var = float(np.var(chs[k][np.isfinite(chs[k])])) if np.any(np.isfinite(chs[k])) else float("nan")
    else:
        k, var = pick_best_channel(chs)

    if k is None:
        raise RuntimeError("No valid channel found in EXR.")

    depth = chs[k]
    finite_vals = depth[np.isfinite(depth)]

    print(f"best channel: {k}")
    print(f"var: {var}")
    if finite_vals.size > 0:
        print(
            "raw min/max:",
            float(np.min(finite_vals)),
            float(np.max(finite_vals))
        )
    else:
        print("raw min/max: no finite values")

    depth01, valid_mask, lo, hi = robust_normalize(
        depth,
        pmin=pmin,
        pmax=pmax,
        invert=invert,
        invalid_fill=0.0,
    )

    print(f"normalize range: lo={lo}, hi={hi}, invert={invert}")

    if out_dir is None:
        out_dir = os.path.dirname(exr_path)
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(exr_path))[0]

    gray_png = os.path.join(out_dir, f"{stem}_{k}_gray_preview.png")
    color_png = os.path.join(out_dir, f"{stem}_{k}_{cmap_name}_preview.png")
    npy_path = os.path.join(out_dir, f"{stem}_{k}_norm.npy")
    raw_npy_path = os.path.join(out_dir, f"{stem}_{k}_raw.npy")

    save_gray_preview(depth01, gray_png)
    save_color_preview(depth01, valid_mask, color_png, cmap_name=cmap_name)

    np.save(npy_path, depth01.astype(np.float32))
    np.save(raw_npy_path, depth.astype(np.float32))

    print("wrote gray preview :", gray_png)
    print("wrote color preview:", color_png)
    print("wrote norm npy     :", npy_path)
    print("wrote raw npy      :", raw_npy_path)


if __name__ == "__main__":
    p = r"E:\City_smaple_rendered\py_export\depth_nx.exr"

    visualize_exr_depth(
        exr_path=p,
        out_dir=None,              # None = 存到 EXR 同目录
        preferred_channel=None,    # 例如 "R" / "Y"；不填就自动挑
        cmap_name="turbo",         # 可改: turbo / viridis / magma / plasma / inferno / jet
        invert=True,               # 常见 depth 预览更像 DepthAnything 的风格
        pmin=1,
        pmax=99,
    )