#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NEUS -> NeRF(Blender-style) 一键打包
输入目录结构：
  dataset_dir/
    ├─ image/                 # 必需
    ├─ mask/                  # 可选，本脚本不写入 transforms
    └─ camera_sphere.npz      # 必需，含 world_mat_0..N

输出目录结构(默认):
  dataset_dir/nerf/<case_name>/
    ├─ train/                 # 复制自 image/
    ├─ test/                  # 复制自 image/
    ├─ transforms_train.json
    └─ transforms_test.json
"""

import os
import json
import glob
import shutil
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# === 你自己的实现：请确保可导入 ===
# from yourmodule import load_K_Rt_from_P
# 为了示意，这里给一个占位符；实际请用你的实现覆盖掉此函数！
def load_K_Rt_from_P(dummy, P34):
    raise NotImplementedError("请导入你自己的 load_K_Rt_from_P 实现。")

def col_normalize_R(R):
    Rn = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)
    return Rn

def cv_w2c_to_nerf_c2w(T_w2c_cv, normalize_R=True, svd_orthogonalize=False):
    """
    与你 read_cameras_from_nerf_json 的逆过程保持一致：
      c2w = inv(w2c_cv) 后，对 y、z 两列取反，得到 NeRF/Blender 坐标系的 c2w。
    """
    c2w = np.linalg.inv(T_w2c_cv).copy()
    if svd_orthogonalize:
        U, _, Vt = np.linalg.svd(c2w[:3, :3])
        c2w[:3, :3] = U @ Vt
    elif normalize_R:
        c2w[:3, :3] = col_normalize_R(c2w[:3, :3])

    c2w[:3, 1] *= -1.0  # Y 轴取反
    c2w[:3, 2] *= -1.0  # Z 轴取反
    return c2w

def fx_to_fovx(fx, W):
    return float(2.0 * np.arctan(0.5 * W / float(fx)))

def build_frames_from_npz(dataset_dir, images_rel="image", npz_name="camera_sphere.npz",
                          normalize_R=True, svd_orthogonalize=False):
    """
    读取 camera_sphere.npz 的 world_mat_i，生成 NeRF 的 frames（不带子目录前缀）。
    返回：frames(list[dict])、W、H、fx_list
    注意：此处 file_path 先用图像文件名（不含 train/test 前缀），
         在写 transforms_{split}.json 时再加上 'train/' 或 'test/' 前缀。
    """
    img_dir = Path(dataset_dir) / images_rel
    npz_path = Path(dataset_dir) / npz_name

    img_paths = sorted([p for p in img_dir.glob("*") if p.is_file()])
    if not img_paths:
        raise FileNotFoundError(f"No images under {img_dir}")

    with Image.open(img_paths[0]) as im0:
        W, H = im0.size

    cam = np.load(str(npz_path))
    world_keys = sorted([k for k in cam.files if k.startswith("world_mat_")],
                        key=lambda s: int(s.split("_")[-1]))
    if not world_keys:
        raise ValueError("No 'world_mat_*' found in npz")

    n_frames = min(len(img_paths), len(world_keys))
    img_paths = img_paths[:n_frames]
    world_keys = world_keys[:n_frames]

    frames = []
    fx_list = []
    for k, img_p in zip(world_keys, img_paths):
        P = cam[k]  # 4x4 或更大；只取 3x4
        K, T_w2c_cv = load_K_Rt_from_P("None", P[:3, :4])

        T_c2w_nerf = cv_w2c_to_nerf_c2w(
            T_w2c_cv, normalize_R=normalize_R, svd_orthogonalize=svd_orthogonalize
        )

        fx = float(K[0, 0])
        fx_list.append(fx)

        # 暂存相对文件名（basename），后续按 split 加前缀
        frames.append({
            "file_path": img_p.name,  # 先只放文件名
            "transform_matrix": T_c2w_nerf.tolist()
        })

    return frames, W, H, fx_list

def write_transforms_json(out_dir, frames, W, H, fx_list, split_name):
    """
    将 frames 的 file_path 前加上 split 前缀（train/ 或 test/），写入 transforms_{split}.json
    """
    frames_split = []
    for fr in frames:
        fr2 = dict(fr)
        fr2["file_path"] = f"{split_name}/{fr['file_path']}"
        frames_split.append(fr2)

    fx_med = float(np.median(fx_list))
    camera_angle_x = fx_to_fovx(fx_med, W)

    out = {
        "camera_angle_x": camera_angle_x,
        "w": int(W),
        "h": int(H),
        # 如果你需要更明确的内参，也可以加上：
        # "fl_x": fx_med, "fl_y": fx_med, "cx": W/2.0, "cy": H/2.0,
        "frames": frames_split
    }

    out_path = Path(out_dir) / f"transforms_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Saved {out_path}")

def copy_images_to_splits(src_image_dir, dst_root_dir):
    """
    将 image/ 复制到 dst_root_dir/train 与 dst_root_dir/test
    若目录存在则先删除再复制，保证与源一致。
    """
    for split in ("train", "test"):
        dst_dir = Path(dst_root_dir) / split
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_image_dir, dst_dir)
        print(f"[OK] Copied images -> {dst_dir}")

def main():
    parser = argparse.ArgumentParser(description="Pack NEUS dataset to NeRF format.")
    parser.add_argument("dataset_dir", type=str, help="NEUS 数据集根目录（含 image/ 和 camera_sphere.npz）")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="输出目录（默认：<dataset_dir>/nerf/<case_name>）")
    parser.add_argument("--images_rel", type=str, default="image", help="图片子目录名，默认 image")
    parser.add_argument("--npz_name", type=str, default="camera_sphere.npz", help="相机 npz 文件名")
    parser.add_argument("--svd_orthogonalize", action="store_true",
                        help="使用 SVD 正交化旋转（默认只做列归一化）")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    case_name = dataset_dir.name
    default_out_root = dataset_dir / "nerf" / case_name
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 构建 frames（先不带 split 前缀）
    frames, W, H, fx_list = build_frames_from_npz(
        dataset_dir=dataset_dir,
        images_rel=args.images_rel,
        npz_name=args.npz_name,
        normalize_R=True,
        svd_orthogonalize=args.svd_orthogonalize
    )

    # 2) 复制 image -> train / test
    src_image_dir = dataset_dir / args.images_rel
    copy_images_to_splits(src_image_dir, out_dir)

    # 3) 写 transforms_train.json 与 transforms_test.json
    write_transforms_json(out_dir, frames, W, H, fx_list, split_name="train")
    write_transforms_json(out_dir, frames, W, H, fx_list, split_name="test")

    print(f"\n[ALL DONE] Output at: {out_dir}")

if __name__ == "__main__":
    main()
"""
python pack_neus_to_nerf.py C:/Users/guanl/Desktop/reg/public_data/bunny_pose1  --out_dir C:/Users/guanl/Desktop/reg/public_data/bunny_pose1/ --images_rel image --npz_name camera_sphere.npz

"""