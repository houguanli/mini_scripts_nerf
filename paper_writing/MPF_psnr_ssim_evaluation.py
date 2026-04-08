import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _extract_first_int(stem: str) -> Optional[int]:
    m = re.search(r"(\d+)", stem)
    if m is None:
        return None
    return int(m.group(1))


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _collect_indexed_images(folder: Path) -> Dict[int, Path]:
    """
    扫描 folder 下所有图片，提取文件名中的第一个整数作为 index
    例如:
      000.png -> 0
      rgb_001_final.png -> 1
    """
    mapping: Dict[int, Path] = {}
    if not folder.exists():
        return mapping

    for p in sorted(folder.iterdir()):
        if not _is_image_file(p):
            continue
        idx = _extract_first_int(p.stem)
        if idx is None:
            print(f"[WARN] skip file without integer index: {p}")
            continue
        if idx in mapping:
            raise RuntimeError(f"Duplicate image index {idx} in folder: {folder}")
        mapping[idx] = p
    return mapping


def _read_image_float01(path: Path) -> np.ndarray:
    """
    返回 HxWxC, float32, [0,1]
    灰度图会扩成 HxWx1
    RGBA 会截断为 RGB
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr /= 255.0

    return arr


def _read_mask_bool(mask_path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    读取 mask，返回 HxW bool
    非零即 True
    """
    mask = Image.open(mask_path)
    mask_arr = np.array(mask)

    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]

    if mask_arr.shape[:2] != target_hw:
        raise RuntimeError(
            f"Mask shape mismatch: {mask_path}, got {mask_arr.shape[:2]}, expect {target_hw}"
        )

    return mask_arr > 0


def _match_channels(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    尽量匹配通道数：
    - 都是 1 通道：直接返回
    - 一个 1 通道，一个 3 通道：把 1 通道复制成 3 通道
    - 其他情况要求通道一致
    """
    if pred.shape[:2] != gt.shape[:2]:
        raise RuntimeError(f"Image size mismatch: pred={pred.shape}, gt={gt.shape}")

    cp = pred.shape[2]
    cg = gt.shape[2]

    if cp == cg:
        return pred, gt

    if cp == 1 and cg == 3:
        pred = np.repeat(pred, 3, axis=2)
        return pred, gt

    if cp == 3 and cg == 1:
        gt = np.repeat(gt, 3, axis=2)
        return pred, gt

    raise RuntimeError(f"Channel mismatch not supported: pred={pred.shape}, gt={gt.shape}")


def _compute_masked_psnr(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """
    pred, gt: HxWxC in [0,1]
    mask: HxW bool or None
    """
    if mask is None:
        return float(peak_signal_noise_ratio(gt, pred, data_range=1.0))

    if mask.shape != gt.shape[:2]:
        raise RuntimeError(f"Mask shape mismatch: mask={mask.shape}, image={gt.shape[:2]}")

    valid = mask > 0
    valid_count = int(valid.sum())
    if valid_count == 0:
        return float("nan")

    diff2 = (pred - gt) ** 2
    mse = diff2[valid].mean()
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _choose_ssim_win_size(h: int, w: int) -> int:
    """
    skimage SSIM 要求 win_size 为奇数且 <= min(H, W)
    """
    m = min(h, w)
    if m < 3:
        raise RuntimeError(f"Image too small for SSIM: {(h, w)}")

    win = min(11, m)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        win = 3
    return win


def _compute_masked_ssim(pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """
    计算 SSIM。
    如果有 mask，则：
      1) 先算 full SSIM map
      2) 再在 mask 区域上平均
    """
    h, w = gt.shape[:2]
    win_size = _choose_ssim_win_size(h, w)

    if gt.shape[2] == 1:
        gt_in = gt[..., 0]
        pred_in = pred[..., 0]
        ssim_mean, ssim_map = structural_similarity(
            gt_in,
            pred_in,
            data_range=1.0,
            full=True,
            win_size=win_size,
        )
    else:
        ssim_mean, ssim_map = structural_similarity(
            gt,
            pred,
            data_range=1.0,
            full=True,
            channel_axis=-1,
            win_size=win_size,
        )
        # 某些版本返回 HxW 或 HxWxC，这里都兼容一下
        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=2)

    if mask is None:
        return float(ssim_mean)

    if mask.shape != gt.shape[:2]:
        raise RuntimeError(f"Mask shape mismatch: mask={mask.shape}, image={gt.shape[:2]}")

    valid = mask > 0
    valid_count = int(valid.sum())
    if valid_count == 0:
        return float("nan")

    return float(ssim_map[valid].mean())


def evaluate_image_pair(
    pred_img_path: Path,
    gt_img_path: Path,
    gt_mask_path: Optional[Path] = None,
) -> Dict:
    pred = _read_image_float01(pred_img_path)
    gt = _read_image_float01(gt_img_path)
    pred, gt = _match_channels(pred, gt)

    mask = None
    if gt_mask_path is not None and gt_mask_path.exists():
        mask = _read_mask_bool(gt_mask_path, target_hw=gt.shape[:2])

    psnr = _compute_masked_psnr(pred, gt, mask)
    ssim = _compute_masked_ssim(pred, gt, mask)

    return {
        "pred_image": str(pred_img_path),
        "gt_image": str(gt_img_path),
        "gt_mask": str(gt_mask_path) if gt_mask_path is not None and gt_mask_path.exists() else None,
        "height": int(gt.shape[0]),
        "width": int(gt.shape[1]),
        "channels": int(gt.shape[2]),
        "has_mask": mask is not None,
        "valid_mask_pixels": int(mask.sum()) if mask is not None else int(gt.shape[0] * gt.shape[1]),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }


def _safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _collect_gt_cases(gt_dir: Path) -> Dict[str, Dict]:
    """
    root/gt/<case_name>/image/*.png
    root/gt/<case_name>/mask/*.png  # optional
    """
    gt_cases: Dict[str, Dict] = {}

    for case_dir in sorted(gt_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        case_name = case_dir.name
        image_dir = case_dir / "image"
        mask_dir = case_dir / "mask"

        if not image_dir.is_dir():
            continue

        gt_images = _collect_indexed_images(image_dir)
        gt_masks = _collect_indexed_images(mask_dir) if mask_dir.is_dir() else {}

        if len(gt_images) == 0:
            continue

        gt_cases[case_name] = {
            "image_dir": image_dir,
            "mask_dir": mask_dir if mask_dir.is_dir() else None,
            "gt_images": gt_images,
            "gt_masks": gt_masks,
        }

    return gt_cases


def _collect_prediction_cases(root: Path, gt_dirname: str = "gt") -> Dict[str, Dict[str, Dict[int, Path]]]:
    """
    兼容两种结构：

    1) root/<case>/raw/<method>/*.png
    2) root/<case>/<method>/*.png

    返回:
    {
      case_name: {
        method_name: {
          idx: image_path
        }
      }
    }
    """
    pred_cases: Dict[str, Dict[str, Dict[int, Path]]] = {}

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        if case_dir.name == gt_dirname:
            continue

        case_name = case_dir.name
        method_map: Dict[str, Dict[int, Path]] = {}

        # 优先 case/raw/*
        raw_dir = case_dir / "raw"
        candidate_parent = raw_dir if raw_dir.is_dir() else case_dir

        for method_dir in sorted(candidate_parent.iterdir()):
            if not method_dir.is_dir():
                continue

            method_name = method_dir.name
            pred_images = _collect_indexed_images(method_dir)
            if len(pred_images) == 0:
                continue

            method_map[method_name] = pred_images

        if len(method_map) > 0:
            pred_cases[case_name] = method_map

    return pred_cases


def evaluate_root(
    root: str,
    gt_dirname: str = "gt",
    output_json: Optional[str] = None,
) -> Dict:
    root = Path(root)
    gt_dir = root / gt_dirname

    if not gt_dir.exists():
        raise FileNotFoundError(f"GT folder not found: {gt_dir}")

    gt_cases = _collect_gt_cases(gt_dir)
    pred_cases = _collect_prediction_cases(root, gt_dirname=gt_dirname)

    per_frame_results = []
    per_case_method = []
    missing_gt_cases = []
    orphan_gt_cases = []

    gt_case_names = set(gt_cases.keys())
    pred_case_names = set(pred_cases.keys())

    for case_name in sorted(pred_case_names):
        if case_name not in gt_cases:
            missing_gt_cases.append(case_name)
            print(f"[WARN] prediction case exists but GT missing: {case_name}")
            continue

        gt_info = gt_cases[case_name]
        gt_images = gt_info["gt_images"]
        gt_masks = gt_info["gt_masks"]

        method_map = pred_cases[case_name]

        for method_name, pred_images in sorted(method_map.items()):
            print(f"[INFO] evaluating case={case_name}, method={method_name}")

            frame_results = []
            skipped_pred_indices = []

            for idx, pred_img_path in sorted(pred_images.items()):
                gt_img_path = gt_images.get(idx, None)
                if gt_img_path is None:
                    skipped_pred_indices.append(idx)
                    print(f"[WARN] case={case_name}, method={method_name}, pred index={idx} not found in GT, skip")
                    continue

                gt_mask_path = gt_masks.get(idx, None)

                try:
                    r = evaluate_image_pair(
                        pred_img_path=pred_img_path,
                        gt_img_path=gt_img_path,
                        gt_mask_path=gt_mask_path,
                    )
                    r["case_name"] = case_name
                    r["method_name"] = method_name
                    r["frame_index"] = int(idx)
                    per_frame_results.append(r)
                    frame_results.append(r)
                except Exception as e:
                    print(f"[WARN] failed on case={case_name}, method={method_name}, idx={idx}: {e}")

            if len(frame_results) == 0:
                continue

            case_method_summary = {
                "case_name": case_name,
                "method_name": method_name,
                "is_post": method_name.endswith("_post"),
                "num_frames": len(frame_results),
                "mean_psnr": _safe_nanmean([x["psnr"] for x in frame_results]),
                "mean_ssim": _safe_nanmean([x["ssim"] for x in frame_results]),
                "skipped_pred_indices": skipped_pred_indices,
            }
            per_case_method.append(case_method_summary)

    for case_name in sorted(gt_case_names - pred_case_names):
        orphan_gt_cases.append(case_name)

    if len(per_case_method) == 0:
        raise RuntimeError("No valid case-method results were evaluated.")

    # 按 method 聚合
    method_groups: Dict[str, List[Dict]] = {}
    for x in per_case_method:
        method_groups.setdefault(x["method_name"], []).append(x)

    per_method_summary = []
    for method_name in sorted(method_groups.keys()):
        items = method_groups[method_name]
        per_method_summary.append({
            "method_name": method_name,
            "is_post": method_name.endswith("_post"),
            "num_cases": len(items),
            "mean_psnr": _safe_nanmean([x["mean_psnr"] for x in items]),
            "mean_ssim": _safe_nanmean([x["mean_ssim"] for x in items]),
        })

    overall_summary = {
        "root": str(root),
        "gt_dir": str(gt_dir),
        "num_case_method_pairs": len(per_case_method),
        "num_methods": len(per_method_summary),
        "overall_mean_psnr": _safe_nanmean([x["mean_psnr"] for x in per_case_method]),
        "overall_mean_ssim": _safe_nanmean([x["mean_ssim"] for x in per_case_method]),
    }

    output = {
        "summary": overall_summary,
        "per_method_summary": per_method_summary,
        "per_case_method": sorted(per_case_method, key=lambda x: (x["case_name"], x["method_name"])),
        "per_frame_results": sorted(per_frame_results, key=lambda x: (x["case_name"], x["method_name"], x["frame_index"])),
        "missing_gt_cases": missing_gt_cases,
        "orphan_gt_cases": orphan_gt_cases,
    }

    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[INFO] saved results to: {output_path}")

    return output


def print_summary(results: Dict):
    summary = results["summary"]

    print("\n========== Overall Summary ==========")
    print(f"num_case_method_pairs : {summary['num_case_method_pairs']}")
    print(f"num_methods           : {summary['num_methods']}")
    print(f"overall_mean_psnr     : {summary['overall_mean_psnr']:.6f}")
    print(f"overall_mean_ssim     : {summary['overall_mean_ssim']:.6f}")

    print("\n========== Per Method Summary ==========")
    for x in results["per_method_summary"]:
        print(
            f"[{x['method_name']}] "
            f"cases={x['num_cases']} | "
            f"PSNR={x['mean_psnr']:.6f}, "
            f"SSIM={x['mean_ssim']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Dataset root")
    parser.add_argument("--gt_dirname", type=str, default="gt",
                        help="GT folder name under root, default: gt")
    parser.add_argument("--output_json", type=str, default=None)

    args = parser.parse_args()

    results = evaluate_root(
        root=args.root,
        gt_dirname=args.gt_dirname,
        output_json=args.output_json,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
# python MPF_psnr_ssim_evaluation.py --root "C:\Users\guanl\Downloads\OneDrive_1_3-11-2026\mesh\custom\synthetic" --output_json "C:\Users\guanl\Downloads\OneDrive_1_3-11-2026\mesh\custom\synthetic\psnr_ssim_eval_results.json"

