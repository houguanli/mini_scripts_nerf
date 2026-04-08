"""
MPF_fscore_evaluation.py

Extends MPF_cd_nc_evaluation.py with F-score (Precision / Recall / F-score)
at multiple distance thresholds.

F-score definition (per threshold τ):
    Precision(τ) = |{p ∈ pred | min_dist(p, gt)  ≤ τ}| / |pred|
    Recall(τ)    = |{g ∈ gt   | min_dist(g, pred) ≤ τ}| / |gt|
    F-score(τ)   = 2·P·R / (P+R)   (0 if P+R == 0)

CD and NC are also computed so that a single run gives the full metric table.

Usage:
    python MPF_fscore_evaluation.py \
        --root "path/to/dataset" \
        --num_samples 100000 \
        --fscore_thresholds 0.005 0.01 0.02 0.05 \
        --cd_filter_threshold 0.02 \
        --output_json "results.json"

Directory convention (same as MPF_cd_nc_evaluation.py):
    root/
      gt/          ← GT meshes: <case_name>.ply
      <case_name>/
        raw/       ← predicted meshes: <method_name>.ply
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import trimesh

# ---------------------------------------------------------------------------
# Shared mesh / sampling helpers (duplicated from MPF_cd_nc_evaluation.py
# to keep this file self-contained)
# ---------------------------------------------------------------------------

def _load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        tm = trimesh.load(mesh_path, force='mesh')
        tm = tm.triangulate()
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(tm.vertices),
            triangles=o3d.utility.Vector3iVector(tm.faces),
        )
    mesh.compute_vertex_normals()
    print(f"[DEBUG] {mesh_path.name}: verts={len(mesh.vertices)}, "
          f"tris={len(mesh.triangles)}, empty={mesh.is_empty()}")
    return mesh


def _sample_points_with_normals(
    mesh: o3d.geometry.TriangleMesh,
    num_samples: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """均匀采样，返回 points (N,3) 和 normals (N,3)。"""
    if seed is not None:
        o3d.utility.random.seed(seed)
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    points  = np.asarray(pcd.points,  dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64)
    if normals.shape[0] != points.shape[0]:
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals, dtype=np.float64)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    normals[valid]  = normals[valid] / norms[valid]
    normals[~valid] = 0.0
    return points, normals


def _safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


# ---------------------------------------------------------------------------
# CD / NC (single-direction)
# ---------------------------------------------------------------------------

def _compute_cd_nc_one_dir(
    src_pts: np.ndarray,
    src_nrm: np.ndarray,
    dst_pts: np.ndarray,
    dst_nrm: np.ndarray,
    cd_filter_threshold: Optional[float] = None,
    use_abs_for_nc: bool = True,
) -> Dict[str, float]:
    """单向 src→dst 的 CD 和 NC。"""
    tree = cKDTree(dst_pts)
    dists, idxs = tree.query(src_pts, k=1, workers=-1)

    matched_nrm = dst_nrm[idxs]
    dots = np.sum(src_nrm * matched_nrm, axis=1)
    if use_abs_for_nc:
        dots = np.abs(dots)
    dots = np.clip(dots, -1.0, 1.0)

    valid_mask = (dists <= cd_filter_threshold) if cd_filter_threshold is not None \
                 else np.ones_like(dists, dtype=bool)

    valid_count = int(valid_mask.sum())
    total_count = int(len(dists))
    valid_ratio = float(valid_count / max(total_count, 1))

    if valid_count == 0:
        return {"cd": float("nan"), "nc": float("nan"),
                "valid_ratio": valid_ratio, "valid_count": valid_count,
                "total_count": total_count, "_dists": dists}

    return {"cd": float(np.mean(dists[valid_mask])),
            "nc": float(np.mean(dots[valid_mask])),
            "valid_ratio": valid_ratio,
            "valid_count": valid_count,
            "total_count": total_count,
            "_dists": dists}          # 保留原始距离供 F-score 复用


# ---------------------------------------------------------------------------
# F-score
# ---------------------------------------------------------------------------

def _compute_fscore(
    pred_to_gt_dists: np.ndarray,
    gt_to_pred_dists: np.ndarray,
    thresholds: List[float],
) -> Dict[str, float]:
    """
    给定两个方向的最近邻距离数组，计算各阈值下的 P / R / F。

    Args:
        pred_to_gt_dists: shape (N_pred,)  pred 各点到最近 GT 点的距离
        gt_to_pred_dists: shape (N_gt,)    GT 各点到最近 pred 点的距离
        thresholds:       评测阈值列表

    Returns:
        扁平字典，key 形如 "fscore_0.010", "precision_0.010", "recall_0.010"
    """
    result = {}
    n_pred = len(pred_to_gt_dists)
    n_gt   = len(gt_to_pred_dists)

    for tau in thresholds:
        key = f"{tau:.4f}".rstrip('0').rstrip('.')   # "0.01" → "0.01"

        precision = float((pred_to_gt_dists <= tau).sum()) / max(n_pred, 1)
        recall    = float((gt_to_pred_dists <= tau).sum()) / max(n_gt,   1)
        fscore    = (2.0 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        result[f"precision_{key}"] = precision
        result[f"recall_{key}"]    = recall
        result[f"fscore_{key}"]    = fscore

    return result


# ---------------------------------------------------------------------------
# Pair-level evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(
    pred_mesh_path: Path,
    gt_mesh_path: Path,
    num_samples: int = 100000,
    seed: int = 0,
    cd_filter_threshold: Optional[float] = None,
    fscore_thresholds: Optional[List[float]] = None,
    use_abs_for_nc: bool = True,
) -> Dict:
    """
    评测一对 (pred, gt) mesh，返回 CD / NC / F-score 全套指标。

    Args:
        fscore_thresholds: F-score 阈值列表，默认 [0.005, 0.01, 0.02, 0.05]
    """
    if fscore_thresholds is None:
        fscore_thresholds = [0.005, 0.01, 0.02, 0.05]

    pred_mesh = _load_mesh(pred_mesh_path)
    gt_mesh   = _load_mesh(gt_mesh_path)

    pred_pts, pred_nrm = _sample_points_with_normals(pred_mesh, num_samples, seed)
    gt_pts,   gt_nrm   = _sample_points_with_normals(gt_mesh,   num_samples, seed)

    fwd = _compute_cd_nc_one_dir(pred_pts, pred_nrm, gt_pts, gt_nrm,
                                  cd_filter_threshold, use_abs_for_nc)
    bwd = _compute_cd_nc_one_dir(gt_pts, gt_nrm, pred_pts, pred_nrm,
                                  cd_filter_threshold, use_abs_for_nc)

    # F-score：使用未经 cd_filter 截断的原始距离，才能正确反映 recall/precision
    fscore_metrics = _compute_fscore(fwd["_dists"], bwd["_dists"], fscore_thresholds)

    cd_bi = float(np.nanmean([fwd["cd"], bwd["cd"]]))
    nc_bi = float(np.nanmean([fwd["nc"], bwd["nc"]]))

    result = {
        "pred_mesh": str(pred_mesh_path),
        "gt_mesh":   str(gt_mesh_path),
        "num_samples": num_samples,
        "cd_filter_threshold": cd_filter_threshold,
        "fscore_thresholds": fscore_thresholds,

        # CD
        "cd_forward":      fwd["cd"],
        "cd_backward":     bwd["cd"],
        "cd_bidirectional": cd_bi,

        # NC
        "nc_forward":      fwd["nc"],
        "nc_backward":     bwd["nc"],
        "nc_bidirectional": nc_bi,

        # validity
        "forward_valid_ratio":  fwd["valid_ratio"],
        "backward_valid_ratio": bwd["valid_ratio"],
        "forward_valid_count":  fwd["valid_count"],
        "backward_valid_count": bwd["valid_count"],
        "forward_total_count":  fwd["total_count"],
        "backward_total_count": bwd["total_count"],
    }
    result.update(fscore_metrics)
    return result


# ---------------------------------------------------------------------------
# Directory-level evaluation (same layout as MPF_cd_nc_evaluation.py)
# ---------------------------------------------------------------------------

def _collect_gt_map(gt_dir: Path) -> Dict[str, Path]:
    gt_map = {}
    for p in sorted(gt_dir.glob("*.ply")):
        if p.stem in gt_map:
            raise RuntimeError(f"Duplicate GT case name: {p.stem}")
        gt_map[p.stem] = p
    return gt_map


def _collect_predictions(root: Path, gt_dirname: str = "gt") -> Dict[str, List[Path]]:
    pred_map = {}
    for item in sorted(root.iterdir()):
        if not item.is_dir() or item.name == gt_dirname:
            continue
        raw_dir = item / "raw"
        if not raw_dir.is_dir():
            continue
        ply_files = sorted(raw_dir.glob("*.ply"))
        if ply_files:
            pred_map[item.name] = ply_files
    return pred_map


def evaluate_root(
    root: str,
    gt_dirname: str = "gt",
    num_samples: int = 100000,
    seed: int = 0,
    cd_filter_threshold: Optional[float] = None,
    fscore_thresholds: Optional[List[float]] = None,
    output_json: Optional[str] = None,
) -> Dict:
    if fscore_thresholds is None:
        fscore_thresholds = [0.005, 0.01, 0.02, 0.05]

    root    = Path(root)
    gt_dir  = root / gt_dirname
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT folder not found: {gt_dir}")

    gt_map   = _collect_gt_map(gt_dir)
    pred_map = _collect_predictions(root, gt_dirname)

    per_case_method   = []
    missing_gt_cases  = []
    orphan_gt_cases   = []

    for case_name in sorted(pred_map.keys()):
        if case_name not in gt_map:
            missing_gt_cases.append(case_name)
            print(f"[WARN] prediction exists but GT missing: {case_name}")
            continue

        gt_path = gt_map[case_name]
        for pred_path in pred_map[case_name]:
            method_name = pred_path.stem
            print(f"[INFO] evaluating case={case_name}, method={method_name}")
            try:
                res = evaluate_pair(
                    pred_mesh_path=pred_path,
                    gt_mesh_path=gt_path,
                    num_samples=num_samples,
                    seed=seed,
                    cd_filter_threshold=cd_filter_threshold,
                    fscore_thresholds=fscore_thresholds,
                )
                res["case_name"]   = case_name
                res["method_name"] = method_name
                res["is_post"]     = method_name.endswith("_post")
                per_case_method.append(res)
            except Exception as e:
                print(f"[WARN] failed case={case_name} method={method_name}: {e}")

    for case_name in sorted(set(gt_map.keys()) - set(pred_map.keys())):
        orphan_gt_cases.append(case_name)

    if not per_case_method:
        raise RuntimeError("No valid case-method pairs were evaluated.")

    # ----- per-method aggregation -----
    method_groups: Dict[str, List[Dict]] = {}
    for x in per_case_method:
        method_groups.setdefault(x["method_name"], []).append(x)

    # F-score key names (derived from first result)
    fscore_keys = [k for k in per_case_method[0] if k.startswith("fscore_")]
    precision_keys = [k for k in per_case_method[0] if k.startswith("precision_")]
    recall_keys    = [k for k in per_case_method[0] if k.startswith("recall_")]

    per_method_summary = []
    for method_name in sorted(method_groups.keys()):
        items = method_groups[method_name]
        entry = {
            "method_name": method_name,
            "is_post":     method_name.endswith("_post"),
            "num_cases":   len(items),

            "mean_cd_forward":      _safe_nanmean([x["cd_forward"]      for x in items]),
            "mean_cd_backward":     _safe_nanmean([x["cd_backward"]     for x in items]),
            "mean_cd_bidirectional":_safe_nanmean([x["cd_bidirectional"] for x in items]),

            "mean_nc_forward":      _safe_nanmean([x["nc_forward"]      for x in items]),
            "mean_nc_backward":     _safe_nanmean([x["nc_backward"]     for x in items]),
            "mean_nc_bidirectional":_safe_nanmean([x["nc_bidirectional"] for x in items]),

            "mean_forward_valid_ratio":  _safe_nanmean([x["forward_valid_ratio"]  for x in items]),
            "mean_backward_valid_ratio": _safe_nanmean([x["backward_valid_ratio"] for x in items]),
        }
        for k in fscore_keys + precision_keys + recall_keys:
            entry[f"mean_{k}"] = _safe_nanmean([x[k] for x in items])
        per_method_summary.append(entry)

    overall_summary = {
        "root":                    str(root),
        "gt_dir":                  str(gt_dir),
        "num_case_method_pairs":   len(per_case_method),
        "num_methods":             len(per_method_summary),
        "num_samples":             num_samples,
        "cd_filter_threshold":     cd_filter_threshold,
        "fscore_thresholds":       fscore_thresholds,

        "overall_mean_cd_forward":      _safe_nanmean([x["cd_forward"]       for x in per_case_method]),
        "overall_mean_cd_backward":     _safe_nanmean([x["cd_backward"]      for x in per_case_method]),
        "overall_mean_cd_bidirectional":_safe_nanmean([x["cd_bidirectional"] for x in per_case_method]),

        "overall_mean_nc_forward":      _safe_nanmean([x["nc_forward"]       for x in per_case_method]),
        "overall_mean_nc_backward":     _safe_nanmean([x["nc_backward"]      for x in per_case_method]),
        "overall_mean_nc_bidirectional":_safe_nanmean([x["nc_bidirectional"] for x in per_case_method]),
    }
    for k in fscore_keys + precision_keys + recall_keys:
        overall_summary[f"overall_mean_{k}"] = _safe_nanmean([x[k] for x in per_case_method])

    output = {
        "summary":            overall_summary,
        "per_method_summary": per_method_summary,
        "per_case_method":    sorted(per_case_method,
                                     key=lambda x: (x["case_name"], x["method_name"])),
        "missing_gt_cases":   missing_gt_cases,
        "orphan_gt_cases":    orphan_gt_cases,
    }

    if output_json is not None:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[INFO] saved results to: {out_path}")

    return output


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_summary(results: Dict):
    summary = results["summary"]
    thresholds = summary.get("fscore_thresholds", [])

    print("\n========== Overall Summary ==========")
    print(f"num_case_method_pairs    : {summary['num_case_method_pairs']}")
    print(f"num_methods              : {summary['num_methods']}")
    print(f"num_samples              : {summary['num_samples']}")
    print(f"cd_filter_threshold      : {summary['cd_filter_threshold']}")
    print(f"fscore_thresholds        : {thresholds}")
    print(f"overall CD_f             : {summary['overall_mean_cd_forward']:.8f}")
    print(f"overall CD_b             : {summary['overall_mean_cd_backward']:.8f}")
    print(f"overall CD_bi            : {summary['overall_mean_cd_bidirectional']:.8f}")
    print(f"overall NC_f             : {summary['overall_mean_nc_forward']:.8f}")
    print(f"overall NC_b             : {summary['overall_mean_nc_backward']:.8f}")
    print(f"overall NC_bi            : {summary['overall_mean_nc_bidirectional']:.8f}")
    for tau in thresholds:
        key = f"{tau:.4f}".rstrip('0').rstrip('.')
        p = summary.get(f"overall_mean_precision_{key}", float("nan"))
        r = summary.get(f"overall_mean_recall_{key}",    float("nan"))
        f = summary.get(f"overall_mean_fscore_{key}",    float("nan"))
        print(f"overall F@{tau:<6}         : P={p:.4f}  R={r:.4f}  F={f:.4f}")

    print("\n========== Per Method Summary ==========")
    for x in results["per_method_summary"]:
        print(f"\n[{x['method_name']}]  cases={x['num_cases']}")
        print(f"  CD  f/b/bi : {x['mean_cd_forward']:.8f} / "
              f"{x['mean_cd_backward']:.8f} / {x['mean_cd_bidirectional']:.8f}")
        print(f"  NC  f/b/bi : {x['mean_nc_forward']:.8f} / "
              f"{x['mean_nc_backward']:.8f} / {x['mean_nc_bidirectional']:.8f}")
        for tau in thresholds:
            key = f"{tau:.4f}".rstrip('0').rstrip('.')
            p = x.get(f"mean_precision_{key}", float("nan"))
            r = x.get(f"mean_recall_{key}",    float("nan"))
            f = x.get(f"mean_fscore_{key}",    float("nan"))
            print(f"  F@{tau:<6}     : P={p:.4f}  R={r:.4f}  F={f:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3D meshes: CD, NC, and F-score at multiple thresholds."
    )
    parser.add_argument("--root", type=str, required=True,
                        help="Dataset root. Expects root/gt/*.ply and "
                             "root/<case_name>/raw/<method>.ply")
    parser.add_argument("--gt_dirname", type=str, default="gt")
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cd_filter_threshold", type=float, default=None,
                        help="Distance threshold for filtering CD/NC (optional)")
    parser.add_argument("--fscore_thresholds", type=float, nargs="+",
                        default=[0.005, 0.01, 0.02, 0.05],
                        help="F-score distance thresholds (default: 0.005 0.01 0.02 0.05)")
    parser.add_argument("--output_json", type=str, default=None)

    args = parser.parse_args()

    results = evaluate_root(
        root=args.root,
        gt_dirname=args.gt_dirname,
        num_samples=args.num_samples,
        seed=args.seed,
        cd_filter_threshold=args.cd_filter_threshold,
        fscore_thresholds=args.fscore_thresholds,
        output_json=args.output_json,
    )
    print_summary(results)


if __name__ == "__main__":
    main()

# Example:
# python MPF_fscore_evaluation.py \
#     --root "C:\path\to\dataset" \
#     --num_samples 100000 \
#     --fscore_thresholds 0.005 0.01 0.02 0.05 \
#     --cd_filter_threshold 0.02 \
#     --output_json "results.json"
