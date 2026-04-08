import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
import multiprocessing as mp
import argparse
import os
import json
from pathlib import Path


def get_path_components(path):
    path = Path(path)
    ppath = str(path.parent)
    stem = str(path.stem)
    ext = str(path.suffix)
    return ppath, stem, ext


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


def find_first_existing_file(dir_path, exts):
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return None

    candidates = []
    for ext in exts:
        candidates.extend(sorted(dir_path.glob(f"*{ext}")))
        candidates.extend(sorted(dir_path.glob(f"**/*{ext}")))

    # 去重，保持顺序
    seen = set()
    unique = []
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen and p.is_file():
            seen.add(rp)
            unique.append(p)

    return unique[0] if len(unique) > 0 else None


def find_gt_file(gt_case_dir):
    # GT 通常是点云
    exts = [".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"]
    return find_first_existing_file(gt_case_dir, exts)


def find_pred_file(method_dir):
    # 预测结果默认还是按 ply 优先找
    exts = [".ply", ".obj", ".stl", ".off", ".pcd"]
    return find_first_existing_file(method_dir, exts)


def load_data_as_points(data_path, mode='mesh', thresh=0.002):
    data_path = str(data_path)

    if mode == 'mesh':
        data_mesh = o3d.io.read_triangle_mesh(data_path)
        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)

        if len(vertices) == 0:
            raise RuntimeError(f"Empty mesh vertices: {data_path}")
        if len(triangles) == 0:
            # 没有三角面时，退化成点云使用
            return vertices

        tri_vert = vertices[triangles]

        v1 = tri_vert[:, 1] - tri_vert[:, 0]
        v2 = tri_vert[:, 2] - tri_vert[:, 0]
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)

        non_zero_area = (area2 > 0)[:, 0]
        l1, l2, area2, v1, v2, tri_vert = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
        ]

        if len(l1) == 0:
            return vertices

        thr = thresh * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(
                sample_single_tri,
                (
                    (n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0])
                    for i in range(len(n1))
                ),
                chunksize=1024
            )

        if len(new_pts) > 0:
            new_pts = np.concatenate(new_pts, axis=0)
            data_pcd = np.concatenate([vertices, new_pts], axis=0)
        else:
            data_pcd = vertices

        return data_pcd

    elif mode == 'pcd':
        data_pcd_o3d = o3d.io.read_point_cloud(data_path)
        data_pcd = np.asarray(data_pcd_o3d.points)
        if len(data_pcd) == 0:
            raise RuntimeError(f"Empty point cloud: {data_path}")
        return data_pcd

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def downsample_points(data_pcd, thresh):
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    nn_engine = skln.NearestNeighbors(
        n_neighbors=1,
        radius=thresh,
        algorithm='kd_tree',
        n_jobs=-1
    )
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)

    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1

    data_down = data_pcd[mask]
    return data_down


def load_gt_points(gt_path):
    stl_pcd = o3d.io.read_point_cloud(str(gt_path))
    stl = np.asarray(stl_pcd.points)
    if len(stl) == 0:
        raise RuntimeError(f"Empty GT point cloud: {gt_path}")
    return stl


def evaluate_single_pair(
    data_path,
    gt_path,
    vis_out_dir,
    case_name,
    method_name,
    mode='mesh',
    downsample_density=0.002,
    max_dist=0.1,
    visualize_threshold=0.01,
    dist_thred1=0.001,
    dist_thred2=0.002,
):
    os.makedirs(vis_out_dir, exist_ok=True)

    pbar_total = 9 if mode == 'mesh' else 8
    pbar = tqdm(total=pbar_total, desc=f"{case_name}/{method_name}", leave=True)

    pbar.set_description(f'{case_name}/{method_name} | read pred')
    data_pcd = load_data_as_points(data_path, mode=mode, thresh=downsample_density)

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | downsample pred')
    data_down = downsample_points(data_pcd, downsample_density)

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | read gt')
    gt_pts = load_gt_points(gt_path)

    nn_engine = skln.NearestNeighbors(
        n_neighbors=1,
        radius=downsample_density,
        algorithm='kd_tree',
        n_jobs=-1
    )

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | compute d2gt')
    nn_engine.fit(gt_pts)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_down, n_neighbors=1, return_distance=True)

    if np.any(dist_d2s < max_dist):
        mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    else:
        mean_d2s = float('nan')

    precision_1 = len(dist_d2s[dist_d2s < dist_thred1]) / len(dist_d2s)
    precision_2 = len(dist_d2s[dist_d2s < dist_thred2]) / len(dist_d2s)

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | compute gt2d')
    nn_engine.fit(data_down)
    dist_s2d, idx_s2d = nn_engine.kneighbors(gt_pts, n_neighbors=1, return_distance=True)

    if np.any(dist_s2d < max_dist):
        mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
    else:
        mean_s2d = float('nan')

    recall_1 = len(dist_s2d[dist_s2d < dist_thred1]) / len(dist_s2d)
    recall_2 = len(dist_s2d[dist_s2d < dist_thred2]) / len(dist_s2d)

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | visualize')
    vis_dist = visualize_threshold
    R = np.array([[1, 0, 0]], dtype=np.float64)
    G = np.array([[0, 1, 0]], dtype=np.float64)
    W = np.array([[1, 1, 1]], dtype=np.float64)

    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color = R * data_alpha + W * (1 - data_alpha)
    data_color[dist_d2s[:, 0] >= max_dist] = G
    write_vis_pcd(str(Path(vis_out_dir) / f'vis_{case_name}_d2gt_{method_name}.ply'), data_down, data_color)

    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color = R * stl_alpha + W * (1 - stl_alpha)
    stl_color[dist_s2d[:, 0] >= max_dist] = G
    write_vis_pcd(str(Path(vis_out_dir) / f'vis_{case_name}_gt2d_{method_name}.ply'), gt_pts, stl_color)

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | finalize')
    pbar.close()

    over_all = (mean_d2s + mean_s2d) / 2
    fscore_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-6)
    fscore_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2 + 1e-6)

    result = {
        "case_name": case_name,
        "method_name": method_name,
        "pred_path": str(data_path),
        "gt_path": str(gt_path),
        "over_all": float(over_all),
        "mean_d2gt": float(mean_d2s),
        "mean_gt2d": float(mean_s2d),
        "precision_1mm": float(precision_1),
        "recall_1mm": float(recall_1),
        "fscore_1mm": float(fscore_1),
        "precision_2mm": float(precision_2),
        "recall_2mm": float(recall_2),
        "fscore_2mm": float(fscore_2),
        "num_pred_points_downsampled": int(len(data_down)),
        "num_gt_points": int(len(gt_pts)),
    }
    return result


def collect_tasks(root):
    root = Path(root)
    gt_root = root / "gt_point_cloud"

    if not gt_root.exists():
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    case_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name != "gt_point_cloud"])
    tasks = []

    gt_exts = [".ply", ".pcd", ".xyz", ".pts"]
    pred_exts = [".ply", ".obj", ".stl", ".off", ".pcd"]

    for case_dir in case_dirs:
        case_name = case_dir.name
        raw_dir = case_dir / "raw"

        if not raw_dir.exists():
            print(f"[WARN] skip case '{case_name}': raw dir not found -> {raw_dir}")
            continue

        # 找 gt_point_cloud/case_name.xxx
        gt_file = None
        for ext in gt_exts:
            candidate = gt_root / f"{case_name}{ext}"
            if candidate.exists():
                gt_file = candidate
                break

        if gt_file is None:
            print(f"[WARN] skip case '{case_name}': GT file not found for basename '{case_name}' in {gt_root}")
            continue

        # raw 下直接就是 methodA.ply / methodB.ply
        pred_files = sorted([
            p for p in raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() in pred_exts
        ])

        if len(pred_files) == 0:
            print(f"[WARN] skip case '{case_name}': no prediction files under {raw_dir}")
            continue

        for pred_file in pred_files:
            method_name = pred_file.stem
            vis_out_dir = raw_dir / f"{method_name}_eval_vis"
            log_path = raw_dir / f"{method_name}_eval_result.txt"

            tasks.append({
                "case_name": case_name,
                "method_name": method_name,
                "pred_file": pred_file,
                "gt_file": gt_file,
                "vis_out_dir": vis_out_dir,
                "log_path": log_path,
            })

    return tasks
def write_single_log(log_path, result):
    with open(log_path, 'w+', encoding='utf-8') as fLog:
        fLog.write(
            f'over_all {np.round(result["over_all"], 6)} '
            f'mean_d2gt {np.round(result["mean_d2gt"], 6)} '
            f'mean_gt2d {np.round(result["mean_gt2d"], 6)} \n'
            f'precision_1mm {np.round(result["precision_1mm"], 6)} '
            f'recall_1mm {np.round(result["recall_1mm"], 6)} '
            f'fscore_1mm {np.round(result["fscore_1mm"], 6)} \n'
            f'precision_2mm {np.round(result["precision_2mm"], 6)} '
            f'recall_2mm {np.round(result["recall_2mm"], 6)} '
            f'fscore_2mm {np.round(result["fscore_2mm"], 6)} \n'
            f'[{result["case_name"]}][{result["method_name"]}] \n'
        )


def summarize_results(results):
    summary = {}
    for r in results:
        m = r["method_name"]
        summary.setdefault(m, [])
        summary[m].append(r)

    per_method = {}
    for m, items in summary.items():
        valid_overall = [x["over_all"] for x in items if np.isfinite(x["over_all"])]
        valid_d2gt = [x["mean_d2gt"] for x in items if np.isfinite(x["mean_d2gt"])]
        valid_gt2d = [x["mean_gt2d"] for x in items if np.isfinite(x["mean_gt2d"])]

        per_method[m] = {
            "num_cases": len(items),
            "mean_over_all": float(np.mean(valid_overall)) if len(valid_overall) > 0 else float("nan"),
            "mean_d2gt": float(np.mean(valid_d2gt)) if len(valid_d2gt) > 0 else float("nan"),
            "mean_gt2d": float(np.mean(valid_gt2d)) if len(valid_gt2d) > 0 else float("nan"),
        }
    return per_method


if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='root path of local dataset')
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--downsample_density', type=float, default=0.002)
    parser.add_argument('--patch_size', type=float, default=60)  # 保留，不动
    parser.add_argument('--max_dist', type=float, default=0.1)
    parser.add_argument('--visualize_threshold', type=float, default=0.01)
    parser.add_argument('--log', type=str, default=None, help='optional global summary log path')
    args = parser.parse_args()

    tasks = collect_tasks(args.root)
    print(f"[INFO] found {len(tasks)} case-method tasks")

    if len(tasks) == 0:
        print("[ERROR] no valid tasks found")
        sys.exit(1)

    all_results = []

    for task in tasks:
        case_name = task["case_name"]
        method_name = task["method_name"]
        pred_file = task["pred_file"]
        gt_file = task["gt_file"]
        vis_out_dir = task["vis_out_dir"]
        log_path = task["log_path"]

        print(f"\n[INFO] processing {case_name}/{method_name}")
        print(f"       pred: {pred_file}")
        print(f"       gt  : {gt_file}")

        try:
            result = evaluate_single_pair(
                data_path=pred_file,
                gt_path=gt_file,
                vis_out_dir=vis_out_dir,
                case_name=case_name,
                method_name=method_name,
                mode=args.mode,
                downsample_density=args.downsample_density,
                max_dist=args.max_dist,
                visualize_threshold=args.visualize_threshold,
            )

            print(
                f'over_all: {result["over_all"]}; '
                f'mean_d2gt: {result["mean_d2gt"]}; '
                f'mean_gt2d: {result["mean_gt2d"]}.'
            )
            print(
                f'precision_1mm: {result["precision_1mm"]};  '
                f'recall_1mm: {result["recall_1mm"]};  '
                f'fscore_1mm: {result["fscore_1mm"]}'
            )
            print(
                f'precision_2mm: {result["precision_2mm"]};  '
                f'recall_2mm: {result["recall_2mm"]};  '
                f'fscore_2mm: {result["fscore_2mm"]}'
            )

            write_single_log(log_path, result)
            all_results.append(result)

        except Exception as e:
            print(f"[ERROR] failed on {case_name}/{method_name}: {e}")

    # 写总汇总
    root = Path(args.root)
    summary_json = root / "eval_summary.json"
    summary_txt = root / "eval_summary.txt"

    per_method_summary = summarize_results(all_results)

    full_summary = {
        "num_tasks_total": len(tasks),
        "num_tasks_success": len(all_results),
        "per_case_method_results": all_results,
        "per_method_summary": per_method_summary,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"num_tasks_total: {len(tasks)}\n")
        f.write(f"num_tasks_success: {len(all_results)}\n\n")
        f.write("=== Per-method summary ===\n")
        for method_name, s in sorted(per_method_summary.items()):
            f.write(
                f"{method_name}: "
                f"num_cases={s['num_cases']}, "
                f"mean_over_all={s['mean_over_all']:.6f}, "
                f"mean_d2gt={s['mean_d2gt']:.6f}, "
                f"mean_gt2d={s['mean_gt2d']:.6f}\n"
            )

    if args.log is not None:
        with open(args.log, "w", encoding="utf-8") as f:
            json.dump(full_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] summary json saved to: {summary_json}")
    print(f"[INFO] summary txt  saved to: {summary_txt}")

"""
python deep_fasion_eval_cd.py --root C:/Users/guanl/Downloads/OneDrive_1_3-11-2026/tmp_test/lego2/ --mode mesh
"""