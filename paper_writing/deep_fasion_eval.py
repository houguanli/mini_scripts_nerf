import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm
import multiprocessing as mp
import argparse
import os
import json
from pathlib import Path
import sys


def get_path_components(path):
    path = Path(path)
    ppath = str(path.parent)
    stem = str(path.stem)
    ext = str(path.suffix)
    return ppath, stem, ext


def safe_normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def sample_single_tri(input_):
    """
    对单个三角形做均匀采样，同时返回采样点法向（直接继承 face normal）
    """
    n1, n2, v1, v2, tri_vert, tri_normal = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert

    tri_normal = np.asarray(tri_normal).reshape(1, 3)
    qn = np.repeat(tri_normal, len(q), axis=0)
    return q, qn


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

    seen = set()
    unique = []
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen and p.is_file():
            seen.add(rp)
            unique.append(p)

    return unique[0] if len(unique) > 0 else None


def find_gt_file(gt_case_dir):
    exts = [".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"]
    return find_first_existing_file(gt_case_dir, exts)


def find_pred_file(method_dir):
    exts = [".ply", ".obj", ".stl", ".off", ".pcd"]
    return find_first_existing_file(method_dir, exts)


def estimate_pcd_normals(points, radius=0.01, max_nn=30, orient_k=30):
    """
    对点云估计 normal
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    try:
        pcd.orient_normals_consistent_tangent_plane(orient_k)
    except Exception:
        pass

    normals = np.asarray(pcd.normals)
    normals = safe_normalize(normals)
    return normals


def voxel_downsample_points_and_normals(points, normals, voxel_size):
    """
    可扩展的 voxel downsample。
    保留每个 voxel 的第一个点及其 normal。
    比 radius_neighbors 稳得多，适合大点云。
    """
    if voxel_size is None or voxel_size <= 0:
        return points, normals

    if len(points) == 0:
        return points, normals

    coords = np.floor(points / voxel_size).astype(np.int64)

    # 用结构化数组做 unique，避免 Python tuple 巨慢
    dtype = np.dtype([('x', np.int64), ('y', np.int64), ('z', np.int64)])
    coords_view = coords.view(dtype).reshape(-1)

    _, unique_indices = np.unique(coords_view, return_index=True)
    unique_indices = np.sort(unique_indices)

    points_ds = points[unique_indices]
    normals_ds = None if normals is None else normals[unique_indices]

    return points_ds, normals_ds


def chunked_1nn_query(tree, query_points, ref_normals=None, query_normals=None, chunk_size=200000):
    """
    分块做 1-NN 查询。
    返回：
        dists: (N,)
        idxs:  (N,)
        ncs:   (N,) 或 None
    """
    all_dists = []
    all_idxs = []
    all_ncs = [] if (ref_normals is not None and query_normals is not None) else None

    total = len(query_points)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        pts_chunk = query_points[start:end]

        dists_chunk, idxs_chunk = tree.query(pts_chunk, k=1, workers=-1)

        all_dists.append(dists_chunk)
        all_idxs.append(idxs_chunk)

        if all_ncs is not None:
            ref_n = ref_normals[idxs_chunk]
            qry_n = query_normals[start:end]
            nc = np.abs(np.sum(qry_n * ref_n, axis=1))
            all_ncs.append(nc)

    dists = np.concatenate(all_dists, axis=0)
    idxs = np.concatenate(all_idxs, axis=0)

    if all_ncs is not None:
        ncs = np.concatenate(all_ncs, axis=0)
    else:
        ncs = None

    return dists, idxs, ncs


def load_data_as_points_and_normals(data_path, mode='mesh', thresh=0.002, pred_downsample_density=None):
    """
    返回:
        data_pcd: (N,3)
        data_nrm: (N,3)
    """
    data_path = str(data_path)

    if mode == 'mesh':
        data_mesh = o3d.io.read_triangle_mesh(data_path)
        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)

        if len(vertices) == 0:
            raise RuntimeError(f"Empty mesh vertices: {data_path}")

        if len(triangles) == 0:
            normals = estimate_pcd_normals(vertices, radius=max(thresh * 4, 1e-4))
            if pred_downsample_density is not None and pred_downsample_density > 0:
                vertices, normals = voxel_downsample_points_and_normals(
                    vertices, normals, pred_downsample_density
                )
            return vertices, normals

        tri_vert = vertices[triangles]

        v1 = tri_vert[:, 1] - tri_vert[:, 0]
        v2 = tri_vert[:, 2] - tri_vert[:, 0]
        tri_normals = np.cross(v1, v2)
        tri_normals = safe_normalize(tri_normals)

        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)

        non_zero_area = (area2 > 0)[:, 0]
        l1, l2, area2, v1, v2, tri_vert, tri_normals = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert, tri_normals]
        ]

        if len(l1) == 0:
            normals = estimate_pcd_normals(vertices, radius=max(thresh * 4, 1e-4))
            if pred_downsample_density is not None and pred_downsample_density > 0:
                vertices, normals = voxel_downsample_points_and_normals(
                    vertices, normals, pred_downsample_density
                )
            return vertices, normals

        thr = thresh * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        with mp.Pool() as mp_pool:
            sampled = mp_pool.map(
                sample_single_tri,
                (
                    (
                        n1[i, 0],
                        n2[i, 0],
                        v1[i:i + 1],
                        v2[i:i + 1],
                        tri_vert[i:i + 1, 0],
                        tri_normals[i]
                    )
                    for i in range(len(n1))
                ),
                chunksize=1024
            )

        if len(sampled) > 0:
            new_pts = np.concatenate([x[0] for x in sampled], axis=0)
            new_nrm = np.concatenate([x[1] for x in sampled], axis=0)

            if data_mesh.has_vertex_normals():
                vertex_normals = np.asarray(data_mesh.vertex_normals)
                if len(vertex_normals) != len(vertices):
                    vertex_normals = estimate_pcd_normals(vertices, radius=max(thresh * 4, 1e-4))
                else:
                    vertex_normals = safe_normalize(vertex_normals)
            else:
                vertex_normals = estimate_pcd_normals(vertices, radius=max(thresh * 4, 1e-4))

            data_pcd = np.concatenate([vertices, new_pts], axis=0)
            data_nrm = np.concatenate([vertex_normals, new_nrm], axis=0)
        else:
            data_pcd = vertices
            if data_mesh.has_vertex_normals():
                data_nrm = np.asarray(data_mesh.vertex_normals)
                if len(data_nrm) != len(data_pcd):
                    data_nrm = estimate_pcd_normals(data_pcd, radius=max(thresh * 4, 1e-4))
                else:
                    data_nrm = safe_normalize(data_nrm)
            else:
                data_nrm = estimate_pcd_normals(data_pcd, radius=max(thresh * 4, 1e-4))

        data_nrm = safe_normalize(data_nrm)

        if pred_downsample_density is not None and pred_downsample_density > 0:
            data_pcd, data_nrm = voxel_downsample_points_and_normals(
                data_pcd, data_nrm, pred_downsample_density
            )

        return data_pcd, data_nrm

    elif mode == 'pcd':
        data_pcd_o3d = o3d.io.read_point_cloud(data_path)
        data_pcd = np.asarray(data_pcd_o3d.points)
        if len(data_pcd) == 0:
            raise RuntimeError(f"Empty point cloud: {data_path}")

        data_nrm = np.asarray(data_pcd_o3d.normals)
        if len(data_nrm) != len(data_pcd):
            data_nrm = estimate_pcd_normals(data_pcd, radius=max(thresh * 4, 1e-4))
        else:
            data_nrm = safe_normalize(data_nrm)

        if pred_downsample_density is not None and pred_downsample_density > 0:
            data_pcd, data_nrm = voxel_downsample_points_and_normals(
                data_pcd, data_nrm, pred_downsample_density
            )

        return data_pcd, data_nrm

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def downsample_points_and_normals(data_pcd, data_nrm, thresh):
    """
    用 voxel downsample 替换原来的 radius_neighbors 版本。
    """
    return voxel_downsample_points_and_normals(data_pcd, data_nrm, thresh)


def load_gt_points_and_normals(gt_path, gt_downsample_density=None):
    gt_pcd = o3d.io.read_point_cloud(str(gt_path))
    gt_pts = np.asarray(gt_pcd.points)
    if len(gt_pts) == 0:
        raise RuntimeError(f"Empty GT point cloud: {gt_path}")

    gt_nrm = np.asarray(gt_pcd.normals)
    if len(gt_nrm) != len(gt_pts):
        raise RuntimeError(f"GT point cloud has no valid normals: {gt_path}")

    gt_nrm = safe_normalize(gt_nrm)

    if gt_downsample_density is not None and gt_downsample_density > 0:
        gt_pts, gt_nrm = voxel_downsample_points_and_normals(
            gt_pts, gt_nrm, gt_downsample_density
        )

    return gt_pts, gt_nrm


def evaluate_single_pair(
    data_path,
    gt_path,
    vis_out_dir,
    case_name,
    method_name,
    mode='mesh',
    downsample_density=0.002,
    gt_downsample_density=None,
    pred_downsample_density=None,
    max_dist=0.1,
    visualize_threshold=0.01,
    dist_thred1=0.001,
    dist_thred2=0.002,
    knn_chunk_size=200000,
):
    os.makedirs(vis_out_dir, exist_ok=True)

    if gt_downsample_density is None:
        gt_downsample_density = downsample_density
    if pred_downsample_density is None:
        pred_downsample_density = downsample_density

    pbar_total = 8
    pbar = tqdm(total=pbar_total, desc=f"{case_name}/{method_name}", leave=True)

    pbar.set_description(f'{case_name}/{method_name} | read pred')
    data_pcd, data_nrm = load_data_as_points_and_normals(
        data_path,
        mode=mode,
        thresh=downsample_density,
        pred_downsample_density=pred_downsample_density
    )

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | downsample pred')
    data_down, nrm_down = downsample_points_and_normals(
        data_pcd, data_nrm, pred_downsample_density
    )

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | read gt')
    gt_pts, gt_nrm = load_gt_points_and_normals(
        gt_path, gt_downsample_density=gt_downsample_density
    )

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | build gt tree')
    gt_tree = cKDTree(gt_pts)

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | compute d2gt')
    dist_d2s, idx_d2s, nc_d2gt = chunked_1nn_query(
        tree=gt_tree,
        query_points=data_down,
        ref_normals=gt_nrm,
        query_normals=nrm_down,
        chunk_size=knn_chunk_size
    )
    nc_d2gt_mean = float(np.mean(nc_d2gt))

    valid_d2s = dist_d2s < max_dist
    if np.any(valid_d2s):
        mean_d2s = dist_d2s[valid_d2s].mean()
    else:
        mean_d2s = float('nan')

    precision_1 = float(np.mean(dist_d2s < dist_thred1))
    precision_2 = float(np.mean(dist_d2s < dist_thred2))

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | build pred tree')
    pred_tree = cKDTree(data_down)

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | compute gt2d')
    dist_s2d, idx_s2d, nc_gt2d = chunked_1nn_query(
        tree=pred_tree,
        query_points=gt_pts,
        ref_normals=nrm_down,
        query_normals=gt_nrm,
        chunk_size=knn_chunk_size
    )
    nc_gt2d_mean = float(np.mean(nc_gt2d))

    valid_s2d = dist_s2d < max_dist
    if np.any(valid_s2d):
        mean_s2d = dist_s2d[valid_s2d].mean()
    else:
        mean_s2d = float('nan')

    recall_1 = float(np.mean(dist_s2d < dist_thred1))
    recall_2 = float(np.mean(dist_s2d < dist_thred2))

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | visualize')
    vis_dist = visualize_threshold
    R = np.array([[1, 0, 0]], dtype=np.float64)
    G = np.array([[0, 1, 0]], dtype=np.float64)
    W = np.array([[1, 1, 1]], dtype=np.float64)

    data_alpha = np.clip(dist_d2s[:, None], None, vis_dist) / vis_dist
    data_color = R * data_alpha + W * (1 - data_alpha)
    data_color[dist_d2s >= max_dist] = G
    write_vis_pcd(str(Path(vis_out_dir) / f'vis_{case_name}_d2gt_{method_name}.ply'), data_down, data_color)

    stl_alpha = np.clip(dist_s2d[:, None], None, vis_dist) / vis_dist
    stl_color = R * stl_alpha + W * (1 - stl_alpha)
    stl_color[dist_s2d >= max_dist] = G
    write_vis_pcd(str(Path(vis_out_dir) / f'vis_{case_name}_gt2d_{method_name}.ply'), gt_pts, stl_color)

    pbar.update(1)
    pbar.set_description(f'{case_name}/{method_name} | finalize')
    pbar.close()

    over_all = (mean_d2s + mean_s2d) / 2
    nc_over_all = (nc_d2gt_mean + nc_gt2d_mean) / 2.0

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

        "mean_nc_d2gt": float(nc_d2gt_mean),
        "mean_nc_gt2d": float(nc_gt2d_mean),
        "mean_nc_over_all": float(nc_over_all),

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

        gt_file = None
        for ext in gt_exts:
            candidate = gt_root / f"{case_name}{ext}"
            if candidate.exists():
                gt_file = candidate
                break

        if gt_file is None:
            print(f"[WARN] skip case '{case_name}': GT file not found for basename '{case_name}' in {gt_root}")
            continue

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
            f'mean_nc_d2gt {np.round(result["mean_nc_d2gt"], 6)} '
            f'mean_nc_gt2d {np.round(result["mean_nc_gt2d"], 6)} '
            f'mean_nc_over_all {np.round(result["mean_nc_over_all"], 6)} \n'
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

        valid_nc_d2gt = [x["mean_nc_d2gt"] for x in items if np.isfinite(x["mean_nc_d2gt"])]
        valid_nc_gt2d = [x["mean_nc_gt2d"] for x in items if np.isfinite(x["mean_nc_gt2d"])]
        valid_nc_overall = [x["mean_nc_over_all"] for x in items if np.isfinite(x["mean_nc_over_all"])]

        per_method[m] = {
            "num_cases": len(items),
            "mean_over_all": float(np.mean(valid_overall)) if len(valid_overall) > 0 else float("nan"),
            "mean_d2gt": float(np.mean(valid_d2gt)) if len(valid_d2gt) > 0 else float("nan"),
            "mean_gt2d": float(np.mean(valid_gt2d)) if len(valid_gt2d) > 0 else float("nan"),
            "mean_nc_d2gt": float(np.mean(valid_nc_d2gt)) if len(valid_nc_d2gt) > 0 else float("nan"),
            "mean_nc_gt2d": float(np.mean(valid_nc_gt2d)) if len(valid_nc_gt2d) > 0 else float("nan"),
            "mean_nc_over_all": float(np.mean(valid_nc_overall)) if len(valid_nc_overall) > 0 else float("nan"),
        }
    return per_method


if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='root path of local dataset')
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])

    parser.add_argument('--downsample_density', type=float, default=0.01)
    parser.add_argument('--gt_downsample_density', type=float, default=None,
                        help='GT voxel downsample size; default = downsample_density')
    parser.add_argument('--pred_downsample_density', type=float, default=None,
                        help='Pred voxel downsample size; default = downsample_density')

    parser.add_argument('--patch_size', type=float, default=60)  # 保留，不动
    parser.add_argument('--max_dist', type=float, default=0.1)
    parser.add_argument('--visualize_threshold', type=float, default=0.01)
    parser.add_argument('--knn_chunk_size', type=int, default=200000)
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
                gt_downsample_density=args.gt_downsample_density,
                pred_downsample_density=args.pred_downsample_density,
                max_dist=args.max_dist,
                visualize_threshold=args.visualize_threshold,
                knn_chunk_size=args.knn_chunk_size,
            )

            print(
                f'over_all: {result["over_all"]}; '
                f'mean_d2gt: {result["mean_d2gt"]}; '
                f'mean_gt2d: {result["mean_gt2d"]}.'
            )
            print(
                f'nc_d2gt: {result["mean_nc_d2gt"]}; '
                f'nc_gt2d: {result["mean_nc_gt2d"]}; '
                f'nc_over_all: {result["mean_nc_over_all"]}.'
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
                f"mean_gt2d={s['mean_gt2d']:.6f}, "
                f"mean_nc_d2gt={s['mean_nc_d2gt']:.6f}, "
                f"mean_nc_gt2d={s['mean_nc_gt2d']:.6f}, "
                f"mean_nc_over_all={s['mean_nc_over_all']:.6f}\n"
            )

    if args.log is not None:
        with open(args.log, "w", encoding="utf-8") as f:
            json.dump(full_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] summary json saved to: {summary_json}")
    print(f"[INFO] summary txt  saved to: {summary_txt}")


"""
示例：
python deep_fasion_eval.py --root C:/Users/guanl/Downloads/OneDrive_1_3-11-2026/mesh/custom/synthetic --mode pcd --downsample_density 0.005 --gt_downsample_density 0.01 --knn_chunk_size 200000
python deep_fasion_eval.py --root C:/Users/guanl/Downloads/OneDrive_1_3-11-2026/tmp_test/lego/ --mode mesh --downsample_density 0.002 --gt_downsample_density 0.002 --knn_chunk_size 200000
"""