# ICP — 迭代最近点点云配准

使用 ICP (Iterative Closest Point) 算法对两个点云 / 网格进行精配准。

## 文件说明

| 文件 | 说明 |
|------|------|
| `o3d_icp.py` | 基于 **Open3D** 的 ICP 配准。将两个 PLY/OBJ 网格转换为点云，体素降采样后执行 Point-to-Point ICP，输出配准后的变换矩阵并合并保存。 |
| `trimesh_icp.py` | 基于 **trimesh** 的 ICP 实现，提供另一套接口。 |

## 使用示例

```python
# o3d_icp.py — 修改脚本顶部路径变量后直接运行
pc0_path = "dragon_pos1.ply"
pc1_path = "dragon_pos2.ply"
out_path = "dragon_merge.ply"
python o3d_icp.py
```

## 依赖

- `open3d`
- `trimesh`
- `numpy`
