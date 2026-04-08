# about_colmap — COLMAP 相机参数提取与转换

从 COLMAP 稀疏重建结果中提取相机内外参，并转换为 NeRF/NeuS 所需的 JSON 格式。

## 文件说明

| 文件 | 说明 |
|------|------|
| `camera_convert.py` | 核心转换脚本。读取 COLMAP 的 `images.txt` / `cameras.txt`，将四元数转旋转矩阵，提取内参矩阵 K 和外参 [R\|t]，输出 JSON。 |
| `cp_3D2obj.py` | 将 COLMAP 生成的稀疏 3D 点云转换为 OBJ 格式，方便在 Blender 等工具中查看。 |
| `get_bbox.py` | 从点云或变换矩阵中计算场景的轴对齐包围盒（AABB），用于确定 NeRF 训练范围。 |
| `colmap_bat.txt` | COLMAP 完整流程命令参考（特征提取 → 匹配 → 稀疏重建）。 |
| `colmap_t1.bat` | Windows 批处理脚本，自动执行 COLMAP 重建流程。 |

## 典型工作流

```
图像目录
  → colmap_t1.bat          (特征提取 + 匹配 + 稀疏重建)
  → camera_convert.py      (导出相机参数 JSON)
  → get_bbox.py            (估算场景包围盒)
```

## 依赖

- `numpy`
- COLMAP 命令行工具（需单独安装）
