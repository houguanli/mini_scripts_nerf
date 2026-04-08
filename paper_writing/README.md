# paper_writing — 论文评估指标工具

用于计算 3D 重建与图像渲染质量的各类定量指标，支持批量评测并输出 JSON 报告。

## 文件说明

### 主要评测脚本

| 文件 | 指标 | 说明 |
|------|------|------|
| `MPF_cd_nc_evaluation.py` | CD、NC | 双向 Chamfer Distance + Normal Consistency，支持距离阈值过滤，批量处理多 case 多方法。 |
| `MPF_fscore_evaluation.py` | CD、NC、**F-score** | 在上表基础上扩展多阈值 F-score（Precision / Recall / F），默认阈值 0.005/0.01/0.02/0.05。 |
| `MPF_psnr_ssim_evaluation.py` | PSNR、SSIM | 渲染图像与 GT 的像素级质量评估，批量处理整个数据集。 |
| `PoseFusion_evaluate_mesh_cd.py` | CD | PoseFusion 方法专用的 Chamfer Distance 评测脚本。 |
| `calc_6DOF_error.py` | 旋转 / 平移误差 | 计算预测位姿与 GT 位姿之间的 6-DOF 误差（旋转角度误差 + 平移距离误差）。 |
| `cd_calculate.py` | CD | 轻量版 Chamfer Distance 计算，适合快速单对评测。 |
| `psnr_iou_calculate.py` | PSNR、IoU | 同时计算图像 PSNR 和分割 IoU 指标。 |
| `deep_fasion_eval.py` | PSNR、SSIM | 针对 DeepFashion 数据集的渲染质量评测。 |
| `deep_fasion_eval_cd.py` | CD | 针对 DeepFashion 数据集的几何质量评测。 |

### 可视化 / 辅助脚本

| 文件 | 说明 |
|------|------|
| `virtualize_2d_sdf.py` | 可视化 2D SDF（有向距离场）热力图。 |
| `virtulize_sdf_grid.py` | 可视化 3D SDF 网格切片。 |
| `view_coverage.py` | 分析相机视角覆盖率，统计场景各区域的可见帧数。 |
| `generate_visible_camera.py` | 生成仅包含可见特定区域的相机子集。 |
| `export_obb.py` | 从点云或网格导出有向包围盒（OBB）。 |
| `remove_imgae_border.py` | 批量裁去图像边缘黑边，用于渲染结果预处理。 |
| `combine_images.py` | 将多张结果图拼接为对比图，便于论文图表制作。 |
| `cacluate_offset.py` | 暴力搜索两张图像之间的像素偏移量。 |
| `sdf.npy` | 预计算的 SDF 数据文件（供可视化脚本使用）。 |

## 批量评测用法（MPF_fscore_evaluation.py）

```bash
python MPF_fscore_evaluation.py \
    --root "path/to/dataset" \
    --num_samples 100000 \
    --fscore_thresholds 0.005 0.01 0.02 0.05 \
    --cd_filter_threshold 0.02 \
    --output_json results.json
```

**目录结构约定**：
```
dataset/
  gt/              ← GT 网格: <case_name>.ply
  <case_name>/
    raw/           ← 预测网格: <method_name>.ply
```

## 依赖

- `numpy` / `scipy`
- `open3d`
- `trimesh`
- `scikit-image`（PSNR / SSIM）
- `Pillow`
