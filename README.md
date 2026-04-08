# mini_scripts — NeRF / 3D 重建工具箱

面向神经渲染（NeRF、VolSDF）及相关 3D 重建研究的实用脚本集合。  
涵盖从数据采集、相机标定、数据集制作，到网格处理、渲染、评估的完整 pipeline。

---

## 目录结构

| 目录 | 功能简述 |
|------|---------|
| [about_colmap/](about_colmap/) | COLMAP 相机参数提取与格式转换 |
| [blender/](blender/) | Blender Python 脚本，渲染与网格生成 |
| [camera_pose_dectect/](camera_pose_dectect/) | 相机位姿估计与可视化 |
| [dataset_maker/](dataset_maker/) | 数据集制作，QR/棋盘格标定，格式转换 |
| [mask_maker/](mask_maker/) | 交互式遮罩生成工具 |
| [paper_writing/](paper_writing/) | 论文评估指标（CD、NC、F-score、PSNR/SSIM 等） |
| [raw_data_deal_with/](raw_data_deal_with/) | 原始数据清洗、对齐、格式转换 |
| [ICP/](ICP/) | 迭代最近点（ICP）点云配准 |
| [reg/](reg/) | RANSAC 配准与图像特征检测 |
| [QEM/](QEM/) | 二次误差度量（QEM）网格简化 |
| [third_party_code/](third_party_code/) | 第三方算法：多视角渲染、可微刚体动力学、SDF |
| [ue_python/](ue_python/) | Unreal Engine 5 Python API，全景图批量渲染 |
| [volsdf/](volsdf/) | VolSDF 辅助工具，OBB 生成，相似位姿去重 |

## 总体 Pipeline

```
图像采集
  ↓
camera_pose_dectect / dataset_maker   (标定 & 位姿估计)
  ↓
about_colmap                           (COLMAP 稀疏重建)
  ↓
mask_maker                             (前景分割)
  ↓
raw_data_deal_with                     (数据清洗 & 对齐)
  ↓
[NeRF / VolSDF 训练]
  ↓
blender / ue_python                    (渲染验证)
  ↓
paper_writing                          (指标评估)
```

## 主要依赖

- **NumPy / SciPy**
- **OpenCV** — 图像处理、ArUco 标定
- **Open3D** — 点云处理、ICP、可视化
- **trimesh / pyrender** — 网格处理与渲染
- **Blender (bpy)** — 3D 建模与渲染脚本
- **Unreal Engine Python API (unreal)** — 全景渲染
- **PyTorch** — 可微物理仿真
- **scikit-image** — PSNR / SSIM 评估
