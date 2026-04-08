# reg — 配准算法

提供基于特征的图像配准和点云配准算法，包括 RANSAC 和 Log-Polar FFT 旋转检测。

## 文件说明

| 文件 | 说明 |
|------|------|
| `RANSAC.py` | 基于 **Open3D** 的点云全局配准。提取 FPFH 特征，使用 RANSAC 求解初始变换矩阵，输出旋转矩阵和平移向量。 |
| `image_feature_detect.py` | 基于 **Log-Polar FFT + 归一化互相关（NCC）** 的图像旋转角度检测，适用于无纹理或弱纹理场景。 |
| `fricp_reg.bat` | 调用 Fast-RICP 工具进行快速鲁棒 ICP 配准的批处理脚本。 |
| `obb_reg.bat` | 基于有向包围盒（OBB）进行初始对齐的批处理脚本。 |
| `ransac.bat` | 批量执行 RANSAC 配准流程的 Windows 脚本。 |
| `loss_heatmap.png` | 配准过程中生成的损失热力图（可视化结果）。 |
| `loss_results.csv` | 配准迭代过程的损失数值记录。 |

## 依赖

- `open3d`
- `numpy`
- `scikit-image`（FFT 配准）
