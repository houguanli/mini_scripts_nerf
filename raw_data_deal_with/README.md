# raw_data_deal_with — 原始数据处理与清洗

对采集到的原始图像、视频、网格和相机数据进行预处理，包括对齐、缩放、重命名和格式转换。

## 文件说明

| 文件 | 说明 |
|------|------|
| `align_cam_mesh_with_visualization.py` | **相机-网格对齐**。使用 PCA 估计主轴方向，通过 Umeyama 相似变换将相机坐标系和网格坐标系对齐，提供 Open3D 可视化验证。 |
| `apply_rt2_obj.py` | 将旋转矩阵 R 和平移向量 t 应用到 OBJ 网格顶点，输出变换后的 OBJ 文件。 |
| `clear_neus_exp.py` | 清理 NeuS 实验输出目录，删除中间检查点文件，保留最终结果。 |
| `exctract_frames.py` | 从视频文件中按指定帧率抽取图像帧，批量保存为 PNG/JPG。 |
| `scale_mesh.py` | 对网格进行等比缩放，通常用于归一化到单位球内。 |
| `rename_files.py` | 按规则批量重命名文件（如统一编号格式 `0001.png`）。 |
| `quad_and_eula.py` | 四元数与欧拉角相互转换的工具函数库。 |
| `test_load.py` | 测试各类数据文件（PLY、JSON、NPZ）的加载是否正确。 |

## 依赖

- `open3d`
- `numpy`
- `opencv-python`（视频帧提取）
