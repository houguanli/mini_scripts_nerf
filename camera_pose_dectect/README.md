# camera_pose_dectect — 相机位姿估计与可视化

从投影矩阵、JSON 或 NPZ 文件中解析相机位姿，支持 ArUco/QR 标定板生成和相机射线可视化。

## 文件说明

| 文件 | 说明 |
|------|------|
| `show_cameras.py` | 加载投影矩阵，分解内参 K 和外参 [R\|t]，用 **Open3D** 在三维空间中可视化相机锥体。 |
| `QR_generator.py` | 生成 ArUco 标定板图案，用作位姿标定的物理靶标。 |
| `show_rays.py` | 根据相机参数可视化各像素对应的射线方向，辅助验证相机模型正确性。 |
| `json2npz.py` | 将 JSON 格式的相机参数批量转换为 NPZ（NumPy 二进制）格式。 |
| `libdregnerf.py` | NeRF 相关的微分渲染辅助库。 |
| `frame.png` | 测试用图像帧。 |
| `test_npz.npz` | 测试用 NPZ 相机参数数据。 |

## 依赖

- `opencv-python`（含 `cv2.aruco`）
- `open3d`
- `numpy`
