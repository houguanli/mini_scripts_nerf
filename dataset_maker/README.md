# dataset_maker — 数据集制作与格式转换

从多种来源（QR 码、ArUco 标定板、棋盘格、EXIF 元数据）估计相机参数，并将 NeuS 格式数据集转换为 NeRF 训练所需格式。

## 文件说明

| 文件 | 说明 |
|------|------|
| `detect_qr_pose.py` | 从图像中检测 QR 码位姿，将旋转矩阵转换为四元数并写入 JSON。 |
| `detect_c2w_from_3x3qrs.py` | 利用 3×3 排列的 QR 码阵列估算相机到世界的变换矩阵（camera-to-world）。 |
| `detect_single_w2c_from_qr.py` | 从单张图像中的 QR 码估算 world-to-camera 矩阵。 |
| `detect_k_from_EXIF.py` | 从图像 EXIF 元数据中推断相机内参矩阵 K（焦距、传感器尺寸）。 |
| `cal_coeff_from_chessboard.py` | 使用棋盘格标定图像计算相机内参与畸变系数。 |
| `pack_neus_to_nerf.py` | 将 NeuS 格式的相机参数和图像路径打包为 NeRF 的 `transforms.json`。 |
| `batch_pack_neus_to_nerf.bat` | 批量执行 `pack_neus_to_nerf.py` 的 Windows 脚本。 |
| `generate_mat_json_from_dict.py` | 从 Python 字典生成相机矩阵 JSON 文件的工具函数。 |
| `qr_detect_test.py` | QR 码检测功能的单元测试脚本。 |
| `zip_imgAcalc_new_K.py` | 批量处理图像并根据新内参矩阵 K 重新计算 / 保存结果。 |

## 依赖

- `opencv-python`（含 ArUco、棋盘格标定模块）
- `numpy`
- `Pillow`（EXIF 读取）
