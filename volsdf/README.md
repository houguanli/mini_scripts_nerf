# volsdf — VolSDF 辅助工具

为 **VolSDF**（体渲染 + SDF 隐式表面重建）训练和后处理提供辅助脚本，包括有向包围盒生成、相机加载和位姿去重。

## 文件说明

| 文件 | 说明 |
|------|------|
| `obb_maker.py` | 从点云或网格计算**有向包围盒（OBB）**。使用 PCA 估计主轴，输出旋转矩阵和尺寸，支持四元数格式导出。 |
| `camera_loader.py` | 加载 VolSDF / NeuS 格式的相机参数文件（JSON / NPZ），统一转换为内参 K 和外参矩阵。 |
| `similar_cam_pose_detect.py` | 检测并过滤数据集中**相似度过高的相机位姿**，通过旋转距离和位移距离双重阈值去重，减少训练冗余。 |
| `white_bk_generate.py` | 为合成数据集中的透明背景图像批量生成白色背景版本。 |
| `run_obb_reg.bat` | 调用 OBB 配准流程的 Windows 批处理脚本。 |

## 使用示例

```bash
# 生成场景 OBB
python obb_maker.py --input scene.ply --output obb.json

# 过滤相似相机位姿
python similar_cam_pose_detect.py \
    --transforms transforms.json \
    --rot_threshold 5.0 \
    --trans_threshold 0.05 \
    --output filtered_transforms.json
```

## 依赖

- `open3d`
- `numpy`
- `matplotlib`
