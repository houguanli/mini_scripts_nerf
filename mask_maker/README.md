# mask_maker — 遮罩生成工具

为 NeRF / 体渲染训练生成前景遮罩（mask），支持手动交互绘制和基于网格的自动生成。

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_mask.py` | **交互式遮罩绘制工具**。打开 OpenCV 窗口，支持鼠标涂抹（可调笔刷大小），将绘制结果保存为二值 PNG 遮罩。 |
| `first_mask_maker.py` | 生成初始遮罩（粗略框选前景区域），供后续精细化使用。 |
| `mesh_mask_maker.py` | 将 3D 网格投影到各相机视角，自动生成对应的遮罩图像，适合有已知网格时的批量处理。 |
| `seggpt_bat_maker.py` | 生成批量调用 **SegGPT** 分割模型的脚本，用于自动前景分割。 |

## 使用方式

```bash
# 交互式绘制遮罩
python generate_mask.py --input images/ --output masks/

# 从网格自动生成遮罩
python mesh_mask_maker.py --mesh model.ply --cameras transforms.json --output masks/
```

## 依赖

- `opencv-python`
- `numpy`
- `open3d`（mesh_mask_maker）
