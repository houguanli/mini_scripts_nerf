# blender — Blender Python 脚本集

在 Blender 内部 Python 环境（`bpy`）中运行的脚本，用于网格生成、场景渲染和视频合成。

## 文件说明

| 文件 | 说明 |
|------|------|
| `render_png_single_camera.py` | 按螺旋轨迹生成相机序列并批量渲染 PNG。设置面积光源，输出用于 NeRF 训练的多视角图像。 |
| `mesh_maker.py` | 程序化生成简单几何体（斜坡、坡道等），导出为 OBJ 文件。 |
| `blender_video_maker.py` | 将渲染好的帧序列合成为视频。 |
| `blend.py` | 对图像的 Alpha 通道进行合成混合操作。 |
| `full_save_scps.py` | 批量保存 Blender 场景脚本。 |
| `save_obj.py` | 将当前 Blender 场景中的选中对象导出为 OBJ。 |
| `print_bbox.py` | 打印场景或选中对象的包围盒信息。 |

## 运行方式

脚本需在 Blender 内置 Python 环境中运行：

```bash
blender --background scene.blend --python render_png_single_camera.py
```

或在 Blender 脚本编辑器中直接粘贴运行。

## 依赖

- `bpy`（Blender 内置）
- `numpy`
- `opencv-python`（部分脚本）
