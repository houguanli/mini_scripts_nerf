# ue_python — Unreal Engine 5 全景渲染脚本

在 **Unreal Engine 5** 内部 Python 环境中运行，实现全景图（Panoramic）的批量渲染、格式转换和多视角合并。

## 文件说明

### 批量渲染

| 文件 | 说明 |
|------|------|
| `batch_render_pano.py` | 按 +X/-X/+Y/-Y 顺序批量渲染全景图。首次运行自动生成 `render_plan.json`，支持断点续渲（已完成的 shot 自动跳过）。 |
| `batch_render_pano_12shot.py` | 12 视角全景渲染变体，覆盖更密集的方向分布。 |
| `batch_export_pano_rgb_normal_depth.py` | 批量导出全景 RGB + 法线 + 深度三通道图像。 |

### 单帧渲染

| 文件 | 说明 |
|------|------|
| `single_raw_pano_rendering.py` | 单张全景图渲染核心函数，被批量脚本调用。 |
| `single_raw_pano_rendering_in_double_cube.py` | 使用双立方体（Double Cube）展开方式的单帧渲染。 |
| `export_pano_image.py` | 导出单张全景图工具函数。 |

### 合并与格式转换

| 文件 | 说明 |
|------|------|
| `merge_double_cube.py` | 将双立方体格式的六个面合并为完整全景图。 |
| `batch_merge_pano_in_double_cube.py` | 批量执行双立方体全景合并。 |
| `convert_exr2png_rgb_normal_depth.py` | 将 UE 导出的 EXR 多层文件拆分转换为 RGB / 法线 / 深度三张 PNG。 |
| `virtualize_exr.py` | 可视化 EXR 文件中的各通道数据。 |

## 运行环境

脚本需在 **UE5 内置 Python 解释器**中运行（通过 Unreal Python 插件启用）：

```python
# 在 UE 编辑器 Python 控制台中执行
import batch_render_pano
batch_render_pano.run(output_dir="D:/renders/")
```

## 依赖

- `unreal`（UE5 内置模块）
- `os`, `sys`, `json`（标准库）
- `OpenEXR` / `Imath`（EXR 格式转换，可选）
