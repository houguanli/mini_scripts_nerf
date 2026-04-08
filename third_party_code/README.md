# third_party_code — 第三方算法与研究代码

收集整理的第三方实现和实验性研究代码，包括物理仿真、多视角渲染和网格处理。

## 文件说明

### 渲染

| 文件 | 说明 |
|------|------|
| `render_mvimg_from_mesh.py` | 使用 **trimesh + pyrender** 从网格生成多视角渲染图像，支持自定义光源和相机位姿。 |

### 物理仿真

| 文件 | 说明 |
|------|------|
| `diff_rigid_body_torch_only.py` | 纯 **PyTorch** 实现的可微刚体动力学，用于梯度优化。 |
| `rigid_body_dynamics.py` | 标准（非可微）刚体动力学仿真。 |
| `tow_body.py` | 两刚体碰撞交互仿真。 |

### 网格 / SDF 处理

| 文件 | 说明 |
|------|------|
| `simplify_meshes.py` | 批量网格简化工具，基于 trimesh。 |
| `sdf_generate.py` | 从三角网格生成 SDF（有向距离场）体素网格。 |
| `class_3d_model.py` | 3D 模型加载与几何计算基础类（与 QEM/ 共享逻辑）。 |
| `cp_imges.py` | 图像批量复制 / 整理工具。 |
| `tmp_show_paper_timeline.py` | 论文相关方法时间线可视化（临时脚本）。 |

### 数据文件

| 文件 / 目录 | 说明 |
|------------|------|
| `mesh_result/` | 仿真生成的网格序列（`0001.obj` … `0060.obj`）及测试场景 `slope_new.obj`。 |
| `output/` | 两体动力学仿真输出的网格帧序列（`0_1.obj`, `0_2.obj` …）。 |
| `output.zip` | `output/` 的压缩存档。 |
| `nerf_surface_timeline.pdf/png` | NeRF 与表面重建方法发展时间线图。 |
| `imgui.ini` | ImGui 界面配置文件。 |

## 依赖

- `trimesh`
- `pyrender`
- `torch`
- `numpy`
- `open3d`
