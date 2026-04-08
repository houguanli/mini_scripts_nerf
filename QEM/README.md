# QEM — 二次误差度量网格简化

基于 **Quadric Error Metrics (QEM)** 的三角网格简化算法，可按比例缩减顶点数量同时保持几何精度。

## 文件说明

| 文件 | 说明 |
|------|------|
| `class_3d_model.py` | 3D 模型基类，读取 OBJ 文件，存储顶点、面、法向，计算各面的平面方程和 Q 矩阵。 |
| `QEM.py` | 核心简化算法。继承 `a_3d_model`，实现有效边对选取、误差计算、堆排序迭代收缩，输出简化后的 OBJ。 |
| `QEM_runner.py` | 命令行入口，指定输入文件、简化比例 `ratio` 和距离阈值 `threshold`。 |

## 使用方式

```bash
python QEM_runner.py --input model.obj --ratio 0.5 --threshold 0.01
# ratio: 目标保留顶点比例 (0, 1]
# threshold: 判定有效边对的距离上限
```

## 依赖

- `numpy`
