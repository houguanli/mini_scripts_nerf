import numpy as np
import sys
import bpy

selected_object = bpy.context.active_object

# 获取包围盒的最小和最大顶点坐标
# bbox_min, bbox_max = selected_object.bound_box.min, selected_object.bound_box.max

# 打印包围盒信息
print("物体名称:", selected_object.name)
print("包围盒坐标:", selected_object.bound_box)
for i in range(0, 8):
    bbox_min = selected_object.bound_box[i]
    print(selected_object.bound_box[i])
    print("包围顶点坐标:")
    print("X:", bbox_min[0])
    print("Y:", bbox_min[1])
    print("Z:", bbox_min[2])