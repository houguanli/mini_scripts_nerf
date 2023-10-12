import bpy
import math
import numpy as np
import json
from bpy import context


def generate_spiral_trajectory(t):
    if t < 1 or t > 30:
        raise ValueError("t should be in the range [1, 30]")

    # 计算螺旋线的参数
    a = 2 * np.pi * 3  # 三圈
    b = 1.5 / (2 * np.pi * 3)  # 半球半径为1.5，对应的 z(t) 参数

    # 计算 x(t)、y(t) 和 z(t)
    theta = np.linspace(0, 3 * 2 * np.pi, t)  # 在[0, 3 * 2 * pi]范围内生成t个均匀分布的角度
    x = 1.5 * np.cos(a * theta)
    y = 1.5 * np.sin(a * theta)
    z = b * theta

    # 创建 NumPy 三维向量
    trajectory = tuple([x, y, z])

    # 最终返回 NumPy 三维向量
    return trajectory


scene = context.scene
# clear old
bpy.ops.object.select_all()
bpy.ops.object.delete()  # 1. 创建一个新的区域光源

pi = 3.14159
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, 0, 2), rotation=(0, 0, 0), )
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, 0, -2), rotation=(0, pi, 0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(2, 0, 0), rotation=(0, pi / 2, 0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(-2, 0, 0), rotation=(0, -pi / 2, 0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, 2, 0), rotation=(-pi / 2, pi / 2, 0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, -2, 0), rotation=(0, pi / 2, -pi / 2))
for obj in bpy.context.scene.objects:
    if obj.type == 'LIGHT':
        obj.data.energy = 50
bpy.ops.object.camera_add(location=(0, 0, 2), rotation=(0, 0, 0))
camera = bpy.context.active_object
scene.camera = context.object
render_width = 512
render_height = 512
bpy.context.scene.render.resolution_x = render_width
bpy.context.scene.render.resolution_y = render_height
all_json_data = {}
json_file_path = "C:/Users/GUANL/Desktop/GenshinNerf/t12/images/camera.json"
for index in range(1, 30):
    # 2. import data
    time_index_str = f"{index:06d}"
    filename = f"0_{time_index_str}"
    file_path = 'C:/Users/GUANL/Desktop/GenshinNerf/t12/models/t_' + filename + '.obj'
    print(file_path)
    bpy.ops.import_scene.obj(filepath=file_path)

    # 3. render a set of pics
    # 设置渲染设置
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    camera.data.lens = 35
    r = 1.5
    # cal rotation
    camera_name = str(index) + "_1"
    bpy.context.scene.render.filepath = "C:/Users/GUANL/Desktop/GenshinNerf/t12/images/" + camera_name + ".png"
    # cal new position
    trajectory = generate_spiral_trajectory(index)
    print("calc tra: " + str(trajectory))
    #  camera.location = (trajectory[0],trajectory[1],trajectory[2])
    camera.location = trajectory

    # toward O
    look_target = bpy.data.objects.new(name="Look Target", object_data=None)
    bpy.context.collection.objects.link(look_target)
    look_target.location = (0, 0, 0)
    # point to O
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = look_target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    # render, save pic, write camera matric
    bpy.ops.render.render(write_still=True)
    M = np.array(camera.matrix_world)
    M_inv = np.linalg.inv(M)
    M_name = camera_name + "_M"
    M_inv_name = camera_name + "_M_inv"
    # generate (HWK) as a scalar mat
    focal_length = camera.data.lens
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height
    pixels_width = bpy.context.scene.render.resolution_x
    pixels_height = bpy.context.scene.render.resolution_y
    # 计算像素尺寸
    pixel_size_x = sensor_width / pixels_width
    pixel_size_y = sensor_height / pixels_height
    fx = focal_length / pixel_size_x
    fy = focal_length / pixel_size_y
    cx = 500 / 2
    cy = pixels_height / 2

    # 创建内部参数矩阵
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    K_name = camera_name + "_K"
    json_data = {M_name: M.tolist(), K_name: K.tolist(), M_inv_name: M_inv.tolist()}
    all_json_data[M_name] = M.tolist()
    all_json_data[K_name] = K.tolist()
    all_json_data[M_inv_name] = M_inv.tolist()
    # delete after
    bpy.data.objects.remove(look_target)
    look_target.select_set(True)
    bpy.ops.object.delete()
    for obj in bpy.data.objects:
        # 选择物体
        if obj.type == 'MESH':
            obj.select_set(True)
            # 删除所有已选中的物体
        bpy.ops.object.delete()

with open(json_file_path, 'w') as json_file:
    json.dump(all_json_data, json_file, indent=4)