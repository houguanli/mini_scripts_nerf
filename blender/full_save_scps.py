###############
#use this to save objs
###############

import numpy as np
import sys   
import bpy 

bpy.ops.object.select_all()
tar_path = 'D:/simulation_obj/test2/_.obj' 
bpy.ops.export_scene.obj(filepath=tar_path, use_animation=True,use_triangles=True,use_materials=True)

########################################################################################################

#################
#use this import obj, add lights and render imgs, also save camera infos
#################
import bpy
import math
import numpy as np
import json
from bpy import context
scene = context.scene
# clear old
bpy.ops.object.select_all()
bpy.ops.object.delete()# 1. 创建一个新的区域光源
pi=3.14159
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, 0, 2), rotation=(0,0,0),)
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, 0, -2), rotation=(0,pi,0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(2, 0, 0), rotation=(0,pi/2,0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(-2, 0, 0), rotation=(0,-pi/2,0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, 2, 0), rotation=(-pi/2,pi/2,0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, -2, 0), rotation=(0,pi/2,-pi/2))
for obj in bpy.context.scene.objects:
    if obj.type == 'LIGHT':
        obj.data.energy=50
bpy.ops.object.camera_add(location=(0, 0, 2),rotation=(0,0,0))
camera = bpy.context.active_object
scene.camera = context.object
render_width = 512
render_height = 512

bpy.context.scene.render.resolution_x = render_width
bpy.context.scene.render.resolution_y = render_height
##a big loop
for index in range(1,10):
    #2. import data
    file_path = 'D:\\simulation_obj\\test2\\__00000' + str(index) + '.obj' 
    file_name = "__00000" + str(index)
    bpy.ops.import_scene.obj(filepath=file_path)
    json_file_path = "D:/simulation_obj/test2/r_shoot/camera.json"
    # 3. render a set of pics
    # 设置渲染设置
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    camera.data.lens = 35
    r = 2.5
    for i in range(5):
        # cal rotation
        angle = 2 * pi * i / 5
        camera_name = str(index) +"_" + str(i + 1)
        bpy.context.scene.render.filepath = "D:/simulation_obj/test2/r_shoot/" + camera_name + ".png"
        # cal new position
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        camera.location = (x, y, 2)

        # toward O
        look_target = bpy.data.objects.new(name="Look Target", object_data=None)
        bpy.context.collection.objects.link(look_target)
        look_target.location = (0,0,0)
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
        cx = pixels_width / 2
        cy = pixels_height / 2

        # 创建内部参数矩阵
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        K_name = camera_name + "_K"
        json_data = {M_name: M.tolist(), K_name : K.tolist(), M_inv_name : M_inv.tolist()}       
        try:
            with open(json_file_path, 'r') as json_file:
                json_content = json.load(json_file)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            json_content = []
        json_content.append(json_data)
        # 重新写入文件
        with open(json_file_path, 'w') as json_file:
            json.dump(json_content, json_file, indent=4)
            
        #delete after        
        bpy.data.objects.remove(look_target)
        
    #delete old obj
    for obj in bpy.data.objects:
        # 选择物体
        if obj.type == 'MESH':
            obj.select_set(True)
    # 删除所有已选中的物体
    bpy.ops.object.delete()
########################################################################################################
#################
#use this to make camera shoot circle
#################
import bpy
import mathutils
import numpy as np
from bpy import context
scene = context.scene
# 1. 添加一个摄像机

bpy.ops.object.camera_add(location=(0, 0, 2),rotation=(0,0,0))
camera = bpy.context.active_object
scene.camera = context.object

# 2. 拍摄一张照片
# 设置渲染设置
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = "D:/simulation_obj/test2/rendered_image.png"
bpy.ops.render.render(write_still=True)

# 3. 获取并打印相机的内外参数
# 内参数
K = camera.data.lens
print(f"Camera focal length (in Blender units): {K}")

# 外参数 - 旋转和平移矩阵
R = camera.matrix_world.to_3x3()
T = camera.location
print(f"Rotation matrix: {R}")
print(f"Translation vector: {T}")

# 如果你需要完整的4x4变换矩阵
M = camera.matrix_world
print(f"Transformation matrix: \n{M}")

#delete after
camera.select_set(True)
bpy.context.view_layer.objects.active = camera
bpy.ops.object.delete()