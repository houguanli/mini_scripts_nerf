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
################################################################################################################
import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np

DEBUG = False

VIEWS = 10
RESOLUTION = 800
RESULTS_PATH = 'C:/Users/guanl/Desktop/GenshinNerf/bunny/0000'
obj_file_path = "C:/Users/guanl/Desktop/GenshinNerf/bunny/0000.obj"
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
UPPER_VIEWS = True
CIRCLE_FIXED_START = (0, 0, 0)
CIRCLE_FIXED_END = (.3, 0, 0)
fp = bpy.path.abspath(f"//{RESULTS_PATH}")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


if not os.path.exists(fp):
    os.makedirs(fp)

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}

id = 0
obj_file_name = f"{id:04}.obj"
# obj_file_path = os.path.join("C:/Users/guanl/Desktop/GenshinNerf/dp_simulation/duck/duck3d_s.obj", obj_file_name)  # 替换为你的.obj文件夹路径

# 检查文件是否存在
if os.path.exists(obj_file_path):
    # 删除前一帧的所有网格对象
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    #        # 导入新的.obj文件
    bpy.ops.import_scene.obj(filepath=obj_file_path)
    #        bpy.ops.import_scene.obj(filepath='P:/pd_eigen_mesh/obs.obj')

    # 选择新导入的对象
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
#    bpy.ops.transform.rotate(value=3.14159/2, orient_axis='X')  # 旋转45度，你可以更改value为其他值来旋转不同的角度


## Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
# bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if 'Custom Outputs' not in tree.nodes:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    if FORMAT == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        map = tree.nodes.new(type="CompositorNodeMapRange")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.inputs['From Min'].default_value = 0
        map.inputs['From Max'].default_value = 8
        map.inputs['To Min'].default_value = 1
        map.inputs['To Max'].default_value = 0
        links.new(render_layers.outputs['Depth'], map.inputs[0])

        links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.name = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background
objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})


def parent_obj_to_camera(b_camera):
    origin = (1.0 / 4, 0, 1.0 / 4)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = 800
scene.render.resolution_y = 600
H, W = scene.render.resolution_y, scene.render.resolution_x
scene.render.resolution_percentage = 100
camera_angle_x = bpy.data.objects['Camera'].data.angle_x
cam = scene.objects['Camera']
cam.location = (0.1, 0.3, 0.0)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians

stepsize = 360.0 / VIEWS
vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]
rotation_mode = 'XYZ'

if not DEBUG:
    for output_node in [tree.nodes['Depth Output'], tree.nodes['Normal Output']]:
        output_node.base_path = ''

out_data['frames'] = []

b_empty.rotation_euler = CIRCLE_FIXED_START
b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + vertical_diff
total_data = {}
for i in range(0, VIEWS):
    if DEBUG:
        i = np.random.randint(0, VIEWS)
        b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize * i)) + 1) / 2 * vertical_diff
        b_empty.rotation_euler[2] += radians(2 * stepsize * i)

    print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
    scene.render.filepath = fp + '/' + f"{(i + 1):04}"

    tree.nodes['Depth Output'].file_slots[0].path = scene.render.filepath + "_depth_"
    tree.nodes['Normal Output'].file_slots[0].path = scene.render.filepath + "_normal_"
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    intrinsic_matrix = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])
    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still
    #    total_data[str(i+1) + '_1_M'] =  listify_matrix(np.linalg.inv(cam.matrix_world))
    c2w_blender = np.array(cam.matrix_world)
    #    print(c2w_blender.type)
    c2w_blender[:, 1:3] = c2w_blender[:, 1:3] * -1
    total_data[str(i + 1) + '_1_M'] = listify_matrix(c2w_blender)
    total_data[str(i + 1) + '_1_K'] = listify_matrix(intrinsic_matrix)
    #    "1_1_M" "1_1_K" "1_2_M" ... etc
    #    frame_data = {
    #        'file_path': scene.render.filepath,
    #        'rotation': radians(stepsize),
    #        'transform_matrix': listify_matrix(cam.matrix_world),
    #        'intrinsic_matrix': listify_matrix(intrinsic_matrix)
    #    }
    #    out_data['frames'].append(frame_data)

    b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize * i)) + 1) / 2 * vertical_diff
    b_empty.rotation_euler[2] += radians(2 * stepsize)

if not DEBUG:
    with open(fp + '/' + 'cameras_blender.json', 'w') as out_file:
        json.dump(total_data, out_file, indent=4)

# 定义导入函数
# def load_obj_for_frame(scene):
#    frame_number = scene.frame_current
#    # 根据帧号生成文件名
#    obj_file_name = f"{frame_number:04}.obj"
#    obj_file_path = os.path.join("P:/pd_eigen_mesh", obj_file_name)  # 替换为你的.obj文件夹路径
#
#    # 检查文件是否存在
#    if os.path.exists(obj_file_path):
#        # 删除前一帧的所有网格对象
#        bpy.ops.object.select_all(action='DESELECT')
#        bpy.ops.object.select_by_type(type='MESH')
#        bpy.ops.object.delete()

##        # 导入新的.obj文件
#        bpy.ops.import_scene.obj(filepath=obj_file_path)
#        bpy.ops.import_scene.obj(filepath='P:/pd_eigen_mesh/obs.obj')

#        # 选择新导入的对象
#        bpy.ops.object.select_all(action='DESELECT')
#        bpy.ops.object.select_by_type(type='MESH')

#        # 旋转选定的对象
#        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
#        bpy.ops.transform.rotate(value=3.14159/2, orient_axis='X')  # 旋转45度，你可以更改value为其他值来旋转不同的角度

## 将函数添加为帧更改处理器
# bpy.app.handlers.frame_change_pre.clear()
# bpy.app.handlers.frame_change_pre.append(load_obj_for_frame)
######################################################################################################################
import argparse, sys, os
import json
import bpy
import mathutils
import numpy as np

DEBUG = False

VIEWS = 30
RESOLUTION = 800
RESULTS_PATH = '/Users/houguanli/Desktop/virtual_data/static/rabbit/image'
RESULTS_PATH_ = '/Users/houguanli/Desktop/virtual_data/static/rabbit/image'
obj_file_path = "/Users/houguanli/Desktop/virtual_data/object/rabbit/rabbit_colored.obj"
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
UPPER_VIEWS = True
CIRCLE_FIXED_START = (0, 0, 0)
CIRCLE_FIXED_END = (.6, 0, 0)
fp = bpy.path.abspath(f"//{RESULTS_PATH}")
for obj in bpy.context.scene.objects:
    if obj.type == 'LIGHT':
        obj.select_set(True)
    else:
        obj.select_set(False)
bpy.ops.object.delete()
pi, l_dis=3.14159, 2.8 # light distance
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, 0, l_dis), rotation=(0,0,0),)
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, 0, -l_dis), rotation=(0,pi,0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(l_dis, 0, 0), rotation=(0,pi/2,0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(-l_dis, 0, 0), rotation=(0,-pi/2,0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, l_dis, 0), rotation=(-pi/2,pi/2,0))
bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, -l_dis, 0), rotation=(0,pi/2,-pi/2))
for obj in bpy.context.scene.objects:
    if obj.type == 'LIGHT':
        obj.data.energy=50
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


if not os.path.exists(fp):
    os.makedirs(fp)

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x
}

id = 0
obj_file_name = f"{id:04}.obj"
# obj_file_path = os.path.join("C:/Users/guanl/Desktop/GenshinNerf/dp_simulation/duck/duck3d_s.obj", obj_file_name)  # 替换为你的.obj文件夹路径

# 检查文件是否存在
if os.path.exists(obj_file_path):
    # 删除前一帧的所有网格对象
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    #        # 导入新的.obj文件
    bpy.ops.import_scene.obj(filepath=obj_file_path)
    #        bpy.ops.import_scene.obj(filepath='P:/pd_eigen_mesh/obs.obj')

    # 选择新导入的对象
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
else:
    print("Invaild input path!!!")
#    bpy.ops.transform.rotate(value=3.14159/2, orient_axis='X')  # 旋转45度，你可以更改value为其他值来旋转不同的角度


## Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
# bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if 'Custom Outputs' not in tree.nodes:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    if FORMAT == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        map = tree.nodes.new(type="CompositorNodeMapRange")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.inputs['From Min'].default_value = 0
        map.inputs['From Max'].default_value = 8
        map.inputs['To Min'].default_value = 1
        map.inputs['To Max'].default_value = 0
        links.new(render_layers.outputs['Depth'], map.inputs[0])

        links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.name = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background
objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})

def parent_obj_to_camera(b_camera):
    origin = (0.0 / 4, 0, 0.0 / 4)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting
    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    return b_empty

scene = bpy.context.scene
scene.render.resolution_x = 800
scene.render.resolution_y = 600
H, W = scene.render.resolution_y, scene.render.resolution_x
scene.render.resolution_percentage = 100
camera_angle_x = bpy.data.objects['Camera'].data.angle_x
cam = scene.objects['Camera']
cam.location = (0.1, 0.25, 0.1)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians

stepsize = 360.0 / VIEWS
vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]
rotation_mode = 'XYZ'

if not DEBUG:
    for output_node in [tree.nodes['Depth Output'], tree.nodes['Normal Output']]:
        output_node.base_path = ''

out_data['frames'] = []

b_empty.rotation_euler = CIRCLE_FIXED_START
b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + vertical_diff
total_data = {}
# two factor rendering to have bottom view
for i in range(0, VIEWS):
    if DEBUG:
        i = np.random.randint(0, VIEWS)
        b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize * i)) + 1) / 2 * vertical_diff
        b_empty.rotation_euler[2] += radians(2 * stepsize * i)

    print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
    scene.render.filepath = fp + '/' + f"{(i):04}"
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    intrinsic_matrix = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])
    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still
    c2w_blender = np.array(cam.matrix_world)
    #    print(c2w_blender.type)
    c2w_blender[:, 1:3] = c2w_blender[:, 1:3] * -1
    total_data[str(i) + '_M'] = listify_matrix(c2w_blender)
    total_data[str(i) + '_K'] = listify_matrix(intrinsic_matrix)
    # render z-filp
    add_z_flip=True
    if add_z_flip:
        b_empty.rotation_euler[1] = b_empty.rotation_euler[1] + 180
        scene.render.filepath = fp + '/' + f"{(i + VIEWS):04}"
        if DEBUG:
            break
        else:
            bpy.ops.render.render(write_still=True)  # render still
        c2w_blender = np.array(cam.matrix_world)
        #    print(c2w_blender.type)
        c2w_blender[:, 1:3] = c2w_blender[:, 1:3] * -1
        total_data[str(i + VIEWS) + '_M'] = listify_matrix(c2w_blender)
        total_data[str(i + VIEWS) + '_K'] = listify_matrix(intrinsic_matrix)
        b_empty.rotation_euler[1] =  b_empty.rotation_euler[1] - 180 # reset the rotation angle
    #upd b_empty
    b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize * i)) + 1) / 2 * vertical_diff
    b_empty.rotation_euler[2] += radians(2 * stepsize)

if not DEBUG:
    with open(fp + '/' + 'cameras_blender.json', 'w') as out_file:
        json.dump(total_data, out_file, indent=4)

if not DEBUG:
    with open(fp + '/' + 'cameras_sphere.json', 'w') as out_file:
        json.dump(total_data, out_file, indent=4)