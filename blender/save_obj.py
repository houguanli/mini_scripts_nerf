import numpy as np
import sys   
import bpy 

bpy.ops.object.select_all()
for i in range(0, 1):
    bpy.context.scene.frame_end = 50
    tar_path = 'C:/Users/guanl/Desktop/GenshinNerf/t12/models/t_' + str(i) + '.obj'
    bpy.ops.export_scene.obj(filepath=tar_path, use_animation=True, use_triangles = True)