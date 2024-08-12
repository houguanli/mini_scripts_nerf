import json
import os
import numpy as np
from scipy.spatial.transform import Rotation


def read_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                if parts[0] == 'v':
                    vertices.append([float(coord) for coord in parts[1:4]])
                elif parts[0] == 'f':
                    faces.append([int(idx.split('/')[0]) for idx in parts[1:]])
    return np.array(vertices), faces

def from_quat(q): #return rotate mat R
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array(
    [[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])
    return R

def write_obj(file_path, vertices, faces, sp_write=False):
    with open(file_path, 'w') as f:
        if sp_write:
            f.write("\nmtllib bunny_original.mtl \no bunny_0000.025\n")
        for vertex in vertices:
            f.write(f"v {' '.join(map(str, vertex))}\n")
        if sp_write:
            f.write("usemtl Default_OBJ.024\n")
        for face in faces:
            f.write(f"f {' '.join(str(idx) for idx in face)}\n")


def transform_obj(input_path, output_path, translation, quaternion, sp_write=True):
    vertices, faces = read_obj(input_path)

    # 计算四元数对应的旋转矩阵
    rotation_matrix = from_quat(quaternion)

    # 应用旋转和平移
    # print(rotation_matrix)
    print(translation.shape)
    transformed_vertices = np.dot(vertices, rotation_matrix.T) + translation
    # transformed_vertices = (np.dot(rotation_matrix, vertices.T)).T  + translation
    # 写入新的OBJ文件
    write_obj(output_path, transformed_vertices, faces, sp_write=sp_write)

def transform_obj_json(json_path, file_path, length, start_index=0, path_pre_fix=""):
    with open(json_path,"r") as json_file:  # load　again for better versibility, like renew the json and dynamicly adapt
        init_RT_from_json = json.load(json_file)
    for image_id in range(start_index, length + start_index):
        R, T = init_RT_from_json[str(image_id) + "_R"], init_RT_from_json[str(image_id) + "_T"]
        R, T = np.array(R), np.array(T)
        output_path = path_pre_fix + str(image_id) + ".obj"
        # print(output_path)
        transform_obj(file_path, output_path, translation=T, quaternion=R)


def batch_transform_obj(input_folder, output_folder, trans, quat, sp_write=False):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.obj'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            transform_obj(input_path, output_path, trans, quat, sp_write=sp_write)

if __name__ == "__main__":
# 示例使用
    input_path   = 'C:/Users/guanli.hou/Desktop/real_world/dynamic_short/exp/bunny_bounce/mesh_result/bunny.obj'
    output_path  = 'C:/Users/guanli.hou/Desktop/real_world/dynamic_short/exp/tree_slide_short/PGA_NeuS/tmp.obj'
    json_path    = 'C:/Users/guanli.hou/Desktop/real_world/dynamic_short/exp/bunny_bounce/mesh_result/pa.json'
    path_pre_fix = "C:/Users/guanli.hou/Desktop/real_world/dynamic_short/exp/bunny_bounce/mesh_result/pga/"
    input_path = 'C:/Users/guanli.hou/Desktop/neural_rig/real_world/xbox/front.obj'
    output_path= 'C:/Users/guanli.hou/Desktop/neural_rig/real_world/xbox/front2back.obj'
    input_path = 'C:/Users/guanli.hou/Desktop/neural_rig/real_world/dragon/dragon2.obj'
    output_path = 'C:/Users/guanli.hou/Desktop/neural_rig/real_world/dragon/dragon2_to_1_fine.obj'
    # input_path = 'C:/Users/guanli.hou/Desktop/neural_rig/synthetic_data/bunny/stand.obj'
    # output_path = 'C:/Users/guanli.hou/Desktop/neural_rig/synthetic_data/bunny/stand2lie_fine.obj'
# translation = np.array([-0.0454,  0.0062,  0.0963])  # 平移向量
#     translation = np.array([ 0.1723, -0.0881,  0.0382])  # 平移向量
#
#     quaternion = np.array([ 0.5740, -0.0091, -0.0223,  0.8176])  # 四元数
    # transform_obj_json(json_path=json_path, file_path=input_path, length=18, start_index=0, path_pre_fix=path_pre_fix)
    # translation = np.array([-0.32199525,  0.16299613,  0.3357338])  # 平移向量
    # quaternion =   [0.4736984355265714 , -0.21672384641841186 , 0.7409593713558119 , -0.4238157308482185]
    # translation = np.array([1, -1, 1])  # 平移向量
    # quaternion = 0.7745190528383291 , -0.15849364905389038 , 0.5915063509461096 , -0.15849364905389038
# refined: 0.7745185343166424 , -0.15849607473668723 , 0.591504834554036 , -0.1584994187496259
    translation = np.array([-0.012965,-0.008527, 0.028259])  # 平移向量
    quaternion =[0.5149449505519659 , -0.7461303121541245 , 0.06547512670513084 , -0.4169343629023269]
    # translation = np.array([-0.3201,  0.1704,  0.3462])  # 平移向量
    # quaternion =  [ 0.4680, -0.2135,  0.7420, -0.4336]
    # batch_transform_obj(input_folder=input_path, output_folder=output_path, trans=translation, quat=quaternion, sp_write = True)
    transform_obj(input_path, output_path, translation, quaternion)
