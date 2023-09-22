import os
import numpy as np
import json


# 定义一个函数，接受文件路径作为参数
def condense_camera(file_path):
    total_add = ""
    with open(file_path, 'r') as file:
        for line in file:
            str_len = len(line)
            if (str_len < 200):
                print(line, end='')
                total_add += line
    base, extension = os.path.splitext(file_path)
    insert_string = "_sphere"
    new_file_path = base + insert_string + extension
    with open(new_file_path, 'w') as file:
        file.write(total_add)


def read_K(cameras_pth):  # this read camera mats in colmap file and return a 3*3 mat
    all_K = {}
    with open(cameras_pth, 'r') as file:
        for line in file:
            if line.startswith("# "):
                continue
            else:
                parts = line.strip().split(' ')
                camera_id = int(parts[0])
                # import pdb; pdb.set_trace()
                f_x, c_x, c_y = map(float, parts[4:7])
                intrinsic_matrix = [
                    [f_x, 0, c_x],
                    [0, f_x, c_y],
                    [0, 0, 1]
                ]
                all_K[camera_id] = intrinsic_matrix
    print(all_K)
    return all_K


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """将四元数转换为旋转矩阵"""
    R = np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
    ])
    return R



def quaternion_to_rotation_matrix__(qw, qx, qy, qz):
    q = np.array([qw, qx,  qy, qz]).reshape(1,4)
    q = q / np.linalg.norm(q, axis=1)
    R = np.ones((3, 3))
    qr = q[0, 0]
    qi = q[0, 1]
    qj = q[0, 2]
    qk = q[0, 3]
    R[0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[0, 1] = 2 * (qj * qi - qk * qr)
    R[0, 2] = 2 * (qi * qk + qr * qj)
    R[1, 0] = 2 * (qj * qi + qk * qr)
    R[1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[1, 2] = 2 * (qj * qk - qi * qr)
    R[2, 0] = 2 * (qk * qi - qj * qr)
    R[2, 1] = 2 * (qj * qk + qi * qr)
    R[2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return R


def read_M(images_pth):  # this read image mats in colmap file and return a 4*4 mat
    all_M = {}
    with open(images_pth, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue

            data = line.split()
            image_id = int(data[0])
            qw, qx, qy, qz = map(float, data[1:5])
            tx, ty, tz = map(float, data[5:8])

            R = quaternion_to_rotation_matrix__(qw, qx, qy, qz)
            T = np.array([[tx], [ty], [tz]])
            # 构造外参矩阵，其实是[R | T]形式
            extrinsic_matrix = np.hstack((R, T))
            matrix_4x4 = np.zeros((4, 4))
            # 将3x4矩阵的值复制到4x4矩阵
            matrix_4x4[:3, :] = extrinsic_matrix  # copy from 3*4
            # 设置第四行为[0, 0, 0, 1]
            matrix_4x4[3, :] = [0, 0, 0, 1]
            extrinsic_matrix = matrix_4x4
            all_M[image_id] = extrinsic_matrix

    # print(all_M)
    return all_M


def to_json(tar_path, all_M, all_K):  # colmap raw
    total_dict = {}

    for key, value in all_M.items():  # M needs to store a inv
        real_f_id = key - 1
        new_index_name = str(real_f_id) + "_" + "M_inv"
        total_dict[new_index_name] = value.tolist()
        # also store an inv here
        try:
            inverse_matrix = np.linalg.inv(value)
            new_index_name = str(real_f_id) + "_" + "M"
            total_dict[new_index_name] = inverse_matrix.tolist()
        except np.linalg.LinAlgError:
            print("error at get inverse C2W mat\n\n see original mat \n\n")
            print(value)
            exit(-1)

    for key, value in all_K.items():
        real_f_id = key - 1
        new_index_name = str(real_f_id) + "_" + "K"
        total_dict[new_index_name] = value

    # print(total_dict)

    with open(tar_path, 'w') as file:
        json.dump(total_dict, file, indent=4)
    return


def to_json_adpt_sp(tar_path, all_M, all_K):  # colmap raw data to adjust to special neus dataset loader as set
    # frame_cnt = 40, camera_count = 1
    total_dict = {}
    # frame_count = len(all_M)
    for key, value in all_M.items():  # M needs to store a inv
        real_f_id = key
        new_index_name = str(real_f_id) + "_1_" + "M_inv"  # as only one camera
        total_dict[new_index_name] = value.tolist()
        # also store an inv here
        try:
            inverse_matrix = np.linalg.inv(value)
            new_index_name = str(real_f_id) + "_1_" + "M"
            total_dict[new_index_name] = inverse_matrix.tolist()
        except np.linalg.LinAlgError:
            print("error at get inverse C2W mat\n\n see original mat \n\n")
            print(value)
            exit(-1)

    for key, value in all_K.items():
        real_f_id = key
        new_index_name = str(real_f_id) + "_1_" + "K"
        total_dict[new_index_name] = value

    # print(total_dict)

    with open(tar_path, 'w') as file:
        json.dump(total_dict, file, indent=4)
    return


def np_test():
    data = {
        "matrix1": np.array([[1, 2], [3, 4]]),
        "matrix2": np.array([[5, 6], [7, 8]])
    }
    all_dict = {}
    for key, value in data.items():
        data[key] = value
        all_dict[key] = value

    for key, value in data.items():
        # also store an inv here
        data[key] = value
        all_dict[key] = value  # add again
        try:
            inverse_matrix = np.linalg.inv(value)
            new_k = str(key) + "_inv"
            all_dict[new_k] = inverse_matrix  # add inv
        except np.linalg.LinAlgError:
            exit(-1)
    with open("a_json.txt", 'w') as file:
        json.dump(all_dict, file, indent=4)
    return


# 调用函数并传入要读取的文件路径

if __name__ == '__main__':
    file_path = "C:/Users/guanl/Desktop/face_video/front/sparse/1/images.txt"
    images_pth = "C:/Users/guanl/Desktop/face_video/front/sparse/1/images_sphere.txt"
    cameras_pth = "C:/Users/guanl/Desktop/face_video/front/sparse/1/cameras.txt"
    condense_camera(file_path)
    all_M = read_M(images_pth)
    all_K = read_K(cameras_pth)
    tar_path = "C:/Users/guanl/Desktop/face_video/front/sparse/1/cameras_sphere.json"
    to_json_adpt_sp(tar_path, all_M, all_K)
    # condense_camera(file_path)
