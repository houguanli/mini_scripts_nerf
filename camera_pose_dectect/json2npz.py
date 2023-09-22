import json
import numpy as np


def convert(frame_count=1, camera_count=1):
    # 从JSON文件读取相机参数
    json_file_path = 'D:/gitwork/NeuS/public_data/sim_ball/camera.json'  # 您的JSON文件路径
    with open(json_file_path, 'r') as json_file:
        camera_params_list = json.load(json_file)

    # 将相机参数转化为NumPy数组并保存为NPZ二进制文件
    output_npz_path = 'D:/gitwork/NeuS/public_data/sim_ball/cameras_sphere.npz'  # 保存的NPZ文件路径
    output_npz_path2 = 'D:/gitwork/NeuS/public_data/sim_ball/cameras_large.npz'  # 保存的NPZ文件路径

    camera_matrices = {}
    # print(camera_params_list)
    for frame_id in range(1, frame_count + 1):
        for camera_id in range(1, camera_count + 1):
            mat_name = str(frame_id) + "_" + str(camera_id)
            idx = frame_id * camera_count + camera_id - camera_count - 1
            camera_matrix_map = camera_params_list[idx]
            camera_matrix = camera_matrix_map[mat_name]
            identity_matrix = np.identity(4)
            camera_idx = "world_mat_" + str(idx)
            scale_idx = "scale_mat_" + str(idx)
            camera_matrices[camera_idx] = camera_matrix
            camera_matrices[scale_idx] = identity_matrix

            print(camera_matrix_map)
            print(camera_matrix)
            print(idx)

    np.savez(output_npz_path, **camera_matrices)
    np.savez(output_npz_path2, **camera_matrices)
    print('Camera parameters saved as NPZ binary file.')


def npz_debug(npz_path):
    data = np.load(npz_path)
    print(data)
    for key in data.files:
        print(f"Key: {key}")
        print(data[key])
        print("\n")
    return


if __name__ == '__main__':
    npz_path = "D:/gitwork/NeuS/public_data/bird/cameras_large.npz"
    npz_debug(npz_path)
    # convert(frame_count=9, camera_count=5)
