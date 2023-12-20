import json
import numpy as np
import cv2 as cv


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def load_npz_to_dict(file_path):
    # 加载npz文件
    data = np.load(file_path)

    # 将数据转换为字典
    data_dict = {key: data[key] for key in data.keys()}

    # 打印字典
    for key, value in data_dict.items():
        print(f"{key}: {value}")

    return data_dict


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
        if key.startswith("world_mat") and not key.startswith("world_mat_inv"):
            print(f"mat Key: {key}")
            intrinsics, pose = load_K_Rt_from_P("none", data[key][:3, :4])
            print(f"pose :\n {pose}")
            print(f"intrinsics :\n {intrinsics}")

        else:
            print(f"Key: {key}")
            # print(data[key])
        # print("\n")
    return

# intrinsics, pose = load_K_Rt_from_P(None, P)
if __name__ == '__main__':
    npz_path = "D:/gitwork/NeuS/public_data/bird/cameras_large.npz"
    npz_path = "C:/Users/guanl/Desktop/GenshinNerf/t22/soap/soap1_qr1/preprocessed/cameras_sphere.npz"
    npz_path = "C:/Users/guanl/Desktop/GenshinNerf/t23/datasets1/soap_dynamic1/cameras_sphere.npz"

    npz_debug(npz_path)
    # convert(frame_count=9, camera_count=5)
