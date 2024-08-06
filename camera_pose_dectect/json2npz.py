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

def calc_KR2P(K, R, is_c2w=True):
    #c2w need to convert
    if is_c2w:
        R = np.linalg.inv(np.array(R))
    # assume K & R is 4*4 mat
    return np.dot(K, R)
def convert(output_npz_path, json_file_path, frame_count=1, camera_count=1, start_index=0, with_camera_prefix=False):
    # 从JSON文件读取相机参数
    with open(json_file_path, 'r') as json_file:
        camera_params_list = json.load(json_file)

    # 将相机参数转化为NumPy数组并保存为NPZ二进制文件
    camera_matrices = {}
    # print(camera_params_list)
    for frame_id in range(start_index, frame_count + start_index):
        for camera_id in range(1, camera_count + 1):
            if with_camera_prefix:
                mat_name = str(frame_id) + "_" + str(camera_id)
            else:
                mat_name = str(frame_id)
            idx = (frame_id - start_index) * camera_count + camera_id - camera_count # make this mat start from 0
            # camera_matrix_map = camera_params_list[idx]
            camera_matrix = camera_params_list[mat_name + "_K"]# maybe 3 * 3
            camera_matrix_4 = np.identity(4)
            camera_matrix_4[:3, :3] = camera_matrix
            c2w_matrix = camera_params_list[mat_name + "_M"]
            world_matrix = calc_KR2P(K= camera_matrix_4, R= c2w_matrix)
            identity_matrix = np.identity(4)
            camera_idx = "camera_mat_" + str(idx)
            world_idx  = "world_mat_" + str(idx)
            scale_idx  = "scale_mat_" + str(idx)
            camera_matrices[camera_idx] = camera_matrix_4
            camera_matrices[world_idx] = world_matrix
            camera_matrices[scale_idx] = identity_matrix
            print("------------------test decompose world mat------------------------")
            K, R = load_K_Rt_from_P(None, world_matrix[:3, :4])
            print("K:\n", str(K))
            print("R:\n", str(R))
            print(idx)

    np.savez(output_npz_path, **camera_matrices)
    print('Camera parameters saved as NPZ binary file.')

def move_all_c2w(npz_path, delta_t=np.zeros(3)):
    data = np.load(npz_path)
    new_dict = {}
    for key in data.files:
        if key.startswith("world_mat") and not key.startswith("world_mat_inv"):
            # print(f"mat Key: {key}")
            # print(data[key])
            intrinsics, pose = load_K_Rt_from_P("none", data[key][:3, :4])
            pose[:3, 3] = pose[:3, 3] + delta_t

            new_P = calc_KR2P(K=intrinsics, R=pose, is_c2w=True)
            new_dict[key] = new_P
            #
            # print(f"pose :\n {pose}")
            # print(f"intrinsics :\n {intrinsics}")
        else:
            new_dict[key] = data[key]
            # print(f"Key: {key}")
    np.savez(npz_path, **new_dict)
    return
def npz_debug(npz_path):
    data = np.load(npz_path)
    for key in data.files:
        # print(f"mat Key: {key}")
        # print(data[key])

        if key.startswith("world_mat") and not key.startswith("world_mat_inv"):
            # print(f"mat Key: {key}")
            # print(data[key])
            intrinsics, pose = load_K_Rt_from_P("none", data[key][:3, :4])
            print(f"pose :\n {pose}")
            print(f"intrinsics :\n {intrinsics}")

        else:
            print(f"Key: {key}")
    return

# intrinsics, pose = load_K_Rt_from_P(None, P)
if __name__ == '__main__':
    npz_path = "D:/gitwork/NeuS/public_data/bird/cameras_large.npz"
    npz_path = "C:/Users/guanl/Desktop/GenshinNerf/t22/soap/soap1_qr1/preprocessed/cameras_sphere.npz"
    npz_path = "C:/Users/guanli.hou/Desktop/tree_circle/cameras_sphere.npz"
    npz_path = "C:/Users/guanli.hou/Desktop/real_world/static/public_data/joyo/cameras_sphere.npz"
    npz_path = "C:/Users/GUANLI.HOU/Desktop/neural_rig/synthetic_data/bunny/bunny_stand/cameras_sphere.npz"

    npz_path_test = "C:/Users/GUANLI.HOU/Desktop/real_world/static/public_data/slide1/cameras_sphere.npz"

    json_path     = "C:/Users/GUANLI.HOU/Desktop/real_world/static/public_data/slope_30/cameras_sphere.json"
    # move_all_c2w(npz_path, delta_t=np.array([0.003, -0.002, -0.031]))
    npz_debug(npz_path)
    exit()
    convert(output_npz_path=npz_path_test, json_file_path=json_path, frame_count=60, camera_count=1, start_index=1, with_camera_prefix=True)
