import camtools as ct
import open3d as o3d
import json
import numpy as np
import copy


def mat_convert(mat):
    rotation_x_90 = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    # Compute the new rotation matrix
    new_matrix = rotation_x_90 @ mat

    mirror_matrix = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    R = new_matrix[:3, :3]
    t = new_matrix[:3, 3]
    mirror_clip = mirror_matrix[:3, :3]
    # new_matrix = mirror_matrix @ new_matrix
    # Construct the new extrinsic matrix
    R = mirror_clip @ R
    new_matrix = np.zeros((4, 4))
    new_matrix[:3, :3] = R
    new_matrix[:3, 3] = t
    new_matrix[3, 3] = 1

    return new_matrix


def look_at_pos(camera_mat, target_pos=None, up_vector=None):
    if target_pos is None:
        target_pos = [0, 0.1, 0]
    if up_vector is None:
        up_vector = [0, 1, 0]
    camera_pos = camera_mat[:3, 3]
    forward = np.array(target_pos) - np.array(camera_pos)
    forward /= np.linalg.norm(forward)

    # 计算右向量（相机的右方向）
    right = np.cross(up_vector, forward)
    right /= np.linalg.norm(right)

    # 计算新的上向量（确保垂直于前向和右向）
    new_up = np.cross(forward, right)

    # 构建观察矩阵
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = np.column_stack((right, new_up, -forward))
    view_matrix[:3, 3] = -np.dot(view_matrix[:3, :3], camera_pos)

    return view_matrix


def sp_re(mat):
    # assert mt = (4, 4)
    R = mat[:3, :3]
    print("R: \n" + str(R))
    R = np.linalg.inv(R)
    mat[:3, :3] = R
    return mat


#  notice: this calculation needs w2c camera mat, not c2w
def read_cameras_from_json(cameras_path, frames=4):
    with open(cameras_path, 'r') as json_file:
        data = json.load(json_file)
    # data = np.load(cameras_path)

    # 将数据转换为字典
    data_dict = {key: data[key] for key in data.keys()}
    Ks, Ts_inv = [], []
    for idx in range(0, frames):
        template_name = str(idx + 1) + "_1"
        K_name = template_name + "_K"
        T_name = template_name + "_M"
        k = np.array(data_dict[K_name])
        t = np.array(data_dict[T_name])
        Ks.append(k)
        # t = mat_convert(t)
        # t = look_at_pos(t)
        # t = np.eye(4, 4)
        # t[2, 3] = 2
        t_inv = np.linalg.inv(t)
        Ts_inv.append(t_inv)
        # print("err at calc mats for " + str(idx))

    return Ks, Ts_inv

def read_cameras_npz(cameras_path, frames=4):
    with open(cameras_path, 'r') as f:
        camera_json = json.load(f)
    print(camera_json)
    Ks, Ts_inv = [], []
    for idx in range(0, frames):
        template_name = str(idx + 1) + "_1"
        K_name = template_name + "_K"
        T_name = template_name + "_M"
        k = np.array(camera_json[K_name])
        t = np.array(camera_json[T_name])
        Ks.append(k)
        # t = mat_convert(t)
        # t = look_at_pos(t)
        t = np.eye(4, 4)
        t[2, 3] = 2
        t_inv = np.linalg.inv(t)
        Ts_inv.append(t_inv)
        # print("err at calc mats for " + str(idx))

    return Ks, Ts_inv


def reformat_blender_mat(mat_path, frames=50):
    all_json = {}
    with open(mat_path, 'r') as f:
        camera_json = json.load(f)
    for idx in range(0, frames):
        template_name = str(idx + 1) + "_1"
        K_name = template_name + "_K"
        T_name = template_name + "_M"
        T_inv_name = template_name + "_M_inv"
        try:
            k = np.array(camera_json[K_name])
            t = np.array(camera_json[T_name])
            t[:, 1:3] *= -1
            # t_inv = np.array(camera_json[T_inv_name])
            all_json[K_name] = k.tolist()
            all_json[T_name] = t.tolist()
            all_json[T_inv_name] = (np.linalg.inv(t)).tolist()
        except:
            print("err at get " + K_name)
            continue
    print("re-f all json \n")
    print(all_json)
    with open(mat_path, 'w') as f:
        json.dump(all_json, f, indent=4)
    return

def replace_k_mat(mat_path, new_K=None, frames=20):
    if new_K is None:
        print("input new K!")
        return
    all_json = {}
    with open(mat_path, 'r') as f:
        camera_json = json.load(f)
    for idx in range(0, frames):
        template_name = str(idx + 1) + "_1"
        K_name = template_name + "_K"
        T_name = template_name + "_M"
        T_inv_name = template_name + "_M_inv"
        try:
            k = np.array(new_K)
            t = np.array(camera_json[T_name])
            # t[:, 1:3] *= -1
            # t_inv = np.array(camera_json[T_inv_name])
            all_json[K_name] = k.tolist()
            all_json[T_name] = t.tolist()
            all_json[T_inv_name] = (np.linalg.inv(t)).tolist()
        except:
            print("err at get " + K_name)
            continue
    print("re-f all json \n")
    print(all_json)
    with open(mat_path, 'w') as f:
        json.dump(all_json, f, indent=4)
    return

def load_npz_to_dict(file_path):
    # 加载npz文件
    data = np.load(file_path)

    # 将数据转换为字典
    data_dict = {key: data[key] for key in data.keys()}

    # 打印字典
    for key, value in data_dict.items():
        print(f"{key}: {value}")

    return data_dict

if __name__ == '__main__':
    cameras_path = 'C:/Users/GUANL/Desktop/GenshinNerf/t11/camera.json'
    cameras_path = 'C:/Users/GUANL/Desktop/GenshinNerf/t16/frames/airphone/t16/camera.json'
    # cameras_path = 'C:/Users/GUANL/Desktop/GenshinNerf/dp_simulation/duck/duck_3d/cameras.json'
    cameras_path = 'D:/gitwork/neus_original/public_data/rws_obj5/cameras_sphere.json'

    cameras_path = 'C:/Users/GUANL/Desktop/GenshinNerf/t21/compress/cameras_sphere.json'
    npz_path = "D:/gitwork/NeuS/public_data/bird/cameras_large.npz"
    # cameras_path = 'C:/Users/guanl/Desktop/face_video/front/sparse/1/cameras_sphere.json'
    # reformat_blender_mat(cameras_path, frames=20)
    # new_K = [[393.1742062283737, 0, 246.57381480968857], [0, 392.47815705069127, 185.1793146543779], [0, 0, 1]]
    # replace_k_mat(cameras_path, new_K=new_K)
    # exit()

    # npz_dict = load_npz_to_dict(npz_path)

    Ks, Ts_inv = read_cameras_from_json(cameras_path, frames=20)
    # import pdbDDD
    # pdb.set_trace()
    cameras = ct.camera.create_camera_ray_frames(Ks, Ts_inv)
    # model1 = o3d.io.read_triangle_mesh("C:/Users/GUANL/Desktop/GenshinNerf/t12/models/t_0_000001.obj")
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[1, 0, 1])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])

    o3d.visualization.draw_geometries([cameras, axis])

    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # # 往x方向平移1.3米
    # mesh_tx = copy.deepcopy(mesh).translate((1.3, 0, 0))
    # # 往y方向平移1.3米
    # mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
    # o3d.visualization.draw_geometries([mesh, mesh_tx, mesh_ty])
