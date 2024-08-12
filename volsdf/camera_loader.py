import os
import torch
import numpy as np
def calc_KR2P(K, R, is_c2w=True):
    #c2w need to convert
    if is_c2w:
        R = np.linalg.inv(np.array(R))
    # assume K & R is 4*4 mat
    return np.dot(K, R)
def read_camera_txt_new(camera_path, img_res=[1024, 1024]):
    with open(camera_path, 'r') as f:
        content = f.read()
    import re
    pattern = r'-?\d+\.\d+'
    matches = re.findall(pattern, content)
    matches = matches[18:]
    matrix_values = [float(value) for value in matches]
    pose_all = np.array(matrix_values).reshape(-1, 4, 4)
    pose_all_list = []
    for pose in pose_all:
        pose[:, 1:3] = pose[:, 1:3] * -1 # term reshape
        pose_all_list.append(pose[:, :])
    # lens_value = 43.456 * (1024 / 36)
    lens_value = 0.5 * 1024 / np.tan(.5 * np.pi /4)
    print(f"Focal Lens: {lens_value} in pixel")
    k = np.eye(4)
    k[0, 0] = k[1, 1] = lens_value
    k[0, 2] = k[1, 2] = img_res[0] / 2
    intrinsic_all = [k[:, :]] * len(pose_all)
    print(pose_all_list)
    print(intrinsic_all)
    return pose_all_list, intrinsic_all

def poses_to_npz(pose_all_list, intrinsic_all, output_npz_path):
    # assume same num
    cnt = len(pose_all_list)
    camera_matrices = {}
    for idx in range(0, cnt):
        mat_name = str(idx)
        camera_matrix_4 = intrinsic_all[idx]
        # camera_matrix_4 =  np.identity(4)
        # camera_matrix_4[:3, :3] = camera_matrix
        c2w_matrix = pose_all_list[idx]
        world_matrix = calc_KR2P(K=camera_matrix_4, R=c2w_matrix)
        identity_matrix = np.identity(4)
        camera_idx = "camera_mat_" + str(idx)
        world_idx = "world_mat_" + str(idx)
        scale_idx = "scale_mat_" + str(idx)
        camera_matrices[camera_idx] = camera_matrix_4
        camera_matrices[world_idx] = world_matrix
        camera_matrices[scale_idx] = identity_matrix
    np.savez(output_npz_path, **camera_matrices)
    print('Camera parameters saved as NPZ binary file.')

if __name__ == '__main__':
    camera_path = "C:/Users/GUANLI.HOU/Desktop/neural_rig/bunny/1024_liedown/camera.txt"
    pose_all_list, intrinsic_all = read_camera_txt_new(camera_path)
    out_npz_path = "C:/Users/GUANLI.HOU/Desktop/neural_rig/bunny/1024_liedown/cameras_sphere.npz"
    poses_to_npz(pose_all_list, intrinsic_all, out_npz_path)