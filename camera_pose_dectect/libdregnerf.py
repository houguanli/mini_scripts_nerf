import json
import numpy as np
import math
import os
import shutil
from PIL import Image
import cv2 as cv
from scipy.spatial.transform import Rotation as R
# from libnpz import *
# from libcamera import *

def convert_json_to_npz(json_path, npz_path, IMAGE_WIDTH=800, IMAGE_HEIGHT=800):
    # 加载 JSON 文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 提取 camera_angle_x 并计算 focal length
    camera_angle_x = data["camera_angle_x"]
    focal = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)

    # 构造内参矩阵（camera_mat），所有帧共享
    K = np.array([
        [focal, 0, IMAGE_WIDTH / 2.0, 0],
        [0, focal, IMAGE_HEIGHT / 2.0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    K_temp = np.array([
        [focal, 0, IMAGE_WIDTH / 2.0],
        [0, focal, IMAGE_HEIGHT / 2.0],
        [0, 0, 1]
    ])  # 3x3 内参矩阵

    # 准备存储数据
    mats = {}

    # 遍历 frames，提取 transform_matrix 作为 world_mat
    for i, frame in enumerate(data["frames"]):
        # 从 transform_matrix 提取 c2w（camera-to-world）
        transform_matrix = np.array(frame["transform_matrix"])
        transform_matrix[:, 1:3] = -transform_matrix[:, 1:3]  # 翻转 Y 和 Z 轴

        # 从 c2w 转换为 w2c（world-to-camera）
        world_to_camera = np.linalg.inv(transform_matrix)
        if i == 0:
            print(f"Original world_to_camera: \n{world_to_camera}")

        # 从 w2c 提取 R 和 t
        R = world_to_camera[:3, :3]
        t = world_to_camera[:3, 3]
        R[:, 0] /= np.linalg.norm(R[:, 0])  # 归一化第一列
        R[:, 1] /= np.linalg.norm(R[:, 1])  # 归一化第二列
        R[:, 2] /= np.linalg.norm(R[:, 2])  # 归一化第三列
        R[:, 1] = -R[:, 1]
        R[:, 2] = -R[:, 2]
        # 构造外参矩阵 [R | t]
        Rt = np.hstack((R, t.reshape(3, 1)))

        # 计算投影矩阵 P
        P = np.dot(K_temp, Rt)
        P = np.vstack((P, np.array([0, 0, 0, 1])))

        if i == 0:
            # print(f"Original frame {i} K: \n{K_temp}")
            # print(f"Original frame {i} P: \n{P}")
            print(f"Original frame {i} Rt: \n{Rt}")

        # 保存矩阵
        mats[f"camera_mat_{i}"] = K
        mats[f"world_mat_{i}"] = P
        mats[f"scale_mat_{i}"] = np.eye(4)  # 假设 scale_mat 为单位矩阵

    # 合并所有矩阵数据并保存为 npz
    np.savez(npz_path, **mats)
    print(f"Conversion completed. File saved to {npz_path}")


def convert_dataset_to_npz(old_dataset_path, new_dataset_path, generate_simple_mask=True):
    """
    Convert a dataset from the old JSON and image directory structure to the new NPZ and standardized image directory.

    Args:
        old_dataset_path (str): Path to the old dataset directory.
        new_dataset_path (str): Path to the new dataset directory.
    """
    # 确保新目录存在
    os.makedirs(os.path.join(new_dataset_path, "image"), exist_ok=True)
    if generate_simple_mask:
        os.makedirs(os.path.join(new_dataset_path, "mask"), exist_ok=True)

    # 转换相机参数
    convert_json_to_npz(os.path.join(old_dataset_path, "transforms.json"),
                        os.path.join(new_dataset_path, "cameras_sphere.npz"))

    json_path = os.path.join(old_dataset_path, "transforms.json")
    with open(json_path, "r") as f:
        transforms = json.load(f)

    # 遍历 frames，复制图像文件并保存相机位姿
    for i, frame in enumerate(transforms["frames"]):
        # 处理图像文件
        old_image_path = os.path.join(old_dataset_path, frame["file_path"].lstrip("./") + ".png")
        new_image_name = f"{i:03d}.png"
        new_image_path = os.path.join(new_dataset_path, "image", new_image_name)
        shutil.copy(old_image_path, new_image_path)  # 复制图像到新目录

        # 生成简单 mask 文件
        if generate_simple_mask:
            with Image.open(old_image_path) as img:
                if img.mode == "RGBA":
                    # 使用 Alpha 通道生成 mask
                    alpha = img.getchannel("A")
                    mask = Image.new("L", img.size, 255)  # 默认白色
                    mask.paste(0, mask=alpha.point(lambda p: p < 255 and 255))
                else:
                    # 如果没有 Alpha 通道，默认全白
                    mask = Image.new("L", img.size, 255)

                # 保存 mask 文件
                new_mask_path = os.path.join(new_dataset_path, "mask", new_image_name)
                mask.save(new_mask_path)

    print(f"Conversion completed. Dataset saved to {new_dataset_path}")


def get_image_dimensions(image_folder):
    """
    获取图片文件夹中第一张图片的宽度和高度。

    Args:
        image_folder (str): 图片文件夹路径。

    Returns:
        tuple: (width, height) 图片的宽度和高度。
    """
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    return img.width, img.height
    raise ValueError("No valid image found in the specified folder.")

def load_K_Rt_from_P(filename, P=None):
    print(f"Loading K and Rt from {filename}...=====================")
    print(f"P: \n{P}")
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
    intrinsics[1, 1] = intrinsics[0, 0]  # Assume square pixels
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def npz_to_json(npz_path, json_path, image_width=800, image_height=800):
    """
    Convert an NPZ camera parameter file back to a JSON format.

    Args:
        npz_path (str): Path to the NPZ file.
        json_path (str): Path to save the JSON file.
        image_width (int): Width of the images (default: 800).
        image_height (int): Height of the images (default: 800).
    """
    # 加载 NPZ 文件
    data = np.load(npz_path)

    # 提取 camera_angle_x
    camera_mat_key = "camera_mat_0"  # 假设所有帧共享相同内参
    world_mat_key = "world_mat_0"  # 假设所有帧共享相同内参
    if camera_mat_key not in data:
        print(f"Warning: Key '{camera_mat_key}' for intrinsics not found. Will attempt to decompose from {world_mat_key}.")
        K, _ = load_K_Rt_from_P(None, data[world_mat_key][:3, :4])
        K = K[:3, :3]  # 提取 3x3 内参矩阵
        print(f"K_Real: \n{K}")

    else:
        K = data[camera_mat_key][:3, :3]  # 提取 3x3 内参矩阵
        # K_Decomp, _ = load_K_Rt_from_P(None, data[world_mat_key][:3, :4])
        # K_Decomp = K_Decomp[:3, :3]  # 提取 3x3 内参矩阵
        # print(f"K_Real: \n{K}")
        # print(f"K_Decomp: \n{K_Decomp}")
    focal_length = K[0, 0]  # 焦距 (f_x)
    camera_angle_x = 2 * np.arctan(image_width / (2 * focal_length))  # 反推

    # 准备 JSON 数据结构
    json_data = {"camera_angle_x": float(camera_angle_x), "frames": []}

    # 获取视点数量
    num_views = sum(1 for key in data.files if key.startswith("world_mat"))

    # 计算每个视点的旋转步长（弧度值）
    stepsize = 360.0 / num_views  # 每个视点的步长（度数）
    rotation_radians = math.radians(stepsize)  # 转为弧度

    # 遍历 world_mat
    for key in data.files:
        if key.startswith("world_mat"):
            frame_idx = int(key.split("_")[-1])  # 提取帧索引
            P = data[key]  # 投影矩阵
            P = P[:3, :4]
            # P 是通过 K 和 [R | t] 点乘得到的，而我们已经有了 K，所以可以通过 P = K * [R | t] 得到 [R | t]
            Rt = np.dot(np.linalg.inv(K), P)

            # if frame_idx == 0:
            #     print(f"Transformed frame {frame_idx} Rt: \n{Rt}")
            c2w = np.linalg.inv(np.vstack((Rt, np.array([0, 0, 0, 1]))))
            c2w[:3, 2] *= -1
            c2w[:3, 1] *= -1
            # 添加到 JSON 数据
            json_data["frames"].append({
                "file_path": f"./train/r_{frame_idx}",
                "rotation": rotation_radians,  # 默认值，可调整
                "transform_matrix": c2w.tolist()  # 转换为列表格式
            })

    # 保存为 JSON 文件
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Conversion completed. JSON file saved to {json_path}")


def analyze_file_name(file_list):
    # 先看看有没有任何文件名是以先导 0 开头的
    has_preceding_zero = False
    pad_length = 0
    for file_name in file_list:
        if file_name.startswith("0"):
            has_preceding_zero = True
            pad_length = len(file_name)
            break

    # 确认一下扩展名
    extension = None
    for file_name in file_list:
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            extension = file_name.split(".")[-1]
            break

    if has_preceding_zero:
        return "{frame_idx:0" + str(pad_length-len(extension)-1) + "d}" + "." + extension
    else:
        return "{frame_idx}" + "." + extension


def convert_npz_to_json_structure(input_dir, output_dir):
    """
    Convert a fixed NPZ directory structure to a JSON-based directory structure.

    Args:
        input_dir (str): Path to the input directory (NPZ structure).
        output_dir (str): Path to the output directory (JSON structure).
    """
    # 路径定义
    npz_path = os.path.join(input_dir, "cameras_sphere.npz")
    input_image_dir = os.path.join(input_dir, "image")
    train_dir = os.path.join(output_dir, "train")
    json_path = os.path.join(output_dir, "transforms.json")
    bbox_path = os.path.join(output_dir, "bbox.txt")

    # 创建输出目录
    os.makedirs(train_dir, exist_ok=True)

    # 加载 NPZ 文件
    data = np.load(npz_path)

    # 动态读取图片文件列表并排序
    image_files = sorted(
        [f for f in os.listdir(input_image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    )
    if not image_files:
        raise ValueError("No image files found in the input image directory.")

    # 动态获取第一张图片的分辨率
    first_image_path = os.path.join(input_image_dir, image_files[0])
    with Image.open(first_image_path) as img:
        image_width, image_height = img.size
        print(f"Image dimensions: {image_width} x {image_height}")

    input_file_format = analyze_file_name(image_files)

    # 遍历 world_mat 和图片文件
    for key in data.files:
        if not key.startswith("world_mat"):
            continue
        frame_idx = int(key.split("_")[-1])  # 提取帧索引
        print(f"Processing key {frame_idx}...")
        # 将图片从 image 文件夹复制到 train 文件夹
        input_image_path = os.path.join(input_image_dir, input_file_format.format(frame_idx=frame_idx))
        output_image_path = os.path.join(train_dir, f"r_{frame_idx}.png")
        shutil.copy(input_image_path, output_image_path)

    # 保存 JSON 文件
    npz_to_json(npz_path, json_path, image_width, image_height)

    print(f"Conversion completed. JSON file saved to {json_path}")
    print(f"Images moved to {train_dir}")


def load_K_from_json_dataset(dir):
    json_path = os.path.join(dir, "transforms.json")

    with open(json_path, "r") as f:
        transforms = json.load(f)
        # 提取 camera_angle_x 并计算 focal length
        camera_angle_x = transforms["camera_angle_x"]

        IMAGE_WIDTH, IMAGE_HEIGHT = get_image_dimensions(os.path.join(dir, "train"))

        focal = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)

        # 构造内参矩阵（camera_mat），所有帧共享
        K = np.array([
            [focal, 0, IMAGE_WIDTH / 2.0],
            [0, focal, IMAGE_HEIGHT / 2.0],
            [0, 0, 1]
        ])
        return K


def create_se3_from_quad_trans(quad, trans):
    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(quad).as_matrix()

    # Construct the SE3 transformation matrix
    se3_matrix = np.eye(4)
    se3_matrix[:3, :3] = rotation_matrix
    se3_matrix[:3, 3] = trans

    return se3_matrix

def forge_world_frame_transforms(mat1, mat2, data_dir):
    data = {
        "0": mat1.tolist(),
        "1": mat2.tolist(),
    }

    json_file = os.path.join(data_dir, 'world_frame_transforms.json')
    json_obj = json.dumps(data, indent=4)
    print(f'Saving world frame transformations to {json_file}')
    with open(json_file, 'w') as f:
        f.write(json_obj)

if __name__ == '__main__':
    # Single Json / NPZ Conversion

    convert_json_to_npz("C:/Users/guanl/Desktop/reg/dreg_nerf_data/reameked_data/fire/transforms.json",
                        "C:/Users/guanl/Desktop/reg/dreg_nerf_data/reameked_data/fire/cameras_sphere.npz")
    # npz_debug("3D_Banana_-_01fb90_npz/cameras_sphere.npz")
    # npz_to_json("3D_Banana_-_01fb90_npz/cameras_sphere.npz", "3D_Banana_-_01fb90/transforms_back.json")

    # Dataset Conversion
    # convert_dataset_to_npz("3D_Banana_-_01fb90", "3D_Banana_-_01fb90_npz")
    # visualize_camera("3D_Banana_-_01fb90/transforms.npz", frames=120)
    # visualize_camera("3D_Banana_-_01fb90_npz/cameras_sphere.npz", frames=120)
    # convert_npz_to_json_structure("3D_Banana_-_01fb90_npz", "3D_Banana_-_01fb90_back")

    # Real Dataset Conversion
    # npz_debug("\\\\wsl.localhost\\Ubuntu-24.04\\home\\ubuntu\\DReg-NeRF\\externals\\formal_dataset\\bunny_pose1\\cameras_sphere.npz")
    # convert_npz_to_json_structure("\\\\wsl.localhost\\Ubuntu-24.04\\home\\ubuntu\\DReg-NeRF\\externals\\formal_dataset\\bunny_pose1", "\\\\wsl.localhost\\Ubuntu-24.04\\home\\ubuntu\\DReg-NeRF\\externals\\formal_dataset_converted\\bunny_pose1")
    # npz_debug("nikon_pose2_npz/cameras_sphere.npz")
    # convert_npz_to_json_structure("bunny_full", "bunny_full_json")