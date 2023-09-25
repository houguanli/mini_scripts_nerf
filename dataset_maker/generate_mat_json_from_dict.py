import os
import detect_k_from_EXIF
import detect_single_w2c_from_qr
import detect_c2w_from_3x3qrs
import numpy as np
import json

detect_from_multi_arcuo = True
debug_mode = True


def rename_images_in_folder(folder_path):
    # 获取文件夹内的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # 按照字典序排序图片文件
    image_files.sort()

    # 生成计数器，从0000开始命名
    counter = 0

    # 重命名图片文件
    for image_file in image_files:
        if counter >= 1000:
            break

        # 构建新的文件名，保持四位数格式，例如 '0000.jpg'
        new_filename = f'{counter + 1:04d}' + os.path.splitext(image_file)[-1]

        # 构建完整的旧文件路径和新文件路径
        old_path = os.path.join(folder_path, image_file)
        new_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)

        counter += 1

    print(f'Renamed {counter} images in {folder_path}.')
    return counter


def calc_json(file_dict, count_imgs, img_type=".jpg"):  # IMPORTANT:: c2w == M for Neus calculation
    dict = {}
    success_count = 0
    for idx in range(0, count_imgs):

        filename = file_dict + "/" + f'{idx + 1:04d}' + img_type
        if debug_mode:
            print("run at " + filename)
        K = detect_k_from_EXIF.get_k_from_exif(filename)
        if K is None:
            print("ERR AT detect K for " + filename)
            continue
        if detect_from_multi_arcuo:
            c2w = detect_c2w_from_3x3qrs.detect_aruco_and_estimate_pose(filename, 0.031, K)
        else:
            c2w = detect_single_w2c_from_qr.detect_aruco_and_estimate_pose(filename, 0.036, None)
        if c2w is None:
            print("ERR AT detect M for " + filename)
            continue
        if debug_mode:
            print("K & M calc success! /n-----------------")
            print(K)
            print(c2w)
            print("-----------------------------------------")

        success_count= success_count + 1
        w2c = np.linalg.inv(c2w)
        K_n = str(idx + 1) + "_1_K"
        M_n = str(idx + 1) + "_1_M"
        M_inv_n = str(idx + 1) + "_1_M_inv"
        dict[K_n] = K.tolist()
        dict[M_n] = c2w.tolist()
        dict[M_inv_n] = w2c.tolist()
    if debug_mode:
        print("-----Calc success with " + str(success_count) + " imgs with all count=" + str(count_imgs) + "------")
    cameras_path = file_dict + "/camera.json"
    with open(cameras_path, 'w') as json_file:
        json.dump(dict, json_file, indent=4)


if __name__ == "__main__":
    folder_path = 'C:/Users/GUANL/Desktop/GenshinNerf/t13/static/test'
    folder_path = 'C:/Users/GUANL/Desktop/GenshinNerf/__tmp'

    # folder_path = 'D:/gitwork/NeuS/public_data/real_world_multi_qrs/mask'

    # count = rename_images_in_folder(folder_path)
    count = 32
    calc_json(folder_path, count)



