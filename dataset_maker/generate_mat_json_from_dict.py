import os

import torch.cuda

import detect_k_from_EXIF
import detect_single_w2c_from_qr
import detect_c2w_from_3x3qrs
import numpy as np
import json
arcuo_detect_type_dict = {'single_arcuo' : "single_arcuo", "multi_3x3": "multi_3x3", "multi_5x7":"multi_5x7"}
detect_from_multi_arcuo = arcuo_detect_type_dict['multi_5x7']
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


def calc_json(file_dict, count_imgs, img_type=".jpg", with_default_mat='ptz_1920', arcuo_type=None):  # IMPORTANT:: c2w == M for Neus calculation
    dict = {}
    success_count = 0
    if arcuo_type is None:
        arcuo_type = detect_from_multi_arcuo
    for idx in range(0, count_imgs):
        filename = file_dict + "/" + f'{idx + 1:04d}' + img_type
        if debug_mode:
            print("run at " + filename)
        K = detect_k_from_EXIF.get_k_from_exif(filename, with_default_mat=with_default_mat)
        if K is None:
            print("ERR AT detect K for " + filename)
            continue
        try:
            if arcuo_type == 'multi_3x3':
                c2w = detect_c2w_from_3x3qrs.detect_aruco_and_estimate_pose(filename, 0.031, K)
            elif arcuo_type == 'single_arcuo':
                c2w = detect_single_w2c_from_qr.detect_aruco_and_estimate_pose(filename, 0.028, K)
            elif arcuo_type == 'multi_5x7':
                c2w = detect_c2w_from_3x3qrs.detect_aruco_and_estimate_pose(filename, 0.0185, K)
            else:
                print("mode set failed! with mode: " + arcuo_type)
                exit(-1)

            if debug_mode:
                print("K & M calc success!-----------------")
                print(K)
                print(c2w)
                print("-----------------------------------------")
        except:
            print("ERR AT detect M for " + filename)
            continue
        success_count = success_count + 1
        w2c = np.linalg.inv(c2w)
        K_n = str(idx + 1) + "_1_K"
        M_n = str(idx + 1) + "_1_M"
        M_inv_n = str(idx + 1) + "_1_M_inv"
        dict[K_n] = K.tolist()
        dict[M_n] = c2w.tolist()
        dict[M_inv_n] = w2c.tolist()
    if debug_mode:
        print("-----Calc success with " + str(success_count) + " imgs with all count=" + str(count_imgs) + "------")
    cameras_path = file_dict + "/cameras_sphere.json"
    with open(cameras_path, 'w') as json_file:
        json.dump(dict, json_file, indent=4)


if __name__ == "__main__":
    folder_path = 'C:/Users/guanl/Desktop/GenshinNerf/tmp'

    # folder_path = 'D:/gitwork/NeuS/public_data/real_world_multi_qrs/mask'

    # count = rename_images_in_folder(folder_path)
    calc_json(folder_path, count_imgs=3, img_type=".jpg", with_default_mat='phone', arcuo_type="multi_5x7")
