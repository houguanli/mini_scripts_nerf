import os

import cv2
import numpy as np

def sort_key(s):
    return int(s.split(".")[0][0:])

def rename_images(directory, new_ext=None, start_idx=1, sp_sort_flag = 1):
    # 获取文件夹中的所有文件
    files = os.listdir(directory)

    # 过滤出.png和.jpg格式的图片
    images = [f for f in files if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG')]

    # 确保图片数量不超过1000
    if len(images) > 1000:
        raise ValueError("图片数量超过1000张!")

    # 对图片进行排序，这样我们在重命名时不会遗漏任何图片
    if not sp_sort_flag:
        images.sort()
    else:
        images = sorted(images, key=sort_key)

    #
    # 开始重命名
    for idx, image in enumerate(images, start_idx):
        # 获取文件扩展名
        ext = os.path.splitext(image)[1]
        # ext = '.png'
        # 新名称格式：0001, 0002, ...
        if new_ext is not None:
            ext = new_ext

        new_name = f"{idx:03}{ext}"
        # 获取图片当前的完整路径和新的完整路径
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)
        # 重命名图片
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_name}")


def reformat_images(directory, new_type=".jpg"):
    # 获取文件夹中的所有文件
    files = os.listdir(directory)

    # 过滤出.png和.jpg格式的图片
    images = [f for f in files if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG')]

    # 确保图片数量不超过1000
    if len(images) > 1000:
        raise ValueError("图片数量超过1000张!")

    # 对图片进行排序，这样我们在重命名时不会遗漏任何图片
    images.sort()

    # 开始重命名h
    for idx, image in enumerate(images, 1):
        # 获取文件扩展名
        ext = new_type
        # ext = '.png'
        # 新名称格式：0001, 0002, ...
        new_name = f"{idx:04}{ext}"
        # 获取图片当前的完整路径和新的完整路径
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)
        image = cv2.imread(old_path)
        cv2.imwrite(new_path, image)  # overwrite the original image
        # 重命名图片
        print(f"Renamed {old_path} to {new_path}")
    return

def apply_gaussian_and_laplacian(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 高斯平滑
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # 拉普拉斯锐化
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    # 将结果转换为 8 位无符号整数
    sharpened = np.uint8(image - laplacian)
    return sharpened

def apply_gal_dir(image_dir):
    files = os.listdir(image_dir)

    # 过滤出.png和.jpg格式的图片
    images = [f for f in files if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG')]
    # 对图片进行排序，这样我们在重命名时不会遗漏任何图片
    images.sort()

    # 开始重命名h
    for idx, image in enumerate(images, 1):
        old_path = os.path.join(image_dir, image)
        gal_img = apply_gaussian_and_laplacian(old_path)
        cv2.imwrite(old_path, gal_img)  # overwrite the original image


if __name__ == "__main__":
    # output_path = "C:/Users/guanl/Desktop/GenshinNerf/t22/hex_s/mask"
    # 替换为你的文件夹路径
    directory_path = 'D:/gitwork/NeuS/public_data/soccer_gal/image'  # 替换为你的文件夹路径
    directory_path = 'C:/Users/guanl/Desktop/GenshinNerf/t22/soap/soap_clash/move2'  # 替换为你的文件夹路径
    directory_path = 'C:/Users/GUANLI.HOU/Desktop/fake_full_render/slide/pga'
    directory_path = 'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic_short/exp/dragon_slip_short/sp_calc/ga_sp'  # 替换为你的文件夹路径
    directory_path = 'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic/exp/bunny_bounce_long/IoU_calc/white_bk'  # 替换为你的文件夹路径
    # directory_path = 'C:/Users/GUANLI.HOU/Desktop/fake_full_render/bunny/ga'  # 替换为你的文件夹路径

    rename_images(directory_path, start_idx=000, new_ext=".png", sp_sort_flag=True)
    # directory_path = 'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic_short/exp/yoyo_book/sp_calc/init_sp'  # 替换为你的文件夹路径
    #
    # rename_images(directory_path, start_idx=0, new_ext=".png", sp_sort_flag=True)
    # directory_path = 'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic_short/exp/yoyo_book/sp_calc/pa_sp'  # 替换为你的文件夹路径
    #
    # rename_images(directory_path, start_idx=0, new_ext=".png", sp_sort_flag=True)
    # directory_path = 'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic_short/exp/yoyo_book/sp_calc/pga_sp'  # 替换为你的文件夹路径
    #
    # rename_images(directory_path, start_idx=0, new_ext=".png", sp_sort_flag=True)
