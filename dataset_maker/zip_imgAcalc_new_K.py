import os
import cv2 as cv
import numpy as np
import json
from PIL import Image
from PIL.Image import Resampling
from scipy import ndimage

default_original_K = [[3.55085455e+03, 0.00000000e+00, 2.23088539e+03],
                      [0.00000000e+00, 3.54865667e+03, 1.67835047e+03],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
dynamic_original_K = [[735.15009953, 0., 961.93174061],
                      [0., 733.3960477, 553.13510509],
                      [0., 0., 1.]]  # this camera K is from a pan-tilt-zoom camera
dynamic_K_1280 = [[490.1000663533333, 0, 641.1211604066666], [0, 488.9306984666667, 368.59007006], [0, 0, 1]]
dynamic_K_512 = []

def resize_images(directory, original_K=None):
    # 获取文件夹中的所有文件
    if original_K is None:
        original_K = dynamic_original_K
    files = os.listdir(directory)

    # 过滤出.png和.jpg格式的图片
    images = [f for f in files if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG')]
    if len(images) > 1000:
        raise ValueError("图片数量超过1000张!")
    # 对图片进行排序，这样我们在重命名时不会遗漏任何图片
    images.sort()
    # assume all imgs has same W & H
    img0_path = os.path.join(directory, images[0])
    img0 = Image.open(img0_path)
    W0, H0 = img0.width, img0.height
    target_W = 1280
    scale_w = target_W / W0
    target_H = int(H0 * target_W / W0)
    scale_h = target_H / H0

    f_new_x = original_K[0][0] * scale_w
    f_new_y = original_K[1][1] * scale_h
    c_new_x, c_new_y = (original_K[0][2] + 0.5) * scale_w - 0.5, (original_K[1][2] + 0.5) * scale_h - 0.5

    new_calc_k = [[f_new_x, 0, c_new_x],
                  [0, f_new_y, c_new_y],
                  [0, 0, 1]]
    print("cacl new K: " + str(new_calc_k))

    # begin compress
    for idx, image in enumerate(images, 1):
        # 获取文件扩展名
        ext = os.path.splitext(image)[1]
        # ext = ".png"
        # 新名称格式：0001, 0002, ...
        new_name = f"{idx:04}{ext}"
        # 获取图片当前的完整路径和新的完整路径
        old_path = os.path.join(directory, image)
        img_i = Image.open(old_path)
        resized_img = img_i.resize((target_W, target_H), Resampling.LANCZOS)

        new_folder = directory + "/compress/"
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        new_path = os.path.join(new_folder, new_name)
        # 重命名图片
        # os.rename(old_path, new_path)
        resized_img.save(new_path)
        print(f"Resized {image} to {new_path}")


if __name__ == "__main__":
    directory_path = 'C:/Users/guanl/Desktop/GenshinNerf/t21/compress'  # 替换为你的文件夹路径
    # directory_path = 'D:/gitwork/neus_original/public_data/rws_obj5/image'  # 替换为你的文件夹路径

    resize_images(directory_path)
