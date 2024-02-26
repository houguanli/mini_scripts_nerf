import os

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from scipy import ndimage

def make_init_mask(img_path, out_path):
    image = cv2.imread(img_path)
    threshold_value = 2
    height, width, _ = image.shape
    highlighted_image = image.copy()
    # 找到大于阈值的像素位置
    for y in range(height):
        for x in range(width):
            if all(pixel > threshold_value for pixel in image[y, x]):
                highlighted_image[y, x] = [255, 255, 255]  # 设置为白色
            else:
                highlighted_image[y, x] = [0, 0, 0]
    cv2.imshow('Original Image', image)
    cv2.imshow('Highlighted Image', highlighted_image)
    cv2.imwrite(out_path, highlighted_image)


def refine_mask(img_path, convert_option=True):
    image = cv2.imread(img_path)
    threshold_value = 66
    height, width, _ = image.shape
    refined_mask = image.copy()
    for y in range(height):
        for x in range(width):
            pixel_sum = sum(image[y, x] ) + 0.
            # if all(pixel > threshold_value for pixel in image[y, x]):
            if pixel_sum > threshold_value:

                if convert_option:
                    refined_mask[y, x] = [0, 0, 0]  # 设置为黑色
                else:
                    refined_mask[y, x] = [255, 255, 255]   # 设置为白色
            else:
                if convert_option:
                    refined_mask[y, x] = [255, 255, 255]
                else:
                    refined_mask[y, x] = [0, 0, 0]  # 设置为黑色
    print("saving refined mask at " + img_path)
    cv2.imwrite(img_path, refined_mask)  # overwrite the original image


def resize_image(input_path, output_path, new_width, new_height):
    try:
        # 打开图像文件
        img = Image.open(input_path)
        # 调整图像大小
        resized_img = img.resize((new_width, new_height), Resampling.LANCZOS)
        # 保存调整大小后的图像
        resized_img.save(output_path)

        print(f"图像已调整大小并保存到 {output_path}")
    except Exception as e:
        print(f"发生错误: {e}")


def refine_all_mask(directory, new_postfix=None):
    files = os.listdir(directory)

    # 过滤出.png和.jpg格式的图片
    images = [f for f in files if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG')]
    if len(images) > 1000:
        raise ValueError("图片数量超过1000张!")
    # 对图片进行排序，这样我们在重命名时不会遗漏任何图片
    images.sort()
    for idx, image in enumerate(images, 1):
        # 获取文件扩展名
        ext = os.path.splitext(image)[1]
        if new_postfix is not None:
            ext = new_postfix
        # ext = ".png"
        # 新名称格式：0001, 0002, ...
        new_name = f"mask_{idx:04}{ext}"
        # 获取图片当前的完整路径和新的完整路径
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)
        refine_mask(old_path, convert_option=False)
        # 重命名图片


def keep_largest_white_area(image_path):
    try:
        # 打开图像文件
        img = Image.open(image_path)

        # 将图像转换为NumPy数组
        img_array = np.array(img)

        # 找到最大的白色区域
        white_areas = (img_array == 255)
        labeled, num_features = ndimage.label(white_areas)

        if num_features > 0:
            largest_area_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
            largest_area = labeled == largest_area_label
            img_array[~largest_area] = 0

            # 将NumPy数组转换回图像
            result_img = Image.fromarray(img_array)

            # 保存结果到源文件
            result_img.save(image_path)
            print(f"最大的白色区域已保存到 {image_path}")
        else:
            print("未找到白色区域。")

    except Exception as e:
        print(f"发生错误: {e}")


def combine_images(image1_path, image2_path):
    # 打开第一张图片
    image1 = cv2.imread(image1_path)

    # 打开第二张图片并转为二值图像
    image2 = cv2.imread(image2_path)
    image2[image2 > 0.1] = 1
    image1 = np.where(image2 > 0, image1, 0)
    cv2.imwrite(image1_path, image1)

def combine_images_dir(imgs_dir, masks_dir):
    files = os.listdir(imgs_dir)

    # 过滤出.png和.jpg格式的图片
    images = [f for f in files if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG')]
    if len(images) > 1000:
        raise ValueError("图片数量超过1000张!")
    # 对图片进行排序，这样我们在重命名时不会遗漏任何图片
    images.sort()
    for idx, image in enumerate(images, 1):
        # 获取文件扩展名
        ext = os.path.splitext(image)[1]
        # ext = ".png"
        # 新名称格式：0001, 0002, ...
        img_name = f"/{idx:04}{ext}"
        image_path = imgs_dir + img_name
        mask_path = masks_dir + img_name
        print(image_path)
        combine_images(image_path, mask_path)




if __name__ == '__main__':
    input_path = "C:/Users/guanl/Desktop/GenshinNerf/t13/static/obj_/0035-removebg.png"  # 输入图像路径
    output_path = "C:/Users/guanl/Desktop/GenshinNerf/t13/static/obj_/0035.png"  # 输出图像路径
    output_path__ = "C:/Users/guanl/Desktop/GenshinNerf/t13/static/obj_/mask_0035.jvpg"  # 输出图像路径
    output_path = "D:/gitwork/neus_original/public_data/rws_obstacle/0042.jpg"
    output_path = "D:/gitwork/NeuS/public_data/soccer_wb/image"
    # output_path = "C:/Users/guanl/Desktop/GenshinNerf/t22/image"
    # output_path = "C:/Users/guanl/Desktop/GenshinNerf/reflect_bunny_torch_base/motion/bunny_only/render_results_cmp"
    # output_path = "C:/Users/guanl/Desktop/GenshinNerf/slip_duck_torch/duck_original/mask"

    new_width = 4624  # 新的宽度
    new_height = 3472  # 新的高度
    # resize_image(input_path, output_path, new_width, new_height)
    # refine_all_mask(output_path, new_postfix='.png')
    # exit()
    # keep_largest_white_area(output_path__)
    #
    # image = "D:/gitwork/neus_original/exp/bunny2/wmask/test.png"
    # mask = "D:/gitwork/neus_original/exp/bunny2/wmask/0001.png"
    # combine_images(image, mask)
    out_mask_path = "D:/gitwork/NeuS/public_data/soccer_wb/mask"
    combine_images_dir(output_path, out_mask_path)

