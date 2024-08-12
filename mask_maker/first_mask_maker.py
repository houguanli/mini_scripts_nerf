import os

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from scipy import ndimage
def sort_key(s):
    return int(s.split(".")[0][0:])
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
def merge_images(img_path1, img_path2, threshold):
    """
    根据x和y坐标的阈值合成一张图片。
    参数：
    img_path1: 第一张图片的路径
    img_path2: 第二张图片的路径
    合成后的图片
    """
    # 读取两张图片
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    # 确保图片具有相同的尺寸
    if img1.shape != img2.shape:
        raise ValueError("两张图片尺寸不同，请确保它们具有相同的分辨率。")

    # 创建一个空白图片用于存放结果
    result_img = np.zeros_like(img1)

    # 遍历每个像素点进行合成
    for y in range(img1.shape[0]): # H, W
        for x in range(img1.shape[1]):
            if x > threshold:
                result_img[y, x] = img1[y, x]
            else:
                result_img[y, x] = img2[y, x]
    cv2.imwrite(img_path1, result_img)
    return

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

def remove_bk(img_path, flag_01=False): # assume png
    # image = cv2.imread(img_path)
    # 读取图像并以RGBA格式存储
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if flag_01: # treat as a bio-pixel image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #reread
        expanded_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        # 将单通道图像的值复制到所有四个通道
        for i in range(4):
            expanded_image[:, :, i] = image
        image = expanded_image

    # 如果图像为三维数组，添加一个维度使其变为四维数组
    if  image.shape[2] == 3:
        print("No alpha！！！")
        # Create an alpha channel with all values set to 255 (fully opaque)
        alpha_channel = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        # Add the alpha channel to the image
        image = np.dstack((image, alpha_channel))
    threshold_value,  threshold_value_max= 0, 255 # black or white bk
    height, width, _ = image.shape
    refined_mask = image.copy()
    for y in range(height):
        for x in range(width):
            pixel_sum = 0. + image[y, x][0] + image[y, x][1] + image[y, x][2]
            # if all(pixel > threshold_value for pixel in image[y, x]):
            if threshold_value < pixel_sum : # not remake bg
                refined_mask[y, x] = refined_mask[y, x]
                # refined_mask[y, x] = [255, 255, 255]   # 设置为白色
            else:
                refined_mask[y, x] = [0, 0, 0, 0]  # 设置为 透明
    print("saving refined mask at " + img_path)
    cv2.imwrite(img_path, refined_mask)  # overwrite the original image
def refine_alpha(img_path, new_alpha): # assume png
    # image = cv2.imread(img_path)
    # 读取图像并以RGBA格式存储
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # 如果图像为三维数组，添加一个维度使其变为四维数组
    if  image.shape[2] == 3:
        print("No alpha！！！")
        # Create an alpha channel with all values set to 255 (fully opaque)
        exit(-1)
    else:
        print("remaking " + img_path + " with alpha " + str(new_alpha))
    threshold_value,  threshold_value_max= 1, 200 # black or white bk
    height, width, _ = image.shape
    refined_mask = image.copy()
    for y in range(height):
        for x in range(width):
            pixel_sum = image[y, x][0] + 0.
            # if all(pixel > threshold_value for pixel in image[y, x]):
            if threshold_value < pixel_sum < threshold_value_max: # not remake bg
                refined_mask[y, x][3] = new_alpha
    cv2.imwrite(img_path, refined_mask)  # overwrite the original image
def resize_image(input_path, output_path, new_width, new_height):
    try:
        # 打开图像文件
        image = Image.open(input_path)
        # 调整图像大小
        resized_img = image.resize((new_width, new_height), Resampling.LANCZOS)
        # 保存调整大小后的图像
        resized_img.save(output_path)

        print(f"图像已调整大小并保存到 {output_path}")
    except Exception as e:
        print(f"发生错误: {e}")
def remove_all_bk(directory, new_postfix=None, flag_01=False):
    files = os.listdir(directory)

    # 过滤出.png和.jpg格式的图片
    images = [f for f in files if f.endswith('.png') or f.endswith('.PNG')]
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
        new_name = f"mask_{idx:03}{ext}"
        # 获取图片当前的完整路径和新的完整路径
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)
        remove_bk(old_path, flag_01=flag_01)
        # 重命名图片

def remove_all_bk_recursive(directory, new_postfix=None):
    idx = 1  # 初始化索引
    for root, dirs, files in os.walk(directory):
        # 过滤出.png和.PNG格式的图片
        images = [f for f in files if f.endswith('.png') or f.endswith('.PNG')]
        if len(images) > 1000:
            raise ValueError("图片数量超过1000张!")
        # 对图片进行排序，这样我们在重命名时不会遗漏任何图片
        images.sort()

        for image in images:
            # 获取文件扩展名
            ext = os.path.splitext(image)[1]
            if new_postfix is not None:
                ext = new_postfix
            # 新名称格式：mask_0001, mask_0002, ...
            new_name = f"mask_{idx:03}{ext}"
            # 获取图片当前的完整路径
            old_path = os.path.join(root, image)
            # 移除背景
            remove_bk(old_path)
            # 如果需要，重命名图片
            if new_postfix is not None:
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)

            idx += 1  # 更新索引
def remake_all_alpha(directory, new_postfix=None, sp_sort_flag=True):
    files = os.listdir(directory)

    # 过滤出.png和.jpg格式的图片
    images = [f for f in files if f.endswith('.png') or f.endswith('.PNG')]
    if len(images) > 1000:
        raise ValueError("图片数量超过1000张!")
    # 对图片进行排序，这样我们在重命名时不会遗漏任何图片
    if not sp_sort_flag:
        images.sort()
    else:
        images = sorted(images, key=sort_key)

    for idx, image in enumerate(images, 1):
        # 获取文件扩展名
        ext = os.path.splitext(image)[1]
        if new_postfix is not None:
            ext = new_postfix
        # ext = ".png"
        # 新名称格式：0001, 0002, ...
        new_name = f"mask_{idx:03}{ext}"
        # 获取图片当前的完整路径和新的完整路径
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)
        refine_alpha(old_path, int((idx + 0.) * 255 / len(images) * 0.7 + 255 * 0.3)) # 30% ~ 100%
        # 重命名图片
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
        new_name = f"mask_{idx:03}{ext}"
        # 获取图片当前的完整路径和新的完整路径
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)
        refine_mask(old_path, convert_option=False)
        # 重命名图片

def keep_largest_white_area(image_path):
    try:
        # 打开图像文件
        image = Image.open(image_path)

        # 将图像转换为NumPy数组
        img_array = np.array(image)

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

def image_and(image1_path, image2_path):
    image1 = cv2.imread(image1_path)

    # 打开第二张图片并转为二值图像
    image2 = cv2.imread(image2_path)
    image1 = cv2.bitwise_and(image1, image2)
    cv2.imwrite(image1_path, image1)
def combine_images(image1_path, image2_path):
    # 打开第一张图片
    image1 = cv2.imread(image1_path)

    # 打开第二张图片并转为二值图像
    image2 = cv2.imread(image2_path)
    image2[image2 > 0.1] = 1
    image1 = np.where(image2 > 0, image1, 0)
    cv2.imwrite(image1_path, image1)

def replace_image(image1_path, image2_path):
    # 打开第一张图片
    image1 = cv2.imread(image1_path)

    # 打开第二张图片并转为二值图像
    image2 = cv2.imread(image2_path)
    image2_cp = image2
    # image2[image2 > 0.1] = 1
    image1 = np.where(image2 > 0, image2_cp, image1)
    cv2.imwrite(image1_path, image1)


def combine_images_dir_recursive(root_imgs_dir, root_masks_dir, start_idx=0):
    idx = start_idx
    for root, dirs, files in os.walk(root_imgs_dir):
        # 过滤出.png和.jpg格式的图片
        images = [f for f in files if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG')]
        if len(images) > 1000:
            raise ValueError("图片数量超过1000张!")
        # 对图片进行排序，这样我们在重命名时不会遗漏任何图片
        images.sort()

        for image in images:
            # 计算当前图片和掩码的相对路径
            relative_path = os.path.relpath(root, root_imgs_dir)
            mask_root = os.path.join(root_masks_dir, relative_path)

            if not os.path.exists(mask_root):
                print(f"掩码文件夹 {mask_root} 不存在，跳过 {image}")
                continue

            # 获取文件扩展名
            ext = os.path.splitext(image)[1]
            img_name = f"{idx:04}{ext}"

            image_path = os.path.join(root, image)
            mask_path = os.path.join(mask_root, img_name)

            if os.path.exists(mask_path):
                combine_images(image_path, mask_path)
                idx += 1
            else:
                print(f"找不到掩码文件：{mask_path}")


def replace_with_alpha(img_path, overlay_img_path):
    """
    使用具有alpha通道的第二张图片覆盖第一张图片的相应像素。
    参数:
    img_path: 第一张图片的路径。
    overlay_img_path: 第二张具有alpha通道的图片的路径。
    返回:
    合成后的图片。
    """
    # 读取第一张图片
    img = cv2.imread(img_path)
    # 读取第二张图片，包括其alpha通道
    overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
    # 确保overlay_img包含alpha通道
    if overlay_img.shape[2] == 4:
        # 分离alpha通道和颜色通道
        overlay_color = overlay_img[:, :, :3]
        alpha_mask = overlay_img[:, :, 3] / 255.0
        # 根据alpha值合成图片
        for c in range(0, 3):
            img[:, :, c] = (1 - alpha_mask) * img[:, :, c] + alpha_mask * overlay_color[:, :, c]
    else:
        raise ValueError("第二张图片没有alpha通道。")
    return img


def replace_images_with_alpha(folder_path, background_img_path, num_images):
    """
    批量处理指定文件夹中的图片，使用给定的背景图片合成。
    参数:
    folder_path: 需要合成的图片所在的文件夹路径。
    background_img_path: 背景图片的路径。
    num_images: 需要处理的图片数量。

    返回:
    None，但会在当前目录下保存合成后的图片。
    """
    for i in range(num_images):
        # 构造文件名，假设文件名是从"000"开始的
        filename = f"{i:03}.png"
        img_path = os.path.join(folder_path, filename)

        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"文件 {img_path} 不存在。")
            continue

        # 合成图片
        merged_img = replace_with_alpha(background_img_path, img_path)

        # 保存合成后的图片
        cv2.imwrite(img_path, merged_img)
        print(f"saved {img_path}")


if __name__ == '__main__':
    input_path = "C:/Users/guanl/Desktop/GenshinNerf/t13/static/obj_/0035-removebg.png"  # 输入图像路径
    output_path = "D:/gitwork/neus_original/public_data/rws_obstacle/0042.jpg"
    output_path = "D:/gitwork/NeuS/public_data/soccer_wb/image"
    output_path = "C:/Users/guanl/Desktop/GenshinNerf/t22/image"
    output_path = "C:/Users/guanl/Desktop/GenshinNerf/reflect_bunny_torch_base/motion/bunny_only/gt"
    output_path = "C:/Users/GUANLI.HOU/Desktop/fake_full_render/bunny/"
    output_path = 'C:/Users/GUANLI.HOU/Desktop/GenshinNerf/PGA_NeuS_Paper_writing/changing_alpha/dragon_sub/dragon_mask_sub/'
    bk_path = output_path + "bk.png"
    cmb_dir = output_path + "tmp"
    # output_path = "C:/Users/GUANLI.HOU/Desktop/real_world/dynamic_short/exp/dragon_slip_short/IoU_calc/pga"
    new_width = 4624  # 新的宽度
    new_height = 3472  # 新的高度
    # refine_all_mask(directory=output_path)
    # replace_images_with_alpha(cmb_dir, bk_path, num_images=35)
    # resize_image(input_path, output_path, new_width, new_height)
    remove_all_bk(output_path, new_postfix='.png', flag_01=True)

    # remake_all_alpha(output_path, new_postfix='.png')
    # merge_images("C:/Users/GUANLI.HOU/Desktop/001.png",
    #              "C:/Users/GUANLI.HOU/Desktop/000.png", threshold=400)
    exit()
    # keep_largest_white_area(output_path__)
    #
    # image = "D:/gitwork/neus_original/exp/bunny2/wmask/test.png"
    # mask = "D:/gitwork/neus_original/exp/bunny2/wmask/0001.png"
    # combine_images(image, mask)
    out_mask_path = "C:/Users/GUANLI.HOU/Desktop/real_world/static/public_data/dragon_pos2/mask"
    # combine_images_dir(output_path, out_mask_path, start_idx=0)

