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
    threshold_value = 150
    height, width, _ = image.shape
    refined_mask = image.copy()
    for y in range(height):
        for x in range(width):
            if all(pixel > threshold_value for pixel in image[y, x]):
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


if __name__ == '__main__':
    input_path = "C:/Users/guanl/Desktop/GenshinNerf/t13/static/obj_/0035-removebg.png"  # 输入图像路径
    output_path = "C:/Users/guanl/Desktop/GenshinNerf/t13/static/obj_/0035.png"  # 输出图像路径
    output_path__ = "C:/Users/guanl/Desktop/GenshinNerf/t13/static/obj_/mask_0035.jpg"  # 输出图像路径
    output_path = "D:\\gitwork\\genshinnerf\\dynamic_test\\mask\\transform0001_mask.png"
    output_path__ = "D:\\gitwork\\genshinnerf\\dynamic_test\\mask\\transform0001_.png"
    new_width = 4624  # 新的宽度
    new_height = 3472  # 新的高度
    # resize_image(input_path, output_path, new_width, new_height)
    make_init_mask(output_path, output_path__)
    # keep_largest_white_area(output_path__)
