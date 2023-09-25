# this file has two functions
# the first function is used to generate a white ball as a moving mask which is obviously white
# the second function is used to generate the mask for the obstacle

import cv2
import numpy as np
import os

def generate_mask_from_image(input_path, display_width=1280, brush_size=200):
    drawing = False
    _flip=1
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    original_height, original_width = image.shape[:2]
    # 计算缩放比例
    scale_factor = display_width / original_width
    display_height = int(original_height * scale_factor)
    print(str(display_width) + " " + str(display_height))
    # 创建一个和原始图像大小对应的全0的mask
    mask_original = np.zeros((original_height, original_width), dtype=np.uint8)

    # 缩放图像
    image_resized = cv2.resize(image, (display_width, display_height))

    def draw_circle(event, x, y, flags, param):
        nonlocal drawing
        nonlocal mask_original
        nonlocal _flip
        nonlocal brush_size
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            if _flip==1:
                brush_size = 10
            else:
                brush_size = 200
            _flip = (_flip + 1) % 2
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # 当鼠标拖动时，为对应区域的 brush_size x brush_size 像素设置为1
            half_size = brush_size // 2
            original_x = int(x / scale_factor)
            original_y = int(y / scale_factor)

            mask_original[original_y - half_size:original_y + half_size + 1,
            original_x - half_size:original_x + half_size + 1] = 1

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', display_width, display_height)  # 设置窗口大小
    cv2.setMouseCallback('Image', draw_circle)

    while True:
        display = image_resized.copy()
        mask_display = cv2.resize(mask_original, (display_width, display_height))
        display[mask_display == 1] = [0, 0, 255]  # 将mask为1的区域在显示图上标红

        cv2.imshow('Image', display)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # 任意键盘按键退出
            break

    # 保存mask为黑白图片
    output_filename = os.path.dirname(input_path) + "/mask_" + input_path.split("/")[-1]
    print(output_filename)
    cv2.imwrite(output_filename, mask_original * 255)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 使用方法
    path = "C:/Users/GUANL/Desktop/GenshinNerf/__tmp/mask/0020.jpg"
    path = "D:/gitwork/NeuS/public_data/rws_object2/image/0011.jpg"
    generate_mask_from_image(path)  # 替换为你的图片路径
