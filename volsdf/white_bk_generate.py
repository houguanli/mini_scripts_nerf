import cv2
import numpy as np
import os

def white_bk_generator(dir_path, w, h, cnt):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for i in range(cnt):
        img = np.ones((h, w, 3), dtype=np.uint8) * 255  # Create a white image
        file_name = f"{i:03d}.png"  # Naming the file starting from 000
        file_path = os.path.join(dir_path, file_name)
        cv2.imwrite(file_path, img)

if __name__ == '__main__':
    
    # 调用函数生成白色图片
    dir_path = "C:/Users/GUANLI.HOU/Desktop/neural_rig/bunny/bunny_stand/mask"
    w = 1024  # 图片宽度
    h = 1024  # 图片高度
    cnt = 40  # 生成图片数量
    white_bk_generator(dir_path, w, h, cnt)
