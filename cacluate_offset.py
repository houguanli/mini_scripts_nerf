import cv2
import numpy as np
import os


def calculate_offset_and_mask(img_path1, img_path2, max_offset=10):
    """
    根据两个图像路径计算偏移量，同时生成参与计算的像素掩码。

    参数：
    - img_path1: 第一幅图像的路径
    - img_path2: 第二幅图像的路径
    - max_offset: 偏移量的最大绝对值

    返回：
    - (delta_x, delta_y): 最佳偏移量
    - mask: 表示参与了计算的像素掩码，0-255的图像，255表示参与计算
    """
    # 使用OpenCV读取图像
    I1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    I2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

    H, W, _ = I1.shape
    min_error = float('inf')
    best_offset = (0, 0)
    best_mask = np.zeros_like(I1)

    # 遍历所有可能的偏移量
    for delta_x in range(-max_offset, max_offset + 1):
        for delta_y in range(-max_offset, max_offset + 1):
            # 创建偏移后的图像和掩码
            M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
            I1_shifted = cv2.warpAffine(I1, M, (W, H))
            mask = cv2.warpAffine(np.ones_like(I1) * 255, M, (W, H))

            # 计算当前偏移量下的误差（只考虑掩码内的像素）
            error = np.sum(((I1_shifted - I2) ** 2) * (mask > 0))

            # 更新最小误差和最佳偏移量及掩码
            if error < min_error:
                min_error = error
                best_offset = (delta_x, delta_y)
                best_mask = mask
    shifted_I1 = cv2.warpAffine(I1, np.float32([[1, 0, best_offset[0]], [0, 1, best_offset[1]]]), (W, H))
    return best_offset, best_mask, shifted_I1


def calc_dir(dir_path1, dir_path2, img_count=1, start_idx=0, ext=".png", max_offset=20):
    """
    对两个文件夹内所有同名图片文件计算偏移量和掩码。

    参数：
    - dir_path1: 第一个文件夹路径
    - dir_path2: 第二个文件夹路径

    返回：
    - offsets_and_masks: 字典，键为文件名，值为(最佳偏移量, 掩码)
    """
    offsets_and_masks = {}
    for idx in range(start_idx, img_count + start_idx):
        filename_gt  = f"{idx:03}{ext}"
        filename_cmp = f"{idx:03}{ext}"
        filename_new, mask_new = f"rmi{idx:03}{ext}", f"rmm{idx:03}{ext}"
        img_path1 = os.path.join(dir_path1, filename_gt)
        img_path2 = os.path.join(dir_path2, filename_cmp)
        if os.path.exists(img_path2):
            offset, mask, shifted_I1 = calculate_offset_and_mask(img_path1, img_path2, max_offset=max_offset)
            print(offset)
            offsets_and_masks[filename_cmp] = (offset, mask)
            img_path_new = os.path.join(dir_path1, filename_new)
            mask_path_new = os.path.join(dir_path1, mask_new)
            cv2.imwrite(mask_path_new, mask)
            cv2.imwrite(img_path_new, shifted_I1)

            # store the offset mask and write a new cmp_image

    return offsets_and_masks


def create_custom_image(width, height, y_threshold):
    """
    创建一个指定宽度和高度的图像，其中y大于某个阈值的部分为黑色，其他为白色。

    参数:
    width (int): 图像的宽度。
    height (int): 图像的高度。
    y_threshold (int): y坐标的阈值，大于此值的像素将被设为黑色。

    返回:
    numpy.ndarray: 创建的定制图像。
    """
    # 首先创建一个全白图像
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    # 如果y坐标大于阈值，则设置像素为黑色
    if y_threshold < height:  # 确保阈值在图像高度范围内
        image[y_threshold:height, :, :] = 0  # 将y大于阈值的部分设为黑色

    return image
if __name__ == "__main__":
    # white = create_custom_image(800, 600, y_threshold=550)
    # for i in range (0, 18):
    #     cv2.imwrite(f"C:/Users/GUANLI.HOU/Desktop/real_world/dynamic_short/exp/bunny_bounce/sp_calc/white_bk/{i:03}.png", white)
    #
    # exit()
    image_count = 24
    folder_gt =   'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic_short/exp/dragon_slip_short/sp_calc/gt'
    folder_cmp =  'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic_short/exp/dragon_slip_short/sp_calc/ga'
    iou_values = calc_dir(dir_path1=folder_cmp, dir_path2=folder_gt, img_count=image_count, start_idx=0, max_offset=10)
    print("generate ", folder_cmp, " done")
    # folder_gt =   'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic_short/exp/yoyo_book/sp_calc/gt'
    # folder_cmp =  'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic_short/exp/yoyo_book/sp_calc/pa'
    # iou_values = calc_dir(dir_path1=folder_cmp, dir_path2=folder_gt, img_count=image_count, start_idx=0, max_offset=5)

