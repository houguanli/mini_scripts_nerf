import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def visualize_mse(mse):
    """
    使用OpenCV根据MSE值（二维数组）可视化一个灰度图。
    参数：
    - mse: 二维数组，表示图像的MSE值
    """
    # 确定MSE的最大值和最小值，用于归一化
    max_mse = np.max(mse)
    min_mse = np.min(mse)

    # 归一化MSE到0-1之间，然后转换为0-255的灰度值
    normalized_mse = ((mse - min_mse) / (max_mse - min_mse) * 255).astype(np.uint8)

    # 使用OpenCV显示MSE图像
    cv2.imshow('MSE Visualization', normalized_mse)
    cv2.waitKey(0)  # 等待直到有按键操作
    cv2.destroyAllWindows()  # 关闭窗口

def calculate_psnr_new(gt, mask, image):
    # gt = gt[:3]

    gt = torch.from_numpy(gt).float()
    mask = torch.from_numpy(mask).float()
    gt /= 255.0
    mask /= 255.0

    image = torch.from_numpy(image).float()
    image /= 255.0
    #
    # image_after_mask = image
    # gt_after_mask = gt
    image_after_mask = image * mask
    gt_after_mask = gt * mask
    mse = ((image_after_mask - gt_after_mask) ** 2).mean()
    mse__ = ((image_after_mask - gt_after_mask) ** 2)
    # visualize_mse(np.array(mse__))
    mse__ = mse__.sum()
    # print(mse, "  ", mse__)
    mask_sum = (mask.sum() + 1e-5)
    # print("cnt ", mask_sum)
    mse = 1.0 / np.sqrt(mse)
    mse__ = mse__ / mask_sum
    mse__ = 1.0 / np.sqrt(mse__)
    psnr = 20 * np.log10(mse__)
    return psnr


def compare_images_psnr_dir_new(folder_gt, folder_cmp, folder_mask, img_count=21, ext=".png", start_idx=0):
    psnr_values = []
    for idx in range(start_idx, img_count + start_idx):
        filename_gt = f"{idx:03}{ext}"
        filename_cmp = f"{idx:03}{ext}"
        filename_mask = f"{idx:03}{ext}"

        gt = cv2.imread(os.path.join(folder_gt, filename_gt))
        image = cv2.imread(os.path.join(folder_cmp, filename_cmp))
        mask = cv2.imread(os.path.join(folder_mask, filename_mask))
        if gt is not None and image is not None:
            psnr = calculate_psnr_new(gt, mask, image)
            psnr_values.append(psnr)
            # print(f"PSNR for {filename_gt}: {psnr}")
            print(f"{psnr}")


    return psnr_values

def calculate_iou(mask1, mask2):
    """
    计算两个掩膜之间的IoU。
    :param mask1: 第一个掩膜图像
    :param mask2: 第二个掩膜图像
    :return: IoU值
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    # print(f"{np.sum(intersection)}, {np.sum(union)}")

    return iou_score


def calculate_iou_dir(ground_truth_folder, prediction_folder,img_count=21,ext=".png", start_idx=0):
    iou_scores = []

    # 获取两个文件夹中的文件列表
    gt_files = sorted(os.listdir(ground_truth_folder))
    pred_files = sorted(os.listdir(prediction_folder))

    # 遍历文件列表，找到重名的文件进行IoU计算
    for idx in range(start_idx, img_count + start_idx):
        filename_gt  = f"{idx:03}{ext}"
        filename_cmp = f"{idx:03}{ext}"
        img1 = cv2.imread(os.path.join(ground_truth_folder, filename_gt))
        img1 = np.where(img1 > 0, img1, 0)
        img2 = cv2.imread(os.path.join(prediction_folder, filename_cmp))
        img2 = np.where(img2 > 0, img2, 0)

        if img1 is not None and img2 is not None:
            iou = calculate_iou(img1, img2)
            iou_scores.append(iou)
            # print(f"iou for {filename_gt}: {iou}")
            print(f"{iou}")

    # 计算平均IoU
    if iou_scores:
        average_iou = sum(iou_scores) / len(iou_scores)
        return iou_scores
    else:
        print("No matching files found.")


if __name__ == "__main__":
    image_count = 35
    # folder_gt =   'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic/exp/bunny_b/sp_calc/gt'
    # folder_cmp =  'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic/exp/dragon_slip_short/sp_calc/ga_sp'
    # folder_mask = 'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic/exp/dragon_slip_short/sp_calc/white_sp'
    folder_gt =   'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic/exp/bunny_bounce_long/IoU_calc/gt_wb'
    folder_cmp =  'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic/exp/bunny_bounce_long/IoU_calc/pga_wb'
    folder_mask = 'C:/Users/GUANLI.HOU/Desktop/real_world/dynamic/exp/bunny_bounce_long/IoU_calc/white_bk'

    # iou_values = calculate_iou_dir(ground_truth_folder=folder_mask, prediction_folder=folder_cmp, img_count=image_count, start_idx=0)

    psnr_values = compare_images_psnr_dir_new(folder_gt=folder_gt, folder_cmp=folder_cmp,folder_mask=folder_mask, img_count=image_count, start_idx=0)

    # if iou_values:
    #     average_psnr = sum(iou_values) / len(iou_values)
    #     print(f"Average IOU: {average_psnr}")
    # else:
    #     print("No images to compare.")

    exit()
    psnr_values = calculate_iou_dir(folder_gt, folder_cmp, img_count=19)

"""
[[1, -1, 0,  2],
 [3, -1, -1, 7],
 [1, -2, -1, 0],
 [4, 4, 2, 18]]
 [7, 26, -6, 90]
"""