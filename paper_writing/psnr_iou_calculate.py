import cv2
import os
import numpy as np

def calculate_psnr(image1, image2):
    general_mask = np.logical_or(image1, image2)
    local_mask = np.logical_and(image1, image2)
    # image1 = image1[general_mask]
    image2 = np.where(local_mask, image2, 0)
    mask_sum = local_mask.sum()
    mse = (((image1 - image2) ** 2 * general_mask).sum() / (mask_sum))
    mse = 255.0 / np.sqrt(mse)
    print(np.array(image1).shape)
    print("AND MASK", np.sum(np.array(local_mask)))
    print("OR MASK", np.sum(np.array(general_mask)))

    # print("mse", mse)

    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(mse)
    return psnr
def compare_images_psnr_dir(folder_gt, folder_cmp, img_count=21, ext=".png", start_idx=0):
    psnr_values = []
    for idx in range(start_idx, img_count + start_idx):
        filename_gt  = f"{idx:04}{ext}"
        filename_cmp = f"{idx}{ext}"
        if 1:
            img1 = cv2.imread(os.path.join(folder_gt, filename_gt))
            img2 = cv2.imread(os.path.join(folder_cmp, filename_cmp))
            if img1 is not None and img2 is not None:
                psnr = calculate_psnr(img1, img2)
                psnr_values.append(psnr)
                print(f"PSNR for {filename_gt}: {psnr}")
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
    return iou_score


def calculate_iou_dir(ground_truth_folder, prediction_folder,img_count=21,ext=".png", start_idx=0):
    iou_scores = []

    # 获取两个文件夹中的文件列表
    gt_files = sorted(os.listdir(ground_truth_folder))
    pred_files = sorted(os.listdir(prediction_folder))

    # 遍历文件列表，找到重名的文件进行IoU计算
    for idx in range(start_idx, img_count + start_idx):
        filename_gt  = f"{idx:04}{ext}"
        filename_cmp = f"debug{idx}{ext}"
        if 1:
            img1 = cv2.imread(os.path.join(folder_gt, filename_gt))
            img2 = cv2.imread(os.path.join(folder_cmp, filename_cmp))
            if img1 is not None and img2 is not None:
                iou = calculate_iou(img1, img2)
                iou_scores.append(iou)
                print(f"PSNR for {filename_gt}: {iou}")
    # 计算平均IoU
    if iou_scores:
        average_iou = sum(iou_scores) / len(iou_scores)
        return iou_scores
    else:
        print("No matching files found.")


if __name__ == "__main__":
    folder_gt = 'C:/Users/guanli.hou/Desktop/real_world/dynamic/exp/soap_geo/image_gt_wmask'
    folder_cmp = 'C:/Users/guanli.hou/Desktop/real_world/dynamic/exp/soap_geo/render_result_sequence_for_refine_RT_soap_init'

    folder_gt = 'C:/Users/guanli.hou/Desktop/real_world/dynamic/exp/bunny_bounce/gt'
    folder_cmp = 'C:/Users/guanli.hou/Desktop/real_world/dynamic/exp/bunny_bounce/iter_0'
    psnr_values = compare_images_psnr_dir(folder_gt, folder_cmp, img_count=19)
    if psnr_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        print(f"Average PSNR: {average_psnr}")
    else:
        print("No images to compare.")