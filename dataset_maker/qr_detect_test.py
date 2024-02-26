import cv2 as cv


def detect_qr(filename):
    dectorParams = cv.aruco.DetectorParameters()
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
    detector = cv.aruco.ArucoDetector(dictionary, dectorParams)

    # Load image
    inputImage = cv.imread(filename)
    # show image
    # cv.imshow("img", inputImage)
    # detect markersj
    makerCorners, makerIds, rejectedImgPoints = detector.detectMarkers(inputImage)
    outpuImage = inputImage.copy()
    original_width, original_height = outpuImage.shape[1], outpuImage.shape[0]
    desired_width = 1080
    aspect_ratio = original_height / original_width
    new_height = int(desired_width * aspect_ratio)

    # draw detected markers
    cv.namedWindow("img2", 0)
    cv.resizeWindow("img2", 1080, new_height)
    cv.aruco.drawDetectedMarkers(outpuImage, makerCorners, makerIds)

    # show image
    cv.imshow("img2", outpuImage)
    # cv.resizeWindow("img", 1080, 720)

    # wait for key
    cv.waitKey(0)
    # destroy all windows
    cv.destroyAllWindows()


import numpy as np


def remove_outliers(rvecs, tvecs, threshold=10.0):
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)

    while True:
        # 计算平均值
        rvecs_mean = np.mean(rvecs, axis=0)
        tvecs_mean = np.mean(tvecs, axis=0)

        # 计算rvecs和tvecs与其对应平均值的差值的比例
        rvecs_diff_ratio = np.abs((rvecs - rvecs_mean) / rvecs_mean) * 100
        tvecs_diff_ratio = np.abs((tvecs - tvecs_mean) / tvecs_mean) * 100

        # 找出超出阈值的项
        rvecs_outliers = np.any(rvecs_diff_ratio > threshold, axis=1)
        tvecs_outliers = np.any(tvecs_diff_ratio > threshold, axis=1)
        outliers = np.logical_or(rvecs_outliers, tvecs_outliers)

        # 如果没有超出阈值的数据，就跳出循环
        if not np.any(outliers):
            break

        # 打印并移除超出阈值的数据
        print("Removing outliers: ")
        print("Rvecs: ", rvecs[outliers])
        print("Tvecs: ", tvecs[outliers])
        rvecs = rvecs[~outliers]
        tvecs = tvecs[~outliers]

    return rvecs, tvecs

def tst():
    vx_count = [10, 50, 58, 25, 158, 45, 25, -25, 30, 15, 5, -60, -20, -48, -30, 30, 14]
    zfb_count = [30, 68, 328, 328, 120, 20, 60, 160, 37]
    vx_count, zfb_count = np.array(vx_count), np.array(zfb_count)
    print(np.sum(vx_count))
    print(np.sum(zfb_count))

# 示例数据

if __name__ == '__main__':
    tst()
    exit()
    filename = 'C:/Users/GUANL/Desktop/GenshinNerf/t10/0001.jpg'

    detect_qr(filename)
    rvecs = [[1.1, 1.2, 1.3], [1.15, 1.25, 1.35], [1.3, 1.5, 1.8], [1.12, 1.22, 1.32]]
    tvecs = [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.3, 0.5, 0.8], [0.12, 0.22, 0.32]]

    # 调用函数
    new_rvecs, new_tvecs = remove_outliers(rvecs, tvecs)
    print("Filtered Rvecs:", new_rvecs)
    print("Filtered Tvecs:", new_tvecs)

