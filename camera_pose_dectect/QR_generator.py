import cv2
import numpy as np
MUTIPLE_FACTOR = 80


def create_aruco_board_image(output_path="frame.png", markersX=1, markersY=1, markerLength=MUTIPLE_FACTOR * 5
                             , markerSeparation= MUTIPLE_FACTOR * 1, margins=MUTIPLE_FACTOR * 1, borderBits=10):
    """
    创建一个ArUco棋盘图像并保存到指定路径。

    参数:
    - output_path: 输出的图像路径
    - markersX: X轴上标记的数量
    - markersY: Y轴上标记的数量
    - markerLength: 标记的长度，单位是像素
    - markerSeparation: 每个标记之间的间隔，单位像素
    - margins: 标记与边界之间的间隔
    - borderBits: 标记的边界所占的bit位数
    """
    width = markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins
    height = markersY * (markerLength + markerSeparation) - markerSeparation + 2 * margins
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    size_t = 10  # 示例，你可以设置为任何整数值
    ids = np.array(list(range(1, size_t + 1)))
    # 生成aruco的格子，或者生成1个aruco,(1, 1)表示生成了1行1列
    board = cv2.aruco.GridBoard((3, 3), 0.015, 0.011, dictionary)
    # board = cv2.aruco.CharucoBoard((11, 8), 0.015, 0.011, dictionary)
    img = cv2.aruco.Board.generateImage(board, (width, height), 0, 1)
    img1 = cv2.aruco.Board.generateImage(board, (width, height), 20, 20)
    # img2 = cv2.aruco.Dictionary.generateImageMarker(1, 9, 9)
    cv2.imshow('Highlighted Image', img1)
    cv2.imwrite(output_path, img1)
    cv2.waitKey(0)


if __name__ == "__main__":
    # 调用函数，生成ArUco棋盘图像并保存为"frame.png"
    create_aruco_board_image()
