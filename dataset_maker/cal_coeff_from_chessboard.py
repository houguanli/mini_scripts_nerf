import numpy as np
import cv2
import glob


def calibrate_camera(images_path_pattern, grid_size, grid_square_size):
    """
    标定摄像机。

    :param images_path_pattern: 使用glob模式匹配的棋盘格图片路径。
    :param grid_size: 一个二元组，表示棋盘的格子数 (corner_count_x, corner_count_y)。
    :param grid_square_size: 棋盘上每个格子的实际尺寸。
    :return: 内参矩阵，畸变系数，外参的旋转和平移向量。
    """

    # 空数组初始化，用于存储所有检测到的角点。
    object_points = []  # 3d point in real world space
    image_points = []  # 2d points in image plane.

    # 准备棋盘格的3D点，例如(0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[1] * grid_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * grid_square_size

    # 读取所有图像并找到角。
    images = glob.glob(images_path_pattern)
    for fname in images:
        img = cv2.imread(fname)
        print("read frame name " + fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        # 如果找到，将其添加到我们的数据中
        if ret:
            print("detect!")
            object_points.append(objp)
            image_points.append(corners)
        else:
            print("no chess detect")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def generate_chessboard(size, h_c, w_c, filename='chessboard.jpg'):
    """
    生成棋盘格图片。

    :param size: 棋盘上每个格子的像素大小。
    :param h_c: 棋盘格的高度（垂直方向上的格子数）。
    :param w_c: 棋盘格的宽度（水平方向上的格子数）。
    :param filename: 输出的图片文件名。
    """
    # 定义棋盘格的宽和高
    board_height = h_c * size
    board_width = w_c * size

    # 创建一个黑色的空图像
    board = np.zeros((board_height, board_width), dtype=np.uint8)

    # 填充白色方块
    for i in range(h_c):
        for j in range(w_c):
            if (i + j) % 2 == 0:
                board[i * size:(i + 1) * size, j * size:(j + 1) * size] = 255

    cv2.imwrite(filename, board)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    images_path_pattern = 'C:/Users/guanl/Desktop/GenshinNerf/t9/*.jpg'
    chess_board_path = 'C:/Users/guanl/Desktop/GenshinNerf/t9/chessboard.jpg'
    # generate_chessboard(100, 9, 6, chess_board_path)
    # exit(0)
    grid_size = (8, 5)  # 表示棋盘上有9x6个角。
    grid_square_size = 0.0265  # 0.265 per set
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(images_path_pattern, grid_size, grid_square_size)
    print(mtx)
    print(ret)
    print(dist)

# 示例调用
# 例如：images_path_pattern = 'path_to_chessboard_images/*.jpg'
# grid_size = (9, 6) 表示棋盘上有9x6个角。
# grid_square_size = 0.025 (假设每个格子是25mm或2.5厘米)
# mtx, dist, rvecs, tvecs = calibrate_camera(images_path_pattern, grid_size, grid_square_size)
