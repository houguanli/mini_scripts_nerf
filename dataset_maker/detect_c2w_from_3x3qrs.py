import cv2
import numpy as np

# red axis refers to X and green refers to Y
debug_mode = False
show_res_img = False
mode_dict = {'EXIF_mode': 'EXIF_mode', 'chessboard_mode': 'chessboard_mode', 'yuanmu_2320_mode': 'yuanmu_2320_mode',
             'dynamic_camera_mode': 'dynamic_camera_mode', 'yuanmu_1920_modes': 'yuanmu_1920_modes', 'yuanmu_1920_moded': 'yuanmu_1920_moded'}
ar_set_modes = ["wood_mode", "a4_mode"]
# K_mode = 'dynamic_camera_mode'
qr_move_vector = []
qr_code_length = 0.024  # the size of the qr code
board_length = 0.61  # the length of the test board
default_marker_size = 0.031
default_5x7_marker_size = 0.0185
arcuo_size_5x7e = 0.0185
default_camera_EXIF_K = [[3.21111111e+03, 0.00000000e+00, 2.31200000e+03],
                         [0.00000000e+00, 3.61666667e+03, 1.73600000e+03],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]  # default camera mat
default_chessboard_K = [[3.55085455e+03, 0.00000000e+00, 2.23088539e+03],
                        [0.00000000e+00, 3.54865667e+03, 1.67835047e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

dynamic_camera_K = [[735.15009953, 0., 961.93174061],
                    [0., 733.3960477, 553.13510509],
                    [0., 0., 1.]]  # this camera K is from a pan-tilt-zoom camera
dynamic_camera_K_2700 = [[1.20204328e+03, 0.00000000e+00, 1.36849027e+03],
                         [0.00000000e+00, 1.19620501e+03, 7.01540732e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
static_phone_K_yuanmu_1920 =  \
    [[1.43084900e+03, -3.29807117e-05,  9.57443726e+02],
 [ 0.00000000e+00,  1.45261902e+03,  5.39865356e+02],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

dynaimic_phone_K_yuanmu_1920 = \
    [[4.37470801e+03, 1.80489751e-05, 1.17978186e+03],
     [0.00000000e+00, 4.38952051e+03, 4.72950500e+02],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

dynaimic_phone_K_yuanmu_1920_tree = \
[[3.27436621e+03,  4.26224760e-05,  9.48091187e+02],
 [ 0.00000000e+00, 3.27436621e+03,  2.97527985e+02],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]

dynaimic_phone_K_yuanmu_2320 =  [[4.34950439e+03, 4.80758608e-05, 1.19272156e+03],
                                 [0.00000000e+00, 4.40551123e+03, 5.40642639e+02],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
cmp_diff_pixel = 1.525774468624755  # if this is less than 1, it will be a good assm
cmp_diff_dynamic_camera_pixel_2700 = 0.9102474884918834
cmp_diff_dynamic_camera_pixel = 0.8204705806151762
default_chessboard_coeffs = np.array([4.25182401e-02, 3.70250319e-01, 4.63505072e-04, -3.93194259e-03])
dynamic_camera_chessboard_coeffs = np.array([0.00948553, -0.02774021, 0.00214638, 0.02973592])
dynamic_camera_chessboard_coeffs_2700 = np.array([0.02870973, -0.04853193, -0.0049702, 0.0004024, 0.03914582])

idol_coeffs = np.zeros(4)
qrs_id_pos_dict_old = {
    "2": [0.550, 0.037], "1": [0.292, 0.070], "0": [0.035, 0.080],
    "5": [0.560, 0.259], "4": [0.259, 0.279], "3": [0.039, 0.293],
    "8": [0.563, 0.500], "7": [0.281, 0.544], "6": [0.010, 0.567],

}  # this dict stores the measured pos of each qr code from the upper right corner as the board

qrs_id_pos_dict__ = {
    "7": [0.564, 0.018], "8": [0.300, 0.014], "6": [0.035, 0.080],
    "5": [0.560, 0.259], "4": [0.259, 0.279], "3": [0.010, 0.300],
    "2": [0.563, 0.500], "1": [0.291, 0.564], "0": [0.010, 0.567],

}  # this dict stores the measured pos of each qr code from the upper right corner as the board

# in the real arcuo direction, X = Y_old, Y = -X_old
qrs_id_pos_dict = {
    "2": [0.037, -0.550], "1": [0.070, -0.292], "0": [0.080, -0.035],
    "5": [0.259, -0.560], "4": [0.279, -0.259], "3": [0.293, -0.039],
    "8": [0.500, -0.563], "7": [0.544, -0.281], "6": [0.567, -0.010],

}  # this dict stores the measured pos of each qr code from the upper right corner as the board

wood_h = [10.6, 28.45, 47.55]  # horizontal distance in wood board, notice that it is cm
wood_v = [10.3, 31.3, 51.3]  # vertical distance for in wood board


def init_qrs_id_dict_wood():
    tmp_set = {}
    for i in range(0, 9):
        tmp_set[str(i)] = [wood_h[i % 3] / 100,
                           wood_v[int((i - i % 3) / 3)] / 100]  # init position, convert it to meter
    tmp_set["8"][1] = tmp_set["8"][1] + 0.005  # this is a small distance error in wood board
    return tmp_set


def init_qrs_id_dict_a4(row_count=5, col_count=7, arcuo_size=0.0185, boarder_size=0.0015):
    # TODO: this func assumes that we recieve a set of arcuo code
    # note that arcuo index start from 1
    tmp_dict = {}

    for i in range(0, row_count):
        for j in range(0, col_count):
            tmp_x, tmp_y = i * (arcuo_size + boarder_size), -j * (arcuo_size + boarder_size)
            index = i + j * row_count
            tmp_dict[str(index)] = [tmp_x, tmp_y]

    return tmp_dict


qrs_id_dict_wood = init_qrs_id_dict_wood()
qrs_id_dict_a4 = init_qrs_id_dict_a4()


def get_paras_fromapi(K=None, dict_type="5X5", K_mode = mode_dict['EXIF_mode']):
    """
    Fetch the camera intrinsic parameters, ArUco dictionary and parameters from an API or predefined settings.

    :return: camera_matrix, dist_coeffs, aruco_dict, aruco_params
    """
    # For this example, we'll use predefined camera parameters.
    # In a real-world scenario, these could be fetched from an API or some calibration data.
    dist_coeffs = default_chessboard_coeffs
    if K is None:  # use a default of xiao mi
        if K_mode == mode_dict['EXIF_mode']:
            K = default_camera_EXIF_K
            dist_coeffs = idol_coeffs
        elif K_mode == mode_dict['chessboard_mode']:
            K = default_chessboard_K
            dist_coeffs = default_chessboard_coeffs  # Assuming no lens distortion
        elif K_mode == mode_dict['dynamic_camera_mode']:
            K = dynamic_camera_K
            dist_coeffs = dynamic_camera_chessboard_coeffs
        elif K_mode == mode_dict['yuanmu_2320_mode']:
            K = dynaimic_phone_K_yuanmu_2320
            dist_coeffs = idol_coeffs
        elif K_mode == mode_dict['yuanmu_1920_modes']:
            K = static_phone_K_yuanmu_1920
            dist_coeffs = idol_coeffs
        elif K_mode == mode_dict['yuanmu_1920_moded']:
            K = dynaimic_phone_K_yuanmu_1920
            dist_coeffs = idol_coeffs
        else:
            print("no such mode! please make sure about it!")
            exit(0)
    camera_matrix = np.array(K)

    # Define ArUco dictionary and parameters
    if dict_type == "5X5":
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    elif dict_type == "6X6":
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    else:
        print("invalid dict type ! Exiting")
        exit(-1)

    aruco_params = cv2.aruco.DetectorParameters()
    return camera_matrix, dist_coeffs, aruco_dict, aruco_params


def estimate_pose_single_marker(corner, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    nada, rvec, tvec = cv2.solvePnP(marker_points, corner, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    trash.append(nada)
    return rvec, tvec, trash


def re_estimate_pose(id__, rvec, tvec, arcuo_mode="a4_mode", id=0):
    """ this cal a new r_ mat, scr it if bug occurs"""
    """the target 3x3 qr code has its own position to original point"""
    rmat, _ = cv2.Rodrigues(rvec)  # cal rotation mat
    if arcuo_mode == "a4_mode":
        tmp_qr2world = np.array([[qrs_id_dict_a4[str(id__)][0]], [qrs_id_dict_a4[str(id__)][1]], [0.]])
    elif arcuo_mode == "wood_mode":
        tmp_qr2world = np.array([[qrs_id_dict_wood[str(id__)][0]], [qrs_id_dict_wood[str(id__)][1]], [0.]])
    elif arcuo_mode == "single_mode":
        tmp_qr2world = np.zeros([3, 1])
    else:
        print("ERR! mode is not set correctly!")
        exit(-1)
    delta_tvec = -np.dot(rmat, tmp_qr2world)
    tvec = tvec + delta_tvec
    return rvec, tvec


def calc_extrinsic_mat_from__avg_vecs(rvecs, tvecs, threshold=20.0, threshold2=20.0, muti_qr_mode="multi_mode"):
    """this function gets a set of rvecs, tvecs generated from arcuo, then cal is avg, and remove outliers"""
    """at last, cal a ex mat (c2w mat) from the avg tvecs & rvecs and return it """

    def group_vectors(rvecs, tvecs, threshold2):
        # 合并rvecs和tvecs为一个6维向量
        combined = np.hstack((rvecs, tvecs))
        if debug_mode:
            print("-------------|original group hold vecs R|-----------")
            print(rvecs)
            print("-------------|original group hold vecs T|-----------")
            print(tvecs)
        # 根据6D向量的模进行排序
        sorted_indices = np.argsort(np.linalg.norm(combined, axis=1))
        sorted_combined = combined[sorted_indices]

        groups = []
        current_group = [sorted_combined[0]]

        for i in range(1, len(sorted_combined)):
            prev_vec = sorted_combined[i - 1]
            curr_vec = sorted_combined[i]

            diff = np.linalg.norm(curr_vec - prev_vec) / np.linalg.norm(prev_vec) * 100
            if diff < threshold2:
                current_group.append(curr_vec)
            else:
                groups.append(current_group)
                current_group = [curr_vec]

        groups.append(current_group)

        # 选取最大的组
        largest_group = max(groups, key=len)
        return np.array(largest_group)[:, :3], np.array(largest_group)[:, 3:]


    print(len(rvecs))
    if muti_qr_mode == "single_mode":
        rvecs, tvecs = rvecs, tvecs
    elif muti_qr_mode is not None:
        rvecs = np.array(rvecs).squeeze()
        tvecs = np.array(tvecs).squeeze()
        rvecs, tvecs = group_vectors(rvecs, tvecs, threshold2)
    else:
        print("err for not specified qr mode! ")
        exit(-1)

    if debug_mode:
        print("-------------|largest group hold vecs R|-----------")
        print(rvecs)
        print("-------------|largest group hold vecs T|-----------")
        print(tvecs)

    if muti_qr_mode == "single_mode":
        rvec = rvecs[0]
        tvec = tvecs[0]
        rvec = rvec.reshape(-1, 1)
        tvec = tvec.reshape(-1, 1)
    elif muti_qr_mode is not None:
        while True:
            # 计算平均值
            rvecs_mean = np.mean(rvecs, axis=0)
            tvecs_mean = np.mean(tvecs, axis=0)
            # 计算rvecs和tvecs与其对应平均值的差值的比例
            outliers = []
            holder_r = []
            holder_t = []
            for i in range(0, rvecs.shape[0]):
                rvecs_diff_ratio = np.linalg.norm(rvecs[i] - rvecs_mean) / np.linalg.norm(rvecs_mean) * 100
                tvecs_diff_ratio = np.linalg.norm(tvecs[i] - tvecs_mean) / np.linalg.norm(tvecs_mean) * 100
                # 找出超出阈值的项
                if rvecs_diff_ratio > threshold or tvecs_diff_ratio > threshold:
                    outliers.append(i)
                    if debug_mode:
                        print("Removing outliers: ")
                        print("Rvecs: \n", rvecs[outliers])
                        print("Tvecs: \n", tvecs[outliers])
                else:
                    holder_r.append(rvecs[i])
                    holder_t.append(tvecs[i])

            # 如果没有超出阈值的数据，就跳出循环
            if not np.any(outliers):
                break
            # 打印并移除超出阈值的数据

            rvecs = np.array(holder_r)
            tvecs = np.array(holder_t)
        rvec = np.mean(rvecs, axis=0)
        tvec = np.mean(tvecs, axis=0)
        rvec = rvec.reshape(-1, 1)
        tvec = tvec.reshape(-1, 1)
    else:
        print("err for not specified qr mode! ")
        exit(-1)
    if debug_mode:
        print("-------------|final hold vecs R|-----------")
        print(rvecs)
        print("-------------|final hold vecs T|-----------")
        print(tvecs)
    # generate world to camera mat
    rmat, _ = cv2.Rodrigues(rvec)
    transform_matrix = np.zeros((4, 4))
    transform_matrix[0:3, 0:3] = rmat
    transform_matrix[0:3, [3]] = tvec
    transform_matrix[3, 3] = 1.0  # this is w2c
    c2w_mat = np.linalg.inv(transform_matrix)
    if debug_mode:
        print("-------------|calc extrinsic mat c2w|-----------")
        print(c2w_mat)

    return transform_matrix, c2w_mat


def detect_aruco_and_estimate_pose(image_path, marker_size, K, require_debug=False, muti_qr_mode="a4_mode", dict_type=None, K_mode="dynamic_camera_mode", only_id=-1):
    """
    Detect ArUco markers and estimate pose.
    :param image_path: Path to the image containing ArUco markers.
    :return: List of detected marker corners and their IDs, and rotation and translation vectors for each marker.

    Args:
        muti_qr_mode:
        require_debug: if use debug as output
        muti_qr_mode: the type that qr code arranges, current is single, 3x3, A4 (5X7) THREE modes
    """
    if dict_type is None:
        dict_type = "5X5"
        #use 5x5 as for debug
    debug_mode = require_debug
    camera_matrix, dist_coeffs, aruco_dict, aruco_params = get_paras_fromapi(K, dict_type, K_mode=K_mode)
    if camera_matrix is None:
        print("must contain camera_matrix value! ")
        exit(-1)
    if debug_mode:
        print("Running with the  calc K_mode as : " + K_mode)
    # Load the image
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Detect ArUco markers
    corners, ids, rejectedImgPoints = detector.detectMarkers(image)

    c2w_mat, w2c_mat = None, None
    rvecs = []
    tvecs_ = []
    if ids is not None:
        if show_res_img:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

        for i, corner__ in enumerate(corners):
            # Draw detected markers
            id__ = int(ids[i][0])  # actually rvec equals rvec_ as all arcuo codes lie in the XOY plane
            rvec, tvec, _objPoints = estimate_pose_single_marker(corner__, marker_size, camera_matrix, dist_coeffs)
            _rvec, _tvec = re_estimate_pose(id__, rvec, tvec, muti_qr_mode)
            rvecs.append(_rvec)
            tvecs_.append(_tvec)
            if debug_mode:
                print(f"Rotation vector for QR code :\n", rvec)
                # print("re-detect Translation vector :\n", _tvec)
                # print(f"Translation vector for QR code :\n", tvec)
                # print(f"c2w :\n", c2w_mat)
                # print("detected id of qr code " + str(id__))
                print("------------------------------------")
            if show_res_img:
                cv2.namedWindow("Aruco Detection", 0)
                img_ = cv2.imread(image_path)
                font_scale = 3  # 设置字体大小
                font_color = (0, 0, 255)  # 设置字体颜色
                font_thickness = 5  # 设置字体厚度
                x, y = int(corner__[0][0][0]), int(corner__[0][0][1])
                cv2.putText(image, str(id__), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color,
                            font_thickness)
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
                # cv2.resizeWindow("Aruco Detection", 1080, 720)
                cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                cv2.imshow("Aruco Detection", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if muti_qr_mode == "single_mode":
                break # only process this mode for once
        w2c_mat, c2w_mat = calc_extrinsic_mat_from__avg_vecs(rvecs, tvecs_, muti_qr_mode=muti_qr_mode)

    else:
        print("detect failed in " + image_path)

    return c2w_mat


if __name__ == '__main__':
    filename = 'C:/Users/guanl/Desktop/GenshinNerf/tmp/0004.jpg'
    # half_marker_size_x = default_marker_size / 2.0
    # half_marker_size_y = half_marker_size_x
    # for key, vec in qrs_id_pos_dict.items():
    #     qrs_id_pos_dict[key] = [vec[0] + half_marker_size_x, vec[1] - half_marker_size_y]
    show_res_img = True
    debug_mode = True
    # c2w = detect_aruco_and_estimate_pose(filename, marker_size=0.022, K=None, K_mode="chessboard_mode")

    # this code is a example for single 6x6 qr detection:
    # the marker size is 2.8 cm, K (intrinsic mode is dynaimic_phone_K_yuanmu_1920)
    filename = 'C:/Users/guanl/Desktop/GenshinNerf/t22/soap/soap_dynamic1/preprocessed/image/021.png'
    filename = 'C:/Users/guanli.hou/Desktop/real_world/dynamic/public_data/tree_slide/tree_qr_static_1qr.png'
    # filename = '/Users/houguanli/Desktop/real_world/object/tree/qr.jpg'

    c2w = detect_aruco_and_estimate_pose(filename, marker_size= 0.03, K=None, dict_type="6X6", require_debug=False, muti_qr_mode="single_mode", K_mode="yuanmu_1920_moded")

    print((c2w))
