import cv2
import numpy as np

debug_mode = False


def get_paras_fromapi(K=None, dict_type="5X5"):
    """
    Fetch the camera intrinsic parameters, ArUco dictionary and parameters from an API or predefined settings.

    :return: camera_matrix, dist_coeffs, aruco_dict, aruco_params
    """
    # For this example, we'll use predefined camera parameters.
    # In a real-world scenario, these could be fetched from an API or some calibration data.
    if K is None:  # use a default
        K = [[3.21111111e+03, 0.00000000e+00, 2.31200000e+03],
         [0.00000000e+00, 3.61666667e+03, 1.73600000e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    camera_matrix = np.array(K)
    dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion

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


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
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
    rvecs = []
    tvecs = []
    i = 0
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


def detect_aruco_and_estimate_pose(image_path, marker_size, K = None):
    """
    Detect ArUco markers and estimate pose.

    :param image_path: Path to the image containing ArUco markers.
    :return: List of detected marker corners and their IDs, and rotation and translation vectors for each marker.
    """
    camera_matrix, dist_coeffs, aruco_dict, aruco_params = get_paras_fromapi(K)

    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    # Detect ArUco markers
    corners, ids, detectedImgPoints = detector.detectMarkers(gray)
    c2w_mat = None
    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        rvecs, tvecs, _objPoints = my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        rvec = rvecs[0]
        tvec = tvecs[0]

        # generate world to camera mat
        rmat, _ = cv2.Rodrigues(rvec)
        transform_matrix = np.zeros((4, 4))
        transform_matrix[0:3, 0:3] = rmat
        transform_matrix[0:3, [3]] = tvec
        transform_matrix[3, 3] = 1.0  # this is w2c
        c2w_mat = np.linalg.inv(transform_matrix)
        if debug_mode:
            print(f"Rotation vector for QR code :", rvec)
            print(f"Translation vector for QR code :", tvec)
            print(f"w2c :\n", transform_matrix)
            print(f"c2w :\n", c2w_mat)
            print("----")
            # cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)
            cv2.namedWindow("Aruco Detection", 0)
            cv2.resizeWindow("Aruco Detection", 1080, 720)
            cv2.imshow("Aruco Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("detect failed in " + image_path)
    return c2w_mat


if __name__ == '__main__':
    filename = 'C:/Users/GUANL/Desktop/GenshinNerf/t10/0001.jpg'

    c2w = detect_aruco_and_estimate_pose(filename, marker_size=0.031, K = None)
    print(c2w)
