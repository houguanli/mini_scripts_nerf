from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np

debug_mode = False

default_ptz1920_mat = np.array([[735.15009953, 0., 961.93174061],
                                [0., 733.3960477, 553.13510509],
                                [0., 0., 1.]])  # this camera K is from a pan-tilt-zoom camera
default_phone_mat = np.array([[3.55085455e+03, 0.00000000e+00, 2.23088539e+03],
                              [0.00000000e+00, 3.54865667e+03, 1.67835047e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
default_ptz_mat_1280 = np.array(
    [[490.1000663533333, 0, 641.1211604066666], [0, 488.9306984666667, 368.59007006], [0, 0, 1]])


def extract_exif(filename):
    """提取图片的EXIF信息"""
    image = Image.open(filename)
    exif_data = image._getexif()

    if not exif_data:
        return None

    labeled_exif_data = {TAGS[key]: exif_data[key] for key in exif_data.keys() if key in TAGS}
    return labeled_exif_data


def calculate_intrinsic_parameters(exif_data):
    focal_length_entry = exif_data.get('FocalLength')  # FocalLength tag
    focal_length_35mm_entry = exif_data.get('FocalLengthIn35mmFilm')  # FocalLengthIn35mmFilm tag

    # Ensure the values are extracted correctly
    if not focal_length_entry or not focal_length_35mm_entry:
        raise ValueError("Cannot extract necessary EXIF data.")

    if isinstance(focal_length_entry, tuple):
        FL_actual = float(focal_length_entry[0]) / focal_length_entry[1]
    else:
        FL_actual = float(focal_length_entry)

    FL_35mm = float(focal_length_35mm_entry)

    # Calculate sensor width
    sensor_width_mm = (FL_actual / FL_35mm) * 36.0
    if debug_mode:
        print("fixed | original sensor width : " + str(sensor_width_mm) + "mm 36.0mm")
    # Assume 3:2 aspect ratio for 35mm film
    sensor_height_mm = (2 / 3) * sensor_width_mm

    image_width = exif_data.get('ExifImageWidth')
    image_height = exif_data.get('ExifImageHeight')
    dx = sensor_width_mm / image_width
    dy = sensor_height_mm / image_height
    if debug_mode:
        print("img w&h " + str(image_width) + " " + str(image_height))
    fx = FL_actual / dx
    fy = FL_actual / dy

    # 假设fy与fx相同，成像中心是图像中心
    cx = image_width / 2.0
    cy = image_height / 2.0

    # 构建相机内参矩阵
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    return K


def get_k_from_exif(filename, with_default_mat='ptz_1920'):
    if with_default_mat == 'ptz_1920':
        return default_ptz1920_mat
    elif with_default_mat == 'phone':
        return default_phone_mat
    elif with_default_mat == 'ptz_1280':
        return default_ptz_mat_1280
    elif with_default_mat == "from_exit":
        exif_data = extract_exif(filename)
        camera_matrix = calculate_intrinsic_parameters(exif_data)
        return camera_matrix
    else:
        print("Invaild mat type!!!! with " + with_default_mat)
        exit(-1)


if __name__ == '__main__':
    filename = 'C:/Users/guanl/Desktop/GenshinNerf/t9/0001.jpg'
    exif_data = extract_exif(filename)
    if exif_data:
        camera_matrix = calculate_intrinsic_parameters(exif_data)
        if debug_mode:
            if camera_matrix is not None:
                print(camera_matrix)
            else:
                print("Cannot compute camera matrix from EXIF data.")
    else:
        print("No EXIF data found.")

# 'C:/Users/GUANLI.HOU/Desktop/GenshinNerf/t4/0.jpg'
