import cv2
import os
import json


def extract_frames_and_calib(video_path, jump_frames=1):
    # 读取视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    # 创建一个文件夹来保存帧
    dir_name = os.path.splitext(video_path)[0] + "/raw_jump"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # 假设内参矩阵 (只作为示例，你应该使用你的摄像头的真实内参)
    fx, fy = 1680, 1080
    cx, cy = cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2
    intrinsic_matrix = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy
    }

    frame_count = 1
    save_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % jump_frames == 0:
            frame_name = os.path.join(dir_name, f"{save_count:06}.png")
            cv2.imwrite(frame_name, frame)
            save_count += 1

        frame_count += 1

    cap.release()
    #
    # # 保存内参矩阵到JSON文件
    # with open(os.path.join(dir_name, "intrinsic_matrix.json"), "w") as json_file:
    #     json.dump(intrinsic_matrix, json_file, indent=4)


if __name__ == "__main__":

    video_path = "C:/Users/guanl/Desktop/GenshinNerf/t22/soap/qr1.mp4"
    video_path = "/Users/houguanli/Desktop/real_world/tree/dynamic/raw/tree_slop.mp4"
    extract_frames_and_calib(video_path, jump_frames=30)



# C:/Users/GUANLI.HOU/Desktop/GenshinNerf/t1.mp4