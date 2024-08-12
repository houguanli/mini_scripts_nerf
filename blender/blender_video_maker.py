from PIL import Image
import imageio
import os

def create_dynamic_video(image_folder, frame_count, frame_duration=0.02, output_format="gif", start_pre_fix=0):
    """
    创建动态视频（GIF或MP4）。

    Args:
        image_folder (str): 包含图像的文件夹路径。
        frame_count (int): 图像的帧数。
        frame_duration (float): 每帧之间的时间间隔（秒）。
        output_format (str): 输出格式，可以是 'gif' 或 'mp4'。
    """
    # 获取文件夹下所有文件名，按文件名排序

    # 创建一个图像列表，用于存储读取的图像
    images = []

    # 读取图像并添加到图像列表
    for i in range(start_pre_fix, frame_count + start_pre_fix):
        image_file = f"{i:03}.png"  # 假设图像格式为PNG
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path)
        images.append(img)


    # 生成输出文件路径
    if output_format == "gif":
        output_file = os.path.join(image_folder, "result.gif")
    elif output_format == "mp4":
        output_file = os.path.join(image_folder, "1.mp4")
    else:
        raise ValueError("Unsupported output format. Please use 'gif' or 'mp4' as the output format.")

    # 保存为动态GIF或MP4
    if output_format == "gif":
        imageio.mimsave(output_file, images, duration=frame_duration)  # 设置每帧之间的时间间隔（秒）
    elif output_format == "mp4":
        imageio.mimsave(output_file, images, format="mp4")


if __name__ == '__main__':
    # 例如，生成一个包含 30 帧的动态 GIF 图像
    output_path = "C:/Users/GUANLI.HOU/Desktop/real_world/dynamic/exp/bunny_bounce_long/IoU_calc/pga_wb"

    create_dynamic_video(output_path, frame_count=35, frame_duration=0.08, start_pre_fix=0)
