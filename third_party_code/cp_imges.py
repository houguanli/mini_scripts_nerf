import random
import shutil
from PIL import Image
import os

def copy_random_images(new_folder_path="/path/to/new/folder", start_folder=4, end_folder=303):
    # 遍历文件夹
    base_folder_path = 'D:/gitwork/revisit_CIL/data/omnibenchmark/train/'

    for i in range(start_folder, end_folder + 1):
        folder_path = base_folder_path + str(i)  # 转换为字符串，因为文件夹名称是字符串格式

        # 获取文件夹中所有.jpg文件的列表
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

        # 如果文件夹中有图片
        if image_files:
            # 随机选择一张图片
            chosen_image = random.choice(image_files)
            chosen_image_path = os.path.join(folder_path, chosen_image)
            print("move " + chosen_image_path + " to " + new_folder_path + chosen_image )
            # 将选中的图片复制到新的文件夹中
            shutil.copy(chosen_image_path, new_folder_path + chosen_image)


def create_and_resize_image(folder_path, output_path, image_size=(200, 200), matrix_size=(15, 15),
                            final_size=(1200, 1200), gap=100):
    # 读取文件夹中的所有图片，并将每张图片缩放到指定大小
    images = [Image.open(os.path.join(folder_path, img)).resize(image_size) for img in os.listdir(folder_path) if
              img.endswith(".jpg")]

    # 如果图片数量不足，抛出异常
    if len(images) < matrix_size[0] * matrix_size[1]:
        raise ValueError("Not enough images in the folder for the specified matrix size")

    # 创建一个新的空白图片，用于拼接
    total_width = image_size[0] * matrix_size[0]
    total_height = image_size[1] * matrix_size[1]
    new_img = Image.new('RGB', (total_width, total_height))

    # 拼接图片
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            img = images[i * matrix_size[0] + j]
            new_img.paste(img, (j * (image_size[0] + gap), i * (image_size[1] + gap)))

    # 压缩图片到指定大小
    new_img = new_img.resize(final_size)

    # 保存图片
    new_img.save(output_path)


if __name__ == "__main__":
    # 起始和结束的文件夹名称
    start_folder = 4
    end_folder = 303

    # 新文件夹的路径
    new_folder_path = "D:/gitwork/revisit_CIL/data/example/"
    copy_random_images(new_folder_path)
    create_and_resize_image(new_folder_path, new_folder_path + "general.jpg")
