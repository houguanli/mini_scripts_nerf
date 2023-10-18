import os


def rename_images(directory):
    # 获取文件夹中的所有文件
    files = os.listdir(directory)

    # 过滤出.png和.jpg格式的图片
    images = [f for f in files if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.JPG')]

    # 确保图片数量不超过1000
    if len(images) > 1000:
        raise ValueError("图片数量超过1000张!")

    # 对图片进行排序，这样我们在重命名时不会遗漏任何图片
    images.sort()

    # 开始重命名
    for idx, image in enumerate(images, 1):
        # 获取文件扩展名
        ext = os.path.splitext(image)[1]
        # ext = '.png'
        # 新名称格式：0001, 0002, ...
        new_name = f"{idx:04}{ext}"
        # 获取图片当前的完整路径和新的完整路径
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)
        # 重命名图片
        os.rename(old_path, new_path)
        print(f"Renamed {image} to {new_name}")


if __name__ == "__main__":
    directory_path = 'C:/Users/guanl/Desktop/GenshinNerf/t21/compress/mask'  # 替换为你的文件夹路径
    # directory_path = 'D:/gitwork/neus_original/public_data/rws_obj5/mask'  # 替换为你的文件夹路径

    rename_images(directory_path)
