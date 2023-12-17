import os
from PIL import Image

def merge_images(folder_path, width, height, w_c, h_c):
    # 获取文件夹中的所有图片文件
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg') or file.endswith('.png')]

    # 如果文件夹中存在图片文件
    if len(image_files) > 0:
        # 计算单个图片的宽度和高度
        original_image = Image.open(os.path.join(folder_path, image_files[0]))
        original_width, original_height = original_image.size
        image_width = width // w_c
        image_height = height // h_c

        # 创建一个空白的大图，用于拼接图片
        total_width = image_width * w_c
        total_height = image_height * h_c
        result_image = Image.new('RGB', (total_width, total_height))

        # 逐个读取图片并拼接到大图上
        for idx, image_file in enumerate(image_files):
            row_idx = idx // w_c
            col_idx = idx % w_c
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            resized_image = image.resize((image_width, image_height))
            result_image.paste(resized_image, (col_idx * image_width, row_idx * image_height))

        # 调整最终的大图尺寸
        result_image = result_image.resize((width, height))

        # 保存拼接后的大图
        result_image.save( folder_path + '/result_image.png')
        print("拼接完成，大图已保存。")
    else:
        print("文件夹中没有图片文件。")

def convert_black_to_transparent(folder_path):
    # 获取文件夹中的所有 PNG 图片文件
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]

    # 如果文件夹中存在 PNG 图片文件
    if len(image_files) > 0:
        # 逐个读取图片并将黑色像素转为透明像素
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            image = image.convert('RGBA')
            pixel_data = image.load()
            width, height = image.size
            for y in range(height):
                for x in range(width):
                    r, g, b, a = pixel_data[x, y]
                    if (r, g, b) == (0, 0, 0):
                        pixel_data[x, y] = (0, 0, 0, 0)

            # 覆盖原有的图片
            image.save(image_path)
        print("处理完成，图片已保存。")
    else:
        print("文件夹中没有 PNG 图片文件。")


def overlap_images(folder_path, opacity, end_opacity=1):
    # 获取文件夹中的所有 PNG 图片文件
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]

    # 如果文件夹中存在 PNG 图片文件
    if len(image_files) > 0:
        # 逐个读取图片并重叠在同一张图片上
        first_image = Image.open(os.path.join(folder_path, image_files[0]))
        width, height = first_image.size
        result_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        cnt, delta = 0, (end_opacity - opacity) / len(image_files)
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            image = image.convert('RGBA')
            alpha = int((opacity + cnt * delta) * 255)
            alpha_layer = Image.new('RGBA', (width, height), (0, 0, 0, alpha))
            # image[:, 3] = alpha_layer[:, 3]

            for y in range(height):
                for x in range(width):
                    # r1, g1, b1, a1 = result_image.getpixel((x, y))
                    r2, g2, b2, a2 = image.getpixel((x, y))
                    if r2 != 0 or g2 != 0:
                        result_image.putpixel((x, y), (r2, g2, b2, alpha))
            cnt = cnt + 1
            # result_image = Image.alpha_composite(result_image, alpha_layer)

        # 保存重叠后的图片
        result_image.save( folder_path + '/result_image.png')

        print("处理完成，图片已保存。")
    else:
        print("文件夹中没有 PNG 图片文件。")



if __name__ == "__main__":
    output_path = "D:/paper_writing/PA-Neus/motion_gt"
    # 替换为你的文件夹路径
    # directory_path = 'D:/gitwork/neus_original/public_data/rws_obj5/mask'  # 替换为你的文件夹路径

    # merge_images(output_path, width=1920, height=1080, w_c=3, h_c=2)
    convert_black_to_transparent(output_path)
    overlap_images(output_path, opacity=0.5, end_opacity=1)
