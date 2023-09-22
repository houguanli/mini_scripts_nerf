# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from seggpt.first_mask_maker import refine_mask, make_init_mask


# original cmd_line in seggpt, output_dir should contain mask 1_1 as default start, use make_init_mask() to generate one
# python seggpt_inference.py --input_image examples/hmbb_2.jpg --prompt_image examples/hmbb_1.jpg
# --prompt_target examples/hmbb_1_target.png --output_dir ./
def make_seggpt_bat(input_dir, output_dir, save_bat_path, frame_count=1, camera_count=1, image_type="png"):
    base_cmd = "python seggpt_inference.py "
    all_cmd_to_write = "cd D:/gitwork/seggpt/SegGPT/SegGPT_inference\nD:\nconda activate seggpt\n"  # switch to base dir
    for frame_id in range(1, frame_count + 1):
        for camera_id in range(1, camera_count + 1):
            if frame_id == 1 and camera_id == 1:
                continue  # 1_1 has its mask in target file
            file_name = str(frame_id) + "_" + str(camera_id)
            input_img_path = input_dir + "/" + file_name + "." + image_type
            # output_img_path = output_dir + "/" + file_name + "." + image_type
            prompt_image_path = input_dir + "/" + "1_1." + image_type
            prompt_img_mask_path = output_dir + "/" + "1_1." + image_type
            input_para = "--input_image " + input_img_path + " "
            prompt_para = "--prompt_image " + prompt_image_path + " "
            prompt_mask_para = "--prompt_target " + prompt_img_mask_path + " "
            output_para = "--output_dir " + output_dir + " "
            output_mask_para = "--out_mask " + output_dir + " "
            output_type_para = "--output_type .png "

            tmp_cmd = (base_cmd + input_para + prompt_para + prompt_mask_para + output_para + output_mask_para +
                       output_type_para + "\n")
            all_cmd_to_write += tmp_cmd
    with open(save_bat_path, 'w') as f:
        f.write(all_cmd_to_write)
    print(all_cmd_to_write)

    return


# this code is written for a series of pics that named from 0001 - 0x_num
# convert in bat for

def one_camera_seggpt__bat_maker(input_dir, output_dir, save_bat_path, frame_count=1, image_type="jpg"):
    base_cmd = "python seggpt_inference.py "
    all_cmd_to_write = "cd D:/gitwork/seggpt/SegGPT/SegGPT_inference\nD:\nconda activate seggpt\n"  # switch to base dir
    for frame_id in range(1, frame_count + 1):
        if frame_id == 1:
            continue  # 1_1 has its mask in target file
        file_name = f"{frame_id:04}"  # name format
        input_img_path = input_dir + "/" + file_name + "." + image_type
        # output_img_path = output_dir + "/" + file_name + "." + image_type
        prompt_image_path = input_dir + "/" + "0001." + image_type
        prompt_img_mask_path = output_dir + "/" + "mask_0001." + image_type
        input_para = "--input_image " + input_img_path + " "
        prompt_para = "--prompt_image " + prompt_image_path + " "
        prompt_mask_para = "--prompt_target " + prompt_img_mask_path + " "
        output_para = "--output_dir " + output_dir + " "
        output_mask_para = "--out_mask " + output_dir + " "
        output_type_para = "--output_type .jpg "

        tmp_cmd = base_cmd + input_para + prompt_para + prompt_mask_para + output_para + output_mask_para + output_type_para + "\n"
        all_cmd_to_write += tmp_cmd
    with open(save_bat_path, 'w') as f:
        f.write(all_cmd_to_write)
    print(all_cmd_to_write)


def one_camera_seggpt_bat_maker_multi_init_mask(input_dir, output_dir, save_bat_path, frame_count=1, image_type="jpg"):
    base_cmd = "python seggpt_inference.py "
    all_cmd_to_write = "cd D:/gitwork/seggpt/SegGPT/SegGPT_inference\nD:\n"  # switch to base dir
    for frame_id in range(1, frame_count + 1):
        if frame_id == 1 or frame_id == 42:
            continue  # 1_1 has its mask in target file
        file_name = f"{frame_id:04}"  # name format
        file_name = str(frame_id) + "_1"
        input_img_path = input_dir + "/" + file_name + "." + image_type
        # output_img_path = output_dir + "/" + file_name + "." + image_type
        prompt_image_path = input_dir + "/" + "1_1." + image_type + " " + input_dir + "/" + "42_1." + image_type
        prompt_img_mask_path = output_dir + "/" + "mask_1_1." + image_type + " " + output_dir + "/" + "mask_42_1." + image_type
        input_para = "--input_image " + input_img_path + " "
        prompt_para = "--prompt_image " + prompt_image_path + " "
        prompt_mask_para = "--prompt_target " + prompt_img_mask_path + " "
        output_para = "--output_dir " + output_dir + " "
        output_mask_para = "--out_mask " + output_dir + " "
        output_type_para = "--output_type ." + image_type

        tmp_cmd = base_cmd + input_para + prompt_para + prompt_mask_para + output_para + output_mask_para + output_type_para + "\n"
        all_cmd_to_write += tmp_cmd
    with open(save_bat_path, 'w') as f:
        f.write(all_cmd_to_write)
    print(all_cmd_to_write)


def one_camera_seggpt_multi_init_static_sense_bat_maker(input_dir, output_dir, save_bat_path, frame_count=1, image_type="jpg"):
    base_cmd = "python seggpt_inference.py "
    all_cmd_to_write = "cd D:/gitwork/seggpt/SegGPT/SegGPT_inference\nD:\n"  # switch to base dir
    for frame_id in range(1, frame_count + 1):
        if frame_id == 1 or frame_id == 35:
            continue  # 1_1 has its mask in target file
        file_name = f"{frame_id:04}"  # name format
        input_img_path = input_dir + "/" + file_name + "." + image_type
        # output_img_path = output_dir + "/" + file_name + "." + image_type
        prompt_image_path = input_dir + "/" + "0001." + image_type + " " + input_dir + "/" + "0035." + image_type
        prompt_img_mask_path = output_dir + "/" + "mask_0001." + image_type + " " + output_dir + "/" + "mask_0035." + image_type
        input_para = "--input_image " + input_img_path + " "
        prompt_para = "--prompt_image " + prompt_image_path + " "
        prompt_mask_para = "--prompt_target " + prompt_img_mask_path + " "
        output_para = "--output_dir " + output_dir + " "
        output_mask_para = "--out_mask " + output_dir + " "
        output_type_para = "--output_type ." + image_type

        tmp_cmd = base_cmd + input_para + prompt_para + prompt_mask_para + output_para + output_mask_para + output_type_para + "\n"
        all_cmd_to_write += tmp_cmd
    with open(save_bat_path, 'w') as f:
        f.write(all_cmd_to_write)
    print(all_cmd_to_write)


def refine_mask_batch_one_camera_(input_dir, frame_count=1, image_type="jpg"):
    for frame_id in range(1, frame_count + 1):
            if frame_id == 1:
                continue  # 1_1 has its mask in target file
            # file_name = "mask_" + f"{frame_id:04}"
            file_name = "mask_" + str(frame_id) + "_1"
            tmp_path = input_dir + "/" + file_name + "." + image_type
            print(tmp_path)
            refine_mask(tmp_path, convert_option=False)


def refine_mask_batch_static_sense_(input_dir, frame_count=1, image_type="jpg"):
    for frame_id in range(1, frame_count + 1):
            if frame_id == 1:
                continue  # 1_1 has its mask in target file
            file_name = "mask_" + f"{frame_id:04}"
            tmp_path = input_dir + "/" + file_name + "." + image_type
            print(tmp_path)
            refine_mask(tmp_path, convert_option=False)


def refine_mask_batch(input_dir, frame_count=1, camera_count=1, image_type="png"):
    for frame_id in range(1, frame_count + 1):
        for camera_id in range(1, camera_count + 1):
            if frame_id == 1 and camera_id == 1:
                continue  # 1_1 has its mask in target file
            file_name = "mask_" + str(frame_id) + "_" + str(camera_id)
            tmp_path = input_dir + "/" + file_name + "." + image_type
            print()
            refine_mask(tmp_path)


def test():
    image_path = "/data/r_shoot/r_shoot/1_1.png"
    input_dir = "/data/r_shoot/r_shoot"
    out_path = "//data/r_shoot/result_tmp/1_1.png"
    out_dir = "/data/r_shoot/result_tmp"
    # make_seggpt_bat(input_dir, out_dir, out_dir + "auto.bat", frame_count=10, camera_count=5)
    mask_path = "/data/r_shoot/result_tmp/mask_1_1.png"

    masks_dir = out_dir
    # refine_mask_batch(masks_dir, frame_count=10, camera_count=5)
    return


def test2():
    image_path = "D:/gitwork/NeuS/public_data/real_world_normal/mask/0001.png"
    input_dir = "D:/gitwork/NeuS/public_data/real_world_normal/image"
    out_path = "D:/gitwork/NeuS/public_data/real_world_normal/mask"
    out_dir = "D:/gitwork/NeuS/public_data/real_world_normal/mask"
    # make_init_mask()
    # one_camera_seggpt__bat_maker(input_dir, out_dir, out_dir + "/auto.bat", frame_count=17)
    mask_path = "D:/gitwork/NeuS/public_data/real_world_normal/mask"
    # refine_mask(mask_path)

    masks_dir = out_dir
    refine_mask_batch_one_camera_(masks_dir, frame_count=17)
    return


def test3():
    image_path = "D:/gitwork/NeuS/public_data/real_world_multi_qrs/mask/0001.png"
    input_dir = "D:/gitwork/NeuS/public_data/real_world_multi_qrs/image"
    out_path = "D:/gitwork/NeuS/public_data/real_world_multi_qrs/mask"
    out_dir = "D:/gitwork/NeuS/public_data/real_world_multi_qrs/mask"
    # make_init_mask()
    one_camera_seggpt_bat_maker_multi_init_mask(input_dir, out_dir, out_dir + "/auto.bat", frame_count=48)
    mask_path = "D:/gitwork/NeuS/public_data/real_world_multi_qrs/mask"
    # refine_mask(mask_path)

    masks_dir = out_dir
    refine_mask_batch_one_camera_(masks_dir, frame_count=48)
    return

def test_blender_static():
    input_dir = "D:/gitwork/NeuS/public_data/rws_object2/image"
    out_dir = "D:/gitwork/NeuS/public_data/rws_object2/mask"
    # one_camera_seggpt_multi_init_static_sense_bat_maker(input_dir, out_dir, out_dir + "/auto.bat", frame_count=38, image_type="jpg")
    masks_dir = out_dir
    refine_mask_batch_static_sense_(masks_dir, frame_count=38, image_type="jpg")
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_blender_static()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
