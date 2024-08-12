import open3d as o3d
import os


def get_bounding_box_from_ply(ply_file):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(ply_file)

    # 计算包围盒
    bounding_box = pcd.get_axis_aligned_bounding_box()

    # 打印包围盒信息
    print("Bounding box min bound:", bounding_box.min_bound)
    print("Bounding box max bound:", bounding_box.max_bound)

    return bounding_box


def scale_obj_vertices(input_file, output_file, scale):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    vertices = []
    other_data = []

    for line in lines:
        if line.startswith('v '):
            vertex = line.strip().split()[1:]
            vertex = [float(coord) * scale for coord in vertex]  # 缩放顶点坐标
            vertices.append(vertex)
        else:
            other_data.append(line)

    with open(output_file, 'w') as f:
        for vertex in vertices:
            f.write(f"v {' '.join(str(coord) for coord in vertex)}\n")
        for data in other_data:
            f.write(data)


def write_string_to_obj_header(file_path, string_to_write):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with open(file_path, 'w') as file:
        inserted = False
        for line in lines:
            if not inserted and line.strip().startswith('f'):
                file.write(string_to_write + '\n')
                inserted = True
            file.write(line)


def process_folder(folder_path, string_to_write):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.obj'):
                file_path = os.path.join(root, file)
                write_string_to_obj_header(file_path, string_to_write)



# 示例使用

if __name__ == "__main__":
    # ply_file_path = "D:/gitwork/NeuS/exp/real_world_normal/womask/meshes/_.ply"  # 替换为你的PLY文件路径
    # bbox = get_bounding_box_from_ply(ply_file_path)
    # input_file = 'C:/Users/guanl/Desktop/GenshinNerf/dp_simulation/duck/duck3d.obj'
    # output_file = 'C:/Users/guanl/Desktop/GenshinNerf/dp_simulation/duck/duck3d_s.obj'
    # scale = 0.25
    # scale_obj_vertices(input_file, output_file, scale)
    # Write "Hello World!" at the beginning of each obj file in the folder "obj_files"
    # string_to_write = "\nmtllib bunny_original.mtl \no bunny_0000.025"
    string_to_write = "s 1"
    folder_path = "C:/Users/GUANLI.HOU/Desktop/to_ori"
    process_folder(folder_path, string_to_write)
