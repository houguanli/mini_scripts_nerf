import open3d as o3d

def get_bounding_box_from_ply(ply_file):
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(ply_file)

    # 计算包围盒
    bounding_box = pcd.get_axis_aligned_bounding_box()

    # 打印包围盒信息
    print("Bounding box min bound:", bounding_box.min_bound)
    print("Bounding box max bound:", bounding_box.max_bound)

    return bounding_box

# 主函数++

if __name__ == "__main__":
    ply_file_path = "D:/gitwork/NeuS/exp/real_world_normal/womask/meshes/_.ply"  # 替换为你的PLY文件路径
    bbox = get_bounding_box_from_ply(ply_file_path)
