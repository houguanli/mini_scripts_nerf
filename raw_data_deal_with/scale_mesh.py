import os


def scale_deform_obj_mesh(path, scale = 1.0, deformation=None):
    """
    Scales a mesh in an OBJ file based on a given factor.
    Args:
        scale (float): Scale factor.
        path (str): Path to the OBJ mesh.

    Returns:
        None
    """
    # Read the obj file
    if deformation is None:
        deformation = [0, 0, 0]
    with open(path, 'r') as f:
        lines = f.readlines()
    add_front_v, add_front_f = False, False
    add_v_content, add_f_content = "mtllib bunny_original.mtl \no bunny_0000.025\n", "usemtl Default_OBJ.024 \n s 1\n"
    # Process each line and scale the vertex positions
    new_lines = []
    for line in lines:
        if line.startswith('v '):  # vertex position
            if add_front_v:
                add_front_v = False
                new_lines.append(add_v_content)
            parts = line.split()
            # Scale x, y, and z coordinates
            x, y, z = float(parts[1])*scale, float(parts[2])*scale, float(parts[3])*scale
            x, y, z = x + deformation[0], y + deformation[1], z + deformation[2]
            new_lines.append(f"v {x} {y} {z}\n")
        elif line.startswith('f '):
            if add_front_f:
                add_front_f = False
                new_lines.append(add_f_content)
            new_lines.append(line)
        else:
            new_lines.append(line)

    # Write the modified lines back to the obj file
    with open(path, 'w') as f:
        f.writelines(new_lines)


def scale_deform_obj_all_mesh(directory):
    scale = 1
    deformation = [-2, 0.0, 0.0]
    # deformation = None

    files = os.listdir(directory)
    meshes = [f for f in files if f.endswith('.obj')]
    meshes.sort()
    for idx, mesh in enumerate(meshes, 1):
        # 获取文件扩展名
        ext = os.path.splitext(mesh)[1]
        # 获取图片当前的完整路径和新的完整路径
        old_path = os.path.join(directory, mesh)
        scale_deform_obj_mesh(scale=scale, deformation=deformation, path=old_path)


if __name__ == '__main__':
    model_dir = "C:/Users/guanl/Desktop/GenshinNerf/reflect_bunny_torch_base/reformat"
    scale_deform_obj_all_mesh(model_dir)
    # for i in range(1, 51):
    #     time_index_str = f"{i:06d}"
    #     filename = f"t_0_{time_index_str}"
    #     path = model_dir + filename + ".obj"
    #     scale_deform_obj_mesh(scale=0.1, path=path, deformation=[-2, 0, 0])

