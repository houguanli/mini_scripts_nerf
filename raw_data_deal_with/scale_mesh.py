def scale_obj_mesh(scale, path):
    """
    Scales a mesh in an OBJ file based on a given factor.

    Args:
        scale (float): Scale factor.
        path (str): Path to the OBJ mesh.

    Returns:
        None
    """
    # Read the obj file
    with open(path, 'r') as f:
        lines = f.readlines()

    # Process each line and scale the vertex positions
    new_lines = []
    for line in lines:
        if line.startswith('v '):  # vertex position
            parts = line.split()
            # Scale x, y, and z coordinates
            x, y, z = float(parts[1])*scale, float(parts[2])*scale, float(parts[3])*scale
            new_lines.append(f"v {x} {y} {z}\n")
        else:
            new_lines.append(line)

    # Write the modified lines back to the obj file
    with open(path, 'w') as f:
        f.writelines(new_lines)

# Example usage:
# scale_obj_mesh(2.0, "path_to_your_mesh.obj")


if __name__ == '__main__':
    model_dir = "C:/Users/guanl/Desktop/GenshinNerf/t12/models/"
    for i in range(1, 51):
        time_index_str = f"{i:06d}"
        filename = f"t_0_{time_index_str}"
        path = model_dir + filename + ".obj"
        scale_obj_mesh(0.1, path)

