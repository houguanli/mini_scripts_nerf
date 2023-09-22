#  this is a file written for colmap 3d point data to obj file to see whether the result is good
def points3D_to_obj(input_file, output_file):
    with open(input_file, 'r') as f, open(output_file, 'w') as out:
        for line in f:
            # Skip comments
            if line.startswith("#"):
                continue

            parts = line.split()
            x, y, z = parts[1:4]
            r, g, b = parts[4:7]

            # Write the vertex (v) with its position and color
            out.write(f"v {x} {y} {z} {r} {g} {b}\n")


if __name__ == '__main__':
    points3D_to_obj("C:/Users/guanl/Desktop/face_video/front/sparse/1/points3D.txt",
                    "C:/Users/guanl/Desktop/face_video/front/sparse/1/output.obj")
