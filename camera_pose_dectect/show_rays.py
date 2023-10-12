import open3d as o3d
import numpy as np

def show_rays(rays_o, rays_d, scale_factor=10):
    # 创建一个空的LineSet对象
    line_set = o3d.geometry.LineSet()

    # 将起点和方向向量转换为numpy数组（如果它们不是）
    rays_o = np.asarray(rays_o)
    rays_d = np.asarray(rays_d)

    # 计算终点，我们通过一个比例因子来扩大方向向量，使得线段看起来像射线
    rays_e = rays_o + scale_factor * rays_d

    # 设置线段的起点和终点
    model1 = o3d.io.read_triangle_mesh("D:/gitwork/NeuS/exp/bird/wmask/meshes/00300000.ply")
    line_set.points = o3d.utility.Vector3dVector(np.vstack((rays_o, rays_e)))
    line_set.lines = o3d.utility.Vector2iVector([[i, i + len(rays_o)] for i in range(len(rays_o))])

    # 显示线段
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([line_set, axis, model1])



if __name__ == '__main__':
    images_path_pattern = 'D:/gitwork/NeuS/dynamic_test/test_render.txt'
    with open(images_path_pattern, 'r') as f:
        lines = f.readlines()
    count = 0
    rays_o, rays_d = [], []

    for arr in lines:
        # print(arr)
        data = arr.strip().replace('[', '').replace(']', '').split(',')

        if count == 0:
            count = 1
            rays_o.append([float(i) for i in data])
        else:
            count = 0
            rays_d.append([float(i) for i in data])
    # print(str(rays_o))

    rays_o, rays_d = np.array([[2.9335, -1.7378, -4.7746]]), np.array([[-0.5976,  0.1954,  0.7776]])
    show_rays(rays_o, rays_d)
    # finish tmp_saving