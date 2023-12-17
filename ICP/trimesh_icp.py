import trimesh
import trimesh.registration as reg
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    m1_path = "C:/Users/GUANLI.HOU/Desktop/r1.ply"
    m2_path = "C:/Users/GUANLI.HOU/Desktop/r3.ply"
    gt_path = "C:/Users/GUANLI.HOU/Desktop/ori.ply"
    m1 = o3d.io.read_triangle_mesh(m1_path)
    m2 = o3d.io.read_triangle_mesh(m2_path)
    mo = o3d.io.read_triangle_mesh(gt_path)
    number_of_points, fact = 100000, 1  # 你想要采样的点数
    pcd1 = m1.sample_points_uniformly(number_of_points=int(number_of_points / fact))
    pcd2 = m2.sample_points_uniformly(number_of_points=number_of_points * fact)
    pcd_mo = mo.sample_points_uniformly(number_of_points=number_of_points * fact)
    p1, p2 = np.array(pcd1.points), np.array(pcd2.points)
    p1 = p1 + np.array([0.01, 0.01, 0.01])
    pts = o3d.cpu.pybind.utility.Vector3dVector(p1)
    # pcd_mo.points = pcd1.points
    pcd1.points = pts
    ones = np.ones_like(len(pcd1.points))
    mat, trans, cost = reg.icp(pcd1.points, pcd2.points, max_iterations=1000, threshold=1/1e6)
    tmp_mat = mat[:3, :]
    p1_to_2 = np.matmul(tmp_mat[None, :3, :3], p1[:, :, None]).squeeze()
    p1_to_2 = p1_to_2 + tmp_mat[:3, 3]
    pts = o3d.cpu.pybind.utility.Vector3dVector(p1_to_2)
    after_ipc = o3d.geometry.PointCloud(pts)
    print(p1_to_2)
    print(mat)
    o3d.visualization.draw_geometries([pcd2, after_ipc])


    # mesh1.show()
    # # print(trans)
    # print(cost)
