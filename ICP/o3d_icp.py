import numpy as np
import open3d as o3d


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) == 3:
            color = np.repeat(np.array(color)[np.newaxis, ...], xyz.shape[0], axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

pc0_path = "C:/Users/guanl/Desktop/00220000_rough.ply"
pc1_path = "C:/Users/guanl/Desktop/00250000.ply"

mesh0 = o3d.io.read_triangle_mesh(pc0_path)
mesh1 = o3d.io.read_triangle_mesh(pc1_path)
pc0 = np.array(mesh0.vertices)
# pc0 = pc0 + np.array([0.005, 0.005, 0.005])
pc1 = np.array(mesh1.vertices)
pcd0 = make_open3d_point_cloud(pc0)
pcd1 = make_open3d_point_cloud(pc1)

o3d.visualization.draw_geometries([pcd0, pcd1])

downsample = 0.001
pcd0 = o3d.geometry.PointCloud.voxel_down_sample(pcd0, voxel_size=downsample)
pcd1 = o3d.geometry.PointCloud.voxel_down_sample(pcd1, voxel_size=downsample)

# fit to unit cube
# o3d.visualization.draw_geometries([pcd1])

reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, downsample*2, np.eye(4),
                                                  o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                  o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
tmp_mat = reg.transformation[:3, :]
p0 = np.array(pcd0.points)

p0_to_1 = np.matmul(tmp_mat[None, :3, :3], p0[:, :, None]).squeeze()
p0_to_1 = p0_to_1 + tmp_mat[:3, 3]
pts = make_open3d_point_cloud(p0_to_1)

o3d.visualization.draw_geometries([pts, pcd1])

print(reg.transformation)
