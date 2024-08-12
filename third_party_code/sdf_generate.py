import os
import numpy as np
import json
from argparse import ArgumentParser

import trimesh
import mesh_to_sdf

def parser():
    parser = ArgumentParser()
    parser.add_argument('--mesh_path', type=str, default='assets/donut.obj', help='path to the mesh file')
    parser.add_argument('--resolution', type=int, default=512, help='resolution of the SDF grid')
    parser.add_argument('--using_narrow_band', type=bool, default=True, help='setting this to True will only compute \
                         SDF values within the narrow band(bounding box) of the surface')
    parser.add_argument('--output_path', type=str, default='nerf_rigid_body/donut.bin', help='path to the output SDF file')
    parser.add_argument('--store_in_numpy',default=False, action = "store_true", help='setting this to True will store the SDF values in numpy array(npy file)')

    args = parser.parse_args()
    return args

def resterize(x_min:np.float32, x_max:np.float32, y_min:np.float32, y_max:np.float32, z_min:np.float32, z_max: np.float32, resolution:np.int32):
    x_spacing = (x_max - x_min) / resolution
    y_spacing = (y_max - y_min) / resolution
    z_spacing = (z_max - z_min) / resolution
    
    x = np.linspace(x_min + x_spacing/2, x_max - x_spacing/2, resolution)
    y = np.linspace(y_min + y_spacing/2, y_max - y_spacing/2, resolution)
    z = np.linspace(z_min + z_spacing/2, z_max - z_spacing/2, resolution)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    coordinates = np.stack((X,Y,Z), axis=-1)
    N = X.size  # number of points
    coordinates = coordinates.reshape((N, 3))

    return coordinates

def generate_sdf(mesh):
    # get the bounding box of the mesh
    mesh_box = mesh.bounding_box.bounds
    # get the center of the bounding box
    center = (mesh_box[0] + mesh_box[1]) / 2
    # get the length of the bounding box
    delta = [mesh_box[1][0] - mesh_box[0][0], mesh_box[1][1] - mesh_box[0][1], mesh_box[1][2] - mesh_box[0][2]]
    # generate grid according to the bounding box and resolution
    grid = resterize(center[0] - delta[0]/2, center[0] + delta[0]/2, center[1] - delta[1]/2, center[1] + delta[1]/2, center[2] - delta[2]/2, center[2] + delta[2]/2, args.resolution)

    # Calculate SDF values for the query points
    sdf_values = mesh_to_sdf.mesh_to_sdf(mesh, grid, surface_point_method='scan', sign_method='normal', \
        bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
    sdf_values = np.ascontiguousarray(sdf_values)

    return sdf_values, mesh_box, center

if __name__ == '__main__':
    args = parser()
    mesh = trimesh.load(args.mesh_path)
    # print(mesh.bounding_box.bounds)
    # test_point = np.array(mesh.bounding_box.bounds[0])
    # test_point = np.array([-0.140000, -0.039921, -0.139724])
    # test_point_reshaped = test_point.reshape(1, 3)

    # sdf_values = mesh_to_sdf.mesh_to_sdf(mesh, test_point_reshaped, surface_point_method='scan', sign_method='normal', \
    #     bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)
    # print(sdf_values)
    if args.using_narrow_band:
        sdf_values, mesh_box, center = generate_sdf(mesh)
        # write sdf_values, mesh_box, center and resolution to numpy file
        if args.store_in_numpy:
            np.savez(args.output_path, sdf_values=sdf_values, mesh_box=mesh_box, center=center, resolution=args.resolution)
        # # write to json file
        # json_data = {'bounding_box': mesh_box.tolist(), 'center': center.tolist(), 'resolution': args.resolution, 'sdf_file': args.output_path}
        # with open('nerf_rigid_body/donut.json', 'w') as f:
        #     json.dump(json_data, f, indent=4)
        # # create output sdf file
        # with open(args.output_path, 'wb') as f:
        #     if args.store_in_numpy:
        #         np.savetxt(args.output_path, sdf_values, encoding='utf-8')
        #     else:
        #         # write size
        #         f.write(np.array([args.resolution, args.resolution, args.resolution], dtype=np.int32).tobytes())
        #         # write sdf values
        #         for i in range(args.resolution):
        #             for j in range(args.resolution):
        #                 for k in range(args.resolution):
        #                     f.write(np.array([sdf_values[i * args.resolution * args.resolution + j * args.resolution + k]], dtype=np.float32))
        # # print first 10 sdf values
        print(sdf_values[:10])


"""
python sdf_generate.py --mesh_path C:/Users/guanl/Desktop/GenshinNerf/slip_bunny_torch_base/bunny_original.obj --output_path C:/Users/guanl/Desktop/GenshinNerf/slip_bunny_torch_base/bunny_original.bin --store_in_numpy

"""