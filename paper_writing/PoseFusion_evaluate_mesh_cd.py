import open3d as o3d
import numpy as np
import os
import sys
import pandas as pd

# we assume the folder structure is as follows:
# base_folder
# ├── case_name_1
# │   ├── ground_truth.ply
# │   ├── pose1.ply
# │   ├── pose2.ply
# │   ├── ...
# |   └── fusion.ply
# ├── case_name_2
# │   ├── ground_truth.ply
# │   ├── pose1.ply
# │   ├── pose2.ply
# │   ├── ...
# |   └── fusion.ply
# └── ...
# in out experiment, we have three comparing meshes for each case.
# the ground truth mesh is in ground_truth.ply
# the other meshes are in pose1.ply, pose2.ply, etc.
# the output csv file will be in base_folder/cd_results.csv
# the csv file will have the following columns:
# case_name, pose1_cd, pose2_cd, pose3_cd, fusion_cd
# the cd is the chamfer distance between the ground truth mesh and the comparing mesh
# the chamfer distance is defined as the average distance from each point in one point cloud to the nearest point in the other point cloud
# the chamfer distance is defined as:
# cd(P, Q) = 1/|P| * sum_{p in P} min_{q in Q} ||p - q||^2 + 1/|Q| * sum_{q in Q} min_{p in P} ||p - q||^2
# the chamfer distance is a measure of similarity between two point clouds
# the smaller the chamfer distance, the more similar the two point clouds are
# the chamfer distance is a measure of how well the two point clouds match

# compute the chamfer distance between two meshes
def compute_cd(ground_truth_mesh, compare_mesh):
    # Compute the point clouds from the meshes
    pcd_gt = ground_truth_mesh.sample_points_uniformly(number_of_points=10000)
    pcd_compare = compare_mesh.sample_points_uniformly(number_of_points=10000)

    # Compute the distances
    dists_gt = pcd_gt.compute_point_cloud_distance(pcd_compare)
    dists_compare = pcd_compare.compute_point_cloud_distance(pcd_gt)

    # Compute the chamfer distance
    cd = np.mean(dists_gt) + np.mean(dists_compare)
    return cd


if __name__ == "__main__":
    if sys.argv[1] == "-h":
        print("Usage: python evaluate_mesh_cd.py <base_folder>")
        exit(0)
    base_folder = sys.argv[1]
    if not os.path.exists(base_folder):
        print(f"Base folder {base_folder} does not exist.")
        exit(1)
    
    # list the folders in the base folder
    case_name_list = os.listdir(base_folder)
    case_name_list = [case_name for case_name in case_name_list if os.path.isdir(os.path.join(base_folder, case_name))]

    output_csv = os.path.join(base_folder, "cd_results.csv")

    # create a dataframe to store the results
    results_df = pd.DataFrame(columns=["case_name", "pose1_cd", "pose2_cd", "pose3_cd", "fusion_cd"])
    # iterate over the folders
    for case_name in case_name_list:
        # load the ground truth mesh
        ground_truth_mesh = o3d.io.read_triangle_mesh(os.path.join(base_folder, case_name, "ground_truth.ply"))
        # load the comparing meshes
        pose1_mesh = o3d.io.read_triangle_mesh(os.path.join(base_folder, case_name, "pose1.ply"))
        pose2_mesh = o3d.io.read_triangle_mesh(os.path.join(base_folder, case_name, "pose2.ply"))
        pose3_mesh = o3d.io.read_triangle_mesh(os.path.join(base_folder, case_name, "pose3.ply"))
        fusion_mesh= o3d.io.read_triangle_mesh(os.path.join(base_folder, case_name, "fusion.ply"))

        # compute the chamfer distance
        pose1_cd = compute_cd(ground_truth_mesh, pose1_mesh)
        pose2_cd = compute_cd(ground_truth_mesh, pose2_mesh)
        pose3_cd = compute_cd(ground_truth_mesh, pose3_mesh)
        fusion_cd = compute_cd(ground_truth_mesh, fusion_mesh)
        print("fusion_cd", pose3_cd )
        # add the results to the dataframe
        results_df = results_df.append({"case_name": case_name, "pose1_cd": pose1_cd, "pose2_cd": pose2_cd, "pose3_cd": pose3_cd, "fusion_cd": fusion_cd}, ignore_index=True)
    
    # save the results to a csv file
    results_df.to_csv(output_csv, index=False)

    # Export as LaTeX table
    latex_table = results_df.to_latex(index=False, float_format="%.4f")
    latex_table_file = os.path.join(base_folder, "cd_results.tex")
    with open(latex_table_file, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {latex_table_file}")
    
    
    print(f"Results saved to {output_csv}")
    print("Chamfer distance computation completed.")
    print("All done.")