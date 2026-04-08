import trimesh
import numpy as np
import pyvista as pv
import tkinter as tk
from tkinter import filedialog
def load_mesh(file_path):
    """加载 OBJ 或 PLY 格式的 mesh 文件"""
    return trimesh.load(file_path, force='mesh')

def compute_obb(mesh):
    """计算 mesh 的 Oriented Bounding Box (OBB)"""
    obb = mesh.bounding_box_oriented
    return obb

def visualize_mesh_and_obb(mesh, obb=None):
    """使用 PyVista 可视化 Mesh 和 OBB"""
    plotter = pv.Plotter()

    # PyVista 处理 trimesh 转换
    mesh_pv = pv.PolyData(mesh.vertices, np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces)).astype(np.int32))
    # Mesh 可视化： 灰色
    plotter.add_mesh(mesh_pv, color="white", show_edges=False)

    # OBB 可视化
    if obb != None:
        obb_pv = pv.PolyData(obb.vertices, np.hstack((np.full((len(obb.faces), 1), 3), obb.faces)).astype(np.int32))
        plotter.add_mesh(obb_pv, color="red", opacity=0.3, style="wireframe")

    plotter.show()

if __name__ == "__main__":
    # file_path = "./assets/bunny.ply"  # 替换为你的 OBJ/PLY 文件路径
    # load file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="请选择一个文件", 
        filetypes=[("PLY 文件", "*.ply"), ("OBJ 文件", "*.obj")]
    )
    mesh = load_mesh(file_path)

    obb = compute_obb(mesh)
    # obb = None
    visualize_mesh_and_obb(mesh, obb)
    # export obb
    save_path = filedialog.asksaveasfilename(
        title="保存 OBB 文件",
        defaultextension=".ply",
        filetypes=[("PLY 文件", "*.ply"), ("OBJ 文件", "*.obj")]
    )
    obb.export(save_path)
    print(f"OBB saved to {save_path}")