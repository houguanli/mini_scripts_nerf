"""
2023, Baixin, NTU.
Copyright 
Based on Hao Zh, NJU https://github.com/zhuhao-nju/facescape
Parametric model fitter..
"""
# render multi-view model
import cv2, json, os, trimesh
import numpy as np
import pyrender
#os.environ['PYOPENGL_PLATFORM']='egl'
from scipy.spatial.transform import Rotation

glob_debug=False
def add_lights_based_on_pose(scene, pose):
    # 获取PX的位置和方向
    position = pose[0:3, 3]
    forward_direction = -pose[0:3, 0:3]

    # 定义一个向量的长度，这将确定光源距离PX的距离
    light_distance = position[0]
    light_intensity = 1

    # 添加X轴上的方向光
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity),
              pose=np.array([[1, 0, 0, light_distance], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    # scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity),
    #           pose=np.array([[1, 0, 0, -light_distance], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    #
    # # 添加Y轴上的方向光
    # scene.add(pyrender.DirectionalLight(color=[0.0, 1.0, 0.0], intensity=light_intensity),
    #           pose=np.array([[1, 0, 0, 0], [0, 1, 0, light_distance], [0, 0, 1, 0], [0, 0, 0, 1]]))


    mat_Y = np.array(
            [[0, 1, 0, 0],
             [-1, 0, 0, 0],
             [0, 0, 1, 0.],
             [0, 0, 0, 1]])
    for i in range(0, 28):
        delt = np.array([i%3-1, (i/3)%3-1, (i/9)%3-1])
        delt *= 2
        delt_Y = np.array(mat_Y)
        delt_Y[0:3, 3] += delt
        scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1),pose=delt_Y)
    #mat_Y[0:3, 3] += position
    # scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1),
    #           pose=mat_Y)
    #
    # # 添加Z轴上的方向光
    # scene.add(pyrender.DirectionalLight(color=[0.0, 0.0, 1.0], intensity=light_intensity),
    #           pose=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, light_distance], [0, 0, 0, 1]]))
    # scene.add(pyrender.DirectionalLight(color=[0.0, 0.0, 1.0], intensity=light_intensity),
    #           pose=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -light_distance], [0, 0, 0, 1]]))


# render with gl camera
def render_glcam(model_in, # model name or trimesh
                 K = None,
                 Rt = None,
                 scale = 1.0,
                 rend_size = (512, 512),
                 light_trans = np.array([[0], [100], [0]]),
                 flat_shading = False):
    
    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()
    #pr_mesh = pyrender.Mesh.from_trimesh(mesh)

    # confirm to contain vertex_color
    if not hasattr(mesh.visual, 'vertex_colors'):
        print("Your trimesh object does not have vertex colors!")
        exit(-10086)
    else:
        #
        pr_mesh = pyrender.Mesh.from_trimesh(mesh)

    # Scene creation
    scene = pyrender.Scene()

    # Adding objects to the scene
    face_node = scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0] * scale, K[1][1] * scale
    cx, cy = K[0][2] * scale, K[1][2] * scale

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, 
                                    znear=0.1, zfar=100000)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = Rt[:3, :3].T
    cam_pose[:3, 3] = -Rt[:3, :3].T.dot(Rt[:, 3])
    scene.add(cam, pose=cam_pose)

    # Set up the light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    light_pose = cam_pose.copy()
    print(light_pose)
    light_pose[0:3, :] += light_trans
    scene.add(light, pose=light_pose)

    #set additional light :
    add_lights_based_on_pose(scene, light_pose)
    ###test r
    if glob_debug:
        pyrender.Viewer(scene, use_raymond_lighting=True)

    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)
    if flat_shading is True:
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    else:
        color, depth = r.render(scene)

    # rgb to bgr for cv2
    color = color[:, :, [2, 1, 0]]

    return depth, color


# render with cv camera
def render_cvcam(model_in, # model name or trimesh
                 K = None,
                 Rt = None,
                 scale = 1.0,
                 rend_size = (512, 512),
                 light_trans = np.array([[0], [100], [0]]),
                 flat_shading = False):
    
    if np.array(K).all() == None:
        K = np.array([[2000, 0, 256],
                      [0, 2000, 256],
                      [0, 0, 1]], dtype=np.float64)
        
    if np.array(Rt).all() == None:
        Rt = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]], dtype=np.float64)
    
    # define R to transform from cvcam to glcam
    R_cv2gl = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
    Rt_cv = R_cv2gl.dot(Rt)
    
    return render_glcam(model_in, K, Rt_cv, scale, rend_size, light_trans, flat_shading)

# render with orth camera
def render_orthcam(model_in, # model name or trimesh
                   xy_mag,
                   rend_size,
                   flat_shading=False,
                   zfar = 10000,
                   znear = 0.05):
    
    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()
    pr_mesh = pyrender.Mesh.from_trimesh(mesh)
  
    # Scene creation
    scene = pyrender.Scene()
    
    # Adding objects to the scene
    face_node = scene.add(pr_mesh)
    
    # Camera Creation
    if type(xy_mag) == float:
        cam = pyrender.OrthographicCamera(xmag = xy_mag, ymag = xy_mag, 
                                          znear=znear, zfar=zfar)
    elif type(xy_mag) == tuple:
        cam = pyrender.OrthographicCamera(xmag = xy_mag[0], ymag = xy_mag[1], 
                                          znear=znear, zfar=zfar)
    else:
        print("Error: xy_mag should be float or tuple")
        return False
        
    scene.add(cam, pose=np.eye(4))
    
    # Set up the light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    scene.add(light, pose=np.eye(4))
    
    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)
    if flat_shading is True:
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    else:
        color, depth = r.render(scene)
    
    # rgb to bgr for cv2
    color = color[:, :, [2, 1, 0]]
    
    # fix pyrender BUG of depth rendering, pyrender version: 0.1.43
    depth[depth!=0] = (zfar + znear - ((2.0 * znear * zfar) / depth[depth!=0]) ) / (zfar - znear)
    depth[depth!=0] = ( ( depth[depth!=0] + (zfar + znear) / (zfar - znear) ) * (zfar - znear) ) / 2.0
    
    return depth, color



def normalize(v):
    """
    Normalize a vector to unit length.
    """
    return v / np.linalg.norm(v)

def look_at(origin, target):
    """
    Compute the rotation matrix to make the camera look at the target from the origin.
    :param origin: The position of the camera.
    :param target: The position of the target to look at.
    :return: The rotation matrix (R) that orients the camera to look at the target.
    """
    forward = normalize(target - origin)
    right = normalize(np.cross(np.array([0, 0, 1]), forward))
    up = np.cross(forward, right)
    R = np.column_stack((right, up, -forward))
    return R

def spherical_to_cartesian(radius, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates.
    :param radius: The radius of the sphere.
    :param theta: The azimuthal angle (in radians) in the xy-plane (0 at x-axis, counterclockwise).
    :param phi: The polar angle (in radians) from the positive z-axis (0 at z-axis, up to xy-plane).
    :return: Cartesian coordinates (x, y, z).
    """
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z])

def generate_camera_trajectory_on_sphere(n_cameras, radius):
    # Step 1: Initialize an empty list to store camera poses (R, t)
    camera_trajectory = []

    # Step 2: Calculate the angular spacing between cameras
    angular_spacing = 2 * np.pi / n_cameras

    # Step 3: Generate camera positions on the sphere
    for i in range(n_cameras):
        # Calculate the azimuthal angle (theta) in radians for this camera
        theta = i * angular_spacing

        # Calculate the polar angle (phi) in radians for this camera (elevation from the z-axis)
        phi = np.pi  # For a camera trajectory around the equator of the sphere

        # Convert spherical coordinates to Cartesian coordinates to get the camera position
        camera_position = spherical_to_cartesian(radius, theta, phi)

        # Calculate the rotation matrix R based on the camera position to make it look at the object's center
        R = look_at(camera_position, np.array([0, 0, 0]))
        # Append the camera pose (R, t) to the trajectory list
        Rt = np.concatenate([R, camera_position.reshape(3,1)], axis=1)
        Rt = -np.concatenate([Rt[1:2,:],Rt[2:3,:],Rt[0:1,:]], axis=0)
        # camera_trajectory.append((R, camera_position))
        camera_trajectory.append(Rt)

    return camera_trajectory

import json
from collections import defaultdict
def save_camera_params(save_path, focal_length, res, Rt):
    camera_param = dict()
    for i in range(Rt.shape[0]):
        item = dict()
        item['focal'] = focal_length
        item['res'] = res
        item['rt'] = Rt[i].tolist()
        camera_param[i] = item
    with open(save_path, 'w') as f:
        json.dump(camera_param, f, indent=4, sort_keys=True)

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, vertex_colors=g.visual.to_color().vertex_colors())
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def merge_mesh(scene):
    mesh_list = scene.dump()
    single_mesh = mesh_list[0]
    tv = single_mesh.visual.to_color()
    import pdb; pdb.set_trace()
    for idx in range(1, len(mesh_list)):
        single_mesh = trimesh.util.concatenate(single_mesh, mesh_list[idx])
    single_mesh.export('./test.obj')

def render_img(mesh_path, K=None, Rt=None, cam_id=0, pre_rot_angles=None, n_cameras=1,save_path='./'):
    tu_base_mesh = trimesh.load(mesh_path)

    # tu_base_mesh = trimesh.exchange.obj.load_obj(mesh_path, group_material=)
    # tu_base_mesh = as_mesh(tu_base_mesh)
    # tu_base_mesh = tu_base_mesh.dump(concatenate=True)
    # tu_base_mesh = merge_mesh(tu_base_mesh)
    # extract K Rt
    if K is None:
        K = np.array([[2000, 0, 256],
                    [0, 2000, 256],
                    [0, 0, 1]], dtype=np.float64)
    if Rt is None:
        Rt = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 10]], dtype=np.float64)
    if pre_rot_angles is not None:
        rot = Rt[:3,:3]
        t = Rt[:3,3:]
        r = Rotation.from_matrix(rot)
        ori_angles = r.as_euler("zxy",degrees=True)
        delta_angles = 540.0 / n_cameras * cam_id
        if cam_id > 9 and cam_id < 20:
            pre_rot_angles = [0, 30, 0]
        elif cam_id > 19 and cam_id < 30:
            pre_rot_angles = [0, -20, 0]
        ori_angles = np.array(ori_angles) + np.array(pre_rot_angles) + np.array([0,0,delta_angles])
        r = Rotation.from_euler("zxy", ori_angles, degrees=True)
        rot = r.as_matrix()
        Rt = np.concatenate([rot, t], axis=1)

    h, w = 512, 512
    #texture_visual = tu_base_mesh.visual.to_texture()
    #tu_base_mesh.visual = texture_visual
    #=tu_base_mesh.visual.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)
    # normalize the mesh
    extents = tu_base_mesh.extents / 2.0
    # print(np.min(tu_base_mesh.vertices,axis=0), np.max(tu_base_mesh.vertices,axis=0))
    # print(extents)
    transform_matrix = np.eye(4)
    transform_matrix[:3,:3] /= np.mean(extents)
    tu_base_mesh.apply_transform(transform_matrix)
    print(tu_base_mesh.extents)
    Rt[:,3] = Rt[:,3] / np.mean(extents)

    #import pdb; pdb.set_trace()
    # Rt[:,:3] = Rt[:,:3] / np.mean(extents)
    # render texture image and depth
    rend_depth, rend_tex = render_cvcam(tu_base_mesh, K, Rt, rend_size=(h, w),
                                                flat_shading=True)
    # render color image
    _, rend_color = render_cvcam(tu_base_mesh, K, Rt, rend_size=(h, w),
                                        flat_shading=False)

    # render shade image
    #tu_base_mesh.visual.material.image = np.ones((1, 1, 3), dtype=np.uint8)*255
    #_, rend_shade = render_cvcam(tu_base_mesh, K, Rt, rend_size=(h, w),
    #                                    flat_shading=False)
    # exit(0)
    # save all
    # rend_depth_vis = rend_depth.copy()
    # rend_depth_vis[rend_depth!=0] = rend_depth_vis[rend_depth!=0] - np.min(rend_depth[rend_depth!=0])
    # rend_depth_vis = (rend_depth_vis / np.max(rend_depth_vis) * 255).astype(np.uint8)

    # save image and depth
    # os.makedirs("./demo_output/", exist_ok = True)
    # cv2.imwrite(f"./demo_output/circle/tu_tex_{cam_id}.jpg", rend_tex)
    cv2.imwrite(f"{save_path}/{cam_id}.png", rend_color)
    return (K[0,0], [K[0,2], K[1,2]], Rt)

if __name__ == '__main__':
    mesh_path = 'D:/simulation_obj/0.obj'
    save_path = 'D:/simulation_obj/vis_/0'
    n_cameras = 5
    if not os.path.exists(save_path):  # create dirs if ness
        os.makedirs(save_path)
    # camera_params_list = generate_camera_trajectory_on_sphere(n_cameras=n_cameras, radius=12.0)
    rt_list = []
    for i in range(n_cameras):
        focal, res, rt = render_img(mesh_path, K=None, Rt=None, cam_id=i,\
                    pre_rot_angles=[0,0,0],\
                    n_cameras=n_cameras, save_path=save_path)
        rt_list.append(rt[None, :,:])
    rt_array = np.concatenate(rt_list,axis=0)
    print(rt_array.shape)
    save_camera_params(save_path+'/camera.json',focal_length=focal, res=res, Rt=rt_array)

    # cv2.imwrite("./demo_output/tu_shade.jpg", rend_shade)
    # rend_depth_vis = rend_depth - np.min(rend_depth[rend_depth!=0])
    # rend_depth_vis = (rend_depth_vis / np.max(rend_depth_vis) * 255).astype(np.uint8)
    # cv2.imwrite("./demo_output/tu_depth.jpg", rend_depth_vis)
    print(f"results saved to {save_path}")