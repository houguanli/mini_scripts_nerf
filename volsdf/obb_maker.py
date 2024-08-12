import numpy as np
import open3d as o3d
class Rect: # _right_hand
    def __init__(self, with_init=False, o = None, x = None, y = None, z = None):
        if not with_init:
            self.o, self.x, self.y, self.z = np.array(3), np.array(3), np.array(3), np.array(3)
        else:
            self.o, self.x, self.y, self.z = o, x, y, z

class Mesh2SDF:
    def __init__(self, mesh_path):
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.aabb_tree = o3d.geometry.KDTreeFlann(self.mesh)
        self.obb = None

    def calculate_sdf(self, points):
        sdf_values = []
        for point in points:
            closest_point_id = self.aabb_tree.search_knn_vector_3d(point, 1)[1][0]
            closest_point = np.asarray(self.mesh.vertices)[closest_point_id]
            sdf_value = np.linalg.norm(closest_point - point)
            sdf_values.append(sdf_value)
        return np.array(sdf_values)

    def calculate_obb_from_point_cloud(self):
        pointArray = np.array(self.mesh.vertices)
        ca = np.cov(pointArray, y=None, rowvar=0, bias=1)
        v, vect = np.linalg.eig(ca)
        tvect = np.transpose(vect)
        # use the inverse of the eigenvectors as a rotation matrix and
        # rotate the points so they align with the x and y axes
        ar = np.dot(pointArray, np.linalg.inv(tvect))

        # get the minimum and maximum x and y
        mina = np.min(ar, axis=0)
        maxa = np.max(ar, axis=0)
        diff = (maxa - mina) * 0.5
        # the center is just half way between the min and max xy
        center = mina + diff

        # get the 8 corners by subtracting and adding half the bounding boxes height and width to the center
        pointShape = pointArray.shape
        if pointShape[1] == 2:
            corners = np.array([center + [-diff[0], -diff[1]],
                                center + [diff[0], -diff[1]],
                                center + [diff[0], diff[1]],
                                center + [-diff[0], diff[1]],
                                center + [-diff[0], -diff[1]]])
        if pointShape[1] == 3:
            # get the 8 corners by subtracting and adding half the bounding boxes height and width to the center
            corners = np.array([center + [-diff[0], -diff[1], -diff[2]],
                                center + [diff[0], -diff[1], -diff[2]],
                                center + [diff[0], diff[1], -diff[2]],
                                center + [-diff[0], diff[1], -diff[2]],
                                center + [-diff[0], diff[1], diff[2]],
                                center + [diff[0], diff[1], diff[2]],
                                center + [diff[0], -diff[1], diff[2]],
                                center + [-diff[0], -diff[1], diff[2]],
                                center + [-diff[0], -diff[1], -diff[2]]])

            # use the the eigenvectors as a rotation matrix and
        # rotate the corners and the centerback
        corners = np.dot(corners, tvect)
        center = np.dot(center, tvect)
        radius = diff
        if pointShape[1] == 2:
            array0, array1 = np.abs(vect[0, :]), np.abs(vect[1, :])
            index0, index1 = np.argmax(array0), np.argmax(array1)
            radius[index0], radius[index1] = diff[0], diff[1]
        if pointShape[1] == 3:
            array0, array1, array2 = np.abs(vect[0, :]), np.abs(vect[1, :]), np.abs(vect[2, :])
            index0, index1, index2 = np.argmax(array0), np.argmax(array1), np.argmax(array2)
            radius[index0], radius[index1], radius[index2] = diff[0], diff[1], diff[2]
        eigenvalue = v
        eigenvector = vect
        return corners
        # return corners, center, radius, eigenvalue, eigenvector


def judge_rect(o: np.array, x: np.array, y: np.array, z: np.array):
    zero_like = 1e-7
    try:
        ox, oy, oz = x - o, y - o, z - o
        # right hand judge ixj=k, jxk=i, kxi=j,
        oxy, oyz, ozx = np.cross(ox, oy), np.cross(oy, oz), np.cross(oz, ox)
        xx, yy, zz = np.dot(oz, oxy), np.dot(oy, ozx), np.dot(ox, oyz)
        cxx, cyy, czz =  np.cross(oz, oxy), np.cross(oy, ozx), np.cross(ox, oyz)
        cxx, cyy, czz = np.abs(np.linalg.norm(cxx)), np.abs(np.linalg.norm(cyy)), np.abs(np.linalg.norm(czz))
        # print("xx, yy, zz:0 ", xx, yy, zz)

        return xx > 0 and yy > 0 and zz > 0 and cxx < zero_like and  cyy < zero_like and czz < zero_like # all are 90 and follow right hand rule
    except:
        print("err at judging")
        return False

def generate_full_rect(o: np.array, x: np.array, y: np.array, z: np.array):
    ox, oy, oz = x - o, y - o, z - o
    pxy = ox + oy + o
    pxz = ox + oz + o
    pyz = oy + oz + o
    pxyz = ox + oy + oz + o # other for points
    return [o, x, y, z, pxy, pxz, pyz, pxyz]

def generate_full_rect_using_rect(r: Rect):
    o, x, y, z = r.o, r.x, r.y, r.z
    ox, oy, oz = x - o, y - o, z - o
    pxy = ox + oy + o
    pxz = ox + oz + o
    pyz = oy + oz + o
    pxyz = ox + oy + oz + o # other for points
    return [o, x, pxy, y, z, pxz, pxyz, pyz] # x-first return for align

def sample_cuboid_surface(vertices, num_samples):
    # 计算长方体的六个面
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]

    samples, random_uvs = [], []
    for face in faces:
        # 生成随机点在面上
        u = np.random.rand(num_samples)
        v = np.random.rand(num_samples)
        random_uvs.append([u, v])
        # 插值计算面上的坐标
        for i in range(num_samples):
            sample_point = (1 - u[i]) * (1 - v[i]) * face[0] + u[i] * (1 - v[i]) * face[1] + u[i] * v[i] * face[2] + (1 - u[i]) * v[i] * face[3]
            samples.append(sample_point)

    return np.array(samples), random_uvs

def corresponding_sample_points(rect: Rect, random_uvs):
    vertices = generate_full_rect_using_rect(rect)
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]
    us = np.array(random_uvs)[:, 0, :]
    vs = np.array(random_uvs)[:, 1, :]
    # print("Array1 shape:", u.shape)
    # print("Array2 shape:", v.shape)
    samples = []
    for index in range(0, 6):
        face = faces[index]
        u = us[index]
        v = vs[index]
        num_samples = len(u)
        for i in range(num_samples):
            sample_point = (1 - u[i]) * (1 - v[i]) * face[0] + u[i] * (1 - v[i]) * face[1] + u[i] * v[i] * face[2] + (
                        1 - u[i]) * v[i] * face[3]
            samples.append(sample_point)
    return samples

def calc_sdf_diff(sdfs1, sdfs2):
    diff = sdfs1 - sdfs2
    diff_sum = np.sum(np.abs(diff))
    return diff, diff_sum

def calc_rect_ori_from_axis(x, y, z): # calc this RT from delta_x y z axis
    # 归一化处理
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    # 构建正交矩阵
    orthogonal_matrix = np.column_stack((x, y, z))
    # rotation_matrix = np.eye(4)
    # rotation_matrix[:3, :3] = orthogonal_matrix
    return orthogonal_matrix # this is R

def calc_rect_ori_from_rect(r: Rect): # calc this RT from
    o, x, y, z = r.o, r.x, r.y, r.z
    ox, oy, oz = x - o, y - o, z - o
    return calc_rect_ori_from_axis(ox, oy, oz)

def pairwise_obb(r1: Rect, r2: Rect, r1_SDF: Mesh2SDF, r2_SDF: Mesh2SDF): # register OBB, also need a sdf calc function
    fr1, fr2 = generate_full_rect_using_rect(r1), generate_full_rect_using_rect(r2) # full rect 1
    # generate generate_enum_index for fr2 and check whether this type of index matches the  fr1 format
    test_indexes = generate_enum_index(r2)
    # i1, j1, k1, l1 = 0, 1, 3, 4 # index in fr1 after generate_enum_index
    static_samples, random_uvs = sample_cuboid_surface(fr1, num_samples=10000)
    raw_R, raw_T, raw_RT = None, None, None # raw translation calculated from OBB
    fr1_sdfs = r1_SDF.calculate_sdf(static_samples)
    fr1_T, fr1_R = r1.o, calc_rect_ori_from_rect(r1) # directly calc this from rect1
    fr1_RT = np.eye(4)
    fr1_RT[:3, :3] = fr1_R
    fr1_RT[:3, 3] = fr1_T
    fr1_RT_inv = np.linalg.inv(fr1_RT)
    record_sdf_diff, coff_index = 1e10, -1 # very large, init
    for index in range(0, 24) : # every rect has 24 types ori in total
        i2, j2, k2, l2 = test_indexes[index]
        tmp_rec2 = Rect(with_init=True, o = fr2[i2], x = fr2[j2], y=fr2[k2], z=fr2[l2])
        tmp_samples = corresponding_sample_points(rect=tmp_rec2, random_uvs=random_uvs)
        fr2_sdfs = r2_SDF.calculate_sdf(tmp_samples)
        _, diff_sum = calc_sdf_diff(fr1_sdfs, fr2_sdfs) # calc sdf diffs
        print("SDF diff sum ", diff_sum)
        if diff_sum < record_sdf_diff:
            coff_index = index
            record_sdf_diff = diff_sum # update
            # print("upd record_sdf_diff to ", record_sdf_diff)
            tmp_r2_T = tmp_rec2.o
            tmp_r2_R = calc_rect_ori_from_rect(tmp_rec2) # calc r2 t2 for absolute axis
            tmp_r2_RT = np.eye(4)
            tmp_r2_RT[:3, :3] = tmp_r2_R
            tmp_r2_RT[:3, 3] = tmp_r2_T
            rel_mat_r12r2 = np.dot(tmp_r2_RT, fr1_RT_inv)
            raw_R = rel_mat_r12r2[:3, :3]
            raw_T = rel_mat_r12r2[:3, 3]
            raw_RT = np.eye(4)
            raw_RT[:3, :3] = raw_R
            raw_RT[:3, 3] = raw_T
            # calc refer (r1 -1 * r2)
    if raw_R is None or raw_T is None:
        print("ERR at pairwise OBBs")
    else:
        print("most approximate index: ", coff_index)
    return raw_RT, record_sdf_diff


def pairwise_obb_plus(r1: Rect, r2: Rect, r1_SDF: Mesh2SDF, r2_SDF: Mesh2SDF, acc_flag=True): #
    fr1 = generate_full_rect_using_rect(r1) # full rect 1
    test_indexes = generate_enum_index(r1)
    raw_RT, record_sdf_diff = None, 1e10

    for index in range(0, 24): # in plus function, we also enum (24 ? 8) in rect 1 to find best suit obb
        if acc_flag:
            if index % 3 == 0:
                i1, j1, k1, l1 = test_indexes[index]
                tmp_rec1 =  Rect(with_init=True, o = fr1[i1], x = fr1[j1], y=fr1[k1], z=fr1[l1])
            else:
                continue
        else:
            i1, j1, k1, l1 = test_indexes[index]
            tmp_rec1 = Rect(with_init=True, o=fr1[i1], x=fr1[j1], y=fr1[k1], z=fr1[l1])
        tmp_raw_RT, tmp_diff = pairwise_obb(r1 = tmp_rec1, r2 = r2, r1_SDF= r1_SDF, r2_SDF=r2_SDF)
        if tmp_diff < record_sdf_diff:
            print("renew rec_diff to ", tmp_diff)
            record_sdf_diff = tmp_diff
            raw_RT = tmp_raw_RT

    return raw_RT, record_sdf_diff


def has_duplicate(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    if len(unique_elements) != len(arr):
        return True
    else:
        return False

def generate_enum_index(rect1: Rect):
    fr1 = generate_full_rect_using_rect(rect1)
    out_form = []
    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 8):
                for l in range(0, 8):
                    if has_duplicate(np.array([i, j, k, l])):
                        continue
                    else:
                        o, x, y, z = fr1[i], fr1[j], fr1[k], fr1[l]
                        if judge_rect(o, x, y, z):
                            # print("valid index seq: ", i, j, k, l)
                            out_form.append([i, j, k, l])
                        else:
                            continue
    return out_form

def select_rect_from_full(fr1): # fr1 is a full rect
    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 8):
                for l in range(0, 8):
                    if has_duplicate(np.array([i, j, k, l])):
                        continue
                    else:
                        o, x, y, z = fr1[i], fr1[j], fr1[k], fr1[l]
                        if judge_rect(o, x, y, z):
                            # print("valid index seq: ", i, j, k, l)
                            return Rect(with_init=True, o=o, x=x, y=y, z=z)
    print("ERR for this rec! ")
    return None

def generate_rec_index():
    return
if __name__ == '__main__':
    # o = np.array([0, 0, 0])
    # z = np.array([0, 0, 1])
    # y = np.array([0, 1, 0])
    # x = np.array([1, 0, 0])
    # r1 = Rect(with_init=True, o=o, x=x, y=y, z=z)
    # generate_enum_index(r1)
    mesh_path_rt60, mesh_path_stand = 'C:/Users/guanli.hou/Desktop/neural_rig/synthetic_data/bunny/lie.obj',  'C:/Users/guanli.hou/Desktop/neural_rig/synthetic_data/bunny/stand.obj'
    # return rt is stand to lie, stand + RT = lie
    mesh_path_rt60, mesh_path_stand = 'C:/Users/guanli.hou/Desktop/neural_rig/real_world/dragon/dragon1.obj',  'C:/Users/guanli.hou/Desktop/neural_rig/real_world/dragon/dragon2.obj'
    # dragon 1 is lying , 2 is standing
    # mesh_path_rt60, mesh_path_stand = 'C:/Users/guanli.hou/Desktop/neural_rig/real_world/xbox/front.obj', 'C:/Users/guanli.hou/Desktop/neural_rig/real_world/xbox/back.obj'

    sdf1, sdf2 = Mesh2SDF(mesh_path_stand), Mesh2SDF(mesh_path_rt60)
    full_rect1, full_rect2 = sdf1.calculate_obb_from_point_cloud(), sdf2.calculate_obb_from_point_cloud()
    rect1, rect2 = select_rect_from_full(full_rect1), select_rect_from_full(full_rect2) # simplify OBB as o x y z
    raw_RT, _ = pairwise_obb(r1 = rect1, r2 = rect2, r1_SDF = sdf1, r2_SDF=sdf2)
    # raw_RT, _ = pairwise_obb_plus(r1 = rect1, r2 = rect2, r1_SDF = sdf1, r2_SDF=sdf2, acc_flag=False)

    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(full_rect1)
    # o3d.io.write_point_cloud('C:/Users/guanli.hou/Desktop/rt_60.ply', point_cloud)
    # point_cloud.points = o3d.utility.Vector3dVector(full_rect2)
    # o3d.io.write_point_cloud('C:/Users/guanli.hou/Desktop/stand.ply', point_cloud)
    print(raw_RT)
    # print(raw_T)

    # res = judge_rect(o, x, y, z)
    # print(res)
    # res = judge_rect(o, x, y, -z)
    # print(res)