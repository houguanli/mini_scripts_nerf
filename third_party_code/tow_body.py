import pdb

import torch
import numpy as np
import trimesh
from pathlib import Path
import os
import math
from tqdm import trange
import json

class RigidBodySimulator(torch.nn.Module):
    def __init__(self, options):
        super(RigidBodySimulator, self).__init__()
        self.options = options
        self.substep = options['substep']
        self.frames = options['frames']
        self.dt = 1.0 / 60.0 / self.substep
        
        self.mesh = trimesh.load_mesh(str(Path(options['mesh'])))
        print('mass_center:{}'.format(self.mesh.center_mass))

        # load collision mesh
        self.voxel_resolution = 512
        # convert vertices to numpy array
        vertices = np.array(self.mesh.vertices) - self.mesh.center_mass

        self.object1_translation = []
        self.object1_quaternion = []
        self.object2_translation = []
        self.object2_quaternion = []

        self.obj1_v = []
        self.obj1_omega = []
        self.obj2_v = []
        self.obj2_omega = []

        self.object1_translation_list = []
        self.object1_quaternion_list = []
        self.object2_translation_list = []
        self.object2_quaternion_list = []

        self.collision_function = []

        self.rigid_body_sdf = np.load(str(Path(options['objects_sdf_path'])))['sdf_values']
        self.rigid_body_sdf_grad = np.array(np.gradient(self.rigid_body_sdf))
        self.bbox = np.load(str(Path(options['objects_sdf_path'])))['mesh_box']
        self.voxel_resolution = np.load(str(Path(options['objects_sdf_path'])))['resolution']
        self.rigid_body_sdf = self.rigid_body_sdf.reshape([self.voxel_resolution, self.voxel_resolution, self.voxel_resolution])
        # copy the sdf values to the torch tensor
        self.rigid_body_sdf = torch.from_numpy(self.rigid_body_sdf.astype(np.float32))
        self.rigid_body_sdf_grad = torch.from_numpy(self.rigid_body_sdf_grad.astype(np.float32))
        self.bbox = torch.from_numpy(self.bbox.astype(np.float32))
        # pdb.set_trace()
        self.mass_center = torch.tensor(self.mesh.center_mass, dtype=torch.float32)
        self.x = torch.tensor(vertices, dtype=torch.float32, requires_grad=True)
        for i in range(self.frames * self.substep):
            self.object1_translation_list.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
            self.object1_quaternion_list.append(torch.zeros(4, dtype=torch.float32, requires_grad=True))
            self.object2_translation_list.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
            self.object2_quaternion_list.append(torch.zeros(4, dtype=torch.float32, requires_grad=True))
            self.obj1_v.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
            self.obj1_omega.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
            self.obj2_v.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
            self.obj2_omega.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
        self.kn = torch.nn.Parameter(torch.tensor([options['kn']], requires_grad=True))
        self.mu = torch.nn.Parameter(torch.tensor([options['mu']], requires_grad=True))
        self.linear_damping = torch.nn.Parameter(torch.tensor([options['linear_damping']]))
        self.angular_damping = torch.nn.Parameter(torch.tensor([options['angular_damping']]))
        self.object1_init_v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
        self.object2_init_v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
        self.mass = torch.tensor([0.0], dtype=torch.float32)
        self.inv_mass = torch.tensor([0.0], dtype=torch.float32)
        self.target = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32)
        self.init()
    
    def set_obj1_init_translation(self, init_translation):
        with torch.no_grad():
            self.object1_translation_list[0] = torch.tensor(init_translation, dtype=torch.float32)
    
    def set_obj1_init_quaternion(self, init_euler_angle):
        with torch.no_grad():
            self.object1_quaternion_list[0] = torch.tensor(self.from_euler(init_euler_angle), dtype=torch.float32)
    
    def set_obj2_init_translation(self, init_translation):
        with torch.no_grad():
            self.object2_translation_list[0] = torch.tensor(init_translation, dtype=torch.float32)
    
    def set_obj2_init_quaternion(self, init_euler_angle):
        with torch.no_grad():
            self.object2_quaternion_list[0] = torch.tensor(self.from_euler(init_euler_angle), dtype=torch.float32)
    
    def set_init_v(self):
        with torch.no_grad():
            self.obj1_v[0] = self.object1_init_v
            self.obj2_v[0] = self.object2_init_v

    def add_planar_contact(self, slope_degree, init_height):
        self.c = np.cos(np.deg2rad(slope_degree))
        self.s = np.sin(np.deg2rad(slope_degree))
        self.init_height = init_height
        # self.contact_normal = torch.tensor([-self.s, self.c, 0.0], dtype=torch.float32)

    # the euler angle is in degree, we first conver it to radian
    def from_euler(self, euler_angle):
        # convert euler angle to quaternion
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        phi = math.radians(euler_angle[0] / 2)
        theta = math.radians(euler_angle[1] / 2)
        psi = math.radians(euler_angle[2] / 2)
        w = math.cos(phi) * math.cos(theta) * math.cos(psi) + math.sin(phi) * math.sin(theta) * math.sin(psi)
        x = math.sin(phi) * math.cos(theta) * math.cos(psi) - math.cos(phi) * math.sin(theta) * math.sin(psi)
        y = math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.cos(theta) * math.sin(psi)
        z = math.cos(phi) * math.cos(theta) * math.sin(psi) - math.sin(phi) * math.sin(theta) * math.cos(psi)
        return [w, x, y, z]

    def quat_mul(self, a, b):
        return torch.tensor([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                      a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
                      a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1]])
    
    def quat_mul_scalar(self, a, b):
        return torch.tensor([a[0] * b, a[1] * b, a[2] * b, a[3] * b])
    
    def quat_add(self, a, b):
        return torch.tensor([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
    
    def quat_subtraction(self, a, b):
        return torch.tensor([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])
    
    def quat_normal(self, a)->torch.int32:
        return torch.tensor([a[0] / torch.norm(a), a[1] / torch.norm(a), a[2] / torch.norm(a), a[3] / torch.norm(a)])
    
    def quat_conjugate(self, a):
        return torch.tensor([a[0], -a[1], -a[2], -a[3]])

    def quat_rotate_vector(self, q, v):
        return self.quat_mul(self.quat_mul(q, torch.tensor([0, v[0], v[1], v[2]])), self.quat_conjugate(q))[1:]
    
    def quat_to_matrix(self, q):
        q = q / torch.norm(q)
        w, x, y, z = q[0], q[1], q[2], q[3]
        return torch.tensor([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                      [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                      [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])
    
    def quat_inverse(self, q):
        return self.quat_conjugate(q) / torch.norm(q)
    
    def GetCrossMatrix(self, a):
        return torch.tensor([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])

    def init(self):
        with torch.no_grad():
            self.mass = 0.0
            self.inertia_referance = torch.zeros(3, 3, dtype=torch.float32)
            self.obj1_v[0] = self.object1_init_v
            self.obj2_v[0] = self.object2_init_v
            mass = 1.0
            for i in range(self.mesh.vertices.shape[0]):
                self.mass += mass
                r = self.x[i] - self.mass_center
                # inertia = \sum_{i=1}^{n} m_i (r_i^T r_i I - r_i r_i^T)  https://en.wikipedia.org/wiki/List_of_moments_of_inertia
                # as r_i is a col vector, r_i^T is a row vector, so r_i^T r_i is a scalar (actually is dot product)
                I_i = mass * (r.dot(r) * torch.eye(3) - torch.outer(r, r))
                self.inertia_referance += I_i
            self.inv_mass = 1.0 / self.mass
            print('inerita_referance:{}'.format(self.inertia_referance))

    def collision_detect(self, f:torch.int32, contact_normal:torch.tensor, translation:torch.tensor, quaternion:torch.tensor, v:torch.tensor, omega:torch.tensor):
        mat_R = self.quat_to_matrix(quaternion)
        xi = translation + torch.matmul(self.x, mat_R.t()) + self.mass_center[None]
        vi = v + torch.cross(omega.unsqueeze(0),  torch.matmul(self.x, mat_R.t()), dim=1)
        d = torch.einsum('bi,i->b', xi, contact_normal)
        rel_v = torch.einsum('bi,i->b', vi, contact_normal)
        contact_condition = (d < 0.0) & (rel_v < 0.0)
        # calculate how many points are in contact with the plane
        num_collision = torch.sum(contact_condition.int())
        v_out = torch.zeros(3, dtype=torch.float32)
        omega_out = torch.zeros(3, dtype=torch.float32)
        if num_collision > 0:
            contact_mask = contact_condition.float()[:, None]  # add a new axis to broadcast
            # calculate the sum of the contact points
            sum_position = torch.sum(self.x * contact_mask, dim=0)
            
            # calculate the average of the contact points
            collision_ri = sum_position / num_collision
            # print('collision_ri:{}'.format(collision_ri))
            collision_Ri = mat_R @ collision_ri
            # calculate the velocity of the contact points
            vi = v + omega.cross(collision_Ri)

            v_i_n = vi.dot(contact_normal) * contact_normal
            v_i_t = vi - v_i_n
            vn_new = -self.kn * v_i_n
            alpha = 1.0 - (self.mu * (1.0 + self.kn) * (torch.norm(v_i_n) / (torch.norm(v_i_t) + 1e-6)))
            if alpha < 0.0:
                alpha = 0.0
            vt_new = alpha  * v_i_t
            vi_new = vn_new + vt_new
            I = mat_R @ self.inertia_referance @ mat_R.t()
            collision_Rri_mat = self.GetCrossMatrix(collision_Ri)
            k = torch.tensor([[self.inv_mass, 0.0, 0.0],\
                           [0.0, self.inv_mass, 0.0],\
                           [0.0, 0.0, self.inv_mass]]) - collision_Rri_mat @ I.inverse() @ collision_Rri_mat
            J = torch.inverse(k) @ (vi_new - vi)
            v_out = v_out + J * self.inv_mass
            omega_out = omega_out + I.inverse() @ collision_Rri_mat @ J
        return v_out, omega_out

    def planar_collision_detect(self, f:torch.int32):
        contact_normal = torch.tensor([-self.s,  self.c, 0.0 ], dtype=torch.float32)
        mat_R = self.quat_to_matrix(self.quaternion[f])
        xi = self.translation[f] +  torch.matmul(self.x, mat_R.t()) + self.mass_center[None]
        vi = self.v[f] + torch.cross(self.omega[f].unsqueeze(0),  torch.matmul(self.x, mat_R.t()), dim=1)

        d = torch.einsum('bi,i->b', xi, contact_normal)
        rel_v = torch.einsum('bi,i->b', vi, contact_normal)
        contact_condition = (d < (-self.c * self.init_height)) & (rel_v < 0.0)
        sum_position = torch.zeros(3, dtype=torch.float32)
        # caculate the how many points are in contact with the plane
        num_collision = torch.sum(contact_condition.int())

        v_out = torch.zeros(3, dtype=torch.float32)
        omega_out = torch.zeros(3, dtype=torch.float32)

        if num_collision > 0:
            contact_mask = contact_condition.float()[:, None]  # add a new axis to broadcast
            # calculate the sum of the contact points
            sum_position = torch.sum(self.x * contact_mask, dim=0)
            
            # calculate the average of the contact points
            collision_ri = sum_position / num_collision
            collision_Ri = mat_R @ collision_ri
            # calculate the velocity of the contact points
            vi = self.v[f] + self.omega[f].cross(collision_Ri)

            v_i_n = vi.dot(contact_normal) * contact_normal
            v_i_t = vi - v_i_n
            vn_new = -self.kn * v_i_n
            alpha = 1.0 - (self.mu * (1.0 + self.kn) * (torch.norm(v_i_n) / (torch.norm(v_i_t) + 1e-6)))
            if alpha < 0.0:
                alpha = 0.0
            vt_new = alpha  * v_i_t
            # print('f: {}, alpha:{}, vt_new:{}, vt:{}, item: vi_t_normal: {}, vi_n_normal: {}'.format(f, alpha, vt_new, v_i_t, torch.norm(v_i_t), torch.norm(v_i_n)))
            vi_new = vn_new + vt_new
            I = mat_R @ self.inertia_referance @ mat_R.t()
            collision_Rri_mat = self.GetCrossMatrix(collision_Ri)
            k = torch.tensor([[self.inv_mass, 0.0, 0.0],\
                        [0.0, self.inv_mass, 0.0],\
                        [0.0, 0.0, self.inv_mass]]) - collision_Rri_mat @ I.inverse() @ collision_Rri_mat
            J = k.inverse() @ (vi_new - vi)
            v_out = v_out + J * self.inv_mass
            omega_out = omega_out + I.inverse() @ collision_Rri_mat @ J
        return v_out, omega_out
    
    def object_collision_detect(self, f: torch.int32):
        v1_out = torch.zeros(3, dtype=torch.float32)
        omega1_out = torch.zeros(3, dtype=torch.float32)
        v2_out = torch.zeros(3, dtype=torch.float32)
        omega2_out = torch.zeros(3, dtype=torch.float32)

        # set object1 as the reference object, transform object2 to object1's coordinate
        mat_R1 = self.quat_to_matrix(self.object1_quaternion_list[f])  # obj1's rotation matrix
        mat_R2 = self.quat_to_matrix(self.object2_quaternion_list[f])  # obj2's rotation matrix

        # Transform obj2 vertices to world coordinate system
        xi2_world = self.object2_translation_list[f] + torch.matmul(self.x, mat_R2.t()) + self.mass_center[None]

        # Translate xi2 to obj1's coordinate system
        # Apply obj1's inverse rotation and translation
        xi2_local_to_obj1 = torch.matmul((xi2_world - self.object1_translation_list[f] - self.mass_center[None]), mat_R1)

        # xi2_local_to_obj1 now contains obj2's vertices in obj1's local coordinate system
        num_collision = 0
        xi2_local_to_obj1_in_bbox = (xi2_local_to_obj1 > self.bbox[0]) & (xi2_local_to_obj1 < self.bbox[1])
        # sum_position = torch.zeros(3, dtype=torch.float32)
        # query xi2_local_to_obj1's sdf value
        # pdb.set_trace()
        sdf_grid = self.rigid_body_sdf[None, None, :, :, :]
        bbox_delta = self.bbox[1] - self.bbox[0]
        xi2_normalized = 2.0 * (xi2_local_to_obj1 / bbox_delta) - 1.0
        xi2_normalized = xi2_normalized[None, :, None, None, :]
        xi2_local_to_obj1_sdf = torch.nn.functional.grid_sample(sdf_grid, xi2_normalized, mode='bilinear',
                                                                padding_mode='border', align_corners=True)
        xi2_local_to_obj1_sdf = xi2_local_to_obj1_sdf.reshape(-1)
        contact_condition = xi2_local_to_obj1_sdf < 0.0  # sdf value less than 0 means in collision
        if num_collision > 0:
            contact_mask = contact_condition.float()[:, None]  # add a new axis to broadcast
            # calculate the sum of the contact points
            sum_position = torch.sum(xi2_local_to_obj1 * contact_mask, dim=0)
            # calculate the average of the contact points
            collision_ri = sum_position / num_collision
            collision_Ri1 = mat_R1 @ collision_ri
            collision_Ri2 = mat_R2 @ collision_ri

            # calculate the velocity of the contact points
            vi1 = self.obj1_v[f] + self.obj1_omega[f].cross(collision_Ri1)
            vi2 = self.obj2_v[f] + self.obj2_omega[f].cross(collision_Ri2)

            if torch.dot(vi1, vi2) < 0.0: # dot value < 0 means two objects are moving away from each other
                return v1_out, omega1_out, v2_out, omega2_out
            
            # obj1 and obj2 have the same mass 
            vi1_new = vi2
            vi2_new = vi1

            I1 = mat_R1 @ self.inertia_referance @ mat_R1.t()
            I2 = mat_R2 @ self.inertia_referance @ mat_R2.t()

            collision_Rri_mat1 = self.GetCrossMatrix(collision_Ri1)
            collision_Rri_mat2 = self.GetCrossMatrix(collision_Ri2)

            k1 = torch.tensor([[self.inv_mass, 0.0, 0.0],
                            [0.0, self.inv_mass, 0.0],
                            [0.0, 0.0, self.inv_mass]]) - collision_Rri_mat1 @ I1.inverse() @ collision_Rri_mat1
            k2 = torch.tensor([[self.inv_mass, 0.0, 0.0],
                            [0.0, self.inv_mass, 0.0],
                            [0.0, 0.0, self.inv_mass]]) - collision_Rri_mat2 @ I2.inverse() @ collision_Rri_mat2
            
            J1 = k1.inverse() @ (vi1_new - vi1)
            J2 = k2.inverse() @ (vi2_new - vi2)

            v1_out = v1_out + J1 * self.inv_mass
            omega1_out = omega1_out + I1.inverse() @ collision_Rri_mat1 @ J1

            v2_out = v2_out + J2 * self.inv_mass
            omega2_out = omega2_out + I2.inverse() @ collision_Rri_mat2 @ J2
    
        return v1_out, omega1_out, v2_out, omega2_out

    def forward(self, f:torch.int32):
        # advect
        v1_out = (self.obj1_v[f] + torch.tensor([0.0,-9.8, 0.0]) * self.dt) * self.linear_damping
        v2_out = (self.obj2_v[f] + torch.tensor([0.0,-9.8, 0.0]) * self.dt) * self.linear_damping
        omega1_out = self.obj1_omega[f] * self.angular_damping
        omega2_out = self.obj2_omega[f] * self.angular_damping
        # check collision between object 1 and object 2
        v_1_out, omega_1_out, v_2_out, omega_2_out = self.object_collision_detect(f=f)

        v1_out = v1_out + v_1_out
        omega1_out = omega1_out + omega_1_out

        v2_out = v2_out + v_2_out
        omega2_out = omega2_out + omega_2_out
    
        # v_out_ , omega_out_ = self.planar_collision_detect(f=f)
        # v_out = v_out + v_out_
        # omega_out = omega_out + omega_out_

        v_1_out_ , omega_1_out_ = self.collision_detect(f=f, contact_normal=torch.tensor([0.0, 1.0, 0.0]), translation=self.object1_translation_list[f],\
                                                     quaternion=self.object1_quaternion_list[f], v=self.obj1_v[f], omega=self.obj1_omega[f])
        v1_out = v1_out + v_1_out_
        omega1_out = omega1_out + omega_1_out_

        v_2_out_, omega_2_out_ = self.collision_detect(f=f, contact_normal=torch.tensor([0.0, 1.0, 0.0]), translation=self.object2_translation_list[f],\
                                                        quaternion=self.object2_quaternion_list[f], v=self.obj2_v[f], omega=self.obj2_omega[f])
        v2_out = v2_out + v_2_out_
        omega2_out = omega2_out + omega_2_out_

        # J = F · Δt = m · Δv,  F = m · Δv / Δt = J / Δt
        # torque = r × F = r × (J / Δt) = (r × J) / Δt
        # Δω = I^(-1) · torque · Δt = I^(-1) · (r × J) / Δt · Δt = I^(-1) · (r × J)
        # update state
        wt1 = omega1_out * self.dt * 0.5
        wt2 = omega2_out * self.dt * 0.5

        dq1 = self.quat_mul(torch.tensor([0.0, wt1[0], wt1[1], wt1[2]], dtype=torch.float32), self.object1_quaternion_list[f])
        dq2 = self.quat_mul(torch.tensor([0.0, wt2[0], wt2[1], wt2[2]], dtype=torch.float32), self.object2_quaternion_list[f])
        
        self.object1_translation_list[f + 1] = self.object1_translation_list[f] + self.dt * v1_out
        self.object2_translation_list[f + 1] = self.object2_translation_list[f] + self.dt * v2_out

        self.obj1_omega[f + 1] = omega1_out
        self.obj2_omega[f + 1] = omega2_out

        self.obj1_v[f + 1] = v1_out
        self.obj2_v[f + 1] = v2_out

        quat_new1 = self.object1_quaternion_list[f] + dq1
        quat_new2 = self.object2_quaternion_list[f] + dq2

        self.object1_quaternion_list[f + 1] = quat_new1 / torch.norm(quat_new1)
        self.object2_quaternion_list[f + 1] = quat_new2 / torch.norm(quat_new2)

        return self.object1_translation_list[f + 1], self.object1_quaternion_list[f + 1], \
            self.object2_translation_list[f + 1], self.object2_quaternion_list[f + 1]
    
    def export_mesh(self, f:torch.int32):
        with torch.no_grad():
            mat_R1 = self.quat_to_matrix(self.object1_quaternion_list[i])
            mat_R2 = self.quat_to_matrix(self.object2_quaternion_list[i])

            xi1 = self.object1_translation_list[f] +  torch.matmul(self.x, mat_R1.t()) + self.mass_center[None]
            xi2 = self.object2_translation_list[f] +  torch.matmul(self.x, mat_R2.t()) + self.mass_center[None]

            faces = self.mesh.faces
            mesh1 = trimesh.Trimesh(vertices=xi1.detach().numpy(), faces=faces)
            mesh2 = trimesh.Trimesh(vertices=xi2.detach().numpy(), faces=faces)
            #
            # self.object1_translation.append(self.object1_translation[f].detach().clone().numpy())
            # self.object1_quaternion.append(self.object1_quaternion_list[f].detach().clone().numpy())
            # self.object2_translation.append(self.object2_translation[f].detach().clone().numpy())
            # self.object2_quaternion.append(self.object2_quaternion_list[f].detach().clone().numpy())
            #
            mesh1.export(str(Path(self.options['output']) / '{}_1.obj'.format(f // self.substep)))
            mesh2.export(str(Path(self.options['output']) / '{}_2.obj'.format(f // self.substep)))
    
    def export_translation_and_quaternion(self):
        # write as json file
        with open(str(Path(self.options['output']) / 'obj1_translation.json'), 'w') as f:
            # import pdb; pdb.set_trace()
            json.dump(self.object1_translation, f, indent=4)
        
        with open(str(Path(self.options['output']) / 'obj2_translation.json'), 'w') as f:
            json.dump(self.object2_translation, f, indent=4)

        with open(str(Path(self.options['output']) / 'obj1_quaternion.json'), 'w') as f:
            json.dump(self.object1_quaternion, f, indent=4)

        with open(str(Path(self.options['output']) / 'obj2_quaternion.json'), 'w') as f:
            json.dump(self.object2_quaternion, f, indent=4)

if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_name = "C:/Users/guanl/Desktop/GenshinNerf/slip_bunny_torch_base/bunny_original.obj"
    obstacle_mesh = 'C:/Users/guanl/Desktop/GenshinNerf/slip_bunny_ti_base/slope_new.obj'
    objects_sdf_path = "C:/Users/guanl/Desktop/GenshinNerf/slip_bunny_torch_base/bunny_original.npz"
    options = {
        'substep': 10,
        'frames': 100,
        'kn': 0.92,
        'mu': 0.1,
        'output': Path(current_directory) / 'output',
        'linear_damping': 0.999,
        'angular_damping': 0.998,
        'mesh': file_name,
        'obstacle_mesh': obstacle_mesh,
        'objects_sdf_path': objects_sdf_path
    }
    rigid = RigidBodySimulator(options=options)

    rigid.set_init_v()

    rigid.set_obj1_init_quaternion([-90, -90.0, -60.0])
    rigid.set_obj1_init_translation([2.1, 0.5, 0.25])

    rigid.set_obj2_init_quaternion([0.0, 0.0, 0.0])
    rigid.set_obj2_init_translation([2.1, 0.2, 0.25])
    # rigid.add_planar_contact(slope_degree=30, init_height=1.0)
    pbar = trange(options['frames'] * options['substep'] - 1)
    for i in pbar:
        translation_obj1, rotation_obj1, translation_obj2, rotation_obj2 = rigid(i)
        torch.cuda.synchronize()
        if i % 10 == 0:
            rigid.export_mesh(i)
    rigid.export_translation_and_quaternion()