import json

import torch
import numpy as np
import trimesh
from pathlib import Path
import os
import math


class RigidBodySimulator(torch.nn.Module):
    def __init__(self, options):
        super(RigidBodySimulator, self).__init__()
        self.options = options
        self.substep = options['substep']
        self.frames = options['frames']
        self.dt = 1.0 / 60.0 / self.substep

        self.mesh = trimesh.load_mesh(str(Path(options['mesh'])))
        print('mass_center:{}'.format(self.mesh.center_mass))
        # convert vertices to numpy array
        self.vertices = np.array(self.mesh.vertices)
        self.translation = []
        self.quaternion = []
        self.v = []
        self.omega = []
        # torch tensors
        self.mass_center = torch.tensor(self.mesh.center_mass, dtype=torch.float32)
        self.x = torch.tensor(self.vertices, dtype=torch.float32, requires_grad=True)
        for i in range(self.frames * self.substep):
            self.translation.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
            self.quaternion.append(torch.zeros(4, dtype=torch.float32, requires_grad=True))
            self.v.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
            self.omega.append(torch.zeros(3, dtype=torch.float32, requires_grad=True))
        self.kn = torch.nn.Parameter(torch.tensor([options['kn']], requires_grad=True))
        self.mu = torch.nn.Parameter(torch.tensor([options['mu']], requires_grad=True))
        self.linear_damping = torch.nn.Parameter(torch.tensor([options['linear_damping']]))
        self.angular_damping = torch.nn.Parameter(torch.tensor([options['angular_damping']]))
        self.init_v = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=True)
        self.mass = torch.tensor([0.0], dtype=torch.float32)
        self.inv_mass = torch.tensor([0.0], dtype=torch.float32)
        self.sum_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.num_collision = torch.tensor([0], dtype=torch.int32)
        self.init()

    def set_init_translation(self, init_translation):
        with torch.no_grad():
            self.translation[0] = torch.tensor(init_translation, dtype=torch.float32)

    def set_init_quaternion(self, init_euler_angle):
        with torch.no_grad():
            self.quaternion[0] = torch.tensor(self.form_euler(init_euler_angle), dtype=torch.float32)

    def set_init_v(self, init_v):
        with torch.no_grad():
            self.init_v = torch.tensor(init_v, dtype=torch.float32)

    # the euler angle is in degree, we first conver it to radian
    def form_euler(self, euler_angle):
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

    def quat_normal(self, a) -> torch.int32:
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
            self.translation[0] = torch.tensor([0.0, 0.0, 0.0])
            self.quaternion[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            self.inertia_referance = torch.zeros(3, 3, dtype=torch.float32)
            mass = 1.0
            for i in range(self.vertices.shape[0]):
                self.mass += mass
                r = self.x[i]
                # inertia = \sum_{i=1}^{n} m_i (r_i^T r_i I - r_i r_i^T)  https://en.wikipedia.org/wiki/List_of_moments_of_inertia
                # as r_i is a col vector, r_i^T is a row vector, so r_i^T r_i is a scalar (actually is dot product)
                I_i = mass * (r.dot(r) * torch.eye(3) - r.outer_product(r))
                self.inertia_referance += I_i
            self.inv_mass = 1.0 / self.mass

    def forward(self, f: torch.int32):
        # advect
        v_out = self.v[f] + torch.tensor([0.0, -9.8, 0.0]) * self.dt * self.linear_damping
        omega_out = self.omega[f] * self.angular_damping

        # collision detect
        self.sum_position = torch.zeros(3, dtype=torch.float32)
        self.num_collision = 0
        inertial = torch.zeros(3, 3, dtype=torch.float32)
        mat_R = self.quat_to_matrix(self.quaternion[f])
        xi = self.translation[f] + mat_R @ (self.x - self.mass_center) + self.mass_center[None]
        planar_contat_normal = torch.tensor([0.0, 1.0, 0.0])  # contact normal (planar contact)
        d = torch.einsum('bi,i->b', xi, planar_contat_normal)
        contact_condition = d < 0.0  # contact condition
        # caculate the how many points are in contact with the plane
        self.num_collision = torch.sum(contact_condition)

        # collision response
        # impluse method
        if self.num_collision > 0:
            # calculate the sum of the contact points
            self.sum_position = torch.einsum('bi,i->b', xi, contact_condition)
            # calculate the average of the contact points
            collision_ri = self.sum_position / self.num_collision
            collision_Ri = mat_R @ collision_ri
            # calculate the velocity of the contact points
            vi = self.v[f] + self.omega[f].cross(mat_R @ collision_ri)

            v_i_n = torch.einsum('bi,i->b', vi, planar_contat_normal) * planar_contat_normal
            v_i_t = vi - v_i_n
            vn_new = -self.kn * v_i_n
            alpha = 1.0 - (self.mu * (1.0 + self.ke) * (torch.norm(v_i_t) / (torch.norm(v_i_n) + 1e-10)))
            if alpha < 0.0:
                alpha = 0.0
            vt_new = alpha * v_i_t
            vi_new = vn_new + vt_new
            inertial = mat_R @ self.inertia_referance @ mat_R.transpose()
            collision_Rri_mat = self.GetCrossMatrix(collision_Ri)
            k = torch.tensor([[self.inv_mass, 0.0, 0.0],
                              [0.0, self.inv_mass, 0.0],
                              [0.0, 0.0, self.inv_mass]]) - collision_Rri_mat @ inertial.inverse() @ collision_Rri_mat
            J = k.inverse() @ (vi_new - vi)
            v_out = v_out + J * self.inv_mass
            omega_out = omega_out + inertial.inverse() @ collision_Rri_mat @ J

        # update
        wt = omega_out * self.dt * 0.5
        dq = self.quat_mul(torch.tensor([0.0, wt[0], wt[1], wt[2]], dtype=torch.float32), self.quat[f])
        self.translation[f + 1] = self.translation[f] + self.dt * v_out
        self.omega[f + 1] = omega_out
        self.v[f + 1] = v_out
        quat_new = self.quat_add(self.quaternion[f], dq)
        self.quaternion[f + 1] = quat_new / torch.norm(quat_new)

        return self.translation[f + 1], self.quaternion[f + 1]

    def run(self):
        data= {}
        data["frames"] = self.frames
        data["1_1_M"] = [[1.0, 0.0, 0.0,0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, -3.0],
                         [0.0, 0.0, 0.0, 1.0]]
        data['dt'] = self.dt
        data['substep'] = self.substep
        data["results"] = []
        while self.window.running:
            if self.window.is_pressed(ti.ui.LEFT, 'b'):
                self.pause = not self.pause
            if not self.pause:
                for i in range(self.substep):
                    self.clear()
                    self.step(self.current_frame)
                new_frame = {}
                new_frame["frame_id"] = self.current_frame
                new_frame["translation"] = self.translation[0].to_numpy().tolist()
                new_frame["rotation"] = self.quad[0].to_numpy().tolist()
                data["results"].append(new_frame)
                self.current_frame += 1
                # output mesh result
                self.get_mesh_now()
                # import pdb
                # pdb.set_trace()

                mesh_vertices = self.x_t.to_numpy()
                mesh_vertices = mesh_vertices.tolist()
                mesh_faces = self.mesh.faces.tolist()
                mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
                if self.current_frame <= data['frames']:
                    mesh.export('mesh_result/{:04d}.obj'.format(self.current_frame))
                else:
                    exit(0)
            # print('x shape', self.x.shape)
            self.get_transform_matrix()
            self.render()
        with open('transform.json', 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_name = Path(current_directory) / 'mesh' / 'bunny.obj'
    options = {
        'substep': 10,
        'frames': 30,
        'kn': 0.92,
        'mu': 0.1,
        'linear_damping': 0.999,
        'angular_damping': 0.999,
        'mesh': file_name,
    }
    rigid = RigidBodySimulator(options=options)
    rigid.run()
