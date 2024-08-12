import torch
import numpy as np
import trimesh
from pathlib import Path
import os
import math
from tqdm import trange


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
        self.voxel_resolution = 64
        # convert vertices to numpy array
        vertices = np.array(self.mesh.vertices) - self.mesh.center_mass
        self.translation = []
        self.quaternion = []
        self.v = []
        self.omega = []
        self.collision_function = []
        # torch tensors
        self.mass_center = torch.tensor(self.mesh.center_mass, dtype=torch.float32)
        self.x = torch.tensor(vertices, dtype=torch.float32, requires_grad=True)
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
        self.target = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32)
        self.init()

    def set_init_translation(self, init_translation):
        with torch.no_grad():
            self.translation[0] = torch.tensor(init_translation, dtype=torch.float32)

    def set_init_quaternion(self, init_euler_angle):
        with torch.no_grad():
            self.quaternion[0] = torch.tensor(self.from_euler(init_euler_angle), dtype=torch.float32)

    def set_init_v(self):
        with torch.no_grad():
            self.v[0] = self.init_v

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

    def clear(self):
        with torch.no_grad():
            for i in range(self.frames * self.substep):
                if i == 0:
                    self.translation[i] = torch.tensor([0.0, 1.09805178e-01, 0.0])
                    self.quaternion[i] = torch.tensor([1.0, 0.0, 0.0, 0.0])
                self.v[i] = torch.zeros(3, dtype=torch.float32)
                self.omega[i] = torch.zeros(3, dtype=torch.float32)

    def init(self):
        with torch.no_grad():
            self.mass = 0
            self.translation[0] = torch.tensor([0.3, 0.5, 0.0])
            self.quaternion[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            self.inertia_referance = torch.zeros(3, 3, dtype=torch.float32)
            self.v[0] = self.init_v
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

    def collision_detect(self, f: torch.int32, contact_normal):
        mat_R = self.quat_to_matrix(self.quaternion[f])
        xi = self.translation[f] + torch.matmul(self.x, mat_R.t()) + self.mass_center[None]
        vi = self.v[f] + torch.cross(self.omega[f].unsqueeze(0), torch.matmul(self.x, mat_R.t()), dim=1)
        d = torch.einsum('bi,i->b', xi, contact_normal)
        rel_v = torch.einsum('bi,i->b', vi, contact_normal)
        contact_condition = (d < 0.0) & (rel_v < 0.0)
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
            # print('collision_ri:{}'.format(collision_ri))
            collision_Ri = mat_R @ collision_ri
            # calculate the velocity of the contact points
            vi = self.v[f] + self.omega[f].cross(collision_Ri)

            v_i_n = vi.dot(contact_normal) * contact_normal
            v_i_t = vi - v_i_n
            vn_new = -self.kn * v_i_n
            alpha = 1.0 - (self.mu * (1.0 + self.kn) * (torch.norm(v_i_n) / (torch.norm(v_i_t) + 1e-6)))
            if alpha < 0.0:
                alpha = 0.0
            vt_new = alpha * v_i_t
            vi_new = vn_new + vt_new
            I = mat_R @ self.inertia_referance @ mat_R.t()
            collision_Rri_mat = self.GetCrossMatrix(collision_Ri)
            k = torch.tensor([[self.inv_mass, 0.0, 0.0], \
                              [0.0, self.inv_mass, 0.0], \
                              [0.0, 0.0, self.inv_mass]]) - collision_Rri_mat @ I.inverse() @ collision_Rri_mat
            J = torch.inverse(k) @ (vi_new - vi)
            v_out = v_out + J * self.inv_mass
            omega_out = omega_out + I.inverse() @ collision_Rri_mat @ J
        return v_out, omega_out

    def planar_collision_detect(self, f: torch.int32):
        contact_normal = torch.tensor([-self.s, self.c, 0.0], dtype=torch.float32)
        mat_R = self.quat_to_matrix(self.quaternion[f])
        xi = self.translation[f] + torch.matmul(self.x, mat_R.t()) + self.mass_center[None]
        vi = self.v[f] + torch.cross(self.omega[f].unsqueeze(0), torch.matmul(self.x, mat_R.t()), dim=1)
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
            vt_new = alpha * v_i_t
            # print('f: {}, alpha:{}, vt_new:{}, vt:{}, item: vi_t_normal: {}, vi_n_normal: {}'.format(f, alpha, vt_new, v_i_t, torch.norm(v_i_t), torch.norm(v_i_n)))
            vi_new = vn_new + vt_new
            I = mat_R @ self.inertia_referance @ mat_R.t()
            collision_Rri_mat = self.GetCrossMatrix(collision_Ri)
            k = torch.tensor([[self.inv_mass, 0.0, 0.0], \
                              [0.0, self.inv_mass, 0.0], \
                              [0.0, 0.0, self.inv_mass]]) - collision_Rri_mat @ I.inverse() @ collision_Rri_mat
            J = k.inverse() @ (vi_new - vi)
            v_out = v_out + J * self.inv_mass
            omega_out = omega_out + I.inverse() @ collision_Rri_mat @ J
        return v_out, omega_out

    def forward(self, f: torch.int32):
        # advect
        v_out = (self.v[f] + torch.tensor([0.0, -9.8, 0.0]) * self.dt) * self.linear_damping
        omega_out = self.omega[f] * self.angular_damping
        v_out_, omega_out_ = self.planar_collision_detect(f=f)
        v_out = v_out + v_out_
        omega_out = omega_out + omega_out_
        v_out_, omega_out_ = self.collision_detect(f=f, contact_normal=torch.tensor([0.0, 1.0, 0.0]))
        v_out = v_out + v_out_
        omega_out = omega_out + omega_out_
        # J = F 路 螖t = m 路 螖v,  F = m 路 螖v / 螖t = J / 螖t
        # torque = r 脳 F = r 脳 (J / 螖t) = (r 脳 J) / 螖t
        # 螖蠅 = I^(-1) 路 torque 路 螖t = I^(-1) 路 (r 脳 J) / 螖t 路 螖t = I^(-1) 路 (r 脳 J)
        # update state
        wt = omega_out * self.dt * 0.5
        dq = self.quat_mul(torch.tensor([0.0, wt[0], wt[1], wt[2]], dtype=torch.float32), self.quaternion[f])
        self.translation[f + 1] = self.translation[f] + self.dt * v_out
        self.omega[f + 1] = omega_out
        self.v[f + 1] = v_out
        quat_new = self.quaternion[f] + dq
        self.quaternion[f + 1] = quat_new / torch.norm(quat_new)

        return self.translation[f + 1], self.quaternion[f + 1]

    def compute_loss(self):
        # compute loss
        loss = torch.norm(self.translation[-1] + self.mass_center - self.target)
        loss.backward()
        print('loss:{}'.format(loss))
        print('kn grad:{}, mu grad: {}'.format(self.kn.grad, self.mu.grad))
        return loss

    def export_mesh(self, f: torch.int32):
        with torch.no_grad():
            mat_R = self.quat_to_matrix(self.quaternion[f])
            xi = self.translation[f] + torch.matmul(self.x, mat_R.t()) + self.mass_center[None]
            faces = self.mesh.faces
            mesh = trimesh.Trimesh(vertices=xi.detach().numpy(), faces=faces)
            mesh.export(str(Path(self.options['output']) / '{}.obj'.format(f // self.substep)))


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # file_name = "C:/Users/guanl/Desktop/GenshinNerf/slip_bunny_ti_base/bunny_original.obj"
    file_name = file_name = Path(current_directory) / 'mesh' / 'bunny.obj'
    obstacle_mesh = file_name = Path(current_directory) / 'mesh' / 'slope_new.obj'
    options = {
        'substep': 10,
        'frames': 60,
        'kn': 0.92,
        'mu': 0.1,
        'output': Path(current_directory) / 'output',
        'linear_damping': 0.999,
        'angular_damping': 0.998,
        'mesh': file_name,
        'obstacle_mesh': obstacle_mesh
    }
    train_iters = 100
    rigid = RigidBodySimulator(options=options)
    optimizer = torch.optim.Adam(
        [
            # {"params": getattr(rigid, 'init_v'), 'lr': 1e-1},
            {"params": getattr(rigid, 'mu'), 'lr': 1e-2}
            # {"params": getattr(rigid, 'kn'), 'lr': 1e-2}
        ]
    )
    # for i in range(train_iters):
    # optimizer.zero_grad()
    # rigid.clear()
    rigid.set_init_v()
    rigid.set_init_quaternion([-90, 180.0, 30.0])
    rigid.set_init_translation([2.1, 0.3, 0.25])
    rigid.add_planar_contact(slope_degree=30, init_height=1.0)
    pbar = trange(options['frames'] * options['substep'] - 1)
    for i in pbar:
        translation, rotation = rigid(i) # move forward
        torch.cuda.synchronize()
        if i % 10 == 0:
            rigid.export_mesh(i)
    # loss = rigid.compute_loss()
    # if loss < 1e-6:
    #     break
    # optimizer.step()
    # print('optimized mu:{}'.format(rigid.mu))
    # print('final position = ', translation)