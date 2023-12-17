import taichi as ti
from argparse import ArgumentParser
import numpy as np
import os
import trimesh
import torch
import math
from pathlib import Path
import json
import tetgen
from common import *

ti.init(arch=ti.cpu)

# quaternion helper functions
@ti.func
def quat_mul(a, b)->ti.Vector:
    return ti.Vector([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                      a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
                      a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1]])

@ti.func
def quat_mul_scalar(a, b)->ti.Vector:
    return ti.Vector([a[0] * b, a[1] * b, a[2] * b, a[3] * b])

@ti.func
def quat_add(a, b)->ti.Vector:
    return ti.Vector([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])

@ti.func
def quat_subtraction(a, b)->ti.Vector:
    return ti.Vector([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])

@ti.func
def quat_normal(a)->ti.f32:
    return ti.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3])

@ti.func
def quat_to_matrix(q)->ti.Matrix:
    q = q.normalized()
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ti.Matrix([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                      [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                      [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])

@ti.func
def quat_inverse(q)->ti.Vector:
    # the inverse of a quaternion is its conjugate divided by its norm
    norm_squared = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    return ti.Vector([q[0], -q[1], -q[2], -q[3]]) / norm_squared

@ti.func
def Get_Cross_Matrix(a)->ti.Matrix:
    return ti.Matrix([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    # A = ti.Matrix.zero(dt=ti.f32, n=4, m=4)
    # A[0, 0] = 0
    # A[0, 1] = -a[2]
    # A[0, 2] = a[1]
    # A[1, 0] = a[2]
    # A[1, 1] = 0
    # A[1, 2] = -a[0]
    # A[2, 0] = -a[1]
    # A[2, 1] = a[0]
    # A[2, 2] = 0
    # A[3, 3] = 1
    # return A

# the euler angle is in degree, we first conver it to radian
def form_euler(euler_angle):
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

@ti.func
def vec_to_quat(vec):
    return ti.Vector([0.0, vec[0], vec[1], vec[2]])

@ti.func
def quat_to_vec(quat):
    return ti.Vector([quat[1], quat[2], quat[3]])

@ti.func
def get_current_position(initial_vertex_position, quaternion, translation, initial_mass_center)->ti.Vector:
    # Step 1: Calculate initial offset
    initial_offset = initial_vertex_position - initial_mass_center

    # Step 2: Apply rotation using quaternion
    rotated_offset = quat_mul(quat_mul(quaternion, vec_to_quat(initial_offset)), quat_inverse(quaternion))

    # Step 3: Apply translation
    current_position = quat_to_vec(rotated_offset) + translation

    return current_position
    
@ti.data_oriented
class rigid_body:

    current_frame = 0
    T = ti.Matrix.field(4, 4, dtype=ti.f32, shape=1)
    frame_dt = 1.0 / 60.0
    substep = 10
    dt = frame_dt / substep

    def __init__(self, mesh_file_name, options):
        assert options is not None
        assert mesh_file_name is not None, 'mesh_file_name is None. You need to privide a mesh file name.'  # floor
        floor_vertices = np.array([[-5.0, 0.0, -5.0], [-5.0, 0.0, 5.0], [5.0, 0.0, 5.0], [5.0, 0.0, -5.0]])
        floor_faces = np.array([[0, 1, 2], [0, 2, 3]]).flatten()
        self.floor_vertices = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.floor_faces = ti.field(dtype=ti.i32, shape=6)
        self.floor_vertices.from_numpy(floor_vertices)
        self.floor_faces.from_numpy(floor_faces)
        # load mesh
        self.mesh = trimesh.load(mesh_file_name)
        # self.mesh.apply_scale(5.0)
        # print('bounding box:', self.mesh.bounding_box.bounds)
        vertices = np.array(self.mesh.vertices) - self.mesh.center_mass
        faces = np.array(self.mesh.faces)
        self.mass_center = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.mass_center[None] = self.mesh.center_mass
        # import pdb; pdb.set_trace()
        # static collision mesh
        self.static_mesh = ti.Vector.field(3, dtype=ti.f32, shape=vertices.shape[0])
        self.static_mesh.from_numpy(vertices)
        self.static_mesh_faces = ti.field(dtype=ti.i32, shape=faces.shape[0] * 3)
        self.static_mesh_faces.from_numpy(faces.flatten())

        if options['frames'] is not None:
            self.frames = options['frames']
        
        if options['transform'] is not None:
            self.initial_translation = ti.Vector(options['transform'][0:3], dt=ti.f32)
            self.initial_quat = ti.Vector(form_euler(options['transform'][3:6]), dt=ti.f32)
        else:
            self.initial_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
            self.initial_translation = ti.Vector([0.0, 0.0, 0.0])

        self.c, self.s = np.cos(np.deg2rad(options["slope_degree"])), np.sin(np.deg2rad(options["slope_degree"]))
        self.init_height = options['init_height']
        self.contact_normal = ti.Vector([-self.s, self.c, 0.0], dt=ti.f32)

        # convert mesh to taichi data structure
        self.x = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.x_t = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, self.mesh.vertices.shape[0]).place(self.x, self.x.grad, self.x_t)

        # self.kf = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        # self.kn = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        # self.mu = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        # self.offset = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        # self.p = ti.field(dtype=ti.f32, shape= (), needs_grad=True)
        # self.offset[None] = options['offset']
        # self.kf[None] = options['kf']
        # self.kn[None] = options['kn']
        # self.mu[None] = options['mu']
        # self.p[None] = options['p']

        # translation 
        self.mass = ti.field(dtype=ti.f32, shape=())
        self.mass[None] = 0.0
        self.inv_mass = 0.0
        self.force = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.v = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        # rotation
        self.omega = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.torque = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.inertia = ti.Matrix.field(3, 3, dtype=ti.f32, needs_grad=True)
        # the indices is constant for all frames, so we don't need to store it in the frame loop, but only in the init_state function
        # and we won't need the grad of indices, so we don't need to set needs_grad=True
        self.indices = ti.field(dtype=ti.i32)
         # transformation information for rigid body
        self.quad = ti.Vector.field(4, dtype=ti.f32, needs_grad=True) 
        self.translation = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        # place data
        ti.root.dense(ti.j, 3 * self.mesh.faces.shape[0]).place(self.indices)
        particles = ti.root.dense(ti.i, self.frames)
        # we only store these variables in the the mass center
        particles.place(self.torque, self.torque.grad, self.inertia, self.inertia.grad, self.quad, self.quad.grad, self.translation, self.translation.grad,
                        self.omega, self.omega.grad, self.v, self.v.grad, self.force, self.force.grad)

        # these variables are stored in each vertex
        # particles.dense(ti.j, self.mesh.vertices.shape[0]).place(self.x, self.x.grad)
        self.test_coord = ti.Vector.field(3, dtype=ti.f32, shape=10)

        self.init_state(vertices, faces)
        self.inv_mass = 1.0 / self.mass[None]

        # rigid body parameters
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        # params need to be optimized
        self.ke = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.mu = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.ke[None] = options['ke']
        self.mu[None] = options['mu']
        # import pdb; pdb.set_trace()
        # set up ggui
        #create a window
        self.window = ti.ui.Window(name='Rigid body dynamics', res=(1280, 720), fps_limit=60, pos=(150,150))
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(1,2,3)
        self.camera.lookat(0,0,0)
        
        position = ti.Vector([1, 2, 3])
        lookat = ti.Vector([0, 0, 0])
        up = ti.Vector([0, 1, 0])
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.scene.set_camera(self.camera)

        self.pause = True

    @ti.kernel
    def get_mesh_now(self):
        mat_R = quat_to_matrix(self.quad[0])
        for i in range(self.x.shape[0]):
            self.x_t[i] = self.translation[0] + mat_R @ self.x[i] + self.mass_center[None]
            

    @ti.kernel
    def clear(self):
        self.force.fill(0.0)
        self.torque.fill(0.0)

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

    @ti.kernel
    def get_transform_matrix(self):
        R = quat_to_matrix(self.quad[0])
        T = ti.Matrix.identity(ti.f32, 4)
        T[0, 3] = self.translation[0][0]
        T[1, 3] = self.translation[0][1]
        T[2, 3] = self.translation[0][2]
        T[0:3, 0:3] = R
        self.T[0] = T

    def render(self, frame=0):
        self.camera.track_user_inputs(self.window, movement_speed=0.05, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(1,2,3), color=(1, 1, 1))
        # draw the floor
        self.scene.mesh(self.floor_vertices, self.floor_faces, color=(0.5, 0.5, 0.5),show_wireframe=True)
        self.scene.mesh_instance(self.x, self.indices, color=(0.5, 0.5, 0.5),show_wireframe=True, transforms=self.T)
        # draw static mesh
        # self.scene.mesh(self.static_mesh, self.static_mesh_faces, color=(0.3, 0.2, 0.1),show_wireframe=False)
        # self.scene.mesh(self.x, self.indices, color=(0.5, 0.5, 0.5),show_wireframe=True)
        self.canvas.scene(self.scene)
        self.window.show()
        # print('mass center:', self.mass_center[None] + self.translation[0])

    
    @ti.kernel
    def init_state(self, vertices:ti.types.ndarray(), faces:ti.types.ndarray()):
        for i in range(vertices.shape[0]):
            self.x[i] = ti.Vector([vertices[i, 0], vertices[i, 1], vertices[i, 2]], dt=ti.f32)
        for i in range(faces.shape[0]):
            for j in ti.static(range(3)):
                self.indices[i * 3 + j] = faces[i, j]
        # set initial transformation
        self.quad[0] = self.initial_quat
        self.translation[0] = ti.Vector([0.0, 0.0, 0.0]) + self.initial_translation
        self.v[0] = ti.Vector([0.0, 0.0, 0.0])
        # set initial rotation
        self.omega[0] = ti.Vector([0.0, 0.0, 0.0])
        self.inertia[0] = ti.Matrix.zero(ti.f32, 3, 3)
        # calculate ref inertia (frame 0)
        mass = 1.0
        for i in range(vertices.shape):
            ti.atomic_add(self.mass[None], mass)
            r = self.x[i] - self.mass_center[None]
            # inertia = \sum_{i=1}^{n} m_i (r_i^T r_i I - r_i r_i^T)  https://en.wikipedia.org/wiki/List_of_moments_of_inertia
            # as r_i is a col vector, r_i^T is a row vector, so r_i^T r_i is a scalar (actually is dot product)
            I_i = mass * (r.dot(r) * ti.Matrix.identity(ti.f32, 3) - r.outer_product(r))
            ti.atomic_add(self.inertia[0], I_i)
        print('inertia:', self.inertia[0])


    @ti.kernel
    def step(self, f:ti.f32):
        self.v[0] += self.dt * self.gravity
        self.v[0] *= 0.999
        print('frame:', f, 'v:', self.v[0])
        self.omega[0] *= 0.998

        # collision Impulse
        avg_collision_point = ti.Vector([0.0, 0.0, 0.0])
        num_collision = 0
        ri = ti.Vector([0.0, 0.0, 0.0])
        vi = ti.Vector([0.0, 0.0, 0.0])
        mat_R = quat_to_matrix(self.quad[0])
        # floor collision
        # implus model
        for i in range(self.x.shape[0]):
            ri = self.x[i]
            xi = self.translation[0] + mat_R @ ri + self.mass_center[None]
            if xi.dot(self.contact_normal) < -self.c * self.init_height:
                vi = self.v[0] + self.omega[0].cross(mat_R @ ri)
                if vi.dot(self.contact_normal) < 0.0:
                    ti.atomic_add(num_collision, 1)
                    ti.atomic_add(avg_collision_point, ri)
        if num_collision > 0:
            ri = avg_collision_point / num_collision
            print('ri:', ri)
            Rri = mat_R @ ri
            vi = self.v[0] + self.omega[0].cross(mat_R @ ri)

            # calculate new velocity
            v_iN = vi.dot(self.contact_normal) * self.contact_normal
            v_iT = vi - v_iN
            # impluse method 
            alpha = 1.0 - (self.mu[None] * (1.0 + self.ke[None]) * (v_iN.norm()/v_iT.norm()))
            if alpha < 0.0:
                alpha = 0.0
            vi_new_N = -self.ke[None] * v_iN
            vi_new_T = alpha * v_iT
            vi_new = vi_new_N + vi_new_T

            # calculate impulse
            I = mat_R @ self.inertia[0] @ mat_R.transpose()
            Rri_mat = Get_Cross_Matrix(Rri)
            k = ti.Matrix([[self.inv_mass, 0.0, 0.0],\
                           [0.0, self.inv_mass, 0.0],\
                           [0.0, 0.0, self.inv_mass]]) - Rri_mat @ I.inverse() @ Rri_mat
            J = k.inverse() @ (vi_new - vi)
            # update velocity
            self.v[0]+= self.inv_mass * J
            self.omega[0] += I.inverse() @ Rri_mat @ J

        planar_contat_normal = ti.Vector([0.0, 1.0, 0.0])
        num_collision = 0
        avg_collision_point = ti.Vector([0.0, 0.0, 0.0])
        for i in range(self.x.shape[0]):
            ri = self.x[i]
            xi = self.translation[0] + mat_R @ ri + self.mass_center[None]
            if xi.dot(planar_contat_normal) < 0.01:
                vi = self.v[0] + self.omega[0].cross(mat_R @ ri)
                if vi.dot(planar_contat_normal) < 0.0:
                    ti.atomic_add(num_collision, 1)
                    ti.atomic_add(avg_collision_point, ri)
        if num_collision > 0:
            ri = avg_collision_point / num_collision
            Rri = mat_R @ ri
            vi = self.v[0] + self.omega[0].cross(Rri)
            # calculate new velocity
            v_iN = vi.dot(planar_contat_normal) * planar_contat_normal
            v_iT = vi - v_iN
            # impluse method 
            alpha = 1.0 - (self.mu[None] * (1.0 + self.ke[None]) * (v_iN.norm()/v_iT.norm()))
            if alpha < 0.0:
                alpha = 0.0
            vi_new_N = -self.ke[None] * v_iN
            vi_new_T = alpha * v_iT
            vi_new = vi_new_N + vi_new_T

            # calculate impulse
            I = mat_R @ self.inertia[0] @ mat_R.transpose()
            Rri_mat = Get_Cross_Matrix(Rri)
            k = ti.Matrix([[self.inv_mass, 0.0, 0.0],\
                           [0.0, self.inv_mass, 0.0],\
                           [0.0, 0.0, self.inv_mass]]) - Rri_mat @ I.inverse() @ Rri_mat
            J = k.inverse() @ (vi_new - vi)
            # update velocity
            self.v[0]+= self.inv_mass * J
            self.omega[0] += I.inverse() @ Rri_mat @ J
        wt = self.omega[0] * self.dt * 0.5
        dq = quat_mul(ti.Vector([0.0, wt[0], wt[1], wt[2]]), self.quad[0])

        self.translation[0] += self.dt * self.v[0]
        self.quad[0] += dq
        self.quad[0] = self.quad[0].normalized()


    @ti.kernel
    def step_soft_contact(self, f:ti.f32):
        self.v[0] += self.dt * self.gravity
        self.v[0] *= 0.999
        self.omega[0] *= 0.998

        # collision forces
        force = ti.Vector([0.0, 0.0, 0.0])
        torque = ti.Vector([0.0, 0.0, 0.0])
        mat_R = quat_to_matrix(self.quad[0])

        # soft contact parameters
        kn = 1e4 # Spring constant
        kf = 0.25    # Damping coefficient
        mu = 1e4
        for i in range(self.x.shape[0]):
            ri = self.x[i] - self.mass_center[None]
            xi = self.translation[0] + mat_R @ ri + self.mass_center[None]
            Rri = mat_R @ ri
            # Check for collision with the plane\
            planar_contat_normal = ti.Vector([0.0, 1.0, 0.0])
            d = planar_contat_normal.dot(xi) - 0.01
            if d < 0.0:
                fn_mag = kn * ti.math.pow(-ti.min(d, 0.0), 3 - 1)
                fn = fn_mag * planar_contat_normal
                # compute the friction force
                vi = self.v[0] + self.omega[0].cross(mat_R @ ri)
                us = vi - vi.dot(planar_contat_normal) * planar_contat_normal
                us_mag = us.norm()
                ff = -ti.min(kf, mu * fn_mag / (us_mag + 1e-6)) * us
                force += (fn + ff)
                torque += Rri.cross(fn + ff)
        I = mat_R @ self.inertia[0] @ mat_R.transpose()
        # Update linear and angular velocity
        self.v[0] += self.dt * self.inv_mass * force
        self.omega[0] += self.dt * I.inverse() @ torque

        # Integrate position and orientation
        self.translation[0] += self.dt * self.v[0]
        wt = self.omega[0] * self.dt * 0.5
        dq = quat_mul(ti.Vector([0.0, wt[0], wt[1], wt[2]]), self.quad[0])

        self.quad[0] += dq
        self.quad[0] = self.quad[0].normalized()


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # file_name = Path(current_directory) / 'assets' / 'bunny_0.obj'

    file_name = "C:/Users/guanl/Desktop/GenshinNerf/slip_bunny_ti_base/bunny_original.obj"
    robot = rigid_body(file_name, { 'frames': 60,
                                    'ke': 0.2,
                                    'kn': 0.0,
                                    'mu': 1,
                                    'init_height': 1,
                                    'offset': 0.0, 
                                    'p': 3,
                                    'slope_degree': 30,
                                    'transform': [2.1, 0.3, 0.25, -90, -90, 30],
                                    })
    robot.run()
    