import math

import numpy as np


def generate_z_up_slope(width, height, length, filename):
    # 创建斜坡的顶点坐标
    alla = [2, 0, 0]
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [width, 0, 0.0],
        [0.0, height, 0.0],
        [width, height, 0.0],

        [0, 0.0, length],
        [width, 0.0, length],
        [0.0, height, length],
        [width, height, length],

    ])

    # 创建斜坡的三角形面片
    faces = np.array([
        [0, 2, 1],
        [1, 2, 3],
        [0, 1, 4],
        [1, 5, 4],
        [2, 5, 3],
        [2, 4, 5],
        [0, 4, 2],
        [1, 3, 5],
        # [5, 11, 2],
        # [5, 10, 11],
        # [3, 4, 11],
        # [3, 11, 10]
    ])

    # 将顶点和面片保存为 OBJ 格式文件
    with open(filename, 'w') as f:
        for v in vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for face in faces:
            f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))


def generate_y_up_slope(width, length, height, filename, alla=np.zeros(3)):
    # 创建斜坡的顶点坐标
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0, 0, width],
        [0.0, height, 0.0],
        [0, height, width],

        [-length, 0.0, 0],
        [-length, 0.0, width],
        [-length, height, 0],
        [-length, height, width],

    ])
    vertices = vertices + alla
    # 创建斜坡的三角形面片
    faces = np.array([
        [0, 2, 1],
        [1, 2, 3],
        [0, 1, 4],
        [1, 5, 4],
        [2, 5, 3],
        [2, 4, 5],
        [0, 4, 2],
        [1, 3, 5],
        # [5, 11, 2],
        # [5, 10, 11],
        # [3, 4, 11],
        # [3, 11, 10]
    ])
    # 将顶点和面片保存为 OBJ 格式文件
    with open(filename, 'w') as f:
        for v in vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for face in faces:
            f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))


def form_euler(euler_angle, d_type="radius"):
    if d_type == "radius":
        euler_angle = euler_angle * 360.0 / math.pi
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

# 调用函数生成斜坡并保存为 OBJ 文件
# alla = np.array([-2, -0.3, -0.25])
scalar = 0.2451
# generate_y_up_slope(scalar * 1.732, scalar * 1.732, scalar, 'C:/Users/guanl/Desktop/GenshinNerf/slip_bunny/slope2.obj', alla=alla)
eular_angle = np.array([-90, -90, -60])
print(form_euler(eular_angle, d_type="angle"))
