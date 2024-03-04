import numpy as np

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix into a quaternion.
    Parameters:
    - R: A 3x3 rotation matrix.
    Returns:
    - q: A quaternion [qw, qx, qy, qz] as a numpy array.
    """
    # Ensure the matrix is numpy array for calculations
    R = np.asarray(R)
    # Allocate space for the quaternion
    q = np.empty((4,))
    # Compute the trace of the matrix
    tr = R.trace()
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q

def decompose_RT(input_mat, flag = "quad"):
    # assume input mat is 4x4 mat
    r = input_mat[:3, :3]
    t = input_mat[:3, 3]

    if flag == "quad": # decompose mat to quad and T
        r = rotation_matrix_to_quaternion(r)
    return r, t

def calcm3(mat1, mat2):
    m2_inv = np.linalg.inv(mat2)
    # m3 = m2_inv @ mat1
    m3 = mat1 @ m2_inv
    print(str(m3))
    return m3


def from_static_and_frame0_to_final(mat1, mat2):
    m1_inv = np.linalg.inv(mat1)
    # m3 = m2_inv @ mat1
    m3 = m1_inv @ mat2

    print(str(m3))
    return m3

m1 = np.array(
    [[-0.03572936, 0.99931851, 0.0092697, 0.05514524],
     [-0.99814468, -0.0361419, 0.0489998, -0.0607912],
     [0.04930147, -0.00750177, 0.99875578, -0.01673644],
     [0., 0., 0., 1.]]
)

m2 = np.array(
    [[0.90419301, -0.41297267, -0.10903509, -0.07296673],
     [0.42601751, 0.89033453, 0.16066592, -0.18126493],
     [0.03072708, -0.19172385, 0.98096782, 0.27248595],
     [0., 0., 0., 1.]]
)
m3 =  np.array(
    [[-0.3735146, -0.7024477, 0.6058498, -0.55369043],
     [-0.9249084, 0.2320759, -0.3011399, 0.22341669],
     [0.0709319, -0.6728357, -0.7363835, 0.6192576],
     [0., 0., 0., 1.]]
       )
m3 = from_static_and_frame0_to_final(m1, m2)

r, t = decompose_RT(m3)
print(r)
print(t)

# use calc m3 as m1 is the the mat calc from the camera from the qr board, and the other one（m2） is RT detected from the
# single other-race single qr code for R0 T0 calculation

"""
m2 = np.array(
                  [[-0.20938785,    0.72913867, -0.65154703,  0.61477604],
                    [0.97783173,    0.15705771, -0.1384846,   0.15637316],
                    [0.00135601,   -0.66610035, -0.7458609,   0.42764357],
                    [0., 0., 0., 1.]]
)

m1 = np.array(
    [[0.96851933,   0.18500824, -0.16655992,  0.18477099],
     [0.24893497,  -0.72322685,  0.644185,   -0.6132438],
     [-0.00128107, -0.66536826, -0.7465142,   0.43042728],
     [0.,          0.,          0.,          1.]]

)

"""