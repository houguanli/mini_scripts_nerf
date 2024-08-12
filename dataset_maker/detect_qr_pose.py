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
def quat_to_matrix(q):
    q = q / np.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])

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


# the m3 (r0 t0) calc from static should be m2 into the next calculation
# the other m3 is the m1 in the next, and calc as following
def from_static_and_frame0_to_final(mat1, mat2):
    m1_inv = np.linalg.inv(mat1)
    # m3 = m2_inv @ mat1
    m3 = m1_inv @ mat2

    print(str(m3))
    return m3

m1 = np.array(
[[ 9.99490481e-01,  4.70476824e-04, -3.19159179e-02, -2.24022906e-03],
 [-5.27567188e-04,  9.99998277e-01, -1.78020684e-03, -7.34188367e-03],
 [ 3.19150001e-02,  1.79618396e-03,  9.99488945e-01,  8.67056969e-04],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
)

m2 = np.array(
    [[-0.94230136, 0.06511499, 0.32837201, 0.06639522],
     [0.07120732, 0.99744004, 0.00654885, -0.00968835],
     [-0.32710497, 0.02955348, -0.94452577, 0.12580145],
     [0., 0., 0., 1.]]
)
m3 =  np.array(
    [[6.09750375e-01, 3.38822651e-01, 7.16521941e-01, -2.28422495e-03],
     [-5.64042232e-01, -4.49618111e-01, 6.92603722e-01, 2.72346680e-04],
     [5.56831070e-01, -8.26464014e-01, -8.30445274e-02, 2.77832139e-02],
     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

)
# m3 = from_static_and_frame0_to_final(m1, m2)

r, t = decompose_RT(m3)
w, x, y, z = r
print(w,",", x, "," , y, ",", z)
print(t)
"""
use calc m3 as m1 is the the mat calc from the camera from the qr board, and the other one（m2） is RT detected from the
single other-race single qr code for R0 T0 calculation
"""
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