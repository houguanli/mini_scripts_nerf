import math
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


def quat_to_matrix(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])

def quaternion_to_euler(w, x, y, z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x axis, pitch is rotation around y axis
    and yaw is rotation around z axis.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # use 90 degrees if out of range
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert euler angles (roll, pitch, yaw) to a quaternion.
    Roll, pitch, and yaw are given in radians.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return w, x, y, z


# Example quaternion
q =[-0.059238496177121146 , 0.998243853695083 , 0.0 , 0.0]
w, x, y, z = q[0], q[1], q[2], q[3]
# Convert quaternion to euler angles
roll, pitch, yaw = quaternion_to_euler(w, x, y, z)

# Convert radians to degrees
roll_deg = math.degrees(roll)
pitch_deg = math.degrees(pitch)
yaw_deg = math.degrees(yaw)
# print(math.cos())
print("Roll: {:.2f} degrees".format(roll_deg))
print("Pitch: {:.2f} degrees".format(pitch_deg))
print("Yaw: {:.2f} degrees".format(yaw_deg))
delat_P = [0,0,0]
roll = math.radians(-36 + delat_P[0])  # Roll: rotation around the x-axis
pitch = math.radians(180+ delat_P[1])  # Pitch: rotation around the y-axis
yaw = math.radians(0 + delat_P[2])  # Yaw: rotation around the z-axis
# roll, pitch, yaw = [math.radians(16.3), math.radians(-1.1), math.radians(60)]
# Convert euler angles to quaternion
w, x, y, z = euler_to_quaternion(roll, pitch, yaw)

print("Quaternion:")
print(w,",", x, "," , y, ",", z)

q = [5.823541592445462e-17 , -1.892183365217075e-17 , 0.9510565162951535 , 0.3090169943749474]
T = [0, 1, -1]

RT = np.eye(4)
RT[:3, :3] = quat_to_matrix(q)
RT[:3, 3] = T
# RT = np.linalg.inv(RT)
RT2str = '[' + ',\n '.join('[' + ', '.join(str(x) for x in row) + ']' for row in RT) + ']'
print(RT2str)


RT4 = RT
RT4[:, 1:3] = RT4[:, 1:3] * -1
RT42str = '[' + ',\n '.join('[' + ', '.join(str(x) for x in row) + ']' for row in RT4) + ']'
print(RT42str)

q = [0.9150, -0.2691, -0.1273,  0.2763]
T = [0.1536, -0.1478,  0.3126]

RT2 = np.eye(4)
RT2[:3, :3] = quat_to_matrix(q)
RT2[:3, 3] = T
RT22str = '[' + ',\n '.join('[' + ', '.join(str(x) for x in row) + ']' for row in RT2) + ']'
print(RT22str)




RT3 = np.matmul(RT2, RT)
RT22str = '[' + ',\n '.join('[' + ', '.join(str(x) for x in row) + ']' for row in RT3) + ']'
q, t = decompose_RT(RT3)
print(RT22str)
w, x, y, z = q
print(w,",", x, "," , y, ",", z)
x, y, z = t
print(x, "," , y, ",", z)

RT4 = RT3
RT4[:, 1:3] = RT4[:, 1:3] * -1
RT42str = '[' + ',\n '.join('[' + ', '.join(str(x) for x in row) + ']' for row in RT4) + ']'
print(RT42str)
