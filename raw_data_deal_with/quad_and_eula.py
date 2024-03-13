import math

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
w, x, y, z = 0.1, 0.2, 0.3, 0.4
q = [0.52133467, -0.01417546, -0.08058844,  0.84942025]
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
delat_P = [-80, -20, 340]
roll = math.radians(-8.74 + delat_P[0])  # Roll: rotation around the x-axis
pitch = math.radians(-3.44+ delat_P[1])  # Pitch: rotation around the y-axis
yaw = math.radians(-117.18 + delat_P[2])  # Yaw: rotation around the z-axis

# Convert euler angles to quaternion
w, x, y, z = euler_to_quaternion(roll, pitch, yaw)

print("Quaternion:")
print(w,",", x, "," , y, ",", z)
