import numpy as np
def calcm3(mat1, mat2):
    m2_inv = np.linalg.inv(mat2)
    m3 = m2_inv @ mat1
    print(str(m3))
    return m3

m2 = np.array([
    [-0.16388532,  0.77591521, -0.60917746, 0.2069921],
     [0.98492957,  0.1633032,  -0.056972,   0.04986992],
     [0.05527518,  -0.60933377, -0.79098483, 0.1501093],
     [0.,          0.,          0.,          1.]]
)

m1 = np.array(
    [[0.96851933,  0.18500824, - 0.16655992,  0.18477099],
     [-0.24893497,  0.72322685, - 0.644185,    0.6132438],
     [0.00128107, 0.66536826, 0.7465142, -0.43042728],
     [0.,          0.,          0.,          1.]]

)
calcm3(m1, m2)
# use calc m3 as m1 is the the mat calc from the camera from the qr board, and the other one is RT detected from the
# single other-race single qr code for R0 T0 calculation
