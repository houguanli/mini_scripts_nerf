import numpy as np
def calcm3(mat1, mat2):
    m2_inv = np.linalg.inv(mat2)
    # m3 = m2_inv @ mat1
    m3 = mat1 @ m2_inv

    print(str(m3))
    return m3

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
calcm3(m1, m2)
# use calc m3 as m1 is the the mat calc from the camera from the qr board, and the other one is RT detected from the
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