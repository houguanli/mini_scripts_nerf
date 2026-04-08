import numpy as np
import transforms3d.quaternions as t3d_quat

def relative_rotation_error_quat(q1_xyzw, q2_xyzw, degrees=True):
    """
    输入:
        q1_xyzw, q2_xyzw : [x, y, z, w] 格式的四元数 (需是单位四元数)
        degrees          : 若 True 则返回角度(度)，否则返回弧度

    输出:
        两个旋转之间的相对旋转角 (标量)

    算法:
        1) 将 [x,y,z,w] 转为 transforms3d 需要的 [w,x,y,z]
        2) quat2mat 得到 R1, R2
        3) R_rel = R1^T * R2
        4) theta = arccos((trace(R_rel) - 1) / 2)
        5) 返回 theta(度) 或 theta(弧度)
    """
    # 1) 适配 transforms3d 的四元数顺序
    wxyz1 = [q1_xyzw[3], q1_xyzw[0], q1_xyzw[1], q1_xyzw[2]]
    wxyz2 = [q2_xyzw[3], q2_xyzw[0], q2_xyzw[1], q2_xyzw[2]]

    # 2) 转为旋转矩阵
    R1 = t3d_quat.quat2mat(wxyz1)  # shape (3,3)
    R2 = t3d_quat.quat2mat(wxyz2)  # shape (3,3)

    # 3) 计算相对旋转 R_rel
    R_rel = R1.T @ R2  # (3,3)

    # 4) 用迹公式求出旋转角, 并做数值截断防止浮点误差
    trace_val = np.trace(R_rel)
    c = (trace_val - 1.0) / 2.0
    c = max(min(c, 1.0), -1.0)  # 避免 arccos 出现超出 [-1,1] 的浮点问题
    theta = np.arccos(c)

    # 若要返回角度，则转为度
    if degrees:
        theta = np.degrees(theta)
    return theta

# ------------------ 测试示例 ------------------
if __name__ == "__main__":
    # 假设有两个四元数 (x,y,z,w)
    q1 = [-0.03880798482805574, 0.9290876532494491, 0.16375871604495607, 0.3293526314033707]  # gt
    q2 = [  -0.0334,  0.9223,  0.1784,  0.3384] #cmp

    err_deg = relative_rotation_error_quat(q1, q2, degrees=True)
    print(f"Relative rotation error = {err_deg:.4f} dre")

"""
gt q:
bunny122
-0.0458062169708968 , 0.4341922640082269 , 0.8995315476898905 , -0.01489506874289095
0.24411868898530442 , -0.0074502091568214406 , 0.03143932579988373


"""