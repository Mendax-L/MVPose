def Center_Distance(R, t, uv1, uv2, K1, K2):
    # 提取旋转矩阵和平移向量
    r31, r32, r33 = R[2, :]
    r11, r12, r13 = R[0, :]
    r21, r22, r23 = R[1, :]
    tx, ty, tz = t

    # 提取坐标点 u1, v1 和 u2, v2
    u1, v1 = uv1
    u2, v2 = uv2

    # 提取内参矩阵 K1 和 K2
    fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
    fx2, fy2, cx2, cy2 = K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]

    # 计算 A1
    A1 = (r31 * (u1 - cx1) * (u2 - cx2)) / (fx1 * fx2) + (r32 * (v1 - cy1) * (u2 - cx2)) / (fy1 * fx2) + (r33 * (u2 - cx2)) / fx2 - (r11 * (u1 - cx1)) / fx1 - (r12 * (v1 - cy1)) / fx1 - r13

    # 计算 A2
    A2 = (r31 * (u1 - cx1) * (v2 - cy2)) / (fx1 * fy2) + (r32 * (v1 - cy1) * (v2 - cy2)) / (fy1 * fy2) + (r33 * (v2 - cy2)) / fy2 - (r21 * (u1 - cx1)) / fy1 - (r22 * (v1 - cy1)) / fy1 - r23

    # 计算 b1 和 b2
    b1 = tx - (tz * (u2 - cx2)) / fx2
    b2 = ty - (tz * (v2 - cy2)) / fy2

    # 计算 Z
    Z = (A1 * b1 + A2 * b2) / (A1**2 + A2**2)

    return Z

def Depth_From_Multiviews(CR_list , Ct_list, uv_list, Kc_list):

    Z = []
    uv_master = uv_list[0]
    Kc_master = Kc_list[0]
    for i, (CR, Ct, uv, Kc) in enumerate(zip(CR_list, Ct_list, uv_list, Kc_list)):
        if i == 0: 
            continue
        else:
            Z = Center_Distance(CR, Ct, uv_master, uv, Kc_master, Kc)
    return sum(Z) / len(Z)