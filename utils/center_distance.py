import numpy as np

def Center_Distance(R, t, u1, v1, u2, v2, K1, K2):
    # 提取旋转矩阵和平移向量
    r31, r32, r33 = R[2, :]
    r11, r12, r13 = R[0, :]
    r21, r22, r23 = R[1, :]
    tx, ty, tz = t

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

# 示例输入
R = np.array([[0.99622298, 0, -0.08683191],  # 旋转矩阵 R
              [0.0171, 0, 0.0147],
              [0.0104, -0.0144, 0.9999]])
R = np.eye(3)

t = np.array([0.1,0, 0])  # 平移向量 t

u1, v1 = 320, 240  # 第一相机的像素坐标 (u1, v1)
u2, v2 = 310, 250  # 第二相机的像素坐标 (u2, v2)

# 相机内参矩阵
K1 = np.array([[1000, 0, 320],  # 第一相机的内参矩阵
               [0, 1000, 240],
               [0, 0, 1]])

K2 = np.array([[1000, 0, 320],  # 第二相机的内参矩阵
               [0, 1000, 240],
               [0, 0, 1]])

# 计算 Z
Z = Center_Distance(R, t, u1, v1, u2, v2, K1, K2)
print("Center_Distance:", Z)
