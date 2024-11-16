import numpy as np

# 将数据转为矩阵（默认行优先）
cam_R_m2c = np.array([
    0.6155426282490462, -0.7872002747152219, -0.03771988906432555,
    -0.19894986552077276, -0.10889926681888931, -0.9739401059834828,
    0.7625789371251613, 0.6070060649389416, -0.22364550906920733
])

# 行优先解析为 3x3 矩阵
# R = cam_R_m2c.reshape(3, 3, order='F')
R = cam_R_m2c.reshape(3, 3)
print("R :\n", R)
print("R*R.T :\n", R@(R.T))

# 验证列向量是否是单位向量
for i in range(3):
    norm = np.linalg.norm(R[:, i])  # 计算每列的范数
    print(f"Norm of column {i+1}: {norm:.6f} (Should be 1)")

# 验证列向量之间是否正交
for i in range(3):
    for j in range(i + 1, 3):
        dot_product = np.dot(R[:, i], R[:, j])  # 计算两列的点积
        print(f"Dot product of column {i+1} and column {j+1}: {dot_product:.6f} (Should be 0)")
