import numpy as np
from scipy.spatial.transform import Rotation as R

def average_rotation_matrices(rot_matrices):
    """
    输入多个旋转矩阵的集合，将它们转换为四元数并计算平均旋转，返回平均后的旋转矩阵。
    
    Parameters:
        rot_matrices (numpy.ndarray): 形状为 (N, 3, 3)，包含 N 个旋转矩阵。
        
    Returns:
        numpy.ndarray: 平均旋转矩阵，形状为 (3, 3)。
    """
    # Step 1: 将旋转矩阵转换为四元数
    quaternions = [R.from_matrix(rot).as_quat() for rot in rot_matrices]
    
    # Step 2: 对四元数进行平均
    quaternions = np.array(quaternions)
    mean_quaternion = np.mean(quaternions, axis=0)
    
    # Step 3: 对平均四元数进行归一化
    mean_quaternion /= np.linalg.norm(mean_quaternion)
    
    # Step 4: 将平均四元数转换为旋转矩阵
    mean_rotation = R.from_quat(mean_quaternion).as_matrix()
    
    return mean_rotation

# 示例旋转矩阵
rot_matrices = [
    np.array([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]]),
    np.array([[1, 0, 0], [0, 0.866, -0.5], [0, 0.5, 0.866]])
]

# 计算平均旋转矩阵
average_rotation = average_rotation_matrices(rot_matrices)
print("平均旋转矩阵:")
print(average_rotation)
