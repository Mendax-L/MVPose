import torch
import math

# 定义从旋转矩阵转换为欧拉角的函数
def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (XYZ order).
    """
    assert R.shape == (3, 3)
    
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z])

if __name__ == "__main__":
    # 定义已知的旋转矩阵和对应的欧拉角
    known_rotations = [
        (torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]), torch.tensor([math.pi/2, 0.0, 0.0])),
        (torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), torch.tensor([0.0, 0.0, -math.pi/2])),
        (torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]), torch.tensor([0.0, -math.pi/2, 0.0]))
    ]

    # 验证代码
    for rotation_matrix, expected_euler_angles in known_rotations:
        computed_euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
        print("Rotation Matrix:\n", rotation_matrix)
        print("Expected Euler Angles:", expected_euler_angles)
        print("Computed Euler Angles:", computed_euler_angles)
        print("Difference:", computed_euler_angles - expected_euler_angles)
        print()
