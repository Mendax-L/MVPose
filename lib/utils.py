from tkinter import Image

import numpy as np

# !!!!!!!注意逆矩阵转换
def project_points(points, rotation, translation):  
    # 创建4x4齐次变换矩阵  
    transform_matrix = np.zeros((4, 4))  
    transform_matrix[:3, :3] = rotation  
    transform_matrix[:3, 3] = translation  
    transform_matrix[3, 3] = 1  
  
    # 将点扩展为齐次坐标  
    ones = np.ones((points.shape[0], 1))  
    points_homogeneous = np.hstack((points, ones))  
  
    # 应用变换矩阵  
    projected_points_homogeneous = points_homogeneous @ transform_matrix.T  
  
    # 归一化到非齐次坐标（除以w分量）  
    w = projected_points_homogeneous[:, 3]  
    projected_points = projected_points_homogeneous[:, :3] / w[:, np.newaxis]  
  
    # 返回投影后的点（可以根据需要返回x, y, z或仅x和y）  
    return projected_points  # 只返回x和y坐标 

def get_rotation_yaw(z_c,w):
    slide =np.sqrt(z_c**2 + w**2)
    c = z_c/slide
    s = w/slide
    
    # 生成旋转矩阵
    rotation_matrix = np.array([
        [c, 0, -s],  # X 轴分量与 Z 轴分量
        [0, 1, 0],  # Y 轴保持不变
        [s, 0, c]  # Z 轴分量的负值
    ])
    return rotation_matrix

def get_min_bbox(mask):
        # 如果 mask 是 PIL 图像，则转换为 numpy 数组
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    # 查找mask中非零像素的坐标
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return None  # 如果没有有效的mask数据

    # 获取最小的外接矩形
    y_min, x_min = np.min(coords, axis=0)
    y_max, x_max = np.max(coords, axis=0)

    # 计算正方形的四个边界点
    bounding_box = (x_min, y_min, x_max, y_max)

    # 处理边界条件，确保bounding_box不超出图像边界
    img_h, img_w = mask.shape
    x_min = max(bounding_box[0], 0)
    y_min = max(bounding_box[1], 0)
    x_max = min(bounding_box[2], img_w -1)
    y_max = min(bounding_box[3], img_h -1)

    return (int(x_min), int(y_min), int(x_max), int(y_max))