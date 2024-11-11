import numpy as np
import open3d as o3d

def generate_point_cloud_from_depth_rgb(depth_image, rgb_image, intrinsics):
    height, width = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    points = []
    colors = []
    
    for v in range(height):
        for u in range(width):
            z = depth_image[v, u]
            if z == 0:  # 跳过无效深度
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append((x, y, z))
            colors.append(rgb_image[v, u] / 255.0)  # 归一化颜色

    points = np.array(points)
    colors = np.array(colors)

    # 创建点云并保存为 PLY
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud("output.ply", point_cloud)
    print("PLY 文件已保存！")

# 示例使用
# depth_image = ... # 深度图的numpy数组
# rgb_image = ...   # RGB图的numpy数组
# intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) # 相机内参
# generate_point_cloud_from_depth_rgb(depth_image, rgb_image, intrinsics)
