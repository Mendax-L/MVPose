import os
import json
import threading
from PIL import Image
import os
import cv2
import numpy as np

# 假设每个视图中只有一个物体

# 切换到脚本所在的目录
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)
with open('/home/mendax/project/SATPose/datasets/lmo/pbr/bop_data/lmo/train_pbr/000000/scene_camera.json') as f:
    scene_camera = json.load(f)

cam_data = scene_camera['0']
cam_K = np.array(cam_data['cam_K']).reshape(3, 3)
fx, fy = cam_K[0, 0], cam_K[1, 1]
cx, cy = cam_K[0, 2], cam_K[1, 2]


def apply_mask_to_rgb(rgb, mask):
    r, g, b = rgb.split()
    # 将 mask 转换为二进制掩码，以确保只有 0 和 255 值
    binary_mask = mask.point(lambda p: p > 0 and 255)
    
    # 将 R、G、B 通道中 mask 为 0 的区域设置为 0
    r = Image.composite(r, Image.new("L", r.size, 0), binary_mask)
    g = Image.composite(g, Image.new("L", g.size, 0), binary_mask)
    b = Image.composite(b, Image.new("L", b.size, 0), binary_mask)

    # 合并 R、G、B 通道和 mask 通道为一个四通道图像
    rgb = Image.merge("RGB", (r, g, b))
    return rgb

def apply_mask_to_depth(depth, mask):
    # 将深度图转换为 numpy 数组，以便进行数值操作
    depth_array = np.array(depth).astype(np.float32)
    
    # 将 depth 映射到 0-255 区间
    min_val, max_val = depth_array.min(), depth_array.max()
    if max_val > min_val:  # 避免除以零
        depth_array = (depth_array - min_val) / (max_val - min_val) * 255.0
    depth_array = depth_array.astype(np.uint8)  # 转换为8位图像

    # 将处理后的深度图重新转换为 PIL 图像
    depth_normalized = Image.fromarray(depth_array)

    # 将 mask 转换为二进制掩码，以确保只有 0 和 255 值
    binary_mask = mask.point(lambda p: 255 if p > 0 else 0)
    
    # 使用掩码将指定区域设置为0
    masked_depth = Image.composite(depth_normalized, Image.new("L", depth.size, 0), binary_mask)

    return masked_depth



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

def generate_point_cloud(rgb, depth, fx, fy, cx, cy):
    print(depth.shape)
    h, w = depth.shape
    points = []
    colors = []
    max_depth_val = 65535  # 假设这是uint16的最大值  
    scale_factor = 1.0  # 假设这是深度图像的比例因子  

    for v in range(h):
        for u in range(w):
            z = depth[v, u] / scale_factor  
            if z > 0 : 
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                colors.append(rgb[v, u])

    return np.array(points), np.array(colors)





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


def render_image(points, colors, img_size):
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    print(img.shape)
    for i, (point, color) in enumerate(zip(points, colors)):
        # print(i)
        v, u = int(point[1]*fy/point[2] + cy), int(point[0]*fx/point[2] + cx)
        if 0 <= u < img_size[1] and 0 <= v < img_size[0]:
            img[v, u] = color
    return img

def crop_and_save(image, crop_area, output_dir, filename):  
    cropped_image = image.crop(crop_area)  
    output_path = os.path.join(output_dir, filename)  
    cropped_image.save(output_path) 

# 函数生成中心点数据并保存裁剪图像
def centerxy_data_generate(target_dir):
    # 创建存储裁剪结果和中心点文件的目录
    mincrop_dir_path = f"{target_dir}/mincrop"
    centercrop_dir_path = f"{target_dir}/centercrop"
    depthcrop_dir_path = f"{target_dir}/depthcrop"
    newview_dir_path = f"{target_dir}/newview"
    newviewcrop_dir_path = f"{target_dir}/newviewcrop"
    if not os.path.exists(mincrop_dir_path):
        os.makedirs(mincrop_dir_path)
        print(f"目录 {mincrop_dir_path} 已创建。")
    if not os.path.exists(centercrop_dir_path):
        os.makedirs(centercrop_dir_path)
        print(f"目录 {centercrop_dir_path} 已创建。")
    if not os.path.exists(depthcrop_dir_path):
        os.makedirs(depthcrop_dir_path)
        print(f"目录 {depthcrop_dir_path} 已创建。")      
    if not os.path.exists(newview_dir_path):
        os.makedirs(newview_dir_path)
        print(f"目录 {newview_dir_path} 已创建。")  
    if not os.path.exists(newviewcrop_dir_path):
        os.makedirs(newviewcrop_dir_path)
        print(f"目录 {newviewcrop_dir_path} 已创建。")  
    
    # 读取中心点数据和边界框信息
    with open(f"{target_dir}/object_info.json") as f:
        items = json.load(f)
        # 获取中心点的坐标（以索引 "0" 为例）

    for item in items:
        img_id = item['img_id']
        obj_idx = item['idx']
        minxyxy = item['minxyxy']
        centerxyxy = item['centerxyxy']
        newviewxyxy = item['minxyxy_newview']
        cam_t_m2c = item['cam_t_m2c']
        # 加载 RGB 图像
        rgb_path = f"{target_dir}/rgb/{str(img_id).zfill(6)}.png"
        mask_path = f"{target_dir}/mask_visib/{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"
        depth_path = f"{target_dir}/depth/{str(img_id).zfill(6)}.png"
        newview_path = f"{target_dir}/newview/{str(img_id).zfill(6)}.png"

        # r, g, b = rgb.split()
        # # 合并 R、G、B 通道和 mask 通道为一个四通道图像
        # rgba = Image.merge("RGBA", (r, g, b, mask))


        rgb = Image.open(rgb_path)
        mask = Image.open(mask_path)
        depth = Image.open(depth_path)


        w = 50
        z_c = cam_t_m2c[2]
        rgb_np, depth_np = np.array(rgb), np.array(depth)
        points, colors = generate_point_cloud(rgb_np, depth_np, fx, fy, cx, cy)
        rotation = get_rotation_yaw(z_c,w) 
        translation = np.array([-w, 0, 0])
        new_points = project_points(points, rotation, translation)
        new_image = render_image(new_points, colors, rgb_np.shape[:2])
        newview = Image.fromarray(new_image)
        newview.save(newview_path)





        # masked_rgb = apply_mask_to_rgb(rgb, mask)
        # masked_depth = apply_mask_to_depth(depth, mask)

        threads = []  
        
        # 为最小区域裁剪创建一个线程  
        t1 = threading.Thread(target=crop_and_save, args=(rgb, minxyxy, mincrop_dir_path, f"{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"))  
        threads.append(t1)  
        
        # 为中心区域裁剪创建一个线程  
        t2 = threading.Thread(target=crop_and_save, args=(rgb, centerxyxy, centercrop_dir_path, f"{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"))  
        threads.append(t2)  
        
        # 为新视角区域裁剪创建一个线程  
        t3 = threading.Thread(target=crop_and_save, args=(newview, newviewxyxy, newviewcrop_dir_path, f"{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"))  
        threads.append(t3)  
        
        # 启动所有线程  
        for t in threads:  
            t.start()  
        
        # 等待所有线程完成  
        for t in threads:  
            t.join()
        # squarecrop = masked_rgb.crop(squarexyxy)
        # squarecrop_path = os.path.join(squarecrop_dir_path, f"{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png")
        # squarecrop.save(squarecrop_path)

        # depthcrop = masked_depth.crop(centerxyxy)
        # depthcrop_path = os.path.join(depthcrop_dir_path, f"{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png")
        # depthcrop.save(depthcrop_path)

if __name__ == "__main__":
    target_dir = f'datasets/lmo/test/000002'
    centerxy_data_generate(target_dir)
    target_dir = '/home/mendax/project/SATPose/datasets/lmo/pbr/bop_data/lmo/train_pbr/000000'
    centerxy_data_generate(target_dir)
    obj_ids = [1,5,6,8,9,10,11,12]
    for obj_id in obj_ids:
        target_dir = f'datasets/lm/{str(obj_id).zfill(6)}'  # RGB 图像目录
        # target_dir = f'datasets/lmo/test/000002'  # RGB 图像目录
        # target_dir = '/home/mendax/project/SATPose/datasets/lmo/pbr/bop_data/lmo/train_pbr/000000'

        centerxy_data_generate(target_dir)

