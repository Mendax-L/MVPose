import json
from PIL import Image
import numpy as np
import os

# # 切换到脚本所在的目录
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

# 确认当前工作目录
print(f"当前工作目录: {os.getcwd()}")
# obj_ids = [1,5,6,8,9,10,11,12]
obj_id = 12
target_dir = f'datasets/lm/{str(obj_id).zfill(6)}'
# target_dir = f'datasets/lmo/test/000002'
# target_dir = '/home/mendax/project/SATPose/datasets/lmo/pbr/bop_data/lmo/train_pbr/000000'
model_dir = f"datasets/lmo/models"

with open(f'datasets/lmo/test_targets_bop19.json') as f:
    test_targets = json.load(f)
# 初始化一个空字典
scene_id = 0 
grouped_by_im_id = {}

# 遍历 test_targets 列表
for entry in test_targets:
    im_id = str(entry["im_id"])
    obj_id = entry["obj_id"]
    
    # 如果 im_id 不在字典中，初始化为一个空列表
    if im_id not in grouped_by_im_id:
        grouped_by_im_id[im_id] = []
    
    # 将 obj_id 添加到对应的 im_id 列表中
    grouped_by_im_id[im_id].append(obj_id)
# print(grouped_by_im_id)
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


# 读取 scene_camera.json、scene_gt.json 和 scene_gt_info.json 文件
with open(f'{target_dir}/scene_camera.json') as f_cam, \
     open(f'{target_dir}/scene_gt.json') as f_gt, \
     open(f'{target_dir}/scene_gt_info.json') as f_info, \
     open(f'{model_dir}/models_info.json') as models_info:

    scene_camera = json.load(f_cam)
    scene_gt = json.load(f_gt)
    scene_gt_info = json.load(f_info)
    models = json.load(models_info)

# 初始化结果
results = []

# 图像的宽度和高度
img_width = 640
img_height = 480

# 遍历所有编号的视图
for img_id, objs in scene_gt.items():
    # if img_id not in grouped_by_im_id:
    #    continue
    for idx, obj in enumerate(objs):
        obj_id = obj['obj_id']
        # if obj_id not in grouped_by_im_id[img_id]:
        #     continue
        img_id = str(img_id)  # 确保键是字符串
        cam_data = scene_camera[img_id]
        gt_data = scene_gt[img_id][idx]  # 假设每个视图中只有一个物体
        gt_info = scene_gt_info[img_id][idx]  # 获取 bbox 信息
        mask_path = f"{target_dir}/mask/{str(img_id).zfill(6)}_{str(idx).zfill(6)}.png"
        mask = Image.open(mask_path)
        # 获取相机内参
        cam_K = np.array(cam_data['cam_K']).reshape(3, 3)
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]

        # 获取平移向量 cam_t_m2c 和物体的 ID
        if np.array(gt_data['obj_id'])!= obj_id:
            raise IndexError("obj_id 不匹配")
        cam_t_m2c = np.array(gt_data['cam_t_m2c'])  # [tx, ty, tz]
        obj_id = gt_data['obj_id']
        tx, ty, tz = cam_t_m2c
        cam_R_m2c = np.array(gt_data['cam_R_m2c']).tolist()  # 3x3 rotation matrix
        cam_t_m2c = cam_t_m2c.tolist()

        # 计算物体中心在图像中的投影坐标
        u = fx * (tx / tz) + cx
        v = fy * (ty / tz) + cy

        u = int(u)
        v = int(v)

        

        img_w, img_h = mask.size
        # print(img_h, img_w)
        # 获取 bbox_obj 的 xyxy 信息
        bbox_obj = gt_info['bbox_obj']
        xmin, ymin, width, height = bbox_obj
        side = max(width, height)
        xmax = xmin + width
        ymax = ymin + height
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_w - 1, xmax)
        ymax = min(img_h - 1, ymax)
        minxyxy = (xmin, ymin, xmax, ymax)


        # minxyxy = get_min_bbox(mask)
        # width, height = minxyxy[2]-minxyxy[0], minxyxy[3]-minxyxy[1]
        halfside = int(max(abs(minxyxy[2]-u), abs(minxyxy[0]-u), abs(minxyxy[3]-v), abs(minxyxy[1]-v)))
        xmin, ymin = u - halfside , v - halfside
        xmax, ymax = u + halfside , v + halfside
        # xmin = max(0, xmin)
        # ymin = max(0, ymin)
        # xmax = min(img_w - 1, xmax)
        # ymax = min(img_h - 1, ymax)
        centerxyxy = (xmin, ymin, xmax, ymax)
        

        w=50
        print(tx,ty,tz)
        rotation = get_rotation_yaw(tz,w) 
        translation = np.array([-w, 0, 0])  # 位移向量
        center = np.array([[tx,ty,tz]])
        center_newview = project_points(center, rotation, translation)
        for i, point in enumerate(center_newview):
        # print(i)
            uv_newview = (int(point[0]*fx/point[2] + cx), int(point[1]*fy/point[2] + cy)) 
        # print(uv_newview)
        
        # 计算相对的中心点坐标 u_relative 和 v_relative
        u_relative = (u - minxyxy[0]) / side
        v_relative = (v - minxyxy[1]) / side
        diameter = models[str(obj_id)]['diameter']
        r = diameter / 2

        center_newview = center_newview[0]
        x_c,y_c,z_c = center_newview
        minxyzxyz_newview = [[x_c-r,y_c-r,z_c],[x_c+r,y_c+r,z_c]]
        minxyxy_newview = []
        for i, point in enumerate(minxyzxyz_newview):
        # print(i)
            minxyxy_newview.append(int(point[0]*fx/point[2] + cx)) 
            minxyxy_newview.append(int(point[1]*fy/point[2] + cy))
        print(minxyxy_newview)

        side_newview = max(abs(minxyxy_newview[2]-minxyxy_newview[0]), abs(minxyxy_newview[3]-minxyxy_newview[1]))
        u_newview_relative = (uv_newview[0] - minxyxy_newview[0]) / side_newview
        v_newview_relative = (uv_newview[1] - minxyxy_newview[1]) / side_newview

        # 检查投影点是否在图像范围内（可选）
        if 0 <= u < img_width and 0 <= v < img_height:
            obj_info = {
                'img_id':int(img_id),
                'idx':idx,                
                'obj_id': obj_id,
                'uv':(u,v),
                'uv_newview':uv_newview,
                'uv_relative': (u_relative,v_relative),
                'uv_newview_relative': (u_newview_relative,v_newview_relative),
                'minxyxy':minxyxy,
                'minxyxy_newview':minxyxy_newview,
                'centerxyxy':centerxyxy,           
                'width': width,
                'height': height,
                'cam_R_m2c':cam_R_m2c,
                'cam_t_m2c':cam_t_m2c,
                'scene_id': scene_id,
                'diameter': diameter
            }
        else:
            obj_info = {
                'img_id':img_id,
                'idx':idx,
                'obj_id': obj_id,
                'uv':(u,v),
                'uv_newview':uv_newview,
                'uv_relative': (u_relative,v_relative),
                'uv_newview_relative': (u_newview_relative,v_newview_relative),
                'minxyxy':minxyxy,
                'minxyxy_newview':minxyxy_newview,
                'centerxyxy':centerxyxy,  
                'width': width,
                'height': height,                
                'cam_R_m2c':cam_R_m2c,
                'cam_t_m2c':cam_t_m2c,
                'scene_id': scene_id,
                'diameter': diameter,
                'note': 'center outside image bounds'
            }

        # 将结果存入字典
        results.append(obj_info)

# 将结果保存为 JSON 文件
with open(f'{target_dir}/object_info.json', 'w') as f_out:
    json.dump(results, f_out, indent=4)

print("物体中心坐标和相对坐标已成功计算并保存至 object_centers.json")
