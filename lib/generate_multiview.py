import json
import numpy as np
import os
from config import views


# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

# 确认当前工作目录
print(f"当前工作目录: {os.getcwd()}")
# obj_ids = [1,5,6,8,9,10,11,12]
obj_id = 12
target_dir = f'../Datasets/lm/{str(obj_id).zfill(6)}'
# target_dir = f'../Datasets/lmo/test/000002'
# target_dir = '/home/mendax/project/SATPose/datasets/lmo/pbr/bop_data/lmo/train_pbr/000000'
model_dir = f"../Datasets/lmo/models"

with open(f'../Datasets/lmo/test_targets_bop19.json') as f:
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
img_w = 640
img_h = 480

# 遍历所有编号的视图

for view_id, (Rc, tc) in views.items():
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
            # 获取相机内参
            Kc = np.array(cam_data['cam_K']).reshape(3, 3)
            fx, fy = Kc[0, 0], Kc[1, 1]
            cx, cy = Kc[0, 2], Kc[1, 2]

            # 获取平移向量 cam_t_m2c 和物体的 ID
            if np.array(gt_data['obj_id'])!= obj_id:
                raise IndexError("obj_id 不匹配")
            

            Ra = np.array(gt_data['cam_R_m2c']).reshape(3, 3)
            ta = np.array(gt_data['cam_t_m2c']).reshape(3, 1)
            tc = tc.reshape(3, 1)

            Rc_inv = np.linalg.inv(Rc)
            Rb = Rc_inv @ Ra
            tb = (Rc_inv @ (ta - tc))

            R = Rb.tolist()
            t = tb.flatten().tolist()


            t_x,t_y,t_z = t[0],t[1],t[2]
            # 计算物体中心在图像中的投影坐标
            uv = (fx * (t_x / t_z) + cx, fy * (t_y / t_z) + cy)


            # 获取 bbox_obj 的 xyxy 信息
            # bbox = gt_info['bbox_obj']

            diameter = models[str(obj_id)]['diameter']
            r = diameter / 2
            
            cors = [[t_x-r,t_y-r,t_z],[t_x+r,t_y+r,t_z]]
            bbox = []
            for i, cor in enumerate(cors):
            # print(i)
                bbox.append(int(cor[0]*fx/cor[2] + cx)) 
                bbox.append(int(cor[1]*fy/cor[2] + cy))
            print(bbox)
            bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            uv_relative = ((uv[0] - bbox[0]) / bbox_w, (uv[1] - bbox[1]) / bbox_h)

            # 检查投影点是否在图像范围内（可选）
            if 0 <= uv[0] < img_w and 0 <= uv[1] < img_h:
                obj_info = {
                    'img_id':int(img_id),
                    'view_id': view_id,
                    'idx':idx,                
                    'obj_id': obj_id,
                    'uv': uv,
                    'uv_relative': uv_relative,
                    'bbox': bbox,
                    'view_R': Rc.tolist(),
                    'view_t': tc.flatten().tolist(),
                    'R': R,
                    't': t,
                    'scene_id': scene_id,
                    'diameter': diameter,
                    'Kc':Kc.tolist()
                }
            else:
                obj_info = {
                    'img_id':int(img_id),
                    'view_id': view_id,
                    'idx':idx,                
                    'obj_id': obj_id,
                    'uv': uv,
                    'uv_relative': uv_relative,
                    'bbox': bbox,
                    'view_R': Rc.tolist(),
                    'view_t': tc.flatten().tolist(),
                    'R': R,
                    't': t,
                    'scene_id': scene_id,
                    'diameter': diameter,
                    'Kc':Kc.tolist(),
                    'note': 'center outside image bounds'
                }

            # 将结果存入字典
            results.append(obj_info)

    # 将结果保存为 JSON 文件
    with open(f'{target_dir}/view{view_id}_info.json', 'w') as f_out:
        json.dump(results, f_out, indent=4)

    print(f"新视角物体位姿相关信息已成功计算并保存至 {target_dir}/view{view_id}_info.json")
