import os
import sys

from lib import to_allo
from lib.depth_from_multiviews import Depth_From_Multiviews
sys.path.insert(0,os.getcwd()) # 把当前路径添加到 sys.path 中

import csv
import json
import time
import numpy as np
import torch
from torchvision import transforms
from lib.SATRot import SATRot
from utils.loader.test_dataloader import get_test_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write2csv(save_path, scene_id, im_id, obj_id, score, R, t, time):
    # 确保目录存在
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    batch_size = scene_id.shape[0]  # 获取 batch_size
    # print(f"scene_id: {scene_id.shape}, im_id: {im_id.shape}, obj_id: {obj_id.shape}, score: {score.shape}, R: {R.shape}, t: {t.shape}, time: {time.shape}")
    # 将数据按批次逐行写入 CSV 文件
    with open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # 遍历每个 batch 的数据
        for i in range(batch_size):
            row = [
                scene_id[i].cpu().item(),               # scene_id 单个元素
                im_id[i].cpu().item(),                  # im_id 单个元素
                obj_id[i].cpu().item(),                 # obj_id 单个元素
                score[i].cpu().item(),                  # score 单个元素
                ' '.join(map(str, R[i].detach().flatten().cpu().numpy())),  # 旋转矩阵 R (展平并以空格分隔)
                ' '.join(map(str, t[i].detach().flatten().cpu().numpy())),  # 平移向量 t (展平并以空格分隔)
                time[i].cpu().item() if isinstance(time, torch.Tensor) else time  # time 单个元素
            ]
            writer.writerow(row)  # 写入一行数据

    print(f"Batch data written to {save_path}")


def test4bop(target_dir = 'datasets/lmo/test/000002', obj_id = 1, save_dir = 'results/lmo/MVPose_lmo-test.csv'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dir = target_dir  # RGB 图像目录
    
    test_loader = get_test_dataloader(target_dir, obj_id)

    # 加载自定义网络模型

    R_net = SATRot(d_model = 240, nhead=4, num_layers=4, num_samples = 8).to(device)
    R_net.load_state_dict(torch.load(f"weights/SATRot_obj_{obj_id}.pth", map_location=device))
    R_net.eval()


    # 打印部分数据样本
    for scene_id, img_id, obj_id, multiviews_cropimg, cropxyxy, uv_gt, R_gt, t_gt in test_loader:
        scene_id, img_id, obj_id, multiviews_cropimg, cropxyxy, uv_gt, R_gt, t_gt = scene_id.to(device), img_id.to(device), obj_id.to(device), half_diameter.to(device), minxyxy.to(device), rgb.to(device), mask.to(device), depth_img.to(device), t_gt.to(device), centerxyxy_gt.to(device), centeruv_gt.to(device), centercrop_gt.to(device), R_gt.to(device)


        uv_preds, R_preds = R_net(multiviews_cropimg)
        R_preds = to_allo(R_preds,uv_preds)

        uvpreds = (uv_preds[0]* + cropxyxy[0, 0:2], uv_preds[1] + cropxyxy[0, 2:4])
        
        R_pred = R_preds[0]# 转换成四元数求平均
        d_pred = Depth_From_Multiviews(CR_list , Ct_list, uv_list, Kc_list)


        fx,fy, cx,cy = Kc_list[0][0,0],Kc_list[0][1,1],Kc_list[0][0,2],Kc_list[0][1,2]
        x_pred, y_pred = fx * uv_pred[0] / d_pred + cx, fy * uv_pred[1] / d_pred + cy
        t_pred = (x_pred, y_pred, d_pred)

        print(f'R_pred:{R_pred}')
        print(f'R_gt:{R_gt[0]}')        
        print(f't_pred:{t_pred}')
        print(f't_gt:{t_gt[0]}')       
        print(f'uv_pred:{uv_preds[0]}')
        print(f'uv_gt:{uv_gt[0]}')


        score=torch.tensor(1).repeat(scene_id.shape[0])
        elapsed_time=torch.tensor(-1).repeat(scene_id.shape[0])


        write2csv(save_dir,scene_id, img_id, obj_id, score,R_gt,t_pred,elapsed_time)
            


if __name__ == '__main__':
        # 清空指定文件
    file_path = 'results/lmo/SplitPose_lmo-test.csv'
    obj_ids = [1,5,6,8,9,10,11,12]
    if os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass  # 清空文件内容
    for obj_id in obj_ids:
        test4bop(obj_id = obj_id)
        