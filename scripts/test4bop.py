import os
import sys


sys.path.insert(0,os.getcwd()) # 把当前路径添加到 sys.path 中

import csv
import json
import time
import numpy as np
import torch
from torchvision import transforms
from lib.SATRot import SATRot
from lib.to_allo import get_allorot
from utils.loader.test_dataloader import SATVal_loader
from lib.config import test_scene_ids, SATRot_test_transform
from lib import to_allo
from lib.multiviews import t_From_Multiviews

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write2csv(save_path, scene_id, im_id, obj_id, score, R, t, time):
    # 确保目录存在
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # print(f"scene_id: {scene_id.shape}, im_id: {im_id.shape}, obj_id: {obj_id.shape}, score: {score.shape}, R: {R.shape}, t: {t.shape}, time: {time.shape}")
    # 将数据按批次逐行写入 CSV 文件
    with open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # 遍历每个 batch 的数据
        for i in range(len(R)):
            row = [
                scene_id,               # scene_id 单个元素
                im_id[i].cpu().item(),                  # im_id 单个元素
                obj_id,                 # obj_id 单个元素
                score[i],                  # score 单个元素
                ' '.join(map(str, R[i].detach().flatten().cpu().numpy())),  # 旋转矩阵 R (展平并以空格分隔)
                ' '.join(map(str, t[i].detach().flatten().cpu().numpy())),  # 平移向量 t (展平并以空格分隔)
                time[i]  # time 单个元素
            ]
            writer.writerow(row)  # 写入一行数据

    print(f"Batch data written to {save_path}")


    
def test4bop(target_dir = '../Datasets/ycbv/test', scene_ids = test_scene_ids, obj_id = 1, save_dir = 'results/lmo/MVPose_lmo-test.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_net = SATRot(d_model = 120, nhead = 6, num_layers=4,).to(device)
    R_net.load_state_dict(torch.load(f"weights/SATR_ycbv_{obj_id}.pth", map_location=device))
    R_net.eval()
    for scene_id in scene_ids:
        val_loader = SATVal_loader(target_dir = target_dir, scene_ids=[scene_id], obj_id = obj_id, transform =SATRot_test_transform, sample_ratio=1)
        if len(val_loader) == 0:
            continue
        R_preds = []
        uv_preds = []
        Rc_list = []
        tc_list = []
        Kc_list = []
        score_list = []
        time_list = []
        img_ids = []
        t_gts = []
        uv_gts = []

        # 打印部分数据样本
        for img_id, rgb_inputs, uv_gt, R_gt, t_gt, bbox_gt, Kc_inv, Kc, Rc, tc  in val_loader:
            img_id, rgb_inputs, uv_gt, R_gt, t_gt, bbox_gt, Kc_inv, Kc, Rc, tc = img_id.to(device), rgb_inputs.to(device), uv_gt.to(device), R_gt.to(device), t_gt.to(device), bbox_gt.to(device), Kc_inv.to(device), Kc.to(device), Rc.to(device), tc.to(device)  # 将数据移到 GPU
            # print(f'scene id: {scene_id}, img id: {img_id} obj_id: {obj_id}')
            # print(f'Rc:{Rc} tc:{tc}')
            w, h = bbox_gt[:, 2] - bbox_gt[:, 0], bbox_gt[:, 3] - bbox_gt[:, 1]

            uv_pred, R_pred = R_net(rgb_inputs)
            R_pred = get_allorot(uv_pred, R_pred, Kc_inv)

            u_pred =(uv_pred[:, 0] * w + bbox_gt[:, 0]) # x坐标恢复
            v_pred =(uv_pred[:, 1] * h + bbox_gt[:, 1]) # y坐标恢复


            uv_pred = torch.cat([u_pred.unsqueeze(1),v_pred.unsqueeze(1)], dim=1)

            
            
            img_ids.append(img_id)
            R_preds.append(R_pred.squeeze(0))
            uv_preds.append(uv_pred.squeeze(0))
            Kc_list.append(Kc.squeeze(0))
            Rc_list.append(Rc.squeeze(0))
            tc_list.append(tc.squeeze(0))
            score_list.append(1)
            time_list.append(-1)

            t_gts.append(t_gt.squeeze(0))
            uv_gts.append(uv_gt.squeeze(0))



        t_preds = t_From_Multiviews(Rc_list , tc_list, uv_preds, Kc_list)
        print(f't_gts:{t_gts}')
        print(f't_preds:{t_preds}')

        write2csv(save_dir,scene_id, img_ids, obj_id, score_list,R_preds,t_preds,time_list)
            


if __name__ == '__main__':
        # 清空指定文件
    file_path = 'results/ycbv/MVPose_ycbv-test.csv'
    if os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass  # 清空文件内容
    for obj_id in range(1, 22):
        test4bop(target_dir = '../Datasets/ycbv/test',scene_ids = test_scene_ids, obj_id = obj_id, save_dir = file_path)
        