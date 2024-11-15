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
from utils.loader.SATRot_loader import SATRot_loader
from lib.config import test_scene_ids
from lib import to_allo
from lib.multiviews import t_From_Multiviews

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


    
def test4bop(target_dir = '../Datasets/ycbv/test', scene_ids = test_scene_ids, obj_id = 1, save_dir = 'results/lmo/MVPose_lmo-test.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R_net = SATRot(d_model = 240, nhead=4, num_layers=4,).to(device)
    R_net.load_state_dict(torch.load(f"weights/SATRot_obj_{obj_id}.pth", map_location=device))
    R_net.eval()
    for scene_id in scene_ids:
        val_loader = SATRot_loader(target_dir = target_dir, scene_ids=list(scene_id), obj_id = obj_id, transform =SATRot_test_transform, sample_ratio=1)
        R_preds = []
        uv_preds = []
        Rc_list = []
        Kc_list = []

        # 打印部分数据样本
        for img_id, rgb_inputs, uv_gt, R_gt, t_gt, bbox_gt, Kc_inv, Kc, Rc,  in val_loader:
            img_id, rgb_inputs, uv_gt, R_gt, t_gt, bbox_gt, Kc_inv, Kc, Rc = img_id.to(device), rgb_inputs.to(device), uv_gt.to(device), R_gt.to(device), bbox_gt.to(device), Kc_inv.to(device), Kc.to(device), Rc.to(device)  # 将数据移到 GPU


            uv_pred, R_pred = R_net(rgb_inputs)
            R_pred = to_allo(R_pred,uv_pred)

            w, h = bbox_gt[:, 2] - bbox_gt[:, 0], bbox_gt[:, 3] - bbox_gt[:, 1]

            uv_pred, R_pred = R_net(rgb_inputs)

            u_pred =(uv_pred[:, 0] * w + bbox_gt[:, 0]) # x坐标恢复
            v_pred =(uv_pred[:, 1] * h + bbox_gt[:, 1]) # y坐标恢复

            uv_pred = torch.cat([u_pred.unsqueeze(1),v_pred.unsqueeze(1)], dim=1)
            R_pred = get_allorot(uv_pred, R_pred, Kc_inv)
            R_preds.append(R_pred)
            uv_preds.append(uv_pred)
            Kc_list.append(Kc)
            Rc_list.append(Rc)

        t_pred = t_From_Multiviews(R_preds , Rc_list, uv_preds, Kc_list)


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
    file_path = 'results/ycbv/MVPose_ycbv-test.csv'
    if os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass  # 清空文件内容
    for obj_id in range(1, 22):
        test4bop(target_dir = 'datasets/lmo/test/000002',scene_ids = test_scene_ids, obj_id = obj_id, save_dir = file_path)
        