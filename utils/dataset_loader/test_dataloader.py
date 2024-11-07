import csv
import os
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from PIL import Image



# 自定义 Dataset 类
class TestDataset(Dataset):
    def __init__(self, obj_id, target_dir, gt_file ,transform=None):
        self.target_dir = target_dir
        self.gt = self._load_gt(gt_file)
        self.obj_id = obj_id
        self.transform = transform
        self.items = []
        for data in self.gt:
            if data["obj_id"] == self.obj_id:
                self.items.append(data)
        
        if self.transform is None:
            self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    # 加载中心点数据
    def _load_gt(self, gt_file):
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        return gt

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        item = self.items[idx]
        img_id = item['img_id']
        obj_idx = item['idx']
        scene_id = item['scene_id']
        obj_id = item['obj_id']
        half_diameter = item['half_diameter']
        minxyxy = item['minxyxy']
        centerxyxy = item['squarexyxy']

        rgb_path = f"{self.target_dir}/rgb/{str(img_id).zfill(6)}.png"
        mask_path = f"{self.target_dir}/mask_visib/{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"
        mincrop_path = f"{self.target_dir}/mincrop/{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"
        centercrop_path = f"{self.target_dir}/centercrop/{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"
        depth_img_path = f"{self.target_dir}/depth/{str(img_id).zfill(6)}.png"
        
        scene_id = torch.tensor(int(scene_id), dtype=torch.int16)
        img_id = torch.tensor(int(img_id), dtype=torch.int16)
        obj_id = torch.tensor(int(obj_id), dtype=torch.int16)
        half_diameter = torch.tensor(half_diameter, dtype=torch.float32)
        minxyxy = torch.tensor(minxyxy, dtype=torch.int32)
        centerxyxy = torch.tensor(centerxyxy, dtype=torch.int32)


        rgb = Image.open(rgb_path)
        mask = Image.open(mask_path)
        depth_img = Image.open(depth_img_path)
        rgb = transforms.ToTensor()(rgb)
        mask = transforms.ToTensor()(mask)
        depth_img = transforms.ToTensor()(depth_img)
        t_gt = np.array(item["cam_t_m2c"])
        t_gt = torch.tensor(t_gt, dtype=torch.float32)
        centeruv_gt = np.array([item["u"], item["v"]]) 
        centeruv_gt = torch.tensor(centeruv_gt, dtype=torch.float32)

        rot_mat = np.array(item["cam_R_m2c"])
        rot_mat = torch.tensor(rot_mat, dtype=torch.float32)

        centercrop = Image.open(centercrop_path)
        centercrop = self.transform(centercrop)

        return scene_id, img_id, obj_id, half_diameter, minxyxy, rgb, mask, depth_img, t_gt, centerxyxy, centeruv_gt, centercrop, rot_mat


# 定义数据加载函数
def get_test_dataloader(target_dir, gt_file ,obj_id=1,batch_size=1, shuffle=True, num_workers=4):
    # 定义必要的预处理步骤
    
    # 实例化自定义数据集
    dataset = TestDataset(obj_id=obj_id, target_dir=target_dir, gt_file = gt_file)
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader


# if __name__ == '__main__':