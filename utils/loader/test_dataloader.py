import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from lib.config import SATRot_test_transform, views
import random

# 自定义 Dataset 类
class ValDataset(Dataset):
    def __init__(self, scene_ids, target_dir, obj_id, transform=None, sample_ratio=0.5):
        self.scene_ids = scene_ids
        self.obj_id = obj_id
        self.rgb_dirs = [f'{target_dir}/{str(scene_id).zfill(6)}/view_000/crop/' for scene_id in scene_ids]
        self.gt_files = [f'{target_dir}/{str(scene_id).zfill(6)}/view_000/view_000_info.json' for scene_id in scene_ids]
        self.transform = transform
        self.items = []

        # 针对每个 scene_id 进行采样
        for rgb_dir, gt_file in zip(self.rgb_dirs, self.gt_files):
            gt = self._load_gt(gt_file)
            scene_items = [(rgb_dir, data) for data in gt if data["obj_id"] == self.obj_id]
            
            # 按比例采样
            if sample_ratio < 1.0:
                sample_size = int(len(scene_items) * sample_ratio)
                scene_items = random.sample(scene_items, sample_size)

            self.items.extend(scene_items) 

        if self.transform is None:
            self.transform = SATRot_test_transform

    # 加载中心点数据
    def _load_gt(self, gt_file):
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        return gt
    
    # 返回数据集大小
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb_dir, item = self.items[idx]
        img_id = item['img_id']
        obj_idx = item['idx']
        bbox = item['bbox']
        uv = np.array([item["uv"]]) 
        uv_relative = item['uv_relative']
        Kc = item['Kc']
        Rc = item['Rc']
        tc = item['tc']
        Kc_inv = item['Kc_inv']
        R = item['R']
        t = item['t']

        img_name = f"{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"
        img_file = os.path.join(rgb_dir, img_name)

        rgb = Image.open(img_file)
        if self.transform:
            rgb = self.transform(rgb)

        uv = torch.tensor(uv, dtype=torch.float32).squeeze(0)
        R = torch.tensor(R, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        Kc_inv = torch.tensor(Kc_inv, dtype=torch.float32) 
        Kc = torch.tensor(Kc, dtype=torch.float32) 
        Rc = torch.tensor(Rc, dtype=torch.float32) 
        tc = torch.tensor(tc, dtype=torch.float32) 
        # uv_relative = torch.tensor(uv_relative, dtype=torch.float32)


        return img_id, rgb, uv, R, t, bbox, Kc_inv, Kc, Rc, tc

# 定义数据加载函数
def SATVal_loader(target_dir, obj_id =1, scene_ids=[1], transform=None, batch_size=1, shuffle=False, num_workers=1, sample_ratio=0.5):
    # 实例化自定义数据集
    dataset = ValDataset(scene_ids=scene_ids, target_dir=target_dir, obj_id = obj_id, transform=transform, sample_ratio=sample_ratio)

    # if split_ratio is None:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
    # else:
    #     train_size = int(len(dataset) * split_ratio)
    #     test_size = len(dataset) - train_size

    #     # 使用 random_split 划分数据集
    #     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    #     # 创建 DataLoader
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    #     # 创建 DataLoader
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    #     return train_loader,test_loader

# 示例调用
if __name__ == '__main__':
    target_dir = '../Datasets/lmo/train/000001'# rgb 图像目录
    # 获取数据加载器
    train_loader = SATVal_loader(target_dir)
