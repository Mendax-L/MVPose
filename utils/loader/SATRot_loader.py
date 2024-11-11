import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from lib.config import SATRot_test_transform, views
import random

# 自定义 Dataset 类
class RotDataset(Dataset):
    def __init__(self, obj_id, target_dir, transform=None):
        self.obj_id = obj_id
        self.rgb_dirs = [f'{target_dir}/view_{str(view_id).zfill(3)}/crop/' for view_id in views.keys()]
        self.gt_files = [f'{target_dir}/view_{str(view_id).zfill(3)}/view_{str(view_id).zfill(3)}_info.json' for view_id in views.keys()]
        self.transform = transform
        self.items = []

        # 加载每个文件夹的 ground truth 数据
        for rgb_dir, gt_file in zip(self.rgb_dirs, self.gt_files):
            gt = self._load_gt(gt_file)
            for data in gt:
                if data["obj_id"] == self.obj_id:
                    self.items.append((rgb_dir, data))

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
        R = item['R']
        # t = item['t']

        img_name = f"{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"
        img_file = os.path.join(rgb_dir, img_name)

        rgb = Image.open(img_file)
        if self.transform:
            rgb = self.transform(rgb)

        uv = torch.tensor(uv, dtype=torch.float32).squeeze(0)
        R = torch.tensor(R, dtype=torch.float32)
        # t = torch.tensor(t, dtype=torch.float32)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        # uv_relative = torch.tensor(uv_relative, dtype=torch.float32)


        return rgb, uv, R, bbox

# 定义数据加载函数
def SATRot_loader(target_dir, obj_id=1, transform=None, batch_size=32, shuffle=True, num_workers=8):
    # 实例化自定义数据集
    dataset = RotDataset(obj_id=obj_id, target_dir=target_dir, transform=transform)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

# 示例调用
if __name__ == '__main__':
    target_dir = '../Datasets/lmo/train/000001'# rgb 图像目录
    # 获取数据加载器
    train_loader = SATRot_loader(target_dir)
