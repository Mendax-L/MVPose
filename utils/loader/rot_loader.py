import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

# 自定义 Dataset 类
class RotDataset(Dataset):
    def __init__(self, obj_id, target_dirs, transform=None):
        self.obj_id = obj_id
        self.rgb_dirs = [f'{dir}/centercrop/' for dir in target_dirs]
        self.gt_files = [f'{dir}/object_info.json' for dir in target_dirs]
        self.transform = transform
        self.items = []

        # 加载每个文件夹的 ground truth 数据
        for rgb_dir, gt_file in zip(self.rgb_dirs, self.gt_files):
            gt = self._load_gt(gt_file)
            for data in gt:
                if data["obj_id"] == self.obj_id:
                    self.items.append((rgb_dir, data))

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

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
        # squarexyxy = item['squarexyxy']
        centerxyxy = item['centerxyxy']
        u_relative, v_relative = item['u_relative'], item['v_relative']

        img_name = f"{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"
        img_file = os.path.join(rgb_dir, img_name)

        uv_gt = np.array([item["u"], item["v"]]) 
        uv_gt = torch.tensor(uv_gt, dtype=torch.float32)

        t_gt = torch.tensor(item["cam_t_m2c"], dtype=torch.float32)

        # 1. 加载 rgb 图像
        rgb_img = Image.open(img_file)
        if self.transform:
            rgb_img = self.transform(rgb_img)
        
        # 2. 获取旋转矩阵并提取前两列
        # size = torch.tensor((squarexyxy[2] - squarexyxy[0]), dtype=torch.float32).unsqueeze(0)

        rot_mat = np.array(item["cam_R_m2c"])
        rot_tensor = torch.tensor(rot_mat, dtype=torch.float32)
        # squarexyxy = torch.tensor(squarexyxy, dtype=torch.float32)
        centerxyxy = torch.tensor(centerxyxy, dtype=torch.float32)
        uv_relative = torch.tensor((u_relative, v_relative), dtype=torch.float32)


        return rgb_img,uv_gt, rot_tensor

# 定义数据加载函数
def get_rot_dataloader(target_dirs, obj_id=1, transform=None, batch_size=32, shuffle=True, num_workers=8):
    # 实例化自定义数据集
    dataset = RotDataset(obj_id=obj_id, target_dirs=target_dirs, transform=transform)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

# 示例调用
if __name__ == '__main__':
    target_dirs = ['../datasets/lmo/train/000001', '../datasets/lmo/train/000002']  # rgb 图像目录
    # 获取数据加载器
    train_loader = get_pose_dataloader(target_dirs)
