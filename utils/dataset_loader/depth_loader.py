import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import torch

class RandomBlockOcclusion:
    def __init__(self, num_blocks, block_size, fill_value_range=(0, 1000)):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.fill_value_range = fill_value_range  # 这里定义随机值范围

    def __call__(self, img):
        """
        img: 输入图像，张量格式 (C, H, W)
        """
        _, H, W = img.shape
        img = img.clone()  # 防止原始张量被修改

        for _ in range(self.num_blocks):
            # 随机选择块的起始位置
            x_start = random.randint(0, W - self.block_size[0])
            y_start = random.randint(0, H - self.block_size[1])

            # 定义随机扰动的填充块
            random_fill = torch.randint(
                self.fill_value_range[0], 
                self.fill_value_range[1], 
                (self.block_size[1], self.block_size[0])  # (block_height, block_width)
            )

            # 将填充块放置在图像中
            img[:, y_start:y_start + self.block_size[1], x_start:x_start + self.block_size[0]] = random_fill

        return img

# 自定义 transforms，包含遮挡增强


# 自定义 Dataset 类
class DepthDataset(Dataset):
    def __init__(self, obj_id, target_dir, transform = None):
        self.obj_id = obj_id
        self.transform = transform
        self.items = []
        self.mask_visib_dir = f'{target_dir}/mask_visib/'    
        self.depth_dir = f'{target_dir}/depth/'
        gt_file = f'{target_dir}/object_info.json'
        self.gt = self._load_gt(gt_file)
        for data in self.gt:
            if data["obj_id"] == self.obj_id:
                self.items.append(data)
        


    # 加载中心点数据
    def _load_gt(self, gt_file):
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        return gt
    
    # 返回数据集大小
    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        item = self.items[idx]
        img_id = item['img_id']
        centerxyxy = item['centerxyxy']
        obj_idx = item['idx']
        half_diameter = item['half_diameter']
        rot_mat = np.array(item["cam_R_m2c"])
        rot_tensor = torch.tensor(rot_mat, dtype=torch.float32)

        # 2. 获取 UV 坐标并在深度图上查找深度值
        depth_img_path = os.path.join(self.depth_dir, f"{str(img_id).zfill(6)}.png")
        depth_img = Image.open(depth_img_path).convert('I')        
        mask_visib_path = os.path.join(self.mask_visib_dir, f"{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png")
        mask_visib = Image.open(mask_visib_path)
        x_min, y_min, x_max, y_max = centerxyxy  # 使用你的 bbox 坐标
        bbox_depth = transforms.ToTensor()(depth_img.crop((x_min, y_min, x_max, y_max)))
        bbox_mask = transforms.ToTensor()(mask_visib.crop((x_min, y_min, x_max, y_max)))
        bbox_depth = transforms.Resize((128,128))(bbox_depth)
        bbox_mask = transforms.Resize((128,128))(bbox_mask)
        bbox_mask = (bbox_mask > 0).float()
        if self.transform:
            bbox_depth  = self.transform(bbox_depth)
        bbox_depth = bbox_depth * bbox_mask
        # transform = transforms.Compose([
        #         # transforms.Pad((0, 0, max(128 - bbox_depth.shape[1], 0), max(128 - bbox_depth.shape[2], 0))),  # 动态填充，确保图像最小为 128x128
        #         transforms.Resize((128, 128)),
        # ])

        depth_gt = torch.tensor(item["cam_t_m2c"][2], dtype=torch.float32).unsqueeze(0)


        return bbox_depth, depth_gt
    
        
# 定义数据加载函数
def get_depth_dataloader(target_dir, obj_id=1, if_transform = False ,batch_size=8, shuffle=True, num_workers=8):
    # 定义必要的预处理步骤
    
    transform = None
    if if_transform:
        transform = transforms.Compose([
            # transforms.Resize((64, 64)),  # 调整大小
            RandomBlockOcclusion(num_blocks=random.randint(1,10), block_size=(10,10), fill_value_range=(300, 1000)),
            RandomBlockOcclusion(num_blocks=random.randint(2,4), block_size=(random.randint(10,40), random.randint(1,40)), fill_value_range=(0, 1)),
        ])

    # 实例化自定义数据集
    dataset = DepthDataset(obj_id=obj_id,target_dir=target_dir, transform=transform)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

# 示例调用
if __name__ == '__main__':

    target_dir = 'datasets/lmo/test/000002'  # rgba 图像目录

    # 获取数据加载器
    train_loader = get_depth_dataloader(target_dir)

    # 打印部分数据样本
    for bbox_depth, depth_gt in train_loader:
        print(f"bbox_depth: {bbox_depth.size()}, depth_gt: {depth_gt.size()}")
        break