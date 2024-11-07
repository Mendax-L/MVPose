import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自定义 Dataset 类
class CeneterUVDataset(Dataset):
    def __init__(self, obj_id, rgb_dir, gt_file, transform=None):
        self.obj_id = obj_id
        self.rgb_dir = rgb_dir
        self.transform = transform
        self.gt = self._load_gt(gt_file)
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

    # 返回数据集大小
    def __len__(self):
        return len(self.items)

    # 返回单个样本（图像和中心点）
    def __getitem__(self, idx):
        item = self.items[idx]
        img_id = item['img_id']
        obj_idx = item['idx']
        img_name = f"{str(img_id).zfill(6)}_{str(obj_idx).zfill(6)}.png"  # 将索引转换为六位编号
        img_path = os.path.join(self.rgb_dir, img_name)

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 加载中心点信息

        center_uv = torch.tensor([item['u_relative'], item['v_relative']], dtype=torch.float32)

        # 对图像进行预处理
        if self.transform:
            image = self.transform(image)

        return image, center_uv

# 定义数据加载函数
def get_centeruv_dataloader(target_dir, obj_id=1, transform=None, batch_size=16, shuffle=True, num_workers=4):
    # 图像的预处理（例如调整大小、转换为 tensor 等）

    
    rgb_dir = f'{target_dir}/mincrop'
    gt_file = f'{target_dir}/object_info.json'

    # 实例化自定义数据集
    dataset = CeneterUVDataset(obj_id, rgb_dir=rgb_dir, gt_file=gt_file, transform=transform)

    # 使用 DataLoader 加载数据集
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

# 测试代码（可选）
if __name__ == '__main__':

    target_dir = '../datasets/lmo/train/000001'  # RGB 图像目录
    obj_id = 1  # 目标对象 ID
    # 获取数据加载器
    train_loader = get_centeruv_dataloader(target_dir, obj_id)

    # 打印部分数据样本
    for rgb_inputs, centeruv_gt in train_loader:
        print(f"RGB Inputs: {rgb_inputs.shape}, Center UV Ground Truth: {centeruv_gt}")
        break
