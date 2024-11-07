import os
import random
import sys

sys.path.insert(0,os.getcwd()) # 把当前路径添加到 sys.path 中

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import os
from utils.dataset_loader.depth_loader import get_depth_dataloader
from network.depth_net import DepthTransformer
import torchvision.models as models
from utils.transform import RandomOcclusionRGB
import torch.nn.functional as F
from utils.earlystop import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Using device: {device}")



criterion = F.l1_loss  # Assuming a regression problem
d_model = 64
nhead = 4
num_layers = 4
model = DepthTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=1, save_path="model.pth"):
    early_stopping = EarlyStopping(patience=5, min_delta=3)
    model.train()
    best_val_loss = float("inf")
    best_model_path = save_path  # 存储损失最小的模型路径
    if len(train_loader.dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data source.")
    for epoch in range(num_epochs):
        running_loss = 0.0
        i = 0
        for bbox_depth, rot_info, depth_gt in tqdm(train_loader):
            bbox_depth, rot_info, depth_gt = bbox_depth.to(device), rot_info.to(device), depth_gt.to(device)
            optimizer.zero_grad()
        # 对 batch 中每个图像处理
            predicted_depth = model(bbox_depth)
            # print(f'{predicted_depths}  vs  {depth_gt}')




            depth_sample = bbox_depth[0]
            min_val = depth_sample.min()
            max_val = depth_sample.max()
            normalized_depth = (depth_sample - min_val) / (max_val - min_val)

            # 再将其映射到 [0, 255]
            depth_img = (normalized_depth * 255).clamp(0, 255).byte()
            depth_img = depth_img.unsqueeze(0)
            transform = transforms.ToPILImage(mode='L')  # 'L' 表示灰度图
            img = transform(depth_img)
            img.save(f"visib/lm_depth/depth_image_{i}.png")
            i = i+1







            loss = criterion(predicted_depth, depth_gt)
            # print(f'loss:{loss}')
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * predicted_depth.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")

        val_loss = 0.0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for validation
             for bbox_depth, rot_info, depth_gt in tqdm(test_loader):
                bbox_depth, rot_info, depth_gt = bbox_depth.to(device), rot_info.to(device), depth_gt.to(device)
                predicted_depth = model(bbox_depth)

                depth_sample = bbox_depth[0]
                min_val = depth_sample.min()
                max_val = depth_sample.max()
                normalized_depth = (depth_sample - min_val) / (max_val - min_val)

                # 再将其映射到 [0, 255]
                depth_img = (normalized_depth * 255).clamp(0, 255).byte()
                depth_img = depth_img.unsqueeze(0)
                transform = transforms.ToPILImage(mode='L')  # 'L' 表示灰度图
                img = transform(depth_img)
                img.save(f"visib/lmo_depth/depth_image_{i}.png")
                i = i+1



                loss = criterion(predicted_depth, depth_gt)
                # print(f'loss:{loss}')
                val_loss += loss.item() * predicted_depth.size(0)                

        
        val_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}")
        
                # 更新最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with Validation Loss: {val_loss:.4f}")

        # Check early stopping
        if val_loss < 10 and early_stopping.check_early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}.")
            break

    print("Training completed. Best model saved with Validation Loss:", best_val_loss)
    print("Saved in:", best_model_path)



# Example usage of training and valing
# Example usage of training and valing
def train_depth(obj_ids):
    # Paths to images and their corresponding targets
    if obj_ids == None:
        obj_ids = [1,5,6,8,9,10,11,12]
    for obj_id in obj_ids:
        target_dir_1 = f'datasets/lm/{str(obj_id).zfill(6)}'  # RGB 图像目录
        target_dir_2 = f'datasets/lmo/train/{str(obj_id).zfill(6)}'  # RGB 图像目录
        target_dir_3 = f'datasets/lmo/test/000002'  # RGB 图像目录

        # train_loader = get_depth_dataloader(target_dir = target_dir_2, obj_id=obj_id, if_transform = True)
        test_loader = get_depth_dataloader(target_dir = target_dir_3, obj_id=obj_id, if_transform = False)
        # train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=3, save_path=f"weights/depth_obj_{obj_id}.pth")
        
        
        train_loader = get_depth_dataloader(target_dir = target_dir_1, obj_id=obj_id, if_transform = True) 
        # model.load_state_dict(torch.load(f"weights/depthv2_obj_{obj_id}.pth", map_location=device))
        train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=45, save_path=f"weights/depth_obj_{obj_id}.pth")


if __name__ == "__main__":
    obj_ids = [1,5,6,8,9,10,11,12]
    train_depth(obj_ids)