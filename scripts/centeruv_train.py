import itertools
import os
import sys
sys.path.insert(0,os.getcwd()) # 把当前路径添加到 sys.path 中

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from network.centeruv_net import CenterUV_Net
import torch.nn.functional as F
import torchvision.models as models
from utils.transform import RandomOcclusionRGB
from utils.dataset_loader.centeruv_loader import get_centeruv_dataloader
from utils.earlystop import EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

re_size = 128
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((re_size, re_size)),
    transforms.RandomInvert(p=0.1),              # 随机反色
    transforms.RandomPosterize(bits=4, p=0.3),   # 随机减少颜色位数
    transforms.RandomSolarize(threshold=128, p=0.3),  # 随机日蚀效果
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # 随机调整锐度
    transforms.RandomAutocontrast(p=0.3),        # 随机自动对比度
    transforms.RandomEqualize(p=0.3),            # 随机直方图均衡化
    RandomOcclusionRGB(p=0.3), 
    transforms.ToTensor()                        # 转换为 Tensor
])

model = CenterUV_Net().to(device)  # 将模型移到 GPU
def criterion(output, target):
    return F.l1_loss(output, target, reduction='mean')  # Assuming a regression problem
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=1, save_path="model.pth"):
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    model.train()
    if len(train_loader.dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data source.")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for rgb_inputs,centeruv_gt in tqdm(train_loader):
            rgb_inputs,centeruv_gt = rgb_inputs.to(device), centeruv_gt.to(device)  # 将数据移到 GPU
            # x, y, crop_size = centeruv_gt[:, 0], centeruv_gt[:, 1], centeruv_gt[:, 2]
            # # 使用比例k来调整x和y
            # adjusted_centeruv_gt = torch.stack((x * re_size/crop_size, y * re_size/crop_size), dim=1)
            optimizer.zero_grad()
            outputs = model(rgb_inputs)
            loss = criterion(outputs, centeruv_gt) #criterion 函数返回的是一个批次中每个样本的平均损失
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * rgb_inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")
        # Validation step
        val_loss = 0.0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for validation
            for rgb_inputs,centeruv_gt in test_loader:
                rgb_inputs,centeruv_gt = rgb_inputs.to(device), centeruv_gt.to(device)  # 将数据移到 GPU
                outputs = model(rgb_inputs)
                loss = criterion(outputs, centeruv_gt) #criterion 函数返回的是一个批次中每个样本的平均损失
                val_loss += loss.item() * rgb_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}")
        
        # Check early stopping
        if early_stopping.check_early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}.")
            break

    torch.save(model.state_dict(), save_path)
    print("Model saved to", save_path)
            # save_path = save_path.replace(f'.pth', '.pth')


def train_centeruv(obj_ids):
    # Paths to images and their corresponding targets
    if obj_ids == None:
        obj_ids = [1,5,6,8,9,10,11,12]
    for obj_id in obj_ids:
        train_target_dir = f'datasets/lm/{str(obj_id).zfill(6)}'  # RGB 图像目录
        test_target_dir = f'datasets/lmo/test/000002'  # RGB 图像目录
        train_loader = get_centeruv_dataloader(target_dir = train_target_dir, obj_id=obj_id, transform =transform)
        test_loader = get_centeruv_dataloader(target_dir = test_target_dir, obj_id=obj_id, transform =None)

        # model.load_state_dict(torch.load(f"weights/centeruv_obj_{obj_id}.pth", map_location=device))
        # Train and val the model
        train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20, save_path=f"weights/centeruv_obj_{obj_id}.pth")
        #val_model(model, val_loader, criterion)

if __name__ == "__main__":
    obj_ids = [1,5,6,8,9,10,11,12]
    train_centeruv(obj_ids)