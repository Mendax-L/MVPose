import itertools
import json
import os
import sys

sys.path.insert(0,os.getcwd()) # 把当前路径添加到 sys.path 中

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import os
import torch.nn.functional as F
from utils.dataset_loader.rot_loader import get_rot_dataloader
from network.rot_net import Rot_Net
from utils.transform import RandomHolesRGB,RandomOcclusionRGB,RandomBlur,RandomPosterizeRGBA,RandomOcclusionRGBA,RandomHolesRGBA,RandomBlackRGBA,RandomSolarizeRGBA, RandomAdjustSharpnessRGBA
from utils.allo_ego import psi_tensor
import torch.nn.functional as F
from utils.earlystop import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

re_size = 128
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((re_size, re_size)),
    RandomOcclusionRGB(p=0.4),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.4),
    RandomHolesRGB(p=0.4),
    transforms.RandomInvert(p=0.1),              # 随机反色
    transforms.RandomPosterize(bits=4, p=0.4),   # 随机减少颜色位数
    transforms.RandomSolarize(threshold=128, p=0.4),  # 随机日蚀效果
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.4),  # 随机调整锐度
    transforms.RandomAutocontrast(p=0.4),        # 随机自动对比度
    transforms.RandomEqualize(p=0.4),            # 随机直方图均衡化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],     # 正则化
                        std=[0.229, 0.224, 0.225])
])

camera_file = '../Datasets/lmo/camera.json'
with open(camera_file, 'r') as c:
    camera = json.load(c)
Kc = torch.tensor([
    [camera['fx'], 0, camera['cx']],  # 第一行: [fx, 0, cx]
    [0, camera['fy'], camera['cy']],  # 第二行: [0, fy, cy]
    [0, 0, 1]                           # 第三行: [0, 0, 1]
], dtype=torch.float32).to(device)
Kc_inv = torch.inverse(Kc).to(device)

def criterion(output, target, alpha=0.3):
    # Normalize both vectors to ensure that we're only comparing directions
    # cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
    # Compute the cosine similarity
    # Compute the cosine similarity
    cosine_sim_1 = torch.sum(output[:,:3] * target[:,:3], dim=1)
    cosine_sim_2 = torch.sum(output[:,3:6] * target[:,3:6], dim=1)
    # cosine_sim_3 = torch.sum(output[:,6:] * target[:,6:], dim=1)
    
    # Compute the angular distance (arccos of cosine similarity)
    angle_1 = torch.acos(torch.clamp(cosine_sim_1, -1 + 1e-7, 1 - 1e-7))  # Clamp to avoid numerical errors
    angle_2 = torch.acos(torch.clamp(cosine_sim_2, -1 + 1e-7, 1 - 1e-7))  # Clamp to avoid numerical errors
    # angle_3 = torch.acos(torch.clamp(cosine_sim_3, -1 + 1e-7, 1 - 1e-7))  # Clamp to avoid numerical errors


    # The loss is the mean of the angle (in radians)
    loss = alpha * torch.mean(angle_1+angle_2) + (1 - alpha) * F.l1_loss(output, target, reduction='mean')  # Mean angle across all samples
    return loss



# def criterion(R_pred, R_gt, alpha=0.3):
#     # 将 [batchsize, 9] 转换回 [batchsize, 3, 3] 的旋转矩阵
#     R_pred = R_pred.view(-1, 3, 3)
#     R_gt = R_gt.view(-1, 3, 3)
    
#     # 计算 Geodesic Loss
#     R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    
#     # 计算旋转差异的迹（trace）
#     trace = torch.sum(torch.diagonal(R_diff, dim1=-2, dim2=-1), dim=-1)
    
#     # 通过 arccos 计算角度差异
#     theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    
#     # Geodesic Loss
#     geodesic_loss = torch.mean(theta)
    
#     # 计算 L1 Loss
#     l1_loss = F.l1_loss(R_pred, R_gt)
    
#     # 组合 Geodesic Loss 和 L1 Loss
#     combined_loss = alpha * geodesic_loss + (1 - alpha) * l1_loss
    
#     return combined_loss


model = Rot_Net(d_model = 240, nhead=4, num_layers=4, num_samples =[1, 5, 10], window_sizes = [128, 64, 32]).to(device)  # 将模型移到 GPU

optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# Training loop
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30, save_path="model.pth"):
    early_stopping = EarlyStopping(patience=5, min_delta=0.02)
    model.train()
    # 初始化验证损失最小值为正无穷
    best_val_loss = float("inf")
    best_model_path = save_path  # 存储损失最小的模型路径
    if len(train_loader.dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data source.")
    # for epoch in range(num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        i = 0
        j = 0
        for rgba_inputs, centeruv, allo_rot in tqdm(train_loader):
            rgba_inputs, centeruv, allo_rot = rgba_inputs.to(device), centeruv.to(device), allo_rot.to(device)  # 将数据移到 GPU


            rgba_sample = rgba_inputs[0]
            # 再将其映射到 [0, 255]
            transform = transforms.ToPILImage()
            img = transform(rgba_sample)
            img.save(f"visib/lm_rgb/rgb_image_{i}.png")
            i = i+1



            # print(rgba_inputs.shape)
            optimizer.zero_grad()
            rot_pred = model(rgba_inputs)
            # print(f'rot_pred.shape:{rot_pred.shape}')
            centeruv_3d = torch.cat([centeruv, torch.ones((centeruv.shape[0],1)).to(device)], dim=1)
            p = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(centeruv.shape[0], 1).to(device)
            # print(f'p:{p.shape}')           
            q = torch.matmul(Kc_inv, centeruv_3d.T).T
            # print(f'q:{q.shape}')

            Rc=psi_tensor(p, q)
            # print(f'Rc:{Rc.shape}')

            rot_pred = rot_pred.view(-1, 3, 3)
            
            R_pred = torch.matmul(Rc, rot_pred)

            rot_pred = R_pred.reshape(-1, 9)
            # print(f'R_pred:{R_pred.shape}')
            # print(f'rot_pred.shape:{rot_pred.shape}')

            loss = criterion(rot_pred, allo_rot)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * rgba_inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")
        # Validation step
        val_loss = 0.0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for validation
            for rgba_inputs, centeruv, allo_rot in test_loader:
                rgba_inputs, centeruv, allo_rot = rgba_inputs.to(device), centeruv.to(device), allo_rot.to(device)
                
                for rgba_sample in rgba_inputs:
                    transform = transforms.ToPILImage()
                    img = transform(rgba_sample)
                    img.save(f"visib/lmo_rgb/rgb_image_{j}.png")
                    j = j+1
                
                rot_pred = model(rgba_inputs)
                centeruv_3d = torch.cat([centeruv, torch.ones((centeruv.shape[0], 1)).to(device)], dim=1)
                p = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(centeruv.shape[0], 1).to(device)
                q = torch.matmul(Kc_inv, centeruv_3d.T).T
                Rc = psi_tensor(p, q)
                
                rot_pred = rot_pred.view(-1, 3, 3)
                
                R_pred = torch.matmul(Rc, rot_pred)

                rot_pred = R_pred.reshape(-1, 9)

                loss = criterion(rot_pred, allo_rot)
                val_loss += loss.item() * rgba_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Validation Loss: {val_loss:.4f}")
        
                # 更新最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with Validation Loss: {val_loss:.4f}")

        # Check early stopping
        if early_stopping.check_early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}.")
            break

    print("Training completed. Best model saved with Validation Loss:", best_val_loss)
    print("Saved in:", best_model_path)


def train_rot(obj_ids):
    # Paths to images and their corresponding targets
    if obj_ids == None:
        obj_ids = [1,5,6,8,9,10,11,12]
    for obj_id in obj_ids:
        train_target_dir = [f'datasets/lmo/train/{str(obj_id).zfill(6)}' ] # RGB 图像目录
        test_target_dir = [f'datasets/lmo/test/000002']  # RGB 图像目录
        train_loader = get_rot_dataloader(target_dirs = train_target_dir, obj_id=obj_id, transform =transform)
        test_loader = get_rot_dataloader(target_dirs = test_target_dir, obj_id=obj_id, transform =None)
    
        # Train and val the model
        train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=35, save_path=f"weights/SATRot_obj_{obj_id}.pth")
        #val_model(model, val_loader, criterion)

        train_target_dir = [f'datasets/lm/{str(obj_id).zfill(6)}']  # RGB 图像目录
        train_loader = get_rot_dataloader(target_dirs = train_target_dir, obj_id=obj_id, transform =transform)
        model.load_state_dict(torch.load(f"weights/SATRot_obj_{obj_id}.pth", map_location=device))
        train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=45, save_path=f"weights/SATRot_obj_{obj_id}.pth")


if __name__ == "__main__":
    obj_ids = [1,5,6,8,9,10,11,12]
    train_rot(obj_ids)