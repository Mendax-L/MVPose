import os
import sys

sys.path.insert(0,os.getcwd()) # 把当前路径添加到 sys.path 中

import torch
import torch.optim as optim
from tqdm import tqdm
import os
from utils.loader.SATRot_loader import SATRot_loader
from lib.SATRotv2 import SATRotv2
from lib.SATRot import SATRot
from utils.earlystop import EarlyStopping
from lib.loss import criterion_R, criterion_uv
from lib.config import SATRot_train_transform, SATRot_test_transform
from lib.to_allo import get_allorot
import torchvision.transforms as transforms
from lib.config import train_scene_ids, test_scene_ids

print(f'test_ycbv_scene_ids:{test_scene_ids}')
print(f'train_ycbv_scene_ids:{train_scene_ids}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SATRot(d_model = 120, nhead=4, num_layers=4).to(device)  # 将模型移到 GPU
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training loop
def train_model(model, train_loader, test_loader, optimizer, num_epochs=30, save_path="model.pth"):
    R_early_stopping = EarlyStopping(patience=5, min_delta=0.02)
    uv_early_stopping = EarlyStopping(patience=5, min_delta=0.02)
    # 初始化验证损失最小值为正无穷
    best_R_loss = float("inf")
    best_uv_loss = float("inf")
    best_model_path = save_path  # 存储损失最小的模型路径

    if len(train_loader.dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data source.")
    
    # for epoch in range(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        R_trainloss = 0.0
        uv_trainloss = 0.0
        for rgb_inputs, uv_gt, R_gt, bbox_gt, Kc_inv in tqdm(train_loader):
            rgb_inputs, uv_gt, R_gt, bbox_gt, Kc_inv = rgb_inputs.to(device), uv_gt.to(device), R_gt.to(device), bbox_gt.to(device), Kc_inv.to(device)  # 将数据移到 GPU
            
            rgb_sample = rgb_inputs[0]
            # 再将其映射到 [0, 255]
            transform = transforms.ToPILImage()
            img = transform(rgb_sample)
            img.save(f"visib/lm_rgb/rgb_image.png")

            w, h = bbox_gt[:, 2] - bbox_gt[:, 0], bbox_gt[:, 3] - bbox_gt[:, 1]

            optimizer.zero_grad()
            uv_pred, R_pred = model(rgb_inputs)

            u_pred =(uv_pred[:, 0] * w + bbox_gt[:, 0]) # x坐标恢复
            v_pred =(uv_pred[:, 1] * h + bbox_gt[:, 1]) # y坐标恢复

            uv_pred = torch.cat([u_pred.unsqueeze(1),v_pred.unsqueeze(1)], dim=1)
            R_pred = get_allorot(uv_pred, R_pred, Kc_inv)

            R_loss = criterion_R(R_pred, R_gt)
            uv_loss = criterion_uv(uv_pred, uv_gt)
            total_loss = R_loss + uv_loss
            total_loss.backward()
            optimizer.step()
            R_trainloss += R_loss.item() * rgb_inputs.size(0)
            uv_trainloss += uv_loss.item() * rgb_inputs.size(0)
        R_trainloss /= len(train_loader.dataset)
        uv_trainloss /= len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, R_loss: {R_trainloss:.4f}, uv_loss: {uv_trainloss}")



        # Validation step
        R_valloss = 0.0
        uv_valloss = 0.0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for validation
            for rgb_inputs, uv_gt, R_gt, bbox_gt, Kc_inv in tqdm(test_loader):
                rgb_inputs, uv_gt, R_gt, bbox_gt, Kc_inv = rgb_inputs.to(device), uv_gt.to(device), R_gt.to(device), bbox_gt.to(device), Kc_inv.to(device)  # 将数据移到 GPU
                

                rgb_sample = rgb_inputs[0]
                # 再将其映射到 [0, 255]
                transform = transforms.ToPILImage()
                img = transform(rgb_sample)
                img.save(f"visib/lmo_rgb/rgb_image.png")


                w, h = bbox_gt[:, 2] - bbox_gt[:, 0], bbox_gt[:, 3] - bbox_gt[:, 1]
                # print(rgb_inputs.shape)
                optimizer.zero_grad()
                uv_pred, R_pred = model(rgb_inputs)
                u_pred =(uv_pred[:, 0] * w + bbox_gt[:, 0]) # x坐标恢复
                v_pred =(uv_pred[:, 1] * h + bbox_gt[:, 1]) # y坐标恢复
                uv_pred = torch.cat([u_pred.unsqueeze(1),v_pred.unsqueeze(1)], dim=1)
                R_pred = get_allorot(uv_pred, R_pred, Kc_inv)

                R_loss = criterion_R(R_pred, R_gt)
                uv_loss = criterion_uv(uv_pred, uv_gt)
                # print(f"R_pred:{R_pred}")
                # print(f"R_gt:{R_gt}")
                R_valloss += R_loss.item() * rgb_inputs.size(0)
                uv_valloss += uv_loss.item() * rgb_inputs.size(0)
        
        R_valloss /= len(test_loader.dataset)
        uv_valloss /= len(test_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Val_R_loss: {R_valloss:.4f},Val_uv_loss: {uv_valloss}")
        
                # 更新最佳模型
        if R_valloss < best_R_loss and uv_valloss < best_uv_loss:
            best_R_loss = R_valloss
            best_uv_loss = uv_valloss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with Val_R_loss: {R_valloss:.4f},Val_uv_loss: {uv_valloss}")

        # Check early stopping
        if R_early_stopping.check_early_stopping(R_valloss) and uv_early_stopping.check_early_stopping(uv_valloss):
            print(f"Early stopping at epoch {epoch}.")
            break
    print(f"Best model saved in: {best_model_path} with loss R: {best_R_loss} uv: {best_uv_loss}")


def train_R():
    # Paths to images and their corresponding targets
    for obj_id in range(1,22):
        train_target_dir = f'../Datasets/ycbv/train_real' # RGB 图像目录
        test_target_dir = f'../Datasets/ycbv/test'  # RGB 图像目录
        # train_loader,test_loader = SATRot_loader(target_dir = test_target_dir, obj_id=obj_id, transform =SATRot_train_transform,split_ratio=0.5)
        train_loader = SATRot_loader(target_dir = train_target_dir, scene_ids=train_scene_ids, obj_id = obj_id, transform =SATRot_train_transform, sample_ratio=0.1)
        test_loader = SATRot_loader(target_dir = test_target_dir, scene_ids=test_scene_ids, obj_id = obj_id, transform =SATRot_test_transform, sample_ratio=1)

        # Train and val the model
        # model.load_state_dict(torch.load(f"weights/SATR_ycbv_{obj_id}.pth", map_location=device))
        train_model(model, train_loader, test_loader, optimizer, num_epochs=45, save_path=f"weights/SATR_ycbv_{obj_id}.pth")
        #val_model(model, val_loader, criterion)



if __name__ == "__main__":
    train_R()