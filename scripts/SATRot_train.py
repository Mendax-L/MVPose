import os
import sys

sys.path.insert(0,os.getcwd()) # 把当前路径添加到 sys.path 中

import torch
import torch.optim as optim
from tqdm import tqdm
import os
from utils.loader.SATRot_loader import SATRot_loader
from lib.SATRot import SATRot
from utils.earlystop import EarlyStopping
from lib.loss import criterion_R, criterion_uv
from lib.config import Kc_lmo, Kc_lmo_inv, SATRot_transform
from lib.to_allo import to_allo


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = SATRot(d_model = 240, nhead=4, num_layers=4, num_samples =10,).to(device)  # 将模型移到 GPU
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# Training loop
def train_model(model, train_loader, test_loader, optimizer, num_epochs=30, save_path="model.pth"):
    R_early_stopping = EarlyStopping(patience=5, min_delta=0.02)
    uv_early_stopping = EarlyStopping(patience=5, min_delta=0.02)
    model.train()
    # 初始化验证损失最小值为正无穷
    best_R_loss = float("inf")
    best_uv_loss = float("inf")
    best_model_path = save_path  # 存储损失最小的模型路径

    if len(train_loader.dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data source.")
    
    # for epoch in range(num_epochs):
    for epoch in range(num_epochs):
        R_trainloss = 0.0
        uv_trainloss = 0.0
        for rgb_inputs, uv_gt, R_gt in tqdm(train_loader):
            rgb_inputs, uv_gt, R_gt = rgb_inputs.to(device), uv_gt.to(device), R_gt.to(device)  # 将数据移到 GPU

            # print(rgb_inputs.shape)
            optimizer.zero_grad()
            uv_pred, R_pred = model(rgb_inputs)

            R_pred = to_allo(R_pred)

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
            for rgb_inputs, uv_gt, R_gt in test_loader:
                rgb_inputs, uv_gt, R_gt = rgb_inputs.to(device), uv_gt.to(device), R_gt.to(device)
                
                uv_pred, R_pred = model(rgb_inputs)

                R_pred = to_allo(R_pred)

                R_loss = criterion_R(R_pred, R_gt)
                uv_loss = criterion_uv(uv_pred, uv_gt)

                R_valloss += R_loss.item() * rgb_inputs.size(0)
                uv_valloss += uv_loss.item() * rgb_inputs.size(0)
        
        R_valloss /= len(test_loader.dataset)
        uv_valloss /= len(test_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, ValR_loss: {R_valloss:.4f},Valuv_loss: {uv_valloss}")
        
                # 更新最佳模型
        if R_valloss < best_R_loss and uv_valloss < best_uv_loss:
            best_R_loss = R_valloss
            best_uv_loss = uv_valloss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with ValR_loss: {R_valloss:.4f},Valuv_loss: {uv_valloss}")

        # Check early stopping
        if R_early_stopping.check_early_stopping(R_valloss) and uv_early_stopping.check_early_stopping(uv_valloss):
            print(f"Early stopping at epoch {epoch}.")
            break
    print(f"Best model saved in: {uv_valloss} with loss R: {R_valloss} uv: {uv_valloss}")


def train_R(obj_ids):
    # Paths to images and their corresponding targets
    if obj_ids == None:
        obj_ids = [1,5,6,8,9,10,11,12]
    for obj_id in obj_ids:
        train_target_dirs = [f'datasets/lmo/train/{str(obj_id).zfill(6)}', f'datasets/lm/{str(obj_id).zfill(6)}'] # RGB 图像目录
        test_target_dirs = [f'datasets/lmo/test/000002']  # RGB 图像目录
        train_loader = SATRot_loader(target_dirs = train_target_dirs, obj_id=obj_id, transform =SATRot_transform)
        test_loader = SATRot_loader(target_dirs = test_target_dirs, obj_id=obj_id, transform =None)
    
        # Train and val the model
        train_model(model, train_loader, test_loader, optimizer, num_epochs=45, save_path=f"weights/SATR_obj_{obj_id}.pth")
        #val_model(model, val_loader, criterion)

        # model.load_state_dict(torch.load(f"weights/SATR_obj_{obj_id}.pth", map_location=device))


if __name__ == "__main__":
    obj_ids = [1,5,6,8,9,10,11,12]
    train_R(obj_ids)