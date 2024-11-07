import os
import sys
sys.path.insert(0,os.getcwd()) # 把当前路径添加到 sys.path 中
import csv
import json
import time
import numpy as np
import torch
from torchvision import transforms
from network.centeruv_net import CenterUV_Net
from network.depth_net import DepthTransformer
import torch.nn.functional as F
from network.SAT6D import Pose_Net
from utils.allo_ego import psi, psi_tensor
from utils.dataset_loader.test_dataloader import get_test_dataloader
from scipy.spatial.transform import Rotation as R

camera_file = 'datasets/lmo/camera.json'
with open(camera_file, 'r') as c:
    camera = json.load(c)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Kc = torch.tensor([
    [camera['fx'], 0, camera['cx']],  # 第一行: [fx, 0, cx]
    [0, camera['fy'], camera['cy']],  # 第二行: [0, fy, cy]
    [0, 0, 1]                           # 第三行: [0, 0, 1]
], dtype=torch.float32).to(device)
Kc_inv = torch.inverse(Kc).to(device)
obj_ids = [1,5,6,8,9,10,11,12]

def write2csv(save_path, scene_id, im_id, obj_id, score, R, t, time):
    # 确保目录存在
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    batch_size = scene_id.shape[0]  # 获取 batch_size
    # print(f"scene_id: {scene_id.shape}, im_id: {im_id.shape}, obj_id: {obj_id.shape}, score: {score.shape}, R: {R.shape}, t: {t.shape}, time: {time.shape}")
    # 将数据按批次逐行写入 CSV 文件
    with open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # 遍历每个 batch 的数据
        for i in range(batch_size):
            row = [
                scene_id[i].cpu().item(),               # scene_id 单个元素
                im_id[i].cpu().item(),                  # im_id 单个元素
                obj_id[i].cpu().item(),                 # obj_id 单个元素
                score[i].cpu().item(),                  # score 单个元素
                ' '.join(map(str, R[i].detach().flatten().cpu().numpy())),  # 旋转矩阵 R (展平并以空格分隔)
                ' '.join(map(str, t[i].detach().flatten().cpu().numpy())),  # 平移向量 t (展平并以空格分隔)
                time[i].cpu().item() if isinstance(time, torch.Tensor) else time  # time 单个元素
            ]
            writer.writerow(row)  # 写入一行数据

    print(f"Batch data written to {save_path}")

def crop_xyxy_advanced(img, xyxy):
    batch_size = img.shape[0]
    
    # 将 xyxy 分解为 xmin, ymin, xmax, ymax 并确保它们是整数
    xmin = xyxy[:, 0].long()  # [batch_size]
    ymin = xyxy[:, 1].long()  # [batch_size]
    xmax = xyxy[:, 2].long()  # [batch_size]
    ymax = xyxy[:, 3].long()  # [batch_size]
    
    # 获取每个图片的高度和宽度
    # 对于每个图片使用不同的xmin,xmax,ymin,ymax进行索引裁剪
    cropped_imgs = []
    for i in range(batch_size):
        cropped_img = img[i, :, ymin[i]:ymax[i], xmin[i]:xmax[i]]  # 对每个图像进行裁剪
        cropped_imgs.append(transforms.Resize((128,128))(cropped_img))
    
    # 堆叠成 batch_size 的张量
    crop = torch.stack(cropped_imgs, dim=0)
    
    return crop

# 假设 rot_mat_gt 的形状为 (batch_size, 9)
def batch_rotmat_to_quat(rot_mat_gt):
    # 将 batch_size x 9 转换为 batch_size x 3 x 3
    batch_size = rot_mat_gt.shape[0]
    rot_matrices = rot_mat_gt.view(batch_size, 3, 3).cpu().detach().numpy()  # 转换为 numpy 数组

    # 使用 scipy 将旋转矩阵转换为四元数
    r = R.from_matrix(rot_matrices)  # 通过旋转矩阵创建 Rotation 对象
    quats = r.as_quat()  # 转换为四元数 (batch_size, 4)

    # 将结果转换为 torch 张量并返回
    quats = torch.tensor(quats, dtype=torch.float32).to(rot_mat_gt.device)
    return quats
def get_uv_depth_median(depth_img, v_pred, u_pred, patch_size=7):
    """
    从depth_img中以v_pred和u_pred为中心提取patch_size x patch_size的区域，并返回区域中值
    :param depth_img: Tensor of shape [batch_size, H, W], 深度图
    :param v_pred: Tensor of shape [batch_size], 每个样本的v坐标
    :param u_pred: Tensor of shape [batch_size], 每个样本的u坐标
    :param patch_size: int, 提取的区域大小，默认是7x7
    :return: Tensor of shape [batch_size], 区域中的中值
    """
    # Ensure patch size is odd for symmetric patches
    assert patch_size % 2 == 1, "Patch size must be odd"
    half_patch = patch_size // 2  # For a 7x7 patch, this will be 3

    # Get the image dimensions
    batch_size, H, W = depth_img.shape
    u_pred = u_pred.long()
    v_pred = v_pred.long()

    # Clamp the indices to ensure the patch doesn't go out of bounds
    v_min = torch.clamp(v_pred - half_patch, 0, H - patch_size)
    v_max = torch.clamp(v_pred + half_patch, half_patch, H)
    u_min = torch.clamp(u_pred - half_patch, 0, W - patch_size)
    u_max = torch.clamp(u_pred + half_patch, half_patch, W)

    # Step 2: Extract the patch around each (v_pred, u_pred)
    uv_depth_patch = []
    for i in range(batch_size):
        # Extract the patch from depth image
        patch = depth_img[i, v_min[i]:v_max[i] + 1, u_min[i]:u_max[i] + 1]
        uv_depth_patch.append(patch)

    # Convert list of patches to a tensor
    uv_depth_patch = torch.stack(uv_depth_patch)

    # Step 3: Flatten the patches and find the median
    uv_depth_median = torch.median(uv_depth_patch.view(batch_size, -1), dim=1)[0]

    return uv_depth_median

def test4bop(target_dir = 'datasets/lmo/test/000002', obj_id = 1, save_dir = 'results/lmo/SplitPose_lmo-test.csv'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_dir = target_dir  # RGB 图像目录
    
    gt_file = os.path.join(target_dir, 'object_info.json')  # GT 文件路径
    
    # 获取数据加载器
    test_loader = get_test_dataloader(target_dir, gt_file , obj_id)

    # 加载自定义网络模型

    rot_net = Pose_Net(d_model = 240, nhead=4, num_layers=4, num_samples =12).to(device)
    # rot_net = Rot_Net(d_model = 240, nhead=4, num_layers=4, num_samples =[1, 5, 10], window_sizes = [128, 64, 32]).to(device)
    center_net = CenterUV_Net().to(device)
    # depth_net = DepthTransformer(d_model=64, nhead=4, num_layers=4).to(device)
    rot_net.load_state_dict(torch.load(f"weights/rotv2_obj_{obj_id}.pth", map_location=device))
    rot_net.eval()
    center_net.load_state_dict(torch.load(f"weights/centeruv_obj_{obj_id}.pth", map_location=device))
    center_net.eval()
    # depth_net.load_state_dict(torch.load(f"weights/depth_obj_{obj_id}.pth", map_location=device))
    # depth_net.eval()



    # 打印部分数据样本
    for scene_id, img_id, obj_id, half_diameter, minxyxy, rgb, mask, depth_img, t_gt, centerxyxy_gt, centeruv_gt, centercrop_gt, rot_gt in test_loader:
        scene_id, img_id, obj_id, half_diameter, minxyxy, rgb, mask, depth_img, t_gt, centerxyxy_gt, centeruv_gt, centercrop_gt, rot_gt = scene_id.to(device), img_id.to(device), obj_id.to(device), half_diameter.to(device), minxyxy.to(device), rgb.to(device), mask.to(device), depth_img.to(device), t_gt.to(device), centerxyxy_gt.to(device), centeruv_gt.to(device), centercrop_gt.to(device), rot_gt.to(device)
        start_time = time.perf_counter()
        print(f'minxyxy:{minxyxy}')
        mincrop = crop_xyxy_advanced(rgb,minxyxy).to(device)
        print(mincrop.shape)
        centeruv_relative_pred = center_net(mincrop)
        print(centeruv_relative_pred.shape)    
        u_pred = centeruv_relative_pred[:, 0]*(minxyxy[:, 2]-minxyxy[:, 0])+minxyxy[:, 0]
        v_pred = centeruv_relative_pred[:, 1]*(minxyxy[:, 3]-minxyxy[:, 1])+minxyxy[:, 1]
        u_pred = torch.clamp(u_pred.long(), min=0, max=rgb.shape[3] - 1).to(device)
        v_pred = torch.clamp(v_pred.long(), min=0, max=rgb.shape[2] - 1).to(device)
        # u_pred,v_pred = centeruv_gt[:, 0],centeruv_gt[:, 1]
        centeruv_pred = torch.stack([u_pred, v_pred], dim=1)
        print(f"u_pred: {u_pred}, v_pred: {v_pred}")
        print(f"centeruv_gt: {centeruv_gt}")
        binary_mask = (mask > 0).float()  # [H, W]

        # 将 mask 扩展为与 RGB 图像通道对齐的形状
        # binary_mask = binary_mask.unsqueeze(0)  # [1, H, W]
        
        # 将 RGB 图像中 mask 为 0 的区域设置为 0
        masked_rgb = rgb * binary_mask 
        # print(f"rgb: {rgb.shape}")
        # print(f"mask: {mask.shape}")



        # print(f"rgba: {rgba.shape}")
        # 计算各个偏移值
        offsets = torch.stack([
            abs(minxyxy[:, 2] - u_pred),  # x 方向右侧的偏移
            abs(minxyxy[:, 0] - u_pred),  # x 方向左侧的偏移
            abs(minxyxy[:, 3] - v_pred),  # y 方向下方的偏移
            abs(minxyxy[:, 1] - v_pred)   # y 方向上方的偏移
        ], dim=1)

        # 在 dim=1 维度上取最大值，获取 halfside
        halfside, _ = torch.max(offsets, dim=1)

        print(f"halfside: {halfside.shape}")

        # 计算中心区域的坐标 (centerxyxy)
        centerxyxy = torch.stack((
            torch.clamp(u_pred.long() - halfside, min=0, max=rgb.shape[3] - 1),  # 左侧 x
            torch.clamp(v_pred.long() - halfside, min=0, max=rgb.shape[2] - 1),  # 上侧 y
            torch.clamp(u_pred.long() + halfside, min=0, max=rgb.shape[3] - 1),  # 右侧 x
            torch.clamp(v_pred.long() + halfside, min=0, max=rgb.shape[2] - 1)   # 下侧 y
        ), dim=1).to(device)

        print(f"centerxyxy: {centerxyxy}")
        print(f'centerxyxy_gt: {centerxyxy_gt}')
        centercrop = crop_xyxy_advanced(masked_rgb,centerxyxy).to(device)
        centercrop = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(centercrop)

        rgba_sample = centercrop[0]
        # 再将其映射到 [0, 255]
        transform = transforms.ToPILImage()
        img = transform(rgba_sample)
        img.save(f"visib/lmo_test_rgba/rgba_image_{img_id}_{obj_id}.png")

        # print(f'centercrop: {centercrop}')
        # print(f'centercrop_gt: {centercrop_gt}')
        # 深度和旋转预测
        
        
        
        sizealpha = 2*halfside/128
        sizealpha = sizealpha.unsqueeze(1)
        print(f'sizealpha: {sizealpha.shape}')

        rot_pred,depth_pred = rot_net(centercrop,sizealpha)
        rot_pred = rot_pred.view(-1, 3, 3)
        print(f"rot_mat:{rot_pred.shape}")
        print(f'centeruv_gt:{centeruv_gt}')
        print(f'centeruv_pred:{centeruv_pred}')
        centeruv_3d = torch.cat([centeruv_pred, torch.ones((centeruv_pred.shape[0],1)).to(device)], dim=1)
        print(f"centeruv_3d:{centeruv_3d.shape}")
        

        p = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(centeruv_pred.shape[0], 1).to(device)
        print(f'p:{p.shape}')

        q = torch.matmul(Kc_inv, centeruv_3d.T).T.to(device)
        print(f'q:{q.shape}')

        Rc=psi_tensor(p, q)
        print(f'Rc:{Rc.shape}')

        R = torch.matmul(Rc, rot_pred)
        print(f'R:{R}')

        print(f'rot_gt =:{rot_gt}')

        np.set_printoptions(threshold=np.inf)
        bbox_depth = crop_xyxy_advanced(depth_img,centerxyxy).to(device)
        bbox_mask = crop_xyxy_advanced(mask,centerxyxy).to(device)
        bbox_mask = (bbox_mask > 0).float()
        bbox_depth  = bbox_depth * bbox_mask
        print(f'bbox_depth:{bbox_depth.shape}')
        bbox_depth = transforms.Resize((32, 32))(bbox_depth)
        bbox_depth = bbox_depth.squeeze(0)
        print(f'bbox_depth:{bbox_depth.shape}')
        
        # depth_pred = depth_net(bbox_depth)
        depth_pred = depth_pred.unsqueeze(0)

        # quaternions = batch_rotmat_to_quat(rot_mat) 
        # depth_pred = t_gt[:, 2]

        x_c = (u_pred - camera['cx']) * depth_pred / camera['fx']
        y_c = (v_pred - camera['cy']) * depth_pred / camera['fy']
        print(f'x_c:{x_c}, y_c:{y_c}, depth_pred:{depth_pred}')
        print(f't_gt:{t_gt}')
        t = torch.stack([x_c, y_c, depth_pred], dim=1).detach()  # 转换为 NumPy 数组
        end_time = time.perf_counter()
        score=torch.tensor(1).repeat(scene_id.shape[0])
        elapsed_time=torch.tensor(-1).repeat(scene_id.shape[0])


        write2csv(save_dir,scene_id, img_id, obj_id, score,R,t,elapsed_time)
            


if __name__ == '__main__':
        # 清空指定文件
    file_path = 'results/lmo/SplitPose_lmo-test.csv'
    if os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass  # 清空文件内容
    for obj_id in obj_ids:
        test4bop(obj_id = obj_id)
        