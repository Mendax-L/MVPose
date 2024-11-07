import torch
import torch.nn.functional as F

def compute_loss_R(me, gt):
    """
    Computes the loss \mathcal{L}_R as the average L1 norm of the difference between reconstructed and target images.
    
    Args:
    - reconstructed_images (torch.Tensor): The tensor of reconstructed images, shape (N, C, H, W)
    - target_images (torch.Tensor): The tensor of target images, shape (N, C, H, W)
    
    Returns:
    - loss (torch.Tensor): The computed loss \mathcal{L}_R
    """

    if isinstance(me, list):
        me = torch.tensor(me)
    if isinstance(gt, list):
        gt = torch.tensor(gt)

    # Compute the L1 loss (mean absolute error) between the reconstructed and target images
    loss = F.l1_loss(me, gt, reduction='mean')
    
    return loss

import torch

def angular_loss(R_pred, R_true):
    """
    计算两个旋转矩阵之间的角度误差
    
    参数:
    R_pred -- 预测的旋转矩阵 (batch_size, 3, 3)
    R_true -- 真实的旋转矩阵 (batch_size, 3, 3)
    
    返回:
    平均角度误差 (标量)
    """
    batch_size = R_pred.shape[0]
    RT_R = (torch.bmm(R_pred.transpose(1, 2), R_true))
    # RT_R = torch.matmul(R_pred.transpose(1, 2), R_true)  # R_pred^T * R_true
    trace = torch.diagonal(RT_R, dim1=1, dim2=2).sum(-1)  # 计算对角线元素之和
    trace_clamped = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    
    # 计算角度误差
    theta = torch.acos(trace_clamped)
    return torch.mean(theta)

import torch

def quaternion_distance(q1, q2):
    """
    计算两个四元数之间的距离
    
    参数:
    q1 -- 预测的四元数 (batch_size, 4)
    q2 -- 真实的四元数 (batch_size, 4)
    
    返回:
    平均四元数距离 (标量)
    """
    q1 = q1 / torch.norm(q1, dim=1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=1, keepdim=True)
    d = torch.abs(torch.sum(q1 * q2, dim=1))
    theta = 2 * torch.acos(d)
    return torch.mean(theta)


if __name__=="__main__":
    # 示例数据
    R_pred = torch.eye(3).unsqueeze(0).repeat(10, 1, 1)  # 预测的旋转矩阵
    R_true = torch.eye(3).unsqueeze(0).repeat(10, 1, 1)  # 真实的旋转矩阵

    loss = angular_loss(R_pred, R_true)
    print("Angular Loss:", loss.item())
    # 示例数据
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(10, 1)  # 预测的四元数
    q2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(10, 1)  # 真实的四元数

    loss = quaternion_distance(q1, q2)
    print("Quaternion Distance:", loss.item())
