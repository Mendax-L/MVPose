import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

def Center_Distance(R, t, uv1, uv2, K1, K2):
    # 提取旋转矩阵和平移向量
    # print(R,t)
    r11, r12, r13 = R[0, :]
    r21, r22, r23 = R[1, :]
    r31, r32, r33 = R[2, :]
    tx, ty, tz = t

    # 提取坐标点 u1, v1 和 u2, v2
    u1, v1 = uv1
    u2, v2 = uv2

    # 提取内参矩阵 K1 和 K2
    fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
    fx2, fy2, cx2, cy2 = K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]

    # 计算 A1
    A1 = (
        (r31 * (u1 - cx1) * (u2 - cx2)) / (fx1 * fx2)
        + (r32 * (v1 - cy1) * (u2 - cx2)) / (fy1 * fx2)
        + (r33 * (u2 - cx2)) / fx2
        - (r11 * (u1 - cx1)) / fx1
        - (r12 * (v1 - cy1)) / fx1
        - r13
    )

    # 计算 A2
    A2 = (
        (r31 * (u1 - cx1) * (v2 - cy2)) / (fx1 * fy2)
        + (r32 * (v1 - cy1) * (v2 - cy2)) / (fy1 * fy2)
        + (r33 * (v2 - cy2)) / fy2
        - (r21 * (u1 - cx1)) / fy1
        - (r22 * (v1 - cy1)) / fy1
        - r23
    )

    # 计算 b1 和 b2
    b1 = tx - (tz * (u2 - cx2)) / fx2
    b2 = ty - (tz * (v2 - cy2)) / fy2

    # 计算 Z
    Z = (A1 * b1 + A2 * b2) / (A1**2 + A2**2)

    return Z


def t_From_Multiviews(Rc_list, tc_list, uv_preds, Kc_list):
    # 初始化
    device = Rc_list[0].device

    frames_len = len(Rc_list)
    step = max(1, int(0.1 * frames_len))  # 确保步长至少为 1
    t_preds = []
    for j in range(len(Rc_list)):
        z_values = []
        for i, (Rc, tc, uv, Kc) in enumerate(zip(Rc_list, tc_list, uv_preds, Kc_list)):
            Rc_r = Rc @ torch.inverse(Rc_list[j])
            tc_r = tc - Rc_r @ tc_list[j] 
            z = Center_Distance(Rc_r, tc_r, uv_preds[j], uv, Kc_list[j], Kc)
            z_values.append(z)
        
        z_values = torch.stack(z_values)  # 将列表转为 PyTorch 张量
        z_positive = z_values[z_values > 0]
        z_values_np = z_positive.cpu().detach().numpy()
        # 使用 DBSCAN 聚类
        dbscan = DBSCAN(eps=15, min_samples=5)  # eps 是距离阈值，min_samples 是最小样本数
        z_values_np_reshaped = z_values_np.reshape(-1, 1)
        labels = dbscan.fit_predict(z_values_np_reshaped)

        # 找到非噪声点的簇
        valid_clusters = labels[labels != -1]  # -1 表示噪声点
        largest_cluster = np.argmax(np.bincount(valid_clusters))

        # 选出最大簇的深度值
        cluster_values = z_values_np[labels == largest_cluster]

        # 计算该簇的平均深度
        z_pred = torch.tensor(cluster_values.mean(), device='cuda')

        # z_positive = z_values[z_values > 0]
        # if len(z_positive) > 0:
        #     z_pred = z_positive.mean()
        # else:
        #     z_pred = z_values.mean()
        fx, fy, cx, cy = Kc_list[j][0, 0], Kc_list[j][1, 1], Kc_list[j][0, 2], Kc_list[j][1, 2]
        x_pred = (uv_preds[j][0] - cx) * z_pred / fx
        y_pred = (uv_preds[j][1] - cy) * z_pred / fy
        t_preds.append(torch.tensor([x_pred, y_pred, z_pred]).to(device))

    return t_preds
        
    # print(f'z_values: {len(z_values)}'
    # print(f'z_values: {z_values}')
    

    # 计算平均深度
    # z_values_np = z_values.cpu().detach().numpy()
    # # 使用 DBSCAN 聚类
    # dbscan = DBSCAN(eps=20, min_samples=5)  # eps 是距离阈值，min_samples 是最小样本数
    # z_values_np_reshaped = z_values_np.reshape(-1, 1)
    # labels = dbscan.fit_predict(z_values_np_reshaped)

    # # 找到非噪声点的簇
    # valid_clusters = labels[labels != -1]  # -1 表示噪声点
    # largest_cluster = np.argmax(np.bincount(valid_clusters))

    # # 选出最大簇的深度值
    # cluster_values = z_values_np[labels == largest_cluster]

    # # 计算该簇的平均深度
    # z_pred = torch.tensor(cluster_values.mean(), device='cuda')

    # print(f'z_pred: {z_pred}')

    # 从主视图反算 x, y, z 的预测值

    # 更新所有视图下的 t_pred

