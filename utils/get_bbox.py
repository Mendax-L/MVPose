import numpy as np
import torch
from PIL import Image

def get_min_bbox(mask):
    # 如果 mask 是 PIL 图像，则将其转换为 PyTorch 张量
    if isinstance(mask, Image.Image):
        mask = torch.tensor(np.array(mask))

    # 查找mask中不为0的像素的坐标
    coords = torch.nonzero(mask > 0, as_tuple=False)
    
    if len(coords) == 0:
        return None  # 如果没有有效的mask数据

    # 获取最小的外接矩形
    x_min = max(torch.min(coords[:, 1]).item(), 0)  # x 是图像的列坐标
    x_max = min(torch.max(coords[:, 1]).item(), mask.shape[1])
    y_min = max(torch.min(coords[:, 0]).item(), 0)  # y 是图像的行坐标
    y_max = min(torch.max(coords[:, 0]).item(), mask.shape[0])
    x_c, y_c = (x_max + x_min) / 2, (y_max + y_min) / 2

    side = max((x_max - x_min), (y_max - y_min)) / 2

    bounding_box = (int(x_c - side), int(y_c - side), int(x_c + side), int(y_c + side))
    # 边界条件未解决
    return bounding_box

def get_center_bbox(cxcy, mask):
    # 如果 mask 是 PIL 图像，则将其转换为 PyTorch 张量
    if isinstance(mask, Image.Image):
        mask = torch.tensor(np.array(mask))

    # 查找mask中不为0的像素的坐标
    coords = torch.nonzero(mask > 0, as_tuple=False)
    
    if len(coords) == 0:
        return None  # 如果没有有效的mask数据

    # 获取最小的外接矩形
    x_min = max(torch.min(coords[:, 1]).item(), 0)  # x 是图像的列坐标
    x_max = min(torch.max(coords[:, 1]).item(), mask.shape[1])
    y_min = max(torch.min(coords[:, 0]).item(), 0)  # y 是图像的行坐标
    y_max = min(torch.max(coords[:, 0]).item(), mask.shape[0])

    side = max(abs(x_max - cxcy[0]), abs(x_min - cxcy[0]), abs(y_max - cxcy[1]), abs(y_min - cxcy[1]))

    bounding_box = (int(cxcy[0] - side), int(cxcy[1] - side), int(cxcy[0] + side), int(cxcy[1] + side))
    # 边界条件未解决
    return bounding_box

# 示例用法
if __name__ == "__main__":
    mask = torch.tensor([[0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]])

    bounding_box = get_min_bbox(mask)
    print("Bounding box coordinates:", bounding_box)
