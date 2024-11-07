import torch

def get_bounding_box_batch(masks):
    bounding_boxes = []
    
    # 假设masks的维度为 (batch_size, height, width)
    for mask in masks:
        # 查找mask中不为0的像素的坐标
        coords = torch.nonzero(mask > 0, as_tuple=False)

        if coords.size(0) == 0:
            bounding_boxes.append(None)  # 如果没有有效的mask数据
            continue

        # 获取最小的外接矩形
        x_min, y_min = torch.min(coords, dim=0)[0]
        x_max, y_max = torch.max(coords, dim=0)[0]

        x_c, y_c = (x_max + x_min).float() / 2, (y_max + y_min).float() / 2

        side = max((x_max - x_min).item(), (y_max - y_min).item()) / 2

        bounding_box = (int(x_c - side), int(y_c - side), int(x_c + side), int(y_c + side))
        bounding_boxes.append(bounding_box)
    
    return bounding_boxes

# 示例用法
# 假设有一个批次的二值mask，维度为 (batch_size, height, width)
masks = torch.tensor([[[0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0]],
                      
                      [[0, 0, 1, 1, 0],
                       [0, 0, 1, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]])

bounding_boxes = get_bounding_box_batch(masks)
print("Bounding box coordinates for batch:", bounding_boxes)
