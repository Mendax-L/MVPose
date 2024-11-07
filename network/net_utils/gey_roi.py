import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

def get_roi_and_resize_batch(image_tensor, bounding_box_tensor, output_size=(128, 128)):
    """
    输入:
    - image_tensor: 输入图片张量，形状为 (B, C, H, W)，B为批次大小
    - bounding_box_tensor: 包围框张量，形状为 (B, 4)，每个包围框 (x_min, y_min, x_max, y_max)
    - output_size: 调整大小后的目标尺寸 (width, height)

    输出:
    - roi_resized: 裁剪并调整大小后的ROI张量，形状为 (B, C, 128, 128)
    """
    batch_size = image_tensor.shape[0]
    rois_resized = []

    for i in range(batch_size):
        x_min, y_min, x_max, y_max = bounding_box_tensor[i]

        # 裁剪每个样本的ROI
        roi = image_tensor[i, :, y_min:y_max, x_min:x_max]

        # 调整ROI大小为指定的output_size
        roi_resized = TF.resize(roi, output_size)
        rois_resized.append(roi_resized)

    # 将所有处理后的ROI张量合并为一个batch
    rois_resized = torch.stack(rois_resized)

    return rois_resized

# 示例用法
# 假设图片张量有批次大小为2，3通道 (C, H, W)，并且包围框为 (x_min, y_min, x_max, y_max)
image_tensor = torch.randn(2, 3, 256, 256)  # 生成一个批量为2的3通道图像张量
bounding_box_tensor = torch.tensor([[50, 50, 200, 200], [30, 30, 180, 180]])  # 批量包围框

roi_resized = get_roi_and_resize_batch(image_tensor, bounding_box_tensor)
print("Resized ROI shape:", roi_resized.shape)  # 应该输出 (2, 3, 128, 128)
