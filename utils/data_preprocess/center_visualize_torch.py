import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image

# 切换到脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 读取图像并转换为 Tensor（使用 PIL 和 torchvision.transforms）
image_path = '../lmo/pbr/train_pbr/000000/mask/000000_000000.png'
image_pil = Image.open(image_path)

# 转换图像为 Tensor
transform = transforms.ToTensor()
image_tensor = transform(image_pil)

# 图像的形状是 (C, H, W)，所以需要把它转换为 (H, W, C) 以符合显示要求
image_np = image_tensor.permute(1, 2, 0).numpy()

# 读取中心点数据和边界框信息
with open('../lmo/pbr/train_pbr/000000/object_centers.json') as f_centers, open('../lmo/pbr/train_pbr/000000/scene_gt_info.json') as f_info:
    centers = json.load(f_centers)
    scene_gt_info = json.load(f_info)

# 获取中心点的坐标（以索引 "0" 为例）
center_data = centers["0"]
u, v = int(center_data['u']), int(center_data['v'])
u_relative = center_data['u_relative']
v_relative = center_data['v_relative']

# 获取 xyxy（bbox_obj）信息
bbox_obj = scene_gt_info["0"][0]['bbox_obj']
xmin, ymin, width, height = bbox_obj
xmax = xmin + width
ymax = ymin + height

# 计算相对中心点的坐标
u_relative_abs = int(xmin + u_relative * width)
v_relative_abs = int(ymin + v_relative * height)

# 在图像上标记绝对坐标的红色圆点和相对坐标的蓝色圆点

# 使用 PyTorch 的方式来模拟 OpenCV 的 circle 标记（直接操作 NumPy 图像数据）
# 标记红色绝对坐标中心点 (u, v)
image_np = np.array(image_np)
image_np = torch.tensor(image_np)
radius = 5

def draw_circle(image_tensor, u, v, color):
    """在给定的图像 tensor 上标记一个颜色圆点"""
    color_tensor = torch.tensor(color)  # 将颜色转换为 (3,) 形状的 tensor
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            if 0 <= u + i < image_tensor.shape[1] and 0 <= v + j < image_tensor.shape[0]:
                if i**2 + j**2 <= radius**2:
                    image_tensor[v + j, u + i, :] = color_tensor  # 赋值给图像张量



# 绝对坐标标记红色圆点 (u, v)
draw_circle(image_np, u, v, [1, 0, 0])  # 红色圆点 RGB 格式

# 转换为 NumPy 以便使用 matplotlib 显示
image_marked = image_np.numpy()
plt.imshow(image_marked)
plt.title("Object Center Marked in Red (absolute)")
plt.show()

# 保存标记了中心点的图像
plt.imsave("media/center_mask_with_abusolute_torch.png", image_marked)

# 相对坐标标记蓝色圆点 (u_relative_abs, v_relative_abs)
draw_circle(image_np, u_relative_abs, v_relative_abs, [0, 0, 1])  # 蓝色圆点 RGB 格式

# 转换为 NumPy 以便使用 matplotlib 显示
image_marked_relative = image_np.numpy()
plt.imshow(image_marked_relative)
plt.title("Object Center Marked in Blue (relative)")
plt.show()

# 保存标记了中心点的图像
plt.imsave("media/center_mask_with_relative_torch.png", image_marked_relative)
