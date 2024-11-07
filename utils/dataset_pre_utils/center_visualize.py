import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# 切换到脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 读取图像
image_path = '/home/mendax/project/SATPose/datasets/lmo/pbr/bop_data/lmo/train_pbr/000000/rgb/000000.png'
image = cv2.imread(image_path)

# 读取中心点数据和边界框信息
with open('/home/mendax/project/SATPose/datasets/lmo/pbr/bop_data/lmo/train_pbr/000000/object_info.json') as f_centers,\
    open('/home/mendax/project/SATPose/datasets/lmo/pbr/bop_data/lmo/train_pbr/000000/scene_gt_info.json') as f_info:
    centers = json.load(f_centers)
    scene_gt_info = json.load(f_info)

# 获取中心点的坐标（以索引 "0" 为例）
center_data = centers[0]
u, v = center_data['uv']
uv_relative = center_data['uv_relative']
uv_newview_relative = center_data['uv_newview_relative']
uv_newview = center_data['uv_newview']

# 获取 xyxy（bbox_obj）信息
minxyxy = center_data['minxyxy']
xmin, ymin, xmax, ymax = minxyxy
width = xmax - xmin
height= ymax - ymin

# 计算相对中心点的坐标
u_relative_abs = int(xmin + uv_relative[0] * width)
v_relative_abs = int(ymin + uv_relative[1] * height)

minxyxy_newview = center_data['minxyxy_newview']
xmin, ymin, xmax, ymax = minxyxy_newview
width = xmax - xmin
height= ymax - ymin

# 计算相对中心点的坐标
u_newview_relative_abs = int(xmin + uv_newview_relative[0] * width)
v_newview_relative_abs = int(ymin + uv_newview_relative[1] * height)


# 在图像上绘制红色圆点（绝对坐标）和蓝色圆点（相对坐标）
cv2.circle(image, (u, v), radius=5, color=(0, 0, 255), thickness=-1)  # 红色圆点 (OpenCV 使用 BGR 格式)

# 显示图像（使用 matplotlib 因为 OpenCV 显示颜色格式是 BGR）
image_redmark = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式以便使用 matplotlib
plt.imshow(image_redmark)
plt.title("Object Center Marked in Red (absolute)")
plt.show()

# 保存标记了中心点的图像
plt.imsave("media/center_mask_with_abusolute.png", image_redmark)

cv2.circle(image, (u_relative_abs, v_relative_abs), radius=5, color=(255, 0, 0), thickness=-1)  # 蓝色圆点
# 显示图像（使用 matplotlib 因为 OpenCV 显示颜色格式是 BGR）
image_bluemark = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式以便使用 matplotlib
plt.imshow(image_bluemark)
plt.title("Object Center Marked in Blue (relative)")
plt.show()

# 保存标记了中心点的图像
plt.imsave("media/center_mask_with_relative.png", image_bluemark)


image = cv2.imread("/home/mendax/project/SATPose/datasets/lmo/pbr/bop_data/lmo/train_pbr/000000/newview/000000.png")
cv2.circle(image, uv_newview, radius=5, color=(0, 255, 0), thickness=-1)  # 绿色圆点
cv2.circle(image, (u_newview_relative_abs, v_newview_relative_abs), radius=5, color=(255, 0, 0), thickness=-1)  # 蓝色圆点
cv2.circle(image, (minxyxy_newview[0],minxyxy_newview[1]), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色圆点
cv2.circle(image, (minxyxy_newview[2],minxyxy_newview[3]), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色圆点
# 显示图像（使用 matplotlib 因为 OpenCV 显示颜色格式是 BGR）
image_greenmark = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式以便使用 matplotlib
plt.imshow(image_greenmark)
plt.title("Object Center Marked in Blue (relative)")
plt.show()

# 保存标记了中心点的图像
plt.imsave("media/center_new_view.png", image_greenmark)

