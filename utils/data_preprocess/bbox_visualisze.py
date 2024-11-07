import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import os
print(os.getcwd())  # 查看当前工作目录

# # 切换到脚本所在的目录
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

def get_min_bbox(mask):
        # 查找 mask 中非零像素的坐标
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return None  # 如果没有有效的 mask 数据

    # 获取最小的外接矩形
    y_min, x_min = np.min(coords, axis=0)
    y_max, x_max = np.max(coords, axis=0)

    bounding_box = (x_min, y_min, x_max, y_max)

    # 处理边界条件，确保 bounding_box 不超出图像边界
    img_h, img_w = mask.shape[:2]
    x_min = max(bounding_box[0], 0)
    y_min = max(bounding_box[1], 0)
    x_max = min(bounding_box[2], img_w)
    y_max = min(bounding_box[3], img_h)

    return (x_min, y_min, x_max, y_max)

# 读取图像
image_path = 'datasets/lmo/train/000001/rgb/000009.png'
image = cv2.imread(image_path)
mask_path = 'datasets/lmo/train/000001/mask/000009_000000.png'
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 读取中心点数据和边界框信息
with open('datasets/lmo/train/000001/scene_gt_info.json') as f_info:
    scene_gt_info = json.load(f_info)
with open(f"datasets/lmo/train/000001/object_info.json") as f_centers:
    centers = json.load(f_centers)

# 获取 xyxy（bbox_obj）信息
bbox_obj = scene_gt_info["1"][0]['bbox_obj']
xmin, ymin, width, height = bbox_obj
xmax = xmin + width
ymax = ymin + height

center_data = centers["8"]
u, v = int(center_data['u']), int(center_data['v'])
real_width, real_height = int(center_data['width']), int(center_data['height'])
real_xmin, real_ymin, real_xmax, real_ymax = center_data['minxyxy']

# 绘制边界框（绿色）
cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
# cv2.rectangle(image, (real_xmin, real_ymin), (real_xmax, real_ymax), color=(0, 255, 255), thickness=1)

# 显示图像（使用 matplotlib，因为 OpenCV 显示颜色格式是 BGR）
image_marked = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式以便使用 matplotlib
plt.imshow(image_marked)
plt.title("Object Center and Bounding Box")
plt.show()

# 保存标记了中心点和边界框的图像
plt.imsave("media/bbox_mask.png", image_marked)
