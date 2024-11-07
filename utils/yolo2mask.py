import os
import cv2
import numpy as np

# 定义文件夹路径
label_folder = '/home/mendax/project/Split6D/dataset/VGA4/test/labels'  # 替换为包含txt标注文件的文件夹
image_folder = '/home/mendax/project/Split6D/dataset/VGA4/test/mask'  # 替换为对应的图像文件夹
output_folder = '/home/mendax/project/Split6D/dataset/VGA4/test'  # 替换为输出mask图像的文件夹

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def yolo_to_mask(txt_file, img_file, output_path):
    # 读取图像以获取尺寸
    img = cv2.imread(img_file)
    img_height, img_width = img.shape[:2]

    # 创建一个空白的mask图像
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    print("working")
    print(txt_file)
    # 打开标注文件
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            class_id = int(data[0])
            points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)

            # 解归一化坐标
            points[:, 0] = points[:, 0] * img_width  # x坐标
            points[:, 1] = points[:, 1] * img_height  # y坐标

            # 将解归一化后的点转为整数
            points = points.astype(np.int32)

            # 将多边形绘制到mask图像上
            cv2.fillPoly(mask, [points], color=255)

    # 保存mask图像

    cv2.imwrite(output_path, mask)
image_path='/home/mendax/project/Split6D/dataset/VGA4/test/mask/mask_test_0.png'
label_path='/home/mendax/project/Split6D/dataset/VGA4/test/labels/rgb_test_0.txt'
output_mask='/home/mendax/project/Split6D/dataset/VGA4/test/test_0_mask.png'

yolo_to_mask(label_path, image_path, output_mask)
# 遍历标注文件夹中的每个txt文件
# for label_file in os.listdir(label_folder):
#     print(label_file)
#     if label_file.endswith(".txt"):
#         label_path = os.path.join(label_folder, label_file)
#         image_file = label_file.replace('.txt', '.png')  # 假设图像文件为.jpg
#         image_path = os.path.join(image_folder, image_file)

#         if os.path.exists(image_path):
#             output_mask = os.path.join(output_folder, label_file.replace('.txt', '_mask.png'))
#             yolo_to_mask(label_path, image_path, output_mask)
