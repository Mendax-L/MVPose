import os
import cv2

# 定义文件夹路径
mode="train"
mask_folder = f'dataset/VGA/{mode}/mask'  # 替换为你 mask 图片所在的文件夹
output_folder = f'dataset/VGA/{mode}/labels'  # 替换为输出标注文件的路径

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def convert_mask_to_yolo(mask_path, output_path, class_id=0):
    # 读取mask图像
    mask = cv2.imread(mask_path, 0)
    img_height, img_width = mask.shape

    # 提取轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 打开输出文件
    with open(output_path, 'w') as f:
        for contour in contours:
            # 归一化轮廓坐标
            normalized_contour = []
            for point in contour:
                x, y = point[0]
                normalized_x = x / img_width
                normalized_y = y / img_height
                normalized_contour.append((normalized_x, normalized_y))
            
            # 写入YOLOv8格式
            f.write(f"{class_id}")
            for point in normalized_contour:
                f.write(f" {point[0]} {point[1]}")
            f.write("\n")

# 遍历mask文件夹中的每个文件
for mask_file in os.listdir(mask_folder):
    if mask_file.endswith(".png"):  # 假设mask图像为.png格式
        print("working")
        mask_path = os.path.join(mask_folder, mask_file)
        output_file = os.path.splitext(mask_file)[0] + ".txt"
        output_path = os.path.join(output_folder, output_file)
        print(output_path)
        convert_mask_to_yolo(mask_path, output_path)
