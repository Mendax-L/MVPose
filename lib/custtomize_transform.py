from PIL import Image
import random

class RandomHolesRGB:
    def __init__(self, p=1, occlusion_size=(0.5, 0.15), num_holes=random.randint(5, 15)):
        """
        p: 应用遮挡的概率
        occlusion_size: 遮挡区域占图像比例的范围 (最小比例, 最大比例)
        num_holes: 生成的遮挡块数量
        """
        self.p = p
        self.occlusion_size = occlusion_size
        self.num_holes = num_holes

    def __call__(self, img):
        if random.random() < self.p:
            # 获取图像尺寸
            img_width, img_height = img.size

            for _ in range(self.num_holes):
                # 随机生成遮挡区域的大小和位置
                occlusion_w = int(random.uniform(self.occlusion_size[0], self.occlusion_size[1]) * img_width)
                occlusion_h = int(random.uniform(self.occlusion_size[0], self.occlusion_size[1]) * img_height)

                occlusion_x = int(random.uniform(0, img_width - occlusion_w))
                occlusion_y = int(random.uniform(0, img_height - occlusion_h))

                # 创建纯黑色遮挡区域
                occlusion_area = Image.new("RGB", (occlusion_w, occlusion_h), color=(0,0,0))

                # 将遮挡区域粘贴到原图上
                img.paste(occlusion_area, (occlusion_x, occlusion_y))

        return img
    
class RandomOcclusionRGB:
    def __init__(self, p=0.6, occlusion_size=(0.2, 0.5)):
        """
        p: 应用遮挡的概率
        occlusion_size: 遮挡区域占图像比例的范围 (最小比例, 最大比例)
        """
        self.p = p
        self.occlusion_size = occlusion_size

    def __call__(self, img):
        if random.random() < self.p:
            # 获取图像尺寸
            img_width, img_height = img.size

            # 随机生成遮挡区域的大小和位置
            occlusion_w = random.uniform(self.occlusion_size[0], self.occlusion_size[1]) * img_width
            occlusion_h = random.uniform(self.occlusion_size[0], self.occlusion_size[1]) * img_height

            occlusion_x = random.uniform(0, img_width - occlusion_w)
            occlusion_y = random.uniform(0, img_height - occlusion_h)
            random_color = (0, 0, 0)
            # 创建遮挡区域并填充为随即色块
            occlusion_area = Image.new("RGB", (int(occlusion_w), int(occlusion_h)), random_color)  # 黑色区域
            
            # 将遮挡区域粘贴到原图上
            img.paste(occlusion_area, (int(occlusion_x), int(occlusion_y)))

        return img