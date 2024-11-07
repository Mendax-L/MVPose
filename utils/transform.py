from PIL import Image, ImageOps, ImageFilter
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF

class RandomInvertRGBA:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            r, g, b, a = img.split()
            r = ImageOps.invert(r)
            g = ImageOps.invert(g)
            b = ImageOps.invert(b)
            a = ImageOps.invert(a)

            img = Image.merge('RGBA', (r, g, b, a))
        return img

class RandomPosterizeRGBA:
    def __init__(self, bits=3, p=0.5):
        self.bits = bits
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            r, g, b, a = img.split()
            r = ImageOps.posterize(r, self.bits)
            g = ImageOps.posterize(g, self.bits)
            b = ImageOps.posterize(b, self.bits)
            img = Image.merge('RGBA', (r, g, b, a))
        return img

class RandomSolarizeRGBA:
    def __init__(self, threshold=128, p=0.5):
        self.threshold = threshold
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            r, g, b, a = img.split()
            r = ImageOps.solarize(r, self.threshold)
            g = ImageOps.solarize(g, self.threshold)
            b = ImageOps.solarize(b, self.threshold)
            img = Image.merge('RGBA', (r, g, b, a))
        return img
    

class RandomBlur:
    def __init__(self, p=0.5, max_radius=5):
        """
        随机模糊：高斯模糊、运动模糊、平均模糊等。
        - p: 施加模糊的概率。
        - max_radius: 控制模糊的半径，值越高模糊越强。
        """
        self.p = p
        self.max_radius = max_radius

    def __call__(self, img):
        if random.random() < self.p:
            blur_type = random.choice(['Gaussian', 'Motion', 'Average'])
            radius = random.uniform(1, self.max_radius)

            if blur_type == 'Gaussian':
                return img.filter(ImageFilter.GaussianBlur(radius=radius))

            elif blur_type == 'Motion':
                return self.motion_blur(img, int(radius * 2))

            elif blur_type == 'Average':
                return img.filter(ImageFilter.BoxBlur(radius))

        return img

    def motion_blur(self, img, length):
        """
        自定义运动模糊。
        - length: 模糊长度。
        """
        # 旋转图片实现运动模糊效果
        angle = random.choice([0, 45, 90, 135])
        img = img.rotate(angle)
        img = img.filter(ImageFilter.GaussianBlur(radius=length / 2))
        return img.rotate(-angle)


import torch

import torch

class NormalizeRGB:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        初始化正则化参数，仅用于RGB通道
        :param mean: RGB通道的均值，默认使用ImageNet的均值
        :param std: RGB通道的标准差，默认使用ImageNet的标准差
        """
        # 确保mean和std是单一RGB通道均值值
        if not isinstance(mean, (tuple, list)) or len(mean) != 3:
            raise ValueError("mean应该是包含三个元素的元组或列表，例如 (0.485, 0.456, 0.406)")
        if not isinstance(std, (tuple, list)) or len(std) != 3:
            raise ValueError("std应该是包含三个元素的元组或列表，例如 (0.229, 0.224, 0.225)")

        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, img):
        """
        将正则化应用到输入图像的前三个通道（RGB），保留A通道不变
        :param img: 输入图像 (Tensor格式，形状为[4, H, W])
        :return: 正则化后的图像
        """
        # 分离RGB和A通道
        rgb = img[:3]
        a = img[3:]
        
        # 对RGB通道进行正则化，A通道保持不变
        rgb = (rgb - self.mean) / self.std
        return torch.cat([rgb, a], dim=0)




class RandomAdjustSharpnessRGBA:
    def __init__(self, sharpness_factor=2, p=0.3):
        self.sharpness_factor = sharpness_factor
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            r, g, b, a = img.split()
            r = r.filter(ImageFilter.SHARPEN)
            g = g.filter(ImageFilter.SHARPEN)
            b = b.filter(ImageFilter.SHARPEN)
            # 锐化因子大于1时，可以多次应用滤波器
            for _ in range(int(self.sharpness_factor) - 1):
                r = r.filter(ImageFilter.SHARPEN)
                g = g.filter(ImageFilter.SHARPEN)
                b = b.filter(ImageFilter.SHARPEN)
            img = Image.merge('RGBA', (r, g, b, a))
        return img
    
class RandomAutocontrastRGBA:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            r, g, b, a = img.split()
            r = ImageOps.autocontrast(r)
            g = ImageOps.autocontrast(g)
            b = ImageOps.autocontrast(b)
            img = Image.merge('RGBA', (r, g, b, a))
        return img

class RandomEqualizeRGBA:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            r, g, b, a = img.split()
            r = ImageOps.equalize(r)
            g = ImageOps.equalize(g)
            b = ImageOps.equalize(b)
            img = Image.merge('RGBA', (r, g, b, a))
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

    
class RandomHolesRGBA:
    def __init__(self, p=1, occlusion_size=(0.15, 0.25), num_holes=random.randint(10, 20)):
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
                occlusion_w = random.uniform(self.occlusion_size[0], self.occlusion_size[1]) * img_width
                occlusion_h = random.uniform(self.occlusion_size[0], self.occlusion_size[1]) * img_height

                occlusion_x = random.uniform(0, img_width - occlusion_w)
                occlusion_y = random.uniform(0, img_height - occlusion_h)

                # 获取原始图像区域的RGB值
                region = img.crop((int(occlusion_x), int(occlusion_y), 
                                   int(occlusion_x + occlusion_w), int(occlusion_y + occlusion_h)))

                # 转换为RGBA并设置A通道为0
                occlusion_area = region.convert("RGBA")
                alpha = occlusion_area.split()[3]
                alpha = Image.new("L", occlusion_area.size, 0)  # 创建A通道，全0表示完全透明
                occlusion_area.putalpha(alpha)

                # 将遮挡区域粘贴到原图上
                img.paste(occlusion_area, (int(occlusion_x), int(occlusion_y)), occlusion_area)

        return img

class RandomBlackRGBA:
    def __init__(self, p=0.5, occlusion_size=(0.3, 0.6), num_spots=random.randint(1, 3)):
        """
        p: 应用遮挡的概率
        occlusion_size: 遮挡区域占图像比例的范围 (最小比例, 最大比例)
        num_spots: 生成的遮挡块数量
        """
        self.p = p
        self.occlusion_size = occlusion_size
        self.num_spots = num_spots

    def __call__(self, img):
        if random.random() < self.p:
            # 获取图像尺寸
            img_width, img_height = img.size

            for _ in range(self.num_spots):
                # 随机选择四个边中的一个进行遮挡
                side = random.choice(['top', 'bottom', 'left', 'right'])

                if side == 'top' or side == 'bottom':
                    occlusion_w = img_width
                    occlusion_h = random.uniform(self.occlusion_size[0], self.occlusion_size[1]) * img_height
                    occlusion_x = 0  # 从最左侧开始
                    occlusion_y = 0 if side == 'top' else img_height - occlusion_h  # 顶部或底部

                elif side == 'left' or side == 'right':
                    occlusion_w = random.uniform(self.occlusion_size[0], self.occlusion_size[1]) * img_width
                    occlusion_h = img_height
                    occlusion_x = 0 if side == 'left' else img_width - occlusion_w  # 左侧或右侧
                    occlusion_y = 0  # 从顶部开始

                # 生成全黑遮挡区域，A通道为128（半透明）
                black_rgba = (0, 0, 0, 0)

                # 创建遮挡区域
                occlusion_area = Image.new("RGBA", (int(occlusion_w), int(occlusion_h)), black_rgba)

                # 将遮挡区域粘贴到原图上
                img.paste(occlusion_area, (int(occlusion_x), int(occlusion_y)), occlusion_area)

        return img

      
class RandomOcclusionRGBA:
    def __init__(self, p=1, occlusion_size=(0.3, 0.5), num_spots=random.randint(1, 3)):
        """
        p: 应用遮挡的概率
        occlusion_size: 遮挡区域占图像比例的范围 (最小比例, 最大比例)
        num_spots: 生成的遮挡块数量
        """
        self.p = p
        self.occlusion_size = occlusion_size
        self.num_spots = num_spots

    def __call__(self, img):
        if random.random() < self.p:
            # 获取图像尺寸
            img_width, img_height = img.size

            for _ in range(self.num_spots):
                # 随机生成遮挡区域的大小和位置
                occlusion_w = random.uniform(self.occlusion_size[0], self.occlusion_size[1]) * img_width
                occlusion_h = random.uniform(self.occlusion_size[0], self.occlusion_size[1]) * img_height

                occlusion_x = random.uniform(0, img_width - occlusion_w)
                occlusion_y = random.uniform(0, img_height - occlusion_h)

                # 随机生成RGB通道的颜色，A通道为黑色（即透明度为0）
                random_rgb = (0, 0, random.randint(0, 255), 0)

                # 创建遮挡区域并设置为随机RGB颜色，A通道为黑色
                occlusion_area = Image.new("RGBA", (int(occlusion_w), int(occlusion_h)), random_rgb)

                # 将遮挡区域粘贴到原图上
                img.paste(occlusion_area, (int(occlusion_x), int(occlusion_y)))

        return img
            
class ToTensorRGBA:
    def __call__(self, img):
        return TF.to_tensor(img)

