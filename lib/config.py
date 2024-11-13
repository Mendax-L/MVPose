import numpy as np
import torch


Kc_lmo = np.array([
    [572.4114, 0, 325.2611],  # 第一行: [fx, 0, cx]
    [0, 573.57043, 242.04899],  # 第二行: [0, fy, cy]
    [0, 0, 1]                           # 第三行: [0, 0, 1]
])

Kc_lmo_inv = np.linalg.inv(Kc_lmo)

Kc_lmo_tensor = torch.tensor(Kc_lmo, dtype=torch.float32)
Kc_lmo_inv_tensor = torch.tensor(Kc_lmo_inv, dtype=torch.float32)


Rc0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
tc0 = np.array([0, 0, 0])

Rc1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
tc1 = np.array([0, 80, 0])

Rc2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
tc2 = np.array([0, 0, 80])

Rc3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
tc3 = np.array([-80, 0, 0])

# Rc4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# tc4 = np.array([0, -80, 0])

# views = {0:(Rc0,tc0), 1:(Rc1,tc1), 2:(Rc2,tc2), 3:(Rc3,tc3)}
views = {0:(Rc0,tc0)}


test_scene_ids = list(range(48,60))
train_scene_ids = list(set(range(0, 92)) - set(test_scene_ids))
train_scene_ids.sort() 




import torchvision.transforms as transforms
# from lib.customized_transform import RandomHolesRGB,RandomOcclusionRGB

SATRot_train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    # RandomOcclusionRGB(p=0.4),
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.4),
    # RandomHolesRGB(p=0.4),
    # transforms.RandomInvert(p=0.1),              # 随机反色
    # transforms.RandomPosterize(bits=4, p=0.4),   # 随机减少颜色位数
    # transforms.RandomSolarize(threshold=128, p=0.4),  # 随机日蚀效果
    # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.4),  # 随机调整锐度
    # transforms.RandomAutocontrast(p=0.4),        # 随机自动对比度
    # transforms.RandomEqualize(p=0.4),            # 随机直方图均衡化
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],     # 正则化
    #                     std=[0.229, 0.224, 0.225])
])

SATRot_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],     # 正则化
    #                     std=[0.229, 0.224, 0.225])
])

