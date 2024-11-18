import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from torchvision.models import ResNet18_Weights,ResNet34_Weights

class MultiScalePositionalEncoding(nn.Module):
    def __init__(self, d_model):
        """
        初始化二维位置编码
        :param d_model: 编码的特征维度（必须是偶数）
        """
        super(MultiScalePositionalEncoding, self).__init__()
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, but got {d_model}")
        
        self.d_model = d_model

    def forward(self, height, width, window_sizes=[16, 32]):
        """
        动态生成二维位置编码，相对于图片中心生成
        :param height: 图像的高度
        :param width: 图像的宽度
        :param window_sizes: 多个窗口尺度
        :return: 对应 (height, width) 的二维位置编码矩阵，大小为 (height, width, d_model)
        """
        # 获取图像中心的坐标
        center_y, center_x = height // 2, width // 2

        # 创建每个像素相对于中心位置的 (y, x) 网格
        y_pos = torch.arange(height, dtype=torch.float32).unsqueeze(1).repeat(1, width) - center_y
        x_pos = torch.arange(width, dtype=torch.float32).unsqueeze(0).repeat(height, 1) - center_x

        # 初始化二维位置编码，分别对 y 和 x 进行编码
        pe_y = torch.zeros(height, width, self.d_model // 3)
        pe_x = torch.zeros(height, width, self.d_model // 3)
        pe_w = torch.zeros(len(window_sizes), self.d_model // 3)

        div_term = torch.exp(torch.arange(0, self.d_model // 3, 2).float() * -(math.log(10000.0) / (self.d_model // 3)))

        # 对 y 方向进行正弦-余弦编码
        pe_y[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)  # 偶数索引使用 sin
        pe_y[:, :, 1::2] = torch.cos(y_pos.unsqueeze(-1) * div_term)  # 奇数索引使用 cos

        # 对 x 方向进行正弦-余弦编码
        pe_x[:, :, 0::2] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe_x[:, :, 1::2] = torch.cos(x_pos.unsqueeze(-1) * div_term)

        # 对 window_size 进行正弦-余弦编码
        for idx, w in enumerate(window_sizes):
            pe_w[idx, 0::2] = torch.sin(w * div_term)  # 偶数索引使用 sin
            pe_w[idx, 1::2] = torch.cos(w * div_term)  # 奇数索引使用 cos

        # 将 y 和 x 的编码结果合并，形成最终的二维位置编码
        pe = torch.cat([pe_y, pe_x], dim=-1)

        # 为每个 window_size 生成独立的编码并加到 pe 中
        pe = pe.unsqueeze(2).repeat(1, 1, len(window_sizes), 1)  # 复制编码为每个 window_size 使用
        pe_w = pe_w.unsqueeze(0).unsqueeze(1).repeat(height, width, 1, 1)  # 扩展 window_size 的编码
        pe = torch.cat([pe, pe_w], dim=-1)

        return pe


class CustomConvNet(nn.Module):
    def __init__(self, d_model, origin_model=models.resnet18(weights=ResNet18_Weights.DEFAULT)):
        # 修改第一层的卷积层，使其接受4个通道的输入
        super(CustomConvNet, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=3, bias=False),  # 修改输入通道为4
            *list(origin_model.children())[0:-1]  # 保留 resnet 的其他层
        )
        self.fc = nn.Sequential(
            nn.Linear(512, d_model)  # 对于回归问题，输出6个连续值
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

    

class SATRot(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_samples =[1, 10], window_sizes = [128, 24]):
        super(SATRot, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_encoder = MultiScalePositionalEncoding(d_model)
        self.num_samples = num_samples
        self.window_sizes = window_sizes
        self.extractor = CustomConvNet(d_model, models.resnet34(weights=ResNet34_Weights.DEFAULT)) 
        self.linear_projection = nn.Linear(window_sizes[-1] * window_sizes[-1] * 3, d_model)

                # 定义可学习的采样坐标, (num_samples, 2) 代表 y 和 x 坐标
        self.learnable_positions = nn.ParameterList([
            nn.Parameter(torch.rand(num_samples[i], 2) * 128)  # 初始化为 (0, 128) 的随机值
            for i in range(len(num_samples))
        ])
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.d_model = d_model

        self.R_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(d_model, 6)  # 回归六个连续值
        )

        self.uv_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(d_model, 2),  # 回归2个连续值
            nn.Sigmoid()
        )

        self.pos_encoding = self.pos_encoder(128, 128, self.window_sizes).detach()
        self.pos_encoding = self.pos_encoding.to(device)         


    def extract_and_sample_features(self, single_image, idx, window_size=16):
        """
        只提取 A 通道非零点的位置和对应的 16x16 窗口卷积特征，并进行均匀采样。
        :param rgb_image: 输入的 RGBA 图像 (batch_size, 4, H, W)
        :param num_samples: 均匀采样的非零点数目
        :param window_size: 16x16 的窗口大小
        :return: 提取的非零点特征和对应的位置信息
        """
        H, W = single_image.shape[1], single_image.shape[2]
        half_window = window_size // 2
        
        learned_positions = self.learnable_positions[idx]
        learned_positions = torch.clamp(learned_positions.round().long(), 0, H - 1)
        

        sampled_windows = []
        sampled_positions = []

        # 对每个采样点建立 16x16 窗口并进行卷积
        for pos in learned_positions:
            y, x = pos[0], pos[1]
            
            # 确保窗口不会超出图像边界
            y1, y2 = max(0, y - half_window), min(H, y + half_window)
            x1, x2 = max(0, x - half_window), min(W, x + half_window)
            
            # 提取 16x16 窗口并进行卷积
            window = single_image[:3, y1:y2, x1:x2]  # (1, 4, window_size, window_size)
            if window.size(1) != window_size or window.size(2) != window_size:
                # 如果窗口大小不足 16x16，则进行填充
                pad = nn.ZeroPad2d((0, window_size - window.size(2), 0, window_size - window.size(1)))
                window = pad(window)
            # print(f'window: {window.shape}')

            sampled_windows.append(window)
            pos_w = torch.cat([pos, torch.tensor([self.window_sizes.index(window_size)], device=pos.device)], dim=0)
            sampled_positions.append(pos_w)

        sampled_windows = torch.stack(sampled_windows, dim=0)  # (num_samples, d_model)
        # print(f'sampled_windows: {sampled_windows.shape}')

        sampled_positions = torch.stack(sampled_positions, dim=0)  # (num_samples, 3)
        # print(f'sampled_features: {sampled_features.shape}, sampled_positions: {sampled_positions.shape}')


        return sampled_positions, sampled_windows

    def forward(self, rgb_image):
        """
        :param rgb_image: 输入的 RGBA 图像 (batch_size, 4, H, W)
        :param num_samples: 均匀采样的非零点数目
        :return: 最终回归的六个连续值
        """

        # 提取 16x16 窗口卷积特征和对应的非零坐标
        pixel_positions = []
        features = []
            

        for idx, (n_s, win) in enumerate(zip(self.num_samples, self.window_sizes)):
            pixel_position = []
            window = []
            for i in range(rgb_image.size(0)):
                
                pixel_position_certainsize, window_certainsize = self.extract_and_sample_features(rgb_image[i], idx, win)
                pixel_position.append(pixel_position_certainsize)
                window.append(window_certainsize)
                # print(f'pixel_position_certainsize:{pixel_position_certainsize.shape}')
                # print(f'window_certainsize:{window_certainsize.shape}')
            pixel_position = torch.stack(pixel_position)
            window = torch.cat(window,dim=0)
            
            # print(f'pixel_position:{pixel_position.shape}')
            # print(f'window:{window.shape}')
            if idx == 0: 
                feature = self.extractor(window)
            else:
                window_flat = window.view(window.size(0), -1)  # 展平窗口
                feature = self.linear_projection(window_flat)  # 直接线性映射到 d_model
     
            # print(f'feature:{feature.shape}')
            feature = feature.view(rgb_image.size(0), n_s, -1)
            # print(f'feature_sorted:{feature.shape}')

            pixel_positions.append(pixel_position)
            features.append(feature)

        pixel_positions = torch.cat(pixel_positions, dim=1)  # Stack along dim 1
        features = torch.cat(features, dim=1)  # Stack along dim 1

        # print(f'Stacked pixel_positions: {pixel_positions.shape}')
        # print(f'Stacked features: {features.shape}')
        # print(f'uv_init:{uv_init.shape}')

        # print(f'device:{device}')
        # print(f'features:{features.device}')
        # print(f'global_feature:{global_feature.shape}')
        # print(f'features:{features.device}')
        # print(f'pixel_positions:{pixel_positions.shape}')

        
        # print(f'pos_encoding:{pos_encoding.shape}')

        # 根据像素位置提取对应的 Positional Encoding
        positional_features = self.pos_encoding[pixel_positions[:, :, 0], pixel_positions[:, :, 1], pixel_positions[:, :, 2]] # (batch_size, num_samples, d_model)

        # print(f'features: {features.shape}')
        # 将卷积提取的特征和位置编码相加
        input_sequence = features + positional_features
        # print(f'input_sequence: {input_sequence.shape}')

        # 通过 Transformer 模型
        transformer_output = self.transformer(input_sequence)

        # 最终通过全连接层得到回归的六个连续值
        # x1 = self.fc_1(transformer_output.mean(dim=1))
        # x2 = self.fc_2(transformer_output.mean(dim=1))
        r = self.R_fc(transformer_output.mean(dim=1))
        r1, r2 = r[:, :3], r[:, 3:]

        r2 = r2 - torch.sum(r1 * r2, dim=1, keepdim=True) * r1
        r3 = torch.cross(r1, r2, dim=1)
        r1 = F.normalize(r1, p=2, dim=1)  # 归一化到单位向量
        r2 = F.normalize(r2, p=2, dim=1)  # 归一化到单位向量
        r3 = F.normalize(r3, p=2, dim=1)  # 归一化到单位向量
        R = torch.cat([r1 , r2, r3], dim=1)

        uv = self.uv_fc(transformer_output.mean(dim=1))


        return uv, R


# Example usage
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设输入 RGBA 图像
    batch_size = 4
    height = 128
    width = 128
    rgb_image = torch.randn(batch_size, 4, height, width).to(device)  # 随机生成的 RGBA 图像
    # 初始化 Transformer 模型
    d_model = 60
    nhead = 4
    num_layers = 3
    model = SATRot(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    

    # 将 RGBA 图像传入 Transformer，进行均匀采样并得到回归结果
    num_samples = 128
    predicted_output = model(rgb_image)

    print(predicted_output)  # 打印回归的六个连续值
