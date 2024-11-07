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

    def forward(self, height, width):
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
        pe_y = torch.zeros(height, width, self.d_model // 2)
        pe_x = torch.zeros(height, width, self.d_model // 2)

        div_term = torch.exp(torch.arange(0, self.d_model // 2, 2).float() * -(math.log(10000.0) / (self.d_model // 2)))

        # 对 y 方向进行正弦-余弦编码
        pe_y[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)  # 偶数索引使用 sin
        pe_y[:, :, 1::2] = torch.cos(y_pos.unsqueeze(-1) * div_term)  # 奇数索引使用 cos

        # 对 x 方向进行正弦-余弦编码
        pe_x[:, :, 0::2] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe_x[:, :, 1::2] = torch.cos(x_pos.unsqueeze(-1) * div_term)


        # 将 y 和 x 的编码结果合并，形成最终的二维位置编码
        pe = torch.cat([pe_y, pe_x], dim=-1)

        return pe

class GlobalConvNet(nn.Module):
    def __init__(self, d_model, origin_model=models.resnet34(weights=ResNet34_Weights.DEFAULT)):
        super(GlobalConvNet, self).__init__()

        # 直接使用原始模型，保存所需的层
        self.forelayer = nn.Sequential(*list(origin_model.children())[:-4]) 
        self.layer3 = origin_model.layer3  # 保存 layer3
        self.layer4 = origin_model.layer4  # 保存 layer4
        self.avgpool = origin_model.avgpool  # 保留平均池化层
        self.fc = nn.Sequential(
            nn.Linear(512, d_model)  # 线性层用于嵌入
        )

    def forward(self, x):
        x = self.forelayer(x)  # 通过 layer1
        feature_map = self.layer3(x)  # 获取 layer3 的特征图
        x = self.layer4(feature_map)  # 通过 layer4

        # 处理以生成嵌入
        x = self.avgpool(x)  # 通过平均池化
        x = torch.flatten(x, 1)  # 展平处理
        embedding = self.fc(x)  # 通过全连接层生成嵌入

        # 打印特征图和嵌入的形状
        # # print(f'feature_map shape: {feature_map.shape}, embedding shape: {embedding.shape}')
        return feature_map, embedding  # 返回 feature_map 和 embedding

class LocalConvNet(nn.Module):
    def __init__(self, d_model, origin_model=models.resnet18(weights=ResNet18_Weights.DEFAULT)):
        super(LocalConvNet, self).__init__()
        
        # 保留 ResNet 的所有层，并访问 layer4 的输出
        self.features = nn.Sequential(*list(origin_model.children())[:-1])  # 去掉最后的全连接层和池化层
        self.fc = nn.Sequential(
            nn.Linear(512, d_model)  # 对于回归问题，输出 d_model 个连续值
        )

    def forward(self, x):
        x = self.features(x)  # 通过前面的层
        # 最后全连接层处理
        x = torch.flatten(x, 1)  # 根据需要对特征图进行展平处理
        embedding = self.fc(x)
        return embedding  # 返回 embedding 和 feature_map


class Keypoints_extractor(nn.Module): 
    def __init__(self, num_samples, feature_map_size = 256, temperature=0.1):
        super(Keypoints_extractor, self).__init__()
        self.num_samples = num_samples
        self.temperature = temperature
        # 学习显著性热力图生成器
        self.saliency_conv = nn.Conv2d(feature_map_size, 1, kernel_size=3, padding=1)
        
    def forward(self, feature_maps):
        """
        生成显著性热力图并根据显著性选择关键点
        :param feature_maps: CNN的特征图，形状为 (batch_size, channels, H, W)
        :return: 关键点坐标 tensor，形状为 (batch_size, num_samples, 2)
        """
        batch_size, _, height, width = feature_maps.shape

        # 生成显著性热力图并进行softmax归一化
        saliency_map = self.saliency_conv(feature_maps)  # (batch_size, 1, H, W)
        saliency_map = saliency_map.view(batch_size, -1)  # 展平为 (batch_size, H * W)
        saliency_probs = F.softmax(saliency_map / self.temperature, dim=-1)  # 转换为概率分布

        # 抽样num_samples个关键点
        indices = torch.multinomial(saliency_probs, self.num_samples, replacement=True)  # (batch_size, num_samples)
        
        # 将平面索引转换为(y, x)坐标
        keypoints_y = indices // width
        keypoints_x = indices % width
        keypoints = torch.stack([keypoints_y, keypoints_x], dim=-1)  # (batch_size, num_samples, 2)
        
        return keypoints
    
class Rot_Net(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_samples =10, window_size = 1):
        super(Rot_Net, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.pos_encoder = MultiScalePositionalEncoding(d_model)
        self.num_samples = num_samples
        self.window_size = window_size
        self.position_embedding = nn.Embedding(8*8 + 1, d_model)
        self.keypoint_extractor = Keypoints_extractor(self.num_samples)
        self.global_extractor = GlobalConvNet(d_model, models.resnet34(weights=ResNet34_Weights.DEFAULT))
        self.local_extractor = nn.ModuleList([
                LocalConvNet(d_model,models.resnet18(weights=ResNet18_Weights.DEFAULT)) 
                for _ in range(num_samples)  # 使用不同的 out_channels
            ])
                # 定义可学习的采样坐标, (num_samples, 2) 代表 y 和 x 坐标

        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.d_model = d_model
        self.local_fc = nn.Sequential(
            nn.Linear(256, 128),  # 第一层，减少维度
            nn.Dropout(p=0.3),
            nn.ReLU(),            # 激活函数
            nn.Linear(128, d_model)  # 最终映射到 d_model
        )
        self.rot_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(d_model, 6)  # 回归六个连续值
        )

        # self.pos_encoding = self.pos_encoder(8, 8).detach()
        # self.pos_encoding = self.pos_encoding.to(device) 
                


    def get_sample_pos_win(self, global_featuremap, num_samples, window_size=1):
        """
        从学习到的关键点采样窗口
        :param global_featuremap: CNN的全局特征图
        :param num_samples: 每个图像的采样关键点数量
        :param window_size: 每个窗口的大小
        :return: 采样窗口和关键点位置
        """
        batch_size, _, height, width = global_featuremap.shape

        # 利用 Keypoints_extractor 生成关键点位置
        centerpixel_position = self.keypoint_extractor.forward(global_featuremap)

        # 在每个关键点周围提取窗口
        windows = []
        half_window = window_size // 2

        for i in range(batch_size):
            batch_windows = []
            for j in range(num_samples):
                y, x = centerpixel_position[i, j]
                
                # 计算窗口边界并处理边缘情况
                top = max(0, y - half_window)
                bottom = min(height, y + half_window)
                left = max(0, x - half_window)
                right = min(width, x + half_window)

                # 从特征图中裁剪窗口
                window = global_featuremap[i, :, top:bottom, left:right]
                
                # 如果窗口大小不足，填充为一致大小
                padded_window = F.pad(window, (0, window_size - window.shape[2], 0, window_size - window.shape[1]))
                
                batch_windows.append(padded_window)
            
            windows.append(torch.stack(batch_windows))  # 将每个关键点的窗口堆叠

        windows = torch.stack(windows)  # 形状: (batch_size, num_samples, channels, window_size, window_size)

        return windows, centerpixel_position

    def forward(self, rgb_image):
        """
        :param rgba_image: 输入的 RGBA 图像 (batch_size, 4, H, W)
        :param num_samples: 均匀采样的非零点数目
        :return: 最终回归的六个连续值
        """
        device = rgb_image.device

        # 1、得到全局的特征图

        global_featuremap, global_embeding = self.global_extractor(rgb_image)
        global_embeding = global_embeding.unsqueeze(1)
        # print(f'global_featuremap: {global_featuremap.shape}, global_embeding: {global_embeding.shape}')
        
        # 2、根据特征图找关键点

        windows, centerpixel_position = self.get_sample_pos_win(global_featuremap, self.num_samples, self.window_size)
        # print(f'windows: {windows.shape},centerpixel_position: {centerpixel_position.shape}')
        # # print(f'centerpixel_position: {centerpixel_position}')
        windows_feature = torch.squeeze(windows, dim=(-1, -2))
        local_embeding = self.local_fc(windows_feature)
        # print(f'local_embeding: {local_embeding.shape}')


        # 4、计算位置编码
        batch_size = global_embeding.shape[0]
        x_coords = centerpixel_position[:, :, 0]  # 取出 x 坐标
        y_coords = centerpixel_position[:, :, 1]  # 取出 y 坐标
        position_indices = y_coords * 8 + x_coords +1
        global_positional_features = self.position_embedding(torch.zeros(batch_size, dtype=torch.long, device=device)).unsqueeze(1)  # 全局为 0
        position_indices = position_indices.clamp(0, self.num_samples - 1)  # 确保索引不越界
        local_positional_features = self.position_embedding(position_indices)  # 通过嵌入层获取局部位置编码

        # print(f'global_positional_features: {global_positional_features.shape}, local_positional_features: {local_positional_features.shape}')


        global_sequence = global_positional_features + global_embeding
        # print(f'global_sequence: {global_sequence.shape}')
        local_sequence = local_positional_features + local_embeding
        # print(f'local_sequence: {local_sequence.shape}')
        input_sequence = torch.cat([global_sequence, local_sequence], dim=1)
        # print(f'input_sequence: {input_sequence.shape}')


        # 通过 Transformer 模型
        transformer_output = self.transformer(input_sequence)


        x = self.rot_fc(transformer_output.mean(dim=1))
        x1, x2 = x[:, :3], x[:, 3:]

        x2 = x2 - torch.sum(x1 * x2, dim=1, keepdim=True) * x1
        x3 = torch.cross(x1, x2, dim=1)
        x1 = F.normalize(x1, p=2, dim=1)  # 归一化到单位向量
        x2 = F.normalize(x2, p=2, dim=1)  # 归一化到单位向量
        x3 = F.normalize(x3, p=2, dim=1)  # 归一化到单位向量
        x = torch.cat([x1 , x2, x3], dim=1)
        return x


# Example usage
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设输入 RGBA 图像
    batch_size = 4
    height = 128
    width = 128
    rgba_image = torch.randn(batch_size, 3, height, width).to(device)  # 随机生成的 RGBA 图像
    # 初始化 Transformer 模型
    d_model = 60
    nhead = 4
    num_layers = 3
    model = Rot_Net(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    

    # 将 RGBA 图像传入 Transformer，进行均匀采样并得到回归结果
    num_samples = 128
    predicted_output = model(rgba_image)

    print(predicted_output)  # 打印回归的六个连续值
