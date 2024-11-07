import torch
import torch.nn as nn
import math

class DynamicPositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        """
        初始化二维位置编码
        :param d_model: 编码的特征维度（必须是偶数）
        """
        super(DynamicPositionalEncoding2D, self).__init__()
        
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, but got {d_model}")
        
        self.d_model = d_model

    def forward(self, height, width):
        """
        动态生成二维位置编码
        :param height: 图像的高度
        :param width: 图像的宽度
        :return: 对应 (height, width) 的二维位置编码矩阵，大小为 (height, width, d_model)
        """
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

class DepthTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_samples=512):
        """
        Transformer 模型用于深度预测
        :param d_model: 特征维度
        :param nhead: 多头注意力的头数
        :param num_layers: Transformer 层数
        """
        super(DepthTransformer, self).__init__()
        self.pos_encoder = DynamicPositionalEncoding2D(d_model)  # 动态位置编码器
        self.depth_embed = nn.Linear(1, d_model)  # 将深度值映射到 d_model 维度
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.num_samples = num_samples
        self.MLP =  nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(p=0.5),
            nn.ReLU(), 
            nn.Linear(d_model, 1),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )  # 最后的全连接层，用于回归深度值        
        self.K_MLP =  nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(p=0.5),
            nn.ReLU(), 
            nn.Linear(d_model, 1),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )  # 最后的全连接层，用于回归深度值

    
    def extract_and_sample_non_zero_points(self, single_depth_map):
        """
        提取所有非零像素并进行均匀采样。如果非零点不足，则用0补齐。
        :param single_depth_map: 输入的深度图 (H, W)
        :param num_samples: 采样的非零点数目
        :return: 均匀采样后的非零像素位置和深度值
        """
        # 获取非零点的掩码
        non_zero_mask = single_depth_map > 0
        # 获取非零点的像素位置（y, x 坐标）
        pixel_positions = torch.nonzero(non_zero_mask, as_tuple=False)
        # print(f'pixel_positions: {pixel_positions}')
        # 获取对应的深度值
        depth_values = single_depth_map[non_zero_mask].unsqueeze(1)  # (N, 1)
        # print(f'pixel_positions: {pixel_positions.shape}')
        # print(f'depth_values: {depth_values.shape}')
        num_non_zero = pixel_positions.size(0)

        if num_non_zero > self.num_samples:
            # 如果非零点数量大于采样数量，则进行均匀采样
            indices = torch.linspace(0, num_non_zero - 1, steps=self.num_samples).long()
            pixel_positions = pixel_positions[indices]
            depth_values = depth_values[indices]
            # print(f'pixel_positions: {pixel_positions.shape}')
            # print(f'depth_values: {depth_values.shape}')
        else:
            # 如果非零点数量少于采样数量，则进行填充
            padding_size = self.num_samples - num_non_zero

            pad_positions = torch.zeros(padding_size, 2, dtype=pixel_positions.dtype, device=pixel_positions.device)
            pad_depth_values = torch.zeros(padding_size, 1, dtype=depth_values.dtype, device=depth_values.device)
            # print(f'pad_positions: {pad_positions.shape}')
            # print(f'pad_depth_values: {pad_depth_values.shape}')
            pixel_positions = torch.cat([pixel_positions, pad_positions], dim=0)
            depth_values = torch.cat([depth_values, pad_depth_values], dim=0)

        return pixel_positions, depth_values

    def forward(self, depth_map):
        """
        :param pixel_positions: 非零像素的位置 (batchsize, N, 2)
        :param depth_values: 非零像素的深度值 (batchsize, N, 1)
        :param height: 输入图像的高度
        :param width: 输入图像的宽度
        :return: 最终预测的深度值
        """
        
        pixel_positions = []
        depth_values = []
        for i in range(depth_map.size(0)):

            pixel_position, depth_value = self.extract_and_sample_non_zero_points(depth_map[i])
            pixel_positions.append(pixel_position)
            depth_values.append(depth_value)
        pixel_positions = torch.stack(pixel_positions)
        depth_values = torch.stack(depth_values)

        min_depth = depth_values.min()  
        max_depth = depth_values.max()  
        depth_values = (depth_values - min_depth) / (max_depth - min_depth)

        # 动态生成对应的二维位置编码
        pos_encoding = self.pos_encoder(depth_map.shape[1], depth_map.shape[2]).to(pixel_positions.device)  # (height, width, d_model)

        # 根据像素位置提取对应的 Positional Encoding
        positional_features = pos_encoding[pixel_positions[:, :, 0], pixel_positions[:, :, 1]]  # (batchsize, N, d_model)
        # print(f'pixel_positions: {pixel_positions.shape}')
        # print(f'positional_features: {positional_features.shape}')
        # 将深度值映射到 d_model 维度
        depth_embedded = self.depth_embed(depth_values)  # (batchsize, N, d_model)
        # print(f'depth_embedded: {depth_embedded.shape}')
        # 将 Positional Encoding 和映射后的深度值相加
        input_sequence = positional_features + depth_embedded  # (batchsize, N, d_model)
        # print(f'input_sequence: {input_sequence.shape}')
        # 通过 Transformer 模型
        transformer_output = self.transformer(input_sequence)
        # print(f'transformer_output: {transformer_output.shape}')
        # 最终通过全连接层得到预测的深度值
        depth_pred = self.MLP(transformer_output.mean(dim=1))
        depth_pred = depth_pred * (max_depth - min_depth) + min_depth 

        return depth_pred




if __name__ == '__main__':
    # 假设输入深度图
    batch_size = 4
    height = 128
    width = 128
    depth_map = torch.randn(batch_size, height, width) * 100 + 300 

    # 初始化 Transformer 模型
    d_model = 64
    nhead = 4
    num_layers = 3
    model = DepthTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers)

    # 将非零像素及其坐标、深度值传入 Transformer，得到预测的深度值
    predicted_depth = model(depth_map)
    # print(f"Predicted depth for image {i}: {predicted_depth.item()}")

    print(predicted_depth)