import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OneDimPositionalEncoding(nn.Module):
    def __init__(self, d_model, width):
        super(OneDimPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.width = width

        # 创建水平位置编码
        x_pos = torch.arange(width, dtype=torch.float32).unsqueeze(1)
        pe_x = torch.zeros(width, d_model)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)

        # 将位置编码设置为不可训练参数
        self.positional_encoding = nn.Parameter(pe_x.unsqueeze(0), requires_grad=False)

    def forward(self, n_patches):
        return self.positional_encoding[:, :n_patches, :]

class SATRot(nn.Module):
    def __init__(self, img_height, img_width, patch_width, d_model, nhead, num_layers):
        super(SATRot, self).__init__()
        self.patch_width = patch_width
        self.d_model = d_model
        self.n_patches = img_width // patch_width

        # 线性层用于条形块嵌入
        self.patch_embed = nn.Linear(patch_width * img_height * 3, d_model)
        
        # 一维位置编码
        self.pos_encoding = OneDimPositionalEncoding(d_model, self.n_patches).forward(self.n_patches).to(device)

        # 分类token用于全局特征汇总
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 旋转和位置预测的输出全连接层
        self.rot_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 6)  # 输出六个旋转参数
        )
        self.uv_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),  # 输出uv位置
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 确保图像宽度可以被条形块宽度整除
        assert W % self.patch_width == 0, "图像宽度必须可以被条形块宽度整除"

        # 将图像划分成垂直的条形块
        x = x.unfold(3, self.patch_width, self.patch_width).permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.n_patches, -1)  # 展平每个条形块
        
        # 条形块通过线性层映射到d_model维度
        x = self.patch_embed(x)

        # 添加分类token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加一维位置编码
        x[:, 1:] += self.pos_encoding

        # Transformer 编码
        x = self.transformer_encoder(x)

        # 获取分类token作为全局特征
        cls_feature = x[:, 0]

        # 通过全连接层进行旋转和位置预测
        R = self.rot_fc(cls_feature)
        uv = self.uv_fc(cls_feature)

        # 正交化旋转矩阵
        r1, r2 = R[:, :3], R[:, 3:]
        r2 = r2 - torch.sum(r1 * r2, dim=1, keepdim=True) * r1
        r3 = torch.cross(r1, r2, dim=1)
        r1 = F.normalize(r1, p=2, dim=1)
        r2 = F.normalize(r2, p=2, dim=1)
        r3 = F.normalize(r3, p=2, dim=1)
        R = torch.cat([r1, r2, r3], dim=1)

        return uv, R

# 示例用法
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 假设输入 RGB 图像
    batch_size = 4
    img_height = 128
    img_width = 128
    patch_width = 16
    d_model = 256
    nhead = 8
    num_layers = 6

    rgb_image = torch.randn(batch_size, 3, img_height, img_width).to(device)  # 随机生成的 RGB 图像
    model = SATRot(img_height, img_width, patch_width, d_model, nhead, num_layers).to(device)

    # 预测输出
    uv, R = model(rgb_image)
    print("Predicted UV:", uv)
    print("Predicted R:", R)
