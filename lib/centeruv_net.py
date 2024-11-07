import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet18_Weights, ResNet34_Weights
# from net_utils.matrix2euler import rotation_matrix_to_euler_angles


class CenterUV_Net(nn.Module):
    def __init__(self):
        super(CenterUV_Net, self).__init__()
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        # 修改最后的全连接层，输出维度为2 (x, y)
        resnet.fc = nn.Linear(resnet.fc.in_features, 2)
        self.resnet = resnet

    def forward(self, x):
        x = self.resnet(x)
        x =torch.sigmoid(x)
        return x
    

# Example usage
if __name__ == "__main__":
    # Create a random tensor with shape (batch_size, 3, 256, 256)
    image_path = 'dataset/VGA/train/origin/images/rgb_train_0.png'
    image = Image.open(image_path)
    print(image.size)
    # 将图片转换为张量，像素值范围从[0, 255]归一化为[0, 1]
    x = transforms.ToTensor()(image)
    print(x.size())
    resize = transforms.Resize((128, 128))
    x = x.unsqueeze(0)
    x = resize(x)
    print(x.size())
    rotation = torch.randn(1, 6)
    model = CenterUV_Net()
    output = model(x,rotation)
    print(output.shape)  # Expected output shape: (batch_size, 6)