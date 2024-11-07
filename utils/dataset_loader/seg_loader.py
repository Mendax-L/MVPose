from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为张量
    # 可以在这里添加其他转换
])

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, mode='train', transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.mode = mode
        self.annotations = []
        self.file_path = f'vga_made/annotations/{self.mode}/rotation_labels.txt'
        self.load_annotations()
        

    def __len__(self):
        return len(self.annotations)
    
    def load_annotations(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                numbers = [float(num) for num in line.strip().split()]
                self.annotations.append(numbers)
        return self.annotations
    
    def getAnnos(self,imgId):
        return self.annotations[imgId]
    
    @staticmethod
    def collate_fn(batch):
        # 解包批次中的图像和目标
        images, targets = zip(*batch)
        # 堆叠图像数据成为一个四维张量 [batch_size, channels, height, width]
        images = torch.stack(images, dim=0)
        # 根据需要处理目标数据，这里简单地转换为列表或其他适合您模型的格式
        targets = torch.tensor(targets)
        return images, targets



    def __getitem__(self, idx):
        image = Image.open(f"vga_made/images/{self.mode}/image_{self.mode}_{idx}.png").convert("RGB")
        
        target = self.getAnnos(idx)
        if self.transform:
            image = self.transform(image)
        return image, target

    def getTarget_annos(self, imgId):
        target_annos = []

        # 假设文件名是 'annotations.txt'，并且每一行的数据格式如下：
        # imgId 对应的行内容：0.8268458843231201 -0.31788110733032227 -0.4639799892902374 0.48673945665359497 0.817773699760437 0.3071333169937134
        
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            # 假设 imgId 从 0 开始，与行号对应
            target_line = lines[imgId].strip()  # 去除行尾的换行符

            # 将这一行的字符串分割为单独的数字
            target_annos = [float(num) for num in target_line.split()]

        return target_annos