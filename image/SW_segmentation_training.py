#SW 分割模型
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm

# 数据集类
class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_image=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.images = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)  # 假设掩码文件名与图像文件名相同

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask

# 路径设置
image_dir = 'data/img_all'
mask_dir = 'data/msk_all'
model_dir = 'data/model_SW'
model_new_dir = 'data/model_new_SW'
model_path = os.path.join(model_dir, 'swin_transformer_model.pth')
new_model_path = os.path.join(model_new_dir, 'swin_transformer_model.pth')

# 检查路径和文件
assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist."
assert os.path.exists(mask_dir), f"Mask directory {mask_dir} does not exist."

# 图像和掩码的变换
transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 创建数据集和数据加载器
train_dataset = ImageDataset(image_dir, mask_dir, transform_image=transform_image, transform_mask=transform_mask)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

class SwinTransformerSegmentation(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=1, pretrained=True):
        super(SwinTransformerSegmentation, self).__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28 -> 56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112x112 -> 224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)  # 保持224x224尺寸不变
        )

    def forward(self, x):
        features = self.encoder.forward_features(x)
        x = features.permute(0, 3, 1, 2)  # 适配解码器输入
        x = self.decoder(x)
        return x

# 可视化函数
def visualize_predictions(images, masks, preds, num_samples=3):
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    masks = masks.squeeze(1).cpu().numpy()  # 确保掩码正确形状
    preds = torch.sigmoid(preds).cpu().numpy()

    fig, axs = plt.subplots(num_samples, 3, figsize=(10, 10))
    for i in range(num_samples):
        axs[i, 0].imshow(images[i])
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(masks[i], cmap='gray')
        axs[i, 1].set_title('True Mask')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(preds[i, 0] > 0.5, cmap='gray')
        axs[i, 2].set_title('Predicted Mask')
        axs[i, 2].axis('off')

    plt.show()

# 主训练代码
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SwinTransformerSegmentation().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# 检查是否有预训练模型
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
else:
    print("No pre-trained model found, starting training from scratch.")

# 训练循环
num_epochs = 10
visualization_interval = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    running_loss /= len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {running_loss}")

    if (epoch + 1) % visualization_interval == 0:
        model.eval()
        with torch.no_grad():
            images, masks = next(iter(train_loader))
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            visualize_predictions(images, masks, preds)

# 保存模型
if not os.path.exists(model_new_dir):
    os.makedirs(model_new_dir)
torch.save(model.state_dict(), new_model_path)
print(f"Model saved to {new_model_path}")