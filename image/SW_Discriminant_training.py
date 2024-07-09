# 测试公开数据集转个人数据集 SW
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import timm
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_file, transform=None):
        self.img_dir = img_dir
        self.mask_file = mask_file
        self.transform = transform
        self.img_labels = self.read_mask_file()

    def read_mask_file(self):
        img_labels = []
        with open(self.mask_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                img_labels.append((parts[0], int(parts[1])))
        return img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"文件 {img_path} 不存在。")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# 数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 创建数据集和数据加载器
img_dir = 'data/img判别'  # 替换为你的图片文件夹路径
mask_file = 'data/MASK.txt'  # 替换为你的掩码文件路径
assert os.path.isdir(img_dir), f"图片文件夹路径 {img_dir} 不存在"
assert os.path.isfile(mask_file), f"掩码文件 {mask_file} 不存在"

# 检查掩码文件中的所有图片路径是否存在
with open(mask_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        img_name = line.strip().split()[0]
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"警告: 文件 {img_path} 在掩码文件中列出，但不存在。")

batch_size = 73
dataset = CustomDataset(img_dir, mask_file, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 设备和模型设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = 'data/公开转个人/model/model'
new_model_dir = 'data/公开转个人/model/最新判别model_dir'
param_dir = 'data/公开转个人/params'  # 保存参数的目录
os.makedirs(new_model_dir, exist_ok=True)
os.makedirs(param_dir, exist_ok=True)  # 确保参数目录存在

model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
scaler = GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=10)

model_loaded = False
if os.path.isdir(model_dir):
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.pth'):
            model_path = os.path.join(model_dir, file_name)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model_loaded = True
            print(f"Loaded model from {model_path}")
if not model_loaded:
    print("No pre-trained model found. Training from scratch.")

# 训练和验证
num_epochs = 40
save_freq = 5
visualize_freq = 5
early_stopping_patience = 10

best_loss = float('inf')
best_model_path = os.path.join(new_model_dir, 'best_model.pth')
patience_counter = 0


def save_parameters(epoch, loss, path):
    with open(path, 'a') as file:
        file.write(f"Epoch: {epoch}, Loss: {loss}\n")


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        running_loss += loss.item()
    average_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    scheduler.step()

    # 验证
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Validation Loss after {epoch + 1} epochs: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path}")

    # 保存模型和参数每10个epoch
    if (epoch + 1) % 20 == 0:
        epoch_model_path = os.path.join(new_model_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved every 10 epochs at {epoch_model_path}")
        save_parameters(epoch + 1, average_loss, os.path.join(param_dir, 'training_params.txt'))