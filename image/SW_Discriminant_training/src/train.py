import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
from src.config import Config
from src.dataset import CustomDataset
from src.model import load_pretrained_model

def train_model():
    config = Config()

    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.old_model_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_dir = 'data/data/img_公开'
    mask_file = 'data/data/output2.txt'
    dataset = CustomDataset(img_dir, mask_file, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(config, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    writer = SummaryWriter(config.log_dir)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.num_epochs}]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            writer.add_scalar('Training loss', loss.item(), epoch)

        average_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {average_loss:.4f}")

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        accuracy, precision, recall, f1_score = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=1)
        writer.add_scalar('Validation loss', val_loss, epoch)
        writer.add_scalar('Accuracy', accuracy, epoch)
        writer.add_scalar('Precision', precision, epoch)
        writer.add_scalar('Recall', recall, epoch)
        if f1_score is not None:
            writer.add_scalar('F1 Score', f1_score, epoch)
        else:
            f1_score = 0  # 设定一个默认值
        print(f"Validation Metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(config.model_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % config.save_freq == 0:
            periodic_save_path = os.path.join(config.model_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), periodic_save_path)
            print(f"Model saved every {config.save_freq} epochs at {periodic_save_path}")

        if patience_counter >= config.early_stopping_patience:
            print("Early stopping triggered.")
            break

        scheduler.step()

    writer.close()

if __name__ == "__main__":
    train_model()
