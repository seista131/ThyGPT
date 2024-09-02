import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import ImageDataset
from src.model import SwinTransformerSegmentation
from src.utils import visualize_predictions
import torchvision.transforms as transforms
from tqdm import tqdm

def train_model(image_dir, mask_dir, model_path, new_model_path, num_epochs=10, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = ImageDataset(image_dir, mask_dir, transform_image=transform_image, transform_mask=transform_mask)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SwinTransformerSegmentation().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print("No pre-trained model found, starting training from scratch.")

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

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                images, masks = next(iter(train_loader))
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images)
                visualize_predictions(images, masks, preds)

    if not os.path.exists(os.path.dirname(new_model_path)):
        os.makedirs(os.path.dirname(new_model_path))
    torch.save(model.state_dict(), new_model_path)
    print(f"Model saved to {new_model_path}")

if __name__ == "__main__":
    image_dir = 'data/img_all'
    mask_dir = 'data/msk_all'
    model_path = 'data/model_SW/swin_transformer_model.pth'
    new_model_path = 'data/model_new_SW/swin_transformer_model.pth'
    train_model(image_dir, mask_dir, model_path, new_model_path)
