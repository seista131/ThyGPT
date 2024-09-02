import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from src.model import ModifiedSwinTransformer

class CustomImageDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.original_image = Image.open(img_path).convert("RGB")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        original_tensor = transforms.ToTensor()(self.original_image)
        transformed_image = self.transform(self.original_image) if self.transform else original_tensor
        return transformed_image, original_tensor

def gradcam_analysis():
    model_directory = 'data/data/model/model'
    img_folder = 'data/data/img测试/m'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = find_first_model(model_directory)
    if model_path is None:
        raise FileNotFoundError("No model file found in the directory.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = ModifiedSwinTransformer(model_path, device).to(device)
    target_layer = model.conv_layer
    gradcam = GradCAM(model=model, target_layers=[target_layer])

    image_index = 521
    img_files = [f for f in os.listdir(img_folder) if f.endswith(('.PNG', '.jpg', '.jpeg'))]
    if image_index < 0 or image_index >= len(img_files):
        raise ValueError("Image index out of range.")

    selected_image_path = os.path.join(img_folder, img_files[image_index])
    print(f"Processing image at index {image_index}: {selected_image_path}")

    dataset = CustomImageDataset(selected_image_path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    save_dir = 'data/grad_cam'
    os.makedirs(save_dir, exist_ok=True)

    for transformed_images, original_tensors in loader:
        transformed_images = transformed_images.to(device)
        targets = [lambda x: x.mean()]  # 假设使用模型输出的均值作为目标
        mask = gradcam(input_tensor=transformed_images, targets=targets)

        img = transformed_images.cpu().squeeze(0)
        img = img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        raw_heatmap_image = show_cam_on_image(img, mask[0], use_rgb=True)

        smooth_mask = cv2.GaussianBlur(mask[0], (5, 5), 0)
        low_threshold = 50
        high_threshold = 150
        canny_edges = cv2.Canny((img * 255).astype(np.uint8), low_threshold, high_threshold)
        canny_edges = canny_edges / 255
        gaussian_filtered_edges = cv2.GaussianBlur(canny_edges, (9, 9), 2)
        enhanced_heatmap = smooth_mask + (smooth_mask * gaussian_filtered_edges * 0.5)
        enhanced_heatmap = np.clip(enhanced_heatmap, 0, 1)

        fig, axs = plt.subplots(1, 6, figsize=(24, 4))
        axs[0].imshow(original_tensors[0].permute(1, 2, 0))
        axs[0].set_title("Original Image")
        axs[1].imshow(img)
        axs[1].set_title("Preprocessed Image")
        axs[2].imshow(raw_heatmap_image)
        axs[2].set_title("Raw Grad-CAM Heatmap")
        axs[3].imshow(show_cam_on_image(img, smooth_mask, use_rgb=True))
        axs[3].set_title("Smooth Heatmap")
        axs[4].imshow(gaussian_filtered_edges, cmap='gray')
        axs[4].set_title("Edges")
        axs[5].imshow(show_cam_on_image(img, enhanced_heatmap, use_rgb=True))
        axs[5].set_title("Enhanced Heatmap")

        plt.show()

        save_image(raw_heatmap_image, 'raw_heatmap.png')
        save_image(show_cam_on_image(img, smooth_mask, use_rgb=True), 'smooth_heatmap.png')
        save_image(show_cam_on_image(img, enhanced_heatmap, use_rgb=True), 'enhanced_heatmap.png')

def save_image(image_array, filename):
    save_dir = 'data/grad_cam'
    image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
    image_pil.save(os.path.join(save_dir, filename))

def find_first_model(directory, extension='.pth'):
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)
    return None

if __name__ == "__main__":
    gradcam_analysis()
