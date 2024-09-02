import os
from torch.utils.data import Dataset
from PIL import Image

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
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageFolderWithLabelsDataset(Dataset):
    def __init__(self, img_folder, labels, transform=None):
        self.img_folder = img_folder
        self.labels = labels
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_folder) if f.endswith(('.PNG', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_name)
        label = self.labels.get(img_name, -1)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path
