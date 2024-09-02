import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
import numpy as np
from src.config import Config
from src.dataset import ImageFolderWithLabelsDataset
from src.model import load_pretrained_model

def test_model():
    config = Config()

    labels = read_labels(config.label_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = load_pretrained_model(config, device)
    model.eval()

    dataset = ImageFolderWithLabelsDataset(config.img_folder, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels, paths in tqdm(loader, desc="Processing Images"):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])

        for path, pred, label in zip(paths, predicted, labels):
            results.append({'Image Path': path, 'Predicted Class': pred.item(), 'Actual Class': label.item()})

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Accuracy: {accuracy * 100:.2f}%')

    results_df = pd.DataFrame(results)
    os.makedirs(config.output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(config.output_dir, config.results_csv), index=False)

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(recall, precision, lw=2, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    pr_curve_path = os.path.join(config.output_dir, 'precision_recall_curve.png')
    plt.savefig(pr_curve_path)
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(config.output_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.show()

    print(f'ROC AUC: {roc_auc:.2f}')
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))

def read_labels(label_file):
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_name, label = parts
                labels[img_name] = int(label)
            else:
                print(f"Skipping malformed line: {line.strip()}")
    return labels

if __name__ == "__main__":
    test_model()
