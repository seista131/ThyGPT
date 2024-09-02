import os
import torch
import numpy as np
from PIL import Image


# Helper function to save images
def save_image(image_array, filename, save_dir='data/grad_cam'):
    """
    Save a NumPy image array as an image file.

    Args:
    - image_array (np.array): The image to save, expected to be in the range [0, 1].
    - filename (str): The name of the file to save.
    - save_dir (str): The directory where the image will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
    image_pil.save(os.path.join(save_dir, filename))


# Helper function to find the first model in a directory
def find_first_model(directory, extension='.pth'):
    """
    Find the first model file in a directory with a given extension.

    Args:
    - directory (str): The directory to search for model files.
    - extension (str): The file extension to look for (default is '.pth').

    Returns:
    - str: The full path to the first model file found, or None if no files are found.
    """
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)
    return None


# Function to read image labels from a file
def read_labels(label_file):
    """
    Read image labels from a text file.

    Args:
    - label_file (str): The path to the label file. Each line should contain an image filename and its label.

    Returns:
    - dict: A dictionary mapping image filenames to labels.
    """
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


# Helper function to calculate accuracy
def calculate_accuracy(preds, labels):
    """
    Calculate the accuracy of predictions.

    Args:
    - preds (np.array): Predicted labels.
    - labels (np.array): Ground truth labels.

    Returns:
    - float: The accuracy as a percentage.
    """
    correct = np.sum(preds == labels)
    total = len(labels)
    return correct / total


# Function to calculate and display performance metrics
def display_classification_metrics(all_labels, all_preds, all_probs):
    """
    Display various classification metrics including accuracy, precision, recall, F1-score, ROC curve, and Precision-Recall curve.

    Args:
    - all_labels (list): Ground truth labels.
    - all_preds (list): Predicted labels.
    - all_probs (list): Probabilities for the positive class.
    """
    from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
    import matplotlib.pyplot as plt

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(recall, precision, lw=2, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print(f'ROC AUC: {roc_auc:.2f}')
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))
