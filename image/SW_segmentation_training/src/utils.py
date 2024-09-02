import torch
import matplotlib.pyplot as plt

def visualize_predictions(images, masks, preds, num_samples=3):
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    masks = masks.squeeze(1).cpu().numpy()
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