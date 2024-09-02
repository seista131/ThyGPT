import torch.nn as nn
import timm

class SwinTransformerSegmentation(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', num_classes=1, pretrained=True):
        super(SwinTransformerSegmentation, self).__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.encoder.forward_features(x)
        x = features.permute(0, 3, 1, 2)
        x = self.decoder(x)
        return x
