import torch
import torch.nn as nn
import timm

class ModifiedSwinTransformer(nn.Module):
    def __init__(self, model_path, device):
        super(ModifiedSwinTransformer, self).__init__()
        self.base_model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=2)
        self.base_model.load_state_dict(torch.load(model_path, map_location=device))
        self.base_model.eval()
        self.conv_layer = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layer(x)
        x = self.activation(x)
        return x

def load_pretrained_model(config, device):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=2)
    model_files = [f for f in os.listdir(config["old_model_dir"]) if f.endswith('.pth')]
    if model_files:
        first_model_file = model_files[0]
        model_path = os.path.join(config["old_model_dir"], first_model_file)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("No .pth model file found in the old model directory. Training from scratch.")
    return model.to(device)
