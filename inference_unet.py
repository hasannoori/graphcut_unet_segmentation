import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(EnhancedUNet, self).__init__()

        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)
    




# Cell 2: Inference and Visualization Function

def run_inference(img_path, mask_path, slice_idx, model_path):
    """
    Loads the trained model, performs inference on a single CT slice,
    and plots the original image, ground truth, and model prediction.
    """
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    # Initialize the model and load pre-trained weights
    model = EnhancedUNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # Handle both dictionary-based and raw state_dict checkpoints
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Set model to evaluation mode (disables Dropout and fixes BatchNorm)
    model.eval()

    # Load image and ground truth mask data
    img_data = nib.load(img_path).dataobj[:, :, slice_idx]
    mask_data = nib.load(mask_path).dataobj[:, :, slice_idx]

    # Preprocessing: Windowing (-200 to 250 HU), Normalization, and Resizing
    img_clipped = np.clip(img_data, -200, 250)
    img_norm = (img_clipped - (-200)) / 450.0
    img_resized = cv2.resize(img_norm, (256, 256), interpolation=cv2.INTER_AREA)

    # Prepare binary ground truth mask (mask > 0 includes Liver + Tumor)
    true_mask = cv2.resize((mask_data > 0).astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)

    # Convert image to PyTorch tensor format: (Batch, Channel, Height, Width)
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Perform prediction without tracking gradients (saves memory & speeds up)
    with torch.no_grad():
        output_logit = model(img_tensor)
        pred_prob = torch.sigmoid(output_logit).squeeze().cpu().numpy()
        pred_mask = (pred_prob > 0.5).astype(np.uint8)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Original CT Scan
    axes[0].imshow(img_resized, cmap='gray')
    axes[0].set_title(f'Original CT Scan (Slice {slice_idx})')
    axes[0].axis('off')

    # Plot 2: Ground Truth
    axes[1].imshow(img_resized, cmap='gray')
    axes[1].imshow(true_mask, cmap='Greens', alpha=0.5)
    axes[1].set_title('Ground Truth (Green)')
    axes[1].axis('off')

    # Plot 3: Model Prediction
    axes[2].imshow(img_resized, cmap='gray')
    axes[2].imshow(pred_mask, cmap='Reds', alpha=0.5)
    axes[2].set_title('Model Prediction (Red)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    bg_mask = 1 - pred_mask

    return img_resized, pred_mask, bg_mask









