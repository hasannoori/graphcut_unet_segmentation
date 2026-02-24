# src/inference/enhanced_unet_infer.py

import numpy as np
import nibabel as nib
import cv2
import torch
import matplotlib.pyplot as plt

from src.models.enhanced_unet import EnhancedUNet


def _preprocess_slice(
    img_slice,
    mask_slice=None,
    img_height=256,
    img_width=256,
    clip_min=-200,
    clip_max=250,
):
    img_clipped = np.clip(img_slice, clip_min, clip_max)
    img_norm = (img_clipped - clip_min) / float(clip_max - clip_min)
    img_resized = cv2.resize(img_norm, (img_width, img_height), interpolation=cv2.INTER_AREA)

    true_mask = None
    if mask_slice is not None:
        true_mask = cv2.resize(
            (mask_slice > 0).astype(np.float32),
            (img_width, img_height),
            interpolation=cv2.INTER_NEAREST,
        )

    return img_resized, true_mask


def load_enhanced_unet(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EnhancedUNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, device


def run_inference(
    img_path,
    mask_path,
    slice_idx,
    model_path,
    img_height=256,
    img_width=256,
    clip_min=-200,
    clip_max=250,
    threshold=0.5,
    show=True,
):
    model, device = load_enhanced_unet(model_path, device=None)
    print(f"Running inference on: {device}")

    img_slice = nib.load(img_path).dataobj[:, :, slice_idx]
    mask_slice = None
    if mask_path is not None:
        mask_slice = nib.load(mask_path).dataobj[:, :, slice_idx]

    img_resized, true_mask = _preprocess_slice(
        img_slice,
        mask_slice=mask_slice,
        img_height=img_height,
        img_width=img_width,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output_logit = model(img_tensor)
        pred_prob = torch.sigmoid(output_logit).squeeze().cpu().numpy()
        pred_mask = (pred_prob > threshold).astype(np.uint8)

    if show:
        _plot_results(img_resized, true_mask, pred_mask, slice_idx)

    return {
        "image": img_resized,
        "true_mask": true_mask,
        "pred_prob": pred_prob,
        "pred_mask": pred_mask,
    }


def _plot_results(img_resized, true_mask, pred_mask, slice_idx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_resized, cmap="gray")
    axes[0].set_title(f"Original CT Scan (Slice {slice_idx})")
    axes[0].axis("off")

    axes[1].imshow(img_resized, cmap="gray")
    if true_mask is not None:
        axes[1].imshow(true_mask, cmap="Greens", alpha=0.5)
    axes[1].set_title("Ground Truth (Green)")
    axes[1].axis("off")

    axes[2].imshow(img_resized, cmap="gray")
    axes[2].imshow(pred_mask, cmap="Reds", alpha=0.5)
    axes[2].set_title("Model Prediction (Red)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()