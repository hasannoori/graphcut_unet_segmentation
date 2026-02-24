# scripts/train_enhanced_unet.py

import torch

from src.data.lits_dataset import make_loaders
from src.models.enhanced_unet import EnhancedUNet, DiceLoss
from src.training.train_loop import train_model


def main():
    BASE_PATH = "./LiTS17"

    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    BATCH_SIZE = 16
    NUM_WORKERS = 2

    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    SAVE_DIR = "./"
    EARLY_STOPPING_PATIENCE = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders(
        base_path=BASE_PATH,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_size=0.2,
        random_state=42,
        clip_min=-200,
        clip_max=250,
    )

    model = EnhancedUNet(in_channels=1, out_channels=1).to(device)
    criterion = DiceLoss()

    print("Starting training...")
    train_losses, val_losses, val_dice_scores = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        save_dir=SAVE_DIR,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        grad_clip_max_norm=1.0,
    )

    print("Training finished.")
    print(f"Best checkpoint saved to: {SAVE_DIR}/models/best_model.pth")


if __name__ == "__main__":
    main()