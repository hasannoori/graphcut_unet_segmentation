# src/training/train_loop.py

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    learning_rate=1e-4,
    num_epochs=50,
    save_dir="./",
    early_stopping_patience=10,
    grad_clip_max_norm=1.0,
):
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_dice = 0.0
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    val_dice_scores = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        model.train()
        epoch_train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()

            epoch_train_loss += loss.item()

            if batch_idx % max(1, len(train_loader) // 4) == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()

                preds = torch.sigmoid(outputs)
                preds_bin = (preds > 0.5).float()

                intersection = (preds_bin * masks).sum()
                union = preds_bin.sum() + masks.sum()
                dice = (2.0 * intersection) / (union + 1e-7)

                val_dice += dice.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_val_dice)

        scheduler.step(avg_val_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Val Dice:   {avg_val_dice:.4f}")
        print(f"LR:         {current_lr:.6f}")

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            epochs_no_improve = 0

            ckpt_path = os.path.join(save_dir, "models", "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_dice_scores": val_dice_scores,
                },
                ckpt_path,
            )
            print(f"New best model saved! Dice: {best_dice:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {epoch + 1} epochs! Best Dice: {best_dice:.4f}"
                )
                break

    return train_losses, val_losses, val_dice_scores