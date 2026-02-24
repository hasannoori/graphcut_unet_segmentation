# src/data/lits_dataset.py

import os
import glob
import numpy as np
import nibabel as nib
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def find_lits_pairs(base_path: str):
    """
    Finds volume-*.nii and corresponding segmentation-*.nii pairs under base_path.
    Returns two lists: volume_paths, mask_paths (paired & aligned).
    """
    volume_paths = sorted(
        glob.glob(os.path.join(base_path, "**", "volume-*.nii"), recursive=True)
    )
    mask_paths_all = sorted(
        glob.glob(os.path.join(base_path, "**", "segmentation-*.nii"), recursive=True)
    )

    if len(volume_paths) == 0 or len(mask_paths_all) == 0:
        raise FileNotFoundError(
            "Could not find .nii files. Check base_path and dataset extraction."
        )

    valid_volumes = []
    valid_masks = []

    for vol_path in volume_paths:
        vol_name = os.path.basename(vol_path)
        vol_num = vol_name.replace("volume-", "").replace(".nii", "")

        mask_path = next(
            (m for m in mask_paths_all if f"segmentation-{vol_num}.nii" in m),
            None,
        )
        if mask_path:
            valid_volumes.append(vol_path)
            valid_masks.append(mask_path)

    return valid_volumes, valid_masks


def split_train_val(volume_paths, mask_paths, val_size=0.2, random_state=42):
    return train_test_split(
        volume_paths, mask_paths, test_size=val_size, random_state=random_state
    )


class LiTSDataset(Dataset):
    """
    Loads LiTS volumes/masks and creates a list of valid slices (slices_info)
    where mask has non-zero pixels. Applies clip/normalize/resize.
    Output:
        image: (1, H, W) float32 in [0,1]
        mask:  (1, H, W) float32 binary {0,1}
    """

    def __init__(
        self,
        vol_paths,
        mask_paths,
        img_height=256,
        img_width=256,
        clip_min=-200,
        clip_max=250,
    ):
        self.vol_paths = vol_paths
        self.mask_paths = mask_paths
        self.img_height = img_height
        self.img_width = img_width
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.slices_info = []
        print(f"Scanning {len(vol_paths)} volumes for valid slices...")

        for v_path, m_path in zip(vol_paths, mask_paths):
            mask_data = nib.load(m_path).get_fdata()
            valid_z = np.where(np.sum(mask_data, axis=(0, 1)) > 0)[0]
            for z in valid_z:
                self.slices_info.append((v_path, m_path, int(z)))

        print(f"Found {len(self.slices_info)} valid slices with liver tissue.")

    def __len__(self):
        return len(self.slices_info)

    def __getitem__(self, idx):
        v_path, m_path, z = self.slices_info[idx]

        img_slice = nib.load(v_path).dataobj[:, :, z]
        mask_slice = nib.load(m_path).dataobj[:, :, z]

        img_clipped = np.clip(img_slice, self.clip_min, self.clip_max)
        img_norm = (img_clipped - self.clip_min) / (self.clip_max - self.clip_min)

        img_resized = cv2.resize(
            img_norm, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA
        )
        mask_resized = cv2.resize(
            mask_slice,
            (self.img_width, self.img_height),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_binary = (mask_resized > 0).astype(np.float32)

        img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_binary, dtype=torch.float32).unsqueeze(0)

        return img_tensor, mask_tensor


def make_loaders(
    base_path: str,
    img_height=256,
    img_width=256,
    batch_size=16,
    num_workers=2,
    val_size=0.2,
    random_state=42,
    clip_min=-200,
    clip_max=250,
):
    vol_paths, mask_paths = find_lits_pairs(base_path)
    train_vol, val_vol, train_mask, val_mask = split_train_val(
        vol_paths, mask_paths, val_size=val_size, random_state=random_state
    )

    train_ds = LiTSDataset(
        train_vol,
        train_mask,
        img_height=img_height,
        img_width=img_width,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    val_ds = LiTSDataset(
        val_vol,
        val_mask,
        img_height=img_height,
        img_width=img_width,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader