import numpy as np
from tqdm import tqdm

def predict_full_volume(model, volume, patch_shape=(128,128,16), stride=(64,64,8), threshold=0.5):
    """
    Full-volume inference using sliding window for patch-based 3D UNet.
    
    Args:
        model: Keras model with TFSMLayer
        volume: 3D numpy array (D,H,W) preprocessed
        patch_shape: tuple, patch size used during training
        stride: tuple, sliding window stride in (D,H,W)
        threshold: float, probability threshold for binarization
    
    Returns:
        pred_mask: 3D numpy array of shape volume.shape
                   0=background,1=liver,2=lesion
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_shape
    sd, sh, sw = stride
    
    pred_liver = np.zeros(volume.shape, dtype=np.float32)
    pred_lesion = np.zeros(volume.shape, dtype=np.float32)
    count_map = np.zeros(volume.shape, dtype=np.float32)
    
    # Sliding window
    z_range = tqdm(range(0, W, sw))
    for z in z_range:
        for y in range(0, H, sh):
            for x in range(0, D, sd):

                # print(f"Processing patch at (x={x}, y={y}, z={z})")
                
                # Compute patch bounds
                x_start = x
                y_start = y
                z_start = z
                x_end = min(x_start + pd, D)
                y_end = min(y_start + ph, H)
                z_end = min(z_start + pw, W)
                
                # Extract patch
                patch = volume[x_start:x_end, y_start:y_end, z_start:z_end]
                
                # Pad if patch is smaller than model input
                pad_width = (
                    (0, pd - patch.shape[0]),
                    (0, ph - patch.shape[1]),
                    (0, pw - patch.shape[2])
                )
                patch = np.pad(patch, pad_width, mode='constant', constant_values=0)
                
                patch_input = patch[None, ..., None].astype(np.float32)
                
                # Predict
                pred_patch = model.predict(patch_input, verbose=0)[0]  # (128,128,16,2)
                
                # Remove padding if needed
                pred_patch = pred_patch[:x_end-x_start, :y_end-y_start, :z_end-z_start, :]
                
                # Accumulate predictions
                pred_liver[x_start:x_end, y_start:y_end, z_start:z_end] += pred_patch[...,0]
                pred_lesion[x_start:x_end, y_start:y_end, z_start:z_end] += pred_patch[...,1]
                count_map[x_start:x_end, y_start:y_end, z_start:z_end] += 1
                
    # Average overlapping predictions
    pred_liver /= count_map
    pred_lesion /= count_map
    
    # Apply threshold
    liver_mask = (pred_liver > threshold).astype(np.uint8)
    lesion_mask = (pred_lesion > threshold).astype(np.uint8)
    
    # Combine
    pred_mask = liver_mask.copy()
    pred_mask[lesion_mask==1] = 2
    
    return pred_mask



