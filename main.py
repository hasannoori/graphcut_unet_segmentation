import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from graphcut import *
import matplotlib.colors as mcolors
import tensorflow as tf
from keras.layers import TFSMLayer
from keras import Model
from inference_unet import *
import os




ct_nii = nib.load("../data/3/volume-11.nii")
ct_volume = ct_nii.get_fdata()

seg_nii = nib.load("../data/3/segmentation-11.nii")
gt_seg = seg_nii.get_fdata()

ct_volume = np.clip(ct_volume, 40, 400)
ct_volume = (ct_volume - 40) / (400 - 40)
ct_volume = ct_volume.astype(np.float32)


slice_idx = 410  
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(ct_volume[:, :, slice_idx], cmap="gray")
plt.title("CT Slice")


slice_gt = gt_seg[:, :, slice_idx]

# Define colors for labels 0,1,2
cmap = mcolors.ListedColormap([
    "black",   # 0 → background
    "white",   # 1 → liver
    "red"      # 2 → tumor
])

# Ensure values map exactly to 0,1,2
bounds = [0, 1, 2, 3]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
plt.subplot(1,2,2)

plt.imshow(slice_gt, cmap=cmap, norm=norm)
plt.title("GT: Liver (white), Tumor (red)")
# plt.axis("off")
plt.show()
















tfsmlayer = TFSMLayer("../final_model", call_endpoint="serving_default")
# Wrap into a Keras model
inputs = tf.keras.Input(shape=(128, 128, 16, 1))  
outputs = tfsmlayer(inputs)["conv3d_14"]
model = Model(inputs, outputs)

print("Model loaded successfully.")



patch_shape = (128,128,16)
stride = (64,64,8)  # overlap for smoother output

pred_mask_full = predict_full_volume(model, ct_volume, patch_shape, stride, threshold=0.5)
print("Full-volume prediction done.", pred_mask_full.shape)


# Ensure mask type is integer
pred_mask_full = pred_mask_full.astype(np.uint8)

# Create new NIfTI image using original affine & header
pred_nii = nib.Nifti1Image(
    pred_mask_full,
    affine=ct_nii.affine,
    header=ct_nii.header
)

# Optional but recommended: fix datatype in header
pred_nii.set_data_dtype(np.uint8)

# Save
save_path = "../data/out3/prediction-11.nii"
nib.save(pred_nii, save_path)

print("Saved prediction to:", save_path)
















for i in range(0, pred_mask_full.shape[2], 5):  # show every 5th slice
    plt.figure(figsize=(6,6))
    plt.imshow(ct_volume[:,:,i], cmap='gray')
    plt.imshow(np.ma.masked_where(pred_mask_full[:,:,i]!=1, pred_mask_full[:,:,i]), cmap='Greys', alpha=0.4)
    plt.imshow(np.ma.masked_where(pred_mask_full[:,:,i]!=2, pred_mask_full[:,:,i]), cmap='Reds', alpha=0.5)
    plt.title(f"Slice {i}")
    plt.axis('off')
    plt.show()


for i in range(0, gt_seg.shape[2], 5):
    plt.figure(figsize=(12,6))
    
    # Ground truth
    plt.subplot(1,2,1)
    plt.imshow(gt_seg[:,:,i], cmap='jet')
    plt.title(f"GT — Slice {i}")
    plt.axis('off')
    
    # Prediction
    plt.subplot(1,2,2)
    plt.imshow(pred_mask_full[:,:,i], cmap='jet')
    plt.title(f"Pred — Slice {i}")
    plt.axis('off')
    
    plt.show()





def dice_score(y_true, y_pred, class_id):
    y_true_c = (y_true == class_id).astype(np.uint8)
    y_pred_c = (y_pred == class_id).astype(np.uint8)
    intersection = np.sum(y_true_c * y_pred_c)
    return 2*intersection / (np.sum(y_true_c) + np.sum(y_pred_c) + 1e-6)

dice_liver = dice_score(gt_seg, pred_mask_full, 1)
dice_lesion = dice_score(gt_seg, pred_mask_full, 2)

print(f"Dice Liver: {dice_liver:.4f}, Dice Lesion: {dice_lesion:.4f}")






# # visualize
# pred_patch = pred[0]  # remove batch dimension → (128,128,16,2)

# # Apply threshold
# threshold = 0.5
# pred_liver = (pred_patch[..., 0] > threshold).astype(np.uint8)
# pred_lesion = (pred_patch[..., 1] > threshold).astype(np.uint8)

# # Combine into single mask: 0=bg,1=liver,2=lesion
# pred_mask = pred_liver.copy()
# pred_mask[pred_lesion == 1] = 2


# for i in range(pred_mask.shape[2]):  # iterate over depth
#     plt.figure(figsize=(6,6))
#     plt.imshow(pred_mask[:, :, i], cmap='jet')
#     plt.title(f"Predicted mask — slice {i}")
#     plt.axis('off')
#     plt.show()














# plt.figure(figsize=(10,4))

# plt.subplot(1,2,1)
# plt.imshow(ct_volume_clipped[:, :, slice_idx], cmap="gray")
# plt.title("CT Slice")


# slice_gt = gt_seg[:, :, slice_idx]

# # Define colors for labels 0,1,2
# cmap = mcolors.ListedColormap([
#     "black",   # 0 → background
#     "white",   # 1 → liver
#     "red"      # 2 → tumor
# ])

# # Ensure values map exactly to 0,1,2
# bounds = [0, 1, 2, 3]
# norm = mcolors.BoundaryNorm(bounds, cmap.N)
# plt.subplot(1,2,2)

# plt.imshow(slice_gt, cmap=cmap, norm=norm)
# plt.title("GT: Liver (white), Tumor (red)")
# # plt.axis("off")
# plt.show()





# height, width = slice_img.shape[:2]

# fg_bbox = [220, 40, 280, 100]
# bg_bbox = [0, 200, 500, 500]

# mask_fg = np.zeros((height, width), dtype=np.uint8)
# mask_bg = np.zeros((height, width), dtype=np.uint8)

# mask_fg[int(fg_bbox[1]):int(fg_bbox[3]), int(fg_bbox[0]):int(fg_bbox[2])] = 1
# fg_seed_mask = mask_fg.astype(bool)

# mask_bg[int(bg_bbox[1]):int(bg_bbox[3]), int(bg_bbox[0]):int(bg_bbox[2])] = 1
# bg_seed_mask = mask_bg.astype(bool)


# # graph_cut(slice_img, 60, 1000, (220, 40, 280, 100), (0, 200, 500, 500))

# seg = graph_cut_iterative(
#     slice_img,
#     fg_seed_mask,                 # <-- arbitrary shape mask (bool)
#     bg_seed_mask,                 # <-- arbitrary shape mask (bool)
#     k=30.0,                       # smoothness scale (gamma)
#     iterations=5,
#     n_components=5,
#     fg_prior=0.05,                # tumor small
#     hard_seeds=True,
#     seed_inf=1e9,
#     spatial_weight=0.25,          # how much x,y help (0 disables)
#     distance_bias=0.0,            # try 0.0..2.0 if needed
#     restrict_to_body=True,        # helps reduce air/background confusion
#     body_threshold_hu=-500,       # air is ~ -1000 HU
#     keep_component_touching_fg=True,
#     verbose=True,
# )

# visualize(slice_img, seg, fg_seed_mask, bg_seed_mask, title="GraphCut")

# bigger k → smoother segmentation
# smaller s → more sensitive to intensity differences



