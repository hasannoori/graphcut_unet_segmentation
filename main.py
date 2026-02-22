import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from graphcut import *
import matplotlib.colors as mcolors

ct_nii = nib.load("../data/2/volume-111.nii")
ct_volume = ct_nii.get_fdata()

seg_nii = nib.load("../data/2/segmentation-111.nii")
gt_seg = seg_nii.get_fdata()

ct_volume_clipped = np.clip(ct_volume, 40, 400)
slice_idx = 300  # for example
slice_img  = ct_volume_clipped[:, :, slice_idx]


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(ct_volume_clipped[:, :, slice_idx], cmap="gray")
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





height, width = slice_img.shape[:2]

fg_bbox = [220, 40, 280, 100]
bg_bbox = [0, 200, 500, 500]

mask_fg = np.zeros((height, width), dtype=np.uint8)
mask_bg = np.zeros((height, width), dtype=np.uint8)

mask_fg[int(fg_bbox[1]):int(fg_bbox[3]), int(fg_bbox[0]):int(fg_bbox[2])] = 1
fg_seed_mask = mask_fg.astype(bool)

mask_bg[int(bg_bbox[1]):int(bg_bbox[3]), int(bg_bbox[0]):int(bg_bbox[2])] = 1
bg_seed_mask = mask_bg.astype(bool)


# graph_cut(slice_img, 60, 1000, (220, 40, 280, 100), (0, 200, 500, 500))

seg = graph_cut_iterative(
    slice_img,
    fg_seed_mask,                 # <-- arbitrary shape mask (bool)
    bg_seed_mask,                 # <-- arbitrary shape mask (bool)
    k=30.0,                       # smoothness scale (gamma)
    iterations=5,
    n_components=5,
    fg_prior=0.05,                # tumor small
    hard_seeds=True,
    seed_inf=1e9,
    spatial_weight=0.25,          # how much x,y help (0 disables)
    distance_bias=0.0,            # try 0.0..2.0 if needed
    restrict_to_body=True,        # helps reduce air/background confusion
    body_threshold_hu=-500,       # air is ~ -1000 HU
    keep_component_touching_fg=True,
    verbose=True,
)

visualize(slice_img, seg, fg_seed_mask, bg_seed_mask, title="GraphCut")

# bigger k → smoother segmentation
# smaller s → more sensitive to intensity differences



