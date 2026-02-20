import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from graphcut import *

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

plt.subplot(1,2,2)
plt.imshow(gt_seg[:, :, slice_idx], cmap="jet")
plt.title("gt Segmentation")
plt.show()



graph_cut(slice_img, 60, 1000, (220, 40, 280, 100), (0, 200, 500, 500))

# bigger k → smoother segmentation
# smaller s → more sensitive to intensity differences



