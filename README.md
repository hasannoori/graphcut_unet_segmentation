# GraphCut + U-Net Segmentation

This repository contains a segmentation project that combines:

- **U-Net (deep learning)**
- **GraphCut (classical / interactive segmentation)**

It includes notebooks, source modules, and a pretrained model checkpoint.

---

##  Important (Regenerating Outputs)

**If you want to regenerate outputs/results, use the notebook(s) in `Online Notebooks/` and the parameter files in that same folder.**

 This is the preferred way to reproduce outputs correctly.  
 Do not rely on random local changes if you want consistent results.

---

## Project Structure

- `Online Notebooks/` → online notebooks + parameters for regeneration (**use this for output reproduction**)
- `src/models/` → U-Net model code
- `src/training/` → training loop
- `src/data/` → dataset loader / preprocessing
- `src/graphcut/` → GraphCut functions
- `train_unet_model.ipynb` → U-Net training notebook
- `main.ipynb` → main experiment notebook
- `enhanced_unet_model.pth` → pretrained model checkpoint

---

## Keep GraphCut Functions (Do Not Delete)

Please **do not delete the GraphCut functions**.

Instead, keep them and **separate/label them clearly** so it is easy to see what is:

- **U-Net code**
- **GraphCut code**
- **Notebook experiment code**

This keeps the project organized and preserves your original work.

---

## Recommended Usage

### To regenerate outputs (recommended)
1. Open the notebook(s) in **`Online Notebooks/`**
2. Use the **parameter files in that same folder**
3. Run cells in order
* The final file is **`inference_Unet_plus_Graphcut.ipynb`** that compares UNet and UNet + Graphcut approaches.
* For regenerating UNet only results you can use. **`inference_enhanced_unet_model.ipynb`** for enhanced UNet and **`train_inference_base_unet_model.ipynb`** (which includes inference code).
* The code used for brain tumor segmentation on private dataset is available in **`braintumor_segmentation_private_dataset.ipynb`** but unfortunately we can not put the link to dataset in the nootebook.
### To train U-Net locally
Use `train_unet_model.ipynb` and the modules inside `src/`.

---

## Notess

This repo is meant to preserve **both**:
- the U-Net pipeline
- the GraphCut pipeline

Please keep both parts documented and clearly separated.