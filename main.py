# -*- coding: utf-8 -*-
#pip install PyMaxflow==1.2.15

import numpy as np
import maxflow
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import nibabel as nib



def graph_cut(slice_img, k, s, fore, back):
    """
    slice_img : input image
    k    : smoothness weight
    s    : sigma parameter (controls similarity decay)
    fore : foreground rectangle (x1,y1,x2,y2)
    back : background rectangle (x1,y1,x2,y2)
    """

    # -------------------------------
    # 1. Load image (grayscale)
    # -------------------------------
    # I_pil = Image.open(file).convert('L')
    I = np.array(slice_img, dtype=np.float32)
    m, n = I.shape

    # -------------------------------
    # 2. Extract user foreground/background samples
    # -------------------------------
    x1, y1, x2, y2 = fore
    If = slice_img[y1:y2, x1:x2].astype(np.float32)

    x1, y1, x2, y2 = back
    Ib = slice_img[y1:y2, x1:x2].astype(np.float32)

    # show foreground and background samples on the original image
    # original_color = np.array(Image.open(file))
    # plt.figure(figsize=(5, 5))
    # plt.title("Foreground and Background Sample")
    # plt.imshow(original_color)
    # plt.gca().add_patch(plt.Rectangle((fore[0], fore[1]), fore[2] - fore[0], fore[3] - fore[1], edgecolor='green', facecolor='none', linewidth=2))
    # plt.gca().add_patch(plt.Rectangle((back[0], back[1]), back[2] - back[0], back[3] - back[1], edgecolor='red', facecolor='none', linewidth=2))
    
    # plt.axis("off") 
    # plt.show()

    # Compute histogram means
    # If_mean = np.mean(cv2.calcHist([If], [0], None, [256], [0, 256]))
    # Ib_mean = np.mean(cv2.calcHist([Ib], [0], None, [256], [0, 256]))

    # If_mean = np.mean(If)
    # Ib_mean = np.mean(Ib)

    If_pixels = If.reshape(-1, 1)
    Ib_pixels = Ib.reshape(-1, 1)

    n_components = 5   # try 2–5 for LiTS

    gmm_f = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm_b = GaussianMixture(n_components=n_components, covariance_type='full')

    gmm_f.fit(If_pixels)
    gmm_b.fit(Ib_pixels)

    # -------------------------------
    # 3. Compute unary terms (data cost)
    # -------------------------------
    F = np.zeros((m, n), dtype=np.float32)
    B = np.zeros((m, n), dtype=np.float32)

    eps = 1e-6  # prevent division by zero

    # for i in range(m):
    #     for j in range(n):
    #         diff_f = abs(I[i, j] - If_mean)
    #         diff_b = abs(I[i, j] - Ib_mean)

    #         denom = diff_f + diff_b + eps

    #         F[i, j] = -np.log(diff_f / denom + eps)
    #         B[i, j] = -np.log(diff_b / denom + eps)

    print("###### ",I.min(), I.max(), I.dtype)

    # reshape full image
    I_flat = I.reshape(-1, 1)

    # log-likelihood under each model
    log_prob_f = -gmm_f.score_samples(I_flat)
    log_prob_b = -gmm_b.score_samples(I_flat)

    # negative log likelihood
    F = -log_prob_f.reshape(m, n)
    B = -log_prob_b.reshape(m, n)

    F = F - F.min()
    B = B - B.min()


    F = F*B.max()/F.max()

    F_mask = F/F.max()>0.97
    F = F_mask*F

    print("F dynamic range:", F.min(), "to", F.max())
    print("B dynamic range:", B.min(), "to", B.max())

    # plt.figure(figsize=(5, 5))
    # plt.title("Foreground and Background Sample")
    # plt.imshow(F)
    # plt.figure(figsize=(5, 5))
    # plt.title("Foreground and Background Sample")
    # plt.imshow(B)
    # plt.show()


    # -------------------------------
    # 4. Create graph
    # -------------------------------
    g = maxflow.Graph[float]()
    nodes = g.add_nodes(m * n)

    # Helper to convert 2D → 1D index
    def node_id(i, j):
        return i * n + j

    # -------------------------------
    # 5. Add edges
    # -------------------------------
    for i in range(m):
        for j in range(n):
            idx = node_id(i, j)

            # # Terminal edges (source/sink)
            # ws = F[i, j] / (F[i, j] + B[i, j] + eps)
            # wt = B[i, j] / (F[i, j] + B[i, j] + eps)

            # g.add_tedge(idx, float(ws), float(wt))
            g.add_tedge(idx, F[i, j], B[i, j])

            # 8-neighborhood smoothness
            if j > 0:
                w = k * np.exp(-((I[i, j] - I[i, j - 1]) ** 2) / 2*s*s)
                # print(f"Edge ({i},{j}) ↔ ({i},{j-1}): weight={w:.2f}")
                g.add_edge(idx, node_id(i, j - 1), float(w), (float(k-w)))

            if j < n - 1:
                w = k * np.exp(-((I[i, j] - I[i, j + 1]) ** 2) / 2*s*s)
                g.add_edge(idx, node_id(i, j + 1), float(w), (float(k-w)))
            if i > 0:
                w = k * np.exp(-((I[i, j] - I[i - 1, j]) ** 2) / 2*s*s)
                g.add_edge(idx, node_id(i - 1, j), float(w), (float(k-w)))

            if i < m - 1:
                w = k * np.exp(-((I[i, j] - I[i + 1, j]) ** 2) / 2*s*s)
                g.add_edge(idx, node_id(i + 1, j), float(w), (float(k-w)))
            
            if i > 0 and j > 0:
                w = k * np.exp(-((I[i, j] - I[i - 1, j - 1]) ** 2) / 2*s*s)
                g.add_edge(idx, node_id(i - 1, j - 1), float(w), (float(k-w)))
            
            if i > 0 and j < n - 1:
                w = k * np.exp(-((I[i, j] - I[i - 1, j + 1]) ** 2) / 2*s*s)
                g.add_edge(idx, node_id(i - 1, j + 1), float(w), (float(k-w)))

            if i < m - 1 and j > 0:
                w = k * np.exp(-((I[i, j] - I[i + 1, j - 1]) ** 2) / 2*s*s)
                g.add_edge(idx, node_id(i + 1, j - 1), float(w), (float(k-w)))
            
            if i < m - 1 and j < n - 1:
                w = k * np.exp(-((I[i, j] - I[i + 1, j + 1]) ** 2) / 2*s*s)
                g.add_edge(idx, node_id(i + 1, j + 1), float(w), (float(k-w)))

    # -------------------------------
    # 6. Run maxflow
    # -------------------------------
    g.maxflow()

    # -------------------------------
    # 7. Get segmentation
    # -------------------------------
    segmentation = np.zeros((m, n), dtype=np.uint8)

    for i in range(m):
        for j in range(n):
            idx = node_id(i, j)
            segmentation[i, j] = g.get_segment(idx)

    # -------------------------------
    # 8. Build output image
    # -------------------------------
    #  Normalize for visualization (only for display!)
    vis = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
    vis = (vis * 255).astype(np.uint8)

    #  Convert grayscale → RGB
    output = np.stack([vis, vis, vis], axis=-1)   # shape becomes (m, n, 3)

    print("output shape:", output.shape)

    #  Color background (segmentation == 1) in red
    output[segmentation == 1] = [255, 0, 0]

    plt.imshow(output)
    plt.axis("off")
    plt.show()

# -------------------------------
# Run examples
# -------------------------------

ct_nii = nib.load("../data/2/volume-111.nii")
ct = ct_nii.get_fdata()

ct2 = np.clip(ct, -100, 400)
slice_idx = 300  
slice_img  = ct2[:, :, slice_idx]

graph_cut(slice_img, 60, 1000, (220, 40, 280, 100), (0, 200, 500, 500))

# bigger k → smoother segmentation
# smaller s → more sensitive to intensity differences