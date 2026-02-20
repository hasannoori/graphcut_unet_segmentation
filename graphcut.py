#pip install PyMaxflow==1.2.15


import numpy as np
import maxflow
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture



def graph_cut(slice_img, k, s, fore, back):
    """
    slice_img : input image
    k    : smoothness weight
    s    : sigma parameter (controls similarity decay)
    fore : foreground sample (x1,y1,x2,y2)
    back : background sample (x1,y1,x2,y2)
    """

    I = np.array(slice_img, dtype=np.float32)
    m, n = I.shape

    x1, y1, x2, y2 = fore
    If = slice_img[y1:y2, x1:x2].astype(np.float32)

    x1, y1, x2, y2 = back
    Ib = slice_img[y1:y2, x1:x2].astype(np.float32)

    # plt.figure(figsize=(5, 5))
    # plt.title("Foreground and Background Sample")
    # plt.imshow(slice_img)
    # plt.gca().add_patch(plt.Rectangle((fore[0], fore[1]), fore[2] - fore[0], fore[3] - fore[1], edgecolor='green', facecolor='none', linewidth=2))
    # plt.gca().add_patch(plt.Rectangle((back[0], back[1]), back[2] - back[0], back[3] - back[1], edgecolor='red', facecolor='none', linewidth=2))
    # plt.show()

    If_pixels = If.reshape(-1, 1)
    Ib_pixels = Ib.reshape(-1, 1)

    n_components = 5  

    gmm_f = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm_b = GaussianMixture(n_components=n_components, covariance_type='full')

    gmm_f.fit(If_pixels)
    gmm_b.fit(Ib_pixels)

    F = np.zeros((m, n), dtype=np.float32)
    B = np.zeros((m, n), dtype=np.float32)

    # print("###### ",I.min(), I.max(), I.dtype)

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

            g.add_tedge(idx, F[i, j], B[i, j])

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

    g.maxflow()
    segmentation = np.zeros((m, n), dtype=np.uint8)

    for i in range(m):
        for j in range(n):
            idx = node_id(i, j)
            segmentation[i, j] = g.get_segment(idx)

    # -------------------------------
    # 8. Build output image
    # -------------------------------
    vis = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
    vis = (vis * 255).astype(np.uint8)

    alpha = 0.5  # transparency

    output = np.stack([vis, vis, vis], axis=-1).astype(np.float32)

    red_mask = np.zeros_like(output)
    red_mask[..., 0] = 255  # red channel

    mask = segmentation == 1
    output[mask] = (1 - alpha) * output[mask] + alpha * red_mask[mask]

    output = output.astype(np.uint8)

    plt.imshow(output)
    plt.axis("off")
    plt.show()