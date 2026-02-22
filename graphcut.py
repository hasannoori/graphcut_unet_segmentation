#pip install PyMaxflow==1.2.15


# import numpy as np
# import maxflow
# import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture



# def graph_cut(slice_img, k, s, fore, back):
#     """
#     slice_img : input image
#     k    : smoothness weight
#     s    : sigma parameter (controls similarity decay)
#     fore : foreground sample (x1,y1,x2,y2)
#     back : background sample (x1,y1,x2,y2)
#     """

#     I = np.array(slice_img, dtype=np.float32)
#     m, n = I.shape

#     x1, y1, x2, y2 = fore
#     If = slice_img[y1:y2, x1:x2].astype(np.float32)

#     x1, y1, x2, y2 = back
#     Ib = slice_img[y1:y2, x1:x2].astype(np.float32)

#     # plt.figure(figsize=(5, 5))
#     # plt.title("Foreground and Background Sample")
#     # plt.imshow(slice_img)
#     # plt.gca().add_patch(plt.Rectangle((fore[0], fore[1]), fore[2] - fore[0], fore[3] - fore[1], edgecolor='green', facecolor='none', linewidth=2))
#     # plt.gca().add_patch(plt.Rectangle((back[0], back[1]), back[2] - back[0], back[3] - back[1], edgecolor='red', facecolor='none', linewidth=2))
#     # plt.show()

#     If_pixels = If.reshape(-1, 1)
#     Ib_pixels = Ib.reshape(-1, 1)

#     n_components = 5  

#     gmm_f = GaussianMixture(n_components=n_components, covariance_type='full')
#     gmm_b = GaussianMixture(n_components=n_components, covariance_type='full')

#     gmm_f.fit(If_pixels)
#     gmm_b.fit(Ib_pixels)

#     F = np.zeros((m, n), dtype=np.float32)
#     B = np.zeros((m, n), dtype=np.float32)

#     # print("###### ",I.min(), I.max(), I.dtype)

#     I_flat = I.reshape(-1, 1)

#     # log-likelihood under each model
#     log_prob_f = -gmm_f.score_samples(I_flat)
#     log_prob_b = -gmm_b.score_samples(I_flat)

#     # negative log likelihood
#     F = -log_prob_f.reshape(m, n)
#     B = -log_prob_b.reshape(m, n)

#     F = F - F.min()
#     B = B - B.min()

#     F = F*B.max()/F.max()

#     F_mask = F/F.max()>0.97
#     F = F_mask*F

#     print("F dynamic range:", F.min(), "to", F.max())
#     print("B dynamic range:", B.min(), "to", B.max())

#     # plt.figure(figsize=(5, 5))
#     # plt.title("Foreground and Background Sample")
#     # plt.imshow(F)
#     # plt.figure(figsize=(5, 5))
#     # plt.title("Foreground and Background Sample")
#     # plt.imshow(B)
#     # plt.show()


#     # -------------------------------
#     # 4. Create graph
#     # -------------------------------
#     g = maxflow.Graph[float]()
#     nodes = g.add_nodes(m * n)

#     # Helper to convert 2D → 1D index
#     def node_id(i, j):
#         return i * n + j

#     # -------------------------------
#     # 5. Add edges
#     # -------------------------------
#     for i in range(m):
#         for j in range(n):
#             idx = node_id(i, j)

#             g.add_tedge(idx, F[i, j], B[i, j])

#             if j > 0:
#                 w = k * np.exp(-((I[i, j] - I[i, j - 1]) ** 2) / 2*s*s)
#                 # print(f"Edge ({i},{j}) ↔ ({i},{j-1}): weight={w:.2f}")
#                 g.add_edge(idx, node_id(i, j - 1), float(w), (float(k-w)))

#             if j < n - 1:
#                 w = k * np.exp(-((I[i, j] - I[i, j + 1]) ** 2) / 2*s*s)
#                 g.add_edge(idx, node_id(i, j + 1), float(w), (float(k-w)))
#             if i > 0:
#                 w = k * np.exp(-((I[i, j] - I[i - 1, j]) ** 2) / 2*s*s)
#                 g.add_edge(idx, node_id(i - 1, j), float(w), (float(k-w)))

#             if i < m - 1:
#                 w = k * np.exp(-((I[i, j] - I[i + 1, j]) ** 2) / 2*s*s)
#                 g.add_edge(idx, node_id(i + 1, j), float(w), (float(k-w)))
            
#             if i > 0 and j > 0:
#                 w = k * np.exp(-((I[i, j] - I[i - 1, j - 1]) ** 2) / 2*s*s)
#                 g.add_edge(idx, node_id(i - 1, j - 1), float(w), (float(k-w)))
            
#             if i > 0 and j < n - 1:
#                 w = k * np.exp(-((I[i, j] - I[i - 1, j + 1]) ** 2) / 2*s*s)
#                 g.add_edge(idx, node_id(i - 1, j + 1), float(w), (float(k-w)))

#             if i < m - 1 and j > 0:
#                 w = k * np.exp(-((I[i, j] - I[i + 1, j - 1]) ** 2) / 2*s*s)
#                 g.add_edge(idx, node_id(i + 1, j - 1), float(w), (float(k-w)))
            
#             if i < m - 1 and j < n - 1:
#                 w = k * np.exp(-((I[i, j] - I[i + 1, j + 1]) ** 2) / 2*s*s)
#                 g.add_edge(idx, node_id(i + 1, j + 1), float(w), (float(k-w)))

#     g.maxflow()
#     segmentation = np.zeros((m, n), dtype=np.uint8)

#     for i in range(m):
#         for j in range(n):
#             idx = node_id(i, j)
#             segmentation[i, j] = g.get_segment(idx)

#     # -------------------------------
#     # 8. Build output image
#     # -------------------------------
#     vis = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
#     vis = (vis * 255).astype(np.uint8)

#     alpha = 0.5  # transparency

#     output = np.stack([vis, vis, vis], axis=-1).astype(np.float32)

#     red_mask = np.zeros_like(output)
#     red_mask[..., 0] = 255  # red channel

#     mask = segmentation == 1
#     output[mask] = (1 - alpha) * output[mask] + alpha * red_mask[mask]

#     output = output.astype(np.uint8)

#     plt.imshow(output)
#     plt.axis("off")
#     plt.show()






import numpy as np
import maxflow
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import nibabel as nib
from scipy import ndimage as ndi
from matplotlib.path import Path


# ============================================================
# Seed mask utilities (ARBITRARY SHAPES)
# ============================================================

def seed_mask_from_rect(shape, rect):
    """rect=(x1,y1,x2,y2) with x2,y2 exclusive."""
    m, n = shape
    x1, y1, x2, y2 = rect
    mask = np.zeros((m, n), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def seed_mask_from_polygon(shape, vertices_xy):
    """
    Fill a polygon given by vertices [(x,y), ...] into a boolean mask.
    Works for arbitrary shapes (not only rectangles).
    """
    m, n = shape
    poly = Path(np.asarray(vertices_xy, dtype=np.float64))

    xs, ys = np.meshgrid(np.arange(n), np.arange(m))
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    inside = poly.contains_points(pts).reshape(m, n)
    return inside


def seed_mask_from_scribble_points(shape, points_xy, radius=3):
    """
    Create a seed mask from a set of scribble points by stamping disks.
    points_xy: iterable of (x,y)
    """
    m, n = shape
    mask = np.zeros((m, n), dtype=bool)
    rr = int(radius)

    for (x, y) in points_xy:
        x = int(round(x))
        y = int(round(y))
        y0, y1 = max(0, y - rr), min(m, y + rr + 1)
        x0, x1 = max(0, x - rr), min(n, x + rr + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - y)**2 + (xx - x)**2 <= rr ** 2
        mask[y0:y1, x0:x1] |= disk

    return mask


# ============================================================
# Feature extraction (more informative than intensity-only)
# ============================================================

def compute_features(I, spatial_weight=0.25, smooth_sigma=1.0):
    """
    Build a per-pixel feature vector:
      [HU, HU_smooth, gradmag, spatial_x, spatial_y]
    """
    I = I.astype(np.float64)
    m, n = I.shape

    Is = ndi.gaussian_filter(I, smooth_sigma)

    gx = ndi.sobel(Is, axis=1)
    gy = ndi.sobel(Is, axis=0)
    grad = np.hypot(gx, gy)

    xs, ys = np.meshgrid(np.arange(n), np.arange(m))
    x = (xs - (n - 1) / 2.0) / max(n, 1)
    y = (ys - (m - 1) / 2.0) / max(m, 1)

    F = np.stack([I, Is, grad, spatial_weight * x, spatial_weight * y], axis=-1)
    return F


def robust_normalize_features(F, roi_mask=None):
    """
    Robust normalize each feature dimension using median and IQR.
    """
    m, n, d = F.shape
    X = F.reshape(-1, d)

    if roi_mask is None:
        Xr = X
    else:
        Xr = X[roi_mask.ravel()]

    med = np.median(Xr, axis=0)
    q75 = np.percentile(Xr, 75, axis=0)
    q25 = np.percentile(Xr, 25, axis=0)
    iqr = (q75 - q25)
    iqr[iqr < 1e-8] = 1.0

    Xn = (X - med) / iqr
    return Xn.reshape(m, n, d)


# ============================================================
# Pairwise term scaling (GrabCut-style beta)
# ============================================================

def grabcut_beta(I):
    I = I.astype(np.float64)
    diffs = []
    diffs.append((I[:, 1:] - I[:, :-1]) ** 2)
    diffs.append((I[1:, :] - I[:-1, :]) ** 2)
    diffs.append((I[1:, 1:] - I[:-1, :-1]) ** 2)
    diffs.append((I[1:, :-1] - I[:-1, 1:]) ** 2)
    mean_diff_sq = np.mean([d.mean() for d in diffs])
    return 1.0 / (2.0 * mean_diff_sq + 1e-12)


def fit_gmm(X, n_components=5, reg_covar=1e-3, random_state=42):
    """
    Fit a GMM robustly; reduce components if too few points.
    X: (N, D)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    N = X.shape[0]
    if N < 5:
        k = 1
    else:
        k = min(n_components, max(1, N // 50))

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=reg_covar,
        random_state=random_state,
        max_iter=200,
        init_params="kmeans",
    )
    gmm.fit(X)
    return gmm




# ============================================================
# Main: iterative GrabCut-like GraphCut WITH ARBITRARY SEEDS
# ============================================================

def graph_cut_iterative(
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
):
    """
    Output segmentation:
      0 = background
      1 = foreground
    """
    I = np.asarray(slice_img, dtype=np.float64)
    m, n = I.shape

    fg_seed_mask = np.asarray(fg_seed_mask, dtype=bool)
    bg_seed_mask = np.asarray(bg_seed_mask, dtype=bool)

    if fg_seed_mask.shape != (m, n) or bg_seed_mask.shape != (m, n):
        raise ValueError("Seed masks must have same shape as slice_img.")

    if np.any(fg_seed_mask & bg_seed_mask):
        raise ValueError("FG and BG seeds overlap.")

    if fg_seed_mask.sum() == 0 or bg_seed_mask.sum() == 0:
        raise ValueError("Both FG and BG seed masks must contain at least 1 pixel.")

    # --- ROI mask (optional but usually helps) ---
    roi_mask = None
    if restrict_to_body:
        # crude "body" mask: everything above air threshold, then keep largest CC
        body = I > body_threshold_hu
        lab, num = ndi.label(body)
        if num > 0:
            sizes = ndi.sum(body, lab, index=np.arange(1, num + 1))
            largest = 1 + int(np.argmax(sizes))
            roi_mask = (lab == largest)
        else:
            roi_mask = body

        # ensure seeds are within ROI; if not, expand ROI to include them
        roi_mask = roi_mask | fg_seed_mask | bg_seed_mask

    # --- features for better unary term (not just intensity) ---
    F = compute_features(I, spatial_weight=spatial_weight, smooth_sigma=1.0)
    F = robust_normalize_features(F, roi_mask=roi_mask)
    d = F.shape[-1]

    # initial segmentation: FG where FG seeds, BG otherwise (within ROI)
    seg = np.zeros((m, n), dtype=np.uint8)
    seg[fg_seed_mask] = 1
    seg[bg_seed_mask] = 0

    beta = grabcut_beta(I)
    neighbors = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2.0)), (1, -1, np.sqrt(2.0))]

    eps = 1e-6
    fg_prior = float(np.clip(fg_prior, eps, 1.0 - eps))
    log_pi_f = np.log(fg_prior)
    log_pi_b = np.log(1.0 - fg_prior)

    # precompute distance maps if using distance bias
    if distance_bias > 0:
        dist_fg = ndi.distance_transform_edt(~fg_seed_mask)
        dist_bg = ndi.distance_transform_edt(~bg_seed_mask)
        dist_fg = dist_fg / (dist_fg.max() + 1e-8)
        dist_bg = dist_bg / (dist_bg.max() + 1e-8)

    for it in range(iterations):
        # ------------------------------------------------------------
        # Fit GMMs on current FG/BG (plus seeds)
        # ------------------------------------------------------------
        fg_train = (seg == 1) | fg_seed_mask
        bg_train = (seg == 0) | bg_seed_mask

        if roi_mask is not None:
            fg_train &= roi_mask
            bg_train &= roi_mask

        Xf = F[fg_train].reshape(-1, d)
        Xb = F[bg_train].reshape(-1, d)

        gmm_f = fit_gmm(Xf, n_components=n_components, reg_covar=1e-3)
        gmm_b = fit_gmm(Xb, n_components=n_components, reg_covar=1e-3)

        # ------------------------------------------------------------
        # Unary costs
        # Df = cost for FG, Db = cost for BG
        # ------------------------------------------------------------
        Xall = F.reshape(-1, d)
        ll_f = gmm_f.score_samples(Xall)
        ll_b = gmm_b.score_samples(Xall)


        Df = -(ll_f + log_pi_f).reshape(m, n)
        Db = -(ll_b + log_pi_b).reshape(m, n)

        # optional distance bias for spatial coherence around seeds
        if distance_bias > 0:
            Df = Df + distance_bias * dist_fg
            Db = Db + distance_bias * dist_bg

        # shift to non-negative
        off = min(Df.min(), Db.min())
        Df = np.maximum(Df - off, 0.0)
        Db = np.maximum(Db - off, 0.0)

        if verbose:
            print(f"[iter {it+1}/{iterations}] Df: {Df.min():.3f}..{Df.max():.3f} | "
                  f"Db: {Db.min():.3f}..{Db.max():.3f}")

        # ------------------------------------------------------------
        # Build graph
        #
        # Labels:
        #   0 = background (SOURCE side)
        #   1 = foreground (SINK side)
        #
        # add_tedge(p, cap_source, cap_sink):
        #   cost(BG=0) = cap_sink
        #   cost(FG=1) = cap_source
        #
        # so: cap_source = Df, cap_sink = Db.
        # ------------------------------------------------------------
        g = maxflow.Graph[float]()
        g.add_nodes(m * n)

        def node_id(i, j):
            return i * n + j

        for i in range(m):
            base = i * n
            for j in range(n):
                idx = base + j

                # If using ROI, force outside ROI to background
                if roi_mask is not None and (not roi_mask[i, j]):
                    g.add_tedge(idx, float(seed_inf), 0.0)  # force BG
                    continue

                if hard_seeds and fg_seed_mask[i, j]:
                    g.add_tedge(idx, 0.0, float(seed_inf))  # force FG
                elif hard_seeds and bg_seed_mask[i, j]:
                    g.add_tedge(idx, float(seed_inf), 0.0)  # force BG
                else:
                    g.add_tedge(idx, float(Df[i, j]), float(Db[i, j]))

                Ip = I[i, j]
                for di, dj, dist in neighbors:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        if roi_mask is not None and (not roi_mask[ni, nj]):
                            continue
                        diff = Ip - I[ni, nj]
                        w = (k / dist) * np.exp(-beta * (diff * diff))
                        if w > 0:
                            g.add_edge(idx, node_id(ni, nj), float(w), float(w))

        flow = g.maxflow()
        if verbose:
            print(f"[iter {it+1}/{iterations}] maxflow: {flow:.3f}")

        new_seg = np.zeros((m, n), dtype=np.uint8)
        for i in range(m):
            base = i * n
            for j in range(n):
                new_seg[i, j] = g.get_segment(base + j)  # 0=BG(source), 1=FG(sink)

        # keep only components that touch FG seeds (reduces scattered false positives)
        if keep_component_touching_fg:
            fg = (new_seg == 1)
            lab, num = ndi.label(fg)
            if num > 0:
                touching = np.unique(lab[fg_seed_mask & (lab > 0)])
                if touching.size > 0:
                    fg = np.isin(lab, touching)
                    new_seg = fg.astype(np.uint8)

        seg = new_seg

    return seg


# ============================================================
# Visualization + Dice (optional)
# ============================================================

def visualize(slice_img, seg, fg_seed_mask=None, bg_seed_mask=None, title="GraphCut"):
    I = np.asarray(slice_img, dtype=np.float64)
    vis = (I - I.min()) / (I.max() - I.min() + 1e-8)
    vis = (vis * 255).astype(np.uint8)

    out = np.stack([vis, vis, vis], axis=-1).astype(np.float32)
    alpha = 0.45
    fg = seg == 1
    bg = seg == 0

    # red FG
    out[fg, 0] = (1 - alpha) * out[fg, 0] + alpha * 255
    out[fg, 1] *= (1 - alpha)
    out[fg, 2] *= (1 - alpha)

    # blue BG
    out[bg, 2] = (1 - alpha) * out[bg, 2] + alpha * 255
    out[bg, 0] *= (1 - alpha)
    out[bg, 1] *= (1 - alpha)

    out = np.clip(out, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(vis, cmap="gray")
    ax[0].set_title("Input")
    ax[0].axis("off")

    ax[1].imshow(seg, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Mask (0=BG, 1=FG)")
    ax[1].axis("off")

    ax[2].imshow(out)
    if fg_seed_mask is not None:
        ax[2].contour(fg_seed_mask.astype(np.uint8), levels=[0.5], colors="lime", linewidths=1)
    if bg_seed_mask is not None:
        ax[2].contour(bg_seed_mask.astype(np.uint8), levels=[0.5], colors="red", linewidths=1)
    ax[2].set_title(title + " (overlay; seeds outlined)")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return (2.0 * inter) / (a.sum() + b.sum() + 1e-8)











