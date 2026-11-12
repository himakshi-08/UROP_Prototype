"""
Generate Step 3 Visualizations:
Visualizes the 8-orientation quadrature Gabor energy maps, total energy,
and dominant orientation on a representative fingerprint to verify that
filters capture genuine ridge structure rather than background noise.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.dataset_loader import default_split_dirs
from preprocessing import preprocess_fingerprint
from features import extract_minutiae, build_template
from features.template import _quadrature_gabor_bank, N_FILTERS

def main():
    splits = default_split_dirs()
    test_dir = splits["test"]

    # Pick representative print from first subject
    subjs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    first_subj = subjs[0]
    img_files = sorted([f for f in os.listdir(os.path.join(test_dir, first_subj)) if f.lower().endswith(('.bmp', '.png', '.tif'))])
    img_path = os.path.join(test_dir, first_subj, img_files[0])

    print(f"[*] Visualizing Gabor responses for: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    prep = preprocess_fingerprint(img)
    minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
    tmpl = build_template(img, prep, minutiae, vector_bits=256)

    # Get prepared polar data
    prepared = tmpl._prepare_polar(img)
    img_w, mask_w, ori_w, center, max_r, inner = prepared
    h, w = img_w.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    ring_mask = ((rr >= inner) & (rr <= max_r) & (mask_w > 0.25)).astype(np.float32)
    img_roi = img_w * ring_mask

    gabor_bank = _quadrature_gabor_bank()
    energies = []
    thetas_deg = []
    for ke, ko, theta_k in gabor_bank:
        re = cv2.filter2D(img_roi, cv2.CV_32F, ke)
        ro = cv2.filter2D(img_roi, cv2.CV_32F, ko)
        e = np.sqrt(re * re + ro * ro) * ring_mask
        energies.append(e)
        thetas_deg.append(np.degrees(theta_k))

    E_total = np.sum(energies, axis=0)

    # Create figure: 2 rows of 5 plots
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    axes = axes.flatten()

    # 1. Original Centered ROI
    axes[0].imshow(img_roi, cmap="gray")
    axes[0].set_title("Aligned ROI (Input to Filters)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # 2. Ring Mask & Center
    axes[1].imshow(ring_mask, cmap="Blues")
    axes[1].plot(center[0], center[1], "rx", markersize=8, label="Centroid")
    axes[1].set_title(f"Ring Mask (R=[{inner:.0f}, {max_r:.0f}])", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # 3-10: 8 Directional Gabor Energy Maps
    for k in range(8):
        ax = axes[k + 2]
        im = ax.imshow(energies[k], cmap="inferno")
        ax.set_title(f"Gabor {k}: {thetas_deg[k]:.1f}° Energy", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    fig_dir = os.path.join(BASE_DIR, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_fig = os.path.join(fig_dir, "step3_gabor_directional_responses.png")
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved directional response visualization to: {out_fig}")

if __name__ == "__main__":
    main()

