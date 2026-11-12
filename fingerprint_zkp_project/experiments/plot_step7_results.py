import os
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
step7_path = os.path.join(BASE_DIR, "results", "step7_radial_sampling_results.json")
step4_path = os.path.join(BASE_DIR, "results", "step4_evaluation_results.json")

with open(step7_path, "r") as f:
    s7 = json.load(f)
with open(step4_path, "r") as f:
    s4 = json.load(f)["exp_b1"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# 1. Ring Valid Cell Fraction Comparison
r_idx = np.arange(8)
old_ring_val = [0.0, 0.0, 100.0, 99.9, 98.8, 94.4, 87.2, 59.3] # Old
new_ring_val = [v * 100.0 for v in s7["ring_valid_fractions"]]

w = 0.35
axes[0].bar(r_idx - w/2, old_ring_val, width=w, label="Old warpPolar (Rings 0 & 1 Dead)", color="#e74c3c", alpha=0.85)
axes[0].bar(r_idx + w/2, new_ring_val, width=w, label="Step 7 remap Annulus (All Active)", color="#2ecc71", alpha=0.85)
axes[0].set_ylabel("Valid Cell Fraction (%) [Coverage >= 0.70]", fontsize=10, fontweight="bold")
axes[0].set_xlabel("Radial Ring (0=Innermost to 7=Outermost)", fontsize=10, fontweight="bold")
axes[0].set_title("Radial Ring Validity Profile\n(Old warpPolar vs. Step 7 remap)", fontsize=11, fontweight="bold")
axes[0].set_xticks(r_idx)
axes[0].set_xticklabels([f"r_{r}" for r in range(8)])
axes[0].legend(loc="lower left", frameon=True, fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")
axes[0].set_ylim(0, 115)

# 2. Score Distributions: Genuine vs Impostor
gen_s7 = [r["aligned_norm_hd"] * 100.0 for r in s7["pair_records"]["genuine"]]
imp_s7 = [r["aligned_norm_hd"] * 100.0 for r in s7["pair_records"]["impostor"]]
gen_s4 = [r["aligned_norm_hd"] * 100.0 for r in s4["pair_records"]["genuine"]]
imp_s4 = [r["aligned_norm_hd"] * 100.0 for r in s4["pair_records"]["impostor"]]

bins = np.linspace(15, 55, 30)
axes[1].hist(gen_s7, bins=bins, density=True, color="#2980b9", alpha=0.6, label=f"Step 7 Genuine (mu={s7['genuine_norm_hd_mean']:.1f}%)")
axes[1].hist(imp_s7, bins=bins, density=True, color="#c0392b", alpha=0.6, label=f"Step 7 Impostor (mu={s7['impostor_norm_hd_mean']:.1f}%)")
axes[1].hist(gen_s4, bins=bins, density=True, histtype="step", linewidth=2.0, color="#154360", label=f"Step 4 B1 Genuine (mu={s4['genuine_norm_hd_mean']:.1f}%)")
axes[1].hist(imp_s4, bins=bins, density=True, histtype="step", linewidth=2.0, color="#78281f", label=f"Step 4 B1 Impostor (mu={s4['impostor_norm_hd_mean']:.1f}%)")

axes[1].set_xlabel("Normalized Aligned HD (%)", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Probability Density", fontsize=10, fontweight="bold")
axes[1].set_title("Aligned Score Distributions\n(Step 4 B1 vs. Step 7)", fontsize=11, fontweight="bold")
axes[1].legend(loc="upper right", frameon=True, fontsize=8.5)
axes[1].grid(True, linestyle="--", alpha=0.4)

# 3. ROC Curves
def calc_roc(gen, imp):
    all_s = np.concatenate([gen, imp])
    thresholds = np.linspace(np.min(all_s) - 1e-4, np.max(all_s) + 1e-4, 501)
    tpr, fpr = [], []
    for tau in thresholds:
        tpr.append(np.mean(np.array(gen) >= tau))
        fpr.append(np.mean(np.array(imp) >= tau))
    s_idx = np.argsort(fpr)
    return np.array(fpr)[s_idx], np.array(tpr)[s_idx]

fpr_s7, tpr_s7 = calc_roc([-x for x in gen_s7], [-x for x in imp_s7])
fpr_s4, tpr_s4 = calc_roc([-x for x in gen_s4], [-x for x in imp_s4])

axes[2].plot(fpr_s7, tpr_s7, color="#27ae60", linewidth=2.2, label=f"Step 7 (remap Annulus, AUC={s7['roc_auc']:.4f})")
axes[2].plot(fpr_s4, tpr_s4, color="#7f8c8d", linewidth=1.8, linestyle="--", label=f"Step 4 B1 (Old warpPolar, AUC={s4['roc_auc']:.4f})")
axes[2].plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Chance (AUC=0.50)")

axes[2].set_xlabel("False Accept Rate (FAR / FPR)", fontsize=10, fontweight="bold")
axes[2].set_ylabel("True Accept Rate (TAR / TPR)", fontsize=10, fontweight="bold")
axes[2].set_title("Biometric Discrimination ROC\n(Subject-Cluster 95% CI: [0.609, 0.772])", fontsize=11, fontweight="bold")
axes[2].legend(loc="lower right", frameon=True, fontsize=9)
axes[2].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
fig_path = os.path.join(BASE_DIR, "results", "figures", "step7_radial_sampling_comparison.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=200)
print(f"[+] Figure saved to: {fig_path}")

