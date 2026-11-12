import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(BASE_DIR, "results", "step4_evaluation_results.json")
with open(results_path, "r") as f:
    res = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Plot 1: Evolutionary Progression across Steps
steps = ['Step 0\nBaseline', 'Step 1\nDouble-Angle', 'Step 2\nMask-Aware', 'Step 3\nGabor Bank', 'Step 4-B1\n512b Gray', 'Step 4-B2\n256b Pooled']
gen_means = [41.06, 38.66, 35.26, 34.03, res['exp_b1']['genuine_norm_hd_mean'], res['exp_b2']['genuine_norm_hd_mean']]
imp_means = [39.50, 40.00, 37.82, 36.71, res['exp_b1']['impostor_norm_hd_mean'], res['exp_b2']['impostor_norm_hd_mean']]

x = np.arange(len(steps))
w = 0.35
axes[0].bar(x - w/2, gen_means, width=w, label='Genuine Aligned HD (%)', color='#2b5c8f', alpha=0.9)
axes[0].bar(x + w/2, imp_means, width=w, label='Impostor Aligned HD (%)', color='#c23b22', alpha=0.9)
axes[0].set_ylabel('Normalized Hamming Distance (%)', fontsize=11, fontweight='bold')
axes[0].set_title('Progression of Genuine vs. Impostor HD', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(steps, fontsize=9)
axes[0].grid(True, linestyle='--', alpha=0.4, axis='y')
axes[0].legend(loc='upper right', frameon=True)
axes[0].set_ylim(0, 50)

# Plot 2: Genuine vs Impostor Distributions for B1 & B2
b1_gen = [r['aligned_norm_hd'] * 100 for r in res['exp_b1']['pair_records']['genuine']]
b1_imp = [r['aligned_norm_hd'] * 100 for r in res['exp_b1']['pair_records']['impostor']]
b2_gen = [r['aligned_norm_hd'] * 100 for r in res['exp_b2']['pair_records']['genuine']]
b2_imp = [r['aligned_norm_hd'] * 100 for r in res['exp_b2']['pair_records']['impostor']]

bins = np.linspace(15, 55, 30)
axes[1].hist(b1_gen, bins=bins, density=True, alpha=0.5, color='#1f77b4', label=f'B1 (512b) Genuine (mu={np.mean(b1_gen):.1f}%)')
axes[1].hist(b1_imp, bins=bins, density=True, alpha=0.5, color='#d62728', label=f'B1 (512b) Impostor (mu={np.mean(b1_imp):.1f}%)')
axes[1].hist(b2_gen, bins=bins, density=True, histtype='step', linewidth=2.5, color='#2ca02c', label=f'B2 (256b) Genuine (mu={np.mean(b2_gen):.1f}%)')
axes[1].hist(b2_imp, bins=bins, density=True, histtype='step', linewidth=2.5, color='#ff7f0e', label=f'B2 (256b) Impostor (mu={np.mean(b2_imp):.1f}%)')
axes[1].set_xlabel('Normalized Aligned HD (%)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Probability Density', fontsize=11, fontweight='bold')
axes[1].set_title('Empirical Score Distributions (Exp B1 vs B2)', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=8.5, frameon=True)
axes[1].grid(True, linestyle='--', alpha=0.4)

# Plot 3: ROC Curves
def get_roc(gen, imp):
    all_s = np.concatenate([gen, imp])
    thresholds = np.linspace(np.min(all_s) - 1e-4, np.max(all_s) + 1e-4, 501)
    tpr, fpr = [], []
    for tau in thresholds:
        tpr.append(np.mean(np.array(gen) >= tau))
        fpr.append(np.mean(np.array(imp) >= tau))
    s_idx = np.argsort(fpr)
    return np.array(fpr)[s_idx], np.array(tpr)[s_idx]

auc_a = res["exp_a"]["roc_auc"]
auc_b1 = res["exp_b1"]["roc_auc"]
auc_b2 = res["exp_b2"]["roc_auc"]
auc_c2 = res["exp_c2"]["roc_auc"]

fpr_a, tpr_a = get_roc([-x for x in [r['aligned_norm_hd']*100 for r in res['exp_a']['pair_records']['genuine']]], [-x for x in [r['aligned_norm_hd']*100 for r in res['exp_a']['pair_records']['impostor']]])
fpr_b1, tpr_b1 = get_roc([-x for x in b1_gen], [-x for x in b1_imp])
fpr_b2, tpr_b2 = get_roc([-x for x in b2_gen], [-x for x in b2_imp])
fpr_c2, tpr_c2 = get_roc([-x for x in [r['aligned_norm_hd']*100 for r in res['exp_c2']['pair_records']['genuine']]], [-x for x in [r['aligned_norm_hd']*100 for r in res['exp_c2']['pair_records']['impostor']]])

axes[2].plot(fpr_a, tpr_a, label=f'Exp A (256b Step 3 Ref, AUC={auc_a:.4f})', color='gray', linestyle='--')
axes[2].plot(fpr_b1, tpr_b1, label=f'Exp B1 (512b Gray, AUC={auc_b1:.4f})', color='#1f77b4', linewidth=2)
axes[2].plot(fpr_b2, tpr_b2, label=f'Exp B2 (256b Pooled, AUC={auc_b2:.4f})', color='#2ca02c', linewidth=2)
axes[2].plot(fpr_c2, tpr_c2, label=f'Exp C2 (256b Pooled+Mask, AUC={auc_c2:.4f})', color='#9467bd', linewidth=1.8, linestyle='-.')
axes[2].plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random Chance (AUC=0.50)')

axes[2].set_xlabel('False Accept Rate (FAR / FPR)', fontsize=11, fontweight='bold')
axes[2].set_ylabel('True Accept Rate (TAR / TPR)', fontsize=11, fontweight='bold')
axes[2].set_title('Biometric Discrimination ROC Curves', fontsize=12, fontweight='bold')
axes[2].legend(loc='lower right', fontsize=8.5, frameon=True)
axes[2].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
fig_path = os.path.join(BASE_DIR, 'results', 'figures', 'step4_experimental_comparison.png')
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=200)
print(f"[+] Figure saved successfully to: {fig_path}")

