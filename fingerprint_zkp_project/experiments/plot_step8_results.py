import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
step8_path = os.path.join(BASE_DIR, "results", "step8_channel_ablation_results.json")

if not os.path.exists(step8_path):
    print(f"Results file not found at {step8_path}")
    sys.exit(1)

with open(step8_path, "r", encoding="utf-8") as f:
    data = json.load(f)

conds = data["conditions"]
boot = data["bootstrap_summary"]
deltas = data["delta_summary"]

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

# -------------------------------------------------------------
# Panel 1: Factorial Distance & Separation Breakdown
# -------------------------------------------------------------
c_keys = ["cond_A", "cond_B", "cond_C", "cond_D", "cond_E"]
labels = [
    "A: 4ch Unw\n(Baseline)",
    "B: 4ch LLR\n(Weighted)",
    "C: 3ch Unw\n(No Energy)",
    "D: 3ch LLR\n(No Energy+W)",
    "E: 3ch Joint\n(Ring x Ch)"
]

gen_means = [conds[k]["genuine_mean"] for k in c_keys]
gen_stds = [conds[k]["genuine_std"] for k in c_keys]
imp_means = [conds[k]["impostor_mean"] for k in c_keys]
imp_stds = [conds[k]["impostor_std"] for k in c_keys]
seps = [conds[k]["separation"] for k in c_keys]

x = np.arange(len(c_keys))
width = 0.35

axes[0].bar(x - width/2, gen_means, width, yerr=gen_stds, capsize=4,
            label="Genuine Mean HD/WHD", color="#2980b9", alpha=0.85)
axes[0].bar(x + width/2, imp_means, width, yerr=imp_stds, capsize=4,
            label="Impostor Mean HD/WHD", color="#c0392b", alpha=0.85)

for i in range(len(c_keys)):
    axes[0].text(x[i], max(gen_means[i], imp_means[i]) + 6.0,
                 f"Δ = {seps[i]:.2f} pp", ha="center", fontsize=9, fontweight="bold", color="#2c3e50")

axes[0].set_ylabel("Normalized Distance (%)", fontsize=11, fontweight="bold")
axes[0].set_title("Genuine vs. Impostor Distance & Separation\n(2x2 Factorial + Joint Ring Ablation)", fontsize=11, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=9)
axes[0].legend(loc="upper left", frameon=True, fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")
axes[0].set_ylim(0, 52)

# -------------------------------------------------------------
# Panel 2: ROC-AUC & Decidability (d') with 95% Bootstrap CIs
# -------------------------------------------------------------
aucs = [conds[k]["auc"] for k in c_keys]
auc_cis = [boot[k]["auc_ci"] for k in c_keys]
auc_errs = np.array([[aucs[i] - auc_cis[i][0], auc_cis[i][1] - aucs[i]] for i in range(len(c_keys))]).T

d_primes = [conds[k]["decidability"] for k in c_keys]
dp_cis = [boot[k]["dp_ci"] for k in c_keys]
dp_errs = np.array([[d_primes[i] - dp_cis[i][0], dp_cis[i][1] - d_primes[i]] for i in range(len(c_keys))]).T

ax2_twin = axes[1].twinx()

b1 = axes[1].bar(x - width/2, aucs, width, yerr=auc_errs, capsize=4,
                 label="ROC-AUC (95% CI)", color="#27ae60", alpha=0.85)
b2 = ax2_twin.bar(x + width/2, d_primes, width, yerr=dp_errs, capsize=4,
                  label="Decidability d' (95% CI)", color="#8e44ad", alpha=0.85)

for i in range(len(c_keys)):
    axes[1].text(x[i] - width/2, aucs[i] + 0.02, f"{aucs[i]:.4f}", ha="center", fontsize=8.5, fontweight="bold", color="#1e8449")
    ax2_twin.text(x[i] + width/2, d_primes[i] + 0.03, f"{d_primes[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold", color="#6c3483")

axes[1].set_ylabel("ROC-AUC", fontsize=11, fontweight="bold", color="#1e8449")
ax2_twin.set_ylabel("Decidability Index d'", fontsize=11, fontweight="bold", color="#6c3483")
axes[1].set_title("Discrimination Performance: ROC-AUC & d'\n(Subject-Cluster Resampled 95% CIs)", fontsize=11, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=9)
axes[1].set_ylim(0.50, 0.85)
ax2_twin.set_ylim(0.40, 1.20)
axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

# -------------------------------------------------------------
# Panel 3: Full ROC Curves
# -------------------------------------------------------------
def calc_roc(gen, imp):
    all_s = np.concatenate([gen, imp])
    thresholds = np.linspace(np.min(all_s) - 1e-4, np.max(all_s) + 1e-4, 501)
    tpr, fpr = [], []
    for tau in thresholds:
        tpr.append(np.mean(np.array(gen) >= tau))
        fpr.append(np.mean(np.array(imp) >= tau))
    s_idx = np.argsort(fpr)
    return np.array(fpr)[s_idx], np.array(tpr)[s_idx]

colors = {
    "cond_A": "#7f8c8d",
    "cond_B": "#3498db",
    "cond_C": "#e67e22",
    "cond_D": "#27ae60",
    "cond_E": "#9b59b6"
}
styles = {
    "cond_A": "--",
    "cond_B": "-.",
    "cond_C": "-",
    "cond_D": "-",
    "cond_E": ":"
}
lw = {
    "cond_A": 1.8,
    "cond_B": 1.8,
    "cond_C": 2.2,
    "cond_D": 2.6,
    "cond_E": 1.8
}

for k in c_keys:
    g = [-r["weighted_hd"] * 100.0 for r in conds[k]["pair_records"]["genuine"]]
    i = [-r["weighted_hd"] * 100.0 for r in conds[k]["pair_records"]["impostor"]]
    fpr, tpr = calc_roc(g, i)
    axes[2].plot(fpr, tpr, color=colors[k], linestyle=styles[k], linewidth=lw[k],
                 label=f"{labels[c_keys.index(k)].split()[0]} (AUC={conds[k]['auc']:.4f})")

axes[2].plot([0, 1], [0, 1], "k:", alpha=0.4, label="Random Chance (0.50)")
axes[2].set_xlabel("False Accept Rate (FAR / FPR)", fontsize=11, fontweight="bold")
axes[2].set_ylabel("True Accept Rate (TAR / TPR)", fontsize=11, fontweight="bold")
axes[2].set_title("Biometric Verification ROC Curves\n(Locked Benchmark: 100 G / 100 I Pairs)", fontsize=11, fontweight="bold")
axes[2].legend(loc="lower right", frameon=True, fontsize=8.5)
axes[2].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

# Save figure in results/figures and in artifact directory
fig_path1 = os.path.join(BASE_DIR, "results", "figures", "step8_channel_ablation_comparison.png")
os.makedirs(os.path.dirname(fig_path1), exist_ok=True)
plt.savefig(fig_path1, dpi=200)

art_dir = r"C:\Users\sofia\.gemini\antigravity\brain\374fe828-0336-41eb-963e-1de0160545a8"
fig_path2 = os.path.join(art_dir, "step8_channel_ablation_comparison.png")
plt.savefig(fig_path2, dpi=200)

print(f"[+] Saved figure to: {fig_path1}")
print(f"[+] Saved figure to artifact dir: {fig_path2}")

