import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
step9_path = os.path.join(BASE_DIR, "results", "step9_operating_points_and_ecc_results.json")

if not os.path.exists(step9_path):
    print(f"Results file not found at {step9_path}")
    sys.exit(1)

with open(step9_path, "r", encoding="utf-8") as f:
    data = json.load(f)

p9a = data["part_9a_biometric"]
p9b = data["part_9b_ecc"]

fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

# -------------------------------------------------------------
# Panel 1: Biometric Verification ROC & Operating Points
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
    "cond_A": "#2980b9",
    "cond_C": "#e67e22",
    "cond_D": "#27ae60"
}
labels = {
    "cond_A": "Cond A (4ch Baseline, 512b)",
    "cond_C": "Cond C (3ch Orientation Unweighted, 384b)",
    "cond_D": "Cond D (3ch Orientation Weighted, 384b)"
}

for k in ["cond_A", "cond_C", "cond_D"]:
    gen_scores = [-r["dist"] for r in p9a[k]["scores"]["gen"]]
    imp_scores = [-r["dist"] for r in p9a[k]["scores"]["imp"]]
    fpr, tpr = calc_roc(gen_scores, imp_scores)
    axes[0].plot(fpr, tpr, color=colors[k], linewidth=2.2,
                 label=f"{labels[k]}\n(AUC={p9a[k]['auc']:.4f}, EER={p9a[k]['eer']*100.0:.1f}%)")
    
    # Mark Zero-FAR operational point
    z_gar = p9a[k]["zero_far"]["gar"]
    axes[0].plot(0.0, z_gar, marker="o", color=colors[k], markersize=6)

axes[0].plot([0, 1], [0, 1], "k:", alpha=0.4, label="Random Chance")
axes[0].set_xlabel("False Accept Rate (FAR / FPR)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("True Accept Rate (TAR / GAR)", fontsize=11, fontweight="bold")
axes[0].set_title("Part 9A: Biometric Verification ROC Curves\n(Locked Benchmark: 100 G / 100 I Pairs)", fontsize=11, fontweight="bold")
axes[0].legend(loc="lower right", frameon=True, fontsize=8.5)
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].set_xlim(-0.02, 1.02)
axes[0].set_ylim(-0.02, 1.02)

# -------------------------------------------------------------
# Panel 2: ECC Secret Recovery Near-Miss Distributions (CDF)
# -------------------------------------------------------------
x_bits = np.arange(0, 11)
for k in ["cond_A", "cond_C", "cond_D"]:
    rec = p9b[k]["records"]["genuine"]
    f_bits = [r["failed_bits"] for r in rec]
    cdf = [np.mean(np.array(f_bits) <= b) * 100.0 for b in x_bits]
    axes[1].plot(x_bits, cdf, marker="s", linewidth=2.2, color=colors[k],
                 label=f"{labels[k]}\n(Full Rec={p9b[k]['genuine_recovery_rate']*100.0:.1f}%, Mean Fail={p9b[k]['mean_failed_bits']:.2f})")

axes[1].axvline(0, color="k", linestyle="--", alpha=0.5, label="Full Recovery (0 errors)")
axes[1].set_xlabel("Max Permitted Secret Bit Errors (t_outer / 32 bits)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Cumulative Genuine Recovery Rate (%)", fontsize=11, fontweight="bold")
axes[1].set_title("Part 9B: ECC Cumulative Recovery CDF\n(Interleaved Repetition Decoder)", fontsize=11, fontweight="bold")
axes[1].set_xticks(x_bits)
axes[1].legend(loc="lower right", frameon=True, fontsize=8.5)
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].set_ylim(0, 100)

# -------------------------------------------------------------
# Panel 3: Cryptographic Impostor FAR vs Alignment Regime
# -------------------------------------------------------------
t_vals = [1, 2, 3, 4]
x_t = np.arange(len(t_vals))
w_bar = 0.35

# Plot Condition D under Regime 1 vs Regime 2
gen_reg1 = [p9b["cond_D"]["outer_code_analysis"][f"t_{t}"]["regime1_prelocked"]["gen_recovery"] * 100.0 for t in t_vals]
imp_reg1 = [p9b["cond_D"]["outer_code_analysis"][f"t_{t}"]["regime1_prelocked"]["imp_far"] * 100.0 for t in t_vals]

gen_reg2 = [p9b["cond_D"]["outer_code_analysis"][f"t_{t}"]["regime2_exhaustive"]["gen_recovery"] * 100.0 for t in t_vals]
imp_reg2 = [p9b["cond_D"]["outer_code_analysis"][f"t_{t}"]["regime2_exhaustive"]["imp_far"] * 100.0 for t in t_vals]

axes[2].bar(x_t - w_bar/2, imp_reg1, width=w_bar, label="Regime 1: Pre-Locked Alignment (Impostor FAR)", color="#27ae60", alpha=0.85)
axes[2].bar(x_t + w_bar/2, imp_reg2, width=w_bar, label="Regime 2: Exhaustive Shift Search (Impostor FAR)", color="#c0392b", alpha=0.85)

for i in range(len(t_vals)):
    axes[2].text(x_t[i] - w_bar/2, imp_reg1[i] + 1.0, f"{imp_reg1[i]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#1e8449")
    axes[2].text(x_t[i] + w_bar/2, imp_reg2[i] + 1.0, f"{imp_reg2[i]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#922b21")

axes[2].set_xlabel("Outer Error Correction Capacity (t_outer / 32 bits)", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Verified Cryptographic Impostor FAR (%)", fontsize=11, fontweight="bold")
axes[2].set_title("Part 9B: Cryptographic Impostor Collision Risk\n(Pre-Locked Prior vs. Post-Decoding Search)", fontsize=11, fontweight="bold")
axes[2].set_xticks(x_t)
axes[2].set_xticklabels([f"t <= {t} bits" for t in t_vals], fontsize=9.5)
axes[2].legend(loc="upper left", frameon=True, fontsize=8.5)
axes[2].grid(True, linestyle="--", alpha=0.4, axis="y")
axes[2].set_ylim(0, max(max(imp_reg2) + 12.0, 30.0))

plt.tight_layout()

# Save
fig_path1 = os.path.join(BASE_DIR, "results", "figures", "step9_operating_points_and_ecc.png")
os.makedirs(os.path.dirname(fig_path1), exist_ok=True)
plt.savefig(fig_path1, dpi=200)

art_dir = r"C:\Users\sofia\.gemini\antigravity\brain\374fe828-0336-41eb-963e-1de0160545a8"
fig_path2 = os.path.join(art_dir, "step9_operating_points_and_ecc.png")
plt.savefig(fig_path2, dpi=200)

print(f"[+] Saved figure to: {fig_path1}")
print(f"[+] Saved figure to artifact dir: {fig_path2}")

