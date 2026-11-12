import os
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(BASE_DIR, "results", "step5_ecc_feasibility_results.json")
with open(results_path, "r") as f:
    res = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# 1. Full-Secret Recovery Rate (%)
cfgs = ["B1-A\n(512b Consec)", "B1-B\n(512b Interl)", "B2-A\n(256b Consec)", "B2-B\n(256b Interl)"]
rec_rates = [res[k]["genuine_recovery_rate_pct"] for k in ["B1-A", "B1-B", "B2-A", "B2-B"]]
frr_rates = [res[k]["genuine_frr_pct"] for k in ["B1-A", "B1-B", "B2-A", "B2-B"]]

x = np.arange(len(cfgs))
w = 0.45
bars = axes[0].bar(x, rec_rates, width=w, color=["#4a7bb0", "#1f77b4", "#52b152", "#2ca02c"], edgecolor="black")
for bar in bars:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)

axes[0].set_ylabel("Full-Secret Recovery Rate (%)", fontsize=11, fontweight="bold")
axes[0].set_title("Genuine 32-Bit Secret Recovery Rate\n(All 32 Blocks Correct & Hash Matched)", fontsize=12, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(cfgs, fontsize=9.5)
axes[0].set_ylim(0, 16)
axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

# 2. Histogram of Failed Bits per Query (B1-B vs B2-B)
labels = ["0 (Match)", "1 bit", "2 bits", "3 bits", "4 bits", "5+ bits"]
b1_b_hist = [res["B1-B"]["failed_bits_histogram"][k] for k in ["0", "1", "2", "3", "4", "5+"]]
b2_b_hist = [res["B2-B"]["failed_bits_histogram"][k] for k in ["0", "1", "2", "3", "4", "5+"]]

x_h = np.arange(len(labels))
w_h = 0.35
axes[1].bar(x_h - w_h/2, b1_b_hist, width=w_h, label="B1-B (512b Interleaved)", color="#1f77b4", alpha=0.9)
axes[1].bar(x_h + w_h/2, b2_b_hist, width=w_h, label="B2-B (256b Interleaved)", color="#2ca02c", alpha=0.9)
axes[1].set_ylabel("Number of Genuine Queries (out of 100)", fontsize=11, fontweight="bold")
axes[1].set_title("Distribution of Failed Secret Bits\n(Queries Near Successful Recovery)", fontsize=12, fontweight="bold")
axes[1].set_xticks(x_h)
axes[1].set_xticklabels(labels, fontsize=9.5)
axes[1].legend(loc="upper right", frameon=True, fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

# 3. Ties & Erasures Breakdown
ties_mean = [res[k]["genuine_ties_mean"] for k in ["B1-A", "B1-B", "B2-A", "B2-B"]]
ties_std = [res[k]["genuine_ties_std"] for k in ["B1-A", "B1-B", "B2-A", "B2-B"]]

axes[2].bar(x, ties_mean, yerr=ties_std, capsize=5, width=w, color="#e67e22", edgecolor="black", alpha=0.85)
for i, v in enumerate(ties_mean):
    axes[2].text(i, v + 0.25, f"{v:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=10)

axes[2].set_ylabel("Mean Ties / Erasures per Query", fontsize=11, fontweight="bold")
axes[2].set_title("Ambiguous Majority Decisions (Ties)\n(Consecutive vs. Interleaved)", fontsize=12, fontweight="bold")
axes[2].set_xticks(x)
axes[2].set_xticklabels(cfgs, fontsize=9.5)
axes[2].set_ylim(0, 9)
axes[2].grid(True, linestyle="--", alpha=0.4, axis="y")

plt.tight_layout()
fig_path = os.path.join(BASE_DIR, "results", "figures", "step5_ecc_feasibility_distributions.png")
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=200)
print(f"[+] Figure saved successfully to: {fig_path}")

