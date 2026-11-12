import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res_path = os.path.join(BASE_DIR, "results", "milestone11_integrated_system_results.json")

if not os.path.exists(res_path):
    print(f"Results file not found at {res_path}")
    sys.exit(1)

with open(res_path, "r", encoding="utf-8") as f:
    data = json.load(f)

m1 = data["model1_legitimate_user"]
m2 = data["model2_zkp_impostor_attack"]
m3 = data["model3_stolen_credential_attack"]
m4 = data["model4_credential_mismatch"]

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

# -------------------------------------------------------------
# Panel 1: End-to-End Threat Model Acceptance Rates
# -------------------------------------------------------------
models = [
    "Model 1:\nLegitimate User\n(Gen + Correct K)",
    "Model 2:\nZKP Impostor Attack\n(Imp + No K)",
    "Model 3:\nStolen Credential\n(Imp + Stolen K)",
    "Model 4:\nWrong Credential\n(Gen + Wrong K)"
]

auth_rates = [
    m1["genuine_acceptance_rate"] * 100.0,
    m2["cryptographic_far"] * 100.0,
    m3["cryptographic_far"] * 100.0,
    m4["acceptance_rate"] * 100.0
]

bio_rates = [
    m1["secret_recovery_rate"] * 100.0,
    m2["biometric_collision_rate"] * 100.0,
    m3["biometric_collision_rate"] * 100.0,
    m4["secret_recovery_rate"] * 100.0
]

x = np.arange(len(models))
w = 0.35

b1 = axes[0].bar(x - w/2, bio_rates, width=w, label="Biometric Recovery (S')", color="#e67e22", alpha=0.85)
b2 = axes[0].bar(x + w/2, auth_rates, width=w, label="End-to-End Auth (ZKP Verified)", color="#27ae60", alpha=0.85)

# Custom color for Model 2 and Model 4
b2[1].set_color("#2980b9") # 0% FAR
b2[2].set_color("#c0392b") # Stolen K
b2[3].set_color("#8e44ad") # Wrong K

for i in range(len(models)):
    axes[0].text(x[i] - w/2, bio_rates[i] + 1.5, f"{bio_rates[i]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#d35400")
    axes[0].text(x[i] + w/2, auth_rates[i] + 1.5, f"{auth_rates[i]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#196f3d" if auth_rates[i]>0 else "#2e4053")

axes[0].set_ylabel("Authentication / Recovery Rate (%)", fontsize=11, fontweight="bold")
axes[0].set_title("Milestone 11: End-to-End Threat Model Evaluation\n(Locked Benchmark: 100 G / 100 I Pairs)", fontsize=11, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=8.5)
axes[0].legend(loc="upper right", frameon=True, fontsize=8.5)
axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")
axes[0].set_ylim(0, 75)

# -------------------------------------------------------------
# Panel 2: Cryptographic Latency Breakdown (Mean ms)
# -------------------------------------------------------------
lat = m1["latency_ms_summary"]
stages = [
    "Feature\nExtraction",
    "ECC\nDecoding",
    "Witness\nDerivation",
    "Ristretto255\nZKP Protocol"
]
lat_vals = [
    lat.get("feature_extraction", 0.0),
    lat.get("ecc_decoding", 0.0),
    lat.get("witness_derivation", 0.0),
    lat.get("zkp_protocol", 0.0)
]
colors_lat = ["#34495e", "#2980b9", "#f39c12", "#27ae60"]

bars = axes[1].bar(stages, lat_vals, color=colors_lat, alpha=0.85, width=0.5)
for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval + (0.05 * max(lat_vals)), f"{yval:.2f} ms",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#2c3e50")

axes[1].set_ylabel("Latency (milliseconds)", fontsize=11, fontweight="bold")
axes[1].set_title(f"Authentication Latency Breakdown\n(Total Round-Trip: {lat.get('total', 0.0):.2f} ms)", fontsize=11, fontweight="bold")
axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")
axes[1].set_ylim(0, max(lat_vals) * 1.25)

# -------------------------------------------------------------
# Panel 3: Two-Factor Defense Mechanism (FAR Suppression)
# -------------------------------------------------------------
labels_pie = [
    f"Rejected by Biometrics\n({100 - m2['biometric_collision_rate']*100:.0f}%)",
    f"Biometric Collision Neutralized by ZKP\n({m2['biometric_collision_rate']*100:.0f}%)",
    f"Accepted Impostor FAR\n({m2['cryptographic_far']*100:.1f}%)"
]
sizes = [
    100.0 - m2["biometric_collision_rate"] * 100.0,
    m2["biometric_collision_rate"] * 100.0,
    m2["cryptographic_far"] * 100.0
]
colors_pie = ["#2ecc71", "#3498db", "#e74c3c"]
explode = (0.0, 0.08, 0.15) if sizes[2] > 0 else (0.0, 0.08, 0.0)

axes[2].pie(sizes[:2] if sizes[2] == 0 else sizes,
            labels=labels_pie[:2] if sizes[2] == 0 else labels_pie,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors_pie[:2] if sizes[2] == 0 else colors_pie,
            explode=explode[:2] if sizes[2] == 0 else explode,
            textprops={"fontsize": 9, "fontweight": "bold"})
axes[2].set_title("Zero-Knowledge Impostor Defense Breakdown\n(100 Impostor Attack Trials)", fontsize=11, fontweight="bold")

plt.tight_layout()

# Save
fig_path1 = os.path.join(BASE_DIR, "results", "figures", "milestone11_integrated_system.png")
os.makedirs(os.path.dirname(fig_path1), exist_ok=True)
plt.savefig(fig_path1, dpi=200)

art_dir = r"C:\Users\sofia\.gemini\antigravity\brain\374fe828-0336-41eb-963e-1de0160545a8"
fig_path2 = os.path.join(art_dir, "milestone11_integrated_system.png")
plt.savefig(fig_path2, dpi=200)

print(f"[+] Saved figure to: {fig_path1}")
print(f"[+] Saved figure to artifact dir: {fig_path2}")

