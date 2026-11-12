"""
Step 6A: Spatial Failure Analysis for Polar FingerCode Representation
Analyzes the spatial and structural error distributions across all 100 genuine benchmark pairs:
1. 2D Error Heatmap across 16 sectors x 8 rings.
2. 2D Erasure Heatmap across 16 sectors x 8 rings.
3. Radial ring-wise error and erasure profiles (rings 0 to 7).
4. Angular sector-wise error and erasure profiles (sectors 0 to 15).
5. Channel-wise error breakdown (Energy bit vs. Gray b2, b1, b0).
6. Secret-bit failure distribution across all 32 secret bits.
"""

import os
import sys
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.evaluate_step4 import extract_step4_all_configs
from experiments.evaluate_step5_ecc import make_permutations, hash_secret, decode_erasure_majority

def main():
    benchmark_json_path = os.path.join(BASE_DIR, "data", "fixed_pairs_benchmark.json")
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    unique_paths = set()
    for item in bench["genuine_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])

    print(f"[*] Extracting templates for {len(unique_paths)} unique benchmark images...")
    template_cache = {p: extract_step4_all_configs(p) for p in unique_paths}
    (p512, inv512), _ = make_permutations(seed=20261112)

    k_bits = 32
    rep_factor = 16
    bps = 32
    tau = 0.7

    # 1. Enrollment Store
    enroll_store = {}
    for p in set(item["path_1"] for item in bench["genuine_pairs"]):
        seed = int(hashlib.sha256(p.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        secret = rng.randint(0, 2, k_bits, dtype=np.uint8)
        commitment = hash_secret(secret)
        feat = template_cache[p]["exp_b1"]
        b_enr = feat["vec"]
        v_enr = (feat["cov"] >= tau).astype(np.uint8)
        tiled = np.tile(secret, rep_factor)
        cw = tiled[p512]
        w = np.bitwise_xor(b_enr, cw)
        enroll_store[p] = {"secret": secret, "commitment": commitment, "w": w, "v_enr": v_enr}

    # Tracking spatial matrices: shape (16 sectors, 8 rings)
    cell_eval_counts = np.zeros((16, 8), dtype=np.float64)
    cell_valid_counts = np.zeros((16, 8), dtype=np.float64)
    cell_error_counts = np.zeros((16, 8), dtype=np.float64)

    # Channel-wise tracking (0=Energy, 1=Gray-b2, 2=Gray-b1, 3=Gray-b0)
    channel_eval_counts = np.zeros(4, dtype=np.float64)
    channel_valid_counts = np.zeros(4, dtype=np.float64)
    channel_error_counts = np.zeros(4, dtype=np.float64)

    # Per-secret-bit failure tracking
    secret_bit_fails = np.zeros(k_bits, dtype=np.int32)
    secret_bit_ties = np.zeros(k_bits, dtype=np.int32)

    for item in bench["genuine_pairs"]:
        enr = enroll_store[item["path_1"]]
        secret = enr["secret"]
        w = enr["w"]
        v1 = enr["v_enr"]
        b1 = template_cache[item["path_1"]]["exp_b1"]["vec"]

        feat_q = template_cache[item["path_2"]]["exp_b1"]
        b2 = feat_q["vec"]
        v2 = (feat_q["cov"] >= tau).astype(np.uint8)

        # Find best shift under ECC decoding
        best_failed_bits = 999
        best_rec = None
        best_shift = 0

        for s in range(16):
            sh_b2 = np.roll(b2, s * bps)
            sh_v2 = np.roll(v2, s * bps)
            m_s = (v1 > 0) & (sh_v2 > 0)
            noisy_cw = np.bitwise_xor(sh_b2, w)
            rec, is_comp, stats = decode_erasure_majority(noisy_cw, m_s.astype(np.uint8), k_bits, rep_factor, "interleaved", inv=inv512)
            failed_count = sum(1 for j in range(k_bits) if rec[j] != secret[j])
            if failed_count < best_failed_bits:
                best_failed_bits = failed_count
                best_rec = rec
                best_shift = s
                if failed_count == 0:
                    break

        # Record secret bit failures
        for j in range(k_bits):
            if best_rec[j] != secret[j]:
                secret_bit_fails[j] += 1
            if best_rec[j] is None:
                secret_bit_ties[j] += 1

        # Use best aligned query for spatial cell-level error tracking
        sh_b2 = np.roll(b2, best_shift * bps)
        sh_v2 = np.roll(v2, best_shift * bps)
        m_s = (v1 > 0) & (sh_v2 > 0)
        raw_diff = (b1 != sh_b2)

        for p_idx in range(512):
            cell_idx = p_idx // 4
            sec = cell_idx // 8
            ring = cell_idx % 8
            ch = p_idx % 4

            cell_eval_counts[sec, ring] += 1.0
            channel_eval_counts[ch] += 1.0

            if m_s[p_idx] > 0:
                cell_valid_counts[sec, ring] += 1.0
                channel_valid_counts[ch] += 1.0
                if raw_diff[p_idx]:
                    cell_error_counts[sec, ring] += 1.0
                    channel_error_counts[ch] += 1.0

    # Calculate spatial failure rates
    cell_erasure_rate = 1.0 - (cell_valid_counts / np.maximum(cell_eval_counts, 1.0))
    cell_error_rate = cell_error_counts / np.maximum(cell_valid_counts, 1.0)

    ring_erasure_rate = np.mean(cell_erasure_rate, axis=0) * 100.0
    ring_error_rate = np.mean(cell_error_rate, axis=0) * 100.0

    sector_erasure_rate = np.mean(cell_erasure_rate, axis=1) * 100.0
    sector_error_rate = np.mean(cell_error_rate, axis=1) * 100.0

    channel_names = ["Energy Bit", "Gray-b2 (MSB)", "Gray-b1", "Gray-b0 (LSB)"]
    channel_error_rate = (channel_error_counts / np.maximum(channel_valid_counts, 1.0)) * 100.0
    channel_erasure_rate = (1.0 - (channel_valid_counts / np.maximum(channel_eval_counts, 1.0))) * 100.0

    print("\n" + "=" * 90)
    print("                    EXPERIMENT 6A: SPATIAL & STRUCTURAL FAILURE ANALYSIS")
    print("=" * 90)

    print("\n--- 1. Radial Ring Failure Profile (Inner Ring 0 -> Outer Ring 7) ---")
    print(f"{'Ring':<6} {'Radius':<15} {'Mean Valid Overlap':<22} {'Erasure Rate':<16} {'Raw Error Rate':<16}")
    print("-" * 75)
    for r in range(8):
        print(f"Ring {r:<2} {f'r_{r}':<15} {f'{100.0 - ring_erasure_rate[r]:.1f}%':<22} {f'{ring_erasure_rate[r]:.1f}%':<16} {f'{ring_error_rate[r]:.2f}%':<16}")

    print("\n--- 2. Feature Channel Failure Profile ---")
    print(f"{'Channel':<16} {'Description':<18} {'Erasure Rate':<16} {'Raw Error Rate':<16}")
    print("-" * 68)
    for c in range(4):
        print(f"Channel {c:<8} {channel_names[c]:<18} {f'{channel_erasure_rate[c]:.1f}%':<16} {f'{channel_error_rate[c]:.2f}%':<16}")

    print("\n--- 3. Angular Sector Profile (Sectors 0 to 15, 22.5 deg steps) ---")
    for s in range(16):
        ang = s * 22.5
        print(f"Sector {s:02d} ({ang:5.1f} deg): Error={sector_error_rate[s]:.2f}%, Erasure={sector_erasure_rate[s]:.1f}%")

    print("\n--- 4. Secret Bit Failure Uniformity ---")
    print(f"Mean failed bit frequency: {np.mean(secret_bit_fails):.1f}% +- {np.std(secret_bit_fails):.1f}%")
    print(f"Range: min = {min(secret_bit_fails)}% (Bit {np.argmin(secret_bit_fails)}), max = {max(secret_bit_fails)}% (Bit {np.argmax(secret_bit_fails)})")

    # Generate 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: 2D Spatial Error Heatmap (Sectors x Rings)
    im1 = axes[0, 0].imshow(cell_error_rate.T * 100.0, aspect="auto", cmap="YlOrRd", origin="lower")
    axes[0, 0].set_title("Genuine Error Rate (%) across Polar Cells (16 Sectors x 8 Rings)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Angular Sector (0 to 15)", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Radial Ring (0=Inner to 7=Outer)", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 0], label="Error Rate (%)")

    # Panel 2: 2D Spatial Erasure Heatmap (Sectors x Rings)
    im2 = axes[0, 1].imshow(cell_erasure_rate.T * 100.0, aspect="auto", cmap="Blues", origin="lower")
    axes[0, 1].set_title("Mask Erasure Rate (%) across Polar Cells (Background / Clipping)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Angular Sector (0 to 15)", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("Radial Ring (0=Inner to 7=Outer)", fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=axes[0, 1], label="Erasure Rate (%)")

    # Panel 3: Radial Ring Profiles (Error vs Erasure)
    r_idx = np.arange(8)
    axes[1, 0].plot(r_idx, ring_error_rate, marker="o", color="#c0392b", linewidth=2.5, label="Raw Error Rate (%)")
    axes[1, 0].set_ylabel("Error Rate (%)", color="#c0392b", fontsize=10, fontweight="bold")
    axes[1, 0].tick_params(axis='y', labelcolor="#c0392b")
    axes[1, 0].set_xlabel("Radial Ring (0=Inner to 7=Outer)", fontsize=10, fontweight="bold")
    axes[1, 0].set_title("Radial Gradient: Error Rate vs. Erasure Rate", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(r_idx)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    ax3_twin = axes[1, 0].twinx()
    ax3_twin.plot(r_idx, ring_erasure_rate, marker="s", color="#2980b9", linewidth=2.5, linestyle="--", label="Erasure Rate (%)")
    ax3_twin.set_ylabel("Erasure Rate (%)", color="#2980b9", fontsize=10, fontweight="bold")
    ax3_twin.tick_params(axis='y', labelcolor="#2980b9")

    # Panel 4: Per-Secret-Bit Failure Frequency
    s_idx = np.arange(k_bits)
    axes[1, 1].bar(s_idx, secret_bit_fails, color="#8e44ad", edgecolor="black", alpha=0.85)
    axes[1, 1].axhline(np.mean(secret_bit_fails), color="red", linestyle="--", label=f"Mean Failure Rate ({np.mean(secret_bit_fails):.1f}%)")
    axes[1, 1].set_xlabel("Secret Bit Index (0 to 31)", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Failure Frequency (%)", fontsize=10, fontweight="bold")
    axes[1, 1].set_title("Secret Bit Failure Frequency across 100 Genuine Queries", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(np.arange(0, 32, 2))
    axes[1, 1].set_ylim(0, 30)
    axes[1, 1].legend(loc="upper right", frameon=True)
    axes[1, 1].grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(BASE_DIR, "results", "figures", "step6a_spatial_failure_analysis.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=200)
    print(f"\n[+] Step 6A diagnostic figure saved to: {fig_path}")

    # Save summary data
    out_json = os.path.join(BASE_DIR, "results", "step6a_spatial_failure_results.json")
    summary = {
        "ring_error_rate_pct": ring_error_rate.tolist(),
        "ring_erasure_rate_pct": ring_erasure_rate.tolist(),
        "sector_error_rate_pct": sector_error_rate.tolist(),
        "sector_erasure_rate_pct": sector_erasure_rate.tolist(),
        "channel_error_rate_pct": channel_error_rate.tolist(),
        "channel_erasure_rate_pct": channel_erasure_rate.tolist(),
        "secret_bit_failures_pct": secret_bit_fails.tolist(),
        "secret_bit_ties_pct": secret_bit_ties.tolist(),
        "mean_secret_bit_failure_pct": float(np.mean(secret_bit_fails)),
        "std_secret_bit_failure_pct": float(np.std(secret_bit_fails)),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Step 6A failure statistics saved to: {out_json}")

if __name__ == "__main__":
    main()

