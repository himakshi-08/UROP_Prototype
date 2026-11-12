"""
Direction 2 Security Evaluation: ISO/IEC 30136 Unlinkability & Template Irreversibility.
Evaluates:
1. ISO/IEC 30136 Unlinkability Metric:
   - Mated helper distribution: D(W1, W2) for same subject enrolled in two different services.
   - Non-mated helper distribution: D(W1, W3) for different subjects enrolled in different services.
   - Local linkability D_leftrightarrow(s) and global system linkability D_leftrightarrow^sys.
2. Helper Cross-Matching Attack:
   - Adversary observes W1 and W2, decodes Delta S = S1 xor S2 using repetition codebook.
   - Measures empirical recovery rate of cross-matching differential.
3. Irreversibility & Brute-Force Complexity Analysis:
   - Work-factor in scalar multiplications and HKDF evaluations under all threat models.
"""

import os
import sys
import glob
import json
import hashlib
import secrets
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from authentication.enrollment import extract_384_template, get_public_interleaver
from authentication.authenticate import DEV_WEIGHTS_3CH

def compute_iso30136_unlinkability(mated_scores, non_mated_scores, num_bins=100):
    """
    Computes ISO/IEC 30136 semi-mated / mated vs non-mated unlinkability metric.
    D_leftrightarrow(s) = [p(s|mated) - p(s|non_mated)] / [p(s|mated) + p(s|non_mated)]
    clipped to [0, 1] if p(s|mated) > p(s|non_mated), else 0.
    D_sys = integral of D_leftrightarrow(s) * p(s|mated) ds.
    """
    all_scores = np.concatenate([mated_scores, non_mated_scores])
    s_min, s_max = np.min(all_scores), np.max(all_scores)
    bins = np.linspace(s_min - 1e-5, s_max + 1e-5, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    p_mated, _ = np.histogram(mated_scores, bins=bins, density=True)
    p_non_mated, _ = np.histogram(non_mated_scores, bins=bins, density=True)

    # Avoid divide-by-zero
    denom = p_mated + p_non_mated
    valid = denom > 1e-9

    lr = np.zeros_like(bin_centers)
    lr[valid] = p_mated[valid] / denom[valid]

    # ISO/IEC 30136 local linkability metric D_lr(s) in [0, 1]
    # D_lr(s) = max(0, 2 * (p_mated / denom) - 1)
    d_lr = np.maximum(0.0, 2.0 * lr - 1.0)

    # Global system linkability metric: expected D_lr under mated distribution
    dbin = bins[1] - bins[0]
    d_sys = float(np.sum(d_lr[valid] * p_mated[valid] * dbin))
    d_sys = min(1.0, max(0.0, d_sys))

    return {
        "d_sys": d_sys,
        "mated_mean": float(np.mean(mated_scores)),
        "mated_std": float(np.std(mated_scores)),
        "non_mated_mean": float(np.mean(non_mated_scores)),
        "non_mated_std": float(np.std(non_mated_scores)),
        "bin_centers": bin_centers.tolist(),
        "p_mated": p_mated.tolist(),
        "p_non_mated": p_non_mated.tolist(),
        "d_lr": d_lr.tolist()
    }

def main():
    train_dir = r"C:\Users\sofia\Downloads\fp_training\fp_training"
    subjs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])[:30]
    print(f"[*] Direction 2 Security Evaluation on 30 Development Subjects: {subjs[0]} to {subjs[-1]}")

    perm, inv = get_public_interleaver(20261112)
    weights_384 = np.tile(DEV_WEIGHTS_3CH, 128)

    # 1. Feature Extraction: Load two distinct impressions per subject (Impression 1 and Impression 2)
    enrollments_svc1 = {} # Service A (Application 1)
    enrollments_svc2 = {} # Service B (Application 2)

    print("[*] Enrolling 30 subjects in two independent services (Service A & Service B)...")
    for sid in subjs:
        spath = os.path.join(train_dir, sid)
        img_files = sorted(glob.glob(os.path.join(spath, "*.[bB][mM][pP]")) +
                           glob.glob(os.path.join(spath, "*.[tT][iI][fF]")) +
                           glob.glob(os.path.join(spath, "*.[pP][nN][gG]")))[:2]
        if len(img_files) < 2:
            continue

        f1, f2 = img_files[0], img_files[1]
        t1, m1 = extract_384_template(f1)
        t2, m2 = extract_384_template(f2)

        if t1 is None or t2 is None:
            continue

        # Service A enrollment
        sec1 = np.random.randint(0, 2, 32, dtype=np.uint8)
        cred1 = secrets.token_bytes(32)
        salt1 = secrets.token_bytes(16)
        cw1 = np.tile(sec1, 12)[perm]
        W1 = np.bitwise_xor(t1, cw1)

        # Service B enrollment (Independent secret, independent credential, independent salt)
        sec2 = np.random.randint(0, 2, 32, dtype=np.uint8)
        cred2 = secrets.token_bytes(32)
        salt2 = secrets.token_bytes(16)
        cw2 = np.tile(sec2, 12)[perm]
        W2 = np.bitwise_xor(t2, cw2)

        enrollments_svc1[sid] = {
            "template": t1, "mask": m1, "W": W1, "secret": sec1, "cred": cred1, "salt": salt1
        }
        enrollments_svc2[sid] = {
            "template": t2, "mask": m2, "W": W2, "secret": sec2, "cred": cred2, "salt": salt2
        }

    valid_sids = [s for s in subjs if s in enrollments_svc1 and s in enrollments_svc2]
    print(f"[+] Successfully cross-enrolled {len(valid_sids)} subjects across two isolated services.")

    # 2. ISO/IEC 30136 Unlinkability Analysis
    # Metric: Cyclic-minimum normalized Hamming distance between helper data instances
    bps = 24
    def helper_min_hd(w_a, w_b):
        min_hd = 999.0
        for s in range(16):
            sh_b = np.roll(w_b, s * bps)
            hd = float(np.mean(w_a != sh_b))
            if hd < min_hd:
                min_hd = hd
        return min_hd

    mated_helper_scores = []
    # Mated: W1 and W2 from the SAME subject
    for sid in valid_sids:
        w1 = enrollments_svc1[sid]["W"]
        w2 = enrollments_svc2[sid]["W"]
        dist = helper_min_hd(w1, w2)
        # Similarity score: 1.0 - distance
        mated_helper_scores.append(1.0 - dist)

    # Non-mated: W1 from subject i and W2 from subject j (i != j)
    non_mated_helper_scores = []
    for i in range(len(valid_sids)):
        for j in range(len(valid_sids)):
            if i != j:
                s_i, s_j = valid_sids[i], valid_sids[j]
                w_i = enrollments_svc1[s_i]["W"]
                w_j = enrollments_svc2[s_j]["W"]
                dist = helper_min_hd(w_i, w_j)
                non_mated_helper_scores.append(1.0 - dist)

    mated_arr = np.array(mated_helper_scores)
    non_mated_arr = np.array(non_mated_helper_scores)

    unlink_res = compute_iso30136_unlinkability(mated_arr, non_mated_arr, num_bins=50)

    # 3. Helper Cross-Matching Differential Attack Simulation
    # Adversary observes W1 and W2, computes W_diff = W1 xor Roll(W2, s)
    # Then decodes candidate Delta S using repetition codebook
    print("\n[*] Simulating Cross-Matching Attack on Helper Data...")
    cross_match_successes = 0
    cross_match_total = len(valid_sids)

    for sid in valid_sids:
        rec1 = enrollments_svc1[sid]
        rec2 = enrollments_svc2[sid]
        w1 = rec1["W"]
        w2 = rec2["W"]
        true_delta_s = np.bitwise_xor(rec1["secret"], rec2["secret"])

        # Adversary searches across 16 shifts
        found_delta = False
        for s in range(16):
            sh_w2 = np.roll(w2, s * bps)
            w_diff = np.bitwise_xor(w1, sh_w2)

            # Majority decode candidate Delta S from w_diff
            deint_diff = w_diff[inv].reshape(12, 32)
            cand_delta = []
            tie = False
            for j in range(32):
                col = deint_diff[:, j]
                ones = int(np.sum(col))
                zeros = 12 - ones
                if ones > zeros:
                    cand_delta.append(1)
                elif zeros > ones:
                    cand_delta.append(0)
                else:
                    tie = True
                    break
            if not tie and np.array_equal(cand_delta, true_delta_s):
                found_delta = True
                break

        if found_delta:
            cross_match_successes += 1

    cross_match_rate = float(cross_match_successes / cross_match_total)

    # 4. Irreversibility & Work-Factor Breakdown
    # Timing benchmark for Ristretto255 scalar multiplication and HKDF
    from crypto.schnorr import Ristretto255Group
    from crypto.key_derivation import derive_witness
    group = Ristretto255Group()

    t_sc_start = time.perf_counter()
    n_ops = 1000
    for _ in range(n_ops):
        sc = group.random_scalar()
        _ = group.base_scalarmult(sc)
    t_sc_elapsed = time.perf_counter() - t_sc_start
    sec_per_sc = t_sc_elapsed / n_ops

    t_hkdf_start = time.perf_counter()
    for _ in range(n_ops):
        _ = derive_witness(secrets.token_bytes(4), secrets.token_bytes(32), secrets.token_bytes(16))
    t_hkdf_elapsed = time.perf_counter() - t_hkdf_start
    sec_per_hkdf = t_hkdf_elapsed / n_ops

    # Display Security Analysis
    print("\n" + "=" * 95)
    print("           DIRECTION 2: ISO/IEC 30136 UNLINKABILITY & IRREVERSIBILITY REPORT")
    print("=" * 95)
    print("1. ISO/IEC 30136 Unlinkability Analysis:")
    print(f"   Mated Helper Similarity Mean ± SD:     {unlink_res['mated_mean']:.4f} ± {unlink_res['mated_std']:.4f}")
    print(f"   Non-Mated Helper Similarity Mean ± SD: {unlink_res['non_mated_mean']:.4f} ± {unlink_res['non_mated_std']:.4f}")
    print(f"   System Unlinkability Score D_sys:      {unlink_res['d_sys']:.4f} (0.0 = Fully Unlinkable, 1.0 = Fully Linkable)")
    print(f"   Interpretation: Standard Fuzzy Commitment helper data retains residual template correlations.")
    print("-" * 95)
    print("2. Cross-Matching Differential Attack:")
    print(f"   Attacker Success in Decoding (S1 ^ S2): {cross_match_successes} / {cross_match_total} ({cross_match_rate*100.0:.1f}%)")
    print(f"   Finding: When two helper records of the same user are intercepted, an attacker who aligns them")
    print(f"   can successfully decode Delta S in {cross_match_rate*100.0:.1f}% of cases, leaking the template differential T1 ^ T2.")
    print("-" * 95)
    print("3. Irreversibility & Computational Work-Factor:")
    print(f"   Benchmarked Scalar Mult Latency: {sec_per_sc*1000.0:.4f} ms per Ristretto255 point multiplication")
    print(f"   Benchmarked HKDF-SHA256 Latency: {sec_per_hkdf*1000.0:.4f} ms per 2-factor derivation")
    print()
    print("   Threat Model A: External Adversary (Attacker observes Helper W, Public Key Y, Salt; lacks K):")
    print("     - Search Space: S (2^32) x K (2^256) = 2^288 candidate combinations.")
    print(f"     - Work Factor: 2^288 HKDF + Scalar Mult operations (~ 4.97 x 10^86 operations).")
    print("     - Security Level: Exceeds 128-bit cryptographic security (UNCONDITIONAL AGAINST EXTERNAL ATTACK).")
    print()
    print("   Threat Model B: Malicious Server / Stolen Credential (Attacker possesses K, W, Y, Salt):")
    print("     - Search Space: S in [0, 2^32 - 1] = 4,294,967,296 candidate biometric secrets.")
    print(f"     - Work Factor: 2^32 operations x ({sec_per_sc*1e6:.1f} us) ~= {2**32 * sec_per_sc / 3600.0:.2f} CPU hours (single core).")
    print(f"     - Security Level: 32 bits (Vulnerable to offline GPU dictionary search if credential K is compromised).")
    print("=" * 95)

    # Save to JSON
    out_file = os.path.join(BASE_DIR, "results", "direction2_security_iso30136_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "iso30136": {
                "d_sys": unlink_res["d_sys"],
                "mated_mean": unlink_res["mated_mean"],
                "mated_std": unlink_res["mated_std"],
                "non_mated_mean": unlink_res["non_mated_mean"],
                "non_mated_std": unlink_res["non_mated_std"]
            },
            "cross_matching_attack": {
                "successes": cross_match_successes,
                "total": cross_match_total,
                "success_rate": cross_match_rate
            },
            "work_factor": {
                "scalar_mult_sec": sec_per_sc,
                "hkdf_sec": sec_per_hkdf,
                "external_attacker_bits": 288,
                "compromised_k_attacker_cpu_hours": float(2**32 * sec_per_sc / 3600.0)
            }
        }, f, indent=2)
    print(f"\n[+] Security evaluation results saved to {out_file}")

if __name__ == "__main__":
    main()

