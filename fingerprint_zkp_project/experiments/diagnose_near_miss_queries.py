"""
Diagnostic Analysis of Genuine Near-Miss Queries (Direction 1).
Investigates the 47 genuine queries that produced candidate secrets but failed exact recovery.

Evaluates:
1. Error Count Distribution (queries failing by exactly 1, 2, 3, 4, 5+ bits).
2. Bit Position Error Heatmap (which of the 32 secret bits fail most often?).
3. Alignment Precision: Did ciphertext-guided alignment select the optimal shift s*?
   Comparison between ciphertext-guided shift and ground-truth template distance shift.
4. Repetition Block & Spatial Analysis: Which physical cells/rings/channels map to the failing bits?
"""

import os
import sys
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from authentication.enrollment import get_public_interleaver
from authentication.authenticate import decode_fuzzy_secret
from experiments.evaluate_step8_channel_ablation import DEV_WEIGHTS_3CH

def main():
    bench_path = os.path.join(BASE_DIR, "data", "fixed_pairs_benchmark.json")
    m11_path = os.path.join(BASE_DIR, "results", "milestone11_integrated_system_results.json")

    with open(bench_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    with open(m11_path, "r", encoding="utf-8") as f:
        m11_data = json.load(f)

    # Re-evaluate with exact secret comparison
    from experiments.evaluate_step7_radial_sampling import extract_step7_features
    from authentication.enrollment import extract_384_template

    perm, inv = get_public_interleaver(20261112)
    weights_384 = np.tile(DEV_WEIGHTS_3CH, 128)

    # Cache feature vectors for the 100 genuine pairs
    print("[*] Loading genuine pair feature vectors for near-miss diagnosis...")
    genuine_pairs = benchmark["genuine_pairs"]

    error_counts = []
    bit_fail_counts = np.zeros(32, dtype=np.int32)
    replica_margin_summary = []
    alignment_comparison = []

    for item in genuine_pairs:
        p1 = item["path_1"]
        p2 = item["path_2"]
        pid = item["pair_id"]
        sid = item["subject_id"]

        t1, m1 = extract_384_template(p1)
        t2, m2 = extract_384_template(p2)

        # Deterministic secret for this identity
        import hashlib
        seed = int(hashlib.sha256(p1.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        secret = rng.randint(0, 2, 32, dtype=np.uint8)

        cw = np.tile(secret, 12)[perm]
        W = np.bitwise_xor(t1, cw)

        # 1. Ground truth biometric optimal shift (using plain templates)
        bps = 24
        best_bio_s = 0
        best_bio_errs = 999.0
        for s in range(16):
            sh_t2 = np.roll(t2, s * bps)
            sh_m2 = np.roll(m2, s * bps)
            mut = (m1 > 0) & (sh_m2 > 0)
            if np.mean(mut) >= 0.50:
                diff = np.sum((t1[mut] != sh_t2[mut]) * weights_384[mut]) / np.sum(weights_384[mut])
                if diff < best_bio_errs:
                    best_bio_errs = diff
                    best_bio_s = s

        # 2. Ciphertext-guided alignment & decoding
        rec_S, opt_s, n_val, overlap, fail_reason = decode_fuzzy_secret(
            t2, m2, W, m1, inv, weights_384
        )

        alignment_comparison.append({
            "pair_id": pid,
            "subject_id": sid,
            "bio_shift": best_bio_s,
            "ciphertext_shift": opt_s,
            "shift_matched": bool(best_bio_s == opt_s),
            "recovered": rec_S is not None
        })

        if rec_S is None:
            # Complete failure (ties or erasures)
            error_counts.append(32)
        else:
            diffs = (rec_S != secret)
            n_errs = int(np.sum(diffs))
            error_counts.append(n_errs)
            bit_fail_counts += diffs.astype(np.int32)

    error_counts = np.array(error_counts)
    
    print("\n" + "=" * 90)
    print("                     DIRECTION 1: NEAR-MISS DIAGNOSTIC RESULTS")
    print("=" * 90)

    # 1. Error Count Distribution
    hist_exact = int(np.sum(error_counts == 0))
    hist_1 = int(np.sum(error_counts == 1))
    hist_2 = int(np.sum(error_counts == 2))
    hist_3 = int(np.sum(error_counts == 3))
    hist_4 = int(np.sum(error_counts == 4))
    hist_5_8 = int(np.sum((error_counts >= 5) & (error_counts <= 8)))
    hist_gt_8 = int(np.sum(error_counts > 8))

    print(f"Total Genuine Queries Evaluated: {len(error_counts)}")
    print(f"  Exact Recovery (0 bit errors)  : {hist_exact:>3} ({hist_exact/len(error_counts)*100.0:.1f}%) [SUCCESS]")
    print(f"  Near-Miss by EXACTLY 1 bit     : {hist_1:>3} ({hist_1/len(error_counts)*100.0:.1f}%)")
    print(f"  Near-Miss by EXACTLY 2 bits    : {hist_2:>3} ({hist_2/len(error_counts)*100.0:.1f}%)")
    print(f"  Near-Miss by EXACTLY 3 bits    : {hist_3:>3} ({hist_3/len(error_counts)*100.0:.1f}%)")
    print(f"  Near-Miss by EXACTLY 4 bits    : {hist_4:>3} ({hist_4/len(error_counts)*100.0:.1f}%)")
    print(f"  Moderate errors (5 to 8 bits)  : {hist_5_8:>3} ({hist_5_8/len(error_counts)*100.0:.1f}%)")
    print(f"  Severe errors (> 8 bits)       : {hist_gt_8:>3} ({hist_gt_8/len(error_counts)*100.0:.1f}%)")
    print("-" * 90)
    
    # Cumulative near misses
    cum_1 = hist_exact + hist_1
    cum_2 = cum_1 + hist_2
    cum_3 = cum_2 + hist_3
    cum_4 = cum_3 + hist_4
    print("Cumulative Recovery if Outer Correction is Permitted:")
    print(f"  Full Recovery (t = 0)         : {hist_exact}%")
    print(f"  If correcting t <= 1 bit error : {cum_1}% (bridges +{hist_1} genuine users!)")
    print(f"  If correcting t <= 2 bit errors: {cum_2}% (bridges +{hist_1 + hist_2} genuine users!)")
    print(f"  If correcting t <= 3 bit errors: {cum_3}% (bridges +{hist_1 + hist_2 + hist_3} genuine users!)")
    print(f"  If correcting t <= 4 bit errors: {cum_4}% (bridges +{hist_1 + hist_2 + hist_3 + hist_4} genuine users!)")
    print("=" * 90)

    # 2. Alignment Diagnosis
    shift_matches = sum(1 for a in alignment_comparison if a["shift_matched"])
    shift_mismatches = len(alignment_comparison) - shift_matches
    print("\nAlignment Precision Analysis:")
    print(f"  Ciphertext-Guided Shift matched Ground-Truth: {shift_matches} / {len(alignment_comparison)} ({shift_matches}%)")
    print(f"  Alignment Mismatches:                         {shift_mismatches} / {len(alignment_comparison)}")
    
    if shift_mismatches > 0:
        print("  Mismatched Pairs:")
        for a in alignment_comparison:
            if not a["shift_matched"]:
                print(f"    Pair {a['pair_id']} (Subj {a['subject_id']}): Bio shift = {a['bio_shift']}, Ciphertext shift = {a['ciphertext_shift']}")

    # 3. Secret Bit Error Heatmap
    print("\nSecret Bit Error Frequencies (Across all 47 near-misses):")
    print("Bit Index: " + " ".join(f"{i:>2}" for i in range(16)))
    print("Errors:    " + " ".join(f"{bit_fail_counts[i]:>2}" for i in range(16)))
    print("-" * 60)
    print("Bit Index: " + " ".join(f"{i:>2}" for i in range(16, 32)))
    print("Errors:    " + " ".join(f"{bit_fail_counts[i]:>2}" for i in range(16, 32)))
    print(f"\nMean errors per bit position: {np.mean(bit_fail_counts):.2f} ± {np.std(bit_fail_counts):.2f}")
    print(f"Max errors on single bit position: Bit {np.argmax(bit_fail_counts)} ({np.max(bit_fail_counts)} errors)")
    print(f"Min errors on single bit position: Bit {np.argmin(bit_fail_counts)} ({np.min(bit_fail_counts)} errors)")

    # 4. Interleaver Mapping to Spatial Channels
    # Deconstruct where the 12 replicas of each bit live
    deint_pos = inv.reshape(12, 32)
    # Check ring distribution of each secret bit's replicas
    # In 384b: cell_idx = pos // 3; sector = cell // 8; ring = cell % 8; channel = pos % 3
    bit_channel_hist = np.zeros((32, 3), dtype=np.int32) # Gray-b2, b1, b0
    bit_ring_hist = np.zeros((32, 8), dtype=np.int32)
    for j in range(32):
        replicas = deint_pos[:, j]
        for p in replicas:
            ch = p % 3
            cell = p // 3
            ring = cell % 8
            bit_channel_hist[j, ch] += 1
            bit_ring_hist[j, ring] += 1

    print("\nPermutation Diversity Verification:")
    print(f"  Mean channels per bit replicas (out of 3): {np.mean([np.count_nonzero(bit_channel_hist[j]) for j in range(32)]):.2f}")
    print(f"  Mean rings per bit replicas (out of 8):    {np.mean([np.count_nonzero(bit_ring_hist[j]) for j in range(32)]):.2f}")
    print("=" * 90)

    # Save diagnostic results
    diag_output = os.path.join(BASE_DIR, "results", "near_miss_diagnostic_results.json")
    with open(diag_output, "w", encoding="utf-8") as f:
        json.dump({
            "error_distribution": {
                "exact_0": hist_exact,
                "fail_1": hist_1,
                "fail_2": hist_2,
                "fail_3": hist_3,
                "fail_4": hist_4,
                "fail_5_8": hist_5_8,
                "fail_gt_8": hist_gt_8
            },
            "cumulative_recovery": {
                "t_0": hist_exact,
                "t_1": cum_1,
                "t_2": cum_2,
                "t_3": cum_3,
                "t_4": cum_4
            },
            "bit_fail_counts": bit_fail_counts.tolist(),
            "alignment": {
                "matches": shift_matches,
                "mismatches": shift_mismatches,
                "details": alignment_comparison
            }
        }, f, indent=2)
    print(f"[+] Diagnostic data saved to {diag_output}")

if __name__ == "__main__":
    main()

