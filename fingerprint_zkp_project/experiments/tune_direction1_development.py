"""
Direction 1 Development-Set Optimization & Near-Miss Investigation.
Evaluates alignment metrics, error distributions, and ECC code-rate trade-offs
strictly on the 30 disjoint development subjects from fp_training.
Controls:
- 300 Genuine Development Pairs (30 subjects x 10 pairs).
- 300 Impostor Development Pairs (drawn across disjoint subjects).
- Never touches or snoops on fp_testing benchmark.
"""

import os
import sys
import glob
import json
import hashlib
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from authentication.enrollment import extract_384_template, get_public_interleaver
from authentication.authenticate import DEV_WEIGHTS_3CH
from experiments.evaluate_step8_channel_ablation import DEV_WEIGHTS_RING

def wilson_score_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = 1.959964
    p_hat = float(k) / float(n)
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower * 100.0, upper * 100.0

def evaluate_alignment_metrics(cand_cw, mut_m, deint_cw, deint_m, deint_w, ring_w_matrix, k_bits=32, rep_factor=12):
    """
    Computes candidate alignment scores for a single shift s.
    Returns dict of candidate scores: M0, M1, M2, M3, M4, M5.
    """
    m0_score = 0.0
    m1_score = 0.0
    m2_score = 0.0
    m3_score = 0.0
    m4_count = 0
    m5_score = 0.0

    for j in range(k_bits):
        col_b = deint_cw[:, j]
        col_m = deint_m[:, j]
        col_w = deint_w[:, j]
        col_rw = ring_w_matrix[:, j]
        val = (col_m > 0)

        if not np.any(val):
            continue

        sgn = 2.0 * col_b[val].astype(np.float64) - 1.0
        w_val = col_w[val]
        rw_val = col_rw[val]

        # Raw weighted polarity sum
        raw_sum = float(np.sum(w_val * sgn))
        tot_w = float(np.sum(w_val))
        abs_sum = abs(raw_sum)

        # M0: Baseline unnormalized absolute sum
        m0_score += abs_sum

        # M1: Normalized conviction per bit: |sum| / tot_w in [0, 1]
        conviction = abs_sum / tot_w if tot_w > 1e-6 else 0.0
        m1_score += conviction

        # M2: Squared conviction: conviction^2 in [0, 1]
        m2_score += (conviction ** 2)

        # M3: Cubed conviction: conviction^3 in [0, 1]
        m3_score += (conviction ** 3)

        # M4: High-conviction bit indicator (agreement >= 60%)
        if conviction >= 0.60:
            m4_count += 1

        # M5: Ring-and-channel joint weighted conviction
        joint_w = w_val * rw_val
        joint_sum = abs(float(np.sum(joint_w * sgn)))
        joint_tot = float(np.sum(joint_w))
        if joint_tot > 1e-6:
            m5_score += (joint_sum / joint_tot) ** 2

    return {
        "m0": m0_score,
        "m1": m1_score,
        "m2": m2_score,
        "m3": m3_score,
        "m4": float(m4_count),
        "m5": m5_score
    }

def decode_at_shift(cand_cw, mut_m, inv_perm, weights_384, k_bits=32, rep_factor=12):
    """
    Decodes noisy codeword at specified shift using LLR-weighted majority decoding.
    Returns (candidate_secret, n_erasures, n_ties).
    """
    deint_cw = cand_cw[inv_perm].reshape(rep_factor, k_bits)
    deint_m = mut_m[inv_perm].reshape(rep_factor, k_bits)
    deint_w = weights_384[inv_perm].reshape(rep_factor, k_bits)

    recovered = []
    n_erasures = 0
    n_ties = 0

    for j in range(k_bits):
        col_b = deint_cw[:, j]
        col_m = deint_m[:, j]
        col_w = deint_w[:, j]
        val = (col_m > 0)

        if not np.any(val):
            n_erasures += 1
            recovered.append(0) # placeholder
            continue

        sgn = 2.0 * col_b[val].astype(np.float64) - 1.0
        sc = float(np.sum(col_w[val] * sgn))

        if sc > 1e-6:
            recovered.append(1)
        elif sc < -1e-6:
            recovered.append(0)
        else:
            n_ties += 1
            recovered.append(0)

    return np.array(recovered, dtype=np.uint8), n_erasures, n_ties

def main():
    train_dir = os.path.join(os.path.expanduser("~"), "Downloads", "fp_training", "fp_training")
    if not os.path.exists(train_dir):
        train_dir = r"C:\Users\sofia\Downloads\fp_training\fp_training"
    if not os.path.exists(train_dir):
        print(f"[!] Directory not found: {train_dir}")
        return

    subjs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])[:30]
    print(f"[*] Processing 30 disjoint development subjects from {train_dir}: {subjs[0]} to {subjs[-1]}")

    perm, inv = get_public_interleaver(20261112)
    weights_384 = np.tile(DEV_WEIGHTS_3CH, 128)

    # Precompute ring weight matrix for M5
    # In 384b template: cell = pos // 3; ring = cell % 8
    cell_rings = np.array([(p // 3) % 8 for p in range(384)], dtype=np.int32)
    ring_weights_384 = DEV_WEIGHTS_RING[cell_rings]
    ring_w_deint = ring_weights_384[inv].reshape(12, 32)

    # 1. Feature Extraction & Caching for all 150 images
    print("[*] Extracting 384-bit annular templates for 30 development subjects...")
    dev_templates = {}
    total_imgs = 0
    t0 = time.time()
    for sid in subjs:
        spath = os.path.join(train_dir, sid)
        img_files = sorted(glob.glob(os.path.join(spath, "*.[bB][mM][pP]")) +
                           glob.glob(os.path.join(spath, "*.[tT][iI][fF]")) +
                           glob.glob(os.path.join(spath, "*.[pP][nN][gG]")))[:5]
        tmpls = []
        for f in img_files:
            t, m = extract_384_template(f)
            if t is not None:
                tmpls.append({
                    "path": f,
                    "filename": os.path.basename(f),
                    "template": t,
                    "mask": m
                })
        dev_templates[sid] = tmpls
        total_imgs += len(tmpls)
    print(f"[+] Loaded {total_imgs} images across 30 subjects in {time.time()-t0:.1f}s.")

    # 2. Form 300 Genuine Development Pairs
    genuine_pairs = []
    for sid, tmpls in dev_templates.items():
        n = len(tmpls)
        for i in range(n):
            for j in range(i+1, n):
                genuine_pairs.append({
                    "subject_id": sid,
                    "img1": tmpls[i],
                    "img2": tmpls[j]
                })

    # 3. Form 300 Impostor Development Pairs
    impostor_pairs = []
    sids = list(dev_templates.keys())
    rng = np.random.RandomState(42)
    while len(impostor_pairs) < 300:
        i, j = rng.choice(len(sids), 2, replace=False)
        s1, s2 = sids[i], sids[j]
        idx1 = rng.randint(0, len(dev_templates[s1]))
        idx2 = rng.randint(0, len(dev_templates[s2]))
        impostor_pairs.append({
            "subject_1": s1,
            "subject_2": s2,
            "img1": dev_templates[s1][idx1],
            "img2": dev_templates[s2][idx2]
        })

    print(f"[+] Formed {len(genuine_pairs)} Genuine and {len(impostor_pairs)} Impostor Development Pairs.")

    # Helper: deterministic secret generation per enrollment image
    def get_secret(path_str):
        seed = int(hashlib.sha256(path_str.encode()).hexdigest()[:8], 16)
        r = np.random.RandomState(seed)
        return r.randint(0, 2, 32, dtype=np.uint8)

    # 4. Evaluation of Alignment Metrics on Genuine Pairs
    bps = 24
    metric_keys = ["m0", "m1", "m2", "m3", "m4", "m5"]
    alignment_correct = {m: 0 for m in metric_keys}
    exact_recovery = {m: 0 for m in metric_keys}
    error_counts_by_metric = {m: [] for m in metric_keys}
    top2_recovery = {m: 0 for m in metric_keys}

    # Tracking spatial failures for M2
    spatial_bit_errors = np.zeros(32, dtype=np.int32)
    replica_error_by_ring = np.zeros(8, dtype=np.int32)
    replica_total_by_ring = np.zeros(8, dtype=np.int32)
    replica_error_by_ch = np.zeros(3, dtype=np.int32)
    replica_total_by_ch = np.zeros(3, dtype=np.int32)

    print("\n[*] Evaluating Alignment Metrics across 300 Genuine Development Pairs...")
    for pair in genuine_pairs:
        t1, m1 = pair["img1"]["template"], pair["img1"]["mask"]
        t2, m2 = pair["img2"]["template"], pair["img2"]["mask"]
        secret = get_secret(pair["img1"]["path"])

        cw = np.tile(secret, 12)[perm]
        W = np.bitwise_xor(t1, cw)

        # Ground truth biometric optimal shift
        best_bio_s = 0
        best_bio_dist = 999.0
        for s in range(16):
            sh_t2 = np.roll(t2, s * bps)
            sh_m2 = np.roll(m2, s * bps)
            mut = (m1 > 0) & (sh_m2 > 0)
            if np.mean(mut) >= 0.50:
                dist = np.sum((t1[mut] != sh_t2[mut]) * weights_384[mut]) / np.sum(weights_384[mut])
                if dist < best_bio_dist:
                    best_bio_dist = dist
                    best_bio_s = s

        # Compute scores across all 16 shifts for each metric
        shift_scores = {m: np.full(16, -1e9, dtype=np.float64) for m in metric_keys}
        shift_cand_cw = []
        shift_mut_m = []

        for s in range(16):
            sh_q = np.roll(t2, s * bps)
            sh_m = np.roll(m2, s * bps)
            mut_m = (m1 > 0) & (sh_m > 0)
            shift_mut_m.append(mut_m)

            if np.mean(mut_m) < 0.50:
                shift_cand_cw.append(None)
                continue

            cand_cw = np.bitwise_xor(W, sh_q)
            shift_cand_cw.append(cand_cw)

            deint_cw = cand_cw[inv].reshape(12, 32)
            deint_m = mut_m[inv].reshape(12, 32)
            deint_w = weights_384[inv].reshape(12, 32)

            scs = evaluate_alignment_metrics(cand_cw, mut_m, deint_cw, deint_m, deint_w, ring_w_deint)
            for m in metric_keys:
                shift_scores[m][s] = scs[m]

        # For each metric, evaluate alignment match and recovery
        for m in metric_keys:
            order = np.argsort(shift_scores[m])[::-1]
            opt_s = int(order[0])
            cand_cw = shift_cand_cw[opt_s]
            mut_m = shift_mut_m[opt_s]

            is_aligned = (opt_s == best_bio_s)
            if is_aligned:
                alignment_correct[m] += 1

            if cand_cw is not None:
                rec_S, n_eras, n_ties = decode_at_shift(cand_cw, mut_m, inv, weights_384)
                if n_eras == 0 and n_ties == 0:
                    n_errs = int(np.sum(rec_S != secret))
                    error_counts_by_metric[m].append(n_errs)
                    if n_errs == 0:
                        exact_recovery[m] += 1
                        top2_recovery[m] += 1
                    else:
                        # Test top-2: if shift 2 succeeds, could top-2 help?
                        s2 = int(order[1])
                        cw2 = shift_cand_cw[s2]
                        m2_mut = shift_mut_m[s2]
                        if cw2 is not None:
                            rec_S2, e2, t2_cnt = decode_at_shift(cw2, m2_mut, inv, weights_384)
                            if e2 == 0 and t2_cnt == 0 and np.sum(rec_S2 != secret) == 0:
                                top2_recovery[m] += 1
                else:
                    error_counts_by_metric[m].append(32)
            else:
                error_counts_by_metric[m].append(32)

            # Record detailed spatial diagnostics for M2
            if m == "m2" and is_aligned and cand_cw is not None:
                rec_S, _, _ = decode_at_shift(cand_cw, mut_m, inv, weights_384)
                diffs = (rec_S != secret)
                spatial_bit_errors += diffs.astype(np.int32)
                # Analyze inverted replicas in decoded blocks
                deint_cw = cand_cw[inv].reshape(12, 32)
                deint_m = mut_m[inv].reshape(12, 32)
                deint_p = inv.reshape(12, 32)
                for j in range(32):
                    sec_bit = secret[j]
                    for r_idx in range(12):
                        if deint_m[r_idx, j] > 0:
                            pos = deint_p[r_idx, j]
                            ch = pos % 3
                            cell = pos // 3
                            ring = cell % 8
                            replica_total_by_ring[ring] += 1
                            replica_total_by_ch[ch] += 1
                            if deint_cw[r_idx, j] != sec_bit:
                                replica_error_by_ring[ring] += 1
                                replica_error_by_ch[ch] += 1

    # 5. Evaluate Impostor Collisions on Development Set (300 Impostor Pairs)
    print("[*] Evaluating Impostor Collision Rates across 300 Development Impostor Pairs...")
    imp_collisions_exact = {m: 0 for m in metric_keys}
    imp_collisions_t1 = {m: 0 for m in metric_keys}
    imp_collisions_t2 = {m: 0 for m in metric_keys}

    for pair in impostor_pairs:
        t1, m1 = pair["img1"]["template"], pair["img1"]["mask"]
        t2, m2 = pair["img2"]["template"], pair["img2"]["mask"]
        victim_secret = get_secret(pair["img1"]["path"])

        cw = np.tile(victim_secret, 12)[perm]
        W = np.bitwise_xor(t1, cw)

        # Evaluate candidate shifts
        shift_scores = {m: np.full(16, -1e9, dtype=np.float64) for m in metric_keys}
        shift_cand_cw = []
        shift_mut_m = []

        for s in range(16):
            sh_q = np.roll(t2, s * bps)
            sh_m = np.roll(m2, s * bps)
            mut_m = (m1 > 0) & (sh_m > 0)
            shift_mut_m.append(mut_m)

            if np.mean(mut_m) < 0.50:
                shift_cand_cw.append(None)
                continue

            cand_cw = np.bitwise_xor(W, sh_q)
            shift_cand_cw.append(cand_cw)

            deint_cw = cand_cw[inv].reshape(12, 32)
            deint_m = mut_m[inv].reshape(12, 32)
            deint_w = weights_384[inv].reshape(12, 32)

            scs = evaluate_alignment_metrics(cand_cw, mut_m, deint_cw, deint_m, deint_w, ring_w_deint)
            for m in metric_keys:
                shift_scores[m][s] = scs[m]

        for m in metric_keys:
            opt_s = int(np.argmax(shift_scores[m]))
            cand_cw = shift_cand_cw[opt_s]
            mut_m = shift_mut_m[opt_s]

            if cand_cw is not None:
                rec_S, n_eras, n_ties = decode_at_shift(cand_cw, mut_m, inv, weights_384)
                if n_eras == 0 and n_ties == 0:
                    n_errs = int(np.sum(rec_S != victim_secret))
                    if n_errs == 0:
                        imp_collisions_exact[m] += 1
                    if n_errs <= 1:
                        imp_collisions_t1[m] += 1
                    if n_errs <= 2:
                        imp_collisions_t2[m] += 1

    # 6. Display Findings
    print("\n" + "=" * 105)
    print("           DIRECTION 1 DEVELOPMENT-SET RESULTS: ALIGNMENT & NEAR-MISS OPTIMIZATION")
    print("=" * 105)
    n_gen = len(genuine_pairs)
    n_imp = len(impostor_pairs)

    metric_names = {
        "m0": "M0: Baseline (Unnormalized Sum of Replica Weights)",
        "m1": "M1: Normalized Conviction (|sum| / total_weight)",
        "m2": "M2: Squared Conviction ((|sum| / total_weight)^2)",
        "m3": "M3: Cubed Conviction ((|sum| / total_weight)^3)",
        "m4": "M4: High-Conviction Count (bits with >= 60% agreement)",
        "m5": "M5: Joint Ring x Channel Squared Conviction"
    }

    print(f"{'Alignment Metric':<55} | {'Align Acc':<10} | {'Exact Rec (t=0)':<16} | {'t<=1 Rec':<10} | {'t<=2 Rec':<10} | {'Imp Coll (t=0)':<14}")
    print("-" * 125)

    summary_records = {}
    for m in metric_keys:
        acc = alignment_correct[m] / n_gen * 100.0
        rec0 = exact_recovery[m] / n_gen * 100.0
        err_arr = np.array(error_counts_by_metric[m])
        rec1 = np.sum(err_arr <= 1) / n_gen * 100.0
        rec2 = np.sum(err_arr <= 2) / n_gen * 100.0
        coll0 = imp_collisions_exact[m] / n_imp * 100.0
        coll1 = imp_collisions_t1[m] / n_imp * 100.0
        coll2 = imp_collisions_t2[m] / n_imp * 100.0

        summary_records[m] = {
            "name": metric_names[m],
            "alignment_accuracy": acc,
            "exact_recovery": rec0,
            "recovery_t1": rec1,
            "recovery_t2": rec2,
            "top2_recovery": top2_recovery[m] / n_gen * 100.0,
            "impostor_collision_t0": coll0,
            "impostor_collision_t1": coll1,
            "impostor_collision_t2": coll2,
            "error_distribution": {
                "0": int(np.sum(err_arr == 0)),
                "1": int(np.sum(err_arr == 1)),
                "2": int(np.sum(err_arr == 2)),
                "3": int(np.sum(err_arr == 3)),
                "4": int(np.sum(err_arr == 4)),
                "5_8": int(np.sum((err_arr >= 5) & (err_arr <= 8))),
                "gt_8": int(np.sum(err_arr > 8))
            }
        }
        print(f"{metric_names[m]:<55} | {acc:>8.1f}% | {rec0:>14.1f}% | {rec1:>8.1f}% | {rec2:>8.1f}% | {coll0:>12.1f}%")

    print("=" * 125)

    # Spatial Error Distribution across Rings and Channels
    print("\nPhysical Failure Distribution across Concentric Rings (Aligned Genuine Pairs under M2):")
    print(f"{'Ring':<6} | {'Radius Interval':<22} | {'Bit Error Rate':<16} | {'Total Tested Replicas':<22}")
    print("-" * 72)
    for r in range(8):
        err = replica_error_by_ring[r]
        tot = replica_total_by_ring[r]
        rate = (err / tot * 100.0) if tot > 0 else 0.0
        print(f"Ring {r} | [{0.18 + r*0.1025:.2f} r_max, {0.18 + (r+1)*0.1025:.2f} r_max] | {rate:>14.2f}% | {tot:>20}")

    print("\nPhysical Failure Distribution across Orientation Channels:")
    print(f"{'Channel':<15} | {'Bit Error Rate':<16} | {'Total Tested Replicas':<22}")
    print("-" * 60)
    ch_names = ["Gray-b2 (MSB)", "Gray-b1 (MID)", "Gray-b0 (LSB)"]
    for c in range(3):
        err = replica_error_by_ch[c]
        tot = replica_total_by_ch[c]
        rate = (err / tot * 100.0) if tot > 0 else 0.0
        print(f"{ch_names[c]:<15} | {rate:>14.2f}% | {tot:>20}")
    print("=" * 105)

    # Save development results
    out_path = os.path.join(BASE_DIR, "results", "direction1_development_optimization_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "fp_training (30 disjoint development subjects)",
            "n_genuine": n_gen,
            "n_impostor": n_imp,
            "metrics": summary_records,
            "spatial_ring_analysis": {
                f"ring_{r}": {
                    "error_rate": float(replica_error_by_ring[r] / replica_total_by_ring[r]) if replica_total_by_ring[r] > 0 else 0.0,
                    "errors": int(replica_error_by_ring[r]),
                    "total": int(replica_total_by_ring[r])
                } for r in range(8)
            },
            "channel_analysis": {
                ch_names[c]: {
                    "error_rate": float(replica_error_by_ch[c] / replica_total_by_ch[c]) if replica_total_by_ch[c] > 0 else 0.0,
                    "errors": int(replica_error_by_ch[c]),
                    "total": int(replica_total_by_ch[c])
                } for c in range(3)
            }
        }, f, indent=2)
    print(f"\n[+] Results saved to {out_path}")

if __name__ == "__main__":
    main()

