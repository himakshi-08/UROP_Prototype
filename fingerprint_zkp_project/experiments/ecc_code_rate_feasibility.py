"""
ECC Code-Rate Feasibility Experiment: 32-bit vs 48-bit vs 64-bit Secret Length.
Strictly development-set only (fp_training, 30 disjoint subjects).
The locked benchmark (fixed_pairs_benchmark.json) is NOT touched.

Configurations under test:
  Cfg-32: K=32, R=12 (N=384), majority tolerance t=5 (existing production)
  Cfg-48: K=48, R= 8 (N=384), majority tolerance t=3 (intermediate)
  Cfg-64: K=64, R= 6 (N=384), majority tolerance t=2 (experiment)

Feature representation: Fixed at 384-bit annular orientation (3-channel Gray).
Alignment metric: M1 (normalized conviction, top-1 shift only).
Credential K: Independent from biometric secret S (distinct security parameters).

Central question: Can 64-bit secret recovery achieve acceptable genuine
authentication on development data without substantially increasing impostor recovery?

Security note: 2^64 is not a 128-bit security level. The goal is to characterize
the genuine/impostor trade-off empirically before any security claims are made.
"""

import os
import sys
import glob
import json
import hashlib
import time
import numpy as np
from scipy.stats import beta as beta_dist

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from authentication.enrollment import extract_384_template, get_public_interleaver
from authentication.authenticate import DEV_WEIGHTS_3CH

# ------------------------------------------------------------------
# Configurations under test
# ------------------------------------------------------------------
CONFIGS = [
    {
        "label": "Cfg-32",
        "k_bits": 32,
        "rep_factor": 12,
        "majority_t": 5,   # tolerance = floor(R/2) - 0 = 5
        "note": "Existing production: N=384, K=32, R=12"
    },
    {
        "label": "Cfg-48",
        "k_bits": 48,
        "rep_factor": 8,
        "majority_t": 3,   # tolerance = floor(8/2) - 1 = 3
        "note": "Intermediate: N=384, K=48, R=8"
    },
    {
        "label": "Cfg-64",
        "k_bits": 64,
        "rep_factor": 6,
        "majority_t": 2,   # tolerance = floor(6/2) - 1 = 2
        "note": "Experiment: N=384, K=64, R=6"
    }
]

# Public permutation seed (frozen, same as production)
INTERLEAVER_SEED = 20261112


def wilson_ci(k, n):
    """Wilson score 95% confidence interval as (lo%, hi%)."""
    if n == 0:
        return 0.0, 100.0
    z = 1.959964
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return max(0.0, center-margin)*100, min(1.0, center+margin)*100


def make_interleaver(k_bits, rep_factor, seed):
    """
    Generate a reproducible public interleaver permutation for a given (K, R, seed).
    Returns (perm, inv_perm) each of length N = K * R.
    """
    n = k_bits * rep_factor
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n).astype(np.int32)
    inv = np.argsort(perm).astype(np.int32)
    return perm, inv


def m1_alignment_score(cand_cw, mut_m, inv_perm, weights_384, k_bits, rep_factor):
    """
    M1 normalized conviction alignment score.
    Returns sum_j [|score_j| / total_weight_j] across all k_bits bits.
    """
    n = k_bits * rep_factor
    deint_cw = cand_cw[inv_perm].reshape(rep_factor, k_bits)
    deint_m  = mut_m[inv_perm[:n]].reshape(rep_factor, k_bits)
    deint_w  = weights_384[inv_perm[:n]].reshape(rep_factor, k_bits)

    score = 0.0
    for j in range(k_bits):
        col_b = deint_cw[:, j]
        col_m = deint_m[:, j]
        col_w = deint_w[:, j]
        val = col_m > 0
        if np.any(val):
            sgn = 2.0 * col_b[val].astype(np.float64) - 1.0
            w_sum = float(np.sum(col_w[val]))
            if w_sum > 1e-9:
                score += abs(float(np.sum(col_w[val] * sgn))) / w_sum
    return score


def decode_at_shift(cand_cw, mut_m, inv_perm, weights_384, k_bits, rep_factor):
    """
    LLR-weighted majority decoding at a given (pre-aligned) shift.
    Returns (recovered_bits, n_erasures, n_ties, per_bit_scores).
    per_bit_scores: normalized conviction for each of k_bits bits.
    """
    n = k_bits * rep_factor
    deint_cw = cand_cw[inv_perm].reshape(rep_factor, k_bits)
    deint_m  = mut_m[inv_perm[:n]].reshape(rep_factor, k_bits)
    deint_w  = weights_384[inv_perm[:n]].reshape(rep_factor, k_bits)

    recovered = []
    n_erasures = 0
    n_ties = 0
    per_bit_scores = []

    for j in range(k_bits):
        col_b = deint_cw[:, j]
        col_m = deint_m[:, j]
        col_w = deint_w[:, j]
        val = col_m > 0

        if not np.any(val):
            n_erasures += 1
            recovered.append(0)
            per_bit_scores.append(0.0)
            continue

        sgn = 2.0 * col_b[val].astype(np.float64) - 1.0
        w_sum = float(np.sum(col_w[val]))
        raw = float(np.sum(col_w[val] * sgn))
        conviction = raw / w_sum if w_sum > 1e-9 else 0.0
        per_bit_scores.append(abs(conviction))

        if raw > 1e-6:
            recovered.append(1)
        elif raw < -1e-6:
            recovered.append(0)
        else:
            n_ties += 1
            recovered.append(0)

    return np.array(recovered, dtype=np.uint8), n_erasures, n_ties, per_bit_scores


def evaluate_pair(t1, m1_mask, t2, m2_mask, secret, perm, inv, weights_384,
                  k_bits, rep_factor, bps=24):
    """
    Encode, align with M1, and decode a single template pair.
    Returns a dict of per-pair diagnostics.
    """
    n = k_bits * rep_factor
    # Clip weights to N (in case N < 384 is ever used — here N=384 always)
    w = weights_384[:n]

    cw = np.tile(secret, rep_factor)[perm]
    W  = np.bitwise_xor(t1[:n], cw)

    # M1 shift search (top-1 only)
    best_s = 0
    best_score = -1e9
    shift_scores = []
    cand_cw_all = []
    mut_all = []

    for s in range(16):
        sh_q  = np.roll(t2, s * bps)[:n]
        sh_m  = np.roll(m2_mask, s * bps)[:n]
        mut   = (m1_mask[:n] > 0) & (sh_m > 0)
        overlap = float(np.mean(mut))

        if overlap < 0.50:
            shift_scores.append(-1e9)
            cand_cw_all.append(None)
            mut_all.append(mut)
            continue

        cand = np.bitwise_xor(W, sh_q)
        cand_cw_all.append(cand)
        mut_all.append(mut)

        sc = m1_alignment_score(cand, mut, inv, weights_384, k_bits, rep_factor)
        shift_scores.append(sc)
        if sc > best_score:
            best_score = sc
            best_s = s

    cand_opt = cand_cw_all[best_s]
    mut_opt  = mut_all[best_s]

    if cand_opt is None:
        return {
            "failed": True, "reason": "No valid overlap at any shift",
            "best_shift": best_s, "n_bit_errors": k_bits,
            "n_erasures": k_bits, "n_ties": 0,
            "per_bit_convictions": [], "overlap": 0.0
        }

    rec, n_eras, n_ties, pbs = decode_at_shift(
        cand_opt, mut_opt, inv, weights_384, k_bits, rep_factor
    )

    n_errs = int(np.sum(rec != secret))
    overlap_v = float(np.mean(mut_opt))

    # Per-replica block analysis (how many of R replicas agree per bit)
    deint_cw = cand_opt[inv].reshape(rep_factor, k_bits)
    deint_m  = mut_opt[inv[:n]].reshape(rep_factor, k_bits)
    block_agreements = []
    for j in range(k_bits):
        col_b = deint_cw[:, j]
        col_m = deint_m[:, j]
        val   = col_m > 0
        n_val = int(np.sum(val))
        if n_val > 0:
            agree = int(np.sum(col_b[val] == secret[j]))
            block_agreements.append(agree / n_val)
        else:
            block_agreements.append(0.0)

    return {
        "failed": False,
        "best_shift": best_s,
        "n_bit_errors": n_errs,
        "n_erasures": n_eras,
        "n_ties": n_ties,
        "exact_recovery": (n_errs == 0 and n_eras == 0 and n_ties == 0),
        "per_bit_convictions": pbs,
        "mean_conviction": float(np.mean(pbs)) if pbs else 0.0,
        "mean_block_agreement": float(np.mean(block_agreements)),
        "overlap": overlap_v
    }


def run_config(cfg, dev_genuine, dev_impostor, weights_384):
    label   = cfg["label"]
    k_bits  = cfg["k_bits"]
    rep     = cfg["rep_factor"]
    t_tol   = cfg["majority_t"]

    print(f"\n{'='*95}")
    print(f"  {label}: K={k_bits} bits, R={rep} replicas, majority t={t_tol}  |  {cfg['note']}")
    print(f"{'='*95}")

    perm, inv = make_interleaver(k_bits, rep, INTERLEAVER_SEED)

    t0_enc = time.perf_counter()

    # ---- Genuine Pairs ----
    gen_results = []
    for pair in dev_genuine:
        t1, m1 = pair["t1"], pair["m1"]
        t2, m2 = pair["t2"], pair["m2"]
        secret  = pair["secret"][:k_bits]  # First k_bits of a 64-bit master secret
        r = evaluate_pair(t1, m1, t2, m2, secret, perm, inv, weights_384, k_bits, rep)
        gen_results.append(r)

    # ---- Impostor Pairs ----
    imp_results = []
    for pair in dev_impostor:
        t1, m1  = pair["enroll_t"], pair["enroll_m"]
        t2, m2  = pair["query_t"], pair["query_m"]
        secret  = pair["victim_secret"][:k_bits]
        r = evaluate_pair(t1, m1, t2, m2, secret, perm, inv, weights_384, k_bits, rep)
        imp_results.append(r)

    t_elapsed = time.perf_counter() - t0_enc
    n_gen = len(gen_results)
    n_imp = len(imp_results)

    # ---- Statistics ----
    err_counts = np.array([r["n_bit_errors"] for r in gen_results], dtype=np.int32)

    exact_rec  = int(np.sum(err_counts == 0))
    rec_t1     = int(np.sum(err_counts <= 1))
    rec_t2     = int(np.sum(err_counts <= 2))
    rec_t3     = int(np.sum(err_counts <= 3))
    rec_t4     = int(np.sum(err_counts <= 4))
    n_erasures = int(np.sum([r["n_erasures"] > 0 for r in gen_results]))
    n_ties     = int(np.sum([r["n_ties"] > 0 for r in gen_results]))

    imp_exact  = int(np.sum([r["n_bit_errors"] == 0 for r in imp_results]))
    imp_t1     = int(np.sum([r["n_bit_errors"] <= 1 for r in imp_results]))
    imp_t2     = int(np.sum([r["n_bit_errors"] <= 2 for r in imp_results]))

    # ECC decoding latency estimate
    ms_per_pair = t_elapsed / (n_gen + n_imp) * 1000.0

    ci_gen = wilson_ci(exact_rec, n_gen)
    ci_imp = wilson_ci(imp_exact, n_imp)

    # Error distribution histogram
    hist = {
        "0":   int(np.sum(err_counts == 0)),
        "1":   int(np.sum(err_counts == 1)),
        "2":   int(np.sum(err_counts == 2)),
        "3":   int(np.sum(err_counts == 3)),
        "4":   int(np.sum(err_counts == 4)),
        "5_8": int(np.sum((err_counts >= 5) & (err_counts <= 8))),
        "gt8": int(np.sum(err_counts > 8))
    }

    print(f"  Genuine pairs: {n_gen}   |   Impostor pairs: {n_imp}")
    print(f"  ECC latency:   {ms_per_pair:.3f} ms/pair  (total {t_elapsed:.1f}s)")
    print()
    print(f"  GENUINE SECRET RECOVERY (development set, {n_gen} pairs):")
    print(f"    Exact (t=0): {exact_rec}/{n_gen} ({exact_rec/n_gen*100:.1f}%)  "
          f"[95% CI: {ci_gen[0]:.1f}%, {ci_gen[1]:.1f}%]")
    print(f"    Bit errors 0–1: {rec_t1}/{n_gen} ({rec_t1/n_gen*100:.1f}%)")
    print(f"    Bit errors 0–2: {rec_t2}/{n_gen} ({rec_t2/n_gen*100:.1f}%)")
    print(f"    Bit errors 0–3: {rec_t3}/{n_gen} ({rec_t3/n_gen*100:.1f}%)")
    print(f"    Bit errors 0–4: {rec_t4}/{n_gen} ({rec_t4/n_gen*100:.1f}%)")
    print(f"    Erasure events: {n_erasures}/{n_gen}  |  Tie events: {n_ties}/{n_gen}")
    print()
    print(f"  IMPOSTOR SECRET RECOVERY (development set, {n_imp} pairs):")
    print(f"    Exact (t=0): {imp_exact}/{n_imp} ({imp_exact/n_imp*100:.1f}%)  "
          f"[95% CI: {ci_imp[0]:.1f}%, {ci_imp[1]:.1f}%]")
    print(f"    Bit errors ≤1: {imp_t1}/{n_imp} ({imp_t1/n_imp*100:.1f}%)")
    print(f"    Bit errors ≤2: {imp_t2}/{n_imp} ({imp_t2/n_imp*100:.1f}%)")
    print()
    print(f"  ERROR DISTRIBUTION (genuine):  "
          f"0:{hist['0']}  1:{hist['1']}  2:{hist['2']}  "
          f"3:{hist['3']}  4:{hist['4']}  5-8:{hist['5_8']}  >8:{hist['gt8']}")

    # Security estimate (stolen K)
    # Work factor = 2^k_bits HKDF+ScalarMult ~= 2^k_bits x 62.3us
    hkdf_ms  = 0.0102   # from Direction 2 benchmark
    sc_ms    = 0.0523
    op_ms    = hkdf_ms + sc_ms
    cpu_hrs  = (2**k_bits) * op_ms / 1000.0 / 3600.0
    gpu_hrs  = cpu_hrs / 1000.0  # rough 1000-core GPU
    print(f"\n  OFFLINE GUESSING (stolen K, {k_bits}-bit secret S):")
    print(f"    Work factor:  2^{k_bits} = {2**k_bits:.2e} operations")
    print(f"    Single-core:  {cpu_hrs:.2e} CPU hours")
    print(f"    1000-core GPU:{gpu_hrs:.2e} GPU hours")
    if k_bits <= 48:
        print(f"    *** WARNING: Practically brute-forceable with modern hardware ***")
    elif k_bits < 64:
        print(f"    *** Marginally resistant on current GPU clusters ***")
    else:
        print(f"    *** Provides meaningful resistance against offline search ***")
    print(f"    Security note: 2^{k_bits} < 2^128 — this is NOT a 128-bit cryptographic guarantee.")

    return {
        "label": label,
        "k_bits": k_bits,
        "rep_factor": rep,
        "majority_t": t_tol,
        "n_genuine": n_gen,
        "n_impostor": n_imp,
        "genuine": {
            "exact_recovery": exact_rec,
            "exact_recovery_rate": exact_rec / n_gen,
            "exact_ci": ci_gen,
            "rec_t1": rec_t1,
            "rec_t2": rec_t2,
            "rec_t3": rec_t3,
            "rec_t4": rec_t4,
            "n_erasure_events": n_erasures,
            "n_tie_events": n_ties,
            "error_distribution": hist,
            "mean_conviction": float(np.mean([r["mean_conviction"] for r in gen_results]))
        },
        "impostor": {
            "exact_recovery": imp_exact,
            "exact_recovery_rate": imp_exact / n_imp,
            "exact_ci": ci_imp,
            "rec_t1": imp_t1,
            "rec_t2": imp_t2
        },
        "latency_ms_per_pair": ms_per_pair,
        "security": {
            "search_space_bits": k_bits,
            "cpu_hours_single_core": cpu_hrs,
            "gpu_hours_1000core": gpu_hrs,
            "note": f"2^{k_bits} is not a 128-bit security level"
        }
    }


def main():
    train_dir = r"C:\Users\sofia\Downloads\fp_training\fp_training"
    if not os.path.exists(train_dir):
        print(f"[!] Cannot find fp_training at: {train_dir}")
        return

    subjs = sorted([d for d in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, d))])[:30]
    print(f"[*] ECC Code-Rate Feasibility Experiment")
    print(f"[*] Development subjects: {subjs[0]} to {subjs[-1]}  ({len(subjs)} subjects)")
    print(f"[*] Feature representation: 384-bit annular orientation (fixed)")
    print(f"[*] Alignment metric: M1 normalized conviction, top-1 shift only")
    print(f"[*] Locked benchmark: NOT ACCESSED")

    weights_384 = np.tile(DEV_WEIGHTS_3CH, 128)  # channel weights tiled over 128 cells

    # ---- Load 150 templates ----
    print("\n[*] Extracting templates from 30 development subjects...")
    t0 = time.time()
    dev_templates = {}
    for sid in subjs:
        spath = os.path.join(train_dir, sid)
        files = sorted(glob.glob(os.path.join(spath, "*.[bB][mM][pP]")) +
                       glob.glob(os.path.join(spath, "*.[tT][iI][fF]")) +
                       glob.glob(os.path.join(spath, "*.[pP][nN][gG]")))[:5]
        tmpls = []
        for f in files:
            t, m = extract_384_template(f)
            if t is not None:
                tmpls.append({"path": f, "t": t, "m": m})
        if len(tmpls) >= 2:
            dev_templates[sid] = tmpls
    print(f"[+] {sum(len(v) for v in dev_templates.values())} templates loaded in {time.time()-t0:.1f}s")

    # ---- Genuine development pairs (all intra-subject pairs, up to 300) ----
    # We need 64-bit secrets as the "master" and slice per configuration
    rng_secret = np.random.RandomState(INTERLEAVER_SEED)

    genuine_pairs = []
    for sid, tmpls in dev_templates.items():
        n = len(tmpls)
        for i in range(n):
            for j in range(i+1, n):
                img1 = tmpls[i]
                img2 = tmpls[j]
                # Deterministic 64-bit secret for this enrollment image
                h = int(hashlib.sha256(img1["path"].encode()).hexdigest()[:16], 16)
                r = np.random.RandomState(h & 0xFFFFFFFF)
                secret64 = r.randint(0, 2, 64, dtype=np.uint8)
                genuine_pairs.append({
                    "subject_id": sid,
                    "t1": img1["t"], "m1": img1["m"],
                    "t2": img2["t"], "m2": img2["m"],
                    "secret": secret64
                })

    # Cap at 300 to keep runtime reasonable
    if len(genuine_pairs) > 300:
        rng_sub = np.random.RandomState(42)
        idx = rng_sub.choice(len(genuine_pairs), 300, replace=False)
        genuine_pairs = [genuine_pairs[i] for i in sorted(idx)]

    # ---- Impostor development pairs (300 cross-subject) ----
    sids = list(dev_templates.keys())
    rng_imp = np.random.RandomState(42)
    impostor_pairs = []
    while len(impostor_pairs) < 300:
        i, j = rng_imp.choice(len(sids), 2, replace=False)
        s_enroll = sids[i]
        s_query  = sids[j]
        idx_e = rng_imp.randint(len(dev_templates[s_enroll]))
        idx_q = rng_imp.randint(len(dev_templates[s_query]))
        t_enroll = dev_templates[s_enroll][idx_e]
        t_query  = dev_templates[s_query][idx_q]
        h = int(hashlib.sha256(t_enroll["path"].encode()).hexdigest()[:16], 16)
        r = np.random.RandomState(h & 0xFFFFFFFF)
        secret64 = r.randint(0, 2, 64, dtype=np.uint8)
        impostor_pairs.append({
            "enroll_t": t_enroll["t"], "enroll_m": t_enroll["m"],
            "query_t":  t_query["t"],  "query_m":  t_query["m"],
            "victim_secret": secret64
        })

    print(f"[+] {len(genuine_pairs)} genuine pairs, {len(impostor_pairs)} impostor pairs formed.")

    # ---- Run each configuration ----
    all_results = []
    for cfg in CONFIGS:
        result = run_config(cfg, genuine_pairs, impostor_pairs, weights_384)
        all_results.append(result)

    # ---- Cross-configuration summary table ----
    print("\n" + "=" * 95)
    print("  CROSS-CONFIGURATION SUMMARY (development set, fp_training, NOT the locked benchmark)")
    print("=" * 95)
    hdr = (f"{'Config':<10} {'K':>5} {'R':>4} {'t':>3} | "
           f"{'Gen (t=0)':>10} {'Gen CI':>16} | "
           f"{'Gen (≤2)':>9} | "
           f"{'Imp (t=0)':>10} {'Imp CI':>16} | "
           f"{'Imp (≤2)':>9} | {'Latency':>8}")
    print(hdr)
    print("-" * 120)
    for r in all_results:
        g = r["genuine"]
        im = r["impostor"]
        gr = g["exact_recovery_rate"] * 100
        ir = im["exact_recovery_rate"] * 100
        gr2= g["rec_t2"] / r["n_genuine"] * 100
        ir2= im["rec_t2"] / r["n_impostor"] * 100
        ci_g = f"[{g['exact_ci'][0]:.1f},{g['exact_ci'][1]:.1f}]"
        ci_i = f"[{im['exact_ci'][0]:.1f},{im['exact_ci'][1]:.1f}]"
        print(f"{r['label']:<10} {r['k_bits']:>5} {r['rep_factor']:>4} {r['majority_t']:>3} | "
              f"{gr:>9.1f}% {ci_g:>16} | "
              f"{gr2:>8.1f}% | "
              f"{ir:>9.1f}% {ci_i:>16} | "
              f"{ir2:>8.1f}% | {r['latency_ms_per_pair']:>7.3f}ms")
    print("=" * 95)
    print()
    print("  OFFLINE GUESSING COST (stolen credential K — distinct from biometric secret S):")
    print(f"  {'Config':<10} {'Secret bits':>12} {'CPU hours (1-core)':>20} {'GPU hours (1000-core)':>22} {'Note'}")
    print("-" * 90)
    for r in all_results:
        sec = r["security"]
        note = "***BRUTE-FORCEABLE***" if r["k_bits"] <= 48 else "Meaningful resistance"
        print(f"  {r['label']:<10} {r['k_bits']:>12} {sec['cpu_hours_single_core']:>20.2e} "
              f"{sec['gpu_hours_1000core']:>22.2e} {note}")
    print()
    print("  Security reminder: none of these configurations reach 2^128 security level.")
    print("=" * 95)

    # ---- Save JSON ----
    out_path = os.path.join(BASE_DIR, "results", "ecc_code_rate_feasibility_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "ECC Code-Rate Feasibility (32 vs 48 vs 64-bit secret, fp_training only)",
            "dataset": "fp_training (30 disjoint development subjects, NOT fp_testing)",
            "n_genuine": len(genuine_pairs),
            "n_impostor": len(impostor_pairs),
            "configurations": all_results
        }, f, indent=2)
    print(f"\n[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()

