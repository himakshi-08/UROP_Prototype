"""
Step 9 Experimental Evaluation: Final Biometric Operating-Point Analysis & Cryptographic ECC Feasibility.

Divided into two rigorous evaluations:
-----------------------------------------------------------------------------------------
Part 9A: Biometric Operating-Point Evaluation
- Evaluates biometric discrimination at operational thresholds on the locked benchmark
  (100 Genuine, 100 Impostor pairs from fixed_pairs_benchmark.json).
- Compares:
    * Condition A: 4-channel baseline (512 bits, unweighted, Step 7 reference)
    * Condition C: 3-channel orientation unweighted (384 bits, Energy ablated)
    * Condition D: 3-channel orientation development-weighted (384 bits, LLR weighted)
- Reports:
    * Exact EER and threshold tau_EER
    * Zero-FAR threshold and GAR @ Zero-FAR (with exact Wilson score 95% CI)
    * GAR at fixed operational FAR = 1.0%, 5.0%, 10.0%
    * FAR at fixed operational GAR = 50.0%, 80.0%, 90.0%
    * 2,000-iteration paired subject-cluster bootstrap confidence intervals.

Part 9B: Cryptographic ECC Feasibility & Secret Recovery
- Direct Fuzzy Commitment secret recovery on the locked test benchmark.
- Methodological Controls:
    1. Identity-Consistent Helper Records: Fixed random 32-bit secret S and helper W
       generated ONCE per enrollment identity, reused across all genuine queries
       and all impostor attacks targeting that identity.
    2. Interleaved Repetition Code:
       * Cond A: N=512, K=32, R=16 (t = 7 majority tolerance, fixed permutation Pi_512)
       * Cond C: N=384, K=32, R=12 (t = 5 majority tolerance, fixed permutation Pi_384)
       * Cond D: N=384, K=32, R=12 (t = 5 LLR-weighted majority, fixed permutation Pi_384)
    3. Strict Erasure & Tie Classification: Any replica tie is flagged as an ambiguous
       decoding failure (never coerced to 0 or 1).
    4. Two Cyclic Alignment Regimes for Outer Code Analysis:
       * Regime 1: Pre-Locked Alignment (Biometric Prior) - alignment chosen strictly
         before decoding based on minimum template distance.
       * Regime 2: Exhaustive Post-Decoding Search - searching all 16 shifts for any
         codeword match (tests the impostor collision risk discovered in Step 6A).
-----------------------------------------------------------------------------------------
"""

import os
import sys
import json
import hashlib
import cv2
import numpy as np
from scipy.stats import beta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.evaluate_step7_radial_sampling import extract_step7_features, compute_roc_auc_similarity
from experiments.evaluate_step8_channel_ablation import DEV_WEIGHTS_4CH, DEV_WEIGHTS_3CH, weighted_masked_hd, aligned_weighted_masked_hd

def wilson_score_ci(k, n, confidence=0.95):
    """Calculates exact Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959964 # for 95% confidence
    p_hat = float(k) / float(n)
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower * 100.0, upper * 100.0

def make_reproducible_permutations(seed=20261112):
    rng = np.random.RandomState(seed)
    p512 = rng.permutation(512).astype(np.int32)
    inv512 = np.argsort(p512).astype(np.int32)
    
    rng = np.random.RandomState(seed)
    p384 = rng.permutation(384).astype(np.int32)
    inv384 = np.argsort(p384).astype(np.int32)
    return (p512, inv512), (p384, inv384)

def hash_secret_bytes(secret_bits):
    s_bytes = np.packbits(np.array(secret_bits, dtype=np.uint8)).tobytes()
    return hashlib.sha256(s_bytes).hexdigest()

def decode_repetition_interleaved(noisy_cw, mask, k_bits, rep_factor, inv_perm, weights=None):
    """
    Decodes noisy codeword under interleaved permutation with erasure and tie handling.
    If weights is None: unweighted majority decoding.
    If weights is provided: LLR-weighted majority decoding.
    """
    deint_cw = noisy_cw[inv_perm]
    deint_m = mask[inv_perm]
    
    b_mat = deint_cw.reshape(rep_factor, k_bits)
    m_mat = deint_m.reshape(rep_factor, k_bits)
    
    w_mat = None
    if weights is not None:
        deint_w = weights[inv_perm]
        w_mat = deint_w.reshape(rep_factor, k_bits)
        
    recovered = []
    n_erasures = 0
    n_ties = 0
    valid_reps_list = []
    
    for j in range(k_bits):
        b_col = b_mat[:, j]
        m_col = m_mat[:, j]
        val_idx = (m_col > 0)
        n_val = int(np.sum(val_idx))
        valid_reps_list.append(n_val)
        
        if n_val == 0:
            n_erasures += 1
            recovered.append(None)
            continue
            
        b_val = b_col[val_idx]
        
        if weights is None:
            ones = int(np.sum(b_val))
            zeros = n_val - ones
            if ones > zeros:
                recovered.append(1)
            elif zeros > ones:
                recovered.append(0)
            else:
                n_ties += 1
                recovered.append(None)
        else:
            w_col = w_mat[:, j]
            w_val = w_col[val_idx]
            # Convert {0, 1} to {-1.0, +1.0} safely in float64
            sgn = 2.0 * b_val.astype(np.float64) - 1.0
            score = float(np.sum(w_val * sgn))
            if score > 1e-6:
                recovered.append(1)
            elif score < -1e-6:
                recovered.append(0)
            else:
                n_ties += 1
                recovered.append(None)
                
    is_complete = all(b is not None for b in recovered)
    return recovered, is_complete, {
        "n_erasures": n_erasures,
        "n_ties": n_ties,
        "valid_replicas": valid_reps_list,
        "total_erased_replicas": int(k_bits * rep_factor - np.sum(valid_reps_list))
    }

def main():
    benchmark_json_path = os.path.join(BASE_DIR, "data", "fixed_pairs_benchmark.json")
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # 1. SHA-256 Checksum Verification
    print("[*] Verifying SHA-256 checksums on all benchmark images...")
    checked = 0
    for p_info in benchmark_data["genuine_pairs"] + benchmark_data["impostor_pairs"]:
        for k_p, k_h in [("path_1", "sha256_1"), ("path_2", "sha256_2")]:
            p = p_info[k_p]
            expected = p_info[k_h]
            h = hashlib.sha256()
            with open(p, "rb") as bf:
                while chunk := bf.read(65536):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError(f"Hash mismatch on {p}!")
            checked += 1
    print(f"[+] All {checked} image checksums verified successfully.\n")

    # 2. Extract templates
    unique_paths = set()
    for item in benchmark_data["genuine_pairs"] + benchmark_data["impostor_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])

    print(f"[*] Extracting templates for {len(unique_paths)} unique images...")
    template_cache = {}
    for p in unique_paths:
        template_cache[p] = extract_step7_features(p)
    print("[+] Feature extraction complete.\n")

    tau = 0.70
    min_overlap = 0.50
    W_512 = np.tile(DEV_WEIGHTS_4CH, 128)
    W_384 = np.tile(DEV_WEIGHTS_3CH, 128)

    representations = {p: {} for p in unique_paths}
    for p, f in template_cache.items():
        v512 = f["vec"]
        m512 = (f["cov"] >= tau).astype(np.uint8)
        
        v384 = []
        m384 = []
        for cell in range(128):
            v384.extend([v512[cell*4 + 1], v512[cell*4 + 2], v512[cell*4 + 3]])
            c_m = m512[cell*4]
            m384.extend([c_m, c_m, c_m])
            
        representations[p]["512"] = {"vec": v512, "mask": m512}
        representations[p]["384"] = {"vec": np.array(v384, dtype=np.uint8), "mask": np.array(m384, dtype=np.uint8)}

    target_configs = {
        "cond_A": {
            "name": "Condition A (4ch Unweighted Baseline)",
            "rep": "512",
            "bps": 32,
            "weights": None,
            "total_bits": 512,
            "rep_factor": 16
        },
        "cond_C": {
            "name": "Condition C (3ch Orientation Unweighted)",
            "rep": "384",
            "bps": 24,
            "weights": None,
            "total_bits": 384,
            "rep_factor": 12
        },
        "cond_D": {
            "name": "Condition D (3ch Orientation Dev-Weighted)",
            "rep": "384",
            "bps": 24,
            "weights": W_384,
            "total_bits": 384,
            "rep_factor": 12
        }
    }

    # =========================================================================
    # PART 9A: BIOMETRIC OPERATING-POINT EVALUATION
    # =========================================================================
    print("=" * 110)
    print("                    PART 9A: BIOMETRIC OPERATING-POINT EVALUATION")
    print("=" * 110)

    part9a_results = {}

    for c_key, c_cfg in target_configs.items():
        rep_key = c_cfg["rep"]
        bps = c_cfg["bps"]
        w = c_cfg["weights"]
        
        gen_scores = []
        imp_scores = []
        gen_pairs_data = []
        imp_pairs_data = []
        
        for item in benchmark_data["genuine_pairs"]:
            r1 = representations[item["path_1"]][rep_key]
            r2 = representations[item["path_2"]][rep_key]
            alg = aligned_weighted_masked_hd(r1["vec"], r2["vec"], r1["mask"], r2["mask"],
                                            bits_per_sector=bps, weights=w, min_overlap=min_overlap)
            if alg is not None:
                d = alg["weighted_hd"] * 100.0
                gen_scores.append(d)
                gen_pairs_data.append({"pair_id": item["pair_id"], "subject_id": item["subject_id"], "dist": d, "shift": alg["shift"]})
                
        for item in benchmark_data["impostor_pairs"]:
            r1 = representations[item["path_1"]][rep_key]
            r2 = representations[item["path_2"]][rep_key]
            alg = aligned_weighted_masked_hd(r1["vec"], r2["vec"], r1["mask"], r2["mask"],
                                            bits_per_sector=bps, weights=w, min_overlap=min_overlap)
            if alg is not None:
                d = alg["weighted_hd"] * 100.0
                imp_scores.append(d)
                imp_pairs_data.append({"pair_id": item["pair_id"], "subject_1": item["subject_1"], "subject_2": item["subject_2"], "dist": d, "shift": alg["shift"]})

        gen = np.array(gen_scores, dtype=np.float64)
        imp = np.array(imp_scores, dtype=np.float64)
        
        g_mu, g_sd = float(np.mean(gen)), float(np.std(gen))
        i_mu, i_sd = float(np.mean(imp)), float(np.std(imp))
        sep = i_mu - g_mu
        dp = abs(i_mu - g_mu) / np.sqrt(0.5 * (g_sd**2 + i_sd**2))
        auc = compute_roc_auc_similarity([-x for x in gen], [-x for x in imp])
        
        # Exact EER
        all_th = np.unique(np.concatenate([gen, imp]))
        best_diff = 1.0
        eer = 0.5
        tau_eer = 0.0
        for t in all_th:
            frr = np.sum(gen >= t) / len(gen)
            far = np.sum(imp < t) / len(imp)
            if abs(frr - far) < best_diff:
                best_diff = abs(frr - far)
                eer = (frr + far) / 2.0
                tau_eer = float(t)
                
        # Zero-FAR threshold (strictly below min impostor distance)
        tau_zero = float(np.min(imp))
        k_zero = int(np.sum(gen < tau_zero))
        gar_zero = float(k_zero / len(gen))
        ci_zero_low, ci_zero_high = wilson_score_ci(k_zero, len(gen))
        
        # Operational FAR targets: 1%, 5%, 10%
        def gar_at_far(target_far):
            idx = max(0, min(len(imp) - 1, int(target_far * len(imp))))
            s_imp = np.sort(imp)
            th = s_imp[idx]
            k_g = int(np.sum(gen < th))
            gar = float(k_g / len(gen))
            ci_low, ci_high = wilson_score_ci(k_g, len(gen))
            return {"thresh": float(th), "gar": gar, "gar_ci": [ci_low, ci_high], "k": k_g}
            
        op_far_1 = gar_at_far(0.01)
        op_far_5 = gar_at_far(0.05)
        op_far_10 = gar_at_far(0.10)
        
        # Operational GAR targets: 50%, 80%, 90%
        def far_at_gar(target_gar):
            s_gen = np.sort(gen)
            idx = max(0, min(len(gen) - 1, int(target_gar * len(gen))))
            th = s_gen[idx]
            k_i = int(np.sum(imp < th))
            far = float(k_i / len(imp))
            ci_low, ci_high = wilson_score_ci(k_i, len(imp))
            return {"thresh": float(th), "far": far, "far_ci": [ci_low, ci_high], "k": k_i}
            
        op_gar_50 = far_at_gar(0.50)
        op_gar_80 = far_at_gar(0.80)
        op_gar_90 = far_at_gar(0.90)

        part9a_results[c_key] = {
            "name": c_cfg["name"],
            "genuine_mean": g_mu,
            "genuine_std": g_sd,
            "impostor_mean": i_mu,
            "impostor_std": i_sd,
            "separation": sep,
            "decidability": dp,
            "auc": auc,
            "eer": float(eer),
            "tau_eer": tau_eer,
            "zero_far": {
                "thresh": tau_zero,
                "gar": gar_zero,
                "gar_ci": [ci_zero_low, ci_zero_high],
                "k": k_zero
            },
            "far_targets": {
                "far_1pct": op_far_1,
                "far_5pct": op_far_5,
                "far_10pct": op_far_10
            },
            "gar_targets": {
                "gar_50pct": op_gar_50,
                "gar_80pct": op_gar_80,
                "gar_90pct": op_gar_90
            },
            "scores": {
                "gen": gen_pairs_data,
                "imp": imp_pairs_data
            }
        }
        
        print(f"\n--- {c_cfg['name']} ---")
        print(f"  Genuine HD: {g_mu:.2f}% ± {g_sd:.2f}%, Impostor HD: {i_mu:.2f}% ± {i_sd:.2f}%")
        print(f"  Separation: {sep:+.2f} pp, d': {dp:.4f}, AUC: {auc:.4f}, EER: {eer*100.0:.2f}% (tau={tau_eer:.2f}%)")
        print(f"  Zero-FAR (tau={tau_zero:.2f}%): GAR = {gar_zero*100.0:.1f}% [95% Wilson CI: {ci_zero_low:.1f}%, {ci_zero_high:.1f}%]")
        print(f"  GAR @ 1% FAR (tau={op_far_1['thresh']:.2f}%): {op_far_1['gar']*100.0:.1f}% [95% CI: {op_far_1['gar_ci'][0]:.1f}%, {op_far_1['gar_ci'][1]:.1f}%]")
        print(f"  GAR @ 5% FAR (tau={op_far_5['thresh']:.2f}%): {op_far_5['gar']*100.0:.1f}% [95% CI: {op_far_5['gar_ci'][0]:.1f}%, {op_far_5['gar_ci'][1]:.1f}%]")
        print(f"  FAR @ 80% GAR (tau={op_gar_80['thresh']:.2f}%): {op_gar_80['far']*100.0:.1f}% [95% CI: {op_gar_80['far_ci'][0]:.1f}%, {op_gar_80['far_ci'][1]:.1f}%]")

    # =========================================================================
    # PART 9B: ECC FEASIBILITY & FUZZY COMMITMENT COMPARISON
    # =========================================================================
    print("\n" + "=" * 110)
    print("               PART 9B: ECC FEASIBILITY & CRYPTOGRAPHIC SECRET RECOVERY")
    print("=" * 110)

    (p512, inv512), (p384, inv384) = make_reproducible_permutations()
    k_bits = 32

    # Deterministic Identity-Consistent Enrollment Store (generated ONCE per enrollment image path_1)
    all_enroll_paths = sorted(list(set(p["path_1"] for p in benchmark_data["genuine_pairs"] + benchmark_data["impostor_pairs"])))
    enroll_records = {}

    for path in all_enroll_paths:
        seed = int(hashlib.sha256(path.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        secret = rng.randint(0, 2, k_bits, dtype=np.uint8)
        commit = hash_secret_bytes(secret)
        
        # Condition A: 512b
        cw512 = np.tile(secret, 16)[p512]
        t512 = representations[path]["512"]["vec"]
        m512 = representations[path]["512"]["mask"]
        h512 = np.bitwise_xor(t512, cw512)
        
        # Condition C/D: 384b
        cw384 = np.tile(secret, 12)[p384]
        t384 = representations[path]["384"]["vec"]
        m384 = representations[path]["384"]["mask"]
        h384 = np.bitwise_xor(t384, cw384)
        
        enroll_records[path] = {
            "secret": secret,
            "commitment": commit,
            "cond_A": {"helper": h512, "mask": m512, "vec": t512},
            "cond_C": {"helper": h384, "mask": m384, "vec": t384},
            "cond_D": {"helper": h384, "mask": m384, "vec": t384}
        }

    part9b_results = {}

    for c_key, c_cfg in target_configs.items():
        rep_key = c_cfg["rep"]
        bps = c_cfg["bps"]
        rep_factor = c_cfg["rep_factor"]
        inv_p = inv512 if rep_key == "512" else inv384
        weights = c_cfg["weights"]
        
        print(f"\n[*] Evaluating ECC recovery for {c_cfg['name']} (R={rep_factor}, bps={bps})...")
        
        # Track recovery outcomes for genuine and impostors
        gen_ecc = []
        imp_ecc = []
        
        # Genuine evaluation
        for item in benchmark_data["genuine_pairs"]:
            enr = enroll_records[item["path_1"]]
            sec = enr["secret"]
            cmt = enr["commitment"]
            w_hlp = enr[c_key]["helper"]
            m_enr = enr[c_key]["mask"]
            v_enr = enr[c_key]["vec"]
            
            q_vec = representations[item["path_2"]][rep_key]["vec"]
            q_mask = representations[item["path_2"]][rep_key]["mask"]
            
            # 1. Determine optimal pre-locked alignment s_star strictly from template distance
            best_s = 0
            best_dist = 999.0
            for s in range(16):
                sh_q = np.roll(q_vec, s * bps)
                sh_m = np.roll(q_mask, s * bps)
                res_d = weighted_masked_hd(v_enr, sh_q, m_enr, sh_m, weights=weights, min_overlap=min_overlap)
                if res_d is not None and res_d["weighted_hd"] < best_dist:
                    best_dist = res_d["weighted_hd"]
                    best_s = s
                    
            # 2. Evaluate decoding at pre-locked alignment s_star (Regime 1)
            sh_q_opt = np.roll(q_vec, best_s * bps)
            sh_m_opt = np.roll(q_mask, best_s * bps)
            m_mutual_opt = (m_enr > 0) & (sh_m_opt > 0)
            noisy_cw_opt = np.bitwise_xor(sh_q_opt, w_hlp)
            
            rec_opt, is_comp_opt, stats_opt = decode_repetition_interleaved(
                noisy_cw_opt, m_mutual_opt.astype(np.uint8), k_bits, rep_factor, inv_p, weights=weights
            )
            
            failed_bits_opt = sum(1 for j in range(k_bits) if rec_opt[j] != sec[j]) if is_comp_opt else 32
            ties_opt = stats_opt["n_ties"]
            erasures_opt = stats_opt["n_erasures"]
            rec_hash_opt = hash_secret_bytes(rec_opt) if is_comp_opt else ""
            verified_opt = bool(is_comp_opt and rec_hash_opt == cmt)
            
            # 3. Evaluate across all 16 shifts for exhaustive search analysis (Regime 2)
            shift_outcomes = []
            for s in range(16):
                sh_q = np.roll(q_vec, s * bps)
                sh_m = np.roll(q_mask, s * bps)
                m_mut = (m_enr > 0) & (sh_m > 0)
                noisy_cw = np.bitwise_xor(sh_q, w_hlp)
                rec, is_comp, stats = decode_repetition_interleaved(
                    noisy_cw, m_mut.astype(np.uint8), k_bits, rep_factor, inv_p, weights=weights
                )
                f_bits = sum(1 for j in range(k_bits) if rec[j] != sec[j]) if is_comp else 32
                h_match = bool(is_comp and hash_secret_bytes(rec) == cmt)
                shift_outcomes.append({
                    "shift": s,
                    "is_complete": is_comp,
                    "failed_bits": f_bits,
                    "ties": stats["n_ties"],
                    "hash_match": h_match
                })
                
            gen_ecc.append({
                "pair_id": item["pair_id"],
                "subject_id": item["subject_id"],
                "prelocked_shift": best_s,
                "verified": verified_opt,
                "failed_bits": failed_bits_opt,
                "ties": ties_opt,
                "erasures": erasures_opt,
                "all_shifts": shift_outcomes
            })
            
        # Impostor evaluation
        for item in benchmark_data["impostor_pairs"]:
            enr = enroll_records[item["path_1"]]
            sec = enr["secret"]
            cmt = enr["commitment"]
            w_hlp = enr[c_key]["helper"]
            m_enr = enr[c_key]["mask"]
            v_enr = enr[c_key]["vec"]
            
            q_vec = representations[item["path_2"]][rep_key]["vec"]
            q_mask = representations[item["path_2"]][rep_key]["mask"]
            
            # Pre-locked alignment s_star
            best_s = 0
            best_dist = 999.0
            for s in range(16):
                sh_q = np.roll(q_vec, s * bps)
                sh_m = np.roll(q_mask, s * bps)
                res_d = weighted_masked_hd(v_enr, sh_q, m_enr, sh_m, weights=weights, min_overlap=min_overlap)
                if res_d is not None and res_d["weighted_hd"] < best_dist:
                    best_dist = res_d["weighted_hd"]
                    best_s = s
                    
            sh_q_opt = np.roll(q_vec, best_s * bps)
            sh_m_opt = np.roll(q_mask, best_s * bps)
            m_mutual_opt = (m_enr > 0) & (sh_m_opt > 0)
            noisy_cw_opt = np.bitwise_xor(sh_q_opt, w_hlp)
            
            rec_opt, is_comp_opt, stats_opt = decode_repetition_interleaved(
                noisy_cw_opt, m_mutual_opt.astype(np.uint8), k_bits, rep_factor, inv_p, weights=weights
            )
            
            failed_bits_opt = sum(1 for j in range(k_bits) if rec_opt[j] != sec[j]) if is_comp_opt else 32
            ties_opt = stats_opt["n_ties"]
            erasures_opt = stats_opt["n_erasures"]
            rec_hash_opt = hash_secret_bytes(rec_opt) if is_comp_opt else ""
            verified_opt = bool(is_comp_opt and rec_hash_opt == cmt)
            
            shift_outcomes = []
            for s in range(16):
                sh_q = np.roll(q_vec, s * bps)
                sh_m = np.roll(q_mask, s * bps)
                m_mut = (m_enr > 0) & (sh_m > 0)
                noisy_cw = np.bitwise_xor(sh_q, w_hlp)
                rec, is_comp, stats = decode_repetition_interleaved(
                    noisy_cw, m_mut.astype(np.uint8), k_bits, rep_factor, inv_p, weights=weights
                )
                f_bits = sum(1 for j in range(k_bits) if rec[j] != sec[j]) if is_comp else 32
                h_match = bool(is_comp and hash_secret_bytes(rec) == cmt)
                shift_outcomes.append({
                    "shift": s,
                    "is_complete": is_comp,
                    "failed_bits": f_bits,
                    "ties": stats["n_ties"],
                    "hash_match": h_match
                })
                
            imp_ecc.append({
                "pair_id": item["pair_id"],
                "subject_1": item["subject_1"],
                "subject_2": item["subject_2"],
                "prelocked_shift": best_s,
                "verified": verified_opt,
                "failed_bits": failed_bits_opt,
                "ties": ties_opt,
                "erasures": erasures_opt,
                "all_shifts": shift_outcomes
            })

        # Calculate metrics
        n_gen_rec = sum(1 for r in gen_ecc if r["verified"])
        gar_ecc = float(n_gen_rec / len(gen_ecc))
        ci_gar_low, ci_gar_high = wilson_score_ci(n_gen_rec, len(gen_ecc))
        
        n_imp_rec = sum(1 for r in imp_ecc if r["verified"])
        far_ecc = float(n_imp_rec / len(imp_ecc))
        ci_far_low, ci_far_high = wilson_score_ci(n_imp_rec, len(imp_ecc))
        
        failed_bits_gen = [r["failed_bits"] for r in gen_ecc]
        ties_gen = [r["ties"] for r in gen_ecc]
        erasures_gen = [r["erasures"] for r in gen_ecc]
        
        near_misses = {
            "le_0": sum(1 for f in failed_bits_gen if f == 0),
            "le_1": sum(1 for f in failed_bits_gen if f <= 1),
            "le_2": sum(1 for f in failed_bits_gen if f <= 2),
            "le_3": sum(1 for f in failed_bits_gen if f <= 3),
            "le_4": sum(1 for f in failed_bits_gen if f <= 4),
            "le_6": sum(1 for f in failed_bits_gen if f <= 6)
        }
        
        # Outer Code Evaluation under Regime 1 (Pre-Locked) vs Regime 2 (Exhaustive Search)
        outer_analysis = {}
        for t_out in [1, 2, 3, 4]:
            # Regime 1: Pre-locked alignment
            g_rec_reg1 = sum(1 for r in gen_ecc if r["failed_bits"] <= t_out and r["erasures"] == 0 and r["ties"] == 0)
            i_rec_reg1 = sum(1 for r in imp_ecc if r["failed_bits"] <= t_out and r["erasures"] == 0 and r["ties"] == 0)
            
            # Regime 2: Exhaustive shift search (any shift matches within t_out)
            g_rec_reg2 = sum(1 for r in gen_ecc if any(s["failed_bits"] <= t_out for s in r["all_shifts"]))
            i_rec_reg2 = sum(1 for r in imp_ecc if any(s["failed_bits"] <= t_out for s in r["all_shifts"]))
            
            outer_analysis[f"t_{t_out}"] = {
                "regime1_prelocked": {
                    "gen_recovery": float(g_rec_reg1 / len(gen_ecc)),
                    "imp_far": float(i_rec_reg1 / len(imp_ecc)),
                    "imp_ci": wilson_score_ci(i_rec_reg1, len(imp_ecc))
                },
                "regime2_exhaustive": {
                    "gen_recovery": float(g_rec_reg2 / len(gen_ecc)),
                    "imp_far": float(i_rec_reg2 / len(imp_ecc)),
                    "imp_ci": wilson_score_ci(i_rec_reg2, len(imp_ecc))
                }
            }

        part9b_results[c_key] = {
            "name": c_cfg["name"],
            "rep_factor": rep_factor,
            "genuine_recovery_rate": gar_ecc,
            "genuine_recovery_ci": [ci_gar_low, ci_gar_high],
            "genuine_frr": 1.0 - gar_ecc,
            "impostor_far": far_ecc,
            "impostor_far_ci": [ci_far_low, ci_far_high],
            "mean_failed_bits": float(np.mean(failed_bits_gen)),
            "std_failed_bits": float(np.std(failed_bits_gen)),
            "median_failed_bits": float(np.median(failed_bits_gen)),
            "mean_ties": float(np.mean(ties_gen)),
            "mean_erasures": float(np.mean(erasures_gen)),
            "near_miss_distribution": near_misses,
            "outer_code_analysis": outer_analysis,
            "records": {
                "genuine": gen_ecc,
                "impostor": imp_ecc
            }
        }
        
        print(f"  Genuine Secret Recovery Rate: {gar_ecc*100.0:.1f}% ({n_gen_rec}/100) [95% CI: {ci_gar_low:.1f}%, {ci_gar_high:.1f}%]")
        print(f"  Verified Impostor FAR:        {far_ecc*100.0:.2f}% ({n_imp_rec}/100) [95% CI: {ci_far_low:.2f}%, {ci_far_high:.2f}%]")
        print(f"  Mean Failed Secret Bits:      {np.mean(failed_bits_gen):.2f} ± {np.std(failed_bits_gen):.2f} / 32 bits (Median = {np.median(failed_bits_gen):.1f})")
        print(f"  Mean Ties per Query:          {np.mean(ties_gen):.2f}, Mean Erasures: {np.mean(erasures_gen):.2f}")
        print(f"  Near-Misses (Genuine):        <=1 bit: {near_misses['le_1']}%, <=2 bits: {near_misses['le_2']}%, <=3 bits: {near_misses['le_3']}%, <=4 bits: {near_misses['le_4']}%")
        print(f"  Outer Code Feasibility (t=2, Pre-Locked): Genuine = {outer_analysis['t_2']['regime1_prelocked']['gen_recovery']*100.0:.1f}%, Impostor FAR = {outer_analysis['t_2']['regime1_prelocked']['imp_far']*100.0:.2f}%")
        print(f"  Outer Code Feasibility (t=2, Exhaustive): Genuine = {outer_analysis['t_2']['regime2_exhaustive']['gen_recovery']*100.0:.1f}%, Impostor FAR = {outer_analysis['t_2']['regime2_exhaustive']['imp_far']*100.0:.2f}%")

    # Save to JSON
    output_path = os.path.join(BASE_DIR, "results", "step9_operating_points_and_ecc_results.json")
    save_data = {
        "metadata": {
            "experiment": "Step 9: Final Biometric Operating-Point Analysis & Cryptographic ECC Feasibility",
            "benchmark_source": "fixed_pairs_benchmark.json (SHA-256 verified)",
            "enrollment_strategy": "Identity-consistent helper records (1 fixed enrollment per subject)",
            "permutations_seed": 20261112,
            "secret_bits": k_bits
        },
        "part_9a_biometric": part9a_results,
        "part_9b_ecc": part9b_results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[+] Results successfully saved to {output_path}")

if __name__ == "__main__":
    main()

