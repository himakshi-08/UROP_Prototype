"""
Step 8 Experimental Evaluation: Controlled Channel Ablation & Reliability Weighting.
Factorial 2x2 design evaluating:
1. Feature Channel Selection: Full 4-channel (512 bits) vs. Orientation-only 3-channel (384 bits).
2. Weighting Strategy: Uniform unweighted vs. Development-trained LLR weights (from fp_training).

Experimental Conditions:
- Condition A: Full 4-channel (Energy + 3-bit Gray), Unweighted (512 bits, bps=32) [Step 7 Baseline]
- Condition B: Full 4-channel, Development-Trained LLR Weights (512 bits, bps=32)
- Condition C: Orientation-Only 3-channel (3-bit Gray), Unweighted (384 bits, bps=24)
- Condition D: Orientation-Only 3-channel, Development-Trained LLR Weights (384 bits, bps=24)
- Condition E (Exploratory): Orientation-Only 3-channel, Joint Ring x Channel Development Weights (384 bits, bps=24)

Methodological Controls:
- All weights frozen strictly on disjoint fp_training development subjects (no test data leakage).
- Evaluated on exact, immutable locked benchmark (100 genuine, 100 impostor pairs from fixed_pairs_benchmark.json).
- SHA-256 verification of all input images.
- Subject-cluster bootstrap resampling (2,000 iterations) for paired delta-AUC and 95% CIs.
"""

import os
import sys
import json
import hashlib
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.evaluate_step7_radial_sampling import extract_step7_features, compute_roc_auc_similarity

# Frozen weights trained strictly on fp_training development subjects (task-1124)
DEV_WEIGHTS_4CH = np.array([0.36357, 1.06450, 1.28243, 0.85054], dtype=np.float64) # Energy, Gray-b2, Gray-b1, Gray-b0
DEV_WEIGHTS_3CH = np.array([1.06450, 1.28243, 0.85054], dtype=np.float64)           # Gray-b2, Gray-b1, Gray-b0
DEV_WEIGHTS_RING = np.array([0.8162, 1.0073, 0.9925, 0.8789, 0.8265, 0.8100, 0.6585, 0.7669], dtype=np.float64)

def weighted_masked_hd(v1, v2, m1, m2, weights=None, min_overlap=0.50):
    """
    Computes normalized Hamming distance and weighted Hamming distance over active bit overlap.
    """
    m = (m1 > 0) & (m2 > 0)
    n_valid = int(np.sum(m))
    total_bits = len(v1)
    overlap = n_valid / float(total_bits) if total_bits > 0 else 0.0
    if overlap < min_overlap or n_valid == 0:
        return None
    
    diffs = (v1[m] != v2[m])
    errs = int(np.sum(diffs))
    norm_hd = errs / float(n_valid)
    
    if weights is None:
        weighted_hd = norm_hd
    else:
        w_valid = weights[m]
        w_sum = float(np.sum(w_valid))
        if w_sum <= 0:
            return None
        weighted_hd = float(np.sum(diffs.astype(np.float64) * w_valid) / w_sum)
        
    return {
        "norm_hd": norm_hd,
        "weighted_hd": weighted_hd,
        "errs": errs,
        "n_valid": n_valid,
        "overlap": overlap
    }

def aligned_weighted_masked_hd(v1, v2, m1, m2, bits_per_sector, weights=None, n_sectors=16, min_overlap=0.50):
    """
    Finds the cyclic shift minimizing weighted_hd (or norm_hd if unweighted).
    """
    best = None
    for s in range(n_sectors):
        sh_v2 = np.roll(v2, s * bits_per_sector)
        sh_m2 = np.roll(m2, s * bits_per_sector)
        res = weighted_masked_hd(v1, sh_v2, m1, sh_m2, weights=weights, min_overlap=min_overlap)
        if res is not None:
            if best is None or res["weighted_hd"] < best["weighted_hd"]:
                best = dict(res)
                best["shift"] = s
    return best

def compute_operational_metrics(gen_scores, imp_scores):
    """
    Computes EER, zero-FAR GAR, and GAR @ 1% FAR.
    Scores are distance metrics (lower is better).
    """
    gen = np.sort(np.array(gen_scores, dtype=np.float64))
    imp = np.sort(np.array(imp_scores, dtype=np.float64))
    
    # Impostor min distance defines Zero-FAR operational threshold
    min_imp = imp[0]
    gar_zero_far = float(np.sum(gen < min_imp) / len(gen))
    
    # 1% FAR threshold (allowing 1% impostor acceptance, so index = int(0.01 * len(imp)))
    far_1pct_idx = max(0, int(0.01 * len(imp)))
    thresh_1pct = imp[far_1pct_idx]
    gar_1pct_far = float(np.sum(gen < thresh_1pct) / len(gen))
    
    # EER calculation
    all_thresholds = np.unique(np.concatenate([gen, imp]))
    best_diff = 1.0
    eer = 0.5
    for t in all_thresholds:
        frr = np.sum(gen >= t) / len(gen)
        far = np.sum(imp < t) / len(imp)
        if abs(frr - far) < best_diff:
            best_diff = abs(frr - far)
            eer = (frr + far) / 2.0
            
    return {
        "zero_far_threshold": float(min_imp),
        "gar_zero_far": gar_zero_far,
        "gar_1pct_far": gar_1pct_far,
        "eer": float(eer)
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

    # 2. Extract Step 7 templates for all unique images
    unique_paths = set()
    for item in benchmark_data["genuine_pairs"] + benchmark_data["impostor_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])

    print(f"[*] Extracting Step 7 templates (annular mapping) for {len(unique_paths)} unique images...")
    template_cache = {}
    for p in unique_paths:
        template_cache[p] = extract_step7_features(p)
    print("[+] Feature extraction complete.\n")

    # Construct Weight Arrays
    # 512-bit weights: tile 4-channel weights across 128 cells
    W_512 = np.tile(DEV_WEIGHTS_4CH, 128)
    
    # 384-bit weights: tile 3-channel weights across 128 cells
    W_384 = np.tile(DEV_WEIGHTS_3CH, 128)
    
    # 384-bit joint weights: ring weight * channel weight for each cell
    W_384_joint = np.zeros(384, dtype=np.float64)
    for s in range(16):
        for r in range(8):
            cell_idx = s * 8 + r
            w_r = DEV_WEIGHTS_RING[r]
            for ch in range(3):
                W_384_joint[cell_idx * 3 + ch] = w_r * DEV_WEIGHTS_3CH[ch]

    tau = 0.70
    min_overlap = 0.50

    # Build representations for all images
    representations = {p: {} for p in unique_paths}
    for p, f in template_cache.items():
        v512 = f["vec"]
        m512 = (f["cov"] >= tau).astype(np.uint8)
        
        # 384-bit representation: extract channels 1, 2, 3 (Gray-b2, b1, b0)
        v384 = []
        m384 = []
        for cell in range(128):
            v384.extend([v512[cell*4 + 1], v512[cell*4 + 2], v512[cell*4 + 3]])
            c_m = m512[cell*4]
            m384.extend([c_m, c_m, c_m])
            
        representations[p]["512"] = {"vec": v512, "mask": m512}
        representations[p]["384"] = {"vec": np.array(v384, dtype=np.uint8), "mask": np.array(m384, dtype=np.uint8)}

    # Define the 5 experimental conditions
    conditions = {
        "cond_A": {
            "name": "Cond A: 4-Channel Unweighted (Baseline)",
            "rep": "512",
            "bps": 32,
            "weights": None,
            "desc": "Energy + 3-bit Gray (512 bits, unweighted, Step 7 reference)"
        },
        "cond_B": {
            "name": "Cond B: 4-Channel Dev-Weighted",
            "rep": "512",
            "bps": 32,
            "weights": W_512,
            "desc": "Energy + 3-bit Gray (512 bits, LLR weighted)"
        },
        "cond_C": {
            "name": "Cond C: 3-Channel Orientation Unweighted",
            "rep": "384",
            "bps": 24,
            "weights": None,
            "desc": "3-bit Gray only (384 bits, unweighted, Energy ablated)"
        },
        "cond_D": {
            "name": "Cond D: 3-Channel Orientation Dev-Weighted",
            "rep": "384",
            "bps": 24,
            "weights": W_384,
            "desc": "3-bit Gray only (384 bits, LLR weighted, Energy ablated)"
        },
        "cond_E": {
            "name": "Cond E: 3-Channel Joint RingxChannel Weighted",
            "rep": "384",
            "bps": 24,
            "weights": W_384_joint,
            "desc": "3-bit Gray only (384 bits, Joint Ring x Channel LLR weighted)"
        }
    }

    results = {}

    for c_key, c_cfg in conditions.items():
        print(f"[*] Running {c_cfg['name']}...")
        rep_key = c_cfg["rep"]
        bps = c_cfg["bps"]
        w = c_cfg["weights"]
        
        gen_res = []
        imp_res = []
        rej_gen = 0
        rej_imp = 0
        
        for item in benchmark_data["genuine_pairs"]:
            r1 = representations[item["path_1"]][rep_key]
            r2 = representations[item["path_2"]][rep_key]
            
            alg = aligned_weighted_masked_hd(r1["vec"], r2["vec"], r1["mask"], r2["mask"],
                                            bits_per_sector=bps, weights=w, min_overlap=min_overlap)
            if alg is None:
                rej_gen += 1
                continue
            
            gen_res.append({
                "pair_id": item["pair_id"],
                "subject_id": item["subject_id"],
                "weighted_hd": alg["weighted_hd"],
                "norm_hd": alg["norm_hd"],
                "errs": alg["errs"],
                "n_valid": alg["n_valid"],
                "overlap": alg["overlap"],
                "shift": alg["shift"]
            })
            
        for item in benchmark_data["impostor_pairs"]:
            r1 = representations[item["path_1"]][rep_key]
            r2 = representations[item["path_2"]][rep_key]
            
            alg = aligned_weighted_masked_hd(r1["vec"], r2["vec"], r1["mask"], r2["mask"],
                                            bits_per_sector=bps, weights=w, min_overlap=min_overlap)
            if alg is None:
                rej_imp += 1
                continue
            
            imp_res.append({
                "pair_id": item["pair_id"],
                "subject_1": item["subject_1"],
                "subject_2": item["subject_2"],
                "weighted_hd": alg["weighted_hd"],
                "norm_hd": alg["norm_hd"],
                "errs": alg["errs"],
                "n_valid": alg["n_valid"],
                "overlap": alg["overlap"],
                "shift": alg["shift"]
            })
            
        # Summary statistics on weighted_hd (which equals norm_hd when unweighted)
        gen_dists = [r["weighted_hd"] * 100.0 for r in gen_res]
        imp_dists = [r["weighted_hd"] * 100.0 for r in imp_res]
        gen_raw_dists = [r["norm_hd"] * 100.0 for r in gen_res]
        imp_raw_dists = [r["norm_hd"] * 100.0 for r in imp_res]
        
        g_mu, g_sig = float(np.mean(gen_dists)), float(np.std(gen_dists))
        i_mu, i_sig = float(np.mean(imp_dists)), float(np.std(imp_dists))
        sep = i_mu - g_mu
        dp = abs(i_mu - g_mu) / np.sqrt(0.5 * (g_sig**2 + i_sig**2)) if (g_sig > 0 or i_sig > 0) else 0.0
        auc = compute_roc_auc_similarity([-x for x in gen_dists], [-x for x in imp_dists])
        
        # Operational security metrics
        op_metrics = compute_operational_metrics(gen_dists, imp_dists)
        
        cfg_copy = dict(c_cfg)
        if cfg_copy["weights"] is not None:
            cfg_copy["weights"] = f"ndarray({len(cfg_copy['weights'])})"
            
        results[c_key] = {
            "config": cfg_copy,
            "genuine_mean": g_mu,
            "genuine_std": g_sig,
            "genuine_median": float(np.median(gen_dists)),
            "genuine_raw_mean": float(np.mean(gen_raw_dists)),
            "impostor_mean": i_mu,
            "impostor_std": i_sig,
            "impostor_median": float(np.median(imp_dists)),
            "impostor_raw_mean": float(np.mean(imp_raw_dists)),
            "separation": sep,
            "decidability": dp,
            "auc": auc,
            "rejections_genuine": rej_gen,
            "rejections_impostor": rej_imp,
            "mean_valid_bits": float(np.mean([r["n_valid"] for r in gen_res])),
            "mean_overlap": float(np.mean([r["overlap"] for r in gen_res])) * 100.0,
            "operational": op_metrics,
            "pair_records": {
                "genuine": gen_res,
                "impostor": imp_res
            }
        }
        print(f"  -> Gen: {g_mu:.2f}% ± {g_sig:.2f}%, Imp: {i_mu:.2f}% ± {i_sig:.2f}%, Sep: {sep:.2f} pp, d': {dp:.4f}, AUC: {auc:.4f}")

    # 3. Subject-Cluster Bootstrap Resampling (2,000 iterations)
    print("\n[*] Computing 2,000-iteration subject-cluster bootstrap confidence intervals...")
    pair_to_subj = {p["pair_id"]: p["subject_id"] for p in benchmark_data["genuine_pairs"]}
    all_gen_ids = sorted(list(set(pair_to_subj.keys())))
    unique_subjs = sorted(list(set(pair_to_subj.values())))
    all_imp_ids = [p["pair_id"] for p in benchmark_data["impostor_pairs"]]
    
    subj_to_pairs = {s: [] for s in unique_subjs}
    for pid, s in pair_to_subj.items():
        subj_to_pairs[s].append(pid)
        
    # Maps for fast lookup: cond -> pid -> score
    score_maps = {}
    for c_key in conditions:
        score_maps[c_key] = {
            "gen": {r["pair_id"]: r["weighted_hd"] * 100.0 for r in results[c_key]["pair_records"]["genuine"]},
            "imp": {r["pair_id"]: r["weighted_hd"] * 100.0 for r in results[c_key]["pair_records"]["impostor"]}
        }
        
    np.random.seed(42)
    n_boot = 2000
    boot_stats = {c_key: {"sep": [], "dp": [], "auc": []} for c_key in conditions}
    boot_deltas = {
        "delta_C_minus_A": [], # Effect of Energy channel removal (unweighted)
        "delta_B_minus_A": [], # Effect of LLR weighting (4-channel)
        "delta_D_minus_A": [], # Effect of both (3-channel weighted vs 4-channel baseline)
        "delta_D_minus_C": [], # Effect of LLR weighting on 3-channel
        "delta_E_minus_D": []  # Effect of ring weighting on 3-channel
    }
    
    for _ in range(n_boot):
        # Cluster sample genuine pairs by subject
        sampled_subjs = np.random.choice(unique_subjs, size=len(unique_subjs), replace=True)
        sampled_gen_ids = []
        for s in sampled_subjs:
            sampled_gen_ids.extend(subj_to_pairs[s])
        sampled_imp_ids = np.random.choice(all_imp_ids, size=len(all_imp_ids), replace=True)
        
        boot_aucs = {}
        for c_key in conditions:
            g_boot = [score_maps[c_key]["gen"][pid] for pid in sampled_gen_ids if pid in score_maps[c_key]["gen"]]
            i_boot = [score_maps[c_key]["imp"][pid] for pid in sampled_imp_ids if pid in score_maps[c_key]["imp"]]
            gm, gs = np.mean(g_boot), np.std(g_boot)
            im, is_ = np.mean(i_boot), np.std(i_boot)
            sp = im - gm
            d_p = abs(im - gm) / np.sqrt(0.5 * (gs**2 + is_**2))
            b_auc = compute_roc_auc_similarity([-x for x in g_boot], [-x for x in i_boot])
            
            boot_stats[c_key]["sep"].append(sp)
            boot_stats[c_key]["dp"].append(d_p)
            boot_stats[c_key]["auc"].append(b_auc)
            boot_aucs[c_key] = b_auc
            
        boot_deltas["delta_C_minus_A"].append(boot_aucs["cond_C"] - boot_aucs["cond_A"])
        boot_deltas["delta_B_minus_A"].append(boot_aucs["cond_B"] - boot_aucs["cond_A"])
        boot_deltas["delta_D_minus_A"].append(boot_aucs["cond_D"] - boot_aucs["cond_A"])
        boot_deltas["delta_D_minus_C"].append(boot_aucs["cond_D"] - boot_aucs["cond_C"])
        boot_deltas["delta_E_minus_D"].append(boot_aucs["cond_E"] - boot_aucs["cond_D"])

    # Compute 95% CIs and p-values
    bootstrap_summary = {}
    for c_key in conditions:
        bootstrap_summary[c_key] = {
            "sep_ci": [float(np.percentile(boot_stats[c_key]["sep"], 2.5)), float(np.percentile(boot_stats[c_key]["sep"], 97.5))],
            "dp_ci": [float(np.percentile(boot_stats[c_key]["dp"], 2.5)), float(np.percentile(boot_stats[c_key]["dp"], 97.5))],
            "auc_ci": [float(np.percentile(boot_stats[c_key]["auc"], 2.5)), float(np.percentile(boot_stats[c_key]["auc"], 97.5))]
        }
        
    delta_summary = {}
    for d_name, d_vals in boot_deltas.items():
        mean_d = float(np.mean(d_vals))
        ci_lower = float(np.percentile(d_vals, 2.5))
        ci_upper = float(np.percentile(d_vals, 97.5))
        # Two-tailed p-value
        p_val = float(2.0 * min(np.mean(np.array(d_vals) <= 0), np.mean(np.array(d_vals) >= 0)))
        delta_summary[d_name] = {
            "mean_delta": mean_d,
            "ci_95": [ci_lower, ci_upper],
            "p_value": p_val,
            "significant": bool(ci_lower > 0 or ci_upper < 0)
        }

    # Print Formatted Results Table
    print("\n" + "=" * 115)
    print("           STEP 8: CONTROLLED CHANNEL ABLATION & RELIABILITY WEIGHTING RESULTS")
    print("=" * 115)
    print(f"{'Condition':<36} {'Genuine HD':<18} {'Impostor HD':<18} {'Sep (pp)':<10} {'d-prime':<10} {'AUC':<10} {'EER'}")
    print("-" * 115)
    for c_key, c_cfg in conditions.items():
        res = results[c_key]
        g_str = f"{res['genuine_mean']:.2f}% ± {res['genuine_std']:.2f}%"
        i_str = f"{res['impostor_mean']:.2f}% ± {res['impostor_std']:.2f}%"
        sep_str = f"{res['separation']:+.2f}"
        dp_str = f"{res['decidability']:.4f}"
        auc_str = f"{res['auc']:.4f}"
        eer_str = f"{res['operational']['eer']*100.0:.2f}%"
        print(f"{c_cfg['name']:<36} {g_str:<18} {i_str:<18} {sep_str:<10} {dp_str:<10} {auc_str:<10} {eer_str}")
    print("=" * 115)

    print("\n=== PAIRED BOOTSTRAP DELTA-AUC COMPARISONS (2,000 iterations) ===")
    comparisons_meta = [
        ("delta_C_minus_A", "Cond C (3ch unweighted) vs Cond A (4ch unweighted) [Energy Ablation]"),
        ("delta_B_minus_A", "Cond B (4ch weighted) vs Cond A (4ch unweighted) [4ch LLR Weighting]"),
        ("delta_D_minus_A", "Cond D (3ch weighted) vs Cond A (4ch unweighted) [3ch Weighted vs Base]"),
        ("delta_D_minus_C", "Cond D (3ch weighted) vs Cond C (3ch unweighted) [3ch LLR Weighting]"),
        ("delta_E_minus_D", "Cond E (3ch joint weighted) vs Cond D (3ch weighted) [Ring Weighting]")
    ]
    for d_key, label in comparisons_meta:
        ds = delta_summary[d_key]
        sig_str = "SIGNIFICANT" if ds["significant"] else "NOT significant"
        print(f"- {label}:")
        print(f"    Delta AUC = {ds['mean_delta']:+.4f}, 95% CI = [{ds['ci_95'][0]:+.4f}, {ds['ci_95'][1]:+.4f}], p = {ds['p_value']:.4f} ({sig_str})")

    # Save to JSON
    output_path = os.path.join(BASE_DIR, "results", "step8_channel_ablation_results.json")
    save_data = {
        "metadata": {
            "experiment": "Step 8: Controlled Channel Ablation & Reliability Weighting",
            "training_source": "fp_training (development subjects strictly)",
            "benchmark_source": "fixed_pairs_benchmark.json (SHA-256 verified)",
            "dev_weights_4ch": DEV_WEIGHTS_4CH.tolist(),
            "dev_weights_3ch": DEV_WEIGHTS_3CH.tolist(),
            "dev_weights_ring": DEV_WEIGHTS_RING.tolist(),
            "n_boot": n_boot
        },
        "conditions": results,
        "bootstrap_summary": bootstrap_summary,
        "delta_summary": delta_summary
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[+] Results successfully saved to {output_path}")

if __name__ == "__main__":
    main()
