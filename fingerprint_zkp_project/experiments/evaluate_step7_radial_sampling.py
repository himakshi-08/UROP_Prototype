"""
Step 7 Experimental Evaluation: Isolated Radial Sampling Fix.
Maps all 8 radial bins over the active annular region [r_inner, r_max] where r_inner = 0.18 * r_max,
using explicit coordinate meshgrid and cv2.remap with 128 samples per cell (256 theta x 64 r).

Isolated Experimental Controls:
- Angular sectors: 16 (22.5 deg each)
- Radial rings: 8 (equally spaced over [r_inner, r_max])
- Quadrature Gabor bank: 8 orientations (unchanged)
- Orientation quantization: 3-bit circular Gray code (unchanged)
- Validity threshold: tau = 0.70 (unchanged)
- Alignment: 16 cyclic shifts of 32 bits (unchanged)
- Benchmark pairs: Exact 100 genuine and 100 impostor pairs from fixed_pairs_benchmark.json
- Statistical rigor: Subject-cluster bootstrap resampling (2000 iterations) for 95% CIs.
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

from preprocessing import preprocess_fingerprint
from features import extract_minutiae, build_template
from features.template import _quadrature_gabor_bank, N_FILTERS, N_SECTORS, N_RINGS

def d_pi(a, b):
    diff = np.abs(a - b) % np.pi
    return np.minimum(diff, np.pi - diff)

def extract_step7_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at {img_path}")
    prep = preprocess_fingerprint(img)
    minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
    tmpl = build_template(img, prep, minutiae, vector_bits=256)

    prepared = tmpl._prepare_polar(img)
    if prepared is None:
        raise ValueError(f"Polar preparation failed for {img_path}")
    img_w, mask_w, ori_w, center, max_r, inner = prepared

    cx, cy = center
    h, w = img_w.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring_mask = ((rr >= inner) & (rr <= max_r) & (mask_w > 0.25)).astype(np.float32)
    img_roi = img_w * ring_mask

    # Step 7 Radial Sampling Grid over [inner, max_r]
    # 256 angular steps x 64 radial steps -> 128 area samples per (16x8) polar cell
    n_th, n_r = 256, 64
    thetas = np.linspace(0, 2 * np.pi, n_th, endpoint=False).astype(np.float32)
    radii = (inner + (np.arange(n_r, dtype=np.float32) + 0.5) * (max_r - inner) / float(n_r))
    R_grid, TH_grid = np.meshgrid(radii, thetas)
    map_x = (cx + R_grid * np.cos(TH_grid)).astype(np.float32)
    map_y = (cy + R_grid * np.sin(TH_grid)).astype(np.float32)

    gabor_bank = _quadrature_gabor_bank()
    p_energies = []
    filter_thetas = []
    for ke, ko, th_k in gabor_bank:
        re = cv2.filter2D(img_roi, cv2.CV_32F, ke)
        ro = cv2.filter2D(img_roi, cv2.CV_32F, ko)
        e = np.sqrt(re * re + ro * ro) * ring_mask
        e_hi = cv2.remap(e, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        p_energies.append(cv2.resize(e_hi, (8, 16), interpolation=cv2.INTER_AREA))
        filter_thetas.append(th_k)

    m_hi = cv2.remap(ring_mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    polar_m = cv2.resize(m_hi, (8, 16), interpolation=cv2.INTER_AREA)

    valid_cells = (polar_m >= 0.70)

    # 1. Total Directional Energy per cell
    E_total = np.sum(p_energies, axis=0) # (16, 8)
    med_energy = float(np.median(E_total[valid_cells])) if np.any(valid_cells) else float(np.median(E_total))
    bits_e = (E_total > med_energy).astype(np.uint8)

    # 2. Dominant Gabor Orientation via circular vector summation
    U = np.sum([p_energies[k] * np.cos(2.0 * filter_thetas[k]) for k in range(N_FILTERS)], axis=0)
    V = np.sum([p_energies[k] * np.sin(2.0 * filter_thetas[k]) for k in range(N_FILTERS)], axis=0)
    theta_gabor = np.mod(0.5 * np.arctan2(V, U), np.pi).astype(np.float32)

    q = np.clip(np.floor(8.0 * theta_gabor / np.pi).astype(np.int32), 0, 7)
    g = q ^ (q >> 1) # 3-bit circular Gray code

    # Assemble 512-bit representation (128 cells x 4 bits)
    vec_512 = []
    cov_512 = []
    for s in range(16):
        for r_idx in range(8):
            c_cov = float(polar_m[s, r_idx])
            g_val = int(g[s, r_idx])
            b2 = (g_val >> 2) & 1
            b1 = (g_val >> 1) & 1
            b0 = (g_val >> 0) & 1
            e_bit = int(bits_e[s, r_idx])

            vec_512.extend([e_bit, b2, b1, b0])
            cov_512.extend([c_cov, c_cov, c_cov, c_cov])

    return {
        "vec": np.array(vec_512, dtype=np.uint8),
        "cov": np.array(cov_512, dtype=np.float32),
        "polar_m": polar_m,
        "valid_cells_count": int(np.sum(valid_cells)),
        "ring_coverage_means": np.mean(polar_m, axis=0).tolist(),
        "ring_valid_fractions": (np.mean(polar_m >= 0.70, axis=0)).tolist(),
        "bits_per_sector": 32,
        "total_bits": 512
    }

def masked_hd(v1, v2, m1, m2, min_overlap=0.50):
    m = (m1 > 0) & (m2 > 0)
    n_valid = int(np.sum(m))
    total_bits = len(v1)
    overlap = n_valid / float(total_bits) if total_bits > 0 else 0.0
    if overlap < min_overlap or n_valid == 0:
        return None
    errs = int(np.sum(v1[m] != v2[m]))
    norm_hd = errs / float(n_valid)
    return {
        "norm_hd": norm_hd,
        "equiv_bits": norm_hd * float(total_bits),
        "errs": errs,
        "n_valid": n_valid,
        "overlap": overlap
    }

def aligned_masked_hd(v1, v2, m1, m2, bits_per_sector=32, n_sectors=16, min_overlap=0.50):
    best = None
    for s in range(n_sectors):
        sh_v2 = np.roll(v2, s * bits_per_sector)
        sh_m2 = np.roll(m2, s * bits_per_sector)
        res = masked_hd(v1, sh_v2, m1, sh_m2, min_overlap=min_overlap)
        if res is not None:
            if best is None or res["norm_hd"] < best["norm_hd"]:
                best = dict(res)
                best["shift"] = s
    return best

def compute_roc_auc_similarity(genuine_scores, impostor_scores):
    gen = np.array(genuine_scores, dtype=np.float64)
    imp = np.array(impostor_scores, dtype=np.float64)
    all_scores = np.concatenate([gen, imp])
    thresholds = np.linspace(np.min(all_scores) - 1e-4, np.max(all_scores) + 1e-4, 1001)

    tpr = []
    fpr = []
    n_gen = len(gen)
    n_imp = len(imp)

    for tau in thresholds:
        tp = np.sum(gen >= tau) / float(n_gen)
        fp = np.sum(imp >= tau) / float(n_imp)
        tpr.append(tp)
        fpr.append(fp)

    sorted_idx = np.argsort(fpr)
    s_fpr = np.array(fpr)[sorted_idx]
    s_tpr = np.array(tpr)[sorted_idx]
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(trapz_func(s_tpr, s_fpr))

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

    # 2. Extract Step 7 templates
    unique_paths = set()
    for item in benchmark_data["genuine_pairs"] + benchmark_data["impostor_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])

    print(f"[*] Extracting Step 7 templates (annulus cv2.remap) for {len(unique_paths)} unique images...")
    template_cache = {}
    for p in unique_paths:
        template_cache[p] = extract_step7_features(p)
    print("[+] Extraction complete.\n")

    # 3. Analyze Ring Coverage Profile
    all_ring_cov = np.array([template_cache[p]["ring_coverage_means"] for p in unique_paths])
    all_ring_val = np.array([template_cache[p]["ring_valid_fractions"] for p in unique_paths])
    all_valid_counts = [template_cache[p]["valid_cells_count"] for p in unique_paths]

    mean_ring_cov = np.mean(all_ring_cov, axis=0) * 100.0
    mean_ring_val = np.mean(all_ring_val, axis=0) * 100.0

    print("=" * 95)
    print("                    STEP 7: RADIAL RING COVERAGE PROFILE")
    print("=" * 95)
    print(f"{'Ring':<8} {'Radial Interval':<22} {'Mean Mask Coverage':<22} {'Valid Cells Fraction (>=0.70)'}")
    print("-" * 80)
    for r in range(8):
        int_str = f"[r_{r}, r_{r+1}]"
        print(f"Ring {r:<3} {int_str:<22} {f'{mean_ring_cov[r]:.1f}%':<22} {f'{mean_ring_val[r]:.1f}%':<20}")
    print("-" * 80)
    print(f"Mean valid cells per template: {np.mean(all_valid_counts):.1f} / 128 cells ({np.mean(all_valid_counts)/128.0*100.0:.1f}%)")
    print("=" * 95)

    # 4. Evaluate Genuine and Impostor Pairs
    tau = 0.70
    min_overlap = 0.50

    gen_results = []
    imp_results = []
    rej_gen = 0
    rej_imp = 0

    for item in benchmark_data["genuine_pairs"]:
        t1 = template_cache[item["path_1"]]
        t2 = template_cache[item["path_2"]]
        m1 = (t1["cov"] >= tau).astype(np.uint8)
        m2 = (t2["cov"] >= tau).astype(np.uint8)

        raw_res = masked_hd(t1["vec"], t2["vec"], m1, m2, min_overlap=min_overlap)
        alg_res = aligned_masked_hd(t1["vec"], t2["vec"], m1, m2, bits_per_sector=32, min_overlap=min_overlap)

        if raw_res is None or alg_res is None:
            rej_gen += 1
            continue

        gen_results.append({
            "pair_id": item["pair_id"],
            "subject_id": item["subject_id"],
            "raw_norm_hd": raw_res["norm_hd"],
            "raw_errs": raw_res["errs"],
            "raw_n_valid": raw_res["n_valid"],
            "raw_overlap": raw_res["overlap"],
            "aligned_norm_hd": alg_res["norm_hd"],
            "aligned_errs": alg_res["errs"],
            "aligned_n_valid": alg_res["n_valid"],
            "aligned_overlap": alg_res["overlap"],
            "selected_shift": alg_res["shift"]
        })

    for item in benchmark_data["impostor_pairs"]:
        t1 = template_cache[item["path_1"]]
        t2 = template_cache[item["path_2"]]
        m1 = (t1["cov"] >= tau).astype(np.uint8)
        m2 = (t2["cov"] >= tau).astype(np.uint8)

        raw_res = masked_hd(t1["vec"], t2["vec"], m1, m2, min_overlap=min_overlap)
        alg_res = aligned_masked_hd(t1["vec"], t2["vec"], m1, m2, bits_per_sector=32, min_overlap=min_overlap)

        if raw_res is None or alg_res is None:
            rej_imp += 1
            continue

        imp_results.append({
            "pair_id": item["pair_id"],
            "subject_1": item["subject_1"],
            "subject_2": item["subject_2"],
            "raw_norm_hd": raw_res["norm_hd"],
            "raw_errs": raw_res["errs"],
            "raw_n_valid": raw_res["n_valid"],
            "raw_overlap": raw_res["overlap"],
            "aligned_norm_hd": alg_res["norm_hd"],
            "aligned_errs": alg_res["errs"],
            "aligned_n_valid": alg_res["n_valid"],
            "aligned_overlap": alg_res["overlap"],
            "selected_shift": alg_res["shift"]
        })

    # Metrics
    gen_alg_hd = [r["aligned_norm_hd"] * 100.0 for r in gen_results]
    imp_alg_hd = [r["aligned_norm_hd"] * 100.0 for r in imp_results]
    gen_alg_errs = [r["aligned_errs"] for r in gen_results]
    imp_alg_errs = [r["aligned_errs"] for r in imp_results]
    gen_n_val = [r["aligned_n_valid"] for r in gen_results]
    imp_n_val = [r["aligned_n_valid"] for r in imp_results]
    gen_ovp = [r["aligned_overlap"] * 100.0 for r in gen_results]
    imp_ovp = [r["aligned_overlap"] * 100.0 for r in imp_results]

    g_mu, g_sig = float(np.mean(gen_alg_hd)), float(np.std(gen_alg_hd))
    i_mu, i_sig = float(np.mean(imp_alg_hd)), float(np.std(imp_alg_hd))
    sep = i_mu - g_mu
    dp = abs(i_mu - g_mu) / np.sqrt(0.5 * (g_sig**2 + i_sig**2)) if (g_sig > 0 or i_sig > 0) else 0.0
    auc = compute_roc_auc_similarity([-x for x in gen_alg_hd], [-x for x in imp_alg_hd])

    # 5. Subject-Cluster Bootstrap Resampling (2000 iterations)
    print("\n[*] Computing 2,000-iteration subject-cluster bootstrap confidence intervals...")
    with open(os.path.join(BASE_DIR, "results", "step4_evaluation_results.json"), "r") as f:
        step4_res = json.load(f)["exp_b1"]

    b1_ref_gen = {r["pair_id"]: r["aligned_norm_hd"] * 100.0 for r in step4_res["pair_records"]["genuine"]}
    b1_ref_imp = {r["pair_id"]: r["aligned_norm_hd"] * 100.0 for r in step4_res["pair_records"]["impostor"]}

    step7_gen_by_id = {r["pair_id"]: r["aligned_norm_hd"] * 100.0 for r in gen_results}
    step7_imp_by_id = {r["pair_id"]: r["aligned_norm_hd"] * 100.0 for r in imp_results}

    common_gen_ids = sorted(list(set(b1_ref_gen.keys()) & set(step7_gen_by_id.keys())))
    common_imp_ids = sorted(list(set(b1_ref_imp.keys()) & set(step7_imp_by_id.keys())))

    pair_to_subj = {p["pair_id"]: p["subject_id"] for p in benchmark_data["genuine_pairs"]}
    subjs_in_common = [pair_to_subj[pid] for pid in common_gen_ids]
    unique_subjs = sorted(list(set(subjs_in_common)))

    subj_to_pairs = {s: [] for s in unique_subjs}
    for pid in common_gen_ids:
        s = pair_to_subj[pid]
        subj_to_pairs[s].append(pid)

    np.random.seed(42)
    n_boot = 2000
    boot_g_mu, boot_i_mu, boot_sep, boot_dp, boot_auc = [], [], [], [], []
    boot_delta_auc = []

    for _ in range(n_boot):
        sampled_subjs = np.random.choice(unique_subjs, size=len(unique_subjs), replace=True)
        sampled_gen_ids = []
        for s in sampled_subjs:
            sampled_gen_ids.extend(subj_to_pairs[s])
        sampled_imp_ids = np.random.choice(common_imp_ids, size=len(common_imp_ids), replace=True)

        # Step 7 resampled scores
        g7 = [step7_gen_by_id[pid] for pid in sampled_gen_ids]
        i7 = [step7_imp_by_id[pid] for pid in sampled_imp_ids]
        g7_m, g7_s = np.mean(g7), np.std(g7)
        i7_m, i7_s = np.mean(i7), np.std(i7)
        sp = i7_m - g7_m
        dpr = abs(i7_m - g7_m) / np.sqrt(0.5 * (g7_s**2 + i7_s**2))
        auc7 = compute_roc_auc_similarity([-x for x in g7], [-x for x in i7])

        # Step 4 B1 resampled AUC
        g_b1 = [b1_ref_gen[pid] for pid in sampled_gen_ids]
        i_b1 = [b1_ref_imp[pid] for pid in sampled_imp_ids]
        auc_b1 = compute_roc_auc_similarity([-x for x in g_b1], [-x for x in i_b1])

        boot_g_mu.append(g7_m)
        boot_i_mu.append(i7_m)
        boot_sep.append(sp)
        boot_dp.append(dpr)
        boot_auc.append(auc7)
        boot_delta_auc.append(auc7 - auc_b1)

    ci_g = np.percentile(boot_g_mu, [2.5, 97.5])
    ci_i = np.percentile(boot_i_mu, [2.5, 97.5])
    ci_sep = np.percentile(boot_sep, [2.5, 97.5])
    ci_dp = np.percentile(boot_dp, [2.5, 97.5])
    ci_auc = np.percentile(boot_auc, [2.5, 97.5])
    ci_delta_auc = np.percentile(boot_delta_auc, [2.5, 97.5])
    p_val_delta = np.mean(np.array(boot_delta_auc) <= 0.0)

    # 6. Print Comprehensive Comparative Report
    print("\n" + "=" * 115)
    print("                    STEP 7 EVALUATION: COMPARISON WITH STEP 4 B1 REFERENCE")
    print("=" * 115)
    print(f"{'Metric':<30} {'Step 4 B1 (Old warpPolar)':<25} {'Step 7 (Active Annulus remap)':<28} {'Change / CI'}")
    print("-" * 115)

    ref_g = f"{step4_res['genuine_norm_hd_mean']:.2f} +- {step4_res['genuine_norm_hd_std']:.2f}%"
    s7_g = f"{g_mu:.2f} +- {g_sig:.2f}%"
    print(f"{'Genuine Aligned HD':<30} {ref_g:<25} {s7_g:<28} 95% CI: [{ci_g[0]:.2f}%, {ci_g[1]:.2f}%]")

    ref_i = f"{step4_res['impostor_norm_hd_mean']:.2f} +- {step4_res['impostor_norm_hd_std']:.2f}%"
    s7_i = f"{i_mu:.2f} +- {i_sig:.2f}%"
    print(f"{'Impostor Aligned HD':<30} {ref_i:<25} {s7_i:<28} 95% CI: [{ci_i[0]:.2f}%, {ci_i[1]:.2f}%]")

    ref_sep = f"+{step4_res['separation_pct']:.2f}%"
    s7_sep = f"+{sep:.2f}%"
    print(f"{'Genuine-Impostor Separation':<30} {ref_sep:<25} {s7_sep:<28} 95% CI: [{ci_sep[0]:.2f}%, {ci_sep[1]:.2f}%]")

    ref_dp = f"{step4_res['d_prime']:.4f}"
    s7_dp = f"{dp:.4f}"
    print(f"{'Decidability d-prime':<30} {ref_dp:<25} {s7_dp:<28} 95% CI: [{ci_dp[0]:.4f}, {ci_dp[1]:.4f}]")

    ref_auc = f"{step4_res['roc_auc']:.4f}"
    s7_auc = f"{auc:.4f}"
    print(f"{'ROC-AUC':<30} {ref_auc:<25} {s7_auc:<28} 95% CI: [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")

    ref_nval = f"{step4_res['n_valid_mean']:.1f} / 512 ({step4_res['overlap_pct_mean']:.1f}%)"
    s7_nval = f"{np.mean(gen_n_val):.1f} / 512 ({np.mean(gen_ovp):.1f}%)"
    print(f"{'Valid Compared Bits (Gen)':<30} {ref_nval:<25} {s7_nval:<28} +{np.mean(gen_n_val) - step4_res['n_valid_mean']:.1f} valid bits (+{np.mean(gen_ovp) - step4_res['overlap_pct_mean']:.1f}%)")

    ref_raw_err = f"{step4_res['genuine_errs_mean']:.1f} / {step4_res['impostor_errs_mean']:.1f}"
    s7_raw_err = f"{np.mean(gen_alg_errs):.1f} / {np.mean(imp_alg_errs):.1f}"
    print(f"{'Raw Bit Errors (Gen / Imp)':<30} {ref_raw_err:<25} {s7_raw_err:<28} Actual unscaled bit counts")

    ref_rej = f"{step4_res['rejected_genuine']} G / {step4_res['rejected_impostor']} I"
    s7_rej = f"{rej_gen} G / {rej_imp} I"
    print(f"{'Rejections (Overlap < 50%)':<30} {ref_rej:<25} {s7_rej:<28} Unchanged guardrail")

    delta_auc_pt = auc - step4_res["roc_auc"]
    print(f"{'Delta AUC (Step 7 - Step 4)':<30} {'---':<25} {f'{delta_auc_pt:+.4f}':<28} 95% CI: [{ci_delta_auc[0]:+.4f}, {ci_delta_auc[1]:+.4f}] (p={p_val_delta:.4f})")
    print("=" * 115)

    # Save results to JSON
    out_json = os.path.join(BASE_DIR, "results", "step7_radial_sampling_results.json")
    results_dict = {
        "ring_coverage_means": mean_ring_cov.tolist(),
        "ring_valid_fractions": mean_ring_val.tolist(),
        "mean_valid_cells": float(np.mean(all_valid_counts)),
        "genuine_norm_hd_mean": g_mu,
        "genuine_norm_hd_std": g_sig,
        "genuine_norm_hd_95_ci": ci_g.tolist(),
        "impostor_norm_hd_mean": i_mu,
        "impostor_norm_hd_std": i_sig,
        "impostor_norm_hd_95_ci": ci_i.tolist(),
        "separation_pct": sep,
        "separation_95_ci": ci_sep.tolist(),
        "d_prime": float(dp),
        "d_prime_95_ci": ci_dp.tolist(),
        "roc_auc": float(auc),
        "roc_auc_95_ci": ci_auc.tolist(),
        "delta_auc_vs_step4": float(delta_auc_pt),
        "delta_auc_95_ci": ci_delta_auc.tolist(),
        "delta_auc_p_val": float(p_val_delta),
        "n_valid_bits_mean": float(np.mean(gen_n_val)),
        "n_valid_bits_std": float(np.std(gen_n_val)),
        "overlap_pct_mean": float(np.mean(gen_ovp)),
        "overlap_pct_std": float(np.std(gen_ovp)),
        "genuine_raw_errs_mean": float(np.mean(gen_alg_errs)),
        "impostor_raw_errs_mean": float(np.mean(imp_alg_errs)),
        "rejected_genuine": rej_gen,
        "rejected_impostor": rej_imp,
        "pair_records": {
            "genuine": gen_results,
            "impostor": imp_results
        }
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[+] Full Step 7 evaluation results saved to: {out_json}")

if __name__ == "__main__":
    main()

