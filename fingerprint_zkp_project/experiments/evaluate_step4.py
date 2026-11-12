"""
Step 4 Evaluation Script:
Evaluates Experiments A, B1, B2, C1, and C2 on the locked benchmark:
fixed_pairs_benchmark.json (100 genuine pairs and 100 impostor pairs from fp_testing).

Configurations:
- Exp A:  256-bit, 128 cells (Step 3 reference, 1-bit energy + 1-bit orientation).
- Exp B1: 512-bit, 128 cells (Isolate 3-bit Gray code orientation quantization).
- Exp B2: 256-bit, 64 cells  (Combined radial pooling [8->4 rings] + 3-bit Gray code).
- Exp C1: 512-bit, 128 cells (B1 + continuous boundary uncertainty mask at delta*).
- Exp C2: 256-bit, 64 cells  (B2 + continuous boundary uncertainty mask at delta*).

Protocol:
- SHA-256 integrity verification of all benchmark images before evaluation.
- Test-set isolation: delta* selected strictly on fp_training development set.
- Detailed metrics: Normalized HD (%), SD, separation, d', ROC-AUC, valid overlap (%),
  actual compared bit counts (N_valid and errors), and rejection counts.
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

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def d_pi(a, b):
    diff = np.abs(a - b) % np.pi
    return np.minimum(diff, np.pi - diff)

def d_boundary(theta):
    boundaries = np.arange(8) * (np.pi / 8.0)
    diffs = [d_pi(theta, b) for b in boundaries]
    return np.min(diffs, axis=0)

def extract_step4_all_configs(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at {img_path}")
    prep = preprocess_fingerprint(img)
    minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
    tmpl = build_template(img, prep, minutiae, vector_bits=256)

    # 1. Step 3 Reference Template (Exp A)
    exp_a_vec = np.array(tmpl.binary_vector, dtype=np.uint8)
    exp_a_cov = np.array(tmpl.coverage_vector, dtype=np.float32)

    # Polar preparation
    prepared = tmpl._prepare_polar(img)
    if prepared is None:
        raise ValueError(f"Polar preparation failed for {img_path}")
    img_w, mask_w, ori_w, center, max_r, inner = prepared
    h, w = img_w.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    ring_mask = ((rr >= inner) & (rr <= max_r) & (mask_w > 0.25)).astype(np.float32)
    img_roi = img_w * ring_mask

    n_sec, n_ring = 16, 8
    dsize = (n_ring, n_sec)
    flags = cv2.WARP_FILL_OUTLIERS + cv2.INTER_AREA

    # 8-orientation quadrature Gabor responses
    gabor_bank = _quadrature_gabor_bank()
    p_energies = []
    thetas = []
    for ke, ko, theta_k in gabor_bank:
        re = cv2.filter2D(img_roi, cv2.CV_32F, ke)
        ro = cv2.filter2D(img_roi, cv2.CV_32F, ko)
        e = np.sqrt(re * re + ro * ro) * ring_mask
        p_energies.append(cv2.warpPolar(e, dsize, center, max_r, flags))
        thetas.append(theta_k)

    polar_m = cv2.warpPolar(ring_mask, dsize, center, max_r, flags)
    valid_cells = (polar_m > 0.2)

    # 2. 128 cells (16x8): Configuration B1
    E_total_128 = np.sum(p_energies, axis=0)
    med_energy_128 = float(np.median(E_total_128[valid_cells])) if np.any(valid_cells) else float(np.median(E_total_128))
    bits_e_128 = (E_total_128 > med_energy_128).astype(np.uint8)

    U_128 = np.sum([p_energies[k] * np.cos(2.0 * thetas[k]) for k in range(N_FILTERS)], axis=0)
    V_128 = np.sum([p_energies[k] * np.sin(2.0 * thetas[k]) for k in range(N_FILTERS)], axis=0)
    theta_128 = np.mod(0.5 * np.arctan2(V_128, U_128), np.pi).astype(np.float32)

    q_128 = np.clip(np.floor(8.0 * theta_128 / np.pi).astype(np.int32), 0, 7)
    g_128 = q_128 ^ (q_128 >> 1)  # 3-bit Gray code
    d_bound_128 = d_boundary(theta_128)

    vec_512, cov_512, dbound_512 = [], [], []
    for s in range(n_sec):
        for r in range(n_ring):
            c_cov = float(polar_m[s, r])
            c_db = float(d_bound_128[s, r])
            g_val = int(g_128[s, r])
            b2 = (g_val >> 2) & 1
            b1 = (g_val >> 1) & 1
            b0 = (g_val >> 0) & 1
            e_bit = int(bits_e_128[s, r])

            vec_512.extend([e_bit, b2, b1, b0])
            cov_512.extend([c_cov, c_cov, c_cov, c_cov])
            dbound_512.extend([999.0, c_db, c_db, c_db])  # Energy bit independent of orientation boundary

    # 3. 64 cells (16x4 pooled rings): Configuration B2
    p_energies_pooled = []
    for k in range(N_FILTERS):
        pe = p_energies[k]
        pe_p = np.zeros((16, 4), dtype=np.float32)
        for p in range(4):
            pe_p[:, p] = 0.5 * (pe[:, 2*p] + pe[:, 2*p+1])
        p_energies_pooled.append(pe_p)

    polar_m_pooled = np.zeros((16, 4), dtype=np.float32)
    for p in range(4):
        polar_m_pooled[:, p] = 0.5 * (polar_m[:, 2*p] + polar_m[:, 2*p+1])

    valid_cells_pooled = (polar_m_pooled > 0.2)
    E_total_64 = np.sum(p_energies_pooled, axis=0)
    med_energy_64 = float(np.median(E_total_64[valid_cells_pooled])) if np.any(valid_cells_pooled) else float(np.median(E_total_64))
    bits_e_64 = (E_total_64 > med_energy_64).astype(np.uint8)

    U_64 = np.sum([p_energies_pooled[k] * np.cos(2.0 * thetas[k]) for k in range(N_FILTERS)], axis=0)
    V_64 = np.sum([p_energies_pooled[k] * np.sin(2.0 * thetas[k]) for k in range(N_FILTERS)], axis=0)
    theta_64 = np.mod(0.5 * np.arctan2(V_64, U_64), np.pi).astype(np.float32)
    q_64 = np.clip(np.floor(8.0 * theta_64 / np.pi).astype(np.int32), 0, 7)
    g_64 = q_64 ^ (q_64 >> 1)
    d_bound_64 = d_boundary(theta_64)

    vec_256_b2, cov_256_b2, dbound_256_b2 = [], [], []
    for s in range(16):
        for p in range(4):
            c_cov = float(polar_m_pooled[s, p])
            c_db = float(d_bound_64[s, p])
            g_val = int(g_64[s, p])
            b2 = (g_val >> 2) & 1
            b1 = (g_val >> 1) & 1
            b0 = (g_val >> 0) & 1
            e_bit = int(bits_e_64[s, p])

            vec_256_b2.extend([e_bit, b2, b1, b0])
            cov_256_b2.extend([c_cov, c_cov, c_cov, c_cov])
            dbound_256_b2.extend([999.0, c_db, c_db, c_db])

    return {
        "exp_a": {
            "vec": exp_a_vec,
            "cov": exp_a_cov,
            "bits_per_sector": 16,
            "total_bits": 256
        },
        "exp_b1": {
            "vec": np.array(vec_512, dtype=np.uint8),
            "cov": np.array(cov_512, dtype=np.float32),
            "dbound": np.array(dbound_512, dtype=np.float32),
            "bits_per_sector": 32,
            "total_bits": 512
        },
        "exp_b2": {
            "vec": np.array(vec_256_b2, dtype=np.uint8),
            "cov": np.array(cov_256_b2, dtype=np.float32),
            "dbound": np.array(dbound_256_b2, dtype=np.float32),
            "bits_per_sector": 16,
            "total_bits": 256
        }
    }

def masked_hd(v1, v2, m1, m2, min_overlap=0.40):
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

def aligned_masked_hd(v1, v2, m1, m2, bits_per_sector, n_sectors=16, min_overlap=0.40):
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

def evaluate_config(benchmark_data, template_cache, cfg_name, get_mask_fn, min_overlap=0.40):
    gen_results = []
    imp_results = []
    rej_gen = 0
    rej_imp = 0

    # Genuine
    for item in benchmark_data["genuine_pairs"]:
        t1 = template_cache[item["path_1"]][cfg_name]
        t2 = template_cache[item["path_2"]][cfg_name]
        v1, v2 = t1["vec"], t2["vec"]
        m1, m2 = get_mask_fn(t1, t2)
        bps = t1["bits_per_sector"]

        raw_res = masked_hd(v1, v2, m1, m2, min_overlap=min_overlap)
        alg_res = aligned_masked_hd(v1, v2, m1, m2, bps, min_overlap=min_overlap)

        if raw_res is None or alg_res is None:
            rej_gen += 1
            continue

        gen_results.append({
            "pair_id": item["pair_id"],
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

    # Impostor
    for item in benchmark_data["impostor_pairs"]:
        t1 = template_cache[item["path_1"]][cfg_name]
        t2 = template_cache[item["path_2"]][cfg_name]
        v1, v2 = t1["vec"], t2["vec"]
        m1, m2 = get_mask_fn(t1, t2)
        bps = t1["bits_per_sector"]

        raw_res = masked_hd(v1, v2, m1, m2, min_overlap=min_overlap)
        alg_res = aligned_masked_hd(v1, v2, m1, m2, bps, min_overlap=min_overlap)

        if raw_res is None or alg_res is None:
            rej_imp += 1
            continue

        imp_results.append({
            "pair_id": item["pair_id"],
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

    return {
        "genuine_norm_hd_mean": g_mu,
        "genuine_norm_hd_std": g_sig,
        "impostor_norm_hd_mean": i_mu,
        "impostor_norm_hd_std": i_sig,
        "separation_pct": sep,
        "d_prime": float(dp),
        "roc_auc": float(auc),
        "genuine_errs_mean": float(np.mean(gen_alg_errs)),
        "genuine_errs_std": float(np.std(gen_alg_errs)),
        "impostor_errs_mean": float(np.mean(imp_alg_errs)),
        "impostor_errs_std": float(np.std(imp_alg_errs)),
        "n_valid_mean": float(np.mean(gen_n_val)),
        "n_valid_std": float(np.std(gen_n_val)),
        "overlap_pct_mean": float(np.mean(gen_ovp)),
        "overlap_pct_std": float(np.std(gen_ovp)),
        "rejected_genuine": rej_gen,
        "rejected_impostor": rej_imp,
        "n_evaluated_genuine": len(gen_results),
        "n_evaluated_impostor": len(imp_results),
        "pair_records": {
            "genuine": gen_results,
            "impostor": imp_results
        }
    }

def main():
    benchmark_json_path = os.path.join(BASE_DIR, "data", "fixed_pairs_benchmark.json")
    print(f"[*] Loading fixed benchmark: {benchmark_json_path}")
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # 1. SHA-256 Checksum Verification
    print("[*] Verifying SHA-256 checksums on all benchmark images...")
    checked = 0
    for p_info in benchmark_data["genuine_pairs"] + benchmark_data["impostor_pairs"]:
        for k_p, k_h in [("path_1", "sha256_1"), ("path_2", "sha256_2")]:
            p = p_info[k_p]
            expected = p_info[k_h]
            actual = sha256_file(p)
            if actual != expected:
                raise ValueError(f"CRITICAL: Hash mismatch for {p}! Expected {expected}, got {actual}")
            checked += 1
    print(f"[+] Verified {checked} image checksums successfully. Benchmark integrity intact.\n")

    # 2. Extract templates for all unique images
    unique_paths = set()
    for item in benchmark_data["genuine_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])
    for item in benchmark_data["impostor_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])

    print(f"[*] Extracting all feature representations for {len(unique_paths)} unique benchmark images...")
    template_cache = {}
    for p in unique_paths:
        template_cache[p] = extract_step4_all_configs(p)
    print("[+] All templates extracted and cached.\n")

    # Delta* selected from development set tuning on fp_training:
    # On B2, delta* = 1.0 deg peaked at AUC=0.6768, d'=0.6890, overlap=53.0%, 0 rejections.
    delta_star_deg = 1.0
    delta_star_rad = np.radians(delta_star_deg)
    tau = 0.7

    print("=" * 115)
    print("                    STEP 4 COMPREHENSIVE EXPERIMENTAL EVALUATION (LOCKED 200 PAIRS)")
    print("=" * 115)

    all_results = {}

    # Experiment A: Step 3 Reference (256 bits, 128 cells)
    print("[*] Running Experiment A: Step 3 Reference (256-bit, 128 cells, min_overlap=0.50)...")
    def mask_exp_a(t1, t2):
        return (t1["cov"] >= tau).astype(np.uint8), (t2["cov"] >= tau).astype(np.uint8)
    res_a = evaluate_config(benchmark_data, template_cache, "exp_a", mask_exp_a, min_overlap=0.50)
    all_results["exp_a"] = res_a

    # Experiment B1: 512 bits, 128 cells (3-bit Gray code orientation)
    print("[*] Running Experiment B1: 512-bit, 128 cells (3-bit Gray code, min_overlap=0.50)...")
    def mask_exp_b1(t1, t2):
        return (t1["cov"] >= tau).astype(np.uint8), (t2["cov"] >= tau).astype(np.uint8)
    res_b1 = evaluate_config(benchmark_data, template_cache, "exp_b1", mask_exp_b1, min_overlap=0.50)
    all_results["exp_b1"] = res_b1

    # Experiment B2: 256 bits, 64 cells (Radial pooling + 3-bit Gray code)
    print("[*] Running Experiment B2: 256-bit, 64 cells (Radial pooling + 3-bit Gray code, min_overlap=0.50)...")
    def mask_exp_b2(t1, t2):
        return (t1["cov"] >= tau).astype(np.uint8), (t2["cov"] >= tau).astype(np.uint8)
    res_b2 = evaluate_config(benchmark_data, template_cache, "exp_b2", mask_exp_b2, min_overlap=0.50)
    all_results["exp_b2"] = res_b2

    # Experiment C1: 512 bits, 128 cells with boundary uncertainty masking (delta*=1.0 deg, min_overlap=0.40)
    print(f"[*] Running Experiment C1: 512-bit + Boundary Masking (delta*={delta_star_deg} deg, min_overlap=0.40)...")
    def mask_exp_c1(t1, t2):
        m1 = (t1["cov"] >= tau) & (t1["dbound"] >= delta_star_rad)
        m2 = (t2["cov"] >= tau) & (t2["dbound"] >= delta_star_rad)
        return m1.astype(np.uint8), m2.astype(np.uint8)
    res_c1 = evaluate_config(benchmark_data, template_cache, "exp_b1", mask_exp_c1, min_overlap=0.40)
    all_results["exp_c1"] = res_c1

    # Experiment C2: 256 bits, 64 cells with boundary uncertainty masking (delta*=1.0 deg, min_overlap=0.40)
    print(f"[*] Running Experiment C2: 256-bit + Radial Pooling + Boundary Masking (delta*={delta_star_deg} deg, min_overlap=0.40)...")
    def mask_exp_c2(t1, t2):
        m1 = (t1["cov"] >= tau) & (t1["dbound"] >= delta_star_rad)
        m2 = (t2["cov"] >= tau) & (t2["dbound"] >= delta_star_rad)
        return m1.astype(np.uint8), m2.astype(np.uint8)
    res_c2 = evaluate_config(benchmark_data, template_cache, "exp_b2", mask_exp_c2, min_overlap=0.40)
    all_results["exp_c2"] = res_c2

    # Print Full Comparative Table
    print("\n" + "=" * 125)
    print(f"{'Experiment':<10} {'Bits':<6} {'Cells':<7} {'Gen Aligned HD':<17} {'Imp Aligned HD':<17} {'Separation':<12} {'d-prime':<9} {'ROC-AUC':<9} {'Overlap':<9} {'N_val (Gen)':<13} {'Rejections'}")
    print("-" * 125)

    configs = [
        ("Exp A",  256, 128, res_a),
        ("Exp B1", 512, 128, res_b1),
        ("Exp B2", 256, 64,  res_b2),
        ("Exp C1", 512, 128, res_c1),
        ("Exp C2", 256, 64,  res_c2),
    ]

    for name, n_bits, n_cells, r in configs:
        g_str = f"{r['genuine_norm_hd_mean']:.2f} +- {r['genuine_norm_hd_std']:.2f}%"
        i_str = f"{r['impostor_norm_hd_mean']:.2f} +- {r['impostor_norm_hd_std']:.2f}%"
        sep_str = f"{r['separation_pct']:+.2f}%"
        dp_str = f"{r['d_prime']:.4f}"
        auc_str = f"{r['roc_auc']:.4f}"
        ovp_str = f"{r['overlap_pct_mean']:.1f}%"
        nval_str = f"{r['n_valid_mean']:.1f}/{n_bits}"
        rej_str = f"{r['rejected_genuine']} G / {r['rejected_impostor']} I"
        print(f"{name:<10} {n_bits:<6} {n_cells:<7} {g_str:<17} {i_str:<17} {sep_str:<12} {dp_str:<9} {auc_str:<9} {ovp_str:<9} {nval_str:<13} {rej_str}")

    print("=" * 125)

    # Save to JSON
    out_json_path = os.path.join(BASE_DIR, "results", "step4_evaluation_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Full Step 4 evaluation results saved to: {out_json_path}")

if __name__ == "__main__":
    main()

