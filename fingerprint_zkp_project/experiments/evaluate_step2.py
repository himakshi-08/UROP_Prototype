"""
Step 2 Evaluation Script: Mask-Aware Polar Cell Validity.
Evaluates cell-level mask coverage and masked Hamming distance across the exact
100 genuine and 100 impostor pairs in fixed_pairs_benchmark.json.
Tests coverage thresholds tau in {0.5, 0.7, 0.8}.
Computes ROC-AUC using -HD as the similarity score (correct direction).
Compares results side-by-side with Step 0 and Step 1.
"""

import os
import sys
import json
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing import preprocess_fingerprint
from features import extract_minutiae, build_template
from features.template import BiometricTemplate, N_SECTORS, BITS_PER_SECTOR

def extract_template_and_coverage(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at {img_path}")
    prep = preprocess_fingerprint(img)
    minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
    tmpl = build_template(img, prep, minutiae, vector_bits=256)
    return tmpl.binary_vector, tmpl.coverage_vector

def compute_roc_auc_similarity(genuine_scores, impostor_scores):
    """
    Computes AUC where higher score = more likely genuine.
    Scores can be negative distances (-HD) or similarities.
    """
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

    # Sort by FPR ascending for numerical integration
    sorted_idx = np.argsort(fpr)
    s_fpr = np.array(fpr)[sorted_idx]
    s_tpr = np.array(tpr)[sorted_idx]
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc = float(trapz_func(s_tpr, s_fpr))
    return auc

def run_evaluation_for_tau(benchmark_data, template_cache, tau=0.7):
    gen_results = []
    imp_results = []
    rejected_gen = 0
    rejected_imp = 0

    # 1. Genuine Pairs
    for item in benchmark_data["genuine_pairs"]:
        v1, cov1 = template_cache[item["path_1"]]
        v2, cov2 = template_cache[item["path_2"]]
        m1 = (cov1 >= tau).astype(np.uint8)
        m2 = (cov2 >= tau).astype(np.uint8)

        # Raw masked HD (no cyclic shift)
        raw_res = BiometricTemplate.masked_hamming_distance(v1, v2, m1, m2)
        # Aligned masked HD (over 16 cyclic shifts)
        alg_res = BiometricTemplate.aligned_masked_hamming_distance(v1, v2, m1, m2)

        if raw_res is None or alg_res is None:
            rejected_gen += 1
            continue

        gen_results.append({
            "pair_id": item["pair_id"],
            "subject_id": item["subject_id"],
            "raw_equiv_bits": raw_res["equiv_bits"],
            "raw_norm_hd": raw_res["norm_hd"],
            "raw_overlap": raw_res["overlap"],
            "aligned_equiv_bits": alg_res["equiv_bits"],
            "aligned_norm_hd": alg_res["norm_hd"],
            "aligned_overlap": alg_res["overlap"]
        })

    # 2. Impostor Pairs
    for item in benchmark_data["impostor_pairs"]:
        v1, cov1 = template_cache[item["path_1"]]
        v2, cov2 = template_cache[item["path_2"]]
        m1 = (cov1 >= tau).astype(np.uint8)
        m2 = (cov2 >= tau).astype(np.uint8)

        raw_res = BiometricTemplate.masked_hamming_distance(v1, v2, m1, m2)
        alg_res = BiometricTemplate.aligned_masked_hamming_distance(v1, v2, m1, m2)

        if raw_res is None or alg_res is None:
            rejected_imp += 1
            continue

        imp_results.append({
            "pair_id": item["pair_id"],
            "subject_1": item["subject_1"],
            "subject_2": item["subject_2"],
            "raw_equiv_bits": raw_res["equiv_bits"],
            "raw_norm_hd": raw_res["norm_hd"],
            "raw_overlap": raw_res["overlap"],
            "aligned_equiv_bits": alg_res["equiv_bits"],
            "aligned_norm_hd": alg_res["norm_hd"],
            "aligned_overlap": alg_res["overlap"]
        })

    gen_raw_bits = [r["raw_equiv_bits"] for r in gen_results]
    gen_alg_bits = [r["aligned_equiv_bits"] for r in gen_results]
    imp_raw_bits = [r["raw_equiv_bits"] for r in imp_results]
    imp_alg_bits = [r["aligned_equiv_bits"] for r in imp_results]

    gen_raw_ovp = [r["raw_overlap"] for r in gen_results]
    gen_alg_ovp = [r["aligned_overlap"] for r in gen_results]
    imp_raw_ovp = [r["raw_overlap"] for r in imp_results]
    imp_alg_ovp = [r["aligned_overlap"] for r in imp_results]

    # Decidability (d')
    d_prime_raw = abs(np.mean(imp_raw_bits) - np.mean(gen_raw_bits)) / np.sqrt(0.5 * (np.std(gen_raw_bits)**2 + np.std(imp_raw_bits)**2))
    d_prime_alg = abs(np.mean(imp_alg_bits) - np.mean(gen_alg_bits)) / np.sqrt(0.5 * (np.std(gen_alg_bits)**2 + np.std(imp_alg_bits)**2))

    # ROC AUC using similarity = -equiv_bits (lower distance = higher similarity!)
    auc_raw = compute_roc_auc_similarity([-b for b in gen_raw_bits], [-b for b in imp_raw_bits])
    auc_alg = compute_roc_auc_similarity([-b for b in gen_alg_bits], [-b for b in imp_alg_bits])

    return {
        "tau": tau,
        "genuine": {
            "raw": {"mean": float(np.mean(gen_raw_bits)), "std": float(np.std(gen_raw_bits)), "min": float(np.min(gen_raw_bits)), "max": float(np.max(gen_raw_bits)), "mean_overlap": float(np.mean(gen_raw_ovp))},
            "aligned": {"mean": float(np.mean(gen_alg_bits)), "std": float(np.std(gen_alg_bits)), "min": float(np.min(gen_alg_bits)), "max": float(np.max(gen_alg_bits)), "mean_overlap": float(np.mean(gen_alg_ovp))}
        },
        "impostor": {
            "raw": {"mean": float(np.mean(imp_raw_bits)), "std": float(np.std(imp_raw_bits)), "min": float(np.min(imp_raw_bits)), "max": float(np.max(imp_raw_bits)), "mean_overlap": float(np.mean(imp_raw_ovp))},
            "aligned": {"mean": float(np.mean(imp_alg_bits)), "std": float(np.std(imp_alg_bits)), "min": float(np.min(imp_alg_bits)), "max": float(np.max(imp_alg_bits)), "mean_overlap": float(np.mean(imp_alg_ovp))}
        },
        "d_prime": {"raw": float(d_prime_raw), "aligned": float(d_prime_alg)},
        "auc": {"raw": float(auc_raw), "aligned": float(auc_alg)},
        "rejections": {"genuine": rejected_gen, "impostor": rejected_imp, "total_valid_genuine": len(gen_results), "total_valid_impostor": len(imp_results)},
        "individual_genuine": gen_results,
        "individual_impostor": imp_results
    }

def main():
    benchmark_json_path = os.path.join(BASE_DIR, "data", "fixed_pairs_benchmark.json")
    step0_json_path = os.path.join(BASE_DIR, "results", "step0_baseline_results.json")
    step1_json_path = os.path.join(BASE_DIR, "results", "step1_evaluation_results.json")

    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    with open(step0_json_path, "r", encoding="utf-8") as f:
        step0_data = json.load(f)

    with open(step1_json_path, "r", encoding="utf-8") as f:
        step1_data = json.load(f)

    # Unique images
    unique_paths = set()
    for item in benchmark_data["genuine_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])
    for item in benchmark_data["impostor_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])

    print(f"[*] Extracting Step 2 templates & coverage for {len(unique_paths)} unique images...")
    template_cache = {}
    for p in unique_paths:
        template_cache[p] = extract_template_and_coverage(p)

    # Test tau = 0.5, 0.7, 0.8
    results_by_tau = {}
    for tau in [0.5, 0.7, 0.8]:
        print(f"[*] Evaluating Step 2 for tau = {tau}...")
        results_by_tau[str(tau)] = run_evaluation_for_tau(benchmark_data, template_cache, tau=tau)

    # Primary result is tau = 0.7
    primary_res = results_by_tau["0.7"]

    # Save to json
    out_path = os.path.join(BASE_DIR, "results", "step2_evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_by_tau, f, indent=2)

    print(f"\n[+] Saved Step 2 evaluation results to: {out_path}")

    # Baseline and Step 1 values for comparison
    s0_gen_alg = step0_data["genuine"]["aligned"]
    s0_imp_alg = step0_data["impostor"]["aligned"]
    s0_d_alg   = step0_data["decidability"]["aligned_d_prime"]
    s0_auc_alg = step0_data["auc"]["aligned_auc"]

    s1_gen_alg = step1_data["genuine"]["aligned"]
    s1_imp_alg = step1_data["impostor"]["aligned"]
    s1_d_alg   = step1_data["decidability"]["aligned_d_prime"]
    s1_auc_alg = step1_data["auc"]["aligned_auc"]

    s2_gen_alg = primary_res["genuine"]["aligned"]
    s2_imp_alg = primary_res["impostor"]["aligned"]
    s2_d_alg   = primary_res["d_prime"]["aligned"]
    s2_auc_alg = primary_res["auc"]["aligned"]

    s0_gen_str = f"{s0_gen_alg['mean']:.2f} +- {s0_gen_alg['std']:.2f}"
    s1_gen_str = f"{s1_gen_alg['mean']:.2f} +- {s1_gen_alg['std']:.2f}"
    s2_gen_str = f"{s2_gen_alg['mean']:.2f} +- {s2_gen_alg['std']:.2f}"

    s0_imp_str = f"{s0_imp_alg['mean']:.2f} +- {s0_imp_alg['std']:.2f}"
    s1_imp_str = f"{s1_imp_alg['mean']:.2f} +- {s1_imp_alg['std']:.2f}"
    s2_imp_str = f"{s2_imp_alg['mean']:.2f} +- {s2_imp_alg['std']:.2f}"

    ovp_str = f"{s2_gen_alg['mean_overlap']:.3f} Gen / {s2_imp_alg['mean_overlap']:.3f} Imp"
    rej_str = f"{primary_res['rejections']['genuine']} Gen / {primary_res['rejections']['impostor']} Imp"

    print("\n" + "=" * 95)
    print("      ISOLATED PROGRESSION: STEP 0 (BASELINE) -> STEP 1 (DOUBLE-ANGLE) -> STEP 2 (MASK-AWARE)")
    print("                      Primary Step 2 Configuration: Coverage Tau = 0.7")
    print("=" * 95)
    print(f"{'Metric':<34} {'Step 0 (Locked)':<20} {'Step 1 (Angle)':<20} {'Step 2 (Masked)':<20}")
    print("-" * 95)
    print(f"{'Genuine Aligned HD (Bits/256)':<34} {s0_gen_str:<20} {s1_gen_str:<20} {s2_gen_str:<20}")
    print(f"{'Genuine Min Aligned HD':<34} {s0_gen_alg['min']:<20} {s1_gen_alg['min']:<20} {s2_gen_alg['min']:<20.1f}")
    print(f"{'Impostor Aligned HD (Bits/256)':<34} {s0_imp_str:<20} {s1_imp_str:<20} {s2_imp_str:<20}")
    print(f"{'Aligned Decidability (d\')':<34} {s0_d_alg:<20.4f} {s1_d_alg:<20.4f} {s2_d_alg:<20.4f}")
    print(f"{'Aligned ROC-AUC':<34} {s0_auc_alg:<20.4f} {s1_auc_alg:<20.4f} {s2_auc_alg:<20.4f}")
    print(f"{'Mean Bit Overlap Fraction':<34} {'1.000 (Fixed)':<20} {'1.000 (Fixed)':<20} {ovp_str:<20}")
    print(f"{'Rejected Comparisons (<15% ovp)':<34} {'0':<20} {'0':<20} {rej_str:<20}")

    print("\n" + "=" * 95)
    print("                     SENSITIVITY ANALYSIS ACROSS COVERAGE TAU")
    print("=" * 95)
    print(f"{'Tau':<8} {'Gen Aligned HD':<18} {'Imp Aligned HD':<18} {'d-prime':<12} {'ROC-AUC':<12} {'Gen Overlap':<14} {'Rejections'}")
    print("-" * 95)
    for t_str, res in results_by_tau.items():
        g_s = f"{res['genuine']['aligned']['mean']:.2f} +- {res['genuine']['aligned']['std']:.2f}"
        i_s = f"{res['impostor']['aligned']['mean']:.2f} +- {res['impostor']['aligned']['std']:.2f}"
        d_s = f"{res['d_prime']['aligned']:.4f}"
        a_s = f"{res['auc']['aligned']:.4f}"
        o_s = f"{res['genuine']['aligned']['mean_overlap']*100:.1f}% ({res['genuine']['aligned']['mean_overlap']*256:.0f} bits)"
        rej = f"{res['rejections']['genuine']} G / {res['rejections']['impostor']} I"
        print(f"{t_str:<8} {g_s:<18} {i_s:<18} {d_s:<12} {a_s:<12} {o_s:<14} {rej}")
    print("=" * 95)

if __name__ == "__main__":
    main()
