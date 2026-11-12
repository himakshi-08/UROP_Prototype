"""
Step 3 Evaluation Script:
Evaluates the 8-orientation quadrature Gabor filter bank on the exact same locked
100 genuine and 100 impostor pairs in fixed_pairs_benchmark.json.
Enforces min_overlap = 0.50 guardrail.
Records selected shift, selected overlap, and minimum masked HD for every pair.
Compares side-by-side: Step 0, Step 1, Step 2, and Step 3.
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
from features.template import BiometricTemplate

def extract_template_and_coverage(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at {img_path}")
    prep = preprocess_fingerprint(img)
    minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
    tmpl = build_template(img, prep, minutiae, vector_bits=256)
    return tmpl.binary_vector, tmpl.coverage_vector

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
    auc = float(trapz_func(s_tpr, s_fpr))
    return auc

def main():
    benchmark_json_path = os.path.join(BASE_DIR, "data", "fixed_pairs_benchmark.json")
    step0_json_path = os.path.join(BASE_DIR, "results", "step0_baseline_results.json")
    step1_json_path = os.path.join(BASE_DIR, "results", "step1_evaluation_results.json")
    step2_json_path = os.path.join(BASE_DIR, "results", "step2_evaluation_results.json")

    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)
    with open(step0_json_path, "r", encoding="utf-8") as f:
        step0_data = json.load(f)
    with open(step1_json_path, "r", encoding="utf-8") as f:
        step1_data = json.load(f)
    with open(step2_json_path, "r", encoding="utf-8") as f:
        step2_data = json.load(f)["0.7"]

    unique_paths = set()
    for item in benchmark_data["genuine_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])
    for item in benchmark_data["impostor_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])

    print(f"[*] Extracting Step 3 (Quadrature Gabor Bank) templates for {len(unique_paths)} unique images...")
    template_cache = {}
    for p in unique_paths:
        template_cache[p] = extract_template_and_coverage(p)

    tau = 0.7
    min_overlap = 0.50

    gen_results = []
    imp_results = []
    rej_gen = 0
    rej_imp = 0

    # 1. Genuine Pairs
    for item in benchmark_data["genuine_pairs"]:
        v1, cov1 = template_cache[item["path_1"]]
        v2, cov2 = template_cache[item["path_2"]]
        m1 = (cov1 >= tau).astype(np.uint8)
        m2 = (cov2 >= tau).astype(np.uint8)

        raw_res = BiometricTemplate.masked_hamming_distance(v1, v2, m1, m2, min_overlap=min_overlap)
        alg_res = BiometricTemplate.aligned_masked_hamming_distance(v1, v2, m1, m2, min_overlap=min_overlap)

        if raw_res is None or alg_res is None:
            rej_gen += 1
            continue

        gen_results.append({
            "pair_id": item["pair_id"],
            "subject_id": item["subject_id"],
            "raw_norm_hd": raw_res["norm_hd"],
            "raw_equiv_bits": raw_res["equiv_bits"],
            "raw_overlap": raw_res["overlap"],
            "aligned_norm_hd": alg_res["norm_hd"],
            "aligned_equiv_bits": alg_res["equiv_bits"],
            "aligned_overlap": alg_res["overlap"],
            "selected_shift": alg_res.get("shift_sector", 0)
        })

    # 2. Impostor Pairs
    for item in benchmark_data["impostor_pairs"]:
        v1, cov1 = template_cache[item["path_1"]]
        v2, cov2 = template_cache[item["path_2"]]
        m1 = (cov1 >= tau).astype(np.uint8)
        m2 = (cov2 >= tau).astype(np.uint8)

        raw_res = BiometricTemplate.masked_hamming_distance(v1, v2, m1, m2, min_overlap=min_overlap)
        alg_res = BiometricTemplate.aligned_masked_hamming_distance(v1, v2, m1, m2, min_overlap=min_overlap)

        if raw_res is None or alg_res is None:
            rej_imp += 1
            continue

        imp_results.append({
            "pair_id": item["pair_id"],
            "subject_1": item["subject_1"],
            "subject_2": item["subject_2"],
            "raw_norm_hd": raw_res["norm_hd"],
            "raw_equiv_bits": raw_res["equiv_bits"],
            "raw_overlap": raw_res["overlap"],
            "aligned_norm_hd": alg_res["norm_hd"],
            "aligned_equiv_bits": alg_res["equiv_bits"],
            "aligned_overlap": alg_res["overlap"],
            "selected_shift": alg_res.get("shift_sector", 0)
        })

    gen_raw_b = [r["raw_equiv_bits"] for r in gen_results]
    gen_alg_b = [r["aligned_equiv_bits"] for r in gen_results]
    imp_raw_b = [r["raw_equiv_bits"] for r in imp_results]
    imp_alg_b = [r["aligned_equiv_bits"] for r in imp_results]

    gen_ovp = [r["aligned_overlap"] for r in gen_results]
    imp_ovp = [r["aligned_overlap"] for r in imp_results]

    d_prime_raw = abs(np.mean(imp_raw_b) - np.mean(gen_raw_b)) / np.sqrt(0.5 * (np.std(gen_raw_b)**2 + np.std(imp_raw_b)**2))
    d_prime_alg = abs(np.mean(imp_alg_b) - np.mean(gen_alg_b)) / np.sqrt(0.5 * (np.std(gen_alg_b)**2 + np.std(imp_alg_b)**2))

    auc_raw = compute_roc_auc_similarity([-b for b in gen_raw_b], [-b for b in imp_raw_b])
    auc_alg = compute_roc_auc_similarity([-b for b in gen_alg_b], [-b for b in imp_alg_b])

    step3_summary = {
        "genuine": {
            "raw": {"mean": float(np.mean(gen_raw_b)), "std": float(np.std(gen_raw_b)), "min": float(np.min(gen_raw_b)), "max": float(np.max(gen_raw_b))},
            "aligned": {"mean": float(np.mean(gen_alg_b)), "std": float(np.std(gen_alg_b)), "min": float(np.min(gen_alg_b)), "max": float(np.max(gen_alg_b)), "mean_overlap": float(np.mean(gen_ovp))}
        },
        "impostor": {
            "raw": {"mean": float(np.mean(imp_raw_b)), "std": float(np.std(imp_raw_b)), "min": float(np.min(imp_raw_b)), "max": float(np.max(imp_raw_b))},
            "aligned": {"mean": float(np.mean(imp_alg_b)), "std": float(np.std(imp_alg_b)), "min": float(np.min(imp_alg_b)), "max": float(np.max(imp_alg_b)), "mean_overlap": float(np.mean(imp_ovp))}
        },
        "d_prime": {"raw": float(d_prime_raw), "aligned": float(d_prime_alg)},
        "auc": {"raw": float(auc_raw), "aligned": float(auc_alg)},
        "rejections": {"genuine": rej_gen, "impostor": rej_imp, "total_valid_genuine": len(gen_results), "total_valid_impostor": len(imp_results)},
        "individual_genuine": gen_results,
        "individual_impostor": imp_results
    }

    out_json = os.path.join(BASE_DIR, "results", "step3_evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(step3_summary, f, indent=2)

    print(f"\n[+] Saved Step 3 results to: {out_json}")

    # Format side-by-side progression table
    s0_g = step0_data["genuine"]["aligned"]
    s0_i = step0_data["impostor"]["aligned"]
    s1_g = step1_data["genuine"]["aligned"]
    s1_i = step1_data["impostor"]["aligned"]
    s2_g = step2_data["genuine"]["aligned"]
    s2_i = step2_data["impostor"]["aligned"]
    s3_g = step3_summary["genuine"]["aligned"]
    s3_i = step3_summary["impostor"]["aligned"]

    sep0 = s0_i["mean"] - s0_g["mean"]
    sep1 = s1_i["mean"] - s1_g["mean"]
    sep2 = s2_i["mean"] - s2_g["mean"]
    sep3 = s3_i["mean"] - s3_g["mean"]

    s0_g_str = f"{s0_g['mean']:.2f} +- {s0_g['std']:.2f}"
    s1_g_str = f"{s1_g['mean']:.2f} +- {s1_g['std']:.2f}"
    s2_g_str = f"{s2_g['mean']:.2f} +- {s2_g['std']:.2f}"
    s3_g_str = f"{s3_g['mean']:.2f} +- {s3_g['std']:.2f}"

    s0_i_str = f"{s0_i['mean']:.2f} +- {s0_i['std']:.2f}"
    s1_i_str = f"{s1_i['mean']:.2f} +- {s1_i['std']:.2f}"
    s2_i_str = f"{s2_i['mean']:.2f} +- {s2_i['std']:.2f}"
    s3_i_str = f"{s3_i['mean']:.2f} +- {s3_i['std']:.2f}"

    s0_d = f"{step0_data['decidability']['aligned_d_prime']:.4f}"
    s1_d = f"{step1_data['decidability']['aligned_d_prime']:.4f}"
    s2_d = f"{step2_data['d_prime']['aligned']:.4f}"
    s3_d = f"{step3_summary['d_prime']['aligned']:.4f}"

    s0_a = f"{step0_data['auc']['aligned_auc']:.4f}"
    s1_a = f"{step1_data['auc']['aligned_auc']:.4f}"
    s2_a = f"{step2_data['auc']['aligned']:.4f}"
    s3_a = f"{step3_summary['auc']['aligned']:.4f}"

    s2_go = f"{s2_g['mean_overlap']*100:.1f}%"
    s3_go = f"{s3_g['mean_overlap']*100:.1f}%"
    s2_io = f"{s2_i['mean_overlap']*100:.1f}%"
    s3_io = f"{s3_i['mean_overlap']*100:.1f}%"
    rej_str = f"{rej_gen} Gen / {rej_imp} Imp"

    print("\n" + "=" * 105)
    print("      FOUR-STAGE PROGRESSION: STEP 0 (BASE) -> STEP 1 (ANGLE) -> STEP 2 (MASK) -> STEP 3 (GABOR)")
    print("                              Locked Benchmark: 100 Genuine vs 100 Impostor Pairs")
    print("=" * 105)
    print(f"{'Metric':<30} {'Step 0 (Base)':<18} {'Step 1 (Angle)':<18} {'Step 2 (Masked)':<18} {'Step 3 (Gabor)':<18}")
    print("-" * 105)
    print(f"{'Genuine Aligned HD (Bits)':<30} {s0_g_str:<18} {s1_g_str:<18} {s2_g_str:<18} {s3_g_str:<18}")
    print(f"{'Impostor Aligned HD (Bits)':<30} {s0_i_str:<18} {s1_i_str:<18} {s2_i_str:<18} {s3_i_str:<18}")
    print(f"{'Separation (Imp - Gen)':<30} {f'{sep0:+.2f} bits':<18} {f'{sep1:+.2f} bits':<18} {f'{sep2:+.2f} bits':<18} {f'{sep3:+.2f} bits':<18}")
    print(f"{'Aligned Decidability (d\')':<30} {s0_d:<18} {s1_d:<18} {s2_d:<18} {s3_d:<18}")
    print(f"{'Aligned ROC-AUC':<30} {s0_a:<18} {s1_a:<18} {s2_a:<18} {s3_a:<18}")
    print(f"{'Mean Genuine Overlap':<30} {'100.0% (Forced)':<18} {'100.0% (Forced)':<18} {s2_go:<18} {s3_go:<18}")
    print(f"{'Mean Impostor Overlap':<30} {'100.0% (Forced)':<18} {'100.0% (Forced)':<18} {s2_io:<18} {s3_io:<18}")
    print(f"{'Rejected Pairs (<50% ovp)':<30} {'0':<18} {'0':<18} {'0':<18} {rej_str:<18}")
    print("=" * 105)

if __name__ == "__main__":
    main()
