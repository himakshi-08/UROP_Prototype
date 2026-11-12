"""
Step 1 Evaluation Script:
Evaluates the double-angle orientation field transformation against the exact
same 100 genuine and 100 impostor pairs in fixed_pairs_benchmark.json.
Compares metrics side-by-side with Step 0 baseline.
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
from features.template import iter_binary_alignments, BITS_PER_SECTOR, N_SECTORS

def compute_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def extract_template(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at {img_path}")
    prep = preprocess_fingerprint(img)
    minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
    tmpl = build_template(img, prep, minutiae, vector_bits=256)
    return tmpl.binary_vector

def compute_auc(genuine_dists, impostor_dists):
    thresholds = np.linspace(0, 256, 1025)
    tpr = []
    fpr = []
    n_gen = len(genuine_dists)
    n_imp = len(impostor_dists)

    for tau in thresholds:
        tp = np.sum(np.array(genuine_dists) <= tau) / float(n_gen)
        fp = np.sum(np.array(impostor_dists) <= tau) / float(n_imp)
        tpr.append(tp)
        fpr.append(fp)

    sorted_indices = np.argsort(fpr)
    sorted_fpr = np.array(fpr)[sorted_indices]
    sorted_tpr = np.array(tpr)[sorted_indices]
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc = float(trapz_func(sorted_tpr, sorted_fpr))
    return auc

def main():
    benchmark_json_path = os.path.join(BASE_DIR, "data", "fixed_pairs_benchmark.json")
    step0_json_path = os.path.join(BASE_DIR, "results", "step0_baseline_results.json")

    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    with open(step0_json_path, "r", encoding="utf-8") as f:
        step0_data = json.load(f)

    print("[*] Verifying benchmark file checksums...")
    checksum_failures = 0
    for p_info in benchmark_data["genuine_pairs"] + benchmark_data["impostor_pairs"]:
        for k_path, k_sha in [("path_1", "sha256_1"), ("path_2", "sha256_2")]:
            actual_sha = compute_file_sha256(p_info[k_path])
            if actual_sha != p_info[k_sha]:
                print(f"[!] Checksum mismatch on {p_info[k_path]}")
                checksum_failures += 1

    if checksum_failures == 0:
        print("[+] Checksum verification PASSED: 100% identical image files.")
    else:
        print(f"[!] WARNING: {checksum_failures} file checksum mismatches detected!")

    # Unique images
    unique_paths = set()
    for item in benchmark_data["genuine_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])
    for item in benchmark_data["impostor_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])

    print(f"[*] Extracting Step 1 templates for {len(unique_paths)} unique images...")
    template_cache = {}
    for p in unique_paths:
        template_cache[p] = extract_template(p)

    step1_genuine_results = []
    step1_impostor_results = []

    for item in benchmark_data["genuine_pairs"]:
        v1 = template_cache[item["path_1"]]
        v2 = template_cache[item["path_2"]]
        raw_hd = int(np.sum(v1 != v2))
        aligned_hd = min(int(np.sum(v1 != np.roll(v2, s * BITS_PER_SECTOR))) for s in range(N_SECTORS))
        step1_genuine_results.append({
            "pair_id": item["pair_id"],
            "subject_id": item["subject_id"],
            "raw_hd": raw_hd,
            "aligned_hd": aligned_hd
        })

    for item in benchmark_data["impostor_pairs"]:
        v1 = template_cache[item["path_1"]]
        v2 = template_cache[item["path_2"]]
        raw_hd = int(np.sum(v1 != v2))
        aligned_hd = min(int(np.sum(v1 != np.roll(v2, s * BITS_PER_SECTOR))) for s in range(N_SECTORS))
        step1_impostor_results.append({
            "pair_id": item["pair_id"],
            "subject_1": item["subject_1"],
            "subject_2": item["subject_2"],
            "raw_hd": raw_hd,
            "aligned_hd": aligned_hd
        })

    # Summary statistics
    gen_raw_1 = [r["raw_hd"] for r in step1_genuine_results]
    gen_alg_1 = [r["aligned_hd"] for r in step1_genuine_results]
    imp_raw_1 = [r["raw_hd"] for r in step1_impostor_results]
    imp_alg_1 = [r["aligned_hd"] for r in step1_impostor_results]

    d_prime_raw_1 = abs(np.mean(imp_raw_1) - np.mean(gen_raw_1)) / np.sqrt(0.5 * (np.std(gen_raw_1)**2 + np.std(imp_raw_1)**2))
    d_prime_alg_1 = abs(np.mean(imp_alg_1) - np.mean(gen_alg_1)) / np.sqrt(0.5 * (np.std(gen_alg_1)**2 + np.std(imp_alg_1)**2))

    auc_raw_1 = compute_auc(gen_raw_1, imp_raw_1)
    auc_alg_1 = compute_auc(gen_alg_1, imp_alg_1)

    # Step 0 baseline values
    s0_gen_raw = [r["raw_hd"] for r in step0_data["individual_genuine"]]
    s0_gen_alg = [r["aligned_hd"] for r in step0_data["individual_genuine"]]
    s0_imp_raw = [r["raw_hd"] for r in step0_data["individual_impostor"]]
    s0_imp_alg = [r["aligned_hd"] for r in step0_data["individual_impostor"]]
    s0_d_prime_raw = step0_data["decidability"]["raw_d_prime"]
    s0_d_prime_alg = step0_data["decidability"]["aligned_d_prime"]
    s0_auc_raw = step0_data["auc"]["raw_auc"]
    s0_auc_alg = step0_data["auc"]["aligned_auc"]

    # Save Step 1 Results
    step1_summary = {
        "genuine": {
            "raw": {"min": int(np.min(gen_raw_1)), "max": int(np.max(gen_raw_1)), "mean": float(np.mean(gen_raw_1)), "median": float(np.median(gen_raw_1)), "std": float(np.std(gen_raw_1))},
            "aligned": {"min": int(np.min(gen_alg_1)), "max": int(np.max(gen_alg_1)), "mean": float(np.mean(gen_alg_1)), "median": float(np.median(gen_alg_1)), "std": float(np.std(gen_alg_1))}
        },
        "impostor": {
            "raw": {"min": int(np.min(imp_raw_1)), "max": int(np.max(imp_raw_1)), "mean": float(np.mean(imp_raw_1)), "median": float(np.median(imp_raw_1)), "std": float(np.std(imp_raw_1))},
            "aligned": {"min": int(np.min(imp_alg_1)), "max": int(np.max(imp_alg_1)), "mean": float(np.mean(imp_alg_1)), "median": float(np.median(imp_alg_1)), "std": float(np.std(imp_alg_1))}
        },
        "decidability": {
            "raw_d_prime": float(d_prime_raw_1),
            "aligned_d_prime": float(d_prime_alg_1)
        },
        "auc": {
            "raw_auc": float(auc_raw_1),
            "aligned_auc": float(auc_alg_1)
        },
        "individual_genuine": step1_genuine_results,
        "individual_impostor": step1_impostor_results
    }

    out_path = os.path.join(BASE_DIR, "results", "step1_evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(step1_summary, f, indent=2)

    print(f"\n[+] Saved Step 1 results to: {out_path}")

    print("\n" + "=" * 90)
    print("        ISOLATED ABLATION COMPARISON: STEP 0 (LOCKED BASELINE) vs STEP 1 (DOUBLE-ANGLE)")
    print("=" * 90)
    print(f"{'Metric':<32} {'Step 0 (Baseline)':<25} {'Step 1 (Double-Angle)':<25} {'Delta'}")
    print("-" * 90)
    print(f"{'Genuine Raw HD (Mean+-Std)':<32} {f'{np.mean(s0_gen_raw):.2f} +- {np.std(s0_gen_raw):.2f}':<25} {f'{np.mean(gen_raw_1):.2f} +- {np.std(gen_raw_1):.2f}':<25} {np.mean(gen_raw_1)-np.mean(s0_gen_raw):+.2f}")
    print(f"{'Genuine Aligned HD (Mean+-Std)':<32} {f'{np.mean(s0_gen_alg):.2f} +- {np.std(s0_gen_alg):.2f}':<25} {f'{np.mean(gen_alg_1):.2f} +- {np.std(gen_alg_1):.2f}':<25} {np.mean(gen_alg_1)-np.mean(s0_gen_alg):+.2f}")
    print(f"{'Genuine Min Aligned HD':<32} {int(np.min(s0_gen_alg)):<25} {int(np.min(gen_alg_1)):<25} {int(np.min(gen_alg_1))-int(np.min(s0_gen_alg)):+d}")
    print(f"{'Impostor Raw HD (Mean+-Std)':<32} {f'{np.mean(s0_imp_raw):.2f} +- {np.std(s0_imp_raw):.2f}':<25} {f'{np.mean(imp_raw_1):.2f} +- {np.std(imp_raw_1):.2f}':<25} {np.mean(imp_raw_1)-np.mean(s0_imp_raw):+.2f}")
    print(f"{'Impostor Aligned HD (Mean+-Std)':<32} {f'{np.mean(s0_imp_alg):.2f} +- {np.std(s0_imp_alg):.2f}':<25} {f'{np.mean(imp_alg_1):.2f} +- {np.std(imp_alg_1):.2f}':<25} {np.mean(imp_alg_1)-np.mean(s0_imp_alg):+.2f}")
    print(f"{'Raw Decidability (d\')':<32} {s0_d_prime_raw:<25.4f} {d_prime_raw_1:<25.4f} {d_prime_raw_1-s0_d_prime_raw:+.4f}")
    print(f"{'Aligned Decidability (d\')':<32} {s0_d_prime_alg:<25.4f} {d_prime_alg_1:<25.4f} {d_prime_alg_1-s0_d_prime_alg:+.4f}")
    print(f"{'Raw AUC':<32} {s0_auc_raw:<25.4f} {auc_raw_1:<25.4f} {auc_raw_1-s0_auc_raw:+.4f}")
    print(f"{'Aligned AUC':<32} {s0_auc_alg:<25.4f} {auc_alg_1:<25.4f} {auc_alg_1-s0_auc_alg:+.4f}")
    print("=" * 90)

if __name__ == "__main__":
    main()

