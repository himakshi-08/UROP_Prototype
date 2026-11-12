"""
Step 0: Benchmark Pair Locking and Baseline Measurement.
Generates fixed_pairs_benchmark.json with 100 genuine and 100 impostor pairs,
including image paths and SHA-256 file checksums, under a fixed random seed.
Evaluates the CURRENT UNCHANGED implementation against these pairs and saves
individual pair results to step0_baseline_results.json.
"""

import os
import sys
import glob
import json
import hashlib
import random
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.dataset_loader import default_split_dirs
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
    # For distance metrics: Genuine has smaller distance, Impostor has larger distance
    # Threshold sweep from 0 to 256
    thresholds = np.linspace(0, 256, 1025)
    tpr = []
    fpr = []
    n_gen = len(genuine_dists)
    n_imp = len(impostor_dists)

    for tau in thresholds:
        # Genuine accepted if dist <= tau
        tp = np.sum(np.array(genuine_dists) <= tau) / float(n_gen)
        # Impostor accepted if dist <= tau
        fp = np.sum(np.array(impostor_dists) <= tau) / float(n_imp)
        tpr.append(tp)
        fpr.append(fp)

    # Sort by FPR ascending for numerical trapezoidal integration
    sorted_indices = np.argsort(fpr)
    sorted_fpr = np.array(fpr)[sorted_indices]
    sorted_tpr = np.array(tpr)[sorted_indices]
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc = float(trapz_func(sorted_tpr, sorted_fpr))
    return auc, thresholds.tolist(), tpr, fpr

def main():
    random.seed(42)
    np.random.seed(42)

    splits = default_split_dirs()
    test_dir = splits["test"]
    print(f"[*] Discovering images in test directory: {test_dir}")

    subject_folders = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    subjects_dict = {}
    for sid in subject_folders:
        spath = os.path.join(test_dir, sid)
        img_files = sorted(glob.glob(os.path.join(spath, "*.[bB][mM][pP]")) +
                           glob.glob(os.path.join(spath, "*.[tT][iI][fF]")) +
                           glob.glob(os.path.join(spath, "*.[pP][nN][gG]")))
        if len(img_files) >= 2:
            subjects_dict[sid] = img_files

    print(f"[+] Found {len(subjects_dict)} subjects with >= 2 impressions.")

    # 1. Generate 100 Genuine Pairs
    all_possible_genuine = []
    for sid, files in subjects_dict.items():
        n = len(files)
        for i in range(n):
            for j in range(i + 1, n):
                all_possible_genuine.append((sid, files[i], files[j]))

    random.shuffle(all_possible_genuine)
    selected_genuine = all_possible_genuine[:100]

    # 2. Generate 100 Impostor Pairs
    all_possible_impostor = []
    sids = list(subjects_dict.keys())
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            s1, s2 = sids[i], sids[j]
            # Take one random impression from s1 and one from s2
            all_possible_impostor.append((s1, s2, subjects_dict[s1][0], subjects_dict[s2][0]))

    random.shuffle(all_possible_impostor)
    selected_impostor = all_possible_impostor[:100]

    # Structure fixed_pairs_benchmark
    benchmark_data = {
        "metadata": {
            "random_seed": 42,
            "dataset_root": test_dir,
            "num_genuine": len(selected_genuine),
            "num_impostor": len(selected_impostor),
        },
        "genuine_pairs": [],
        "impostor_pairs": []
    }

    print("[*] Computing file checksums for genuine pairs...")
    for idx, (sid, p1, p2) in enumerate(selected_genuine):
        cs1 = compute_file_sha256(p1)
        cs2 = compute_file_sha256(p2)
        benchmark_data["genuine_pairs"].append({
            "pair_id": idx,
            "subject_id": sid,
            "path_1": p1,
            "sha256_1": cs1,
            "path_2": p2,
            "sha256_2": cs2
        })

    print("[*] Computing file checksums for impostor pairs...")
    for idx, (s1, s2, p1, p2) in enumerate(selected_impostor):
        cs1 = compute_file_sha256(p1)
        cs2 = compute_file_sha256(p2)
        benchmark_data["impostor_pairs"].append({
            "pair_id": idx,
            "subject_1": s1,
            "subject_2": s2,
            "path_1": p1,
            "sha256_1": cs1,
            "path_2": p2,
            "sha256_2": cs2
        })

    benchmark_json_path = os.path.join(BASE_DIR, "data", "fixed_pairs_benchmark.json")
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"[+] Saved fixed benchmark pairs to: {benchmark_json_path}")

    # 3. Run Step 0 (Current Baseline Evaluation)
    print("\n[*] Evaluating Step 0 (Current Unchanged Implementation)...")
    
    # Cache unique image templates to avoid recomputing identical images
    unique_paths = set()
    for item in benchmark_data["genuine_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])
    for item in benchmark_data["impostor_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])

    print(f"[*] Extracting templates for {len(unique_paths)} unique images...")
    template_cache = {}
    for p in unique_paths:
        template_cache[p] = extract_template(p)

    step0_genuine_results = []
    step0_impostor_results = []

    for item in benchmark_data["genuine_pairs"]:
        v1 = template_cache[item["path_1"]]
        v2 = template_cache[item["path_2"]]
        raw_hd = int(np.sum(v1 != v2))
        aligned_hd = min(int(np.sum(v1 != np.roll(v2, s * BITS_PER_SECTOR))) for s in range(N_SECTORS))
        step0_genuine_results.append({
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
        step0_impostor_results.append({
            "pair_id": item["pair_id"],
            "subject_1": item["subject_1"],
            "subject_2": item["subject_2"],
            "raw_hd": raw_hd,
            "aligned_hd": aligned_hd
        })

    # Summary statistics
    gen_raw = [r["raw_hd"] for r in step0_genuine_results]
    gen_alg = [r["aligned_hd"] for r in step0_genuine_results]
    imp_raw = [r["raw_hd"] for r in step0_impostor_results]
    imp_alg = [r["aligned_hd"] for r in step0_impostor_results]

    d_prime_raw = abs(np.mean(imp_raw) - np.mean(gen_raw)) / np.sqrt(0.5 * (np.std(gen_raw)**2 + np.std(imp_raw)**2))
    d_prime_alg = abs(np.mean(imp_alg) - np.mean(gen_alg)) / np.sqrt(0.5 * (np.std(gen_alg)**2 + np.std(imp_alg)**2))

    auc_raw, _, _, _ = compute_auc(gen_raw, imp_raw)
    auc_alg, _, _, _ = compute_auc(gen_alg, imp_alg)

    step0_summary = {
        "genuine": {
            "raw": {"min": int(np.min(gen_raw)), "max": int(np.max(gen_raw)), "mean": float(np.mean(gen_raw)), "median": float(np.median(gen_raw)), "std": float(np.std(gen_raw))},
            "aligned": {"min": int(np.min(gen_alg)), "max": int(np.max(gen_alg)), "mean": float(np.mean(gen_alg)), "median": float(np.median(gen_alg)), "std": float(np.std(gen_alg))}
        },
        "impostor": {
            "raw": {"min": int(np.min(imp_raw)), "max": int(np.max(imp_raw)), "mean": float(np.mean(imp_raw)), "median": float(np.median(imp_raw)), "std": float(np.std(imp_raw))},
            "aligned": {"min": int(np.min(imp_alg)), "max": int(np.max(imp_alg)), "mean": float(np.mean(imp_alg)), "median": float(np.median(imp_alg)), "std": float(np.std(imp_alg))}
        },
        "decidability": {
            "raw_d_prime": float(d_prime_raw),
            "aligned_d_prime": float(d_prime_alg)
        },
        "auc": {
            "raw_auc": float(auc_raw),
            "aligned_auc": float(auc_alg)
        },
        "individual_genuine": step0_genuine_results,
        "individual_impostor": step0_impostor_results
    }

    results_json_path = os.path.join(BASE_DIR, "results", "step0_baseline_results.json")
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(step0_summary, f, indent=2)

    print(f"[+] Saved Step 0 baseline results to: {results_json_path}")

    print("\n" + "=" * 80)
    print("                    STEP 0: LOCKED BASELINE RESULTS")
    print("=" * 80)
    print(f"  Genuine Raw HD     : {np.mean(gen_raw):.2f} +- {np.std(gen_raw):.2f}  [Min: {np.min(gen_raw)}, Max: {np.max(gen_raw)}]")
    print(f"  Genuine Aligned HD : {np.mean(gen_alg):.2f} +- {np.std(gen_alg):.2f}  [Min: {np.min(gen_alg)}, Max: {np.max(gen_alg)}]")
    print(f"  Impostor Raw HD    : {np.mean(imp_raw):.2f} +- {np.std(imp_raw):.2f}  [Min: {np.min(imp_raw)}, Max: {np.max(imp_raw)}]")
    print(f"  Impostor Aligned HD: {np.mean(imp_alg):.2f} +- {np.std(imp_alg):.2f}  [Min: {np.min(imp_alg)}, Max: {np.max(imp_alg)}]")
    print(f"  Raw Decidability (d'): {d_prime_raw:.4f}  |  Raw AUC: {auc_raw:.4f}")
    print(f"  Aligned Decidability (d'): {d_prime_alg:.4f}  |  Aligned AUC: {auc_alg:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
