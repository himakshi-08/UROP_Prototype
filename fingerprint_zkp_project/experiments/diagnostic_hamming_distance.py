"""
Diagnostic Experiment: Raw and Aligned Hamming Distance Distributions.
Measures the Hamming distance between 100 genuine pairs and 100 impostor pairs
from fp_testing BEFORE and AFTER rotational alignment, without ECC or Fuzzy Commitment.
"""

import os
import sys
import glob
import cv2
import numpy as np

# Ensure fingerprint_zkp_project is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.dataset_loader import default_split_dirs
from preprocessing import preprocess_fingerprint
from features import extract_minutiae, build_template
from features.template import iter_binary_alignments, BITS_PER_SECTOR, N_SECTORS

def load_fingerprint_template(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    prep = preprocess_fingerprint(img)
    minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
    tmpl = build_template(img, prep, minutiae, vector_bits=256)
    return tmpl.binary_vector

def main():
    splits = default_split_dirs()
    test_dir = splits["test"]
    print(f"[*] Loading testing images from: {test_dir}")

    # Discover subjects and their impressions
    subject_dirs = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    print(f"[*] Found {len(subject_dirs)} subjects in test set.")

    # Cache templates for subjects
    subject_templates = {}
    total_loaded = 0
    max_subjects_to_cache = 30  # 30 subjects * ~4 impressions = ~120 templates

    for s_id in subject_dirs[:max_subjects_to_cache]:
        s_path = os.path.join(test_dir, s_id)
        img_files = sorted(glob.glob(os.path.join(s_path, "*.[bB][mM][pP]")) +
                           glob.glob(os.path.join(s_path, "*.[tT][iI][fF]")) +
                           glob.glob(os.path.join(s_path, "*.[pP][nN][gG]")))
        
        tmpls = []
        for f in img_files:
            vec = load_fingerprint_template(f)
            if vec is not None:
                tmpls.append((os.path.basename(f), vec))
                total_loaded += 1
        
        if len(tmpls) >= 2:
            subject_templates[s_id] = tmpls

    print(f"[+] Loaded {total_loaded} fingerprint templates across {len(subject_templates)} subjects.")

    # 1. Collect exactly 100 Genuine Pairs
    genuine_raw_hd = []
    genuine_aligned_hd = []
    genuine_pairs_info = []

    for s_id, tmpls in subject_templates.items():
        n = len(tmpls)
        for i in range(n):
            for j in range(i + 1, n):
                v1 = tmpls[i][1]
                v2 = tmpls[j][1]

                # Raw HD
                raw_hd = int(np.sum(v1 != v2))
                
                # Aligned HD (min over 16 cyclic shifts)
                aligned_hd = min(int(np.sum(v1 != shifted)) for shifted in iter_binary_alignments(v2, N_SECTORS, BITS_PER_SECTOR))

                genuine_raw_hd.append(raw_hd)
                genuine_aligned_hd.append(aligned_hd)
                genuine_pairs_info.append((s_id, tmpls[i][0], tmpls[j][0], raw_hd, aligned_hd))

                if len(genuine_raw_hd) == 100:
                    break
            if len(genuine_raw_hd) == 100:
                break
        if len(genuine_raw_hd) == 100:
            break

    # 2. Collect exactly 100 Impostor Pairs
    impostor_raw_hd = []
    impostor_aligned_hd = []
    impostor_pairs_info = []

    s_ids = list(subject_templates.keys())
    for i in range(len(s_ids)):
        for j in range(i + 1, len(s_ids)):
            s1, s2 = s_ids[i], s_ids[j]
            # Compare sample 0 of s1 with sample 0 of s2
            v1 = subject_templates[s1][0][1]
            v2 = subject_templates[s2][0][1]

            raw_hd = int(np.sum(v1 != v2))
            aligned_hd = min(int(np.sum(v1 != shifted)) for shifted in iter_binary_alignments(v2, N_SECTORS, BITS_PER_SECTOR))

            impostor_raw_hd.append(raw_hd)
            impostor_aligned_hd.append(aligned_hd)
            impostor_pairs_info.append((s1, s2, raw_hd, aligned_hd))

            if len(impostor_raw_hd) == 100:
                break
        if len(impostor_raw_hd) == 100:
            break

    # Compute Statistics
    def stats(arr):
        a = np.array(arr, dtype=np.float64)
        return {
            "count": len(a),
            "min": int(np.min(a)),
            "max": int(np.max(a)),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
            "mean_pct": float(np.mean(a)) / 256.0 * 100.0,
            "min_pct": float(np.min(a)) / 256.0 * 100.0,
            "max_pct": float(np.max(a)) / 256.0 * 100.0,
        }

    gen_raw_st = stats(genuine_raw_hd)
    gen_alg_st = stats(genuine_aligned_hd)
    imp_raw_st = stats(impostor_raw_hd)
    imp_alg_st = stats(impostor_aligned_hd)

    # Decidability Index (d')
    # d' = |mean_imp - mean_gen| / sqrt(0.5 * (std_imp^2 + std_gen^2))
    d_prime_raw = abs(imp_raw_st["mean"] - gen_raw_st["mean"]) / np.sqrt(0.5 * (imp_raw_st["std"]**2 + gen_raw_st["std"]**2))
    d_prime_alg = abs(imp_alg_st["mean"] - gen_alg_st["mean"]) / np.sqrt(0.5 * (imp_alg_st["std"]**2 + gen_alg_st["std"]**2))

    print("\n" + "=" * 80)
    print("        DIAGNOSTIC EXPERIMENT: 100 GENUINE PAIRS vs 100 IMPOSTOR PAIRS")
    print("                     Vector Bit Length = 256 bits")
    print("=" * 80)
    
    print("\n[TABLE 1: GENUINE PAIR HAMMING DISTANCES (100 PAIRS)]")
    print(f"  Metric                     Before Alignment (Raw)       After Rotational Alignment")
    print(f"  ----------------------------------------------------------------------------------")
    print(f"  Minimum HD                 {gen_raw_st['min']:>4} ({gen_raw_st['min_pct']:>5.2f}%)               {gen_alg_st['min']:>4} ({gen_alg_st['min_pct']:>5.2f}%)")
    print(f"  Maximum HD                 {gen_raw_st['max']:>4} ({gen_raw_st['max_pct']:>5.2f}%)               {gen_alg_st['max']:>4} ({gen_alg_st['max_pct']:>5.2f}%)")
    print(f"  Mean HD                    {gen_raw_st['mean']:>6.2f} ({gen_raw_st['mean_pct']:>5.2f}%)             {gen_alg_st['mean']:>6.2f} ({gen_alg_st['mean_pct']:>5.2f}%)")
    print(f"  Median HD                  {gen_raw_st['median']:>6.2f} ({gen_raw_st['median']/256*100:>5.2f}%)             {gen_alg_st['median']:>6.2f} ({gen_alg_st['median']/256*100:>5.2f}%)")
    print(f"  Standard Deviation (sigma) {gen_raw_st['std']:>6.2f}                      {gen_alg_st['std']:>6.2f}")

    print("\n[TABLE 2: IMPOSTOR PAIR HAMMING DISTANCES (100 PAIRS)]")
    print(f"  Metric                     Before Alignment (Raw)       After Rotational Alignment")
    print(f"  ----------------------------------------------------------------------------------")
    print(f"  Minimum HD                 {imp_raw_st['min']:>4} ({imp_raw_st['min_pct']:>5.2f}%)               {imp_alg_st['min']:>4} ({imp_alg_st['min_pct']:>5.2f}%)")
    print(f"  Maximum HD                 {imp_raw_st['max']:>4} ({imp_raw_st['max_pct']:>5.2f}%)               {imp_alg_st['max']:>4} ({imp_alg_st['max_pct']:>5.2f}%)")
    print(f"  Mean HD                    {imp_raw_st['mean']:>6.2f} ({imp_raw_st['mean_pct']:>5.2f}%)             {imp_alg_st['mean']:>6.2f} ({imp_alg_st['mean_pct']:>5.2f}%)")
    print(f"  Median HD                  {imp_raw_st['median']:>6.2f} ({imp_raw_st['median']/256*100:>5.2f}%)             {imp_alg_st['median']:>6.2f} ({imp_alg_st['median']/256*100:>5.2f}%)")
    print(f"  Standard Deviation (sigma) {imp_raw_st['std']:>6.2f}                      {imp_alg_st['std']:>6.2f}")

    print("\n[TABLE 3: SEPARATION & DECIDABILITY]")
    print(f"  Raw Decidability Index (d')     : {d_prime_raw:.3f}")
    print(f"  Aligned Decidability Index (d') : {d_prime_alg:.3f}")
    print(f"  Theoretical Random Chance HD    : 128.00 (50.00%)")
    print("=" * 80)

if __name__ == "__main__":
    main()

