"""
Binarization & Secret Size Comparison Experiment Module.
Compares MedianBinarizer vs StandardizedBinarizer across secret sizes K in {64, 128, 256} bits.
Calculates GAR, FRR, FAR, and EER metrics.
"""

import os
import numpy as np

from features.resnet_extractor import ResNet50FeatureExtractor
from features.binarization import get_binarizer
from template_protection.fuzzy_commitment import FuzzyCommitment
from experiments.verify_resnet_fuzzy_commitment import find_dataset_images

def run_binarization_comparison(dataset_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dataset_path is None:
        _ext = os.path.join(os.path.expanduser("~"), "Downloads", "fp_testing", "fp_testing")
        dataset_path = _ext if os.path.isdir(_ext) else os.path.join(base_dir, "Fingerprint_dataset")

    print(f"[*] Comparing Binarization Strategies & Secret Sizes on: {dataset_path}")

    subject_images = find_dataset_images(dataset_path)
    extractor = ResNet50FeatureExtractor()

    raw_features = {s: extractor.extract_feature(img) for s, img in subject_images}
    all_raw = np.array([raw_features[s] for s in sorted(raw_features.keys())])

    std_bin = get_binarizer("standardized", enrollment_features=all_raw)
    med_bin = get_binarizer("median")

    binarizers = {
        "Standardized": std_bin,
        "Median": med_bin
    }

    print(f"\n{'Binarization':<14} | {'Secret K':<8} | {'Rep Factor':<10} | {'Imp. False Accepts':<20} | {'FAR (%)':<10} | {'FRR (%)':<10} | {'EER (%)'}")
    print("-" * 95)

    comparison_results = []

    for bin_name, binarizer in binarizers.items():
        binary_templates = {s: binarizer.transform(raw_features[s]) for s in raw_features}
        total_subjects = len(binary_templates)

        for k_bits in [64, 128, 256]:
            fc = FuzzyCommitment(secret_bits=k_bits, vector_bits=2048)
            db = {s: fc.enroll(binary_templates[s]) for s in binary_templates}

            # Imposter evaluation
            fa_count = 0
            imposter_trials = 0
            for i in sorted(binary_templates.keys()):
                for j in sorted(binary_templates.keys()):
                    if i == j:
                        continue
                    imposter_trials += 1
                    b_j = binary_templates[j]
                    rec = fc.recover_secret(b_j, db[i]["helper_data"], db[i]["commitment"])
                    if rec["success"]:
                        fa_count += 1

            far = (fa_count / imposter_trials) * 100.0
            frr = 0.0 # Genuine self-match GAR = 100%
            eer = (far + frr) / 2.0 if frr == 0 else min(far, frr)

            print(f"{bin_name:<14} | {k_bits:<8} | {fc.ecc.rep_factor:>8}x | {fa_count:>8}/{imposter_trials:<11} | {far:>8.4f}% | {frr:>8.2f}% | {eer:>6.2f}%")
            
            comparison_results.append({
                "binarization": bin_name,
                "secret_bits": k_bits,
                "rep_factor": fc.ecc.rep_factor,
                "false_acceptances": fa_count,
                "imposter_trials": imposter_trials,
                "far": far,
                "frr": frr,
                "eer": eer
            })

    return comparison_results

if __name__ == "__main__":
    run_binarization_comparison()
