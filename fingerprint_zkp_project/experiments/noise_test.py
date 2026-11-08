"""
Synthetic Bit-Flip Robustness Experiment Module.
Evaluates Fuzzy Commitment error recovery under controlled synthetic bit-flip noise.
Labeled strictly as a synthetic noise robustness experiment.
"""

import os
import numpy as np

from features.resnet_extractor import ResNet50FeatureExtractor
from features.binarization import get_binarizer
from template_protection.fuzzy_commitment import FuzzyCommitment
from experiments.verify_resnet_fuzzy_commitment import find_dataset_images

def run_noise_test(dataset_path=None, binarization_method="standardized", secret_bits=128):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dataset_path is None:
        _ext = os.path.join(os.path.expanduser("~"), "Downloads", "fp_testing", "fp_testing")
        dataset_path = _ext if os.path.isdir(_ext) else os.path.join(base_dir, "Fingerprint_dataset")

    print(f"[*] Running Synthetic Bit-Flip Robustness Experiment on: {dataset_path}")

    subject_images = find_dataset_images(dataset_path)
    extractor = ResNet50FeatureExtractor()

    raw_features = {s: extractor.extract_feature(img) for s, img in subject_images}
    all_raw = np.array([raw_features[s] for s in sorted(raw_features.keys())])

    binarizer = get_binarizer(binarization_method, enrollment_features=all_raw)
    binary_templates = {s: binarizer.transform(raw_features[s]) for s in raw_features}

    fc = FuzzyCommitment(secret_bits=secret_bits, vector_bits=2048)
    db = {s: fc.enroll(binary_templates[s]) for s in binary_templates}

    noise_levels = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    total_subjects = len(binary_templates)

    print(f"\n{'Synthetic Noise (%)':<20} | {'Expected Bit Errors':<20} | {'Successful Verifications':<25} | {'Recovery Rate (%)'}")
    print("-" * 85)

    results = []
    np.random.seed(42)

    for noise_p in noise_levels:
        successes = 0
        for s in sorted(binary_templates.keys()):
            orig_b = binary_templates[s].copy()
            noise_mask = np.random.binomial(1, noise_p, size=len(orig_b)).astype(np.uint8)
            noisy_b = np.bitwise_xor(orig_b, noise_mask)

            rec = fc.recover_secret(noisy_b, db[s]["helper_data"], db[s]["commitment"])
            if rec["success"]:
                successes += 1

        rate = (successes / total_subjects) * 100.0
        exp_bits = int(round(noise_p * 2048))
        print(f"{noise_p * 100:>16.1f}% | {exp_bits:>16} bits | {successes:>10}/{total_subjects:<12} | {rate:>16.2f}%")
        results.append({
            "noise_pct": noise_p * 100,
            "expected_errors": exp_bits,
            "successes": successes,
            "total": total_subjects,
            "recovery_rate": rate
        })

    return results

if __name__ == "__main__":
    run_noise_test()
