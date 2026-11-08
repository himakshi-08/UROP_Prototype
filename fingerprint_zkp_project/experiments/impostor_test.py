"""
Imposter Authentication Experiment Module.
Evaluates 1:1 imposter verification across all pairs (j != i).
Calculates imposter Hamming distances d_H(Bi, Bj), false acceptances, and False Acceptance Rate (FAR).
"""

import os
import glob
import numpy as np

from features.resnet_extractor import ResNet50FeatureExtractor
from features.binarization import get_binarizer
from template_protection.fuzzy_commitment import FuzzyCommitment
from experiments.verify_resnet_fuzzy_commitment import find_dataset_images

def run_impostor_test(dataset_path=None, binarization_method="standardized", secret_bits=128):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dataset_path is None:
        _ext = os.path.join(os.path.expanduser("~"), "Downloads", "fp_testing", "fp_testing")
        dataset_path = _ext if os.path.isdir(_ext) else os.path.join(base_dir, "Fingerprint_dataset")

    print(f"[*] Running Imposter Cross-Matching Test on: {dataset_path}")
    print(f"[*] Configuration: Binarization={binarization_method}, Secret K={secret_bits} bits")

    subject_images = find_dataset_images(dataset_path)
    extractor = ResNet50FeatureExtractor()

    raw_features = {s: extractor.extract_feature(img) for s, img in subject_images}
    all_raw = np.array([raw_features[s] for s in sorted(raw_features.keys())])

    binarizer = get_binarizer(binarization_method, enrollment_features=all_raw)
    binary_templates = {s: binarizer.transform(raw_features[s]) for s in raw_features}

    fc = FuzzyCommitment(secret_bits=secret_bits, vector_bits=2048)

    db = {s: fc.enroll(binary_templates[s]) for s in binary_templates}

    total_trials = 0
    false_acceptances = 0
    imposter_distances = []

    for i in sorted(binary_templates.keys()):
        for j in sorted(binary_templates.keys()):
            if i == j:
                continue
            total_trials += 1
            b_j = binary_templates[j]
            d_h = int(np.sum(binary_templates[i] != b_j))
            imposter_distances.append(d_h)

            rec = fc.recover_secret(b_j, db[i]["helper_data"], db[i]["commitment"])
            if rec["success"]:
                false_acceptances += 1

    far = (false_acceptances / total_trials) * 100.0 if total_trials > 0 else 0.0
    avg_dh = np.mean(imposter_distances) if imposter_distances else 0.0
    min_dh = np.min(imposter_distances) if imposter_distances else 0

    print("-" * 75)
    print(f"[*] Imposter Trials Evaluated: {total_trials}")
    print(f"[*] False Acceptances: {false_acceptances}")
    print(f"[*] False Acceptance Rate (FAR): {far:.4f}%")
    print(f"[*] Average Imposter Hamming Distance: {avg_dh:.1f} bits ({avg_dh/2048*100:.2f}%)")
    print(f"[*] Minimum Imposter Hamming Distance: {min_dh} bits ({min_dh/2048*100:.2f}%)")

    return {
        "trials": total_trials,
        "false_acceptances": false_acceptances,
        "far": far,
        "avg_hamming": avg_dh,
        "min_hamming": min_dh
    }

if __name__ == "__main__":
    run_impostor_test()
