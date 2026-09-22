"""
Experiment 3: Biometric Template Protection Evaluation (Fuzzy Commitment).
Evaluates Genuine Secret Recovery Rate, Impostor Secret Recovery Rate, and Template Protection EER.
Saves results, plots, and metrics to results/ directory.
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data import FingerprintDataset, default_split_dirs
from preprocessing import preprocess_fingerprint
from features import extract_minutiae, BiometricTemplate, build_template
from template_protection import FuzzyCommitment
from evaluation import compute_eer

def run_protected_experiment(max_subjects=12):
    print("=" * 70)
    print("  RUNNING EXPERIMENT 3: BIOMETRIC TEMPLATE PROTECTION (FUZZY COMMITMENT)")
    print("=" * 70)

    splits = default_split_dirs()
    loader = FingerprintDataset(splits["test"])
    dataset = loader.load_dataset(max_subjects=max_subjects)

    print("\n[Step 1] Preprocessing and generating binary templates B for Fuzzy Commitment...")
    templates = {}
    for subject_id, samples in dataset.items():
        templates[subject_id] = {}
        for sample_id, img in samples.items():
            prep = preprocess_fingerprint(img)
            minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
            tmpl = build_template(img, prep, minutiae, vector_bits=256)
            templates[subject_id][sample_id] = tmpl

    from authentication import SecureBiometricPipeline

    pipeline = SecureBiometricPipeline(secret_bits=32, vector_bits=256)
    
    # Enroll all subjects (sample 1)
    for s_id, samples in dataset.items():
        sample_1_img = samples[list(samples.keys())[0]]
        pipeline.enroll(s_id, sample_1_img)

    print("\n[Step 2] Testing Genuine Secret Recovery (same subject, sample 2+)...")
    genuine_successes = 0
    total_genuine = 0
    genuine_scores = []

    for s_id, samples in dataset.items():
        sample_ids = list(samples.keys())
        for sample_id in sample_ids[1:]:
            query_img = samples[sample_id]
            res = pipeline.authenticate(s_id, query_img)
            total_genuine += 1
            if res.get("authenticated", False) or res.get("secret_recovered", False):
                genuine_successes += 1
                score = 1.0 - (res.get("errors_corrected", 0) / 256.0)
            else:
                score = 0.5 - (res.get("errors_corrected", 64) / 256.0)
            genuine_scores.append(float(score))

    print("\n[Step 3] Testing Impostor Secret Recovery (different subjects)...")
    impostor_successes = 0
    total_impostor = 0
    impostor_scores = []
    subjects = list(dataset.keys())

    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            s1, s2 = subjects[i], subjects[j]
            impostor_img = dataset[s2][list(dataset[s2].keys())[0]]
            res = pipeline.authenticate(s1, impostor_img)
            total_impostor += 1
            if res.get("authenticated", False) or res.get("secret_recovered", False):
                impostor_successes += 1
                score = 1.0 - (res.get("errors_corrected", 0) / 256.0)
            else:
                score = 0.5 - (res.get("errors_corrected", 64) / 256.0)
            impostor_scores.append(float(score))

    genuine_rec_rate = (genuine_successes / max(1, total_genuine)) * 100.0
    impostor_rec_rate = (impostor_successes / max(1, total_impostor)) * 100.0
    eer_metrics = compute_eer(genuine_scores, impostor_scores)

    print(f"\n[FUZZY COMMITMENT RESULTS]")
    print(f"  -> Genuine Secret Recovery Rate: {genuine_rec_rate:.2f}% ({genuine_successes}/{total_genuine})")
    print(f"  -> Impostor Secret Recovery Rate: {impostor_rec_rate:.2f}% ({impostor_successes}/{total_impostor})")
    print(f"  -> Protected System EER: {eer_metrics['eer'] * 100:.2f}%")

    return {
        "genuine_recovery_rate": genuine_rec_rate,
        "impostor_recovery_rate": impostor_rec_rate,
        "eer": eer_metrics["eer"]
    }

if __name__ == "__main__":
    run_protected_experiment()
