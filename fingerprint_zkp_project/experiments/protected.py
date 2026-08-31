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

from data import FingerprintDataset
from preprocessing import preprocess_fingerprint
from features import extract_minutiae, BiometricTemplate
from template_protection import FuzzyCommitment
from evaluation import compute_eer

def run_protected_experiment():
    print("=" * 70)
    print("  RUNNING EXPERIMENT 3: BIOMETRIC TEMPLATE PROTECTION (FUZZY COMMITMENT)")
    print("=" * 70)

    loader = FingerprintDataset()
    dataset = loader.load_dataset()

    print("\n[Step 1] Preprocessing and generating binary templates B for Fuzzy Commitment...")
    templates = {}
    for subject_id, samples in dataset.items():
        templates[subject_id] = {}
        for sample_id, img in samples.items():
            prep = preprocess_fingerprint(img)
            minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
            tmpl = BiometricTemplate(minutiae, image_shape=img.shape, vector_bits=256)
            templates[subject_id][sample_id] = tmpl

    fc = FuzzyCommitment(secret_bits=64, vector_bits=256)
    
    # Enroll all subjects (using first sample)
    enrolled_data = {}
    for s_id in templates:
        sample_1 = list(templates[s_id].keys())[0]
        t1 = templates[s_id][sample_1]
        enrolled_data[s_id] = fc.enroll(t1.binary_vector)

    print("\n[Step 2] Testing Genuine Secret Recovery (same subject, sample 2+)...")
    genuine_successes = 0
    total_genuine = 0
    genuine_scores = []

    for s_id in templates:
        helper = enrolled_data[s_id]["helper_data"]
        commitment = enrolled_data[s_id]["commitment"]
        samples = list(templates[s_id].keys())

        for sample_id in samples[1:]:
            t_query = templates[s_id][sample_id]
            res = fc.recover_secret(t_query.binary_vector, helper, commitment)
            total_genuine += 1
            if res["success"]:
                genuine_successes += 1
            
            # Score as 1.0 - (corrected_errors / total_bits)
            score = 1.0 - (res["errors_corrected"] / 256.0)
            genuine_scores.append(score)

    print("\n[Step 3] Testing Impostor Secret Recovery (different subjects)...")
    impostor_successes = 0
    total_impostor = 0
    impostor_scores = []
    subjects = list(templates.keys())

    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            s1, s2 = subjects[i], subjects[j]
            helper = enrolled_data[s1]["helper_data"]
            commitment = enrolled_data[s1]["commitment"]

            t_impostor = templates[s2][list(templates[s2].keys())[0]]
            res = fc.recover_secret(t_impostor.binary_vector, helper, commitment)
            total_impostor += 1
            if res["success"]:
                impostor_successes += 1
            
            score = 1.0 - (res["errors_corrected"] / 256.0)
            impostor_scores.append(score)

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
