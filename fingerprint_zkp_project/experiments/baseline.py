"""
Experiment 1 & 2: Baseline Biometric Matching Evaluation.
Calculates Genuine/Impostor score distributions, FAR, FRR, ROC curves, and Baseline EER.
Saves results, plots, and metrics to results/ directory.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data import FingerprintDataset, default_split_dirs
from preprocessing import preprocess_fingerprint
from features import extract_minutiae, BiometricTemplate, build_template
from matching import FingerprintMatcher
from evaluation import compute_far_frr, compute_roc, compute_eer

def run_baseline_experiment(max_subjects=12):
    print("=" * 70)
    print("  RUNNING EXPERIMENT 1 & 2: BASELINE BIOMETRIC MATCHING EVALUATION")
    print("=" * 70)

    # 1. Load Dataset
    splits = default_split_dirs()
    loader = FingerprintDataset(splits["test"])
    dataset = loader.load_dataset(max_subjects=max_subjects)

    # Preprocess all images and extract templates
    print("\n[Step 1] Preprocessing fingerprint dataset and extracting templates...")
    templates = {}
    
    for subject_id, samples in dataset.items():
        templates[subject_id] = {}
        for sample_id, img in samples.items():
            prep = preprocess_fingerprint(img)
            minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
            tmpl = build_template(img, prep, minutiae, vector_bits=256)
            templates[subject_id][sample_id] = tmpl

    matcher = FingerprintMatcher(mode="combined")
    genuine_scores = []
    impostor_scores = []

    print("\n[Step 2] Generating Genuine and Impostor pairwise comparisons...")
    subjects = list(templates.keys())

    # Genuine comparisons (same subject, different samples)
    for s_id in subjects:
        samples = list(templates[s_id].keys())
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                t1 = templates[s_id][samples[i]]
                t2 = templates[s_id][samples[j]]
                res = matcher.match(t1, t2)
                genuine_scores.append(res["score"])

    # Impostor comparisons (different subjects, first sample)
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            s1, s2 = subjects[i], subjects[j]
            t1 = templates[s1][list(templates[s1].keys())[0]]
            t2 = templates[s2][list(templates[s2].keys())[0]]
            res = matcher.match(t1, t2)
            impostor_scores.append(res["score"])

    print(f"Total Genuine comparisons: {len(genuine_scores)}")
    print(f"Total Impostor comparisons: {len(impostor_scores)}")

    # 3. Calculate FAR, FRR, ROC, EER
    print("\n[Step 3] Computing FAR, FRR, ROC, and Baseline EER...")
    eer_metrics = compute_eer(genuine_scores, impostor_scores)
    roc_metrics = compute_roc(eer_metrics["far_curve"], 1.0 - eer_metrics["frr_curve"])

    baseline_eer = eer_metrics["eer"]
    optimal_tau = eer_metrics["threshold"]

    print(f"\n[BASELINE RESULTS]")
    print(f"  -> Baseline EER: {baseline_eer * 100:.2f}%")
    print(f"  -> Optimal Decision Threshold (Tau): {optimal_tau:.4f}")
    print(f"  -> ROC AUC: {roc_metrics['auc']:.4f}")

    # 4. Save Figures & Tables
    fig_dir = os.path.join(BASE_DIR, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Plot 1: Score Distributions
    plt.figure(figsize=(8, 5))
    plt.hist(genuine_scores, bins=30, alpha=0.6, label='Genuine Scores', color='green', density=True)
    plt.hist(impostor_scores, bins=30, alpha=0.6, label='Impostor Scores', color='red', density=True)
    plt.axvline(optimal_tau, color='black', linestyle='--', label=f'EER Threshold ({optimal_tau:.2f})')
    plt.title('Baseline Genuine vs. Impostor Score Distributions')
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(fig_dir, "baseline_score_distribution.png"), dpi=300)
    plt.close()

    # Plot 2: FAR vs FRR Curve
    plt.figure(figsize=(8, 5))
    plt.plot(eer_metrics["thresholds"], eer_metrics["far_curve"], label='FAR (False Accept Rate)', color='red')
    plt.plot(eer_metrics["thresholds"], eer_metrics["frr_curve"], label='FRR (False Reject Rate)', color='blue')
    plt.axvline(optimal_tau, color='black', linestyle='--', label=f'EER = {baseline_eer*100:.2f}%')
    plt.title('Baseline FAR and FRR Curves')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(fig_dir, "baseline_far_frr_curve.png"), dpi=300)
    plt.close()

    # Plot 3: ROC Curve
    plt.figure(figsize=(7, 6))
    plt.plot(roc_metrics["far"], roc_metrics["tar"], label=f'ROC Curve (AUC = {roc_metrics["auc"]:.3f})', color='purple', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.title('Baseline ROC Curve')
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(fig_dir, "baseline_roc_curve.png"), dpi=300)
    plt.close()

    print(f"\n[Artifacts Saved] Baseline plots written to {fig_dir}")
    return eer_metrics

if __name__ == "__main__":
    run_baseline_experiment()
