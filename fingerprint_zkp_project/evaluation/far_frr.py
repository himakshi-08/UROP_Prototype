"""
FAR / FRR Evaluation Module.
Computes False Acceptance Rate (FAR) and False Rejection Rate (FRR) curves across threshold sweeps.
"""

import numpy as np

def compute_far_frr(genuine_scores, impostor_scores, thresholds=None):
    """
    Computes FAR, FRR, and TAR (True Acceptance Rate) arrays for given genuine and impostor similarity scores.
    
    Args:
        genuine_scores (list or np.ndarray): Scores from genuine comparisons (same subject).
        impostor_scores (list or np.ndarray): Scores from impostor comparisons (different subjects).
        thresholds (np.ndarray): Array of decision threshold values in range [0, 1].
        
    Returns:
        dict containing:
            - thresholds (np.ndarray)
            - far (np.ndarray)
            - frr (np.ndarray)
            - tar (np.ndarray)
    """
    genuine = np.array(genuine_scores, dtype=np.float64)
    impostor = np.array(impostor_scores, dtype=np.float64)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    far = np.zeros(len(thresholds))
    frr = np.zeros(len(thresholds))

    num_genuine = max(1, len(genuine))
    num_impostor = max(1, len(impostor))

    for idx, tau in enumerate(thresholds):
        # False Accept: Impostors accepted (score >= threshold)
        far[idx] = np.sum(impostor >= tau) / float(num_impostor)
        # False Reject: Genuines rejected (score < threshold)
        frr[idx] = np.sum(genuine < tau) / float(num_genuine)

    tar = 1.0 - frr

    return {
        "thresholds": thresholds,
        "far": far,
        "frr": frr,
        "tar": tar
    }
